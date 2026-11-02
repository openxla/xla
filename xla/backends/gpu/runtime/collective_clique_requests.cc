/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/gpu/runtime/collective_clique_requests.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/runtime/device_id.h"
#include "xla/util.h"

namespace xla::gpu {

absl::Status CollectiveCliqueRequests::RequestClique(
    const GpuCliqueKey& clique_key,
    absl::Span<const std::vector<GlobalDeviceId>> device_groups,
    const CliqueRequirements& requirements) {
  if (device_groups.empty()) {
    return absl::InvalidArgumentError(
        "Collective clique requests require a nonempty device group partition");
  }
  // Each replica group should have devices in sorted order.
  DCHECK(absl::c_all_of(device_groups, [](const auto& g) {
    return absl::c_is_sorted(g);
  })) << "Device group must be sorted in ascending order";

  // And replica groups themselves should be sorted (lexicographic order on
  // sorted inner groups is equivalent to sorting by first device id).
  DCHECK(absl::c_is_sorted(device_groups))
      << "Device groups must be sorted by first device id";

  VLOG(5) << absl::StreamFormat(
      "Add collective clique request: %v; device_groups=[%s]; "
      "requirements=%v",
      clique_key, HumanReadableDeviceGroups(device_groups), requirements);

  // A singleton clique can be requested from multiple partitions. Preserve
  // every partition before deduplicating by the rank-local clique key.
  if (!graph_requests_.empty() &&
      !absl::c_equal(graph_requests_.front().device_groups, device_groups)) {
    has_mixed_device_groups_ = true;
  }
  if (!absl::c_any_of(graph_requests_, [&](const GraphRequest& request) {
        return request.communication_id == clique_key.communication_id() &&
               absl::c_equal(request.device_groups, device_groups);
      })) {
    graph_requests_.push_back({clique_key.communication_id(),
                               {device_groups.begin(), device_groups.end()}});
  }

  // If the clique already exists, update it with new requirements.
  if (auto it = cliques_.find(clique_key); it != cliques_.end()) {
    CliqueRequest& req = it->second;

    // It is illegal to request the same multi-device GPU clique with different
    // device groups under SPMD. Single-device (singleton) cliques are exempt:
    // they involve no communication and are never split, so connected-component
    // collective-permute may legitimately request the same singleton key with
    // different surrounding device groups.
    if (clique_key.devices().size() > 1 &&
        !absl::c_equal(req.device_groups, device_groups)) {
      return InvalidArgument(
          "GPU clique %v requested from different device groups: [%s] vs [%s]",
          clique_key, HumanReadableDeviceGroups(req.device_groups),
          HumanReadableDeviceGroups(device_groups));
    }

    if (requirements.dev_comm) {
      req.dev_comms.insert(*requirements.dev_comm);
    }

    if (requirements.barrier_reqs.has_value()) {
      req.use_barrier_semaphores |=
          requirements.barrier_reqs->use_barrier_semaphores;
      req.use_cross_device_barrier_requested |=
          requirements.barrier_reqs->use_cross_device_barrier;
    }

    req.use_gxl_requested |= requirements.use_gxl;

    return absl::OkStatus();
  }

  // XLA compiler guarantees that all collective operations have the same
  // order on all replicas. We rely on this property to assign unique id to
  // clique requests simply based on the number of already recorded requests.
  CliqueRequest req{/*id=*/cliques_.size(),
                    clique_key,
                    {device_groups.begin(), device_groups.end()}};
  if (requirements.dev_comm) {
    req.dev_comms.insert(*requirements.dev_comm);
  }

  if (requirements.barrier_reqs.has_value()) {
    req.use_cross_device_barrier_requested |=
        requirements.barrier_reqs->use_cross_device_barrier;
    req.use_barrier_semaphores |=
        requirements.barrier_reqs->use_barrier_semaphores;
  }

  req.use_gxl_requested |= requirements.use_gxl;

  cliques_.try_emplace(clique_key, std::move(req));
  return absl::OkStatus();
}

std::vector<GpuCliqueKey> CollectiveCliqueRequests::RequestedCliques() const {
  std::vector<GpuCliqueKey> clique_keys;
  clique_keys.reserve(cliques_.size());
  for (const auto& [key, _] : cliques_) {
    clique_keys.push_back(key);
  }

  return clique_keys;
}

std::string CollectiveCliqueRequests::CollectiveGraphSignature() const {
  std::string signature;
  for (const GraphRequest& request : graph_requests_) {
    absl::StrAppend(&signature, request.communication_id.value(), ":",
                    request.device_groups.size(), ":");
    for (const auto& group : request.device_groups) {
      absl::StrAppend(&signature, group.size(), ":");
      for (GlobalDeviceId device : group) {
        absl::StrAppend(&signature, device.value(), ",");
      }
    }
    absl::StrAppend(&signature, ";");
  }
  return signature;
}

std::vector<GlobalDeviceId>
CollectiveCliqueRequests::CollectiveGraphParticipants(
    GlobalDeviceId device) const {
  absl::flat_hash_set<GlobalDeviceId> participants = {device};
  size_t previous_size;
  do {
    previous_size = participants.size();
    for (const GraphRequest& request : graph_requests_) {
      for (const auto& group : request.device_groups) {
        if (absl::c_any_of(group, [&](GlobalDeviceId member) {
              return participants.contains(member);
            })) {
          participants.insert(group.begin(), group.end());
        }
      }
    }
  } while (participants.size() != previous_size);

  std::vector<GlobalDeviceId> result(participants.begin(), participants.end());
  absl::c_sort(result);
  return result;
}

std::vector<CollectiveCliqueRequests::CliqueRequest>
CollectiveCliqueRequests::OrderedRequestedCliques() const {
  std::vector<CliqueRequest> cliques;
  cliques.reserve(cliques_.size());
  for (const auto& [_, request] : cliques_) {
    cliques.push_back(request);
  }

  absl::c_sort(cliques, [](const CliqueRequest& a, const CliqueRequest& b) {
    // Acquire larger cliques first to be able to split them later.
    if (a.key.devices().size() > b.key.devices().size()) {
      return true;
    }
    if (b.key.devices().size() > a.key.devices().size()) {
      return false;
    }

    // Prefer cliques with smaller id (comes earlier in execution order).
    return a.id < b.id;
  });

  return cliques;
}

}  // namespace xla::gpu
