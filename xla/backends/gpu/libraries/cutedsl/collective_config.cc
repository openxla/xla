/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/libraries/cutedsl/collective_config.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/libraries/cutedsl/collective_config.pb.h"
#include "xla/backends/gpu/libraries/cutedsl/module_image.h"
#include "tsl/platform/protobuf.h"

namespace xla::gpu::cutedsl {

using ::xla::CollectiveOpGroupMode;
using ::xla::ReplicaGroup;
namespace wire = ::xla::gpu::cutedsl::proto;

namespace {

absl::Status MissingField(absl::string_view field) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "Missing CuTeDSL collective v3 ProtoJSON field `%s`", field));
}

absl::Status MissingRepeatedField(absl::string_view repeated_field,
                                  size_t index, absl::string_view field) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "Missing CuTeDSL collective v3 ProtoJSON field `%s[%d].%s`",
      repeated_field, index, field));
}

absl::StatusOr<CollectiveOpGroupMode> ParseGroupMode(int64_t value) {
  switch (value) {
    case static_cast<int64_t>(
        CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA):
    case static_cast<int64_t>(
        CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION):
    case static_cast<int64_t>(
        CollectiveOpGroupMode::
            COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION):
    case static_cast<int64_t>(
        CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID):
      return static_cast<CollectiveOpGroupMode>(value);
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported collective group mode %d", value));
  }
}

absl::StatusOr<std::vector<ReplicaGroup>> ParseReplicaGroups(
    const wire::CollectiveCallConfigV3& proto) {
  if (proto.replica_groups().empty()) {
    return absl::InvalidArgumentError(
        "`replica_groups` must contain at least one group");
  }

  std::vector<ReplicaGroup> groups;
  groups.reserve(proto.replica_groups_size());
  std::set<int64_t> unique_members;
  int64_t group_size = -1;

  for (int group_index = 0; group_index < proto.replica_groups_size();
       ++group_index) {
    const ReplicaGroup& group = proto.replica_groups(group_index);
    if (group.replica_ids().empty()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Replica group %d must not be empty", group_index));
    }
    if (group_size == -1) {
      group_size = group.replica_ids_size();
    } else if (group.replica_ids_size() != group_size) {
      return absl::InvalidArgumentError(
          "All replica groups must have equal cardinality");
    }

    for (int64_t member : group.replica_ids()) {
      if (member < 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Replica-group member IDs must be nonnegative; got %d", member));
      }
      if (!unique_members.insert(member).second) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Replica-group member ID %d appears more than once", member));
      }
    }
    groups.push_back(group);
  }

  return groups;
}

absl::Status ValidatePeerRegionFields(const wire::PeerRegionProto& region,
                                      size_t region_index) {
  if (!region.has_endpoint()) {
    return MissingRepeatedField("peer_regions", region_index, "endpoint");
  }
  if (!region.has_buffer_index()) {
    return MissingRepeatedField("peer_regions", region_index, "buffer_index");
  }
  if (!region.has_byte_offset()) {
    return MissingRepeatedField("peer_regions", region_index, "byte_offset");
  }
  if (!region.has_byte_size()) {
    return MissingRepeatedField("peer_regions", region_index, "byte_size");
  }
  if (!region.has_required_alignment()) {
    return MissingRepeatedField("peer_regions", region_index,
                                "required_alignment");
  }
  if (!region.has_memory_kind()) {
    return MissingRepeatedField("peer_regions", region_index, "memory_kind");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<PeerRegionV3>> ParsePeerRegions(
    const wire::CollectiveCallConfigV3& proto) {
  using PeerRegionKey =
      std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
  std::set<PeerRegionKey> unique_regions;
  std::vector<PeerRegionV3> regions;
  regions.reserve(proto.peer_regions_size());

  for (int region_index = 0; region_index < proto.peer_regions_size();
       ++region_index) {
    const wire::PeerRegionProto& region = proto.peer_regions(region_index);
    RETURN_IF_ERROR(ValidatePeerRegionFields(region, region_index));

    PeerRegionEndpointV3 endpoint;
    switch (region.endpoint()) {
      case wire::PEER_REGION_ENDPOINT_PROTO_ARGUMENT:
        endpoint = PeerRegionEndpointV3::kArgument;
        break;
      case wire::PEER_REGION_ENDPOINT_PROTO_RESULT:
        endpoint = PeerRegionEndpointV3::kResult;
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrFormat("Unsupported endpoint %d for peer region %d",
                            region.endpoint(), region_index));
    }
    if (region.buffer_index() < 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Buffer index for peer region %d must be nonnegative", region_index));
    }
    if (region.byte_offset() < 0 || region.byte_size() <= 0 ||
        region.byte_offset() >
            std::numeric_limits<int64_t>::max() - region.byte_size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid or overflowing byte range for peer region %d",
          region_index));
    }
    if (region.required_alignment() <= 0 ||
        (region.required_alignment() & (region.required_alignment() - 1)) !=
            0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Required alignment for peer region %d must be a positive power of "
          "two",
          region_index));
    }
    if (region.memory_kind() != wire::PEER_MEMORY_KIND_PROTO_SYMMETRIC) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported memory kind %d for peer region %d",
                          region.memory_kind(), region_index));
    }

    PeerRegionKey key = {static_cast<int64_t>(region.endpoint()),
                         region.buffer_index(),
                         region.byte_offset(),
                         region.byte_size(),
                         region.required_alignment(),
                         static_cast<int64_t>(region.memory_kind())};
    if (!unique_regions.insert(key).second) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Peer region record %d duplicates an earlier record", region_index));
    }

    regions.push_back(PeerRegionV3{endpoint, region.buffer_index(),
                                   region.byte_offset(), region.byte_size(),
                                   region.required_alignment(),
                                   PeerMemoryKindV3::kSymmetric});
  }

  return regions;
}

absl::StatusOr<std::vector<CollectiveStepV3>> ParseSteps(
    const wire::CollectiveCallConfigV3& proto) {
  if (proto.steps().empty()) {
    return absl::InvalidArgumentError(
        "`steps` must contain complete [kind, operand] records");
  }

  std::vector<CollectiveStepV3> steps;
  steps.reserve(proto.steps_size());
  bool has_launch = false;
  for (int step_index = 0; step_index < proto.steps_size(); ++step_index) {
    const wire::CollectiveStepProto& step = proto.steps(step_index);
    if (!step.has_kind()) {
      return MissingRepeatedField("steps", step_index, "kind");
    }
    if (!step.has_operand()) {
      return MissingRepeatedField("steps", step_index, "operand");
    }

    switch (step.kind()) {
      case wire::COLLECTIVE_STEP_KIND_PROTO_BARRIER:
        if (step.operand() != 0) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Barrier step %d must have operand zero", step_index));
        }
        if (has_launch) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Barrier step %d follows a launch; v3 requires all barriers "
              "before the first launch until cross-rank launch-error "
              "agreement is implemented",
              step_index));
        }
        steps.push_back(CollectiveStepV3{CollectiveStepKindV3::kBarrier, 0});
        break;
      case wire::COLLECTIVE_STEP_KIND_PROTO_LAUNCH:
        if (step.operand() < 0) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Launch step %d must have a nonnegative function ordinal",
              step_index));
        }
        has_launch = true;
        steps.push_back(
            CollectiveStepV3{CollectiveStepKindV3::kLaunch, step.operand()});
        break;
      default:
        return absl::InvalidArgumentError(absl::StrFormat(
            "Unsupported step kind %d at step %d", step.kind(), step_index));
    }
  }

  if (!has_launch) {
    return absl::InvalidArgumentError(
        "`steps` must contain at least one launch");
  }
  return steps;
}

}  // namespace

absl::StatusOr<CollectiveCallConfigV3> ParseCollectiveCallConfigV3(
    absl::string_view json_config) {
  wire::CollectiveCallConfigV3 proto;
  tsl::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  absl::Status parsed =
      tsl::protobuf::util::JsonStringToMessage(json_config, &proto, options);
  if (!parsed.ok()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to parse CuTeDSL collective v3 ProtoJSON configuration: %s",
        parsed.message()));
  }

  if (!proto.has_abi_clique_size()) return MissingField("abi_clique_size");
  if (proto.abi_clique_size() <= 0 ||
      proto.abi_clique_size() > std::numeric_limits<int32_t>::max()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`abi_clique_size` must be in [1, %d]; got %d",
        std::numeric_limits<int32_t>::max(), proto.abi_clique_size()));
  }

  if (!proto.has_group_mode()) return MissingField("group_mode");
  ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                   ParseGroupMode(static_cast<int64_t>(proto.group_mode())));

  if (!proto.has_communication_id()) return MissingField("communication_id");
  if (proto.communication_id() < 0) {
    return absl::InvalidArgumentError("`communication_id` must be nonnegative");
  }

  ASSIGN_OR_RETURN(std::vector<ReplicaGroup> replica_groups,
                   ParseReplicaGroups(proto));
  if (!proto.has_module()) return MissingField("module");
  ASSIGN_OR_RETURN(ModuleImage module, ModuleImage::Create(proto.module()));
  ASSIGN_OR_RETURN(std::vector<PeerRegionV3> peer_regions,
                   ParsePeerRegions(proto));
  ASSIGN_OR_RETURN(std::vector<CollectiveStepV3> steps, ParseSteps(proto));

  return CollectiveCallConfigV3{
      static_cast<int32_t>(proto.abi_clique_size()),
      group_mode,
      proto.communication_id(),
      std::move(replica_groups),
      std::move(module),
      std::move(peer_regions),
      std::move(steps),
  };
}

}  // namespace xla::gpu::cutedsl
