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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"
#include "xla/ffi/ffi.h"

namespace xla::gpu::cutedsl {
namespace {

constexpr size_t kPeerRegionRecordWidth = 6;
constexpr size_t kStepRecordWidth = 2;

constexpr std::array<absl::string_view, 11> kAllAttributeNames = {
    "schema_version",
    "group_mode",
    "communication_id",
    "replica_group_offsets",
    "replica_group_members",
    "module_blob",
    "module_offsets",
    "module_keys",
    "module_index_by_rank",
    "peer_regions",
    "steps",
};

bool IsExpectedAttribute(absl::string_view name) {
  return std::find(kAllAttributeNames.begin(), kAllAttributeNames.end(),
                   name) != kAllAttributeNames.end();
}

absl::Status ValidateAttributeNames(const ffi::Dictionary& attributes) {
  for (std::string_view name : attributes) {
    absl::string_view attribute_name(name.data(), name.size());
    if (!IsExpectedAttribute(attribute_name)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unknown CuTeDSL collective v3 attribute `%s`", attribute_name));
    }
  }

  for (absl::string_view name : kAllAttributeNames) {
    if (!attributes.contains(name)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Missing CuTeDSL collective v3 attribute `%s`", name));
    }
  }

  return absl::OkStatus();
}

template <typename T>
absl::StatusOr<T> GetAttribute(const ffi::Dictionary& attributes,
                               absl::string_view name) {
  absl::StatusOr<T> value = attributes.get<T>(name);
  if (!value.ok()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid CuTeDSL collective v3 attribute `%s`: %s",
                        name, value.status().message()));
  }
  return *value;
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
    absl::Span<const int64_t> offsets, absl::Span<const int64_t> members) {
  if (offsets.size() < 2) {
    return absl::InvalidArgumentError(
        "`replica_group_offsets` must describe at least one group");
  }
  if (offsets.front() != 0) {
    return absl::InvalidArgumentError(
        "`replica_group_offsets` must start with zero");
  }
  if (members.size() >
          static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      offsets.back() != static_cast<int64_t>(members.size())) {
    return absl::InvalidArgumentError(
        "`replica_group_offsets` must end at `replica_group_members` size");
  }

  std::vector<ReplicaGroup> groups;
  groups.reserve(offsets.size() - 1);
  std::set<int64_t> unique_members;
  int64_t group_size = -1;

  for (size_t group_index = 0; group_index + 1 < offsets.size();
       ++group_index) {
    int64_t begin = offsets[group_index];
    int64_t end = offsets[group_index + 1];
    if (begin < 0 || end <= begin ||
        end > static_cast<int64_t>(members.size())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid replica-group row split [%d, %d) at group %d", begin, end,
          group_index));
    }

    int64_t size = end - begin;
    if (group_size == -1) {
      group_size = size;
    } else if (size != group_size) {
      return absl::InvalidArgumentError(
          "All replica groups must have equal cardinality");
    }

    ReplicaGroup group;
    for (int64_t member_index = begin; member_index < end; ++member_index) {
      int64_t member = members[member_index];
      if (member < 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Replica-group member IDs must be nonnegative; got %d", member));
      }
      if (!unique_members.insert(member).second) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Replica-group member ID %d appears more than once", member));
      }
      group.add_replica_ids(member);
    }
    groups.push_back(std::move(group));
  }

  return groups;
}

absl::StatusOr<std::vector<CollectiveModuleImageV3>> ParseModules(
    absl::string_view blob, absl::Span<const int64_t> offsets,
    absl::string_view keys) {
  if (blob.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return absl::InvalidArgumentError("`module_blob` is too large");
  }
  if (offsets.size() < 2) {
    return absl::InvalidArgumentError(
        "`module_offsets` must describe at least one module image");
  }
  if (offsets.front() != 0) {
    return absl::InvalidArgumentError("`module_offsets` must start with zero");
  }
  if (offsets.back() != static_cast<int64_t>(blob.size())) {
    return absl::InvalidArgumentError(
        "`module_offsets` must end at `module_blob` size");
  }

  size_t module_count = offsets.size() - 1;
  if (keys.size() % kCollectiveModuleDigestSizeV3 != 0 ||
      keys.size() / kCollectiveModuleDigestSizeV3 != module_count) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`module_keys` must contain one %d-byte SHA-256 digest per module",
        kCollectiveModuleDigestSizeV3));
  }

  std::vector<CollectiveModuleImageV3> modules;
  modules.reserve(module_count);
  for (size_t module_index = 0; module_index < module_count; ++module_index) {
    int64_t begin = offsets[module_index];
    int64_t end = offsets[module_index + 1];
    if (begin < 0 || end <= begin || end > static_cast<int64_t>(blob.size())) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid or empty module range [%d, %d) at module %d",
                          begin, end, module_index));
    }

    absl::string_view image = blob.substr(begin, end - begin);
    absl::string_view key =
        keys.substr(module_index * kCollectiveModuleDigestSizeV3,
                    kCollectiveModuleDigestSizeV3);

    llvm::SHA256 hasher;
    hasher.update(llvm::StringRef(image.data(), image.size()));
    std::array<uint8_t, kCollectiveModuleDigestSizeV3> digest = hasher.final();
    if (!std::equal(digest.begin(), digest.end(),
                    reinterpret_cast<const uint8_t*>(key.data()))) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "SHA-256 digest does not match module image %d", module_index));
    }

    CollectiveModuleImageV3 module;
    module.bytes.assign(image.data(), image.size());
    module.sha256 = digest;
    modules.push_back(std::move(module));
  }

  return modules;
}

absl::Status ValidateModuleSelection(
    absl::Span<const int64_t> module_index_by_rank, size_t module_count) {
  if (module_index_by_rank.empty()) {
    return absl::InvalidArgumentError(
        "`module_index_by_rank` must not be empty");
  }
  for (size_t rank = 0; rank < module_index_by_rank.size(); ++rank) {
    int64_t module_index = module_index_by_rank[rank];
    if (module_index < 0 ||
        static_cast<uint64_t>(module_index) >= module_count) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Module index %d for clique rank %d is out of range [0, %d)",
          module_index, rank, module_count));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<PeerRegionV3>> ParsePeerRegions(
    absl::Span<const int64_t> records) {
  if (records.size() % kPeerRegionRecordWidth != 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("`peer_regions` size must be a multiple of %d",
                        kPeerRegionRecordWidth));
  }

  using PeerRegionKey =
      std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
  std::set<PeerRegionKey> unique_regions;
  std::vector<PeerRegionV3> regions;
  regions.reserve(records.size() / kPeerRegionRecordWidth);

  for (size_t offset = 0; offset < records.size();
       offset += kPeerRegionRecordWidth) {
    int64_t endpoint_value = records[offset];
    int64_t buffer_index = records[offset + 1];
    int64_t byte_offset = records[offset + 2];
    int64_t byte_size = records[offset + 3];
    int64_t required_alignment = records[offset + 4];
    int64_t memory_kind_value = records[offset + 5];
    size_t region_index = offset / kPeerRegionRecordWidth;

    PeerRegionEndpointV3 endpoint;
    switch (endpoint_value) {
      case static_cast<int64_t>(PeerRegionEndpointV3::kArgument):
        endpoint = PeerRegionEndpointV3::kArgument;
        break;
      case static_cast<int64_t>(PeerRegionEndpointV3::kResult):
        endpoint = PeerRegionEndpointV3::kResult;
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrFormat("Unsupported endpoint %d for peer region %d",
                            endpoint_value, region_index));
    }
    if (buffer_index < 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Buffer index for peer region %d must be nonnegative", region_index));
    }
    if (byte_offset < 0 || byte_size <= 0 ||
        byte_offset > std::numeric_limits<int64_t>::max() - byte_size) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid or overflowing byte range for peer region %d",
          region_index));
    }
    if (required_alignment <= 0 ||
        (required_alignment & (required_alignment - 1)) != 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Required alignment for peer region %d must be a positive power of "
          "two",
          region_index));
    }
    if (memory_kind_value !=
        static_cast<int64_t>(PeerMemoryKindV3::kSymmetric)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported memory kind %d for peer region %d",
                          memory_kind_value, region_index));
    }

    PeerRegionKey key = {endpoint_value, buffer_index,       byte_offset,
                         byte_size,      required_alignment, memory_kind_value};
    if (!unique_regions.insert(key).second) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Peer region record %d duplicates an earlier record", region_index));
    }

    regions.push_back(PeerRegionV3{endpoint, buffer_index, byte_offset,
                                   byte_size, required_alignment,
                                   PeerMemoryKindV3::kSymmetric});
  }

  return regions;
}

absl::StatusOr<std::vector<CollectiveStepV3>> ParseSteps(
    absl::Span<const int64_t> records) {
  if (records.empty() || records.size() % kStepRecordWidth != 0) {
    return absl::InvalidArgumentError(
        "`steps` must contain complete [kind, operand] records");
  }

  std::vector<CollectiveStepV3> steps;
  steps.reserve(records.size() / kStepRecordWidth);
  bool has_launch = false;
  for (size_t offset = 0; offset < records.size(); offset += kStepRecordWidth) {
    int64_t kind_value = records[offset];
    int64_t operand = records[offset + 1];
    size_t step_index = offset / kStepRecordWidth;
    switch (kind_value) {
      case static_cast<int64_t>(CollectiveStepKindV3::kBarrier):
        if (operand != 0) {
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
      case static_cast<int64_t>(CollectiveStepKindV3::kLaunch):
        if (operand < 0) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Launch step %d must have a nonnegative function ordinal",
              step_index));
        }
        has_launch = true;
        steps.push_back(
            CollectiveStepV3{CollectiveStepKindV3::kLaunch, operand});
        break;
      default:
        return absl::InvalidArgumentError(absl::StrFormat(
            "Unsupported step kind %d at step %d", kind_value, step_index));
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
    const ffi::Dictionary& attributes) {
  RETURN_IF_ERROR(ValidateAttributeNames(attributes));

  ASSIGN_OR_RETURN(int64_t schema_version,
                   GetAttribute<int64_t>(attributes, "schema_version"));
  if (schema_version != kCollectiveCallSchemaVersionV3) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unsupported CuTeDSL collective schema version %d; expected %d",
        schema_version, kCollectiveCallSchemaVersionV3));
  }

  ASSIGN_OR_RETURN(int64_t group_mode_value,
                   GetAttribute<int64_t>(attributes, "group_mode"));
  ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                   ParseGroupMode(group_mode_value));

  ASSIGN_OR_RETURN(int64_t communication_id,
                   GetAttribute<int64_t>(attributes, "communication_id"));
  if (communication_id < 0) {
    return absl::InvalidArgumentError("`communication_id` must be nonnegative");
  }

  ASSIGN_OR_RETURN(absl::Span<const int64_t> replica_group_offsets,
                   GetAttribute<absl::Span<const int64_t>>(
                       attributes, "replica_group_offsets"));
  ASSIGN_OR_RETURN(absl::Span<const int64_t> replica_group_members,
                   GetAttribute<absl::Span<const int64_t>>(
                       attributes, "replica_group_members"));
  ASSIGN_OR_RETURN(
      std::vector<ReplicaGroup> replica_groups,
      ParseReplicaGroups(replica_group_offsets, replica_group_members));

  ASSIGN_OR_RETURN(absl::string_view module_blob,
                   GetAttribute<absl::string_view>(attributes, "module_blob"));
  ASSIGN_OR_RETURN(
      absl::Span<const int64_t> module_offsets,
      GetAttribute<absl::Span<const int64_t>>(attributes, "module_offsets"));
  ASSIGN_OR_RETURN(absl::string_view module_keys,
                   GetAttribute<absl::string_view>(attributes, "module_keys"));
  ASSIGN_OR_RETURN(std::vector<CollectiveModuleImageV3> modules,
                   ParseModules(module_blob, module_offsets, module_keys));

  ASSIGN_OR_RETURN(absl::Span<const int64_t> module_index_by_rank,
                   GetAttribute<absl::Span<const int64_t>>(
                       attributes, "module_index_by_rank"));
  // The final clique size depends on group mode and the runtime device
  // assignment. Prepare validates this map's length against the derived key.
  RETURN_IF_ERROR(
      ValidateModuleSelection(module_index_by_rank, modules.size()));

  ASSIGN_OR_RETURN(
      absl::Span<const int64_t> peer_region_records,
      GetAttribute<absl::Span<const int64_t>>(attributes, "peer_regions"));
  ASSIGN_OR_RETURN(std::vector<PeerRegionV3> peer_regions,
                   ParsePeerRegions(peer_region_records));

  ASSIGN_OR_RETURN(
      absl::Span<const int64_t> step_records,
      GetAttribute<absl::Span<const int64_t>>(attributes, "steps"));
  ASSIGN_OR_RETURN(std::vector<CollectiveStepV3> steps,
                   ParseSteps(step_records));

  return CollectiveCallConfigV3{
      group_mode,
      communication_id,
      std::move(replica_groups),
      std::move(modules),
      std::vector<int64_t>(module_index_by_rank.begin(),
                           module_index_by_rank.end()),
      std::move(peer_regions),
      std::move(steps),
  };
}

}  // namespace xla::gpu::cutedsl
