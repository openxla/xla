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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_COLLECTIVE_CONFIG_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_COLLECTIVE_CONFIG_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/libraries/cutedsl/module_image.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::cutedsl {

enum class PeerRegionEndpointV3 : int64_t {
  kArgument = 0,
  kResult = 1,
};

// V3 supports only XLA load/store-accessible symmetric memory. Keeping this
// field explicit permits a future version to add other generic memory kinds.
enum class PeerMemoryKindV3 : int64_t {
  kSymmetric = 0,
};

enum class CollectiveStepKindV3 : int64_t {
  kBarrier = 0,
  kLaunch = 1,
};

struct PeerRegionV3 {
  PeerRegionEndpointV3 endpoint;
  int64_t buffer_index;
  int64_t byte_offset;
  int64_t byte_size;
  int64_t required_alignment;
  PeerMemoryKindV3 memory_kind;
};

struct CollectiveStepV3 {
  CollectiveStepKindV3 kind;
  // Must be zero for a barrier and is the function ordinal for a launch.
  int64_t operand;
};

struct CollectiveCallConfigV3 {
  // Number of peer-address entries per region referenced by the compiled
  // context descriptor. Prepare verifies this against the runtime clique
  // before loading `module`.
  int32_t abi_clique_size;
  xla::CollectiveOpGroupMode group_mode;
  int64_t communication_id;
  std::vector<xla::ReplicaGroup> replica_groups;
  ModuleImage module;
  std::vector<PeerRegionV3> peer_regions;
  std::vector<CollectiveStepV3> steps;
};

// Parses and validates a ProtoJSON v3 configuration. Unknown JSON fields are
// ignored, and all string-backed data is copied into the returned config.
absl::StatusOr<CollectiveCallConfigV3> ParseCollectiveCallConfigV3(
    absl::string_view json_config);

}  // namespace xla::gpu::cutedsl

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_COLLECTIVE_CONFIG_H_
