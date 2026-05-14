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

#ifndef XLA_BACKENDS_GPU_RUNTIME_ALL_GATHER_H_
#define XLA_BACKENDS_GPU_RUNTIME_ALL_GATHER_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Encapsulates the information needed to perform an all-gather.
struct AllGatherInfo {
  int64_t num_devices;
  int64_t num_elements;
  PrimitiveType element_type;
};

// Returns absl::OkStatus() if the all-gather kernel is supported for the given
// element type and number of elements, or an error status detailing why it is
// not supported.
absl::Status IsAllGatherKernelSupported(int64_t num_ranks, int64_t num_elements,
                                        PrimitiveType element_type);

// A broader check for all-gather kernel support.
// Returns absl::OkStatus() if supported, or an error status detailing why
// it is not supported.
absl::Status IsAllGatherKernelSupported(
    bool is_collective_kernel_enabled, const se::DeviceDescription& device_info,
    int32_t num_operands, int64_t num_devices, int64_t num_elements,
    PrimitiveType element_type, bool is_local,
    const std::vector<ReplicaGroup>& replica_groups);

// Constructs an AllGatherInfo object for the given all-gather instruction.
// Returns an error status if the all-gather kernel is not supported.
absl::StatusOr<AllGatherInfo> BuildAllGatherInfo(
    bool is_collective_kernel_enabled, const se::DeviceDescription& device_info,
    const HloAllGatherInstruction* all_gather);

// Returns the launch dimensions for the all-gather kernel.
// All-gather uses a fixed one-shot tiling strategy: each rank contributes its
// local slice to the output.
LaunchDimensions AllGatherLaunchDimensions(int64_t elements, int64_t num_ranks);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_ALL_GATHER_H_
