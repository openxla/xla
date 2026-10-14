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

#ifndef XLA_BACKENDS_GPU_RUNTIME_ALL_TO_ALL_H_
#define XLA_BACKENDS_GPU_RUNTIME_ALL_TO_ALL_H_

#include <array>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu_topology.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Element types supported by the Triton all-to-all kernel.
// Complex types, tokens, tuples, and exotic types (e.g. 4-bit integers, 8-bit
// floats) are not supported.
inline constexpr auto kSupportedAllToAllTypes =
    std::array{PRED, F16, BF16, F32, F64, S8, S16, S32, S64, U8, U16, U32, U64};

// Maximum number of GPU thread-blocks launched per all-to-all kernel.
inline constexpr int64_t kAllToAllMaxBlocksPerGrid = 64;

// Maximum total transported data size in bytes across all participating devices
// for the one-shot all-to-all strategy (1-barrier double-buffered scratch
// push). Divided by the number of participating devices when compared against
// per-rank input/output buffer size.
inline constexpr int64_t kMaxOneShotAllToAllSizeBytes =
    32LL * 1024 * 1024;  // 32 MB

// Default threshold for custom all-to-all in bytes of total transported data
// across all participating devices for the collective kernel. Divided by the
// number of participating devices when compared against per-rank input/output
// buffer size.
inline constexpr int64_t kMaxAllToAllSizeBytes = 256LL * 1024 * 1024;  // 256 MB

// Encapsulates the information needed to perform an all-to-all via the Triton
// collective kernel backend.
struct AllToAllInfo {
  int64_t num_devices;
  int64_t num_elements;
  PrimitiveType element_type;
  bool is_one_shot = true;
};

// Returns absl::OkStatus() if the all-to-all kernel is supported for the given
// element type, total per-rank element count, and number of devices, or an
// error status detailing why it is not supported.
absl::Status IsAllToAllKernelSupported(int64_t num_elements,
                                       int64_t num_devices,
                                       PrimitiveType element_type);

// A broader check for all-to-all kernel support that verifies device, operand
// count, replica group, split dimension, and element-type constraints.
// Returns absl::OkStatus() if supported, or an absl::UnimplementedError
// explaining why not.
absl::Status IsAllToAllKernelSupported(
    bool is_collective_kernel_enabled, const se::DeviceDescription& device_info,
    int32_t num_operands, bool has_split_dimension, int64_t num_devices,
    int64_t num_elements, int64_t per_rank_split_dim_size,
    PrimitiveType element_type, bool is_local,
    const std::vector<ReplicaGroup>& replica_groups);

// Constructs an AllToAllInfo object for the given all-to-all instruction.
// Returns absl::UnimplementedError if the Triton all-to-all kernel is not
// supported for this instruction's configuration.
absl::StatusOr<AllToAllInfo> BuildAllToAllInfo(
    bool is_collective_kernel_enabled, const GpuTopology& gpu_topology,
    const HloAllToAllInstruction* all_to_all,
    const DeviceAssignment* device_assignment);

// Returns the launch dimensions for the all-to-all kernel.
// All-to-all uses a one-shot strategy: each rank stages its local slices into
// its symmetric buffer and then pulls its destined slice from each peer.
LaunchDimensions AllToAllLaunchDimensions(
    int64_t elements, int64_t num_ranks,
    const se::DeviceDescription& device_info);

// Creates a CollectiveKernelSpec describing the resource requirements of a
// Triton all-to-all kernel. The kernel argument layout matches all-gather:
//   [0] local input buffer (kInputBuffer, index 0)
//   [1] output buffer (kOutputBuffer, index 0)
//   [2] runtime rank  (kRuntimeRank)
//   [3] invocation count (kInvocationCount)
//   [4] signal flags (kScratchBuffer, index 0)
//   [5] remote scratch buffer pointer table (kScratchBuffer, index 1)
// `scratch_memory_type` is used for both scratch buffers.
absl::StatusOr<CollectiveKernelSpec> CreateAllToAllKernelSpec(
    const HloInstruction* instr, const LaunchDimensions& launch_dimensions,
    SymmetricMemoryType scratch_memory_type);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_ALL_TO_ALL_H_
