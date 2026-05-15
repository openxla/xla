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

#include "xla/backends/gpu/runtime/all_gather.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/bit.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

absl::Status IsAllGatherKernelSupported(int64_t num_ranks, int64_t num_elements,
                                        PrimitiveType element_type) {
  // Unsigned integer types are not supported by the Triton all-gather kernel.
  if (element_type == U32 || element_type == U16 || element_type == U64) {
    return absl::UnimplementedError(
        absl::StrCat("Element type (",
                     primitive_util::LowercasePrimitiveTypeName(element_type),
                     ") is not supported for all-gather kernel."));
  }
  // The number of elements must be aligned to kNumElementsPerThread.
  if (num_elements % se::gpu::kNumElementsPerThread != 0) {
    return absl::UnimplementedError(
        absl::StrCat("Number of elements (", num_elements,
                     ") is not aligned to the alignment requirement (",
                     se::gpu::kNumElementsPerThread, ")."));
  }
  return absl::OkStatus();
}

absl::Status IsAllGatherKernelSupported(
    bool is_collective_kernel_enabled, const se::DeviceDescription& device_info,
    int32_t num_operands, int64_t num_devices, int64_t num_elements,
    PrimitiveType element_type, bool is_local,
    const std::vector<ReplicaGroup>& replica_groups) {
  if (!is_collective_kernel_enabled) {
    return absl::UnimplementedError("Collective kernel is not enabled.");
  }
  // Check if the device supports Triton collective codegen:
  // CUDA: Requires compute capability 9.0+ (Hopper or newer)
  // ROCm: All versions with Triton support are enabled
  if (!device_info.cuda_compute_capability().IsAtLeastHopper() &&
      !device_info.gpu_compute_capability().IsRocm()) {
    return absl::UnimplementedError(absl::StrCat(
        "Triton collective codegen requires CUDA compute capability >= 9.0 "
        "(Hopper or newer) or a ROCm device with Triton support. "
        "Got: ",
        device_info.gpu_compute_capability().ToString(), "."));
  }
  // TODO(b/383125489): Support variadic arguments.
  if (num_operands != 1) {
    return absl::UnimplementedError(
        absl::StrCat("Collective kernel is not supported for number of "
                     "operands not equal to 1. Got ",
                     num_operands, "."));
  }
  if (replica_groups.empty()) {
    return absl::UnimplementedError(
        "Replica groups must be explicitly provided for collective kernels.");
  }
  if (!is_local) {
    return absl::UnimplementedError(
        "Cross-host symmetric memory collectives are not supported.");
  }
  if (!llvm::has_single_bit(static_cast<uint64_t>(num_devices))) {
    return absl::UnimplementedError(
        absl::StrCat("Collective kernels are only supported for power of 2 "
                     "number of devices. Got ",
                     num_devices, "."));
  }
  return IsAllGatherKernelSupported(num_devices, num_elements, element_type);
}

absl::StatusOr<AllGatherInfo> BuildAllGatherInfo(
    bool is_collective_kernel_enabled, const se::DeviceDescription& device_info,
    const HloAllGatherInstruction* all_gather) {
  if (!all_gather->device_list()) {
    return absl::UnimplementedError(
        "Replica groups must be explicitly provided for collective kernels.");
  }
  const int64_t num_devices =
      all_gather->device_list()->num_devices_per_group();
  const int64_t num_elements =
      ShapeUtil::ElementsIn(all_gather->operand(0)->shape());
  const PrimitiveType element_type =
      all_gather->operand(0)->shape().element_type();
  const int32_t num_operands = all_gather->operand_count();
  TF_RETURN_IF_ERROR(IsAllGatherKernelSupported(
      is_collective_kernel_enabled, device_info, num_operands, num_devices,
      num_elements, element_type, /*is_local=*/true,
      all_gather->replica_groups()));
  return AllGatherInfo{
      /*.num_devices =*/num_devices,
      /*.num_elements =*/num_elements,
      /*.element_type =*/element_type,
  };
}

namespace {
// All-gather always uses a one-shot strategy: every rank reads its peers'
// slices directly, so the work is proportional to the full element count
// with no rank-based partitioning.
static constexpr int64_t kMaxBlocksPerGrid = 32;
static constexpr uint64_t kMaxThreadsPerBlock = 512;
static constexpr int64_t kWarpSize = 32;
}  // namespace

LaunchDimensions AllGatherLaunchDimensions(int64_t elements,
                                           int64_t num_ranks) {
  // Maximum number of threads such that each thread has elements to process.
  const int64_t total_threads = RoundUpTo(
      CeilOfRatio(elements, se::gpu::kNumElementsPerThread), kWarpSize);
  // Triton expects power of 2 for threads_per_block / threads_per_warp.
  const int64_t threads_per_block =
      std::min(kMaxThreadsPerBlock,
               llvm::bit_ceil(static_cast<uint64_t>(total_threads)));
  const int64_t blocks_per_grid = std::min(
      kMaxBlocksPerGrid, CeilOfRatio(total_threads, threads_per_block));
  return LaunchDimensions(blocks_per_grid, threads_per_block);
}

}  // namespace xla::gpu
