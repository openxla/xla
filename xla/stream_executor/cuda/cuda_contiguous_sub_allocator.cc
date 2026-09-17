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

#include "xla/stream_executor/cuda/cuda_contiguous_sub_allocator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_device_allocator.h"
#include "xla/stream_executor/cuda/cuda_memory_reservation.h"
#include "xla/stream_executor/cuda/cuda_raw_memory_allocation.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/integrations/contiguous_sub_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"

namespace stream_executor::gpu {

absl::StatusOr<std::unique_ptr<ContiguousSubAllocator>>
CreateCudaContiguousSubAllocator(
    StreamExecutor* executor, size_t capacity, size_t headroom,
    const std::vector<tsl::SubAllocator::Visitor>& alloc_visitors,
    const std::vector<tsl::SubAllocator::Visitor>& free_visitors) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  CUdevice device;
  ABSL_RETURN_IF_ERROR(
      cuda::ToStatus(cuDeviceGet(&device, executor->device_ordinal())));
  ABSL_ASSIGN_OR_RETURN(auto options, QueryDeviceAllocatorOptions(device));
  CUmemAllocationProp props = BuildVmmAllocationProp(device, options);
  size_t granularity;
  ABSL_RETURN_IF_ERROR(cuda::ToStatus(cuMemGetAllocationGranularity(
      &granularity, &props, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)));
  ABSL_ASSIGN_OR_RETURN(uint64_t collective_alignment,
                        executor->GetCollectiveMemoryGranularity());
  granularity =
      std::max(granularity, static_cast<size_t>(collective_alignment));
  capacity = capacity / granularity * granularity;
  if (capacity == 0)
    return absl::InvalidArgumentError(
        "VMM capacity is below mapping granularity");
  ABSL_ASSIGN_OR_RETURN(auto reservation, CudaMemoryReservation::Create(
                                              executor, capacity, granularity));
  auto allocate =
      [executor, headroom](
          size_t bytes) -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
    int64_t free_bytes, total_bytes;
    if (!executor->DeviceMemoryUsage(&free_bytes, &total_bytes)) {
      return absl::UnavailableError(
          "Cannot query device memory before BFC growth");
    }
    if (free_bytes < 0 || static_cast<uint64_t>(free_bytes) < headroom ||
        bytes > static_cast<uint64_t>(free_bytes) - headroom) {
      return absl::ResourceExhaustedError(
          "BFC extension would consume device memory headroom");
    }
    return CudaRawMemoryAllocation::Create(executor, bytes);
  };
  return std::make_unique<ContiguousSubAllocator>(
      std::move(reservation), std::move(allocate), granularity,
      executor->device_ordinal(), alloc_visitors, free_visitors);
}

}  // namespace stream_executor::gpu
