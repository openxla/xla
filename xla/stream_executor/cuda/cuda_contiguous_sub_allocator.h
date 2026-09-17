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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_CONTIGUOUS_SUB_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_CONTIGUOUS_SUB_ALLOCATOR_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/stream_executor/integrations/contiguous_sub_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"

namespace stream_executor::gpu {

// Reserves up to capacity bytes, rounded down to CUDA's mapping granularity.
// Each physical allocation leaves at least headroom bytes of currently free
// device memory. This is a best-effort allowance for allocations outside BFC,
// not a reservation against other users of the device.
absl::StatusOr<std::unique_ptr<ContiguousSubAllocator>>
CreateCudaContiguousSubAllocator(
    StreamExecutor* executor, size_t capacity, size_t headroom,
    const std::vector<tsl::SubAllocator::Visitor>& alloc_visitors = {},
    const std::vector<tsl::SubAllocator::Visitor>& free_visitors = {});

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_CONTIGUOUS_SUB_ALLOCATOR_H_
