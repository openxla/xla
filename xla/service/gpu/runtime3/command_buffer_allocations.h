/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_RUNTIME3_COMMAND_BUFFER_ALLOCATIONS_H_
#define XLA_SERVICE_GPU_RUNTIME3_COMMAND_BUFFER_ALLOCATIONS_H_

#include <sys/types.h>
#include "absl/container/flat_hash_map.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/runtime3/command_buffer_thunk.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_memory.h"

namespace xla::gpu {

// Command buffer allocations tracks external buffer allocations done via the
// CommandBuffer API and owned by the XLA executable (via instantiated command
// buffers and memory allocation Gpu graph nodes).
class CommandBufferAllocations : public BufferAllocations::ExternalAllocations {
 public:

  // key is the allocation index where it has remapped (copied), and the value
  // is the index where we put the original allocation in the buffers_ list.
  using RemappedAllocations =
      std::map<BufferAllocation::Index, BufferAllocation::Index>;

  StatusOr<se::DeviceMemoryBase> GetDeviceAddress(
      BufferAllocation::Index index) const override;

  // Adds an external allocation for a given buffer index. Returns error if
  // allocation already exists.
  Status AddAllocation(BufferAllocation::Index index,
                       se::DeviceMemoryBase memory) override;

  bool IsAllocated(BufferAllocation::Index index) const override;

  // Erases an external allocation for a given buffer index. Returns error if
  // allocation does not exists.
  Status EraseAllocation(BufferAllocation::Index index) override;

  // Given a BufferAllocation object filled with external allocation marker,
  // this function constructs the command buffer thunk to do the allocation.
  Status Allocate(const BufferAllocations& buffer_allocations,
                  const ServiceExecutableRunOptions& run_options);

  // Free the allocations that was previously allocated with command buffer
  // thunk, keep the allocations in set `live_allocation_indexes`.
  Status Free(const BufferAllocations& buffer_allocations,
              const absl::flat_hash_set<BufferAllocation::Index>&
                  live_allocation_indexes,
              const ServiceExecutableRunOptions& run_optionsconst);

  // Clear all previous allocations.
  Status Clear() override {
    allocs_.clear();
    return OkStatus();
  };

  // When running a module through GpuExecutable::ExecuteAsyncOnStreamImpl,
  // there are cases that we need to copy the parameter allocation into a new
  // allocation(e.g. the XLA module was compiled with input/output buffer
  // alised, but the user does not donate the input buffer, or we want to copy
  // the parameter allocation to some fix memory address to get stable input
  // memory pointer). To enable copying between original parameter allocation
  // and copied allocation through command buffer, we need to include both
  // the original and copied allocations in the BufferAllocations list.

  // The original allocation's address is added to the end of BufferAllocations,
  // and the copied allocation takes the index of the original allocation.
  // RemappedAllocations describes where the copied allocation is copied from
  // which allocation in the BufferAllocations.
  void SetRemappedAllocations(const RemappedAllocations& remapped_allocation) {
    remapped_allocations_ = remapped_allocation;
  };

  uint Size() {
    return allocs_.size();
  };

 private:
  absl::flat_hash_map<BufferAllocation::Index, se::DeviceMemoryBase> allocs_;
  std::unique_ptr<CommandBufferThunk> alloc_thunk_;
  std::unique_ptr<CommandBufferThunk> free_thunk_;
  std::optional<RemappedAllocations> remapped_allocations_ = std::nullopt;
  absl::flat_hash_map<BufferAllocation::Index, BufferAllocation> allocations_map_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME3_COMMAND_BUFFER_ALLOCATIONS_H_
