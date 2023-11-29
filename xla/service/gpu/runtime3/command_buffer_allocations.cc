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

#include "xla/service/gpu/runtime3/command_buffer_allocations.h"

#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/runtime3/command_buffer_cmd.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_memory.h"

namespace xla::gpu {

bool CommandBufferAllocations::IsAllocated(
    BufferAllocation::Index index) const {
  return allocs_.find(index) != allocs_.end();
}

StatusOr<se::DeviceMemoryBase> CommandBufferAllocations::GetDeviceAddress(
    BufferAllocation::Index index) const {
  auto base = allocs_.find(index);
  if (base == allocs_.end()) {
    return absl::InternalError(absl::StrCat("Command buffer allocation #",
                                            index, " was not allocated"));
  }
  return allocs_.at(index);
}

Status CommandBufferAllocations::AddAllocation(BufferAllocation::Index index,
                                               se::DeviceMemoryBase memory) {
  VLOG(2) << "Add comand buffer allocation: index=" << index
          << "; ptr=" << memory.opaque() << " to allocation "
          << reinterpret_cast<void*>(this);

  auto emplaced = allocs_.try_emplace(index, std::move(memory));
  if (emplaced.second == false) {
    return absl::InternalError(absl::StrCat("Command buffer allocation #",
                                            index, " was already allocated"));
  }
  return OkStatus();
}

Status CommandBufferAllocations::EraseAllocation(
    BufferAllocation::Index index) {
  VLOG(2) << "Erase comand buffer allocation: index=" << index;

  if (allocs_.erase(index) == 0) {
    return absl::InternalError(absl::StrCat("Command buffer allocation #",
                                            index, " was not allocated"));
  }
  return OkStatus();
}

Status CommandBufferAllocations::Allocate(
    const BufferAllocations& buffer_allocations,
    const ServiceExecutableRunOptions& run_options) {
  if (!alloc_thunk_) {
    VLOG(2)
        << "Constructing command buffer allocations for buffer_allocations: "
        << buffer_allocations.ToString();

    // Add allocate command for allocation marked with kExternalAllocationMarker
    // address.
    CommandBufferCmdSequence commands;
    for (BufferAllocation::Index i = 0; i < buffer_allocations.size(); i++) {
      if (buffer_allocations.IsExternalAllocation(i)) {
        allocations_map_.emplace(
            i, BufferAllocation{/*index=*/i,
                                (int64_t)buffer_allocations.GetAllocationSize(i),
                                /*color=*/0});
        commands.Emplace<AllocateCmd>(allocations_map_.at(i));
        VLOG(2) << "Adding AllocateCmd for allocation: " << i;
      }
    }

    // For allocations that are copied from other allocations, add Copy command.
    if (remapped_allocations_) {
      for (const auto& item : remapped_allocations_.value()) {
        BufferAllocation& dst = allocations_map_.at(item.first);
        BufferAllocation& src = allocations_map_.at(item.second);
        BufferAllocation::Slice dst_slice(&dst, 0, dst.size());
        BufferAllocation::Slice src_slice(&src, 0, src.size());
        commands.Emplace<MemcpyDeviceToDeviceCmd>(dst_slice, src_slice,
                                                  dst_slice.size());
        VLOG(2) << "Adding MemcpyDeviceToDeviceCmd from allocation "
                << src.index() << "to allocation " << dst.index();
      }
    }

    if (commands.size() == 0) {
      return OkStatus();
    }

    alloc_thunk_ = std::move(std::make_unique<CommandBufferThunk>(
        std::move(commands), Thunk::ThunkInfo(nullptr)));
  }

  TF_RETURN_IF_ERROR(
      buffer_allocations.GetMutableExternalAllocations()->Clear());

  VLOG(2) << "Allocating through command buffer for "
          << buffer_allocations.ToString();
  Thunk::ExecuteParams params(run_options, buffer_allocations,
                              run_options.stream(), nullptr, {});
  TF_RETURN_IF_ERROR(alloc_thunk_->ExecuteOnStream(params));

  se::DeviceMemoryAllocator* const memory_allocator = run_options.allocator();
  int device_ordinal = run_options.stream()->parent()->device_ordinal();
 
  for (auto item : allocs_) {
    TF_RETURN_IF_ERROR(
        memory_allocator->NotifyExternalAllocate(device_ordinal, item.second));
  }

  TF_RETURN_IF_ERROR(run_options.stream()->BlockHostUntilDone());
  VLOG(2) << "Allocating results: " << buffer_allocations.ToString()
          << " External allocation size " << Size();
  return OkStatus();
}

Status CommandBufferAllocations::Free(const BufferAllocations& buffer_allocations,
            const absl::flat_hash_set<BufferAllocation::Index>& live_allocation_indexes,
            const ServiceExecutableRunOptions& run_options) {
  se::DeviceMemoryAllocator* const memory_allocator = run_options.allocator();
  int device_ordinal = run_options.stream()->parent()->device_ordinal();

  // Notify memory allocator the release of external allocation.
  for (BufferAllocation::Index i = 0; i < buffer_allocations.size(); i++) {
    if (buffer_allocations.IsExternalAllocation(i) &&
        !live_allocation_indexes.contains(i)) {
      TF_RETURN_IF_ERROR(
          memory_allocator->NotifyExternalFree(device_ordinal, allocs_.at(i)));
    }
  }

  if (!free_thunk_) {
    CommandBufferCmdSequence commands;
    for (BufferAllocation::Index i = 0; i < buffer_allocations.size(); i++) {
      // Only allocations that are allocated through command buffer are assumed
      // to be freed here.
      if (buffer_allocations.IsExternalAllocation(i) &&
          !live_allocation_indexes.contains(i)) {
        CHECK(allocations_map_.find(i) != allocations_map_.end())
            << "CommandBuffer freed region " << i << " is not allocated yet";
        commands.Emplace<FreeCmd>(allocations_map_.at(i));
        VLOG(2) << "Adding FreeCmd for allocation: " << i;
      }
    }

    if (commands.size() == 0) {
      return OkStatus();
    }

    VLOG(2) << "Constructing command buffer free for buffer_allocations: "
            << buffer_allocations.ToString();
    free_thunk_ = std::move(std::make_unique<CommandBufferThunk>(
        std::move(commands), Thunk::ThunkInfo(nullptr)));
  }

  Thunk::ExecuteParams params(run_options, buffer_allocations,
                              run_options.stream(), nullptr, {});
  TF_RETURN_IF_ERROR(free_thunk_->ExecuteOnStream(params));
  TF_RETURN_IF_ERROR(run_options.stream()->BlockHostUntilDone());
  return OkStatus();
}


}  // namespace xla::gpu
