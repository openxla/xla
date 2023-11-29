/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/buffer_allocations.h"

#include <cstdint>
#include <set>

#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {

Status BufferAllocations::TearDown(
    const absl::flat_hash_set<BufferAllocation::Index>& live_allocations,
    absl::Span<const BufferAllocation> allocations) {
  // Deallocate temporary buffers, taking care to try to deallocate all of them
  // even if one of the deallocations fails.
  Status status;
  const int64_t num_buffers = allocations.size();
  for (BufferAllocation::Index i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = allocations[i];
    // Deallocate buffers marked "maybe_live_out" but aren't actually live out,
    // and temp buffers. For external allocations, the free should performed by
    // external allocator, so skip external allocations free here.
    if (((allocation.maybe_live_out() &&
          !live_allocations.count(allocation.index())) ||
         allocation.IsPreallocatedTempBuffer()) &&
        !IsExternalAllocation(i)) {
      se::DeviceMemoryBase buffer_address =
          GetDeviceAddress(allocation.index());
      auto dealloc_result =
          memory_allocator_->Deallocate(device_ordinal_, buffer_address);
      if (!dealloc_result.ok() && status.ok()) {
        status = dealloc_result;
      }
    }
  }
  return status;
}

bool BufferAllocations::IsExternalAllocation(
    BufferAllocation::Index index) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, buffers_.size());
  return reinterpret_cast<uintptr_t>(buffers_[index].opaque()) ==
         kExternalAllocationMarker;
};

se::DeviceMemoryBase BufferAllocations::GetDeviceAddress(
    BufferAllocation::Index buffer_index) const {
  CHECK_GE(buffer_index, 0);
  CHECK_LT(buffer_index, buffers_.size());
  se::DeviceMemoryBase base = buffers_[buffer_index];
  if (reinterpret_cast<uintptr_t>(base.opaque()) == kExternalAllocationMarker) {
    if (!external_allocations_) {
      LOG(ERROR) << "Does not have external allocations for buffer "
                 << buffer_index;
      return se::DeviceMemoryBase();
    }

    if (!external_allocations_->IsAllocated(buffer_index)) {
        return base;
    }

    VLOG(2) << "GetDeviceAddress from external allocations "
            << reinterpret_cast<void*>(external_allocations_);

    auto external_address =
        external_allocations_->GetDeviceAddress(buffer_index);
    if (external_address.ok()) {
      return external_address.value();
    }
    LOG(ERROR) << "External address for allocation" << buffer_index
               << " is not allocated yet";
    return se::DeviceMemoryBase();
  }
  return base;
}

se::DeviceMemoryBase& BufferAllocations::GetMutableDeviceAddress(
    BufferAllocation::Index buffer_index) {
  CHECK_GE(buffer_index, 0);
  CHECK_LT(buffer_index, buffers_.size());
  return buffers_[buffer_index];
}

uint64_t BufferAllocations::GetAllocationSize(
    BufferAllocation::Index buffer_index) const {
  CHECK_GE(buffer_index, 0);
  CHECK_LT(buffer_index, buffers_.size());
  return buffers_[buffer_index].size();
}

se::DeviceMemoryBase BufferAllocations::GetDeviceAddress(
    const BufferAllocation::Slice& buffer_slice) const {
  int64_t index = buffer_slice.index();
  se::DeviceMemoryBase base = GetDeviceAddress(index);

  int64_t offset = buffer_slice.offset();
  CHECK_LE(buffer_slice.offset(), base.size())
      << "slice offset " << offset << " must be smaller than buffer #" << index
      << " size " << base.size();

  int64_t extent = offset + buffer_slice.size();
  CHECK_LE(extent, base.size())
      << "slice extent " << extent << " must be smaller than buffer #" << index
      << " size " << base.size();

  return base.GetByteSlice(buffer_slice.offset(), buffer_slice.size());
}

Status BufferAllocations::AddExternalAllocation(
    BufferAllocation::Index index, se::DeviceMemoryBase memory) const {
  if (external_allocations_ == nullptr) {
    return InternalError(
        "Calling external allocations, but no allocation tracker is provided"
        "for allocation %d",
        index);
  }
  return external_allocations_->AddAllocation(index, memory);
}

Status BufferAllocations::EraseExternalAllocation(
    BufferAllocation::Index index) const {
  if (external_allocations_ == nullptr) {
    return InternalError(
        "Calling external allocations, but no allocation tracker is provided"
        "for allocation %d",
        index);
  }
  return external_allocations_->EraseAllocation(index);
}

}  // namespace gpu
}  // namespace xla
