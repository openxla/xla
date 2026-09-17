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

#include "xla/stream_executor/integrations/contiguous_sub_allocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/tsl/framework/allocator.h"

namespace stream_executor {

ContiguousSubAllocator::ContiguousSubAllocator(
    std::unique_ptr<MemoryReservation> reservation, Allocate allocate,
    size_t granularity, int device_ordinal,
    const std::vector<Visitor>& alloc_visitors,
    const std::vector<Visitor>& free_visitors)
    : SubAllocator(alloc_visitors, free_visitors),
      reservation_(std::move(reservation)),
      allocate_(std::move(allocate)),
      granularity_(granularity),
      device_ordinal_(device_ordinal) {
  CHECK_GT(granularity_, 0);
  CHECK_EQ(capacity() % granularity_, 0);
  CHECK_EQ(reinterpret_cast<uintptr_t>(reservation_->address().opaque()) %
               granularity_,
           0);
}

ContiguousSubAllocator::~ContiguousSubAllocator() {
  if (mapped_bytes_ != 0) Free(reservation_->address().opaque(), mapped_bytes_);
}

void* ContiguousSubAllocator::Alloc(size_t alignment, size_t num_bytes,
                                    size_t* bytes_received) {
  *bytes_received = 0;
  const size_t available = capacity() - mapped_bytes_;
  if (num_bytes == 0 || num_bytes > available) return nullptr;
  // available is granularity-aligned, so rounding cannot overflow or exceed it.
  const size_t bytes = ((num_bytes - 1) / granularity_ + 1) * granularity_;
  auto allocation = allocate_(bytes);
  if (!allocation.ok()) {
    VLOG(1) << "Unable to back contiguous arena extension: "
            << allocation.status();
    return nullptr;
  }
  if ((*allocation)->address().size() != bytes) {
    LOG(ERROR) << "Physical allocation size does not match mapping granularity";
    return nullptr;
  }
  auto mapping = reservation_->MapTo(mapped_bytes_, 0, bytes, **allocation);
  if (!mapping.ok()) {
    VLOG(1) << "Unable to map contiguous arena extension: " << mapping.status();
    return nullptr;
  }
  void* ptr = mapping->mapped_address().opaque();
  regions_.push_back({std::move(*allocation), std::move(*mapping)});
  mapped_bytes_ += bytes;
  *bytes_received = bytes;
  VisitAlloc(ptr, device_ordinal_, bytes);
  return ptr;
}

void ContiguousSubAllocator::Free(void* ptr, size_t num_bytes) {
  const uintptr_t base =
      reinterpret_cast<uintptr_t>(reservation_->address().opaque());
  const uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
  CHECK_GE(start, base);
  CHECK_EQ(start - base + num_bytes, mapped_bytes_);
  // Visit each original backing allocation, matching the allocation callbacks
  // even when BFC returns the whole coalesced region at shutdown.
  while (num_bytes != 0) {
    CHECK(!regions_.empty());
    DeviceAddressBase address = regions_.back().mapping.mapped_address();
    CHECK_LE(address.size(), num_bytes);
    VisitFree(address.opaque(), device_ordinal_, address.size());
    num_bytes -= address.size();
    mapped_bytes_ -= address.size();
    regions_.pop_back();
  }
}

}  // namespace stream_executor
