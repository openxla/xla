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

#ifndef XLA_STREAM_EXECUTOR_INTEGRATIONS_CONTIGUOUS_SUB_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_INTEGRATIONS_CONTIGUOUS_SUB_ALLOCATOR_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/tsl/framework/allocator.h"

namespace stream_executor {

// Appends physical allocations to one reserved virtual address range. Existing
// mappings never move or change backing memory. Intended for BFC with region
// coalescing enabled and garbage collection disabled. Calls must be serialized.
class ContiguousSubAllocator : public tsl::SubAllocator {
 public:
  using Allocate =
      absl::AnyInvocable<absl::StatusOr<std::unique_ptr<MemoryAllocation>>(
          size_t)>;

  ContiguousSubAllocator(std::unique_ptr<MemoryReservation> reservation,
                         Allocate allocate, size_t granularity,
                         int device_ordinal,
                         const std::vector<Visitor>& alloc_visitors,
                         const std::vector<Visitor>& free_visitors);
  ~ContiguousSubAllocator() override;

  // Regions are aligned to granularity(), not necessarily `alignment`. BFC
  // handles larger client alignments by splitting chunks within these regions.
  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override;
  // Releases a suffix comprising whole mappings, or the entire coalesced arena.
  void Free(void* ptr, size_t num_bytes) override;
  bool SupportsCoalescing() const override { return true; }
  tsl::AllocatorMemoryType GetMemoryType() const override {
    return tsl::AllocatorMemoryType::kDevice;
  }
  size_t granularity() const { return granularity_; }
  size_t capacity() const { return reservation_->address().size(); }

 private:
  struct Region {
    // The mapping must be destroyed before its physical allocation.
    std::unique_ptr<MemoryAllocation> allocation;
    MemoryReservation::ScopedMapping mapping;
  };
  // The reservation must outlive all mappings.
  std::unique_ptr<MemoryReservation> reservation_;
  Allocate allocate_;
  size_t granularity_;
  int device_ordinal_;
  size_t mapped_bytes_ = 0;
  std::vector<Region> regions_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_INTEGRATIONS_CONTIGUOUS_SUB_ALLOCATOR_H_
