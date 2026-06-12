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

#include "xla/stream_executor/device_address_vmm_allocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/service/computation_placer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {

namespace {
thread_local const xla::DeviceAssignment* current_device_assignment = nullptr;
}  // namespace

DeviceAddressVmmAllocator::DeviceAssignmentScope::DeviceAssignmentScope(
    const xla::DeviceAssignment* device_assignment)
    : previous_(current_device_assignment) {
  current_device_assignment = device_assignment;
}

DeviceAddressVmmAllocator::DeviceAssignmentScope::~DeviceAssignmentScope() {
  current_device_assignment = previous_;
}

bool DeviceAddressVmmAllocator::CurrentMultiDevice() {
  const xla::DeviceAssignment* device_assignment = current_device_assignment;
  return device_assignment != nullptr &&
         device_assignment->replica_count() *
                 device_assignment->computation_count() >
             1;
}

// Interval between CPU polls of the GPU-written deallocation timeline while
// waiting for deferred frees to become safe. The 50us value is a conservative
// initial tradeoff: long enough to avoid busy-spinning a CPU core and short
// enough to keep forced allocator synchronization responsive; it has not been
// benchmark-tuned, so workload-specific tests could refine it if this wait
// shows up in profiles.
static constexpr absl::Duration kGpuTimelinePollInterval =
    absl::Microseconds(50);

// Returns the completed timeline value from pinned host memory using an
// acquire load, so all GPU writes prior to this value are visible.
// Uses __atomic_load_n rather than std::atomic<> because the pointer is
// volatile (GPU-written pinned memory) and reinterpret_cast to
// std::atomic<uint64_t>* would discard the volatile qualifier.
static uint64_t LoadTimeline(const volatile uint64_t* pinned_timeline) {
  return __atomic_load_n(pinned_timeline, __ATOMIC_ACQUIRE);
}

DeviceAddressVmmAllocator::DeviceAddressVmmAllocator(const Platform* platform)
    : DeviceAddressAllocator(platform) {}

absl::Status DeviceAddressVmmAllocator::PopulateDevices(
    DeviceAddressVmmAllocator* allocator,
    absl::Span<const DeviceConfig> devices) {
  absl::flat_hash_set<int> seen_ordinals;
  for (const DeviceConfig& cfg : devices) {
    DCHECK_NE(cfg.executor, nullptr);
    DCHECK_NE(cfg.stream, nullptr);
    int ordinal = cfg.executor->device_ordinal();
    DCHECK(seen_ordinals.insert(ordinal).second)
        << "Duplicate device ordinal: " << ordinal;
  }

  for (const DeviceConfig& cfg : devices) {
    int ordinal = cfg.executor->device_ordinal();

    auto state = std::make_unique<PerDeviceState>();
    state->executor = cfg.executor;
    state->stream = cfg.stream;
    state->pa_budget = cfg.pa_budget;

    RETURN_IF_ERROR(allocator->InitializeDeviceState(*state));

    VLOG(3) << "DeviceAddressVmmAllocator: registering device " << ordinal
            << " with pa_budget " << cfg.pa_budget;
    allocator->per_device_.emplace(ordinal, std::move(state));
  }

  return absl::OkStatus();
}

DeviceAddressVmmAllocator::~DeviceAddressVmmAllocator() {
  absl::Status status = SynchronizeAllPendingOperations();
  CHECK(status.ok()) << status;

  for (auto& [ordinal, state] : per_device_) {
    // Free platform-specific per-device resources (e.g. pinned timeline).
    if (state->destroy_fn) {
      state->destroy_fn();
    }
  }
}

absl::Status DeviceAddressVmmAllocator::SynchronizeAllPendingOperations() {
  for (auto& [ordinal, state] : per_device_) {
    RETURN_IF_ERROR(SynchronizePendingOperations(ordinal));
  }
  return absl::OkStatus();
}

absl::StatusOr<DeviceAddressVmmAllocator::PerDeviceState*>
DeviceAddressVmmAllocator::GetPerDeviceState(int device_ordinal) const {
  auto it = per_device_.find(device_ordinal);
  if (it == per_device_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("No device with ordinal %d registered in "
                        "DeviceAddressVmmAllocator",
                        device_ordinal));
  }
  return it->second.get();
}

absl::StatusOr<DeviceAddressBase>
DeviceAddressVmmAllocator::ValidateReservationRange(
    MemoryReservation* reservation, uint64_t reservation_offset,
    uint64_t size) const {
  if (reservation == nullptr) {
    return absl::InvalidArgumentError("reservation must not be null");
  }

  DeviceAddressBase address = reservation->address();
  if (reservation_offset > address.size() ||
      size > address.size() - reservation_offset) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "reservation range is out of bounds: offset=%uB, size=%uB, "
        "reservation_size=%uB",
        reservation_offset, size, address.size()));
  }

  return address.GetByteSlice(reservation_offset, size);
}

void DeviceAddressVmmAllocator::ProcessCompletedPendingDeallocations(
    PerDeviceState& state) {
  // Single atomic read covers all entries whose seqno is <= completed.
  uint64_t completed = LoadTimeline(state.pinned_timeline);
  while (!state.pending_deallocations.empty()) {
    if (state.pending_deallocations.front().seqno > completed) {
      break;
    }
    if (state.pending_deallocations.front().kind !=
        PendingDeallocationKind::kMap) {
      DoDeallocate(state, state.pending_deallocations.front().addr);
    } else {
      DoUnMap(state, state.pending_deallocations.front().addr);
    }
    state.pending_deallocations.pop_front();
  }
}

void DeviceAddressVmmAllocator::WaitPendingDeallocationsToComplete(
    PerDeviceState& state, uint64_t size) {
  if (state.pending_deallocations.empty()) {
    return;
  }

  uint64_t accumulated_size = 0;
  size_t count_to_wait = 0;
  uint64_t rounded_size = RoundUpToGranularity(state, size);
  uint64_t target_seqno = 0;

  // Target 1.1x the requested size to provide some headroom.
  uint64_t target_size = rounded_size + rounded_size / 10;

  for (const auto& pending : state.pending_deallocations) {
    if (pending.kind != PendingDeallocationKind::kMap) {
      accumulated_size += pending.reclaimable_bytes;
    }
    target_seqno = pending.seqno;
    ++count_to_wait;
    if (accumulated_size >= target_size) {
      break;
    }
  }

  // Move selected entries out of the deque while holding the lock, so no
  // other thread can observe or free them.
  std::vector<PendingDeallocation> selected;
  selected.reserve(count_to_wait);
  for (size_t i = 0; i < count_to_wait; ++i) {
    selected.push_back(std::move(state.pending_deallocations.front()));
    state.pending_deallocations.pop_front();
  }

  // Release the lock before spin-waiting to avoid stalling other threads for
  // potentially milliseconds while the GPU drains its work queue.
  state.mu.unlock();

  // Poll until the GPU writes a timeline value >= target_seqno.
  // Since timeline values are written in stream order, this guarantees all
  // earlier pending deallocations have also completed.
  while (LoadTimeline(state.pinned_timeline) < target_seqno) {
    absl::SleepFor(kGpuTimelinePollInterval);
  }

  // Reacquire the lock before modifying the maps.
  state.mu.lock();

  for (auto& item : selected) {
    if (item.kind != PendingDeallocationKind::kMap) {
      DoDeallocate(state, item.addr);
    } else {
      DoUnMap(state, item.addr);
    }
  }
}

void DeviceAddressVmmAllocator::DoDeallocate(PerDeviceState& state,
                                             DeviceAddressBase mem) {
  VLOG(3) << absl::StreamFormat(
      "Actually freeing virtual address %p (size=%uB) on device ordinal %d",
      mem.opaque(), mem.size(), state.executor->device_ordinal());

  auto record_it = state.records_by_allocator_address.find(mem.opaque());
  CHECK(record_it != state.records_by_allocator_address.end());
  CHECK(record_it->second->allocator_stale);
  CHECK(record_it->second->allocator_address.IsSameAs(mem));
  AllocationRecord& record = *record_it->second;
  CHECK(!record.allocator_active);
  record.allocator_address_mapping.reset();
  record.allocator_address_reservation.reset();

  if (record.raw_allocation != nullptr) {
    uint64_t rounded_size =
        RoundUpToGranularity(state, record.raw_allocation->address().size());
    DCHECK_GE(state.pa_allocated, rounded_size);
    state.pa_allocated -= rounded_size;
  }
  record.raw_allocation.reset();
  CHECK_EQ(state.records_by_allocator_address.erase(mem.opaque()), 1);
}

void DeviceAddressVmmAllocator::DoUnMap(PerDeviceState& state,
                                        DeviceAddressBase mem) {
  VLOG(3) << absl::StreamFormat(
      "Actually unmapping reservation address %p (size=%uB) on device ordinal "
      "%d",
      mem.opaque(), mem.size(), state.executor->device_ordinal());
  state.stale_reservation_mappings.erase(mem.opaque());
}

void* DeviceAddressVmmAllocator::TrackAllocatorAddressMappedAllocation(
    PerDeviceState& state, PendingDeallocationKind kind,
    DeviceAddressBase allocator_address,
    std::shared_ptr<MemoryAllocation> raw_allocation,
    std::unique_ptr<MemoryReservation> reservation,
    MemoryReservation::ScopedMapping mapping, uint64_t allocated_size,
    bool multi_device) {
  void* va_ptr = allocator_address.opaque();
  auto record = std::make_unique<AllocationRecord>();
  record->kind = kind;
  record->allocator_address = allocator_address;
  record->raw_allocation = std::move(raw_allocation);
  record->multi_device = multi_device;
  record->allocator_address_reservation = std::move(reservation);
  record->allocator_address_mapping.emplace(std::move(mapping));
  record->allocator_active = true;
  auto insert_result =
      state.records_by_allocator_address.emplace(va_ptr, std::move(record));
  CHECK(insert_result.second);
  state.pa_allocated += allocated_size;
  return va_ptr;
}

absl::StatusOr<DeviceAddressBase> DeviceAddressVmmAllocator::AllocateWithBudget(
    PerDeviceState& state, uint64_t size, bool multi_device) {
  uint64_t rounded_size = RoundUpToGranularity(state, size);
  if (state.pa_allocated + rounded_size > state.pa_budget) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Not enough PA budget for allocation: pa_allocated=%uB, "
        "rounded_size=%uB, pa_budget=%uB",
        state.pa_allocated, rounded_size, state.pa_budget));
  }

  // Create physical memory allocation (e.g. cuMemCreate).
  ASSIGN_OR_RETURN(auto raw_alloc, CreateAllocation(state.executor, size));
  const uint64_t padded_size = raw_alloc->address().size();

  // Reserve virtual address range (e.g. cuMemAddressReserve).
  ASSIGN_OR_RETURN(auto reservation, CreateReservation(state.executor, size));

  // Map physical memory into the virtual address range and enable access.
  ASSIGN_OR_RETURN(
      auto scoped_mapping,
      reservation->MapTo(/*reservation_offset=*/0, /*allocation_offset=*/0,
                         padded_size, *raw_alloc));

  auto shared_raw = std::shared_ptr<MemoryAllocation>(std::move(raw_alloc));
  DeviceAddressBase allocator_address(reservation->address().opaque(), size);
  void* va_ptr = TrackAllocatorAddressMappedAllocation(
      state, PendingDeallocationKind::kAllocate, allocator_address,
      std::move(shared_raw), std::move(reservation), std::move(scoped_mapping),
      rounded_size, multi_device);
  // Return the original requested size, not the padded size.
  return DeviceAddressBase(va_ptr, size);
}

absl::StatusOr<ScopedDeviceAddress<uint8_t>>
DeviceAddressVmmAllocator::Allocate(
    int device_ordinal, uint64_t allocation_size, bool /*retry_on_failure*/,
    int64_t /*memory_space*/, MemoryReservation* reservation,
    uint64_t reservation_offset, uint64_t mapping_size,
    bool return_reservation_address) {
  if (allocation_size != mapping_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "VMM mapped allocation size (%u) must equal mapping size (%u)",
        allocation_size, mapping_size));
  }
  if (allocation_size == 0) {
    return ScopedDeviceAddress<uint8_t>(DeviceAddressBase(), device_ordinal,
                                        this);
  }
  ASSIGN_OR_RETURN(
      DeviceAddressBase reservation_address,
      ValidateReservationRange(reservation, reservation_offset, mapping_size));

  ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));

  const bool multi_device = CurrentMultiDevice();

  absl::MutexLock lock(state->mu);
  if (state->active_reservation_mappings.contains(
          reservation_address.opaque()) ||
      state->stale_reservation_mappings.contains(
          reservation_address.opaque()) ||
      state->records_by_allocator_address.contains(
          reservation_address.opaque())) {
    return absl::FailedPreconditionError(
        "Reservation address is already tracked by this allocator");
  }

  uint64_t rounded_size = RoundUpToGranularity(*state, allocation_size);
  if (state->pa_allocated + rounded_size > state->pa_budget) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Not enough PA budget for allocation: pa_allocated=%uB, "
        "rounded_size=%uB, pa_budget=%uB",
        state->pa_allocated, rounded_size, state->pa_budget));
  }

  ASSIGN_OR_RETURN(auto raw_alloc,
                   CreateAllocation(state->executor, allocation_size));
  const uint64_t padded_size = raw_alloc->address().size();
  if (mapping_size > padded_size) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Mapping size %u exceeds raw allocation size %u",
                        mapping_size, padded_size));
  }

  ASSIGN_OR_RETURN(
      MemoryReservation::ScopedMapping reservation_mapping,
      reservation->MapTo(reservation_offset, /*allocation_offset=*/0,
                         mapping_size, *raw_alloc));
  auto shared_raw = std::shared_ptr<MemoryAllocation>(std::move(raw_alloc));

  if (return_reservation_address) {
    TrackAllocatorAddressMappedAllocation(
        *state, PendingDeallocationKind::kAllocateAndMapReturnMapAddr,
        reservation_address, std::move(shared_raw), nullptr,
        std::move(reservation_mapping), rounded_size, multi_device);
    return ScopedDeviceAddress<uint8_t>(reservation_address, device_ordinal,
                                        this);
  }

  ASSIGN_OR_RETURN(auto allocator_reservation,
                   CreateReservation(state->executor, allocation_size));
  ASSIGN_OR_RETURN(auto allocator_mapping,
                   allocator_reservation->MapTo(
                       /*reservation_offset=*/0, /*allocation_offset=*/0,
                       padded_size, *shared_raw));
  DeviceAddressBase allocator_address(allocator_reservation->address().opaque(),
                                      allocation_size);
  TrackAllocatorAddressMappedAllocation(
      *state, PendingDeallocationKind::kAllocateAndMapReturnNewAddr,
      allocator_address, std::move(shared_raw), std::move(allocator_reservation),
      std::move(allocator_mapping), rounded_size, multi_device);
  state->active_reservation_mappings.emplace(
      reservation_address.opaque(),
      ReservationMapping{allocator_address, reservation_address, reservation,
                         reservation_offset, mapping_size,
                         std::move(reservation_mapping)});

  return ScopedDeviceAddress<uint8_t>(allocator_address, device_ordinal, this);
}

// Allocation flow with retry:
//
// Allocate(device_ordinal, size)
//           │
//           ▼
// ┌─────────────────────────────────┐
// │ Reuse pending deallocation      │──found──► return
// │ with matching size?             │
// └─────────────────────────────────┘
//           │ not found
//           ▼
// ┌─────────────────────────────────┐
// │ Allocate new physical +         │──OK──► return
// │ virtual memory                  │
// └─────────────────────────────────┘
//           │ failed
//           ▼
// ┌─────────────────────────────────┐
// │ Free any GPU-completed          │
// │ pending deallocations           │
// │ (non-blocking)                  │
// └─────────────────────────────────┘
//           │
//           ▼
// ┌─────────────────────────────────┐
// │ Allocate new physical +         │──OK──► return
// │ virtual memory                  │
// └─────────────────────────────────┘
//           │ failed
//           ▼
// ┌─────────────────────────────────┐
// │ Block until GPU frees           │
// │ enough pending memory           │
// └─────────────────────────────────┘
//           │
//           ▼
// ┌─────────────────────────────────┐
// │ Allocate new physical +         │──OK──► return
// │ virtual memory                  │
// └─────────────────────────────────┘
//           │ failed
//           ▼
//    ResourceExhaustedError
absl::StatusOr<ScopedDeviceAddress<uint8_t>>
DeviceAddressVmmAllocator::Allocate(int device_ordinal, uint64_t size,
                                    bool /*retry_on_failure*/,
                                    int64_t /*memory_space*/) {
  if (size == 0) {
    return ScopedDeviceAddress<uint8_t>(DeviceAddressBase(), device_ordinal,
                                        this);
  }

  ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));

  const bool multi_device = CurrentMultiDevice();

  absl::MutexLock lock(state->mu);

  // Try to reuse a completed pending deallocation with matching size.
  std::optional<DeviceAddressBase> reused =
      TryReusePendingDeallocation(*state, size, multi_device);
  if (reused.has_value()) {
    return ScopedDeviceAddress<uint8_t>(*reused, device_ordinal, this);
  }

  absl::StatusOr<DeviceAddressBase> result =
      AllocateWithBudget(*state, size, multi_device);

  // If allocation failed (e.g., out of memory), try processing pending
  // deallocations to free memory, then retry.
  if (!result.ok()) {
    ProcessCompletedPendingDeallocations(*state);
    result = AllocateWithBudget(*state, size, multi_device);
  }

  if (!result.ok()) {
    WaitPendingDeallocationsToComplete(*state, size);
    result = AllocateWithBudget(*state, size, multi_device);
  }

  if (!result.ok()) {
    return result.status();
  }

  VLOG(3) << absl::StreamFormat(
      "Allocated virtual address %p (%uB) on device ordinal %d",
      result->opaque(), size, device_ordinal);

  return ScopedDeviceAddress<uint8_t>(*result, device_ordinal, this);
}

absl::Status DeviceAddressVmmAllocator::Deallocate(int device_ordinal,
                                                   DeviceAddressBase mem) {
  if (mem.is_null()) {
    return absl::OkStatus();
  }

  ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));

  absl::MutexLock lock(state->mu);

  auto record_it = state->records_by_allocator_address.find(mem.opaque());
  if (record_it == state->records_by_allocator_address.end() ||
      !record_it->second->allocator_active ||
      !record_it->second->allocator_address.IsSameAs(mem)) {
    if (state->active_reservation_mappings.contains(mem.opaque()) ||
        state->stale_reservation_mappings.contains(mem.opaque())) {
      return absl::InvalidArgumentError(
          "DeviceAddressVmmAllocator::Deallocate does not accept reservation "
          "alias addresses; use UnMap instead");
    }
    return absl::InvalidArgumentError(absl::StrFormat(
        "DeviceAddressVmmAllocator::Deallocate received an unknown address %p",
        mem.opaque()));
  }
  AllocationRecord& record = *record_it->second;

  for (const auto& [_, mapping] : state->active_reservation_mappings) {
    if (mapping.allocator_address.IsSameAs(mem)) {
      return absl::FailedPreconditionError(
          "DeviceAddressVmmAllocator::Deallocate requires active reservation "
          "aliases to be released with UnMap first");
    }
  }

  VLOG(3) << absl::StreamFormat(
      "Queueing deferred deallocation for virtual address %p (size=%uB) "
      "on device ordinal %d",
      mem.opaque(), mem.size(), device_ordinal);

  // Assign the next sequence number and enqueue a GPU write to the pinned
  // timeline when the stream reaches this point. The CPU polls the timeline
  // value to know when it is safe to free the memory.
  uint64_t seqno = state->next_seqno++;
  RETURN_IF_ERROR(EnqueueDeferredDeallocation(*state, seqno));

  CHECK(record.allocator_active);
  CHECK(!record.allocator_stale);
  CHECK(record.allocator_address_mapping.has_value());
  record.allocator_active = false;
  record.allocator_stale = true;
  record.allocator_stale_seqno = seqno;
  const uint64_t reclaimable_bytes =
      RoundUpToGranularity(*state, record.raw_allocation->address().size());
  state->pending_deallocations.push_back(
      {record.kind, seqno, record.allocator_address, reclaimable_bytes});

  return absl::OkStatus();
}

absl::Status DeviceAddressVmmAllocator::Map(int device_ordinal,
                                            DeviceAddressBase addr,
                                            MemoryReservation* reservation,
                                            uint64_t reservation_offset,
                                            uint64_t size) {
  if (size == 0) {
    return absl::OkStatus();
  }
  if (addr.is_null()) {
    return absl::InvalidArgumentError(
        "DeviceAddressVmmAllocator::Map requires a non-null source address");
  }
  ASSIGN_OR_RETURN(
      DeviceAddressBase reservation_address,
      ValidateReservationRange(reservation, reservation_offset, size));

  ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));

  absl::MutexLock lock(state->mu);
  auto record_it = state->records_by_allocator_address.find(addr.opaque());
  if (record_it == state->records_by_allocator_address.end() ||
      !record_it->second->allocator_active ||
      !record_it->second->allocator_address.IsSameAs(addr)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "DeviceAddressVmmAllocator::Map received an unknown allocator address "
        "%p",
        addr.opaque()));
  }
  AllocationRecord& record = *record_it->second;
  MemoryAllocation* raw_allocation = record.raw_allocation.get();
  if (size > raw_allocation->address().size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "DeviceAddressVmmAllocator::Map size %u exceeds raw allocation size "
        "%u",
        size, raw_allocation->address().size()));
  }
  if (state->active_reservation_mappings.contains(
          reservation_address.opaque()) ||
      state->stale_reservation_mappings.contains(
          reservation_address.opaque())) {
    return absl::FailedPreconditionError(
        "Reservation address is already tracked by this allocator");
  }
  for (const auto& [_, mapping] : state->active_reservation_mappings) {
    if (mapping.allocator_address.IsSameAs(addr)) {
      return absl::FailedPreconditionError(
          "Allocator address already has an active reservation alias");
    }
  }

  ASSIGN_OR_RETURN(
      MemoryReservation::ScopedMapping scoped_mapping,
      reservation->MapTo(reservation_offset, /*allocation_offset=*/0, size,
                         *raw_allocation));
  state->active_reservation_mappings.emplace(
      reservation_address.opaque(),
      ReservationMapping{addr, reservation_address, reservation,
                         reservation_offset, size, std::move(scoped_mapping)});
  return absl::OkStatus();
}

absl::Status DeviceAddressVmmAllocator::UnMap(int device_ordinal,
                                              MemoryReservation* reservation,
                                              uint64_t reservation_offset,
                                              uint64_t size) {
  if (size == 0) {
    return absl::OkStatus();
  }
  ASSIGN_OR_RETURN(
      DeviceAddressBase reservation_address,
      ValidateReservationRange(reservation, reservation_offset, size));

  ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));

  absl::MutexLock lock(state->mu);
  auto it =
      state->active_reservation_mappings.find(reservation_address.opaque());
  if (it == state->active_reservation_mappings.end()) {
    return absl::InvalidArgumentError(
        "DeviceAddressVmmAllocator::UnMap received an untracked reservation "
        "address");
  }
  if (it->second.reservation != reservation ||
      it->second.reservation_offset != reservation_offset ||
      it->second.size != size) {
    return absl::InvalidArgumentError(
        "DeviceAddressVmmAllocator::UnMap requires the same full reservation "
        "range passed to Map");
  }

  uint64_t seqno = state->next_seqno++;
  RETURN_IF_ERROR(EnqueueDeferredDeallocation(*state, seqno));

  ReservationMapping mapping = std::move(it->second);
  state->active_reservation_mappings.erase(it);
  state->stale_reservation_mappings.emplace(reservation_address.opaque(),
                                            std::move(mapping));
  state->pending_deallocations.push_back(
      {PendingDeallocationKind::kMap, seqno, reservation_address,
       /*reclaimable_bytes=*/0});
  return absl::OkStatus();
}

absl::StatusOr<Stream*> DeviceAddressVmmAllocator::GetStream(
    int device_ordinal) {
  ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));
  return state->stream;
}

absl::Status DeviceAddressVmmAllocator::SynchronizePendingOperations(
    int device_ordinal) {
  ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));

  uint64_t target_seqno;
  {
    absl::MutexLock lock(state->mu);
    if (state->pending_deallocations.empty()) {
      return absl::OkStatus();
    }
    target_seqno = state->pending_deallocations.back().seqno;
  }

  while (LoadTimeline(state->pinned_timeline) < target_seqno) {
    absl::SleepFor(kGpuTimelinePollInterval);
  }

  {
    absl::MutexLock lock(state->mu);
    while (!state->pending_deallocations.empty() &&
           state->pending_deallocations.front().seqno <= target_seqno) {
      if (state->pending_deallocations.front().kind !=
          PendingDeallocationKind::kMap) {
        DoDeallocate(*state, state->pending_deallocations.front().addr);
      } else {
        DoUnMap(*state, state->pending_deallocations.front().addr);
      }
      state->pending_deallocations.pop_front();
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<StreamExecutor*> DeviceAddressVmmAllocator::GetStreamExecutor(
    int device_ordinal) const {
  ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));
  return state->executor;
}

MemoryAllocation* DeviceAddressVmmAllocator::GetRawAllocation(
    int device_ordinal, DeviceAddressBase addr) const {
  absl::StatusOr<PerDeviceState*> state_or = GetPerDeviceState(device_ordinal);
  if (!state_or.ok()) {
    return nullptr;
  }
  PerDeviceState* state = *state_or;
  absl::MutexLock lock(state->mu);
  auto it = state->records_by_allocator_address.find(addr.opaque());
  if (it == state->records_by_allocator_address.end() ||
      !it->second->allocator_active ||
      !it->second->allocator_address.IsSameAs(addr)) {
    return nullptr;
  }
  return it->second->raw_allocation.get();
}

MemoryReservation* DeviceAddressVmmAllocator::GetReservation(
    int device_ordinal, DeviceAddressBase addr) const {
  absl::StatusOr<PerDeviceState*> state_or = GetPerDeviceState(device_ordinal);
  if (!state_or.ok()) {
    return nullptr;
  }
  PerDeviceState* state = *state_or;
  absl::MutexLock lock(state->mu);
  auto it = state->records_by_allocator_address.find(addr.opaque());
  if (it == state->records_by_allocator_address.end() ||
      !it->second->allocator_active ||
      !it->second->allocator_address.IsSameAs(addr)) {
    return nullptr;
  }
  return it->second->allocator_address_reservation.get();
}

uint64_t DeviceAddressVmmAllocator::GetAllocationGranularity(
    StreamExecutor* executor) const {
  absl::StatusOr<PerDeviceState*> state_or =
      GetPerDeviceState(executor->device_ordinal());
  if (!state_or.ok()) {
    return 0;
  }
  PerDeviceState* state = *state_or;
  return state->allocation_granularity;
}

void DeviceAddressVmmAllocator::MoveAllocatorRecordToActive(
    PerDeviceState& state, AllocationRecord& record, uint64_t new_size) {
  CHECK(!record.allocator_active);
  CHECK(record.allocator_stale);
  CHECK(record.allocator_address_mapping.has_value());
  void* allocator_va = record.allocator_address.opaque();
  auto record_it = state.records_by_allocator_address.find(allocator_va);
  CHECK(record_it != state.records_by_allocator_address.end());
  CHECK_EQ(record_it->second.get(), &record);
  record.allocator_address = DeviceAddressBase(allocator_va, new_size);
  record.allocator_active = true;
  record.allocator_stale = false;
  record.allocator_stale_seqno = 0;
}

std::optional<DeviceAddressBase>
DeviceAddressVmmAllocator::TryReusePendingDeallocation(PerDeviceState& state,
                                                       uint64_t size,
                                                       bool multi_device) {
  uint64_t rounded_size = RoundUpToGranularity(state, size);
  for (auto it = state.pending_deallocations.begin();
       it != state.pending_deallocations.end(); ++it) {
    if (it->kind != PendingDeallocationKind::kAllocate) {
      continue;
    }
    auto record_it = state.records_by_allocator_address.find(it->addr.opaque());
    CHECK(record_it != state.records_by_allocator_address.end());
    AllocationRecord& record = *record_it->second;
    CHECK(record.allocator_stale);
    CHECK(record.allocator_address.IsSameAs(it->addr));
    if (record.multi_device != multi_device) {
      continue;
    }
    if (RoundUpToGranularity(state, record.allocator_address.size()) !=
        rounded_size) {
      continue;
    }

    DeviceAddressBase reused_mem(record.allocator_address.opaque(), size);
    VLOG(3) << absl::StreamFormat(
        "Reusing pending deallocation: address=%p original_size=%uB "
        "new_size=%uB rounded_size=%uB device=%d",
        reused_mem.opaque(), record.allocator_address.size(), size, rounded_size,
        state.executor->device_ordinal());
    MoveAllocatorRecordToActive(state, record, size);
    state.pending_deallocations.erase(it);

    return reused_mem;
  }

  return std::nullopt;
}

uint64_t DeviceAddressVmmAllocator::RoundUpToGranularity(
    const PerDeviceState& state, uint64_t size) const {
  if (state.allocation_granularity == 0) {
    return size;
  }
  return ((size + state.allocation_granularity - 1) /
          state.allocation_granularity) *
         state.allocation_granularity;
}

}  // namespace stream_executor
