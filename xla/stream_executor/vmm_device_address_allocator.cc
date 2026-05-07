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

#include "xla/stream_executor/vmm_device_address_allocator.h"

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
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {

static absl::Status DeviceNotFoundError(int device_ordinal) {
  return absl::NotFoundError(
      absl::StrFormat("No device with ordinal %d registered in "
                      "DeviceAddressVmmAllocator",
                      device_ordinal));
}

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

    TF_RETURN_IF_ERROR(allocator->InitializeDeviceState(*state));

    VLOG(3) << "DeviceAddressVmmAllocator: registering device " << ordinal
            << " with pa_budget " << cfg.pa_budget;
    allocator->per_device_.emplace(ordinal, std::move(state));
  }

  return absl::OkStatus();
}

DeviceAddressVmmAllocator::~DeviceAddressVmmAllocator() {
  for (auto& [ordinal, state] : per_device_) {
    // Briefly acquire the lock to read the last pending seqno.
    uint64_t last_seqno = 0;
    {
      absl::MutexLock lock(state->mu);
      if (!state->pending_deallocations.empty()) {
        last_seqno = state->pending_deallocations.back().seqno;
      }
    }

    // Spin-wait for any pending GPU work to complete before freeing physical
    // memory. pinned_timeline is not ABSL_GUARDED_BY and last_seqno is a local.
    if (state->pinned_timeline != nullptr && last_seqno > 0) {
      while (LoadTimeline(state->pinned_timeline) < last_seqno) {
        absl::SleepFor(absl::Microseconds(50));
      }
    }

    {
      absl::MutexLock lock(state->mu);
      for (auto& pending : state->pending_deallocations) {
        CompletePendingDeallocation(*state, pending);
      }
      state->pending_deallocations.clear();
    }

    // Free platform-specific per-device resources (e.g. pinned timeline).
    if (state->destroy_fn) {
      state->destroy_fn();
    }
  }
}

DeviceAddressVmmAllocator::PerDeviceState*
DeviceAddressVmmAllocator::GetPerDeviceState(int device_ordinal) const {
  auto it = per_device_.find(device_ordinal);
  if (it == per_device_.end()) {
    return nullptr;
  }
  return it->second.get();
}

DeviceAddressVmmAllocator::PendingDeallocation
DeviceAddressVmmAllocator::PendingDeallocate(DeviceAddressBase mem,
                                             uint64_t seqno) {
  PendingDeallocation pending;
  pending.kind = PendingDeallocationKind::kDeallocate;
  pending.mem = mem;
  pending.seqno = seqno;
  return pending;
}

DeviceAddressVmmAllocator::PendingDeallocation
DeviceAddressVmmAllocator::PendingDeallocateRawAndUnMap(
    uint64_t seqno, MemoryReservation::ScopedMapping mapping) {
  PendingDeallocation pending;
  pending.kind = PendingDeallocationKind::kDeallocateRawAndUnMap;
  pending.seqno = seqno;
  pending.mapping.emplace(std::move(mapping));
  return pending;
}

DeviceAddressVmmAllocator::PendingDeallocation
DeviceAddressVmmAllocator::PendingDeallocateAndUnMap(
    DeviceAddressBase mem, uint64_t seqno,
    MemoryReservation::ScopedMapping mapping) {
  PendingDeallocation pending;
  pending.kind = PendingDeallocationKind::kDeallocateAndUnMap;
  pending.mem = mem;
  pending.seqno = seqno;
  pending.mapping.emplace(std::move(mapping));
  return pending;
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

bool DeviceAddressVmmAllocator::IsTrackedRawAllocation(
    const PerDeviceState& state, const MemoryAllocation* raw_allocation) const {
  if (raw_allocation == nullptr) {
    return false;
  }
  for (const auto& raw_allocation_entry : state.raw_allocations) {
    if (raw_allocation_entry.second.get() == raw_allocation) {
      return true;
    }
  }
  return false;
}

bool DeviceAddressVmmAllocator::HasActiveRawMappings(
    const PerDeviceState& state, const MemoryAllocation* raw_allocation) const {
  if (raw_allocation == nullptr) {
    return false;
  }
  for (const auto& active_mapping : state.active_raw_mapping_keys) {
    if (active_mapping.second == raw_allocation) {
      return true;
    }
  }
  return false;
}

absl::Status DeviceAddressVmmAllocator::ValidateNoActiveRawMappings(
    const PerDeviceState& state, const MemoryAllocation* raw_allocation,
    const char* operation) const {
  if (raw_allocation == nullptr) {
    return absl::OkStatus();
  }
  for (const auto& active_mapping : state.active_raw_mapping_keys) {
    if (active_mapping.second == raw_allocation) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "%s cannot release raw allocation %p while MapToRaw mapping at "
          "virtual address %p is still active; call UnMapToRaw before "
          "deallocating the raw allocation",
          operation, raw_allocation, active_mapping.first));
    }
  }
  return absl::OkStatus();
}

void DeviceAddressVmmAllocator::CompleteDeferredRawMappings(
    PerDeviceState& state, MemoryAllocation* raw_allocation) {
  if (raw_allocation == nullptr) {
    return;
  }
  DCHECK(!HasActiveRawMappings(state, raw_allocation))
      << "Completing a raw allocation while caller-owned MapToRaw mappings are "
         "still active";
  state.deferred_raw_mappings.erase(raw_allocation);
}

uint64_t DeviceAddressVmmAllocator::EraseRawAllocationKey(PerDeviceState& state,
                                                          void* key) {
  auto it = state.raw_allocations.find(key);
  if (it == state.raw_allocations.end()) {
    return 0;
  }

  uint64_t released_size = 0;
  if (it->second.use_count() == 1) {
    CompleteDeferredRawMappings(state, it->second.get());
    released_size = RoundUpToGranularity(state, it->second->address().size());
  }
  state.raw_allocations.erase(it);

  if (released_size > 0) {
    DCHECK_GE(state.pa_allocated, released_size);
    state.pa_allocated -= released_size;
  }
  return released_size;
}

absl::StatusOr<uint64_t> DeviceAddressVmmAllocator::EnqueuePendingOperation(
    PerDeviceState& state) {
  uint64_t seqno = state.next_seqno++;
  TF_RETURN_IF_ERROR(EnqueueDeferredDeallocation(state, seqno));
  return seqno;
}

absl::Status DeviceAddressVmmAllocator::QueueDeallocate(PerDeviceState& state,
                                                        DeviceAddressBase mem) {
  TF_ASSIGN_OR_RETURN(uint64_t seqno, EnqueuePendingOperation(state));
  state.pending_deallocations.push_back(PendingDeallocate(mem, seqno));
  return absl::OkStatus();
}

absl::Status DeviceAddressVmmAllocator::QueueDeallocateRawAndUnMap(
    PerDeviceState& state, MemoryReservation::ScopedMapping& mapping) {
  TF_ASSIGN_OR_RETURN(uint64_t seqno, EnqueuePendingOperation(state));
  state.pending_deallocations.push_back(
      PendingDeallocateRawAndUnMap(seqno, std::move(mapping)));
  return absl::OkStatus();
}

absl::Status DeviceAddressVmmAllocator::QueueDeallocateAndUnMap(
    PerDeviceState& state, DeviceAddressBase mem,
    MemoryReservation::ScopedMapping& mapping) {
  TF_ASSIGN_OR_RETURN(uint64_t seqno, EnqueuePendingOperation(state));
  state.pending_deallocations.push_back(
      PendingDeallocateAndUnMap(mem, seqno, std::move(mapping)));
  return absl::OkStatus();
}

uint64_t DeviceAddressVmmAllocator::PendingDeallocationReclaimableSize(
    const PerDeviceState& state, const PendingDeallocation& pending) const {
  if (!pending.mem.is_null()) {
    return RoundUpToGranularity(state, pending.mem.size());
  }
  if (pending.mapping.has_value()) {
    return RoundUpToGranularity(state,
                                pending.mapping->mapped_address().size());
  }
  return 0;
}

void DeviceAddressVmmAllocator::CompletePendingDeallocation(
    PerDeviceState& state, PendingDeallocation& pending) {
  if (pending.mapping.has_value()) {
    DeviceAddressBase mapped = pending.mapping->mapped_address();
    pending.mapping.reset();
    state.scoped_mappings.erase(mapped.opaque());
    EraseRawAllocationKey(state, mapped.opaque());
  }

  if (!pending.mem.is_null()) {
    DoDeallocate(state, pending.mem);
  }
}

void DeviceAddressVmmAllocator::ProcessCompletedPendingDeallocations(
    PerDeviceState& state) {
  uint64_t completed = LoadTimeline(state.pinned_timeline);
  while (!state.pending_deallocations.empty()) {
    if (state.pending_deallocations.front().seqno > completed) {
      break;
    }
    CompletePendingDeallocation(state, state.pending_deallocations.front());
    state.pending_deallocations.pop_front();
  }
}

void DeviceAddressVmmAllocator::WaitPendingDeallocationsToComplete(
    PerDeviceState& state, uint64_t size) {
  if (state.pending_deallocations.empty()) {
    return;
  }

  uint64_t accumulated_size = 0;
  uint64_t rounded_size = RoundUpToGranularity(state, size);
  uint64_t target_seqno = 0;

  // Target 1.1x the requested size to provide some headroom.
  uint64_t target_size = rounded_size + rounded_size / 10;

  std::vector<PendingDeallocation> selected;
  while (!state.pending_deallocations.empty() &&
         (accumulated_size < target_size || selected.empty())) {
    accumulated_size += PendingDeallocationReclaimableSize(
        state, state.pending_deallocations.front());
    target_seqno = state.pending_deallocations.front().seqno;
    selected.push_back(std::move(state.pending_deallocations.front()));
    state.pending_deallocations.pop_front();
  }

  if (selected.empty()) {
    return;
  }

  // Release the lock before spin-waiting to avoid stalling other threads for
  // potentially milliseconds while the GPU drains its work queue.
  state.mu.unlock();

  // Poll until the GPU writes a timeline value >= target_seqno.
  // Since timeline values are written in stream order, this guarantees all
  // selected pending operations have completed.
  while (LoadTimeline(state.pinned_timeline) < target_seqno) {
    absl::SleepFor(absl::Microseconds(50));
  }

  state.mu.lock();

  for (auto& item : selected) {
    CompletePendingDeallocation(state, item);
  }
}

void DeviceAddressVmmAllocator::DoDeallocate(PerDeviceState& state,
                                             DeviceAddressBase mem) {
  if (mem.is_null()) {
    return;
  }

  VLOG(3) << absl::StreamFormat(
      "Actually freeing virtual address %p (size=%uB) on device ordinal %d",
      mem.opaque(), mem.size(), state.executor->device_ordinal());

  // Erase the ScopedMapping first: its destructor unmaps the physical memory
  // from the virtual address range.
  state.scoped_mappings.erase(mem.opaque());
  // Erase the reservation next: its destructor frees the virtual address range.
  state.reservations.erase(mem.opaque());
  // Erase the raw allocation last: its destructor releases the physical memory.
  EraseRawAllocationKey(state, mem.opaque());
}

void* DeviceAddressVmmAllocator::TrackOwnedAllocation(
    PerDeviceState& state, std::shared_ptr<MemoryAllocation> raw_allocation,
    std::unique_ptr<MemoryReservation> reservation,
    MemoryReservation::ScopedMapping mapping, uint64_t allocated_size) {
  void* va_ptr = reservation->address().opaque();
  state.raw_allocations.emplace(va_ptr, std::move(raw_allocation));
  state.reservations.emplace(va_ptr, std::move(reservation));
  state.scoped_mappings.emplace(va_ptr, std::move(mapping));
  state.pa_allocated += allocated_size;
  return va_ptr;
}

void DeviceAddressVmmAllocator::TrackRawAndExternalMapping(
    PerDeviceState& state, DeviceAddressBase target,
    std::shared_ptr<MemoryAllocation> raw_allocation,
    MemoryReservation::ScopedMapping mapping, uint64_t allocated_size) {
  state.raw_allocations.emplace(target.opaque(), std::move(raw_allocation));
  state.scoped_mappings.emplace(target.opaque(), std::move(mapping));
  state.pa_allocated += allocated_size;
}

void* DeviceAddressVmmAllocator::TrackOwnedAndExternalMapping(
    PerDeviceState& state, DeviceAddressBase target,
    std::shared_ptr<MemoryAllocation> raw_allocation,
    std::unique_ptr<MemoryReservation> owned_reservation,
    MemoryReservation::ScopedMapping owned_mapping,
    MemoryReservation::ScopedMapping external_mapping,
    uint64_t allocated_size) {
  void* owned_va = owned_reservation->address().opaque();
  state.raw_allocations.emplace(owned_va, raw_allocation);
  state.raw_allocations.emplace(target.opaque(), std::move(raw_allocation));
  state.reservations.emplace(owned_va, std::move(owned_reservation));
  state.scoped_mappings.emplace(owned_va, std::move(owned_mapping));
  state.scoped_mappings.emplace(target.opaque(), std::move(external_mapping));
  state.pa_allocated += allocated_size;
  return owned_va;
}

absl::StatusOr<DeviceAddressBase> DeviceAddressVmmAllocator::TryFreshAllocate(
    PerDeviceState& state, uint64_t size) {
  uint64_t rounded_size = RoundUpToGranularity(state, size);
  if (state.pa_allocated + rounded_size > state.pa_budget) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Not enough PA budget for allocation: pa_allocated=%uB, "
        "rounded_size=%uB, pa_budget=%uB",
        state.pa_allocated, rounded_size, state.pa_budget));
  }

  TF_ASSIGN_OR_RETURN(auto raw_alloc, CreateAllocation(state.executor, size));
  const uint64_t padded_size = raw_alloc->address().size();

  TF_ASSIGN_OR_RETURN(auto reservation,
                      CreateReservation(state.executor, size));

  TF_ASSIGN_OR_RETURN(
      auto scoped_mapping,
      reservation->MapTo(/*reservation_offset=*/0, /*allocation_offset=*/0,
                         padded_size, *raw_alloc));

  auto shared_raw = std::shared_ptr<MemoryAllocation>(std::move(raw_alloc));
  void* va_ptr =
      TrackOwnedAllocation(state, std::move(shared_raw), std::move(reservation),
                           std::move(scoped_mapping), rounded_size);
  // Return the original requested size, not the padded size.
  return DeviceAddressBase(va_ptr, size);
}

absl::StatusOr<DeviceAddressBase>
DeviceAddressVmmAllocator::TryFreshAllocateRawAndMap(
    PerDeviceState& state, MemoryReservation* reservation,
    uint64_t reservation_offset, uint64_t size) {
  TF_ASSIGN_OR_RETURN(
      DeviceAddressBase target,
      ValidateReservationRange(reservation, reservation_offset, size));
  if (state.raw_allocations.contains(target.opaque()) ||
      state.scoped_mappings.contains(target.opaque())) {
    return absl::AlreadyExistsError(absl::StrFormat(
        "reservation range is already tracked at virtual address %p",
        target.opaque()));
  }

  uint64_t rounded_size = RoundUpToGranularity(state, size);
  if (state.pa_allocated + rounded_size > state.pa_budget) {
    return absl::ResourceExhaustedError(
        absl::StrFormat("Not enough PA budget for mapping: pa_allocated=%uB, "
                        "rounded_size=%uB, pa_budget=%uB",
                        state.pa_allocated, rounded_size, state.pa_budget));
  }

  TF_ASSIGN_OR_RETURN(auto raw_alloc, CreateAllocation(state.executor, size));
  if (size > raw_alloc->address().size()) {
    return absl::InternalError(absl::StrFormat(
        "physical allocation is smaller than requested mapping: "
        "allocation_size=%uB, mapping_size=%uB",
        raw_alloc->address().size(), size));
  }

  TF_ASSIGN_OR_RETURN(
      auto scoped_mapping,
      reservation->MapTo(reservation_offset, /*allocation_offset=*/0, size,
                         *raw_alloc));
  auto shared_raw = std::shared_ptr<MemoryAllocation>(std::move(raw_alloc));

  TrackRawAndExternalMapping(state, target, std::move(shared_raw),
                             std::move(scoped_mapping), rounded_size);

  return target;
}

absl::StatusOr<DeviceAddressBase>
DeviceAddressVmmAllocator::TryFreshAllocateAndMap(
    PerDeviceState& state, uint64_t allocation_size,
    MemoryReservation* reservation, uint64_t reservation_offset,
    uint64_t mapping_size) {
  TF_ASSIGN_OR_RETURN(
      DeviceAddressBase target,
      ValidateReservationRange(reservation, reservation_offset, mapping_size));
  if (state.raw_allocations.contains(target.opaque()) ||
      state.scoped_mappings.contains(target.opaque())) {
    return absl::AlreadyExistsError(absl::StrFormat(
        "reservation range is already tracked at virtual address %p",
        target.opaque()));
  }

  uint64_t rounded_size = RoundUpToGranularity(state, allocation_size);
  if (state.pa_allocated + rounded_size > state.pa_budget) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Not enough PA budget for allocation: pa_allocated=%uB, "
        "rounded_size=%uB, pa_budget=%uB",
        state.pa_allocated, rounded_size, state.pa_budget));
  }

  TF_ASSIGN_OR_RETURN(auto raw_alloc,
                      CreateAllocation(state.executor, allocation_size));
  const uint64_t padded_size = raw_alloc->address().size();
  if (mapping_size > padded_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "mapping size must not exceed physical allocation size: "
        "mapping_size=%uB, allocation_size=%uB",
        mapping_size, padded_size));
  }

  TF_ASSIGN_OR_RETURN(auto owned_reservation,
                      CreateReservation(state.executor, allocation_size));
  TF_ASSIGN_OR_RETURN(auto owned_mapping,
                      owned_reservation->MapTo(/*reservation_offset=*/0,
                                               /*allocation_offset=*/0,
                                               padded_size, *raw_alloc));
  TF_ASSIGN_OR_RETURN(
      auto external_mapping,
      reservation->MapTo(reservation_offset, /*allocation_offset=*/0,
                         mapping_size, *raw_alloc));

  auto shared_raw = std::shared_ptr<MemoryAllocation>(std::move(raw_alloc));
  void* owned_va = TrackOwnedAndExternalMapping(
      state, target, std::move(shared_raw), std::move(owned_reservation),
      std::move(owned_mapping), std::move(external_mapping), rounded_size);

  return DeviceAddressBase(owned_va, allocation_size);
}

void DeviceAddressVmmAllocator::LogFreshAllocationResult(
    PerDeviceState& state, const char* attempt, uint64_t reclaim_size,
    const absl::StatusOr<DeviceAddressBase>& result) const {
  if (result.ok()) {
    VLOG(3) << absl::StreamFormat(
        "VMM allocator %s fresh allocation succeeded: address=%p size=%uB "
        "device_ordinal=%d pa_allocated=%uB pa_budget=%uB",
        attempt, result->opaque(), result->size(),
        state.executor->device_ordinal(), state.pa_allocated, state.pa_budget);
    return;
  }
  VLOG(3) << absl::StreamFormat(
      "VMM allocator %s fresh allocation failed: status=%s device_ordinal=%d "
      "reclaim_size=%uB pa_allocated=%uB pa_budget=%uB pending_count=%u",
      attempt, result.status().ToString(), state.executor->device_ordinal(),
      reclaim_size, state.pa_allocated, state.pa_budget,
      static_cast<uint64_t>(state.pending_deallocations.size()));
}

void DeviceAddressVmmAllocator::LogPendingReclaim(PerDeviceState& state,
                                                  const char* reclaim_action,
                                                  uint64_t reclaim_size) const {
  VLOG(3) << absl::StreamFormat(
      "VMM allocator reclaim: %s pending operations across all kinds "
      "(Deallocate, DeallocateRawAndUnMap, DeallocateAndUnMap): "
      "device_ordinal=%d reclaim_size=%uB pending_count=%u",
      reclaim_action, state.executor->device_ordinal(), reclaim_size,
      static_cast<uint64_t>(state.pending_deallocations.size()));
}

// Shared pending-reclaim retry flow:
//
// TryWithPendingReclaim(reclaim_size, try_reuse, try_fresh)
//           │
//           ▼
// ┌─────────────────────────────────┐
// │ try_reuse()                     │──found──► return reused address
// └─────────────────────────────────┘
//           │ not found
//           ▼
// ┌─────────────────────────────────┐
// │ try_fresh()                     │──OK──► return fresh address
// └─────────────────────────────────┘
//           │ ResourceExhausted
//           ▼
// ┌─────────────────────────────────┐
// │ Process completed pending       │
// │ operations                      │
// └─────────────────────────────────┘
//           │
//           ▼
// ┌─────────────────────────────────┐
// │ try_fresh()                     │──OK──► return fresh address
// └─────────────────────────────────┘
//           │ ResourceExhausted
//           ▼
// ┌─────────────────────────────────┐
// │ Wait for pending operations     │
// │ to reclaim enough memory        │
// └─────────────────────────────────┘
//           │
//           ▼
// ┌─────────────────────────────────┐
// │ try_fresh()                     │──OK──► return fresh address
// └─────────────────────────────────┘
//           │ failed
//           ▼
//       return error
template <typename TryReuseFn, typename TryFreshFn>
absl::StatusOr<DeviceAddressBase>
DeviceAddressVmmAllocator::TryWithPendingReclaim(PerDeviceState& state,
                                                 uint64_t reclaim_size,
                                                 TryReuseFn try_reuse,
                                                 TryFreshFn try_fresh) {
  std::optional<DeviceAddressBase> reused = try_reuse();
  if (reused.has_value()) {
    return *reused;
  }

  absl::StatusOr<DeviceAddressBase> result = try_fresh();
  LogFreshAllocationResult(state, "initial", reclaim_size, result);

  if (absl::IsResourceExhausted(result.status())) {
    LogPendingReclaim(state, "processing completed", reclaim_size);
    ProcessCompletedPendingDeallocations(state);
    result = try_fresh();
    LogFreshAllocationResult(state, "post-completed-reclaim retry",
                             reclaim_size, result);
  }

  if (absl::IsResourceExhausted(result.status())) {
    LogPendingReclaim(state, "waiting for", reclaim_size);
    WaitPendingDeallocationsToComplete(state, reclaim_size);
    result = try_fresh();
    LogFreshAllocationResult(state, "post-wait-reclaim retry", reclaim_size,
                             result);
  }

  return result;
}

// Allocate() reuses pending Deallocate entries, otherwise tries fresh owned
// allocation through TryFreshAllocate().
absl::StatusOr<ScopedDeviceAddress<uint8_t>>
DeviceAddressVmmAllocator::Allocate(int device_ordinal, uint64_t size,
                                    bool /*retry_on_failure*/,
                                    int64_t /*memory_space*/) {
  if (size == 0) {
    return ScopedDeviceAddress<uint8_t>(DeviceAddressBase(), device_ordinal,
                                        this);
  }

  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }

  absl::MutexLock lock(state->mu);
  auto try_reuse = [&]() ABSL_NO_THREAD_SAFETY_ANALYSIS {
    return TryReusePendingDeallocate(*state, size);
  };
  auto try_fresh = [&]() ABSL_NO_THREAD_SAFETY_ANALYSIS {
    return TryFreshAllocate(*state, size);
  };

  absl::StatusOr<DeviceAddressBase> result =
      TryWithPendingReclaim(*state, size, try_reuse, try_fresh);

  if (!result.ok()) {
    return result.status();
  }

  VLOG(3) << absl::StreamFormat(
      "Allocated virtual address %p (%uB) on device ordinal %d",
      result->opaque(), size, device_ordinal);

  return ScopedDeviceAddress<uint8_t>(*result, device_ordinal, this);
}

// AllocateRawAndMap() reuses pending DeallocateRawAndUnMap entries, otherwise
// tries fresh raw allocation and maps it into the caller reservation.
absl::Status DeviceAddressVmmAllocator::AllocateRawAndMap(
    int device_ordinal, MemoryReservation* reservation,
    uint64_t reservation_offset, uint64_t size) {
  if (size == 0) {
    return absl::OkStatus();
  }

  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }

  TF_ASSIGN_OR_RETURN(
      DeviceAddressBase target,
      ValidateReservationRange(reservation, reservation_offset, size));

  absl::MutexLock lock(state->mu);
  auto try_reuse =
      [&]() ABSL_NO_THREAD_SAFETY_ANALYSIS -> std::optional<DeviceAddressBase> {
    if (TryReuseDeallocateRawAndUnMap(*state, target)) {
      return target;
    }
    return std::nullopt;
  };
  auto try_fresh = [&]() ABSL_NO_THREAD_SAFETY_ANALYSIS {
    return TryFreshAllocateRawAndMap(*state, reservation, reservation_offset,
                                     size);
  };

  absl::StatusOr<DeviceAddressBase> result =
      TryWithPendingReclaim(*state, size, try_reuse, try_fresh);

  return result.status();
}

absl::Status DeviceAddressVmmAllocator::DeallocateRawAndUnMap(
    int device_ordinal, MemoryReservation* reservation,
    uint64_t reservation_offset, uint64_t size) {
  if (size == 0) {
    return absl::OkStatus();
  }

  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }

  TF_ASSIGN_OR_RETURN(
      DeviceAddressBase target,
      ValidateReservationRange(reservation, reservation_offset, size));

  absl::MutexLock lock(state->mu);
  auto mapping_it = state->scoped_mappings.find(target.opaque());
  if (mapping_it == state->scoped_mappings.end() ||
      !mapping_it->second.mapped_address().IsSameAs(target)) {
    return absl::NotFoundError(absl::StrFormat(
        "no mapping tracked at virtual address %p with size %uB",
        target.opaque(), target.size()));
  }

  auto raw_it = state->raw_allocations.find(target.opaque());
  if (raw_it != state->raw_allocations.end()) {
    TF_RETURN_IF_ERROR(ValidateNoActiveRawMappings(*state, raw_it->second.get(),
                                                   "DeallocateRawAndUnMap"));
  }

  TF_RETURN_IF_ERROR(QueueDeallocateRawAndUnMap(*state, mapping_it->second));
  state->scoped_mappings.erase(mapping_it);

  return absl::OkStatus();
}

absl::StatusOr<MemoryReservation::ScopedMapping>
DeviceAddressVmmAllocator::MapToRaw(int device_ordinal,
                                    MemoryAllocation* raw_allocation,
                                    MemoryReservation* reservation,
                                    uint64_t reservation_offset,
                                    uint64_t size) {
  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }
  if (raw_allocation == nullptr) {
    return absl::InvalidArgumentError("raw_allocation must not be null");
  }
  if (size == 0) {
    return MemoryReservation::ScopedMapping();
  }

  TF_ASSIGN_OR_RETURN(
      DeviceAddressBase target,
      ValidateReservationRange(reservation, reservation_offset, size));

  absl::MutexLock lock(state->mu);
  if (!IsTrackedRawAllocation(*state, raw_allocation)) {
    return absl::NotFoundError(
        "raw_allocation is not tracked by this allocator");
  }
  if (size > raw_allocation->address().size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "mapping size must not exceed physical allocation size: "
        "mapping_size=%uB, allocation_size=%uB",
        size, raw_allocation->address().size()));
  }
  if (state->raw_allocations.contains(target.opaque()) ||
      state->scoped_mappings.contains(target.opaque()) ||
      state->active_raw_mapping_keys.contains(target.opaque())) {
    return absl::AlreadyExistsError(absl::StrFormat(
        "reservation range is already tracked at virtual address %p",
        target.opaque()));
  }

  auto deferred_it = state->deferred_raw_mappings.find(raw_allocation);
  if (deferred_it != state->deferred_raw_mappings.end()) {
    std::vector<MemoryReservation::ScopedMapping>& mappings =
        deferred_it->second;
    for (auto mapping_it = mappings.begin(); mapping_it != mappings.end();
         ++mapping_it) {
      if (!mapping_it->mapped_address().IsSameAs(target)) {
        continue;
      }

      MemoryReservation::ScopedMapping mapping = std::move(*mapping_it);
      mappings.erase(mapping_it);
      if (mappings.empty()) {
        state->deferred_raw_mappings.erase(deferred_it);
      }
      state->active_raw_mapping_keys.emplace(target.opaque(), raw_allocation);
      return mapping;
    }
  }

  TF_ASSIGN_OR_RETURN(auto mapping, reservation->MapTo(reservation_offset,
                                                       /*allocation_offset=*/0,
                                                       size, *raw_allocation));
  DeviceAddressBase mapped = mapping.mapped_address();
  state->active_raw_mapping_keys.emplace(mapped.opaque(), raw_allocation);
  return mapping;
}

absl::Status DeviceAddressVmmAllocator::UnMapToRaw(
    int device_ordinal, MemoryAllocation* raw_allocation,
    MemoryReservation::ScopedMapping&& mapping) {
  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }
  if (raw_allocation == nullptr) {
    return absl::InvalidArgumentError("raw_allocation must not be null");
  }
  if (mapping.is_null()) {
    return absl::OkStatus();
  }

  DeviceAddressBase mapped = mapping.mapped_address();

  absl::MutexLock lock(state->mu);
  auto active_it = state->active_raw_mapping_keys.find(mapped.opaque());
  if (active_it == state->active_raw_mapping_keys.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "mapping at virtual address %p is not tracked by MapToRaw",
        mapped.opaque()));
  }

  if (active_it->second != raw_allocation) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "mapping at virtual address %p belongs to a different raw allocation",
        mapped.opaque()));
  }
  if (!IsTrackedRawAllocation(*state, raw_allocation)) {
    return absl::NotFoundError(
        "raw_allocation is not tracked by this allocator");
  }

  state->active_raw_mapping_keys.erase(active_it);
  state->deferred_raw_mappings[raw_allocation].push_back(std::move(mapping));
  return absl::OkStatus();
}

// AllocateAndMap() reuses pending DeallocateAndUnMap entries, otherwise tries
// fresh owned allocation plus external mapping.
absl::StatusOr<ScopedDeviceAddress<uint8_t>>
DeviceAddressVmmAllocator::AllocateAndMap(
    int device_ordinal, uint64_t allocation_size, bool retry_on_failure,
    int64_t memory_space, MemoryReservation* reservation,
    uint64_t reservation_offset, uint64_t mapping_size) {
  if (allocation_size == 0) {
    if (mapping_size != 0) {
      return absl::InvalidArgumentError(
          "mapping_size must be zero when allocation_size is zero");
    }
    return ScopedDeviceAddress<uint8_t>(DeviceAddressBase(), device_ordinal,
                                        this);
  }

  if (mapping_size == 0) {
    return Allocate(device_ordinal, allocation_size, retry_on_failure,
                    memory_space);
  }

  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }

  TF_ASSIGN_OR_RETURN(
      DeviceAddressBase target,
      ValidateReservationRange(reservation, reservation_offset, mapping_size));

  absl::MutexLock lock(state->mu);
  auto try_reuse = [&]() ABSL_NO_THREAD_SAFETY_ANALYSIS {
    return TryReuseDeallocateAndUnMap(*state, target, allocation_size);
  };
  auto try_fresh = [&]() ABSL_NO_THREAD_SAFETY_ANALYSIS {
    return TryFreshAllocateAndMap(*state, allocation_size, reservation,
                                  reservation_offset, mapping_size);
  };

  absl::StatusOr<DeviceAddressBase> result =
      TryWithPendingReclaim(*state, allocation_size, try_reuse, try_fresh);

  if (!result.ok()) {
    return result.status();
  }

  return ScopedDeviceAddress<uint8_t>(*result, device_ordinal, this);
}

absl::Status DeviceAddressVmmAllocator::DeallocateAndUnMap(
    int device_ordinal, DeviceAddressBase mem, MemoryReservation* reservation,
    uint64_t reservation_offset, uint64_t size) {
  if (mem.is_null()) {
    return DeallocateRawAndUnMap(device_ordinal, reservation,
                                 reservation_offset, size);
  }
  if (size == 0) {
    return Deallocate(device_ordinal, mem);
  }

  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }

  TF_ASSIGN_OR_RETURN(
      DeviceAddressBase target,
      ValidateReservationRange(reservation, reservation_offset, size));

  absl::MutexLock lock(state->mu);
  auto raw_it = state->raw_allocations.find(mem.opaque());
  if (raw_it == state->raw_allocations.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "no owned allocation tracked at virtual address %p", mem.opaque()));
  }
  TF_RETURN_IF_ERROR(ValidateNoActiveRawMappings(*state, raw_it->second.get(),
                                                 "DeallocateAndUnMap"));

  auto mapping_it = state->scoped_mappings.find(target.opaque());
  if (mapping_it == state->scoped_mappings.end() ||
      !mapping_it->second.mapped_address().IsSameAs(target)) {
    return absl::NotFoundError(absl::StrFormat(
        "no mapping tracked at virtual address %p with size %uB",
        target.opaque(), target.size()));
  }

  TF_RETURN_IF_ERROR(QueueDeallocateAndUnMap(*state, mem, mapping_it->second));
  state->scoped_mappings.erase(mapping_it);

  return absl::OkStatus();
}

absl::Status DeviceAddressVmmAllocator::Deallocate(int device_ordinal,
                                                   DeviceAddressBase mem) {
  if (mem.is_null()) {
    return absl::OkStatus();
  }

  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }

  absl::MutexLock lock(state->mu);
  auto raw_it = state->raw_allocations.find(mem.opaque());
  if (raw_it != state->raw_allocations.end()) {
    TF_RETURN_IF_ERROR(ValidateNoActiveRawMappings(*state, raw_it->second.get(),
                                                   "Deallocate"));
  }

  VLOG(3) << absl::StreamFormat(
      "Queueing deferred deallocation for virtual address %p (size=%uB) "
      "on device ordinal %d",
      mem.opaque(), mem.size(), device_ordinal);

  TF_RETURN_IF_ERROR(QueueDeallocate(*state, mem));

  return absl::OkStatus();
}

absl::StatusOr<Stream*> DeviceAddressVmmAllocator::GetStream(
    int device_ordinal) {
  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }
  return state->stream;
}

absl::StatusOr<StreamExecutor*> DeviceAddressVmmAllocator::GetStreamExecutor(
    int device_ordinal) const {
  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }
  return state->executor;
}

MemoryAllocation* DeviceAddressVmmAllocator::GetRawAllocation(
    int device_ordinal, DeviceAddressBase addr) const {
  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return nullptr;
  }
  absl::MutexLock lock(state->mu);
  auto it = state->raw_allocations.find(addr.opaque());
  if (it == state->raw_allocations.end()) {
    return nullptr;
  }
  return it->second.get();
}

MemoryReservation* DeviceAddressVmmAllocator::GetReservation(
    int device_ordinal, DeviceAddressBase addr) const {
  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return nullptr;
  }
  absl::MutexLock lock(state->mu);
  auto it = state->reservations.find(addr.opaque());
  if (it == state->reservations.end()) {
    return nullptr;
  }
  return it->second.get();
}

uint64_t DeviceAddressVmmAllocator::GetAllocationGranularity(
    StreamExecutor* executor) const {
  PerDeviceState* state = GetPerDeviceState(executor->device_ordinal());
  if (state == nullptr) {
    return 0;
  }
  return state->allocation_granularity;
}

std::optional<DeviceAddressBase>
DeviceAddressVmmAllocator::TryReusePendingDeallocate(PerDeviceState& state,
                                                     uint64_t size) {
  uint64_t rounded_size = RoundUpToGranularity(state, size);
  for (auto it = state.pending_deallocations.begin();
       it != state.pending_deallocations.end(); ++it) {
    if (it->kind != PendingDeallocationKind::kDeallocate ||
        it->mapping.has_value() || it->mem.is_null()) {
      continue;
    }
    if (RoundUpToGranularity(state, it->mem.size()) != rounded_size) {
      continue;
    }
    auto raw_it = state.raw_allocations.find(it->mem.opaque());
    if (raw_it != state.raw_allocations.end() &&
        state.deferred_raw_mappings.contains(raw_it->second.get())) {
      continue;
    }

    DeviceAddressBase reused_mem(it->mem.opaque(), size);
    VLOG(3) << absl::StreamFormat(
        "Reusing pending Deallocation: address=%p original_size=%uB "
        "new_size=%uB rounded_size=%uB device=%d",
        reused_mem.opaque(), it->mem.size(), size, rounded_size,
        state.executor->device_ordinal());
    state.pending_deallocations.erase(it);

    return reused_mem;
  }

  return std::nullopt;
}

bool DeviceAddressVmmAllocator::TryReuseDeallocateRawAndUnMap(
    PerDeviceState& state, DeviceAddressBase target) {
  for (auto it = state.pending_deallocations.begin();
       it != state.pending_deallocations.end(); ++it) {
    if (it->kind != PendingDeallocationKind::kDeallocateRawAndUnMap ||
        !it->mapping.has_value()) {
      continue;
    }
    if (!it->mapping->mapped_address().IsSameAs(target)) {
      continue;
    }

    VLOG(3) << absl::StreamFormat(
        "Reusing pending DeallocateRawAndUnMap: address=%p size=%uB "
        "device=%d",
        target.opaque(), target.size(), state.executor->device_ordinal());
    auto insert_result = state.scoped_mappings.emplace(
        target.opaque(), std::move(it->mapping.value()));
    CHECK(insert_result.second);
    state.pending_deallocations.erase(it);
    return true;
  }

  return false;
}

std::optional<DeviceAddressBase>
DeviceAddressVmmAllocator::TryReuseDeallocateAndUnMap(
    PerDeviceState& state, DeviceAddressBase target, uint64_t allocation_size) {
  for (auto it = state.pending_deallocations.begin();
       it != state.pending_deallocations.end(); ++it) {
    if (it->kind != PendingDeallocationKind::kDeallocateAndUnMap ||
        it->mem.is_null() || it->mem.size() != allocation_size ||
        !it->mapping.has_value()) {
      continue;
    }
    if (!it->mapping->mapped_address().IsSameAs(target)) {
      continue;
    }

    DeviceAddressBase reused_mem(it->mem.opaque(), allocation_size);
    VLOG(3) << absl::StreamFormat(
        "Reusing pending DeallocateAndUnMap: owned_address=%p "
        "external_address=%p size=%uB device=%d",
        reused_mem.opaque(), target.opaque(), allocation_size,
        state.executor->device_ordinal());
    auto insert_result = state.scoped_mappings.emplace(
        target.opaque(), std::move(it->mapping.value()));
    CHECK(insert_result.second);
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
