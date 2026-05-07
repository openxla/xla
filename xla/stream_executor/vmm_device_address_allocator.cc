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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
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

// Returns the completed timeline value from pinned host memory using an
// acquire load, so all GPU writes prior to this value are visible.
// Uses __atomic_load_n rather than std::atomic<> because the pointer is
// volatile (GPU-written pinned memory) and reinterpret_cast to
// std::atomic<uint64_t>* would discard the volatile qualifier.
static uint64_t LoadTimeline(const volatile uint64_t* pinned_timeline) {
  return __atomic_load_n(pinned_timeline, __ATOMIC_ACQUIRE);
}

static uintptr_t AddressStart(DeviceAddressBase address) {
  return reinterpret_cast<uintptr_t>(address.opaque());
}

static uintptr_t AddressEnd(DeviceAddressBase address) {
  uintptr_t start = AddressStart(address);
  if (std::numeric_limits<uintptr_t>::max() - start < address.size()) {
    return std::numeric_limits<uintptr_t>::max();
  }
  return start + address.size();
}

static bool AddressRangesOverlap(DeviceAddressBase lhs, DeviceAddressBase rhs) {
  if (lhs.is_null() || rhs.is_null() || lhs.size() == 0 || rhs.size() == 0) {
    return false;
  }
  return AddressStart(lhs) < AddressEnd(rhs) &&
         AddressStart(rhs) < AddressEnd(lhs);
}

struct DebugStatsRow {
  std::string operation;
  std::string api_calls;
  std::string allocation_reuse;
  std::string allocator_va_reuse;
  std::string reservation_va_reuse;
};

struct DebugStatsWidths {
  size_t operation;
  size_t api_calls;
  size_t allocation_reuse;
  size_t allocator_va_reuse;
  size_t reservation_va_reuse;
};

static std::string DebugStatsCount(uint64_t count) {
  return absl::StrFormat("%u", count);
}

static std::string DebugStatsReuse(uint64_t reuse, uint64_t api_calls) {
  double percent = api_calls == 0 ? 0.0 : 100.0 * reuse / api_calls;
  return absl::StrFormat("%u (%5.2f%%)", reuse, percent);
}

static std::string PadLeft(std::string value, size_t width) {
  if (value.size() < width) {
    value.insert(0, width - value.size(), ' ');
  }
  return value;
}

static std::string PadRight(std::string value, size_t width) {
  if (value.size() < width) {
    value.append(width - value.size(), ' ');
  }
  return value;
}

static void AppendDebugStatsSeparator(std::string* table,
                                      const DebugStatsWidths& widths) {
  absl::StrAppend(table, "|", std::string(widths.operation + 2, '-'), "|",
                  std::string(widths.api_calls + 2, '-'), "|",
                  std::string(widths.allocation_reuse + 2, '-'), "|",
                  std::string(widths.allocator_va_reuse + 2, '-'), "|",
                  std::string(widths.reservation_va_reuse + 2, '-'), "|\n");
}

static void AppendDebugStatsRow(std::string* table, const DebugStatsRow& row,
                                const DebugStatsWidths& widths) {
  absl::StrAppend(
      table, "| ", PadRight(row.operation, widths.operation), " | ",
      PadLeft(row.api_calls, widths.api_calls), " | ",
      PadLeft(row.allocation_reuse, widths.allocation_reuse), " | ",
      PadLeft(row.allocator_va_reuse, widths.allocator_va_reuse), " | ",
      PadLeft(row.reservation_va_reuse, widths.reservation_va_reuse), " |\n");
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
  if (VLOG_IS_ON(1)) {
    VLOG(1) << "\n" << DebugStatsTable();
  }

  for (auto& [ordinal, state] : per_device_) {
    // Briefly acquire the lock to read the last pending seqno.
    uint64_t last_seqno = 0;
    {
      absl::MutexLock lock(state->mu);
      if (!state->pending_deallocations.empty()) {
        last_seqno = state->pending_deallocations.back().seqno;
      }
    }

    if (last_seqno > 0) {
      absl::MutexLock lock(state->mu);
      WaitAndDrainPendingDeallocationsUntilSeqno(*state, last_seqno);
    }

    // Free platform-specific per-device resources (e.g. pinned timeline).
    if (state->destroy_fn) {
      state->destroy_fn();
    }
  }
}

std::string DeviceAddressVmmAllocator::DebugStatsTable() const {
  auto load = [](const std::atomic<uint64_t>& counter) {
    return counter.load(std::memory_order_relaxed);
  };

  const uint64_t allocate_calls = load(debug_stats_.allocate_calls);
  const uint64_t mapped_allocate_without_va_calls =
      load(debug_stats_.mapped_allocate_without_va_calls);
  const uint64_t mapped_allocate_with_va_calls =
      load(debug_stats_.mapped_allocate_with_va_calls);
  const uint64_t map_calls = load(debug_stats_.map_calls);

  std::array<DebugStatsRow, 7> rows = {{
      {"Operation", "API Calls", "Allocation Reuse", "Allocator VA Reuse",
       "Reservation VA Reuse"},
      {"Allocate(size)", DebugStatsCount(allocate_calls),
       DebugStatsReuse(load(debug_stats_.allocate_allocation_reuse),
                       allocate_calls),
       "-", "-"},
      {"Allocate(..., allocate_va_address=false)",
       DebugStatsCount(mapped_allocate_without_va_calls),
       DebugStatsReuse(
           load(debug_stats_.mapped_allocate_without_va_allocation_reuse),
           mapped_allocate_without_va_calls),
       DebugStatsReuse(
           load(debug_stats_.mapped_allocate_without_va_allocator_va_reuse),
           mapped_allocate_without_va_calls),
       "-"},
      {"Allocate(..., allocate_va_address=true)",
       DebugStatsCount(mapped_allocate_with_va_calls),
       DebugStatsReuse(
           load(debug_stats_.mapped_allocate_with_va_allocation_reuse),
           mapped_allocate_with_va_calls),
       DebugStatsReuse(
           load(debug_stats_.mapped_allocate_with_va_allocator_va_reuse),
           mapped_allocate_with_va_calls),
       DebugStatsReuse(
           load(debug_stats_.mapped_allocate_with_va_reservation_va_reuse),
           mapped_allocate_with_va_calls)},
      {"Map(addr, reservation, ...)", DebugStatsCount(map_calls), "-", "-",
       DebugStatsReuse(load(debug_stats_.map_reservation_va_reuse), map_calls)},
      {"Deallocate()", DebugStatsCount(load(debug_stats_.deallocate_calls)),
       "-", "-", "-"},
      {"UnMap()", DebugStatsCount(load(debug_stats_.unmap_calls)), "-", "-",
       "-"},
  }};

  DebugStatsWidths widths{
      /*operation=*/0,
      /*api_calls=*/0,
      /*allocation_reuse=*/0,
      /*allocator_va_reuse=*/0,
      /*reservation_va_reuse=*/0,
  };
  for (const DebugStatsRow& row : rows) {
    widths.operation = std::max(widths.operation, row.operation.size());
    widths.api_calls = std::max(widths.api_calls, row.api_calls.size());
    widths.allocation_reuse =
        std::max(widths.allocation_reuse, row.allocation_reuse.size());
    widths.allocator_va_reuse =
        std::max(widths.allocator_va_reuse, row.allocator_va_reuse.size());
    widths.reservation_va_reuse =
        std::max(widths.reservation_va_reuse, row.reservation_va_reuse.size());
  }

  std::string table = "DeviceAddressVmmAllocator debug statistics:\n";
  AppendDebugStatsSeparator(&table, widths);
  AppendDebugStatsRow(&table, rows[0], widths);
  AppendDebugStatsSeparator(&table, widths);
  for (size_t i = 1; i < rows.size(); ++i) {
    AppendDebugStatsRow(&table, rows[i], widths);
  }
  AppendDebugStatsSeparator(&table, widths);
  return table;
}

// Common helpers and accessors.

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

uint64_t DeviceAddressVmmAllocator::RoundUpToGranularity(
    const PerDeviceState& state, uint64_t size) const {
  if (state.allocation_granularity == 0) {
    return size;
  }
  return ((size + state.allocation_granularity - 1) /
          state.allocation_granularity) *
         state.allocation_granularity;
}

absl::StatusOr<Stream*> DeviceAddressVmmAllocator::GetStream(
    int device_ordinal) {
  TF_ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));
  return state->stream;
}

absl::Status DeviceAddressVmmAllocator::SynchronizePendingOperations(
    int device_ordinal) {
  TF_ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));
  absl::MutexLock lock(state->mu);
  if (state->pending_deallocations.empty()) {
    return absl::OkStatus();
  }
  WaitAndDrainPendingDeallocationsUntilSeqno(
      *state, state->pending_deallocations.back().seqno);
  return absl::OkStatus();
}

absl::StatusOr<StreamExecutor*> DeviceAddressVmmAllocator::GetStreamExecutor(
    int device_ordinal) const {
  TF_ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));
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
  auto allocation_it = state->records_by_allocator_address.find(addr.opaque());
  if (allocation_it != state->records_by_allocator_address.end() &&
      allocation_it->second->allocator_active &&
      allocation_it->second->allocator_address.IsSameAs(addr)) {
    return allocation_it->second->raw_allocation.get();
  }
  auto reservation_it = state->active_reservation_records.find(addr.opaque());
  if (reservation_it != state->active_reservation_records.end()) {
    return reservation_it->second->raw_allocation.get();
  }
  return nullptr;
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

// Allocate helpers.

void* DeviceAddressVmmAllocator::TrackAllocatorAddressMappedAllocation(
    PerDeviceState& state, PendingDeallocationKind kind,
    DeviceAddressBase allocator_address,
    std::shared_ptr<MemoryAllocation> raw_allocation,
    std::unique_ptr<MemoryReservation> reservation,
    MemoryReservation::ScopedMapping mapping, uint64_t allocated_size) {
  void* va_ptr = allocator_address.opaque();
  auto record = std::make_unique<AllocationRecord>();
  record->kind = kind;
  record->allocator_address = allocator_address;
  record->raw_allocation = std::move(raw_allocation);
  record->allocator_address_reservation = std::move(reservation);
  record->allocator_address_mapping.emplace(std::move(mapping));
  record->allocator_active = true;
  auto insert_result =
      state.records_by_allocator_address.emplace(va_ptr, std::move(record));
  CHECK(insert_result.second);
  state.pa_allocated += allocated_size;
  return va_ptr;
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
      "(allocator-address allocation, reservation-backed allocation, "
      "allocation plus reservation mapping, reservation unmap): "
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
  // First try to reactivate a compatible pending deallocation without waiting.
  // Reuse is stream-order safe and avoids both a fresh VMM allocation and any
  // host-side wait for the GPU timeline.
  std::optional<DeviceAddressBase> reused = try_reuse();
  if (reused.has_value()) {
    return *reused;
  }

  // If no pending entry matches, try the normal fresh allocation path. Most
  // calls should finish here; the reclaim paths below are only for PA budget
  // pressure or allocator-level allocation failures.
  absl::StatusOr<DeviceAddressBase> result = try_fresh();
  LogFreshAllocationResult(state, "initial", reclaim_size, result);

  if (absl::IsResourceExhausted(result.status())) {
    // A ResourceExhausted error may be stale: some pending deallocations can
    // already be past their stream timeline point. Complete ready allocator
    // deallocations first, without blocking for later pending work and without
    // destroying unrelated stale reservation mappings that may be reused.
    LogPendingReclaim(state, "processing completed", reclaim_size);
    CompleteReadyAllocatorDeallocationsForReclaim(
        state, LoadTimeline(state.pinned_timeline));
    result = try_fresh();
    LogFreshAllocationResult(state, "post-completed-reclaim retry",
                             reclaim_size, result);
  }

  if (absl::IsResourceExhausted(result.status())) {
    // If completed pending work was not enough, wait until enough queued frees
    // should be reclaimable for this request, then retry once more. This is the
    // only path that may block while the GPU drains earlier stream work.
    LogPendingReclaim(state, "waiting for", reclaim_size);
    // Select enough pending allocator-address deallocations to cover this
    // request, then wait for the selected tail seqno to become safe. Unrelated
    // kMap entries do not own physical memory, so leave them stale and
    // reusable.
    if (!state.pending_deallocations.empty()) {
      uint64_t accumulated_size = 0;
      uint64_t rounded_size = RoundUpToGranularity(state, reclaim_size);
      uint64_t target_seqno = 0;
      std::vector<PendingDeallocationKey> selected;

      // Target 1.1x the requested size to provide some headroom.
      uint64_t target_size = rounded_size + rounded_size / 10;

      for (const PendingDeallocation& pending : state.pending_deallocations) {
        if (pending.kind == PendingDeallocationKind::kMap) {
          continue;
        }
        auto record_it =
            state.records_by_allocator_address.find(pending.addr.opaque());
        CHECK(record_it != state.records_by_allocator_address.end());
        CHECK(record_it->second->allocator_stale);
        CHECK(record_it->second->allocator_address.IsSameAs(pending.addr));
        CHECK(record_it->second->raw_allocation != nullptr);
        accumulated_size += RoundUpToGranularity(
            state, record_it->second->raw_allocation->address().size());
        target_seqno = std::max(target_seqno, pending.seqno);
        selected.push_back(
            PendingDeallocationKey{pending.kind, pending.seqno, pending.addr});
        if (accumulated_size >= target_size) {
          break;
        }
      }

      if (!selected.empty()) {
        WaitUntilSeqno(state, target_seqno);
        for (const PendingDeallocationKey& key : selected) {
          CompletePendingDeallocationByKey(state, key);
        }
      }
    }
    result = try_fresh();
    LogFreshAllocationResult(state, "post-wait-reclaim retry", reclaim_size,
                             result);
  }

  return result;
}

// Allocate() reuses pending kAllocate entries, otherwise tries a fresh
// allocator-address mapping.
absl::StatusOr<ScopedDeviceAddress<uint8_t>>
DeviceAddressVmmAllocator::Allocate(int device_ordinal, uint64_t size,
                                    bool /*retry_on_failure*/,
                                    int64_t /*memory_space*/) {
  debug_stats_.allocate_calls.fetch_add(1, std::memory_order_relaxed);
  if (size == 0) {
    return ScopedDeviceAddress<uint8_t>(DeviceAddressBase(), device_ordinal,
                                        this);
  }

  TF_ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));

  absl::MutexLock lock(state->mu);
  auto try_reuse =
      [&]() ABSL_NO_THREAD_SAFETY_ANALYSIS -> std::optional<DeviceAddressBase> {
    uint64_t rounded_size = RoundUpToGranularity(*state, size);
    for (auto it = state->pending_deallocations.begin();
         it != state->pending_deallocations.end(); ++it) {
      if (it->kind != PendingDeallocationKind::kAllocate) {
        continue;
      }
      auto record_it =
          state->records_by_allocator_address.find(it->addr.opaque());
      CHECK(record_it != state->records_by_allocator_address.end());
      AllocationRecord& record = *record_it->second;
      CHECK(record.allocator_stale);
      CHECK(record.allocator_address.IsSameAs(it->addr));
      CHECK_EQ(record.kind, PendingDeallocationKind::kAllocate);
      if (RoundUpToGranularity(*state, record.allocator_address.size()) !=
          rounded_size) {
        continue;
      }

      DeviceAddressBase reused_mem(record.allocator_address.opaque(), size);
      VLOG(3) << absl::StreamFormat(
          "Reusing pending kAllocate: address=%p "
          "original_size=%uB "
          "new_size=%uB rounded_size=%uB device=%d",
          reused_mem.opaque(), record.allocator_address.size(), size,
          rounded_size, state->executor->device_ordinal());
      debug_stats_.allocate_allocation_reuse.fetch_add(
          1, std::memory_order_relaxed);
      const uint64_t pending_seqno = it->seqno;
      MoveAllocatorRecordToActive(*state, record, size);
      if (record.reservation_stale &&
          record.reservation_stale_seqno == pending_seqno) {
        // Reusing the allocator address cancels only allocator teardown. If
        // Deallocate() also auto-staled a reservation alias at the same stream
        // point, preserve that unmap as its own pending operation.
        CHECK(record.reservation_address.has_value());
        *it = PendingDeallocation{PendingDeallocationKind::kMap, pending_seqno,
                                  *record.reservation_address};
      } else {
        state->pending_deallocations.erase(it);
      }

      return std::optional<DeviceAddressBase>(reused_mem);
    }

    return std::optional<DeviceAddressBase>();
  };
  auto try_fresh =
      [&]()
          ABSL_NO_THREAD_SAFETY_ANALYSIS -> absl::StatusOr<DeviceAddressBase> {
    uint64_t rounded_size = RoundUpToGranularity(*state, size);
    if (state->pa_allocated + rounded_size > state->pa_budget) {
      return absl::StatusOr<DeviceAddressBase>(
          absl::ResourceExhaustedError(absl::StrFormat(
              "Not enough PA budget for allocation: pa_allocated=%uB, "
              "rounded_size=%uB, pa_budget=%uB",
              state->pa_allocated, rounded_size, state->pa_budget)));
    }

    TF_ASSIGN_OR_RETURN(auto raw_alloc,
                        CreateAllocation(state->executor, size));
    const uint64_t padded_size = raw_alloc->address().size();

    TF_ASSIGN_OR_RETURN(auto reservation,
                        CreateReservation(state->executor, size));

    TF_ASSIGN_OR_RETURN(
        auto scoped_mapping,
        reservation->MapTo(/*reservation_offset=*/0, /*allocation_offset=*/0,
                           padded_size, *raw_alloc));

    auto shared_raw = std::shared_ptr<MemoryAllocation>(std::move(raw_alloc));
    DeviceAddressBase allocator_address(reservation->address().opaque(), size);
    void* va_ptr = TrackAllocatorAddressMappedAllocation(
        *state, PendingDeallocationKind::kAllocate, allocator_address,
        std::move(shared_raw), std::move(reservation),
        std::move(scoped_mapping), rounded_size);
    // Return the original requested size, not the padded size.
    return absl::StatusOr<DeviceAddressBase>(DeviceAddressBase(va_ptr, size));
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

// Mapped Allocate() reuses matching pending mapped deallocations, otherwise
// tries fresh physical allocation and maps it into the caller reservation.
absl::StatusOr<ScopedDeviceAddress<uint8_t>>
DeviceAddressVmmAllocator::Allocate(
    int device_ordinal, uint64_t allocation_size, bool /*retry_on_failure*/,
    int64_t /*memory_space*/, MemoryReservation* reservation,
    uint64_t reservation_offset, uint64_t mapping_size,
    bool allocate_va_address) {
  if (allocate_va_address) {
    debug_stats_.mapped_allocate_with_va_calls.fetch_add(
        1, std::memory_order_relaxed);
  } else {
    debug_stats_.mapped_allocate_without_va_calls.fetch_add(
        1, std::memory_order_relaxed);
  }

  // Keep zero-sized mapped allocation consistent with regular Allocate(): no
  // physical allocation or mapping is created, so the requested mapping size
  // must also be zero.
  if (allocation_size == 0) {
    if (mapping_size != 0) {
      return absl::InvalidArgumentError(
          "mapping_size must be zero when allocation_size is zero");
    }
    return ScopedDeviceAddress<uint8_t>(DeviceAddressBase(), device_ordinal,
                                        this);
  }
  // A mapped allocation with a nonzero physical allocation must establish a
  // nonempty mapping into the caller-owned reservation.
  if (mapping_size == 0) {
    return absl::InvalidArgumentError(
        "mapping_size must be nonzero for mapped Allocate");
  }
  if (allocation_size != mapping_size) {
    return absl::InvalidArgumentError(
        "allocation_size must equal mapping_size for mapped Allocate");
  }

  TF_ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));

  // Validate the caller-owned reservation slice before taking the allocator
  // lock. `reservation_address` is the VA that must either be reactivated from
  // a pending deallocation or freshly mapped below.
  TF_ASSIGN_OR_RETURN(
      DeviceAddressBase reservation_address,
      ValidateReservationRange(reservation, reservation_offset, mapping_size));

  absl::MutexLock lock(state->mu);
  // First try to satisfy the request from a compatible pending deallocation
  // for the same reservation-derived returned allocator address.
  auto try_reuse =
      [&]() ABSL_NO_THREAD_SAFETY_ANALYSIS -> std::optional<DeviceAddressBase> {
    if (allocate_va_address) {
      // This mode returns a distinct allocator-owned VA while also mapping that
      // allocation into the caller-owned reservation. Reuse is possible only
      // when a pending kAllocateWithAddressAndMap record still has both sides
      // stale and its reservation side exactly matches this request.
      for (auto it = state->pending_deallocations.begin();
           it != state->pending_deallocations.end(); ++it) {
        if (it->kind != PendingDeallocationKind::kAllocateWithAddressAndMap) {
          continue;
        }
        auto record_it =
            state->records_by_allocator_address.find(it->addr.opaque());
        CHECK(record_it != state->records_by_allocator_address.end());
        AllocationRecord& record = *record_it->second;
        CHECK(record.allocator_stale);
        CHECK(record.allocator_address.IsSameAs(it->addr));
        CHECK_EQ(record.kind,
                 PendingDeallocationKind::kAllocateWithAddressAndMap);
        if (!record.reservation_stale) {
          continue;
        }
        CHECK(record.reservation_address.has_value());
        // The allocator address can be reused for command-buffer update-free
        // execution only if the external reservation VA is also the same VA the
        // command buffer captured.
        if (!record.reservation_address->IsSameAs(reservation_address)) {
          continue;
        }
        if (record.raw_allocation->address().size() < allocation_size) {
          // The old mapping is the right VA but not enough physical memory.
          // Wait for its deferred teardown to finish, then let the fresh path
          // create a larger allocation and install a new mapping.
          WaitAndCompleteStaleAllocatorDeallocation(
              *state,
              PendingDeallocationKey{record.kind, record.allocator_stale_seqno,
                                     record.allocator_address});
          return std::nullopt;
        }

        DeviceAddressBase reused_mem(record.allocator_address.opaque(),
                                     allocation_size);
        VLOG(3) << absl::StreamFormat(
            "Reusing pending kAllocateWithAddressAndMap: allocator_address=%p "
            "reservation_address=%p size=%uB device=%d",
            reused_mem.opaque(), reservation_address.opaque(), allocation_size,
            state->executor->device_ordinal());
        debug_stats_.mapped_allocate_with_va_allocation_reuse.fetch_add(
            1, std::memory_order_relaxed);
        debug_stats_.mapped_allocate_with_va_allocator_va_reuse.fetch_add(
            1, std::memory_order_relaxed);
        debug_stats_.mapped_allocate_with_va_reservation_va_reuse.fetch_add(
            1, std::memory_order_relaxed);
        // Reactivate both aliases: the returned allocator VA and the external
        // reservation VA. This cancels the pending allocator teardown and the
        // paired pending kMap unmap for the reservation mapping.
        MoveAllocatorRecordToActive(*state, record, allocation_size);
        MoveReservationRecordToActive(*state, record);
        state->pending_deallocations.erase(it);
        ErasePendingDeallocation(*state, PendingDeallocationKind::kMap,
                                 reservation_address);
        return reused_mem;
      }
      return std::nullopt;
    }

    // Look for a pending deallocation that already owns the requested
    // reservation VA as its returned allocator address. Reusing it keeps the
    // same virtual address mapped and avoids waiting for the GPU timeline when
    // the pending raw allocation is compatible with this request.
    auto record_it =
        state->records_by_allocator_address.find(reservation_address.opaque());
    if (record_it == state->records_by_allocator_address.end()) {
      return std::nullopt;
    }
    AllocationRecord& record = *record_it->second;
    if (!record.allocator_stale) {
      return std::nullopt;
    }
    if (record.kind != PendingDeallocationKind::kAllocateWithMap) {
      return std::nullopt;
    }
    if (!record.allocator_address.IsSameAs(reservation_address)) {
      return std::nullopt;
    }

    // Allocate(..., allocate_va_address=false) returns the reservation VA as an
    // owning allocator address. If the pending raw allocation is too small for
    // the new request, wait for the old mapping to drain so the fresh path can
    // remap this reservation VA to a larger raw allocation.
    if (record.raw_allocation->address().size() < allocation_size) {
      WaitAndCompleteStaleAllocatorDeallocation(
          *state,
          PendingDeallocationKey{record.kind, record.allocator_stale_seqno,
                                 record.allocator_address});
      return std::nullopt;
    }

    VLOG(3) << absl::StreamFormat(
        "Reusing pending reservation-backed allocation: address=%p size=%uB "
        "device=%d",
        reservation_address.opaque(), reservation_address.size(),
        state->executor->device_ordinal());
    debug_stats_.mapped_allocate_without_va_allocation_reuse.fetch_add(
        1, std::memory_order_relaxed);
    debug_stats_.mapped_allocate_without_va_allocator_va_reuse.fetch_add(
        1, std::memory_order_relaxed);
    auto pending_it = state->pending_deallocations.end();
    // The record is indexed by allocator address, but the FIFO queue owns the
    // stream-ordered teardown. Find the queue entry so reuse can either cancel
    // it or narrow it to the reservation unmap that still needs to happen.
    for (auto it = state->pending_deallocations.begin();
         it != state->pending_deallocations.end(); ++it) {
      if (it->kind == PendingDeallocationKind::kAllocateWithMap &&
          it->addr.IsSameAs(reservation_address)) {
        pending_it = it;
        break;
      }
    }
    CHECK(pending_it != state->pending_deallocations.end());
    const uint64_t pending_seqno = pending_it->seqno;
    MoveAllocatorRecordToActive(*state, record, allocation_size);
    if (record.reservation_stale &&
        record.reservation_stale_seqno == pending_seqno) {
      // Reusing the allocator address cancels only allocator teardown. If
      // Deallocate() also auto-staled a reservation alias at the same stream
      // point, preserve that unmap as its own pending operation.
      CHECK(record.reservation_address.has_value());
      *pending_it =
          PendingDeallocation{PendingDeallocationKind::kMap, pending_seqno,
                              *record.reservation_address};
    } else {
      state->pending_deallocations.erase(pending_it);
    }
    return reservation_address;
  };
  // If no pending entry can be reused, allocate fresh physical memory and map
  // it into the caller reservation. When allocate_va_address is true this also
  // creates an allocator-owned address; otherwise the reservation address is
  // the returned allocator address.
  auto try_fresh =
      [&]()
          ABSL_NO_THREAD_SAFETY_ANALYSIS -> absl::StatusOr<DeviceAddressBase> {
    // If the requested reservation VA is only present in the deferred queue,
    // wait for that queued unmap/deallocation to complete before installing a
    // fresh mapping. Active mappings are still rejected below.
    while (true) {
      // Partial overlaps are never reusable: the allocator tracks whole mapped
      // ranges, so a caller must request the exact same reservation slice
      // before stale state can be waited on or reactivated.
      if (auto overlap = FindOverlappingRecord(
              *state, reservation_address, /*include_allocator=*/true,
              /*include_reservation=*/true, /*include_active=*/true,
              /*include_stale=*/true, /*exact_only=*/false,
              /*partial_only=*/true)) {
        return absl::FailedPreconditionError(absl::StrFormat(
            "reservation range at %p (%uB) partially overlaps %s %s range at "
            "%p "
            "(%uB); reservation mappings must be managed with the same full "
            "range",
            reservation_address.opaque(), reservation_address.size(),
            overlap->is_active ? "active" : "stale",
            overlap->is_allocator ? "allocator" : "reservation",
            overlap->tracked_address.opaque(),
            overlap->tracked_address.size()));
      }
      // An exact stale overlap means the previous mapping for this reservation
      // VA is still protected by stream order. Complete only that conflicting
      // stale record, then rescan because another thread may have changed the
      // allocator state while the lock was released.
      auto stale_overlap = FindOverlappingRecord(
          *state, reservation_address, /*include_allocator=*/true,
          /*include_reservation=*/true, /*include_active=*/false,
          /*include_stale=*/true, /*exact_only=*/true,
          /*partial_only=*/false);
      if (!stale_overlap.has_value()) {
        break;
      }
      WaitAndCompleteStaleOverlap(*state, *stale_overlap);
    }

    // At this point stale exact overlaps have been drained. Any remaining
    // overlap is active ownership of the requested reservation range and must
    // be reported as a duplicate mapping attempt instead of remapped underneath
    // existing users.
    if (FindOverlappingRecord(*state, reservation_address,
                              /*include_allocator=*/true,
                              /*include_reservation=*/true,
                              /*include_active=*/true,
                              /*include_stale=*/false,
                              /*exact_only=*/false,
                              /*partial_only=*/false)
            .has_value()) {
      return absl::AlreadyExistsError(absl::StrFormat(
          "reservation range is already tracked at virtual address %p",
          reservation_address.opaque()));
    }

    uint64_t rounded_size = RoundUpToGranularity(*state, allocation_size);
    if (!allocate_va_address) {
      // The returned allocator address is the caller-owned reservation VA. The
      // record is keyed by that VA and owns only the raw allocation plus the
      // scoped mapping into the external reservation.
      if (state->pa_allocated + rounded_size > state->pa_budget) {
        return absl::ResourceExhaustedError(absl::StrFormat(
            "Not enough PA budget for mapping: pa_allocated=%uB, "
            "rounded_size=%uB, pa_budget=%uB",
            state->pa_allocated, rounded_size, state->pa_budget));
      }

      TF_ASSIGN_OR_RETURN(auto raw_alloc,
                          CreateAllocation(state->executor, allocation_size));
      if (mapping_size > raw_alloc->address().size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "physical allocation is smaller than requested mapping: "
            "allocation_size=%uB, mapping_size=%uB",
            raw_alloc->address().size(), mapping_size));
      }

      TF_ASSIGN_OR_RETURN(
          auto scoped_mapping,
          reservation->MapTo(reservation_offset, /*allocation_offset=*/0,
                             mapping_size, *raw_alloc));
      auto shared_raw = std::shared_ptr<MemoryAllocation>(std::move(raw_alloc));

      TrackAllocatorAddressMappedAllocation(
          *state, PendingDeallocationKind::kAllocateWithMap,
          reservation_address, std::move(shared_raw), nullptr,
          std::move(scoped_mapping), rounded_size);

      return reservation_address;
    }

    // This mode creates two VAs for the same raw allocation: an allocator-owned
    // VA returned to the caller, and a non-owning alias in the caller
    // reservation used by captured command buffers.
    if (state->pa_allocated + rounded_size > state->pa_budget) {
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Not enough PA budget for allocation: pa_allocated=%uB, "
          "rounded_size=%uB, pa_budget=%uB",
          state->pa_allocated, rounded_size, state->pa_budget));
    }

    TF_ASSIGN_OR_RETURN(auto raw_alloc,
                        CreateAllocation(state->executor, allocation_size));
    const uint64_t padded_size = raw_alloc->address().size();
    if (mapping_size > padded_size) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "mapping size must not exceed physical allocation size: "
          "mapping_size=%uB, allocation_size=%uB",
          mapping_size, padded_size));
    }

    TF_ASSIGN_OR_RETURN(auto allocator_address_reservation,
                        CreateReservation(state->executor, allocation_size));
    TF_ASSIGN_OR_RETURN(auto allocator_address_mapping,
                        allocator_address_reservation->MapTo(
                            /*reservation_offset=*/0, /*allocation_offset=*/0,
                            padded_size, *raw_alloc));
    TF_ASSIGN_OR_RETURN(
        auto reservation_address_mapping,
        reservation->MapTo(reservation_offset, /*allocation_offset=*/0,
                           mapping_size, *raw_alloc));

    auto shared_raw = std::shared_ptr<MemoryAllocation>(std::move(raw_alloc));
    // Record the paired allocation: the allocator-owned returned VA owns the
    // raw allocation, and the caller reservation VA is a non-owning alias.
    void* allocator_va = allocator_address_reservation->address().opaque();
    auto record = std::make_unique<AllocationRecord>();
    DeviceAddressBase allocator_address(allocator_va, allocation_size);
    record->kind = PendingDeallocationKind::kAllocateWithAddressAndMap;
    record->allocator_address = allocator_address;
    record->raw_allocation = std::move(shared_raw);
    record->allocator_address_reservation =
        std::move(allocator_address_reservation);
    record->allocator_address_mapping.emplace(
        std::move(allocator_address_mapping));
    record->reservation_address = reservation_address;
    record->reservation_address_mapping.emplace(
        std::move(reservation_address_mapping));
    record->allocator_active = true;
    record->reservation_active = true;
    AllocationRecord* record_ptr = record.get();
    auto record_insert = state->records_by_allocator_address.emplace(
        allocator_va, std::move(record));
    CHECK(record_insert.second);
    auto reservation_insert = state->active_reservation_records.emplace(
        reservation_address.opaque(), record_ptr);
    CHECK(reservation_insert.second);
    state->pa_allocated += rounded_size;

    return DeviceAddressBase(allocator_va, allocation_size);
  };

  // The shared retry helper handles PA-budget pressure: try reuse, try fresh,
  // complete already-finished pending work on ResourceExhausted, and finally
  // wait for enough pending deallocations only if necessary.
  absl::StatusOr<DeviceAddressBase> result =
      TryWithPendingReclaim(*state, allocation_size, try_reuse, try_fresh);

  if (!result.ok()) {
    return result.status();
  }

  // For allocate_va_address=false this is `reservation_address`; for
  // allocate_va_address=true it is the allocator-owned address paired with the
  // reservation mapping.
  return ScopedDeviceAddress<uint8_t>(*result, device_ordinal, this);
}

absl::Status DeviceAddressVmmAllocator::Deallocate(int device_ordinal,
                                                   DeviceAddressBase mem) {
  debug_stats_.deallocate_calls.fetch_add(1, std::memory_order_relaxed);
  if (mem.is_null()) {
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));

  absl::MutexLock lock(state->mu);

  auto record_it = state->records_by_allocator_address.find(mem.opaque());
  if (record_it == state->records_by_allocator_address.end() ||
      !record_it->second->allocator_active ||
      !record_it->second->allocator_address.IsSameAs(mem)) {
    return absl::NotFoundError(absl::StrFormat(
        "virtual address %p is not an active allocator address returned by "
        "Allocate()",
        mem.opaque()));
  }
  AllocationRecord& record = *record_it->second;
  CHECK(!state->active_reservation_records.contains(mem.opaque()));

  VLOG(3) << absl::StreamFormat(
      "Queueing deferred deallocation for virtual address %p (size=%uB) "
      "on device ordinal %d",
      mem.opaque(), mem.size(), state->executor->device_ordinal());

  // Reserve the stream timeline sequence for this deferred Deallocate and
  // enqueue the platform marker that will make the stale allocator address safe
  // to destroy.
  uint64_t seqno = state->next_seqno++;
  TF_RETURN_IF_ERROR(EnqueueDeferredDeallocation(*state, seqno));
  if (record.reservation_active) {
    MoveReservationRecordToStale(*state, record, seqno);
  }
  // Move the returned allocator address out of active ownership and keep its
  // mapping alive as stale state until the stream reaches `seqno`.
  CHECK(record.allocator_active);
  CHECK(!record.allocator_stale);
  CHECK(record.allocator_address_mapping.has_value());
  void* allocator_va = record.allocator_address.opaque();
  auto allocator_record_it =
      state->records_by_allocator_address.find(allocator_va);
  CHECK(allocator_record_it != state->records_by_allocator_address.end());
  CHECK_EQ(allocator_record_it->second.get(), &record);
  record.allocator_active = false;
  record.allocator_stale = true;
  record.allocator_stale_seqno = seqno;
  state->pending_deallocations.push_back(
      PendingDeallocation{record.kind, seqno, record.allocator_address});
  return absl::OkStatus();
}

// Map helpers.

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

std::optional<DeviceAddressVmmAllocator::OverlappingRecord>
DeviceAddressVmmAllocator::FindOverlappingRecord(
    PerDeviceState& state, DeviceAddressBase address, bool include_allocator,
    bool include_reservation, bool include_active, bool include_stale,
    bool exact_only, bool partial_only) const {
  CHECK(!(exact_only && partial_only));

  auto matches = [&](DeviceAddressBase tracked_address) {
    if (exact_only) {
      return tracked_address.IsSameAs(address);
    }
    if (partial_only) {
      // Partial overlap means the ranges intersect but are not the same full
      // ownership range.
      return AddressRangesOverlap(tracked_address, address) &&
             !tracked_address.IsSameAs(address);
    }
    return AddressRangesOverlap(tracked_address, address);
  };

  auto check_record = [&](AllocationRecord* record,
                          DeviceAddressBase tracked_address, bool is_allocator,
                          bool is_active) -> std::optional<OverlappingRecord> {
    if (matches(tracked_address)) {
      return OverlappingRecord{record, tracked_address, is_allocator,
                               is_active};
    }
    return std::nullopt;
  };

  if (include_allocator) {
    for (const auto& [_, record_owner] : state.records_by_allocator_address) {
      AllocationRecord* record = record_owner.get();
      CHECK_NE(record->allocator_active, record->allocator_stale);
      bool include_record = (include_active && record->allocator_active) ||
                            (include_stale && record->allocator_stale);
      if (!include_record) {
        continue;
      }
      if (auto overlap = check_record(record, record->allocator_address,
                                      /*is_allocator=*/true,
                                      /*is_active=*/record->allocator_active)) {
        return overlap;
      }
    }
  }
  if (include_reservation && include_active) {
    for (const auto& [_, record] : state.active_reservation_records) {
      CHECK(record->reservation_address.has_value());
      if (auto overlap =
              check_record(record, *record->reservation_address,
                           /*is_allocator=*/false, /*is_active=*/true)) {
        return overlap;
      }
    }
  }
  if (include_reservation && include_stale) {
    for (const auto& [_, record] : state.stale_reservation_records) {
      CHECK(record->reservation_address.has_value());
      if (auto overlap =
              check_record(record, *record->reservation_address,
                           /*is_allocator=*/false, /*is_active=*/false)) {
        return overlap;
      }
    }
  }

  return std::nullopt;
}

absl::Status DeviceAddressVmmAllocator::Map(int device_ordinal,
                                            DeviceAddressBase addr,
                                            MemoryReservation* reservation,
                                            uint64_t reservation_offset,
                                            uint64_t size) {
  debug_stats_.map_calls.fetch_add(1, std::memory_order_relaxed);
  TF_ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));
  if (size == 0) {
    return absl::OkStatus();
  }
  if (addr.is_null()) {
    return absl::InvalidArgumentError("addr must not be null");
  }

  // Map() does not allocate a VA range. It maps the physical allocation backing
  // `addr` into the caller-owned reservation slice, so validate the slice
  // before taking the allocator lock.
  TF_ASSIGN_OR_RETURN(
      DeviceAddressBase reservation_address,
      ValidateReservationRange(reservation, reservation_offset, size));

  absl::MutexLock lock(state->mu);
  auto resolve_source_record =
      [&]()
          ABSL_NO_THREAD_SAFETY_ANALYSIS -> absl::StatusOr<AllocationRecord*> {
    auto allocation_it =
        state->records_by_allocator_address.find(addr.opaque());
    if (allocation_it == state->records_by_allocator_address.end() ||
        !allocation_it->second->allocator_active ||
        !allocation_it->second->allocator_address.IsSameAs(addr)) {
      return absl::NotFoundError(absl::StrFormat(
          "addr %p is not an active allocator address, when trying to "
          "do map of VA reservation to existing physical allocation, we "
          "requires the buffer being mapped to is being allocated through "
          "DeviceAddressVmmAllocator, check the allocator type for the "
          "buffer.",
          addr.opaque()));
    }
    return allocation_it->second.get();
  };

  // Resolve the source address to the raw physical allocation that is currently
  // mapped there. Any active allocator address returned by this allocator is
  // accepted as a Map() source.
  TF_ASSIGN_OR_RETURN(AllocationRecord * source_record,
                      resolve_source_record());
  MemoryAllocation* raw_allocation = source_record->raw_allocation.get();
  if (size > raw_allocation->address().size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "mapping size must not exceed physical allocation size: "
        "mapping_size=%uB, allocation_size=%uB",
        size, raw_allocation->address().size()));
  }
  auto reject_partial_overlap =
      [&]() ABSL_NO_THREAD_SAFETY_ANALYSIS -> absl::Status {
    if (auto overlap = FindOverlappingRecord(
            *state, reservation_address, /*include_allocator=*/true,
            /*include_reservation=*/true, /*include_active=*/true,
            /*include_stale=*/true, /*exact_only=*/false,
            /*partial_only=*/true)) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "reservation range at %p (%uB) partially overlaps %s %s range at %p "
          "(%uB); reservation mappings must be managed with the same full "
          "range",
          reservation_address.opaque(), reservation_address.size(),
          overlap->is_active ? "active" : "stale",
          overlap->is_allocator ? "allocator" : "reservation",
          overlap->tracked_address.opaque(), overlap->tracked_address.size()));
    }
    return absl::OkStatus();
  };
  TF_RETURN_IF_ERROR(reject_partial_overlap());

  while (true) {
    std::optional<PendingDeallocationKey> pending_completion_key;

    if (source_record->reservation_active) {
      return absl::AlreadyExistsError(absl::StrFormat(
          "allocator address %p already has an active reservation mapping at "
          "%p",
          addr.opaque(), source_record->reservation_address->opaque()));
    }

    if (source_record->reservation_stale) {
      CHECK(source_record->reservation_address.has_value());
      if (!source_record->reservation_address->IsSameAs(reservation_address)) {
        pending_completion_key =
            PendingDeallocationKey{PendingDeallocationKind::kMap,
                                   source_record->reservation_stale_seqno,
                                   *source_record->reservation_address};
      }
    }
    if (!pending_completion_key.has_value()) {
      auto stale_reservation_overlap = FindOverlappingRecord(
          *state, reservation_address, /*include_allocator=*/false,
          /*include_reservation=*/true, /*include_active=*/false,
          /*include_stale=*/true, /*exact_only=*/true,
          /*partial_only=*/false);
      if (stale_reservation_overlap.has_value()) {
        AllocationRecord& stale_record = *stale_reservation_overlap->record;

        // UnMap() defers destroying the ScopedMapping until the GPU reaches the
        // recorded stream point. If the caller maps the same reservation
        // address to the same raw allocation before that point, the old mapping
        // is still valid; move it back to the active maps instead of unmapping
        // and remapping.
        CHECK(stale_record.reservation_address.has_value());
        CHECK(stale_record.reservation_address->IsSameAs(reservation_address));
        if (stale_record.raw_allocation.get() == raw_allocation) {
          debug_stats_.map_reservation_va_reuse.fetch_add(
              1, std::memory_order_relaxed);
          MoveReservationRecordToActive(*state, stale_record);
          ErasePendingDeallocation(*state, PendingDeallocationKind::kMap,
                                   reservation_address);
          return absl::OkStatus();
        }

        // The same reservation address is waiting to unmap from a different raw
        // allocation. Creating a new mapping now would overwrite an in-flight
        // mapping that earlier GPU work may still use, so wait until that
        // deferred unmap has completed, then rescan from the start.
        pending_completion_key = PendingDeallocationKey{
            PendingDeallocationKind::kMap, stale_record.reservation_stale_seqno,
            *stale_record.reservation_address};
      } else {
        auto stale_allocator_overlap = FindOverlappingRecord(
            *state, reservation_address, /*include_allocator=*/true,
            /*include_reservation=*/false, /*include_active=*/false,
            /*include_stale=*/true, /*exact_only=*/true,
            /*partial_only=*/false);
        if (stale_allocator_overlap.has_value()) {
          AllocationRecord& stale_record = *stale_allocator_overlap->record;
          pending_completion_key = PendingDeallocationKey{
              stale_record.kind, stale_record.allocator_stale_seqno,
              stale_record.allocator_address};
        }
      }
    }

    if (!pending_completion_key.has_value()) {
      break;
    }
    if (pending_completion_key->kind == PendingDeallocationKind::kMap) {
      WaitAndCompleteStaleReservationMapping(*state, *pending_completion_key);
    } else {
      WaitAndCompleteStaleAllocatorDeallocation(*state,
                                                *pending_completion_key);
    }
    // Waiting releases the allocator lock. Another thread may have deallocated
    // or remapped `addr` while this thread was waiting, so resolve it again
    // before either reactivating another pending unmap or creating a fresh map.
    TF_ASSIGN_OR_RETURN(source_record, resolve_source_record());
    raw_allocation = source_record->raw_allocation.get();
    TF_RETURN_IF_ERROR(reject_partial_overlap());
    if (size > raw_allocation->address().size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "mapping size must not exceed physical allocation size: "
          "mapping_size=%uB, allocation_size=%uB",
          size, raw_allocation->address().size()));
    }
  }
  // A fresh Map() must have exclusive ownership of the reservation address.
  // Reject active mappings and still-stale deferred mappings before installing
  // the new ScopedMapping.
  if (FindOverlappingRecord(*state, reservation_address,
                            /*include_allocator=*/true,
                            /*include_reservation=*/true,
                            /*include_active=*/true,
                            /*include_stale=*/true,
                            /*exact_only=*/false,
                            /*partial_only=*/false)
          .has_value()) {
    return absl::AlreadyExistsError(absl::StrFormat(
        "reservation range is already tracked at virtual address %p",
        reservation_address.opaque()));
  }

  // Install the reservation address mapping to the raw physical allocation. The
  // allocation_offset is zero because Map() aliases the beginning of
  // the source allocation; callers pass the target VA location through
  // `reservation_offset`.
  TF_ASSIGN_OR_RETURN(auto mapping, reservation->MapTo(reservation_offset,
                                                       /*allocation_offset=*/0,
                                                       size, *raw_allocation));
  DeviceAddressBase mapped = mapping.mapped_address();
  // The reservation slice was computed before locking. Verify the platform
  // returned the exact reservation address before recording allocator
  // bookkeeping.
  if (!mapped.IsSameAs(reservation_address)) {
    return absl::InternalError(absl::StrFormat(
        "Map() mapped unexpected virtual address: expected=%p, actual=%p",
        reservation_address.opaque(), mapped.opaque()));
  }
  // Track this as a Map()-owned reservation alias. This only updates the
  // reservation-address index; no new physical allocation is created, so
  // pa_allocated does not change.
  CHECK(!source_record->reservation_active);
  CHECK(!source_record->reservation_stale);
  source_record->reservation_address = mapped;
  source_record->reservation_address_mapping.emplace(std::move(mapping));
  source_record->reservation_active = true;
  auto mapping_insert_result =
      state->active_reservation_records.emplace(mapped.opaque(), source_record);
  CHECK(mapping_insert_result.second);
  return absl::OkStatus();
}

// UnMap/deferred teardown helpers.

void DeviceAddressVmmAllocator::ErasePendingDeallocation(
    PerDeviceState& state, PendingDeallocationKind kind,
    DeviceAddressBase addr) {
  for (auto it = state.pending_deallocations.begin();
       it != state.pending_deallocations.end(); ++it) {
    if (it->kind == kind && it->addr.IsSameAs(addr)) {
      state.pending_deallocations.erase(it);
      return;
    }
  }
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

void DeviceAddressVmmAllocator::MoveReservationRecordToStale(
    PerDeviceState& state, AllocationRecord& record, uint64_t seqno) {
  CHECK(record.reservation_active);
  CHECK(!record.reservation_stale);
  CHECK(record.reservation_address.has_value());
  CHECK(record.reservation_address_mapping.has_value());
  void* reservation_va = record.reservation_address->opaque();
  CHECK_EQ(state.active_reservation_records.erase(reservation_va), 1);
  auto insert_result =
      state.stale_reservation_records.emplace(reservation_va, &record);
  CHECK(insert_result.second);
  record.reservation_active = false;
  record.reservation_stale = true;
  record.reservation_stale_seqno = seqno;
}

void DeviceAddressVmmAllocator::MoveReservationRecordToActive(
    PerDeviceState& state, AllocationRecord& record) {
  CHECK(!record.reservation_active);
  CHECK(record.reservation_stale);
  CHECK(record.reservation_address.has_value());
  CHECK(record.reservation_address_mapping.has_value());
  void* reservation_va = record.reservation_address->opaque();
  CHECK_EQ(state.stale_reservation_records.erase(reservation_va), 1);
  auto insert_result =
      state.active_reservation_records.emplace(reservation_va, &record);
  CHECK(insert_result.second);
  record.reservation_active = true;
  record.reservation_stale = false;
  record.reservation_stale_seqno = 0;
}

void DeviceAddressVmmAllocator::CompleteStaleReservationMapping(
    PerDeviceState& state, AllocationRecord& record) {
  if (!record.reservation_stale) {
    return;
  }
  CHECK(!record.reservation_active);
  CHECK(record.reservation_address.has_value());
  void* reservation_va = record.reservation_address->opaque();
  auto stale_it = state.stale_reservation_records.find(reservation_va);
  if (stale_it != state.stale_reservation_records.end()) {
    CHECK_EQ(stale_it->second, &record);
    state.stale_reservation_records.erase(stale_it);
  }
  record.reservation_address_mapping.reset();
  record.reservation_address.reset();
  record.reservation_stale = false;
  record.reservation_stale_seqno = 0;
}

void DeviceAddressVmmAllocator::WaitUntilSeqno(PerDeviceState& state,
                                               uint64_t target_seqno) {
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
}

void DeviceAddressVmmAllocator::WaitAndDrainPendingDeallocationsUntilSeqno(
    PerDeviceState& state, uint64_t target_seqno) {
  WaitUntilSeqno(state, target_seqno);
  while (!state.pending_deallocations.empty() &&
         state.pending_deallocations.front().seqno <= target_seqno) {
    PendingDeallocation pending = state.pending_deallocations.front();
    state.pending_deallocations.pop_front();
    CompletePendingDeallocation(state, pending);
  }
}

void DeviceAddressVmmAllocator::CompleteReadyAllocatorDeallocationsForReclaim(
    PerDeviceState& state, uint64_t completed_seqno) {
  std::vector<PendingDeallocationKey> selected;
  for (const PendingDeallocation& pending : state.pending_deallocations) {
    if (pending.seqno > completed_seqno ||
        pending.kind == PendingDeallocationKind::kMap) {
      continue;
    }
    selected.push_back(
        PendingDeallocationKey{pending.kind, pending.seqno, pending.addr});
  }
  for (const PendingDeallocationKey& key : selected) {
    CompletePendingDeallocationByKey(state, key);
  }
}

bool DeviceAddressVmmAllocator::CompletePendingDeallocationByKey(
    PerDeviceState& state, const PendingDeallocationKey& key) {
  for (auto it = state.pending_deallocations.begin();
       it != state.pending_deallocations.end(); ++it) {
    if (it->kind == key.kind && it->seqno == key.seqno &&
        it->addr.IsSameAs(key.addr)) {
      PendingDeallocation pending = *it;
      state.pending_deallocations.erase(it);
      CompletePendingDeallocation(state, pending);
      return true;
    }
  }
  return false;
}

void DeviceAddressVmmAllocator::WaitAndCompleteStaleAllocatorDeallocation(
    PerDeviceState& state, const PendingDeallocationKey& key) {
  CHECK_NE(key.kind, PendingDeallocationKind::kMap);
  WaitUntilSeqno(state, key.seqno);
  CompletePendingDeallocationByKey(state, key);
}

void DeviceAddressVmmAllocator::WaitAndCompleteStaleReservationMapping(
    PerDeviceState& state, const PendingDeallocationKey& key) {
  CHECK_EQ(key.kind, PendingDeallocationKind::kMap);
  WaitUntilSeqno(state, key.seqno);
  if (CompletePendingDeallocationByKey(state, key)) {
    return;
  }

  // Deallocate() can stale a reservation alias and queue only the allocator
  // deallocation. In that case there is no kMap queue entry to complete, but a
  // reservation-conflict waiter still needs to drop the stale mapping once its
  // stream point is reached.
  auto stale_it = state.stale_reservation_records.find(key.addr.opaque());
  if (stale_it == state.stale_reservation_records.end()) {
    return;
  }
  AllocationRecord& record = *stale_it->second;
  if (!record.reservation_stale || !record.reservation_address.has_value() ||
      record.reservation_stale_seqno != key.seqno ||
      !record.reservation_address->IsSameAs(key.addr)) {
    return;
  }
  CompleteStaleReservationMapping(state, record);
}

void DeviceAddressVmmAllocator::WaitAndCompleteStaleOverlap(
    PerDeviceState& state, const OverlappingRecord& overlap) {
  CHECK(!overlap.is_active);
  if (overlap.is_allocator) {
    AllocationRecord& record = *overlap.record;
    WaitAndCompleteStaleAllocatorDeallocation(
        state, PendingDeallocationKey{record.kind, record.allocator_stale_seqno,
                                      record.allocator_address});
    return;
  }
  CHECK(overlap.record->reservation_stale);
  CHECK(overlap.record->reservation_address.has_value());
  WaitAndCompleteStaleReservationMapping(
      state, PendingDeallocationKey{PendingDeallocationKind::kMap,
                                    overlap.record->reservation_stale_seqno,
                                    *overlap.record->reservation_address});
}

void DeviceAddressVmmAllocator::CompletePendingDeallocation(
    PerDeviceState& state, const PendingDeallocation& pending) {
  if (pending.kind == PendingDeallocationKind::kMap) {
    auto record_it =
        state.stale_reservation_records.find(pending.addr.opaque());
    CHECK(record_it != state.stale_reservation_records.end());
    CHECK_EQ(record_it->second->reservation_stale_seqno, pending.seqno);
    CompleteStaleReservationMapping(state, *record_it->second);
    return;
  }

  auto record_it =
      state.records_by_allocator_address.find(pending.addr.opaque());
  CHECK(record_it != state.records_by_allocator_address.end());
  CHECK(record_it->second->allocator_stale);
  CHECK(record_it->second->allocator_address.IsSameAs(pending.addr));
  CHECK_EQ(record_it->second->kind, pending.kind);
  CHECK_EQ(record_it->second->allocator_stale_seqno, pending.seqno);
  // Complete allocator-address teardown. If this allocation still has a stale
  // reservation alias, drop that mapping first, then release allocator VA state
  // and physical allocation accounting.
  AllocationRecord& record = *record_it->second;
  CHECK(!record.allocator_active);
  CHECK(!record.reservation_active);
  if (record.reservation_stale) {
    CHECK(record.reservation_address.has_value());
    PendingDeallocationKey reservation_key{PendingDeallocationKind::kMap,
                                           record.reservation_stale_seqno,
                                           *record.reservation_address};
    for (auto it = state.pending_deallocations.begin();
         it != state.pending_deallocations.end(); ++it) {
      if (it->kind == reservation_key.kind &&
          it->seqno == reservation_key.seqno &&
          it->addr.IsSameAs(reservation_key.addr)) {
        state.pending_deallocations.erase(it);
        break;
      }
    }
    CompleteStaleReservationMapping(state, record);
  }
  void* allocator_va = record.allocator_address.opaque();
  auto owning_record_it = state.records_by_allocator_address.find(allocator_va);
  CHECK(owning_record_it != state.records_by_allocator_address.end());
  CHECK_EQ(owning_record_it->second.get(), &record);
  record.allocator_address_mapping.reset();
  record.allocator_address_reservation.reset();

  if (record.raw_allocation != nullptr) {
    uint64_t released_size =
        RoundUpToGranularity(state, record.raw_allocation->address().size());
    DCHECK_GE(state.pa_allocated, released_size);
    state.pa_allocated -= released_size;
  }
  record.raw_allocation.reset();
  CHECK_EQ(state.records_by_allocator_address.erase(allocator_va), 1);
}

absl::Status DeviceAddressVmmAllocator::UnMap(int device_ordinal,
                                              MemoryReservation* reservation,
                                              uint64_t reservation_offset,
                                              uint64_t size) {
  debug_stats_.unmap_calls.fetch_add(1, std::memory_order_relaxed);
  TF_ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));
  if (size == 0) {
    return absl::OkStatus();
  }

  // Map() and Allocate(..., allocate_va_address=true) record reservation
  // mappings by the mapped reservation VA. Reconstruct the same reservation
  // slice here so callers do not need to hold a ScopedMapping.
  TF_ASSIGN_OR_RETURN(
      DeviceAddressBase reservation_address,
      ValidateReservationRange(reservation, reservation_offset, size));

  absl::MutexLock lock(state->mu);
  // UnMap() only accepts the exact active reservation range previously created
  // by Map() or Allocate(..., allocate_va_address=true). Allocator addresses
  // and subranges are not valid UnMap() inputs.
  auto active_it =
      state->active_reservation_records.find(reservation_address.opaque());
  if (active_it == state->active_reservation_records.end()) {
    auto stale_it =
        state->stale_reservation_records.find(reservation_address.opaque());
    if (stale_it != state->stale_reservation_records.end()) {
      CHECK(stale_it->second->reservation_address.has_value());
    }
    if (stale_it != state->stale_reservation_records.end() &&
        stale_it->second->reservation_address->IsSameAs(reservation_address)) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "reservation range at virtual address %p (%uB) is already pending "
          "UnMap()",
          reservation_address.opaque(), reservation_address.size()));
    }
    return absl::NotFoundError(absl::StrFormat(
        "UnMap() requires an exact active reservation range created by Map() "
        "or Allocate(..., allocate_va_address=true): virtual address %p "
        "(%uB)",
        reservation_address.opaque(), reservation_address.size()));
  }
  AllocationRecord* record = active_it->second;
  CHECK(record->reservation_active);
  CHECK(!record->reservation_stale);
  CHECK(record->reservation_address.has_value());
  CHECK(record->reservation_address->IsSameAs(reservation_address));
  CHECK(record->reservation_address_mapping.has_value());
  CHECK(record->reservation_address_mapping->mapped_address().IsSameAs(
      reservation_address));

  // Reserve the stream timeline sequence for this deferred UnMap and enqueue
  // the platform marker that will make the stale mapping safe to destroy.
  uint64_t seqno = state->next_seqno++;
  TF_RETURN_IF_ERROR(EnqueueDeferredDeallocation(*state, seqno));
  MoveReservationRecordToStale(*state, *record, seqno);
  state->pending_deallocations.push_back(PendingDeallocation{
      PendingDeallocationKind::kMap, seqno, reservation_address});
  return absl::OkStatus();
}

}  // namespace stream_executor
