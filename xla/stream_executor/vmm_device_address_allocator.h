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

#ifndef XLA_STREAM_EXECUTOR_VMM_DEVICE_ADDRESS_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_VMM_DEVICE_ADDRESS_ALLOCATOR_H_

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {

// Abstract base class for a DeviceAddressAllocator backed by virtual memory
// management (VMM). VMM lets the allocator manage device memory in three
// separate steps:
//
//  1. Allocate raw physical memory. This is the real device memory capacity.
//  2. Reserve a virtual address (VA) range. This creates addresses but does not
//     make them usable yet.
//  3. Map a VA range to raw physical memory. Device kernels access memory
//     through the mapped VA.
//
// A concrete subclass provides the platform-specific operations for those
// steps, plus a stream-ordered timeline used to know when old mappings and
// allocations are safe to release.
//
// Caller-visible address roles:
//
//  * Allocator address: any address returned by Allocate(). It owns the raw
//    physical allocation, can be used as the source address for Map(), and must
//    eventually be released with Deallocate().
//  * Reservation address: a caller-owned MemoryReservation slice
//    [reservation_base + offset, reservation_base + offset + size) that is
//    mapped as a non-owning alias of an allocator address. It must be released
//    with UnMap(), not Deallocate().
//
// clang-format off
// Allowed address behavior:
//
// +-------------------------------------------------+---------------------+------------+-----+-------+
// | Address                                         | Role                | Deallocate | Map | UnMap |
// +-------------------------------------------------+---------------------+------------+-----+-------+
// | Allocate() return                               | allocator address   | yes        | yes | no    |
// | Allocate(..., allocate_va_address=false) return | allocator address   | yes        | yes | no    |
// | Allocate(..., allocate_va_address=true) return  | allocator address   | yes        | yes | no    |
// | reservation slice from Allocate(..., true)      | reservation address | no         | no  | yes   |
// | reservation slice from Map()                    | reservation address | no         | no  | yes   |
// +-------------------------------------------------+---------------------+------------+-----+-------+
// clang-format on
//
// The table uses "yes" for API calls that accept the address in that row. For
// example, Map() takes an allocator address as its source, while UnMap() takes
// a reservation address to tear down. Map() still requires the allocator
// address to have no active reservation-address alias; for example, an
// Allocate(..., allocate_va_address=true) result can be remapped only after its
// initial reservation-address alias is released with UnMap().
//
// The main API flows are:
//
//  1. Allocate(size) creates an allocator-owned VA reservation, allocates raw
//     physical memory, maps that memory into the owned reservation, and returns
//     the allocator address.
//  2. Allocate(..., allocate_va_address=false) allocates raw physical memory
//     and maps it directly into the caller reservation. The returned VA comes
//     from the caller reservation, but it is still the allocator address for
//     this allocation.
//  3. Allocate(..., allocate_va_address=true) returns a separate
//     allocator-owned address and also maps the same raw physical allocation
//     into the caller reservation as a reservation address.
//  4. Map(addr, reservation, ...) maps the raw physical allocation currently
//     backing allocator address `addr` into one caller reservation slice.
//     UnMap(reservation, ...) removes that reservation-address alias.
//
// Deallocate() accepts only allocator addresses. If the allocator address has
// an active reservation-address alias, Deallocate() first schedules that alias
// to be unmapped, then schedules the allocator address and raw physical
// allocation to be released. UnMap() accepts only reservation addresses created
// by Map() or by Allocate(..., allocate_va_address=true). Passing an allocator
// address to UnMap(), or a reservation address to Deallocate(), is an error.
// Each allocator address may have at most one active reservation-address alias.
// A reservation mapping is owned as the same full range that created it:
// partial UnMap(), Map(), or Allocate() operations that overlap an active or
// stale reservation mapping are rejected.
//
// Deallocate() and UnMap() are stream-ordered deferred operations. The
// allocator enqueues a timeline write on the device stream, moves the affected
// address record from active tracking to stale tracking, and appends a pending
// entry with the operation kind, sequence number, and address. The stale
// AllocationRecord keeps the raw allocation, any allocator-owned reservation,
// and ScopedMapping objects alive until the stream reaches that sequence
// number, so kernels already submitted to the stream can keep using the old VA.
// When the sequence completes, dropping the ScopedMapping objects performs the
// real unmap, then the allocator releases any owned reservation and raw
// physical memory.
//
// Stale records are also the fast reuse path. Allocate() first looks for a
// compatible stale allocator address before creating new VMM state. Map() does
// the same for a stale reservation mapping: if the requested reservation
// address is still mapped to the same raw physical allocation, the allocator
// reactivates the old mapping instead of unmapping and remapping. If a
// requested reservation address is still stale for a different raw allocation,
// Map() waits for that deferred unmap to complete before installing the new
// mapping.
//
// Each registered device has independent state protected by its own mutex, so
// operations on different devices can proceed in parallel. The per-device map
// is populated at construction time and is not modified afterward. Concrete
// subclasses implement the platform-specific virtual methods
// (InitializeDeviceState, CreateAllocation, CreateReservation,
// EnqueueDeferredDeallocation) and expose platform-specific Create() factories.
// Subclasses must also set PerDeviceState::destroy_fn in InitializeDeviceState
// to release platform-specific resources such as pinned timeline memory.
class DeviceAddressVmmAllocator : public DeviceAddressAllocator {
 public:
  // Per-device configuration supplied at construction.
  struct DeviceConfig {
    // StreamExecutor for this device. Must outlive the allocator.
    StreamExecutor* executor;
    // Stream used for deferred deallocation. Must outlive the allocator.
    Stream* stream;
    // Maximum bytes of physical memory that may be allocated simultaneously on
    // this device. Defaults to unlimited.
    uint64_t pa_budget = UINT64_MAX;
  };

  ~DeviceAddressVmmAllocator() override;

  absl::StatusOr<ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override;

  // Allocates raw physical memory and maps it into a caller-owned
  // MemoryReservation range.
  // `allocation_size` and `mapping_size` must be equal.
  //
  // There are two modes:
  //
  //  * `allocate_va_address=false`: the mapped reservation slice is returned
  //    and is treated as the allocator address. The caller releases it with
  //    Deallocate(), may use it as a Map() source, and must not pass it to
  //    UnMap().
  //  * `allocate_va_address=true`: the allocator creates and returns a
  //    separate allocator-owned address. The same raw physical allocation is
  //    also mapped into the caller reservation as a reservation address. The
  //    returned allocator address is released with Deallocate(); the
  //    reservation-address alias may be released earlier with UnMap().
  //
  // The caller owns `reservation` and must keep it alive while any mapping into
  // it is active or waiting for deferred unmap completion. Deallocate() never
  // destroys or takes ownership of `reservation`.
  absl::StatusOr<ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t allocation_size, bool retry_on_failure,
      int64_t memory_space, MemoryReservation* reservation,
      uint64_t reservation_offset, uint64_t mapping_size,
      bool allocate_va_address);

  // Deallocates an allocator address asynchronously. `mem` must be an address
  // returned by Allocate(), including reservation-derived addresses returned by
  // Allocate(..., allocate_va_address=false). Reservation addresses created by
  // Map() or by Allocate(..., allocate_va_address=true) must not be passed to
  // Deallocate(). If `mem` has an active reservation-address alias,
  // Deallocate() automatically defers unmapping that alias before deferring the
  // allocator-address deallocation. The caller can call this function while
  // device kernels are still consuming the data; the actual release is deferred
  // until earlier work on the device stream completes.
  absl::Status Deallocate(int device_ordinal, DeviceAddressBase mem) override;

  // Adds a reservation-address alias for an existing allocator address by
  // mapping the physical allocation currently backing `addr` into
  // `reservation` at `reservation_offset`.
  //
  // `addr` must be an active allocator address returned by this allocator,
  // including reservation-derived addresses returned by
  // Allocate(..., allocate_va_address=false). Non-owning reservation addresses
  // created by Map() or by Allocate(..., allocate_va_address=true), and
  // addresses from other allocators, are not supported. The physical allocation
  // backing `addr` must be at least `size` bytes. Each allocator address may
  // have at most one active reservation-address alias at a time. The caller
  // owns `reservation` and must keep it alive until UnMap() is called, or until
  // Deallocate(addr) automatically defers the alias unmap and the allocator
  // stream reaches that deferred unmap point.
  absl::Status Map(int device_ordinal, DeviceAddressBase addr,
                   MemoryReservation* reservation, uint64_t reservation_offset,
                   uint64_t size);

  // Defers unmapping the reservation address created by Map() or by
  // Allocate(..., allocate_va_address=true) for the given reservation range
  // until all previously enqueued work on the allocator stream has completed.
  // The caller must pass the same full reservation range that created the
  // mapping; partial ranges that overlap a tracked mapping are rejected.
  // The reservation-derived allocator address returned by
  // Allocate(..., allocate_va_address=false) is not a reservation address for
  // this API and must be released with Deallocate() instead.
  //
  // On success this method moves the active mapping to the deferred unmap
  // queue. On error, active bookkeeping is unchanged. Empty mappings, such as
  // zero-size Map(), are treated as no-ops.
  absl::Status UnMap(int device_ordinal, MemoryReservation* reservation,
                     uint64_t reservation_offset, uint64_t size);

  // Pull in two-arg overload that sets retry_on_failure to true.
  using DeviceAddressAllocator::Allocate;

  // Returns true: this allocator supports asynchronous deallocation.
  bool AllowsAsynchronousDeallocation() const override { return true; }

  // Returns the stream for the given device ordinal.
  absl::StatusOr<Stream*> GetStream(int device_ordinal) override;

  // Waits for all pending stream-ordered deallocations and unmaps on the given
  // device to complete, then drops the corresponding deferred bookkeeping.
  absl::Status SynchronizePendingOperations(int device_ordinal);

  // Returns the StreamExecutor for the given device ordinal.
  absl::StatusOr<StreamExecutor*> GetStreamExecutor(int device_ordinal) const;

  // Returns the MemoryAllocation (physical memory) backing the given virtual
  // address on the specified device, or nullptr if the address was not
  // allocated by this allocator. The returned pointer is valid until the
  // allocation is deallocated.
  MemoryAllocation* GetRawAllocation(int device_ordinal,
                                     DeviceAddressBase addr) const;

  // Returns the VMM allocation granularity for the device associated with
  // `executor`, or 0 if the device is not registered or granularity is unknown.
  uint64_t GetAllocationGranularity(StreamExecutor* executor) const;

  // Creates a virtual address reservation of the given size.
  virtual absl::StatusOr<std::unique_ptr<MemoryReservation>> CreateReservation(
      StreamExecutor* executor, uint64_t size) = 0;

 protected:
  enum class PendingDeallocationKind {
    // Deferred Deallocate() of an Allocate() result backed by an
    // allocator-owned reservation.
    kAllocate,
    // Deferred Deallocate() of an Allocate(..., allocate_va_address=false)
    // result. The allocator address is a caller-owned reservation range.
    kAllocateWithMap,
    // Deferred Deallocate() of an Allocate(..., allocate_va_address=true)
    // result. The record has an allocator-owned returned address and may also
    // have a non-owning caller reservation mapping to unmap.
    kAllocateWithAddressAndMap,
    // Deferred completion of a Map()-owned reservation address. The reservation
    // address is a non-owning alias of an existing raw allocation.
    kMap,
  };

  // Lifetime record for one raw physical allocation.
  //
  // The record is owned by records_by_allocator_address while either the
  // allocator address or a reservation-address alias is active, stale, or
  // pending completion. Active indexes are callable by public APIs. Stale
  // indexes are no longer callable by users, but still keep mappings alive
  // until the stream-ordered deferred operation completes or a later Allocate()
  // or Map() reuses them.
  struct AllocationRecord {
    PendingDeallocationKind kind = PendingDeallocationKind::kAllocate;
    DeviceAddressBase allocator_address;
    std::shared_ptr<MemoryAllocation> raw_allocation;

    // Present for Allocate() and Allocate(..., allocate_va_address=true).
    std::unique_ptr<MemoryReservation> allocator_address_reservation;
    // Present while the allocator address is active or stale.
    std::optional<MemoryReservation::ScopedMapping> allocator_address_mapping;

    // Present while a reservation alias is active or stale.
    std::optional<DeviceAddressBase> reservation_address;
    std::optional<MemoryReservation::ScopedMapping> reservation_address_mapping;

    // Allocator address state. Every live AllocationRecord has an allocator
    // address, and exactly one of allocator_active/allocator_stale is true
    // until the deferred allocator-address deallocation completes and the
    // record is destroyed.
    bool allocator_active = false;
    bool allocator_stale = false;

    // Reservation-address alias state. A record may have no reservation alias;
    // in that case reservation_address/reservation_address_mapping are empty
    // and both flags are false. If a reservation alias exists, exactly one of
    // reservation_active/reservation_stale is true until the deferred unmap
    // completes or the alias is reactivated.
    bool reservation_active = false;
    bool reservation_stale = false;

    // Valid only while the corresponding address is stale. These seqnos
    // identify the stream point at which it is safe to destroy the stale
    // mapping and, for allocator addresses, release the raw physical
    // allocation.
    uint64_t allocator_stale_seqno = 0;
    uint64_t reservation_stale_seqno = 0;
  };

  // Queue entry for a stream-ordered deferred operation. The heavy resources
  // live in AllocationRecord; this entry only says which stale address becomes
  // safe to complete when the GPU timeline reaches `seqno`.
  struct PendingDeallocation {
    PendingDeallocationKind kind = PendingDeallocationKind::kAllocate;
    // GPU stream sequence number recorded at deallocation time. When the
    // pinned_timeline value reaches this seqno, the memory is safe to free.
    uint64_t seqno = 0;
    // Allocator address for allocation deallocations; reservation address for
    // kMap.
    DeviceAddressBase addr;
  };

  // Stable identity for a pending operation. Iterators into
  // pending_deallocations must not be kept across waits because
  // WaitUntilSeqno() releases state.mu.
  struct PendingDeallocationKey {
    PendingDeallocationKind kind = PendingDeallocationKind::kAllocate;
    uint64_t seqno = 0;
    DeviceAddressBase addr;
  };

  struct PerDeviceState {
    StreamExecutor* executor;
    Stream* stream;
    uint64_t pa_budget;
    // VMM allocation granularity for this device. Set once in
    // InitializeDeviceState(); 0 means the query failed.
    uint64_t allocation_granularity = 0;

    // Host-visible timeline counter. The GPU writes an increasing sequence
    // number to this location as each deallocation point is reached in the
    // stream. The CPU reads it atomically to determine which pending
    // deallocations are safe to execute.
    // Allocated at construction in InitializeDeviceState(); freed via
    // destroy_fn in ~DeviceAddressVmmAllocator().
    // Never modified after construction other than by the GPU.
    volatile uint64_t* pinned_timeline = nullptr;
    // Device-mapped pointer to pinned_timeline (as uint64_t to avoid
    // platform-specific types in this header). Passed to the platform-specific
    // stream write operation.
    uint64_t timeline_dev_ptr = 0;

    // Called at the end of ~DeviceAddressVmmAllocator() to release
    // platform-specific resources (e.g. pinned timeline memory). Set once by
    // InitializeDeviceState(); must not reference the subclass instance.
    std::function<void()> destroy_fn;

    mutable absl::Mutex mu;
    uint64_t pa_allocated ABSL_GUARDED_BY(mu) = 0;
    // Monotonically increasing counter for timeline sequence numbers.
    uint64_t next_seqno ABSL_GUARDED_BY(mu) = 1;
    std::deque<PendingDeallocation> pending_deallocations ABSL_GUARDED_BY(mu);
    // Owns AllocationRecord objects. Key is the allocator address pointer
    // (`AllocationRecord::allocator_address.opaque()`), including the
    // reservation-derived allocator address returned by
    // Allocate(..., allocate_va_address=false). Allocator-address active/stale
    // state is stored in AllocationRecord::allocator_active/allocator_stale.
    absl::flat_hash_map<void*, std::unique_ptr<AllocationRecord>>
        records_by_allocator_address ABSL_GUARDED_BY(mu);

    // Active/stale reservation-address indexes. Keys are reservation alias
    // pointers (`AllocationRecord::reservation_address->opaque()`) created by
    // Map() or by Allocate(..., allocate_va_address=true).
    absl::flat_hash_map<void*, AllocationRecord*> active_reservation_records
        ABSL_GUARDED_BY(mu);
    absl::flat_hash_map<void*, AllocationRecord*> stale_reservation_records
        ABSL_GUARDED_BY(mu);
  };

  explicit DeviceAddressVmmAllocator(const Platform* platform);

  // Validates no duplicate ordinals in `devices`, then iterates over each
  // device config, constructs a PerDeviceState (setting executor, stream,
  // pa_budget), calls InitializeDeviceState() for platform-specific
  // initialization, and registers the state in allocator->per_device_.
  //
  // Called by platform-specific Create() factories.
  static absl::Status PopulateDevices(DeviceAddressVmmAllocator* allocator,
                                      absl::Span<const DeviceConfig> devices);

  // Validates device capabilities and initializes timeline fields
  // (pinned_timeline, timeline_dev_ptr, allocation_granularity) in state.
  // state.executor, state.stream, and state.pa_budget are already set.
  virtual absl::Status InitializeDeviceState(PerDeviceState& state) = 0;

  // Creates a physical memory allocation of the given size.
  virtual absl::StatusOr<std::unique_ptr<MemoryAllocation>> CreateAllocation(
      StreamExecutor* executor, uint64_t size) = 0;

  // Enqueues a GPU timeline write at the given seqno on the device's stream.
  virtual absl::Status EnqueueDeferredDeallocation(PerDeviceState& state,
                                                   uint64_t seqno) = 0;

 private:
  struct DebugStats {
    std::atomic<uint64_t> allocate_calls{0};
    std::atomic<uint64_t> allocate_allocation_reuse{0};

    std::atomic<uint64_t> mapped_allocate_without_va_calls{0};
    std::atomic<uint64_t> mapped_allocate_without_va_allocation_reuse{0};
    std::atomic<uint64_t> mapped_allocate_without_va_allocator_va_reuse{0};

    std::atomic<uint64_t> mapped_allocate_with_va_calls{0};
    std::atomic<uint64_t> mapped_allocate_with_va_allocation_reuse{0};
    std::atomic<uint64_t> mapped_allocate_with_va_allocator_va_reuse{0};
    std::atomic<uint64_t> mapped_allocate_with_va_reservation_va_reuse{0};

    std::atomic<uint64_t> map_calls{0};
    std::atomic<uint64_t> map_reservation_va_reuse{0};

    std::atomic<uint64_t> deallocate_calls{0};
    std::atomic<uint64_t> unmap_calls{0};
  };

  std::string DebugStatsTable() const;

  // Common helpers.

  // Returns pointer into per_device_ map, or NotFound if device_ordinal is not
  // registered. No lock needed: per_device_ is read-only after construction.
  absl::StatusOr<PerDeviceState*> GetPerDeviceState(int device_ordinal) const;

  // Round up size to the device's allocation granularity.
  uint64_t RoundUpToGranularity(const PerDeviceState& state,
                                uint64_t size) const;

  // Allocate helpers.

  // Records a raw allocation mapped at an owning allocator address. Takes
  // ownership of `reservation` when the allocator address was allocator-owned;
  // reservation-backed returned addresses pass nullptr here. Charges
  // `allocated_size` to the PA budget and returns the allocator VA pointer.
  void* TrackAllocatorAddressMappedAllocation(
      PerDeviceState& state, PendingDeallocationKind kind,
      DeviceAddressBase allocator_address,
      std::shared_ptr<MemoryAllocation> raw_allocation,
      std::unique_ptr<MemoryReservation> reservation,
      MemoryReservation::ScopedMapping mapping, uint64_t allocated_size)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Shared allocation retry policy. First calls `try_reuse` to reactivate
  // compatible pending state without blocking, then calls `try_fresh`. On
  // ResourceExhausted, it completes ready pending entries and, if needed, waits
  // for enough pending frees to reclaim approximately `reclaim_size` bytes.
  template <typename TryReuseFn, typename TryFreshFn>
  absl::StatusOr<DeviceAddressBase> TryWithPendingReclaim(PerDeviceState& state,
                                                          uint64_t reclaim_size,
                                                          TryReuseFn try_reuse,
                                                          TryFreshFn try_fresh)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Emits VLOG diagnostics for a fresh allocation attempt made by
  // TryWithPendingReclaim(), including PA budget state and pending queue size
  // when the attempt fails.
  void LogFreshAllocationResult(
      PerDeviceState& state, const char* attempt, uint64_t reclaim_size,
      const absl::StatusOr<DeviceAddressBase>& result) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Emits VLOG diagnostics before a pending-reclaim action, such as completing
  // already-ready entries or waiting for enough pending frees to drain.
  void LogPendingReclaim(PerDeviceState& state, const char* reclaim_action,
                         uint64_t reclaim_size) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Map helpers.

  // Validates a caller-owned reservation slice and returns the corresponding
  // DeviceAddressBase. Rejects null reservations and out-of-bounds
  // offset/size pairs before any allocator bookkeeping is mutated.
  absl::StatusOr<DeviceAddressBase> ValidateReservationRange(
      MemoryReservation* reservation, uint64_t reservation_offset,
      uint64_t size) const;

  struct OverlappingRecord {
    AllocationRecord* record = nullptr;
    DeviceAddressBase tracked_address;
    bool is_allocator = false;
    bool is_active = false;
  };

  // Finds a tracked allocator or reservation range that overlaps `address`.
  // `exact_only` returns only identical ranges; `partial_only` returns only
  // non-identical overlapping ranges; both false returns any overlap.
  std::optional<OverlappingRecord> FindOverlappingRecord(
      PerDeviceState& state, DeviceAddressBase address, bool include_allocator,
      bool include_reservation, bool include_active, bool include_stale,
      bool exact_only, bool partial_only) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // UnMap/deferred teardown helpers.

  // Removes the matching pending entry when a stale record is reused.
  void ErasePendingDeallocation(PerDeviceState& state,
                                PendingDeallocationKind kind,
                                DeviceAddressBase addr)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  void MoveAllocatorRecordToActive(PerDeviceState& state,
                                   AllocationRecord& record, uint64_t new_size)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  void MoveReservationRecordToStale(PerDeviceState& state,
                                    AllocationRecord& record, uint64_t seqno)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  void MoveReservationRecordToActive(PerDeviceState& state,
                                     AllocationRecord& record)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  void CompleteStaleReservationMapping(PerDeviceState& state,
                                       AllocationRecord& record)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Waits for the device timeline to reach `target_seqno`. Temporarily releases
  // and reacquires state.mu around the blocking wait. This does not complete
  // pending entries by itself.
  void WaitUntilSeqno(PerDeviceState& state, uint64_t target_seqno)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Waits for pending operations through `target_seqno`, then completes all
  // still-pending operations up to that sequence. Used only when preserving
  // stale mappings for future reuse is no longer useful.
  void WaitAndDrainPendingDeallocationsUntilSeqno(PerDeviceState& state,
                                                  uint64_t target_seqno)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Completes ready allocator-address deallocations for PA reclaim while
  // leaving unrelated kMap entries stale and reusable.
  void CompleteReadyAllocatorDeallocationsForReclaim(PerDeviceState& state,
                                                     uint64_t completed_seqno)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Completes a pending operation whose stream sequence has passed by dropping
  // its ScopedMappings, allocator-owned reservation, and raw allocation
  // reference. This is where VA unmap, reservation release, and PA budget
  // accounting happen.
  void CompletePendingDeallocation(PerDeviceState& state,
                                   const PendingDeallocation& pending)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Finds, erases, and completes the selected pending entry if it is still
  // present. Returns false if another thread already reused or completed it
  // while state.mu was released.
  bool CompletePendingDeallocationByKey(PerDeviceState& state,
                                        const PendingDeallocationKey& key)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Waits for and completes the selected allocator-address deallocation, if it
  // is still pending after the wait.
  void WaitAndCompleteStaleAllocatorDeallocation(
      PerDeviceState& state, const PendingDeallocationKey& key)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Waits for and completes a stale reservation-address mapping. The
  // reservation mapping may have a kMap queue entry, or it may be stale because
  // Deallocate() auto-staled the alias before queuing allocator deallocation.
  void WaitAndCompleteStaleReservationMapping(PerDeviceState& state,
                                              const PendingDeallocationKey& key)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Completes only the stale allocator or reservation mapping that conflicts
  // with the current request, leaving unrelated stale mappings reusable.
  void WaitAndCompleteStaleOverlap(PerDeviceState& state,
                                   const OverlappingRecord& overlap)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Device ordinal -> per-device allocator state. Populated at construction by
  // PopulateDevices() and never modified afterward, so map lookup is safe
  // without an allocator-wide lock. Each PerDeviceState owns its own mutex for
  // mutable allocation and pending-deallocation state.
  absl::flat_hash_map<int, std::unique_ptr<PerDeviceState>> per_device_;
  DebugStats debug_stats_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_VMM_DEVICE_ADDRESS_ALLOCATOR_H_
