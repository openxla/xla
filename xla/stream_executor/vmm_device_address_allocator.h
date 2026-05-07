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

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

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

// Abstract base class for virtual memory map (VMM) allocators that separate
// physical memory allocation, virtual address reservation, and mapping. A
// concrete subclass owns per-device VMM state and implements the platform
// operations used to allocate physical memory, reserve virtual addresses, map
// ranges, and enqueue deferred-free timeline updates.
//
// The allocator supports several mapping shapes:
//  1. Allocate()/Deallocate() create an owned virtual reservation, allocate
//     physical memory, map the allocation into that reservation, and later
//     defer both unmap and release.
//  2. AllocateRawAndMap()/DeallocateRawAndUnMap() allocate physical memory and
//     map it into a caller-owned MemoryReservation range.
//  3. AllocateAndMap()/DeallocateAndUnMap() create an owned allocation and also
//     map the same physical allocation into a caller-owned reservation range.
//  4. MapToRaw()/UnMapToRaw() temporarily map an already tracked physical
//     allocation into additional caller-owned reservation ranges.
//
// Deallocation and unmap requests are asynchronous. The allocator records a GPU
// timeline write on the device stream and keeps the MemoryAllocation,
// MemoryReservation, and ScopedMapping objects alive until all earlier stream
// work has completed. This lets callers release allocator-owned addresses or
// return extra mappings while kernels may still be using them.
//
// Each registered device has independent state protected by its own mutex, so
// operations on different devices can proceed in parallel. The per-device map
// is populated at construction time and is not modified afterward.
//
// Concrete subclasses implement the platform-specific virtual methods
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

  // Deallocates memory asynchronously. The caller can call this function even
  // if device kernels are still consuming the data — the actual deallocation
  // will be deferred until all previously enqueued work on the device's stream
  // completes.
  absl::Status Deallocate(int device_ordinal, DeviceAddressBase mem) override;

  // Allocates raw physical memory on `device_ordinal` and maps it into an
  // existing virtual address `reservation` at `reservation_offset`.
  //
  // The caller owns `reservation` and must keep it alive until the mapping is
  // explicitly deallocated. The mapped range starts at `reservation_offset` and
  // has size `size`. This method does not destroy or otherwise take ownership
  // of `reservation`.
  absl::Status AllocateRawAndMap(int device_ordinal,
                                 MemoryReservation* reservation,
                                 uint64_t reservation_offset, uint64_t size);

  // UnMaps a range previously mapped by AllocateRawAndMap() and releases the
  // raw physical allocation associated with that mapping.
  //
  // The caller owns `reservation`. This method unmaps the range starting at
  // `reservation_offset` with size `size`, but does not destroy or otherwise
  // take ownership of `reservation`.
  absl::Status DeallocateRawAndUnMap(int device_ordinal,
                                     MemoryReservation* reservation,
                                     uint64_t reservation_offset,
                                     uint64_t size);

  // Maps an already tracked raw physical allocation into an existing virtual
  // address `reservation` at `reservation_offset`.
  //
  // `raw_allocation` must be a physical allocation tracked by this allocator,
  // typically obtained from GetRawAllocation(). The caller owns `reservation`
  // and must keep it alive until the returned ScopedMapping is unmapped. If
  // the mapping is passed to UnMapToRaw(), keep `reservation` alive until the
  // raw allocation is deallocated and the deferred unmap has completed.
  absl::StatusOr<MemoryReservation::ScopedMapping> MapToRaw(
      int device_ordinal, MemoryAllocation* raw_allocation,
      MemoryReservation* reservation, uint64_t reservation_offset,
      uint64_t size);

  // Defers unmapping a mapping created by MapToRaw() until the underlying raw
  // physical allocation is deallocated.
  //
  // On success this method consumes `mapping` and returns ownership to the
  // allocator for deferred unmap. On error, `mapping` remains owned by the
  // caller and active MapToRaw() bookkeeping is unchanged. `raw_allocation`
  // must be the same tracked raw allocation that was passed to MapToRaw() for
  // this mapping. Empty mappings, such as the result of a zero-size MapToRaw(),
  // are treated as no-ops.
  absl::Status UnMapToRaw(int device_ordinal, MemoryAllocation* raw_allocation,
                          MemoryReservation::ScopedMapping&& mapping);

  // Allocates `allocation_size` bytes on `device_ordinal`, returns the newly
  // allocated device address, and also maps `mapping_size` bytes of the same
  // physical allocation into an existing virtual address `reservation` at
  // `reservation_offset`.
  //
  // The caller owns `reservation` and must keep it alive until the external
  // mapping is explicitly unmapped. To release both the returned allocation and
  // the external mapping, release the returned ScopedDeviceAddress and pass the
  // raw address to DeallocateAndUnMap(). DeallocateAndUnMap() does not destroy
  // or otherwise take ownership of `reservation`.
  absl::StatusOr<ScopedDeviceAddress<uint8_t>> AllocateAndMap(
      int device_ordinal, uint64_t allocation_size, bool retry_on_failure,
      int64_t memory_space, MemoryReservation* reservation,
      uint64_t reservation_offset, uint64_t mapping_size);

  // Deallocates `mem` and unmaps it from an existing virtual address
  // `reservation` at `reservation_offset`.
  //
  // The caller owns `reservation`. This method releases the device allocation
  // and unmaps the range starting at `reservation_offset` with size `size`, but
  // does not destroy or otherwise take ownership of `reservation`.
  absl::Status DeallocateAndUnMap(int device_ordinal, DeviceAddressBase mem,
                                  MemoryReservation* reservation,
                                  uint64_t reservation_offset, uint64_t size);

  // Pull in two-arg overload that sets retry_on_failure to true.
  using DeviceAddressAllocator::Allocate;

  // Returns true — this allocator supports asynchronous deallocation.
  bool AllowsAsynchronousDeallocation() const override { return true; }

  // Returns the stream for the given device ordinal.
  absl::StatusOr<Stream*> GetStream(int device_ordinal) override;

  // Returns the StreamExecutor for the given device ordinal.
  absl::StatusOr<StreamExecutor*> GetStreamExecutor(int device_ordinal) const;

  // Returns the MemoryAllocation (physical memory) backing the given virtual
  // address on the specified device, or nullptr if the address was not
  // allocated by this allocator. The returned pointer is valid until the
  // allocation is deallocated.
  MemoryAllocation* GetRawAllocation(int device_ordinal,
                                     DeviceAddressBase addr) const;

  // Returns the MemoryReservation (virtual address range) for the given
  // virtual address on the specified device, or nullptr if the address was not
  // allocated by this allocator. The returned pointer is valid until the
  // allocation is deallocated.
  MemoryReservation* GetReservation(int device_ordinal,
                                    DeviceAddressBase addr) const;

  // Returns the VMM allocation granularity for the device associated with
  // `executor`, or 0 if the device is not registered or granularity is unknown.
  uint64_t GetAllocationGranularity(StreamExecutor* executor) const;

  // Creates a virtual address reservation of the given size.
  virtual absl::StatusOr<std::unique_ptr<MemoryReservation>> CreateReservation(
      StreamExecutor* executor, uint64_t size) = 0;

 protected:
  enum class PendingDeallocationKind {
    kDeallocate,
    kDeallocateRawAndUnMap,
    kDeallocateAndUnMap,
  };

  struct PendingDeallocation {
    PendingDeallocationKind kind = PendingDeallocationKind::kDeallocate;
    // Device address to free after the GPU reaches `seqno`. This is an address
    // allocated through Allocate() or AllocateAndMap(). It can be empty when
    // the pending operation only needs to defer an unmap.
    DeviceAddressBase mem;
    // GPU stream sequence number recorded at deallocation time. When the
    // pinned_timeline value reaches this seqno, the memory is safe to free.
    uint64_t seqno;
    // Pending mapping to keep alive until the GPU reaches `seqno`. Destroying
    // this ScopedMapping performs the unmap for AllocateRawAndMap() or
    // DeallocateAndUnMap(). Empty when the pending operation only needs to free
    // `mem`.
    std::optional<MemoryReservation::ScopedMapping> mapping;
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
    absl::flat_hash_map<void*, std::shared_ptr<MemoryAllocation>>
        raw_allocations ABSL_GUARDED_BY(mu);
    absl::flat_hash_map<void*, std::unique_ptr<MemoryReservation>> reservations
        ABSL_GUARDED_BY(mu);
    absl::flat_hash_map<void*, MemoryReservation::ScopedMapping> scoped_mappings
        ABSL_GUARDED_BY(mu);
    // Extra MapToRaw() mappings returned to callers through ScopedMapping
    // ownership and later returned to the allocator through UnMapToRaw().
    // These mappings are destroyed when the underlying raw allocation is
    // completed.
    absl::flat_hash_map<MemoryAllocation*,
                        std::vector<MemoryReservation::ScopedMapping>>
        deferred_raw_mappings ABSL_GUARDED_BY(mu);
    // Extra MapToRaw() mappings currently owned by callers, keyed by mapped
    // virtual address. A raw allocation must not have entries in this map when
    // its PendingDeallocation is completed; callers must transfer active
    // mappings back through UnMapToRaw() before deallocating the raw
    // allocation.
    absl::flat_hash_map<void*, MemoryAllocation*> active_raw_mapping_keys
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
  // Returns pointer into per_device_ map; null if device_ordinal not
  // registered. No lock needed — per_device_ is read-only after construction.
  PerDeviceState* GetPerDeviceState(int device_ordinal) const;

  static PendingDeallocation PendingDeallocate(DeviceAddressBase mem,
                                               uint64_t seqno);

  static PendingDeallocation PendingDeallocateRawAndUnMap(
      uint64_t seqno, MemoryReservation::ScopedMapping mapping);

  static PendingDeallocation PendingDeallocateAndUnMap(
      DeviceAddressBase mem, uint64_t seqno,
      MemoryReservation::ScopedMapping mapping);

  absl::StatusOr<uint64_t> EnqueuePendingOperation(PerDeviceState& state)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  absl::Status QueueDeallocate(PerDeviceState& state, DeviceAddressBase mem)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  absl::Status QueueDeallocateRawAndUnMap(
      PerDeviceState& state, MemoryReservation::ScopedMapping& mapping)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  absl::Status QueueDeallocateAndUnMap(
      PerDeviceState& state, DeviceAddressBase mem,
      MemoryReservation::ScopedMapping& mapping)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  void* TrackOwnedAllocation(PerDeviceState& state,
                             std::shared_ptr<MemoryAllocation> raw_allocation,
                             std::unique_ptr<MemoryReservation> reservation,
                             MemoryReservation::ScopedMapping mapping,
                             uint64_t allocated_size)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  void TrackRawAndExternalMapping(
      PerDeviceState& state, DeviceAddressBase target,
      std::shared_ptr<MemoryAllocation> raw_allocation,
      MemoryReservation::ScopedMapping mapping, uint64_t allocated_size)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  void* TrackOwnedAndExternalMapping(
      PerDeviceState& state, DeviceAddressBase target,
      std::shared_ptr<MemoryAllocation> raw_allocation,
      std::unique_ptr<MemoryReservation> owned_reservation,
      MemoryReservation::ScopedMapping owned_mapping,
      MemoryReservation::ScopedMapping external_mapping,
      uint64_t allocated_size) ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  absl::StatusOr<DeviceAddressBase> TryFreshAllocate(PerDeviceState& state,
                                                     uint64_t size)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  absl::StatusOr<DeviceAddressBase> TryFreshAllocateRawAndMap(
      PerDeviceState& state, MemoryReservation* reservation,
      uint64_t reservation_offset, uint64_t size)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  absl::StatusOr<DeviceAddressBase> TryFreshAllocateAndMap(
      PerDeviceState& state, uint64_t allocation_size,
      MemoryReservation* reservation, uint64_t reservation_offset,
      uint64_t mapping_size) ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  void LogFreshAllocationResult(
      PerDeviceState& state, const char* attempt, uint64_t reclaim_size,
      const absl::StatusOr<DeviceAddressBase>& result) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  void LogPendingReclaim(PerDeviceState& state, const char* reclaim_action,
                         uint64_t reclaim_size) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  template <typename TryReuseFn, typename TryFreshFn>
  absl::StatusOr<DeviceAddressBase> TryWithPendingReclaim(PerDeviceState& state,
                                                          uint64_t reclaim_size,
                                                          TryReuseFn try_reuse,
                                                          TryFreshFn try_fresh)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  absl::StatusOr<DeviceAddressBase> ValidateReservationRange(
      MemoryReservation* reservation, uint64_t reservation_offset,
      uint64_t size) const;

  // Process any pending deallocations whose timeline sequence numbers have
  // been passed by the GPU.
  void ProcessCompletedPendingDeallocations(PerDeviceState& state)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Wait for enough pending operations to complete to reclaim at least 'size'
  // bytes. Selects pending operations from the front of the queue until their
  // cumulative size meets or exceeds the requested size, then spin-waits on
  // the GPU timeline counter and completes the selected operations.
  // Temporarily releases and reacquires state.mu around the blocking wait.
  void WaitPendingDeallocationsToComplete(PerDeviceState& state, uint64_t size)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Completes a pending operation whose stream sequence has passed. If this
  // releases a raw allocation, active_raw_mapping_keys must not contain any
  // caller-owned MapToRaw() mappings for that raw allocation.
  void CompletePendingDeallocation(PerDeviceState& state,
                                   PendingDeallocation& pending)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Actually perform the synchronous deallocation.
  void DoDeallocate(PerDeviceState& state, DeviceAddressBase mem)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  uint64_t EraseRawAllocationKey(PerDeviceState& state, void* key)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  bool IsTrackedRawAllocation(const PerDeviceState& state,
                              const MemoryAllocation* raw_allocation) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  bool HasActiveRawMappings(const PerDeviceState& state,
                            const MemoryAllocation* raw_allocation) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  absl::Status ValidateNoActiveRawMappings(
      const PerDeviceState& state, const MemoryAllocation* raw_allocation,
      const char* operation) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Destroys deferred mappings returned through UnMapToRaw() for
  // `raw_allocation`. Active MapToRaw() mappings are caller-owned and must
  // already have been returned through UnMapToRaw() before this is called.
  void CompleteDeferredRawMappings(PerDeviceState& state,
                                   MemoryAllocation* raw_allocation)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  uint64_t PendingDeallocationReclaimableSize(
      const PerDeviceState& state, const PendingDeallocation& pending) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Try to reuse a pending deallocation with matching rounded size.
  // Returns the reused address if found, or std::nullopt if no match.
  // Reuse is safe because any new work submitted after Allocate() returns is
  // enqueued on the same stream after the recorded deallocation event, so GPU
  // stream ordering guarantees the old work finishes before the new work runs.
  std::optional<DeviceAddressBase> TryReusePendingDeallocate(
      PerDeviceState& state, uint64_t size)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  bool TryReuseDeallocateRawAndUnMap(PerDeviceState& state,
                                     DeviceAddressBase target)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  std::optional<DeviceAddressBase> TryReuseDeallocateAndUnMap(
      PerDeviceState& state, DeviceAddressBase target, uint64_t allocation_size)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Round up size to the device's allocation granularity.
  uint64_t RoundUpToGranularity(const PerDeviceState& state,
                                uint64_t size) const;

  // Populated at construction; never modified. Safe to read without a lock.
  absl::flat_hash_map<int, std::unique_ptr<PerDeviceState>> per_device_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_VMM_DEVICE_ADDRESS_ALLOCATOR_H_
