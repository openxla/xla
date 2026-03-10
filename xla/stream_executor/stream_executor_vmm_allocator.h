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

#ifndef XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_VMM_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_VMM_ALLOCATOR_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>

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

// Virtual address allocator that separates virtual address reservation from
// physical memory allocation. It can be bound to one or more GPU devices at
// construction time and routes Allocate/Deallocate calls to per-device state
// based on device_ordinal.
//
// Concurrency model: each device has its own absl::Mutex, so operations on
// different devices run fully in parallel. The per-device map is populated
// entirely at construction and never modified afterward, so lookups require no
// global lock.
//
// For each device the allocator:
//  1. Allocates physical memory via MemoryAllocation (e.g. cuMemCreate).
//  2. Reserves virtual address space via MemoryReservation
//     (e.g. cuMemAddressReserve).
//  3. Maps physical memory to virtual address via
//     MemoryReservation::ScopedMapping (e.g. cuMemMap + cuMemSetAccess), which
//     automatically unmaps on destruction.
//
// The allocator tracks the ScopedMapping and underlying MemoryAllocation and
// MemoryReservation objects for each returned DeviceAddressBase. Callers can
// retrieve these via GetRawAllocation() and GetReservation().
//
// This allocator supports asynchronous deallocation: when Deallocate() is
// called, it records a GPU timeline write on the device's stream and defers
// the actual deallocation until the GPU reaches that point in the stream. This
// allows callers to deallocate memory while device kernels may still be
// consuming the data.
//
// Requires compute capability >= 7.0 (Volta and later) for
// cuStreamWriteValue64 support. Use Create() to obtain an instance; it
// returns an error if the device does not meet this requirement.
//
// This allocator is thread-safe for concurrent use by multiple threads across
// any registered devices.
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

  // Creates an allocator supporting multiple devices.
  //
  // Returns an error if any device does not support cuStreamWriteValue64
  // (compute capability < 7.0).
  //
  // Precondition: all entries in `devices` have distinct device ordinals.
  static absl::StatusOr<std::unique_ptr<DeviceAddressVmmAllocator>> Create(
      const Platform* platform, absl::Span<const DeviceConfig> devices);

  // Creates an allocator for a single device.
  //
  // Returns an error if the device does not support cuStreamWriteValue64
  // (compute capability < 7.0).
  //
  // Parameters:
  //   executor:  StreamExecutor for this device. Must outlive the allocator.
  //   stream:    Stream used for deferred deallocation. Must outlive the
  //              allocator. This should typically be the main compute stream
  //              from ServiceExecutableRunOptions.
  //   pa_budget: Maximum bytes of physical memory that may be simultaneously
  //              allocated on this device. Defaults to unlimited.
  static absl::StatusOr<std::unique_ptr<DeviceAddressVmmAllocator>> Create(
      StreamExecutor* executor, Stream* stream,
      uint64_t pa_budget = UINT64_MAX);

  ~DeviceAddressVmmAllocator() override;

  absl::StatusOr<ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override;

  // Pull in two-arg overload that sets retry_on_failure to true.
  using DeviceAddressAllocator::Allocate;

  // Deallocates memory asynchronously. The caller can call this function even
  // if device kernels are still consuming the data — the actual deallocation
  // will be deferred until all previously enqueued work on the device's stream
  // completes.
  absl::Status Deallocate(int device_ordinal, DeviceAddressBase mem) override;

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

 private:
  struct PendingDeallocation {
    DeviceAddressBase mem;
    // GPU stream sequence number recorded at deallocation time. When the
    // pinned_timeline value reaches this seqno, the memory is safe to free.
    uint64_t seqno;
  };

  struct PerDeviceState {
    StreamExecutor* executor;
    Stream* stream;
    uint64_t pa_budget;
    // VMM allocation granularity for this device. Set once in Create() via
    // cuMemGetAllocationGranularity; 0 means the query failed.
    uint64_t allocation_granularity = 0;

    // Host-visible timeline counter. The GPU writes an increasing sequence
    // number to this location via cuStreamWriteValue64 as each deallocation
    // point is reached in the stream. The CPU reads it atomically to determine
    // which pending deallocations are safe to execute.
    // Allocated at construction via cuMemHostAlloc; freed in the destructor.
    // Never modified after construction other than by the GPU.
    volatile uint64_t* pinned_timeline = nullptr;
    // Device-mapped pointer to pinned_timeline (as uint64_t to avoid CUDA
    // types in this header). Passed to cuStreamWriteValue64.
    uint64_t timeline_dev_ptr = 0;

    mutable absl::Mutex mu;
    uint64_t pa_allocated ABSL_GUARDED_BY(mu) = 0;
    // Monotonically increasing counter for timeline sequence numbers.
    uint64_t next_seqno ABSL_GUARDED_BY(mu) = 1;
    std::deque<PendingDeallocation> pending_deallocations ABSL_GUARDED_BY(mu);
    absl::flat_hash_map<void*, std::unique_ptr<MemoryAllocation>>
        raw_allocations ABSL_GUARDED_BY(mu);
    absl::flat_hash_map<void*, std::unique_ptr<MemoryReservation>> reservations
        ABSL_GUARDED_BY(mu);
    absl::flat_hash_map<void*, MemoryReservation::ScopedMapping> scoped_mappings
        ABSL_GUARDED_BY(mu);
  };

  explicit DeviceAddressVmmAllocator(const Platform* platform);

  // Returns pointer into per_device_ map; null if device_ordinal not
  // registered. No lock needed — per_device_ is read-only after construction.
  PerDeviceState* GetPerDeviceState(int device_ordinal) const;

  absl::StatusOr<DeviceAddressBase> AllocateWithBudget(PerDeviceState& state,
                                                       uint64_t size)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Process any pending deallocations whose timeline sequence numbers have
  // been passed by the GPU.
  void ProcessCompletedPendingDeallocations(PerDeviceState& state)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Wait for enough pending deallocations to complete to free at least 'size'
  // bytes. Selects deallocations from the front of the queue until their
  // cumulative size meets or exceeds the requested size, then spin-waits on
  // the GPU timeline counter and performs the deallocations.
  // Temporarily releases and reacquires state.mu around the blocking wait.
  void WaitPendingDeallocationsToComplete(PerDeviceState& state, uint64_t size);

  // Actually perform the synchronous deallocation.
  void DoDeallocate(PerDeviceState& state, DeviceAddressBase mem)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Try to reuse a pending deallocation with matching rounded size.
  // Returns the reused address if found, or std::nullopt if no match.
  // Reuse is safe because any new work submitted after Allocate() returns is
  // enqueued on the same stream after the recorded deallocation event, so GPU
  // stream ordering guarantees the old work finishes before the new work runs.
  std::optional<DeviceAddressBase> TryReusePendingDeallocation(
      PerDeviceState& state, uint64_t size)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Round up size to the device's allocation granularity.
  uint64_t RoundUpToGranularity(const PerDeviceState& state,
                                uint64_t size) const;

  // Populated at construction; never modified. Safe to read without a lock.
  absl::flat_hash_map<int, std::unique_ptr<PerDeviceState>> per_device_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_VMM_ALLOCATOR_H_
