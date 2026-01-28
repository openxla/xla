/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_VIRTUAL_ADDRESS_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_VIRTUAL_ADDRESS_ALLOCATOR_H_

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {

// Virtual address allocator that separates virtual address reservation from
// physical memory allocation. This allocator:
// 1. Reserves virtual address space using StreamExecutor::ReserveAddress
// 2. Allocates unmapped physical memory using StreamExecutor::AllocateRaw
// 3. Maps physical memory to virtual address using StreamExecutor::MapRawAddress
//
// The returned DeviceAddress will have the raw_handle set to the underlying
// physical memory handle.
//
// This allocator supports asynchronous deallocation: when Deallocate() is
// called, it records an event on the stream and defers the actual deallocation
// until the event completes. This allows callers to deallocate memory while
// device kernels may still be consuming the data.
class DeviceVirtualAddressAllocator : public DeviceAddressAllocator {
 public:
  // Information needed per device for the allocator.
  struct DeviceInfo {
    StreamExecutor* executor;
    Stream* stream;  // The stream to use for this device. Must outlive the
                     // allocator.
  };

  // Create an allocator supporting a single device.
  //
  // Parameters:
  //   executor: The stream executor to use for allocation operations.
  //   stream: The stream to use for deferred deallocation. Must outlive the
  //           allocator. This should typically be the main compute stream from
  //           ServiceExecutableRunOptions.
  DeviceVirtualAddressAllocator(StreamExecutor* executor, Stream* stream);

  // Create an allocator supporting multiple devices.
  //
  // Precondition: all devices have different device ordinals.
  //
  // Parameters:
  //   platform: The platform for this allocator.
  //   devices: List of device info (executor + stream) to support. Each stream
  //            must outlive the allocator and should typically be the main
  //            compute stream from ServiceExecutableRunOptions.
  DeviceVirtualAddressAllocator(const Platform* platform,
                                absl::Span<const DeviceInfo> devices);

  ~DeviceVirtualAddressAllocator() override;

  absl::StatusOr<ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override;

  // Pull in two-arg overload that sets retry_on_failure to true.
  using DeviceAddressAllocator::Allocate;

  // Deallocates memory asynchronously. The caller can call this function even
  // if device kernels are still consuming the data - the actual deallocation
  // will be deferred until all previously enqueued work on the stream
  // completes.
  absl::Status Deallocate(int device_ordinal, DeviceAddressBase mem) override;

  // Returns true - this allocator supports asynchronous deallocation.
  bool AllowsAsynchronousDeallocation() const override { return true; }

  // Returns the stream for the given device ordinal. This is the stream that
  // was passed to the constructor and should be the main compute stream from
  // ServiceExecutableRunOptions.
  absl::StatusOr<Stream*> GetStream(int device_ordinal) override;

  // Gets the stream executor for given device ordinal.
  absl::StatusOr<StreamExecutor*> GetStreamExecutor(int device_ordinal) const;

 private:
  // Structure to track pending deallocations.
  struct PendingDeallocation {
    int device_ordinal;
    DeviceAddressBase mem;
    std::unique_ptr<Event> event;
  };

  // Process any pending deallocations whose events have completed.
  void ProcessPendingDeallocations() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Actually perform the synchronous deallocation.
  void DoDeallocate(int device_ordinal, DeviceAddressBase mem);

  // Round up size to allocation granularity.
  uint64_t RoundUpToGranularity(uint64_t size) const;

  // Try to reuse a pending deallocation with matching rounded size.
  // Returns the reused address if found, or std::nullopt if no match.
  // Only reuses deallocations whose events have already completed (non-blocking).
  std::optional<DeviceAddressBase> TryReusePendingDeallocation(
      int device_ordinal, uint64_t size, uint64_t rounded_size)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Available stream executors. Each stream executor has a different device
  // ordinal.
  std::vector<StreamExecutor*> stream_executors_;

  // Streams for each device ordinal. These are not owned by this allocator;
  // they must outlive the allocator.
  std::map<int, Stream*> streams_;

  // Allocation granularity for rounding sizes.
  uint64_t allocation_granularity_ = 0;

  absl::Mutex mutex_;

  // Queue of pending deallocations waiting for their events to complete.
  std::deque<PendingDeallocation> pending_deallocations_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_VIRTUAL_ADDRESS_ALLOCATOR_H_
