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

#ifndef XLA_PJRT_SWAP_RAW_BUFFER_H_
#define XLA_PJRT_SWAP_RAW_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/raw_pjrt_client.h"
#include "xla/runtime/device_id.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {

// AsyncValue payload type representing completion of a host-side swap buffer
// operation.
struct SwapEvent {};

// Device event promise backed by an `IndirectAsyncValue` typed on `SwapEvent`.
class SwapDeviceEventPromise : public PjRtDeviceEventPromise {
 public:
  explicit SwapDeviceEventPromise(tsl::RCReference<tsl::IndirectAsyncValue> av);

  PjRtDeviceEventPtr event() const override {
    return PjRtDeviceEventPtr(tsl::AsyncValuePtr<SwapEvent>(av_.get()));
  }

  void Set(PjRtDeviceEventRef event) override;

  void SetError(absl::Status s) override { av_->SetError(std::move(s)); }

  void SetReady() override;

 private:
  // Underlying indirect async value fulfilled with a `SwapEvent` or error.
  tsl::RCReference<tsl::IndirectAsyncValue> av_;
};

// Creates a linked `(PjRtDeviceEventPromiseRef, PjRtDeviceEventRef)` pair for
// `SwapMemorySpace`.
std::pair<PjRtDeviceEventPromiseRef, PjRtDeviceEventRef>
CreateSwapLinkedEventPromise();

// Bridges `future` into a `PjRtDeviceEventRef` backed by `SwapEvent`.
PjRtDeviceEventRef ToSwapEvent(tsl::Future<> future);

// Owns the sequence of fixed-size 8 MiB chunks (plus an exact-size final
// remainder chunk, if any) allocated from a `HostMemoryAllocator`.
class SwapChunkStorage {
 public:
  struct Chunk {
    // Pinned host allocation for this chunk.
    HostMemoryAllocator::OwnedPtr data;
    // Exact byte size of this chunk.
    size_t size;
  };

  static absl::StatusOr<SwapChunkStorage> Allocate(
      size_t total_bytes, HostMemoryAllocator* allocator, int numa_node,
      std::optional<LocalDeviceId> local_device_id);

  SwapChunkStorage(size_t total_bytes, std::vector<Chunk> chunks);
  SwapChunkStorage(SwapChunkStorage&&) = default;
  SwapChunkStorage& operator=(SwapChunkStorage&&) = default;

  size_t total_bytes() const { return total_bytes_; }
  absl::Span<const Chunk> chunks() const { return chunks_; }

 private:
  // Total logical byte capacity across all chunks.
  size_t total_bytes_;
  // Ordered list of allocated chunks.
  std::vector<Chunk> chunks_;
};

// Chunked raw buffer implementation for `SwapMemorySpace`, shared across CPU,
// GPU, and TPU `CommonPjRtClient` backends.
class SwapRawBuffer : public CommonPjRtRawBufferImpl {
 public:
  static absl::StatusOr<PjRtRawBufferRef> Allocate(
      PjRtRawClient* raw_client, PJRT_Memory* memory_space,
      size_t on_device_bytes_count);

  static absl::StatusOr<tsl::RCReference<SwapRawBuffer>> Allocate(
      PJRT_Memory* memory_space, size_t on_device_bytes_count,
      HostMemoryAllocator* allocator, int numa_node,
      std::optional<LocalDeviceId> local_device_id,
      AsyncWorkRunner* async_work_runner);

  SwapRawBuffer(PJRT_Memory* memory_space,
                tsl::AsyncValueRef<SwapChunkStorage> storage,
                int64_t slice_offset, size_t byte_size,
                AsyncWorkRunner* async_work_runner);

  PJRT_Memory* c_memory_space() const override { return memory_space_; }

  void* GetHostPointer() const override { return nullptr; }

  void* OpaqueDeviceMemoryDataPointer() const override { return nullptr; }

  size_t GetOnDeviceSizeInBytes() const override { return byte_size_; }

  const SwapChunkStorage& storage() const { return *storage_; }
  int64_t slice_offset() const { return slice_offset_; }

  absl::Status ValidateSlice(int64_t offset, int64_t slice_size) const;

  absl::StatusOr<PjRtRawBufferRef> Slice(int64_t offset, int64_t size) override;

  absl::StatusOr<PjRtDeviceEventRef> CopyRawHostToDeviceAndReturnEvent(
      const void* src, int64_t offset, int64_t transfer_size,
      PjRtDeviceEventRefVector dependencies) override;

  absl::StatusOr<PjRtDeviceEventRef> CopyRawDeviceToHostAndReturnEvent(
      void* dst, int64_t offset, int64_t transfer_size,
      PjRtDeviceEventRefVector dependencies) override;

  absl::StatusOr<PjRtDeviceEventRef> MakeAllocationReadyEvent() override;

  PjRtDeviceEventPtr GetRawBufferAsyncValue() override {
    return PjRtDeviceEventPtr::FromAsyncValue(storage_.GetAsyncValue());
  }

  void CopyTo(
      PjRtRawBufferRef dst_raw_buffer,
      PjRtDeviceEventPromiseRef definition_event_promise,
      PjRtDeviceEventPromiseRef src_usage_event_promise,
      absl::AnyInvocable<void(absl::Status) &&> allocation_event) override;

  void ScheduleCopyTo(
      PjRtDeviceEventRefVector transfer_dependency_events,
      PjRtRawBufferRef dst_raw_buffer,
      PjRtDeviceEventPromiseRef definition_event_promise,
      PjRtDeviceEventPromiseRef src_usage_event_promise,
      absl::AnyInvocable<void(absl::Status) &&> allocation_event) override;

  // Schedules a direct chunked copy from `src_raw_buffer` into `this` swap
  // buffer without allocating a contiguous host bounce buffer.
  void ScheduleCopyFrom(
      PjRtDeviceEventRefVector transfer_dependency_events,
      PjRtRawBufferRef src_raw_buffer,
      PjRtDeviceEventPromiseRef definition_event_promise,
      PjRtDeviceEventPromiseRef src_usage_event_promise,
      absl::AnyInvocable<void(absl::Status) &&> allocation_event);

 private:
  // Memory space that owns this raw buffer.
  PJRT_Memory* memory_space_;
  // Concrete reference-counted chunked host allocation storage.
  tsl::AsyncValueRef<SwapChunkStorage> storage_;
  // Byte offset into `*storage_` for `Slice()` views.
  int64_t slice_offset_;
  // Logical byte size of this raw buffer (or slice).
  size_t byte_size_;
  // Executor used to run asynchronous host copy tasks.
  AsyncWorkRunner* async_work_runner_;
};

}  // namespace xla

#endif  // XLA_PJRT_SWAP_RAW_BUFFER_H_
