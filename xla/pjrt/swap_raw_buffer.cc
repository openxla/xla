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

#include "xla/pjrt/swap_raw_buffer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/device_event_utils.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/raw_pjrt_client.h"
#include "xla/runtime/device_id.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace {

// Fixed chunk size (8 MiB) used for chunked allocations in SwapMemorySpace.
constexpr size_t kSwapChunkSizeBytes = 8 * 1024 * 1024;

// Iterates over the sub-ranges of `storage` covering `[abs_offset, abs_offset +
// size)`, invoking `fn(uint8_t* chunk_ptr, int64_t rel_offset, int64_t bytes)`.
template <typename Fn>
void ForEachChunkSpan(const SwapChunkStorage& storage, int64_t abs_offset,
                      int64_t size, Fn&& fn) {
  if (size <= 0) {
    return;
  }
  absl::Span<const SwapChunkStorage::Chunk> chunks = storage.chunks();
  size_t chunk_index = abs_offset / kSwapChunkSizeBytes;
  size_t offset_in_chunk = abs_offset % kSwapChunkSizeBytes;
  int64_t remaining = size;
  int64_t rel_offset = 0;
  while (remaining > 0) {
    CHECK_LT(chunk_index, chunks.size());
    const SwapChunkStorage::Chunk& chunk = chunks[chunk_index];
    CHECK_LT(offset_in_chunk, chunk.size);
    const size_t available = chunk.size - offset_in_chunk;
    const size_t copy_bytes = std::min<size_t>(available, remaining);
    fn(chunk.data.get() + offset_in_chunk, rel_offset, copy_bytes);
    rel_offset += copy_bytes;
    remaining -= copy_bytes;
    ++chunk_index;
    offset_in_chunk = 0;
  }
}

struct HostChunkSpan {
  // Pointer to the host chunk data.
  uint8_t* ptr;
  // Byte offset within the peer device buffer.
  int64_t offset;
  // Number of bytes to transfer for this chunk.
  int64_t size;
};

absl::InlinedVector<HostChunkSpan, 8> BuildChunkSpans(
    const SwapChunkStorage& storage, int64_t abs_offset, size_t byte_size) {
  absl::InlinedVector<HostChunkSpan, 8> chunks;
  ForEachChunkSpan(
      storage, abs_offset, byte_size,
      [&chunks](uint8_t* chunk_ptr, int64_t rel_offset, int64_t copy_bytes) {
        chunks.push_back(HostChunkSpan{
            .ptr = chunk_ptr,
            .offset = rel_offset,
            .size = copy_bytes,
        });
      });
  return chunks;
}

void CopyHostToChunks(const void* src, const SwapChunkStorage& dst_storage,
                      int64_t dst_abs_offset, int64_t size) {
  const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
  ForEachChunkSpan(
      dst_storage, dst_abs_offset, size,
      [src_bytes](uint8_t* chunk_ptr, int64_t rel_offset, int64_t copy_bytes) {
        std::memcpy(chunk_ptr, src_bytes + rel_offset, copy_bytes);
      });
}

void CopyChunksToHost(const SwapChunkStorage& src_storage,
                      int64_t src_abs_offset, void* dst, int64_t size) {
  uint8_t* dst_bytes = static_cast<uint8_t*>(dst);
  ForEachChunkSpan(
      src_storage, src_abs_offset, size,
      [dst_bytes](uint8_t* chunk_ptr, int64_t rel_offset, int64_t copy_bytes) {
        std::memcpy(dst_bytes + rel_offset, chunk_ptr, copy_bytes);
      });
}

void CopyChunksToChunks(const SwapChunkStorage& src_storage,
                        int64_t src_abs_offset,
                        const SwapChunkStorage& dst_storage,
                        int64_t dst_abs_offset, int64_t size) {
  ForEachChunkSpan(
      src_storage, src_abs_offset, size,
      [&dst_storage, dst_abs_offset](uint8_t* src_chunk_ptr, int64_t rel_offset,
                                     int64_t copy_bytes) {
        CopyHostToChunks(src_chunk_ptr, dst_storage,
                         dst_abs_offset + rel_offset, copy_bytes);
      });
}

// Shared implementation for `SwapRawBuffer::CopyRawHostToDeviceAndReturnEvent`
// and `CopyRawDeviceToHostAndReturnEvent`.
template <typename CopyFn>
absl::StatusOr<PjRtDeviceEventRef> CopyRawAndReturnEvent(
    tsl::RCReference<SwapRawBuffer> self, int64_t offset, int64_t transfer_size,
    PjRtDeviceEventRefVector dependencies, AsyncWorkRunner* async_work_runner,
    CopyFn copy_fn) {
  ABSL_RETURN_IF_ERROR(self->ValidateSlice(offset, transfer_size));
  tsl::AsyncValueRef<SwapEvent> event =
      tsl::MakeUnconstructedAsyncValueRef<SwapEvent>();
  ExecuteWhenReady(PjRtDeviceEventSpan(dependencies), async_work_runner,
                   [self = std::move(self), offset, transfer_size,
                    dependencies = dependencies, event,
                    copy_fn = std::move(copy_fn)]() mutable {
                     if (absl::Status dep_status = GetErrors(dependencies);
                         !dep_status.ok()) {
                       event.SetError(std::move(dep_status));
                       return;
                     }
                     if (transfer_size > 0) {
                       copy_fn(self->storage(), self->slice_offset() + offset,
                               transfer_size);
                     }
                     event.emplace();
                   });
  return PjRtDeviceEventRef(std::move(event));
}

// Joins per-chunk device events into a single event on `memory_space`. If
// `partial_error` is non-OK, waits for any already-launched `events` to finish
// before surfacing `partial_error` so in-flight DMAs cannot outlive buffers.
absl::StatusOr<PjRtDeviceEventRef> JoinChunkEvents(
    PjRtMemorySpace* memory_space,
    absl::InlinedVector<PjRtDeviceEventRef, 8> events,
    absl::Status partial_error = absl::OkStatus()) {
  auto* common_client =
      absl::down_cast<CommonPjRtClient*>(memory_space->client());
  if (events.empty()) {
    if (!partial_error.ok()) {
      return partial_error;
    }
    PjRtDeviceEventPromiseRef promise;
    PjRtDeviceEventRef ready_event;
    ABSL_ASSIGN_OR_RETURN(
        std::tie(promise, ready_event),
        common_client->CreateLinkedEventPromise(memory_space, "CopyChunks"));
    promise.SetReady();
    return ready_event;
  }
  if (partial_error.ok() && events.size() == 1) {
    return std::move(events[0]);
  }

  // Copy dependency refs before moving `events` into the lambda capture so
  // argument evaluation order cannot pass an empty moved-from span.
  absl::InlinedVector<PjRtDeviceEventRef, 8> deps = events;
  PjRtDeviceEventPromiseRef promise;
  PjRtDeviceEventRef combined_event;
  ABSL_ASSIGN_OR_RETURN(
      std::tie(promise, combined_event),
      common_client->CreateLinkedEventPromise(memory_space, "CopyChunks"));
  if (common_client->event_tracking_enabled()) {
    common_client->AddEventDependencies(memory_space, promise.event(), events);
  }
  ExecuteWhenReady(
      absl::MakeConstSpan(deps), common_client->async_work_runner(),
      [promise = std::move(promise), events = std::move(events),
       partial_error = std::move(partial_error)]() mutable {
        if (!partial_error.ok()) {
          promise.SetError(std::move(partial_error));
        } else if (absl::Status status = GetErrors(events); !status.ok()) {
          promise.SetError(std::move(status));
        } else {
          promise.SetReady();
        }
      });
  return combined_event;
}

template <typename ChunkT, typename TransferOneFn>
absl::StatusOr<PjRtDeviceEventRef> CopyChunksWithEvents(
    PjRtMemorySpace* memory_space, absl::Span<const ChunkT> chunks,
    TransferOneFn transfer_one) {
  absl::InlinedVector<PjRtDeviceEventRef, 8> chunk_events;
  chunk_events.reserve(chunks.size());
  for (const ChunkT& chunk : chunks) {
    absl::StatusOr<PjRtDeviceEventRef> ev = transfer_one(chunk);
    if (!ev.ok()) {
      return JoinChunkEvents(memory_space, std::move(chunk_events),
                             ev.status());
    }
    chunk_events.push_back(*std::move(ev));
  }
  return JoinChunkEvents(memory_space, std::move(chunk_events));
}

}  // namespace

absl::StatusOr<SwapChunkStorage> SwapChunkStorage::Allocate(
    size_t total_bytes, HostMemoryAllocator* allocator, int numa_node,
    std::optional<LocalDeviceId> local_device_id) {
  if (allocator == nullptr) {
    return absl::FailedPreconditionError(
        "HostMemoryAllocator is not initialized for swap memory space.");
  }
  if (total_bytes == 0) {
    return SwapChunkStorage(/*total_bytes=*/0, /*chunks=*/{});
  }

  const size_t num_full_chunks = total_bytes / kSwapChunkSizeBytes;
  const size_t remainder_bytes = total_bytes % kSwapChunkSizeBytes;
  const size_t total_chunks = num_full_chunks + (remainder_bytes > 0 ? 1 : 0);

  HostMemoryAllocator::AllocateOptions alloc_options = {
      .numa_node = numa_node,
      .local_device_id = local_device_id,
  };

  std::vector<Chunk> chunks;
  chunks.reserve(total_chunks);
  for (size_t i = 0; i < num_full_chunks; ++i) {
    HostMemoryAllocator::OwnedPtr ptr =
        allocator->Allocate(kSwapChunkSizeBytes, alloc_options);
    if (ptr == nullptr) {
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Failed to allocate %zu MiB swap chunk %zu of %zu (total_bytes: "
          "%zu).",
          kSwapChunkSizeBytes / (1024 * 1024), i, total_chunks, total_bytes));
    }
    chunks.push_back(Chunk{
        .data = std::move(ptr),
        .size = kSwapChunkSizeBytes,
    });
  }
  if (remainder_bytes > 0) {
    HostMemoryAllocator::OwnedPtr ptr =
        allocator->Allocate(remainder_bytes, alloc_options);
    if (ptr == nullptr) {
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Failed to allocate remainder swap chunk (remainder_bytes: %zu, "
          "total_bytes: %zu).",
          remainder_bytes, total_bytes));
    }
    chunks.push_back(Chunk{
        .data = std::move(ptr),
        .size = remainder_bytes,
    });
  }

  return SwapChunkStorage(total_bytes, std::move(chunks));
}

SwapChunkStorage::SwapChunkStorage(size_t total_bytes,
                                   std::vector<Chunk> chunks)
    : total_bytes_(total_bytes), chunks_(std::move(chunks)) {}

SwapDeviceEventPromise::SwapDeviceEventPromise(
    tsl::RCReference<tsl::IndirectAsyncValue> av)
    : av_(std::move(av)) {}

void SwapDeviceEventPromise::Set(PjRtDeviceEventRef event) {
  tsl::AsyncValueRef<SwapEvent> swap_event =
      std::move(event).down_cast<SwapEvent>();
  CHECK(swap_event) << "SwapDeviceEventPromise::Set requires a SwapEvent.";
  av_->ForwardTo(std::move(swap_event));
}

void SwapDeviceEventPromise::SetReady() {
  av_->ForwardTo(tsl::MakeAvailableAsyncValueRef<SwapEvent>());
}

std::pair<PjRtDeviceEventPromiseRef, PjRtDeviceEventRef>
CreateSwapLinkedEventPromise() {
  tsl::RCReference<tsl::IndirectAsyncValue> av = tsl::MakeIndirectAsyncValue();
  PjRtDeviceEventRef event =
      PjRtDeviceEventRef(tsl::AsyncValueRef<SwapEvent>(av));
  auto promise = tsl::MakeRef<SwapDeviceEventPromise>(std::move(av));
  return std::make_pair(PjRtDeviceEventPromiseRef(std::move(promise)),
                        std::move(event));
}

PjRtDeviceEventRef ToSwapEvent(tsl::Future<> future) {
  tsl::AsyncValueRef<SwapEvent> event =
      tsl::MakeUnconstructedAsyncValueRef<SwapEvent>();
  std::move(future).OnReady([event](absl::Status status) mutable {
    if (!status.ok()) {
      event.SetError(std::move(status));
    } else {
      event.emplace();
    }
  });
  return PjRtDeviceEventRef(std::move(event));
}

absl::StatusOr<PjRtRawBufferRef> SwapRawBuffer::Allocate(
    PjRtRawClient* raw_client, PJRT_Memory* memory_space,
    size_t on_device_bytes_count) {
  const LocalDeviceId local_device_id(memory_space->local_device_id);
  ABSL_ASSIGN_OR_RETURN(
      tsl::RCReference<SwapRawBuffer> buffer,
      Allocate(memory_space, on_device_bytes_count,
               raw_client->GetHostMemoryAllocator(),
               raw_client->GetNumaNode(local_device_id), local_device_id,
               raw_client->async_work_runner()));
  return PjRtRawBufferRef(std::move(buffer));
}

absl::StatusOr<tsl::RCReference<SwapRawBuffer>> SwapRawBuffer::Allocate(
    PJRT_Memory* memory_space, size_t on_device_bytes_count,
    HostMemoryAllocator* allocator, int numa_node,
    std::optional<LocalDeviceId> local_device_id,
    AsyncWorkRunner* async_work_runner) {
  ABSL_ASSIGN_OR_RETURN(
      SwapChunkStorage storage,
      SwapChunkStorage::Allocate(on_device_bytes_count, allocator, numa_node,
                                 local_device_id));
  return tsl::MakeRef<SwapRawBuffer>(
      memory_space,
      tsl::MakeAvailableAsyncValueRef<SwapChunkStorage>(std::move(storage)),
      /*slice_offset=*/0, on_device_bytes_count, async_work_runner);
}

SwapRawBuffer::SwapRawBuffer(PJRT_Memory* memory_space,
                             tsl::AsyncValueRef<SwapChunkStorage> storage,
                             int64_t slice_offset, size_t byte_size,
                             AsyncWorkRunner* async_work_runner)
    : memory_space_(memory_space),
      storage_(std::move(storage)),
      slice_offset_(slice_offset),
      byte_size_(byte_size),
      async_work_runner_(async_work_runner) {
  CHECK(storage_.IsConcrete());
}

absl::Status SwapRawBuffer::ValidateSlice(int64_t offset,
                                          int64_t slice_size) const {
  const int64_t buffer_size = byte_size_;
  if (offset < 0 || slice_size < 0 || offset > buffer_size ||
      buffer_size - offset < slice_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid slice of swap buffer (buffer_size: %d, offset: %d, "
        "slice_size: %d).",
        buffer_size, offset, slice_size));
  }
  return absl::OkStatus();
}

absl::StatusOr<PjRtRawBufferRef> SwapRawBuffer::Slice(int64_t offset,
                                                      int64_t size) {
  ABSL_RETURN_IF_ERROR(ValidateSlice(offset, size));
  return tsl::MakeRef<SwapRawBuffer>(memory_space_, storage_,
                                     slice_offset_ + offset, size,
                                     async_work_runner_);
}

absl::StatusOr<PjRtDeviceEventRef>
SwapRawBuffer::CopyRawHostToDeviceAndReturnEvent(
    const void* src, int64_t offset, int64_t transfer_size,
    PjRtDeviceEventRefVector dependencies) {
  return CopyRawAndReturnEvent(
      tsl::FormRef(this), offset, transfer_size, std::move(dependencies),
      async_work_runner_,
      [src](const SwapChunkStorage& storage, int64_t abs_offset, int64_t size) {
        CopyHostToChunks(src, storage, abs_offset, size);
      });
}

absl::StatusOr<PjRtDeviceEventRef>
SwapRawBuffer::CopyRawDeviceToHostAndReturnEvent(
    void* dst, int64_t offset, int64_t transfer_size,
    PjRtDeviceEventRefVector dependencies) {
  return CopyRawAndReturnEvent(
      tsl::FormRef(this), offset, transfer_size, std::move(dependencies),
      async_work_runner_,
      [dst](const SwapChunkStorage& storage, int64_t abs_offset, int64_t size) {
        CopyChunksToHost(storage, abs_offset, dst, size);
      });
}

absl::StatusOr<PjRtDeviceEventRef> SwapRawBuffer::MakeAllocationReadyEvent() {
  return PjRtDeviceEventRef(tsl::MakeAvailableAsyncValueRef<SwapEvent>());
}

void SwapRawBuffer::CopyTo(
    PjRtRawBufferRef dst_raw_buffer,
    PjRtDeviceEventPromiseRef definition_event_promise,
    PjRtDeviceEventPromiseRef src_usage_event_promise,
    absl::AnyInvocable<void(absl::Status) &&> allocation_event) {
  ScheduleCopyTo(/*transfer_dependency_events=*/{}, std::move(dst_raw_buffer),
                 std::move(definition_event_promise),
                 std::move(src_usage_event_promise),
                 std::move(allocation_event));
}

void SwapRawBuffer::ScheduleCopyTo(
    PjRtDeviceEventRefVector transfer_dependency_events,
    PjRtRawBufferRef dst_raw_buffer,
    PjRtDeviceEventPromiseRef definition_event_promise,
    PjRtDeviceEventPromiseRef src_usage_event_promise,
    absl::AnyInvocable<void(absl::Status) &&> allocation_event) {
  ExecuteWhenReady(
      PjRtDeviceEventSpan(transfer_dependency_events), async_work_runner_,
      [src_buf = tsl::FormRef(this), dst_raw_buffer = std::move(dst_raw_buffer),
       definition_event_promise = std::move(definition_event_promise),
       src_usage_event_promise = std::move(src_usage_event_promise),
       allocation_event = std::move(allocation_event),
       transfer_dependency_events = transfer_dependency_events]() mutable {
        auto fail = [&](absl::Status status) {
          if (allocation_event) {
            std::move(allocation_event)(status);
          }
          definition_event_promise.SetError(status);
          src_usage_event_promise.SetError(status);
        };

        if (absl::Status dep_status = GetErrors(transfer_dependency_events);
            !dep_status.ok()) {
          fail(std::move(dep_status));
          return;
        }
        if (src_buf->byte_size_ != dst_raw_buffer->GetOnDeviceSizeInBytes()) {
          fail(absl::InvalidArgumentError(absl::StrFormat(
              "Size mismatch in SwapRawBuffer::ScheduleCopyTo (src_bytes: %zu, "
              "dst_bytes: %zu).",
              src_buf->byte_size_, dst_raw_buffer->GetOnDeviceSizeInBytes())));
          return;
        }
        if (allocation_event) {
          std::move(allocation_event)(absl::OkStatus());
        }

        if (auto* dst_swap = dst_raw_buffer->down_cast<SwapRawBuffer>()) {
          CopyChunksToChunks(src_buf->storage(), src_buf->slice_offset_,
                             dst_swap->storage(), dst_swap->slice_offset_,
                             src_buf->byte_size_);
          definition_event_promise.SetReady();
          src_usage_event_promise.SetReady();
          return;
        }

        if (void* dst_host_ptr = dst_raw_buffer->GetHostPointer();
            dst_host_ptr != nullptr) {
          CopyChunksToHost(src_buf->storage(), src_buf->slice_offset_,
                           dst_host_ptr, src_buf->byte_size_);
          definition_event_promise.SetReady();
          src_usage_event_promise.SetReady();
          return;
        }

        absl::InlinedVector<HostChunkSpan, 8> chunks = BuildChunkSpans(
            src_buf->storage(), src_buf->slice_offset_, src_buf->byte_size_);
        absl::StatusOr<PjRtDeviceEventRef> h2d_event = CopyChunksWithEvents(
            dst_raw_buffer->memory_space(), absl::MakeConstSpan(chunks),
            [&](const HostChunkSpan& chunk) {
              return dst_raw_buffer->CopyRawHostToDeviceAndReturnEvent(
                  chunk.ptr, chunk.offset, chunk.size);
            });

        if (!h2d_event.ok()) {
          definition_event_promise.SetError(h2d_event.status());
          src_usage_event_promise.SetError(h2d_event.status());
          return;
        }

        h2d_event->AndThen(
            [src_buf = std::move(src_buf), dst_raw_buffer,
             src_usage_event_promise = std::move(src_usage_event_promise),
             ev = *h2d_event]() {
              if (std::optional<absl::Status> error = ev.GetErrorIfPresent()) {
                src_usage_event_promise.SetError(std::move(*error));
              } else {
                src_usage_event_promise.SetReady();
              }
            });
        definition_event_promise.Set(*std::move(h2d_event));
      });
}

void SwapRawBuffer::ScheduleCopyFrom(
    PjRtDeviceEventRefVector transfer_dependency_events,
    PjRtRawBufferRef src_raw_buffer,
    PjRtDeviceEventPromiseRef definition_event_promise,
    PjRtDeviceEventPromiseRef src_usage_event_promise,
    absl::AnyInvocable<void(absl::Status) &&> allocation_event) {
  ExecuteWhenReady(
      PjRtDeviceEventSpan(transfer_dependency_events), async_work_runner_,
      [dst_buf = tsl::FormRef(this), src_raw_buffer = std::move(src_raw_buffer),
       definition_event_promise = std::move(definition_event_promise),
       src_usage_event_promise = std::move(src_usage_event_promise),
       allocation_event = std::move(allocation_event),
       transfer_dependency_events = transfer_dependency_events]() mutable {
        auto fail = [&](absl::Status status) {
          if (allocation_event) {
            std::move(allocation_event)(status);
          }
          definition_event_promise.SetError(status);
          src_usage_event_promise.SetError(status);
        };

        if (absl::Status dep_status = GetErrors(transfer_dependency_events);
            !dep_status.ok()) {
          fail(std::move(dep_status));
          return;
        }
        if (dst_buf->byte_size_ != src_raw_buffer->GetOnDeviceSizeInBytes()) {
          fail(absl::InvalidArgumentError(absl::StrFormat(
              "Size mismatch in SwapRawBuffer::ScheduleCopyFrom (src_bytes: "
              "%zu, dst_bytes: %zu).",
              src_raw_buffer->GetOnDeviceSizeInBytes(), dst_buf->byte_size_)));
          return;
        }
        if (allocation_event) {
          std::move(allocation_event)(absl::OkStatus());
        }

        if (auto* src_swap = src_raw_buffer->down_cast<SwapRawBuffer>()) {
          CopyChunksToChunks(src_swap->storage(), src_swap->slice_offset_,
                             dst_buf->storage(), dst_buf->slice_offset_,
                             dst_buf->byte_size_);
          definition_event_promise.SetReady();
          src_usage_event_promise.SetReady();
          return;
        }

        if (void* src_host_ptr = src_raw_buffer->GetHostPointer();
            src_host_ptr != nullptr) {
          CopyHostToChunks(src_host_ptr, dst_buf->storage(),
                           dst_buf->slice_offset_, dst_buf->byte_size_);
          definition_event_promise.SetReady();
          src_usage_event_promise.SetReady();
          return;
        }

        absl::InlinedVector<HostChunkSpan, 8> chunks = BuildChunkSpans(
            dst_buf->storage(), dst_buf->slice_offset_, dst_buf->byte_size_);
        absl::StatusOr<PjRtDeviceEventRef> d2h_event = CopyChunksWithEvents(
            src_raw_buffer->memory_space(), absl::MakeConstSpan(chunks),
            [&](const HostChunkSpan& chunk) {
              return src_raw_buffer->CopyRawDeviceToHostAndReturnEvent(
                  chunk.ptr, chunk.offset, chunk.size);
            });

        if (!d2h_event.ok()) {
          definition_event_promise.SetError(d2h_event.status());
          src_usage_event_promise.SetError(d2h_event.status());
          return;
        }

        d2h_event->AndThen(
            [dst_buf = std::move(dst_buf), src_raw_buffer,
             definition_event_promise = std::move(definition_event_promise),
             ev = *d2h_event]() {
              if (std::optional<absl::Status> error = ev.GetErrorIfPresent()) {
                definition_event_promise.SetError(std::move(*error));
              } else {
                definition_event_promise.SetReady();
              }
            });
        src_usage_event_promise.Set(*std::move(d2h_event));
      });
}

}  // namespace xla
