/* Copyright 2025 The OpenXLA Authors.

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
#include "xla/python/transfer/streaming_ifrt.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/python/transfer/streaming.h"
#include "xla/python/transfer/transfer_socket.pb.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace aux {

class PremappedPjRtBuffer {
 public:
  PremappedPjRtBuffer(absl::Span<uint8_t> data,
                      std::shared_ptr<xla::PjRtClient> client,
                      std::shared_ptr<void> owner)
      : data_(data), client_(client), owner_(std::move(owner)) {}
  ~PremappedPjRtBuffer();

  absl::Span<uint8_t>* data() { return &data_; }

 private:
  absl::Span<uint8_t> data_;
  std::shared_ptr<xla::PjRtClient> client_;
  std::shared_ptr<void> owner_;
};

PremappedPjRtBuffer::~PremappedPjRtBuffer() {
  if (client_) {
    CHECK_OK(client_->DmaUnmap(data_.data()));
  }
}

absl::StatusOr<std::shared_ptr<absl::Span<uint8_t>>> MapPjrtMemory(
    std::shared_ptr<xla::PjRtClient> pjrt_client, void* data,
    size_t buffer_size, std::shared_ptr<void> owner) {
  auto s = pjrt_client->DmaMap(data, buffer_size);
  if (s.code() == ::absl::StatusCode::kUnimplemented) {
    pjrt_client = nullptr;
  } else if (!s.ok()) {
    return s;
  }
  auto result = std::make_shared<PremappedPjRtBuffer>(
      absl::MakeSpan(reinterpret_cast<uint8_t*>(data), buffer_size),
      pjrt_client, std::move(owner));
  return std::shared_ptr<absl::Span<uint8_t>>(result, result->data());
}

absl::StatusOr<std::shared_ptr<absl::Span<uint8_t>>> AllocateAndMapPjrtMemory(
    std::shared_ptr<xla::PjRtClient> client, size_t buffer_size) {
  void* data = nullptr;
  if (posix_memalign(&data, kCpuPageSize, buffer_size) != 0) {
    return absl::InternalError("error in posix_memalign.");
  }
  std::shared_ptr<void> owner =
      std::shared_ptr<void>(data, [](void* data) { free(data); });
  return MapPjrtMemory(client, data, buffer_size, std::move(owner));
}

absl::StatusOr<std::vector<DmaCopyChunk>>
DmaCopyChunk::DivideBufferCopiesEvenly(std::shared_ptr<xla::PjRtBuffer> buffer,
                                       size_t xfer_size, size_t buffer_id) {
  ABSL_ASSIGN_OR_RETURN(size_t copy_size, buffer->GetOnDeviceSizeInBytes());
  size_t total_num_copies = (copy_size + xfer_size - 1) / xfer_size;
  std::vector<DmaCopyChunk> work_units;
  work_units.reserve(total_num_copies);
  for (size_t i = 0; i < total_num_copies; ++i) {
    work_units.push_back(
        DmaCopyChunk{[buffer](void* dst, int64_t offset,
                              int64_t transfer_size) -> tsl::Future<> {
                       return buffer->CopyRawToHost(dst, offset, transfer_size);
                     },
                     buffer_id, i * xfer_size,
                     std::min(copy_size - i * xfer_size, xfer_size)});
  }
  return work_units;
}

PremappedCopierState::PremappedCopierState(
    std::shared_ptr<absl::Span<uint8_t>> scratch,
    size_t max_num_parallel_copies, size_t xfer_size)
    : scratch_(scratch),
      max_num_parallel_copies_(max_num_parallel_copies),
      xfer_size_(xfer_size) {
  max_copies_ = scratch->size() / xfer_size_;
  available_copy_offsets_.reserve(max_copies_);
  for (size_t i = 0; i < max_copies_; ++i) {
    available_copy_offsets_.push_back(reinterpret_cast<char*>(scratch->data()) +
                                      i * xfer_size_);
  }
}

void PremappedCopierState::ScheduleCopy(
    DmaCopyChunk blob, absl::AnyInvocable<void(PremappedCopierState* state,
                                               absl::StatusOr<void*> buf,
                                               const DmaCopyChunk& chunk) &&>
                           on_done) {
  WorkList work_list;
  {
    absl::MutexLock l(mu_);
    work_queue_.push_back(WorkQueueItem{std::move(blob),
                                        nullptr,
                                        base_seq_id_ + work_queue_.size(),
                                        false,
                                        {},
                                        std::move(on_done)});
    work_list = FindWorkLocked();
  }
  StartWorkUnlocked(work_list);
}

void PremappedCopierState::ReturnBuffer(void* buffer) {
  WorkList work_list;
  {
    absl::MutexLock l(mu_);
    available_copy_offsets_.push_back(buffer);
    work_list = FindWorkLocked();
  }
  StartWorkUnlocked(work_list);
}

PremappedCopierState::WorkList PremappedCopierState::FindWorkLocked() {
  WorkList out;
  while (read_seq_id_ < base_seq_id_ + work_queue_.size() &&
         num_parallel_copies_ < max_num_parallel_copies_ &&
         !available_copy_offsets_.empty()) {
    auto* temp = &work_queue_[read_seq_id_ - base_seq_id_];
    temp->dest_buffer = available_copy_offsets_.back();
    ++num_parallel_copies_;
    ++read_seq_id_;
    available_copy_offsets_.pop_back();
    out.push_back(temp);
  }
  return out;
}

void PremappedCopierState::StartWorkUnlocked(const WorkList& work_list) {
  for (WorkQueueItem* work_item : work_list) {
    auto& wu = work_item->work;
    auto copy_fn = std::move(wu.copy_fn);
    std::move(copy_fn)(work_item->dest_buffer, wu.offset, wu.size)
        .OnReady([this, this_shared = shared_from_this(),
                  work_item](absl::Status s) {
          WorkList work_list2;
          {
            absl::MutexLock l(mu_);
            --num_parallel_copies_;
            work_item->is_ready = true;
            work_item->result_status = s;
            if (!currently_flushing_) {
              FlushReadyWorkItemsInOrder();
            }
            work_list2 = FindWorkLocked();
          }
          StartWorkUnlocked(work_list2);
        });
  }
}

void PremappedCopierState::FlushReadyWorkItemsInOrder() {
  while (!work_queue_.empty()) {
    auto* work_item = &work_queue_.front();
    if (!work_item->is_ready) {
      return;
    }
    if (!work_item->result_status.ok()) {
      available_copy_offsets_.push_back(work_item->dest_buffer);
    }
    currently_flushing_ = true;
    mu_.unlock();
    {
      auto on_done_fn = std::move(work_item->on_done);
      if (work_item->result_status.ok()) {
        std::move(on_done_fn)(this, work_item->dest_buffer, work_item->work);
      } else {
        std::move(on_done_fn)(this, work_item->result_status, work_item->work);
      }
    }
    mu_.lock();
    currently_flushing_ = false;
    work_queue_.pop_front();
    ++base_seq_id_;
  }
}

// Invokes on_done at most once, either explicitly via Run() or upon destruction
// when a transfer fails early.
class ScopedOnDone {
 public:
  explicit ScopedOnDone(absl::AnyInvocable<void() &&> on_done)
      : on_done_(std::move(on_done)) {}
  ScopedOnDone(ScopedOnDone&&) noexcept = default;
  ScopedOnDone& operator=(ScopedOnDone&&) = delete;
  ScopedOnDone(const ScopedOnDone&) = delete;
  ScopedOnDone& operator=(const ScopedOnDone&) = delete;
  ~ScopedOnDone() { Run(); }

  void Run() {
    if (on_done_) {
      absl::AnyInvocable<void() &&> cb = std::move(on_done_);
      on_done_ = nullptr;
      std::move(cb)();
    }
  }

 private:
  absl::AnyInvocable<void() &&> on_done_;
};

class DmaDestination : public ChunkDestination {
 public:
  DmaDestination(
      std::shared_ptr<xla::PjRtClient::AsyncHostToDeviceTransferManager> atm,
      int buffer_index, size_t transfer_size)
      : atm_(std::move(atm)),
        buffer_index_(buffer_index),
        buffer_size_(atm_->buffer_size(buffer_index)),
        semaphore_(transfer_size) {}
  ~DmaDestination() override = default;

  absl::Status Put(const void* data, int64_t offset, size_t size,
                   absl::AnyInvocable<void() &&> on_done) override {
    ScopedOnDone on_done_guard(std::move(on_done));
    if (offset < 0 || static_cast<size_t>(offset) > buffer_size_ ||
        buffer_size_ - static_cast<size_t>(offset) < size) {
      Poison(absl::InvalidArgumentError(
          absl::StrFormat("Invalid slicing of buffer size %lld with "
                          "invalid offset %lld, slice size %lld",
                          buffer_size_, offset, size)));
      return absl::OkStatus();
    }
    absl::Status status = semaphore_.DoWork(size, [&](bool is_last_transfer) {
      return atm_->TransferRawDataToSubBuffer(
          buffer_index_, data, offset, size, is_last_transfer,
          [atm = atm_, guard = std::move(on_done_guard)]() mutable {
            guard.Run();
          });
    });
    if (!status.ok()) {
      Poison(status);
    }
    return status;
  }

  void Poison(absl::Status s) override {
    if (s.ok()) {
      return;
    }
    if (semaphore_.Poison()) {
      atm_->SetBufferError(buffer_index_, std::move(s));
    }
  }

 private:
  std::shared_ptr<xla::PjRtClient::AsyncHostToDeviceTransferManager> atm_;
  int buffer_index_;
  size_t buffer_size_;
  // Some small modification of AsyncHostToDeviceTransferManager (eg: optional
  // expected send amount) would make this unnecessary.
  internal::IsLastSemaphore semaphore_;
};

tsl::RCReference<ChunkDestination> MakeDmaDestination(
    std::shared_ptr<xla::PjRtClient::AsyncHostToDeviceTransferManager> atm,
    int buffer_index, size_t transfer_size) {
  return tsl::MakeRef<DmaDestination>(atm, buffer_index, transfer_size);
}

class SlicedRawBufferChunkDestination : public ChunkDestination {
 public:
  SlicedRawBufferChunkDestination(xla::PjRtRawBufferRef raw_buffer,
                                  size_t offset, size_t size,
                                  tsl::Promise<> promise)
      : raw_buffer_(raw_buffer),
        slice_offset_(offset),
        slice_size_(size),
        promise_(std::move(promise)) {}

  absl::Status Put(const void* data, int64_t offset, size_t size,
                   absl::AnyInvocable<void() &&> on_done) override {
    if (offset < 0 || static_cast<size_t>(offset) > slice_size_ ||
        slice_size_ - static_cast<size_t>(offset) < size) {
      std::move(on_done)();
      absl::Status bounds_err = absl::InvalidArgumentError(absl::StrFormat(
          "Out of bounds SlicedRawBufferChunkDestination Copy: %d+%d vs %d",
          offset, size, slice_size_));
      Poison(bounds_err);
      return bounds_err;
    }
    bool already_failed = false;
    bool over_sent = false;
    {
      absl::MutexLock l(mu_);
      if (!saved_status_.ok()) {
        already_failed = true;
      } else if (all_bytes_sent_ || slice_size_ - sent_bytes_ < size) {
        over_sent = true;
      } else {
        sent_bytes_ += size;
        if (sent_bytes_ == slice_size_) {
          all_bytes_sent_ = true;
        }
        ++pending_copies_;
      }
    }
    if (already_failed) {
      std::move(on_done)();
      return absl::OkStatus();
    }
    if (over_sent) {
      std::move(on_done)();
      absl::Status extra_err = absl::InvalidArgumentError(absl::StrFormat(
          "Received extra bytes for SlicedRawBufferChunkDestination of size %d",
          slice_size_));
      Poison(extra_err);
      return extra_err;
    }
    auto future =
        raw_buffer_->CopyRawHostToDevice(data, offset + slice_offset_, size);
    future.OnReady([state = tsl::FormRef(this),
                    on_done = std::move(on_done)](absl::Status s) mutable {
      tsl::Promise<> ready_promise;
      absl::Status final_status;
      {
        absl::MutexLock l(state->mu_);
        --state->pending_copies_;
        state->TakePromiseIfDone(std::move(s), ready_promise, final_status);
      }
      if (ready_promise) {
        ready_promise.Set(std::move(final_status));
      }
      std::move(on_done)();
    });
    return absl::OkStatus();
  }

  void TakePromiseIfDone(absl::Status s, tsl::Promise<>& out_promise,
                         absl::Status& out_status)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (!s.ok() && saved_status_.ok()) {
      saved_status_ = std::move(s);
    }
    if (!promise_ || pending_copies_ > 0) {
      return;
    }
    if (!saved_status_.ok() || all_bytes_sent_) {
      out_promise = std::move(promise_);
      out_status = saved_status_;
    }
  }

  void Poison(absl::Status s) override {
    if (s.ok()) {
      return;
    }
    tsl::Promise<> ready_promise;
    absl::Status final_status;
    {
      absl::MutexLock l(mu_);
      TakePromiseIfDone(std::move(s), ready_promise, final_status);
    }
    if (ready_promise) {
      ready_promise.Set(std::move(final_status));
    }
  }

 private:
  xla::PjRtRawBufferRef raw_buffer_;
  size_t slice_offset_;
  size_t slice_size_;
  absl::Mutex mu_;
  size_t sent_bytes_ ABSL_GUARDED_BY(mu_) = 0;
  size_t pending_copies_ ABSL_GUARDED_BY(mu_) = 0;
  bool all_bytes_sent_ ABSL_GUARDED_BY(mu_) = false;
  absl::Status saved_status_ ABSL_GUARDED_BY(mu_);
  tsl::Promise<> promise_ ABSL_GUARDED_BY(mu_);
};

absl::StatusOr<std::pair<tsl::RCReference<ChunkDestination>, tsl::Future<>>>
CreateSlicedRawBufferDest(xla::PjRtRawBufferRef raw_buffer, size_t offset,
                          size_t size) {
  auto [promise, future] = tsl::MakePromise();
  auto dest = tsl::MakeRef<SlicedRawBufferChunkDestination>(
      raw_buffer, offset, size, std::move(promise));
  return std::make_pair(std::move(dest), std::move(future));
}

RawBufferEntry::RawBufferEntry(std::vector<BufferRef> arrs,
                               std::shared_ptr<PremappedCopierState> state,
                               size_t xfer_size)
    : arrs_(std::move(arrs)),
      consumed_(arrs_.size(), false),
      state_(std::move(state)),
      xfer_size_(xfer_size) {
  for (BufferRef& arr : arrs_) {
    if (!arr.ready_future.IsValid()) {
      arr.ready_future = tsl::Future<>(absl::OkStatus());
    }
  }
}

bool RawBufferEntry::Handle(tsl::RCReference<ConnectionState> state,
                            const SocketTransferPullRequest& req,
                            size_t base_req_id) {
  for (uint64_t bid : req.buffer_ids()) {
    const size_t req_id = base_req_id;
    ++base_req_id;

    size_t num_arrs = 0;
    size_t buf_size = 0;
    bool out_of_range = false;
    bool already_fetched = false;
    tsl::Future<> ready_future;
    xla::PjRtRawBufferRef buffer;
    {
      absl::MutexLock l(mu_);
      num_arrs = arrs_.size();
      if (bid >= num_arrs) {
        out_of_range = true;
      } else {
        buf_size = arrs_[bid].buf_size;
        if (consumed_[bid]) {
          already_fetched = true;
        } else {
          consumed_[bid] = true;
          ++num_consumed_bufs_;
          ready_future = std::move(arrs_[bid].ready_future);
          buffer = std::move(arrs_[bid].buffer);
        }
      }
    }

    if (out_of_range) {
      state->SendError(req_id, 0, 0, true,
                       absl::InternalError(absl::StrFormat(
                           "Buffer id: %d out of range %d", bid, num_arrs)));
      continue;
    }
    if (already_fetched) {
      state->SendError(req_id, 0, buf_size, true,
                       absl::InternalError(absl::StrFormat(
                           "Buffer id: %d has already been fetched", bid)));
      continue;
    }

    std::move(ready_future)
        .OnReady([state, copier_state = state_, xfer_size = xfer_size_,
                  buf_size, req_id, bid,
                  buffer = std::move(buffer)](absl::Status s) {
          if (!s.ok()) {
            state->SendError(req_id, 0, buf_size, true, s);
            return;
          }
          if (!buffer) {
            state->SendError(req_id, 0, buf_size, true,
                             absl::InternalError(absl::StrFormat(
                                 "Buffer id: %d is null", bid)));
            return;
          }
          if (buf_size == 0) {
            state->Send(req_id, "", 0, 0, /*is_largest=*/true, [buffer]() {});
            return;
          }
          if (xfer_size == 0) {
            state->SendError(
                req_id, 0, buf_size, true,
                absl::InvalidArgumentError(
                    "Transfer chunk size must be greater than zero"));
            return;
          }
          void* host_pointer = buffer->GetHostPointer();
          if (host_pointer == nullptr && copier_state == nullptr) {
            state->SendError(
                req_id, 0, buf_size, true,
                absl::InternalError("PremappedCopierState is null"));
            return;
          }
          for (size_t offset = 0; offset < buf_size;) {
            const size_t size = std::min(xfer_size, buf_size - offset);
            const bool is_largest = offset + size == buf_size;
            if (host_pointer != nullptr) {
              state->Send(req_id,
                          reinterpret_cast<char*>(host_pointer) + offset,
                          offset, size, is_largest, [buffer]() {});
            } else {
              DmaCopyChunk blob;
              blob.copy_fn = [buffer](void* dst, int64_t copy_offset,
                                      int64_t transfer_size) -> tsl::Future<> {
                return buffer->CopyRawDeviceToHost(dst, copy_offset,
                                                   transfer_size);
              };
              blob.buffer_id = bid;
              blob.offset = offset;
              blob.size = size;
              copier_state->ScheduleCopy(
                  std::move(blob),
                  [req_id, state, copier_state, is_largest](
                      PremappedCopierState* copier_state_ptr,
                      absl::StatusOr<void*> buf, const DmaCopyChunk& chunk) {
                    if (!buf.ok()) {
                      state->SendError(req_id, chunk.offset, chunk.size,
                                       is_largest, buf.status());
                      return;
                    }
                    state->Send(req_id, buf.value(), chunk.offset, chunk.size,
                                is_largest,
                                [copier_state, buf = buf.value()]() {
                                  copier_state->ReturnBuffer(buf);
                                });
                  });
            }
            offset += size;
          }
        });
  }

  absl::MutexLock l(mu_);
  return num_consumed_bufs_ == arrs_.size();
}

bool RawBufferEntry::IsComplete() const {
  absl::MutexLock l(mu_);
  return num_consumed_bufs_ == arrs_.size();
}

PjRtBufferEntry::PjRtBufferEntry(std::vector<BufferRef> arrs,
                                 std::shared_ptr<PremappedCopierState> state,
                                 size_t xfer_size)
    : arrs_(std::move(arrs)),
      consumed_(arrs_.size(), false),
      state_(std::move(state)),
      xfer_size_(xfer_size) {
  for (BufferRef& arr : arrs_) {
    if (!arr.ready_future.IsValid()) {
      arr.ready_future = arr.buffer != nullptr
                             ? arr.buffer->GetReadyFuture()
                             : tsl::Future<>(absl::OkStatus());
    }
  }
}

bool PjRtBufferEntry::Handle(tsl::RCReference<ConnectionState> state,
                             const SocketTransferPullRequest& req,
                             size_t base_req_id) {
  for (uint64_t bid : req.buffer_ids()) {
    const size_t req_id = base_req_id;
    ++base_req_id;

    size_t num_arrs = 0;
    size_t buf_size = 0;
    bool out_of_range = false;
    bool already_fetched = false;
    tsl::Future<> ready_future;
    std::shared_ptr<xla::PjRtBuffer> buffer;
    {
      absl::MutexLock l(mu_);
      num_arrs = arrs_.size();
      if (bid >= num_arrs) {
        out_of_range = true;
      } else {
        buf_size = arrs_[bid].buf_size;
        if (consumed_[bid]) {
          already_fetched = true;
        } else {
          consumed_[bid] = true;
          ++num_consumed_bufs_;
          ready_future = std::move(arrs_[bid].ready_future);
          buffer = std::move(arrs_[bid].buffer);
        }
      }
    }

    if (out_of_range) {
      state->SendError(req_id, 0, 0, true,
                       absl::InternalError(absl::StrFormat(
                           "Buffer id: %d out of range %d", bid, num_arrs)));
      continue;
    }
    if (already_fetched) {
      state->SendError(req_id, 0, buf_size, true,
                       absl::InternalError(absl::StrFormat(
                           "Buffer id: %d has already been fetched", bid)));
      continue;
    }

    std::move(ready_future)
        .OnReady([state, copier_state = state_, xfer_size = xfer_size_,
                  buf_size, req_id, bid,
                  buffer = std::move(buffer)](absl::Status s) {
          if (!s.ok()) {
            state->SendError(req_id, 0, buf_size, true, s);
            return;
          }
          if (!buffer) {
            state->SendError(req_id, 0, buf_size, true,
                             absl::InternalError(absl::StrFormat(
                                 "Buffer id: %d is null", bid)));
            return;
          }
          if (buf_size == 0) {
            state->Send(req_id, "", 0, 0, /*is_largest=*/true, [buffer]() {});
            return;
          }
          if (xfer_size == 0) {
            state->SendError(
                req_id, 0, buf_size, true,
                absl::InvalidArgumentError(
                    "Transfer chunk size must be greater than zero"));
            return;
          }
          if (copier_state == nullptr) {
            state->SendError(
                req_id, 0, buf_size, true,
                absl::InternalError("PremappedCopierState is null"));
            return;
          }
          for (size_t offset = 0; offset < buf_size;) {
            const size_t size = std::min(xfer_size, buf_size - offset);
            const bool is_largest = offset + size == buf_size;
            DmaCopyChunk blob;
            blob.copy_fn = [buffer](void* dst, int64_t copy_offset,
                                    int64_t transfer_size) -> tsl::Future<> {
              return buffer->CopyRawToHost(dst, copy_offset, transfer_size);
            };
            blob.buffer_id = bid;
            blob.offset = offset;
            blob.size = size;
            copier_state->ScheduleCopy(
                std::move(blob),
                [req_id, state, copier_state, is_largest](
                    PremappedCopierState* copier_state_ptr,
                    absl::StatusOr<void*> buf, const DmaCopyChunk& chunk) {
                  if (!buf.ok()) {
                    state->SendError(req_id, chunk.offset, chunk.size,
                                     is_largest, buf.status());
                    return;
                  }
                  state->Send(req_id, buf.value(), chunk.offset, chunk.size,
                              is_largest, [copier_state, buf = buf.value()]() {
                                copier_state->ReturnBuffer(buf);
                              });
                });
            offset += size;
          }
        });
  }

  absl::MutexLock l(mu_);
  return num_consumed_bufs_ == arrs_.size();
}

bool PjRtBufferEntry::IsComplete() const {
  absl::MutexLock l(mu_);
  return num_consumed_bufs_ == arrs_.size();
}

}  // namespace aux
