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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "tsl/platform/casts.h"
#include "xla/future.h"
#include "xla/literal.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"
#include "xla/python/transfer/streaming.h"
#include "xla/python/transfer/test_pattern.h"
#include "xla/python/transfer/transfer_socket.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace aux {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

xla::ifrt::PjRtDevice* GetOtherDevice(xla::ifrt::ArrayRef arr) {
  auto* ifrt_client =
      xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtClient>(arr->client());
  return xla::ifrt::dyn_cast<xla::ifrt::PjRtDevice>(ifrt_client->devices()[1]);
}

xla::ifrt::PjRtClient* GetIfrtClient(xla::ifrt::ArrayRef arr) {
  return xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtClient>(arr->client());
}

xla::Shape ShapeFromIfrt(xla::ifrt::ArrayRef arr) {
  auto* pjrt_arr =
      xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleArray>(arr.get());
  auto buffer = pjrt_arr->pjrt_buffers()[0].get();
  return buffer->on_device_shape();
}

struct SingleBufferCopyPlan {
  std::vector<tsl::RCReference<ChunkDestination>> dests;
  std::vector<xla::ifrt::ArrayRef> arrays;
};

// Single buffer copy plan example.
absl::StatusOr<SingleBufferCopyPlan> SetupTransferDestList(
    xla::Shape shape, xla::ifrt::PjRtDevice* device,
    xla::ifrt::PjRtClient* ifrt_client, size_t xfer_size) {
  auto* pjrt_client = ifrt_client->pjrt_client();
  // CHECK_EQ(pjrt_client->platform_id(), xla::TpuId());
  ABSL_ASSIGN_OR_RETURN(auto* pjrt_memory_space,
                        device->pjrt_device()->default_memory_space());
  ABSL_ASSIGN_OR_RETURN(auto atm_owned,
                        pjrt_client->CreateBuffersForAsyncHostToDevice(
                            {shape}, pjrt_memory_space));
  auto atm = std::shared_ptr<xla::PjRtClient::AsyncHostToDeviceTransferManager>(
      std::move(atm_owned));
  SingleBufferCopyPlan results;
  size_t copy_size = xla::ShapeUtil::ByteSizeOf(shape);

  results.dests.push_back(MakeDmaDestination(atm, 0, copy_size));
  // `CreateBuffersForAsyncHostToDevice` uses a default layout.
  ABSL_ASSIGN_OR_RETURN(
      auto arr, ifrt_client->CreatePjRtArray(atm->RetrieveBuffer(0),
                                             /*has_custom_layout=*/false));
  results.arrays.push_back(std::move(arr));
  return results;
}

void CopyIntoDest(tsl::RCReference<ChunkDestination> dest,
                  tsl::RCReference<xla::ifrt::Array> arr, size_t xfer_size,
                  size_t buffer_id) {
  std::shared_ptr<xla::PjRtClient> pjrt_client =
      absl::down_cast<xla::ifrt::PjRtClient*>(arr->client())
          ->shared_ptr_pjrt_client();

  auto* array = static_cast<xla::ifrt::PjRtCompatibleArray*>(arr.get());
  TF_ASSERT_OK_AND_ASSIGN(
      auto scratch, AllocateAndMapPjrtMemory(pjrt_client, 1024 * 1024 * 16));
  TF_ASSERT_OK_AND_ASSIGN(auto src_work_units,
                          DmaCopyChunk::DivideBufferCopiesEvenly(
                              array->pjrt_buffers()[0], xfer_size, 0));
  auto cstate = std::make_shared<PremappedCopierState>(scratch, 4, xfer_size);

  absl::Mutex mu;
  struct LocalQueueInfo {
    void* buff;
    size_t offset;
    size_t size;
  };
  std::deque<LocalQueueInfo> local_queue;

  for (size_t i = 0; i < src_work_units.size(); ++i) {
    cstate->ScheduleCopy(
        std::move(src_work_units[i]),
        [&mu, &local_queue](PremappedCopierState* state,
                            absl::StatusOr<void*> buf,
                            const DmaCopyChunk& chunk) {
          CHECK_OK(buf.status());
          absl::MutexLock l(mu);
          local_queue.push_back(LocalQueueInfo{*buf, chunk.offset, chunk.size});
        });
  }
  for (size_t i = 0; i < src_work_units.size(); ++i) {
    auto cond = [&]() -> bool { return !local_queue.empty(); };
    mu.LockWhen(absl::Condition(&cond));
    auto state = local_queue.front();
    local_queue.pop_front();
    mu.unlock();
    TF_ASSERT_OK(
        dest->Put(state.buff, state.offset, state.size,
                  [cstate, buf = state.buff]() { cstate->ReturnBuffer(buf); }));
  }
}

absl::StatusOr<std::vector<int32_t>> FetchResult(
    tsl::RCReference<xla::ifrt::Array> arr, size_t result_size) {
  std::vector<int32_t> result;
  result.resize(result_size);
  ABSL_RETURN_IF_ERROR(
      arr->CopyToHostBuffer(result.data(), std::nullopt,
                            xla::ifrt::ArrayCopySemantics::kReuseInput)
          .Await());
  return result;
}

TEST(PremappedCopierState, FreeCycle) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  std::shared_ptr<xla::PjRtClient> pjrt_client =
      absl::down_cast<xla::ifrt::PjRtClient*>(client.get())
          ->shared_ptr_pjrt_client();
  TF_ASSERT_OK_AND_ASSIGN(
      auto scratch, AllocateAndMapPjrtMemory(pjrt_client, 1024 * 1024 * 16));
  auto cstate = std::make_shared<PremappedCopierState>(scratch, 4, 4096);
  std::vector<void*> buffers_to_return;
  for (size_t i = 0; i < 2; ++i) {
    cstate->ScheduleCopy(
        {/*copy_fn=*/[](void* dst, int64_t offset,
                        int64_t transfer_size) -> xla::Future<> {
           return xla::Future<>(absl::OkStatus());
         },
         /*buffer_id=*/0,
         /*offset=*/100,
         /*size=*/100},
        [&buffers_to_return](PremappedCopierState* state,
                             absl::StatusOr<void*> buf,
                             const DmaCopyChunk& chunk) {
          CHECK_OK(buf.status());
          buffers_to_return.push_back(buf.value());
        });
  }
  class BufferReturner {
   public:
    explicit BufferReturner(absl::AnyInvocable<void() &&> on_done)
        : on_done_(std::move(on_done)) {}
    ~BufferReturner() { std::move(on_done_)(); }

   private:
    absl::AnyInvocable<void() &&> on_done_;
  };
  cstate->ScheduleCopy(
      {/*copy_fn=*/[buffer = std::make_unique<BufferReturner>(
                        [b = buffers_to_return[0], cstate]() {
                          cstate->ReturnBuffer(b);
                        })](void* dst, int64_t offset,
                            int64_t transfer_size) -> xla::Future<> {
         return xla::Future<>(absl::OkStatus());
       },
       /*buffer_id=*/0,
       /*offset=*/100,
       /*size=*/100},
      [buffer = std::make_unique<BufferReturner>(
           [b = buffers_to_return[1], cstate]() { cstate->ReturnBuffer(b); })](
          PremappedCopierState* state, absl::StatusOr<void*> buf,
          const DmaCopyChunk& chunk) {
        CHECK_OK(buf.status());
        state->ReturnBuffer(buf.value());
      });
}

TEST(PremappedCopierState, RoundTrip) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  size_t xfer_size = 1024 * 1024;
  auto test_pattern = tests::CreateTestPattern(0, 16l * 1024 * 1024);
  TF_ASSERT_OK_AND_ASSIGN(
      auto arr, tests::CopyTestPatternToDevice(
                    client.get(), client->devices()[0], test_pattern));
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest_copy_plan,
      SetupTransferDestList(ShapeFromIfrt(arr), GetOtherDevice(arr),
                            GetIfrtClient(arr), xfer_size));
  CopyIntoDest(dest_copy_plan.dests[0], arr, xfer_size, 0);

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, FetchResult(dest_copy_plan.arrays[0], test_pattern.size()));
  EXPECT_EQ(result, test_pattern);
}

TEST(PremappedCopierState, RoundTripSlicedRaw) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  size_t xfer_size = 1024 * 1024;
  auto test_pattern = tests::CreateTestPattern(0, 16l * 1024 * 1024);
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest_arr, tests::CopyTestPatternToDevice(
                         client.get(), client->devices()[0],
                         std::vector<int32_t>(test_pattern.size() * 3 / 2, 0)));
  TF_ASSERT_OK(dest_arr->GetReadyFuture().Await());
  TF_ASSERT_OK_AND_ASSIGN(
      auto arr, tests::CopyTestPatternToDevice(
                    client.get(), client->devices()[0], test_pattern));
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest_raw_buffer,
      xla::PjRtRawBuffer::CreateRawAliasOfBuffer(
          static_cast<xla::ifrt::PjRtCompatibleArray*>(dest_arr.get())
              ->pjrt_buffers()[0]
              .get()));
  TF_ASSERT_OK_AND_ASSIGN(
      auto fut_and_dest,
      CreateSlicedRawBufferDest(
          dest_raw_buffer, dest_raw_buffer->GetOnDeviceSizeInBytes() / 3,
          dest_raw_buffer->GetOnDeviceSizeInBytes() * 2 / 3));
  CopyIntoDest(std::move(fut_and_dest.first), arr, xfer_size, 0);
  TF_ASSERT_OK(fut_and_dest.second.Await());

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          FetchResult(dest_arr, test_pattern.size() * 3 / 2));
  std::vector<int32_t> padded_test_pattern(test_pattern.size() / 2, 0);
  padded_test_pattern.insert(padded_test_pattern.end(), test_pattern.begin(),
                             test_pattern.end());
  EXPECT_EQ(result, padded_test_pattern);
}

TEST(PremappedCopierState, PoisonSlicedRaw) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(auto dest_arr, tests::CopyTestPatternToDevice(
                                             client.get(), client->devices()[0],
                                             std::vector<int32_t>(4096, 0)));
  TF_ASSERT_OK(dest_arr->GetReadyFuture().Await());
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest_raw_buffer,
      xla::PjRtRawBuffer::CreateRawAliasOfBuffer(
          static_cast<xla::ifrt::PjRtCompatibleArray*>(dest_arr.get())
              ->pjrt_buffers()[0]
              .get()));
  TF_ASSERT_OK_AND_ASSIGN(auto fut_and_dest,
                          CreateSlicedRawBufferDest(dest_raw_buffer, 0, 4096));
  fut_and_dest.first->Poison(absl::InternalError("Poisoning."));
  ASSERT_FALSE(fut_and_dest.second.Await().ok());
}

TEST(Semaphore, Basic) {
  internal::IsLastSemaphore semaphore(15);
  for (size_t i = 0; i < 10; ++i) {
    CHECK_OK(semaphore.DoWork(1, [&](bool is_last) -> absl::Status {
      EXPECT_FALSE(is_last);
      return absl::OkStatus();
    }));
  }
  CHECK_OK(semaphore.DoWork(5, [&](bool is_last) -> absl::Status {
    EXPECT_TRUE(is_last);
    return absl::OkStatus();
  }));
}

TEST(Semaphore, Async) {
  internal::IsLastSemaphore o_semaphore(16);

  absl::Notification t1_done;
  absl::Notification t2_done;
  absl::Mutex mu;
  size_t thread_id = 0;
  auto thread_wait_flip = [&thread_id, &mu](size_t my_thread_id) {
    auto cond = [&] { return thread_id == my_thread_id; };
    mu.LockWhen(absl::Condition(&cond));
    mu.unlock();
  };
  auto thread_flip = [&thread_id, &mu](size_t my_thread_id) {
    absl::MutexLock l(mu);
    thread_id = 1 - thread_id;
  };

  std::unique_ptr<tsl::Thread> t1(
      tsl::Env::Default()->StartThread({}, "t1", [&]() {
        for (size_t i = 0; i < 8; ++i) {
          thread_wait_flip(0);
          CHECK_OK(o_semaphore.DoWork(1, [&](bool is_last) -> absl::Status {
            thread_flip(0);
            EXPECT_FALSE(is_last);
            return absl::OkStatus();
          }));
        }
      }));
  std::unique_ptr<tsl::Thread> t2(
      tsl::Env::Default()->StartThread({}, "t2", [&]() {
        for (size_t i = 0; i < 8; ++i) {
          thread_wait_flip(1);
          CHECK_OK(o_semaphore.DoWork(1, [&](bool is_last) -> absl::Status {
            thread_flip(1);
            if (i == 7) {
              EXPECT_TRUE(is_last);
            } else {
              EXPECT_FALSE(is_last);
            }
            return absl::OkStatus();
          }));
        }
      }));
}

TEST(Semaphore, Poison) {
  internal::IsLastSemaphore o_semaphore(16);

  EXPECT_TRUE(o_semaphore.Poison());
  EXPECT_FALSE(o_semaphore.Poison());
  EXPECT_FALSE(o_semaphore.Poison());
}

TEST(Semaphore, ConcurrentPoisonWaitersWakeOnceWorkCompletes) {
  // When a final DoWork completes with OkStatus while Poison is waiting in
  // Await, Poison must observe completed_ == true after waking and return
  // false.
  {
    internal::IsLastSemaphore sem(4);
    absl::Notification in_cb;
    absl::Notification finish_cb;
    absl::Notification poison_started;
    std::unique_ptr<tsl::Thread> worker(
        tsl::Env::Default()->StartThread({}, "sem-last-worker", [&]() {
          EXPECT_OK(sem.DoWork(4, [&](bool is_last) {
            EXPECT_TRUE(is_last);
            in_cb.Notify();
            finish_cb.WaitForNotification();
            return absl::OkStatus();
          }));
        }));
    in_cb.WaitForNotification();
    bool poison_result = true;
    std::unique_ptr<tsl::Thread> poisoner(
        tsl::Env::Default()->StartThread({}, "sem-poisoner", [&]() {
          poison_started.Notify();
          poison_result = sem.Poison();
        }));
    poison_started.WaitForNotification();
    finish_cb.Notify();
    worker.reset();
    poisoner.reset();
    EXPECT_FALSE(poison_result);

    bool post_complete_cb_called = false;
    EXPECT_OK(sem.DoWork(/*value=*/0, [&](bool is_last) {
      post_complete_cb_called = true;
      return absl::OkStatus();
    }));
    EXPECT_FALSE(post_complete_cb_called);
  }

  // After Poison succeeds on an incomplete semaphore, subsequent DoWork returns
  // OkStatus without invoking the callback.
  {
    internal::IsLastSemaphore sem(8);
    EXPECT_TRUE(sem.Poison());
    bool post_poison_cb_called = false;
    EXPECT_OK(sem.DoWork(/*value=*/4, [&](bool is_last) {
      post_poison_cb_called = true;
      return absl::OkStatus();
    }));
    EXPECT_FALSE(post_poison_cb_called);
  }

  // When two callers race in Poison while non final DoWork is active, exactly
  // one returns true and the second wakes and returns false.
  {
    internal::IsLastSemaphore sem(8);
    absl::Notification in_cb;
    absl::Notification finish_cb;
    std::unique_ptr<tsl::Thread> worker(
        tsl::Env::Default()->StartThread({}, "sem-nonlast-worker", [&]() {
          EXPECT_OK(sem.DoWork(4, [&](bool is_last) {
            EXPECT_FALSE(is_last);
            in_cb.Notify();
            finish_cb.WaitForNotification();
            return absl::OkStatus();
          }));
        }));
    in_cb.WaitForNotification();
    bool p1 = false;
    bool p2 = false;
    absl::Notification p1_started;
    absl::Notification p2_started;
    std::unique_ptr<tsl::Thread> t1(
        tsl::Env::Default()->StartThread({}, "poison-1", [&]() {
          p1_started.Notify();
          p1 = sem.Poison();
        }));
    std::unique_ptr<tsl::Thread> t2(
        tsl::Env::Default()->StartThread({}, "poison-2", [&]() {
          p2_started.Notify();
          p2 = sem.Poison();
        }));
    p1_started.WaitForNotification();
    p2_started.WaitForNotification();
    finish_cb.Notify();
    worker.reset();
    t1.reset();
    t2.reset();
    EXPECT_NE(p1, p2);
  }
}

class RecordingBufferConnectionState : public ConnectionState {
 public:
  explicit RecordingBufferConnectionState(
      absl::Notification* on_first_error = nullptr)
      : on_first_error_(on_first_error) {}

  void Send(size_t req_id, const void* data, size_t offset, size_t size,
            bool is_largest, absl::AnyInvocable<void() &&> on_done) override {
    std::string chunk =
        size == 0 ? std::string()
                  : std::string(reinterpret_cast<const char*>(data), size);
    std::move(on_done)();

    absl::MutexLock l(mu_);
    InFlightBuffer& in_flight = in_flight_[req_id];
    in_flight.chunks.emplace_back(offset, std::move(chunk));
    if (is_largest) {
      in_flight.transferred_size += static_cast<uint64_t>(offset);
    } else {
      in_flight.transferred_size -= static_cast<uint64_t>(size);
    }
    if (in_flight.transferred_size == 0) {
      std::sort(in_flight.chunks.begin(), in_flight.chunks.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
      std::string assembled;
      for (const auto& [chunk_offset, bytes] : in_flight.chunks) {
        EXPECT_EQ(chunk_offset, assembled.size());
        assembled += bytes;
      }
      completed_buffers_[req_id] = std::move(assembled);
      in_flight_.erase(req_id);
      ++completed_events_;
    }
  }

  void SendError(size_t req_id, size_t offset, size_t size, bool is_largest,
                 absl::Status status) override {
    bool should_notify = false;
    {
      absl::MutexLock l(mu_);
      errors_[req_id] = std::move(status);
      error_sizes_[req_id] = size;
      ++completed_events_;
      if (on_first_error_ != nullptr && !notified_first_error_) {
        notified_first_error_ = true;
        should_notify = true;
      }
    }
    if (should_notify) {
      on_first_error_->Notify();
    }
  }

  void WaitForEvents(size_t expected_events) {
    absl::MutexLock l(mu_);
    auto cond = [this, expected_events]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
      return completed_events_ >= expected_events;
    };
    mu_.Await(absl::Condition(&cond));
  }

  absl::flat_hash_map<size_t, std::string> completed_buffers() {
    absl::MutexLock l(mu_);
    return completed_buffers_;
  }

  absl::flat_hash_map<size_t, absl::Status> errors() {
    absl::MutexLock l(mu_);
    return errors_;
  }

  absl::flat_hash_map<size_t, size_t> error_sizes() {
    absl::MutexLock l(mu_);
    return error_sizes_;
  }

 private:
  struct InFlightBuffer {
    uint64_t transferred_size = 0;
    std::vector<std::pair<size_t, std::string>> chunks;
  };

  absl::Notification* on_first_error_;
  absl::Mutex mu_;
  bool notified_first_error_ ABSL_GUARDED_BY(mu_) = false;
  size_t completed_events_ ABSL_GUARDED_BY(mu_) = 0;
  absl::flat_hash_map<size_t, InFlightBuffer> in_flight_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<size_t, std::string> completed_buffers_
      ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<size_t, absl::Status> errors_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<size_t, size_t> error_sizes_ ABSL_GUARDED_BY(mu_);
};

class BlockingCallbackConnectionState : public ConnectionState {
 public:
  BlockingCallbackConnectionState(absl::Notification* entered,
                                  absl::Notification* unblock)
      : entered_(entered), unblock_(unblock) {}

  void Send(size_t req_id, const void* data, size_t offset, size_t size,
            bool is_largest, absl::AnyInvocable<void() &&> on_done) override {
    std::move(on_done)();
    BlockOnce();
  }

  void SendError(size_t req_id, size_t offset, size_t size, bool is_largest,
                 absl::Status status) override {
    BlockOnce();
  }

 private:
  void BlockOnce() {
    if (!entered_->HasBeenNotified()) {
      entered_->Notify();
      unblock_->WaitForNotification();
    }
  }

  absl::Notification* entered_;
  absl::Notification* unblock_;
};

absl::StatusOr<std::shared_ptr<PremappedCopierState>> MakeTestCopier(
    xla::ifrt::Client* client, size_t xfer_size) {
  std::shared_ptr<xla::PjRtClient> pjrt_client =
      absl::down_cast<xla::ifrt::PjRtClient*>(client)->shared_ptr_pjrt_client();
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<absl::Span<uint8_t>> scratch,
                        AllocateAndMapPjrtMemory(pjrt_client, xfer_size * 8));
  return std::make_shared<PremappedCopierState>(
      scratch, /*max_num_parallel_copies=*/4, xfer_size);
}

absl::StatusOr<std::vector<int32_t>> DecodeInt32Payload(
    absl::string_view bytes) {
  if (bytes.size() % sizeof(int32_t) != 0) {
    return absl::InvalidArgumentError("Payload size is not a multiple of 4");
  }
  const size_t count = bytes.size() / sizeof(int32_t);
  std::vector<int32_t> values(count);
  for (size_t i = 0; i < count; ++i) {
    std::array<char, sizeof(int32_t)> raw;
    std::copy_n(bytes.data() + i * sizeof(int32_t), sizeof(int32_t),
                raw.begin());
    values[i] = absl::bit_cast<int32_t>(raw);
  }
  return values;
}

TEST(PjRtBufferEntryTest, ConcurrentAwaitPullReplayAndHandle) {
  ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  constexpr size_t kXferSize = 4096;
  constexpr size_t kElementsPerBuffer = 3072;  // 12 KiB (3 chunks per buffer).
  constexpr int kNumBuffers = 4;
  constexpr uint64_t kUuid = 0;

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<PremappedCopierState> copier,
                       MakeTestCopier(client.get(), kXferSize));

  for (int iter = 0; iter < 8; ++iter) {
    std::vector<std::vector<int32_t>> expected_patterns;
    std::vector<xla::ifrt::ArrayRef> keep_alive_arrays;
    std::vector<PjRtBufferEntry::BufferRef> refs;
    expected_patterns.reserve(kNumBuffers);
    keep_alive_arrays.reserve(kNumBuffers);
    refs.reserve(kNumBuffers);

    for (int b = 0; b < kNumBuffers; ++b) {
      expected_patterns.push_back(
          tests::CreateTestPattern(b * 1000 + iter, kElementsPerBuffer));
      ASSERT_OK_AND_ASSIGN(
          xla::ifrt::ArrayRef arr,
          tests::CopyTestPatternToDevice(client.get(), client->devices()[0],
                                         expected_patterns.back()));
      auto* pjrt_arr =
          absl::down_cast<xla::ifrt::PjRtCompatibleArray*>(arr.get());
      std::shared_ptr<xla::PjRtBuffer> pjrt_buf = pjrt_arr->pjrt_buffers()[0];
      ASSERT_OK_AND_ASSIGN(size_t buf_size, pjrt_buf->GetOnDeviceSizeInBytes());
      refs.push_back({std::move(pjrt_buf), buf_size});
      keep_alive_arrays.push_back(std::move(arr));
    }

    PullTable table;
    absl::Notification replay_started;
    auto replay_conn =
        tsl::MakeRef<RecordingBufferConnectionState>(&replay_started);

    // Plant a paused pull request before AwaitPull with out of range padding,
    // valid slots, and a duplicate slot within the same request.
    SocketTransferPullRequest planted;
    planted.set_uuid(kUuid);
    for (int pad = 0; pad < 256; ++pad) {
      planted.add_buffer_ids(9999);
    }
    planted.add_buffer_ids(0);
    planted.add_buffer_ids(0);
    planted.add_buffer_ids(1);
    const size_t planted_count = planted.buffer_ids_size();
    table.Handle(replay_conn, planted, /*base_req_id=*/0);

    auto worker_conn = tsl::MakeRef<RecordingBufferConnectionState>();
    std::unique_ptr<tsl::Thread> worker(
        tsl::Env::Default()->StartThread({}, "pjrt-pull-worker", [&]() {
          replay_started.WaitForNotification();
          for (int bid = 0; bid < kNumBuffers; ++bid) {
            SocketTransferPullRequest req;
            req.set_uuid(kUuid);
            req.add_buffer_ids(bid);
            table.Handle(worker_conn, req, /*base_req_id=*/1000 + bid);
          }
        }));

    auto entry =
        tsl::MakeRef<PjRtBufferEntry>(std::move(refs), copier, kXferSize);
    table.AwaitPull(kUuid, std::move(entry));
    worker.reset();

    replay_conn->WaitForEvents(planted_count);
    worker_conn->WaitForEvents(kNumBuffers);

    auto replay_buffers = replay_conn->completed_buffers();
    auto worker_buffers = worker_conn->completed_buffers();
    auto replay_errors = replay_conn->errors();
    auto worker_errors = worker_conn->errors();

    // Exactly kNumBuffers total buffer transfers must succeed across both
    // racing callers, and every buffer must match its expected pattern.
    ASSERT_EQ(replay_buffers.size() + worker_buffers.size(),
              static_cast<size_t>(kNumBuffers));
    for (int b = 0; b < kNumBuffers; ++b) {
      size_t wins = 0;
      if (b < 2) {
        const size_t planted_req_id = (b == 0) ? 256u : 258u;
        auto it = replay_buffers.find(planted_req_id);
        if (it != replay_buffers.end()) {
          EXPECT_THAT(DecodeInt32Payload(it->second),
                      IsOkAndHolds(expected_patterns[b]));
          ++wins;
        } else {
          EXPECT_THAT(replay_errors[planted_req_id],
                      StatusIs(absl::StatusCode::kInternal,
                               HasSubstr("already been fetched")));
        }
      }
      auto worker_it = worker_buffers.find(1000 + b);
      if (worker_it != worker_buffers.end()) {
        EXPECT_THAT(DecodeInt32Payload(worker_it->second),
                    IsOkAndHolds(expected_patterns[b]));
        ++wins;
      } else {
        EXPECT_THAT(worker_errors[1000 + b],
                    StatusIs(absl::StatusCode::kInternal,
                             HasSubstr("already been fetched")));
      }
      EXPECT_EQ(wins, 1u);
    }

    // The duplicate request for buffer 0 at index 257 must always fail with
    // the buffer's size, while out of range index 0 fails with size 0.
    auto replay_error_sizes = replay_conn->error_sizes();
    EXPECT_THAT(replay_errors[257],
                StatusIs(absl::StatusCode::kInternal,
                         HasSubstr("already been fetched")));
    EXPECT_EQ(replay_error_sizes[257], kElementsPerBuffer * sizeof(int32_t));
    EXPECT_EQ(replay_error_sizes[0], 0u);
  }
}

TEST(RawBufferEntryTest, ConcurrentAwaitPullReplayAndHandle) {
  ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  constexpr size_t kXferSize = 4096;
  constexpr size_t kElementsPerBuffer = 3072;
  constexpr int kNumBuffers = 4;
  constexpr uint64_t kUuid = 0;

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<PremappedCopierState> copier,
                       MakeTestCopier(client.get(), kXferSize));

  for (int iter = 0; iter < 8; ++iter) {
    std::vector<std::vector<int32_t>> expected_patterns;
    std::vector<xla::ifrt::ArrayRef> keep_alive_arrays;
    std::vector<RawBufferEntry::BufferRef> refs;
    expected_patterns.reserve(kNumBuffers);
    keep_alive_arrays.reserve(kNumBuffers);
    refs.reserve(kNumBuffers);

    for (int b = 0; b < kNumBuffers; ++b) {
      expected_patterns.push_back(
          tests::CreateTestPattern(b * 2000 + iter, kElementsPerBuffer));
      ASSERT_OK_AND_ASSIGN(
          xla::ifrt::ArrayRef arr,
          tests::CopyTestPatternToDevice(client.get(), client->devices()[0],
                                         expected_patterns.back()));
      auto* pjrt_arr =
          absl::down_cast<xla::ifrt::PjRtCompatibleArray*>(arr.get());
      std::shared_ptr<xla::PjRtBuffer> pjrt_buf = pjrt_arr->pjrt_buffers()[0];
      ASSERT_OK_AND_ASSIGN(size_t buf_size, pjrt_buf->GetOnDeviceSizeInBytes());
      ASSERT_OK_AND_ASSIGN(
          xla::PjRtRawBufferRef raw_buf,
          xla::PjRtRawBuffer::CreateRawAliasOfBuffer(pjrt_buf.get()));
      refs.push_back(
          {pjrt_buf->GetReadyFuture(), std::move(raw_buf), buf_size});
      keep_alive_arrays.push_back(std::move(arr));
    }

    PullTable table;
    absl::Notification replay_started;
    auto replay_conn =
        tsl::MakeRef<RecordingBufferConnectionState>(&replay_started);

    SocketTransferPullRequest planted;
    planted.set_uuid(kUuid);
    for (int pad = 0; pad < 256; ++pad) {
      planted.add_buffer_ids(9999);
    }
    planted.add_buffer_ids(0);
    planted.add_buffer_ids(0);
    planted.add_buffer_ids(1);
    const size_t planted_count = planted.buffer_ids_size();
    table.Handle(replay_conn, planted, /*base_req_id=*/0);

    auto worker_conn = tsl::MakeRef<RecordingBufferConnectionState>();
    std::unique_ptr<tsl::Thread> worker(
        tsl::Env::Default()->StartThread({}, "raw-pull-worker", [&]() {
          replay_started.WaitForNotification();
          for (int bid = 0; bid < kNumBuffers; ++bid) {
            SocketTransferPullRequest req;
            req.set_uuid(kUuid);
            req.add_buffer_ids(bid);
            table.Handle(worker_conn, req, /*base_req_id=*/1000 + bid);
          }
        }));

    auto entry =
        tsl::MakeRef<RawBufferEntry>(std::move(refs), copier, kXferSize);
    table.AwaitPull(kUuid, std::move(entry));
    worker.reset();

    replay_conn->WaitForEvents(planted_count);
    worker_conn->WaitForEvents(kNumBuffers);

    auto replay_buffers = replay_conn->completed_buffers();
    auto worker_buffers = worker_conn->completed_buffers();
    auto replay_errors = replay_conn->errors();
    auto worker_errors = worker_conn->errors();

    ASSERT_EQ(replay_buffers.size() + worker_buffers.size(),
              static_cast<size_t>(kNumBuffers));
    for (int b = 0; b < kNumBuffers; ++b) {
      size_t wins = 0;
      if (b < 2) {
        const size_t planted_req_id = (b == 0) ? 256u : 258u;
        auto it = replay_buffers.find(planted_req_id);
        if (it != replay_buffers.end()) {
          EXPECT_THAT(DecodeInt32Payload(it->second),
                      IsOkAndHolds(expected_patterns[b]));
          ++wins;
        } else {
          EXPECT_THAT(replay_errors[planted_req_id],
                      StatusIs(absl::StatusCode::kInternal,
                               HasSubstr("already been fetched")));
        }
      }
      auto worker_it = worker_buffers.find(1000 + b);
      if (worker_it != worker_buffers.end()) {
        EXPECT_THAT(DecodeInt32Payload(worker_it->second),
                    IsOkAndHolds(expected_patterns[b]));
        ++wins;
      } else {
        EXPECT_THAT(worker_errors[1000 + b],
                    StatusIs(absl::StatusCode::kInternal,
                             HasSubstr("already been fetched")));
      }
      EXPECT_EQ(wins, 1u);
    }
    auto replay_error_sizes = replay_conn->error_sizes();
    EXPECT_THAT(replay_errors[257],
                StatusIs(absl::StatusCode::kInternal,
                         HasSubstr("already been fetched")));
    EXPECT_EQ(replay_error_sizes[257], kElementsPerBuffer * sizeof(int32_t));
    EXPECT_EQ(replay_error_sizes[0], 0u);
  }
}

TEST(PjRtBufferEntryTest, InvalidAndDuplicateIdsDoNotCorruptCompletionCount) {
  ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  constexpr size_t kXferSize = 4096;
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<PremappedCopierState> copier,
                       MakeTestCopier(client.get(), kXferSize));

  std::vector<int32_t> pattern0 = tests::CreateTestPattern(10, 512);
  std::vector<int32_t> pattern1 = tests::CreateTestPattern(20, 512);
  ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef arr0,
                       tests::CopyTestPatternToDevice(
                           client.get(), client->devices()[0], pattern0));
  ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef arr1,
                       tests::CopyTestPatternToDevice(
                           client.get(), client->devices()[0], pattern1));
  std::shared_ptr<xla::PjRtBuffer> buf0 =
      absl::down_cast<xla::ifrt::PjRtCompatibleArray*>(arr0.get())
          ->pjrt_buffers()[0];
  std::shared_ptr<xla::PjRtBuffer> buf1 =
      absl::down_cast<xla::ifrt::PjRtCompatibleArray*>(arr1.get())
          ->pjrt_buffers()[0];
  ASSERT_OK_AND_ASSIGN(size_t size0, buf0->GetOnDeviceSizeInBytes());
  ASSERT_OK_AND_ASSIGN(size_t size1, buf1->GetOnDeviceSizeInBytes());

  // Case 1: Out of range and duplicate IDs equal in count to arrs_.size() must
  // not prematurely evict the entry before buffer 1 is pulled.
  {
    PullTable table;
    constexpr uint64_t kUuid = 101;
    std::vector<PjRtBufferEntry::BufferRef> refs = {{buf0, size0},
                                                    {buf1, size1}};
    table.AwaitPull(kUuid, tsl::MakeRef<PjRtBufferEntry>(std::move(refs),
                                                         copier, kXferSize));

    auto conn = tsl::MakeRef<RecordingBufferConnectionState>();
    SocketTransferPullRequest oob_req;
    oob_req.set_uuid(kUuid);
    oob_req.add_buffer_ids(50);
    oob_req.add_buffer_ids(51);
    table.Handle(conn, oob_req, /*base_req_id=*/0);
    conn->WaitForEvents(2);
    EXPECT_THAT(conn->errors()[0], StatusIs(absl::StatusCode::kInternal,
                                            HasSubstr("out of range")));
    EXPECT_THAT(conn->errors()[1], StatusIs(absl::StatusCode::kInternal,
                                            HasSubstr("out of range")));

    SocketTransferPullRequest dup_req;
    dup_req.set_uuid(kUuid);
    dup_req.add_buffer_ids(0);
    dup_req.add_buffer_ids(0);
    table.Handle(conn, dup_req, /*base_req_id=*/10);
    conn->WaitForEvents(4);
    EXPECT_THAT(DecodeInt32Payload(conn->completed_buffers()[10]),
                IsOkAndHolds(pattern0));
    EXPECT_THAT(conn->errors()[11],
                StatusIs(absl::StatusCode::kInternal,
                         HasSubstr("already been fetched")));

    SocketTransferPullRequest final_req;
    final_req.set_uuid(kUuid);
    final_req.add_buffer_ids(1);
    table.Handle(conn, final_req, /*base_req_id=*/20);
    conn->WaitForEvents(5);
    EXPECT_THAT(DecodeInt32Payload(conn->completed_buffers()[20]),
                IsOkAndHolds(pattern1));
  }

  // Case 2: Request with more buffer_ids than arrs_.size() (overshoot) must
  // still return true once all valid slots are consumed and evict the entry.
  {
    PullTable table;
    constexpr uint64_t kUuid = 102;
    std::vector<PjRtBufferEntry::BufferRef> refs = {{buf0, size0},
                                                    {buf1, size1}};
    table.AwaitPull(kUuid, tsl::MakeRef<PjRtBufferEntry>(std::move(refs),
                                                         copier, kXferSize));

    auto conn = tsl::MakeRef<RecordingBufferConnectionState>();
    SocketTransferPullRequest req;
    req.set_uuid(kUuid);
    req.add_buffer_ids(99);
    req.add_buffer_ids(0);
    req.add_buffer_ids(1);
    table.Handle(conn, req, /*base_req_id=*/0);
    conn->WaitForEvents(3);
    EXPECT_THAT(DecodeInt32Payload(conn->completed_buffers()[1]),
                IsOkAndHolds(pattern0));
    EXPECT_THAT(DecodeInt32Payload(conn->completed_buffers()[2]),
                IsOkAndHolds(pattern1));

    // Entry must now be evicted from PullTable.
    auto post_conn = tsl::MakeRef<RecordingBufferConnectionState>();
    SocketTransferPullRequest post_req;
    post_req.set_uuid(kUuid);
    post_req.add_buffer_ids(0);
    table.Handle(post_conn, post_req, /*base_req_id=*/30);
    table.Reset();
    post_conn->WaitForEvents(1);
    EXPECT_THAT(post_conn->errors()[30],
                StatusIs(absl::StatusCode::kInternal, HasSubstr("Reset")));
  }
}

TEST(RawBufferEntryTest, InvalidAndDuplicateIdsDoNotCorruptCompletionCount) {
  ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  constexpr size_t kXferSize = 4096;
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<PremappedCopierState> copier,
                       MakeTestCopier(client.get(), kXferSize));

  std::vector<int32_t> pattern0 = tests::CreateTestPattern(30, 512);
  std::vector<int32_t> pattern1 = tests::CreateTestPattern(40, 512);
  ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef arr0,
                       tests::CopyTestPatternToDevice(
                           client.get(), client->devices()[0], pattern0));
  ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef arr1,
                       tests::CopyTestPatternToDevice(
                           client.get(), client->devices()[0], pattern1));
  std::shared_ptr<xla::PjRtBuffer> buf0 =
      absl::down_cast<xla::ifrt::PjRtCompatibleArray*>(arr0.get())
          ->pjrt_buffers()[0];
  std::shared_ptr<xla::PjRtBuffer> buf1 =
      absl::down_cast<xla::ifrt::PjRtCompatibleArray*>(arr1.get())
          ->pjrt_buffers()[0];
  ASSERT_OK_AND_ASSIGN(size_t size0, buf0->GetOnDeviceSizeInBytes());
  ASSERT_OK_AND_ASSIGN(size_t size1, buf1->GetOnDeviceSizeInBytes());
  ASSERT_OK_AND_ASSIGN(xla::PjRtRawBufferRef raw0,
                       xla::PjRtRawBuffer::CreateRawAliasOfBuffer(buf0.get()));
  ASSERT_OK_AND_ASSIGN(xla::PjRtRawBufferRef raw1,
                       xla::PjRtRawBuffer::CreateRawAliasOfBuffer(buf1.get()));

  PullTable table;
  constexpr uint64_t kUuid = 201;
  std::vector<RawBufferEntry::BufferRef> refs = {
      {buf0->GetReadyFuture(), raw0, size0},
      {buf1->GetReadyFuture(), raw1, size1},
  };
  table.AwaitPull(
      kUuid, tsl::MakeRef<RawBufferEntry>(std::move(refs), copier, kXferSize));

  auto conn = tsl::MakeRef<RecordingBufferConnectionState>();
  SocketTransferPullRequest req1;
  req1.set_uuid(kUuid);
  req1.add_buffer_ids(77);
  req1.add_buffer_ids(0);
  req1.add_buffer_ids(0);
  table.Handle(conn, req1, /*base_req_id=*/0);
  conn->WaitForEvents(3);
  EXPECT_THAT(conn->errors()[0],
              StatusIs(absl::StatusCode::kInternal, HasSubstr("out of range")));
  EXPECT_THAT(DecodeInt32Payload(conn->completed_buffers()[1]),
              IsOkAndHolds(pattern0));
  EXPECT_THAT(conn->errors()[2], StatusIs(absl::StatusCode::kInternal,
                                          HasSubstr("already been fetched")));

  SocketTransferPullRequest req2;
  req2.set_uuid(kUuid);
  req2.add_buffer_ids(1);
  table.Handle(conn, req2, /*base_req_id=*/10);
  conn->WaitForEvents(4);
  EXPECT_THAT(DecodeInt32Payload(conn->completed_buffers()[10]),
              IsOkAndHolds(pattern1));

  auto post_conn = tsl::MakeRef<RecordingBufferConnectionState>();
  SocketTransferPullRequest post_req;
  post_req.set_uuid(kUuid);
  post_req.add_buffer_ids(1);
  table.Handle(post_conn, post_req, /*base_req_id=*/20);
  table.Reset();
  post_conn->WaitForEvents(1);
  EXPECT_THAT(post_conn->errors()[20],
              StatusIs(absl::StatusCode::kInternal, HasSubstr("Reset")));
}

TEST(PjRtBufferEntryTest, DoesNotHoldMutexAcrossOnReadyOrSendCallbacks) {
  ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  constexpr size_t kXferSize = 4096;
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<PremappedCopierState> copier,
                       MakeTestCopier(client.get(), kXferSize));

  std::vector<int32_t> pattern = tests::CreateTestPattern(5, 64);
  ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef arr,
                       tests::CopyTestPatternToDevice(
                           client.get(), client->devices()[0], pattern));
  std::shared_ptr<xla::PjRtBuffer> buf =
      absl::down_cast<xla::ifrt::PjRtCompatibleArray*>(arr.get())
          ->pjrt_buffers()[0];
  ASSERT_OK_AND_ASSIGN(size_t buf_size, buf->GetOnDeviceSizeInBytes());

  // Slot 0 has an immediately ready error future so OnReady runs synchronously
  // on the calling thread and calls SendError. If PjRtBufferEntry::mu_ were
  // held across OnReady or SendError, the concurrent pull of slot 1 would
  // deadlock.
  std::vector<PjRtBufferEntry::BufferRef> refs = {
      {nullptr, buf_size, tsl::Future<>(absl::InternalError("ready error"))},
      {buf, buf_size, buf->GetReadyFuture()},
  };
  auto entry =
      tsl::MakeRef<PjRtBufferEntry>(std::move(refs), copier, kXferSize);

  absl::Notification entered_callback;
  absl::Notification unblock_callback;
  auto blocking_conn = tsl::MakeRef<BlockingCallbackConnectionState>(
      &entered_callback, &unblock_callback);
  auto other_conn = tsl::MakeRef<RecordingBufferConnectionState>();

  std::unique_ptr<tsl::Thread> other_thread(
      tsl::Env::Default()->StartThread({}, "concurrent-pull", [&]() {
        entered_callback.WaitForNotification();
        SocketTransferPullRequest req1;
        req1.add_buffer_ids(1);
        EXPECT_TRUE(entry->Handle(other_conn, req1, /*base_req_id=*/1));
        other_conn->WaitForEvents(1);
        unblock_callback.Notify();
      }));

  SocketTransferPullRequest req0;
  req0.add_buffer_ids(0);
  EXPECT_TRUE(entry->Handle(blocking_conn, req0, /*base_req_id=*/0));
  other_thread.reset();
  EXPECT_THAT(DecodeInt32Payload(other_conn->completed_buffers()[1]),
              IsOkAndHolds(pattern));
}

TEST(RawBufferEntryTest, DoesNotHoldMutexAcrossOnReadyOrSendCallbacks) {
  ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  constexpr size_t kXferSize = 4096;
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<PremappedCopierState> copier,
                       MakeTestCopier(client.get(), kXferSize));

  std::vector<int32_t> pattern = tests::CreateTestPattern(7, 64);
  ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef arr0,
                       tests::CopyTestPatternToDevice(
                           client.get(), client->devices()[0], pattern));
  ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef arr1,
                       tests::CopyTestPatternToDevice(
                           client.get(), client->devices()[0], pattern));
  std::shared_ptr<xla::PjRtBuffer> buf0 =
      absl::down_cast<xla::ifrt::PjRtCompatibleArray*>(arr0.get())
          ->pjrt_buffers()[0];
  std::shared_ptr<xla::PjRtBuffer> buf1 =
      absl::down_cast<xla::ifrt::PjRtCompatibleArray*>(arr1.get())
          ->pjrt_buffers()[0];
  ASSERT_OK(buf0->GetReadyFuture().Await());
  ASSERT_OK(buf1->GetReadyFuture().Await());
  ASSERT_OK_AND_ASSIGN(size_t buf_size, buf0->GetOnDeviceSizeInBytes());
  ASSERT_OK_AND_ASSIGN(xla::PjRtRawBufferRef raw0,
                       xla::PjRtRawBuffer::CreateRawAliasOfBuffer(buf0.get()));
  ASSERT_OK_AND_ASSIGN(xla::PjRtRawBufferRef raw1,
                       xla::PjRtRawBuffer::CreateRawAliasOfBuffer(buf1.get()));

  std::vector<RawBufferEntry::BufferRef> refs = {
      {tsl::Future<>(absl::OkStatus()), raw0, buf_size},
      {tsl::Future<>(absl::OkStatus()), raw1, buf_size},
  };
  auto entry = tsl::MakeRef<RawBufferEntry>(std::move(refs), copier, kXferSize);

  absl::Notification entered_callback;
  absl::Notification unblock_callback;
  auto blocking_conn = tsl::MakeRef<BlockingCallbackConnectionState>(
      &entered_callback, &unblock_callback);
  auto other_conn = tsl::MakeRef<RecordingBufferConnectionState>();

  std::unique_ptr<tsl::Thread> other_thread(
      tsl::Env::Default()->StartThread({}, "concurrent-raw-pull", [&]() {
        entered_callback.WaitForNotification();
        SocketTransferPullRequest req1;
        req1.add_buffer_ids(1);
        EXPECT_TRUE(entry->Handle(other_conn, req1, /*base_req_id=*/1));
        other_conn->WaitForEvents(1);
        unblock_callback.Notify();
      }));

  SocketTransferPullRequest req0;
  req0.add_buffer_ids(0);
  EXPECT_TRUE(entry->Handle(blocking_conn, req0, /*base_req_id=*/0));
  other_thread.reset();
  EXPECT_THAT(DecodeInt32Payload(other_conn->completed_buffers()[1]),
              IsOkAndHolds(pattern));
}

TEST(BufferEntriesEdgeCaseTest, EmptyAndZeroByteBuffers) {
  ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  constexpr size_t kXferSize = 4096;
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<PremappedCopierState> copier,
                       MakeTestCopier(client.get(), kXferSize));

  // Empty arrs_ entries complete immediately and are not retained by AwaitPull.
  {
    auto empty_pjrt = tsl::MakeRef<PjRtBufferEntry>(
        std::vector<PjRtBufferEntry::BufferRef>{}, copier, kXferSize);
    auto empty_raw = tsl::MakeRef<RawBufferEntry>(
        std::vector<RawBufferEntry::BufferRef>{}, copier, kXferSize);
    EXPECT_TRUE(empty_pjrt->IsComplete());
    EXPECT_TRUE(empty_raw->IsComplete());
    auto conn = tsl::MakeRef<RecordingBufferConnectionState>();
    SocketTransferPullRequest empty_req;
    EXPECT_TRUE(empty_pjrt->Handle(conn, empty_req, /*base_req_id=*/0));
    EXPECT_TRUE(empty_raw->Handle(conn, empty_req, /*base_req_id=*/0));

    SocketTransferPullRequest oob_req;
    oob_req.add_buffer_ids(0);
    EXPECT_TRUE(empty_pjrt->Handle(conn, oob_req, /*base_req_id=*/10));
    EXPECT_TRUE(empty_raw->Handle(conn, oob_req, /*base_req_id=*/20));
    conn->WaitForEvents(2);
    EXPECT_THAT(conn->errors()[10], StatusIs(absl::StatusCode::kInternal,
                                             HasSubstr("out of range")));
    EXPECT_THAT(conn->errors()[20], StatusIs(absl::StatusCode::kInternal,
                                             HasSubstr("out of range")));

    PullTable table;
    const int64_t initial_use_count = copier.use_count();
    table.AwaitPull(
        /*uuid=*/501,
        tsl::MakeRef<PjRtBufferEntry>(std::vector<PjRtBufferEntry::BufferRef>{},
                                      copier, kXferSize));
    table.AwaitPull(
        /*uuid=*/502,
        tsl::MakeRef<RawBufferEntry>(std::vector<RawBufferEntry::BufferRef>{},
                                     copier, kXferSize));
    EXPECT_EQ(copier.use_count(), initial_use_count);
  }

  // Device buffers of size zero must send a single empty terminal frame so the
  // receiver destination completes instead of hanging.
  std::vector<int32_t> empty_pattern;
  ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef zero_arr,
                       tests::CopyTestPatternToDevice(
                           client.get(), client->devices()[0], empty_pattern));
  std::shared_ptr<xla::PjRtBuffer> zero_buf =
      absl::down_cast<xla::ifrt::PjRtCompatibleArray*>(zero_arr.get())
          ->pjrt_buffers()[0];
  ASSERT_OK_AND_ASSIGN(size_t zero_size, zero_buf->GetOnDeviceSizeInBytes());
  ASSERT_EQ(zero_size, 0u);
  ASSERT_OK_AND_ASSIGN(
      xla::PjRtRawBufferRef zero_raw,
      xla::PjRtRawBuffer::CreateRawAliasOfBuffer(zero_buf.get()));

  {
    auto pjrt_entry = tsl::MakeRef<PjRtBufferEntry>(
        std::vector<PjRtBufferEntry::BufferRef>{{zero_buf, zero_size}}, copier,
        kXferSize);
    auto conn = tsl::MakeRef<RecordingBufferConnectionState>();
    SocketTransferPullRequest req;
    req.add_buffer_ids(0);
    EXPECT_TRUE(pjrt_entry->Handle(conn, req, /*base_req_id=*/0));
    conn->WaitForEvents(1);
    ASSERT_TRUE(conn->completed_buffers().contains(0));
    EXPECT_EQ(conn->completed_buffers()[0], "");
  }

  {
    auto raw_entry = tsl::MakeRef<RawBufferEntry>(
        std::vector<RawBufferEntry::BufferRef>{
            {zero_buf->GetReadyFuture(), zero_raw, zero_size}},
        copier, kXferSize);
    auto conn = tsl::MakeRef<RecordingBufferConnectionState>();
    SocketTransferPullRequest req;
    req.add_buffer_ids(0);
    EXPECT_TRUE(raw_entry->Handle(conn, req, /*base_req_id=*/0));
    conn->WaitForEvents(1);
    ASSERT_TRUE(conn->completed_buffers().contains(0));
    EXPECT_EQ(conn->completed_buffers()[0], "");
  }
}

TEST(SlicedRawBufferDestTest, RepeatedPoisonAndOverflowBoundsCheck) {
  ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef dest_arr,
      tests::CopyTestPatternToDevice(client.get(), client->devices()[0],
                                     std::vector<int32_t>(64, 0)));
  ASSERT_OK(dest_arr->GetReadyFuture().Await());
  ASSERT_OK_AND_ASSIGN(
      xla::PjRtRawBufferRef dest_raw_buffer,
      xla::PjRtRawBuffer::CreateRawAliasOfBuffer(
          absl::down_cast<xla::ifrt::PjRtCompatibleArray*>(dest_arr.get())
              ->pjrt_buffers()[0]
              .get()));
  ASSERT_OK_AND_ASSIGN(auto fut_and_dest,
                       CreateSlicedRawBufferDest(dest_raw_buffer, 0, 256));

  const char dummy[256] = {};
  // Exercise offset <= slice_size_ with offset + size > slice_size_ (third
  // bounds check clause).
  ASSERT_OK_AND_ASSIGN(auto in_range_offset_oob,
                       CreateSlicedRawBufferDest(dest_raw_buffer, 0, 256));
  bool third_clause_done = false;
  EXPECT_THAT(
      in_range_offset_oob.first->Put(
          dummy, /*offset=*/128, /*size=*/200,
          [&third_clause_done]() { third_clause_done = true; }),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Out of bounds")));
  EXPECT_TRUE(third_clause_done);
  EXPECT_THAT(
      in_range_offset_oob.second.Await(),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Out of bounds")));

  bool bounds_on_done_called = false;
  EXPECT_THAT(
      fut_and_dest.first->Put(
          dummy, /*offset=*/0x7fffffffffffffffLL,
          /*size=*/0x8000000000000010ULL,
          [&bounds_on_done_called]() { bounds_on_done_called = true; }),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Out of bounds")));
  EXPECT_TRUE(bounds_on_done_called);

  // Subsequent Put after Poison must still invoke on_done and return OkStatus.
  bool poisoned_put_on_done = false;
  EXPECT_THAT(fut_and_dest.first->Put(
                  dummy, /*offset=*/0, /*size=*/0,
                  [&poisoned_put_on_done]() { poisoned_put_on_done = true; }),
              ::absl_testing::IsOk());
  EXPECT_TRUE(poisoned_put_on_done);

  fut_and_dest.first->Poison(absl::InternalError("second poison"));
  EXPECT_THAT(
      fut_and_dest.second.Await(),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Out of bounds")));

  // Zero byte slices must report Poison errors rather than hanging.
  ASSERT_OK_AND_ASSIGN(auto zero_slice_poisoned,
                       CreateSlicedRawBufferDest(dest_raw_buffer, 0, 0));
  zero_slice_poisoned.first->Poison(absl::InternalError("zero slice poison"));
  EXPECT_THAT(
      zero_slice_poisoned.second.Await(),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("zero slice poison")));

  // Zero byte slices must also complete cleanly on a 0 byte Put.
  ASSERT_OK_AND_ASSIGN(auto zero_slice_ok,
                       CreateSlicedRawBufferDest(dest_raw_buffer, 0, 0));
  absl::Notification zero_put_done;
  ASSERT_OK(
      zero_slice_ok.first->Put(dummy, /*offset=*/0, /*size=*/0,
                               [&zero_put_done]() { zero_put_done.Notify(); }));
  zero_put_done.WaitForNotification();
  EXPECT_THAT(zero_slice_ok.second.Await(), ::absl_testing::IsOk());

  // DmaDestination::Put must also invoke on_done on out of bounds and poisoned
  // paths so SlabAllocator slots are never leaked.
  auto* ifrt_client = GetIfrtClient(dest_arr);
  ASSERT_OK_AND_ASSIGN(
      SingleBufferCopyPlan dma_plan,
      SetupTransferDestList(
          ShapeFromIfrt(dest_arr),
          absl::down_cast<xla::ifrt::PjRtDevice*>(ifrt_client->devices()[0]),
          ifrt_client, /*xfer_size=*/256));
  bool dma_oob_done = false;
  EXPECT_THAT(
      dma_plan.dests[0]->Put(dummy, /*offset=*/-1, /*size=*/4,
                             [&dma_oob_done]() { dma_oob_done = true; }),
      ::absl_testing::IsOk());
  EXPECT_TRUE(dma_oob_done);

  bool dma_poisoned_done = false;
  EXPECT_THAT(dma_plan.dests[0]->Put(
                  dummy, /*offset=*/0, /*size=*/4,
                  [&dma_poisoned_done]() { dma_poisoned_done = true; }),
              ::absl_testing::IsOk());
  EXPECT_TRUE(dma_poisoned_done);
}

class FakeAsyncHostToDeviceTransferManager
    : public xla::PjRtClient::AsyncHostToDeviceTransferManager {
 public:
  FakeAsyncHostToDeviceTransferManager(
      size_t buffer_size, absl::Status sub_buffer_status,
      absl::Notification* non_last_entered = nullptr,
      absl::Notification* release_non_last = nullptr,
      std::vector<absl::AnyInvocable<void() &&>>* retained_callbacks = nullptr)
      : buffer_size_(buffer_size),
        sub_buffer_status_(std::move(sub_buffer_status)),
        non_last_entered_(non_last_entered),
        release_non_last_(release_non_last),
        retained_callbacks_(retained_callbacks) {}

  size_t buffer_count() const override { return 1; }
  xla::PjRtDevice* device() const override { return nullptr; }
  std::unique_ptr<xla::PjRtBuffer> RetrieveBuffer(int buffer_index) override {
    return nullptr;
  }
  absl::Status TransferLiteralToBuffer(
      int buffer_index, const xla::LiteralSlice& literal,
      absl::AnyInvocable<void() &&> on_done) override {
    std::move(on_done)();
    return absl::OkStatus();
  }
  size_t buffer_size(int buffer_index) const override { return buffer_size_; }
  absl::Status TransferRawDataToBuffer(
      int buffer_index, absl::string_view data,
      absl::AnyInvocable<void() &&> on_done) override {
    std::move(on_done)();
    return absl::OkStatus();
  }
  absl::Status TransferRawDataToSubBuffer(
      int buffer_index, const void* data, int64_t offset, int64_t transfer_size,
      bool is_last_transfer, absl::AnyInvocable<void() &&> on_done) override {
    if (!is_last_transfer && non_last_entered_ != nullptr) {
      non_last_entered_->Notify();
      release_non_last_->WaitForNotification();
    }
    absl::MutexLock l(mu_);
    ++transfer_calls_;
    if (is_last_transfer) {
      ++last_transfer_calls_;
    }
    if (!sub_buffer_status_.ok()) {
      // Simulate synchronous error where on_done is destroyed without running.
      return sub_buffer_status_;
    }
    if (retained_callbacks_ != nullptr) {
      retained_callbacks_->push_back(std::move(on_done));
      std::move(retained_callbacks_->back())();
    } else {
      std::move(on_done)();
    }
    return absl::OkStatus();
  }
  void SetBufferError(int buffer_index, absl::Status error) override {
    absl::MutexLock l(mu_);
    ++error_count_;
    last_error_ = std::move(error);
  }
  void AddTransferMetadata(const TransferMetadata& metadata) override {}

  size_t transfer_calls() const {
    absl::MutexLock l(mu_);
    return transfer_calls_;
  }
  size_t last_transfer_calls() const {
    absl::MutexLock l(mu_);
    return last_transfer_calls_;
  }
  size_t error_count() const {
    absl::MutexLock l(mu_);
    return error_count_;
  }
  absl::Status last_error() const {
    absl::MutexLock l(mu_);
    return last_error_;
  }

 private:
  const size_t buffer_size_;
  const absl::Status sub_buffer_status_;
  absl::Notification* non_last_entered_;
  absl::Notification* release_non_last_;
  std::vector<absl::AnyInvocable<void() &&>>* retained_callbacks_;
  mutable absl::Mutex mu_;
  size_t transfer_calls_ ABSL_GUARDED_BY(mu_) = 0;
  size_t last_transfer_calls_ ABSL_GUARDED_BY(mu_) = 0;
  size_t error_count_ ABSL_GUARDED_BY(mu_) = 0;
  absl::Status last_error_ ABSL_GUARDED_BY(mu_);
};

TEST(DmaDestinationTest, SyncTransferFailureAndCumulativeOverflowInvokeOnDone) {
  const char dummy[256] = {};

  // Case 1a: Non final chunk synchronous failure in TransferRawDataToSubBuffer
  // must invoke on_done and record the buffer error once.
  {
    auto atm = std::make_shared<FakeAsyncHostToDeviceTransferManager>(
        /*buffer_size=*/256, absl::InternalError("sync transfer failure"));
    tsl::RCReference<ChunkDestination> dest = MakeDmaDestination(atm, 0, 256);
    bool on_done_called = false;
    EXPECT_THAT(dest->Put(dummy, /*offset=*/0, /*size=*/128,
                          [&on_done_called]() { on_done_called = true; }),
                StatusIs(absl::StatusCode::kInternal,
                         HasSubstr("sync transfer failure")));
    EXPECT_TRUE(on_done_called);
    EXPECT_EQ(atm->error_count(), 1u);
    EXPECT_THAT(atm->last_error(),
                StatusIs(absl::StatusCode::kInternal,
                         HasSubstr("sync transfer failure")));
  }

  // Case 1b: Single chunk (is_last == true) synchronous failure in
  // TransferRawDataToSubBuffer must also invoke on_done and record the buffer
  // error once.
  {
    auto atm = std::make_shared<FakeAsyncHostToDeviceTransferManager>(
        /*buffer_size=*/256, absl::InternalError("last chunk sync failure"));
    tsl::RCReference<ChunkDestination> dest = MakeDmaDestination(atm, 0, 256);
    bool on_done_called = false;
    EXPECT_THAT(dest->Put(dummy, /*offset=*/0, /*size=*/256,
                          [&on_done_called]() { on_done_called = true; }),
                StatusIs(absl::StatusCode::kInternal,
                         HasSubstr("last chunk sync failure")));
    EXPECT_TRUE(on_done_called);
    dest->Poison(absl::InternalError("duplicate poison"));
    EXPECT_EQ(atm->error_count(), 1u);
    EXPECT_THAT(atm->last_error(),
                StatusIs(absl::StatusCode::kInternal,
                         HasSubstr("last chunk sync failure")));
  }

  // Case 1c: Concurrent failure of an earlier chunk while the final chunk is
  // waiting in IsLastSemaphore must abort the final chunk before calling
  // TransferRawDataToSubBuffer(..., is_last_transfer=true) and record the
  // buffer error once.
  {
    absl::Notification non_last_entered;
    absl::Notification release_non_last;
    auto atm = std::make_shared<FakeAsyncHostToDeviceTransferManager>(
        /*buffer_size=*/256, absl::InternalError("earlier chunk failed"),
        &non_last_entered, &release_non_last);
    tsl::RCReference<ChunkDestination> dest = MakeDmaDestination(atm, 0, 256);

    bool first_done = false;
    bool last_done = false;
    absl::Notification last_put_started;
    std::unique_ptr<tsl::Thread> last_thread(
        tsl::Env::Default()->StartThread({}, "last-chunk-put", [&]() {
          non_last_entered.WaitForNotification();
          last_put_started.Notify();
          EXPECT_OK(dest->Put(dummy, /*offset=*/128, /*size=*/128,
                              [&last_done]() { last_done = true; }));
        }));

    std::unique_ptr<tsl::Thread> first_thread(
        tsl::Env::Default()->StartThread({}, "first-chunk-put", [&]() {
          EXPECT_THAT(dest->Put(dummy, /*offset=*/0, /*size=*/128,
                                [&first_done]() { first_done = true; }),
                      StatusIs(absl::StatusCode::kInternal,
                               HasSubstr("earlier chunk failed")));
        }));

    last_put_started.WaitForNotification();
    release_non_last.Notify();
    first_thread.reset();
    last_thread.reset();

    EXPECT_TRUE(first_done);
    EXPECT_TRUE(last_done);
    EXPECT_EQ(atm->last_transfer_calls(), 0u);
    EXPECT_EQ(atm->error_count(), 1u);
    EXPECT_THAT(atm->last_error(), StatusIs(absl::StatusCode::kInternal,
                                            HasSubstr("earlier chunk failed")));
  }

  // Case 2: Cumulative chunk bytes exceeding transfer_size must not underflow
  // IsLastSemaphore, must invoke on_done immediately upon callback execution
  // even when the AnyInvocable wrapper stays alive, and must poison the buffer
  // once.
  {
    std::vector<absl::AnyInvocable<void() &&>> retained_callbacks;
    auto atm = std::make_shared<FakeAsyncHostToDeviceTransferManager>(
        /*buffer_size=*/256, absl::OkStatus(), /*non_last_entered=*/nullptr,
        /*release_non_last=*/nullptr, &retained_callbacks);
    tsl::RCReference<ChunkDestination> dest = MakeDmaDestination(atm, 0, 256);

    int first_done_count = 0;
    ASSERT_OK(dest->Put(dummy, /*offset=*/0, /*size=*/200,
                        [&first_done_count]() { ++first_done_count; }));
    EXPECT_EQ(first_done_count, 1);
    retained_callbacks.clear();
    EXPECT_EQ(first_done_count, 1);
    EXPECT_EQ(atm->transfer_calls(), 1u);

    bool second_done = false;
    EXPECT_THAT(
        dest->Put(dummy, /*offset=*/0, /*size=*/200,
                  [&second_done]() { second_done = true; }),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("extra bytes")));
    EXPECT_TRUE(second_done);
    EXPECT_EQ(atm->transfer_calls(), 1u);
    EXPECT_EQ(atm->error_count(), 1u);
  }

  // Case 3: Poison with OkStatus is a no op, while out of bounds offset > size
  // and offset + size > buffer_size_ each poison the buffer independently.
  {
    auto atm = std::make_shared<FakeAsyncHostToDeviceTransferManager>(
        /*buffer_size=*/256, absl::OkStatus());
    tsl::RCReference<ChunkDestination> dest = MakeDmaDestination(atm, 0, 256);
    dest->Poison(absl::OkStatus());
    EXPECT_EQ(atm->error_count(), 0u);

    bool oob_offset_done = false;
    ASSERT_OK(dest->Put(dummy, /*offset=*/300, /*size=*/0,
                        [&oob_offset_done]() { oob_offset_done = true; }));
    EXPECT_TRUE(oob_offset_done);
    EXPECT_EQ(atm->error_count(), 1u);
  }

  {
    auto atm = std::make_shared<FakeAsyncHostToDeviceTransferManager>(
        /*buffer_size=*/256, absl::OkStatus());
    tsl::RCReference<ChunkDestination> dest = MakeDmaDestination(atm, 0, 256);
    bool oob_end_done = false;
    ASSERT_OK(dest->Put(dummy, /*offset=*/128, /*size=*/200,
                        [&oob_end_done]() { oob_end_done = true; }));
    EXPECT_TRUE(oob_end_done);
    EXPECT_EQ(atm->error_count(), 1u);
  }
}

class FakeNonHostRawBuffer : public xla::PjRtRawBuffer {
 public:
  FakeNonHostRawBuffer(size_t size, tsl::AsyncValueRef<tsl::Chain> h2d_event,
                       absl::Status d2h_status, uint8_t fill_byte = 0x5a)
      : size_(size),
        h2d_event_(std::move(h2d_event)),
        d2h_status_(std::move(d2h_status)),
        fill_byte_(fill_byte) {}

  PJRT_Memory* c_memory_space() const override { return nullptr; }
  size_t GetOnDeviceSizeInBytes() const override { return size_; }
  xla::Future<> CopyRawHostToDevice(const void* src, int64_t offset,
                                    int64_t transfer_size) override {
    return xla::Future<>(absl::UnimplementedError("unused"));
  }
  xla::Future<> CopyRawDeviceToHost(void* dst, int64_t offset,
                                    int64_t transfer_size) override {
    return xla::Future<>(absl::UnimplementedError("unused"));
  }
  void* OpaqueDeviceMemoryDataPointer() const override { return nullptr; }
  absl::StatusOr<xla::PjRtDeviceEventRef> CopyRawHostToDeviceAndReturnEvent(
      const void* src, int64_t offset, int64_t transfer_size,
      xla::PjRtDeviceEventRefVector dependencies) override {
    return xla::PjRtDeviceEventRef(h2d_event_.CopyRef());
  }
  absl::StatusOr<xla::PjRtDeviceEventRef> CopyRawDeviceToHostAndReturnEvent(
      void* dst, int64_t offset, int64_t transfer_size,
      xla::PjRtDeviceEventRefVector dependencies) override {
    if (!d2h_status_.ok()) {
      return d2h_status_;
    }
    char* out = reinterpret_cast<char*>(dst);
    for (int64_t i = 0; i < transfer_size; ++i) {
      out[i] = static_cast<char>(fill_byte_ + (((offset + i) >> 6) & 0xf));
    }
    return xla::PjRtDeviceEventRef(
        tsl::MakeAvailableAsyncValueRef<tsl::Chain>());
  }
  absl::StatusOr<xla::PjRtRawBufferRef> Slice(int64_t offset,
                                              int64_t size) override {
    return absl::UnimplementedError("not needed");
  }
  absl::StatusOr<xla::PjRtDeviceEventRef> MakeAllocationReadyEvent() override {
    return absl::UnimplementedError("not needed");
  }
  void CopyTo(
      xla::PjRtRawBufferRef dst_raw_buffer,
      xla::PjRtDeviceEventPromiseRef definition_event_promise,
      xla::PjRtDeviceEventPromiseRef src_usage_event_promise,
      absl::AnyInvocable<void(absl::Status) &&> allocation_event) override {}
  void ScheduleCopyTo(
      xla::PjRtDeviceEventRefVector transfer_dependency_events,
      xla::PjRtRawBufferRef dst_raw_buffer,
      xla::PjRtDeviceEventPromiseRef definition_event_promise,
      xla::PjRtDeviceEventPromiseRef src_usage_event_promise,
      absl::AnyInvocable<void(absl::Status) &&> allocation_event) override {}
  xla::PjRtDeviceEventPtr GetRawBufferAsyncValue() override { return {}; }

 private:
  const size_t size_;
  tsl::AsyncValueRef<tsl::Chain> h2d_event_;
  const absl::Status d2h_status_;
  const uint8_t fill_byte_;
};

TEST(SlicedRawBufferDestTest, PoisonAfterAllBytesSentWhileCopyInFlight) {
  tsl::AsyncValueRef<tsl::Chain> h2d_event =
      tsl::MakeConstructedAsyncValueRef<tsl::Chain>();
  auto raw_buffer = tsl::MakeRef<FakeNonHostRawBuffer>(
      /*size=*/256, h2d_event.CopyRef(), absl::OkStatus());
  ASSERT_OK_AND_ASSIGN(auto fut_and_dest,
                       CreateSlicedRawBufferDest(raw_buffer, 0, 256));

  const char dummy[256] = {};
  bool first_done = false;
  ASSERT_OK(fut_and_dest.first->Put(dummy, /*offset=*/0, /*size=*/256,
                                    [&first_done]() { first_done = true; }));

  // While the 256 byte copy is still in flight (all_bytes_sent_ == true and
  // pending_copies_ == 1), even a zero byte extra chunk triggers over_sent via
  // all_bytes_sent_.
  bool extra_done = false;
  EXPECT_THAT(
      fut_and_dest.first->Put(dummy, /*offset=*/0, /*size=*/0,
                              [&extra_done]() { extra_done = true; }),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("extra bytes")));
  EXPECT_TRUE(extra_done);

  // Subsequent Put after Poison must take the already_failed path and return
  // OkStatus, and completing the in flight copy must fail the future.
  bool post_poison_done = false;
  EXPECT_OK(fut_and_dest.first->Put(
      dummy, /*offset=*/0, /*size=*/0,
      [&post_poison_done]() { post_poison_done = true; }));
  EXPECT_TRUE(post_poison_done);

  EXPECT_FALSE(fut_and_dest.second.IsReady());
  h2d_event.SetStateConcrete();
  EXPECT_TRUE(first_done);
  EXPECT_THAT(
      fut_and_dest.second.Await(),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("extra bytes")));

  // Completing a partial chunk must not resolve the destination future while
  // bytes remain unsent, and a subsequent cumulative over_sent (128 + 200 >
  // 256) must fail the still pending future.
  tsl::AsyncValueRef<tsl::Chain> partial_event =
      tsl::MakeConstructedAsyncValueRef<tsl::Chain>();
  auto partial_raw = tsl::MakeRef<FakeNonHostRawBuffer>(
      /*size=*/256, partial_event.CopyRef(), absl::OkStatus());
  ASSERT_OK_AND_ASSIGN(auto partial_fut_and_dest,
                       CreateSlicedRawBufferDest(partial_raw, 0, 256));
  bool partial_first_done = false;
  ASSERT_OK(partial_fut_and_dest.first->Put(
      dummy, /*offset=*/0, /*size=*/128,
      [&partial_first_done]() { partial_first_done = true; }));
  partial_event.SetStateConcrete();
  EXPECT_TRUE(partial_first_done);
  EXPECT_FALSE(partial_fut_and_dest.second.IsReady());

  bool partial_over_done = false;
  EXPECT_THAT(
      partial_fut_and_dest.first->Put(
          dummy, /*offset=*/0, /*size=*/200,
          [&partial_over_done]() { partial_over_done = true; }),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("extra bytes")));
  EXPECT_TRUE(partial_over_done);
  EXPECT_THAT(
      partial_fut_and_dest.second.Await(),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("extra bytes")));
}

TEST(RawAndPjRtEntryDefensivePathsTest, NonHostPointerDmaAndInvalidConfigs) {
  ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  constexpr size_t kXferSize = 128;
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<PremappedCopierState> copier,
                       MakeTestCopier(client.get(), kXferSize));

  // Exercise RawBufferEntry non host pointer schedule copy success and error.
  {
    auto ok_raw = tsl::MakeRef<FakeNonHostRawBuffer>(
        /*size=*/256, tsl::MakeAvailableAsyncValueRef<tsl::Chain>(),
        absl::OkStatus(), 0x3c);
    auto fail_raw = tsl::MakeRef<FakeNonHostRawBuffer>(
        /*size=*/128, tsl::MakeAvailableAsyncValueRef<tsl::Chain>(),
        absl::InternalError("d2h failed"));
    std::vector<RawBufferEntry::BufferRef> refs = {
        {tsl::Future<>(), ok_raw, 256},
        {tsl::Future<>(), fail_raw, 128},
    };
    auto entry =
        tsl::MakeRef<RawBufferEntry>(std::move(refs), copier, kXferSize);
    auto conn = tsl::MakeRef<RecordingBufferConnectionState>();
    SocketTransferPullRequest req;
    req.add_buffer_ids(0);
    req.add_buffer_ids(1);
    EXPECT_TRUE(entry->Handle(conn, req, /*base_req_id=*/0));
    conn->WaitForEvents(2);
    std::string expected_d2h(256, '\0');
    for (int64_t i = 0; i < 256; ++i) {
      expected_d2h[i] = static_cast<char>(0x3c + ((i >> 6) & 0xf));
    }
    EXPECT_EQ(conn->completed_buffers()[0], expected_d2h);
    EXPECT_THAT(conn->errors()[1],
                StatusIs(absl::StatusCode::kInternal, HasSubstr("d2h failed")));
    EXPECT_EQ(conn->error_sizes()[1], kXferSize);
  }

  // Exercise null buffer, xfer_size == 0, and null PremappedCopierState guards.
  {
    auto ok_raw = tsl::MakeRef<FakeNonHostRawBuffer>(
        /*size=*/64, tsl::MakeAvailableAsyncValueRef<tsl::Chain>(),
        absl::OkStatus());
    auto null_buf_entry = tsl::MakeRef<RawBufferEntry>(
        std::vector<RawBufferEntry::BufferRef>{
            {tsl::Future<>(), xla::PjRtRawBufferRef(), 64}},
        copier, kXferSize);
    auto zero_xfer_entry = tsl::MakeRef<RawBufferEntry>(
        std::vector<RawBufferEntry::BufferRef>{{tsl::Future<>(), ok_raw, 64}},
        copier, /*xfer_size=*/0);
    auto null_copier_entry = tsl::MakeRef<RawBufferEntry>(
        std::vector<RawBufferEntry::BufferRef>{{tsl::Future<>(), ok_raw, 64}},
        /*state=*/nullptr, kXferSize);

    auto conn = tsl::MakeRef<RecordingBufferConnectionState>();
    SocketTransferPullRequest req;
    req.add_buffer_ids(0);
    EXPECT_TRUE(null_buf_entry->Handle(conn, req, /*base_req_id=*/0));
    EXPECT_TRUE(zero_xfer_entry->Handle(conn, req, /*base_req_id=*/1));
    EXPECT_TRUE(null_copier_entry->Handle(conn, req, /*base_req_id=*/2));
    conn->WaitForEvents(3);
    EXPECT_THAT(conn->errors()[0],
                StatusIs(absl::StatusCode::kInternal, HasSubstr("is null")));
    EXPECT_EQ(conn->error_sizes()[0], 64u);
    EXPECT_THAT(conn->errors()[1], StatusIs(absl::StatusCode::kInvalidArgument,
                                            HasSubstr("greater than zero")));
    EXPECT_EQ(conn->error_sizes()[1], 64u);
    EXPECT_THAT(conn->errors()[2], StatusIs(absl::StatusCode::kInternal,
                                            HasSubstr("PremappedCopierState")));
    EXPECT_EQ(conn->error_sizes()[2], 64u);
  }

  // Exercise matching guards on PjRtBufferEntry, host pointer RawBufferEntry
  // with null PremappedCopierState, and unready PjRtBuffer GetReadyFuture().
  {
    std::vector<int32_t> pattern = tests::CreateTestPattern(9, 16);
    ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef arr,
                         tests::CopyTestPatternToDevice(
                             client.get(), client->devices()[0], pattern));
    std::shared_ptr<xla::PjRtBuffer> buf =
        absl::down_cast<xla::ifrt::PjRtCompatibleArray*>(arr.get())
            ->pjrt_buffers()[0];
    ASSERT_OK_AND_ASSIGN(size_t buf_size, buf->GetOnDeviceSizeInBytes());
    ASSERT_OK_AND_ASSIGN(xla::PjRtRawBufferRef host_raw_buf,
                         xla::PjRtRawBuffer::CreateRawAliasOfBuffer(buf.get()));
    ASSERT_NE(host_raw_buf->GetHostPointer(), nullptr);

    auto host_null_copier_entry = tsl::MakeRef<RawBufferEntry>(
        std::vector<RawBufferEntry::BufferRef>{
            {tsl::Future<>(), std::move(host_raw_buf), buf_size}},
        /*state=*/nullptr, kXferSize);

    auto* ifrt_pjrt_client =
        absl::down_cast<xla::ifrt::PjRtClient*>(client.get());
    auto* pjrt_device =
        absl::down_cast<xla::ifrt::PjRtDevice*>(client->devices()[0])
            ->pjrt_device();
    ASSERT_OK_AND_ASSIGN(xla::PjRtMemorySpace * mem_space,
                         pjrt_device->default_memory_space());
    ASSERT_OK_AND_ASSIGN(
        auto atm,
        ifrt_pjrt_client->pjrt_client()->CreateBuffersForAsyncHostToDevice(
            {buf->on_device_shape()}, mem_space));
    std::shared_ptr<xla::PjRtBuffer> unready_buf = atm->RetrieveBuffer(0);
    auto unready_pjrt_entry = tsl::MakeRef<PjRtBufferEntry>(
        std::vector<PjRtBufferEntry::BufferRef>{{unready_buf, buf_size}},
        copier, kXferSize);

    auto null_buf_entry = tsl::MakeRef<PjRtBufferEntry>(
        std::vector<PjRtBufferEntry::BufferRef>{{nullptr, 64, tsl::Future<>()}},
        copier, kXferSize);
    auto zero_xfer_entry = tsl::MakeRef<PjRtBufferEntry>(
        std::vector<PjRtBufferEntry::BufferRef>{{buf, buf_size}}, copier,
        /*xfer_size=*/0);
    auto null_copier_entry = tsl::MakeRef<PjRtBufferEntry>(
        std::vector<PjRtBufferEntry::BufferRef>{{buf, buf_size}},
        /*state=*/nullptr, kXferSize);

    auto conn = tsl::MakeRef<RecordingBufferConnectionState>();
    SocketTransferPullRequest req;
    req.add_buffer_ids(0);
    EXPECT_TRUE(null_buf_entry->Handle(conn, req, /*base_req_id=*/0));
    EXPECT_TRUE(zero_xfer_entry->Handle(conn, req, /*base_req_id=*/1));
    EXPECT_TRUE(null_copier_entry->Handle(conn, req, /*base_req_id=*/2));
    EXPECT_TRUE(host_null_copier_entry->Handle(conn, req, /*base_req_id=*/3));
    EXPECT_TRUE(unready_pjrt_entry->Handle(conn, req, /*base_req_id=*/4));
    conn->WaitForEvents(4);
    EXPECT_EQ(conn->errors().count(4), 0u);
    atm->SetBufferError(0, absl::InternalError("unready buffer failed"));
    conn->WaitForEvents(5);

    EXPECT_THAT(conn->errors()[0],
                StatusIs(absl::StatusCode::kInternal, HasSubstr("is null")));
    EXPECT_EQ(conn->error_sizes()[0], 64u);
    EXPECT_THAT(conn->errors()[1], StatusIs(absl::StatusCode::kInvalidArgument,
                                            HasSubstr("greater than zero")));
    EXPECT_EQ(conn->error_sizes()[1], buf_size);
    EXPECT_THAT(conn->errors()[2], StatusIs(absl::StatusCode::kInternal,
                                            HasSubstr("PremappedCopierState")));
    EXPECT_EQ(conn->error_sizes()[2], buf_size);
    EXPECT_THAT(DecodeInt32Payload(conn->completed_buffers()[3]),
                IsOkAndHolds(pattern));
    EXPECT_THAT(conn->errors()[4],
                StatusIs(absl::StatusCode::kInternal,
                         HasSubstr("unready buffer failed")));
    EXPECT_EQ(conn->error_sizes()[4], buf_size);
  }
}

}  // namespace
}  // namespace aux
