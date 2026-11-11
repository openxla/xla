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

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "tsl/platform/casts.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/python/ifrt/array.h"
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
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace aux {
namespace {

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

constexpr int kNumRacers = 2;

// Outcome of the pulls issued by racing PullTable::Entry::Handle() calls.
struct RaceResults {
  absl::Mutex mu;
  int num_parked = 0;
  int num_sent = 0;
  int num_errors = 0;
};

// Parks the first kNumRacers SendError() calls until all of them have arrived,
// which lines up racing Handle() calls right before their next buffer id.
class RacingConnectionState : public ConnectionState {
 public:
  explicit RacingConnectionState(std::shared_ptr<RaceResults> results)
      : results_(std::move(results)) {}

  void Send(size_t req_id, const void* data, size_t offset, size_t size,
            bool is_largest, absl::AnyInvocable<void() &&> on_done) override {
    std::move(on_done)();
    absl::MutexLock l(results_->mu);
    ++results_->num_sent;
  }

  void SendError(size_t req_id, size_t offset, size_t size, bool is_largest,
                 absl::Status status) override {
    absl::MutexLock l(results_->mu);
    if (results_->num_parked < kNumRacers) {
      ++results_->num_parked;
      results_->mu.Await(absl::Condition(
          +[](RaceResults* r) { return r->num_parked == kNumRacers; },
          results_.get()));
      return;
    }
    ++results_->num_errors;
  }

 private:
  std::shared_ptr<RaceResults> results_;
};

// Pulls buffer 0 of `entry` from kNumRacers threads at once and expects exactly
// one of them to get it. `ready_promise` gates the readiness of that buffer.
void ExpectBufferFetchedOnce(tsl::RCReference<PullTable::Entry> entry,
                             tsl::Promise<> ready_promise) {
  SocketTransferPullRequest req;
  req.add_buffer_ids(1);  // Out of range, so that Handle() calls SendError().
  req.add_buffer_ids(0);
  auto results = std::make_shared<RaceResults>();
  {
    std::vector<std::unique_ptr<tsl::Thread>> racers;
    for (int i = 0; i < kNumRacers; ++i) {
      racers.emplace_back(
          tsl::Env::Default()->StartThread({}, "racer", [&, i]() {
            entry->Handle(tsl::MakeRef<RacingConnectionState>(results), req,
                          i * req.buffer_ids_size());
          }));
    }
  }
  ready_promise.Set();
  absl::MutexLock l(results->mu);
  results->mu.Await(absl::Condition(
      +[](RaceResults* r) { return r->num_sent + r->num_errors == kNumRacers; },
      results.get()));
  EXPECT_EQ(results->num_sent, 1);
  EXPECT_EQ(results->num_errors, kNumRacers - 1);
}

TEST(PullTableEntry, ConcurrentHandleFetchesBufferOnce) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  std::shared_ptr<xla::PjRtClient> pjrt_client =
      absl::down_cast<xla::ifrt::PjRtClient*>(client.get())
          ->shared_ptr_pjrt_client();
  size_t xfer_size = 1024 * 1024;
  TF_ASSERT_OK_AND_ASSIGN(auto scratch,
                          AllocateAndMapPjrtMemory(pjrt_client, 4 * xfer_size));
  auto cstate = std::make_shared<PremappedCopierState>(scratch, 4, xfer_size);
  TF_ASSERT_OK_AND_ASSIGN(auto arr, tests::CopyTestPatternToDevice(
                                        client.get(), client->devices()[0],
                                        tests::CreateTestPattern(0, 1024)));
  TF_ASSERT_OK(arr->GetReadyFuture().Await());
  std::shared_ptr<xla::PjRtBuffer> buffer =
      static_cast<xla::ifrt::PjRtCompatibleArray*>(arr.get())
          ->pjrt_buffers()[0];
  TF_ASSERT_OK_AND_ASSIGN(size_t buf_size, buffer->GetOnDeviceSizeInBytes());
  {
    SCOPED_TRACE("PjRtBufferEntry");
    auto [promise, future] = tsl::MakePromise<>();
    std::vector<PjRtBufferEntry::BufferRef> refs;
    refs.push_back({buffer, buf_size, std::move(future)});
    ExpectBufferFetchedOnce(
        tsl::MakeRef<PjRtBufferEntry>(std::move(refs), cstate, xfer_size),
        std::move(promise));
  }
  {
    SCOPED_TRACE("RawBufferEntry");
    TF_ASSERT_OK_AND_ASSIGN(
        auto raw_buffer,
        xla::PjRtRawBuffer::CreateRawAliasOfBuffer(buffer.get()));
    auto [promise, future] = tsl::MakePromise<>();
    std::vector<RawBufferEntry::BufferRef> refs;
    refs.push_back({std::move(future), std::move(raw_buffer), buf_size});
    ExpectBufferFetchedOnce(
        tsl::MakeRef<RawBufferEntry>(std::move(refs), cstate, xfer_size),
        std::move(promise));
  }
}

class CountingConnectionState : public ConnectionState {
 public:
  void Send(size_t req_id, const void* data, size_t offset, size_t size,
            bool is_largest, absl::AnyInvocable<void() &&> on_done) override {
    std::move(on_done)();
    absl::MutexLock l(mu_);
    ++num_sent_;
  }

  void SendError(size_t req_id, size_t offset, size_t size, bool is_largest,
                 absl::Status status) override {
    absl::MutexLock l(mu_);
    ++num_errors_;
  }

  void AwaitTotal(int expected) {
    absl::MutexLock l(mu_);
    auto done = [this, expected]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
      return num_sent_ + num_errors_ == expected;
    };
    mu_.Await(absl::Condition(&done));
  }

  int num_sent() {
    absl::MutexLock l(mu_);
    return num_sent_;
  }

  int num_errors() {
    absl::MutexLock l(mu_);
    return num_errors_;
  }

 private:
  absl::Mutex mu_;
  int num_sent_ ABSL_GUARDED_BY(mu_) = 0;
  int num_errors_ ABSL_GUARDED_BY(mu_) = 0;
};

void ExpectCompletionOnlyAfterAllDistinctBuffersConsumed(
    tsl::RCReference<PullTable::Entry> entry) {
  auto state = tsl::MakeRef<CountingConnectionState>();

  // Requesting an out-of-range buffer ID and duplicate buffer 0 must only
  // count buffer 0 once, so buffer 1 remains unconsumed.
  SocketTransferPullRequest req1;
  req1.add_buffer_ids(99);
  req1.add_buffer_ids(0);
  req1.add_buffer_ids(0);
  EXPECT_FALSE(entry->Handle(state, req1, /*base_req_id=*/0));

  // Re-requesting already-consumed buffer 0 must still not complete the entry.
  SocketTransferPullRequest req2;
  req2.add_buffer_ids(0);
  EXPECT_FALSE(entry->Handle(state, req2, /*base_req_id=*/3));

  // Consuming buffer 1 completes the 2-buffer entry.
  SocketTransferPullRequest req3;
  req3.add_buffer_ids(1);
  EXPECT_TRUE(entry->Handle(state, req3, /*base_req_id=*/4));

  state->AwaitTotal(5);
  EXPECT_EQ(state->num_sent(), 2);
  EXPECT_EQ(state->num_errors(), 3);
}

TEST(PullTableEntry, DuplicateAndOutOfRangeIdsDoNotPrematurelyCompleteEntry) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  std::shared_ptr<xla::PjRtClient> pjrt_client =
      absl::down_cast<xla::ifrt::PjRtClient*>(client.get())
          ->shared_ptr_pjrt_client();
  size_t xfer_size = 1024 * 1024;
  TF_ASSERT_OK_AND_ASSIGN(auto scratch,
                          AllocateAndMapPjrtMemory(pjrt_client, 4 * xfer_size));
  auto cstate = std::make_shared<PremappedCopierState>(scratch, 4, xfer_size);
  TF_ASSERT_OK_AND_ASSIGN(auto arr0, tests::CopyTestPatternToDevice(
                                         client.get(), client->devices()[0],
                                         tests::CreateTestPattern(0, 1024)));
  TF_ASSERT_OK_AND_ASSIGN(auto arr1, tests::CopyTestPatternToDevice(
                                         client.get(), client->devices()[0],
                                         tests::CreateTestPattern(1, 1024)));
  TF_ASSERT_OK(arr0->GetReadyFuture().Await());
  TF_ASSERT_OK(arr1->GetReadyFuture().Await());
  std::shared_ptr<xla::PjRtBuffer> buf0 =
      static_cast<xla::ifrt::PjRtCompatibleArray*>(arr0.get())
          ->pjrt_buffers()[0];
  std::shared_ptr<xla::PjRtBuffer> buf1 =
      static_cast<xla::ifrt::PjRtCompatibleArray*>(arr1.get())
          ->pjrt_buffers()[0];
  TF_ASSERT_OK_AND_ASSIGN(size_t buf0_size, buf0->GetOnDeviceSizeInBytes());
  TF_ASSERT_OK_AND_ASSIGN(size_t buf1_size, buf1->GetOnDeviceSizeInBytes());

  {
    SCOPED_TRACE("PjRtBufferEntry");
    std::vector<PjRtBufferEntry::BufferRef> refs = {
        {buf0, buf0_size},
        {buf1, buf1_size},
    };
    ExpectCompletionOnlyAfterAllDistinctBuffersConsumed(
        tsl::MakeRef<PjRtBufferEntry>(std::move(refs), cstate, xfer_size));
  }
  {
    SCOPED_TRACE("RawBufferEntry");
    TF_ASSERT_OK_AND_ASSIGN(
        auto raw0, xla::PjRtRawBuffer::CreateRawAliasOfBuffer(buf0.get()));
    TF_ASSERT_OK_AND_ASSIGN(
        auto raw1, xla::PjRtRawBuffer::CreateRawAliasOfBuffer(buf1.get()));
    std::vector<RawBufferEntry::BufferRef> refs = {
        {tsl::Future<>(absl::OkStatus()), std::move(raw0), buf0_size},
        {tsl::Future<>(absl::OkStatus()), std::move(raw1), buf1_size},
    };
    ExpectCompletionOnlyAfterAllDistinctBuffersConsumed(
        tsl::MakeRef<RawBufferEntry>(std::move(refs), cstate, xfer_size));
  }
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

}  // namespace
}  // namespace aux
