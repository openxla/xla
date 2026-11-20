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

#include "xla/pjrt/linearize_throttler.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "tsl/platform/numa.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/manual_async_work_runner.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/staging_buffer.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

// Allocates with malloc and counts allocations.
class CountingHostMemoryAllocator final : public HostMemoryAllocator {
 public:
  absl_nullable OwnedPtr Allocate(size_t size,
                                  const AllocateOptions& options) override {
    ++num_allocations_;
    return OwnedPtr(static_cast<uint8_t*>(std::malloc(size)),
                    {+[](void* ptr, void* arg) { std::free(ptr); }, nullptr});
  }

  int num_allocations() const { return num_allocations_; }

 private:
  int num_allocations_ = 0;
};

constexpr size_t kBudget = 100;

LinearizeThrottler::Options BudgetOptions() {
  LinearizeThrottler::Options options;
  options.max_inflight_linearization_bytes = kBudget;
  options.max_inflight_delinearization_bytes = kBudget;
  return options;
}

class LinearizeThrottlerTest : public ::testing::Test {
 protected:
  tsl::AsyncValueRef<PjRtStagingBuffer> FromData(size_t size) {
    return throttler_.CreateFromData(host_data_.data(), size,
                                     /*on_done_with_host_buffer=*/nullptr);
  }

  tsl::AsyncValueRef<PjRtStagingBuffer> ForDelinearization(size_t size) {
    return throttler_.AllocateForDelinearizationAsync(
        size, tsl::port::kNUMANoAffinity, /*local_device_id=*/std::nullopt);
  }

  // The throttler's destructor waits until every credit is returned. Buffers a
  // test still holds are released when its body returns; that can queue
  // deferred tasks, which hold a credit until they run, so run them here.
  void TearDown() override { runner_.RunAll(); }

  std::vector<uint8_t> host_data_ = std::vector<uint8_t>(4 * kBudget);
  ManualAsyncWorkRunner runner_;
  CountingHostMemoryAllocator allocator_;
  LinearizeThrottler throttler_{&allocator_, &runner_, BudgetOptions()};
};

TEST_F(LinearizeThrottlerTest, WithinBudgetRunsInlineInCaller) {
  tsl::AsyncValueRef<PjRtStagingBuffer> buffer = FromData(kBudget);

  EXPECT_TRUE(buffer.IsAvailable());
  EXPECT_EQ(runner_.num_pending(), 0);
  EXPECT_EQ(buffer->data().size(), kBudget);
}

TEST_F(LinearizeThrottlerTest, ZeroBytesRunsInlineEvenWhenBudgetIsExhausted) {
  tsl::AsyncValueRef<PjRtStagingBuffer> full = FromData(kBudget);

  tsl::AsyncValueRef<PjRtStagingBuffer> empty = FromData(0);

  EXPECT_TRUE(empty.IsAvailable());
}

TEST_F(LinearizeThrottlerTest, OverBudgetIsDeferredWithoutBlockingTheCaller) {
  tsl::AsyncValueRef<PjRtStagingBuffer> first = FromData(60);

  // Returns immediately with an unavailable buffer rather than blocking.
  tsl::AsyncValueRef<PjRtStagingBuffer> second = FromData(60);
  EXPECT_FALSE(second.IsAvailable());
  EXPECT_EQ(runner_.num_pending(), 0);

  // Releasing the first buffer schedules the deferred one on the runner.
  first.reset();
  EXPECT_FALSE(second.IsAvailable());
  EXPECT_EQ(runner_.num_pending(), 1);

  runner_.RunAll();
  EXPECT_TRUE(second.IsAvailable());
}

TEST_F(LinearizeThrottlerTest, AdmittedBytesNeverExceedBudget) {
  std::vector<tsl::AsyncValueRef<PjRtStagingBuffer>> buffers;
  for (int i = 0; i < 5; ++i) {
    buffers.push_back(FromData(30));
  }

  // 3 * 30 <= 100 < 4 * 30.
  EXPECT_TRUE(buffers[0].IsAvailable());
  EXPECT_TRUE(buffers[1].IsAvailable());
  EXPECT_TRUE(buffers[2].IsAvailable());
  EXPECT_FALSE(buffers[3].IsAvailable());
  EXPECT_FALSE(buffers[4].IsAvailable());

  // Freeing 30 bytes admits exactly one more.
  buffers[0].reset();
  runner_.RunAll();
  EXPECT_TRUE(buffers[3].IsAvailable());
  EXPECT_FALSE(buffers[4].IsAvailable());
}

TEST_F(LinearizeThrottlerTest, ManyRequestsStayDeferredWhileBudgetIsFull) {
  tsl::AsyncValueRef<PjRtStagingBuffer> full = FromData(kBudget);

  std::vector<tsl::AsyncValueRef<PjRtStagingBuffer>> queued;
  for (int i = 0; i < 1000; ++i) {
    queued.push_back(FromData(1));
  }

  for (const tsl::AsyncValueRef<PjRtStagingBuffer>& buffer : queued) {
    EXPECT_FALSE(buffer.IsAvailable());
  }
}

TEST_F(LinearizeThrottlerTest, NewRequestThatFitsOvertakesQueuedRequests) {
  tsl::AsyncValueRef<PjRtStagingBuffer> held = FromData(90);
  tsl::AsyncValueRef<PjRtStagingBuffer> big = FromData(50);
  EXPECT_FALSE(big.IsAvailable());

  // Fits in the remaining 10 bytes, so it's admitted ahead of `big`.
  tsl::AsyncValueRef<PjRtStagingBuffer> small = FromData(5);
  EXPECT_TRUE(small.IsAvailable());

  held.reset();
  runner_.RunAll();
  EXPECT_TRUE(big.IsAvailable());
}

TEST_F(LinearizeThrottlerTest, QueuedRequestsAreAdmittedInFifoOrder) {
  tsl::AsyncValueRef<PjRtStagingBuffer> first_half = FromData(kBudget / 2);
  tsl::AsyncValueRef<PjRtStagingBuffer> second_half = FromData(kBudget / 2);
  tsl::AsyncValueRef<PjRtStagingBuffer> big = FromData(60);
  tsl::AsyncValueRef<PjRtStagingBuffer> small = FromData(30);

  // 50 free bytes would fit `small` but not `big`, which is ahead of it.
  first_half.reset();
  EXPECT_EQ(runner_.num_pending(), 0);

  second_half.reset();
  EXPECT_EQ(runner_.num_pending(), 2);
  runner_.RunAll();
  EXPECT_TRUE(big.IsAvailable());
  EXPECT_TRUE(small.IsAvailable());
}

TEST_F(LinearizeThrottlerTest, OversizeRequestIsCappedAndAdmittedAlone) {
  tsl::AsyncValueRef<PjRtStagingBuffer> oversize = FromData(3 * kBudget);

  EXPECT_TRUE(oversize.IsAvailable());
  EXPECT_EQ(oversize->data().size(), 3 * kBudget);

  // It holds the whole budget (capped at kBudget), so nothing else fits.
  tsl::AsyncValueRef<PjRtStagingBuffer> next = FromData(1);
  EXPECT_FALSE(next.IsAvailable());

  oversize.reset();
  runner_.RunAll();
  EXPECT_TRUE(next.IsAvailable());
}

TEST_F(LinearizeThrottlerTest, OversizeRequestWaitsForAnEmptyThrottler) {
  tsl::AsyncValueRef<PjRtStagingBuffer> held = FromData(1);

  tsl::AsyncValueRef<PjRtStagingBuffer> oversize = FromData(3 * kBudget);
  EXPECT_FALSE(oversize.IsAvailable());

  held.reset();
  runner_.RunAll();
  EXPECT_TRUE(oversize.IsAvailable());
}

TEST_F(LinearizeThrottlerTest, OnDoneWithHostBufferRunsWhenBufferIsReleased) {
  bool done = false;
  tsl::AsyncValueRef<PjRtStagingBuffer> buffer = throttler_.CreateFromData(
      host_data_.data(), 10, [&done]() { done = true; });
  EXPECT_TRUE(buffer.IsAvailable());
  EXPECT_FALSE(done);

  buffer.reset();
  EXPECT_TRUE(done);
}

TEST_F(LinearizeThrottlerTest, SyncStagingDestOverdraftsTheBudget) {
  tsl::AsyncValueRef<PjRtStagingBuffer> full = FromData(kBudget);

  LinearizeThrottler::StagingReservation sync_dest =
      throttler_.AllocateStagingDest(/*sync=*/true, 50,
                                     /*reused_buffer=*/PjRtRawBufferRef());
  EXPECT_TRUE(sync_dest.IsAvailable());

  // The overdraft counts against the budget: after `full` is released the
  // in-flight bytes are still 50, so 60 more bytes don't fit.
  tsl::AsyncValueRef<PjRtStagingBuffer> next = FromData(60);
  full.reset();
  runner_.RunAll();
  EXPECT_FALSE(next.IsAvailable());

  sync_dest = LinearizeThrottler::StagingReservation();
  runner_.RunAll();
  EXPECT_TRUE(next.IsAvailable());
}

TEST_F(LinearizeThrottlerTest, AsyncStagingDestIsDeferredAndAllocatesLazily) {
  tsl::AsyncValueRef<PjRtStagingBuffer> full = FromData(kBudget);

  LinearizeThrottler::StagingReservation dest = throttler_.AllocateStagingDest(
      /*sync=*/false, 50, /*reused_buffer=*/PjRtRawBufferRef());
  EXPECT_FALSE(dest.IsAvailable());

  full.reset();
  runner_.RunAll();
  ASSERT_TRUE(dest.IsAvailable());
  EXPECT_EQ(allocator_.num_allocations(), 0);

  absl::StatusOr<tsl::AsyncValueRef<PjRtStagingBuffer>> buffer =
      std::move(dest).Allocate();
  ASSERT_THAT(buffer, IsOk());
  EXPECT_EQ(allocator_.num_allocations(), 1);
  EXPECT_EQ((*buffer)->data().size(), 50);
}

TEST_F(LinearizeThrottlerTest, DelinearizationHasItsOwnBudget) {
  tsl::AsyncValueRef<PjRtStagingBuffer> linearize = FromData(kBudget);

  tsl::AsyncValueRef<PjRtStagingBuffer> delinearize =
      ForDelinearization(kBudget);

  EXPECT_TRUE(delinearize.IsAvailable());
  EXPECT_EQ(allocator_.num_allocations(), 1);
}

TEST_F(LinearizeThrottlerTest, DelinearizationOverBudgetIsDeferred) {
  tsl::AsyncValueRef<PjRtStagingBuffer> first = ForDelinearization(kBudget);

  tsl::AsyncValueRef<PjRtStagingBuffer> second = ForDelinearization(1);
  EXPECT_FALSE(second.IsAvailable());
  EXPECT_EQ(allocator_.num_allocations(), 1);

  first.reset();
  runner_.RunAll();
  EXPECT_TRUE(second.IsAvailable());
  EXPECT_EQ(allocator_.num_allocations(), 2);
}

TEST_F(LinearizeThrottlerTest, SyncDelinearizeOverdraftsAndRunsInline) {
  tsl::AsyncValueRef<PjRtStagingBuffer> full = ForDelinearization(kBudget);

  bool ran = false;
  absl::Status status = throttler_.ThrottleSyncDelinearize(50, [&ran]() {
    ran = true;
    return absl::OkStatus();
  });

  EXPECT_TRUE(ran);
  EXPECT_THAT(status, IsOk());
}

TEST_F(LinearizeThrottlerTest, SyncDelinearizeReturnsTheCallbackStatus) {
  absl::Status status = throttler_.ThrottleSyncDelinearize(
      10, []() { return absl::InternalError("delinearize failed"); });

  EXPECT_THAT(status,
              StatusIs(absl::StatusCode::kInternal, "delinearize failed"));
}

}  // namespace
}  // namespace xla
