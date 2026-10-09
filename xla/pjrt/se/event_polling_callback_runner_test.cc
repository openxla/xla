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

#include "xla/pjrt/se/event_polling_callback_runner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/tsl/platform/env.h"

namespace xla {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Invoke;
using ::testing::Return;

// A fake device stream. Each recorded event is assigned the next sequence
// number on the stream, and completes once the test advances the stream's
// completion point past it (stream work completes in order).
class FakeStream {
 public:
  class FakeEvent : public se::Event {
   public:
    Status PollForStatus() override {
      if (stream_->failed_.load()) {
        return Status::kError;
      }
      return stream_->completed_.load() >= seq_ ? Status::kComplete
                                                : Status::kPending;
    }
    // Set when recorded on a stream.
    FakeStream* stream_ = nullptr;
    int64_t seq_ = 0;
  };

  explicit FakeStream(se::StreamExecutor* executor) {
    ON_CALL(stream_, parent()).WillByDefault(Return(executor));
    ON_CALL(stream_, RecordEvent(_)).WillByDefault(Invoke([this](se::Event* e) {
      int64_t seq = ++num_recorded_;
      auto* event = static_cast<FakeEvent*>(e);
      event->stream_ = this;
      event->seq_ = seq;
      last_recorded_seq_ = seq;
      return absl::OkStatus();
    }));
  }

  // Sequence number of the last event recorded by the calling thread.
  static int64_t last_recorded_seq_on_this_thread() {
    return last_recorded_seq_;
  }

  se::Stream* stream() { return &stream_; }
  // Completes all work recorded so far.
  void CompleteAll() { completed_.store(num_recorded_.load()); }
  // Completes the first `n` recorded events.
  void CompleteUpTo(int64_t n) { completed_.store(n); }
  void Fail() { failed_.store(true); }
  void Recover() { failed_.store(false); }

 private:
  static thread_local int64_t last_recorded_seq_;

  ::testing::NiceMock<se::MockStream> stream_;
  std::atomic<int64_t> num_recorded_{0};
  std::atomic<int64_t> completed_{0};
  std::atomic<bool> failed_{false};
};

thread_local int64_t FakeStream::last_recorded_seq_ = 0;

// Records invocation order of callbacks.
class Recorder {
 public:
  EventPollingCallbackRunner::Callback Make(int id) {
    return [this, id](absl::Status status) {
      absl::MutexLock lock(mu_);
      ids_.push_back(id);
      statuses_.push_back(status);
    };
  }
  // Returns true once at least `n` callbacks ran, or false on timeout.
  bool WaitForCountWithTimeout(size_t n, absl::Duration timeout) {
    absl::MutexLock lock(mu_);
    auto reached = [this, n]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
      return ids_.size() >= n;
    };
    return mu_.AwaitWithTimeout(absl::Condition(&reached), timeout);
  }
  std::vector<int> ids() {
    absl::MutexLock lock(mu_);
    return ids_;
  }
  std::vector<absl::Status> statuses() {
    absl::MutexLock lock(mu_);
    return statuses_;
  }

 private:
  absl::Mutex mu_;
  std::vector<int> ids_ ABSL_GUARDED_BY(mu_);
  std::vector<absl::Status> statuses_ ABSL_GUARDED_BY(mu_);
};

class EventPollingCallbackRunnerTest : public ::testing::Test {
 protected:
  EventPollingCallbackRunnerTest() {
    ON_CALL(executor_, CreateEvent()).WillByDefault(Invoke([this]() {
      ++num_events_created_;
      return absl::StatusOr<std::unique_ptr<se::Event>>(
          std::make_unique<FakeStream::FakeEvent>());
    }));
  }

  std::unique_ptr<EventPollingCallbackRunner> MakeRunner() {
    return std::make_unique<EventPollingCallbackRunner>(
        &executor_, tsl::Env::Default(), tsl::ThreadOptions(),
        "test_event_poller");
  }

  ::testing::NiceMock<se::MockStreamExecutor> executor_;
  std::atomic<int64_t> num_events_created_{0};
};

TEST_F(EventPollingCallbackRunnerTest, CallbackWaitsForEventCompletion) {
  FakeStream s(&executor_);
  Recorder recorder;
  auto runner = MakeRunner();
  ASSERT_OK(runner->ThenCall(s.stream(), recorder.Make(1)));
  EXPECT_FALSE(recorder.WaitForCountWithTimeout(1, absl::Milliseconds(50)));
  s.CompleteAll();
  EXPECT_TRUE(recorder.WaitForCountWithTimeout(1, absl::Seconds(10)));
  EXPECT_THAT(recorder.ids(), ElementsAre(1));
  EXPECT_OK(recorder.statuses()[0]);
}

TEST_F(EventPollingCallbackRunnerTest, CallbacksForSameStreamRunInOrder) {
  FakeStream s(&executor_);
  Recorder recorder;
  auto runner = MakeRunner();
  constexpr int kNum = 100;
  for (int i = 0; i < kNum; ++i) {
    ASSERT_OK(runner->ThenCall(s.stream(), recorder.Make(i)));
  }
  // Complete in several steps, so callbacks become ready in batches.
  s.CompleteUpTo(10);
  EXPECT_TRUE(recorder.WaitForCountWithTimeout(10, absl::Seconds(10)));
  EXPECT_FALSE(recorder.WaitForCountWithTimeout(11, absl::Milliseconds(20)));
  s.CompleteUpTo(55);
  EXPECT_TRUE(recorder.WaitForCountWithTimeout(55, absl::Seconds(10)));
  s.CompleteAll();
  EXPECT_TRUE(recorder.WaitForCountWithTimeout(kNum, absl::Seconds(10)));
  std::vector<int> expected(kNum);
  for (int i = 0; i < kNum; ++i) expected[i] = i;
  EXPECT_EQ(recorder.ids(), expected);
}

TEST_F(EventPollingCallbackRunnerTest, StreamsAreIndependent) {
  FakeStream a(&executor_);
  FakeStream b(&executor_);
  Recorder recorder;
  auto runner = MakeRunner();
  ASSERT_OK(runner->ThenCall(a.stream(), recorder.Make(1)));
  ASSERT_OK(runner->ThenCall(b.stream(), recorder.Make(2)));
  // Completing `b` must not wait for the pending event on `a`.
  b.CompleteAll();
  EXPECT_TRUE(recorder.WaitForCountWithTimeout(1, absl::Seconds(10)));
  EXPECT_THAT(recorder.ids(), ElementsAre(2));
  a.CompleteAll();
  EXPECT_TRUE(recorder.WaitForCountWithTimeout(2, absl::Seconds(10)));
  EXPECT_THAT(recorder.ids(), ElementsAre(2, 1));
}

TEST_F(EventPollingCallbackRunnerTest, ConcurrentEnqueuersKeepPerStreamOrder) {
  FakeStream s(&executor_);
  absl::Mutex mu;
  // Stream sequence numbers of the events, in callback invocation order.
  std::vector<int64_t> callback_order;
  auto runner = MakeRunner();
  constexpr int kThreads = 8;
  constexpr int kPerThread = 200;
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&]() {
      for (int i = 0; i < kPerThread; ++i) {
        auto seq = std::make_shared<int64_t>(0);
        ASSERT_OK(runner->ThenCall(s.stream(), [&, seq](absl::Status) {
          absl::MutexLock lock(mu);
          callback_order.push_back(*seq);
        }));
        // No event completes before all threads joined, so setting the
        // sequence number after ThenCall returns happens before the callback.
        *seq = FakeStream::last_recorded_seq_on_this_thread();
      }
    });
  }
  for (auto& t : threads) t.join();
  s.CompleteAll();
  runner.reset();  // Drains.
  absl::MutexLock lock(mu);
  ASSERT_EQ(callback_order.size(), kThreads * kPerThread);
  // Callbacks ran in the order their events were recorded on the stream.
  for (size_t i = 0; i < callback_order.size(); ++i) {
    EXPECT_EQ(callback_order[i], i + 1);
  }
}

TEST_F(EventPollingCallbackRunnerTest, EventErrorIsReportedAndEventReused) {
  FakeStream s(&executor_);
  Recorder recorder;
  auto runner = MakeRunner();
  ASSERT_OK(runner->ThenCall(s.stream(), recorder.Make(0)));
  s.Fail();
  ASSERT_TRUE(recorder.WaitForCountWithTimeout(1, absl::Seconds(10)));
  EXPECT_FALSE(recorder.statuses()[0].ok());
  s.Recover();
  for (int i = 1; i < 5; ++i) {
    ASSERT_OK(runner->ThenCall(s.stream(), recorder.Make(i)));
    s.CompleteAll();
    ASSERT_TRUE(recorder.WaitForCountWithTimeout(i + 1, absl::Seconds(10)));
    EXPECT_OK(recorder.statuses()[i]);
  }
  // Events are recycled before their callbacks run, including errored ones.
  EXPECT_EQ(num_events_created_.load(), 1);
}

TEST_F(EventPollingCallbackRunnerTest, RecordFailureIsReturned) {
  FakeStream s(&executor_);
  auto runner = MakeRunner();
  EXPECT_CALL(*static_cast<se::MockStream*>(s.stream()), RecordEvent(_))
      .WillOnce(Return(absl::InternalError("record failed")))
      .WillRepeatedly(::testing::DoDefault());
  bool called = false;
  EXPECT_FALSE(
      runner->ThenCall(s.stream(), [&](absl::Status) { called = true; }).ok());
  EXPECT_FALSE(called);
  // The event was returned for reuse.
  ASSERT_OK(runner->ThenCall(s.stream(), [](absl::Status) {}));
  EXPECT_EQ(num_events_created_.load(), 1);
  s.CompleteAll();
}

TEST_F(EventPollingCallbackRunnerTest, ShutdownDrainsThenRejectsNewCallbacks) {
  FakeStream s(&executor_);
  Recorder recorder;
  auto runner = MakeRunner();
  for (int i = 0; i < 10; ++i) {
    ASSERT_OK(runner->ThenCall(s.stream(), recorder.Make(i)));
  }
  absl::Notification shut_down;
  std::thread shutdown_thread([&]() {
    runner->Shutdown();
    shut_down.Notify();
  });
  // Shutdown blocks while events are pending.
  EXPECT_FALSE(
      shut_down.WaitForNotificationWithTimeout(absl::Milliseconds(50)));
  EXPECT_TRUE(recorder.ids().empty());
  s.CompleteAll();
  shutdown_thread.join();
  EXPECT_EQ(recorder.ids().size(), 10);
  EXPECT_THAT(runner->ThenCall(s.stream(), [](absl::Status) {}),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace xla
