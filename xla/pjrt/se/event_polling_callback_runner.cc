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

#include <algorithm>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"

namespace xla {

// While events are pending but none completes, the polling thread sleeps
// between polls, doubling the sleep from 20us up to a 200us cap (which bounds
// the latency added to a callback). Any completion resets it, and enqueueing a
// callback wakes the thread and resets it.
constexpr absl::Duration kMinPollInterval = absl::Microseconds(20);
constexpr absl::Duration kMaxPollInterval = absl::Microseconds(200);

EventPollingCallbackRunner::EventPollingCallbackRunner(
    se::StreamExecutor* executor, tsl::Env* env,
    const tsl::ThreadOptions& thread_options, absl::string_view thread_name)
    : executor_(executor) {
  thread_.reset(env->StartThread(thread_options, std::string(thread_name),
                                 [this]() { PollLoop(); }));
}

EventPollingCallbackRunner::~EventPollingCallbackRunner() { Shutdown(); }

void EventPollingCallbackRunner::Shutdown() {
  {
    absl::MutexLock lock(mu_);
    shutting_down_ = true;
  }
  // Joins the polling thread, which exits once no callbacks are pending.
  thread_.reset();
}

absl::StatusOr<std::unique_ptr<se::Event>>
EventPollingCallbackRunner::AcquireEvent() {
  {
    absl::MutexLock lock(free_events_mu_);
    if (!free_events_.empty()) {
      std::unique_ptr<se::Event> event = std::move(free_events_.back());
      free_events_.pop_back();
      return event;
    }
  }
  return executor_->CreateEvent();
}

void EventPollingCallbackRunner::ReleaseEvent(
    std::unique_ptr<se::Event> event) {
  absl::MutexLock lock(free_events_mu_);
  free_events_.push_back(std::move(event));
}

absl::Status EventPollingCallbackRunner::ThenCall(se::Stream* stream,
                                                  Callback callback) {
  DCHECK_EQ(stream->parent(), executor_);
  absl::StatusOr<std::unique_ptr<se::Event>> event = AcquireEvent();
  if (!event.ok()) {
    return event.status();
  }
  absl::MutexLock record_lock(record_mu_);
  if (absl::Status status = stream->RecordEvent(event->get()); !status.ok()) {
    ReleaseEvent(*std::move(event));
    return status;
  }
  absl::MutexLock lock(mu_);
  // The polling thread decides to exit under `mu_` when nothing is pending,
  // so either it sees this callback or this sees `stopped_`.
  if (stopped_) {
    return absl::FailedPreconditionError(
        "EventPollingCallbackRunner has been shut down.");
  }
  queues_[stream].push_back({*std::move(event), std::move(callback)});
  ++num_pending_;
  new_work_ = true;
  return absl::OkStatus();
}

bool EventPollingCallbackRunner::PollOnce() {
  // Snapshot the oldest pending event of every stream. Events are owned by
  // their queue entries, which only this thread removes, so the pointers stay
  // valid after `mu_` is released.
  std::vector<std::pair<const se::Stream*, se::Event*>> heads;
  {
    absl::MutexLock lock(mu_);
    heads.reserve(queues_.size());
    for (const auto& [stream, queue] : queues_) {
      heads.emplace_back(stream, queue.front().event.get());
    }
  }

  bool progress = false;
  for (auto [stream, event] : heads) {
    // Work on a stream completes in order, so stop at the first pending event.
    while (event != nullptr) {
      se::Event::Status event_status = event->PollForStatus();
      if (event_status == se::Event::Status::kPending) {
        break;
      }
      PendingCallback done;
      {
        absl::MutexLock lock(mu_);
        auto it = queues_.find(stream);
        std::deque<PendingCallback>& queue = it->second;
        done = std::move(queue.front());
        queue.pop_front();
        if (queue.empty()) {
          queues_.erase(it);
          event = nullptr;
        } else {
          event = queue.front().event.get();
        }
        --num_pending_;
      }
      // Recycle even errored events (re-recording re-arms them), so that the
      // polling thread never destroys events (e.g. cuEventDestroy), which can
      // contend on driver locks.
      ReleaseEvent(std::move(done.event));
      tsl::profiler::TraceMe trace("EventPollingCallbackRunner::Callback");
      std::move(done.callback)(
          event_status == se::Event::Status::kComplete
              ? absl::OkStatus()
              : absl::InternalError(
                    "Error polling for event status of a stream callback."));
      progress = true;
    }
  }
  return progress;
}

void EventPollingCallbackRunner::PollLoop() {
  absl::Duration interval = kMinPollInterval;
  while (true) {
    {
      absl::MutexLock lock(mu_);
      auto has_work_or_shutting_down =
          [this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
            return num_pending_ > 0 || shutting_down_;
          };
      mu_.Await(absl::Condition(&has_work_or_shutting_down));
      if (num_pending_ == 0) {
        // Shutting down and fully drained.
        stopped_ = true;
        return;
      }
      // The poll below sees everything enqueued so far.
      new_work_ = false;
    }
    if (PollOnce()) {
      interval = kMinPollInterval;
      continue;
    }
    absl::MutexLock lock(mu_);
    if (mu_.AwaitWithTimeout(absl::Condition(&new_work_), interval)) {
      interval = kMinPollInterval;
    } else {
      interval = std::min(2 * interval, kMaxPollInterval);
    }
  }
}

}  // namespace xla
