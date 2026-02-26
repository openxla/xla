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

#include "xla/runtime/hang_watchdog.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/env.h"

namespace xla {

struct HangWatchdog::Guard {
  Guard(absl::string_view action, absl::Duration duration,
        CancelCallback cancel)
      : action(action),
        duration(duration),
        deadline(absl::Now() + duration),
        cancel(std::move(cancel)) {}

  ~Guard() {
    absl::Duration elapsed = absl::Now() - (deadline - duration);
    VLOG(5) << absl::StreamFormat(
        "%s completed after %v (was added to watch for %v)", action, elapsed,
        duration);
  }

  std::string action;
  absl::Duration duration;
  absl::Time deadline;
  CancelCallback cancel;
};

HangWatchdog::HangWatchdog(tsl::Env* env, absl::string_view name,
                           size_t num_threads)
    : thread_pool_(env, absl::StrCat(name, "-hang-watchdog"), num_threads) {}

HangWatchdog::CancelCallback HangWatchdog::Abort(absl::string_view action,
                                                 absl::Duration duration,
                                                 CancelCallback pre_abort) {
  // Only one thread actually aborts the process. We wait 10 seconds before
  // aborting to give other hang watchdogs a chance to trigger their cancel
  // callbacks and log useful debugging information.
  static absl::once_flag abort_once;

  return [action = std::string(action), duration,
          pre_abort = std::move(pre_abort)]() mutable {
    LOG(ERROR) << absl::StreamFormat("%s failed to finish in %v.", action,
                                     duration);
    if (pre_abort) {
      std::move(pre_abort)();
    }

    absl::call_once(abort_once, [&] {
      LOG(ERROR) << absl::StreamFormat(
          "%s: prepare to abort the process to avoid infinite hangs.", action);
      absl::SleepFor(absl::Seconds(10));
      LOG(FATAL) << absl::StreamFormat(
          "%s: abort the process to avoid infinite hangs.", action);
    });
  };
}

std::shared_ptr<HangWatchdog::Guard> HangWatchdog::Watch(
    absl::string_view action, absl::Duration duration, CancelCallback cancel) {
  // Guard computes a deadline from the current time and the requested duration
  // so that accumulated sleep/wake-up jitter doesn't extend the timeout.
  auto guard = std::make_shared<Guard>(action, duration, std::move(cancel));

  VLOG(3) << absl::StreamFormat(
      "Watch %s for %v (deadline %s)", action, duration,
      absl::FormatTime("%Y-%m-%d %H:%M:%S", guard->deadline,
                       absl::LocalTimeZone()));

  {  // Track newly created guard.
    absl::MutexLock lock(&mu_);
    guards_.push_back(guard);
  }

  // Schedule the first check. Subsequent checks are scheduled recursively by
  // ScheduleCheck to avoid head-of-line blocking a thread pool thread.
  ScheduleCheck(guard, /*sleep_interval=*/absl::Milliseconds(100));

  return guard;
}

void HangWatchdog::ScheduleCheck(std::weak_ptr<Guard> guard,
                                 absl::Duration sleep_interval) {
  static constexpr absl::Duration kMaxSleepInterval = absl::Seconds(5);

  thread_pool_.Schedule([this, guard = std::move(guard), sleep_interval] {
    // If the guard is expired, we are done. We schedule one task per guard
    // because it's easier than dealing with persistent tasks and the
    // synchronization required to make it safe. Missing a timed out guard
    // is a guaranteed way to get a deadlock; we prefer to keep things simple!
    if (guard.expired()) {
      return;
    }

    // Process pending guards: collect expired guards for later cancellation.
    absl::Time deadline = absl::InfiniteFuture();
    std::vector<std::shared_ptr<Guard>> expired;
    {
      absl::MutexLock lock(mu_);
      auto is_expired = [&](std::weak_ptr<Guard>& weak_guard) {
        if (std::shared_ptr<Guard> guard = weak_guard.lock()) {
          // Guard is still active, keep it in the list of pending guards.
          if (absl::Now() < guard->deadline) {
            deadline = std::min(deadline, guard->deadline);
            return false;
          }
          // Guard is expired, we pass the ownership to `expired` vector and
          // will call the cancellation callback from the current task.
          expired.push_back(std::move(guard));
          return true;
        }
        return true;
      };
      auto it = std::remove_if(guards_.begin(), guards_.end(), is_expired);
      guards_.erase(it, guards_.end());
    }

    // Process expired guards and call cancel callbacks, we do it without
    // holding the mutex to allow running multiple cancellation callbacks
    // concurrently.
    for (std::shared_ptr<Guard>& guard : expired) {
      VLOG(3) << absl::StreamFormat(
          "%s didn't finish in %v, calling cancel callback.", guard->action,
          guard->duration);
      std::move(guard->cancel)();
    }

    // We have no more guards to process, stop recursive scheduling loop.
    if (deadline == absl::InfiniteFuture()) {
      return;
    }

    // We must not hold `mu_` here because `Schedule` can execute the task in
    // the caller thread if the task queue is full, which can lead to a
    // deadlock.

    // Sleep for the current interval (capped at remaining time to deadline)
    // and schedule the next check with a doubled interval.
    absl::SleepFor(std::min(sleep_interval, deadline - absl::Now()));
    ScheduleCheck(guard, std::min(sleep_interval * 2, kMaxSleepInterval));
  });
}

}  // namespace xla
