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

#ifndef XLA_PJRT_SE_EVENT_POLLING_CALLBACK_RUNNER_H_
#define XLA_PJRT_SE_EVENT_POLLING_CALLBACK_RUNNER_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/types.h"

namespace xla {

// Runs host callbacks after preceding work on a stream completes, without
// stream host callbacks (e.g. cuLaunchHostFunc).
//
// `ThenCall` records an `se::Event` on the stream and hands it to a dedicated
// polling thread, which polls the event (e.g. cuEventQuery) and invokes the
// callback once the event completes. Under GPU Confidential Computing,
// enqueuing host functions on streams can deadlock inside the driver with
// concurrent kernel setup (e.g. cuFuncSetAttribute) or event recording on
// other threads; recording and querying events does not.
//
// Ordering: callbacks enqueued for the same stream are invoked in the order of
// their `ThenCall` calls, which matches the order in which their events were
// recorded on that stream. There is no ordering across streams.
//
// Callbacks run on the polling thread and must be short and non-blocking;
// expensive work should be forwarded to another thread.
class EventPollingCallbackRunner {
 public:
  // Called with OK once the event completes, or with an error if the event
  // reports an error.
  using Callback = absl::AnyInvocable<void(absl::Status) &&>;

  // Events are created on, and all streams must belong to, `executor`.
  EventPollingCallbackRunner(se::StreamExecutor* executor, tsl::Env* env,
                             const tsl::ThreadOptions& thread_options,
                             absl::string_view thread_name);

  // Calls Shutdown().
  ~EventPollingCallbackRunner();

  EventPollingCallbackRunner(const EventPollingCallbackRunner&) = delete;
  EventPollingCallbackRunner& operator=(const EventPollingCallbackRunner&) =
      delete;

  // Records an event on `stream` and arranges for `callback` to be called from
  // the polling thread once all work enqueued on `stream` before this call has
  // completed. Returns FailedPrecondition after Shutdown(). If an error is
  // returned, `callback` is destroyed without being called.
  absl::Status ThenCall(se::Stream* stream, Callback callback);

  // Blocks until every enqueued event has completed and its callback has
  // returned, then stops the polling thread. Callbacks may call ThenCall while
  // Shutdown() is in progress. Must not be called concurrently with itself.
  void Shutdown();

 private:
  struct PendingCallback {
    std::unique_ptr<se::Event> event;
    Callback callback;
  };

  absl::StatusOr<std::unique_ptr<se::Event>> AcquireEvent();
  void ReleaseEvent(std::unique_ptr<se::Event> event);

  // Polls the oldest pending events of every stream and invokes the callbacks
  // of completed events. Returns true if any callback was invoked.
  bool PollOnce();
  void PollLoop();

  se::StreamExecutor* const executor_;

  // Serializes event recording with enqueueing, so that per-stream queue order
  // matches event record order. This also serializes RecordEvent calls made
  // through this runner (one per device). Never acquired by the polling
  // thread.
  absl::Mutex record_mu_ ABSL_ACQUIRED_BEFORE(mu_);

  // Guards the queues. Never held while calling into the device runtime or
  // invoking callbacks.
  absl::Mutex mu_;
  // Pending callbacks per stream, oldest first. Keys are used only for
  // identity and are never dereferenced. Only the polling thread pops from or
  // erases queues; empty queues are erased.
  absl::flat_hash_map<const se::Stream*, std::deque<PendingCallback>> queues_
      ABSL_GUARDED_BY(mu_);
  int64_t num_pending_ ABSL_GUARDED_BY(mu_) = 0;
  // Set by ThenCall; wakes the polling thread from its backoff sleep.
  bool new_work_ ABSL_GUARDED_BY(mu_) = false;
  bool shutting_down_ ABSL_GUARDED_BY(mu_) = false;
  // Set by the polling thread when it exits.
  bool stopped_ ABSL_GUARDED_BY(mu_) = false;

  absl::Mutex free_events_mu_;
  std::vector<std::unique_ptr<se::Event>> free_events_
      ABSL_GUARDED_BY(free_events_mu_);

  // Declared last so that the thread starts after the state above is
  // initialized.
  std::unique_ptr<tsl::Thread> thread_;
};

}  // namespace xla

#endif  // XLA_PJRT_SE_EVENT_POLLING_CALLBACK_RUNNER_H_
