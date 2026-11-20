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

#ifndef XLA_PJRT_MANUAL_ASYNC_WORK_RUNNER_H_
#define XLA_PJRT_MANUAL_ASYNC_WORK_RUNNER_H_

#include <cstddef>
#include <deque>
#include <utility>

#include "xla/pjrt/async_work_runner.h"

namespace xla {

// An `AsyncWorkRunner` for tests that queues tasks until the test runs them, so
// the test decides when asynchronous work happens and stays single-threaded.
class ManualAsyncWorkRunner final : public AsyncWorkRunner {
 public:
  void Execute(Task task) override { tasks_.push_back(std::move(task)); }

  // Returns the number of queued tasks that have not run yet.
  size_t num_pending() const { return tasks_.size(); }

  // Runs queued tasks on the calling thread, including ones queued while
  // running, until none remain.
  void RunAll() {
    while (!tasks_.empty()) {
      Task task = std::move(tasks_.front());
      tasks_.pop_front();
      std::move(task)();
    }
  }

 private:
  std::deque<Task> tasks_;
};

}  // namespace xla

#endif  // XLA_PJRT_MANUAL_ASYNC_WORK_RUNNER_H_
