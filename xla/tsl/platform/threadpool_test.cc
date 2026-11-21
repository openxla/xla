/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/platform/threadpool.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/synchronization/notification.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"

namespace tsl::thread {
namespace {

TEST(ThreadPoolTest, AsExecutor) {
  ThreadPool thread_pool(Env::Default(), "test", 4);
  Executor* executor = thread_pool.AsExecutor();

  absl::Notification notification;
  executor->Execute([&] { notification.Notify(); });
  notification.WaitForNotification();
}

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
TEST(ThreadPoolTest, RunsInlineWithoutPthreads) {
  ThreadPool thread_pool(Env::Default(), "test", 8);
  EXPECT_EQ(thread_pool.NumThreads(), 1);
  EXPECT_EQ(thread_pool.CurrentThreadId(), -1);

  std::vector<int> order;
  thread_pool.Schedule([&] {
    order.push_back(1);
    thread_pool.Schedule([&] { order.push_back(2); });
    order.push_back(3);
  });
  EXPECT_EQ(order, (std::vector<int>{1, 2, 3}));

  int result = 0;
  thread_pool.AsExecutor()->Execute(
      [value = std::make_unique<int>(42), &result] { result = *value; });
  EXPECT_EQ(result, 42);

  std::vector<int> visits(37);
  for (const auto& params : {ThreadPool::SchedulingParams::Fixed(3),
                             ThreadPool::SchedulingParams::Adaptive(100)}) {
    thread_pool.ParallelFor(37, params, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) ++visits[i];
    });
  }
  EXPECT_EQ(visits, std::vector<int>(37, 2));
}
#endif

}  // namespace
}  // namespace tsl::thread
