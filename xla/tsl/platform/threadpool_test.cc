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

#include <atomic>
#include <cstdint>
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

TEST(ThreadPoolTest, TransformRangeConcurrentlyCoversRange) {
  ThreadPool thread_pool(Env::Default(), "test", 4);
  for (int64_t total : {1, 2, 7, 64, 1000}) {
    for (int64_t block_size : {1, 3, 16, 2000}) {
      std::vector<std::atomic<int>> hits(total);
      thread_pool.TransformRangeConcurrently(
          block_size, total, [&](int64_t first, int64_t last) {
            ASSERT_LE(last - first, block_size);
            for (int64_t i = first; i < last; ++i) {
              hits[i].fetch_add(1, std::memory_order_relaxed);
            }
          });
      for (int64_t i = 0; i < total; ++i) {
        EXPECT_EQ(hits[i].load(), 1)
            << "total=" << total << " block_size=" << block_size << " i=" << i;
      }
    }
  }
}

// Nested calls where every pool thread is itself blocked inside an outer call
// used to deadlock, since the inner work could never be scheduled. The caller
// now runs unclaimed blocks itself, so this must complete.
TEST(ThreadPoolTest, TransformRangeConcurrentlyNestedDoesNotDeadlock) {
  ThreadPool thread_pool(Env::Default(), "test", 2);
  constexpr int64_t kOuter = 16;
  constexpr int64_t kInner = 16;
  std::atomic<int64_t> count = 0;
  thread_pool.TransformRangeConcurrently(
      /*block_size=*/1, kOuter, [&](int64_t first, int64_t last) {
        for (int64_t i = first; i < last; ++i) {
          thread_pool.TransformRangeConcurrently(
              /*block_size=*/1, kInner, [&](int64_t lo, int64_t hi) {
                count.fetch_add(hi - lo, std::memory_order_relaxed);
              });
        }
      });
  EXPECT_EQ(count.load(), kOuter * kInner);
}

// Even if every pool thread is busy for the full duration of the call, the
// caller should complete all of the work on its own.
TEST(ThreadPoolTest, TransformRangeConcurrentlyWithBusyPool) {
  ThreadPool thread_pool(Env::Default(), "test", 2);
  absl::Notification release;
  for (int i = 0; i < thread_pool.NumThreads(); ++i) {
    thread_pool.Schedule([&] { release.WaitForNotification(); });
  }
  std::atomic<int64_t> count = 0;
  thread_pool.TransformRangeConcurrently(
      /*block_size=*/10, 100, [&](int64_t first, int64_t last) {
        count.fetch_add(last - first, std::memory_order_relaxed);
      });
  EXPECT_EQ(count.load(), 100);
  release.Notify();
}

}  // namespace
}  // namespace tsl::thread
