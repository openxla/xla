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

#include "xla/backends/cpu/runtime/ynnpack/ynn_memory_pool.h"

#include <cstddef>
#include <cstdint>
#include <thread>  // NOLINT
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"

namespace xla::cpu {
namespace {

TEST(YnnMemoryPoolTest, ReusesBuffersAcrossInvocations) {
  YnnMemoryPool pool;
  void* first = YnnMemoryPool::Allocate(&pool, 4096, 64);
  ASSERT_NE(first, nullptr);
  YnnMemoryPool::Free(&pool, first, 4096);
  for (int i = 0; i < 3; ++i) {
    void* ptr = YnnMemoryPool::Allocate(&pool, 4096, 64);
    EXPECT_EQ(ptr, first);
    YnnMemoryPool::Free(&pool, ptr, 4096);
  }
  EXPECT_EQ(pool.retained_bytes(), 4096);
}

TEST(YnnMemoryPoolTest, ReturnsReusedBuffersWithTheirOwnSize) {
  YnnMemoryPool pool;
  void* ptr = YnnMemoryPool::Allocate(&pool, 4096, 64);
  YnnMemoryPool::Free(&pool, ptr, 4096);
  // A smaller request reuses the buffer, which keeps its size in the pool.
  void* reused = YnnMemoryPool::Allocate(&pool, 3000, 64);
  EXPECT_EQ(reused, ptr);
  YnnMemoryPool::Free(&pool, reused, 3000);
  EXPECT_EQ(pool.retained_bytes(), 4096);
}

TEST(YnnMemoryPoolTest, ReusesBuffersAcrossThreads) {
  YnnMemoryPool pool;
  void* other = nullptr;
  std::thread([&] {
    other = YnnMemoryPool::Allocate(&pool, 4096, 64);
    YnnMemoryPool::Free(&pool, other, 4096);
  }).join();
  void* ptr = YnnMemoryPool::Allocate(&pool, 4096, 64);
  EXPECT_EQ(ptr, other);
  YnnMemoryPool::Free(&pool, ptr, 4096);
  EXPECT_EQ(pool.retained_bytes(), 4096);
}

TEST(YnnMemoryPoolTest, RespectsAlignment) {
  YnnMemoryPool pool;
  void* ptr = YnnMemoryPool::Allocate(&pool, 1024, 4096);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 4096, 0);
  YnnMemoryPool::Free(&pool, ptr, 1024);
}

TEST(YnnMemoryPoolTest, IsThreadSafe) {
  YnnMemoryPool pool;
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&pool, t] {
      for (int i = 0; i < 1000; ++i) {
        size_t size = 1024 * (t + 1);
        void* ptr = YnnMemoryPool::Allocate(&pool, size, 64);
        CHECK_NE(ptr, nullptr);
        YnnMemoryPool::Free(&pool, ptr, size);
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
}

}  // namespace
}  // namespace xla::cpu
