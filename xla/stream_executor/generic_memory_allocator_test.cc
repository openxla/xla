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

#include "xla/stream_executor/generic_memory_allocator.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace {

TEST(GenericMemoryAllocatorTest, AllocateReturnsCorrectMemoryAllocation) {
  char array[64];
  bool deleter_called = false;
  auto allocator = GenericMemoryAllocator(
      [&array, &deleter_called](
          uint64_t size) -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
        EXPECT_EQ(size, 64);
        return std::make_unique<GenericMemoryAllocation>(
            array, ARRAYSIZE(array),
            [&deleter_called](void*, uint64_t) { deleter_called = true; });
      });
  TF_ASSERT_OK_AND_ASSIGN(auto allocation, allocator.Allocate(64));
  EXPECT_FALSE(deleter_called);
  allocation.reset();
  EXPECT_TRUE(deleter_called);
}

TEST(GenericMemoryAllocatorTest, AllocateReturnsError) {
  auto allocator = GenericMemoryAllocator(
      [](uint64_t size) -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
        return absl::InternalError("Failed to allocate memory");
      });
  EXPECT_THAT(allocator.Allocate(64), testing::Not(testing::status::IsOk()));
}

}  // namespace
}  // namespace stream_executor
