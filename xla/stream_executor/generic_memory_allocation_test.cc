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

#include "xla/stream_executor/generic_memory_allocation.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "xla/tsl/platform/macros.h"

namespace stream_executor {
namespace {

TEST(GenericMemoryAllocationTest, DeleterIsCalledWithCorrectArguments) {
  char array[64];
  bool deleter_called = false;
  auto deleter = [&array, &deleter_called](void *ptr, uint64_t size) {
    EXPECT_EQ(ptr, array);
    EXPECT_EQ(size, ARRAYSIZE(array));
    deleter_called = true;
  };
  {
    GenericMemoryAllocation allocation(array, 64, deleter);
    EXPECT_EQ(allocation.opaque(), array);
    EXPECT_EQ(allocation.size(), 64);
    EXPECT_FALSE(deleter_called);
  }
  EXPECT_TRUE(deleter_called);
}

}  // namespace
}  // namespace stream_executor
