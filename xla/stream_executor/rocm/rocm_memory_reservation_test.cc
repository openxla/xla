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

#include "xla/stream_executor/rocm/rocm_memory_reservation.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/rocm/rocm_raw_memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {

using absl_testing::IsOk;
using absl_testing::StatusIs;

static constexpr uint64_t kTestSize = 1024 * 1024;

class FakeAllocation : public MemoryAllocation {
 public:
  DeviceAddressBase address() const override { return DeviceAddressBase(); }
};

class RocmMemoryReservationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto platform_or = PlatformManager::PlatformWithName("ROCM");
    if (!platform_or.ok()) {
      GTEST_SKIP() << "ROCM platform not available";
    }
    auto executor_or = platform_or.value()->ExecutorForDevice(0);
    if (!executor_or.ok()) {
      GTEST_SKIP() << "ROCM executor not available: " << executor_or.status();
    }
    executor_ = executor_or.value();
  }

  StreamExecutor* executor_ = nullptr;
};

TEST_F(RocmMemoryReservationTest, CreateReservation) {
  TF_ASSERT_OK_AND_ASSIGN(auto res,
                          RocmMemoryReservation::Create(executor_, kTestSize));

  EXPECT_NE(res->address().opaque(), nullptr);
  EXPECT_GE(res->address().size(), kTestSize);
}

TEST_F(RocmMemoryReservationTest, MapToWrongType) {
  TF_ASSERT_OK_AND_ASSIGN(auto res,
                          RocmMemoryReservation::Create(executor_, kTestSize));

  FakeAllocation fake;
  EXPECT_THAT(res->MapTo(0, 0, kTestSize, fake),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(RocmMemoryReservationTest, MapToSingleAllocation) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc, RocmRawMemoryAllocation::Create(executor_, kTestSize));
  TF_ASSERT_OK_AND_ASSIGN(auto res,
                          RocmMemoryReservation::Create(executor_, kTestSize));

  const size_t alloc_size = alloc->address().size();
  TF_ASSERT_OK_AND_ASSIGN(auto mapping, res->MapTo(0, 0, alloc_size, *alloc));

  EXPECT_EQ(mapping.mapped_address().opaque(), res->address().opaque());
  EXPECT_EQ(mapping.mapped_address().size(), alloc_size);
}

TEST_F(RocmMemoryReservationTest, ScopedMappingUnmapsOnDestruction) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc, RocmRawMemoryAllocation::Create(executor_, kTestSize));
  TF_ASSERT_OK_AND_ASSIGN(auto res,
                          RocmMemoryReservation::Create(executor_, kTestSize));

  const size_t alloc_size = alloc->address().size();
  {
    TF_ASSERT_OK_AND_ASSIGN(auto mapping, res->MapTo(0, 0, alloc_size, *alloc));
  }

  TF_ASSERT_OK_AND_ASSIGN(auto mapping2, res->MapTo(0, 0, alloc_size, *alloc));
  EXPECT_NE(mapping2.mapped_address().opaque(), nullptr);
}

TEST_F(RocmMemoryReservationTest, MapToMultipleAllocations) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc1, RocmRawMemoryAllocation::Create(executor_, kTestSize));
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc2, RocmRawMemoryAllocation::Create(executor_, kTestSize));

  const size_t size1 = alloc1->address().size();
  const size_t size2 = alloc2->address().size();

  TF_ASSERT_OK_AND_ASSIGN(
      auto res, RocmMemoryReservation::Create(executor_, size1 + size2));
  ASSERT_GE(res->address().size(), size1 + size2);

  MemoryReservation::MappingDescriptor descs[] = {
      {/*reservation_offset=*/0, /*allocation_offset=*/0, size1, alloc1.get()},
      {/*reservation_offset=*/size1, /*allocation_offset=*/0, size2,
       alloc2.get()},
  };
  TF_ASSERT_OK_AND_ASSIGN(auto mapping, res->MapTo(absl::MakeSpan(descs)));

  EXPECT_EQ(mapping.mapped_address().opaque(), res->address().opaque());
  EXPECT_EQ(mapping.mapped_address().size(), size1 + size2);
}

TEST_F(RocmMemoryReservationTest, TwoReservationsDifferentAddresses) {
  TF_ASSERT_OK_AND_ASSIGN(auto res1,
                          RocmMemoryReservation::Create(executor_, kTestSize));
  TF_ASSERT_OK_AND_ASSIGN(auto res2,
                          RocmMemoryReservation::Create(executor_, kTestSize));

  EXPECT_NE(res1->address().opaque(), res2->address().opaque());
}

}  // namespace
}  // namespace stream_executor::gpu
