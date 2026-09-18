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

#include "xla/stream_executor/integrations/device_mem_allocator.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/tsl/framework/device_id.h"

namespace stream_executor {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Return;

constexpr uintptr_t kBase = 0x10000000;
void* Address(size_t offset) { return reinterpret_cast<void*>(kBase + offset); }

// Independent bookkeeping models physical handles and active mappings. In
// particular, releasing backing that is still mapped is an error in this fake.
struct MemoryState {
  absl::Status reserve_status;
  absl::Status allocate_status;
  absl::Status map_status;
  absl::Status access_status;
  size_t reserved_bytes = 0;
  size_t physical_live = 0;
  size_t next_handle = 0;
  bool reservation_live = false;
  std::map<size_t, size_t> mappings;  // offset -> physical handle
  std::vector<size_t> allocation_requests;
  std::vector<size_t> mapped_offsets;
  std::vector<size_t> unmapped_offsets;
};

class FakeReservation : public MemoryReservation {
 public:
  FakeReservation(MemoryState& state, size_t capacity)
      : state_(state), capacity_(capacity) {
    state_.reservation_live = true;
  }
  ~FakeReservation() override {
    EXPECT_THAT(state_.mappings, IsEmpty());
    EXPECT_EQ(state_.physical_live, 0);
    state_.reservation_live = false;
  }
  DeviceAddressBase address() const override {
    return DeviceAddressBase(Address(0), capacity_);
  }
  size_t granularity() const override { return 256; }

 private:
  absl::Status Map(size_t offset, size_t allocation_offset, size_t size,
                   MemoryAllocation& allocation) override {
    EXPECT_EQ(allocation_offset, 0);
    EXPECT_EQ(size, allocation.address().size());
    EXPECT_LE(offset + size, capacity_);
    if (!state_.map_status.ok()) return state_.map_status;
    EXPECT_TRUE(state_.mappings
                    .emplace(offset, reinterpret_cast<uintptr_t>(
                                         allocation.address().opaque()))
                    .second);
    state_.mapped_offsets.push_back(offset);
    return absl::OkStatus();
  }
  absl::Status SetAccess(uint64_t offset, size_t size) override {
    return state_.access_status;
  }
  absl::Status UnMap(size_t offset, size_t size) override {
    EXPECT_EQ(state_.mappings.erase(offset), 1);
    state_.unmapped_offsets.push_back(offset);
    return absl::OkStatus();
  }
  MemoryState& state_;
  size_t capacity_;
};

class ReservationExecutor : public MockStreamExecutor {
 public:
  MemoryState state;
  absl::StatusOr<std::unique_ptr<MemoryReservation>> CreateMemoryReservation(
      uint64_t size) override {
    if (!state.reserve_status.ok()) return state.reserve_status;
    state.reserved_bytes = size;
    return std::make_unique<FakeReservation>(state, size);
  }
  absl::StatusOr<std::unique_ptr<MemoryAllocation>>
  CreatePhysicalMemoryAllocation(uint64_t size) override {
    state.allocation_requests.push_back(size);
    if (!state.allocate_status.ok()) return state.allocate_status;
    ++state.physical_live;
    return std::make_unique<GenericMemoryAllocation>(
        reinterpret_cast<void*>(++state.next_handle), (size + 255) / 256 * 256,
        [this](void* handle, uint64_t size) {
          for (const auto& [offset, mapped_handle] : state.mappings) {
            EXPECT_NE(mapped_handle, reinterpret_cast<uintptr_t>(handle));
          }
          --state.physical_live;
        });
  }
};

TEST(DeviceMemAllocatorTest, ReservesOnlyVaAndAppendsPhysicalBacking) {
  ReservationExecutor executor;
  std::vector<std::pair<void*, size_t>> allocated, freed;
  {
    DeviceMemAllocator allocator(
        &executor, tsl::PlatformDeviceId(0), {[&](void* ptr, int, size_t size) {
          allocated.emplace_back(ptr, size);
        }},
        {[&](void* ptr, int, size_t size) { freed.emplace_back(ptr, size); }});
    EXPECT_FALSE(allocator.SupportsCoalescing());
    ASSERT_OK(allocator.ReserveMemory(16384));
    EXPECT_TRUE(allocator.SupportsCoalescing());
    EXPECT_EQ(allocator.GetAllocationGranularity(), 256);
    EXPECT_EQ(executor.state.reserved_bytes, 16384);
    EXPECT_THAT(executor.state.allocation_requests, IsEmpty());
    EXPECT_THAT(executor.state.mappings, IsEmpty());

    size_t received = 0;
    EXPECT_EQ(allocator.Alloc(256, 4096, &received), Address(0));
    EXPECT_EQ(received, 4096);
    EXPECT_EQ(allocator.Alloc(256, 2048, &received), Address(4096));
    EXPECT_EQ(received, 2048);
    EXPECT_THAT(executor.state.mapped_offsets, ElementsAre(0, 4096));
    EXPECT_EQ(executor.state.physical_live, 2);
    // BFC coalesces these allocations and returns them as one region.
    allocator.Free(Address(0), 6144);
    EXPECT_EQ(executor.state.physical_live, 0);
    EXPECT_THAT(executor.state.unmapped_offsets, ElementsAre(4096, 0));
    ASSERT_EQ(allocated.size(), 2);
    EXPECT_THAT(freed, ElementsAre(allocated[1], allocated[0]));
  }
  EXPECT_FALSE(executor.state.reservation_live);
}

enum class Failure { kPhysical, kMap, kAccess };
class DeviceMemAllocatorFailureTest : public ::testing::TestWithParam<Failure> {
};

TEST_P(DeviceMemAllocatorFailureTest,
       FailedExtensionPreservesPrefixAndCanRetry) {
  ReservationExecutor executor;
  DeviceMemAllocator allocator(&executor, tsl::PlatformDeviceId(0));
  ASSERT_OK(allocator.ReserveMemory(16384));
  size_t received = 0;
  ASSERT_EQ(allocator.Alloc(256, 4096, &received), Address(0));
  const size_t original_handle = executor.state.mappings.at(0);
  absl::Status* failure_status = nullptr;
  switch (GetParam()) {
    case Failure::kPhysical:
      failure_status = &executor.state.allocate_status;
      break;
    case Failure::kMap:
      failure_status = &executor.state.map_status;
      break;
    case Failure::kAccess:
      failure_status = &executor.state.access_status;
      break;
  }
  *failure_status = absl::ResourceExhaustedError("injected growth failure");
  EXPECT_EQ(allocator.Alloc(256, 2048, &received), nullptr);
  EXPECT_EQ(received, 0);
  ASSERT_EQ(executor.state.mappings.size(), 1);
  EXPECT_EQ(executor.state.mappings.at(0), original_handle);
  EXPECT_EQ(executor.state.physical_live, 1);
  if (GetParam() == Failure::kAccess) {
    EXPECT_THAT(executor.state.unmapped_offsets, ElementsAre(4096));
  } else {
    EXPECT_THAT(executor.state.unmapped_offsets, IsEmpty());
  }
  *failure_status = absl::OkStatus();
  EXPECT_EQ(allocator.Alloc(256, 2048, &received), Address(4096));
  EXPECT_EQ(received, 2048);
  EXPECT_EQ(executor.state.mappings.at(0), original_handle);
  // Destructor also handles a partially backed reservation.
}

INSTANTIATE_TEST_SUITE_P(Growth, DeviceMemAllocatorFailureTest,
                         ::testing::Values(Failure::kPhysical, Failure::kMap,
                                           Failure::kAccess));

TEST(DeviceMemAllocatorTest, BackendPaddingCannotExceedCapacity) {
  ReservationExecutor executor;
  DeviceMemAllocator allocator(&executor, tsl::PlatformDeviceId(0));
  ASSERT_OK(allocator.ReserveMemory(1025));
  size_t received = 0;
  ASSERT_EQ(allocator.Alloc(256, 512, &received), Address(0));
  // 513 rounds to 768, which cannot fit in the remaining 513 bytes.
  EXPECT_EQ(allocator.Alloc(256, 513, &received), nullptr);
  EXPECT_EQ(received, 0);
  EXPECT_EQ(executor.state.physical_live, 1);
  EXPECT_THAT(executor.state.mapped_offsets, ElementsAre(0));
  EXPECT_EQ(allocator.Alloc(256, 512, &received), Address(512));
  EXPECT_EQ(allocator.Alloc(256, 1, &received), nullptr);
  EXPECT_EQ(received, 0);
  EXPECT_EQ(allocator.Alloc(256, 0, &received), nullptr);
  EXPECT_EQ(received, 0);
}

TEST(DeviceMemAllocatorTest, ReturnedSuffixCanBeMappedAgain) {
  ReservationExecutor executor;
  DeviceMemAllocator allocator(&executor, tsl::PlatformDeviceId(0));
  ASSERT_OK(allocator.ReserveMemory(1024));
  size_t received = 0;
  ASSERT_EQ(allocator.Alloc(256, 512, &received), Address(0));
  ASSERT_EQ(allocator.Alloc(256, 256, &received), Address(512));
  allocator.Free(Address(512), 256);
  EXPECT_THAT(executor.state.unmapped_offsets, ElementsAre(512));
  EXPECT_EQ(allocator.Alloc(256, 512, &received), Address(512));
  EXPECT_EQ(received, 512);
  EXPECT_EQ(executor.state.physical_live, 2);
}

TEST(DeviceMemAllocatorTest, ReservationErrorsAreReported) {
  ReservationExecutor executor;
  DeviceMemAllocator allocator(&executor, tsl::PlatformDeviceId(0));
  EXPECT_THAT(allocator.ReserveMemory(0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  executor.state.reserve_status = absl::ResourceExhaustedError("VA exhausted");
  EXPECT_THAT(allocator.ReserveMemory(1024),
              StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_FALSE(allocator.SupportsCoalescing());
  executor.state.reserve_status = absl::OkStatus();
  ASSERT_OK(allocator.ReserveMemory(1024));
  EXPECT_THAT(allocator.ReserveMemory(2048),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(DeviceMemAllocatorTest, LegacyModeUsesExecutorAllocation) {
  MockStreamExecutor executor;
  DeviceMemAllocator allocator(&executor, tsl::PlatformDeviceId(0));
  EXPECT_THAT(allocator.ReserveMemory(1024),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_FALSE(allocator.SupportsCoalescing());
  EXPECT_CALL(executor, Allocate(256, 0))
      .WillOnce(Return(DeviceAddressBase(Address(0), 512)));
  EXPECT_CALL(executor, Deallocate(::testing::_)).Times(1);
  size_t received = 0;
  EXPECT_EQ(allocator.Alloc(256, 256, &received), Address(0));
  EXPECT_EQ(received, 512);
  EXPECT_THAT(allocator.ReserveMemory(1024),
              StatusIs(absl::StatusCode::kInvalidArgument));
  allocator.Free(Address(0), received);
}

}  // namespace
}  // namespace stream_executor
