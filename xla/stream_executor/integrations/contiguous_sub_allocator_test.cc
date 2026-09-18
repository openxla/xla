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

#include "xla/stream_executor/integrations/contiguous_sub_allocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"

namespace stream_executor {
namespace {

constexpr uintptr_t kBase = uintptr_t{1} << 40;
constexpr size_t kPage = 2 << 20;

struct Events {
  std::vector<std::pair<size_t, size_t>> maps;
  std::vector<std::pair<size_t, size_t>> unmaps;
  int live_allocations = 0;
  bool fail_allocation = false;
  bool fail_map = false;
  bool fail_access = false;
};

class FakeAllocation : public MemoryAllocation {
 public:
  FakeAllocation(Events* events, size_t bytes)
      : events_(events), bytes_(bytes) {
    ++events_->live_allocations;
  }
  ~FakeAllocation() override { --events_->live_allocations; }
  DeviceAddressBase address() const override {
    return DeviceAddressBase(nullptr, bytes_);
  }

 private:
  Events* events_;
  size_t bytes_;
};

class FakeReservation : public MemoryReservation {
 public:
  explicit FakeReservation(Events* events) : events_(events) {}
  DeviceAddressBase address() const override {
    return DeviceAddressBase(reinterpret_cast<void*>(kBase), 4 * kPage);
  }

 private:
  absl::Status Map(size_t offset, size_t allocation_offset, size_t bytes,
                   MemoryAllocation& allocation) override {
    if (events_->fail_map) return absl::ResourceExhaustedError("map failed");
    events_->maps.emplace_back(offset, bytes);
    return absl::OkStatus();
  }
  absl::Status SetAccess(uint64_t offset, size_t bytes) override {
    return events_->fail_access ? absl::InternalError("access failed")
                                : absl::OkStatus();
  }
  absl::Status UnMap(size_t offset, size_t bytes) override {
    events_->unmaps.emplace_back(offset, bytes);
    return absl::OkStatus();
  }
  Events* events_;
};

ContiguousSubAllocator::Allocate MakeAllocate(Events* events) {
  return
      [events](
          size_t bytes) -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
        if (events->fail_allocation)
          return absl::ResourceExhaustedError("allocation failed");
        return std::make_unique<FakeAllocation>(events, bytes);
      };
}

TEST(ContiguousSubAllocatorTest, AppendsWithoutTouchingExistingMappings) {
  Events events;
  std::vector<std::pair<void*, size_t>> allocations, frees;
  {
    ContiguousSubAllocator allocator(
        std::make_unique<FakeReservation>(&events), MakeAllocate(&events),
        kPage, 7, {[&](void* ptr, int device, size_t bytes) {
          EXPECT_EQ(device, 7);
          allocations.emplace_back(ptr, bytes);
        }},
        {[&](void* ptr, int device, size_t bytes) {
          EXPECT_EQ(device, 7);
          // The free callback runs while the memory is still mapped.
          EXPECT_EQ(events.unmaps.size(), frees.size());
          frees.emplace_back(ptr, bytes);
        }});
    size_t received;
    void* first = allocator.Alloc(256, 1, &received);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(first), kBase);
    EXPECT_EQ(received, kPage);
    void* second = allocator.Alloc(256, kPage + 1, &received);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(second), kBase + kPage);
    EXPECT_EQ(received, 2 * kPage);
    EXPECT_EQ(events.live_allocations, 2);
    EXPECT_TRUE(events.unmaps.empty());
    EXPECT_EQ(events.maps, (std::vector<std::pair<size_t, size_t>>{
                               {0, kPage}, {kPage, 2 * kPage}}));
    EXPECT_EQ(allocator.Alloc(256, 2 * kPage, &received), nullptr);
    EXPECT_EQ(received, 0);
    EXPECT_EQ(events.maps.size(), 2);
    allocator.Free(first, 3 * kPage);
    EXPECT_EQ(events.live_allocations, 0);
    ASSERT_EQ(frees.size(), 2);
    EXPECT_EQ(frees[0], allocations[1]);
    EXPECT_EQ(frees[1], allocations[0]);
  }
  EXPECT_EQ(events.unmaps.size(), 2);
}

TEST(ContiguousSubAllocatorTest, FailedExtensionRollsBackOnlyNewMemory) {
  for (int failure = 0; failure < 3; ++failure) {
    Events events;
    ContiguousSubAllocator allocator(std::make_unique<FakeReservation>(&events),
                                     MakeAllocate(&events), kPage, 0, {}, {});
    size_t received;
    ASSERT_NE(allocator.Alloc(256, kPage, &received), nullptr);
    events.fail_allocation = failure == 0;
    events.fail_map = failure == 1;
    events.fail_access = failure == 2;
    EXPECT_EQ(allocator.Alloc(256, kPage, &received), nullptr);
    EXPECT_EQ(received, 0);
    EXPECT_EQ(events.live_allocations, 1);
    for (auto [offset, bytes] : events.unmaps) EXPECT_GE(offset, kPage);
    events.fail_allocation = events.fail_map = events.fail_access = false;
    EXPECT_EQ(
        reinterpret_cast<uintptr_t>(allocator.Alloc(256, kPage, &received)),
        kBase + kPage);
    EXPECT_EQ(events.live_allocations, 2);
  }
}

TEST(ContiguousSubAllocatorTest, CanReleaseAndRetryLastMapping) {
  Events events;
  ContiguousSubAllocator allocator(std::make_unique<FakeReservation>(&events),
                                   MakeAllocate(&events), kPage, 0, {}, {});
  size_t received;
  ASSERT_NE(allocator.Alloc(256, kPage, &received), nullptr);
  void* second = allocator.Alloc(256, kPage, &received);
  ASSERT_NE(second, nullptr);
  allocator.Free(second, kPage);
  EXPECT_EQ(allocator.Alloc(256, kPage, &received), second);
  EXPECT_EQ(events.live_allocations, 2);
  ASSERT_EQ(events.unmaps.size(), 1);
  EXPECT_EQ(events.unmaps[0].first, kPage);
}

}  // namespace
}  // namespace stream_executor
