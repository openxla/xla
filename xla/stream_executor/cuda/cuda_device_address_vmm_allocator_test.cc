/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_device_address_vmm_allocator.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/globals.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/cuda/cuda_memory_reservation.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_address_vmm_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace {

using ::absl_testing::IsOk;

// 1 MB, rounded up to the VMM allocation granularity by CUDA VMM.
static constexpr uint64_t kVmmTestSize = 1024 * 1024;

class DeviceAddressVmmAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto platform_or = PlatformManager::PlatformWithName("CUDA");
    if (!platform_or.ok()) {
      GTEST_SKIP() << "CUDA platform not available";
    }
    platform_ = platform_or.value();

    auto executor_or = platform_->ExecutorForDevice(0);
    if (!executor_or.ok()) {
      GTEST_SKIP() << "CUDA executor not available";
    }
    executor_ = executor_or.value();

    auto stream_or = executor_->CreateStream();
    if (!stream_or.ok()) {
      GTEST_SKIP() << "Failed to create stream";
    }
    stream_ = std::move(stream_or.value());

    // Probe for cuStreamWriteValue64 support (requires CC >= 7.0).
    auto probe =
        gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get());
    if (absl::IsUnimplemented(probe.status())) {
      GTEST_SKIP() << "Device does not support cuStreamWriteValue64 "
                      "(requires compute capability >= 7.0): "
                   << probe.status();
    }
  }

  uint64_t GetVmmGranularity() {
    auto probe_or =
        gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get());
    if (!probe_or.ok()) {
      ADD_FAILURE() << "Failed to create VMM allocator: " << probe_or.status();
      return 0;
    }
    uint64_t granularity = (*probe_or)->GetAllocationGranularity(executor_);
    EXPECT_GT(granularity, 0);
    return granularity;
  }

  Platform* platform_ = nullptr;
  StreamExecutor* executor_ = nullptr;
  std::unique_ptr<Stream> stream_;
};

absl::StatusOr<ScopedDeviceAddress<uint8_t>> AllocateExternalMappedForTest(
    DeviceAddressVmmAllocator& allocator, int ordinal, uint64_t allocation_size,
    MemoryReservation* reservation, uint64_t reservation_offset,
    uint64_t mapping_size) {
  return allocator.Allocate(ordinal, allocation_size,
                            /*retry_on_failure=*/true,
                            static_cast<int64_t>(MemorySpace::kCollective),
                            reservation, reservation_offset, mapping_size,
                            /*return_reservation_address=*/true);
}

absl::StatusOr<ScopedDeviceAddress<uint8_t>> AllocateInternalMappedForTest(
    DeviceAddressVmmAllocator& allocator, int ordinal, uint64_t allocation_size,
    MemoryReservation* reservation, uint64_t reservation_offset,
    uint64_t mapping_size) {
  return allocator.Allocate(ordinal, allocation_size,
                            /*retry_on_failure=*/true,
                            static_cast<int64_t>(MemorySpace::kCollective),
                            reservation, reservation_offset, mapping_size,
                            /*return_reservation_address=*/false);
}

class CountingCudaDeviceAddressVmmAllocator
    : public gpu::CudaDeviceAddressVmmAllocator {
 public:
  static absl::StatusOr<std::unique_ptr<CountingCudaDeviceAddressVmmAllocator>>
  Create(StreamExecutor* executor, Stream* stream,
         uint64_t pa_budget = UINT64_MAX) {
    auto allocator = std::unique_ptr<CountingCudaDeviceAddressVmmAllocator>(
        new CountingCudaDeviceAddressVmmAllocator(executor->GetPlatform()));
    std::vector<DeviceConfig> configs = {{executor, stream, pa_budget}};
    absl::Status status = PopulateDevices(allocator.get(), configs);
    if (!status.ok()) {
      return status;
    }
    return allocator;
  }

  int64_t timeline_write_count() const { return timeline_write_count_; }

 protected:
  absl::Status EnqueueDeferredDeallocation(PerDeviceState& state,
                                           uint64_t seqno) override {
    ++timeline_write_count_;
    return gpu::CudaDeviceAddressVmmAllocator::EnqueueDeferredDeallocation(
        state, seqno);
  }

 private:
  explicit CountingCudaDeviceAddressVmmAllocator(const Platform* platform)
      : gpu::CudaDeviceAddressVmmAllocator(platform) {}

  int64_t timeline_write_count_ = 0;
};

TEST_F(DeviceAddressVmmAllocatorTest, AllocateAndDeallocate) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Allocate memory.
  ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator->Allocate(executor_->device_ordinal(), 1024,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  EXPECT_FALSE(scoped_address.is_null());
  EXPECT_EQ(scoped_address->size(), 1024);
  EXPECT_NE(
      allocator->GetRawAllocation(executor_->device_ordinal(), *scoped_address),
      nullptr);

  // The ScopedDeviceAddress will automatically deallocate when it goes out of
  // scope.
}


TEST_F(DeviceAddressVmmAllocatorTest, AllocateZeroSize) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Allocate zero-size memory.
  ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator->Allocate(executor_->device_ordinal(), 0,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  // Zero-size allocation should return a null address.
  EXPECT_TRUE(scoped_address.is_null());
}


TEST_F(DeviceAddressVmmAllocatorTest, AllocateMultiple) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Allocate multiple memory regions.
  ASSERT_OK_AND_ASSIGN(
      auto addr1,
      allocator->Allocate(executor_->device_ordinal(), 1024,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  ASSERT_OK_AND_ASSIGN(
      auto addr2,
      allocator->Allocate(executor_->device_ordinal(), 2048,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  // Both allocations should be valid and distinct.
  EXPECT_FALSE(addr1.is_null());
  EXPECT_FALSE(addr2.is_null());
  EXPECT_NE(addr1->opaque(), addr2->opaque());
  EXPECT_EQ(addr1.cref().size(), 1024);
  EXPECT_EQ(addr2.cref().size(), 2048);
}


TEST_F(DeviceAddressVmmAllocatorTest, MemoryReadWrite) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Allocate memory.
  ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator->Allocate(executor_->device_ordinal(), 1024,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  ASSERT_NE(scoped_address->opaque(), nullptr);

  // Create a stream for memory operations.
  ASSERT_OK_AND_ASSIGN(auto stream, executor_->CreateStream());

  // Write data to the allocated memory.
  constexpr uint64_t kTestValue = 0xDEADBEEFCAFEBABE;
  DeviceAddressBase addr = scoped_address.cref();
  EXPECT_THAT(stream->Memcpy(&addr, &kTestValue, sizeof(kTestValue)),
              absl_testing::IsOk());
  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());

  // Read data back.
  uint64_t read_value = 0;
  EXPECT_THAT(stream->Memcpy(&read_value, addr, sizeof(read_value)),
              absl_testing::IsOk());
  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());

  EXPECT_EQ(read_value, kTestValue);
}


TEST_F(DeviceAddressVmmAllocatorTest, GetStream) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Get the stream - should return the same stream that was provided at
  // construction.
  ASSERT_OK_AND_ASSIGN(Stream * stream,
                       allocator->GetStream(executor_->device_ordinal()));
  EXPECT_EQ(stream, stream_.get());

  // Getting the stream again should return the same pointer.
  ASSERT_OK_AND_ASSIGN(Stream * stream2,
                       allocator->GetStream(executor_->device_ordinal()));
  EXPECT_EQ(stream, stream2);
}


TEST_F(DeviceAddressVmmAllocatorTest, GetStreamExecutor) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  ASSERT_OK_AND_ASSIGN(StreamExecutor * se, allocator->GetStreamExecutor(
                                                executor_->device_ordinal()));
  EXPECT_EQ(se, executor_);
}


TEST_F(DeviceAddressVmmAllocatorTest, AllowsAsynchronousDeallocation) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Virtual address allocator supports asynchronous deallocation via
  // GPU timeline-based processing.
  EXPECT_TRUE(allocator->AllowsAsynchronousDeallocation());
}


TEST_F(DeviceAddressVmmAllocatorTest, ExplicitDeallocate) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Allocate memory.
  ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator->Allocate(executor_->device_ordinal(), 1024,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  ASSERT_NE(scoped_address->opaque(), nullptr);
  DeviceAddressBase addr = scoped_address.cref();

  // Explicitly deallocate.
  EXPECT_THAT(allocator->Deallocate(executor_->device_ordinal(), addr),
              absl_testing::IsOk());

  // Release ownership to prevent double-free.
  scoped_address.Release();
}


TEST_F(DeviceAddressVmmAllocatorTest, DeallocateNull) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Deallocating null address should succeed.
  DeviceAddressBase null_addr;
  EXPECT_THAT(allocator->Deallocate(executor_->device_ordinal(), null_addr),
              absl_testing::IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       PendingDeallocationReusesSameVirtualAddress) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  constexpr uint64_t kSize = 1024;

  ASSERT_OK_AND_ASSIGN(
      auto addr1,
      allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  void* const va = addr1->opaque();

  // Deallocate: timeline write is enqueued but VA is not freed yet.
  DeviceAddressBase raw = addr1.cref();
  addr1.Release();
  ASSERT_THAT(allocator->Deallocate(ordinal, raw), IsOk());

  // Allocate the same size — TryReusePendingDeallocation should match the
  // pending entry and return the identical virtual address.
  ASSERT_OK_AND_ASSIGN(
      auto addr2,
      allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  EXPECT_EQ(addr2->opaque(), va);

  // Sync to drain all pending GPU timeline writes before the allocator
  // is destroyed.
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       AllocateExternalMappedAndDeallocateTrackExternalReservation) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  DeviceAddressBase external =
      reservation->address().GetByteSlice(/*offset_bytes=*/0, granularity);

  ASSERT_OK_AND_ASSIGN(auto mapped,
                       AllocateExternalMappedForTest(
                           *allocator, ordinal, granularity, reservation.get(),
                           /*reservation_offset=*/0, granularity));
  MemoryAllocation* raw = allocator->GetRawAllocation(ordinal, external);
  ASSERT_NE(raw, nullptr);

  ASSERT_THAT(allocator->Deallocate(ordinal, mapped.Release()), IsOk());

  // Mapping the same external VA after Deallocate() should reuse the
  // pending mapping without creating a new physical allocation.
  ASSERT_OK_AND_ASSIGN(auto remapped,
                       AllocateExternalMappedForTest(
                           *allocator, ordinal, granularity, reservation.get(),
                           /*reservation_offset=*/0, granularity));
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, external), raw);

  ASSERT_THAT(allocator->Deallocate(ordinal, remapped.Release()), IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       MappedAllocateRejectsAllocationAndMappingSizeMismatch) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  EXPECT_FALSE(AllocateExternalMappedForTest(
                   *allocator, ordinal, granularity, reservation.get(),
                   /*reservation_offset=*/0, granularity / 2)
                   .ok());
  EXPECT_FALSE(AllocateInternalMappedForTest(
                   *allocator, ordinal, granularity, reservation.get(),
                   /*reservation_offset=*/0, granularity / 2)
                   .ok());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       InternalMappedAllocateRequiresUnMapBeforeDeallocate) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  DeviceAddressBase external =
      reservation->address().GetByteSlice(/*offset_bytes=*/0, granularity);

  ASSERT_OK_AND_ASSIGN(auto addr,
                       AllocateInternalMappedForTest(
                           *allocator, ordinal, granularity, reservation.get(),
                           /*reservation_offset=*/0, granularity));
  DeviceAddressBase internal = addr.cref();

  MemoryAllocation* internal_raw =
      allocator->GetRawAllocation(ordinal, internal);
  ASSERT_NE(internal_raw, nullptr);
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, external), internal_raw);

  EXPECT_FALSE(allocator->Deallocate(ordinal, external).ok());

  EXPECT_THAT(allocator->Deallocate(ordinal, addr.cref()),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                     ::testing::HasSubstr("UnMap() first")));
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, internal), internal_raw);
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, external), internal_raw);

  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, internal), nullptr);
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, external), nullptr);
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(
    DeviceAddressVmmAllocatorTest,
    InternalMappedAllocateUnMapDeallocateCanReuseAllocatorAndReservationRecord) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  DeviceAddressBase external =
      reservation->address().GetByteSlice(/*offset_bytes=*/0, granularity);
  ASSERT_OK_AND_ASSIGN(auto addr,
                       AllocateInternalMappedForTest(
                           *allocator, ordinal, granularity, reservation.get(),
                           /*reservation_offset=*/0, granularity));
  void* const allocator_va = addr->opaque();
  MemoryAllocation* raw = allocator->GetRawAllocation(ordinal, addr.cref());
  ASSERT_NE(raw, nullptr);
  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());

  ASSERT_OK_AND_ASSIGN(auto reused,
                       AllocateInternalMappedForTest(
                           *allocator, ordinal, granularity, reservation.get(),
                           /*reservation_offset=*/0, granularity));
  EXPECT_EQ(reused->opaque(), allocator_va);
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, reused.cref()), raw);
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, external), raw);

  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, reused.Release()), IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       InternalMappedAllocateExternalRangeCanBeReusedAfterUnMap) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  DeviceAddressBase external =
      reservation->address().GetByteSlice(/*offset_bytes=*/0, granularity);
  ASSERT_OK_AND_ASSIGN(auto mapped,
                       AllocateInternalMappedForTest(
                           *allocator, ordinal, granularity, reservation.get(),
                           /*reservation_offset=*/0, granularity));
  DeviceAddressBase internal = mapped.cref();
  MemoryAllocation* raw = allocator->GetRawAllocation(ordinal, internal);
  ASSERT_NE(raw, nullptr);

  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());

  ASSERT_OK_AND_ASSIGN(auto remapped,
                       AllocateExternalMappedForTest(
                           *allocator, ordinal, granularity, reservation.get(),
                           /*reservation_offset=*/0, granularity));
  EXPECT_NE(allocator->GetRawAllocation(ordinal, external), nullptr);

  ASSERT_THAT(allocator->Deallocate(ordinal, remapped.Release()), IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, mapped.Release()), IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       DeallocateRejectsInternalMappedExternalAddress) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  DeviceAddressBase external =
      reservation->address().GetByteSlice(/*offset_bytes=*/0, granularity);
  ASSERT_OK_AND_ASSIGN(auto mapped,
                       AllocateInternalMappedForTest(
                           *allocator, ordinal, granularity, reservation.get(),
                           /*reservation_offset=*/0, granularity));
  DeviceAddressBase internal = mapped.cref();
  MemoryAllocation* raw = allocator->GetRawAllocation(ordinal, internal);
  ASSERT_NE(raw, nullptr);

  EXPECT_FALSE(allocator->Deallocate(ordinal, external).ok());

  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, mapped.Release()), IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest, MapTracksRawAllocation) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  ASSERT_OK_AND_ASSIGN(
      auto addr,
      allocator->Allocate(ordinal, kVmmTestSize,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  MemoryAllocation* raw = allocator->GetRawAllocation(ordinal, addr.cref());
  ASSERT_NE(raw, nullptr);

  DeviceAddressBase external =
      reservation->address().GetByteSlice(/*offset_bytes=*/0, granularity);

  ASSERT_THAT(allocator->Map(ordinal, addr.cref(), reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, external), raw);

  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest, UnMapWithZeroSizeMappingIsNoOp) {
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, kVmmTestSize));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  ASSERT_OK_AND_ASSIGN(
      auto addr,
      allocator->Allocate(ordinal, kVmmTestSize,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  ASSERT_THAT(allocator->Map(ordinal, addr.cref(), reservation.get(),
                             /*reservation_offset=*/0, /*size=*/0),
              IsOk());

  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, /*size=*/0),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       MapRejectsSecondActiveMappingForSameAllocatorAddress) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  ASSERT_OK_AND_ASSIGN(auto reservation1, gpu::CudaMemoryReservation::Create(
                                              executor_, granularity));
  ASSERT_OK_AND_ASSIGN(auto reservation2, gpu::CudaMemoryReservation::Create(
                                              executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  ASSERT_OK_AND_ASSIGN(
      auto addr,
      allocator->Allocate(ordinal, kVmmTestSize,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  MemoryAllocation* raw = allocator->GetRawAllocation(ordinal, addr.cref());
  ASSERT_NE(raw, nullptr);

  DeviceAddressBase target1 =
      reservation1->address().GetByteSlice(/*offset_bytes=*/0, granularity);
  DeviceAddressBase target2 =
      reservation2->address().GetByteSlice(/*offset_bytes=*/0, granularity);
  ASSERT_THAT(allocator->Map(ordinal, addr.cref(), reservation1.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  EXPECT_FALSE(allocator
                   ->Map(ordinal, addr.cref(), reservation2.get(),
                         /*reservation_offset=*/0, granularity)
                   .ok());

  EXPECT_NE(target1.opaque(), target2.opaque());
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, target1), raw);
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, target2), nullptr);

  ASSERT_THAT(allocator->UnMap(ordinal, reservation1.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, target1), nullptr);
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, target2), nullptr);
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       UnMapRejectsPartialRangeOfActiveReservationMapping) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  const uint64_t mapping_size = granularity * 2;
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, mapping_size));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  ASSERT_OK_AND_ASSIGN(
      auto addr,
      allocator->Allocate(ordinal, mapping_size,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  ASSERT_THAT(allocator->Map(ordinal, addr.cref(), reservation.get(),
                             /*reservation_offset=*/0, mapping_size),
              IsOk());

  EXPECT_FALSE(allocator
                   ->UnMap(ordinal, reservation.get(),
                           /*reservation_offset=*/granularity, granularity)
                   .ok());

  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, mapping_size),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       MapRejectsPartialOverlapWithActiveReservationMapping) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  const uint64_t mapping_size = granularity * 2;
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, mapping_size));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  ASSERT_OK_AND_ASSIGN(
      auto addr1,
      allocator->Allocate(ordinal, mapping_size,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  ASSERT_OK_AND_ASSIGN(
      auto addr2,
      allocator->Allocate(ordinal, granularity,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  ASSERT_THAT(allocator->Map(ordinal, addr1.cref(), reservation.get(),
                             /*reservation_offset=*/0, mapping_size),
              IsOk());

  EXPECT_FALSE(allocator
                   ->Map(ordinal, addr2.cref(), reservation.get(),
                         /*reservation_offset=*/granularity, granularity)
                   .ok());

  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, mapping_size),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, addr1.Release()), IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, addr2.Release()), IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       MapReactivatesPendingUnMapForSameRawAndRange) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  ASSERT_OK_AND_ASSIGN(
      auto addr,
      allocator->Allocate(ordinal, kVmmTestSize,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  MemoryAllocation* raw = allocator->GetRawAllocation(ordinal, addr.cref());
  ASSERT_NE(raw, nullptr);

  DeviceAddressBase external =
      reservation->address().GetByteSlice(/*offset_bytes=*/0, granularity);

  ASSERT_THAT(allocator->Map(ordinal, addr.cref(), reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());

  ASSERT_THAT(allocator->Map(ordinal, addr.cref(), reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, external), raw);

  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest, DeallocateRejectsActiveMapMapping) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  ASSERT_OK_AND_ASSIGN(
      auto addr,
      allocator->Allocate(ordinal, kVmmTestSize,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  DeviceAddressBase target =
      reservation->address().GetByteSlice(/*offset_bytes=*/0, granularity);
  MemoryAllocation* raw = allocator->GetRawAllocation(ordinal, addr.cref());
  ASSERT_NE(raw, nullptr);
  ASSERT_THAT(allocator->Map(ordinal, addr.cref(), reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, target), raw);

  EXPECT_THAT(allocator->Deallocate(ordinal, addr.cref()),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                     ::testing::HasSubstr("UnMap() first")));
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, target), raw);

  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, target), nullptr);
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       ReusingRegularAllocatorAddressPreservesExplicitUnMapPending) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  ASSERT_OK_AND_ASSIGN(auto reservation1, gpu::CudaMemoryReservation::Create(
                                              executor_, granularity));
  ASSERT_OK_AND_ASSIGN(auto reservation2, gpu::CudaMemoryReservation::Create(
                                              executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  ASSERT_OK_AND_ASSIGN(
      auto addr,
      allocator->Allocate(ordinal, kVmmTestSize,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  void* const allocator_va = addr->opaque();
  ASSERT_THAT(allocator->Map(ordinal, addr.cref(), reservation1.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->UnMap(ordinal, reservation1.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());

  ASSERT_OK_AND_ASSIGN(
      auto reused,
      allocator->Allocate(ordinal, kVmmTestSize, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  EXPECT_EQ(reused->opaque(), allocator_va);

  ASSERT_THAT(allocator->Map(ordinal, reused.cref(), reservation2.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->UnMap(ordinal, reservation2.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, reused.Release()), IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       ReusingReservationBackedAllocatorAddressPreservesExplicitUnMapPending) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  ASSERT_OK_AND_ASSIGN(
      auto source_reservation,
      gpu::CudaMemoryReservation::Create(executor_, granularity));
  ASSERT_OK_AND_ASSIGN(auto reservation1, gpu::CudaMemoryReservation::Create(
                                              executor_, granularity));
  ASSERT_OK_AND_ASSIGN(auto reservation2, gpu::CudaMemoryReservation::Create(
                                              executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  ASSERT_OK_AND_ASSIGN(
      auto addr, AllocateExternalMappedForTest(
                     *allocator, ordinal, granularity, source_reservation.get(),
                     /*reservation_offset=*/0, granularity));
  void* const allocator_va = addr->opaque();
  ASSERT_THAT(allocator->Map(ordinal, addr.cref(), reservation1.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->UnMap(ordinal, reservation1.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());

  ASSERT_OK_AND_ASSIGN(auto reused, AllocateExternalMappedForTest(
                                        *allocator, ordinal, granularity,
                                        source_reservation.get(),
                                        /*reservation_offset=*/0, granularity));
  EXPECT_EQ(reused->opaque(), allocator_va);

  ASSERT_THAT(allocator->Map(ordinal, reused.cref(), reservation2.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->UnMap(ordinal, reservation2.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, reused.Release()), IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       PendingDeallocationWithPendingMapUnMapCanBeReused) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  ASSERT_OK_AND_ASSIGN(
      auto addr,
      allocator->Allocate(ordinal, kVmmTestSize,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  void* const old_va = addr->opaque();
  ASSERT_THAT(allocator->Map(ordinal, addr.cref(), reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->UnMap(ordinal, reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());

  ASSERT_OK_AND_ASSIGN(
      auto fresh,
      allocator->Allocate(ordinal, kVmmTestSize, /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  EXPECT_EQ(fresh->opaque(), old_va);

  ASSERT_THAT(allocator->Deallocate(ordinal, fresh.Release()), IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest, MapRejectsNullAddress) {
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, kVmmTestSize));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));
  EXPECT_FALSE(allocator
                   ->Map(executor_->device_ordinal(), DeviceAddressBase(),
                         reservation.get(), /*reservation_offset=*/0,
                         kVmmTestSize)
                   .ok());
}


TEST_F(DeviceAddressVmmAllocatorTest, MapRejectsAddressFromDifferentAllocator) {
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, kVmmTestSize));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));
  ASSERT_OK_AND_ASSIGN(
      auto other_allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  ASSERT_OK_AND_ASSIGN(auto other_addr,
                       other_allocator->Allocate(
                           ordinal, kVmmTestSize,
                           /*retry_on_failure=*/true,
                           static_cast<int64_t>(MemorySpace::kCollective)));
  EXPECT_FALSE(allocator
                   ->Map(ordinal, other_addr.cref(), reservation.get(),
                         /*reservation_offset=*/0, kVmmTestSize)
                   .ok());

  ASSERT_THAT(other_allocator->Deallocate(ordinal, other_addr.Release()),
              IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       MapAcceptsReservationBackedAllocatorAddress) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  ASSERT_OK_AND_ASSIGN(
      auto source_reservation,
      gpu::CudaMemoryReservation::Create(executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto target_reservation,
      gpu::CudaMemoryReservation::Create(executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  ASSERT_OK_AND_ASSIGN(auto mapped, AllocateExternalMappedForTest(
                                        *allocator, ordinal, granularity,
                                        source_reservation.get(),
                                        /*reservation_offset=*/0, granularity));
  MemoryAllocation* raw = allocator->GetRawAllocation(ordinal, mapped.cref());
  ASSERT_NE(raw, nullptr);
  DeviceAddressBase target = target_reservation->address().GetByteSlice(
      /*offset_bytes=*/0, granularity);

  ASSERT_THAT(allocator->Map(ordinal, mapped.cref(), target_reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, target), raw);

  EXPECT_FALSE(allocator
                   ->UnMap(ordinal, source_reservation.get(),
                           /*reservation_offset=*/0, granularity)
                   .ok());
  ASSERT_THAT(allocator->UnMap(ordinal, target_reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, mapped.Release()), IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       MapAcceptsAllocatorAddressFromMappedAllocate) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  ASSERT_OK_AND_ASSIGN(
      auto source_reservation,
      gpu::CudaMemoryReservation::Create(executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto target_reservation,
      gpu::CudaMemoryReservation::Create(executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  DeviceAddressBase reservation_address =
      source_reservation->address().GetByteSlice(/*offset_bytes=*/0,
                                                 granularity);
  ASSERT_OK_AND_ASSIGN(auto mapped, AllocateInternalMappedForTest(
                                        *allocator, ordinal, granularity,
                                        source_reservation.get(),
                                        /*reservation_offset=*/0, granularity));
  MemoryAllocation* raw = allocator->GetRawAllocation(ordinal, mapped.cref());
  ASSERT_NE(raw, nullptr);
  DeviceAddressBase target = target_reservation->address().GetByteSlice(
      /*offset_bytes=*/0, granularity);

  EXPECT_FALSE(allocator
                   ->Map(ordinal, reservation_address, target_reservation.get(),
                         /*reservation_offset=*/0, granularity)
                   .ok());
  EXPECT_FALSE(allocator
                   ->Map(ordinal, mapped.cref(), target_reservation.get(),
                         /*reservation_offset=*/0, granularity)
                   .ok());
  ASSERT_THAT(allocator->UnMap(ordinal, source_reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Map(ordinal, mapped.cref(), target_reservation.get(),
                             /*reservation_offset=*/0, granularity),
              IsOk());
  EXPECT_EQ(allocator->GetRawAllocation(ordinal, target), raw);
  ASSERT_THAT(allocator->UnMap(ordinal, target_reservation.get(),
                               /*reservation_offset=*/0, granularity),
              IsOk());
  ASSERT_THAT(allocator->Deallocate(ordinal, mapped.Release()), IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       MapRejectsSizeLargerThanSourceAllocation) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, granularity * 2));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  ASSERT_OK_AND_ASSIGN(
      auto addr,
      allocator->Allocate(ordinal, granularity,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  EXPECT_FALSE(allocator
                   ->Map(ordinal, addr.cref(), reservation.get(),
                         /*reservation_offset=*/0, granularity + 1)
                   .ok());

  ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest, MapRejectsDeallocatedAddress) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  ASSERT_OK_AND_ASSIGN(
      auto addr,
      allocator->Allocate(ordinal, granularity,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  DeviceAddressBase released = addr.Release();
  ASSERT_THAT(allocator->Deallocate(ordinal, released), IsOk());

  EXPECT_FALSE(allocator
                   ->Map(ordinal, released, reservation.get(),
                         /*reservation_offset=*/0, granularity)
                   .ok());

  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest, UnMapRejectsUntrackedMapping) {
  const uint64_t granularity = GetVmmGranularity();
  ASSERT_GT(granularity, 0);
  ASSERT_OK_AND_ASSIGN(auto reservation, gpu::CudaMemoryReservation::Create(
                                             executor_, granularity));
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  ASSERT_OK_AND_ASSIGN(
      auto addr,
      allocator->Allocate(ordinal, kVmmTestSize,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  MemoryAllocation* raw = allocator->GetRawAllocation(ordinal, addr.cref());
  ASSERT_NE(raw, nullptr);

  ASSERT_OK_AND_ASSIGN(
      auto mapping,
      reservation->MapTo(/*reservation_offset=*/0,
                         /*allocation_offset=*/0, granularity, *raw));

  EXPECT_FALSE(allocator
                   ->UnMap(ordinal, reservation.get(),
                           /*reservation_offset=*/0, granularity)
                   .ok());
  mapping = MemoryReservation::ScopedMapping();
  ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       DeferredDeallocationSafeWhileGpuWritesData) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();

  ASSERT_OK_AND_ASSIGN(
      auto addr,
      allocator->Allocate(ordinal, sizeof(uint64_t), /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));

  // Enqueue a memcpy to the allocated buffer on stream_.
  constexpr uint64_t kPattern = 0xCAFEBABEDEADBEEFULL;
  DeviceAddressBase dev_addr = addr.cref();
  ASSERT_THAT(stream_->Memcpy(&dev_addr, &kPattern, sizeof(kPattern)), IsOk());

  // Deallocate while the memcpy is still queued. The seqno timeline write is
  // appended to the stream AFTER the memcpy, so the VA cannot be reused until
  // the GPU advances past it.
  ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());

  // Sync: both the memcpy and the timeline write execute in order.
  // No crash here means the physical memory was not freed prematurely.
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       MultipleSeqnosAllCompleteAfterStreamSync) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  constexpr int kCount = 8;
  constexpr uint64_t kSize = 1024;

  // Allocate kCount buffers and immediately queue their deallocation.
  // Each Deallocate increments next_seqno and enqueues a timeline write.
  for (int i = 0; i < kCount; ++i) {
    ASSERT_OK_AND_ASSIGN(
        auto addr,
        allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                            static_cast<int64_t>(MemorySpace::kCollective)));
    ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());
  }

  // Sync the stream: all kCount timeline writes (seqnos 1..kCount) complete.
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());

  // Each new Allocate call finds a matching pending entry via
  // TryReusePendingDeallocation (or via ProcessCompletedPendingDeallocations
  // once the pending queue is exhausted).
  for (int i = 0; i < kCount; ++i) {
    ASSERT_OK_AND_ASSIGN(
        auto addr,
        allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                            static_cast<int64_t>(MemorySpace::kCollective)));
    EXPECT_FALSE(addr.is_null());
  }

  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}


TEST_F(DeviceAddressVmmAllocatorTest,
       DestructorWithPendingDeallocationsDoesNotCrash) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();

  // Queue several deallocations without syncing the stream first.
  for (int i = 0; i < 4; ++i) {
    ASSERT_OK_AND_ASSIGN(
        auto addr,
        allocator->Allocate(ordinal, 1024, /*retry_on_failure=*/true,
                            static_cast<int64_t>(MemorySpace::kCollective)));
    ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());
  }

  // Destroy without an explicit stream sync. The destructor must spin on the
  // pinned_timeline until the GPU writes all pending seqnos, then free
  // each virtual address safely.
  allocator.reset();  // Must not crash or leak.
}


TEST_F(DeviceAddressVmmAllocatorTest, UnknownDeviceOrdinalReturnsError) {
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int unknown_ordinal = 9999;
  EXPECT_FALSE(allocator
                   ->Allocate(unknown_ordinal, 1024, /*retry_on_failure=*/true,
                              static_cast<int64_t>(MemorySpace::kCollective))
                   .ok());
  // Null Deallocate always succeeds (early return before ordinal lookup).
  EXPECT_THAT(allocator->Deallocate(unknown_ordinal, DeviceAddressBase{}),
              absl_testing::IsOk());
  // Non-null address on unknown ordinal returns error.
  DeviceAddressBase fake_addr(reinterpret_cast<void*>(0x1000), 64);
  EXPECT_FALSE(allocator->Deallocate(unknown_ordinal, fake_addr).ok());
  EXPECT_FALSE(allocator->GetStream(unknown_ordinal).ok());
  EXPECT_FALSE(allocator->GetStreamExecutor(unknown_ordinal).ok());
}


// Multi-device fixture: skips the test if fewer than two CUDA devices are
// available.
class MultiDeviceVmmAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto platform_or = PlatformManager::PlatformWithName("CUDA");
    if (!platform_or.ok()) {
      GTEST_SKIP() << "CUDA platform not available";
    }
    platform_ = platform_or.value();

    if (platform_->VisibleDeviceCount() < 2) {
      GTEST_SKIP() << "Fewer than two CUDA devices available";
    }

    for (int i = 0; i < 2; ++i) {
      auto executor_or = platform_->ExecutorForDevice(i);
      if (!executor_or.ok()) {
        GTEST_SKIP() << "CUDA executor not available for device " << i;
      }
      executors_.push_back(executor_or.value());

      auto stream_or = executors_.back()->CreateStream();
      if (!stream_or.ok()) {
        GTEST_SKIP() << "Failed to create stream for device " << i;
      }
      streams_.push_back(std::move(stream_or.value()));
    }

    // Probe for cuStreamWriteValue64 support using device 0 (requires CC
    // >= 7.0).
    auto probe = gpu::CudaDeviceAddressVmmAllocator::Create(executors_[0],
                                                            streams_[0].get());
    if (absl::IsUnimplemented(probe.status())) {
      GTEST_SKIP() << "Device does not support cuStreamWriteValue64 "
                      "(requires compute capability >= 7.0): "
                   << probe.status();
    }
  }

  Platform* platform_ = nullptr;
  std::vector<StreamExecutor*> executors_;
  std::vector<std::unique_ptr<Stream>> streams_;
};

TEST_F(MultiDeviceVmmAllocatorTest, AllocateOnBothDevices) {
  std::vector<DeviceAddressVmmAllocator::DeviceConfig> configs;
  for (int i = 0; i < 2; ++i) {
    configs.push_back({executors_[i], streams_[i].get()});
  }
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(platform_, configs));

  for (int i = 0; i < 2; ++i) {
    ASSERT_OK_AND_ASSIGN(
        auto addr,
        allocator->Allocate(executors_[i]->device_ordinal(), 1024,
                            /*retry_on_failure=*/true,
                            static_cast<int64_t>(MemorySpace::kCollective)));
    EXPECT_FALSE(addr.is_null());
    EXPECT_EQ(addr->size(), 1024);
    EXPECT_NE(
        allocator->GetRawAllocation(executors_[i]->device_ordinal(), *addr),
        nullptr);
    ASSERT_OK_AND_ASSIGN(
        StreamExecutor * se,
        allocator->GetStreamExecutor(executors_[i]->device_ordinal()));
    EXPECT_EQ(se, executors_[i]);
    ASSERT_OK_AND_ASSIGN(Stream * stream,
                         allocator->GetStream(executors_[i]->device_ordinal()));
    EXPECT_EQ(stream, streams_[i].get());
  }
}


TEST_F(MultiDeviceVmmAllocatorTest, AllocationOnOneDeviceDoesNotAffectOther) {
  std::vector<DeviceAddressVmmAllocator::DeviceConfig> configs;
  for (int i = 0; i < 2; ++i) {
    configs.push_back({executors_[i], streams_[i].get()});
  }
  ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(platform_, configs));

  // Allocate on device 0.
  ASSERT_OK_AND_ASSIGN(
      auto addr0,
      allocator->Allocate(executors_[0]->device_ordinal(), 4096,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  EXPECT_FALSE(addr0.is_null());

  // GetRawAllocation for addr0's pointer on device 1 should return nullptr
  // (different per-device map).
  EXPECT_EQ(
      allocator->GetRawAllocation(executors_[1]->device_ordinal(), *addr0),
      nullptr);
}



}  // namespace
}  // namespace stream_executor
