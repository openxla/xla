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

#include "xla/stream_executor/stream_executor_vmm_allocator.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Ne;

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
  }

  Platform* platform_ = nullptr;
  StreamExecutor* executor_ = nullptr;
  std::unique_ptr<Stream> stream_;
};

TEST_F(DeviceAddressVmmAllocatorTest, AllocateAndDeallocate) {
  DeviceAddressVmmAllocator allocator(executor_, stream_.get());

  // Allocate memory.
  TF_ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator.Allocate(executor_->device_ordinal(), 1024,
                         /*retry_on_failure=*/true,
                         static_cast<int64_t>(MemorySpace::kP2P)));

  EXPECT_FALSE(scoped_address.is_null());
  EXPECT_EQ(scoped_address->size(), 1024);
  EXPECT_TRUE(scoped_address->raw_handle().has_value());

  // The ScopedDeviceAddress will automatically deallocate when it goes out of
  // scope.
}

TEST_F(DeviceAddressVmmAllocatorTest, AllocateZeroSize) {
  DeviceAddressVmmAllocator allocator(executor_, stream_.get());

  // Allocate zero-size memory.
  TF_ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator.Allocate(executor_->device_ordinal(), 0,
                         /*retry_on_failure=*/true,
                         static_cast<int64_t>(MemorySpace::kP2P)));

  // Zero-size allocation should return a null address.
  EXPECT_TRUE(scoped_address.is_null());
}

TEST_F(DeviceAddressVmmAllocatorTest, AllocateMultiple) {
  DeviceAddressVmmAllocator allocator(executor_, stream_.get());

  // Allocate multiple memory regions.
  TF_ASSERT_OK_AND_ASSIGN(
      auto addr1,
      allocator.Allocate(executor_->device_ordinal(), 1024,
                         /*retry_on_failure=*/true,
                         static_cast<int64_t>(MemorySpace::kP2P)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto addr2,
      allocator.Allocate(executor_->device_ordinal(), 2048,
                         /*retry_on_failure=*/true,
                         static_cast<int64_t>(MemorySpace::kP2P)));

  // Both allocations should be valid and distinct.
  EXPECT_FALSE(addr1.is_null());
  EXPECT_FALSE(addr2.is_null());
  EXPECT_NE(addr1->opaque(), addr2->opaque());
  EXPECT_EQ(addr1.cref().size(), 1024);
  EXPECT_EQ(addr2.cref().size(), 2048);
}

TEST_F(DeviceAddressVmmAllocatorTest, MemoryReadWrite) {
  DeviceAddressVmmAllocator allocator(executor_, stream_.get());

  // Allocate memory.
  TF_ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator.Allocate(executor_->device_ordinal(), 1024,
                         /*retry_on_failure=*/true,
                         static_cast<int64_t>(MemorySpace::kP2P)));

  ASSERT_NE(scoped_address->opaque(), nullptr);

  // Create a stream for memory operations.
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor_->CreateStream());

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
  DeviceAddressVmmAllocator allocator(executor_, stream_.get());

  // Get the stream - should return the same stream that was provided at
  // construction.
  TF_ASSERT_OK_AND_ASSIGN(Stream * stream,
                          allocator.GetStream(executor_->device_ordinal()));
  EXPECT_EQ(stream, stream_.get());

  // Getting the stream again should return the same pointer.
  TF_ASSERT_OK_AND_ASSIGN(Stream * stream2,
                          allocator.GetStream(executor_->device_ordinal()));
  EXPECT_EQ(stream, stream2);
}

TEST_F(DeviceAddressVmmAllocatorTest, GetStreamInvalidOrdinal) {
  DeviceAddressVmmAllocator allocator(executor_, stream_.get());

  // Non-existent device ordinal should fail.
  EXPECT_THAT(allocator.GetStream(9999),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("No stream registered")));
}

TEST_F(DeviceAddressVmmAllocatorTest, GetStreamExecutor) {
  DeviceAddressVmmAllocator allocator(executor_, stream_.get());

  TF_ASSERT_OK_AND_ASSIGN(
      StreamExecutor * retrieved_executor,
      allocator.GetStreamExecutor(executor_->device_ordinal()));
  EXPECT_EQ(retrieved_executor, executor_);
}

TEST_F(DeviceAddressVmmAllocatorTest, GetStreamExecutorInvalidOrdinal) {
  DeviceAddressVmmAllocator allocator(executor_, stream_.get());

  // Negative device ordinal should fail.
  EXPECT_THAT(allocator.GetStreamExecutor(-1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("non-negative")));

  // Non-existent device ordinal should fail.
  EXPECT_THAT(allocator.GetStreamExecutor(9999),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(DeviceAddressVmmAllocatorTest, AllowsAsynchronousDeallocation) {
  DeviceAddressVmmAllocator allocator(executor_, stream_.get());

  // Virtual address allocator supports asynchronous deallocation via deferred
  // event-based processing.
  EXPECT_TRUE(allocator.AllowsAsynchronousDeallocation());
}

TEST_F(DeviceAddressVmmAllocatorTest, MultipleExecutors) {
  // Get all available executors and create streams for each.
  std::vector<DeviceAddressVmmAllocator::DeviceInfo> device_infos;
  std::vector<std::unique_ptr<Stream>> streams;
  int device_count = platform_->VisibleDeviceCount();
  for (int i = 0; i < device_count; ++i) {
    auto executor_or = platform_->ExecutorForDevice(i);
    if (executor_or.ok()) {
      StreamExecutor* executor = executor_or.value();
      auto stream_or = executor->CreateStream();
      if (stream_or.ok()) {
        streams.push_back(std::move(stream_or.value()));
        device_infos.push_back({executor, streams.back().get()});
      }
    }
  }

  if (device_infos.size() < 2) {
    GTEST_SKIP() << "Need at least 2 GPUs for multi-executor test";
  }

  // Create allocator with multiple devices.
  DeviceAddressVmmAllocator allocator(platform_, device_infos);

  // Allocate on each device.
  for (const auto& device_info : device_infos) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto scoped_address,
        allocator.Allocate(device_info.executor->device_ordinal(), 1024,
                           /*retry_on_failure=*/true,
                           static_cast<int64_t>(MemorySpace::kP2P)));

    EXPECT_NE(scoped_address->opaque(), nullptr);
    EXPECT_EQ(scoped_address.cref().size(), 1024);
  }
}

TEST_F(DeviceAddressVmmAllocatorTest, ExplicitDeallocate) {
  DeviceAddressVmmAllocator allocator(executor_, stream_.get());

  // Allocate memory.
  TF_ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator.Allocate(executor_->device_ordinal(), 1024,
                         /*retry_on_failure=*/true,
                         static_cast<int64_t>(MemorySpace::kP2P)));

  ASSERT_NE(scoped_address->opaque(), nullptr);
  DeviceAddressBase addr = scoped_address.cref();

  // Explicitly deallocate.
  EXPECT_THAT(allocator.Deallocate(executor_->device_ordinal(), addr),
              absl_testing::IsOk());

  // Release ownership to prevent double-free.
  scoped_address.Release();
}

TEST_F(DeviceAddressVmmAllocatorTest, DeallocateNull) {
  DeviceAddressVmmAllocator allocator(executor_, stream_.get());

  // Deallocating null address should succeed.
  DeviceAddressBase null_addr;
  EXPECT_THAT(allocator.Deallocate(executor_->device_ordinal(), null_addr),
              absl_testing::IsOk());
}

}  // namespace
}  // namespace stream_executor
