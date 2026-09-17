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

#include "xla/stream_executor/cuda/cuda_contiguous_sub_allocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/integrations/contiguous_sub_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/framework/bfc_allocator.h"

namespace stream_executor::gpu {
namespace {

class CudaContiguousSubAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(
        auto* platform, PlatformManager::PlatformWithId(cuda::kCudaPlatformId));
    ASSERT_OK_AND_ASSIGN(executor_, platform->ExecutorForDevice(0));
  }
  StreamExecutor* executor_ = nullptr;
};

TEST_F(CudaContiguousSubAllocatorTest, GrowthPreservesLiveDeviceBuffers) {
  ASSERT_OK_AND_ASSIGN(
      auto sub, CreateCudaContiguousSubAllocator(executor_, 128 << 20, 0));
  const size_t page = sub->granularity();
  ASSERT_LE(8 * page, sub->capacity());
  tsl::BFCAllocator::Options opts;
  opts.enable_spatial_partitioning = true;
  opts.allow_growth = true;
  opts.allow_retry_on_failure = false;
  opts.initial_region_bytes = 4 * page;
  opts.growth_increment_bytes = 2 * page;
  opts.lower_end_policy = {tsl::BFCAllocator::HoleOrder::kAscendingAddress,
                           tsl::BFCAllocator::SplitPolicy::kExact,
                           tsl::BFCAllocator::SplitPolicy::kExact};
  opts.upper_end_policy = {tsl::BFCAllocator::HoleOrder::kDescendingAddress,
                           tsl::BFCAllocator::SplitPolicy::kBfc,
                           tsl::BFCAllocator::SplitPolicy::kBfc};
  tsl::BFCAllocator allocator(std::move(sub), 8 * page, "cuda_growth", opts);
  const tsl::AllocationAttributes lower(false, false, nullptr,
                                        tsl::AllocationEnd::kLower);
  const tsl::AllocationAttributes upper(false, false, nullptr,
                                        tsl::AllocationEnd::kUpper);
  void* collective = allocator.AllocateRaw(page, 2 * page, lower);
  void* ordinary = allocator.AllocateRaw(256, 2 * page, upper);
  ASSERT_NE(collective, nullptr);
  ASSERT_NE(ordinary, nullptr);
  auto activation = executor_->Activate();
  const uint32_t collective_value = 0x12345678;
  const uint32_t ordinary_value = 0xabcdef12;
  ASSERT_EQ(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(collective),
                         &collective_value, sizeof(uint32_t)),
            CUDA_SUCCESS);
  ASSERT_EQ(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(ordinary),
                         &ordinary_value, sizeof(uint32_t)),
            CUDA_SUCCESS);

  void* extension = allocator.AllocateRaw(256, 2 * page, upper);
  ASSERT_NE(extension, nullptr);
  EXPECT_GE(reinterpret_cast<uintptr_t>(extension),
            reinterpret_cast<uintptr_t>(collective) + 4 * page);
  EXPECT_EQ(allocator.GetStats()->pool_bytes, 6 * page);
  ASSERT_EQ(
      cuMemsetD8(reinterpret_cast<CUdeviceptr>(extension), 0xa5, 2 * page),
      CUDA_SUCCESS);
  uint32_t result = 0;
  ASSERT_EQ(cuMemcpyDtoH(&result, reinterpret_cast<CUdeviceptr>(collective),
                         sizeof(result)),
            CUDA_SUCCESS);
  EXPECT_EQ(result, collective_value);
  ASSERT_EQ(cuMemcpyDtoH(&result, reinterpret_cast<CUdeviceptr>(ordinary),
                         sizeof(result)),
            CUDA_SUCCESS);
  EXPECT_EQ(result, ordinary_value);
  ASSERT_EQ(cuMemcpyDtoH(&result, reinterpret_cast<CUdeviceptr>(extension),
                         sizeof(result)),
            CUDA_SUCCESS);
  EXPECT_EQ(result, 0xa5a5a5a5);
  allocator.DeallocateRaw(extension);
  EXPECT_EQ(allocator.AllocateRaw(page, page, lower), nullptr);
  allocator.DeallocateRaw(ordinary);
  allocator.DeallocateRaw(collective);
  EXPECT_EQ(allocator.AllocateRaw(page, 5 * page, lower), nullptr);
}

TEST_F(CudaContiguousSubAllocatorTest, PreservesExternalHeadroom) {
  int64_t free_bytes, total_bytes;
  ASSERT_TRUE(executor_->DeviceMemoryUsage(&free_bytes, &total_bytes));
  ASSERT_OK_AND_ASSIGN(auto sub, CreateCudaContiguousSubAllocator(
                                     executor_, 128 << 20, total_bytes));
  size_t received;
  EXPECT_EQ(sub->Alloc(256, sub->granularity(), &received), nullptr);
  EXPECT_EQ(received, 0);
}

}  // namespace
}  // namespace stream_executor::gpu
