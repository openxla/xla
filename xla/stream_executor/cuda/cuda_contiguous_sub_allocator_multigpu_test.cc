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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_contiguous_sub_allocator.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/integrations/contiguous_sub_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/framework/bfc_allocator.h"

// Include NCCL after XLA headers.
#include "third_party/nccl/nccl.h"

namespace stream_executor::gpu {
namespace {

// Registration is collective, whereas extension allocation is rank-local.
class CudaBfcSymmetricGrowthTest : public ::testing::Test {
 protected:
  template <typename F>
  void OnBothRanks(F function) {
    std::thread first([&] { function(0); });
    std::thread second([&] { function(1); });
    first.join();
    second.join();
  }

  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(
        auto* platform, PlatformManager::PlatformWithId(cuda::kCudaPlatformId));
    if (platform->VisibleDeviceCount() < 2)
      GTEST_SKIP() << "Requires two CUDA devices";
    for (int rank = 0; rank < 2; ++rank) {
      ASSERT_OK_AND_ASSIGN(executors_[rank], platform->ExecutorForDevice(rank));
      ASSERT_OK_AND_ASSIGN(auto sub, CreateCudaContiguousSubAllocator(
                                         executors_[rank], 128 << 20, 0));
      if (rank == 0) page_ = sub->granularity();
      ASSERT_EQ(sub->granularity(), page_);
      ASSERT_LE(16 * page_, sub->capacity());
      tsl::BFCAllocator::Options opts;
      opts.allow_growth = true;
      opts.allow_retry_on_failure = false;
      opts.enable_spatial_partitioning = true;
      opts.initial_region_bytes = 4 * page_;
      opts.growth_increment_bytes = 2 * page_;
      opts.lower_end_policy = {tsl::BFCAllocator::HoleOrder::kAscendingAddress,
                               tsl::BFCAllocator::SplitPolicy::kExact,
                               tsl::BFCAllocator::SplitPolicy::kExact};
      opts.upper_end_policy = {tsl::BFCAllocator::HoleOrder::kDescendingAddress,
                               tsl::BFCAllocator::SplitPolicy::kBfc,
                               tsl::BFCAllocator::SplitPolicy::kBfc};
      allocators_[rank] = std::make_unique<tsl::BFCAllocator>(
          std::move(sub), 16 * page_, "symmetric_growth", opts);
      buffers_[rank][0] = allocators_[rank]->AllocateRaw(page_, page_, lower_);
      ASSERT_NE(buffers_[rank][0], nullptr);
      ASSERT_NE(allocators_[rank]->AllocateRaw(256, page_, upper_), nullptr);
      auto activation = executors_[rank]->Activate();
      ASSERT_EQ(cuStreamCreate(&streams_[rank], CU_STREAM_NON_BLOCKING),
                CUDA_SUCCESS);
    }
    const int devices[] = {0, 1};
    ASSERT_EQ(ncclCommInitAll(comms_.data(), 2, devices), ncclSuccess);
  }

  void TearDown() override {
    OnBothRanks([&](int rank) {
      if (!executors_[rank]) return;
      auto activation = executors_[rank]->Activate();
      for (auto window : windows_[rank]) {
        if (window)
          EXPECT_EQ(ncclCommWindowDeregister(comms_[rank], window),
                    ncclSuccess);
      }
      if (comms_[rank]) EXPECT_EQ(ncclCommDestroy(comms_[rank]), ncclSuccess);
      if (streams_[rank])
        EXPECT_EQ(cuStreamDestroy(streams_[rank]), CUDA_SUCCESS);
      allocators_[rank].reset();
    });
  }

  void Register(int index) {
    OnBothRanks([&](int rank) {
      auto activation = executors_[rank]->Activate();
      EXPECT_EQ(ncclCommWindowRegister(comms_[rank], buffers_[rank][index],
                                       page_, &windows_[rank][index],
                                       NCCL_WIN_COLL_SYMMETRIC),
                ncclSuccess);
    });
    for (int rank = 0; rank < 2; ++rank)
      ASSERT_NE(windows_[rank][index], nullptr);
  }

  void AllReduce(int index) {
    for (int rank = 0; rank < 2; ++rank) {
      auto activation = executors_[rank]->Activate();
      const float input = rank + 1;
      ASSERT_EQ(
          cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(buffers_[rank][index]),
                       &input, sizeof(input)),
          CUDA_SUCCESS);
    }
    ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    for (int rank = 0; rank < 2; ++rank) {
      auto activation = executors_[rank]->Activate();
      EXPECT_EQ(ncclAllReduce(buffers_[rank][index], buffers_[rank][index], 1,
                              ncclFloat, ncclSum, comms_[rank], streams_[rank]),
                ncclSuccess);
    }
    ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
    for (int rank = 0; rank < 2; ++rank) {
      auto activation = executors_[rank]->Activate();
      ASSERT_EQ(cuStreamSynchronize(streams_[rank]), CUDA_SUCCESS);
      float output;
      ASSERT_EQ(
          cuMemcpyDtoH(&output,
                       reinterpret_cast<CUdeviceptr>(buffers_[rank][index]),
                       sizeof(output)),
          CUDA_SUCCESS);
      EXPECT_EQ(output, 3.0f);
    }
  }

  size_t page_ = 0;
  std::array<StreamExecutor*, 2> executors_ = {};
  std::array<std::unique_ptr<tsl::BFCAllocator>, 2> allocators_;
  std::array<ncclComm_t, 2> comms_ = {};
  std::array<CUstream, 2> streams_ = {};
  std::array<std::array<void*, 2>, 2> buffers_ = {};
  std::array<std::array<ncclWindow_t, 2>, 2> windows_ = {};
  const tsl::AllocationAttributes lower_{false, false, nullptr,
                                         tsl::AllocationEnd::kLower};
  const tsl::AllocationAttributes upper_{false, false, nullptr,
                                         tsl::AllocationEnd::kUpper};
};

TEST_F(CudaBfcSymmetricGrowthTest, WindowsSurviveAsymmetricGrowth) {
  ASSERT_NO_FATAL_FAILURE(Register(0));
  ASSERT_NO_FATAL_FAILURE(AllReduce(0));
  // Only rank zero grows, while its first symmetric window remains registered.
  void* extension = allocators_[0]->AllocateRaw(256, 8 * page_, upper_);
  ASSERT_NE(extension, nullptr);
  EXPECT_GT(allocators_[0]->GetStats()->pool_bytes.value(), 4 * page_);
  EXPECT_EQ(allocators_[1]->GetStats()->pool_bytes.value(), 4 * page_);
  ASSERT_NO_FATAL_FAILURE(AllReduce(0));
  for (int rank = 0; rank < 2; ++rank) {
    buffers_[rank][1] = allocators_[rank]->AllocateRaw(page_, page_, lower_);
    ASSERT_NE(buffers_[rank][1], nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(buffers_[rank][1]) -
                  reinterpret_cast<uintptr_t>(buffers_[rank][0]),
              page_);
  }
  ASSERT_NO_FATAL_FAILURE(Register(1));
  ASSERT_NO_FATAL_FAILURE(AllReduce(1));
  allocators_[0]->DeallocateRaw(extension);
}

}  // namespace
}  // namespace stream_executor::gpu
