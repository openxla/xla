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

#include "xla/tools/cost_model/gpu_bandwidth_kernels.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Gt;

// Whether CUDA device 0 is Hopper (compute capability 9.0) or newer, i.e.
// whether it supports the TMA bulk writes the kernel is built on.
absl::StatusOr<bool> IsAtLeastHopper() {
  ABSL_ASSIGN_OR_RETURN(
      stream_executor::Platform * platform,
      stream_executor::PlatformManager::PlatformWithName("CUDA"));
  ABSL_ASSIGN_OR_RETURN(stream_executor::StreamExecutor * executor,
                        platform->ExecutorForDevice(0));
  return executor->GetDeviceDescription()
      .cuda_compute_capability()
      .IsAtLeastHopper();
}

TEST(GpuBandwidthKernelsTest, RejectsNonPositiveTransferSize) {
  EXPECT_THAT(MeasureWriteUblkcpBandwidthBytesPerSec(/*ordinal=*/0,
                                                     /*dma_size_bytes=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GpuBandwidthKernelsTest, RejectsTransferSizeNotMultipleOf16) {
  EXPECT_THAT(MeasureWriteUblkcpBandwidthBytesPerSec(/*ordinal=*/0,
                                                     /*dma_size_bytes=*/8200),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GpuBandwidthKernelsTest, RejectsUnknownDevice) {
  constexpr int kOrdinalBeyondAnyDevice = 1000;
  EXPECT_THAT(MeasureWriteUblkcpBandwidthBytesPerSec(kOrdinalBeyondAnyDevice,
                                                     /*dma_size_bytes=*/8192),
              StatusIs(absl::StatusCode::kUnavailable));
}

TEST(GpuBandwidthKernelsTest, MeasuresPositiveBandwidthOnHopper) {
  ASSERT_OK_AND_ASSIGN(bool is_at_least_hopper, IsAtLeastHopper());
  if (!is_at_least_hopper) {
    GTEST_SKIP() << "TMA bulk writes require compute capability 9.0+.";
  }
  EXPECT_THAT(MeasureWriteUblkcpBandwidthBytesPerSec(/*ordinal=*/0,
                                                     /*dma_size_bytes=*/8192),
              IsOkAndHolds(Gt(0.0)));
}

TEST(GpuBandwidthKernelsTest, RejectsPreHopperDevice) {
  ASSERT_OK_AND_ASSIGN(bool is_at_least_hopper, IsAtLeastHopper());
  if (is_at_least_hopper) {
    GTEST_SKIP() << "Device supports TMA bulk writes.";
  }
  EXPECT_THAT(MeasureWriteUblkcpBandwidthBytesPerSec(/*ordinal=*/0,
                                                     /*dma_size_bytes=*/8192),
              StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace
}  // namespace xla::gpu
