/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/model/hlo_op_profiler.h"

#include <unordered_set>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

class HloOpProfilerTest : public HloTestBase {
 protected:
  void SetUp() override {
    platform_id_ = HloTestBase::GetTestPlatform()->id();
    if (platform_id_ != se::cuda::kCudaPlatformId &&
        platform_id_ != se::rocm::kROCmPlatformId) {
      GTEST_SKIP() << "Not built with --config=cuda or --config=rocm";
    }
  }

  const int kMinClockCyclesAddF32_ = 0;
  const int kMinClockCyclesDivideF64_ =
      platform_id_ == se::cuda::kCudaPlatformId ? 280 : 100;
  const int kMinClockCyclesSqrtC128_ = 1000;
  se::Platform::Id platform_id_;
};

TEST_F(HloOpProfilerTest, BasicMeasurementsAreCorrect) {
  HloOpProfiler profiler(test_runner_as_hlo_runner());
  // f32 is fast but measurable.
  EXPECT_GT(profiler.MeasureClockCyclesPerOp(HloOpcode::kAdd, F32)
                .value()
                .clock_cycles(),
            kMinClockCyclesAddF32_);
  // f64 divide is somewhat slow.
  EXPECT_GT(profiler.MeasureClockCyclesPerOp(HloOpcode::kDivide, F64)
                .value()
                .clock_cycles(),
            kMinClockCyclesDivideF64_);
  // c128 sqrt is slow.
  EXPECT_GT(profiler.MeasureClockCyclesPerOp(HloOpcode::kSqrt, C128)
                .value()
                .clock_cycles(),
            kMinClockCyclesSqrtC128_);
}

TEST_F(HloOpProfilerTest, UnsupportedCombinationsDoNotCrash) {
  HloOpProfiler profiler(test_runner_as_hlo_runner());
  EXPECT_THAT(profiler.MeasureClockCyclesPerOp(HloOpcode::kCbrt, S8),
              absl_testing::StatusIs(tsl::error::INVALID_ARGUMENT));
}

TEST_F(HloOpProfilerTest, AllSupportedCombinationsAreMeasurable) {
  std::unordered_set<HloOpcode> FloatTypes = {
      // go/keep-sorted start
      HloOpcode::kAtan2,
      HloOpcode::kCbrt,
      HloOpcode::kCeil,
      HloOpcode::kCos,
      HloOpcode::kErf,
      HloOpcode::kExp,
      HloOpcode::kExpm1,
      HloOpcode::kFloor,
      HloOpcode::kImag,
      HloOpcode::kIsFinite,
      HloOpcode::kLog,
      HloOpcode::kLog1p,
      HloOpcode::kLogistic,
      HloOpcode::kReal,
      HloOpcode::kRoundNearestAfz,
      HloOpcode::kRoundNearestEven,
      HloOpcode::kRsqrt,
      HloOpcode::kSin,
      HloOpcode::kSqrt,
      HloOpcode::kTan,
      HloOpcode::kTanh
      // go/keep-sorted end
  };
  std::unordered_set<HloOpcode> MeasurebleInFloat = {
      // go/keep-sorted start
      HloOpcode::kAdd,
      HloOpcode::kMultiply,
      HloOpcode::kSubtract,
      // go/keep-sorted end
  };

  // TODO(esjoblom): These ops fail with too fast to measure on ROCm
  // but let's just skip them for now.
  const std::unordered_set<HloOpcode> skip_on_rocm = {
      HloOpcode::kPopulationCount,
      HloOpcode::kRoundNearestAfz,
      HloOpcode::kRoundNearestEven,
  };

  FloatTypes.insert(MeasurebleInFloat.begin(), MeasurebleInFloat.end());
  HloOpProfiler profiler(test_runner_as_hlo_runner());

  const bool is_rocm = platform_id_ == se::rocm::kROCmPlatformId;
  for (const HloOpcode op : HloOpProfiler::AllSupportedOps()) {
    if (!HloOpProfiler::TooFastToMeasure().count(op) &&
        !HloOpProfiler::Unsupported().count(op) &&
        !(is_rocm && skip_on_rocm.count(op))) {
      auto Type = FloatTypes.count(op) ? F32 : S32;
      TF_EXPECT_OK(profiler.MeasureClockCyclesPerOp(op, Type));
    }
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla
