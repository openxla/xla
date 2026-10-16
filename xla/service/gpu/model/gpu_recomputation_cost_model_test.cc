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

#include "xla/service/gpu/model/gpu_recomputation_cost_model.h"

#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/str_replace.h"
#include "xla/backends/gpu/transforms/recompute_fusion_side_outputs.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/decision.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"

namespace xla::gpu {
namespace {
constexpr char kLoopHlo[] = R"(
HloModule loop_side_output
producer_body {
  x = f32[256]{0} parameter(0)
  y = f32[256]{0} parameter(1)
  side = f32[256]{0} add(x, y)
  other = f32[256]{0} negate(x)
  ROOT outputs = (f32[256]{0}, f32[256]{0}) tuple(other, side)
}
consumer_body {
  saved = f32[256]{0} parameter(0)
  x = f32[256]{0} parameter(1)
  y = f32[256]{0} parameter(2)
  product = f32[256]{0} multiply(x, y)
  ROOT output = f32[256]{0} add(saved, product)
}
ENTRY main {
  x = f32[256]{0} parameter(0)
  y = f32[256]{0} parameter(1)
  producer = (f32[256]{0}, f32[256]{0}) fusion(x, y), kind=kLoop, calls=producer_body
  other = f32[256]{0} get-tuple-element(producer), index=0
  saved = f32[256]{0} get-tuple-element(producer), index=1
  consumer = f32[256]{0} fusion(saved, x, y), kind=kLoop, calls=consumer_body
  ROOT result = (f32[256]{0}, f32[256]{0}) tuple(other, consumer)
}
)";

using RecomputationEstimateTest = HloHardwareIndependentTestBase;
TEST_F(RecomputationEstimateTest, AccountsForDtypeBytesWithoutMergingKernels) {
  for (const char* dtype : {"f16", "bf16", "f32"}) {
    SCOPED_TRACE(dtype);
    ASSERT_OK_AND_ASSIGN(auto module,
                         ParseAndReturnVerifiedModule(
                             absl::StrReplaceAll(kLoopHlo, {{"f32", dtype}})));
    int evaluated = 0;
    RecomputeFusionSideOutputs pass(
        [&](const FusionRecomputeVariants& v) {
          ++evaluated;
          auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
          auto before = EstimateRecomputationRegion(*v.before, device);
          auto after = EstimateRecomputationRegion(*v.after, device);
          EXPECT_OK(before.status());
          EXPECT_OK(after.status());
          if (before.ok() && after.ok()) {
            EXPECT_EQ(before->kernels, 2);
            EXPECT_EQ(after->kernels, 2);
            EXPECT_EQ(before->bytes_written - after->bytes_written,
                      v.eliminated_bytes);
            EXPECT_GT(before->bytes_read, after->bytes_read);
            EXPECT_GT(before->duration_ns, 0);
            EXPECT_GT(after->duration_ns, 0);
          }
          return Decision::Forbid("estimate alone must not approve");
        },
        0);
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
    EXPECT_FALSE(changed);
    EXPECT_EQ(evaluated, 1);
  }
}

TEST(RecomputationMeasurementTest, AcceptsStableGain) {
  std::vector<RecomputationSample> samples(25, {100000, 80000, 80000, 100000});
  auto result = EvaluateRecomputationMeasurements(samples);
  EXPECT_TRUE(result.decision.IsAllowed());
  EXPECT_DOUBLE_EQ(result.gain_ns, 20000);
  EXPECT_DOUBLE_EQ(result.uncertainty_ns, 0);
}
TEST(RecomputationMeasurementTest, RejectsRegressionsAndSmallGains) {
  for (double after : {110000, 100000, 99500}) {
    std::vector<RecomputationSample> samples(25,
                                             {100000, after, after, 100000});
    EXPECT_FALSE(
        EvaluateRecomputationMeasurements(samples).decision.IsAllowed());
  }
  std::vector<RecomputationSample> samples(25, {2000, 1500, 1500, 2000});
  EXPECT_FALSE(EvaluateRecomputationMeasurements(samples).decision.IsAllowed());
}
TEST(RecomputationMeasurementTest, RejectsNoiseDespitePositiveMean) {
  std::vector<RecomputationSample> samples;
  for (int i = 0; i < 25; ++i) {
    double after = (i % 2 == 0) ? 50000 : 140000;
    samples.push_back({100000, after, after, 100000});
  }
  auto result = EvaluateRecomputationMeasurements(samples);
  EXPECT_GT(result.gain_ns, 0);
  EXPECT_FALSE(result.decision.IsAllowed());
}
TEST(RecomputationMeasurementTest, RejectsOrderBias) {
  std::vector<RecomputationSample> samples(25, {100000, 50000, 110000, 100000});
  auto result = EvaluateRecomputationMeasurements(samples);
  EXPECT_GT(result.gain_ns, 0);
  EXPECT_FALSE(result.decision.IsAllowed());
}
TEST(RecomputationMeasurementTest, RejectsInsufficientAndInvalidData) {
  std::vector<RecomputationSample> samples(24, {100000, 80000, 80000, 100000});
  EXPECT_FALSE(EvaluateRecomputationMeasurements(samples).decision.IsAllowed());
  samples.push_back(samples.front());
  for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    samples.back().before_first_ns = invalid;
    EXPECT_FALSE(
        EvaluateRecomputationMeasurements(samples).decision.IsAllowed());
  }
}
}  // namespace
}  // namespace xla::gpu
