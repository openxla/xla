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

#include "xla/backends/gpu/autotuner/fusion_recomputation.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/backends/gpu/transforms/recompute_fusion_side_outputs.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"

namespace xla::gpu {
namespace {
using FusionRecomputationTest = HloPjRtGpuTestBase;
TEST_F(FusionRecomputationTest, MeasuresNativeRegionAndChecksOutputs) {
  constexpr char kHlo[] = R"(
HloModule region
producer_body {
  x = f32[4096]{0} parameter(0)
  y = f32[4096]{0} exponential(x)
  z = f32[4096]{0} negate(x)
  ROOT outputs = (f32[4096]{0}, f32[4096]{0}) tuple(y, z)
}
consumer_body {
  y = f32[4096]{0} parameter(0)
  x = f32[4096]{0} parameter(1)
  ROOT result = f32[4096]{0} multiply(y, x)
}
ENTRY main {
  x = f32[4096]{0} parameter(0)
  p = (f32[4096]{0}, f32[4096]{0}) fusion(x), kind=kLoop, calls=producer_body
  y = f32[4096]{0} get-tuple-element(p), index=0
  z = f32[4096]{0} get-tuple-element(p), index=1
  c = f32[4096]{0} fusion(y, x), kind=kLoop, calls=consumer_body
  ROOT result = (f32[4096]{0}, f32[4096]{0}) tuple(c, z)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(auto* platform, se::PlatformManager::PlatformWithId(
                                           stream_executor_platform_id()));
  ASSERT_OK_AND_ASSIGN(auto* executor, platform->ExecutorForDevice(0));
  auto evaluate = MakeFusionRecomputationEvaluator(
      compiler(), gpu_target_config(), executor);
  int evaluated = 0;
  RecomputeFusionSideOutputs pass(
      [&](const FusionRecomputeVariants& variants) {
        ++evaluated;
        auto decision = evaluate(variants);
        EXPECT_OK(decision.status());
        // Exercise the compilation-local cache on the exact same pair of
        // graphs.
        auto cached = evaluate(variants);
        EXPECT_OK(cached.status());
        if (decision.ok() && cached.ok()) {
          EXPECT_EQ(decision->IsAllowed(), cached->IsAllowed());
        }
        return decision;
      },
      /*min_bytes=*/0, /*max_candidates=*/1, /*analyze_only=*/true);
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(evaluated, 1);
  // Never assert a particular performance decision on shared CI hardware.
}
}  // namespace
}  // namespace xla::gpu
