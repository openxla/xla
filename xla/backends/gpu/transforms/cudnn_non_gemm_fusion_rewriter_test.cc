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

#include "xla/backends/gpu/transforms/cudnn_non_gemm_fusion_rewriter.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class CudnnNonGemmFusionRewriterTest : public HloPjRtGpuTestBase {
 protected:
  se::StreamExecutor* stream_executor() const {
    auto platform =
        se::PlatformManager::PlatformWithId(stream_executor_platform_id());
    CHECK_OK(platform);
    absl::StatusOr<se::StreamExecutor*> executor =
        (*platform)->ExecutorForDevice(0);
    CHECK_OK(executor);
    return *executor;
  }
};

// Baseline: a fusion containing a "well-behaved" concatenate (all operands have
// the same size along the concat dimension and all operands are used
// downstream) should be marked as a cuDNN custom fusion.
TEST_F(CudnnNonGemmFusionRewriterTest, WellFormedConcatFusionIsRewritten) {
  constexpr char kHlo[] = R"(
HloModule m

fused_computation {
  p0 = f32[4,8] parameter(0)
  p1 = f32[4,8] parameter(1)
  ROOT concat = f32[4,16] concatenate(p0, p1), dimensions={1}
}

ENTRY e {
  a = f32[4,8] parameter(0)
  b = f32[4,8] parameter(1)
  ROOT fusion = f32[4,16] fusion(a, b), kind=kLoop, calls=fused_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      CudnnNonGemmFusionRewriter(stream_executor(), device_description())
          .Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(IsGpuFusionKind(*module->entry_computation()->root_instruction(),
                              kCuDnnFusionKind));
}

// Case 1: concat with different operand sizes in the concat dimension is not
// supported by the downstream lowering, so the fusion must NOT be marked as a
// cuDNN custom fusion.
TEST_F(CudnnNonGemmFusionRewriterTest,
       ConcatWithDifferentOperandSizesIsNotRewritten) {
  constexpr char kHlo[] = R"(
HloModule m

fused_computation {
  p0 = f32[4,8] parameter(0)
  p1 = f32[4,4] parameter(1)
  ROOT concat = f32[4,12] concatenate(p0, p1), dimensions={1}
}

ENTRY e {
  a = f32[4,8] parameter(0)
  b = f32[4,4] parameter(1)
  ROOT fusion = f32[4,12] fusion(a, b), kind=kLoop, calls=fused_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      CudnnNonGemmFusionRewriter(stream_executor(), device_description())
          .Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_FALSE(IsGpuFusionKind(*module->entry_computation()->root_instruction(),
                               kCuDnnFusionKind));
}

// Case 2: a downstream slice prunes some of the concat operands (the slice
// only covers a subset of the concatenated operands' regions). This pattern is
// not supported by the downstream lowering, so the fusion must NOT be marked
// as a cuDNN custom fusion.
TEST_F(CudnnNonGemmFusionRewriterTest,
       ConcatWithDownstreamSliceThatPrunesOperandsIsNotRewritten) {
  // Concat of three [4,8] tensors along dim 1 -> [4,24].
  // The slice picks [0:4, 0:8] which lies entirely inside operand 0, so
  // operands 1 and 2 are effectively pruned by the downstream indexing.
  constexpr char kHlo[] = R"(
HloModule m

fused_computation {
  p0 = f32[4,8] parameter(0)
  p1 = f32[4,8] parameter(1)
  p2 = f32[4,8] parameter(2)
  concat = f32[4,24] concatenate(p0, p1, p2), dimensions={1}
  ROOT slice = f32[4,8] slice(concat), slice={[0:4], [0:8]}
}

ENTRY e {
  a = f32[4,8] parameter(0)
  b = f32[4,8] parameter(1)
  c = f32[4,8] parameter(2)
  ROOT fusion = f32[4,8] fusion(a, b, c), kind=kLoop, calls=fused_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      CudnnNonGemmFusionRewriter(stream_executor(), device_description())
          .Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_FALSE(IsGpuFusionKind(*module->entry_computation()->root_instruction(),
                               kCuDnnFusionKind));
}

}  // namespace
}  // namespace gpu
}  // namespace xla