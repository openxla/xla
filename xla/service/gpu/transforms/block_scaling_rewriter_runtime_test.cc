/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/block_scaling_rewriter.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::testing::IsOkAndHolds;

class BlockScalingRewriterRuntimeTest
    : public HloPjRtInterpreterReferenceMixin<HloPjRtTestBase> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloPjRtTestBase::GetDebugOptionsForTest();
    debug_options
        .set_xla_gpu_experimental_enable_subchannel_dequantisation_fusion(true);
    return debug_options;
  }

  // Set the fusion backend config to prevent the autotuner from running.
  absl::Status UpdateGemmBackendConfig(HloInstruction* root) {
    GpuBackendConfig backend_config =
        root->backend_config<GpuBackendConfig>().value();
    FusionBackendConfig& fusion_backend_config =
        *backend_config.mutable_fusion_backend_config();
    auto& config = *fusion_backend_config.mutable_cudnn_fusion_config();
    config.set_plan_id(1);
    return root->set_backend_config(backend_config);
  }

  // Run block scaled rewriter on the given module.
  void RunBlockScalingRewriter(HloModule* module, bool expect_rewrite) {
    BlockScalingRewriter pass(/*allow_cudnn=*/true);
    EXPECT_THAT(RunHloPass(&pass, module), IsOkAndHolds(expect_rewrite));
    if (expect_rewrite) {
      HloInstruction* root = module->entry_computation()->root_instruction();
      EXPECT_TRUE(UpdateGemmBackendConfig(root).ok());
    }
  }

  // Make sure the call at the module root is not a composite call.
  void EnsureRootIsNotCompositeCall(HloModule* module) {
    HloInstruction* root = module->entry_computation()->root_instruction();
    EXPECT_FALSE(root->is_composite());
  }
};

TEST_F(BlockScalingRewriterRuntimeTest, Mxfp8) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

block_scaled_dot {
  %lhs = f8e4m3fn[4,192,96] parameter(0)
  %rhs = f8e4m3fn[4,512,96] parameter(1)
  %lhs_scale = f8e8m0fnu[4,192,3] parameter(2)
  %rhs_scale = f8e8m0fnu[4,512,3] parameter(3)
  %a_conv = f16[4,192,96] convert(%lhs)
  %b_conv = f16[4,512,96] convert(%rhs)
  %a_scale_conv = f16[4,192,3] convert(%lhs_scale)
  %b_scale_conv = f16[4,512,3] convert(%rhs_scale)
  %a_scale_bc = f16[4,192,3,32] broadcast(%a_scale_conv), dimensions={0,1,2}
  %b_scale_bc = f16[4,512,3,32] broadcast(%b_scale_conv), dimensions={0,1,2}
  %a_scale = f16[4,192,96] reshape(%a_scale_bc)
  %b_scale = f16[4,512,96] reshape(%b_scale_bc)
  %lhs_dq = f16[4,192,96] multiply(%a_conv, %a_scale)
  %rhs_dq = f16[4,512,96] multiply(%b_conv, %b_scale)
  ROOT %result = f32[4,192,512] dot(%lhs_dq, %rhs_dq),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
}

ENTRY main {
  %lhs = f8e4m3fn[4,192,96] parameter(0)
  %rhs = f8e4m3fn[4,512,96] parameter(1)
  %lhs_scale = f8e8m0fnu[4,192,3] parameter(2)
  %rhs_scale = f8e8m0fnu[4,512,3] parameter(3)
  ROOT %result = f32[4,192,512] call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      to_apply=block_scaled_dot, is_composite=true,
      frontend_attributes={composite.name="mx.block_scaled_dot",composite.version="1"}
})";

  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [&](HloModule* reference_module) {
        EnsureRootIsNotCompositeCall(reference_module);
      },
      /*test_preprocessor=*/
      [&](HloModule* test_module) {
        RunBlockScalingRewriter(test_module, /*expect_rewrite=*/true);
      }));
}

TEST_F(BlockScalingRewriterRuntimeTest, Mxfp8_e5m2_e4m3) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

block_scaled_dot {
  %lhs = f8e5m2[4,192,96] parameter(0)
  %rhs = f8e4m3fn[4,96,512] parameter(1)
  %lhs_scale = f8e8m0fnu[4,192,3] parameter(2)
  %rhs_scale = f8e8m0fnu[4,3,512] parameter(3)
  %a_conv = f16[4,192,96] convert(%lhs)
  %b_conv = f16[4,96,512] convert(%rhs)
  %a_scale_conv = f16[4,192,3] convert(%lhs_scale)
  %b_scale_conv = f16[4,3,512] convert(%rhs_scale)
  %a_scale_bc = f16[4,192,3,32] broadcast(%a_scale_conv), dimensions={0,1,2}
  %b_scale_bc = f16[4,3,32,512] broadcast(%b_scale_conv), dimensions={0,1,3}
  %a_scale = f16[4,192,96] reshape(%a_scale_bc)
  %b_scale = f16[4,96,512] reshape(%b_scale_bc)
  %lhs_dq = f16[4,192,96] multiply(%a_conv, %a_scale)
  %rhs_dq = f16[4,96,512] multiply(%b_conv, %b_scale)
  ROOT %result = f32[4,192,512] dot(%lhs_dq, %rhs_dq),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  %lhs = f8e5m2[4,192,96] parameter(0)
  %rhs = f8e4m3fn[4,96,512] parameter(1)
  %lhs_scale = f8e8m0fnu[4,192,3] parameter(2)
  %rhs_scale = f8e8m0fnu[4,3,512] parameter(3)
  ROOT %result = f32[4,192,512] call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      to_apply=block_scaled_dot, is_composite=true,
      frontend_attributes={composite.name="mx.block_scaled_dot",composite.version="1"}
})";

  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [&](HloModule* reference_module) {
        EnsureRootIsNotCompositeCall(reference_module);
      },
      /*test_preprocessor=*/
      [&](HloModule* test_module) {
        RunBlockScalingRewriter(test_module, /*expect_rewrite=*/true);
      }));
}

TEST_F(BlockScalingRewriterRuntimeTest, Mxfp8_e5m2_e5m2) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

block_scaled_dot {
  %lhs = f8e5m2[4,192,96] parameter(0)
  %rhs = f8e5m2[4,96,512] parameter(1)
  %lhs_scale = f8e8m0fnu[4,192,3] parameter(2)
  %rhs_scale = f8e8m0fnu[4,3,512] parameter(3)
  %a_conv = f16[4,192,96] convert(%lhs)
  %b_conv = f16[4,96,512] convert(%rhs)
  %a_scale_conv = f16[4,192,3] convert(%lhs_scale)
  %b_scale_conv = f16[4,3,512] convert(%rhs_scale)
  %a_scale_bc = f16[4,192,3,32] broadcast(%a_scale_conv), dimensions={0,1,2}
  %b_scale_bc = f16[4,3,32,512] broadcast(%b_scale_conv), dimensions={0,1,3}
  %a_scale = f16[4,192,96] reshape(%a_scale_bc)
  %b_scale = f16[4,96,512] reshape(%b_scale_bc)
  %lhs_dq = f16[4,192,96] multiply(%a_conv, %a_scale)
  %rhs_dq = f16[4,96,512] multiply(%b_conv, %b_scale)
  ROOT %result = f32[4,192,512] dot(%lhs_dq, %rhs_dq),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  %lhs = f8e5m2[4,192,96] parameter(0)
  %rhs = f8e5m2[4,96,512] parameter(1)
  %lhs_scale = f8e8m0fnu[4,192,3] parameter(2)
  %rhs_scale = f8e8m0fnu[4,3,512] parameter(3)
  ROOT %result = f32[4,192,512] call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      to_apply=block_scaled_dot, is_composite=true,
      frontend_attributes={composite.name="mx.block_scaled_dot",composite.version="1"}
})";

  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [&](HloModule* reference_module) {
        EnsureRootIsNotCompositeCall(reference_module);
      },
      /*test_preprocessor=*/
      [&](HloModule* test_module) {
        RunBlockScalingRewriter(test_module, /*expect_rewrite=*/false);
      }));
}

// Scale E2M1FN inputs, as otherwise they become all zeros for the random
// distribution produced by the test due to low type precision.
// Use positive block scale values, as Blackwell MMA discards the sign bit on
// the scale tensor.
TEST_F(BlockScalingRewriterRuntimeTest, Nvfp4) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

block_scaled_dot {
  %lhs = f4e2m1fn[4,192,96] parameter(0)
  %rhs = f4e2m1fn[4,512,96] parameter(1)
  %lhs_scale = f8e4m3fn[4,192,6] parameter(2)
  %rhs_scale = f8e4m3fn[4,512,6] parameter(3)
  %a_conv = f16[4,192,96] convert(%lhs)
  %b_conv = f16[4,512,96] convert(%rhs)
  %a_scale_conv = f16[4,192,6] convert(%lhs_scale)
  %b_scale_conv = f16[4,512,6] convert(%rhs_scale)
  %a_scale_bc = f16[4,192,6,16] broadcast(%a_scale_conv), dimensions={0,1,2}
  %b_scale_bc = f16[4,512,6,16] broadcast(%b_scale_conv), dimensions={0,1,2}
  %a_scale = f16[4,192,96] reshape(%a_scale_bc)
  %b_scale = f16[4,512,96] reshape(%b_scale_bc)
  %lhs_dq = f16[4,192,96] multiply(%a_conv, %a_scale)
  %rhs_dq = f16[4,512,96] multiply(%b_conv, %b_scale)
  ROOT %result = f32[4,192,512] dot(%lhs_dq, %rhs_dq),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
}

ENTRY main {
  %mult_scalar = f16[] constant(6)
  %mult0 = f16[4,192,96] broadcast(%mult_scalar), dimensions={}
  %mult1 = f16[4,512,96] broadcast(%mult_scalar), dimensions={}
  %p0 = f16[4,192,96] parameter(0)
  %p1 = f16[4,512,96] parameter(1)
  %lhs = f4e2m1fn[4,192,96] convert(f16[4,192,96] multiply(%p0, %mult0))
  %rhs = f4e2m1fn[4,512,96] convert(f16[4,512,96] multiply(%p1, %mult1))
  %lhs_scale = f8e4m3fn[4,192,6] parameter(2)
  %rhs_scale = f8e4m3fn[4,512,6] parameter(3)
  %lhs_scale_abs = f8e4m3fn[4,192,6] abs(%lhs_scale)
  %rhs_scale_abs = f8e4m3fn[4,512,6] abs(%rhs_scale)
  ROOT %result = f32[4,192,512] call(%lhs, %rhs, %lhs_scale_abs, %rhs_scale_abs),
      to_apply=block_scaled_dot, is_composite=true,
      frontend_attributes={composite.name="mx.block_scaled_dot",composite.version="1"}
})";

  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [&](HloModule* reference_module) {
        EnsureRootIsNotCompositeCall(reference_module);
      },
      /*test_preprocessor=*/
      [&](HloModule* test_module) {
        RunBlockScalingRewriter(test_module, /*expect_rewrite=*/true);
      }));
}

}  // namespace
}  // namespace xla::gpu
