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
#include "xla/tests/hlo_test_base.h"

namespace xla::gpu {
namespace {

using BlockScalingRewriterTest = HloTestBase;

// Converts custom call to a fusion.
TEST_F(BlockScalingRewriterTest, CustomCallToFusion) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[4,16,256] parameter(0)
  %rhs = f8e4m3fn[4,32,256] parameter(1)
  %lhs_scale = f8e8m0fnu[4,16,8] parameter(2)
  %rhs_scale = f8e8m0fnu[4,32,8] parameter(3)
  ROOT %result = f32[4,16,32] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  BlockScalingRewriter pass(/*allow_cudnn=*/true);
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: [[lhs_input:%.+]] = f8e4m3fn[4,16,256]{2,1,0} parameter(0)
  CHECK: [[lhs_input_conv:%.+]] = f32[4,16,256]{2,1,0} convert([[lhs_input]])
  CHECK: [[lhs_scale:%.+]] = f8e8m0fnu[4,16,8]{2,1,0} parameter(2)
  CHECK: [[lhs_scale_conv:%.+]] = f32[4,16,8]{2,1,0} convert([[lhs_scale]])
  CHECK: [[lhs_scale_bc:%.+]] = f32[4,16,8,32]{3,2,1,0} broadcast([[lhs_scale_conv]]), dimensions={0,1,2}
  CHECK: [[lhs_scale_rs:%.+]] = f32[4,16,256]{2,1,0} reshape([[lhs_scale_bc]])
  CHECK: [[lhs:%.+]] = f32[4,16,256]{2,1,0} multiply([[lhs_input_conv]], [[lhs_scale_rs]])
  CHECK: [[rhs_input:%.+]] = f8e4m3fn[4,32,256]{2,1,0} parameter(1)
  CHECK: [[rhs_input_conv:%.+]] = f32[4,32,256]{2,1,0} convert([[rhs_input]])
  CHECK: [[rhs_scale:%.+]] = f8e8m0fnu[4,32,8]{2,1,0} parameter(3)
  CHECK: [[rhs_scale_conv:%.+]] = f32[4,32,8]{2,1,0} convert([[rhs_scale]])
  CHECK: [[rhs_scale_bc:%.+]] = f32[4,32,8,32]{3,2,1,0} broadcast([[rhs_scale_conv]]), dimensions={0,1,2}
  CHECK: [[rhs_scale_rs:%.+]] = f32[4,32,256]{2,1,0} reshape([[rhs_scale_bc]])
  CHECK: [[rhs:%.+]] = f32[4,32,256]{2,1,0} multiply([[rhs_input_conv]], [[rhs_scale_rs]])
  CHECK: f32[4,16,32]{2,1,0} dot([[lhs]], [[rhs]])
  CHECK-SAME: lhs_batch_dims={0}, lhs_contracting_dims={2}
  CHECK-SAME: rhs_batch_dims={0}, rhs_contracting_dims={2}
  CHECK: f32[4,16,32]{2,1,0} fusion({{.*}}, {{.*}}, {{.*}}, {{.*}}), kind=kCustom
  CHECK-SAME: composite.name="mx.block_scaled_dot"
  CHECK-SAME: "kind":"__cudnn$fusion"
})");
}

// Converts composite op to a fusion (computation is the same).
TEST_F(BlockScalingRewriterTest, CompositeCallToFusion) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

block_scaled_dot {
  %lhs = f8e4m3fn[4,16,256] parameter(0)
  %rhs = f8e4m3fn[4,32,256] parameter(1)
  %lhs_scale = f8e8m0fnu[4,16,8] parameter(2)
  %rhs_scale = f8e8m0fnu[4,32,8] parameter(3)
  %a_conv = f16[4,16,256] convert(%lhs)
  %b_conv = f16[4,32,256] convert(%rhs)
  %a_scale_conv = f16[4,16,8] convert(%lhs_scale)
  %b_scale_conv = f16[4,32,8] convert(%rhs_scale)
  %a_scale_bc = f16[4,16,8,32] broadcast(%a_scale_conv), dimensions={0,1,2}
  %b_scale_bc = f16[4,32,8,32] broadcast(%b_scale_conv), dimensions={0,1,2}
  %a_scale = f16[4,16,256] reshape(%a_scale_bc)
  %b_scale = f16[4,32,256] reshape(%b_scale_bc)
  %lhs_dq = f16[4,16,256] multiply(%a_conv, %a_scale)
  %rhs_dq = f16[4,32,256] multiply(%b_conv, %b_scale)
  ROOT %result = f32[4,16,32] dot(%lhs_dq, %rhs_dq),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
}

ENTRY main {
  %lhs = f8e4m3fn[4,16,256] parameter(0)
  %rhs = f8e4m3fn[4,32,256] parameter(1)
  %lhs_scale = f8e8m0fnu[4,16,8] parameter(2)
  %rhs_scale = f8e8m0fnu[4,32,8] parameter(3)
  ROOT %result = f32[4,16,32] call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      to_apply=block_scaled_dot, is_composite=true,
      frontend_attributes={composite.name="mx.block_scaled_dot",composite.version="1"}
})";

  BlockScalingRewriter pass(/*allow_cudnn=*/true);
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: f32[4,16,32]{2,1,0} fusion({{.*}}, {{.*}}, {{.*}}, {{.*}}), kind=kCustom
  CHECK-SAME: calls=%block_scaled_dot
  CHECK-SAME: composite.name="mx.block_scaled_dot"
  CHECK-SAME: "kind":"__cudnn$fusion"
})");
}

// Converts NVFP4 call to cuDNN fusion.
TEST_F(BlockScalingRewriterTest, Nvfp4ToCudnnSupported) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f4e2m1fn[4,16,128] parameter(0)
  %rhs = f4e2m1fn[4,32,128] parameter(1)
  %lhs_scale = f8e4m3fn[4,16,8] parameter(2)
  %rhs_scale = f8e4m3fn[4,32,8] parameter(3)
  ROOT %result = f32[4,16,32] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  BlockScalingRewriter pass(/*allow_cudnn=*/true);
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: f32[4,16,32]{2,1,0} fusion({{.*}}, {{.*}}, {{.*}}, {{.*}}), kind=kCustom
  CHECK-SAME: composite.name="mx.block_scaled_dot"
  CHECK-SAME: "kind":"__cudnn$fusion"
})");
}

// Converts MXFP8 call (E4M3/E5M2) to cuDNN fusion.
TEST_F(BlockScalingRewriterTest, Mxfp8ToCudnnSupported) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[4,16,256] parameter(0)
  %rhs = f8e5m2[4,32,256] parameter(1)
  %lhs_scale = f8e8m0fnu[4,16,8] parameter(2)
  %rhs_scale = f8e8m0fnu[4,32,8] parameter(3)
  ROOT %result = f32[4,16,32] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  BlockScalingRewriter pass(/*allow_cudnn=*/true);
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: f32[4,16,32]{2,1,0} fusion({{.*}}, {{.*}}, {{.*}}, {{.*}}), kind=kCustom
  CHECK-SAME: composite.name="mx.block_scaled_dot"
  CHECK-SAME: "kind":"__cudnn$fusion"
})");
}

// Converts MXFP8 call (E5M2/E5M2) to composite op (not supported by cuDNN).
TEST_F(BlockScalingRewriterTest, Mxfp8ToCudnnUnsupported) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e5m2[4,16,256] parameter(0)
  %rhs = f8e5m2[4,32,256] parameter(1)
  %lhs_scale = f8e8m0fnu[4,16,8] parameter(2)
  %rhs_scale = f8e8m0fnu[4,32,8] parameter(3)
  ROOT %result = f32[4,16,32] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  BlockScalingRewriter pass(/*allow_cudnn=*/true);
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: f32[4,16,32]{2,1,0} call({{.*}}, {{.*}}, {{.*}}, {{.*}})
  CHECK-SAME: is_composite=true
  CHECK-SAME: composite.name="mx.block_scaled_dot"
})");
}

// cuDNN simple test (MXFP8).
TEST_F(BlockScalingRewriterTest, CudnnSwizzle2D) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

block_scaled_dot {
  %lhs = f8e4m3fn[256,128] parameter(0)
  %rhs = f8e4m3fn[512,128] parameter(1)
  %lhs_scale = f8e8m0fnu[256,4] parameter(2)
  %rhs_scale = f8e8m0fnu[512,4] parameter(3)
  %a_conv = f16[256,128] convert(%lhs)
  %b_conv = f16[512,128] convert(%rhs)
  %a_scale_conv = f16[256,4] convert(%lhs_scale)
  %b_scale_conv = f16[512,4] convert(%rhs_scale)
  %a_scale_bc = f16[256,4,32] broadcast(%a_scale_conv), dimensions={0,1}
  %b_scale_bc = f16[512,4,32] broadcast(%b_scale_conv), dimensions={0,1}
  %a_scale = f16[256,128] reshape(%a_scale_bc)
  %b_scale = f16[512,128] reshape(%b_scale_bc)
  %lhs_dq = f16[256,128] multiply(%a_conv, %a_scale)
  %rhs_dq = f16[512,128] multiply(%b_conv, %b_scale)
  ROOT %result = f32[256,512] dot(%lhs_dq, %rhs_dq),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY main {
  %lhs = f8e4m3fn[256,128] parameter(0)
  %rhs = f8e4m3fn[512,128] parameter(1)
  %lhs_scale = f8e8m0fnu[256,4] parameter(2)
  %rhs_scale = f8e8m0fnu[512,4] parameter(3)
  ROOT %result = f32[256,512] fusion(%lhs, %rhs, %lhs_scale, %rhs_scale),
      kind=kCustom, calls=block_scaled_dot,
      backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion"}},
      frontend_attributes={composite.name="mx.block_scaled_dot",composite.version="1"}
})";

  CudnnBlockScalingRewriter pass;
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: [[swizzle:%.+]] ({{.+}}: f8e8m0fnu[256,4], {{.+}}: f8e8m0fnu[512,4])
  CHECK: [[lhs_scale:%.+]] = f8e8m0fnu[256,4]{1,0} parameter(0)
  CHECK: [[lhs_scale_rs:%.+]] = f8e8m0fnu[1,2,4,32,1,4]{5,4,3,2,1,0} reshape([[lhs_scale]])
  CHECK: [[lhs_scale_tr:%.+]] = f8e8m0fnu[1,2,1,32,4,4]{5,4,3,2,1,0} transpose([[lhs_scale_rs]]), dimensions={0,1,4,3,2,5}
  CHECK: [[lhs_scale_swizzle:%.+]] = f8e8m0fnu[256,4]{1,0} reshape([[lhs_scale_tr]])
  CHECK: [[rhs_scale:%.+]] = f8e8m0fnu[512,4]{1,0} parameter(1)
  CHECK: [[rhs_scale_rs:%.+]] = f8e8m0fnu[1,4,4,32,1,4]{5,4,3,2,1,0} reshape([[rhs_scale]])
  CHECK: [[rhs_scale_tr:%.+]] = f8e8m0fnu[1,4,1,32,4,4]{5,4,3,2,1,0} transpose([[rhs_scale_rs]]), dimensions={0,1,4,3,2,5}
  CHECK: [[rhs_scale_swizzle:%.+]] = f8e8m0fnu[512,4]{1,0} reshape([[rhs_scale_tr]])
  CHECK: ROOT {{.+}} = (f8e8m0fnu[256,4]{1,0}, f8e8m0fnu[512,4]{1,0}) tuple([[lhs_scale_swizzle]], [[rhs_scale_swizzle]])
  CHECK: [[scale_tuple:%.+]] = (f8e8m0fnu[256,4]{1,0}, f8e8m0fnu[512,4]{1,0}) call({{.+}}, {{.+}}), to_apply=[[swizzle]]
  CHECK: [[lhs_swizzled:%.+]] = f8e8m0fnu[256,4]{1,0} get-tuple-element([[scale_tuple]]), index=0
  CHECK: [[rhs_swizzled:%.+]] = f8e8m0fnu[512,4]{1,0} get-tuple-element([[scale_tuple]]), index=1
  CHECK: f32[256,512]{1,0} fusion({{.+}}, {{.+}}, [[lhs_swizzled]], [[rhs_swizzled]])
})");
}

TEST_F(BlockScalingRewriterTest, CudnnSwizzleRhsTransposed) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

block_scaled_dot {
  %lhs = f8e4m3fn[8,256,128] parameter(0)
  %rhs = f8e4m3fn[8,128,512] parameter(1)
  %lhs_scale = f8e8m0fnu[8,256,4] parameter(2)
  %rhs_scale = f8e8m0fnu[8,4,512] parameter(3)
  %a_conv = f16[8,256,128] convert(%lhs)
  %b_conv = f16[8,128,512] convert(%rhs)
  %a_scale_conv = f16[8,256,4] convert(%lhs_scale)
  %b_scale_conv = f16[8,4,512] convert(%rhs_scale)
  %a_scale_bc = f16[8,256,4,32] broadcast(%a_scale_conv), dimensions={0,1,2}
  %b_scale_bc = f16[8,4,32,512] broadcast(%b_scale_conv), dimensions={0,1,3}
  %a_scale = f16[8,256,128] reshape(%a_scale_bc)
  %b_scale = f16[8,128,512] reshape(%b_scale_bc)
  %lhs_dq = f16[8,256,128] multiply(%a_conv, %a_scale)
  %rhs_dq = f16[8,128,512] multiply(%b_conv, %b_scale)
  ROOT %result = f32[8,256,512] dot(%lhs_dq, %rhs_dq),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={1}
}

ENTRY main {
  %lhs = f8e4m3fn[8,256,128] parameter(0)
  %rhs = f8e4m3fn[8,128,512] parameter(1)
  %lhs_scale = f8e8m0fnu[8,256,4] parameter(2)
  %rhs_scale = f8e8m0fnu[8,4,512] parameter(3)
  ROOT %result = f32[8,256,512] fusion(%lhs, %rhs, %lhs_scale, %rhs_scale),
      kind=kCustom, calls=block_scaled_dot,
      backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion"}},
      frontend_attributes={composite.name="mx.block_scaled_dot",composite.version="1"}
})";

  CudnnBlockScalingRewriter pass;
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: [[swizzle:%.+]] ({{.+}}: f8e8m0fnu[8,256,4], {{.+}}: f8e8m0fnu[8,4,512])
  CHECK: [[lhs_scale:%.+]] = f8e8m0fnu[8,256,4]{2,1,0} parameter(0)
  CHECK: [[lhs_scale_rs:%.+]] = f8e8m0fnu[8,2,4,32,1,4]{5,4,3,2,1,0} reshape([[lhs_scale]])
  CHECK: [[lhs_scale_tr:%.+]] = f8e8m0fnu[8,2,1,32,4,4]{5,4,3,2,1,0} transpose([[lhs_scale_rs]]), dimensions={0,1,4,3,2,5}
  CHECK: [[lhs_scale_swizzle:%.+]] = f8e8m0fnu[8,256,4]{2,1,0} reshape([[lhs_scale_tr]])
  CHECK: [[rhs_scale:%.+]] = f8e8m0fnu[8,4,512]{2,1,0} parameter(1)
  CHECK: [[rhs_scale_rs:%.+]] = f8e8m0fnu[8,1,4,4,4,32]{5,4,3,2,1,0} reshape([[rhs_scale]])
  CHECK: [[rhs_scale_tr:%.+]] = f8e8m0fnu[8,4,1,32,4,4]{5,4,3,2,1,0} transpose([[rhs_scale_rs]]), dimensions={0,3,1,5,4,2}
  CHECK: [[rhs_scale_swizzle:%.+]] = f8e8m0fnu[8,512,4]{2,1,0} reshape([[rhs_scale_tr]])
  CHECK: ROOT {{.+}} = (f8e8m0fnu[8,256,4]{2,1,0}, f8e8m0fnu[8,512,4]{2,1,0}) tuple([[lhs_scale_swizzle]], [[rhs_scale_swizzle]])
  CHECK: [[rhs_scale_param:%.+]] = f8e8m0fnu[8,512,4]{2,1,0} parameter(3)
  CHECK: f8e8m0fnu[8,4,512]{2,1,0} transpose([[rhs_scale_param]]), dimensions={0,2,1}
  CHECK: [[scale_tuple:%.+]] = (f8e8m0fnu[8,256,4]{2,1,0}, f8e8m0fnu[8,512,4]{2,1,0}) call({{.+}}, {{.+}}), to_apply=[[swizzle]]
  CHECK: [[lhs_swizzled:%.+]] = f8e8m0fnu[8,256,4]{2,1,0} get-tuple-element([[scale_tuple]]), index=0
  CHECK: [[rhs_swizzled:%.+]] = f8e8m0fnu[8,512,4]{2,1,0} get-tuple-element([[scale_tuple]]), index=1
  CHECK: f32[8,256,512]{2,1,0} fusion({{.+}}, {{.+}}, [[lhs_swizzled]], [[rhs_swizzled]])
})");
}

TEST_F(BlockScalingRewriterTest, CudnnSwizzleRhsMixed) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

block_scaled_dot {
  %lhs = f8e4m3fn[8,256,128] parameter(0)
  %rhs = f8e4m3fn[8,128,512] parameter(1)
  %lhs_scale = f8e8m0fnu[8,256,4] parameter(2)
  %rhs_scale = f8e8m0fnu[8,512,4] parameter(3)
  %rhs_trans = f8e4m3fn[8,512,128] transpose(%rhs), dimensions={0,2,1}
  %a_conv = f16[8,256,128] convert(%lhs)
  %b_conv = f16[8,512,128] convert(%rhs_trans)
  %a_scale_conv = f16[8,256,4] convert(%lhs_scale)
  %b_scale_conv = f16[8,512,4] convert(%rhs_scale)
  %a_scale_bc = f16[8,256,4,32] broadcast(%a_scale_conv), dimensions={0,1,2}
  %b_scale_bc = f16[8,512,4,32] broadcast(%b_scale_conv), dimensions={0,1,2}
  %a_scale = f16[8,256,128] reshape(%a_scale_bc)
  %b_scale = f16[8,512,128] reshape(%b_scale_bc)
  %lhs_dq = f16[8,256,128] multiply(%a_conv, %a_scale)
  %rhs_dq = f16[8,512,128] multiply(%b_conv, %b_scale)
  ROOT %result = f32[8,256,512] dot(%lhs_dq, %rhs_dq),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={2}
}

ENTRY main {
  %lhs = f8e4m3fn[8,256,128] parameter(0)
  %rhs = f8e4m3fn[8,128,512] parameter(1)
  %lhs_scale = f8e8m0fnu[8,256,4] parameter(2)
  %rhs_scale = f8e8m0fnu[8,512,4] parameter(3)
  ROOT %result = f32[8,256,512] fusion(%lhs, %rhs, %lhs_scale, %rhs_scale),
      kind=kCustom, calls=block_scaled_dot,
      backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion"}},
      frontend_attributes={composite.name="mx.block_scaled_dot",composite.version="1"}
})";

  CudnnBlockScalingRewriter pass;
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: [[swizzle:%.+]] ({{.+}}: f8e8m0fnu[8,256,4], {{.+}}: f8e8m0fnu[8,512,4])
  CHECK: [[lhs_scale:%.+]] = f8e8m0fnu[8,256,4]{2,1,0} parameter(0)
  CHECK: [[lhs_scale_rs:%.+]] = f8e8m0fnu[8,2,4,32,1,4]{5,4,3,2,1,0} reshape([[lhs_scale]])
  CHECK: [[lhs_scale_tr:%.+]] = f8e8m0fnu[8,2,1,32,4,4]{5,4,3,2,1,0} transpose([[lhs_scale_rs]]), dimensions={0,1,4,3,2,5}
  CHECK: [[lhs_scale_swizzle:%.+]] = f8e8m0fnu[8,256,4]{2,1,0} reshape([[lhs_scale_tr]])
  CHECK: [[rhs_scale:%.+]] = f8e8m0fnu[8,512,4]{2,1,0} parameter(1)
  CHECK: [[rhs_scale_rs:%.+]] = f8e8m0fnu[8,4,4,32,1,4]{5,4,3,2,1,0} reshape([[rhs_scale]])
  CHECK: [[rhs_scale_tr:%.+]] = f8e8m0fnu[8,4,1,32,4,4]{5,4,3,2,1,0} transpose([[rhs_scale_rs]]), dimensions={0,1,4,3,2,5}
  CHECK: [[rhs_scale_swizzle:%.+]] = f8e8m0fnu[8,512,4]{2,1,0} reshape([[rhs_scale_tr]])
  CHECK: ROOT {{.+}} = (f8e8m0fnu[8,256,4]{2,1,0}, f8e8m0fnu[8,512,4]{2,1,0}) tuple([[lhs_scale_swizzle]], [[rhs_scale_swizzle]])
  CHECK: [[scale_tuple:%.+]] = (f8e8m0fnu[8,256,4]{2,1,0}, f8e8m0fnu[8,512,4]{2,1,0}) call({{.+}}, {{.+}}), to_apply=[[swizzle]]
  CHECK: [[lhs_swizzled:%.+]] = f8e8m0fnu[8,256,4]{2,1,0} get-tuple-element([[scale_tuple]]), index=0
  CHECK: [[rhs_swizzled:%.+]] = f8e8m0fnu[8,512,4]{2,1,0} get-tuple-element([[scale_tuple]]), index=1
  CHECK: f32[8,256,512]{2,1,0} fusion({{.+}}, {{.+}}, [[lhs_swizzled]], [[rhs_swizzled]])
})");
}

// Verify that the scales are padded to 128x4 tile.
TEST_F(BlockScalingRewriterTest, CudnnPadScales) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

block_scaled_dot {
  %lhs = f8e4m3fn[8,192,96] parameter(0)
  %rhs = f8e4m3fn[8,96,512] parameter(1)
  %lhs_scale = f8e8m0fnu[8,192,3] parameter(2)
  %rhs_scale = f8e8m0fnu[8,3,512] parameter(3)
  %a_conv = f16[8,192,96] convert(%lhs)
  %b_conv = f16[8,96,512] convert(%rhs)
  %a_scale_conv = f16[8,192,3] convert(%lhs_scale)
  %b_scale_conv = f16[8,3,512] convert(%rhs_scale)
  %a_scale_bc = f16[8,192,3,32] broadcast(%a_scale_conv), dimensions={0,1,2}
  %b_scale_bc = f16[8,3,32,512] broadcast(%b_scale_conv), dimensions={0,1,3}
  %a_scale = f16[8,192,96] reshape(%a_scale_bc)
  %b_scale = f16[8,96,512] reshape(%b_scale_bc)
  %lhs_dq = f16[8,192,96] multiply(%a_conv, %a_scale)
  %rhs_dq = f16[8,96,512] multiply(%b_conv, %b_scale)
  ROOT %result = f32[8,192,512] dot(%lhs_dq, %rhs_dq),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={1}
}

ENTRY main {
  %lhs = f8e4m3fn[8,192,96] parameter(0)
  %rhs = f8e4m3fn[8,96,512] parameter(1)
  %lhs_scale = f8e8m0fnu[8,192,3] parameter(2)
  %rhs_scale = f8e8m0fnu[8,3,512] parameter(3)
  ROOT %result = f32[8,192,512] fusion(%lhs, %rhs, %lhs_scale, %rhs_scale),
      kind=kCustom, calls=block_scaled_dot,
      backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion"}},
      frontend_attributes={composite.name="mx.block_scaled_dot",composite.version="1"}
})";

  CudnnBlockScalingRewriter pass;
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: [[swizzle:%.+]] ({{.+}}: f8e8m0fnu[8,192,3], {{.+}}: f8e8m0fnu[8,3,512])
  CHECK: [[lhs_scale:%.+]] = f8e8m0fnu[8,192,3]{2,1,0} parameter(0)
  CHECK: [[c0_1:%.+]] = f8e8m0fnu[] constant(5.9e-39)
  CHECK: [[lhs_scale_padded:%.+]] = f8e8m0fnu[8,256,4]{2,1,0} pad([[lhs_scale]], [[c0_1]]), padding=0_0x0_64x0_1
  CHECK: [[lhs_scale_rs:%.+]] = f8e8m0fnu[8,2,4,32,1,4]{5,4,3,2,1,0} reshape([[lhs_scale_padded]])
  CHECK: [[lhs_scale_tr:%.+]] = f8e8m0fnu[8,2,1,32,4,4]{5,4,3,2,1,0} transpose([[lhs_scale_rs]]), dimensions={0,1,4,3,2,5}
  CHECK: [[lhs_scale_swizzle:%.+]] = f8e8m0fnu[8,256,4]{2,1,0} reshape([[lhs_scale_tr]])
  CHECK: [[rhs_scale:%.+]] = f8e8m0fnu[8,3,512]{2,1,0} parameter(1)
  CHECK: [[c0_2:%.+]] = f8e8m0fnu[] constant(5.9e-39)
  CHECK: [[rhs_scale_padded:%.+]] = f8e8m0fnu[8,4,512]{2,1,0} pad([[rhs_scale]], [[c0_2]]), padding=0_0x0_1x0_0
  CHECK: [[rhs_scale_rs:%.+]] = f8e8m0fnu[8,1,4,4,4,32]{5,4,3,2,1,0} reshape([[rhs_scale_padded]])
  CHECK: [[rhs_scale_tr:%.+]] = f8e8m0fnu[8,4,1,32,4,4]{5,4,3,2,1,0} transpose([[rhs_scale_rs]]), dimensions={0,3,1,5,4,2}
  CHECK: [[rhs_scale_swizzle:%.+]] = f8e8m0fnu[8,512,4]{2,1,0} reshape([[rhs_scale_tr]])
  CHECK: ROOT {{.+}} = (f8e8m0fnu[8,256,4]{2,1,0}, f8e8m0fnu[8,512,4]{2,1,0}) tuple([[lhs_scale_swizzle]], [[rhs_scale_swizzle]])
  CHECK: [[lhs_scale_param:%.+]] = f8e8m0fnu[8,256,4]{2,1,0} parameter(2)
  CHECK: f8e8m0fnu[8,192,3]{2,1,0} slice([[lhs_scale_param]]), slice={[0:8], [0:192], [0:3]}
  CHECK: [[rhs_scale_param:%.+]] = f8e8m0fnu[8,512,4]{2,1,0} parameter(3)
  CHECK: [[rhs_scale_trans:%.+]] = f8e8m0fnu[8,4,512]{2,1,0} transpose([[rhs_scale_param]]), dimensions={0,2,1}
  CHECK: f8e8m0fnu[8,3,512]{2,1,0} slice([[rhs_scale_trans]]), slice={[0:8], [0:3], [0:512]}
  CHECK: [[scale_tuple:%.+]] = (f8e8m0fnu[8,256,4]{2,1,0}, f8e8m0fnu[8,512,4]{2,1,0}) call({{.+}}, {{.+}}), to_apply=[[swizzle]]
  CHECK: [[lhs_swizzled:%.+]] = f8e8m0fnu[8,256,4]{2,1,0} get-tuple-element([[scale_tuple]]), index=0
  CHECK: [[rhs_swizzled:%.+]] = f8e8m0fnu[8,512,4]{2,1,0} get-tuple-element([[scale_tuple]]), index=1
  CHECK: f32[8,192,512]{2,1,0} fusion({{.+}}, {{.+}}, [[lhs_swizzled]], [[rhs_swizzled]])
})");
}

// Verify that pre-padded scales are supported.
TEST_F(BlockScalingRewriterTest, CudnnPrePaddedScales) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

block_scaled_dot {
  %lhs = f8e4m3fn[8,192,96] parameter(0)
  %rhs = f8e4m3fn[8,512,96] parameter(1)
  %lhs_scale = f8e8m0fnu[8,256,4] parameter(2)
  %rhs_scale = f8e8m0fnu[8,512,4] parameter(3)
  %lhs_scale_slice = f8e8m0fnu[8,192,3] slice(%lhs_scale), slice={[0:8], [0:192], [0:3]}
  %rhs_scale_slice = f8e8m0fnu[8,512,3] slice(%rhs_scale), slice={[0:8], [0:512], [0:3]}
  %a_conv = f16[8,192,96] convert(%lhs)
  %b_conv = f16[8,512,96] convert(%rhs)
  %a_scale_conv = f16[8,192,3] convert(%lhs_scale_slice)
  %b_scale_conv = f16[8,512,3] convert(%rhs_scale_slice)
  %a_scale_bc = f16[8,192,3,32] broadcast(%a_scale_conv), dimensions={0,1,2}
  %b_scale_bc = f16[8,512,3,32] broadcast(%b_scale_conv), dimensions={0,1,2}
  %a_scale = f16[8,192,96] reshape(%a_scale_bc)
  %b_scale = f16[8,512,96] reshape(%b_scale_bc)
  %lhs_dq = f16[8,192,96] multiply(%a_conv, %a_scale)
  %rhs_dq = f16[8,512,96] multiply(%b_conv, %b_scale)
  ROOT %result = f32[8,192,512] dot(%lhs_dq, %rhs_dq),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={2}
}

ENTRY main {
  %lhs = f8e4m3fn[8,192,96] parameter(0)
  %rhs = f8e4m3fn[8,512,96] parameter(1)
  %lhs_scale = f8e8m0fnu[8,256,4] parameter(2)
  %rhs_scale = f8e8m0fnu[8,512,4] parameter(3)
  ROOT %result = f32[8,192,512] fusion(%lhs, %rhs, %lhs_scale, %rhs_scale),
      kind=kCustom, calls=block_scaled_dot,
      backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion"}},
      frontend_attributes={composite.name="mx.block_scaled_dot",composite.version="1"}
})";

  CudnnBlockScalingRewriter pass;
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK: [[swizzle:%.+]] ({{.+}}: f8e8m0fnu[8,256,4], {{.+}}: f8e8m0fnu[8,512,4])
  CHECK: [[lhs_scale:%.+]] = f8e8m0fnu[8,256,4]{2,1,0} parameter(0)
  CHECK: [[lhs_scale_rs:%.+]] = f8e8m0fnu[8,2,4,32,1,4]{5,4,3,2,1,0} reshape([[lhs_scale]])
  CHECK: [[lhs_scale_tr:%.+]] = f8e8m0fnu[8,2,1,32,4,4]{5,4,3,2,1,0} transpose([[lhs_scale_rs]]), dimensions={0,1,4,3,2,5}
  CHECK: [[lhs_scale_swizzle:%.+]] = f8e8m0fnu[8,256,4]{2,1,0} reshape([[lhs_scale_tr]])
  CHECK: [[rhs_scale:%.+]] = f8e8m0fnu[8,512,4]{2,1,0} parameter(1)
  CHECK: [[rhs_scale_rs:%.+]] = f8e8m0fnu[8,4,4,32,1,4]{5,4,3,2,1,0} reshape([[rhs_scale]])
  CHECK: [[rhs_scale_tr:%.+]] = f8e8m0fnu[8,4,1,32,4,4]{5,4,3,2,1,0} transpose([[rhs_scale_rs]]), dimensions={0,1,4,3,2,5}
  CHECK: [[rhs_scale_swizzle:%.+]] = f8e8m0fnu[8,512,4]{2,1,0} reshape([[rhs_scale_tr]])
  CHECK: ROOT {{.+}} = (f8e8m0fnu[8,256,4]{2,1,0}, f8e8m0fnu[8,512,4]{2,1,0}) tuple([[lhs_scale_swizzle]], [[rhs_scale_swizzle]])
  CHECK: [[scale_tuple:%.+]] = (f8e8m0fnu[8,256,4]{2,1,0}, f8e8m0fnu[8,512,4]{2,1,0}) call({{.+}}, {{.+}}), to_apply=[[swizzle]]
  CHECK: [[lhs_swizzled:%.+]] = f8e8m0fnu[8,256,4]{2,1,0} get-tuple-element([[scale_tuple]]), index=0
  CHECK: [[rhs_swizzled:%.+]] = f8e8m0fnu[8,512,4]{2,1,0} get-tuple-element([[scale_tuple]]), index=1
  CHECK: f32[8,192,512]{2,1,0} fusion({{.+}}, {{.+}}, [[lhs_swizzled]], [[rhs_swizzled]])
})");
}

}  // namespace
}  // namespace xla::gpu
