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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/transforms/block_scaling_rewriter.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"

namespace xla::gpu {
namespace {

using ::tsl::testing::IsOkAndHolds;

class BlockScalingRewriterHipblasltTest : public GpuCodegenTest {
 protected:
  const auto& device_desc() const {
    return backend().default_stream_executor()->GetDeviceDescription();
  }

  const auto& GpuCapability() const {
    return device_desc().gpu_compute_capability();
  }

  bool IsRocm() const {
    return std::holds_alternative<stream_executor::RocmComputeCapability>(
        GpuCapability());
  }

  void SetUp() override {
    if (!IsRocm()) { GTEST_SKIP(); }
    auto rocm_cc = std::get<se::RocmComputeCapability>(GpuCapability());
    if (rocm_cc.gfx_version() != "gfx950") { GTEST_SKIP(); }
  };
};

TEST_F(BlockScalingRewriterHipblasltTest, Mxfp8) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[32,256] parameter(0)
  %rhs = f8e4m3fn[16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[16,8] parameter(3)
  ROOT %result = f32[32,16] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";
  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [&](HloModule* reference_module) {
        BlockScalingRewriter pass(this->device_desc(), /*allow_cudnn=*/false,
                                  /*allow_hipblaslt=*/false);
        EXPECT_THAT(RunHloPass(&pass, reference_module), IsOkAndHolds(true));
      },
      /*test_preprocessor=*/
      [&](HloModule* test_module) {
        BlockScalingRewriter pass(this->device_desc(), /*allow_cudnn=*/false,
                                  /*allow_hipblaslt=*/true);
        EXPECT_THAT(RunHloPass(&pass, test_module), IsOkAndHolds(true));
      }));

  RunAndFilecheckHloRewrite(
      hlo_string,
      BlockScalingRewriter(this->device_desc(), /*allow_cudnn=*/false,
                           /*allow_hipblaslt=*/false),
      "CHECK-NOT: __cublas$lt$matmul$mx");
  RunAndFilecheckHloRewrite(
      hlo_string,
      BlockScalingRewriter(this->device_desc(), /*allow_cudnn=*/false,
                           /*allow_hipblaslt=*/true),
      "CHECK: __cublas$lt$matmul$mx");
}

TEST_F(BlockScalingRewriterHipblasltTest, BatchedMxfp8) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[1,32,256] parameter(0)
  %rhs = f8e4m3fn[1,16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  ROOT %result = f32[1,32,16] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";
  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [&](HloModule* reference_module) {
        BlockScalingRewriter pass(this->device_desc(), /*allow_cudnn=*/false,
                                  /*allow_hipblaslt=*/false);
        EXPECT_THAT(RunHloPass(&pass, reference_module), IsOkAndHolds(true));
      },
      /*test_preprocessor=*/
      [&](HloModule* test_module) {
        BlockScalingRewriter pass(this->device_desc(), /*allow_cudnn=*/false,
                                  /*allow_hipblaslt=*/true);
        EXPECT_THAT(RunHloPass(&pass, test_module), IsOkAndHolds(true));
      }));

  RunAndFilecheckHloRewrite(
      hlo_string,
      BlockScalingRewriter(this->device_desc(), /*allow_cudnn=*/false,
                           /*allow_hipblaslt=*/false),
      "CHECK-NOT: __cublas$lt$matmul$mx");
  RunAndFilecheckHloRewrite(
      hlo_string,
      BlockScalingRewriter(this->device_desc(), /*allow_cudnn=*/false,
                           /*allow_hipblaslt=*/true),
      "CHECK: __cublas$lt$matmul$mx");
}

TEST_F(BlockScalingRewriterHipblasltTest, BatchedMxfp8_MixedTypes) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[1,32,256] parameter(0)
  %rhs = f8e5m2[1,16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  ROOT %result = f32[1,32,16] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";
  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [&](HloModule* reference_module) {
        BlockScalingRewriter pass(this->device_desc(), /*allow_cudnn=*/false,
                                  /*allow_hipblaslt=*/false);
        EXPECT_THAT(RunHloPass(&pass, reference_module), IsOkAndHolds(true));
      },
      /*test_preprocessor=*/
      [&](HloModule* test_module) {
        BlockScalingRewriter pass(this->device_desc(), /*allow_cudnn=*/false,
                                  /*allow_hipblaslt=*/true);
        EXPECT_THAT(RunHloPass(&pass, test_module), IsOkAndHolds(true));
      }));

  RunAndFilecheckHloRewrite(
      hlo_string,
      BlockScalingRewriter(this->device_desc(), /*allow_cudnn=*/false,
                           /*allow_hipblaslt=*/false),
      "CHECK-NOT: __cublas$lt$matmul$mx");
  RunAndFilecheckHloRewrite(
      hlo_string,
      BlockScalingRewriter(this->device_desc(), /*allow_cudnn=*/false,
                           /*allow_hipblaslt=*/true),
      "CHECK: __cublas$lt$matmul$mx");
}

TEST_F(BlockScalingRewriterHipblasltTest, BatchedMxfp4) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f4e2m1fn[1,32,256] parameter(0)
  %rhs = f4e2m1fn[1,16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  ROOT %result = f32[1,32,16] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";
  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [&](HloModule* reference_module) {
        BlockScalingRewriter pass(this->device_desc(), /*allow_cudnn=*/false,
                                  /*allow_hipblaslt=*/false);
        EXPECT_THAT(RunHloPass(&pass, reference_module), IsOkAndHolds(true));
      },
      /*test_preprocessor=*/
      [&](HloModule* test_module) {
        BlockScalingRewriter pass(this->device_desc(), /*allow_cudnn=*/false,
                                  /*allow_hipblaslt=*/true);
        EXPECT_THAT(RunHloPass(&pass, test_module), IsOkAndHolds(true));
      }));

  RunAndFilecheckHloRewrite(
      hlo_string,
      BlockScalingRewriter(this->device_desc(), /*allow_cudnn=*/false,
                           /*allow_hipblaslt=*/false),
      "CHECK-NOT: __cublas$lt$matmul$mx");
  RunAndFilecheckHloRewrite(
      hlo_string,
      BlockScalingRewriter(this->device_desc(), /*allow_cudnn=*/false,
                           /*allow_hipblaslt=*/true),
      "CHECK: __cublas$lt$matmul$mx");
}

TEST_F(BlockScalingRewriterHipblasltTest, BatchedMxfp4fp8) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f4e2m1fn[1,32,256] parameter(0)
  %rhs = f8e4m3fn[1,16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  ROOT %result = f32[1,32,16] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";
  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [&](HloModule* reference_module) {
        BlockScalingRewriter pass(this->device_desc(), /*allow_cudnn=*/false,
                                  /*allow_hipblaslt=*/false);
        EXPECT_THAT(RunHloPass(&pass, reference_module), IsOkAndHolds(true));
      },
      /*test_preprocessor=*/
      [&](HloModule* test_module) {
        BlockScalingRewriter pass(this->device_desc(), /*allow_cudnn=*/false,
                                  /*allow_hipblaslt=*/true);
        EXPECT_THAT(RunHloPass(&pass, test_module), IsOkAndHolds(true));
      }));

  RunAndFilecheckHloRewrite(
      hlo_string,
      BlockScalingRewriter(this->device_desc(), /*allow_cudnn=*/false,
                           /*allow_hipblaslt=*/false),
      "CHECK-NOT: __cublas$lt$matmul$mx");
  RunAndFilecheckHloRewrite(
      hlo_string,
      BlockScalingRewriter(this->device_desc(), /*allow_cudnn=*/false,
                           /*allow_hipblaslt=*/true),
      "CHECK: __cublas$lt$matmul$mx");
}

}  // namespace
}  // namespace xla::gpu
