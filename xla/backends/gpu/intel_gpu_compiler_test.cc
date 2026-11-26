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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/substitute.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/backends/gpu/hlo/transforms/onednn_workspace_pass.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace xla {
namespace gpu {
namespace {

using IntelGpuCompilerTest = HloPjRtGpuTestBase;

TEST_F(IntelGpuCompilerTest, CheckCompiler) {
  EXPECT_EQ(stream_executor_platform_id(),
            stream_executor::sycl::kSyclPlatformId);
}

// Checks that OneDnnWorkspacePass rewrites the cuBLASLt matmul workspace
// from the 4 MiB default to the oneDNN scratchpad size (0 for a tiny matmul).
TEST_F(IntelGpuCompilerTest, ResizesMatmulWorkspace) {
  const char* hlo_text = R"(
    HloModule matmul

    ENTRY main {
      a = bf16[8,8] parameter(0)
      b = bf16[8,8] parameter(1)
      ROOT dot = bf16[8,8] dot(a, b),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";

  const char* check_pattern = R"(
    CHECK: (bf16[8,8]{{.*}}, s8[0]{0}) custom-call
    CHECK-SAME: custom_call_target="__cublas$lt$matmul"
  )";
  MatchOptimizedHlo(hlo_text, check_pattern);
}

// Complex-typed matmuls are skipped because oneDNN's matmul primitive does
// not support complex element types; the 4 MiB default workspace from
// GemmRewriter is retained.
TEST_F(IntelGpuCompilerTest, SkipsWorkspaceResizeForComplexMatmul) {
  const char* hlo_text = R"(
    HloModule complex_matmul

    ENTRY main {
      a = c64[8,8] parameter(0)
      b = c64[8,8] parameter(1)
      ROOT dot = c64[8,8] dot(a, b),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";

  const char* check_pattern = R"(
    CHECK: (c64[8,8]{{.*}}, s8[$0]{0}) custom-call
    CHECK-SAME: custom_call_target="__cublas$$lt$$matmul"
  )";
  MatchOptimizedHlo(
      hlo_text, absl::Substitute(check_pattern, GemmConfig::kDefaultWorkspace));
}

// Checks that the convolution workspace, zero-sized by default, is resized to
// a non-zero size. The autotuner does not run for convolutions on Intel GPU,
// so only OneDnnWorkspacePass can resize it.
TEST_F(IntelGpuCompilerTest, ResizesConvWorkspace) {
  const char* hlo_text = R"(
    HloModule conv

    ENTRY main {
      input = f32[1,56,56,64] parameter(0)
      filter = f32[3,3,64,64] parameter(1)
      ROOT conv = f32[1,56,56,64] convolution(input, filter),
        window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
    }
  )";

  const char* check_pattern = R"(
    CHECK: (f32[1,56,56,64]{3,2,1,0}, u8[{{[1-9][0-9]*}}]{0}) custom-call
    CHECK-SAME: custom_call_target="__cudnn$convForward"
  )";
  MatchOptimizedHlo(hlo_text, check_pattern);
}

// Checks that the workspace of a fused convolution, which runs through the
// fused oneDNN runner, is resized.
TEST_F(IntelGpuCompilerTest, ResizesFusedConvWorkspace) {
  const char* hlo_text = R"(
    HloModule fused_conv

    ENTRY main {
      input = f32[1,56,56,64] parameter(0)
      filter = f32[3,3,64,64] parameter(1)
      bias = f32[64] parameter(2)
      conv = f32[1,56,56,64] convolution(input, filter),
        window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
      broadcast_bias = f32[1,56,56,64] broadcast(bias), dimensions={3}
      add = f32[1,56,56,64] add(conv, broadcast_bias)
      zero = f32[] constant(0)
      zeros = f32[1,56,56,64] broadcast(zero), dimensions={}
      ROOT relu = f32[1,56,56,64] maximum(add, zeros)
    }
  )";

  const char* check_pattern = R"(
    CHECK: (f32[1,56,56,64]{3,2,1,0}, u8[{{[1-9][0-9]*}}]{0}) custom-call
    CHECK-SAME: custom_call_target="__cudnn$convBiasActivationForward"
  )";
  MatchOptimizedHlo(hlo_text, check_pattern);
}

// Checks that the workspace of a backward-data convolution is resized.
TEST_F(IntelGpuCompilerTest, ResizesBackwardDataConvWorkspace) {
  const char* hlo_text = R"(
    HloModule backward_data_conv

    ENTRY main {
      grad = f32[8,32,32,32] parameter(0)
      filter = f32[3,3,16,32] parameter(1)
      reverse = f32[3,3,16,32] reverse(filter), dimensions={0,1}
      ROOT conv = f32[8,32,32,16] convolution(grad, reverse),
        window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01oi->b01f
    }
  )";

  const char* check_pattern = R"(
    CHECK: (f32[8,32,32,16]{3,2,1,0}, u8[{{[1-9][0-9]*}}]{0}) custom-call
    CHECK-SAME: custom_call_target="__cudnn$convBackwardInput"
  )";
  MatchOptimizedHlo(hlo_text, check_pattern);
}

// Checks that the workspace of a backward-filter convolution is resized.
TEST_F(IntelGpuCompilerTest, ResizesBackwardFilterConvWorkspace) {
  const char* hlo_text = R"(
    HloModule backward_filter_conv

    ENTRY main {
      input = f32[8,32,32,16] parameter(0)
      grad = f32[8,32,32,32] parameter(1)
      ROOT conv = f32[3,3,16,32] convolution(input, grad),
        window={size=32x32 pad=1_1x1_1}, dim_labels=f01b_i01o->01bf
    }
  )";

  const char* check_pattern = R"(
    CHECK: ({{.*}}, u8[{{[1-9][0-9]*}}]{0}) custom-call
    CHECK-SAME: custom_call_target="__cudnn$convBackwardFilter"
  )";
  MatchOptimizedHlo(hlo_text, check_pattern);
}

// Running the pass again on already-resized workspaces must be a no-op rather
// than failing to replace the convolution with a mismatched shape.
TEST_F(IntelGpuCompilerTest, WorkspacePassIsIdempotent) {
  const char* hlo_text = R"(
    HloModule matmul_and_conv

    ENTRY main {
      a = bf16[8,8] parameter(0)
      b = bf16[8,8] parameter(1)
      dot = bf16[8,8] dot(a, b),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      input = f32[1,56,56,64] parameter(2)
      filter = f32[3,3,64,64] parameter(3)
      conv = f32[1,56,56,64] convolution(input, filter),
        window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
      ROOT result = (bf16[8,8], f32[1,56,56,64]) tuple(dot, conv)
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       GetOptimizedModule(hlo_text));
  OneDnnWorkspacePass pass(device_description().gpu_compute_capability(),
                           /*device_ordinal=*/0);
  EXPECT_THAT(pass.Run(module.get()), absl_testing::IsOkAndHolds(false));
}

// Resizing a convolution workspace replaces the call with a new one. Its
// control predecessors must move to the new call and its control successors
// must still run after it.
TEST_F(IntelGpuCompilerTest, ConvWorkspaceResizeKeepsControlDependencies) {
  const char* hlo_text = R"hlo(
    HloModule conv_with_control_deps

    ENTRY main {
      input = f32[1,56,56,64]{3,2,1,0} parameter(0)
      filter = f32[64,3,3,64]{3,2,1,0} parameter(1)
      x = f32[4]{0} parameter(2)
      before = f32[4]{0} negate(x)
      conv = (f32[1,56,56,64]{3,2,1,0}, u8[0]{0}) custom-call(input, filter),
        window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f,
        custom_call_target="__cudnn$convForward",
        backend_config={"cudnn_conv_backend_config":{"conv_result_scale":1}},
        control-predecessors={before}
      after = f32[4]{0} negate(before), control-predecessors={conv}
      result = f32[1,56,56,64]{3,2,1,0} get-tuple-element(conv), index=0
      ROOT root = (f32[1,56,56,64]{3,2,1,0}, f32[4]{0}) tuple(result, after)
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  OneDnnWorkspacePass pass(device_description().gpu_compute_capability(),
                           /*device_ordinal=*/0);
  ASSERT_THAT(pass.Run(module.get()), absl_testing::IsOkAndHolds(true));

  const HloInstruction* new_conv = nullptr;
  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kCustomCall) {
      new_conv = instr;
    }
  }
  ASSERT_NE(new_conv, nullptr);
  EXPECT_GT(ShapeUtil::ByteSizeOf(new_conv->shape().tuple_shapes(1)), 0);

  const HloInstruction* before = FindInstruction(module.get(), "before");
  const HloInstruction* after = FindInstruction(module.get(), "after");
  ASSERT_NE(before, nullptr);
  ASSERT_NE(after, nullptr);
  EXPECT_THAT(new_conv->control_predecessors(), ::testing::ElementsAre(before));
  ASSERT_THAT(after->control_predecessors(), ::testing::SizeIs(1));
  EXPECT_EQ(after->control_predecessors()[0]->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(after->control_predecessors()[0]->operand(0)->operand(0), new_conv);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
