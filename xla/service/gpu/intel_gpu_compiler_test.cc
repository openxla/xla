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

#include <gtest/gtest.h>

#include "absl/strings/substitute.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace xla {
namespace gpu {
namespace {

using IntelGpuCompilerTest = HloPjRtGpuTestBase;

TEST_F(IntelGpuCompilerTest, CheckCompiler) {
  EXPECT_EQ(stream_executor_platform_id(),
            stream_executor::sycl::kSyclPlatformId);
}

// Checks that SyclOneDnnWorkspacePass rewrites the cuBLASLt matmul workspace
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
// so only SyclOneDnnWorkspacePass can resize it.
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

}  // namespace
}  // namespace gpu
}  // namespace xla
