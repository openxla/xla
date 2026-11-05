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

#include "xla/backends/gpu/autotuner/hipblaslt_scale_utils.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace xla::gpu {
namespace {

using HipblasLtScaleUtilsTest = HloHardwareIndependentTestBase;

std::string ScaledDotHlo(int64_t m = 64, int64_t n = 96, int64_t k = 512,
                         const char* dtype = "f8e4m3fn",
                         const char* input_layout = "1,0",
                         const char* scale_layout = "1,0") {
  return absl::Substitute(R"(
HloModule scales
ENTRY main {
  a = $4[$0,$2]{$5} parameter(0)
  b = $4[$1,$2]{$5} parameter(1)
  sa = f8e8m0fnu[$0,$3]{$6} parameter(2)
  sb = f8e8m0fnu[$1,$3]{$6} parameter(3)
  ROOT dot = f32[$0,$1] scaled-dot(a, b, sa, sb),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
})",
                          m, n, k, k / 32, dtype, input_layout, scale_layout);
}

std::string BatchedScaledDotHlo(int batch, bool fp4) {
  return absl::Substitute(R"(
HloModule batch
ENTRY main {
  a = $1[$0,64,512]{2,1,0$2} parameter(0)
  b = $1[$0,96,512]{2,1,0$2} parameter(1)
  sa = f8e8m0fnu[$0,64,16]{2,1,0} parameter(2)
  sb = f8e8m0fnu[$0,96,16]{2,1,0} parameter(3)
  ROOT dot = f32[$0,64,96] scaled-dot(a, b, sa, sb),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2}
})",
                          batch, fp4 ? "f4e2m1fn" : "f8e4m3fn",
                          fp4 ? ":E(4)" : "");
}

std::string MixedTypesHlo(bool lhs_is_fp4) {
  return absl::Substitute(R"(
HloModule mixed_types
ENTRY main {
  a = $0[64,512]{1,0$2} parameter(0)
  b = $1[96,512]{1,0$3} parameter(1)
  sa = f8e8m0fnu[64,16]{1,0} parameter(2)
  sb = f8e8m0fnu[96,16]{1,0} parameter(3)
  ROOT dot = f32[64,96] scaled-dot(a, b, sa, sb),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
})",
                          lhs_is_fp4 ? "f4e2m1fn" : "f8e4m3fn",
                          lhs_is_fp4 ? "f8e4m3fn" : "f4e2m1fn",
                          lhs_is_fp4 ? ":E(4)" : "", lhs_is_fp4 ? "" : ":E(4)");
}

bool Allowed(const HloModule& module, const char* arch = "gfx950") {
  return CanUseHipblasLtScale32x8(
             *module.entry_computation()->root_instruction(),
             stream_executor::GpuComputeCapability(
                 stream_executor::RocmComputeCapability(arch)))
      .IsAllowed();
}

TEST_F(HipblasLtScaleUtilsTest, AcceptsAlignedInputsOnGfx950) {
  for (const std::string& hlo :
       {ScaledDotHlo(), ScaledDotHlo(64, 96, 512, "f4e2m1fn", "1,0:E(4)"),
        BatchedScaledDotHlo(1, false), BatchedScaledDotHlo(1, true)}) {
    auto module_or = ParseAndReturnVerifiedModule(hlo);
    ASSERT_THAT(module_or, absl_testing::IsOk());
    EXPECT_TRUE(Allowed(**module_or)) << hlo;
  }
}

TEST_F(HipblasLtScaleUtilsTest, RejectsPaddingTypesLayoutsAndBatches) {
  for (const std::string& hlo :
       {ScaledDotHlo(16, 96, 512), ScaledDotHlo(64, 16, 512),
        ScaledDotHlo(64, 96, 288), ScaledDotHlo(64, 96, 512, "f8e5m2"),
        ScaledDotHlo(64, 96, 512, "f8e4m3fn", "0,1"),
        ScaledDotHlo(64, 96, 512, "f8e4m3fn", "1,0", "0,1"),
        ScaledDotHlo(16, 96, 512, "f4e2m1fn", "1,0:E(4)"),
        ScaledDotHlo(64, 16, 512, "f4e2m1fn", "1,0:E(4)"),
        ScaledDotHlo(64, 96, 288, "f4e2m1fn", "1,0:E(4)"),
        ScaledDotHlo(64, 96, 512, "f4e2m1fn", "0,1:E(4)"),
        ScaledDotHlo(64, 96, 512, "f4e2m1fn", "1,0:E(4)", "0,1"),
        MixedTypesHlo(false), MixedTypesHlo(true),
        BatchedScaledDotHlo(2, false), BatchedScaledDotHlo(2, true)}) {
    auto module_or = ParseAndReturnVerifiedModule(hlo);
    ASSERT_THAT(module_or, absl_testing::IsOk());
    EXPECT_FALSE(Allowed(**module_or)) << hlo;
  }
}

TEST_F(HipblasLtScaleUtilsTest, RejectsOtherMxDevices) {
  auto module_or = ParseAndReturnVerifiedModule(ScaledDotHlo());
  ASSERT_THAT(module_or, absl_testing::IsOk());
  for (const char* arch : {"gfx942", "gfx1250"}) {
    EXPECT_FALSE(Allowed(**module_or, arch)) << arch;
  }
}

TEST_F(HipblasLtScaleUtilsTest, PreservesBytesAtHardwareScaleOffsets) {
  // Several tiles in both dimensions, with and without a unit batch, against an
  // address formula independent of the reshape/transpose; all bytes differ.
  for (auto [rows, cols] :
       std::vector<std::pair<int64_t, int64_t>>{{32, 8}, {64, 16}, {96, 24}}) {
    for (bool batched : {false, true}) {
      std::string hlo = absl::Substitute(R"(
HloModule shuffle
ENTRY main {
  ROOT scale = f8e8m0fnu[$0$1,$2]{$3} parameter(0)
})",
                                         batched ? "1," : "", rows, cols,
                                         batched ? "2,1,0" : "1,0");
      auto module_or = ParseAndReturnVerifiedModule(hlo);
      ASSERT_THAT(module_or, absl_testing::IsOk());
      auto module = std::move(module_or).value();
      HloInstruction* scale = module->entry_computation()->root_instruction();
      Literal input(scale->shape());
      auto* src = static_cast<uint8_t*>(input.untyped_data());
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          src[r * cols + c] = (r * 19 + c * 73) % 255;
        }
      }
      auto fusion_or = SwizzleHipblasLtScale32x8(scale);
      ASSERT_THAT(fusion_or, absl_testing::IsOk());
      HloInstruction* fusion = std::move(fusion_or).value();
      EXPECT_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kLoop);
      EXPECT_EQ(fusion->operand(0), scale);
      auto result_or = HloEvaluator().Evaluate(
          *fusion->fused_instructions_computation(), {&input});
      ASSERT_THAT(result_or, absl_testing::IsOk());
      Literal result = std::move(result_or).value();
      const auto* dst = static_cast<const uint8_t*>(result.untyped_data());
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          int64_t tile = (r / 32) * (cols / 8) + c / 8;
          int64_t offset = tile * 256 + (c % 4) * 64 + (r % 16) * 4 +
                           ((c % 8) / 4) * 2 + (r % 32) / 16;
          EXPECT_EQ(dst[offset], src[r * cols + c])
              << "r=" << r << " c=" << c << " rows=" << rows
              << " cols=" << cols;
        }
      }
    }
  }
}

TEST_F(HipblasLtScaleUtilsTest, MultipleScaleFusionsHaveUniqueNames) {
  auto module_or = ParseAndReturnVerifiedModule(R"(
HloModule two_scales
ENTRY main {
  a = f8e8m0fnu[64,16]{1,0} parameter(0)
  b = f8e8m0fnu[96,16]{1,0} parameter(1)
  ROOT scales = (f8e8m0fnu[64,16]{1,0}, f8e8m0fnu[96,16]{1,0}) tuple(a,b)
})");
  ASSERT_THAT(module_or, absl_testing::IsOk());
  auto module = std::move(module_or).value();
  HloInstruction* root = module->entry_computation()->root_instruction();
  auto a_or = SwizzleHipblasLtScale32x8(root->mutable_operand(0));
  ASSERT_THAT(a_or, absl_testing::IsOk());
  auto b_or = SwizzleHipblasLtScale32x8(root->mutable_operand(1));
  ASSERT_THAT(b_or, absl_testing::IsOk());
  EXPECT_NE((*a_or)->name(), (*b_or)->name());
  ASSERT_THAT(root->ReplaceOperandWith(0, *a_or), absl_testing::IsOk());
  ASSERT_THAT(root->ReplaceOperandWith(1, *b_or), absl_testing::IsOk());
  EXPECT_THAT(module->Verify(), absl_testing::IsOk());
}

}  // namespace
}  // namespace xla::gpu
