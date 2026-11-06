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
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;

class HipblasLtScaleUtilsTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<bool> CanUse(const std::string& hlo,
                              const char* arch = "gfx950") {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
    return CanUseHipblasLtScale32x8(
        *module->entry_computation()->root_instruction(),
        stream_executor::GpuComputeCapability(
            stream_executor::RocmComputeCapability(arch)));
  }

  // A scaled-dot of [m, k] by [n, k], K in blocks of 32.
  absl::StatusOr<bool> CanUse(int64_t m, int64_t n, int64_t k,
                              const char* type = "f8e4m3fn",
                              const char* arch = "gfx950") {
    return CanUse(absl::Substitute(R"(
HloModule m
ENTRY main {
  a = $0[$1,$3] parameter(0)
  b = $0[$2,$3] parameter(1)
  sa = f8e8m0fnu[$1,$4] parameter(2)
  sb = f8e8m0fnu[$2,$4] parameter(3)
  ROOT dot = f32[$1,$2] scaled-dot(a, b, sa, sb),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
})",
                                   type, m, n, k, k / 32),
                  arch);
  }
};

TEST_F(HipblasLtScaleUtilsTest, AcceptsAlignedFp8AndFp4) {
  EXPECT_THAT(CanUse(64, 96, 512, "f8e4m3fn"), IsOkAndHolds(true));
  EXPECT_THAT(CanUse(64, 96, 512, "f4e2m1fn"), IsOkAndHolds(true));
}

TEST_F(HipblasLtScaleUtilsTest, RejectsMOrNNotMultipleOf32) {
  EXPECT_THAT(CanUse(16, 96, 512), IsOkAndHolds(false));
  EXPECT_THAT(CanUse(64, 16, 512), IsOkAndHolds(false));
}

TEST_F(HipblasLtScaleUtilsTest, RejectsKNotMultipleOf256) {
  EXPECT_THAT(CanUse(64, 96, 288), IsOkAndHolds(false));
}

TEST_F(HipblasLtScaleUtilsTest, RejectsOtherTypes) {
  EXPECT_THAT(CanUse(64, 96, 512, "f8e5m2"), IsOkAndHolds(false));
}

TEST_F(HipblasLtScaleUtilsTest, RejectsOtherDevices) {
  EXPECT_THAT(CanUse(64, 96, 512, "f8e4m3fn", "gfx1250"), IsOkAndHolds(false));
}

TEST_F(HipblasLtScaleUtilsTest, RejectsKNotLast) {
  EXPECT_THAT(CanUse(64, 96, 1024), IsOkAndHolds(true));
  // The same sizes with the LHS as [K, M].
  EXPECT_THAT(CanUse(R"(
HloModule m
ENTRY main {
  a = f8e4m3fn[1024,64] parameter(0)
  b = f8e4m3fn[96,1024] parameter(1)
  sa = f8e8m0fnu[32,64] parameter(2)
  sb = f8e8m0fnu[96,32] parameter(3)
  ROOT dot = f32[64,96] scaled-dot(a, b, sa, sb),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
})"),
              IsOkAndHolds(false));
}

TEST_F(HipblasLtScaleUtilsTest, SwizzleMovesScalesIntoTiles) {
  // Two tiles in each dimension.
  const int64_t rows = 64;
  const int64_t cols = 16;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule m
ENTRY main {
  ROOT scale = f8e8m0fnu[64,16] parameter(0)
})"));
  HloInstruction* scale = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(HloInstruction * fusion,
                       SwizzleHipblasLtScale32x8(scale));

  Literal input(scale->shape());
  auto* src = static_cast<uint8_t*>(input.untyped_data());
  for (int64_t i = 0; i < rows * cols; ++i) {
    src[i] = (i * 37) % 251;
  }
  ASSERT_OK_AND_ASSIGN(
      Literal output, HloEvaluator().Evaluate(
                          *fusion->fused_instructions_computation(), {&input}));
  const auto* dst = static_cast<const uint8_t*>(output.untyped_data());
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      // Where hipBLASLt reads the scale of row r and block c.
      int64_t tile = (r / 32) * (cols / 8) + c / 8;
      int64_t offset = tile * 256 + (c % 4) * 64 + (r % 16) * 4 +
                       ((c % 8) / 4) * 2 + (r % 32) / 16;
      EXPECT_EQ(dst[offset], src[r * cols + c]) << "r=" << r << " c=" << c;
    }
  }
}

}  // namespace
}  // namespace xla::gpu
