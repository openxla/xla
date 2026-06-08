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

#include "xla/backends/gpu/transforms/triton_scaled_dot_support.h"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

constexpr absl::string_view kCanonical = R"(
HloModule m
ENTRY e {
  lhs = $lhs[128,256] parameter(0)
  rhs = $rhs[256,128] parameter(1)
  lhs_scale = $lhs_scale[128,8] parameter(2)
  rhs_scale = $rhs_scale[8,128] parameter(3)
  ROOT _ = bf16[128,128] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

constexpr absl::string_view kTransposedRhs = R"(
HloModule m
ENTRY e {
  lhs = $lhs[128,256] parameter(0)
  rhs = $rhs[128,256] parameter(1)
  lhs_scale = $lhs_scale[128,8] parameter(2)
  rhs_scale = $rhs_scale[128,8] parameter(3)
  ROOT _ = bf16[128,128] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}
)";

absl::StatusOr<std::unique_ptr<HloModule>> ParseScaledDot(
    absl::string_view layout_template, absl::string_view lhs_type,
    absl::string_view rhs_type,
    absl::string_view lhs_scale_type = "f8e8m0fnu",
    absl::string_view rhs_scale_type = "f8e8m0fnu") {
  return ParseAndReturnUnverifiedModule(absl::StrReplaceAll(
      layout_template, {{"$lhs", lhs_type},
                        {"$rhs", rhs_type},
                        {"$lhs_scale", lhs_scale_type},
                        {"$rhs_scale", rhs_scale_type}}));
}

const HloScaledDotInstruction& GetScaledDot(const HloModule& module) {
  return *Cast<HloScaledDotInstruction>(
      hlo_query::GetFirstInstructionWithOpcode(*module.entry_computation(),
                                               HloOpcode::kScaledDot));
}

void ExpectSupported(absl::string_view layout, absl::string_view lhs_type,
                     absl::string_view rhs_type,
                     const se::CudaComputeCapability& cc, bool expected,
                     absl::string_view lhs_scale_type = "f8e8m0fnu",
                     absl::string_view rhs_scale_type = "f8e8m0fnu") {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseScaledDot(layout, lhs_type, rhs_type, lhs_scale_type,
                                  rhs_scale_type));
  EXPECT_EQ(IsTritonSupportedScaledDot(GetScaledDot(*module), cc), expected);
}

TEST(IsSupportedTest, RejectsFp4BeforeHopper) {
  ExpectSupported(kCanonical, "f4e2m1fn", "f4e2m1fn",
                  se::CudaComputeCapability::Ampere(), false);
}

TEST(IsSupportedTest, RejectsFp4OnOnlyLhs) {
  ExpectSupported(kCanonical, "f4e2m1fn", "f8e4m3fn",
                  se::CudaComputeCapability::Blackwell(), false);
}

TEST(IsSupportedTest, RejectsFp4OnOnlyRhs) {
  ExpectSupported(kCanonical, "f8e4m3fn", "f4e2m1fn",
                  se::CudaComputeCapability::Blackwell(), false);
}

TEST(IsSupportedTest, RejectsNvfp4OnHopper) {
  ExpectSupported(kTransposedRhs, "f4e2m1fn", "f4e2m1fn",
                  se::CudaComputeCapability::Hopper(), false, "f8e4m3fn",
                  "f8e4m3fn");
}

TEST(IsSupportedTest, RejectsNvfp4CanonicalLayout) {
  ExpectSupported(kCanonical, "f4e2m1fn", "f4e2m1fn",
                  se::CudaComputeCapability::Blackwell(), false, "f8e4m3fn",
                  "f8e4m3fn");
}

TEST(IsSupportedTest, AllowsNvfp4KPackedOperandsOnBlackwell) {
  ExpectSupported(kTransposedRhs, "f4e2m1fn", "f4e2m1fn",
                  se::CudaComputeCapability::Blackwell(), true, "f8e4m3fn",
                  "f8e4m3fn");
}

TEST(IsSupportedTest, RejectsScaleEncodingMismatch) {
  ExpectSupported(kCanonical, "f4e2m1fn", "f4e2m1fn",
                  se::CudaComputeCapability::Blackwell(), false, "f8e8m0fnu",
                  "f8e4m3fn");
}

TEST(IsSupportedTest, RejectsBlackwell10TransposedLhsFp4Layout) {
  constexpr absl::string_view kHloText = R"(
HloModule m
ENTRY e {
  lhs = $lhs[256,128] parameter(0)
  rhs = $rhs[256,128] parameter(1)
  lhs_scale = $lhs_scale[8,128] parameter(2)
  rhs_scale = $rhs_scale[8,128] parameter(3)
  ROOT _ = bf16[128,128] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
}
)";
  ExpectSupported(kHloText, "f4e2m1fn", "f4e2m1fn",
                  se::CudaComputeCapability::Blackwell(), false);
}

TEST(IsSupportedTest, RejectsBlackwell10TransposedRhsFp4Layout) {
  ExpectSupported(kTransposedRhs, "f4e2m1fn", "f4e2m1fn",
                  se::CudaComputeCapability::Blackwell(), false);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
