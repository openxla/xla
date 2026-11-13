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

#include "xla/backends/gpu/transforms/cudnn_custom_reduction_analysis.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

std::string ScalarReduction(absl::string_view type, absl::string_view operation,
                            absl::string_view identity) {
  return absl::Substitute(R"hlo(
HloModule scalar_reduce
body {
  lhs = $0[] parameter(0)
  rhs = $0[] parameter(1)
  ROOT value = $0[] $1(lhs, rhs)
}
ENTRY entry {
  input = $0[2,129,5] parameter(0)
  init = $0[] constant($2)
  ROOT reduction = $0[2,5] reduce(input, init), dimensions={1}, to_apply=body
})hlo",
                          type, operation, identity);
}

// The JAX argmax: the first NaN, or else the first maximum.
constexpr absl::string_view kArgMax = R"hlo(
HloModule arg_reduce
body {
  lhs = f32[] parameter(0)
  li = s32[] parameter(1)
  rhs = f32[] parameter(2)
  ri = s32[] parameter(3)
  better = pred[] compare(lhs, rhs), direction=GT
  lhs_nan = pred[] compare(lhs, lhs), direction=NE
  keep_value = pred[] or(better, lhs_nan)
  value = f32[] select(keep_value, lhs, rhs)
  equal = pred[] compare(lhs, rhs), direction=EQ
  lower = pred[] compare(li, ri), direction=LT
  tie = pred[] and(equal, lower)
  keep_index = pred[] or(keep_value, tie)
  index = s32[] select(keep_index, li, ri)
  ROOT pair = (f32[], s32[]) tuple(value, index)
}
ENTRY entry {
  input = f32[2,129] parameter(0)
  indices = s32[2,129] iota(), iota_dimension=1
  init = f32[] constant(-inf)
  zero = s32[] constant(0)
  reduction = (f32[2], s32[2]) reduce(input, indices, init, zero), dimensions={1}, to_apply=body
  ROOT result = s32[2] get-tuple-element(reduction), index=1
})hlo";

std::string PairReduction(absl::string_view index_identity = "constant(0)") {
  return absl::Substitute(R"hlo(
HloModule pair_reduce
body {
  lhs = f32[] parameter(0)
  li = s32[] parameter(1)
  rhs = f32[] parameter(2)
  ri = s32[] parameter(3)
  value = f32[] add(lhs, rhs)
  index = s32[] add(li, ri)
  ROOT pair = (f32[], s32[]) tuple(value, index)
}
ENTRY entry {
  input = f32[2,129] parameter(0)
  indices = s32[2,129] parameter(1)
  init = f32[] constant(0)
  izero = s32[] $0
  reduction = (f32[2], s32[2]) reduce(input, indices, init, izero), dimensions={1}, to_apply=body
  ROOT result = f32[2] get-tuple-element(reduction), index=0
})hlo",
                          index_identity);
}

constexpr absl::string_view kSaturatingAdd = R"hlo(
HloModule saturated_sum
body {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  sum = s32[] add(lhs, rhs)
  cap = s32[] constant(127)
  ROOT value = s32[] minimum(sum, cap)
}
ENTRY entry {
  input = s32[2,129] parameter(0)
  init = s32[] constant(0)
  ROOT reduction = s32[2] reduce(input, init), dimensions={1}, to_apply=body
})hlo";

using CudnnCustomReductionAnalysisTest = HloHardwareIndependentTestBase;

TEST_F(CudnnCustomReductionAnalysisTest, FrontendStreamTypesAreAccepted) {
  for (const auto& [type, operation, identity] :
       std::vector<std::tuple<std::string, std::string, std::string>>{
           {"f32", "add", "0"},
           {"f32", "multiply", "1"},
           {"f32", "maximum", "-inf"},
           {"f32", "minimum", "inf"},
           {"f16", "add", "0"},
           {"bf16", "maximum", "-inf"},
           {"f64", "add", "0"},
           {"f8e4m3fn", "add", "0"},
           {"f8e5m2", "minimum", "inf"},
           {"s8", "add", "0"},
           {"s32", "minimum", "2147483647"},
           {"s64", "add", "0"},
           {"pred", "and", "true"},
           {"pred", "or", "false"},
           {"pred", "xor", "false"}}) {
    SCOPED_TRACE(type + ":" + operation);
    ASSERT_OK_AND_ASSIGN(auto module,
                         ParseAndReturnVerifiedModule(
                             ScalarReduction(type, operation, identity)));
    EXPECT_THAT(AnalyzeCudnnCustomReduction(
                    *module->entry_computation()->root_instruction()),
                IsOk());
  }
}

TEST_F(CudnnCustomReductionAnalysisTest,
       StreamTypesWithoutIdentityTypeMappingAreRefused) {
  // The frontend has no S16, U16 or U64 identity. The cuDNN fusion compiler
  // does not map U8 or U32 tensors.
  for (const std::string type : {"s16", "u16", "u64", "u8", "u32"}) {
    SCOPED_TRACE(type);
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                          ScalarReduction(type, "add", "0")));
    EXPECT_THAT(AnalyzeCudnnCustomReduction(
                    *module->entry_computation()->root_instruction()),
                StatusIs(absl::StatusCode::kUnimplemented,
                         HasSubstr("stream type " + type +
                                   " has no cuDNN identity type mapping")));
  }
}

TEST_F(CudnnCustomReductionAnalysisTest, MultiOperationCustomBodyIsAccepted) {
  // min(a + b, 127) is not associative for all s32 inputs (for example,
  // negative inputs). The analysis does not check associativity.
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kSaturatingAdd));
  EXPECT_THAT(AnalyzeCudnnCustomReduction(
                  *module->entry_computation()->root_instruction()),
              IsOk());
}

TEST_F(CudnnCustomReductionAnalysisTest, ClampBodyIsAccepted) {
  ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(absl::StrReplaceAll(
          kSaturatingAdd, {{"ROOT value = s32[] minimum(sum, cap)",
                            "floor = s32[] constant(0)\n"
                            "  ROOT value = s32[] clamp(floor, sum, cap)"}})));
  EXPECT_THAT(AnalyzeCudnnCustomReduction(
                  *module->entry_computation()->root_instruction()),
              IsOk());
}

TEST_F(CudnnCustomReductionAnalysisTest,
       InBodyConversionKeepsTheExistingXlaPath) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"hlo(
HloModule converted_sum
body {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  wide_lhs = f32[] convert(lhs)
  wide_rhs = f32[] convert(rhs)
  sum = f32[] add(wide_lhs, wide_rhs)
  ROOT value = bf16[] convert(sum)
}
ENTRY entry {
  input = bf16[2,129] parameter(0)
  init = bf16[] constant(0)
  ROOT reduction = bf16[2] reduce(input, init), dimensions={1}, to_apply=body
})hlo"));
  EXPECT_THAT(AnalyzeCudnnCustomReduction(
                  *module->entry_computation()->root_instruction()),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("scalar custom reduction body")));
}

TEST_F(CudnnCustomReductionAnalysisTest,
       UnsortedMultiAxisAndScalarResultAreAccepted) {
  const std::string text =
      absl::StrReplaceAll(ScalarReduction("f32", "add", "0"),
                          {{"f32[2,5] reduce", "f32[] reduce"},
                           {"dimensions={1}", "dimensions={2,0,1}"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));
  EXPECT_THAT(AnalyzeCudnnCustomReduction(
                  *module->entry_computation()->root_instruction()),
              IsOk());
}

TEST_F(CudnnCustomReductionAnalysisTest, TotalOrderFloatCompareIsRefused) {
  ASSERT_OK_AND_ASSIGN(auto control, ParseAndReturnVerifiedModule(kArgMax));
  EXPECT_THAT(
      AnalyzeCudnnCustomReduction(*FindInstruction(control.get(), "reduction")),
      IsOk());
  // The cuDNN frontend compares floating-point values in IEEE order only.
  for (const std::string direction : {"GT", "NE", "EQ"}) {
    SCOPED_TRACE(direction);
    const std::string before = "direction=" + direction;
    ASSERT_OK_AND_ASSIGN(auto module,
                         ParseAndReturnVerifiedModule(absl::StrReplaceAll(
                             kArgMax, {{before, before + ", order=TOTAL"}})));
    EXPECT_THAT(AnalyzeCudnnCustomReduction(
                    *FindInstruction(module.get(), "reduction")),
                StatusIs(absl::StatusCode::kUnimplemented,
                         HasSubstr("scalar custom reduction body")));
  }
}

TEST_F(CudnnCustomReductionAnalysisTest,
       RuntimeIdentityInLaterStreamIsRefused) {
  ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(PairReduction("parameter(2)")));
  EXPECT_THAT(
      AnalyzeCudnnCustomReduction(*FindInstruction(module.get(), "reduction")),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("typed constant scalar identity")));
}

TEST_F(CudnnCustomReductionAnalysisTest,
       MalformedLaterStreamFieldsAreSafelyRefused) {
  for (const auto& [before, after, reason] :
       std::vector<std::tuple<std::string, std::string, std::string>>{
           {"indices = s32[2,129] parameter(1)",
            "indices = s32[3,129] parameter(1)", "input shape"},
           {"izero = s32[] constant(0)", "izero = f32[] constant(0)",
            "typed constant scalar identity"},
           {"reduction = (f32[2], s32[2])", "reduction = (f32[2], s64[2])",
            "result shape or type"}}) {
    SCOPED_TRACE(after);
    ASSERT_OK_AND_ASSIGN(auto module,
                         ParseAndReturnUnverifiedModule(absl::StrReplaceAll(
                             PairReduction(), {{before, after}})));
    EXPECT_THAT(AnalyzeCudnnCustomReduction(
                    *FindInstruction(module.get(), "reduction")),
                StatusIs(absl::StatusCode::kUnimplemented, HasSubstr(reason)));
  }
}

TEST_F(CudnnCustomReductionAnalysisTest, UnsupportedScalarBodyIsRefused) {
  // The cuDNN frontend has no bitwise integer modes.
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        ScalarReduction("s32", "xor", "0")));
  EXPECT_THAT(AnalyzeCudnnCustomReduction(
                  *module->entry_computation()->root_instruction()),
              StatusIs(absl::StatusCode::kUnimplemented,
                       AllOf(HasSubstr("scalar custom reduction body"),
                             HasSubstr("xor("))));
}

TEST_F(CudnnCustomReductionAnalysisTest,
       BodyConstantsWithoutFrontendConstantTypeAreRefused) {
  for (const auto& [type, accepted] : std::vector<std::pair<std::string, bool>>{
           {"s64", true}, {"f64", true}, {"s8", false}}) {
    SCOPED_TRACE(type);
    // The cap is a reducer constant of the stream type.
    ASSERT_OK_AND_ASSIGN(
        auto module, ParseAndReturnVerifiedModule(
                         absl::StrReplaceAll(kSaturatingAdd, {{"s32", type}})));
    const absl::Status status = AnalyzeCudnnCustomReduction(
        *module->entry_computation()->root_instruction());
    if (accepted) {
      EXPECT_THAT(status, IsOk());
    } else {
      EXPECT_THAT(status, StatusIs(absl::StatusCode::kUnimplemented,
                                   HasSubstr("scalar custom reduction body")));
    }
  }
  // The frontend has no boolean compile-time constant.
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"hlo(
HloModule predicate_constant
body {
  lhs = pred[] parameter(0)
  rhs = pred[] parameter(1)
  either = pred[] or(lhs, rhs)
  always = pred[] constant(true)
  ROOT value = pred[] and(either, always)
}
ENTRY entry {
  input = pred[2,129] parameter(0)
  init = pred[] constant(false)
  ROOT reduction = pred[2] reduce(input, init), dimensions={1}, to_apply=body
})hlo"));
  EXPECT_THAT(AnalyzeCudnnCustomReduction(
                  *module->entry_computation()->root_instruction()),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("scalar custom reduction body")));
}

TEST_F(CudnnCustomReductionAnalysisTest, EmptyAxesAreRefused) {
  const std::string text =
      absl::StrReplaceAll(ScalarReduction("f32", "add", "0"),
                          {{"f32[2,5] reduce", "f32[2,129,5] reduce"},
                           {"dimensions={1}", "dimensions={}"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));
  EXPECT_THAT(
      AnalyzeCudnnCustomReduction(
          *module->entry_computation()->root_instruction()),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("nonempty axes")));
}

TEST_F(CudnnCustomReductionAnalysisTest, ZeroAndDynamicInputExtentsAreRefused) {
  for (const std::string shape : {"[2,0,5]", "[2,<=129,5]"}) {
    SCOPED_TRACE(shape);
    ASSERT_OK_AND_ASSIGN(
        auto module,
        ParseAndReturnVerifiedModule(absl::StrReplaceAll(
            ScalarReduction("f32", "add", "0"), {{"[2,129,5]", shape}})));
    EXPECT_THAT(
        AnalyzeCudnnCustomReduction(
            *module->entry_computation()->root_instruction()),
        StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("input shape")));
  }
}

TEST_F(CudnnCustomReductionAnalysisTest, InvalidAxisFormsAreSafelyRefused) {
  for (const std::string axes : {"1,1", "-1", "3"}) {
    SCOPED_TRACE(axes);
    // Intentionally bypass verification: this directly probes the public
    // analysis helper's refusal of malformed HLO before indexing any axis.
    ASSERT_OK_AND_ASSIGN(
        auto module, ParseAndReturnUnverifiedModule(absl::StrReplaceAll(
                         ScalarReduction("f32", "add", "0"),
                         {{"dimensions={1}", "dimensions={" + axes + "}"}})));
    EXPECT_THAT(AnalyzeCudnnCustomReduction(
                    *module->entry_computation()->root_instruction()),
                StatusIs(absl::StatusCode::kUnimplemented,
                         HasSubstr("distinct input dimensions")));
  }
}

TEST_F(CudnnCustomReductionAnalysisTest,
       SemanticKeyPreservesConstantsAndIgnoresNames) {
  ASSERT_OK_AND_ASSIGN(auto original,
                       ParseAndReturnVerifiedModule(kSaturatingAdd));
  ASSERT_OK_AND_ASSIGN(auto renamed,
                       ParseAndReturnVerifiedModule(absl::StrReplaceAll(
                           kSaturatingAdd, {{"body", "renamed_body"},
                                            {"lhs", "left"},
                                            {"rhs", "right"},
                                            {"cap", "limit"}})));
  ASSERT_OK_AND_ASSIGN(
      auto changed, ParseAndReturnVerifiedModule(absl::StrReplaceAll(
                        kSaturatingAdd, {{"constant(127)", "constant(128)"}})));
  const auto key = [](const HloModule& module) {
    return CudnnCustomReductionSemanticKey(
        *module.entry_computation()->root_instruction()->to_apply());
  };
  EXPECT_EQ(key(*original), key(*renamed));
  EXPECT_NE(key(*original), key(*changed));
}

TEST_F(CudnnCustomReductionAnalysisTest,
       SemanticKeyPreservesFloatConstantStorageBits) {
  const std::string text = absl::StrReplaceAll(
      kSaturatingAdd, {{"s32", "f32"}, {"constant(127)", "constant(nan)"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));
  HloInstruction* reduction = module->entry_computation()->root_instruction();
  HloConstantInstruction* constant =
      Cast<HloConstantInstruction>(FindInstruction(module.get(), "cap"));
  constant->mutable_literal()->Set<float>(
      {}, absl::bit_cast<float>(uint32_t{0x7fc00001}));
  const std::string first =
      CudnnCustomReductionSemanticKey(*reduction->to_apply());
  constant->mutable_literal()->Set<float>(
      {}, absl::bit_cast<float>(uint32_t{0x7fc00002}));
  EXPECT_NE(first, CudnnCustomReductionSemanticKey(*reduction->to_apply()));
  constant->mutable_literal()->Set<float>({}, 0.0f);
  const std::string positive_zero =
      CudnnCustomReductionSemanticKey(*reduction->to_apply());
  constant->mutable_literal()->Set<float>({}, -0.0f);
  EXPECT_NE(positive_zero,
            CudnnCustomReductionSemanticKey(*reduction->to_apply()));
}

}  // namespace
}  // namespace xla::gpu
