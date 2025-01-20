/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/rematerialize_large_all_gather.h"

#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "tsl/platform/statusor.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

class RematerializeLargeAllGatherTest : public HloHardwareIndependentTestBase {
 protected:
  RematerializeLargeAllGatherTest() = default;
};

TEST_F(RematerializeLargeAllGatherTest, ReduceScatterDotPatternWithOptBarrier) {
  constexpr std::string_view hlo = R"(
HloModule main
%reduction {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}
ENTRY main {
  %param0 = f32[4096,4096] parameter(0)
  %param1 = f32[4096,4096] parameter(1)
  %dot = f32[4096,4096] dot(%param0, %param1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %reduce-scatter = f32[2048,4096] reduce-scatter(%dot), dimensions={0}, replica_groups={{0,1}}, to_apply=%reduction
  %multiply = f32[2048,4096] multiply(%reduce-scatter, %reduce-scatter)
  %all-gather = f32[4096,4096] all-gather(%multiply), replica_groups={{0,1}}, dimensions={0}
  %tuple = (f32[4096,4096]) tuple(%all-gather)
  %opt-barrier = (f32[4096,4096]) opt-barrier(%tuple)
  ROOT %gte = f32[4096,4096] get-tuple-element(%opt-barrier), index=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  RematerializeLargeAllGather pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  HloDCE dce;
  TF_RETURN_IF_ERROR(dce.Run(module.get()).status());
  EXPECT_TRUE(changed);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::AllGather(op::GetTupleElement(op::OptimizationBarrier())));
}

TEST_F(RematerializeLargeAllGatherTest,
       ReduceScatterDotPatternWithOptBarrierAndDisablePatternMatch) {
  constexpr std::string_view hlo = R"(
HloModule main
%reduction {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}
ENTRY main {
  %param0 = f32[4096,4096] parameter(0)
  %param1 = f32[4096,4096] parameter(1)
  %dot = f32[4096,4096] dot(%param0, %param1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %reduce-scatter = f32[2048,4096] reduce-scatter(%dot), dimensions={0}, replica_groups={{0,1}}, to_apply=%reduction
  %multiply = f32[2048,4096] multiply(%reduce-scatter, %reduce-scatter)
  %all-gather = f32[4096,4096] all-gather(%multiply), dimensions={0}
  %tuple = (f32[4096,4096]) tuple(%all-gather)
  %opt-barrier = (f32[4096,4096]) opt-barrier(%tuple)
  ROOT %gte = f32[4096,4096] get-tuple-element(%opt-barrier), index=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  RematerializeLargeAllGather pass(/*remat_size_in_bytes=*/4096 * 4096 * 2,
                                   /*disable_pattern_match=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  HloDCE dce;
  TF_RETURN_IF_ERROR(dce.Run(module.get()).status());
  EXPECT_TRUE(changed);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::AllGather(op::GetTupleElement(op::OptimizationBarrier())));
}

TEST_F(RematerializeLargeAllGatherTest,
       ReduceScatterDotPatternWithOptBarrierAndHighBytes) {
  constexpr std::string_view hlo = R"(
HloModule main
%reduction {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}
ENTRY main {
  %param0 = f32[4096,4096] parameter(0)
  %param1 = f32[4096,4096] parameter(1)
  %dot = f32[4096,4096] dot(%param0, %param1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %reduce-scatter = f32[2048,4096] reduce-scatter(%dot), dimensions={0}, replica_groups={{0,1}}, to_apply=%reduction
  %multiply = f32[2048,4096] multiply(%reduce-scatter, %reduce-scatter)
  %all-gather = f32[4096,4096] all-gather(%multiply), dimensions={0}
  %tuple = (f32[4096,4096]) tuple(%all-gather)
  %opt-barrier = (f32[4096,4096]) opt-barrier(%tuple)
  ROOT %gte = f32[4096,4096] get-tuple-element(%opt-barrier), index=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  RematerializeLargeAllGather pass(/*remat_size_in_bytes=*/(8192 * 8192 * 2),
                                   /*disable_pattern_match=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RematerializeLargeAllGatherTest,
       ReduceScatterDotPatternWithOptBarrierAndLowBytes) {
  constexpr std::string_view hlo = R"(
HloModule main
%reduction {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}
ENTRY main {
  %param0 = f32[4096,4096] parameter(0)
  %param1 = f32[4096,4096] parameter(1)
  %dot = f32[4096,4096] dot(%param0, %param1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %reduce-scatter = f32[2048,4096] reduce-scatter(%dot), dimensions={0}, replica_groups={{0,1}}, to_apply=%reduction
  %multiply = f32[2048,4096] multiply(%reduce-scatter, %reduce-scatter)
  %all-gather = f32[4096,4096] all-gather(%multiply), dimensions={0}
  %tuple = (f32[4096,4096]) tuple(%all-gather)
  %opt-barrier = (f32[4096,4096]) opt-barrier(%tuple)
  ROOT %gte = f32[4096,4096] get-tuple-element(%opt-barrier), index=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  RematerializeLargeAllGather pass(/*remat_size_in_bytes=*/1024,
                                   /*disable_pattern_match=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  HloDCE dce;
  TF_RETURN_IF_ERROR(dce.Run(module.get()).status());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::AllGather(op::GetTupleElement(op::OptimizationBarrier())));
}

TEST_F(RematerializeLargeAllGatherTest, NoOptBarrier) {
  constexpr std::string_view hlo = R"(
HloModule main
%reduction {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}
ENTRY main {
  %param0 = f32[4096,4096] parameter(0)
  %param1 = f32[4096,4096] parameter(1)
  %dot = f32[4096,4096] dot(%param0, %param1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %reduce-scatter = f32[2048,4096] reduce-scatter(%dot), dimensions={0}, replica_groups={{0,1}}, to_apply=%reduction
  %multiply = f32[2048,4096] multiply(%reduce-scatter, %reduce-scatter)
  ROOT %all-gather = f32[4096,4096] all-gather(%multiply), dimensions={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  RematerializeLargeAllGather pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RematerializeLargeAllGatherTest,
       NoReduceScatterDotPatternFallbackToDefaultRematSize) {
  constexpr std::string_view hlo = R"(
HloModule main
ENTRY main {
  %param0 = f32[4096,4096] parameter(0)
  %all-gather = f32[8192,4096] all-gather(%param0), dimensions={0}
  %tuple = (f32[8192,4096]) tuple(%all-gather)
  %opt-barrier = (f32[8192,4096]) opt-barrier(%tuple)
  ROOT %gte = f32[8192,4096] get-tuple-element(%opt-barrier), index=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  RematerializeLargeAllGather pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  HloDCE dce;
  TF_RETURN_IF_ERROR(dce.Run(module.get()).status());
  EXPECT_TRUE(changed);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::AllGather(op::GetTupleElement(op::OptimizationBarrier())));
}

}  // namespace
}  // namespace xla