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

#include "xla/service/cpu/cpu_multi_output_fusion.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"

namespace op = xla::testing::opcode_matchers;

namespace xla::cpu {
namespace {

class MultiOutputFusionTest : public HloHardwareIndependentTestBase {
 protected:
  AliasInfo alias_info_;
};

TEST_F(MultiOutputFusionTest, TrivialReusedInput) {
  // The current implementation of the multi-output fusion pass only fuses when
  // one of the instructions is a fusion.
  // TODO(willfroom): Fix this to enable fusing of two un-fused instructions.
  static constexpr absl::string_view kTrivialReusedInput = R"(
    HloModule module

    %add_fn {
      %arg0 = f32[100] parameter(0)
      %arg1 = f32[100] parameter(1)
      ROOT %add = f32[100] add(%arg0, %arg1)
    }


    ENTRY %main (arg0: f32[100], arg1: f32[100]) -> (f32[100], f32[100]) {
      %arg0 = f32[100] parameter(0)
      %arg1 = f32[100] parameter(1)
      %double = f32[100] fusion(%arg0, %arg1), kind=kLoop, calls=%add_fn
      %square = f32[100] multiply(%arg0, %arg1)
      ROOT %result = (f32[100], f32[100]) tuple(%double, %square)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       ParseAndReturnVerifiedModule(kTrivialReusedInput));
  ASSERT_OK_AND_ASSIGN(
      bool changed, CpuMultiOutputFusion(&alias_info_).Run(hlo_module.get()));
  EXPECT_TRUE(changed);
  HloComputation* entry_computation = hlo_module->entry_computation();
  EXPECT_THAT(entry_computation->instructions(),
              ::testing::Contains(op::Fusion()).Times(1));
  auto fusion_match = op::Fusion(op::Parameter(0), op::Parameter(1));
  EXPECT_THAT(entry_computation->root_instruction(),
              op::Tuple(op::GetTupleElement(fusion_match),
                        op::GetTupleElement(fusion_match)));
}

// Regression test for the scan-body fusion gate in
// InstructionFusion::ShouldFuseIntoMultiOutput.
TEST_F(MultiOutputFusionTest, DoesNotFuseInsideScanBody) {
  static constexpr absl::string_view kHloModule = R"(
    HloModule module

    %scan_body {
      %carry = f32[] parameter(0)
      %input = f32[] parameter(1)
      %abs1 = f32[] abs(%input)
      %mul1 = f32[] multiply(%abs1, %abs1)
      %mul2 = f32[] multiply(%abs1, %carry)
      %next = f32[] add(%mul1, %mul2)
      ROOT %t = (f32[], f32[]) tuple(%next, %next)
    }

    ENTRY %main (input: f32[128], init: f32[]) -> (f32[128], f32[]) {
      %input = f32[128]{0} parameter(0)
      %init = f32[] parameter(1)
      ROOT %scan = (f32[128]{0}, f32[]) scan(%input, %init), dimensions={0},
          num_carries=1, is_reverse=false, to_apply=%scan_body,
          is_associative=true
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       ParseAndReturnVerifiedModule(kHloModule));
  ASSERT_OK_AND_ASSIGN(
      bool changed, CpuMultiOutputFusion(&alias_info_).Run(hlo_module.get()));
  EXPECT_FALSE(changed) << hlo_module->ToString();

  // Locate the scan body and verify it stays flat (no kFusion instructions).
  HloComputation* body = nullptr;
  for (HloComputation* c : hlo_module->MakeNonfusionComputations()) {
    if (c->name() == "scan_body") {
      body = c;
      break;
    }
  }
  ASSERT_NE(body, nullptr) << "Scan body computation not found";
  for (const HloInstruction* instr : body->instructions()) {
    EXPECT_NE(instr->opcode(), HloOpcode::kFusion)
        << "Found a kFusion inside a scan body: " << instr->ToString();
  }
}

TEST_F(MultiOutputFusionTest, DoesNotFuseInsideSortComparator) {
  static constexpr absl::string_view kHloModule = R"(
    HloModule module

    %cmp {
      %a = f32[] parameter(0)
      %b = f32[] parameter(1)
      %abs_a = f32[] abs(%a)
      %abs_b = f32[] abs(%b)
      %sum_a = f32[] add(%abs_a, %abs_a)
      %sum_b = f32[] add(%abs_b, %abs_b)
      ROOT %lt = pred[] compare(%sum_a, %sum_b), direction=LT
    }

    ENTRY %main (input: f32[128]) -> f32[128] {
      %input = f32[128]{0} parameter(0)
      ROOT %sorted = f32[128]{0} sort(%input), dimensions={0}, to_apply=%cmp
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       ParseAndReturnVerifiedModule(kHloModule));
  ASSERT_OK_AND_ASSIGN(
      bool changed, CpuMultiOutputFusion(&alias_info_).Run(hlo_module.get()));
  EXPECT_FALSE(changed) << hlo_module->ToString();

  HloComputation* body = nullptr;
  for (HloComputation* c : hlo_module->MakeNonfusionComputations()) {
    if (c->name() == "cmp") {
      body = c;
      break;
    }
  }
  ASSERT_NE(body, nullptr) << "Sort comparator computation not found";
  for (const HloInstruction* instr : body->instructions()) {
    EXPECT_NE(instr->opcode(), HloOpcode::kFusion)
        << "Found a kFusion inside a sort comparator: " << instr->ToString();
  }
}

// The pass fuses r into l, then x into y, then s into y. Fused into l, r stays
// behind with its control edge to s, so s does not depend on l although the
// merged reachability says so. A deferred merge of x and y would then make s
// depend on y, because s seems to depend on x, and keep s out of y. The merges
// after a fusion with control dependencies must be applied eagerly.
TEST_F(MultiOutputFusionTest, MergesAfterControlDependenciesStayExact) {
  static constexpr absl::string_view kHloModule = R"(
    HloModule module

    %x_fn {
      %p0 = f32[256] parameter(0)
      ROOT %negate = f32[256] negate(%p0)
    }

    %y_fn {
      %p0 = f32[256] parameter(0)
      %p1 = f32[16] parameter(1)
      %broadcast = f32[16,16] broadcast(%p1), dimensions={1}
      %reshape = f32[256] reshape(%broadcast)
      ROOT %add = f32[256] add(%p0, %reshape)
    }

    %l_fn {
      %p0 = f32[256] parameter(0)
      %p1 = f32[256] parameter(1)
      %p2 = f32[256] parameter(2)
      %add = f32[256] add(%p0, %p1)
      ROOT %add.1 = f32[256] add(%add, %p2)
    }

    %s_fn {
      %p0 = f32[16] parameter(0)
      %broadcast = f32[16,16] broadcast(%p0), dimensions={1}
      ROOT %reshape = f32[256] reshape(%broadcast)
    }

    ENTRY %main {
      %a = f32[256] parameter(0)
      %a2 = f32[256] parameter(1)
      %b = f32[256] parameter(2)
      %c = f32[16] parameter(3)
      %x = f32[256] fusion(%b), kind=kLoop, calls=%x_fn
      %l = f32[256] fusion(%a, %a2, %x), kind=kLoop, calls=%l_fn
      %r = f32[256] add(%a, %a2)
      %s = f32[256] fusion(%c), kind=kLoop, calls=%s_fn,
          control-predecessors={%r}
      %y = f32[256] fusion(%b, %c), kind=kLoop, calls=%y_fn
      ROOT %t = (f32[256], f32[256], f32[256], f32[256]) tuple(%l, %r, %s, %y)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       ParseAndReturnVerifiedModule(kHloModule));
  ASSERT_OK_AND_ASSIGN(
      bool changed, CpuMultiOutputFusion(&alias_info_).Run(hlo_module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::GetTupleElement(), op::GetTupleElement(),
                              op::GetTupleElement(), op::GetTupleElement()))
      << hlo_module->ToString();
  EXPECT_EQ(root->operand(2)->operand(0), root->operand(3)->operand(0))
      << hlo_module->ToString();
}

// The pass fuses g into f, then pp into q. Fused into f, g moves the control
// predecessor pp of its get-tuple-element g0 onto f, so f comes to depend on
// pp, which the merged reachability does not say. A deferred merge of pp and q
// would then leave f independent of q, and the pass would fuse all four into
// one fusion, dropping the order of pp before g0. The merges after such a
// fusion must be applied eagerly.
TEST_F(MultiOutputFusionTest, MergesAfterTupleControlDependenciesStayExact) {
  static constexpr absl::string_view kHloModule = R"(
    HloModule module

    %g_fn {
      %p0 = f32[1024] parameter(0)
      %slice = f32[256] slice(%p0), slice={[0:256]}
      %negate = f32[256] negate(%slice)
      %exponential = f32[256] exponential(%slice)
      ROOT %tuple = (f32[256], f32[256]) tuple(%negate, %exponential)
    }

    %f_fn {
      %p0 = f32[1024] parameter(0)
      %p1 = f32[256] parameter(1)
      %slice = f32[256] slice(%p0), slice={[256:512]}
      %add = f32[256] add(%slice, %p1)
      %sqrt = f32[256] sqrt(%slice)
      ROOT %tuple = (f32[256], f32[256]) tuple(%add, %sqrt)
    }

    %p_fn {
      %p0 = f32[512] parameter(0)
      %slice = f32[256] slice(%p0), slice={[0:256]}
      ROOT %negate = f32[256] negate(%slice)
    }

    %q_fn {
      %p0 = f32[512] parameter(0)
      %p1 = f32[256] parameter(1)
      %slice = f32[256] slice(%p0), slice={[256:512]}
      ROOT %multiply = f32[256] multiply(%slice, %p1)
    }

    ENTRY %main {
      %a = f32[1024] parameter(0)
      %b = f32[512] parameter(1)
      %c = f32[256] parameter(2)
      %pp = f32[256] fusion(%b), kind=kLoop, calls=%p_fn
      %g = (f32[256], f32[256]) fusion(%a), kind=kLoop, calls=%g_fn
      %g0 = f32[256] get-tuple-element(%g), index=0, control-predecessors={%pp}
      %g1 = f32[256] get-tuple-element(%g), index=1
      %f = (f32[256], f32[256]) fusion(%a, %c), kind=kLoop, calls=%f_fn
      %f0 = f32[256] get-tuple-element(%f), index=0, control-predecessors={%pp}
      %f1 = f32[256] get-tuple-element(%f), index=1
      %q = f32[256] fusion(%b, %c), kind=kLoop, calls=%q_fn
      ROOT %t = (f32[256], f32[256], f32[256], f32[256], f32[256], f32[256])
          tuple(%g0, %g1, %f0, %f1, %pp, %q)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       ParseAndReturnVerifiedModule(kHloModule));
  ASSERT_OK_AND_ASSIGN(
      bool changed, CpuMultiOutputFusion(&alias_info_).Run(hlo_module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  ASSERT_THAT(root, op::Tuple(op::GetTupleElement(), op::GetTupleElement(),
                              op::GetTupleElement(), op::GetTupleElement(),
                              op::GetTupleElement(), op::GetTupleElement()))
      << hlo_module->ToString();
  const HloInstruction* fg = root->operand(0)->operand(0);
  const HloInstruction* pq = root->operand(4)->operand(0);
  EXPECT_NE(fg, pq) << hlo_module->ToString();
  EXPECT_EQ(root->operand(2)->operand(0), fg) << hlo_module->ToString();
  EXPECT_EQ(root->operand(5)->operand(0), pq) << hlo_module->ToString();
  EXPECT_THAT(fg->control_predecessors(), ::testing::ElementsAre(pq))
      << hlo_module->ToString();
}

}  // namespace
}  // namespace xla::cpu
