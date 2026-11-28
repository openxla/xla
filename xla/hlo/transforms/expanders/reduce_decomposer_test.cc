/* Copyright 2022 The OpenXLA Authors.

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
#include "xla/hlo/transforms/expanders/reduce_decomposer.h"

#include <gtest/gtest.h>

#include <optional>
#include <vector>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"

namespace xla {
namespace {

class ReduceDecomposerTest : public HloHardwareIndependentTestBase {};

TEST_F(ReduceDecomposerTest, ReducePerformsTransposition) {
  // Reshape is already a bitcast, nothing should be changed.
  const char* hlo = R"(
HloModule module

add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT out = add(a, b)
}

ENTRY c {
    p = f32[5,3,4]{2,1,0} parameter(0)
    z = f32[] constant(0)
    ROOT r = f32[5,4]{0,1} reduce(p, z), dimensions={1}, to_apply=add, metadata={op_name="reduce"}
}
)";

  RunAndFilecheckHloRewrite(
      hlo,
      ReduceDecomposer{/*custom_layout_allowed=*/[&](const HloInstruction*) {
        return true;
      }},
      std::nullopt);

  RunAndFilecheckHloRewrite(hlo, ReduceDecomposer{},
                            R"(
// CHECK: [[reduce_0:%[^ ]+]] = f32[5,4]{1,0} reduce([[p_1:%[^ ]+]], [[z_2:%[^ ]+]]), dimensions={1}, to_apply=[[add_3:%[^ ]+]], metadata={op_name="reduce"}
// CHECK-NEXT: ROOT [[copy_4:%[^ ]+]] = f32[5,4]{0,1} copy([[reduce_0]])
      )");
}

TEST_F(ReduceDecomposerTest, ReducePerformsTranspositionAfterConvert) {
  const char* hlo = R"(
add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT out = add(a, b)
}

ENTRY c {
    p = f32[5,3,4]{2,1,0} parameter(0)
    z = f32[] constant(0)
    r = f32[5,4]{0,1} reduce(p, z), dimensions={1}, to_apply=add, metadata={op_name="reduce"}
    ROOT c = bf16[5,4]{0,1} convert(r), metadata={op_name="convert"}
})";

  RunAndFilecheckHloRewrite(
      hlo,
      ReduceDecomposer{/*custom_layout_allowed=*/[&](const HloInstruction*) {
        return true;
      }},
      std::nullopt);

  RunAndFilecheckHloRewrite(hlo, ReduceDecomposer{},
                            R"(
// CHECK: [[reduce:%[^ ]+]] = f32[5,4]{1,0} reduce([[p_1:%[^ ]+]], [[z_2:%[^ ]+]]), dimensions={1}, to_apply=[[add:%[^ ]+]], metadata={op_name="reduce"}
// CHECK-NEXT: [[convert:%[^ ]+]] = bf16[5,4]{1,0} convert([[reduce]]), metadata={op_name="convert"}
// CHECK-NEXT: ROOT [[copy:%[^ ]+]] = bf16[5,4]{0,1} copy([[convert]])
      )");
}

TEST_F(ReduceDecomposerTest, ReduceNaturalLayout) {
  // Reshape is already a bitcast, nothing should be changed.
  const char* hlo = R"(
HloModule module

add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT out = add(a, b)
}

ENTRY c {
    p = f32[5,3,4]{2,1,0} parameter(0)
    z = f32[] constant(0)
    ROOT r = reduce(p, z), dimensions={1}, to_apply=add
}
)";

  RunAndFilecheckHloRewrite(hlo, ReduceDecomposer{}, std::nullopt);
}

TEST_F(ReduceDecomposerTest, VariadicReductionWithTranspose) {
  const char* hlo = R"(
HloModule ReduceWithLayoutChangeVariadicDifferent

argmax {
  running_max = f32[] parameter(0)
  running_max_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  current = (f32[], u32[]) tuple(running_max, running_max_idx)
  potential = (f32[], u32[]) tuple(current_value, current_value_idx)

  cmp_code = pred[] compare(current_value, running_max), direction=GT

  new_max = f32[] select(cmp_code, current_value, running_max)
  new_idx = u32[] select(cmp_code, current_value_idx, running_max_idx)

  ROOT out = (f32[], u32[]) tuple(new_max, new_idx)
}

ENTRY main {
  arg0 = f32[2,3,4,1024]{3,2,1,0}  parameter(0)
  idxs = u32[2,3,4,1024]{3,2,1,0}  parameter(1)
  constant0 = f32[] constant(0)
  constant1 = u32[] constant(0)
  ROOT reduce0 = (
      f32[2,3,4]{0,1,2},
      u32[2,3,4]{0,1,2}
    ) reduce(arg0, idxs, constant0,constant1), dimensions={3}, to_apply=argmax
}
  )";

  RunAndFilecheckHloRewrite(hlo, ReduceDecomposer{},
                            R"(
// CHECK:  [[_reduce_0:%[^ ]+]] = (f32[2,3,4]{2,1,0}, u32[2,3,4]{2,1,0}) reduce([[_arg0_1:%[^ ]+]], [[_idxs_2:%[^ ]+]], [[_constant0_3:%[^ ]+]], [[_constant1_4:%[^ ]+]]), dimensions={3}, to_apply=[[_argmax_5:%[^ ]+]]
// CHECK-NEXT:  [[_get_tuple_element_6:%[^ ]+]] = f32[2,3,4]{2,1,0} get-tuple-element([[_reduce_0]]), index=0
// CHECK-NEXT:  [[_copy_7:%[^ ]+]] = f32[2,3,4]{0,1,2} copy([[_get_tuple_element_6]])
// CHECK-NEXT:  [[_get_tuple_element_1_8:%[^ ]+]] = u32[2,3,4]{2,1,0} get-tuple-element([[_reduce_0]]), index=1
// CHECK-NEXT:  [[_copy_1_9:%[^ ]+]] = u32[2,3,4]{0,1,2} copy([[_get_tuple_element_1_8]])
// CHECK-NEXT:  ROOT [[_tuple_10:%[^ ]+]] = (f32[2,3,4]{0,1,2}, u32[2,3,4]{0,1,2}) tuple([[_copy_7]], [[_copy_1_9]])
                        )");
}

TEST_F(ReduceDecomposerTest, VariadicReductionDescendingLayout) {
  const char* hlo = R"(
HloModule ReduceWithLayoutChangeVariadicDifferent

argmax {
  running_max = f32[] parameter(0)
  running_max_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  current = (f32[], u32[]) tuple(running_max, running_max_idx)
  potential = (f32[], u32[]) tuple(current_value, current_value_idx)

  cmp_code = pred[] compare(current_value, running_max), direction=GT

  new_max = f32[] select(cmp_code, current_value, running_max)
  new_idx = u32[] select(cmp_code, current_value_idx, running_max_idx)

  ROOT out = (f32[], u32[]) tuple(new_max, new_idx)
}

ENTRY main {
  arg0 = f32[2,3,4,1024]{3,2,1,0}  parameter(0)
  idxs = u32[2,3,4,1024]{3,2,1,0}  parameter(1)
  constant0 = f32[] constant(0)
  constant1 = u32[] constant(0)
  ROOT reduce0 = (
      f32[2,3,4]{2,1,0},
      u32[2,3,4]{2,1,0}
    ) reduce(arg0, idxs, constant0,constant1), dimensions={3}, to_apply=argmax
}
  )";

  RunAndFilecheckHloRewrite(hlo, ReduceDecomposer{}, std::nullopt);
}

TEST_F(ReduceDecomposerTest, VariadicReductionInputsDifferentLayout) {
  const char* hlo = R"(
HloModule ReduceWithLayoutChangeVariadicDifferent

argmax {
  running_max = f32[] parameter(0)
  running_max_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  current = (f32[], u32[]) tuple(running_max, running_max_idx)
  potential = (f32[], u32[]) tuple(current_value, current_value_idx)

  cmp_code = pred[] compare(current_value, running_max), direction=GT

  new_max = f32[] select(cmp_code, current_value, running_max)
  new_idx = u32[] select(cmp_code, current_value_idx, running_max_idx)

  ROOT out = (f32[], u32[]) tuple(new_max, new_idx)
}

ENTRY main {
  arg0 = f32[2,3,4,1024]{3,2,1,0}  parameter(0)
  idxs = u32[2,3,4,1024]{2,1,3,0}  parameter(1)
  constant0 = f32[] constant(0)
  constant1 = u32[] constant(0)
  ROOT reduce0 = (
      f32[2,3,4]{2,1,0},
      u32[2,3,4]{2,1,0}
    ) reduce(arg0, idxs, constant0,constant1), dimensions={3}, to_apply=argmax
}
  )";

  RunAndFilecheckHloRewrite(hlo, ReduceDecomposer{}, R"(
// CHECK: [[arg0_0:%[^ ]+]] = f32[2,3,4,1024]{3,2,1,0} parameter(0)
// CHECK: [[idxs_1:%[^ ]+]] = u32[2,3,4,1024]{2,1,3,0} parameter(1)
// CHECK: [[copy_2:%[^ ]+]] = u32[2,3,4,1024]{3,2,1,0} copy([[idxs_1]])
// CHECK: [[constant0_3:%[^ ]+]] = f32[] constant(0)
// CHECK: [[constant1_4:%[^ ]+]] = u32[] constant(0)
// CHECK: ROOT [[reduce0_5:%[^ ]+]] = (f32[2,3,4]{2,1,0}, u32[2,3,4]{2,1,0}) reduce([[arg0_0]], [[copy_2]], [[constant0_3]], [[constant1_4]]), dimensions={3}, to_apply=[[argmax_6:%[^ ]+]]
)");
}

class LayoutSensitiveReduceDecomposerTest
    : public HloHardwareIndependentTestBase {
 protected:
  LayoutSensitiveReduceDecomposerTest()
      : HloHardwareIndependentTestBase(/*verifier_layout_sensitive=*/true) {}
};

TEST_F(LayoutSensitiveReduceDecomposerTest, DivergentVariadicOutputLayouts) {
  // Regression test for b/591175249: leaf 0 already has the natural {1,0}
  // layout (no copy) while leaf 1 requires {0,1} (copied).
  constexpr const char* kHlo = R"(
HloModule DivergentVariadicOutputs

argmax {
  v0 = f32[] parameter(0)
  i0 = s32[] parameter(1)
  v1 = f32[] parameter(2)
  i1 = s32[] parameter(3)
  gt = pred[] compare(v1, v0), direction=GT
  ROOT out = (f32[], s32[]) tuple(
      f32[] select(gt, v1, v0), s32[] select(gt, i1, i0))
}

ENTRY main {
  p0 = f32[3,3,3]{2,1,0} parameter(0)
  p1 = s32[3,3,3]{2,1,0} parameter(1)
  c0 = f32[] constant(-inf)
  c1 = s32[] constant(0)
  ROOT r = (f32[3,3]{1,0}, s32[3,3]{0,1}) reduce(p0, p1, c0, c1),
      dimensions={0}, to_apply=argmax
}
)";
  RunAndFilecheckHloRewrite(kHlo, ReduceDecomposer{}, R"(
// CHECK: [[RED:%[^ ]+]] = (f32[3,3]{1,0}, s32[3,3]{1,0}) reduce(
// CHECK-NEXT: [[GTE0:%[^ ]+]] = f32[3,3]{1,0} get-tuple-element([[RED]]), index=0
// CHECK-NEXT: [[GTE1:%[^ ]+]] = s32[3,3]{1,0} get-tuple-element([[RED]]), index=1
// CHECK-NEXT: [[CPY1:%[^ ]+]] = s32[3,3]{0,1} copy([[GTE1]])
// CHECK-NEXT: ROOT {{%[^ ]+}} = (f32[3,3]{1,0}, s32[3,3]{0,1}) tuple([[GTE0]], [[CPY1]])
)");
}

TEST_F(LayoutSensitiveReduceDecomposerTest,
       VariadicInputEqualizationPreservesOutputLayouts) {
  constexpr const char* kHlo = R"(
HloModule VariadicEqualizeKeepOutputs

argmax {
  v0 = f32[] parameter(0)
  i0 = u32[] parameter(1)
  v1 = f32[] parameter(2)
  i1 = u32[] parameter(3)
  gt = pred[] compare(v1, v0), direction=GT
  ROOT out = (f32[], u32[]) tuple(
      f32[] select(gt, v1, v0), u32[] select(gt, i1, i0))
}

ENTRY main {
  arg0 = f32[2,3,4,8]{3,2,1,0} parameter(0)
  idxs = u32[2,3,4,8]{2,1,3,0} parameter(1)
  c0 = f32[] constant(0)
  c1 = u32[] constant(0)
  r0 = (f32[2,3,4]{0,1,2}, u32[2,3,4]{0,1,2}) reduce(arg0, idxs, c0, c1),
      dimensions={3}, to_apply=argmax
  r1 = (f32[2,3,4]{2,1,0}, u32[2,3,4]{0,1,2}) reduce(arg0, idxs, c0, c1),
      dimensions={3}, to_apply=argmax
  ROOT out = ((f32[2,3,4]{0,1,2}, u32[2,3,4]{0,1,2}),
              (f32[2,3,4]{2,1,0}, u32[2,3,4]{0,1,2})) tuple(r0, r1)
}
)";
  RunAndFilecheckHloRewrite(kHlo, ReduceDecomposer{}, R"(
// CHECK: [[IC0:%[^ ]+]] = u32[2,3,4,8]{3,2,1,0} copy(
// CHECK: [[R0:%[^ ]+]] = (f32[2,3,4]{2,1,0}, u32[2,3,4]{2,1,0}) reduce({{%[^ ]+}}, [[IC0]],
// CHECK-NEXT: [[R0G0:%[^ ]+]] = f32[2,3,4]{2,1,0} get-tuple-element([[R0]]), index=0
// CHECK-NEXT: [[R0C0:%[^ ]+]] = f32[2,3,4]{0,1,2} copy([[R0G0]])
// CHECK-NEXT: [[R0G1:%[^ ]+]] = u32[2,3,4]{2,1,0} get-tuple-element([[R0]]), index=1
// CHECK-NEXT: [[R0C1:%[^ ]+]] = u32[2,3,4]{0,1,2} copy([[R0G1]])
// CHECK-NEXT: {{%[^ ]+}} = (f32[2,3,4]{0,1,2}, u32[2,3,4]{0,1,2}) tuple([[R0C0]], [[R0C1]])
// CHECK: [[R1:%[^ ]+]] = (f32[2,3,4]{2,1,0}, u32[2,3,4]{2,1,0}) reduce(
// CHECK-NEXT: [[R1G0:%[^ ]+]] = f32[2,3,4]{2,1,0} get-tuple-element([[R1]]), index=0
// CHECK-NEXT: [[R1G1:%[^ ]+]] = u32[2,3,4]{2,1,0} get-tuple-element([[R1]]), index=1
// CHECK-NEXT: [[R1C1:%[^ ]+]] = u32[2,3,4]{0,1,2} copy([[R1G1]])
// CHECK-NEXT: {{%[^ ]+}} = (f32[2,3,4]{2,1,0}, u32[2,3,4]{0,1,2}) tuple([[R1G0]], [[R1C1]])
)");

  // Equalizer still aligns input minor-to-major without wiping outputs when
  // custom_layout_allowed returns true.
  RunAndFilecheckHloRewrite(
      kHlo, ReduceDecomposer{[](const HloInstruction*) { return true; }}, R"(
// CHECK: = (f32[2,3,4]{0,1,2}, u32[2,3,4]{0,1,2}) reduce(
// CHECK: = (f32[2,3,4]{2,1,0}, u32[2,3,4]{0,1,2}) reduce(
)");
}

TEST_F(LayoutSensitiveReduceDecomposerTest,
       VariadicSubByteMemorySpaceDynamicAndTiles) {
  // Identical minor-to-major with differing :E(4) is a no-op in both orders.
  constexpr const char* kNoopHlo = R"(
HloModule SubByteVariadicNoop

red_a {
  a0 = s4[] parameter(0)
  b0 = s32[] parameter(1)
  a1 = s4[] parameter(2)
  b1 = s32[] parameter(3)
  ROOT t = (s4[], s32[]) tuple(s4[] add(a0, a1), s32[] add(b0, b1))
}

red_b {
  b0 = s32[] parameter(0)
  a0 = s4[] parameter(1)
  b1 = s32[] parameter(2)
  a1 = s4[] parameter(3)
  ROOT t = (s32[], s4[]) tuple(s32[] add(b0, b1), s4[] add(a0, a1))
}

ENTRY main {
  p_s4 = s4[4,8]{1,0:E(4)} parameter(0)
  p_s32 = s32[4,8]{1,0} parameter(1)
  z4 = s4[] constant(0)
  z32 = s32[] constant(0)
  r0 = (s4[4]{0:E(4)}, s32[4]{0}) reduce(p_s4, p_s32, z4, z32),
      dimensions={1}, to_apply=red_a
  r1 = (s32[4]{0}, s4[4]{0:E(4)}) reduce(p_s32, p_s4, z32, z4),
      dimensions={1}, to_apply=red_b
  ROOT out = ((s4[4]{0:E(4)}, s32[4]{0}), (s32[4]{0}, s4[4]{0:E(4)}))
      tuple(r0, r1)
}
)";
  RunAndFilecheckHloRewrite(kNoopHlo, ReduceDecomposer{}, std::nullopt);

  // Equalizing a transposed input preserves :E(4), memory_space, and dynamic
  // dimensions while clearing dimension-order-dependent tile and tail fields.
  constexpr const char* kDynSubByteHlo = R"(
HloModule SubByteDynamicEqualize

red {
  a0 = f32[] parameter(0)
  b0 = s4[] parameter(1)
  a1 = f32[] parameter(2)
  b1 = s4[] parameter(3)
  ROOT t = (f32[], s4[]) tuple(f32[] add(a0, a1), s4[] add(b0, b1))
}

ENTRY main {
  p0 = f32[<=4,8]{1,0:S(1)} parameter(0)
  p1 = s4[<=4,8]{0,1:T(2,4)L(2)E(4)S(1)} parameter(1)
  z0 = f32[] constant(0)
  z1 = s4[] constant(0)
  ROOT r = (f32[<=4]{0:S(1)}, s4[<=4]{0:E(4)S(1)}) reduce(p0, p1, z0, z1),
      dimensions={1}, to_apply=red
}
)";
  auto mod_or = ParseAndReturnVerifiedModule(kDynSubByteHlo);
  ASSERT_TRUE(mod_or.ok()) << mod_or.status();
  auto changed_or = ReduceDecomposer{}.Run(mod_or->get());
  ASSERT_TRUE(changed_or.ok()) << changed_or.status();
  EXPECT_TRUE(*changed_or);
  const HloInstruction* root =
      (*mod_or)->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kReduce);
  const HloInstruction* in1 = root->operand(1);
  ASSERT_EQ(in1->opcode(), HloOpcode::kCopy);
  EXPECT_TRUE(in1->shape().is_dynamic_dimension(0));
  EXPECT_EQ(in1->shape().layout().element_size_in_bits(), 4);
  EXPECT_EQ(in1->shape().layout().memory_space(), 1);
  EXPECT_TRUE(in1->shape().layout().tiles().empty());
  EXPECT_EQ(in1->shape().layout().tail_padding_alignment_in_elements(), 1);
  EXPECT_EQ(in1->shape().layout().minor_to_major(),
            root->operand(0)->shape().layout().minor_to_major());
}

TEST_F(ReduceDecomposerTest, SkipsUnlayoutedAndAutoLayoutReduces) {
  constexpr const char* kHlo = R"(
HloModule AutoLayoutReduce

add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

pair_add {
  a0 = f32[] parameter(0)
  b0 = f32[] parameter(1)
  a1 = f32[] parameter(2)
  b1 = f32[] parameter(3)
  ROOT t = (f32[], f32[]) tuple(f32[] add(a0, a1), f32[] add(b0, b1))
}

ENTRY main {
  p0 = f32[4,8]{1,0} parameter(0)
  p1 = f32[4,8]{0,1} parameter(1)
  z = f32[] constant(0)
  r0 = f32[4]{0} reduce(p0, z), dimensions={1}, to_apply=add
  r1 = (f32[4]{0}, f32[4]{0}) reduce(p0, p1, z, z),
      dimensions={1}, to_apply=pair_add
  ROOT out = (f32[4]{0}, (f32[4]{0}, f32[4]{0})) tuple(r0, r1)
}
)";
  auto mod_or = ParseAndReturnVerifiedModule(kHlo);
  ASSERT_TRUE(mod_or.ok()) << mod_or.status();
  // Strip minor_to_major on p0 (leaving memory_space=1 AUTO layout) and clear
  // layout on r0 completely.
  HloComputation* entry = (*mod_or)->entry_computation();
  Layout* p0_layout =
      entry->parameter_instruction(0)->mutable_shape()->mutable_layout();
  p0_layout->clear_minor_to_major();
  p0_layout->set_memory_space(1);
  entry->GetInstructionWithName("r0")->mutable_shape()->clear_layout();

  bool predicate_called = false;
  auto changed_or = ReduceDecomposer{[&](const HloInstruction*) {
                      predicate_called = true;
                      return false;
                    }}.Run(mod_or->get());
  ASSERT_TRUE(changed_or.ok()) << changed_or.status();
  EXPECT_FALSE(*changed_or);
  EXPECT_FALSE(predicate_called);
}

TEST_F(LayoutSensitiveReduceDecomposerTest,
       ConvertFoldingPreservesLayoutSubByteAndMemorySpace) {
  constexpr const char* kHlo = R"(
HloModule ConvertFoldingLayouts

add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

add_s4 {
  a = s4[] parameter(0)
  b = s4[] parameter(1)
  ROOT s = s4[] add(a, b)
}

ENTRY main {
  pf = f32[5,3,4]{2,1,0} parameter(0)
  p3 = f32[2,3,4,5]{3,2,1,0} parameter(1)
  ps4 = s4[5,3,4]{2,1,0:E(4)} parameter(2)
  zf = f32[] constant(0)
  zs4 = s4[] constant(0)

  // Convert already in r_prime {1,0} layout: elides redundant copy.
  r_elide = f32[5,4]{0,1} reduce(pf, zf), dimensions={1}, to_apply=add_f32
  c_elide = bf16[5,4]{1,0} convert(r_elide)

  // 3D convert with layout distinct from both reduce and r_prime.
  r_3d = f32[2,4,5]{0,1,2} reduce(p3, zf), dimensions={1}, to_apply=add_f32
  c_3d = bf16[2,4,5]{1,2,0} convert(r_3d)

  // f32 -> packed s4:E(4), f32 -> unpacked s4, s4:E(4) -> f32, s4:E(4) -> s2:E(2).
  r_p4 = f32[5,4]{0,1} reduce(pf, zf), dimensions={1}, to_apply=add_f32
  c_p4 = s4[5,4]{0,1:E(4)} convert(r_p4)

  r_u4 = f32[5,4]{0,1} reduce(pf, zf), dimensions={1}, to_apply=add_f32
  c_u4 = s4[5,4]{0,1} convert(r_u4)

  r_sf = s4[5,4]{0,1:E(4)} reduce(ps4, zs4), dimensions={1}, to_apply=add_s4
  c_sf = f32[5,4]{0,1} convert(r_sf)

  r_s2 = s4[5,4]{0,1:E(4)} reduce(ps4, zs4), dimensions={1}, to_apply=add_s4
  c_s2 = s2[5,4]{0,1:E(2)} convert(r_s2)

  // Cross-memory-space convert does not fold across memory spaces.
  r_ms = f32[5,4]{0,1} reduce(pf, zf), dimensions={1}, to_apply=add_f32
  c_ms = bf16[5,4]{0,1:S(1)} convert(r_ms)

  ROOT out = (bf16[5,4]{1,0}, bf16[2,4,5]{1,2,0}, s4[5,4]{0,1:E(4)},
              s4[5,4]{0,1}, f32[5,4]{0,1}, s2[5,4]{0,1:E(2)},
              bf16[5,4]{0,1:S(1)})
      tuple(c_elide, c_3d, c_p4, c_u4, c_sf, c_s2, c_ms)
}
)";
  auto mod_or = ParseAndReturnVerifiedModule(kHlo);
  ASSERT_TRUE(mod_or.ok()) << mod_or.status();
  auto changed_or = ReduceDecomposer{}.Run(mod_or->get());
  ASSERT_TRUE(changed_or.ok()) << changed_or.status();
  EXPECT_TRUE(*changed_or);
  ASSERT_TRUE((*mod_or)->Verify().ok());

  const HloInstruction* root =
      (*mod_or)->entry_computation()->root_instruction();
  // c_elide is the convert directly without an extra copy.
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(root->operand(2)->shape().layout().element_size_in_bits(), 4);
  EXPECT_EQ(
      root->operand(2)->operand(0)->shape().layout().element_size_in_bits(), 4);
  EXPECT_EQ(root->operand(3)->shape().layout().element_size_in_bits(), 0);
  EXPECT_EQ(
      root->operand(3)->operand(0)->shape().layout().element_size_in_bits(), 0);
  EXPECT_EQ(root->operand(4)->shape().layout().element_size_in_bits(), 0);
  EXPECT_EQ(root->operand(5)->shape().layout().element_size_in_bits(), 2);
  // c_ms keeps copy on the f32 reduce before the S(1) convert.
  EXPECT_EQ(root->operand(6)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(root->operand(6)->operand(0)->opcode(), HloOpcode::kCopy);
}

TEST_F(LayoutSensitiveReduceDecomposerTest,
       ConvertFoldingRootEscapeAndControlDependencies) {
  // Root reduce with a dead kConvert user must still decompose the root reduce.
  constexpr const char* kRootHlo = R"(
HloModule RootReduceWithConvertUser

add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

ENTRY main {
  p = f32[5,3,4]{2,1,0} parameter(0)
  z = f32[] constant(0)
  ROOT r = f32[5,4]{0,1} reduce(p, z), dimensions={1}, to_apply=add
  dead_c = bf16[5,4]{0,1} convert(r)
}
)";
  RunAndFilecheckHloRewrite(kRootHlo, ReduceDecomposer{}, R"(
// CHECK: [[R:%[^ ]+]] = f32[5,4]{1,0} reduce(
// CHECK-NEXT: ROOT [[CPY:%[^ ]+]] = f32[5,4]{0,1} copy([[R]])
// CHECK-NEXT: {{%[^ ]+}} = bf16[5,4]{0,1} convert([[CPY]])
)");

  // Control edges pred -> reduce -> mid -> convert -> succ (plus direct
  // reduce -> convert and reduce -> succ) on a same-type layout-aligned convert
  // must preserve every barrier with zero cycles.
  constexpr const char* kCtrlHlo = R"(
HloModule ConvertControlDiamond

add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

ENTRY main {
  p = f32[5,3,4]{2,1,0} parameter(0)
  z = f32[] constant(0)
  pred_op = f32[] negate(z)
  mid_op = f32[] negate(z)
  succ_op = f32[] negate(z)
  r = f32[5,4]{0,1} reduce(p, z), dimensions={1}, to_apply=add
  c = f32[5,4]{1,0} convert(r)
  add_mid = f32[] add(mid_op, z)
  ROOT out = (f32[5,4]{1,0}, f32[], f32[]) tuple(c, add_mid, succ_op)
}
)";
  auto mod_or = ParseAndReturnVerifiedModule(kCtrlHlo);
  ASSERT_TRUE(mod_or.ok()) << mod_or.status();
  HloComputation* entry = (*mod_or)->entry_computation();
  HloInstruction* r = entry->GetInstructionWithName("r");
  HloInstruction* c = entry->GetInstructionWithName("c");
  HloInstruction* pred_op = entry->GetInstructionWithName("pred_op");
  HloInstruction* mid_op = entry->GetInstructionWithName("mid_op");
  HloInstruction* succ_op = entry->GetInstructionWithName("succ_op");
  ASSERT_TRUE(pred_op->AddControlDependencyTo(r).ok());
  ASSERT_TRUE(r->AddControlDependencyTo(mid_op).ok());
  ASSERT_TRUE(r->AddControlDependencyTo(c).ok());
  ASSERT_TRUE(mid_op->AddControlDependencyTo(c).ok());
  ASSERT_TRUE(r->AddControlDependencyTo(succ_op).ok());
  ASSERT_TRUE(c->AddControlDependencyTo(succ_op).ok());

  auto changed_or = ReduceDecomposer{}.Run(mod_or->get());
  ASSERT_TRUE(changed_or.ok()) << changed_or.status();
  EXPECT_TRUE(*changed_or);
  ASSERT_TRUE((*mod_or)->Verify().ok());

  HloInstruction* new_c = entry->root_instruction()->mutable_operand(0);
  ASSERT_EQ(new_c->opcode(), HloOpcode::kConvert);
  HloInstruction* r_prime = new_c->mutable_operand(0);
  ASSERT_EQ(r_prime->opcode(), HloOpcode::kReduce);
  auto to_vec = [](const auto& ptrs) {
    return std::vector<HloInstruction*>(ptrs.begin(), ptrs.end());
  };
  EXPECT_EQ(to_vec(r_prime->control_predecessors()),
            std::vector<HloInstruction*>{pred_op});
  EXPECT_EQ(to_vec(mid_op->control_predecessors()),
            std::vector<HloInstruction*>{r_prime});
  EXPECT_EQ(to_vec(r_prime->control_successors()),
            (std::vector<HloInstruction*>{mid_op, succ_op}));
  EXPECT_EQ(to_vec(new_c->control_predecessors()),
            std::vector<HloInstruction*>{mid_op});
  EXPECT_EQ(to_vec(new_c->control_successors()),
            std::vector<HloInstruction*>{succ_op});
}

TEST_F(LayoutSensitiveReduceDecomposerTest,
       PreservesShardingAttributesOneTupleAndMixedPrecision) {
  constexpr const char* kHlo = R"(
HloModule ShardingAndAttributes

add_pair {
  a0 = f32[] parameter(0)
  b0 = s32[] parameter(1)
  a1 = f32[] parameter(2)
  b1 = s32[] parameter(3)
  ROOT t = (f32[], s32[]) tuple(f32[] add(a0, a1), s32[] add(b0, b1))
}

add_one {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT t = (f32[]) tuple(f32[] add(a, b))
}

add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[4,3,8]{2,1,0} parameter(0)
  p1 = s32[4,3,8]{2,1,0} parameter(1)
  pb = bf16[4,3,8]{2,1,0} parameter(2)
  zf = f32[] constant(0)
  zs = s32[] constant(0)

  rv = (f32[4,8]{1,0}, s32[4,8]{0,1}) reduce(p0, p1, zf, zs),
      dimensions={1}, to_apply=add_pair,
      sharding={{devices=[2,2]<=[4]}, {replicated}},
      frontend_attributes={_xla_compute_type="host", custom="keep"},
      origin={({"rv_a"}, {"rv_b"})}

  rs = (f32[4,8]{0,1}, s32[4,8]{0,1}) reduce(p0, p1, zf, zs),
      dimensions={1}, to_apply=add_pair, sharding={replicated}

  r1 = (f32[4,8]{0,1}) reduce(p0, zf), dimensions={1}, to_apply=add_one

  rm = f32[4,8]{0,1} reduce(pb, zf), dimensions={1}, to_apply=add_f32,
      sharding={devices=[2,2]<=[4]},
      frontend_attributes={_xla_compute_type="dense"},
      origin={{"rm_orig"}}
  cm = bf16[4,8]{0,1} convert(rm), sharding={devices=[4,1]<=[4]},
      frontend_attributes={conv_attr="yes"}, backend_config="keep_cfg"

  ROOT out = ((f32[4,8]{1,0}, s32[4,8]{0,1}),
              (f32[4,8]{0,1}, s32[4,8]{0,1}),
              (f32[4,8]{0,1}), bf16[4,8]{0,1}) tuple(rv, rs, r1, cm)
}
)";
  auto mod_or = ParseAndReturnVerifiedModule(kHlo);
  ASSERT_TRUE(mod_or.ok()) << mod_or.status();
  auto changed_or = ReduceDecomposer{}.Run(mod_or->get());
  ASSERT_TRUE(changed_or.ok()) << changed_or.status();
  EXPECT_TRUE(*changed_or);
  ASSERT_TRUE((*mod_or)->Verify().ok());

  const HloInstruction* root =
      (*mod_or)->entry_computation()->root_instruction();
  const HloInstruction* rv_tup = root->operand(0);
  EXPECT_TRUE(rv_tup->has_sharding());
  EXPECT_TRUE(rv_tup->frontend_attributes().map().empty());
  const HloInstruction* gte0 = rv_tup->operand(0);
  const HloInstruction* cpy1 = rv_tup->operand(1);
  EXPECT_EQ(gte0->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(cpy1->opcode(), HloOpcode::kCopy);
  EXPECT_TRUE(cpy1->frontend_attributes().map().empty());
  ASSERT_TRUE(gte0->has_sharding());
  ASSERT_TRUE(cpy1->has_sharding());
  EXPECT_FALSE(gte0->sharding().IsReplicated());
  EXPECT_TRUE(cpy1->sharding().IsReplicated());
  const HloInstruction* rv_prime = gte0->operand(0);
  EXPECT_EQ(rv_prime->frontend_attributes().map().at("_xla_compute_type"),
            "host");
  EXPECT_EQ(rv_prime->frontend_attributes().map().at("custom"), "keep");
  EXPECT_NE(rv_prime->original_value(), nullptr);

  // Singleton replicated sharding on tuple reduce propagates to leaf copies.
  const HloInstruction* rs_tup = root->operand(1);
  ASSERT_TRUE(rs_tup->operand(0)->has_sharding());
  EXPECT_TRUE(rs_tup->operand(0)->sharding().IsReplicated());

  // 1-tuple reduce stays a 1-tuple.
  EXPECT_TRUE(root->operand(2)->shape().IsTuple());
  EXPECT_EQ(root->operand(2)->operand(0)->opcode(), HloOpcode::kCopy);

  // Mixed-precision reduce -> convert keeps separate shardings, attributes,
  // backend_config, and original_value without leaking attributes to kCopy.
  const HloInstruction* cm_cpy = root->operand(3);
  ASSERT_EQ(cm_cpy->opcode(), HloOpcode::kCopy);
  EXPECT_TRUE(cm_cpy->frontend_attributes().map().empty());
  const HloInstruction* cm_conv = cm_cpy->operand(0);
  ASSERT_EQ(cm_conv->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(cm_conv->frontend_attributes().map().at("conv_attr"), "yes");
  EXPECT_EQ(cm_conv->raw_backend_config_string(), "keep_cfg");
  const HloInstruction* rm_prime = cm_conv->operand(0);
  EXPECT_EQ(rm_prime->frontend_attributes().map().at("_xla_compute_type"),
            "dense");
  EXPECT_NE(rm_prime->original_value(), nullptr);
  EXPECT_NE(rm_prime->sharding(), cm_conv->sharding());
}

}  // namespace
}  // namespace xla
