/* Copyright 2018 The OpenXLA Authors.

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

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/error_spec.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/cpu_instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {
namespace {

class CpuFusionTest : public HloTestBase {
 protected:
  ErrorSpec error_spec_{0.0001, 1e-5};
  AliasInfo alias_info_;

 private:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.add_xla_disable_hlo_passes("layout-assignment");
    return debug_options;
  }
};

TEST_F(CpuFusionTest, FuseTwoElementwiseOps) {
  auto builder = HloComputation::Builder(TestName());
  auto input_literal1 = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0});
  auto input_literal2 = LiteralUtil::CreateR1<float>({-2.0, -42.0, 2.0});
  Shape vshape = input_literal1.shape();

  auto input1 = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal1)));
  auto input2 = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal2)));

  auto add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(vshape, HloOpcode::kAdd, input1, input2));
  builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kNegate, add1));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  CpuInstructionFusion fusion(&alias_info_);
  EXPECT_TRUE(fusion.Run(module.get()).value());

  // The computation root instruction was fused. Verify the fusion instruction
  // is now the root.
  auto computation = module->entry_computation();
  auto fusion_instruction = computation->root_instruction();
  EXPECT_EQ(HloOpcode::kFusion, fusion_instruction->opcode());
  EXPECT_EQ(HloOpcode::kNegate,
            fusion_instruction->fused_expression_root()->opcode());
  // There should be four fused instructions: 2 parameters, the add, and the
  // negate.
  EXPECT_EQ(4, fusion_instruction->fused_instruction_count());

  // Compile and execute the computation.
  ASSERT_OK_AND_ASSIGN(const Literal result, Execute(std::move(module), {}));

  // Check the output correctness.
  LiteralTestUtil::ExpectR1Near<float>({1.0, 40.0, -5.0}, result, error_spec_);
}

TEST_F(CpuFusionTest, FuseElementwiseOpChain) {
  auto builder = HloComputation::Builder(TestName());
  auto input_literal = LiteralUtil::CreateR1<float>({-1.5, -2.5, -3.0});
  Shape vshape = input_literal.shape();

  auto input = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kNegate, input));
  auto ceil = builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kCeil, negate));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kExp, ceil));
  auto floor = builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kFloor, exp));
  auto two = builder.AddInstruction(HloInstruction::CreateBroadcast(
      vshape,
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0))),
      {}));
  builder.AddInstruction(
      HloInstruction::CreateBinary(vshape, HloOpcode::kMultiply, two, floor));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  CpuInstructionFusion fusion(&alias_info_);
  EXPECT_TRUE(fusion.Run(module.get()).value());

  // The computation root instruction was fused. Verify the fusion instruction
  // is now the root.
  auto computation = module->entry_computation();
  auto fusion_instruction = computation->root_instruction();
  EXPECT_EQ(HloOpcode::kFusion, fusion_instruction->opcode());
  EXPECT_EQ(HloOpcode::kMultiply,
            fusion_instruction->fused_expression_root()->opcode());
  // There should be 8 fused instructions: 2 parameters and the fused
  // operations.
  EXPECT_EQ(8, fusion_instruction->fused_instruction_count());

  // Compile and execute the computation.
  ASSERT_OK_AND_ASSIGN(const Literal result, Execute(std::move(module), {}));

  // Check the output correctness.
  LiteralTestUtil::ExpectR1Near<float>({14.0, 40.0, 40.0}, result, error_spec_);
}

TEST_F(CpuFusionTest, ElementwiseOpChainWithNonfusibleInstruction) {
  // Test a chain of fusible ops with a non-fusible op (a reduce) thrown in the
  // middle.
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto input_literal = LiteralUtil::CreateR1<float>({-1.5, -2.5, -3.0});
  Shape vshape = input_literal.shape();

  auto input = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kNegate, input));
  auto ceil = builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kCeil, negate));

  auto cshape = ShapeUtil::MakeShape(F32, {6});
  auto concatenate = builder.AddInstruction(
      HloInstruction::CreateConcatenate(cshape, {ceil, ceil}, /*dimension=*/0));

  // Build an x+y computation to use in a reduce.
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  auto embedded_builder = HloComputation::Builder("f32+f32");
  embedded_builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kAdd,
      embedded_builder.AddInstruction(
          HloInstruction::CreateParameter(0, r0f32, "x")),
      embedded_builder.AddInstruction(
          HloInstruction::CreateParameter(1, r0f32, "y"))));
  auto add_f32 = module->AddEmbeddedComputation(embedded_builder.Build());

  Window window;
  WindowDimension* window_dimension = window.add_dimensions();
  window_dimension->set_size(1);
  window_dimension->set_stride(1);
  window_dimension->set_base_dilation(1);
  window_dimension->set_window_dilation(1);
  // This is a nop reduction.
  auto reduce = builder.AddInstruction(HloInstruction::CreateReduceWindow(
      cshape, concatenate,
      /*init_value=*/
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0))),
      window, add_f32));

  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(cshape, HloOpcode::kExp, reduce));
  auto floor = builder.AddInstruction(
      HloInstruction::CreateUnary(cshape, HloOpcode::kFloor, exp));
  auto two = builder.AddInstruction(HloInstruction::CreateBroadcast(
      cshape,
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0))),
      {}));
  builder.AddInstruction(
      HloInstruction::CreateBinary(cshape, HloOpcode::kMultiply, two, floor));

  module->AddEntryComputation(builder.Build());

  CpuInstructionFusion fusion(&alias_info_);
  EXPECT_TRUE(fusion.Run(module.get()).value());

  // The computation root instruction was fused. Verify the fusion instruction
  // is now the root.
  auto computation = module->entry_computation();

  auto fusion_instruction1 = computation->root_instruction();
  EXPECT_EQ(HloOpcode::kFusion, fusion_instruction1->opcode());
  EXPECT_EQ(HloOpcode::kMultiply,
            fusion_instruction1->fused_expression_root()->opcode());
  // There should be 6 fused instructions in the root fusion instruction: 2
  // parameters, multiply, floor, and exp.
  EXPECT_EQ(6, fusion_instruction1->fused_instruction_count())
      << fusion_instruction1->fused_instructions_computation()->ToString();

  auto fusion_instruction2 = reduce->operand(0);
  EXPECT_EQ(HloOpcode::kFusion, fusion_instruction1->opcode());
  EXPECT_EQ(HloOpcode::kConcatenate,
            fusion_instruction2->fused_expression_root()->opcode());
  // There should be 5 fused instructions in the second fusion instruction: 1
  // parameter, negate, ceil and concat.
  EXPECT_EQ(4, fusion_instruction2->fused_instruction_count())
      << fusion_instruction2->fused_instructions_computation()->ToString();

  // Compile and execute the computation.
  ASSERT_OK_AND_ASSIGN(const Literal result, Execute(std::move(module), {}));

  // Check the output correctness.
  LiteralTestUtil::ExpectR1Near<float>({14.0, 40.0, 40.0, 14.0, 40.0, 40.0},
                                       result, error_spec_);
}

TEST_F(CpuFusionTest, TestOperandOrderToAvoidDuplication) {
  // Test that the operands of an instruction to be fused are considered in the
  // proper order to avoid duplication. Test input:
  //
  //   constant = {...}
  //   negate    = neg(constant)
  //   ceil      = ceil(negate)
  //   add1      = add(negate, ceil)
  //   add2      = add(ceil, negate)
  //
  // In this example, the operands of both add1 and add2 should be fused in the
  // order {ceil, negate} even though they have different orders in their
  // operand vectors. Test for this problem by counting the number of nodes in
  // each fusion instruction to ensure that negate is not duplicated.
  auto builder = HloComputation::Builder(TestName());
  auto input_literal = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0});
  Shape vshape = input_literal.shape();

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kNegate, constant));
  auto ceil = builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kCeil, negate));

  auto add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(vshape, HloOpcode::kMultiply, negate, ceil));
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(vshape, HloOpcode::kMultiply, ceil, negate));

  // Tie together the two adds with a tuple to create a single root.
  auto result =
      builder.AddInstruction(HloInstruction::CreateTuple({add1, add2}));

  // Create computation and module.
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  // Run fusion.
  CpuInstructionFusion fusion(&alias_info_);
  EXPECT_TRUE(fusion.Run(module.get()).value());

  auto fusion1 = result->operand(0);
  auto fusion2 = result->operand(1);
  EXPECT_EQ(HloOpcode::kFusion, fusion1->opcode());
  EXPECT_EQ(HloOpcode::kFusion, fusion2->opcode());

  // Each fusion instruction should have 4 fused instruction inside: add, ceil,
  // negate, and the fused parameter.
  EXPECT_EQ(4, fusion1->fused_instruction_count());
  EXPECT_EQ(4, fusion2->fused_instruction_count());

  // The fusion has a single constant parameter.
  EXPECT_EQ(1, fusion1->operand_count());
  EXPECT_EQ(1, fusion2->operand_count());
}

TEST_F(CpuFusionTest, DoNotDuplicateExpensiveOps) {
  // Verify that expensive operations will not be fused if the fusion results in
  // duplication. Test code:
  //
  //   constant = 42.0
  //   exp1 = exp(constant)
  //   negate1 = negate(exp1)
  //   exp2 = exp(constant)
  //   negate2 = negate(exp2)
  //   tuple = tuple(negate1, negate2, exp2)
  //
  // exp1 should be fused down into negate1, but exp2 will not be fused into
  // negate2 because this will result in duplication of the expensive exp
  // computation. The duplication is caused by the other use of exp2 in the
  // tuple.
  auto builder = HloComputation::Builder(TestName());
  auto input_literal1 = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0});
  auto input_literal2 = LiteralUtil::CreateR1<float>({-2.0, -42.0, 2.0});
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));
  Shape shape = constant->shape();

  auto exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kExp, constant));
  auto negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, exp1));

  auto exp2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kExp, constant));
  auto negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, exp2));

  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({negate1, negate2, exp2}));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  CpuInstructionFusion fusion(&alias_info_);
  EXPECT_TRUE(fusion.Run(module.get()).value());

  // The only fusion instruction should be operand 0 of the tuple (formerly
  // negate1).
  EXPECT_EQ(HloOpcode::kFusion, tuple->operand(0)->opcode());
  EXPECT_EQ(HloOpcode::kNegate, tuple->operand(1)->opcode());
  EXPECT_EQ(HloOpcode::kExp, tuple->operand(2)->opcode());

  auto fusion_inst = tuple->operand(0);
  // There should be three fused instructions: negate2, exp2, and the fused
  // constant.
  EXPECT_EQ(3, fusion_inst->fused_instruction_count());
  EXPECT_EQ(0, fusion_inst->operand_count());
}

TEST_F(CpuFusionTest, DynamicUpdateSliceOutOfPlace) {
  const char* const hlo_string = R"(
HloModule OutOfPlaceDus

dus_fusion {
  base = f32[4,4] parameter(0)
  update = f32[2,2] parameter(1)
  idx0 = s32[] parameter(2)
  idx1 = s32[] parameter(3)
  ROOT dus = f32[4,4] dynamic-update-slice(base, update, idx0, idx1)
}

ENTRY main {
  p0 = f32[4,4] parameter(0)
  p1 = f32[2,2] parameter(1)
  p2 = s32[] parameter(2)
  p3 = s32[] parameter(3)
  fusion = f32[4,4] fusion(p0, p1, p2, p3), kind=kLoop, calls=dus_fusion
  neg = f32[4,4] negate(p0)
  ROOT tuple = (f32[4,4], f32[4,4]) tuple(fusion, neg)
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  auto p0_literal = LiteralUtil::CreateR2<float>({
      {1.0f, 2.0f, 3.0f, 4.0f},
      {5.0f, 6.0f, 7.0f, 8.0f},
      {9.0f, 10.0f, 11.0f, 12.0f},
      {13.0f, 14.0f, 15.0f, 16.0f},
  });
  auto p1_literal = LiteralUtil::CreateR2<float>({
      {100.0f, 200.0f},
      {300.0f, 400.0f},
  });
  auto p2_literal = LiteralUtil::CreateR0<int32_t>(1);
  auto p3_literal = LiteralUtil::CreateR0<int32_t>(1);

  ASSERT_OK_AND_ASSIGN(const Literal result,
                       Execute(std::move(module), {&p0_literal, &p1_literal,
                                                   &p2_literal, &p3_literal}));

  LiteralTestUtil::ExpectR2Near<float>({{1.0f, 2.0f, 3.0f, 4.0f},
                                        {5.0f, 100.0f, 200.0f, 8.0f},
                                        {9.0f, 300.0f, 400.0f, 12.0f},
                                        {13.0f, 14.0f, 15.0f, 16.0f}},
                                       LiteralSlice(result, {0}), error_spec_);

  LiteralTestUtil::ExpectR2Near<float>({{-1.0f, -2.0f, -3.0f, -4.0f},
                                        {-5.0f, -6.0f, -7.0f, -8.0f},
                                        {-9.0f, -10.0f, -11.0f, -12.0f},
                                        {-13.0f, -14.0f, -15.0f, -16.0f}},
                                       LiteralSlice(result, {1}), error_spec_);
}

TEST_F(CpuFusionTest, DynamicUpdateSliceOutOfPlaceWithBitcast) {
  const char* const hlo_string = R"(
HloModule OutOfPlaceDusWithBitcast

dus_fusion {
  base = f32[16] parameter(0)
  bitcast_base = f32[4,4] bitcast(base)
  update = f32[2,2] parameter(1)
  idx0 = s32[] parameter(2)
  idx1 = s32[] parameter(3)
  dus = f32[4,4] dynamic-update-slice(bitcast_base, update, idx0, idx1)
  ROOT bitcast_dus = f32[16] bitcast(dus)
}

ENTRY main {
  p0 = f32[16] parameter(0)
  p1 = f32[2,2] parameter(1)
  p2 = s32[] parameter(2)
  p3 = s32[] parameter(3)
  fusion = f32[16] fusion(p0, p1, p2, p3), kind=kLoop, calls=dus_fusion
  neg = f32[16] negate(p0)
  ROOT tuple = (f32[16], f32[16]) tuple(fusion, neg)
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  auto p0_literal = LiteralUtil::CreateR1<float>({
      1.0f,
      2.0f,
      3.0f,
      4.0f,
      5.0f,
      6.0f,
      7.0f,
      8.0f,
      9.0f,
      10.0f,
      11.0f,
      12.0f,
      13.0f,
      14.0f,
      15.0f,
      16.0f,
  });
  auto p1_literal = LiteralUtil::CreateR2<float>({
      {100.0f, 200.0f},
      {300.0f, 400.0f},
  });
  auto p2_literal = LiteralUtil::CreateR0<int32_t>(1);
  auto p3_literal = LiteralUtil::CreateR0<int32_t>(1);

  ASSERT_OK_AND_ASSIGN(const Literal result,
                       Execute(std::move(module), {&p0_literal, &p1_literal,
                                                   &p2_literal, &p3_literal}));

  LiteralTestUtil::ExpectR1Near<float>(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 100.0f, 200.0f, 8.0f, 9.0f, 300.0f, 400.0f,
       12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
      LiteralSlice(result, {0}), error_spec_);

  LiteralTestUtil::ExpectR1Near<float>(
      {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, -9.0f, -10.0f,
       -11.0f, -12.0f, -13.0f, -14.0f, -15.0f, -16.0f},
      LiteralSlice(result, {1}), error_spec_);
}

TEST_F(CpuFusionTest, ScatterFusionZeroUpdates) {
  // Run an unfused variadic zero update scatter through the full CPU HLO
  // pipeline with algebraic simplification disabled so ScatterSimplifier,
  // FusionWrapper, CopyInsertion, and CpuScatterFusion all handle it.
  constexpr absl::string_view kUnfusedVariadicZeroUpdatesHlo = R"hlo(
    HloModule UnfusedVariadicZeroUpdates

    reducer {
      lhs0 = s32[] parameter(0)
      lhs1 = f64[] parameter(1)
      rhs0 = s32[] parameter(2)
      rhs1 = f64[] parameter(3)
      sum0 = s32[] add(lhs0, rhs0)
      sum1 = f64[] add(lhs1, rhs1)
      ROOT out = (s32[], f64[]) tuple(sum0, sum1)
    }

    ENTRY main {
      op0 = s32[3,2]{1,0} parameter(0)
      op1 = f64[3,2]{1,0} parameter(1)
      idx = s32[0]{0} parameter(2)
      upd0 = s32[0,2]{1,0} parameter(3)
      upd1 = f64[0,2]{1,0} parameter(4)
      ROOT scatter = (s32[3,2]{1,0}, f64[3,2]{1,0}) scatter(
          op0, op1, idx, upd0, upd1),
        update_window_dims={1},
        inserted_window_dims={0},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=reducer
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> variadic_module,
      ParseAndReturnVerifiedModule(kUnfusedVariadicZeroUpdatesHlo));
  variadic_module->mutable_config()
      .mutable_debug_options()
      .add_xla_disable_hlo_passes("algsimp");
  Literal operand = LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}, {5, 6}});
  Literal op1 =
      LiteralUtil::CreateR2<double>({{1.5, 2.5}, {3.5, 4.5}, {5.5, 6.5}});
  Literal idx_1d = LiteralUtil::CreateR1<int32_t>({});
  Literal upd0(ShapeUtil::MakeShape(S32, {0, 2}));
  Literal upd1(ShapeUtil::MakeShape(F64, {0, 2}));
  ASSERT_OK_AND_ASSIGN(const Literal variadic_result,
                       Execute(std::move(variadic_module),
                               {&operand, &op1, &idx_1d, &upd0, &upd1}));
  EXPECT_EQ(LiteralSlice(variadic_result, {0}), operand);
  EXPECT_EQ(LiteralSlice(variadic_result, {1}), op1);
}

TEST_F(CpuFusionTest, ScatterFusionZeroSizedUpdateSlice) {
  // Wrapped fusion on a zero sized operand (distinct from the unfused case
  // below because ZeroSizedHloElimination folds zero sized outputs).
  constexpr absl::string_view kZeroSizedOperandHlo = R"hlo(
    HloModule ScatterFusionZeroSliceZeroOperand

    add_s32 {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }

    scatter_computation {
      operand = s32[3,0]{1,0} parameter(0)
      indices = s32[2,1]{1,0} parameter(1)
      updates = s32[2,1,0]{2,1,0} parameter(2)
      ROOT scatter = s32[3,0]{1,0} scatter(operand, indices, updates),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=add_s32
    }

    ENTRY main {
      p0 = s32[3,0]{1,0} parameter(0)
      operand_copy = s32[3,0]{1,0} copy(p0)
      p1 = s32[2,1]{1,0} parameter(1)
      p2 = s32[2,1,0]{2,1,0} parameter(2)
      ROOT wrapped_scatter = s32[3,0]{1,0} fusion(operand_copy, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> empty_operand_module,
                       ParseAndReturnVerifiedModule(kZeroSizedOperandHlo));
  Literal empty_operand(ShapeUtil::MakeShape(S32, {3, 0}));
  Literal empty_op_indices = LiteralUtil::CreateR2<int32_t>({{0}, {2}});
  Literal empty_op_updates(ShapeUtil::MakeShape(S32, {2, 1, 0}));
  ASSERT_OK_AND_ASSIGN(
      const Literal empty_result,
      Execute(std::move(empty_operand_module),
              {&empty_operand, &empty_op_indices, &empty_op_updates},
              /*run_hlo_passes=*/false));
  EXPECT_EQ(empty_result, empty_operand);

  // Full CPU HLO pipeline with algebraic simplification disabled on a zero
  // sized update slice into a populated operand.
  constexpr absl::string_view kUnfusedZeroSliceHlo = R"hlo(
    HloModule UnfusedZeroSliceScatter

    add_s32 {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }

    ENTRY main {
      operand = s32[3,4]{1,0} parameter(0)
      indices = s32[2,2]{1,0} parameter(1)
      updates = s32[2,0]{1,0} parameter(2)
      ROOT scatter = s32[3,4]{1,0} scatter(operand, indices, updates),
        update_window_dims={1},
        inserted_window_dims={0},
        scatter_dims_to_operand_dims={0,1},
        index_vector_dim=1,
        to_apply=add_s32
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> unfused_module,
                       ParseAndReturnVerifiedModule(kUnfusedZeroSliceHlo));
  unfused_module->mutable_config()
      .mutable_debug_options()
      .add_xla_disable_hlo_passes("algsimp");
  Literal operand = LiteralUtil::CreateR2<int32_t>(
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
  Literal indices = LiteralUtil::CreateR2<int32_t>({{1, 2}, {9, 9}});
  Literal updates_2d(ShapeUtil::MakeShape(S32, {2, 0}));
  ASSERT_OK_AND_ASSIGN(
      const Literal unfused_result,
      Execute(std::move(unfused_module), {&operand, &indices, &updates_2d}));
  EXPECT_EQ(unfused_result, operand);
}

TEST_F(CpuFusionTest, ScatterFusionRankZeroScalarOperands) {
  // Unfused variadic rank 0 scatter going through the full CPU HLO pipeline
  // (ScatterSimplifier + FusionWrapper + CpuScatterFusion).
  constexpr absl::string_view kUnfusedVariadicRankZeroHlo = R"hlo(
    HloModule UnfusedVariadicRankZeroScatter

    reducer {
      lhs0 = s32[] parameter(0)
      lhs1 = f64[] parameter(1)
      rhs0 = s32[] parameter(2)
      rhs1 = f64[] parameter(3)
      sum0 = s32[] add(lhs0, rhs0)
      sum1 = f64[] add(lhs1, rhs1)
      ROOT out = (s32[], f64[]) tuple(sum0, sum1)
    }

    ENTRY main {
      op0 = s32[] parameter(0)
      op1 = f64[] parameter(1)
      idx = s32[2,2,0]{2,1,0} parameter(2)
      upd0 = s32[2,2]{1,0} parameter(3)
      upd1 = f64[2,2]{1,0} parameter(4)
      ROOT scatter = (s32[], f64[]) scatter(op0, op1, idx, upd0, upd1),
        update_window_dims={},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={},
        index_vector_dim=2,
        to_apply=reducer
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> variadic_module,
      ParseAndReturnVerifiedModule(kUnfusedVariadicRankZeroHlo));
  Literal op0 = LiteralUtil::CreateR0<int32_t>(100);
  Literal op1 = LiteralUtil::CreateR0<double>(1.5);
  Literal idx_2x2(ShapeUtil::MakeShape(S32, {2, 2, 0}));
  Literal upd0 = LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}});
  Literal upd1 = LiteralUtil::CreateR2<double>({{0.5, 1.0}, {2.0, 3.0}});
  ASSERT_OK_AND_ASSIGN(const Literal variadic_result,
                       Execute(std::move(variadic_module),
                               {&op0, &op1, &idx_2x2, &upd0, &upd1}));
  EXPECT_EQ(LiteralSlice(variadic_result, {0}),
            LiteralUtil::CreateR0<int32_t>(110));
  LiteralTestUtil::ExpectR0Near<double>(8.0, LiteralSlice(variadic_result, {1}),
                                        error_spec_);

  // Wrapped fusion with zero updates into a rank 0 scalar operand.
  constexpr absl::string_view kZeroUpdatesScalarHlo = R"hlo(
    HloModule ScatterFusionRankZeroZeroUpdates

    add_s32 {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }

    scatter_computation {
      operand = s32[] parameter(0)
      indices = s32[0,0]{1,0} parameter(1)
      updates = s32[0]{0} parameter(2)
      ROOT scatter = s32[] scatter(operand, indices, updates),
        update_window_dims={},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={},
        index_vector_dim=1,
        to_apply=add_s32
    }

    ENTRY main {
      p0 = s32[] parameter(0)
      operand_copy = s32[] copy(p0)
      p1 = s32[0,0]{1,0} parameter(1)
      p2 = s32[0]{0} parameter(2)
      ROOT wrapped_scatter = s32[] fusion(operand_copy, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> zero_upd_scalar_module,
                       ParseAndReturnVerifiedModule(kZeroUpdatesScalarHlo));
  Literal operand = LiteralUtil::CreateR0<int32_t>(10);
  Literal empty_scalar_idx(ShapeUtil::MakeShape(S32, {0, 0}));
  Literal empty_scalar_upd = LiteralUtil::CreateR1<int32_t>({});
  ASSERT_OK_AND_ASSIGN(const Literal zero_upd_scalar_result,
                       Execute(std::move(zero_upd_scalar_module),
                               {&operand, &empty_scalar_idx, &empty_scalar_upd},
                               /*run_hlo_passes=*/false));
  EXPECT_EQ(zero_upd_scalar_result, operand);
}

TEST_F(CpuFusionTest, ScatterFusionMixedWidthVariadic) {
  constexpr absl::string_view kMixedWidthVariadicHlo = R"hlo(
    HloModule ScatterFusionMixedWidthVariadic

    reducer {
      lhs0 = s8[] parameter(0)
      lhs1 = c128[] parameter(1)
      rhs0 = s8[] parameter(2)
      rhs1 = c128[] parameter(3)
      sum0 = s8[] add(lhs0, rhs0)
      sum1 = c128[] add(lhs1, rhs1)
      ROOT out = (s8[], c128[]) tuple(sum0, sum1)
    }

    ENTRY main {
      op0 = s8[4,16]{1,0} parameter(0)
      op1 = c128[4,16]{1,0} parameter(1)
      indices = s32[2,1]{1,0} parameter(2)
      upd0 = s8[2,1,16]{2,1,0} parameter(3)
      upd1 = c128[2,1,16]{2,1,0} parameter(4)
      ROOT scatter = (s8[4,16]{1,0}, c128[4,16]{1,0}) scatter(
          op0, op1, indices, upd0, upd1),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=reducer
    }
  )hlo";

  Literal op0(ShapeUtil::MakeShape(S8, {4, 16}));
  Literal op1(ShapeUtil::MakeShape(C128, {4, 16}));
  Literal upd0(ShapeUtil::MakeShape(S8, {2, 1, 16}));
  Literal upd1(ShapeUtil::MakeShape(C128, {2, 1, 16}));
  Literal expected0(ShapeUtil::MakeShape(S8, {4, 16}));
  Literal expected1(ShapeUtil::MakeShape(C128, {4, 16}));
  for (int64_t r = 0; r < 4; ++r) {
    for (int64_t c = 0; c < 16; ++c) {
      const int8_t v0 = static_cast<int8_t>(r * 16 + c - 32);
      const complex128 v1(static_cast<double>(r * 16 + c),
                          -static_cast<double>(c + 1));
      op0.Set<int8_t>({r, c}, v0);
      op1.Set<complex128>({r, c}, v1);
      expected0.Set<int8_t>({r, c}, v0);
      expected1.Set<complex128>({r, c}, v1);
    }
  }
  Literal indices = LiteralUtil::CreateR2<int32_t>({{1}, {3}});
  const int64_t target_rows[2] = {1, 3};
  for (int64_t u = 0; u < 2; ++u) {
    const int64_t r = target_rows[u];
    for (int64_t c = 0; c < 16; ++c) {
      const int8_t d0 = static_cast<int8_t>((u + 1) * 3 + (c % 5));
      const complex128 d1(static_cast<double>(u + 1) * 0.5,
                          static_cast<double>(c) * 0.25);
      upd0.Set<int8_t>({u, 0, c}, d0);
      upd1.Set<complex128>({u, 0, c}, d1);
      expected0.Set<int8_t>(
          {r, c}, static_cast<int8_t>(expected0.Get<int8_t>({r, c}) + d0));
      expected1.Set<complex128>({r, c}, expected1.Get<complex128>({r, c}) + d1);
    }
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kMixedWidthVariadicHlo));
  ASSERT_OK_AND_ASSIGN(
      const Literal result,
      Execute(std::move(module), {&op0, &op1, &indices, &upd0, &upd1}));
  EXPECT_EQ(LiteralSlice(result, {0}), expected0);
  EXPECT_EQ(LiteralSlice(result, {1}), expected1);
}

class CpuScatterMultiWorkgroupTest : public HloTestBase {
 protected:
  static std::string BuildForced1DScatterHlo(
      absl::string_view module_name, absl::string_view dtype,
      absl::string_view layout_suffix, int64_t num_bins, int64_t num_updates,
      int64_t partitions, bool unique_indices, absl::string_view reducer_body) {
    return absl::StrFormat(
        R"hlo(
HloModule %s

reducer {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
%s
}

scatter_computation {
  operand = %s[%d]{0%s} parameter(0)
  indices = s32[%d,1]{1,0} parameter(1)
  updates = %s[%d,1]{1,0%s} parameter(2)
  ROOT scatter = %s[%d]{0%s} scatter(operand, indices, updates),
    update_window_dims={1},
    inserted_window_dims={},
    scatter_dims_to_operand_dims={0},
    index_vector_dim=1,
    unique_indices=%s,
    to_apply=reducer
}

ENTRY main {
  p0 = %s[%d]{0%s} parameter(0)
  operand_copy = %s[%d]{0%s} copy(p0)
  p1 = s32[%d,1]{1,0} parameter(1)
  p2 = %s[%d,1]{1,0%s} parameter(2)
  ROOT wrapped_scatter = %s[%d]{0%s} fusion(operand_copy, p1, p2),
    kind=kLoop, calls=scatter_computation,
    backend_config={"outer_dimension_partitions":["%d"]}
}
)hlo",
        module_name, dtype, dtype, reducer_body, dtype, num_bins, layout_suffix,
        num_updates, dtype, num_updates, layout_suffix, dtype, num_bins,
        layout_suffix, unique_indices ? "true" : "false", dtype, num_bins,
        layout_suffix, dtype, num_bins, layout_suffix, num_updates, dtype,
        num_updates, layout_suffix, dtype, num_bins, layout_suffix, partitions);
  }

  absl::StatusOr<Literal> RunForced1DScatter(
      absl::string_view module_name, absl::string_view dtype,
      absl::string_view layout_suffix, int64_t num_bins, int64_t num_updates,
      int64_t partitions, bool unique_indices, absl::string_view reducer_body,
      const Literal& operand, const Literal& indices, const Literal& updates) {
    const std::string hlo = BuildForced1DScatterHlo(
        module_name, dtype, layout_suffix, num_bins, num_updates, partitions,
        unique_indices, reducer_body);
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
    return Execute(std::move(module), {&operand, &indices, &updates},
                   /*run_hlo_passes=*/false);
  }

  // Builds indices in [0, active_bins) with heavy collisions plus negative and
  // out of bounds entries, leaving bins [active_bins, num_bins) untouched.
  static std::vector<int32_t> MakeCollidingIndicesWithOob(int64_t num_updates,
                                                          int32_t num_bins,
                                                          int32_t active_bins) {
    std::vector<int32_t> idx(num_updates);
    for (int64_t i = 0; i < num_updates; ++i) {
      if (i % 29 == 3) {
        idx[i] = -7;
      } else if (i % 29 == 11) {
        idx[i] = -1;
      } else if (i % 29 == 19) {
        idx[i] = num_bins;
      } else if (i % 29 == 27) {
        idx[i] = std::numeric_limits<int32_t>::max();
      } else {
        idx[i] = static_cast<int32_t>((i * 7 + (i / 13) * 3) % active_bins);
      }
    }
    return idx;
  }

  template <typename T>
  static Literal MakeColLiteral(PrimitiveType ptype,
                                absl::Span<const T> values) {
    Literal lit(
        ShapeUtil::MakeShape(ptype, {static_cast<int64_t>(values.size()), 1}));
    absl::Span<T> dst = lit.data<T>();
    for (size_t i = 0; i < values.size(); ++i) {
      dst[i] = values[i];
    }
    return lit;
  }

  template <typename IntT>
  static IntT WrapAdd(IntT a, IntT b) {
    using U = std::make_unsigned_t<IntT>;
    return static_cast<IntT>(static_cast<U>(a) + static_cast<U>(b));
  }

  template <typename IntT>
  static IntT WrapMul(IntT a, IntT b) {
    using U = std::make_unsigned_t<IntT>;
    return static_cast<IntT>(static_cast<U>(a) * static_cast<U>(b));
  }

  template <typename IntT>
  absl::Status VerifyIntegerScatterSuite(PrimitiveType ptype,
                                         absl::string_view dtype) {
    constexpr int64_t kNumUpdates = 517;
    constexpr int32_t kBins = 19;
    constexpr int32_t kActiveBins = 16;
    const std::vector<int32_t> idx =
        MakeCollidingIndicesWithOob(kNumUpdates, kBins, kActiveBins);
    const Literal idx_lit = MakeColLiteral<int32_t>(S32, idx);

    std::vector<IntT> init(kBins);
    for (int32_t b = 0; b < kBins; ++b) {
      if constexpr (std::is_signed_v<IntT>) {
        init[b] = static_cast<IntT>((b % 9) - 4);
      } else {
        init[b] = static_cast<IntT>((b % 9) + 1);
      }
    }
    std::vector<IntT> upd(kNumUpdates);
    for (int64_t i = 0; i < kNumUpdates; ++i) {
      const int32_t small = static_cast<int32_t>(i % 5);
      if constexpr (std::is_signed_v<IntT>) {
        upd[i] = static_cast<IntT>(small - 2);
      } else {
        upd[i] = static_cast<IntT>(small + 1);
      }
    }
    const Literal init_lit = LiteralUtil::CreateR1<IntT>(init);
    const Literal upd_lit = MakeColLiteral<IntT>(ptype, upd);

    // Overwrite: every updated bin holds one of the values written to it.
    ABSL_ASSIGN_OR_RETURN(
        Literal ovw_res,
        RunForced1DScatter(
            absl::StrFormat("ScatterOvw_%s", dtype), dtype, "", kBins,
            kNumUpdates, 8, false,
            "  ROOT out = " + std::string(dtype) + "[] copy(rhs)", init_lit,
            idx_lit, upd_lit));
    std::vector<absl::flat_hash_set<IntT>> allowed(kBins);
    for (int64_t i = 0; i < kNumUpdates; ++i) {
      if (idx[i] >= 0 && idx[i] < kBins) {
        allowed[idx[i]].insert(upd[i]);
      }
    }
    for (int32_t b = 0; b < kBins; ++b) {
      const IntT got = ovw_res.Get<IntT>({b});
      if (allowed[b].empty()) {
        EXPECT_EQ(got, init[b]) << dtype << " untouched bin " << b;
      } else {
        EXPECT_TRUE(allowed[b].contains(got)) << dtype << " ovw bin " << b;
      }
    }

    auto check_exact = [&](absl::string_view tag, absl::string_view body,
                           auto combine) -> absl::Status {
      ABSL_ASSIGN_OR_RETURN(
          Literal res,
          RunForced1DScatter(absl::StrFormat("Scatter_%s_%s", dtype, tag),
                             dtype, "", kBins, kNumUpdates, 8, false, body,
                             init_lit, idx_lit, upd_lit));
      std::vector<IntT> expected = init;
      for (int64_t i = 0; i < kNumUpdates; ++i) {
        if (idx[i] >= 0 && idx[i] < kBins) {
          expected[idx[i]] = combine(expected[idx[i]], upd[i]);
        }
      }
      for (int32_t b = 0; b < kBins; ++b) {
        EXPECT_EQ(res.Get<IntT>({b}), expected[b])
            << dtype << " " << tag << " bin " << b;
      }
      return absl::OkStatus();
    };

    ABSL_RETURN_IF_ERROR(check_exact(
        "add", absl::StrFormat("  ROOT out = %s[] add(lhs, rhs)", dtype),
        [](IntT a, IntT b) { return WrapAdd(a, b); }));
    ABSL_RETURN_IF_ERROR(check_exact(
        "mul", absl::StrFormat("  ROOT out = %s[] multiply(lhs, rhs)", dtype),
        [](IntT a, IntT b) { return WrapMul(a, b); }));
    ABSL_RETURN_IF_ERROR(check_exact(
        "min", absl::StrFormat("  ROOT out = %s[] minimum(lhs, rhs)", dtype),
        [](IntT a, IntT b) { return std::min(a, b); }));
    ABSL_RETURN_IF_ERROR(check_exact(
        "max", absl::StrFormat("  ROOT out = %s[] maximum(lhs, rhs)", dtype),
        [](IntT a, IntT b) { return std::max(a, b); }));
    const std::string custom_body = absl::StrFormat(
        "  three = %s[] constant(3)\n"
        "  scaled = %s[] multiply(rhs, three)\n"
        "  ROOT out = %s[] add(lhs, scaled)",
        dtype, dtype, dtype);
    ABSL_RETURN_IF_ERROR(check_exact("linear", custom_body, [](IntT a, IntT b) {
      return WrapAdd(a, WrapMul(b, static_cast<IntT>(3)));
    }));
    return absl::OkStatus();
  }
};

TEST_F(CpuScatterMultiWorkgroupTest, IntegerAndPredDtypesAndReducers) {
  constexpr int64_t kNumUpdates = 517;
  constexpr int32_t kBins = 19;
  constexpr int32_t kActiveBins = 16;
  const std::vector<int32_t> idx =
      MakeCollidingIndicesWithOob(kNumUpdates, kBins, kActiveBins);
  const Literal idx_lit = MakeColLiteral<int32_t>(S32, idx);

  // Pred overwrite: colliding updates write true to even active bins (initially
  // false) and false to odd active bins (initially true); out of bounds updates
  // carry inverted values and untouched bins preserve their initial values.
  Literal pred_ovw_init(ShapeUtil::MakeShape(PRED, {kBins}));
  absl::Span<bool> pred_ovw_init_data = pred_ovw_init.data<bool>();
  for (int32_t b = 0; b < kBins; ++b) {
    pred_ovw_init_data[b] = (b % 2 != 0);
  }
  Literal pred_ovw_upd(ShapeUtil::MakeShape(PRED, {kNumUpdates, 1}));
  absl::Span<bool> pred_ovw_upd_data = pred_ovw_upd.data<bool>();
  for (int64_t i = 0; i < kNumUpdates; ++i) {
    pred_ovw_upd_data[i] =
        (idx[i] >= 0 && idx[i] < kBins) ? (idx[i] % 2 == 0) : true;
  }
  ASSERT_OK_AND_ASSIGN(
      const Literal pred_ovw_res,
      RunForced1DScatter("ScatterPredOvw", "pred", "", kBins, kNumUpdates, 8,
                         false, "  ROOT out = pred[] copy(rhs)", pred_ovw_init,
                         idx_lit, pred_ovw_upd));
  for (int32_t b = 0; b < kBins; ++b) {
    const bool want = (b < kActiveBins) ? (b % 2 == 0) : (b % 2 != 0);
    EXPECT_EQ(pred_ovw_res.Get<bool>({b}), want) << "pred ovw bin " << b;
  }

  // Pred associative logical reducers: or, and, xor under 8 workgroup races.
  Literal pred_init(ShapeUtil::MakeShape(PRED, {kBins}));
  Literal pred_upd(ShapeUtil::MakeShape(PRED, {kNumUpdates, 1}));
  absl::Span<bool> pred_init_data = pred_init.data<bool>();
  absl::Span<bool> pred_upd_data = pred_upd.data<bool>();
  for (int32_t b = 0; b < kBins; ++b) {
    pred_init_data[b] = (b % 3 == 0);
  }
  for (int64_t i = 0; i < kNumUpdates; ++i) {
    // Keep bin 2 receiving only false and bin 5 receiving only true so logical
    // or and logical and both leave mixed false and true active bins.
    if (idx[i] == 2) {
      pred_upd_data[i] = false;
    } else if (idx[i] == 5) {
      pred_upd_data[i] = true;
    } else {
      pred_upd_data[i] = ((i * 5 + 1) % 3 == 0);
    }
  }
  auto check_pred = [&](absl::string_view op, auto combine) {
    ASSERT_OK_AND_ASSIGN(
        const Literal res,
        RunForced1DScatter(
            absl::StrFormat("ScatterPred_%s", op), "pred", "", kBins,
            kNumUpdates, 8, false,
            absl::StrFormat("  ROOT out = pred[] %s(lhs, rhs)", op), pred_init,
            idx_lit, pred_upd));
    std::vector<uint8_t> expected(kBins);
    for (int32_t b = 0; b < kBins; ++b) {
      expected[b] = pred_init_data[b] ? 1 : 0;
    }
    for (int64_t i = 0; i < kNumUpdates; ++i) {
      if (idx[i] >= 0 && idx[i] < kBins) {
        expected[idx[i]] =
            combine(expected[idx[i]] != 0, pred_upd_data[i]) ? 1 : 0;
      }
    }
    for (int32_t b = 0; b < kBins; ++b) {
      EXPECT_EQ(res.Get<bool>({b}), expected[b] != 0)
          << "pred " << op << " bin " << b;
    }
  };
  check_pred("or", [](bool a, bool b) { return a || b; });
  check_pred("and", [](bool a, bool b) { return a && b; });
  check_pred("xor", [](bool a, bool b) { return a != b; });

  ASSERT_OK(VerifyIntegerScatterSuite<int8_t>(S8, "s8"));
  ASSERT_OK(VerifyIntegerScatterSuite<uint8_t>(U8, "u8"));
  ASSERT_OK(VerifyIntegerScatterSuite<int16_t>(S16, "s16"));
  ASSERT_OK(VerifyIntegerScatterSuite<int32_t>(S32, "s32"));
  ASSERT_OK(VerifyIntegerScatterSuite<int64_t>(S64, "s64"));
}

TEST_F(CpuScatterMultiWorkgroupTest, PackedSubByteS4AndU4MultiWorkgroup) {
  auto wrap_add_s4 = [](s4 a, s4 b) -> s4 {
    const int sum = (static_cast<int>(a) + static_cast<int>(b)) & 0xF;
    return static_cast<s4>(sum >= 8 ? sum - 16 : sum);
  };
  auto wrap_add_u4 = [](u4 a, u4 b) -> u4 {
    return static_cast<u4>((static_cast<int>(a) + static_cast<int>(b)) & 0xF);
  };

  auto run_sub_byte_suite = [&](auto sample, PrimitiveType ptype,
                                absl::string_view dtype, auto from_index,
                                auto add_fn) -> absl::Status {
    using SubT = decltype(sample);
    // Split even and odd indices across distant update partitions so separate
    // workgroups concurrently modify adjacent nibbles sharing every byte.
    constexpr int32_t kUniqueBins = 263;
    constexpr int64_t kUniqueUpdates = 259;
    std::vector<int32_t> uniq_idx;
    uniq_idx.reserve(kUniqueUpdates);
    for (int32_t b = 0; b <= 256; b += 2) {
      uniq_idx.push_back(b);
    }
    uniq_idx.push_back(-3);
    for (int32_t b = 1; b <= 255; b += 2) {
      uniq_idx.push_back(b);
    }
    uniq_idx.push_back(kUniqueBins + 5);
    const Literal uniq_idx_lit = MakeColLiteral<int32_t>(S32, uniq_idx);

    std::vector<SubT> uniq_init(kUniqueBins);
    for (int32_t b = 0; b < kUniqueBins; ++b) {
      uniq_init[b] = from_index(b * 3 + 1);
    }
    std::vector<SubT> uniq_upd(kUniqueUpdates);
    for (int64_t i = 0; i < kUniqueUpdates; ++i) {
      uniq_upd[i] = from_index(static_cast<int32_t>(i * 5 + 2));
    }
    const Literal uniq_init_lit = LiteralUtil::CreateR1<SubT>(uniq_init);
    const Literal uniq_upd_lit = MakeColLiteral<SubT>(ptype, uniq_upd);

    // Forced 8 workgroups with unique_indices=true: overwrite and add.
    ABSL_ASSIGN_OR_RETURN(
        const Literal uniq_ovw,
        RunForced1DScatter(
            absl::StrFormat("SubByteUniqOvw_%s", dtype), dtype, ":E(4)",
            kUniqueBins, kUniqueUpdates, 8, true,
            absl::StrFormat("  ROOT out = %s[] copy(rhs)", dtype),
            uniq_init_lit, uniq_idx_lit, uniq_upd_lit));
    ABSL_ASSIGN_OR_RETURN(
        const Literal uniq_add,
        RunForced1DScatter(
            absl::StrFormat("SubByteUniqAdd_%s", dtype), dtype, ":E(4)",
            kUniqueBins, kUniqueUpdates, 8, true,
            absl::StrFormat("  ROOT out = %s[] add(lhs, rhs)", dtype),
            uniq_init_lit, uniq_idx_lit, uniq_upd_lit));
    std::vector<SubT> want_ovw = uniq_init;
    std::vector<SubT> want_add = uniq_init;
    for (int64_t i = 0; i < kUniqueUpdates; ++i) {
      if (uniq_idx[i] >= 0 && uniq_idx[i] < kUniqueBins) {
        want_ovw[uniq_idx[i]] = uniq_upd[i];
        want_add[uniq_idx[i]] = add_fn(want_add[uniq_idx[i]], uniq_upd[i]);
      }
    }
    for (int32_t b = 0; b < kUniqueBins; ++b) {
      EXPECT_EQ(uniq_ovw.Get<SubT>({b}), want_ovw[b]) << dtype << " uovw " << b;
      EXPECT_EQ(uniq_add.Get<SubT>({b}), want_add[b]) << dtype << " uadd " << b;
    }

    // Forced 8 workgroups with unique_indices=false: colliding add and maximum.
    constexpr int32_t kBins = 23;
    constexpr int64_t kNumUpdates = 517;
    const std::vector<int32_t> col_idx =
        MakeCollidingIndicesWithOob(kNumUpdates, kBins, 19);
    const Literal col_idx_lit = MakeColLiteral<int32_t>(S32, col_idx);
    std::vector<SubT> col_init(kBins);
    for (int32_t b = 0; b < kBins; ++b) {
      col_init[b] = from_index(b);
    }
    std::vector<SubT> col_upd(kNumUpdates);
    for (int64_t i = 0; i < kNumUpdates; ++i) {
      col_upd[i] = from_index(static_cast<int32_t>(i * 3 + 7));
    }
    const Literal col_init_lit = LiteralUtil::CreateR1<SubT>(col_init);
    const Literal col_upd_lit = MakeColLiteral<SubT>(ptype, col_upd);

    ABSL_ASSIGN_OR_RETURN(
        const Literal col_add,
        RunForced1DScatter(
            absl::StrFormat("SubByteColAdd_%s", dtype), dtype, ":E(4)", kBins,
            kNumUpdates, 8, false,
            absl::StrFormat("  ROOT out = %s[] add(lhs, rhs)", dtype),
            col_init_lit, col_idx_lit, col_upd_lit));
    ABSL_ASSIGN_OR_RETURN(
        const Literal col_max,
        RunForced1DScatter(
            absl::StrFormat("SubByteColMax_%s", dtype), dtype, ":E(4)", kBins,
            kNumUpdates, 8, false,
            absl::StrFormat("  ROOT out = %s[] maximum(lhs, rhs)", dtype),
            col_init_lit, col_idx_lit, col_upd_lit));
    std::vector<SubT> want_col_add = col_init;
    std::vector<SubT> want_col_max = col_init;
    for (int64_t i = 0; i < kNumUpdates; ++i) {
      if (col_idx[i] >= 0 && col_idx[i] < kBins) {
        want_col_add[col_idx[i]] = add_fn(want_col_add[col_idx[i]], col_upd[i]);
        want_col_max[col_idx[i]] =
            std::max(want_col_max[col_idx[i]], col_upd[i]);
      }
    }
    for (int32_t b = 0; b < kBins; ++b) {
      EXPECT_EQ(col_add.Get<SubT>({b}), want_col_add[b]) << dtype << " add";
      EXPECT_EQ(col_max.Get<SubT>({b}), want_col_max[b]) << dtype << " max";
    }

    // Full CPU HLO pipeline execution on unfused packed sub byte scatter.
    const std::string pipeline_hlo = absl::StrFormat(
        R"hlo(
HloModule FullPipelineSubByte_%s
add_op {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT out = %s[] add(lhs, rhs)
}
ENTRY main {
  operand = %s[%d] parameter(0)
  indices = s32[%d,1] parameter(1)
  updates = %s[%d] parameter(2)
  ROOT res = %s[%d] scatter(operand, indices, updates),
    update_window_dims={},
    inserted_window_dims={0},
    scatter_dims_to_operand_dims={0},
    index_vector_dim=1,
    to_apply=add_op
}
)hlo",
        dtype, dtype, dtype, dtype, dtype, kBins, kNumUpdates, dtype,
        kNumUpdates, dtype, kBins);
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> mod,
                          ParseAndReturnVerifiedModule(pipeline_hlo));
    const Literal upd_1d = LiteralUtil::CreateR1<SubT>(col_upd);
    ABSL_ASSIGN_OR_RETURN(
        const Literal pipe_res,
        Execute(std::move(mod), {&col_init_lit, &col_idx_lit, &upd_1d}));
    for (int32_t b = 0; b < kBins; ++b) {
      EXPECT_EQ(pipe_res.Get<SubT>({b}), want_col_add[b]) << dtype << " pipe";
    }
    return absl::OkStatus();
  };

  ASSERT_OK(run_sub_byte_suite(
      s4(0), S4, "s4",
      [](int32_t v) { return static_cast<s4>(((v % 16 + 16) % 16) - 8); },
      wrap_add_s4));
  ASSERT_OK(run_sub_byte_suite(
      u4(0), U4, "u4",
      [](int32_t v) { return static_cast<u4>((v % 16 + 16) % 16); },
      wrap_add_u4));
}

TEST_F(CpuScatterMultiWorkgroupTest, FloatAndComplexDtypesMultiWorkgroup) {
  constexpr int64_t kNumUpdates = 517;
  constexpr int32_t kBins = 19;
  constexpr int32_t kActiveBins = 16;
  const std::vector<int32_t> idx =
      MakeCollidingIndicesWithOob(kNumUpdates, kBins, kActiveBins);
  const Literal idx_lit = MakeColLiteral<int32_t>(S32, idx);

  // Every possible partial subset sum per bin stays within [-90, 90], so all
  // intermediate sums are bit exact in bf16, f16, f32, and f64 under any order.
  auto check_real_add = [&](auto sample, PrimitiveType ptype,
                            absl::string_view dtype,
                            bool also_run_pipeline) -> absl::Status {
    using RealT = decltype(sample);
    std::vector<RealT> init(kBins);
    std::vector<int32_t> expected(kBins);
    for (int32_t b = 0; b < kBins; ++b) {
      expected[b] = (b % 7) - 3;
      init[b] = static_cast<RealT>(static_cast<float>(expected[b]));
    }
    constexpr int32_t kSteps[4] = {-2, -1, 1, 2};
    std::vector<RealT> upd(kNumUpdates);
    for (int64_t i = 0; i < kNumUpdates; ++i) {
      const int32_t delta = kSteps[i % 4];
      upd[i] = static_cast<RealT>(static_cast<float>(delta));
      if (idx[i] >= 0 && idx[i] < kBins) {
        expected[idx[i]] += delta;
      }
    }
    const Literal init_lit = LiteralUtil::CreateR1<RealT>(init);
    const Literal upd_lit = MakeColLiteral<RealT>(ptype, upd);
    ABSL_ASSIGN_OR_RETURN(
        const Literal res,
        RunForced1DScatter(
            absl::StrFormat("FloatAdd_%s", dtype), dtype, "", kBins,
            kNumUpdates, 8, false,
            absl::StrFormat("  ROOT out = %s[] add(lhs, rhs)", dtype), init_lit,
            idx_lit, upd_lit));
    auto to_double = [](RealT v) -> double {
      if constexpr (std::is_same_v<RealT, double>) {
        return v;
      } else {
        return static_cast<double>(static_cast<float>(v));
      }
    };
    for (int32_t b = 0; b < kBins; ++b) {
      EXPECT_EQ(to_double(res.Get<RealT>({b})),
                static_cast<double>(expected[b]))
          << dtype << " add bin " << b;
    }
    if (also_run_pipeline) {
      const std::string pipe_hlo = absl::StrFormat(
          R"hlo(
HloModule FullPipelineFloat_%s
add_op {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT out = %s[] add(lhs, rhs)
}
ENTRY main {
  operand = %s[%d] parameter(0)
  indices = s32[%d,1] parameter(1)
  updates = %s[%d] parameter(2)
  ROOT res = %s[%d] scatter(operand, indices, updates),
    update_window_dims={}, inserted_window_dims={0},
    scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=add_op
}
)hlo",
          dtype, dtype, dtype, dtype, dtype, kBins, kNumUpdates, dtype,
          kNumUpdates, dtype, kBins);
      ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> mod,
                            ParseAndReturnVerifiedModule(pipe_hlo));
      const Literal upd_1d = LiteralUtil::CreateR1<RealT>(upd);
      ABSL_ASSIGN_OR_RETURN(
          const Literal pipe_res,
          Execute(std::move(mod), {&init_lit, &idx_lit, &upd_1d}));
      for (int32_t b = 0; b < kBins; ++b) {
        EXPECT_EQ(to_double(pipe_res.Get<RealT>({b})),
                  static_cast<double>(expected[b]))
            << dtype << " pipeline bin " << b;
      }
    }
    return absl::OkStatus();
  };

  ASSERT_OK(check_real_add(half(0.0f), F16, "f16", true));
  ASSERT_OK(check_real_add(bfloat16(0.0f), BF16, "bf16", true));
  ASSERT_OK(check_real_add(0.0f, F32, "f32", true));
  ASSERT_OK(check_real_add(0.0, F64, "f64", false));

  // Extra f32 reducers: minimum, maximum, overwrite, and custom lhs + rhs * 3.
  std::vector<float> f32_init(kBins);
  for (int32_t b = 0; b < kBins; ++b) {
    f32_init[b] = static_cast<float>(b - 9);
  }
  std::vector<float> f32_upd(kNumUpdates);
  std::vector<absl::flat_hash_set<int32_t>> f32_allowed(kBins);
  std::vector<float> want_min = f32_init;
  std::vector<float> want_max = f32_init;
  std::vector<float> want_lin = f32_init;
  for (int64_t i = 0; i < kNumUpdates; ++i) {
    const int32_t val = static_cast<int32_t>((i % 41) - 20);
    f32_upd[i] = static_cast<float>(val);
    if (idx[i] >= 0 && idx[i] < kBins) {
      f32_allowed[idx[i]].insert(val);
      want_min[idx[i]] = std::min(want_min[idx[i]], f32_upd[i]);
      want_max[idx[i]] = std::max(want_max[idx[i]], f32_upd[i]);
      want_lin[idx[i]] += 3.0f * f32_upd[i];
    }
  }
  const Literal f32_init_lit = LiteralUtil::CreateR1<float>(f32_init);
  const Literal f32_upd_lit = MakeColLiteral<float>(F32, f32_upd);
  ASSERT_OK_AND_ASSIGN(
      const Literal f32_min,
      RunForced1DScatter("F32Min", "f32", "", kBins, kNumUpdates, 8, false,
                         "  ROOT out = f32[] minimum(lhs, rhs)", f32_init_lit,
                         idx_lit, f32_upd_lit));
  ASSERT_OK_AND_ASSIGN(
      const Literal f32_max,
      RunForced1DScatter("F32Max", "f32", "", kBins, kNumUpdates, 8, false,
                         "  ROOT out = f32[] maximum(lhs, rhs)", f32_init_lit,
                         idx_lit, f32_upd_lit));
  ASSERT_OK_AND_ASSIGN(
      const Literal f32_ovw,
      RunForced1DScatter("F32Ovw", "f32", "", kBins, kNumUpdates, 8, false,
                         "  ROOT out = f32[] copy(rhs)", f32_init_lit, idx_lit,
                         f32_upd_lit));
  ASSERT_OK_AND_ASSIGN(
      const Literal f32_lin,
      RunForced1DScatter("F32Lin", "f32", "", kBins, kNumUpdates, 8, false,
                         "  c3 = f32[] constant(3)\n"
                         "  scaled = f32[] multiply(rhs, c3)\n"
                         "  ROOT out = f32[] add(lhs, scaled)",
                         f32_init_lit, idx_lit, f32_upd_lit));
  for (int32_t b = 0; b < kBins; ++b) {
    EXPECT_EQ(f32_min.Get<float>({b}), want_min[b]);
    EXPECT_EQ(f32_max.Get<float>({b}), want_max[b]);
    EXPECT_EQ(f32_lin.Get<float>({b}), want_lin[b]);
    if (f32_allowed[b].empty()) {
      EXPECT_EQ(f32_ovw.Get<float>({b}), f32_init[b]);
    } else {
      EXPECT_TRUE(f32_allowed[b].contains(
          static_cast<int32_t>(f32_ovw.Get<float>({b}))));
    }
  }

  // Complex c64 and c128: exact integer colliding add, torn write detection on
  // colliding overwrite (every written value satisfies imag == -real - 1), and
  // full CPU HLO pipeline execution.
  constexpr int64_t kComplexUpdates = 1031;
  constexpr int32_t kComplexBins = 11;
  constexpr int32_t kComplexActive = 8;
  const std::vector<int32_t> cidx = MakeCollidingIndicesWithOob(
      kComplexUpdates, kComplexBins, kComplexActive);
  const Literal cidx_lit = MakeColLiteral<int32_t>(S32, cidx);

  auto check_complex = [&](auto sample, PrimitiveType ptype,
                           absl::string_view dtype) -> absl::Status {
    using ComplexT = decltype(sample);
    using ScalarT = typename ComplexT::value_type;
    std::vector<ComplexT> init(kComplexBins);
    std::vector<ComplexT> want_add(kComplexBins);
    for (int32_t b = 0; b < kComplexBins; ++b) {
      const ScalarT re = static_cast<ScalarT>(-(b + 10));
      init[b] = ComplexT(re, -re - static_cast<ScalarT>(1));
      want_add[b] = init[b];
    }
    std::vector<ComplexT> add_upd(kComplexUpdates);
    std::vector<ComplexT> ovw_upd(kComplexUpdates);
    std::vector<absl::flat_hash_set<int64_t>> ovw_reals(kComplexBins);
    for (int64_t i = 0; i < kComplexUpdates; ++i) {
      add_upd[i] = ComplexT(static_cast<ScalarT>((i % 7) - 3),
                            static_cast<ScalarT>(((i * 3) % 5) - 2));
      if (cidx[i] >= 0 && cidx[i] < kComplexBins) {
        want_add[cidx[i]] += add_upd[i];
        const int64_t re_i = i + 1;
        const ScalarT re = static_cast<ScalarT>(re_i);
        ovw_upd[i] = ComplexT(re, -re - static_cast<ScalarT>(1));
        ovw_reals[cidx[i]].insert(re_i);
      } else {
        // Deliberately violate imag == -real - 1 on out of bounds entries.
        ovw_upd[i] =
            ComplexT(static_cast<ScalarT>(999), static_cast<ScalarT>(999));
      }
    }
    const Literal init_lit = LiteralUtil::CreateR1<ComplexT>(init);
    const Literal add_upd_lit = MakeColLiteral<ComplexT>(ptype, add_upd);
    const Literal ovw_upd_lit = MakeColLiteral<ComplexT>(ptype, ovw_upd);

    ABSL_ASSIGN_OR_RETURN(
        const Literal add_res,
        RunForced1DScatter(
            absl::StrFormat("ComplexAdd_%s", dtype), dtype, "", kComplexBins,
            kComplexUpdates, 8, false,
            absl::StrFormat("  ROOT out = %s[] add(lhs, rhs)", dtype), init_lit,
            cidx_lit, add_upd_lit));
    ABSL_ASSIGN_OR_RETURN(
        const Literal ovw_res,
        RunForced1DScatter(
            absl::StrFormat("ComplexOvw_%s", dtype), dtype, "", kComplexBins,
            kComplexUpdates, 8, false,
            absl::StrFormat("  ROOT out = %s[] copy(rhs)", dtype), init_lit,
            cidx_lit, ovw_upd_lit));
    for (int32_t b = 0; b < kComplexBins; ++b) {
      EXPECT_EQ(add_res.Get<ComplexT>({b}), want_add[b]) << dtype << " add";
      const ComplexT got = ovw_res.Get<ComplexT>({b});
      EXPECT_EQ(got.imag(), -got.real() - static_cast<ScalarT>(1))
          << dtype << " torn overwrite at bin " << b;
      if (ovw_reals[b].empty()) {
        EXPECT_EQ(got, init[b]) << dtype << " untouched bin " << b;
      } else {
        EXPECT_TRUE(ovw_reals[b].contains(static_cast<int64_t>(got.real())))
            << dtype << " unexpected real at bin " << b;
      }
    }

    const std::string pipe_hlo = absl::StrFormat(
        R"hlo(
HloModule FullPipelineComplex_%s
add_op {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT out = %s[] add(lhs, rhs)
}
ENTRY main {
  operand = %s[%d] parameter(0)
  indices = s32[%d,1] parameter(1)
  updates = %s[%d] parameter(2)
  ROOT res = %s[%d] scatter(operand, indices, updates),
    update_window_dims={}, inserted_window_dims={0},
    scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=add_op
}
)hlo",
        dtype, dtype, dtype, dtype, dtype, kComplexBins, kComplexUpdates, dtype,
        kComplexUpdates, dtype, kComplexBins);
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> mod,
                          ParseAndReturnVerifiedModule(pipe_hlo));
    const Literal upd_1d = LiteralUtil::CreateR1<ComplexT>(add_upd);
    ABSL_ASSIGN_OR_RETURN(
        const Literal pipe_res,
        Execute(std::move(mod), {&init_lit, &cidx_lit, &upd_1d}));
    for (int32_t b = 0; b < kComplexBins; ++b) {
      EXPECT_EQ(pipe_res.Get<ComplexT>({b}), want_add[b]) << dtype << " pipe";
    }
    return absl::OkStatus();
  };

  ASSERT_OK(check_complex(complex64(0.0f, 0.0f), C64, "c64"));
  ASSERT_OK(check_complex(complex128(0.0, 0.0), C128, "c128"));
}

TEST_F(CpuScatterMultiWorkgroupTest, VariadicRank0ClampingAndWindowSlices) {
  auto build_variadic_hlo = [](absl::string_view name, int32_t bins,
                               int64_t num_upd, bool unique_indices,
                               absl::string_view reducer_body) {
    return absl::StrFormat(
        R"hlo(
HloModule %s
reducer {
  lhs0 = s32[] parameter(0)
  lhs1 = f32[] parameter(1)
  rhs0 = s32[] parameter(2)
  rhs1 = f32[] parameter(3)
%s
}
scatter_computation {
  op0 = s32[%d]{0} parameter(0)
  op1 = f32[%d]{0} parameter(1)
  idx = s32[%d,1]{1,0} parameter(2)
  upd0 = s32[%d,1]{1,0} parameter(3)
  upd1 = f32[%d,1]{1,0} parameter(4)
  ROOT res = (s32[%d]{0}, f32[%d]{0}) scatter(op0, op1, idx, upd0, upd1),
    update_window_dims={1}, inserted_window_dims={},
    scatter_dims_to_operand_dims={0}, index_vector_dim=1,
    unique_indices=%s, to_apply=reducer
}
ENTRY main {
  p0 = s32[%d]{0} parameter(0)
  c0 = s32[%d]{0} copy(p0)
  p1 = f32[%d]{0} parameter(1)
  c1 = f32[%d]{0} copy(p1)
  p2 = s32[%d,1]{1,0} parameter(2)
  p3 = s32[%d,1]{1,0} parameter(3)
  p4 = f32[%d,1]{1,0} parameter(4)
  ROOT wrapped = (s32[%d]{0}, f32[%d]{0}) fusion(c0, c1, p2, p3, p4),
    kind=kLoop, calls=scatter_computation,
    backend_config={"outer_dimension_partitions":["8"]}
}
)hlo",
        name, reducer_body, bins, bins, num_upd, num_upd, num_upd, bins, bins,
        unique_indices ? "true" : "false", bins, bins, bins, bins, num_upd,
        num_upd, num_upd, bins, bins);
  };

  // Variadic (s32, f32) with unique_indices=true runs across 8 workgroups;
  // unique_indices=false falls back to 1 workgroup (exact last write wins).
  constexpr int32_t kVarBins = 269;
  constexpr int64_t kVarUpd = 259;
  std::vector<int32_t> op0_init(kVarBins);
  std::vector<float> op1_init(kVarBins);
  for (int32_t b = 0; b < kVarBins; ++b) {
    op0_init[b] = 1000 + b;
    op1_init[b] = static_cast<float>(-100 - b);
  }
  std::vector<int32_t> uniq_idx(kVarUpd);
  std::vector<int32_t> upd0_vec(kVarUpd);
  std::vector<float> upd1_vec(kVarUpd);
  for (int64_t i = 0; i < kVarUpd; ++i) {
    // 100 is coprime to 257, producing a permutation of [0, 257) plus 2 OOBs.
    if (i == 257) {
      uniq_idx[i] = -5;
    } else if (i == 258) {
      uniq_idx[i] = kVarBins;
    } else {
      uniq_idx[i] = static_cast<int32_t>((i * 100 + 17) % 257);
    }
    upd0_vec[i] = static_cast<int32_t>(i * 3 + 1);
    upd1_vec[i] = static_cast<float>(i + 1) * 0.5f;
  }
  const Literal op0_lit = LiteralUtil::CreateR1<int32_t>(op0_init);
  const Literal op1_lit = LiteralUtil::CreateR1<float>(op1_init);
  const Literal uniq_idx_lit = MakeColLiteral<int32_t>(S32, uniq_idx);
  const Literal upd0_lit = MakeColLiteral<int32_t>(S32, upd0_vec);
  const Literal upd1_lit = MakeColLiteral<float>(F32, upd1_vec);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> var_uniq_mod,
                       ParseAndReturnVerifiedModule(build_variadic_hlo(
                           "VariadicUnique", kVarBins, kVarUpd, true,
                           "  sum0 = s32[] add(lhs0, rhs0)\n"
                           "  ROOT out = (s32[], f32[]) tuple(sum0, rhs1)")));
  ASSERT_OK_AND_ASSIGN(
      const Literal var_uniq_res,
      Execute(std::move(var_uniq_mod),
              {&op0_lit, &op1_lit, &uniq_idx_lit, &upd0_lit, &upd1_lit},
              /*run_hlo_passes=*/false));
  std::vector<int32_t> want_u0 = op0_init;
  std::vector<float> want_u1 = op1_init;
  for (int64_t i = 0; i < kVarUpd; ++i) {
    if (uniq_idx[i] >= 0 && uniq_idx[i] < kVarBins) {
      want_u0[uniq_idx[i]] += upd0_vec[i];
      want_u1[uniq_idx[i]] = upd1_vec[i];
    }
  }
  const LiteralSlice var_uniq_0(var_uniq_res, {0});
  const LiteralSlice var_uniq_1(var_uniq_res, {1});
  for (int32_t b = 0; b < kVarBins; ++b) {
    EXPECT_EQ(var_uniq_0.Get<int32_t>({b}), want_u0[b]);
    EXPECT_EQ(var_uniq_1.Get<float>({b}), want_u1[b]);
  }

  const std::vector<int32_t> col_idx =
      MakeCollidingIndicesWithOob(kVarUpd, kVarBins, 13);
  const Literal col_idx_lit = MakeColLiteral<int32_t>(S32, col_idx);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> var_nonuniq_mod,
      ParseAndReturnVerifiedModule(build_variadic_hlo(
          "VariadicNonUniqueFallback", kVarBins, kVarUpd, false,
          "  ROOT out = (s32[], f32[]) tuple(rhs0, rhs1)")));
  ASSERT_OK_AND_ASSIGN(
      const Literal var_nonuniq_res,
      Execute(std::move(var_nonuniq_mod),
              {&op0_lit, &op1_lit, &col_idx_lit, &upd0_lit, &upd1_lit},
              /*run_hlo_passes=*/false));
  std::vector<int32_t> want_seq0 = op0_init;
  std::vector<float> want_seq1 = op1_init;
  for (int64_t i = 0; i < kVarUpd; ++i) {
    if (col_idx[i] >= 0 && col_idx[i] < kVarBins) {
      want_seq0[col_idx[i]] = upd0_vec[i];
      want_seq1[col_idx[i]] = upd1_vec[i];
    }
  }
  const LiteralSlice var_seq_0(var_nonuniq_res, {0});
  const LiteralSlice var_seq_1(var_nonuniq_res, {1});
  for (int32_t b = 0; b < kVarBins; ++b) {
    EXPECT_EQ(var_seq_0.Get<int32_t>({b}), want_seq0[b]);
    EXPECT_EQ(var_seq_1.Get<float>({b}), want_seq1[b]);
  }

  // Single element s32[1] with 8 workgroups and N = 257 (every valid update
  // contends on element 0, mixed with negative and out of bounds indices).
  constexpr int64_t kScalarUpd = 257;
  std::vector<int32_t> one_idx(kScalarUpd);
  std::vector<int32_t> one_upd(kScalarUpd);
  std::vector<double> f64_scalar_upd(kScalarUpd);
  int32_t want_one = 42;
  int32_t want_r0_s32 = 100;
  double want_r0_f64 = 12.0;
  for (int64_t i = 0; i < kScalarUpd; ++i) {
    one_upd[i] = static_cast<int32_t>((i % 11) - 5);
    f64_scalar_upd[i] = static_cast<double>((i % 9) - 4);
    want_r0_s32 += one_upd[i];
    want_r0_f64 += f64_scalar_upd[i];
    if (i % 17 == 1) {
      one_idx[i] = -1;
    } else if (i % 17 == 9) {
      one_idx[i] = 1;
    } else {
      one_idx[i] = 0;
      want_one += one_upd[i];
    }
  }
  const Literal one_init = LiteralUtil::CreateR1<int32_t>({42});
  const Literal one_idx_lit = MakeColLiteral<int32_t>(S32, one_idx);
  const Literal one_upd_lit = MakeColLiteral<int32_t>(S32, one_upd);
  ASSERT_OK_AND_ASSIGN(
      const Literal one_res,
      RunForced1DScatter("SingleElementS32", "s32", "", 1, kScalarUpd, 8, false,
                         "  ROOT out = s32[] add(lhs, rhs)", one_init,
                         one_idx_lit, one_upd_lit));
  EXPECT_EQ(one_res.Get<int32_t>({0}), want_one);

  // Rank 0 s32[] and f64[] operands with 8 workgroups and N = 257.
  auto run_rank0 = [&](absl::string_view dtype, const Literal& init,
                       const Literal& upd) -> absl::StatusOr<Literal> {
    const std::string hlo = absl::StrFormat(
        R"hlo(
HloModule Rank0MultiWorkgroup_%s
add_op {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT out = %s[] add(lhs, rhs)
}
scatter_computation {
  operand = %s[] parameter(0)
  indices = s32[%d,0]{1,0} parameter(1)
  updates = %s[%d]{0} parameter(2)
  ROOT res = %s[] scatter(operand, indices, updates),
    update_window_dims={}, inserted_window_dims={},
    scatter_dims_to_operand_dims={}, index_vector_dim=1, to_apply=add_op
}
ENTRY main {
  p0 = %s[] parameter(0)
  c0 = %s[] copy(p0)
  p1 = s32[%d,0]{1,0} parameter(1)
  p2 = %s[%d]{0} parameter(2)
  ROOT wrapped = %s[] fusion(c0, p1, p2),
    kind=kLoop, calls=scatter_computation,
    backend_config={"outer_dimension_partitions":["8"]}
}
)hlo",
        dtype, dtype, dtype, dtype, dtype, kScalarUpd, dtype, kScalarUpd, dtype,
        dtype, dtype, kScalarUpd, dtype, kScalarUpd, dtype);
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> mod,
                          ParseAndReturnVerifiedModule(hlo));
    const Literal empty_idx(ShapeUtil::MakeShape(S32, {kScalarUpd, 0}));
    return Execute(std::move(mod), {&init, &empty_idx, &upd},
                   /*run_hlo_passes=*/false);
  };
  ASSERT_OK_AND_ASSIGN(const Literal r0_s32_res,
                       run_rank0("s32", LiteralUtil::CreateR0<int32_t>(100),
                                 LiteralUtil::CreateR1<int32_t>(one_upd)));
  EXPECT_EQ(r0_s32_res.Get<int32_t>({}), want_r0_s32);
  ASSERT_OK_AND_ASSIGN(
      const Literal r0_f64_res,
      run_rank0("f64", LiteralUtil::CreateR0<double>(12.0),
                LiteralUtil::CreateR1<double>(f64_scalar_upd)));
  EXPECT_EQ(r0_f64_res.Get<double>({}), want_r0_f64);

  // Requested 16 workgroups clamped down to N = 5 updates.
  const Literal clamp_init =
      LiteralUtil::CreateR1<int32_t>({10, 20, 30, 40, 50, 60});
  const std::vector<int32_t> clamp_idx = {2, -1, 2, 6, 4};
  const std::vector<int32_t> clamp_upd = {7, 99, -3, 88, 15};
  ASSERT_OK_AND_ASSIGN(
      const Literal clamp_res,
      RunForced1DScatter("ClampPartitions16To5", "s32", "", 6, 5, 16, false,
                         "  ROOT out = s32[] add(lhs, rhs)", clamp_init,
                         MakeColLiteral<int32_t>(S32, clamp_idx),
                         MakeColLiteral<int32_t>(S32, clamp_upd)));
  EXPECT_EQ(clamp_res,
            LiteralUtil::CreateR1<int32_t>({10, 20, 34, 40, 65, 60}));

  // Window slice f32[32, 64] with slice [1, 64]: row major contiguous add,
  // overwrite, unique overwrite with out of bounds indices, plus column major
  // {0,1} layout window slice.
  constexpr int32_t kRows = 32;
  constexpr int32_t kCols = 64;
  auto run_window_slice =
      [&](absl::string_view name, absl::string_view op_layout, int64_t num_upd,
          bool unique_indices, absl::string_view reducer, const Literal& init,
          const Literal& idx_lit,
          const Literal& upd_lit) -> absl::StatusOr<Literal> {
    const std::string hlo = absl::StrFormat(
        R"hlo(
HloModule %s
reducer {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
%s
}
scatter_computation {
  operand = f32[32,64]{%s} parameter(0)
  indices = s32[%d,1]{1,0} parameter(1)
  updates = f32[%d,1,64]{2,1,0} parameter(2)
  ROOT res = f32[32,64]{%s} scatter(operand, indices, updates),
    update_window_dims={1,2}, inserted_window_dims={},
    scatter_dims_to_operand_dims={0}, index_vector_dim=1,
    unique_indices=%s, to_apply=reducer
}
ENTRY main {
  p0 = f32[32,64]{%s} parameter(0)
  c0 = f32[32,64]{%s} copy(p0)
  p1 = s32[%d,1]{1,0} parameter(1)
  p2 = f32[%d,1,64]{2,1,0} parameter(2)
  ROOT wrapped = f32[32,64]{%s} fusion(c0, p1, p2),
    kind=kLoop, calls=scatter_computation,
    backend_config={"outer_dimension_partitions":["8"]}
}
)hlo",
        name, reducer, op_layout, num_upd, num_upd, op_layout,
        unique_indices ? "true" : "false", op_layout, op_layout, num_upd,
        num_upd, op_layout);
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> mod,
                          ParseAndReturnVerifiedModule(hlo));
    return Execute(std::move(mod), {&init, &idx_lit, &upd_lit},
                   /*run_hlo_passes=*/false);
  };

  constexpr int64_t kWinUpd = 259;
  const std::vector<int32_t> win_idx =
      MakeCollidingIndicesWithOob(kWinUpd, kRows, 28);
  const Literal win_idx_lit = MakeColLiteral<int32_t>(S32, win_idx);
  Literal row_init(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {kRows, kCols}, {1, 0}));
  Literal col_init(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {kRows, kCols}, {0, 1}));
  Array2D<float> want_win_add(kRows, kCols);
  for (int32_t r = 0; r < kRows; ++r) {
    for (int32_t c = 0; c < kCols; ++c) {
      const float v = static_cast<float>(r * 10 - c);
      row_init.Set<float>({r, c}, v);
      col_init.Set<float>({r, c}, v);
      want_win_add(r, c) = v;
    }
  }
  Literal win_upd(ShapeUtil::MakeShape(F32, {kWinUpd, 1, kCols}));
  std::vector<absl::flat_hash_set<int32_t>> row_tags(kRows);
  for (int64_t i = 0; i < kWinUpd; ++i) {
    const int32_t tag = static_cast<int32_t>((i % 7) + 1);
    if (win_idx[i] >= 0 && win_idx[i] < kRows) {
      row_tags[win_idx[i]].insert(tag);
    }
    for (int32_t c = 0; c < kCols; ++c) {
      const float v = static_cast<float>(tag * 128 + c);
      win_upd.Set<float>({i, 0, c}, v);
      if (win_idx[i] >= 0 && win_idx[i] < kRows) {
        want_win_add(win_idx[i], c) += v;
      }
    }
  }

  ASSERT_OK_AND_ASSIGN(
      const Literal win_add_res,
      run_window_slice("WindowSliceRowMajorAdd", "1,0", kWinUpd, false,
                       "  ROOT out = f32[] add(lhs, rhs)", row_init,
                       win_idx_lit, win_upd));
  ASSERT_OK_AND_ASSIGN(
      const Literal win_ovw_res,
      run_window_slice("WindowSliceRowMajorOvw", "1,0", kWinUpd, false,
                       "  ROOT out = f32[] copy(rhs)", row_init, win_idx_lit,
                       win_upd));
  ASSERT_OK_AND_ASSIGN(
      const Literal win_col_res,
      run_window_slice("WindowSliceColMajorAdd", "0,1", kWinUpd, false,
                       "  ROOT out = f32[] add(lhs, rhs)", col_init,
                       win_idx_lit, win_upd));
  for (int32_t r = 0; r < kRows; ++r) {
    for (int32_t c = 0; c < kCols; ++c) {
      EXPECT_EQ(win_add_res.Get<float>({r, c}), want_win_add(r, c));
      EXPECT_EQ(win_col_res.Get<float>({r, c}), want_win_add(r, c));
      const float ovw_v = win_ovw_res.Get<float>({r, c});
      if (row_tags[r].empty()) {
        EXPECT_EQ(ovw_v, row_init.Get<float>({r, c}));
      } else {
        const int32_t delta = static_cast<int32_t>(ovw_v) - c;
        EXPECT_EQ(delta % 128, 0);
        EXPECT_TRUE(row_tags[r].contains(delta / 128));
      }
    }
  }

  // Unique window slice overwrite with negative and out of bounds indices.
  constexpr int64_t kUniqWinUpd = 31;
  std::vector<int32_t> uwin_idx(kUniqWinUpd);
  Literal uwin_upd(ShapeUtil::MakeShape(F32, {kUniqWinUpd, 1, kCols}));
  Array2D<float> want_uwin(kRows, kCols);
  for (int32_t r = 0; r < kRows; ++r) {
    for (int32_t c = 0; c < kCols; ++c) {
      want_uwin(r, c) = row_init.Get<float>({r, c});
    }
  }
  for (int64_t i = 0; i < kUniqWinUpd; ++i) {
    if (i == 10) {
      uwin_idx[i] = -3;
    } else if (i == 22) {
      uwin_idx[i] = kRows;
    } else {
      uwin_idx[i] = static_cast<int32_t>((i * 9 + 5) % kRows);
    }
    for (int32_t c = 0; c < kCols; ++c) {
      const float v = static_cast<float>(5000 + i * 64 + c);
      uwin_upd.Set<float>({i, 0, c}, v);
      if (uwin_idx[i] >= 0 && uwin_idx[i] < kRows) {
        want_uwin(uwin_idx[i], c) = v;
      }
    }
  }
  ASSERT_OK_AND_ASSIGN(
      const Literal uwin_res,
      run_window_slice("WindowSliceUniqueOvw", "1,0", kUniqWinUpd, true,
                       "  ROOT out = f32[] copy(rhs)", row_init,
                       MakeColLiteral<int32_t>(S32, uwin_idx), uwin_upd));
  for (int32_t r = 0; r < kRows; ++r) {
    for (int32_t c = 0; c < kCols; ++c) {
      EXPECT_EQ(uwin_res.Get<float>({r, c}), want_uwin(r, c));
    }
  }
}

TEST_F(CpuScatterMultiWorkgroupTest,
       FusedProducersAliasingDeterminismAndKernelCache) {
  // 1. Fused index and update producers under forced 8 workgroup execution.
  constexpr absl::string_view kFusedProducersHlo = R"hlo(
HloModule ScatterFusedIndexAndUpdateProducers
add_s32 {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT out = s32[] add(lhs, rhs)
}
scatter_computation {
  operand = s32[19]{0} parameter(0)
  raw_idx = s32[517,1]{1,0} parameter(1)
  idx_bias = s32[517,1]{1,0} parameter(2)
  raw_upd = s32[517,1]{1,0} parameter(3)
  upd_bias = s32[517,1]{1,0} parameter(4)
  idx = s32[517,1]{1,0} add(raw_idx, idx_bias)
  c2 = s32[] constant(2)
  c2_bcast = s32[517,1]{1,0} broadcast(c2), dimensions={}
  scaled = s32[517,1]{1,0} multiply(raw_upd, c2_bcast)
  upd = s32[517,1]{1,0} subtract(scaled, upd_bias)
  ROOT res = s32[19]{0} scatter(operand, idx, upd),
    update_window_dims={1}, inserted_window_dims={},
    scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=add_s32
}
ENTRY main {
  p0 = s32[19]{0} parameter(0)
  c0 = s32[19]{0} copy(p0)
  p1 = s32[517,1]{1,0} parameter(1)
  p2 = s32[517,1]{1,0} parameter(2)
  p3 = s32[517,1]{1,0} parameter(3)
  p4 = s32[517,1]{1,0} parameter(4)
  ROOT wrapped = s32[19]{0} fusion(c0, p1, p2, p3, p4),
    kind=kLoop, calls=scatter_computation,
    backend_config={"outer_dimension_partitions":["8"]}
}
)hlo";
  constexpr int64_t kNumUpdates = 517;
  constexpr int32_t kBins = 19;
  const std::vector<int32_t> target_idx =
      MakeCollidingIndicesWithOob(kNumUpdates, kBins, 16);
  std::vector<int32_t> raw_idx(kNumUpdates);
  std::vector<int32_t> idx_bias(kNumUpdates);
  std::vector<int32_t> raw_upd(kNumUpdates);
  std::vector<int32_t> upd_bias(kNumUpdates);
  std::vector<int32_t> init(kBins);
  for (int32_t b = 0; b < kBins; ++b) {
    init[b] = b * 5 - 20;
  }
  std::vector<int32_t> want_fused = init;
  for (int64_t i = 0; i < kNumUpdates; ++i) {
    const int32_t shift = static_cast<int32_t>((i % 7) - 3);
    // Avoid signed overflow when target_idx[i] is INT32_MAX.
    idx_bias[i] = (target_idx[i] > kBins) ? 0 : shift;
    raw_idx[i] = target_idx[i] - idx_bias[i];
    raw_upd[i] = static_cast<int32_t>((i % 13) - 6);
    upd_bias[i] = static_cast<int32_t>((i % 5) - 2);
    if (target_idx[i] >= 0 && target_idx[i] < kBins) {
      want_fused[target_idx[i]] += 2 * raw_upd[i] - upd_bias[i];
    }
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> fused_prod_mod,
                       ParseAndReturnVerifiedModule(kFusedProducersHlo));
  const Literal init_lit = LiteralUtil::CreateR1<int32_t>(init);
  const Literal p1_lit = MakeColLiteral<int32_t>(S32, raw_idx);
  const Literal p2_lit = MakeColLiteral<int32_t>(S32, idx_bias);
  const Literal p3_lit = MakeColLiteral<int32_t>(S32, raw_upd);
  const Literal p4_lit = MakeColLiteral<int32_t>(S32, upd_bias);
  ASSERT_OK_AND_ASSIGN(const Literal fused_prod_res,
                       Execute(std::move(fused_prod_mod),
                               {&init_lit, &p1_lit, &p2_lit, &p3_lit, &p4_lit},
                               /*run_hlo_passes=*/false));
  EXPECT_EQ(fused_prod_res, LiteralUtil::CreateR1<int32_t>(want_fused));

  // 2a. Self aliasing scatter(x, idx, x + x) through the full CPU pipeline.
  constexpr int64_t kAliasN = 257;
  constexpr absl::string_view kAliasPipelineHlo = R"hlo(
HloModule SelfAliasingScatterFullPipeline
add_s32 {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT out = s32[] add(lhs, rhs)
}
ENTRY main {
  x = s32[257] parameter(0)
  idx = s32[257,1] parameter(1)
  doubled = s32[257] add(x, x)
  ROOT res = s32[257] scatter(x, idx, doubled),
    update_window_dims={}, inserted_window_dims={0},
    scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=add_s32
}
)hlo";
  std::vector<int32_t> alias_x(kAliasN);
  std::vector<int32_t> alias_rev_idx(kAliasN);
  for (int64_t i = 0; i < kAliasN; ++i) {
    alias_x[i] = static_cast<int32_t>(i * 3 + 5);
    alias_rev_idx[i] = static_cast<int32_t>((kAliasN - 1 - i) % 29);
  }
  std::vector<int32_t> want_alias_pipe = alias_x;
  for (int64_t i = 0; i < kAliasN; ++i) {
    want_alias_pipe[alias_rev_idx[i]] += 2 * alias_x[i];
  }
  const Literal alias_x_lit = LiteralUtil::CreateR1<int32_t>(alias_x);
  const Literal alias_rev_idx_lit = MakeColLiteral<int32_t>(S32, alias_rev_idx);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> alias_pipe_mod,
                       ParseAndReturnVerifiedModule(kAliasPipelineHlo));
  ASSERT_OK_AND_ASSIGN(
      const Literal alias_pipe_res,
      Execute(std::move(alias_pipe_mod), {&alias_x_lit, &alias_rev_idx_lit}));
  EXPECT_EQ(alias_pipe_res, LiteralUtil::CreateR1<int32_t>(want_alias_pipe));

  // 2b. Pre fused self aliasing scatter with run_hlo_passes=false falls back
  // from 8 workgroups to single threaded execution without data races.
  constexpr absl::string_view kAliasFallbackHlo = R"hlo(
HloModule SelfAliasingScatterForcedFallback
ovw_s32 {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT out = s32[] copy(rhs)
}
scatter_computation {
  operand = s32[257]{0} parameter(0)
  idx = s32[257,1]{1,0} parameter(1)
  aliased_src = s32[257,1]{1,0} parameter(2)
  doubled = s32[257,1]{1,0} add(aliased_src, aliased_src)
  ROOT res = s32[257]{0} scatter(operand, idx, doubled),
    update_window_dims={1}, inserted_window_dims={},
    scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=ovw_s32
}
ENTRY main {
  p0 = s32[257]{0} parameter(0)
  c0 = s32[257]{0} copy(p0)
  p1 = s32[257,1]{1,0} parameter(1)
  bc = s32[257,1]{1,0} bitcast(c0)
  ROOT wrapped = s32[257]{0} fusion(c0, p1, bc),
    kind=kLoop, calls=scatter_computation,
    backend_config={"outer_dimension_partitions":["8"]}
}
)hlo";
  std::vector<int32_t> alias_fb_idx(kAliasN);
  std::vector<int32_t> want_alias_fb = alias_x;
  for (int64_t i = 0; i < kAliasN; ++i) {
    alias_fb_idx[i] = static_cast<int32_t>(i % 8);
    const int32_t upd_val = 2 * want_alias_fb[i];
    want_alias_fb[alias_fb_idx[i]] = upd_val;
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> alias_fb_mod,
                       ParseAndReturnVerifiedModule(kAliasFallbackHlo));
  const Literal alias_fb_idx_lit = MakeColLiteral<int32_t>(S32, alias_fb_idx);
  ASSERT_OK_AND_ASSIGN(
      const Literal alias_fb_res,
      Execute(std::move(alias_fb_mod), {&alias_x_lit, &alias_fb_idx_lit},
              /*run_hlo_passes=*/false));
  EXPECT_EQ(alias_fb_res, LiteralUtil::CreateR1<int32_t>(want_alias_fb));

  // 3. Deterministic mode forces single threaded sequential execution for order
  // sensitive overwrite and floating point cancellation sum even with ["8"].
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> det_ovw_mod,
      ParseAndReturnVerifiedModule(BuildForced1DScatterHlo(
          "DeterministicOverwrite", "s32", "", kBins, kNumUpdates, 8, false,
          "  ROOT out = s32[] copy(rhs)")));
  det_ovw_mod->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_exclude_nondeterministic_ops(true);
  std::vector<int32_t> det_upd(kNumUpdates);
  std::vector<int32_t> want_det_ovw = init;
  for (int64_t i = 0; i < kNumUpdates; ++i) {
    det_upd[i] = static_cast<int32_t>(10000 + i);
    if (target_idx[i] >= 0 && target_idx[i] < kBins) {
      want_det_ovw[target_idx[i]] = det_upd[i];
    }
  }
  const Literal target_idx_lit = MakeColLiteral<int32_t>(S32, target_idx);
  const Literal det_upd_lit = MakeColLiteral<int32_t>(S32, det_upd);
  ASSERT_OK_AND_ASSIGN(const Literal det_ovw_res,
                       Execute(std::move(det_ovw_mod),
                               {&init_lit, &target_idx_lit, &det_upd_lit},
                               /*run_hlo_passes=*/false));
  EXPECT_EQ(det_ovw_res, LiteralUtil::CreateR1<int32_t>(want_det_ovw));

  constexpr int64_t kCancelUpd = 261;
  std::vector<int32_t> cancel_idx(kCancelUpd, 0);
  std::vector<float> cancel_upd(kCancelUpd);
  constexpr float kTriple[3] = {1e7f, -1e7f, 1.0f};
  float want_cancel = 0.0f;
  for (int64_t i = 0; i < kCancelUpd; ++i) {
    cancel_upd[i] = kTriple[i % 3];
    want_cancel += cancel_upd[i];
  }
  ASSERT_EQ(want_cancel, 87.0f);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> det_f32_mod,
      ParseAndReturnVerifiedModule(BuildForced1DScatterHlo(
          "DeterministicF32Cancellation", "f32", "", 2, kCancelUpd, 8, false,
          "  ROOT out = f32[] add(lhs, rhs)")));
  det_f32_mod->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_exclude_nondeterministic_ops(true);
  const Literal cancel_init = LiteralUtil::CreateR1<float>({0.0f, -5.5f});
  const Literal cancel_idx_lit = MakeColLiteral<int32_t>(S32, cancel_idx);
  const Literal cancel_upd_lit = MakeColLiteral<float>(F32, cancel_upd);
  ASSERT_OK_AND_ASSIGN(const Literal det_f32_res,
                       Execute(std::move(det_f32_mod),
                               {&cancel_init, &cancel_idx_lit, &cancel_upd_lit},
                               /*run_hlo_passes=*/false));
  EXPECT_EQ(det_f32_res.Get<float>({0}), want_cancel);
  EXPECT_EQ(det_f32_res.Get<float>({1}), -5.5f);

  // 4. Two identical scatter fusions in one module with ["1"] vs ["8"] must not
  // collide in kernel_spec_cache_ (tested in both compilation orders).
  constexpr absl::string_view kKernelCacheHlo = R"hlo(
HloModule ScatterKernelSpecCacheSeparation
add_s32 {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT out = s32[] add(lhs, rhs)
}
ovw_s32 {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT out = s32[] copy(rhs)
}
scatter_add_a {
  op = s32[19]{0} parameter(0)
  idx = s32[517,1]{1,0} parameter(1)
  upd = s32[517,1]{1,0} parameter(2)
  ROOT res = s32[19]{0} scatter(op, idx, upd),
    update_window_dims={1}, inserted_window_dims={},
    scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=add_s32
}
scatter_add_b {
  op = s32[19]{0} parameter(0)
  idx = s32[517,1]{1,0} parameter(1)
  upd = s32[517,1]{1,0} parameter(2)
  ROOT res = s32[19]{0} scatter(op, idx, upd),
    update_window_dims={1}, inserted_window_dims={},
    scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=add_s32
}
scatter_ovw_a {
  op = s32[19]{0} parameter(0)
  idx = s32[517,1]{1,0} parameter(1)
  upd = s32[517,1]{1,0} parameter(2)
  ROOT res = s32[19]{0} scatter(op, idx, upd),
    update_window_dims={1}, inserted_window_dims={},
    scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=ovw_s32
}
scatter_ovw_b {
  op = s32[19]{0} parameter(0)
  idx = s32[517,1]{1,0} parameter(1)
  upd = s32[517,1]{1,0} parameter(2)
  ROOT res = s32[19]{0} scatter(op, idx, upd),
    update_window_dims={1}, inserted_window_dims={},
    scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=ovw_s32
}
ENTRY main {
  p0 = s32[19]{0} parameter(0)
  idx = s32[517,1]{1,0} parameter(1)
  upd = s32[517,1]{1,0} parameter(2)
  c_add_1 = s32[19]{0} copy(p0)
  f_add_1 = s32[19]{0} fusion(c_add_1, idx, upd),
    kind=kLoop, calls=scatter_add_a,
    backend_config={"outer_dimension_partitions":["1"]}
  c_add_8 = s32[19]{0} copy(p0)
  f_add_8 = s32[19]{0} fusion(c_add_8, idx, upd),
    kind=kLoop, calls=scatter_add_b,
    backend_config={"outer_dimension_partitions":["8"]}
  c_ovw_8 = s32[19]{0} copy(p0)
  f_ovw_8 = s32[19]{0} fusion(c_ovw_8, idx, upd),
    kind=kLoop, calls=scatter_ovw_a,
    backend_config={"outer_dimension_partitions":["8"]}
  c_ovw_1 = s32[19]{0} copy(p0)
  f_ovw_1 = s32[19]{0} fusion(c_ovw_1, idx, upd),
    kind=kLoop, calls=scatter_ovw_b,
    backend_config={"outer_dimension_partitions":["1"]}
  ROOT out = (s32[19]{0}, s32[19]{0}, s32[19]{0}, s32[19]{0}) tuple(
      f_add_1, f_add_8, f_ovw_8, f_ovw_1)
}
)hlo";
  std::vector<int32_t> want_cache_add = init;
  std::vector<absl::flat_hash_set<int32_t>> cache_allowed(kBins);
  for (int64_t i = 0; i < kNumUpdates; ++i) {
    if (target_idx[i] >= 0 && target_idx[i] < kBins) {
      want_cache_add[target_idx[i]] += det_upd[i];
      cache_allowed[target_idx[i]].insert(det_upd[i]);
    }
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> cache_mod,
                       ParseAndReturnVerifiedModule(kKernelCacheHlo));
  ASSERT_OK_AND_ASSIGN(
      const Literal cache_res,
      Execute(std::move(cache_mod), {&init_lit, &target_idx_lit, &det_upd_lit},
              /*run_hlo_passes=*/false));
  const Literal want_cache_add_lit =
      LiteralUtil::CreateR1<int32_t>(want_cache_add);
  EXPECT_EQ(LiteralSlice(cache_res, {0}), want_cache_add_lit);
  EXPECT_EQ(LiteralSlice(cache_res, {1}), want_cache_add_lit);
  EXPECT_EQ(LiteralSlice(cache_res, {3}),
            LiteralUtil::CreateR1<int32_t>(want_det_ovw));
  const LiteralSlice cache_ovw_8(cache_res, {2});
  for (int32_t b = 0; b < kBins; ++b) {
    const int32_t got = cache_ovw_8.Get<int32_t>({b});
    if (cache_allowed[b].empty()) {
      EXPECT_EQ(got, init[b]);
    } else {
      EXPECT_TRUE(cache_allowed[b].contains(got));
    }
  }
}

}  // namespace
}  // namespace cpu
}  // namespace xla
