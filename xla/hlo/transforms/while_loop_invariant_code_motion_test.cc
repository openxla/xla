/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/transforms/while_loop_invariant_code_motion.h"

#include <gmock/gmock.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/transforms/copy_insertion.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class WhileLoopInvariantCodeMotionTest : public HloHardwareIndependentTestBase {
 public:
  // Makes a computation which has one parameter, of the given shape, and always
  // returns PRED[]{true}.  This is useful as a dummy loop condition.
  HloComputation* MakeAlwaysTrueComputation(const Shape& param_shape,
                                            HloModule* module);
};

static void FindOnlyWhileInstruction(HloComputation* computation,
                                     HloInstruction** while_instruction) {
  *while_instruction = nullptr;
  for (auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kWhile) {
      ASSERT_EQ(*while_instruction, nullptr);
      *while_instruction = instr;
    }
  }

  ASSERT_NE(*while_instruction, nullptr);
}

HloComputation* WhileLoopInvariantCodeMotionTest::MakeAlwaysTrueComputation(
    const Shape& param_shape, HloModule* module) {
  HloComputation::Builder builder(TestName() + ".always_true");
  builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  return module->AddEmbeddedComputation(builder.Build());
}

TEST_F(WhileLoopInvariantCodeMotionTest, HoistOneInvariantOperation) {
  auto m = CreateNewVerifiedModule();
  auto array_s32 = ShapeUtil::MakeShape(S32, {2});
  Shape while_shape =
      ShapeUtil::MakeTupleShape({array_s32, array_s32, array_s32});

  HloComputation* while_body = [&]() {
    HloComputation::Builder builder(TestName() + ".while_body");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloInstruction* gte_0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(array_s32, param, 0));
    HloInstruction* gte_1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(array_s32, param, 1));
    HloInstruction* add_result = builder.AddInstruction(
        HloInstruction::CreateBinary(array_s32, HloOpcode::kAdd, gte_0, gte_1));
    builder.AddInstruction(
        HloInstruction::CreateTuple({gte_0, gte_1, add_result}));

    return m->AddEmbeddedComputation(builder.Build());
  }();

  HloComputation::Builder builder(TestName());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateParameter(0, while_shape, "init_value"));
  builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, m.get()), while_body,
      init_value));
  HloComputation* entry_computation = m->AddEntryComputation(builder.Build());
  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(m.get()));
  EXPECT_TRUE(simplified_loop);

  HloInstruction* transformed_while;
  FindOnlyWhileInstruction(entry_computation, &transformed_while);

  EXPECT_THAT(entry_computation->instructions(), Contains(op::Add()));
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(op::Add())));
}

TEST_F(WhileLoopInvariantCodeMotionTest, HoistInvariantOperationTree) {
  auto m = CreateNewVerifiedModule();
  auto array_s32 = ShapeUtil::MakeShape(S32, {2});
  Shape while_shape =
      ShapeUtil::MakeTupleShape({array_s32, array_s32, array_s32});

  HloComputation* while_body = [&]() {
    HloComputation::Builder builder(TestName() + ".while_body");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloInstruction* gte_0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(array_s32, param, 0));
    HloInstruction* gte_1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(array_s32, param, 1));
    HloInstruction* gte_2_loop_variant = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(array_s32, param, 2));

    HloInstruction* add_result = builder.AddInstruction(
        HloInstruction::CreateBinary(array_s32, HloOpcode::kAdd, gte_0, gte_1));
    HloInstruction* mul_result =
        builder.AddInstruction(HloInstruction::CreateBinary(
            array_s32, HloOpcode::kMultiply, add_result, gte_1));
    HloInstruction* negate_result = builder.AddInstruction(
        HloInstruction::CreateUnary(array_s32, HloOpcode::kNegate, mul_result));
    HloInstruction* constant = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<int32_t>({4, 4})));
    HloInstruction* sub_result =
        builder.AddInstruction(HloInstruction::CreateBinary(
            array_s32, HloOpcode::kSubtract, negate_result, constant));
    HloInstruction* divide_result =
        builder.AddInstruction(HloInstruction::CreateBinary(
            array_s32, HloOpcode::kDivide, sub_result, gte_2_loop_variant));
    builder.AddInstruction(
        HloInstruction::CreateTuple({gte_0, gte_1, divide_result}));

    return m->AddEmbeddedComputation(builder.Build());
  }();

  HloComputation::Builder builder(TestName());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateParameter(0, while_shape, "init_value"));
  builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, m.get()), while_body,
      init_value));
  HloComputation* entry_computation = m->AddEntryComputation(builder.Build());
  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(m.get()));
  EXPECT_TRUE(simplified_loop);

  HloInstruction* transformed_while;
  FindOnlyWhileInstruction(entry_computation, &transformed_while);

  EXPECT_THAT(entry_computation->instructions(),
              AllOf(Contains(op::Add()), Contains(op::Multiply()),
                    Contains(op::Negate()), Contains(op::Subtract()),
                    Contains(op::Constant()),

                    // The division had a loop varying operand so that better
                    // not be hoisted.
                    Not(Contains(op::Divide()))));

  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(AnyOf(op::Add(), op::Multiply(), op::Negate(),
                             op::Subtract(), op::Constant()))));

  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Contains(op::Divide()));
}

TEST_F(WhileLoopInvariantCodeMotionTest,
       DontHoistTriviallyLoopVaryingComputation) {
  // Basic negative test: the add expression is not loop invariant.
  auto m = CreateNewVerifiedModule();
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  Shape while_shape = ShapeUtil::MakeTupleShape({scalar_s32, scalar_s32});

  HloComputation* while_body = [&]() {
    HloComputation::Builder builder(TestName() + ".while_body");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloInstruction* gte_0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 0));
    HloInstruction* gte_1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 1));
    HloInstruction* add_result =
        builder.AddInstruction(HloInstruction::CreateBinary(
            scalar_s32, HloOpcode::kAdd, gte_0, gte_1));
    builder.AddInstruction(HloInstruction::CreateTuple({gte_0, add_result}));

    return m->AddEmbeddedComputation(builder.Build());
  }();

  HloComputation::Builder builder(TestName());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateParameter(0, while_shape, "init_value"));
  auto* while_inst = builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, m.get()), while_body,
      init_value));

  m->AddEntryComputation(builder.Build());

  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(m.get()));
  EXPECT_FALSE(simplified_loop);

  EXPECT_THAT(while_inst->while_body()->instructions(), Contains(op::Add()));
}

TEST_F(WhileLoopInvariantCodeMotionTest,
       DontHoistLoopVaryingComputationWithAlternatingTuples) {
  auto m = CreateNewVerifiedModule();
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  Shape while_shape =
      ShapeUtil::MakeTupleShape({scalar_s32, scalar_s32, scalar_s32});

  HloComputation* while_body = [&]() {
    HloComputation::Builder builder(TestName() + ".while_body");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloInstruction* gte_0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 0));
    HloInstruction* gte_1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 1));
    HloInstruction* add_result =
        builder.AddInstruction(HloInstruction::CreateBinary(
            scalar_s32, HloOpcode::kAdd, gte_0, gte_1));
    builder.AddInstruction(
        HloInstruction::CreateTuple({gte_1, gte_0, add_result}));

    return m->AddEmbeddedComputation(builder.Build());
  }();

  HloComputation::Builder builder(TestName());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateParameter(0, while_shape, "init_value"));
  auto* while_inst = builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, m.get()), while_body,
      init_value));

  m->AddEntryComputation(builder.Build());
  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(m.get()));
  EXPECT_FALSE(simplified_loop);

  EXPECT_THAT(while_inst->while_body()->instructions(), Contains(op::Add()));
}

TEST_F(WhileLoopInvariantCodeMotionTest, DontHoistInstructionWithSideEffects) {
  auto m = CreateNewVerifiedModule();
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  auto token_shape = ShapeUtil::MakeTokenShape();
  Shape while_shape =
      ShapeUtil::MakeTupleShape({scalar_s32, scalar_s32, token_shape});

  HloComputation* while_body = [&]() {
    HloComputation::Builder builder(TestName() + ".while_body");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloInstruction* gte_0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 0));
    HloInstruction* gte_1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 1));
    HloInstruction* in_token = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(token_shape, param, 2));
    HloInstruction* out_token = builder.AddInstruction(
        HloInstruction::CreateOutfeed(scalar_s32, gte_0, in_token, ""));
    builder.AddInstruction(
        HloInstruction::CreateTuple({gte_0, gte_1, out_token}));

    return m->AddEmbeddedComputation(builder.Build());
  }();

  HloComputation::Builder builder(TestName());
  auto* scalar_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_s32, "param"));
  auto* token = builder.AddInstruction(HloInstruction::CreateToken());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateTuple({scalar_param, scalar_param, token}));
  auto* while_inst = builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, m.get()), while_body,
      init_value));
  builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_s32, while_inst, 0));
  m->AddEntryComputation(builder.Build());

  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(m.get()));
  ASSERT_FALSE(simplified_loop);

  EXPECT_THAT(while_inst->while_body()->instructions(),
              Contains(op::Outfeed()));
}

TEST_F(WhileLoopInvariantCodeMotionTest, DontHoistBitcastAlone) {
  // The bitcast's user, an outfeed, can't be hoisted, so don't hoist the
  // bitcast either.
  auto m = CreateNewVerifiedModule();
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  auto effective_scalar_s32 = ShapeUtil::MakeShape(S32, {1});
  auto token_shape = ShapeUtil::MakeTokenShape();
  Shape while_shape =
      ShapeUtil::MakeTupleShape({scalar_s32, scalar_s32, token_shape});

  HloComputation* while_body = [&]() {
    HloComputation::Builder builder(TestName() + ".while_body");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloInstruction* gte_0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 0));
    HloInstruction* gte_1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 1));
    HloInstruction* in_token = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(token_shape, param, 2));
    HloInstruction* bitcast_inst =
        builder.AddInstruction(HloInstruction::CreateUnary(
            effective_scalar_s32, HloOpcode::kBitcast, gte_0));
    HloInstruction* out_token =
        builder.AddInstruction(HloInstruction::CreateOutfeed(
            effective_scalar_s32, bitcast_inst, in_token, ""));
    builder.AddInstruction(
        HloInstruction::CreateTuple({gte_0, gte_1, out_token}));

    return m->AddEmbeddedComputation(builder.Build());
  }();

  HloComputation::Builder builder(TestName());
  auto* scalar_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_s32, "param"));
  auto* token = builder.AddInstruction(HloInstruction::CreateToken());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateTuple({scalar_param, scalar_param, token}));
  auto* while_inst = builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, m.get()), while_body,
      init_value));
  builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_s32, while_inst, 0));

  m->AddEntryComputation(builder.Build());

  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(m.get()));
  EXPECT_FALSE(simplified_loop);

  EXPECT_THAT(while_inst->while_body()->instructions(),
              Contains(op::Outfeed()));
  EXPECT_THAT(while_inst->while_body()->instructions(),
              Contains(op::Bitcast()));
}

TEST_F(WhileLoopInvariantCodeMotionTest, HoistBitcastIfNeeded) {
  // The bitcast's user can be hoisted, so hoist the bitcast too.
  auto m = CreateNewVerifiedModule();
  auto array_s32 = ShapeUtil::MakeShape(S32, {2});
  Shape while_shape =
      ShapeUtil::MakeTupleShape({array_s32, array_s32, array_s32});

  HloComputation* while_body = [&]() {
    HloComputation::Builder builder(TestName() + ".while_body");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloInstruction* gte_0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(array_s32, param, 0));
    HloInstruction* gte_1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(array_s32, param, 1));
    HloInstruction* bitcast_inst = builder.AddInstruction(
        HloInstruction::CreateUnary(array_s32, HloOpcode::kBitcast, gte_0));
    HloInstruction* add_inst =
        builder.AddInstruction(HloInstruction::CreateBinary(
            array_s32, HloOpcode::kAdd, bitcast_inst, gte_1));
    builder.AddInstruction(
        HloInstruction::CreateTuple({gte_0, gte_1, add_inst}));

    return m->AddEmbeddedComputation(builder.Build());
  }();

  HloComputation::Builder builder(TestName());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateParameter(0, while_shape, "init_value"));
  builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, m.get()), while_body,
      init_value));

  HloComputation* entry_computation = m->AddEntryComputation(builder.Build());

  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(m.get()));
  EXPECT_TRUE(simplified_loop);

  HloInstruction* transformed_while;
  FindOnlyWhileInstruction(entry_computation, &transformed_while);

  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(op::Add())));
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(op::Bitcast())));
  EXPECT_THAT(entry_computation->instructions(), Contains(op::Add()));
  EXPECT_THAT(entry_computation->instructions(), Contains(op::Bitcast()));
}

TEST_F(WhileLoopInvariantCodeMotionTest, DontHoistControlDependencies) {
  auto m = CreateNewVerifiedModule();
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  Shape while_shape =
      ShapeUtil::MakeTupleShape({scalar_s32, scalar_s32, scalar_s32});

  HloComputation* while_body;
  {
    HloComputation::Builder builder(TestName() + ".while_body");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloInstruction* gte_0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 0));
    HloInstruction* gte_1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 1));
    HloInstruction* add_result =
        builder.AddInstruction(HloInstruction::CreateBinary(
            scalar_s32, HloOpcode::kAdd, gte_0, gte_1));
    ASSERT_OK(param->AddControlDependencyTo(add_result));
    builder.AddInstruction(
        HloInstruction::CreateTuple({gte_0, gte_1, add_result}));

    while_body = m->AddEmbeddedComputation(builder.Build());
  }

  HloComputation::Builder builder(TestName());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateParameter(0, while_shape, "init_value"));
  builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, m.get()), while_body,
      init_value));
  m->AddEntryComputation(builder.Build());
  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(m.get()));
  EXPECT_FALSE(simplified_loop);
}

TEST_F(WhileLoopInvariantCodeMotionTest, BodyHasNonTupleRoot) {
  auto m = CreateNewVerifiedModule();
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  Shape while_shape = ShapeUtil::MakeTupleShape({scalar_s32, scalar_s32});

  HloComputation* while_body = [&]() {
    HloComputation::Builder builder(TestName() + ".passthrough");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloComputation* result = m->AddEmbeddedComputation(builder.Build());

    result->AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 1));
    return result;
  }();

  HloComputation::Builder builder(TestName());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateParameter(0, while_shape, "init_value"));
  builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, m.get()), while_body,
      init_value));
  m->AddEntryComputation(builder.Build());
  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(m.get()));
  EXPECT_FALSE(simplified_loop);
}

const char* const kConstantHoistingTestCase = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[2]{0}) parameter(0)
  p_body.1 = f32[2]{0} get-tuple-element(p_body), index=0
  const = f32[2]{0} constant({3, 4})
  add.0 = f32[2]{0} add(p_body.1, const)
  ROOT root = (f32[2]{0}) tuple(add.0)
}

condition {
  p_cond = (f32[2]{0}) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[2]{0} constant({1, 2})
  while_init = (f32[2]{0}) tuple(const_0)
  ROOT while = (f32[2]{0}) while(while_init), condition=condition, body=body
}
)";

TEST_F(WhileLoopInvariantCodeMotionTest, HoistsConstantWhenAsked) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(kConstantHoistingTestCase));

  ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      WhileLoopInvariantCodeMotion{/*hoist_constants=*/true}.Run(m.get()));
  EXPECT_TRUE(simplified_loop);

  HloComputation* while_body = m->GetComputationWithName("wide.body");
  ASSERT_NE(while_body, nullptr);

  // We expect the while body to be the equivalent of:
  //
  //  wide.body {
  //    wide.p_body = (f32[2]{0}, f32[2]{0}) parameter(0)
  //    p_body.2 = get-tuple-element(wide.p_body), index=0
  //    wide.body.in.0 = get-tuple-element(wide.p_body), index=1
  //    add.1 = add(p_body.2, wide.body.in.0)
  //    wide.body.through.0 = get-tuple-element(wide.p_body), index=1
  //    ROOT tuple = (f32[2]{0}, f32[2]{0}) tuple(add.1, wide.body.through.0)
  //  }

  auto wide_p_body = op::Parameter(0);
  auto p_body_2 = op::GetTupleElement(wide_p_body, 0);
  auto wide_body_in_0 = op::GetTupleElement(wide_p_body, 1);
  auto add_1 = op::Add(p_body_2, wide_body_in_0);
  auto wide_body_through_0 = op::GetTupleElement(wide_p_body, 1);
  auto tuple = op::Tuple(add_1, wide_body_through_0);

  EXPECT_THAT(while_body->root_instruction(), tuple);
}

TEST_F(WhileLoopInvariantCodeMotionTest, DoesNotHoistConstantByDefault) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(kConstantHoistingTestCase));

  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(m.get()));
  EXPECT_FALSE(simplified_loop);
}

TEST_F(WhileLoopInvariantCodeMotionTest, DoNotHoistOutOfSingleIteration) {
  const char* const kHloModule = R"(
    HloModule ModuleWithWhile

    body {
      p_body = (f32[2], f32[2], f32[2], s32[]) parameter(0)
      val.0 = f32[2] get-tuple-element(p_body), index=0
      val.1 = f32[2] get-tuple-element(p_body), index=1
      add = f32[2] add(val.0, val.1)
      const = s32[] constant(-1)
      ROOT root = (f32[2], f32[2], f32[2], s32[]) tuple(val.0, val.1, add, const)
    }

    condition {
      p_cond = (f32[2], f32[2], f32[2], s32[]) parameter(0)
      gte = s32[] get-tuple-element(p_cond), index=3
      const = s32[] constant(42)
      ROOT result = pred[] compare(gte, const), direction=EQ
    }

    ENTRY entry {
      param.0 = f32[2] parameter(0)
      param.1 = s32[] parameter(1)
      while_init = (f32[2], f32[2], f32[2], s32[]) tuple(param.0, param.0, param.0, param.1)
      ROOT while = (f32[2], f32[2], f32[2], s32[]) while(while_init), condition=condition, body=body
    })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));

  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(module.get()));
  EXPECT_FALSE(simplified_loop);
}

const char* const kInflatingTestCase = R"(
HloModule ModuleWithWhile

mul {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT mul = f32[] multiply(lhs, rhs)
}

body {
  p_body = (f32[]) parameter(0)
  iota = f32[1024, 1024] iota(), iota_dimension=0
  add = f32[1024, 1024] add(iota, iota)
  constant = f32[] constant(1.0)
  reduce = f32[] reduce(f32[1024, 1024] add, f32[] constant), dimensions={0,1}, to_apply=mul
  ROOT root = (f32[]) tuple(reduce)
}

condition {
  p_cond = (f32[]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  param = f32[] parameter(0)
  while_init = (f32[]) tuple(param)
  ROOT while = (f32[]) while(while_init), condition=condition, body=body
}
)";

TEST_F(WhileLoopInvariantCodeMotionTest, HoistsInflatingByDefault) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(kInflatingTestCase));

  ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      WhileLoopInvariantCodeMotion(/*hoist_constants=*/true).Run(m.get()));
  EXPECT_TRUE(simplified_loop);

  HloComputation* while_body = m->GetComputationWithName("wide.body");
  ASSERT_NE(while_body, nullptr);
  EXPECT_THAT(while_body->instructions(), Not(Contains(op::Iota())));
}

// The previous call misbound its positional arguments (1.0 landed on
// hoist_other), so the ratio was never set and the check never ran.
TEST_F(WhileLoopInvariantCodeMotionTest, NoHoistInflating) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(kInflatingTestCase));

  ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      WhileLoopInvariantCodeMotion(/*hoist_constants=*/true,
                                   /*hoist_reshapes=*/false,
                                   /*hoist_other=*/true,
                                   /*hoist_size_inflation_ratio=*/1.0)
          .Run(m.get()));
  // The constant is hoisted; the iota inflates and stays in the loop.
  EXPECT_TRUE(simplified_loop);

  HloComputation* while_body = m->GetComputationWithName("wide.body");
  ASSERT_NE(while_body, nullptr);
  EXPECT_THAT(while_body->instructions(), Contains(op::Iota()));
  EXPECT_THAT(while_body->instructions(), Not(Contains(op::Constant())));
}

const char* const kInflatingChainTestCase = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[8], f32[8], f32[8,1024]) parameter(0)
  invariant = f32[8] get-tuple-element(p_body), index=0
  negate = f32[8] negate(invariant)
  broadcast = f32[8,1024] broadcast(invariant), dimensions={0}
  negate_broadcast = f32[8,1024] negate(broadcast)
  ROOT root = (f32[8], f32[8], f32[8,1024]) tuple(invariant, negate, negate_broadcast)
}

condition {
  p_cond = (f32[8], f32[8], f32[8,1024]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  param = (f32[8], f32[8], f32[8,1024]) parameter(0)
  ROOT while = (f32[8], f32[8], f32[8,1024]) while(param), condition=condition, body=body
}
)";

TEST_F(WhileLoopInvariantCodeMotionTest, HoistsInflatingChainWithoutRatio) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(kInflatingChainTestCase));

  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(m.get()));
  EXPECT_TRUE(simplified_loop);

  HloInstruction* transformed_while;
  FindOnlyWhileInstruction(m->entry_computation(), &transformed_while);
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(AnyOf(op::Negate(), op::Broadcast()))));
  EXPECT_THAT(m->entry_computation()->instructions(),
              Contains(op::Negate(op::Broadcast())));
}

TEST_F(WhileLoopInvariantCodeMotionTest,
       SizeInflationRatioRejectsOnlyInflatingInstructions) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(kInflatingChainTestCase));

  ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      WhileLoopInvariantCodeMotion(/*hoist_constants=*/false,
                                   /*hoist_reshapes=*/false,
                                   /*hoist_other=*/true,
                                   /*hoist_size_inflation_ratio=*/1.0)
          .Run(m.get()));
  EXPECT_TRUE(simplified_loop);

  // negate(invariant) does not inflate and is hoisted. The broadcast inflates
  // and stays in the loop, so its negate has an unhoisted operand and stays
  // too.
  HloInstruction* transformed_while;
  FindOnlyWhileInstruction(m->entry_computation(), &transformed_while);
  EXPECT_THAT(
      m->entry_computation()->instructions(),
      AllOf(Contains(op::Negate(op::GetTupleElement(op::Parameter(0), 0))),
            Not(Contains(op::Broadcast()))));
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              AllOf(Contains(op::Negate(op::Broadcast())),
                    Each(Not(op::Negate(op::GetTupleElement())))));
}

TEST_F(WhileLoopInvariantCodeMotionTest,
       SizeInflationRatioNeverHoistsLoopVaryingOperands) {
  // Both converts shrink their input, so the ratio permits them; only the one
  // with an invariant operand may be hoisted.
  const char* const kHloModule = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[8,128], f32[8,128], bf16[8,128], bf16[8,128]) parameter(0)
  invariant = f32[8,128] get-tuple-element(p_body), index=0
  varying = f32[8,128] get-tuple-element(p_body), index=1
  convert_invariant = bf16[8,128] convert(invariant)
  convert_varying = bf16[8,128] convert(varying)
  negate = f32[8,128] negate(varying)
  ROOT root = (f32[8,128], f32[8,128], bf16[8,128], bf16[8,128]) tuple(invariant, negate, convert_invariant, convert_varying)
}

condition {
  p_cond = (f32[8,128], f32[8,128], bf16[8,128], bf16[8,128]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  param = (f32[8,128], f32[8,128], bf16[8,128], bf16[8,128]) parameter(0)
  ROOT while = (f32[8,128], f32[8,128], bf16[8,128], bf16[8,128]) while(param), condition=condition, body=body
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(kHloModule));

  ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      WhileLoopInvariantCodeMotion(/*hoist_constants=*/false,
                                   /*hoist_reshapes=*/false,
                                   /*hoist_other=*/true,
                                   /*hoist_size_inflation_ratio=*/1.0)
          .Run(m.get()));
  EXPECT_TRUE(simplified_loop);

  HloInstruction* transformed_while;
  FindOnlyWhileInstruction(m->entry_computation(), &transformed_while);
  EXPECT_THAT(m->entry_computation()->instructions(),
              Contains(op::Convert(op::GetTupleElement(op::Parameter(0), 0))));
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Contains(op::Convert(op::GetTupleElement(op::Parameter(0), 1))));
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(op::Convert(op::GetTupleElement(op::Parameter(0), 0)))));
}

TEST_F(WhileLoopInvariantCodeMotionTest,
       SizeInflationCountsAllLeavesOfTupleOperands) {
  // get-tuple-element(custom-call), index=1 produces 2048 bytes out of a 2080
  // byte tuple operand: hoistable at ratio 1.0 only if the operand tuple's
  // leaves are summed; a tuple counted as size 0 rejects it.
  const char* const kHloModule = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[8], f32[8,64], f32[8,64]) parameter(0)
  invariant_a = f32[8] get-tuple-element(p_body), index=0
  invariant_b = f32[8,64] get-tuple-element(p_body), index=1
  custom_call = (f32[8], f32[8,64]) custom-call(invariant_a, invariant_b), custom_call_target="Foo"
  element = f32[8,64] get-tuple-element(custom_call), index=1
  negate = f32[8,64] negate(element)
  ROOT root = (f32[8], f32[8,64], f32[8,64]) tuple(invariant_a, invariant_b, negate)
}

condition {
  p_cond = (f32[8], f32[8,64], f32[8,64]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  param = (f32[8], f32[8,64], f32[8,64]) parameter(0)
  ROOT while = (f32[8], f32[8,64], f32[8,64]) while(param), condition=condition, body=body
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(kHloModule));

  ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      WhileLoopInvariantCodeMotion(/*hoist_constants=*/false,
                                   /*hoist_reshapes=*/false,
                                   /*hoist_other=*/true,
                                   /*hoist_size_inflation_ratio=*/1.0)
          .Run(m.get()));
  EXPECT_TRUE(simplified_loop);

  HloInstruction* transformed_while;
  FindOnlyWhileInstruction(m->entry_computation(), &transformed_while);
  EXPECT_THAT(m->entry_computation()->instructions(),
              Contains(op::Negate(op::GetTupleElement(op::CustomCall(), 1))));
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(AnyOf(op::CustomCall(), op::Negate()))));
}

// Runs the pass with hoist_size_inflation_ratio = 1.0 and counts the
// shape_size_function calls, which is how often the size inflation test
// visits an array leaf.
class ShapeSizeCallCountTest : public WhileLoopInvariantCodeMotionTest {
 protected:
  static constexpr int kStateWidth = 64;
  // Loose linear bound on shape_size_function calls: the fixed pass makes
  // 2 * kStateWidth + 2 calls in the test below; a walk of the tuple shape per
  // user makes about kStateWidth * kStateWidth.
  static constexpr int kMaxLinearCalls = 4 * kStateWidth + 16;

  absl::StatusOr<bool> RunWithCountingShapeSizeFunction(HloModule* module) {
    return WhileLoopInvariantCodeMotion(
               /*hoist_constants=*/false, /*hoist_reshapes=*/false,
               /*hoist_other=*/true, /*hoist_size_inflation_ratio=*/1.0,
               /*shape_size_function=*/
               [this](const Shape& shape) {
                 ++shape_size_calls_;
                 return ShapeUtil::ByteSizeOfElements(shape);
               })
        .Run(module);
  }

  int64_t shape_size_calls_ = 0;
};

TEST_F(ShapeSizeCallCountTest, LinearInLoopStateWidth) {
  auto m = CreateNewVerifiedModule();
  Shape array_f32 = ShapeUtil::MakeShape(F32, {8});
  Shape while_shape =
      ShapeUtil::MakeTupleShape(std::vector<Shape>(kStateWidth, array_f32));

  // All state elements are loop invariant except the last, which receives
  // negate(element 0): the single hoisting candidate.
  HloComputation* while_body = [&]() {
    HloComputation::Builder builder(TestName() + ".while_body");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    std::vector<HloInstruction*> root_operands;
    for (int i = 0; i + 1 < kStateWidth; ++i) {
      root_operands.push_back(builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(array_f32, param, i)));
    }
    root_operands.push_back(builder.AddInstruction(HloInstruction::CreateUnary(
        array_f32, HloOpcode::kNegate, root_operands.front())));
    builder.AddInstruction(HloInstruction::CreateTuple(root_operands));
    return m->AddEmbeddedComputation(builder.Build());
  }();

  HloComputation::Builder builder(TestName());
  HloInstruction* init_value = builder.AddInstruction(
      HloInstruction::CreateParameter(0, while_shape, "init_value"));
  builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, m.get()), while_body,
      init_value));
  HloComputation* entry_computation = m->AddEntryComputation(builder.Build());

  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       RunWithCountingShapeSizeFunction(m.get()));
  EXPECT_TRUE(simplified_loop);
  EXPECT_THAT(entry_computation->instructions(), Contains(op::Negate()));

  // Every get-tuple-element of the state has the parameter as its operand and
  // cannot be hoisted. Walking the kStateWidth element parameter shape for
  // each of them would cost kStateWidth * kStateWidth calls.
  EXPECT_GT(shape_size_calls_, 0);
  EXPECT_LE(shape_size_calls_, kMaxLinearCalls);
}

TEST_F(WhileLoopInvariantCodeMotionTest, DoesNotHoistSPMDFullToShardShape) {
  auto m = CreateNewVerifiedModule();
  auto array_s32 = ShapeUtil::MakeShape(S32, {4});
  Shape while_shape =
      ShapeUtil::MakeTupleShape({array_s32, array_s32, array_s32});

  HloComputation* while_body = [&]() {
    HloComputation::Builder builder(TestName() + ".while_body");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloInstruction* gte_0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(array_s32, param, 0));
    HloInstruction* gte_1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(array_s32, param, 1));
    HloInstruction* sharded_gte_1 = builder.AddInstruction(
        HloInstruction::CreateCustomCall(array_s32, {gte_1}, "Sharding"));
    sharded_gte_1->set_sharding(HloSharding::IotaTile({4}));
    HloInstruction* manually_sharded_gte_1 =
        builder.AddInstruction(HloInstruction::CreateCustomCall(
            array_s32, {sharded_gte_1}, "SPMDFullToShardShape"));

    manually_sharded_gte_1->set_sharding(HloSharding::Manual());
    HloInstruction* add_result =
        builder.AddInstruction(HloInstruction::CreateBinary(
            array_s32, HloOpcode::kAdd, gte_0, manually_sharded_gte_1));
    HloInstruction* manually_sharded_add_result = builder.AddInstruction(
        HloInstruction::CreateCustomCall(array_s32, {add_result}, "Sharding"));
    manually_sharded_add_result->set_sharding(HloSharding::Manual());
    HloInstruction* sharded_add_result =
        builder.AddInstruction(HloInstruction::CreateCustomCall(
            array_s32, {manually_sharded_add_result}, "SPMDShardShapeToFull"));
    sharded_add_result->set_sharding(HloSharding::IotaTile({4}));
    builder.AddInstruction(
        HloInstruction::CreateTuple({gte_0, gte_1, sharded_add_result}));

    return m->AddEmbeddedComputation(builder.Build());
  }();

  HloComputation::Builder builder(TestName());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateParameter(0, while_shape, "init_value"));
  builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, m.get()), while_body,
      init_value));
  m->AddEntryComputation(builder.Build());
  LOG(INFO) << "my_test: " << m->ToString();
  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(m.get()));
  EXPECT_FALSE(simplified_loop);
}

TEST_F(WhileLoopInvariantCodeMotionTest, DoesNotHoistShardingCustomCalls) {
  const char* const kHloModule = R"(
    HloModule ModuleWithWhile

    body {
      p_body = (f32[2], f32[2], s32[]) parameter(0)
      gte.0 = f32[2] get-tuple-element(p_body), index=0
      gte.1 = f32[2] get-tuple-element(p_body), index=1
      sharding.0 = f32[2] custom-call(gte.0), custom_call_target="Sharding", sharding={devices=[2]<=[2]}
      sharding.1 = f32[2] custom-call(gte.1), custom_call_target="Sharding", sharding={replicated}
      add.0 = f32[2] add(sharding.0, sharding.1)
      gte.2 = s32[] get-tuple-element(p_body), index=2
      const = s32[] constant(1)
      add.1 = s32[] add(gte.2, const)
      ROOT root = (f32[2], f32[2], s32[]) tuple(gte.0, add.0, add.1)
    }

    condition {
      p_cond = (f32[2], f32[2], s32[]) parameter(0)
      gte = s32[] get-tuple-element(p_cond), index=2
      const = s32[] constant(5)
      ROOT result = pred[] compare(gte, const), direction=LT
    }

    ENTRY entry {
      param.0 = f32[2] parameter(0)
      param.1 = s32[] parameter(1)
      while_init = (f32[2], f32[2], s32[]) tuple(param.0, param.0, param.1)
      ROOT while = (f32[2], f32[2], s32[]) while(while_init), condition=condition, body=body
    })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));

  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(module.get()));
  EXPECT_FALSE(simplified_loop);
}

TEST_F(WhileLoopInvariantCodeMotionTest, RespectFrontendAttrDisablingHoisting) {
  const char* const kHloModule = R"(
  HloModule jit_countdown, entry_computation_layout={(f32[]{:T(128)}, f32[]{:T(128)}, f32[10,10]{1,0:T(8,128)})->(f32[]{:T(128)}, f32[]{:T(128)}, f32[10,10]{1,0:T(8,128)})}

  %region_1.4 (Arg_0.5: f32[], Arg_1.6: f32[]) -> f32[] {
    %Arg_0.5 = f32[] parameter(0)
    %Arg_1.6 = f32[] parameter(1)
    ROOT %add.7 = f32[] add(%Arg_0.5, %Arg_1.6)
  }

  %region_0.8 (arg_tuple.9: (f32[], f32[], f32[10,10])) -> (f32[], f32[], f32[10,10]) {
    %arg_tuple.9 = (f32[], f32[], f32[10,10]{1,0}) parameter(0)
    %get-tuple-element.10 = f32[] get-tuple-element(%arg_tuple.9), index=0
    %constant.14 = f32[] constant(1)
    %negate = f32[] negate(%constant.14)
    %add = f32[] add(%get-tuple-element.10, %negate)
    %get-tuple-element.11 = f32[] get-tuple-element(%arg_tuple.9), index=1
    %get-tuple-element.12 = f32[10,10]{1,0} get-tuple-element(%arg_tuple.9), index=2
    %constant.13 = f32[] constant(0)
    %reduce.16 = f32[] reduce(%get-tuple-element.12, %constant.13), dimensions={0,1}, to_apply=%region_1.4
    %add.17 = f32[] add(%get-tuple-element.11, %reduce.16)
    ROOT %tuple.18 = (f32[], f32[], f32[10,10]{1,0}) tuple(%add, %add.17, %get-tuple-element.12)
  }

  %region_2.19 (arg_tuple.20: (f32[], f32[], f32[10,10])) -> pred[] {
    %arg_tuple.20 = (f32[], f32[], f32[10,10]{1,0}) parameter(0)
    %get-tuple-element.21 = f32[] get-tuple-element(%arg_tuple.20), index=0
    %constant.24 = f32[] constant(0)
    ROOT %compare.25 = pred[] compare(%get-tuple-element.21, %constant.24), direction=GT
  }

  ENTRY %main.40 (Arg_0.1: f32[], Arg_1.2: f32[], Arg_2.3: f32[10,10]) -> (f32[], f32[], f32[10,10]) {
    %Arg_0.1 = f32[] parameter(0)
    %Arg_1.2 = f32[] parameter(1)
    %Arg_2.3 = f32[10,10]{1,0} parameter(2)
    %tuple.0 = (f32[], f32[], f32[10,10]{1,0}) tuple(%Arg_0.1, %Arg_1.2, %Arg_2.3)
    %while.0 = (f32[], f32[], f32[10,10]{1,0}) while(%tuple.0), condition=%region_2.19, body=%region_0.8, frontend_attributes={_xla_disable_loop_instr_hoisting="true"}
    %get-tuple-element.36 = f32[] get-tuple-element(%while.0), index=0
    %get-tuple-element.37 = f32[] get-tuple-element(%while.0), index=1
    ROOT %tuple.39 = (f32[], f32[], f32[10,10]{1,0}) tuple(%get-tuple-element.36, %get-tuple-element.37, %Arg_2.3)
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(module.get()));
  EXPECT_FALSE(simplified_loop);
}

TEST_F(WhileLoopInvariantCodeMotionTest, HoistWithOriginalValue) {
  const char* const hlo_string = R"(
HloModule licm_ov_test

body {
  p_body = (s32[2], s32[2]) parameter(0)
  gte0 = s32[2] get-tuple-element(p_body), index=0
  c = s32[2] constant({1, 1}), origin={{"c.1"}}
  add = s32[2] add(gte0, c), origin={{"add.1"}}
  ROOT tuple = (s32[2], s32[2]) tuple(gte0, add)
}

cond {
  p_cond = (s32[2], s32[2]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  p_entry_0 = s32[2] parameter(0)
  while_init = (s32[2], s32[2]) tuple(p_entry_0, p_entry_0)
  ROOT while0 = (s32[2], s32[2]) while(while_init), condition=cond, body=body, origin={({"while.5" {0}},{"while.5" {1}})}
}
)";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloComputation* body = m->GetComputationWithName("body");
  HloInstruction* c = body->GetInstructionWithName("c");

  ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      WhileLoopInvariantCodeMotion{/*hoist_constants=*/true}.Run(m.get()));
  EXPECT_TRUE(simplified_loop);

  HloInstruction* transformed_while;
  FindOnlyWhileInstruction(m->entry_computation(), &transformed_while);

  HloInstruction* hoisted_c = nullptr;
  HloInstruction* hoisted_add = nullptr;
  for (auto* instr : m->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kConstant &&
        instr->shape() == c->shape()) {
      hoisted_c = instr;
    }
    if (instr->opcode() == HloOpcode::kAdd) {
      hoisted_add = instr;
    }
  }
  ASSERT_NE(hoisted_c, nullptr);
  ASSERT_NE(hoisted_add, nullptr);
  ASSERT_NE(hoisted_c->original_value(), nullptr);
  EXPECT_EQ(hoisted_c->original_value()->ToString(), "{\"while.5#*/c.1\"}");
  ASSERT_NE(hoisted_add->original_value(), nullptr);
  EXPECT_EQ(hoisted_add->original_value()->ToString(), "{\"while.5#*/add.1\"}");
  ASSERT_NE(transformed_while->original_value(), nullptr);
  EXPECT_EQ(transformed_while->original_value()->ToString(),
            "({\"while.5\" {0}}, {\"while.5\" {1}}, {\"while.5#*/c.1\"}, "
            "{\"while.5#*/add.1\"})");
}

TEST_F(WhileLoopInvariantCodeMotionTest,
       HoistWithOriginalValueAndCallHierarchy) {
  const char* const hlo_string = R"(
HloModule licm_ov_call_hierarchy_test

body {
  p_body = (s32[2], s32[2]) parameter(0)
  gte0 = s32[2] get-tuple-element(p_body), index=0
  c = s32[2] constant({1, 1}), origin={{"c.1"}}
  add = s32[2] add(gte0, c), origin={{"add.1"},["inner#$"]}
  ROOT tuple = (s32[2], s32[2]) tuple(gte0, add)
}

cond {
  p_cond = (s32[2], s32[2]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  p_entry_0 = s32[2] parameter(0)
  while_init = (s32[2], s32[2]) tuple(p_entry_0, p_entry_0)
  ROOT while0 = (s32[2], s32[2]) while(while_init), condition=cond, body=body, origin={({"while.5" {0}}, {"while.5" {1}}),["w1#$"]}
}
)";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloComputation* body = m->GetComputationWithName("body");
  HloInstruction* c = body->GetInstructionWithName("c");

  ASSERT_OK_AND_ASSIGN(
      bool simplified_loop,
      WhileLoopInvariantCodeMotion{/*hoist_constants=*/true}.Run(m.get()));
  EXPECT_TRUE(simplified_loop);

  HloInstruction* transformed_while;
  FindOnlyWhileInstruction(m->entry_computation(), &transformed_while);

  HloInstruction* hoisted_c = nullptr;
  HloInstruction* hoisted_add = nullptr;
  for (auto* instr : m->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kConstant &&
        instr->shape() == c->shape()) {
      hoisted_c = instr;
    }
    if (instr->opcode() == HloOpcode::kAdd) {
      hoisted_add = instr;
    }
  }
  ASSERT_NE(hoisted_c, nullptr);
  ASSERT_NE(hoisted_add, nullptr);
  ASSERT_NE(hoisted_c->original_value(), nullptr);
  EXPECT_EQ(hoisted_c->original_value()->ToString(),
            "{\"w1#*/c.1\"},[\"w1#*\"]");
  ASSERT_NE(hoisted_add->original_value(), nullptr);
  EXPECT_EQ(hoisted_add->original_value()->ToString(),
            "{\"w1#*/add.1\"},[\"w1#*/inner#$\"]");
  ASSERT_NE(transformed_while->original_value(), nullptr);
  EXPECT_EQ(transformed_while->original_value()->ToString(),
            "({\"while.5\" {0}}, {\"while.5\" {1}}, {\"w1#*/c.1\"}, "
            "{\"w1#*/add.1\"}),[\"w1#$\"]");
}

TEST_F(WhileLoopInvariantCodeMotionTest, HoistsGetRngSeed) {
  const char* const hlo_string = R"(
HloModule licm_get_rng_seed_test

body {
  p_body = (u64[1], u64[1]) parameter(0)
  gte0 = u64[1] get-tuple-element(p_body), index=0
  rng_seed = u64[] custom-call(), custom_call_target="GetRngSeed"
  bcast = u64[1] broadcast(rng_seed), dimensions={}
  add = u64[1] add(gte0, bcast)
  ROOT tuple = (u64[1], u64[1]) tuple(gte0, add)
}

cond {
  p_cond = (u64[1], u64[1]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  p_entry_0 = (u64[1], u64[1]) parameter(0)
  ROOT while0 = (u64[1], u64[1]) while(p_entry_0), condition=cond, body=body
}
)";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(m.get()));
  EXPECT_TRUE(simplified_loop);

  EXPECT_THAT(m->entry_computation()->instructions(),
              Contains(op::CustomCall()));
}

TEST_F(WhileLoopInvariantCodeMotionTest, HoistInvariantCall) {
  const char* const hlo_string = R"(
HloModule HoistInvariantCall

foo (p0: s32[2], p1: s32[2]) -> s32[2] {
  p0 = s32[2]{0} parameter(0)
  p1 = s32[2]{0} parameter(1)
  ROOT add = s32[2]{0} add(p0, p1)
}

body (param: (s32[2], s32[2], s32[2])) -> (s32[2], s32[2], s32[2]) {
  param = (s32[2], s32[2], s32[2]) parameter(0)
  gte_0 = s32[2]{0} get-tuple-element(param), index=0
  gte_1 = s32[2]{0} get-tuple-element(param), index=1
  call_result = s32[2]{0} call(gte_0, gte_1), to_apply=foo
  ROOT tuple = (s32[2], s32[2], s32[2]) tuple(gte_0, gte_1, call_result)
}

cond (param: (s32[2], s32[2], s32[2])) -> pred[] {
  param = (s32[2], s32[2], s32[2]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  init_value = (s32[2], s32[2], s32[2]) parameter(0)
  // CHECK: ENTRY %entry
  // CHECK: %[[call:.+]] = s32[2]{0} call({{.+}}, {{.+}}), to_apply=%foo
  // CHECK: %[[while_init:.+]] = (s32[2]{0}, s32[2]{0}, s32[2]{0}, s32[2]{0}) tuple({{.+}}, {{.+}}, {{.+}}, %[[call]])
  // CHECK: %[[while:.+]] = (s32[2]{0}, s32[2]{0}, s32[2]{0}, s32[2]{0}) while(%[[while_init]]){{.*}}
  ROOT while = (s32[2], s32[2], s32[2]) while(init_value), condition=cond, body=body
}
)";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(m.get()));
  EXPECT_TRUE(simplified_loop);

  EXPECT_THAT(RunFileCheck(m->ToString(), hlo_string),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(WhileLoopInvariantCodeMotionTest, DontHoistCallWithSideEffects) {
  const char* const hlo_string = R"(
HloModule DontHoistCallWithSideEffects

foo (p0: s32[2], p1: s32[2], tok: token[]) -> (s32[2], token[]) {
  p0 = s32[2]{0} parameter(0)
  p1 = s32[2]{0} parameter(1)
  tok = token[] parameter(2)
  outfeed = token[] outfeed(p0, tok)
  add = s32[2]{0} add(p0, p1)
  ROOT tuple = (s32[2]{0}, token[]) tuple(add, outfeed)
}

body (param: (s32[2], s32[2], s32[2], token[])) -> (s32[2], s32[2], s32[2], token[]) {
  param = (s32[2], s32[2], s32[2], token[]) parameter(0)
  gte_0 = s32[2]{0} get-tuple-element(param), index=0
  gte_1 = s32[2]{0} get-tuple-element(param), index=1
  token_val = token[] get-tuple-element(param), index=3
  // CHECK: %body (
  // CHECK: %[[call:.+]] = (s32[2]{0}, token[]) call({{.+}}, {{.+}}, {{.+}}), to_apply=%foo
  call_result = (s32[2]{0}, token[]) call(gte_0, gte_1, token_val), to_apply=foo
  call_res_0 = s32[2]{0} get-tuple-element(call_result), index=0
  call_res_token = token[] get-tuple-element(call_result), index=1
  ROOT tuple = (s32[2]{0}, s32[2]{0}, s32[2]{0}, token[]) tuple(gte_0, gte_1, call_res_0, call_res_token)
}

cond (param: (s32[2], s32[2], s32[2], token[])) -> pred[] {
  param = (s32[2], s32[2], s32[2], token[]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  init_value = (s32[2], s32[2], s32[2], token[]) parameter(0)
  ROOT while = (s32[2], s32[2], s32[2], token[]) while(init_value), condition=cond, body=body
}
)";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(m.get()));
  EXPECT_FALSE(simplified_loop);

  EXPECT_THAT(RunFileCheck(m->ToString(), hlo_string),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(WhileLoopInvariantCodeMotionTest, SharedCalleeOneHoistedOneNot) {
  const char* const hlo_string = R"(
HloModule SharedCalleeOneHoistedOneNot

foo (p0: s32[2]) -> s32[2] {
  p0 = s32[2]{0} parameter(0)
  ROOT negate = s32[2]{0} negate(p0)
}

body_hoist (param: (s32[2], s32[2])) -> (s32[2], s32[2]) {
  param = (s32[2], s32[2]) parameter(0)
  gte_0 = s32[2]{0} get-tuple-element(param), index=0
  call_res = s32[2]{0} call(gte_0), to_apply=foo
  ROOT tuple = (s32[2], s32[2]) tuple(gte_0, call_res)
}

body_no_hoist (param: (s32[2], s32[2])) -> (s32[2], s32[2]) {
  param = (s32[2], s32[2]) parameter(0)
  gte_0 = s32[2]{0} get-tuple-element(param), index=0
  gte_1 = s32[2]{0} get-tuple-element(param), index=1
  // CHECK: %body_no_hoist (
  // CHECK: call({{.*}}), to_apply=%foo
  // CHECK: %{{.*}}body_hoist (
  // CHECK-NOT: call(
  call_res = s32[2]{0} call(gte_1), to_apply=foo
  one = s32[2]{0} constant({1, 1})
  add = s32[2]{0} add(gte_1, one)
  ROOT tuple = (s32[2], s32[2]) tuple(call_res, add)
}

cond (param: (s32[2], s32[2])) -> pred[] {
  param = (s32[2], s32[2]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  init_value = (s32[2], s32[2]) parameter(0)
  // CHECK: ENTRY %entry
  // CHECK: call({{.*}}), to_apply=%foo
  while_hoist = (s32[2], s32[2]) while(init_value), condition=cond, body=body_hoist
  while_no_hoist = (s32[2], s32[2]) while(while_hoist), condition=cond, body=body_no_hoist
  ROOT result = (s32[2], s32[2]) copy(while_no_hoist)
}
)";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                       WhileLoopInvariantCodeMotion{}.Run(m.get()));
  EXPECT_TRUE(simplified_loop);

  EXPECT_THAT(RunFileCheck(m->ToString(), hlo_string),
              absl_testing::IsOkAndHolds(true));
}

WhileLoopInvariantCodeMotion MakeReshapeLicm(float ratio = 1.0f) {
  return WhileLoopInvariantCodeMotion(
      /*hoist_constants=*/false, /*hoist_reshapes=*/true, /*hoist_other=*/false,
      /*hoist_size_inflation_ratio=*/ratio, ShapeUtil::ArraySize);
}

TEST_F(WhileLoopInvariantCodeMotionTest,
       HoistsLayoutCopyAndBitcastChainWithoutStandaloneBitcast) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m, entry_computation_layout={
  (f32[16,8,128]{2,1,0:T(8,128)}, f32[16,1024]{1,0:T(8,128)},
   f32[128,128]{1,0:T(8,128)})->f32[16,1024]{1,0:T(8,128)}}
body {
  p = (f32[16,8,128]{2,1,0:T(8,128)}, f32[16,1024]{1,0:T(8,128)},
       f32[128,128]{1,0:T(8,128)}) parameter(0)
  inv = get-tuple-element(p), index=0
  a0 = get-tuple-element(p), index=1
  a1 = get-tuple-element(p), index=2
  bc_in = f32[2,8,8,128]{3,2,1,0:T(8,128)} bitcast(inv)
  cp = f32[2,8,8,128]{3,1,2,0:T(8,128)} copy(bc_in), origin={{"cp.1"}},
    sharding={devices=[2,1,1,1]<=[2]}, frontend_attributes={_k="v"},
    metadata={op_name="kv"}
  bc_out = f32[16,1024]{1,0:T(8,128)} bitcast(cp)
  bc_alone = f32[128,128]{1,0:T(8,128)} bitcast(inv)
  n0 = add(a0, bc_out)
  n1 = add(a1, bc_alone)
  ROOT r = tuple(inv, n0, n1)
}
cond {
  p = (f32[16,8,128]{2,1,0:T(8,128)}, f32[16,1024]{1,0:T(8,128)},
       f32[128,128]{1,0:T(8,128)}) parameter(0)
  ROOT c = pred[] constant(true)
}
ENTRY e {
  p0 = f32[16,8,128]{2,1,0:T(8,128)} parameter(0)
  p1 = f32[16,1024]{1,0:T(8,128)} parameter(1)
  p2 = f32[128,128]{1,0:T(8,128)} parameter(2)
  init = tuple(p0, p1, p2)
  w = while(init), condition=cond, body=body,
    origin={({"w.1" {0}}, {"w.1" {1}}, {"w.1" {2}})}
  ROOT out = get-tuple-element(w), index=1
})hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(WhileLoopInvariantCodeMotion{}.Run(m.get()),
              absl_testing::IsOkAndHolds(false));
  EXPECT_THAT(MakeReshapeLicm().Run(m.get()), absl_testing::IsOkAndHolds(true));

  HloInstruction* loop = nullptr;
  FindOnlyWhileInstruction(m->entry_computation(), &loop);
  EXPECT_EQ(loop->shape().tuple_shapes().size(), 4);
  EXPECT_EQ(absl::c_count_if(m->entry_computation()->instructions(),
                             HloPredicateIsOp<HloOpcode::kBitcast>),
            1);
  EXPECT_THAT(
      loop->while_body()->instructions(),
      AllOf(Contains(op::Bitcast(op::GetTupleElement(op::Parameter(0), 3))),
            Contains(op::Bitcast(op::GetTupleElement(op::Parameter(0), 0)))));
  const HloInstruction* cp = FindInstruction(m.get(), HloOpcode::kCopy);
  ASSERT_NE(cp, nullptr);
  EXPECT_THAT(cp, op::Copy(op::Bitcast(op::Parameter(0))));
  EXPECT_TRUE(LayoutUtil::Equal(
      cp->shape().layout(),
      LayoutUtil::MakeLayout({3, 1, 2, 0}, {Tile({8, 128})})));
  EXPECT_EQ(cp->sharding(), HloSharding::IotaTile({2, 1, 1, 1}));
  EXPECT_EQ(cp->metadata().op_name(), "kv");
  EXPECT_EQ(cp->get_frontend_attribute("_k"), "v");
  ASSERT_NE(cp->original_value(), nullptr);
  EXPECT_EQ(cp->original_value()->ToString(), "{\"w.1#*/cp.1\"}");

  AliasInfo alias_info;
  EXPECT_OK(CopyInsertion(&alias_info).Run(m.get()).status());
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/true,
                        /*allow_mixed_precision=*/false)
                .Run(m.get())
                .status());
  FindOnlyWhileInstruction(m->entry_computation(), &loop);
  EXPECT_THAT(loop->while_body()->instructions(), Each(Not(op::Copy())));
}

TEST_F(WhileLoopInvariantCodeMotionTest,
       HoistsBitcastReshapeAndFiltersLayoutCopies) {
  auto parse = [this](absl::string_view s, absl::string_view d,
                      absl::string_view body_op = "") {
    const std::string def =
        body_op.empty() ? absl::Substitute("v = f32[8,128]$0 copy(inv)", d)
                        : std::string(body_op);
    return ParseAndReturnVerifiedModule(absl::Substitute(
        R"hlo(
HloModule m
body {
  p = (f32[8,128]$0, f32[8,128]$1) parameter(0)
  inv = get-tuple-element(p), index=0
  acc = get-tuple-element(p), index=1
  $2
  neg = negate(v)
  next = add(acc, neg)
  ROOT r = tuple(inv, next)
}
cond {
  p = (f32[8,128]$0, f32[8,128]$1) parameter(0)
  ROOT c = pred[] constant(true)
}
ENTRY e {
  p = (f32[8,128]$0, f32[8,128]$1) parameter(0)
  ROOT w = while(p), condition=cond, body=body
})hlo",
        s, d, def));
  };
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> cp, parse("{1,0}", "{0,1}"));
  EXPECT_THAT(MakeReshapeLicm().Run(cp.get()),
              absl_testing::IsOkAndHolds(true));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> rs,
      parse("{1,0}", "{1,0}",
            "b = f32[1024]{0} bitcast(inv)\n  v = f32[8,128]{1,0} reshape(b)"));
  EXPECT_THAT(MakeReshapeLicm().Run(rs.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(rs->entry_computation()->instructions(),
              Contains(op::Reshape(op::Bitcast(op::GetTupleElement()))));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> bc,
      parse("{1,0}", "{1,0}", "v = f32[8,128]{1,0} bitcast(inv)"));
  EXPECT_THAT(MakeReshapeLicm().Run(bc.get()),
              absl_testing::IsOkAndHolds(false));

  const struct {
    absl::string_view s, d, body_op;
  } kRejected[] = {
      {"{1,0:T(8,128)}", "{1,0:T(8,128)}", ""},
      {"{1,0}", "{0,1:S(1)}", ""},
      {"{1,0}", "{0,1:S(3)}", ""},
      {"{1,0}", "{1,0:S(5)}", ""},
      {"{1,0}", "{0,1:S(6)}", ""},
      {"{1,0:S(1)}", "{0,1}", ""},
      {"{1,0:S(1)}", "{0,1:S(1)}", ""},
      {"{1,0:S(1)}", "{1,0:S(1)}", "v = f32[8,128]{1,0:S(1)} bitcast(inv)"},
      {"{1,0:S(1)}", "{1,0:S(1)}", "v = f32[8,128]{1,0:S(1)} reshape(inv)"}};
  for (const auto& [s, d, body_op] : kRejected) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m, parse(s, d, body_op));
    EXPECT_THAT(MakeReshapeLicm().Run(m.get()),
                absl_testing::IsOkAndHolds(false));
    EXPECT_THAT(WhileLoopInvariantCodeMotion(/*hoist_constants=*/false,
                                             /*hoist_reshapes=*/true,
                                             /*hoist_other=*/true)
                    .Run(m.get()),
                absl_testing::IsOkAndHolds(false));
  }

  // {1,0:T(8,128)} is 4 KiB while {0,1:T(8,128)} pads dim 0 to 128 (64 KiB).
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> inf,
                       parse("{1,0:T(8,128)}", "{0,1:T(8,128)}"));
  EXPECT_THAT(MakeReshapeLicm(1.0f).Run(inf.get()),
              absl_testing::IsOkAndHolds(false));
  EXPECT_THAT(WhileLoopInvariantCodeMotion(/*hoist_constants=*/true,
                                           /*hoist_reshapes=*/false,
                                           /*hoist_other=*/false)
                  .Run(inf.get()),
              absl_testing::IsOkAndHolds(false));
  EXPECT_THAT(MakeReshapeLicm(16.0f).Run(inf.get()),
              absl_testing::IsOkAndHolds(true));
}

}  // namespace
}  // namespace xla
