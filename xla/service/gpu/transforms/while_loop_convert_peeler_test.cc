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

#include "xla/service/gpu/transforms/while_loop_convert_peeler.h"
#include "gtest/gtest.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/platform_util.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tests/test_utils.h"
#include "xla/service/pattern_matcher.h"

namespace xla::gpu {

namespace {

namespace m = ::xla::match;

class WhileLoopConvertPeelerTest : public HloRunnerAgnosticTestBase {
 public:
  WhileLoopConvertPeelerTest()
      : HloRunnerAgnosticTestBase(std::make_unique<HloRunner>(
            PlatformUtil::GetDefaultPlatform().value())) {}
};

TEST_F(WhileLoopConvertPeelerTest, DynamicSliceRootWhileOneBuffer) {
  const std::string hlo = R"(
  HloModule test
  body {
    param = (s32[], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    data = f32[8,8] get-tuple-element(param), index=1
    bf16_data = bf16[1,8] get-tuple-element(param), index=2
    c1 = s32[] constant(1)
    iter_plus_one = s32[] add(iter, c1)
    c0 = s32[] constant(0)
    ds = f32[1,8] dynamic-slice(data, iter, c0), dynamic_slice_sizes={1,8}
    convert = bf16[1,8] convert(ds)
    add = bf16[1,8] add(convert, bf16_data)
    ROOT tuple = (s32[], f32[8,8], bf16[1,8]) tuple(iter_plus_one, data, add)
  }
  condition {
    param = (s32[], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c8 = s32[] constant(8)
    ROOT lt = pred[] compare(iter, c8), direction=LT
  }
  ENTRY main {
    %c0 = s32[] constant(0)
    %p0 = f32[8,8] parameter(0)
    %p1 = bf16[1,8] parameter(1)
    tuple = (s32[], f32[8,8], bf16[1,8]) tuple(c0, %p0, %p1)
    ROOT while = (s32[], f32[8,8], bf16[1,8]) while(tuple), body=body, condition=condition
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConvertPeeler().Run(module.get(), {}));
  EXPECT_TRUE(changed);
  HloInstruction* while_op = hlo_query::FindInstruction(
      module->entry_computation(), HloOpcode::kWhile);

  // If the convert is peeled, then the while operation type should change from
  // f32[] to bf16[] at index 1.
  EXPECT_EQ(while_op->shape().tuple_shapes(1),
            ShapeUtil::MakeShape(BF16, {8, 8}));

  // We also expect that the while operation is no longer the root, and that the
  // root operation is a tuple of while results and the original buffer, without
  // the convert operation.
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  ASSERT_EQ(root->operand_count(), 3);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(2)->opcode(), HloOpcode::kGetTupleElement);

  // The while operation should ingest the converted buffer at index 1.
  const HloInstruction* possible_convert = while_op->while_init()->operand(1);
  EXPECT_EQ(possible_convert->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(possible_convert->operand(0),
            module->entry_computation()->parameter_instruction(0));

  // The changes in while body and condition should be verified by the verifier.
  TF_CHECK_OK(VerifyHloModule(module.get(), true, false));

  EXPECT_TRUE(
      RunAndCompareTwoModules(module->ToString(), hlo, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(WhileLoopConvertPeelerTest, DynamicSliceRootWhileTwoBuffers) {
  const std::string hlo = R"(
  HloModule test
  body {
    param = (s32[], f32[8,8], bf16[1,8], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    data = f32[8,8] get-tuple-element(param), index=1
    bf16_data = bf16[1,8] get-tuple-element(param), index=2
    data2 = f32[8,8] get-tuple-element(param), index=3
    bf16_data2 = bf16[1,8] get-tuple-element(param), index=4
    c1 = s32[] constant(1)
    iter_plus_one = s32[] add(iter, c1)
    c0 = s32[] constant(0)
    ds = f32[1,8] dynamic-slice(data, iter, c0), dynamic_slice_sizes={1,8}
    ds2 = f32[1,8] dynamic-slice(data2, iter, c0), dynamic_slice_sizes={1,8}
    convert = bf16[1,8] convert(ds)
    convert2 = bf16[1,8] convert(ds2)
    add = bf16[1,8] add(convert, bf16_data)
    add2 = bf16[1,8] add(convert2, bf16_data2)
    ROOT tuple = (s32[], f32[8,8], bf16[1,8], f32[8,8], bf16[1,8]) tuple(iter_plus_one, data, add, data2, add2)
  }
  condition {
    param = (s32[], f32[8,8], bf16[1,8], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c8 = s32[] constant(8)
    ROOT lt = pred[] compare(iter, c8), direction=LT
  }
  ENTRY main {
    %c0 = s32[] constant(0)
    %p0 = f32[8,8] parameter(0)
    %p1 = bf16[1,8] parameter(1)
    %p2 = f32[8,8] parameter(2)
    %p3 = bf16[1,8] parameter(3)
    tuple = (s32[], f32[8,8], bf16[1,8], f32[8,8], bf16[1,8]) tuple(c0, %p0, %p1, %p2, %p3)
    ROOT while = (s32[], f32[8,8], bf16[1,8], f32[8,8], bf16[1,8]) while(tuple), body=body, condition=condition
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConvertPeeler().Run(module.get(), {}));
  EXPECT_TRUE(changed);

  // Verify the general structure:
  //  * root is a tuple of gte from while output, and the original buffers.
  //  * while operation is accepting the original buffers and the converted
  //  buffers.
  //  * the converted buffers are at index 1 and 3 of the while operation.
  auto while_op_matcher = m::While(
      m::Tuple(m::Constant(), m::Convert(m::Parameter(0)), m::Parameter(1),
               m::Convert(m::Parameter(2)), m::Parameter(3)));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::GetTupleElement(while_op_matcher), m::Parameter(0),
                          m::GetTupleElement(while_op_matcher), m::Parameter(2),
                          m::GetTupleElement(while_op_matcher))));

  // Verify the type of the converted buffers.
  HloInstruction* while_op = hlo_query::FindInstruction(
      module->entry_computation(), HloOpcode::kWhile);
  ASSERT_NE(while_op, nullptr);
  EXPECT_EQ(while_op->shape().tuple_shapes(1),
            ShapeUtil::MakeShape(BF16, {8, 8}));
  EXPECT_EQ(while_op->shape().tuple_shapes(3),
            ShapeUtil::MakeShape(BF16, {8, 8}));

  // The rest of the verification is done by the verifier.
  TF_CHECK_OK(VerifyHloModule(module.get(), true, false));
  EXPECT_TRUE(
      RunAndCompareTwoModules(module->ToString(), hlo, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(WhileLoopConvertPeelerTest, DynamicSliceNonRootWhileOneBuffer) {
  const char* hlo = R"(  HloModule test
  body {
    param = (s32[], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    data = f32[8,8] get-tuple-element(param), index=1
    bf16_data = bf16[1,8] get-tuple-element(param), index=2
    c1 = s32[] constant(1)
    iter_plus_one = s32[] add(iter, c1)
    c0 = s32[] constant(0)
    ds = f32[1,8] dynamic-slice(data, iter, c0), dynamic_slice_sizes={1,8}
    convert = bf16[1,8] convert(ds)
    add = bf16[1,8] add(convert, bf16_data)
    ROOT tuple = (s32[], f32[8,8], bf16[1,8]) tuple(iter_plus_one, data, add)
  }
  condition {
    param = (s32[], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c8 = s32[] constant(8)
    ROOT lt = pred[] compare(iter, c8), direction=LT
  }
  ENTRY main {
    %c0 = s32[] constant(0)
    %p0 = f32[8,8] parameter(0)
    %p1 = bf16[1,8] parameter(1)
    tuple = (s32[], f32[8,8], bf16[1,8]) tuple(c0, %p0, %p1)
    while = (s32[], f32[8,8], bf16[1,8]) while(tuple), body=body, condition=condition
    ROOT res = bf16[1,8] get-tuple-element(while), index=2
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConvertPeeler().Run(module.get(), {}));
  EXPECT_TRUE(changed);

  auto while_op_matcher = m::While(
      m::Tuple(m::Constant(), m::Convert(m::Parameter(0)), m::Parameter(1)));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(while_op_matcher)));

  // Verify the type of the converted buffer.
  HloInstruction* while_op = hlo_query::FindInstruction(
      module->entry_computation(), HloOpcode::kWhile);
  ASSERT_NE(while_op, nullptr);
  EXPECT_EQ(while_op->shape().tuple_shapes(1),
            ShapeUtil::MakeShape(BF16, {8, 8}));

  // The rest of the verification is done by the verifier.
  TF_CHECK_OK(VerifyHloModule(module.get(), true, false));
  EXPECT_TRUE(
      RunAndCompareTwoModules(module->ToString(), hlo, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(WhileLoopConvertPeelerTest, DynamicSliceNonRootWhileTwoBuffers) {
  const char* hlo = R"(HloModule test
  body {
    param = (s32[], f32[8,8], bf16[1,8], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    data = f32[8,8] get-tuple-element(param), index=1
    bf16_data = bf16[1,8] get-tuple-element(param), index=2
    data2 = f32[8,8] get-tuple-element(param), index=3
    bf16_data2 = bf16[1,8] get-tuple-element(param), index=4
    c1 = s32[] constant(1)
    iter_plus_one = s32[] add(iter, c1)
    c0 = s32[] constant(0)
    ds = f32[1,8] dynamic-slice(data, iter, c0), dynamic_slice_sizes={1,8}
    ds2 = f32[1,8] dynamic-slice(data2, iter, c0), dynamic_slice_sizes={1,8}
    convert = bf16[1,8] convert(ds)
    convert2 = bf16[1,8] convert(ds2)
    add = bf16[1,8] add(convert, bf16_data)
    add2 = bf16[1,8] add(convert2, bf16_data2)
    ROOT tuple = (s32[], f32[8,8], bf16[1,8], f32[8,8], bf16[1,8]) tuple(iter_plus_one, data, add, data2, add2)
  }
  condition {
    param = (s32[], f32[8,8], bf16[1,8], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c8 = s32[] constant(8)
    ROOT lt = pred[] compare(iter, c8), direction=LT
  }
  ENTRY main {
    %c0 = s32[] constant(0)
    %p0 = f32[8,8] parameter(0)
    %p1 = bf16[1,8] parameter(1)
    %p2 = f32[8,8] parameter(2)
    %p3 = bf16[1,8] parameter(3)
    tuple_init = (s32[], f32[8,8], bf16[1,8], f32[8,8], bf16[1,8]) tuple(c0, %p0, %p1, %p2, %p3)
    while = (s32[], f32[8,8], bf16[1,8], f32[8,8], bf16[1,8]) while(tuple_init), body=body, condition=condition
    gte0 = s32[] get-tuple-element(while), index=0
    gte1 = f32[8,8] get-tuple-element(while), index=1
    gte2 = bf16[1,8] get-tuple-element(while), index=2
    gte3 = f32[8,8] get-tuple-element(while), index=3
    gte4 = bf16[1,8] get-tuple-element(while), index=4
    add1 = f32[8,8] add(gte1, gte3)
    add2 = bf16[1,8] add(gte2, gte4)
    ROOT tuple_res = (f32[8,8], bf16[1,8]) tuple(add1, add2)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConvertPeeler().Run(module.get(), {}));
  EXPECT_TRUE(changed);

  auto while_op_matcher = m::While(
      m::Tuple(m::Constant(), m::Convert(m::Parameter(0)), m::Parameter(1),
               m::Convert(m::Parameter(2)), m::Parameter(3)));
  HloInstruction* while_op = hlo_query::FindInstruction(
      module->entry_computation(), HloOpcode::kWhile);
  ASSERT_NE(while_op, nullptr);
  EXPECT_EQ(while_op->shape().tuple_shapes(1),
            ShapeUtil::MakeShape(BF16, {8, 8}));
  EXPECT_EQ(while_op->shape().tuple_shapes(3),
            ShapeUtil::MakeShape(BF16, {8, 8}));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Add(m::Parameter(0), m::Parameter(2)),
                          m::Add(m::GetTupleElement(while_op_matcher),
                                 m::GetTupleElement(while_op_matcher)))));

  // The rest of the verification is done by the verifier.
  TF_CHECK_OK(VerifyHloModule(module.get(), true, false));
  EXPECT_TRUE(
      RunAndCompareTwoModules(module->ToString(), hlo, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace

}  // namespace xla::gpu