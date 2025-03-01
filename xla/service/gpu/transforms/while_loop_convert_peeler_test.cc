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

TEST_F(WhileLoopConvertPeelerTest, NoDynamicSliceOperationShouldNotChange) {
  const char* hlo = R"(
  HloModule test
  body {
    param = (s32[], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    data = f32[8,8] get-tuple-element(param), index=1
    bf16_data = bf16[1,8] get-tuple-element(param), index=2
    data_new = f32[8,8] add(data, data)
    bf16_data_new = bf16[1,8] add(bf16_data, bf16_data)
    c1 = s32[] constant(1)
    iter_plus_one = s32[] add(iter, c1)
    ROOT tuple = (s32[], f32[8,8], bf16[1,8]) tuple(iter_plus_one, data_new, bf16_data_new)
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
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConvertPeeler().Run(module.get(), {}));
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopConvertPeelerTest,
       NonDeterminableInductionVariableIdxShouldNotChange) {
  const char* hlo = R"(
  HloModule test
  body {
    param = (s32[], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    data = f32[8,8] get-tuple-element(param), index=1
    bf16_data = bf16[1,8] get-tuple-element(param), index=2
    c0 = s32[] constant(0)
    ds = f32[1,8] dynamic-slice(data, iter, c0), dynamic_slice_sizes={1,8}
    convert = bf16[1,8] convert(ds)
    add = bf16[1,8] add(convert, bf16_data)
    c1 = s32[] constant(1)
    iter_plus_one = s32[] add(iter, c1)
    ROOT tuple = (s32[], f32[8,8], bf16[1,8]) tuple(iter_plus_one, data, add)
  }
  condition {
    param = (s32[], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c8 = s32[] constant(8)
    ROOT lt = pred[] compare(iter, c8), direction=LT
  }
  ENTRY main {
    // The induction variable cannot be determined because initialization is not a constant.
    %iter_init = s32[] parameter(0)
    %data_init = f32[8,8] parameter(1)
    %bf16_data_init = bf16[1,8] parameter(2)
    tuple_init = (s32[], f32[8,8], bf16[1,8]) tuple(%iter_init, %data_init, %bf16_data_init)
    ROOT while = (s32[], f32[8,8], bf16[1,8]) while(tuple_init), body=body, condition=condition
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConvertPeeler().Run(module.get(), {}));
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopConvertPeelerTest, BufferIsNotGetTupleElementShouldNotChange) {
  const char* hlo = R"(
  HloModule test
  body {
    param = (s32[], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    data = f32[8,8] get-tuple-element(param), index=1
    bf16_data = bf16[1,8] get-tuple-element(param), index=2
    buffer = f32[8,8] add(data, data)
    c0 = s32[] constant(0)
    ds = f32[1,8] dynamic-slice(buffer, iter, c0), dynamic_slice_sizes={1,8}
    convert = bf16[1,8] convert(ds)
    add = bf16[1,8] add(convert, bf16_data)
    c1 = s32[] constant(1)
    iter_plus_one = s32[] add(iter, c1)
    ROOT tuple = (s32[], f32[8,8], bf16[1,8]) tuple(iter_plus_one, data, add)
  }
  condition {
    param = (s32[], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c8 = s32[] constant(8)
    ROOT lt = pred[] compare(iter, c8), direction=LT
  }
  ENTRY main {
    %iter_init = s32[] constant(0)
    %data_init = f32[8,8] parameter(0)
    %bf16_data_init = bf16[1,8] parameter(1)
    tuple_init = (s32[], f32[8,8], bf16[1,8]) tuple(%iter_init, %data_init, %bf16_data_init)
    ROOT while = (s32[], f32[8,8], bf16[1,8]) while(tuple_init), body=body, condition=condition
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConvertPeeler().Run(module.get(), {}));
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopConvertPeelerTest,
       BufferIsNotGetTupleElementOnParameterShouldNotChange) {
  const char* hlo = R"(
  HloModule test
  body {
    param = (s32[], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    data = f32[8,8] get-tuple-element(param), index=1
    bf16_data = bf16[1,8] get-tuple-element(param), index=2
    custom_call = (f32[8,8], s32[8]) custom-call(data, iter), custom_call_target="my_custom_call"
    buffer = f32[8,8] get-tuple-element(custom_call), index=0
    c0 = s32[] constant(0)
    ds = f32[1,8] dynamic-slice(buffer, iter, c0), dynamic_slice_sizes={1,8}
    convert = bf16[1,8] convert(ds)
    add = bf16[1,8] add(convert, bf16_data)
    c1 = s32[] constant(1)
    iter_plus_one = s32[] add(iter, c1)
    ROOT tuple = (s32[], f32[8,8], bf16[1,8]) tuple(iter_plus_one, data, add)
  }
  condition {
    param = (s32[], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c8 = s32[] constant(8)
    ROOT lt = pred[] compare(iter, c8), direction=LT
  }
  ENTRY main {
    %iter_init = s32[] constant(0)
    %data_init = f32[8,8] parameter(0)
    %bf16_data_init = bf16[1,8] parameter(1)
    tuple_init = (s32[], f32[8,8], bf16[1,8]) tuple(%iter_init, %data_init, %bf16_data_init)
    ROOT while = (s32[], f32[8,8], bf16[1,8]) while(tuple_init), body=body, condition=condition
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConvertPeeler().Run(module.get(), {}));
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopConvertPeelerTest,
       DynamicSliceMultipleVariableIndicesShouldNotChange) {
  const char* hlo = R"(
  HloModule test
  body {
    param = (s32[], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    data = f32[8,8] get-tuple-element(param), index=1
    bf16_data = bf16[1,8] get-tuple-element(param), index=2
    c0 = s32[] constant(0)
    ds = f32[1,8] dynamic-slice(data, iter, iter), dynamic_slice_sizes={1,8}
    convert = bf16[1,8] convert(ds)
    add = bf16[1,8] add(convert, bf16_data)
    c1 = s32[] constant(1)
    iter_plus_one = s32[] add(iter, c1)
    ROOT tuple = (s32[], f32[8,8], bf16[1,8]) tuple(iter_plus_one, data, add)
  }
  condition {
    param = (s32[], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c8 = s32[] constant(8)
    ROOT lt = pred[] compare(iter, c8), direction=LT
  }
  ENTRY main {
    %iter_init = s32[] constant(0)
    %data_init = f32[8,8] parameter(0)
    %bf16_data_init = bf16[1,8] parameter(1)
    tuple_init = (s32[], f32[8,8], bf16[1,8]) tuple(%iter_init, %data_init, %bf16_data_init)
    ROOT while = (s32[], f32[8,8], bf16[1,8]) while(tuple_init), body=body, condition=condition
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConvertPeeler().Run(module.get(), {}));
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopConvertPeelerTest,
       DynamicSliceNoVariableIndicesShouldNotChange) {
  // The aim of this test is to ensure that if we have no variable indices, we
  // should not change the instruction. We also have the condition "For all
  // indices i < k, we should have that dimension[i]=slice_size[i]=1 and
  // offset[i]=0." In this case, all indices should satisfy this condition. This
  // means that the dimensions of buffer should be [1,1,...,1], which should be
  // the same as dimensions of dynamic-slice. This means that the dynamic-slice
  // is a no-op. Such a condition should never arise in practice, because the
  // algebraic simplifier should already remove such a dynamic-slice.

  // This testcase is only to ensure that the peeler does nothing in the case of
  // all constant offsets, if the algebraic simplifier is not run. However, in
  // the future, it is okay if the peeler peels this convert too because peeling
  // it does not change the semantics of the module.
  const char* hlo = R"(
  HloModule test
  body {
    param = (s32[], f32[1,1], bf16[1,1]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    data = f32[1,1] get-tuple-element(param), index=1
    bf16_data = bf16[1,1] get-tuple-element(param), index=2
    c0 = s32[] constant(0)
    ds = f32[1,1] dynamic-slice(data, c0, c0), dynamic_slice_sizes={1,1}
    convert = bf16[1,1] convert(ds)
    add = bf16[1,1] add(convert, bf16_data)
    c1 = s32[] constant(1)
    iter_plus_one = s32[] add(iter, c1)
    ROOT tuple = (s32[], f32[1,1], bf16[1,1]) tuple(iter_plus_one, data, add)
  }
  condition {
    param = (s32[], f32[1,1], bf16[1,1]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c8 = s32[] constant(8)
    ROOT lt = pred[] compare(iter, c8), direction=LT
  }
  ENTRY main {
    %iter_init = s32[] constant(0)
    %data_init = f32[1,1] parameter(0)
    %bf16_data_init = bf16[1,1] parameter(1)
    tuple_init = (s32[], f32[1,1], bf16[1,1]) tuple(%iter_init, %data_init, %bf16_data_init)
    ROOT while = (s32[], f32[1,1], bf16[1,1]) while(tuple_init), body=body, condition=condition
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConvertPeeler().Run(module.get(), {}));
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopConvertPeelerTest,
       DynamicSliceNonContigousSliceOffsetShouldNotChange) {
  const char* hlo = R"(
  HloModule test
  body {
    param = (s32[], f32[8,12], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    data = f32[8,12] get-tuple-element(param), index=1
    bf16_data = bf16[1,8] get-tuple-element(param), index=2
    c4 = s32[] constant(4)
    ds = f32[1,8] dynamic-slice(data, iter, c4), dynamic_slice_sizes={1,8}
    convert = bf16[1,8] convert(ds)
    add = bf16[1,8] add(convert, bf16_data)
    c1 = s32[] constant(1)
    iter_plus_one = s32[] add(iter, c1)
    ROOT tuple = (s32[], f32[8,12], bf16[1,8]) tuple(iter_plus_one, data, add)
  }
  condition {
    param = (s32[], f32[8,12], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c8 = s32[] constant(8)
    ROOT lt = pred[] compare(iter, c8), direction=LT
  }
  ENTRY main {
    %iter_init = s32[] constant(0)
    %data_init = f32[8,12] parameter(0)
    %bf16_data_init = bf16[1,8] parameter(1)
    tuple_init = (s32[], f32[8,12], bf16[1,8]) tuple(%iter_init, %data_init, %bf16_data_init)
    ROOT while = (s32[], f32[8,12], bf16[1,8]) while(tuple_init), body=body, condition=condition
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConvertPeeler().Run(module.get(), {}));
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopConvertPeelerTest,
       DynamicSliceNonContiguousSliceSizeShouldNotChange) {
  const char* hlo = R"(
  HloModule test
  body {
    param = (s32[], f32[8,12], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    data = f32[8,12] get-tuple-element(param), index=1
    bf16_data = bf16[1,8] get-tuple-element(param), index=2
    c0 = s32[] constant(0)
    ds = f32[1,8] dynamic-slice(data, iter, c0), dynamic_slice_sizes={1,8}
    convert = bf16[1,8] convert(ds)
    add = bf16[1,8] add(convert, bf16_data)
    c1 = s32[] constant(1)
    iter_plus_one = s32[] add(iter, c1)
    ROOT tuple = (s32[], f32[8,12], bf16[1,8]) tuple(iter_plus_one, data, add)
  }
  condition {
    param = (s32[], f32[8,12], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c8 = s32[] constant(8)
    ROOT lt = pred[] compare(iter, c8), direction=LT
  }
  ENTRY main {
    %iter_init = s32[] constant(0)
    %data_init = f32[8,12] parameter(0)
    %bf16_data_init = bf16[1,8] parameter(1)
    tuple_init = (s32[], f32[8,12], bf16[1,8]) tuple(%iter_init, %data_init, %bf16_data_init)
    ROOT while = (s32[], f32[8,12], bf16[1,8]) while(tuple_init), body=body, condition=condition
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConvertPeeler().Run(module.get(), {}));
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopConvertPeelerTest,
       DynamicSliceVariableIndexDoesNotCoverFullBufferShouldNotChange) {
  const char* hlo = R"(
  HloModule test
  body {
    param = (s32[], f32[9,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    data = f32[9,8] get-tuple-element(param), index=1
    bf16_data = bf16[1,8] get-tuple-element(param), index=2
    c1 = s32[] constant(1)
    iter_plus_one = s32[] add(iter, c1)
    c0 = s32[] constant(0)
    ds = f32[1,8] dynamic-slice(data, iter, c0), dynamic_slice_sizes={1,8}
    convert = bf16[1,8] convert(ds)
    add = bf16[1,8] add(convert, bf16_data)
    ROOT tuple = (s32[], f32[9,8], bf16[1,8]) tuple(iter_plus_one, data, add)
  }
  condition {
    param = (s32[], f32[9,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c8 = s32[] constant(8)
    ROOT lt = pred[] compare(iter, c8), direction=LT
  }
  ENTRY main {
    %c0 = s32[] constant(0)
    %p0 = f32[9,8] parameter(0)
    %p1 = bf16[1,8] parameter(1)
    tuple = (s32[], f32[9,8], bf16[1,8]) tuple(c0, %p0, %p1)
    ROOT while = (s32[], f32[9,8], bf16[1,8]) while(tuple), body=body, condition=condition
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConvertPeeler().Run(module.get(), {}));
  EXPECT_FALSE(changed);
}

TEST_F(
    WhileLoopConvertPeelerTest,
    DynamicSliceVariableIndexDoesNotCoverFullBufferStepNotOneShouldNotChange) {
  const char* hlo = R"(
  HloModule test
  body {
    param = (s32[], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    data = f32[8,8] get-tuple-element(param), index=1
    bf16_data = bf16[1,8] get-tuple-element(param), index=2
    c2 = s32[] constant(2)
    iter_plus_two = s32[] add(iter, c2)
    c0 = s32[] constant(0)
    ds = f32[1,8] dynamic-slice(data, iter, c0), dynamic_slice_sizes={1,8}
    convert = bf16[1,8] convert(ds)
    add = bf16[1,8] add(convert, bf16_data)
    ROOT tuple = (s32[], f32[8,8], bf16[1,8]) tuple(iter_plus_two, data, add)
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
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopConvertPeelerTest, DynamicSliceMoreThanOneUserShouldNotChange) {
  const char* hlo = R"(
  HloModule test
  body {
    param = (s32[], f32[8,8], bf16[1,8], f32[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    data = f32[8,8] get-tuple-element(param), index=1
    bf16_data = bf16[1,8] get-tuple-element(param), index=2
    f32_accum = f32[1,8] get-tuple-element(param), index=3
    c1 = s32[] constant(1)
    iter_plus_one = s32[] add(iter, c1)
    c0 = s32[] constant(0)
    ds = f32[1,8] dynamic-slice(data, iter, c0), dynamic_slice_sizes={1,8}
    convert = bf16[1,8] convert(ds)
    add.1 = f32[1,8] add(ds, f32_accum)
    add = bf16[1,8] add(convert, bf16_data)
    ROOT tuple = (s32[], f32[8,8], bf16[1,8], f32[1,8]) tuple(iter_plus_one, data, add, add.1)
  }
  condition {
    param = (s32[], f32[8,8], bf16[1,8], f32[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c8 = s32[] constant(8)
    ROOT lt = pred[] compare(iter, c8), direction=LT
  }
  ENTRY main {
    %c0 = s32[] constant(0)
    %p0 = f32[8,8] parameter(0)
    %p1 = bf16[1,8] parameter(1)
    %p2 = f32[1,8] parameter(2)
    tuple = (s32[], f32[8,8], bf16[1,8], f32[1,8]) tuple(c0, %p0, %p1, %p2)
    ROOT while = (s32[], f32[8,8], bf16[1,8], f32[1,8]) while(tuple), body=body, condition=condition
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConvertPeeler().Run(module.get(), {}));
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopConvertPeelerTest,
       DynamicSliceMoreThanOneUserBufferShouldNotChange) {
  const char* hlo = R"(
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
    custom_call = bf16[1,8] custom-call(data), custom_call_target="my_custom_call"
    convert = bf16[1,8] convert(ds)
    add = bf16[1,8] add(convert, bf16_data)
    add.1 = bf16[1,8] add(custom_call, add)
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
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopConvertPeelerTest,
       DynamicSliceBufferNotPassedToNextIterationShouldNotChange) {
  const char* hlo = R"(
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
    custom_call = f32[8,8] custom-call(data), custom_call_target="my_custom_call"
    convert = bf16[1,8] convert(ds)
    add = bf16[1,8] add(convert, bf16_data)
    ROOT tuple = (s32[], f32[8,8], bf16[1,8]) tuple(iter_plus_one, custom_call, add)
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
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopConvertPeelerTest,
       DynamicSliceBufferPassedAtDifferentIndexShouldNotChange) {
  const char* hlo = R"(
  HloModule test
  body {
    param = (s32[], f32[8,8], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    data = f32[8,8] get-tuple-element(param), index=1
    f32_data = f32[8,8] get-tuple-element(param), index=2
    bf16_data = bf16[1,8] get-tuple-element(param), index=3
    c1 = s32[] constant(1)
    iter_plus_one = s32[] add(iter, c1)
    c0 = s32[] constant(0)
    ds = f32[1,8] dynamic-slice(data, iter, c0), dynamic_slice_sizes={1,8}
    convert = bf16[1,8] convert(ds)
    add = bf16[1,8] add(convert, bf16_data)
    ROOT tuple = (s32[], f32[8,8], f32[8,8], bf16[1,8]) tuple(iter_plus_one, f32_data, data, add)
  }
  condition {
    param = (s32[], f32[8,8], f32[8,8], bf16[1,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c8 = s32[] constant(8)
    ROOT lt = pred[] compare(iter, c8), direction=LT
  }
  ENTRY main {
    %c0 = s32[] constant(0)
    %p0 = f32[8,8] parameter(0)
    %p1 = f32[8,8] parameter(1)
    %p2 = bf16[1,8] parameter(2)
    tuple = (s32[], f32[8,8], f32[8,8], bf16[1,8]) tuple(c0, %p0, %p1, %p2)
    ROOT while = (s32[], f32[8,8], f32[8,8], bf16[1,8]) while(tuple), body=body, condition=condition
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConvertPeeler().Run(module.get(), {}));
  EXPECT_FALSE(changed);
}
}  // namespace

}  // namespace xla::gpu
