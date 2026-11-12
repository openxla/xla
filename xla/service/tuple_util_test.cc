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

#include "xla/service/tuple_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;

using TupleUtilTest = HloHardwareIndependentTestBase;

TEST_F(TupleUtilTest, ExtractPrefix) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[32,32]{1,0},f32[32,32]{1,0},f32[32,32]{1,0}) parameter(0)
  ROOT p1 = f32[32,32]{1,0} parameter(1)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* param0 =
      module->entry_computation()->parameter_instruction(0);
  HloInstruction* prefix = TupleUtil::ExtractPrefix(param0, 2);

  EXPECT_THAT(prefix, op::Tuple(op::GetTupleElement(op::Parameter(0), 0),
                                op::GetTupleElement(op::Parameter(0), 1)));
}

TEST_F(TupleUtilTest, AppendSuffix) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[32,32]{1,0},f32[32,32]{1,0},f32[32,32]{1,0}) parameter(0)
  ROOT p1 = f32[32,32]{1,0} parameter(1)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* param0 =
      module->entry_computation()->parameter_instruction(0);
  HloInstruction* param1 =
      module->entry_computation()->parameter_instruction(1);

  HloInstruction* with_suffix =
      TupleUtil::AppendSuffix(param0, {param1, param1});

  EXPECT_THAT(with_suffix, op::Tuple(op::GetTupleElement(op::Parameter(0), 0),
                                     op::GetTupleElement(op::Parameter(0), 1),
                                     op::GetTupleElement(op::Parameter(0), 2),
                                     op::Parameter(1), op::Parameter(1)));
}

TEST_F(TupleUtilTest, ReplaceTupleWithTupleInst) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = f32[32,32]{1,0} parameter(0)
  p1 = f32[32,32]{1,0} parameter(1)
  ROOT tuple = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(p0, p1)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* tuple = FindInstruction(module.get(), "tuple");

  ASSERT_OK_AND_ASSIGN(HloInstruction * new_tuple,
                       TupleUtil::ReplaceTupleWith(p0, tuple, {1}));

  EXPECT_THAT(new_tuple, op::Tuple(op::Parameter(0), op::Parameter(0)));
}

TEST_F(TupleUtilTest, ReplaceTupleWithNonTupleInst) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  ROOT p0 = (f32[32,32]{1,0}, f32[32,32]{1,0}) parameter(0)
  p1 = f32[32,32]{1,0} parameter(1)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* p1 = FindInstruction(module.get(), "p1");

  ASSERT_OK_AND_ASSIGN(HloInstruction * new_tuple,
                       TupleUtil::ReplaceTupleWith(p1, p0, {0}));

  EXPECT_THAT(new_tuple, op::Tuple(op::Parameter(1),
                                   op::GetTupleElement(op::Parameter(0), 1)));
}

TEST_F(TupleUtilTest, ReplaceTupleWithNonTupleInstNested) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  ROOT p0 = (f32[32,32]{1,0}, (f32[32,32]{1,0}, f32[32,32]{1,0})) parameter(0)
  p1 = f32[32,32]{1,0} parameter(1)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* p1 = FindInstruction(module.get(), "p1");

  ASSERT_OK_AND_ASSIGN(HloInstruction * new_tuple,
                       TupleUtil::ReplaceTupleWith(p1, p0, {1, 0}));

  EXPECT_THAT(
      new_tuple,
      op::Tuple(op::GetTupleElement(op::Parameter(0), 0),
                op::Tuple(op::Parameter(1),
                          op::GetTupleElement(
                              op::GetTupleElement(op::Parameter(0), 1), 1))));
}

TEST_F(TupleUtilTest, AddGetTupleElements) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[32,32]{1,0}, (f32[32,32]{1,0}, f32[32,32]{1,0})) parameter(0)
  gte = (f32[32,32]{1,0}, f32[32,32]{1,0}) get-tuple-element(p0), index=1
  ROOT root = f32[32,32]{1,0} get-tuple-element(gte), index=1
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* existing_gte = FindInstruction(module.get(), "gte");

  HloInstruction* new_gte = TupleUtil::AddGetTupleElements({p0, {1, 0}});

  EXPECT_THAT(new_gte, op::GetTupleElement(existing_gte, 0));
}

TEST_F(TupleUtilTest, ExtractPrefixPropagatesOriginalValue) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[], f32[], f32[]) parameter(0), origin={({"a"}, {"b"}, {"c"})}
  ROOT p1 = f32[] parameter(1)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* param0 =
      module->entry_computation()->parameter_instruction(0);

  HloInstruction* prefix = TupleUtil::ExtractPrefix(param0, 2);

  ASSERT_NE(prefix->original_value(), nullptr);
  EXPECT_EQ(prefix->original_value()->ToString(), R"(({"a"}, {"b"}))");
  ASSERT_NE(prefix->operand(1)->original_value(), nullptr);
  EXPECT_EQ(prefix->operand(1)->original_value()->ToString(), R"({"b"})");
}

TEST_F(TupleUtilTest, ExtractPrefixPreservesCallHierarchy) {
  const std::string hlo_string = R"(
HloModule Module

body {
  ROOT p = (f32[], f32[]) parameter(0)
}

cond {
  p = (f32[], f32[]) parameter(0)
  ROOT c = pred[] constant(true)
}

ENTRY entry {
  init = (f32[], f32[]) parameter(0)
  ROOT while = (f32[], f32[]) while(init), condition=cond, body=body, origin={({"w0"}, {"w1"}),["while#$"]}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* while_instr = FindInstruction(module.get(), "while");

  HloInstruction* prefix = TupleUtil::ExtractPrefix(while_instr, 1);

  ASSERT_NE(prefix->original_value(), nullptr);
  EXPECT_EQ(prefix->original_value()->ToString(), R"(({"w0"}),["while#$"])");
  // A get-tuple-element of a while loop doesn't carry its call hierarchy.
  ASSERT_NE(prefix->operand(0)->original_value(), nullptr);
  EXPECT_EQ(prefix->operand(0)->original_value()->ToString(), R"({"w0"})");
}

TEST_F(TupleUtilTest, ExtractPrefixIgnoresStaleOriginalValue) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[], f32[]) parameter(0), origin={({"a"}, {"b"})}
  ROOT p1 = f32[] parameter(1)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo_string));
  HloInstruction* param0 =
      module->entry_computation()->parameter_instruction(0);
  // Widen the shape of `param0` in place without updating its original value,
  // as a pass might do in the middle of a transformation.
  *param0->mutable_shape()->add_tuple_shapes() =
      ShapeUtil::MakeScalarShape(F32);

  HloInstruction* prefix = TupleUtil::ExtractPrefix(param0, 2);

  EXPECT_EQ(prefix->original_value(), nullptr);
  EXPECT_EQ(prefix->operand(0)->original_value(), nullptr);
  EXPECT_EQ(prefix->operand(1)->original_value(), nullptr);
}

TEST_F(TupleUtilTest, ExtractPrefixWithoutOriginalValue) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[], f32[]) parameter(0)
  ROOT p1 = f32[] parameter(1)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* param0 =
      module->entry_computation()->parameter_instruction(0);

  HloInstruction* prefix = TupleUtil::ExtractPrefix(param0, 2);

  EXPECT_EQ(prefix->original_value(), nullptr);
  EXPECT_EQ(prefix->operand(0)->original_value(), nullptr);
}

TEST_F(TupleUtilTest, AppendSuffixPropagatesOriginalValue) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[], f32[]) parameter(0), origin={({"a"}, {"b"}),["call"]}
  p1 = f32[] parameter(1), origin={{"p1"}}
  ROOT p2 = f32[] parameter(2)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  HloComputation* entry = module->entry_computation();

  HloInstruction* with_suffix = TupleUtil::AppendSuffix(
      entry->parameter_instruction(0),
      {entry->parameter_instruction(1), entry->parameter_instruction(2)});

  ASSERT_NE(with_suffix->original_value(), nullptr);
  EXPECT_EQ(with_suffix->original_value()->ToString(),
            R"(({"a"}, {"b"}, {"p1"}, {}),["call"])");
}

TEST_F(TupleUtilTest, AppendSuffixToTupleWithoutOriginalValue) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[], f32[]) parameter(0)
  ROOT p1 = f32[] parameter(1), origin={{"p1"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  HloComputation* entry = module->entry_computation();

  HloInstruction* with_suffix = TupleUtil::AppendSuffix(
      entry->parameter_instruction(0), {entry->parameter_instruction(1)});

  ASSERT_NE(with_suffix->original_value(), nullptr);
  EXPECT_EQ(with_suffix->original_value()->ToString(), R"(({}, {}, {"p1"}))");
}

TEST_F(TupleUtilTest, ReplaceTupleWithPropagatesOriginalValue) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = f32[] parameter(0), origin={{"p0"}}
  p1 = f32[] parameter(1), origin={{"p1"}}
  p2 = f32[] parameter(2), origin={{"p2"}}
  ROOT tuple = (f32[], f32[]) tuple(p0, p1), origin={({"t0"}, {"t1"})}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* tuple = FindInstruction(module.get(), "tuple");
  HloInstruction* p2 = FindInstruction(module.get(), "p2");

  ASSERT_OK_AND_ASSIGN(HloInstruction * replaced,
                       TupleUtil::ReplaceTupleWith(p2, tuple, {1}));
  ASSERT_NE(replaced->original_value(), nullptr);
  EXPECT_EQ(replaced->original_value()->ToString(), R"(({"t0"}, {"p2"}))");

  // Replacing at an index past the end appends the new instruction.
  ASSERT_OK_AND_ASSIGN(HloInstruction * appended,
                       TupleUtil::ReplaceTupleWith(p2, tuple, {2}));
  ASSERT_NE(appended->original_value(), nullptr);
  EXPECT_EQ(appended->original_value()->ToString(),
            R"(({"t0"}, {"t1"}, {"p2"}))");
}

TEST_F(TupleUtilTest,
       ReplaceTupleWithClearsReplacedElementWithoutOriginalValue) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = f32[] parameter(0), origin={{"p0"}}
  p1 = f32[] parameter(1), origin={{"p1"}}
  p2 = f32[] parameter(2)
  ROOT tuple = (f32[], f32[]) tuple(p0, p1), origin={({"t0"}, {"t1"})}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* tuple = FindInstruction(module.get(), "tuple");
  HloInstruction* p2 = FindInstruction(module.get(), "p2");

  ASSERT_OK_AND_ASSIGN(HloInstruction * replaced,
                       TupleUtil::ReplaceTupleWith(p2, tuple, {1}));

  // `p2` replaces the value of element 1, so element 1 no longer holds the
  // value that "t1" refers to.
  ASSERT_NE(replaced->original_value(), nullptr);
  EXPECT_EQ(replaced->original_value()->ToString(), R"(({"t0"}, {}))");
}

TEST_F(TupleUtilTest, ReplaceTupleWithNestedPropagatesOriginalValue) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[], (f32[], f32[])) parameter(0), origin={({"a"}, ({"b"}, {"c"}))}
  ROOT p1 = f32[] parameter(1), origin={{"p1"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* p1 = FindInstruction(module.get(), "p1");

  ASSERT_OK_AND_ASSIGN(HloInstruction * replaced,
                       TupleUtil::ReplaceTupleWith(p1, p0, {1, 0}));

  EXPECT_THAT(
      replaced,
      op::Tuple(op::GetTupleElement(op::Parameter(0), 0),
                op::Tuple(op::Parameter(1),
                          op::GetTupleElement(
                              op::GetTupleElement(op::Parameter(0), 1), 1))));
  ASSERT_NE(replaced->original_value(), nullptr);
  EXPECT_EQ(replaced->original_value()->ToString(),
            R"(({"a"}, ({"p1"}, {"c"})))");
  // The intermediate instructions created for the nested tuple also get
  // original values.
  const HloInstruction* inner_tuple = replaced->operand(1);
  ASSERT_NE(inner_tuple->original_value(), nullptr);
  EXPECT_EQ(inner_tuple->original_value()->ToString(), R"(({"p1"}, {"c"}))");
  const HloInstruction* inner_gte = inner_tuple->operand(1)->operand(0);
  ASSERT_NE(inner_gte->original_value(), nullptr);
  EXPECT_EQ(inner_gte->original_value()->ToString(), R"(({"b"}, {"c"}))");
}

TEST_F(TupleUtilTest, AddGetTupleElementsPropagatesOriginalValue) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[], (f32[], f32[])) parameter(0), origin={({"a"}, ({"b"}, {"c"}))}
  ROOT p1 = f32[] parameter(1)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* p0 = FindInstruction(module.get(), "p0");

  HloInstruction* gte = TupleUtil::AddGetTupleElements(HloPosition{p0, {1, 1}});

  EXPECT_THAT(gte, op::GetTupleElement(op::GetTupleElement(p0, 1), 1));
  ASSERT_NE(gte->original_value(), nullptr);
  EXPECT_EQ(gte->original_value()->ToString(), R"({"c"})");
  ASSERT_NE(gte->operand(0)->original_value(), nullptr);
  EXPECT_EQ(gte->operand(0)->original_value()->ToString(), R"(({"b"}, {"c"}))");
}

TEST_F(TupleUtilTest, GetTupleInstructionAtIndexTest) {
  const std::string hlo_string = R"(
HloModule GetTupleInstructionAtIndexTest

ENTRY entry {
  p0 = (f32[32,32]{1,0}, (f32[32,32]{1,0}, f32[32,32]{1,0}, (f32[32,32]{1,0})), f32[32,32]) parameter(0)
  gte = (f32[32,32]{1,0}, f32[32,32]{1,0}, (f32[32,32]{1,0})) get-tuple-element(p0), index=1
  gte.1 = f32[32,32]{1,0} get-tuple-element(p0), index=0
  gte.2 =  (f32[32,32]{1,0}) get-tuple-element(gte), index=2
  gte.3 = f32[32,32]{1,0} get-tuple-element(gte.2), index=0
  ROOT root = f32[32,32]{1,0} get-tuple-element(gte), index=1
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  ASSERT_NE(p0, nullptr);
  HloInstruction* gte = FindInstruction(module.get(), "gte");
  ASSERT_NE(gte, nullptr);
  HloInstruction* gte1 = FindInstruction(module.get(), "gte.1");
  ASSERT_NE(gte1, nullptr);
  HloInstruction* gte2 = FindInstruction(module.get(), "gte.2");
  ASSERT_NE(gte2, nullptr);

  // Valid queries.
  EXPECT_THAT(TupleUtil::GetTupleInstructionAtIndex(*p0, {1, 1}),
              op::GetTupleElement(gte, 1));
  EXPECT_THAT(TupleUtil::GetTupleInstructionAtIndex(*p0, {1}),
              op::GetTupleElement(p0, 1));
  EXPECT_THAT(TupleUtil::GetTupleInstructionAtIndex(*p0, {1, 2, 0}),
              op::GetTupleElement(gte2, 0));
  EXPECT_THAT(TupleUtil::GetTupleInstructionAtIndex(*p0, {1, 2}),
              op::GetTupleElement(gte, 2));
  EXPECT_THAT(TupleUtil::GetTupleInstructionAtIndex(*p0, {1, 2}),
              op::GetTupleElement(gte, 2));
  EXPECT_THAT(TupleUtil::GetTupleInstructionAtIndex(*p0, {0}),
              op::GetTupleElement(p0, 0));

  // Invalid queries.
  // Out of bounds
  EXPECT_EQ(TupleUtil::GetTupleInstructionAtIndex(*p0, {3}), nullptr);
  EXPECT_EQ(TupleUtil::GetTupleInstructionAtIndex(*p0, {-1}), nullptr);
  EXPECT_EQ(TupleUtil::GetTupleInstructionAtIndex(*p0, {1, 3}), nullptr);
  EXPECT_EQ(TupleUtil::GetTupleInstructionAtIndex(*p0, {0, -1}), nullptr);
  EXPECT_EQ(TupleUtil::GetTupleInstructionAtIndex(*p0, {1, 2, 3}), nullptr);
  EXPECT_EQ(TupleUtil::GetTupleInstructionAtIndex(*p0, {1, 2, -1}), nullptr);
  // Valid index but no gte present.
  EXPECT_EQ(TupleUtil::GetTupleInstructionAtIndex(*p0, {2}), nullptr);
}

}  // namespace
}  // namespace xla
