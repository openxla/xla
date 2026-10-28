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

#include "xla/service/while_loop_fusible_sinking.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/while_loop_simplifier.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;
using ::testing::_;
using WhileLoopFusibleSinkingTest = HloHardwareIndependentTestBase;

TEST_F(WhileLoopFusibleSinkingTest, SinkOneFusible) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[2],f32[2]) parameter(0)
  p_body.0 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=0
  p_body.1 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=1

  add.0 = f32[2] add(p_body.0, p_body.1)
  ROOT root = (f32[2],f32[2]) tuple(add.0, p_body.1)
}

condition {
  p_cond = (f32[2],f32[2]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[2] parameter(0)
  const_1 = f32[2] iota(), iota_dimension=0
  while_init = (f32[2],f32[2]) tuple(const_0, const_1)
  ROOT while = (f32[2],f32[2]) while(while_init), condition=condition, body=body
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(bool changed,
                       WhileLoopFusibleSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_body = module->GetComputationWithName("body");
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::Add(_, op::Iota()), _));
}

TEST_F(WhileLoopFusibleSinkingTest, SinkMask) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[5,7],f32[5,7]) parameter(0)
  p_body.0 = get-tuple-element(p_body), index=0
  p_body.1 = get-tuple-element(p_body), index=1

  add.0 = add(p_body.0, p_body.1)
  ROOT root = tuple(add.0, p_body.1)
}

condition {
  p_cond = (f32[5,7],f32[5,7]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[5,7] parameter(0)
  p = f32[5] parameter(1)
  a = f32[5,7] iota(), iota_dimension=0
  b = f32[5,7] iota(), iota_dimension=1
  c = add(a, b)
  d = f32[5,7] broadcast(p), dimensions={0}
  mask = multiply(c,d)
  while_init = tuple(const_0, mask)
  ROOT while = while(while_init), condition=condition, body=body
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(bool changed,
                       WhileLoopFusibleSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_body = module->GetComputationWithName("body");
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::Add(_, op::Multiply(op::Add(op::Iota(), op::Iota()),
                                                op::Broadcast())),
                        _, _));
}

TEST_F(WhileLoopFusibleSinkingTest, NoSinkSlicedMask) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[5,7],f32[5,7]) parameter(0)
  p_body.0 = get-tuple-element(p_body), index=0
  p_body.1 = get-tuple-element(p_body), index=1
  z = s32[] constant(0)
  j = s32[] constant(3)
  ds = f32[1,7] dynamic-slice(p_body.1, j, z), dynamic_slice_sizes={1,7}
  r = f32[7] reshape(ds)
  b = f32[5,7] broadcast(r), dimensions={1}
  a = add(b, p_body.0)
  add.0 = add(a, p_body.1)
  ROOT root = tuple(add.0, p_body.1)
}

condition {
  p_cond = (f32[5,7],f32[5,7]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[5,7] parameter(0)
  p = f32[5] parameter(1)
  a = f32[5,7] iota(), iota_dimension=0
  b = f32[5,7] iota(), iota_dimension=1
  c = add(a, b)
  d = f32[5,7] broadcast(p), dimensions={0}
  mask = multiply(c,d)
  while_init = tuple(const_0, mask)
  ROOT while = while(while_init), condition=condition, body=body
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(bool changed,
                       WhileLoopFusibleSinking{}.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopFusibleSinkingTest, TestPlumbSingleBroadcast) {
  const std::string hlo_string_before = R"(
  HloModule test

  loop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[4,3,5]{2,1,0} get-tuple-element(loop_var.1), index=2
    bitcast.12855 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} bitcast(get-tuple-element.3)
    add.40974 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.2, bitcast.12855)
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(idx, add.40974, get-tuple-element.3)
  }

  loop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.3 = s32[]{:T(128)} get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant(4)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }

  ENTRY %main {
    param.1 = s32[4,3,5]{2,1,0} iota(), iota_dimension=0
    zero = s32[]{:T(128)} constant(0)
    zeros32 = s32[]{:T(128)} constant(0)
    broadcast = s32[1,1,1,4,3,5]{5,4,3,2,1,0} broadcast(zeros32)
    input = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(zero, broadcast, param.1)
    ROOT while = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) while(input), condition=loop.condition, body=loop.body
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_before,
                       ParseAndReturnVerifiedModule(hlo_string_before));
  ASSERT_OK_AND_ASSIGN(bool changed,
                       WhileLoopFusibleSinking{}.Run(module_before.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module_before.get(), "while"),
              op::While(op::Tuple(_, op::CustomCall(), _, _)));
}

TEST_F(WhileLoopFusibleSinkingTest, TestDontSinkBroadcast) {
  const std::string hlo_string_before = R"(
  HloModule test

  loop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[4,3,5]{2,1,0} get-tuple-element(loop_var.1), index=2
    bitcast.12855 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} bitcast(get-tuple-element.3)
    add.40974 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.2, bitcast.12855)
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(idx, add.40974, get-tuple-element.3)
  }

  loop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.3 = s32[]{:T(128)} get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant(4)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }

  ENTRY %main {
    param.1 = s32[4,3,5]{2,1,0} parameter(0)
    zero = s32[]{:T(128)} constant(0)
    zeros32 = s32[]{:T(128)} constant(0)
    broadcast = s32[1,1,1,4,3,5]{5,4,3,2,1,0} broadcast(zeros32)
    input = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(zero, broadcast, param.1)
    ROOT while = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) while(input), condition=loop.condition, body=loop.body
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_before,
                       ParseAndReturnVerifiedModule(hlo_string_before));
  ASSERT_OK_AND_ASSIGN(bool changed, WhileLoopFusibleSinking(
                                         /*sink_broadcast_of_constant=*/false)
                                         .Run(module_before.get()));
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopFusibleSinkingTest,
       TestPlumbSingleBroadcastNotFlattenCallGraph) {
  const std::string hlo_string_before = R"(
  HloModule test

  loop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[4,3,5]{2,1,0} get-tuple-element(loop_var.1), index=2
    bitcast.12855 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} bitcast(get-tuple-element.3)
    add.40974 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.2, bitcast.12855)
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(idx, add.40974, get-tuple-element.3)
  }

  loop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.3 = s32[]{:T(128)} get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant(4)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }

  ENTRY %main {
    param.1 = s32[4,3,5]{2,1,0} iota(), iota_dimension=0
    zero = s32[]{:T(128)} constant(0)
    zeros32 = s32[]{:T(128)} constant(0)
    broadcast = s32[1,1,1,4,3,5]{5,4,3,2,1,0} broadcast(zeros32)
    input = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(zero, broadcast, param.1)
    while1 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) while(input), condition=loop.condition, body=loop.body
    input2 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(zero, broadcast, param.1)
    ROOT while2 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) while(input2), condition=loop.condition, body=loop.body
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_before,
                       ParseAndReturnVerifiedModule(hlo_string_before));
  CHECK_OK(FlattenCallGraph{}.Run(module_before.get()).status());
  ASSERT_OK_AND_ASSIGN(bool changed,
                       WhileLoopFusibleSinking{}.Run(module_before.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module_before.get(), "while1"),
              op::While(op::Tuple(_, op::CustomCall(), _, _)));
  EXPECT_THAT(FindInstruction(module_before.get(), "while2"),
              op::While(op::Tuple(_, op::CustomCall(), _, _)));
}

TEST_F(WhileLoopFusibleSinkingTest, TestPlumbMultipleBroadcast) {
  const std::string hlo_string_before = R"(
  HloModule test

  loop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.4 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=2
    get-tuple-element.3 = s32[4,3,5]{2,1,0} get-tuple-element(loop_var.1), index=3
    bitcast.12855 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} bitcast(get-tuple-element.3)
    add.40974 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.2, bitcast.12855)
    add.1 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.4, add.40974)
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(idx, add.40974, add.1, get-tuple-element.3)
  }

  loop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.3 = s32[]{:T(128)} get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant(4)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }

  ENTRY %main {
    param.1 = s32[4,3,5]{2,1,0} iota(), iota_dimension=0
    zero = s32[]{:T(128)} constant(0)
    zeros32 = s32[]{:T(128)} constant(0)
    broadcast = s32[1,1,1,4,3,5]{5,4,3,2,1,0} broadcast(zeros32)
    input = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(zero, broadcast, broadcast, param.1)
    ROOT while = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) while(input), condition=loop.condition, body=loop.body
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_before,
                       ParseAndReturnVerifiedModule(hlo_string_before));
  ASSERT_OK_AND_ASSIGN(bool changed,
                       WhileLoopFusibleSinking{}.Run(module_before.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      FindInstruction(module_before.get(), "while"),
      op::While(op::Tuple(_, op::CustomCall(), op::CustomCall(), _, _)));
}

TEST_F(WhileLoopFusibleSinkingTest, TestNoPlumbWithBadCondition) {
  const std::string hlo_string_before = R"(
  HloModule test
  tmp {
    x = s32[] parameter(0)
    y = s32[] parameter(1)
    ROOT add = s32[] add(x, y)
  }

  loop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.4 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=2
    get-tuple-element.3 = s32[4,3,5]{2,1,0} get-tuple-element(loop_var.1), index=3
    bitcast.12855 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} bitcast(get-tuple-element.3)
    add.40974 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.2, bitcast.12855)
    add.1 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.4, add.40974)
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(idx, add.40974, add.1, get-tuple-element.3)
  }

  loop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.3 = s32[]{:T(128)} get-tuple-element(loop_var.2), index=0
    get-tuple-element.4 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.2), index=1
    z = s32[]{:T(128)} constant(0)
    r = s32[]{:T(128)} reduce(get-tuple-element.4, z), dimensions={0,1,2,3,4,5}, to_apply=tmp
    ROOT less-than = pred[] compare(get-tuple-element.3, r), direction=LT
  }

  ENTRY %main {
    param.1 = s32[4,3,5]{2,1,0} parameter(0)
    zero = s32[]{:T(128)} constant(0)
    zeros32 = s32[]{:T(128)} constant(0)
    broadcast = s32[1,1,1,4,3,5]{5,4,3,2,1,0} broadcast(zeros32)
    input = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(zero, broadcast, broadcast, param.1)
    ROOT while = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) while(input), condition=loop.condition, body=loop.body
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_before,
                       ParseAndReturnVerifiedModule(hlo_string_before));
  ASSERT_OK_AND_ASSIGN(bool changed,
                       WhileLoopFusibleSinking{}.Run(module_before.get()));
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopFusibleSinkingTest, TestNoPlumbWithUnknonwnTripCount) {
  const std::string hlo_string_before = R"(
  HloModule test
  tmp {
    x = s32[] parameter(0)
    y = s32[] parameter(1)
    ROOT add = s32[] add(x, y)
  }

  loop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.4 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=2
    get-tuple-element.3 = s32[4,3,5]{2,1,0} get-tuple-element(loop_var.1), index=3
    bitcast.12855 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} bitcast(get-tuple-element.3)
    add.40974 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.2, bitcast.12855)
    add.1 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.4, add.40974)
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(idx, add.40974, add.1, get-tuple-element.3)
  }

  loop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.3 = s32[]{:T(128)} get-tuple-element(loop_var.2), index=0
    m = s32[] constant(0)
    v = s32[] constant(10000)
    rng = s32[] rng(m, v), distribution=rng_uniform
    ROOT less-than = pred[] compare(get-tuple-element.3, rng), direction=LT
  }

  ENTRY %main {
    param.1 = s32[4,3,5]{2,1,0} parameter(0)
    zero = s32[]{:T(128)} constant(0)
    zeros32 = s32[]{:T(128)} constant(0)
    broadcast = s32[1,1,1,4,3,5]{5,4,3,2,1,0} broadcast(zeros32)
    input = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(zero, broadcast, broadcast, param.1)
    ROOT while = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) while(input), condition=loop.condition, body=loop.body
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_before,
                       ParseAndReturnVerifiedModule(hlo_string_before));
  ASSERT_OK_AND_ASSIGN(bool changed,
                       WhileLoopFusibleSinking{}.Run(module_before.get()));
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopFusibleSinkingTest, SinkMaskWithOriginalValue) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[5,7],f32[5,7]) parameter(0)
  p_body.0 = get-tuple-element(p_body), index=0
  p_body.1 = get-tuple-element(p_body), index=1

  add.0 = add(p_body.0, p_body.1)
  ROOT root = tuple(add.0, p_body.1)
}

condition {
  p_cond = (f32[5,7],f32[5,7]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[5,7] parameter(0), origin={{"constant"}}
  p = f32[5] parameter(1), origin={{"parameter"}}
  a = f32[5,7] iota(), iota_dimension=0
  b = f32[5,7] iota(), iota_dimension=1
  c = add(a, b)
  d = f32[5,7] broadcast(p), dimensions={0}
  mask = multiply(c,d), origin={{"mask"}}
  while_init = tuple(const_0, mask), origin={({"constant"}, {"mask"})}
  ROOT while = while(while_init), condition=condition, body=body, origin={({"while" {0}}, {"while" {1}})}
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(bool changed,
                       WhileLoopFusibleSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  HloInstruction* while_instr = FindInstruction(module.get(), "while");
  ASSERT_NE(while_instr->original_value(), nullptr);
  EXPECT_EQ(while_instr->original_value()->ToString(),
            R"(({"while" {0}}, {"while" {1}}, {"parameter"}))");
  HloInstruction* while_init = while_instr->while_init();
  ASSERT_NE(while_init->original_value(), nullptr);
  EXPECT_EQ(while_init->original_value()->ToString(),
            R"(({"constant"}, {"mask"}, {"parameter"}))");
}

TEST_F(WhileLoopFusibleSinkingTest, PlumbSingleBroadcastWithOriginalValue) {
  const std::string hlo_string_before = R"(
  HloModule test

  loop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[4,3,5]{2,1,0} get-tuple-element(loop_var.1), index=2
    bitcast.12855 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} bitcast(get-tuple-element.3)
    add.40974 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.2, bitcast.12855)
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(idx, add.40974, get-tuple-element.3)
  }

  loop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.3 = s32[]{:T(128)} get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant(4)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }

  ENTRY %main {
    param.1 = s32[4,3,5]{2,1,0} iota(), iota_dimension=0
    zero = s32[]{:T(128)} constant(0), origin={{"zero"}}
    zeros32 = s32[]{:T(128)} constant(0), origin={{"zeros32"}}
    broadcast = s32[1,1,1,4,3,5]{5,4,3,2,1,0} broadcast(zeros32), origin={{"broadcast"}}
    input = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(zero, broadcast, param.1), origin={({"zero"}, {"zeros32"}, {"broadcast"})}
    ROOT while = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) while(input), condition=loop.condition, body=loop.body, origin={({"while" {0}}, {"while" {1}}, {"while" {2}})}
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string_before));
  ASSERT_OK_AND_ASSIGN(bool changed,
                       WhileLoopFusibleSinking{}.Run(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* while_instr = FindInstruction(module.get(), "while");
  ASSERT_NE(while_instr->original_value(), nullptr);
  EXPECT_EQ(while_instr->original_value()->ToString(),
            R"(({"while" {0}}, {"while" {1}}, {"while" {2}}, {"zero"}))");
  HloInstruction* while_init = while_instr->while_init();
  ASSERT_NE(while_init->original_value(), nullptr);
  EXPECT_EQ(while_init->original_value()->ToString(),
            R"(({"zero"}, {"zeros32"}, {"broadcast"}, {"zero"}))");
}

TEST_F(WhileLoopFusibleSinkingTest, SinkMaskWithOriginalValueOnBodyRoot) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[5,7],f32[5,7]) parameter(0)
  p_body.0 = get-tuple-element(p_body), index=0
  p_body.1 = get-tuple-element(p_body), index=1

  add.0 = add(p_body.0, p_body.1)
  ROOT root = tuple(add.0, p_body.1), origin={({"while" {0}}, {"while" {1}})}
}

condition {
  p_cond = (f32[5,7],f32[5,7]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[5,7] parameter(0), origin={{"constant"}}
  p = f32[5] parameter(1), origin={{"parameter"}}
  a = f32[5,7] iota(), iota_dimension=0
  b = f32[5,7] iota(), iota_dimension=1
  c = add(a, b)
  d = f32[5,7] broadcast(p), dimensions={0}
  mask = multiply(c,d), origin={{"mask"}}
  while_init = tuple(const_0, mask), origin={({"constant"}, {"mask"})}
  ROOT while = while(while_init), condition=condition, body=body, origin={({"while" {0}}, {"while" {1}})}
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(bool changed,
                       WhileLoopFusibleSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  HloInstruction* while_instr = FindInstruction(module.get(), "while");
  HloInstruction* root = while_instr->while_body()->root_instruction();
  ASSERT_NE(root->original_value(), nullptr);
  EXPECT_EQ(root->original_value()->ToString(),
            R"(({"while" {0}}, {"while" {1}}, {"parameter"}))");

  xla::HloVerifier verifier(/*layout_sensitive=*/false,
                            /*allow_mixed_precision=*/false);
  EXPECT_OK(verifier.Run(module.get()).status());
}

// A loop that runs zero or one times and writes a slice of `ring` into `buf`,
// as a conditional in place update is written in JAX, which disables hoisting
// from such loops. The `$` placeholders vary per test.
constexpr absl::string_view kSliceUpdateLoop = R"(
HloModule SliceUpdateLoop

body {
  p = (s32[], f32[8,4,16], s32[], f32[4,16], s32[]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  buf = f32[8,4,16] get-tuple-element(p), index=1
  n = s32[] get-tuple-element(p), index=2
  v = f32[4,16] get-tuple-element(p), index=3
  slot = s32[] get-tuple-element(p), index=4
  one = s32[] constant(1)
  i1 = s32[] add(i, one)
  $body_update
  zero = s32[] constant(0)
  dus = f32[8,4,16] dynamic-update-slice(buf, update, slot, zero, zero)
  ROOT t = (s32[], f32[8,4,16], s32[], f32[4,16], s32[]) tuple(i1, dus, n, v, slot)
}

cond {
  p = (s32[], f32[8,4,16], s32[], f32[4,16], s32[]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  n = s32[] get-tuple-element(p), index=2
  ROOT lt = pred[] compare(i, n), direction=LT
}

ENTRY entry {
  ring = f32[2,4,16] parameter(0)
  j = s32[] parameter(1)
  buf = f32[8,4,16] parameter(2)
  write = pred[] parameter(3)
  slot = s32[] parameter(4)
  zero = s32[] constant(0)
  one = s32[] constant(1)
  n = s32[] select(write, one, zero)
  $prelude
  ds = f32[1,4,16] dynamic-slice($sliced, j, zero, zero), dynamic_slice_sizes={1,4,16}
  v = f32[4,16] reshape(ds)
  init = (s32[], f32[8,4,16], s32[], f32[4,16], s32[]) tuple(zero, buf, n, v, slot)
  w = (s32[], f32[8,4,16], s32[], f32[4,16], s32[]) while(init), condition=cond, body=body$attrs
  $entry_root
}
)";

constexpr absl::string_view kUpdateFromSlice =
    "update = f32[1,4,16] reshape(v)";
constexpr absl::string_view kReturnBuffer =
    "ROOT out = f32[8,4,16] get-tuple-element(w), index=1";
constexpr absl::string_view kNoHoisting =
    ", frontend_attributes={_xla_disable_loop_instr_hoisting=\"true\"}";
std::string SliceUpdateLoop(absl::string_view body_update,
                            absl::string_view entry_root,
                            absl::string_view attrs = kNoHoisting,
                            absl::string_view sliced = "ring",
                            absl::string_view prelude = "") {
  return absl::StrReplaceAll(kSliceUpdateLoop, {{"$body_update", body_update},
                                                {"$entry_root", entry_root},
                                                {"$attrs", attrs},
                                                {"$sliced", sliced},
                                                {"$prelude", prelude}});
}

TEST_F(WhileLoopFusibleSinkingTest, SinkDynamicSliceIntoUpdate) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(
                           SliceUpdateLoop(kUpdateFromSlice, kReturnBuffer)));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      WhileLoopFusibleSinking(/*sink_broadcast_of_constant=*/true,
                              /*sink_dynamic_slices_into_updates=*/true)
          .Run(module.get()));
  ASSERT_TRUE(changed);

  // The update now slices the ring buffer, which the loop carries.
  HloComputation* body = module->GetComputationWithName("body");
  EXPECT_THAT(body->root_instruction()->operand(1),
              op::DynamicUpdateSlice(_,
                                     op::Reshape(op::Reshape(op::DynamicSlice(
                                         op::GetTupleElement(op::Parameter(0)),
                                         op::GetTupleElement(op::Parameter(0)),
                                         op::Constant(), op::Constant()))),
                                     _, _, _));

  // Once the old loop element is dead, nothing slices outside the loop.
  ASSERT_OK(WhileLoopSimplifier().Run(module.get()).status());
  ASSERT_OK(TupleSimplifier().Run(module.get()).status());
  ASSERT_OK(HloDCE().Run(module.get()).status());
  EXPECT_FALSE(absl::c_any_of(module->entry_computation()->instructions(),
                              HloPredicateIsOp<HloOpcode::kDynamicSlice>));
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/false)
                .Run(module.get())
                .status());
}

// Two such loops write the same slice into two buffers; the second gets it
// through the first loop's output.
TEST_F(WhileLoopFusibleSinkingTest, SinkDynamicSliceIntoChainedUpdates) {
  constexpr absl::string_view kLoop = R"(
body$n {
  p = (s32[], f32[8,4,16], s32[], f32[4,16], s32[]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  buf = f32[8,4,16] get-tuple-element(p), index=1
  n = s32[] get-tuple-element(p), index=2
  v = f32[4,16] get-tuple-element(p), index=3
  slot = s32[] get-tuple-element(p), index=4
  one = s32[] constant(1)
  i1 = s32[] add(i, one)
  update = f32[1,4,16] reshape(v)
  zero = s32[] constant(0)
  dus = f32[8,4,16] dynamic-update-slice(buf, update, slot, zero, zero)
  ROOT t = (s32[], f32[8,4,16], s32[], f32[4,16], s32[]) tuple(i1, dus, n, v, slot)
}

cond$n {
  p = (s32[], f32[8,4,16], s32[], f32[4,16], s32[]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  n = s32[] get-tuple-element(p), index=2
  ROOT lt = pred[] compare(i, n), direction=LT
}
)";
  const std::string hlo =
      absl::StrCat("HloModule ChainedSliceUpdateLoops\n",
                   absl::StrReplaceAll(kLoop, {{"$n", ""}}),
                   absl::StrReplaceAll(kLoop, {{"$n", "2"}}),
                   R"(
ENTRY entry {
  ring = f32[2,4,16] parameter(0)
  j = s32[] parameter(1)
  buf = f32[8,4,16] parameter(2)
  write = pred[] parameter(3)
  slot = s32[] parameter(4)
  buf2 = f32[8,4,16] parameter(5)
  zero = s32[] constant(0)
  one = s32[] constant(1)
  n = s32[] select(write, one, zero)
  ds = f32[1,4,16] dynamic-slice(ring, j, zero, zero), dynamic_slice_sizes={1,4,16}
  v = f32[4,16] reshape(ds)
  init = (s32[], f32[8,4,16], s32[], f32[4,16], s32[]) tuple(zero, buf, n, v, slot)
  w = (s32[], f32[8,4,16], s32[], f32[4,16], s32[]) while(init), condition=cond, body=body, frontend_attributes={_xla_disable_loop_instr_hoisting="true"}
  v_out = f32[4,16] get-tuple-element(w), index=3
  init2 = (s32[], f32[8,4,16], s32[], f32[4,16], s32[]) tuple(zero, buf2, n, v_out, slot)
  w2 = (s32[], f32[8,4,16], s32[], f32[4,16], s32[]) while(init2), condition=cond2, body=body2, frontend_attributes={_xla_disable_loop_instr_hoisting="true"}
  out = f32[8,4,16] get-tuple-element(w), index=1
  out2 = f32[8,4,16] get-tuple-element(w2), index=1
  ROOT r = (f32[8,4,16], f32[8,4,16]) tuple(out, out2)
}
)");
  // Also with the first loop last in the instruction list, as after a pass
  // rebuilt it, so the pass meets the reading loop first.
  for (bool producer_last : {false, true}) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    if (producer_last) {
      HloComputation* entry = module->entry_computation();
      HloInstruction* w = FindInstruction(module.get(), "w");
      HloInstruction* rebuilt = entry->AddInstruction(w->Clone());
      ASSERT_OK(w->ReplaceAllUsesWith(rebuilt));
      ASSERT_OK(entry->RemoveInstruction(w));
    }
    ASSERT_OK_AND_ASSIGN(
        bool changed,
        WhileLoopFusibleSinking(/*sink_broadcast_of_constant=*/true,
                                /*sink_dynamic_slices_into_updates=*/true)
            .Run(module.get()));
    ASSERT_TRUE(changed);
    for (absl::string_view body : {"body", "body2"}) {
      EXPECT_THAT(module->GetComputationWithName(body)
                      ->root_instruction()
                      ->operand(1)
                      ->operand(1),
                  op::Reshape(op::Reshape(op::DynamicSlice())))
          << body << " " << producer_last;
    }
    ASSERT_OK(WhileLoopSimplifier().Run(module.get()).status());
    ASSERT_OK(TupleSimplifier().Run(module.get()).status());
    ASSERT_OK(HloDCE().Run(module.get()).status());
    EXPECT_FALSE(absl::c_any_of(module->entry_computation()->instructions(),
                                HloPredicateIsOp<HloOpcode::kDynamicSlice>))
        << producer_last;
  }
}

// The slice stays outside when a loop that reads it through the first loop's
// output cannot take it: it computes on the value, or it uses the value only
// as an update but writes another value in its place.
TEST_F(WhileLoopFusibleSinkingTest, NoSinkDynamicSliceIfAReaderLoopCannot) {
  for (absl::string_view body2 : {
           R"(
  neg = f32[4,16] negate(v)
  ROOT t = (f32[4,16], f32[8,4,16]) tuple(neg, b))",
           R"(
  u = f32[1,4,16] reshape(v)
  zero = s32[] constant(0)
  dus = f32[8,4,16] dynamic-update-slice(b, u, zero, zero, zero)
  fzero = f32[] constant(0)
  other = f32[4,16] broadcast(fzero), dimensions={}
  ROOT t = (f32[4,16], f32[8,4,16]) tuple(other, dus))",
       }) {
    const std::string hlo = absl::StrReplaceAll(
        SliceUpdateLoop(
            kUpdateFromSlice,
            "v_out = f32[4,16] get-tuple-element(w), index=3\n"
            "  init2 = (f32[4,16], f32[8,4,16]) tuple(v_out, buf)\n"
            "  w2 = (f32[4,16], f32[8,4,16]) while(init2), condition=cond2, "
            "body=body2, "
            "frontend_attributes={_xla_disable_loop_instr_hoisting=\"true\"}\n"
            "  out = f32[8,4,16] get-tuple-element(w), index=1\n"
            "  out2 = f32[8,4,16] get-tuple-element(w2), index=1\n"
            "  ROOT r = (f32[8,4,16], f32[8,4,16]) tuple(out, out2)"),
        {{"HloModule SliceUpdateLoop\n",
          absl::StrCat(R"(HloModule SliceUpdateLoop

body2 {
  p = (f32[4,16], f32[8,4,16]) parameter(0)
  v = f32[4,16] get-tuple-element(p), index=0
  b = f32[8,4,16] get-tuple-element(p), index=1)",
                       body2, R"(
}

cond2 {
  p = (f32[4,16], f32[8,4,16]) parameter(0)
  ROOT f = pred[] constant(false)
}
)")}});
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    ASSERT_OK_AND_ASSIGN(
        bool changed,
        WhileLoopFusibleSinking(/*sink_broadcast_of_constant=*/true,
                                /*sink_dynamic_slices_into_updates=*/true)
            .Run(module.get()));
    EXPECT_FALSE(changed) << body2;
  }
}

// The slice stays outside when something besides the update needs it (code
// outside the loop reading the slice or its reshape, a computation in the
// body, code after the loop), when the loop allows hoisting, which would move
// the slice back out, or when sinking would make the loop carry more than the
// slice: a sliced operand that is computed (not yet materialized) or that the
// loop itself updates. The first case is the same loop where only the update
// reads the slice.
TEST_F(WhileLoopFusibleSinkingTest, SinkDynamicSliceOnlyIntoUpdates) {
  struct Case {
    absl::string_view body_update;
    absl::string_view entry_root;
    absl::string_view attrs;
    absl::string_view sliced;
    absl::string_view prelude;
    bool sinks;
  };
  for (const Case& c : {
           Case{kUpdateFromSlice, kReturnBuffer, kNoHoisting, "ring", "", true},
           Case{kUpdateFromSlice,
                "out = f32[8,4,16] get-tuple-element(w), index=1\n"
                "  ROOT r = (f32[8,4,16], f32[4,16]) tuple(out, v)",
                kNoHoisting, "ring", "", false},
           Case{kUpdateFromSlice,
                "out = f32[8,4,16] get-tuple-element(w), index=1\n"
                "  ROOT r = (f32[8,4,16], f32[1,4,16]) tuple(out, ds)",
                kNoHoisting, "ring", "", false},
           Case{"doubled = f32[4,16] add(v, v)\n"
                "  update = f32[1,4,16] reshape(doubled)",
                kReturnBuffer, kNoHoisting, "ring", "", false},
           Case{kUpdateFromSlice,
                "out = f32[8,4,16] get-tuple-element(w), index=1\n"
                "  last = f32[4,16] get-tuple-element(w), index=3\n"
                "  ROOT r = (f32[8,4,16], f32[4,16]) tuple(out, last)",
                kNoHoisting, "ring", "", false},
           Case{kUpdateFromSlice, kReturnBuffer, "", "ring", "", false},
           Case{kUpdateFromSlice, kReturnBuffer, kNoHoisting, "neg",
                "neg = f32[2,4,16] negate(ring)", false},
           Case{kUpdateFromSlice, kReturnBuffer, kNoHoisting, "buf", "", false},
       }) {
    ASSERT_OK_AND_ASSIGN(
        auto module,
        ParseAndReturnVerifiedModule(SliceUpdateLoop(
            c.body_update, c.entry_root, c.attrs, c.sliced, c.prelude)));
    ASSERT_OK_AND_ASSIGN(
        bool changed,
        WhileLoopFusibleSinking(/*sink_broadcast_of_constant=*/true,
                                /*sink_dynamic_slices_into_updates=*/true)
            .Run(module.get()));
    EXPECT_EQ(changed, c.sinks) << c.body_update << "\n"
                                << c.entry_root << "\n"
                                << c.attrs << "\n"
                                << c.prelude << " " << c.sliced;
  }
}

}  // namespace
}  // namespace xla
