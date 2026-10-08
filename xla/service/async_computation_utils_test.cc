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

#include "xla/service/async_computation_utils.h"

#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;

class AsyncComputationUtilsTest : public HloHardwareIndependentTestBase {
 protected:
  AsyncComputationUtilsTest()
      : HloHardwareIndependentTestBase(/*verifier_layout_sensitive=*/true) {}
};

// The input copy satisfies the custom call's layout constraint, but makes the
// body invalid as a single-instruction async computation.
constexpr char kInvalidBody[] = R"(
HloModule test
body {
  p = f32[2,3]{1,0} parameter(0)
  input_copy = f32[2,3]{0,1} copy(p)
  ROOT op = f32[2,3]{0,1} custom-call(input_copy), custom_call_target="test",
    operand_layout_constraints={f32[2,3]{0,1}}
}
ENTRY main {
  p0 = f32[2,3]{1,0} parameter(0)
  start = ((f32[2,3]{1,0}), f32[2,3]{0,1}, s32[]) async-start(p0), calls=body
  update = ((f32[2,3]{1,0}), f32[2,3]{0,1}, s32[]) async-update(start)
  ROOT done = f32[2,3]{0,1} async-done(update)
}
)";

TEST_F(AsyncComputationUtilsTest, WrapsInvalidBody) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(kInvalidBody));
  HloComputation* body = module->GetComputationWithName("body");
  const std::string original_body = body->ToString();

  // The extra copy makes the original async body fail verification.
  EXPECT_FALSE(verifier().Run(module.get()).ok());
  ASSERT_THAT(LegalizeAsyncComputations(module.get()), IsOkAndHolds(true));
  HloComputation* wrapper = module->entry_computation()
                                ->root_instruction()
                                ->async_wrapped_computation();

  // The async body is now a single call into the unchanged original body.
  ASSERT_EQ(wrapper->root_instruction()->opcode(), HloOpcode::kCall);
  EXPECT_EQ(wrapper->root_instruction()->to_apply(), body);
  EXPECT_TRUE(wrapper->CanExpandIntoSingleInstruction());
  EXPECT_EQ(body->ToString(), original_body);

  // The transformed module passes the verifier.
  EXPECT_OK(verifier().Run(module.get()).status());
}

TEST_F(AsyncComputationUtilsTest, PreservesInterface) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(kInvalidBody));
  HloComputation* body = module->GetComputationWithName("body");
  ASSERT_THAT(LegalizeAsyncComputations(module.get()), IsOkAndHolds(true));
  HloComputation* wrapper = module->entry_computation()
                                ->root_instruction()
                                ->async_wrapped_computation();

  // The wrapper preserves the input/output shapes, layouts, and thread.
  EXPECT_TRUE(ShapeUtil::Equal(wrapper->parameter_instruction(0)->shape(),
                               body->parameter_instruction(0)->shape()));
  EXPECT_TRUE(ShapeUtil::Equal(wrapper->root_instruction()->shape(),
                               body->root_instruction()->shape()));
  EXPECT_EQ(wrapper->execution_thread(), body->execution_thread());

  // The root call forwards the wrapper's parameter to the original body.
  EXPECT_THAT(wrapper->root_instruction()->operands(),
              ElementsAre(wrapper->parameter_instruction(0)));
  EXPECT_OK(verifier().Run(module.get()).status());
}

TEST_F(AsyncComputationUtilsTest, CopiesMetadataToWrapper) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(kInvalidBody));
  HloComputation* body = module->GetComputationWithName("body");
  body->parameter_instruction(0)->set_metadata_op_name("input");
  body->root_instruction()->set_metadata_op_name("output");

  ASSERT_THAT(LegalizeAsyncComputations(module.get()), IsOkAndHolds(true));
  HloComputation* wrapper = module->entry_computation()
                                ->root_instruction()
                                ->async_wrapped_computation();

  // Parameter and result metadata carry over to the wrapper.
  EXPECT_EQ(wrapper->parameter_instruction(0)->metadata().op_name(), "input");
  EXPECT_EQ(wrapper->root_instruction()->metadata().op_name(), "output");
}

TEST_F(AsyncComputationUtilsTest, RequiresSingleInstructionForAsyncBodyOnly) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(kInvalidBody));
  HloComputation* body = module->GetComputationWithName("body");

  // The restriction applies to the async body, not its ordinary caller.
  EXPECT_TRUE(AsyncComputationRequiresSingleInstruction(*body));
  EXPECT_FALSE(
      AsyncComputationRequiresSingleInstruction(*module->entry_computation()));

  ASSERT_THAT(LegalizeAsyncComputations(module.get()), IsOkAndHolds(true));

  // After wrapping, the original body is an ordinary callee and is exempt.
  EXPECT_FALSE(body->IsAsyncComputation());
  EXPECT_FALSE(AsyncComputationRequiresSingleInstruction(*body));
}

TEST_F(AsyncComputationUtilsTest, RedirectsImplicitUpdateAndDoneCallees) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(kInvalidBody));
  HloComputation* entry = module->entry_computation();
  HloComputation* body = module->GetComputationWithName("body");
  HloInstruction* update = entry->GetInstructionWithName("update");
  HloInstruction* done = entry->root_instruction();

  // Before legalization, both instructions refer to the original async body.
  EXPECT_EQ(update->async_wrapped_computation(), body);
  EXPECT_EQ(done->async_wrapped_computation(), body);

  ASSERT_THAT(LegalizeAsyncComputations(module.get()), IsOkAndHolds(true));
  HloComputation* wrapper =
      entry->GetInstructionWithName("start")->async_wrapped_computation();

  // Updating async-start also redirects the implicit update/done callees.
  ASSERT_NE(wrapper, body);
  EXPECT_EQ(update->async_wrapped_computation(), wrapper);
  EXPECT_EQ(done->async_wrapped_computation(), wrapper);
  EXPECT_OK(verifier().Run(module.get()).status());
}

TEST_F(AsyncComputationUtilsTest,
       RedirectsLoopCrossingExplicitUpdateAndDoneCallees) {
  constexpr char hlo[] = R"(
HloModule test
fn {
  p = f32[2,3]{0,1} parameter(0)
  input_copy = f32[2,3]{1,0} copy(p)
  ROOT op = f32[2,3]{0,1} custom-call(input_copy), custom_call_target="test",
    operand_layout_constraints={f32[2,3]{1,0}}
}
body {
  curr_state = (s32[], ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[])) parameter(0)
  i = s32[] get-tuple-element(curr_state), index=0
  one = s32[] constant(1)
  next_i = s32[] add(i, one)
  prev_start = ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[]) get-tuple-element(curr_state), index=1
  prev_update = ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[]) async-update(prev_start), calls=fn
  prev_done = f32[2,3]{0,1} async-done(prev_update)
  next_start = ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[]) async-start(prev_done), calls=fn
  ROOT next_state = (s32[], ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[])) tuple(next_i, next_start)
}
cond {
  param = (s32[], ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[])) parameter(0)
  i = s32[] get-tuple-element(param), index=0
  limit = s32[] constant(1)
  ROOT cond = pred[] compare(i, limit), direction=LT
}
ENTRY main {
  p = f32[2,3]{0,1} parameter(0)
  c0 = s32[] constant(0)
  first_start = ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[]) async-start(p), calls=fn
  init_state = (s32[], ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[])) tuple(c0, first_start)
  loop = (s32[], ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[])) while(init_state), condition=cond, body=body
  final_start = ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[]) get-tuple-element(loop), index=1
  ROOT final_done = f32[2,3]{0,1} async-done(final_start), calls=fn
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloComputation* entry = module->entry_computation();
  HloInstruction* first_start = entry->GetInstructionWithName("first_start");
  HloInstruction* final_done = entry->GetInstructionWithName("final_done");

  HloComputation* body = module->GetComputationWithName("body");
  HloInstruction* prev_update = body->GetInstructionWithName("prev_update");
  HloInstruction* prev_done = body->GetInstructionWithName("prev_done");
  HloInstruction* next_start = body->GetInstructionWithName("next_start");

  HloComputation* fn = module->GetComputationWithName("fn");
  EXPECT_FALSE(fn->CanExpandIntoSingleInstruction());

  // These consumers receive loop-carried state and store explicit callees.
  for (HloInstruction* instr : {prev_update, final_done}) {
    EXPECT_THAT(instr->called_computations(), ElementsAre(fn));
    EXPECT_EQ(instr->async_wrapped_computation(), fn);
  }

  // prev_done inherits its callee from prev_update.
  EXPECT_TRUE(prev_done->called_computations().empty());
  EXPECT_EQ(prev_done->async_wrapped_computation(), fn);
  EXPECT_EQ(first_start->async_wrapped_computation(), fn);
  EXPECT_EQ(next_start->async_wrapped_computation(), fn);

  // Legalize the HLO.
  EXPECT_THAT(LegalizeAsyncComputations(module.get()), IsOkAndHolds(true));

  // We insert a wrapper for fn.
  HloComputation* wrapper = first_start->async_wrapped_computation();
  EXPECT_NE(wrapper, fn);
  EXPECT_TRUE(wrapper->CanExpandIntoSingleInstruction());

  // next_start calls into wrapper.
  EXPECT_EQ(next_start->async_wrapped_computation(), wrapper);

  // Legalization must rewrite stored callees, not just the starts' references.
  for (HloInstruction* instr : {prev_update, final_done}) {
    EXPECT_THAT(instr->called_computations(), ElementsAre(wrapper));
    EXPECT_EQ(instr->async_wrapped_computation(), wrapper);
  }

  // prev_done continues to inherit its callee from prev_update.
  EXPECT_TRUE(prev_done->called_computations().empty());
  EXPECT_EQ(prev_done->async_wrapped_computation(), wrapper);

  EXPECT_OK(verifier().Run(module.get()).status());
}

TEST_F(AsyncComputationUtilsTest, SharesWrapperBetweenAsyncCallers) {
  constexpr char hlo[] = R"(
HloModule test
body {
  p = f32[2,3]{1,0} parameter(0)
  input_copy = f32[2,3]{0,1} copy(p)
  ROOT op = f32[2,3]{0,1} custom-call(input_copy), custom_call_target="test",
    operand_layout_constraints={f32[2,3]{0,1}}
}
ENTRY main {
  p0 = f32[2,3]{1,0} parameter(0)
  start0 = ((f32[2,3]{1,0}), f32[2,3]{0,1}, s32[]) async-start(p0), calls=body
  start1 = ((f32[2,3]{1,0}), f32[2,3]{0,1}, s32[]) async-start(p0), calls=body
  done0 = f32[2,3]{0,1} async-done(start0)
  done1 = f32[2,3]{0,1} async-done(start1)
  ROOT result = (f32[2,3]{0,1}, f32[2,3]{0,1}) tuple(done0, done1)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloComputation* entry = module->entry_computation();
  HloInstruction* first_start = entry->GetInstructionWithName("start0");
  HloInstruction* second_start = entry->GetInstructionWithName("start1");

  // Before legalization we have 'main' and 'body'.
  EXPECT_EQ(module->computation_count(), 2);

  ASSERT_THAT(LegalizeAsyncComputations(module.get()), IsOkAndHolds(true));

  // Legalization adds 'body.async_wrapper' shared between the two async-starts.
  EXPECT_EQ(module->computation_count(), 3);
  EXPECT_EQ(first_start->async_wrapped_computation(),
            second_start->async_wrapped_computation());
  EXPECT_OK(verifier().Run(module.get()).status());
}

TEST_F(AsyncComputationUtilsTest, LeavesOrdinaryCallsAlone) {
  constexpr char hlo[] = R"(
HloModule test
called_by_async {
  p = f32[2,3]{1,0} parameter(0)
  input_copy = f32[2,3]{0,1} copy(p)
  ROOT op = f32[2,3]{0,1} custom-call(input_copy), custom_call_target="test",
    operand_layout_constraints={f32[2,3]{0,1}}
}
called_by_sync {
  p = f32[2,3]{1,0} parameter(0)
  input_copy = f32[2,3]{0,1} copy(p)
  ROOT op = f32[2,3]{0,1} custom-call(input_copy), custom_call_target="test",
    operand_layout_constraints={f32[2,3]{0,1}}
}
ENTRY main {
  p0 = f32[2,3]{1,0} parameter(0)
  start = ((f32[2,3]{1,0}), f32[2,3]{0,1}, s32[]) async-start(p0), calls=called_by_async
  done = f32[2,3]{0,1} async-done(start)
  ordinary_call = f32[2,3]{0,1} call(p0), to_apply=called_by_sync
  ROOT result = (f32[2,3]{0,1}, f32[2,3]{0,1}) tuple(done, ordinary_call)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloComputation* called_by_sync =
      module->GetComputationWithName("called_by_sync");
  const std::string original_called_by_sync = called_by_sync->ToString();
  HloComputation* entry = module->entry_computation();
  HloInstruction* start = entry->GetInstructionWithName("start");
  HloInstruction* call = entry->GetInstructionWithName("ordinary_call");
  ASSERT_THAT(LegalizeAsyncComputations(module.get()), IsOkAndHolds(true));

  // The ordinary call and its multi-instruction body remain unchanged.
  EXPECT_EQ(call->to_apply(), called_by_sync);
  EXPECT_NE(call->to_apply(), start->async_wrapped_computation());
  EXPECT_EQ(called_by_sync->ToString(), original_called_by_sync);
  EXPECT_OK(verifier().Run(module.get()).status());
}

TEST_F(AsyncComputationUtilsTest, PreservesTupleAliasesAndSideEffects) {
  constexpr char hlo[] = R"(
HloModule test
body {
  p = (f32[2,3]{1,0}) parameter(0)
  effect = () custom-call(p), custom_call_target="effect", custom_call_has_side_effect=true
  a = f32[2,3]{1,0} get-tuple-element(p), index=0
  ROOT result = (f32[2,3]{1,0}) tuple(a), control-predecessors={effect}
}
ENTRY main {
  p0 = (f32[2,3]{1,0}) parameter(0)
  start = (((f32[2,3]{1,0})), (f32[2,3]{1,0}), s32[]) async-start(p0), calls=body, output_to_operand_aliasing={{1,0}: (0, {0})}
  ROOT done = (f32[2,3]{1,0}) async-done(start)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloComputation* body = module->GetComputationWithName("body");
  HloInstruction* start =
      module->entry_computation()->GetInstructionWithName("start");
  const auto aliases = start->output_operand_aliasing();

  ASSERT_THAT(LegalizeAsyncComputations(module.get()), IsOkAndHolds(true));

  // The tuple result retains its explicit alias to the input tuple element.
  EXPECT_EQ(start->output_operand_aliasing(), aliases);

  // The call remains side-effecting and the body's control dependency survives.
  EXPECT_TRUE(start->async_wrapped_instruction()->HasSideEffect());
  EXPECT_THAT(body->root_instruction()->control_predecessors(),
              ElementsAre(body->GetInstructionWithName("effect")));

  // Tuple interfaces and aliases remain valid across the new call boundary.
  EXPECT_OK(verifier().Run(module.get()).status());
}

TEST_F(AsyncComputationUtilsTest, LeavesSingleInstructionBodyAlone) {
  constexpr char hlo[] = R"(
HloModule test
body {
  p = f32[2,3]{1,0} parameter(0)
  ROOT neg = f32[2,3]{1,0} negate(p)
}
ENTRY main {
  p0 = f32[2,3]{1,0} parameter(0)
  start = ((f32[2,3]{1,0}), f32[2,3]{1,0}, s32[]) async-start(p0), calls=body
  ROOT done = f32[2,3]{1,0} async-done(start)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  const std::string before = module->ToString();

  // A body containing only its parameter and root needs no wrapper.
  EXPECT_THAT(LegalizeAsyncComputations(module.get()), IsOkAndHolds(false));
  EXPECT_EQ(module->ToString(), before);
}

TEST_F(AsyncComputationUtilsTest, LeavesOtherExecutionThreadsAlone) {
  constexpr char hlo[] = R"(
HloModule test
body {
  p = f32[2,3]{1,0} parameter(0)
  input_copy = f32[2,3]{0,1} copy(p)
  ROOT op = f32[2,3]{0,1} custom-call(input_copy), custom_call_target="test",
    operand_layout_constraints={f32[2,3]{0,1}}
}
ENTRY main {
  p0 = f32[2,3]{1,0} parameter(0)
  start = ((f32[2,3]{1,0}), f32[2,3]{0,1}, s32[]) async-start(p0), calls=body, async_execution_thread="worker"
  ROOT done = f32[2,3]{0,1} async-done(start)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));

  // Multi-instruction async bodies are permitted on other threads.
  EXPECT_FALSE(AsyncComputationRequiresSingleInstruction(
      *module->GetComputationWithName("body")));
  const std::string before = module->ToString();

  // Legalization leaves this already-permitted body unchanged.
  EXPECT_THAT(LegalizeAsyncComputations(module.get()), IsOkAndHolds(false));
  EXPECT_EQ(module->ToString(), before);
  EXPECT_OK(verifier().Run(module.get()).status());
}

TEST_F(AsyncComputationUtilsTest, RespectsExecutionThreadFilter) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(kInvalidBody));
  const std::string before = module->ToString();

  // Excluding the body's thread leaves the module unchanged.
  EXPECT_THAT(LegalizeAsyncComputations(module.get(), {"worker"}),
              IsOkAndHolds(false));
  EXPECT_EQ(module->ToString(), before);

  // Including the main thread allows the body to be legalized.
  EXPECT_THAT(LegalizeAsyncComputations(module.get(),
                                        {HloInstruction::kMainExecutionThread}),
              IsOkAndHolds(true));
  EXPECT_OK(verifier().Run(module.get()).status());
}

TEST_F(AsyncComputationUtilsTest, LeavesCollectivesGroupAlone) {
  constexpr char hlo[] = R"(
HloModule test
body {
  p = f32[2,3]{1,0} parameter(0)
  input_copy = f32[2,3]{0,1} copy(p)
  ROOT op = f32[2,3]{0,1} custom-call(input_copy), custom_call_target="test",
    operand_layout_constraints={f32[2,3]{0,1}}
}
ENTRY main {
  p0 = f32[2,3]{1,0} parameter(0)
  start = ((f32[2,3]{1,0}), f32[2,3]{0,1}, s32[]) async-start(p0), calls=body, frontend_attributes={_collectives_group=""}
  ROOT done = f32[2,3]{0,1} async-done(start)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));

  // The collective-group marker exempts this body from the restriction.
  EXPECT_FALSE(AsyncComputationRequiresSingleInstruction(
      *module->GetComputationWithName("body")));
  const std::string before = module->ToString();

  // Legalization leaves this already-permitted body unchanged.
  EXPECT_THAT(LegalizeAsyncComputations(module.get()), IsOkAndHolds(false));
  EXPECT_EQ(module->ToString(), before);
  EXPECT_OK(verifier().Run(module.get()).status());
}

TEST_F(AsyncComputationUtilsTest,
       LeavesCollectivesGroupWithMultipleCallersAlone) {
  constexpr char hlo[] = R"(
HloModule test
body {
  p = f32[2,3]{1,0} parameter(0)
  input_copy = f32[2,3]{0,1} copy(p)
  ROOT op = f32[2,3]{0,1} custom-call(input_copy), custom_call_target="test",
    operand_layout_constraints={f32[2,3]{0,1}}
}
ENTRY main {
  p0 = f32[2,3]{1,0} parameter(0)
  start_1 = ((f32[2,3]{1,0}), f32[2,3]{0,1}, s32[]) async-start(p0), calls=body, frontend_attributes={_collectives_group=""}
  start_2 = ((f32[2,3]{1,0}), f32[2,3]{0,1}, s32[]) async-start(p0), calls=body, frontend_attributes={_collectives_group=""}
  done_1 = f32[2,3]{0,1} async-done(start_1)
  done_2 = f32[2,3]{0,1} async-done(start_2)
  ROOT out = (f32[2,3]{0,1},f32[2,3]{0,1}) tuple(done_1, done_2)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));

  // The collective-group marker exempts this body from the restriction.
  EXPECT_FALSE(AsyncComputationRequiresSingleInstruction(
      *module->GetComputationWithName("body")));
  const std::string before = module->ToString();

  // Legalization leaves this already-permitted body unchanged.
  EXPECT_THAT(LegalizeAsyncComputations(module.get()), IsOkAndHolds(false));
  EXPECT_EQ(module->ToString(), before);
  EXPECT_OK(verifier().Run(module.get()).status());
}

TEST_F(AsyncComputationUtilsTest,
       LeavesComputationWithOneSyncCallerOneCollectivesGroupAsyncCallerAlone) {
  constexpr char hlo[] = R"(
HloModule test
body {
  p = f32[2,3]{1,0} parameter(0)
  input_copy = f32[2,3]{0,1} copy(p)
  ROOT op = f32[2,3]{0,1} custom-call(input_copy), custom_call_target="test",
    operand_layout_constraints={f32[2,3]{0,1}}
}
ENTRY main {
  p0 = f32[2,3]{1,0} parameter(0)
  start = ((f32[2,3]{1,0}), f32[2,3]{0,1}, s32[]) async-start(p0), calls=body, frontend_attributes={_collectives_group=""}
  ROOT done = f32[2,3]{0,1} async-done(start)
}
)";

  // The parser rejects HLO where an async computation has both async and sync
  // callers. Insert a sync caller after parsing.
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloComputation* entry = module->entry_computation();
  HloComputation* body = module->GetComputationWithName("body");
  entry->AddInstruction(
      HloInstruction::CreateCall(body->root_instruction()->shape(),
                                 {entry->parameter_instruction(0)}, body),
      "val");

  // There is no caller which is async and does not have a collectives group
  // annotation, so `body` does not need to be outlined.
  EXPECT_FALSE(AsyncComputationRequiresSingleInstruction(
      *module->GetComputationWithName("body")));
  const std::string before = module->ToString();

  // Legalization leaves this already-permitted body unchanged.
  EXPECT_THAT(LegalizeAsyncComputations(module.get()), IsOkAndHolds(false));
  EXPECT_EQ(module->ToString(), before);
  EXPECT_OK(verifier().Run(module.get()).status());
}

TEST_F(AsyncComputationUtilsTest,
       OutlinesComputationWithOneAsyncCallerAndOneAsyncCollectivesGroupCaller) {
  constexpr char hlo[] = R"(
HloModule test
body {
  p = f32[2,3]{1,0} parameter(0)
  input_copy = f32[2,3]{0,1} copy(p)
  ROOT op = f32[2,3]{0,1} custom-call(input_copy), custom_call_target="test",
    operand_layout_constraints={f32[2,3]{0,1}}
}
ENTRY main {
  p0 = f32[2,3]{1,0} parameter(0)
  start_1 = ((f32[2,3]{1,0}), f32[2,3]{0,1}, s32[]) async-start(p0), calls=body
  start_2 = ((f32[2,3]{1,0}), f32[2,3]{0,1}, s32[]) async-start(p0), calls=body, frontend_attributes={_collectives_group=""}
  done_1 = f32[2,3]{0,1} async-done(start_1)
  done_2 = f32[2,3]{0,1} async-done(start_2)
  ROOT out = (f32[2,3]{0,1},f32[2,3]{0,1}) tuple(done_1, done_2)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start_1 =
      module->entry_computation()->GetInstructionWithName("start_1");
  HloInstruction* start_2 =
      module->entry_computation()->GetInstructionWithName("start_2");
  HloComputation* body = module->GetComputationWithName("body");

  ASSERT_EQ(start_1->called_computations().size(), 1);
  ASSERT_EQ(start_1->called_computations()[0], body);
  ASSERT_EQ(start_2->called_computations().size(), 1);
  ASSERT_EQ(start_2->called_computations()[0], body);

  // start_1 has no collectives group annotation so outlining is required.
  EXPECT_TRUE(AsyncComputationRequiresSingleInstruction(
      *module->GetComputationWithName("body")));

  // Legalization makes start_2 call a wrapper but leaves start_2 alone.
  EXPECT_THAT(LegalizeAsyncComputations(module.get()), IsOkAndHolds(true));
  EXPECT_NE(start_1->called_computations()[0], body);
  EXPECT_TRUE(
      start_1->called_computations()[0]->CanExpandIntoSingleInstruction());
  EXPECT_EQ(start_2->called_computations()[0], body);

  EXPECT_OK(verifier().Run(module.get()).status());
}

TEST_F(AsyncComputationUtilsTest,
       ResolvesCollectiveGroupAnnotationViaAsyncStartForLoopCarriedFuture) {
  constexpr char hlo[] = R"(
HloModule test
fn {
  p = f32[2,3]{0,1} parameter(0)
  input_copy = f32[2,3]{1,0} copy(p)
  ROOT op = f32[2,3]{0,1} custom-call(input_copy), custom_call_target="test",
    operand_layout_constraints={f32[2,3]{1,0}}
}
body {
  curr_state = (s32[], ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[])) parameter(0)
  i = s32[] get-tuple-element(curr_state), index=0
  one = s32[] constant(1)
  next_i = s32[] add(i, one)
  prev_start = ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[]) get-tuple-element(curr_state), index=1
  prev_update = ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[]) async-update(prev_start), calls=fn
  prev_done = f32[2,3]{0,1} async-done(prev_update)
  next_start = ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[]) async-start(prev_done), calls=fn, frontend_attributes={_collectives_group=""}
  ROOT next_state = (s32[], ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[])) tuple(next_i, next_start)
}
cond {
  param = (s32[], ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[])) parameter(0)
  i = s32[] get-tuple-element(param), index=0
  limit = s32[] constant(1)
  ROOT cond = pred[] compare(i, limit), direction=LT
}
ENTRY main {
  p = f32[2,3]{0,1} parameter(0)
  c0 = s32[] constant(0)
  first_start = ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[]) async-start(p), calls=fn, frontend_attributes={_collectives_group=""}
  init_state = (s32[], ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[])) tuple(c0, first_start)
  loop = (s32[], ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[])) while(init_state), condition=cond, body=body
  final_start = ((f32[2,3]{0,1}), f32[2,3]{0,1}, s32[]) get-tuple-element(loop), index=1
  ROOT final_done = f32[2,3]{0,1} async-done(final_start), calls=fn
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));

  HloComputation* fn = module->GetComputationWithName("fn");
  ASSERT_FALSE(fn->CanExpandIntoSingleInstruction());

  // HLO is already legal since all async-starts have collective group
  // annotations.
  EXPECT_FALSE(AsyncComputationRequiresSingleInstruction(*fn));
  const std::string before = module->ToString();

  // No caller needs a wrapper, including the explicit update/done callers.
  EXPECT_THAT(LegalizeAsyncComputations(module.get()), IsOkAndHolds(false));
  EXPECT_EQ(module->ToString(), before);

  EXPECT_OK(verifier().Run(module.get()).status());
}

TEST_F(AsyncComputationUtilsTest, LeavesAsyncBarrierAlone) {
  constexpr char hlo[] = R"(
HloModule test
body {
  p = (f32[2,3]{1,0}) parameter(0)
  a = f32[2,3]{1,0} get-tuple-element(p), index=0
  ROOT result = (f32[2,3]{1,0}) tuple(a)
}
ENTRY main {
  p0 = (f32[2,3]{1,0}) parameter(0)
  start = (((f32[2,3]{1,0})), (f32[2,3]{1,0}), s32[]) async-start(p0), calls=body
  ROOT done = (f32[2,3]{1,0}) async-done(start)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  // The extra GTE is allowed by the async-barrier exemption.
  EXPECT_FALSE(
      module->GetComputationWithName("body")->CanExpandIntoSingleInstruction());
  EXPECT_FALSE(AsyncComputationRequiresSingleInstruction(
      *module->GetComputationWithName("body")));
  const std::string before = module->ToString();

  // Legalization preserves this permitted multi-instruction body.
  EXPECT_THAT(LegalizeAsyncComputations(module.get()), IsOkAndHolds(false));
  EXPECT_EQ(module->ToString(), before);
  EXPECT_OK(verifier().Run(module.get()).status());
}

}  // namespace
}  // namespace xla
