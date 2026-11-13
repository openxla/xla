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

#include "xla/hlo/transforms/simplifiers/conditional_residual_elision.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/literal_test_util.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;

// A conditional as JAX partially evaluates it: `known` computes the residuals
// of both branches and `staged` consumes those of its branch. The true branch
// saves its operand, the false branch a value that it computes.
constexpr absl::string_view kBranches = R"hlo(
HloModule m

known_true {
  p = (f32[2]) parameter(0)
  x = f32[2] get-tuple-element(p), index=0
  out = f32[2] negate(x)
  zero = f32[] constant(0)
  zeros = f32[2] broadcast(zero), dimensions={}
  ROOT t = (f32[2], f32[2], f32[2]) tuple(out, x, zeros)
}

known_false {
  p = (f32[2]) parameter(0)
  x = f32[2] get-tuple-element(p), index=0
  res = f32[2] exponential(x)
  out = f32[2] multiply(res, res)
  zero = f32[] constant(0)
  zeros = f32[2] broadcast(zero), dimensions={}
  ROOT t = (f32[2], f32[2], f32[2]) tuple(out, zeros, res)
}

staged_true {
  p = (f32[2], f32[2]) parameter(0)
  res = f32[2] get-tuple-element(p), index=0
  g = f32[2] get-tuple-element(p), index=1
  ROOT d = f32[2] multiply(res, g)
}

staged_false {
  p = (f32[2], f32[2]) parameter(0)
  res = f32[2] get-tuple-element(p), index=0
  g = f32[2] get-tuple-element(p), index=1
  ROOT d = f32[2] subtract(res, g)
}
)hlo";

// The start of an entry computation that runs `known`.
constexpr absl::string_view kKnown = R"hlo(
ENTRY entry {
  sel = pred[] parameter(0)
  x = f32[2] parameter(1)
  in = (f32[2]) tuple(x)
  known = (f32[2], f32[2], f32[2]) conditional(sel, in, in),
      true_computation=known_true, false_computation=known_false,
      origin={({"known" {0}}, {"known" {1}}, {"known" {2}})}
  out = f32[2] get-tuple-element(known), index=0
  res_true = f32[2] get-tuple-element(known), index=1
  res_false = f32[2] get-tuple-element(known), index=2
)hlo";

class ConditionalResidualElisionTest : public HloHardwareIndependentTestBase {
 protected:
  // Evaluates `module` with `fill` in every uninitialized buffer. Counts the
  // custom calls with a side effect that it runs, which return their operand.
  absl::StatusOr<Literal> Evaluate(const HloModule& module,
                                   absl::Span<const Literal> arguments,
                                   float fill) {
    HloEvaluator evaluator;
    evaluator.set_custom_call_handler(
        [this, fill](
            const HloInstruction* call,
            absl::Span<const Literal*> operands) -> absl::StatusOr<Literal> {
          if (call->HasSideEffect()) {
            ++side_effects_;
            return operands[0]->Clone();
          }
          if (!call->IsCustomCall("AllocateBuffer")) {
            return absl::UnimplementedError(call->ToString());
          }
          ABSL_ASSIGN_OR_RETURN(Literal buffer, Literal::Make(call->shape()));
          buffer.PopulateWithValue(fill);
          return buffer;
        });
    return evaluator.Evaluate(module, arguments);
  }

  // Runs the pass and HloDCE on `hlo`. Checks that the pass changes the module
  // only if `expect_change`, and that for each of `arguments` the module
  // returns the same bits and runs as many side effects, whatever the
  // uninitialized buffers hold.
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> RunAndCompare(
      absl::string_view hlo, const std::vector<std::vector<Literal>>& arguments,
      bool expect_change) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
    std::unique_ptr<HloModule> reference = module->Clone();
    ABSL_ASSIGN_OR_RETURN(
        bool changed, RunHloPass(ConditionalResidualElision(), module.get()));
    EXPECT_EQ(changed, expect_change);
    ABSL_RETURN_IF_ERROR(module->Verify());
    ABSL_RETURN_IF_ERROR(RunHloPass(HloDCE(), module.get()).status());
    for (const std::vector<Literal>& args : arguments) {
      side_effects_ = 0;
      ABSL_ASSIGN_OR_RETURN(Literal expected, Evaluate(*reference, args, 0));
      const int64_t expected_side_effects = side_effects_;
      side_effects_ = 0;
      ABSL_ASSIGN_OR_RETURN(Literal actual, Evaluate(*module, args, 7));
      EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
      EXPECT_EQ(side_effects_, expected_side_effects);
    }
    return module;
  }

  // Arguments (selector, x) for each of `selectors`.
  template <typename T>
  static std::vector<std::vector<Literal>> Arguments(
      const std::vector<T>& selectors) {
    std::vector<std::vector<Literal>> arguments;
    for (T selector : selectors) {
      std::vector<Literal>& args = arguments.emplace_back();
      args.push_back(LiteralUtil::CreateR0<T>(selector));
      args.push_back(LiteralUtil::CreateR1<float>({1.5f, -2.0f}));
    }
    return arguments;
  }

  static int64_t Count(const HloModule& module, HloOpcode opcode) {
    int64_t count = 0;
    for (const HloComputation* computation : module.computations()) {
      count += absl::c_count_if(computation->instructions(),
                                [&](const HloInstruction* instruction) {
                                  return instruction->opcode() == opcode;
                                });
    }
    return count;
  }

  int64_t side_effects_ = 0;
};

TEST_F(ConditionalResidualElisionTest, ForwardsAndElidesResiduals) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      RunAndCompare(absl::StrCat(kBranches, kKnown, R"hlo(
  args_true = (f32[2], f32[2]) tuple(res_true, x)
  args_false = (f32[2], f32[2]) tuple(res_false, out)
  ROOT staged = f32[2] conditional(sel, args_true, args_false),
      true_computation=staged_true, false_computation=staged_false
}
)hlo"),
                    Arguments<bool>({true, false}), /*expect_change=*/true));
  // `staged` reads the saved operand itself, and `known` no longer returns it.
  const HloInstruction* staged =
      module->entry_computation()->root_instruction();
  EXPECT_EQ(staged->operand(1)->operand(0), staged->operand(1)->operand(1));
  const HloInstruction* known = FindInstruction(module.get(), "known");
  EXPECT_EQ(known->shape().tuple_shapes().size(), 2);
  EXPECT_EQ(known->original_value()->ToString(),
            R"(({"known" {0}}, {"known" {2}}))");
  // The true branch computes none of the two outputs that only the false
  // branch reads.
  EXPECT_EQ(Count(*module, HloOpcode::kNegate), 0);
  EXPECT_EQ(Count(*module, HloOpcode::kBroadcast), 0);
  EXPECT_EQ(Count(*module, HloOpcode::kCustomCall), 2);
  EXPECT_THAT(RunHloPass(ConditionalResidualElision(), module.get()),
              IsOkAndHolds(false));
}

// The true branch of `known` returns a bitcast of its operand through an
// optimization barrier, and `staged` reads a bitcast of the residual.
TEST_F(ConditionalResidualElisionTest, ForwardsThroughBitcastsAndBarriers) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       RunAndCompare(R"hlo(
HloModule m

known_true {
  p = (f32[2]) parameter(0)
  x = f32[2] get-tuple-element(p), index=0
  b = f32[2,1] bitcast(x)
  e = f32[2,1] exponential(b)
  saved = (f32[2,1], f32[2,1]) tuple(e, b)
  barrier = (f32[2,1], f32[2,1]) opt-barrier(saved)
  res = f32[2,1] get-tuple-element(barrier), index=1
  ROOT t = (f32[2,1]) tuple(res)
}

known_false {
  p = (f32[2]) parameter(0)
  x = f32[2] get-tuple-element(p), index=0
  b = f32[2,1] bitcast(x)
  e = f32[2,1] exponential(b)
  ROOT t = (f32[2,1]) tuple(e)
}

staged_true {
  p = (f32[1,2]) parameter(0)
  r = f32[1,2] get-tuple-element(p), index=0
  ROOT n = f32[1,2] negate(r)
}

staged_false {
  p = (f32[1,2]) parameter(0)
  r = f32[1,2] get-tuple-element(p), index=0
  ROOT a = f32[1,2] abs(r)
}

ENTRY entry {
  sel = pred[] parameter(0)
  x = f32[2] parameter(1)
  in = (f32[2]) tuple(x)
  guarded = (f32[2]) opt-barrier(in)
  known = (f32[2,1]) conditional(sel, guarded, guarded),
      true_computation=known_true, false_computation=known_false
  res = f32[2,1] get-tuple-element(known), index=0
  b = f32[1,2] bitcast(res)
  args_true = (f32[1,2]) tuple(b)
  args_false = (f32[1,2]) tuple(b)
  ROOT staged = f32[1,2] conditional(sel, args_true, args_false),
      true_computation=staged_true, false_computation=staged_false
}
)hlo",
                                     Arguments<bool>({true, false}),
                                     /*expect_change=*/true));
  // The true branch of `staged` reads both bitcasts of the operand of
  // `known`, and the true branch of `known` no longer returns it.
  const HloInstruction* forwarded =
      module->entry_computation()->root_instruction()->operand(1)->operand(0);
  ASSERT_EQ(forwarded->opcode(), HloOpcode::kBitcast);
  ASSERT_EQ(forwarded->operand(0)->opcode(), HloOpcode::kBitcast);
  EXPECT_EQ(forwarded->operand(0)->operand(0)->operand(0),
            FindInstruction(module.get(), "guarded"));
  EXPECT_EQ(Count(*module, HloOpcode::kCustomCall), 1);
}

TEST_F(ConditionalResidualElisionTest, IndexedConditionals) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       RunAndCompare(R"hlo(
HloModule m

known_0 {
  p = (f32[2]) parameter(0)
  x = f32[2] get-tuple-element(p), index=0
  zero = f32[] constant(0)
  zeros = f32[2] broadcast(zero), dimensions={}
  ROOT t = (f32[2], f32[2]) tuple(x, zeros)
}

known_1 {
  p = (f32[2]) parameter(0)
  x = f32[2] get-tuple-element(p), index=0
  e = f32[2] exponential(x)
  ROOT t = (f32[2], f32[2]) tuple(x, e)
}

known_2 {
  p = (f32[2]) parameter(0)
  zero = f32[] constant(0)
  zeros = f32[2] broadcast(zero), dimensions={}
  ROOT t = (f32[2], f32[2]) tuple(zeros, zeros)
}

staged_0 {
  p = (f32[2]) parameter(0)
  ROOT r = f32[2] get-tuple-element(p), index=0
}

staged_1 {
  p = (f32[2], f32[2]) parameter(0)
  r0 = f32[2] get-tuple-element(p), index=0
  r1 = f32[2] get-tuple-element(p), index=1
  ROOT s = f32[2] subtract(r0, r1)
}

staged_2 {
  p = (f32[2]) parameter(0)
  r = f32[2] get-tuple-element(p), index=0
  ROOT a = f32[2] abs(r)
}

ENTRY entry {
  i = s32[] parameter(0)
  x_0 = f32[2] parameter(1)
  x_1 = f32[2] negate(x_0)
  in_0 = (f32[2]) tuple(x_0)
  in_1 = (f32[2]) tuple(x_1)
  known = (f32[2], f32[2]) conditional(i, in_0, in_1, in_1),
      branch_computations={known_0, known_1, known_2}
  r0 = f32[2] get-tuple-element(known), index=0
  r1 = f32[2] get-tuple-element(known), index=1
  a0 = (f32[2]) tuple(r0)
  a1 = (f32[2], f32[2]) tuple(r0, r1)
  a2 = (f32[2]) tuple(r0)
  ROOT staged = f32[2] conditional(i, a0, a1, a2),
      branch_computations={staged_0, staged_1, staged_2}
}
)hlo",
                                     // Out of range indices take the last
                                     // branch.
                                     Arguments<int32_t>({-1, 0, 1, 2, 5}),
                                     /*expect_change=*/true));
  // Branches 0 and 1 read x_0 and x_1 themselves. Then only branch 2 reads r0
  // and only branch 1 reads r1: the other two branches of `known` leave each
  // uninitialized.
  const HloInstruction* staged =
      module->entry_computation()->root_instruction();
  EXPECT_EQ(staged->operand(1)->operand(0),
            FindInstruction(module.get(), "x_0"));
  EXPECT_EQ(staged->operand(2)->operand(0),
            FindInstruction(module.get(), "x_1"));
  EXPECT_EQ(Count(*module, HloOpcode::kCustomCall), 4);
}

// The conditionals of a loop body take another branch in each iteration.
TEST_F(ConditionalResidualElisionTest, ConditionalsInLoopBodies) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      RunAndCompare(absl::StrCat(kBranches, R"hlo(
body {
  state = (s32[], f32[2]) parameter(0)
  i = s32[] get-tuple-element(state), index=0
  x = f32[2] get-tuple-element(state), index=1
  two = s32[] constant(2)
  parity = s32[] remainder(i, two)
  zero = s32[] constant(0)
  sel = pred[] compare(parity, zero), direction=EQ
  in = (f32[2]) tuple(x)
  known = (f32[2], f32[2], f32[2]) conditional(sel, in, in),
      true_computation=known_true, false_computation=known_false
  out = f32[2] get-tuple-element(known), index=0
  res_true = f32[2] get-tuple-element(known), index=1
  res_false = f32[2] get-tuple-element(known), index=2
  args_true = (f32[2], f32[2]) tuple(res_true, x)
  args_false = (f32[2], f32[2]) tuple(res_false, out)
  staged = f32[2] conditional(sel, args_true, args_false),
      true_computation=staged_true, false_computation=staged_false
  one = s32[] constant(1)
  next = s32[] add(i, one)
  ROOT t = (s32[], f32[2]) tuple(next, staged)
}

condition {
  state = (s32[], f32[2]) parameter(0)
  i = s32[] get-tuple-element(state), index=0
  end = s32[] constant(3)
  ROOT lt = pred[] compare(i, end), direction=LT
}

ENTRY entry {
  start = s32[] parameter(0)
  x = f32[2] parameter(1)
  init = (s32[], f32[2]) tuple(start, x)
  loop = (s32[], f32[2]) while(init), condition=condition, body=body
  ROOT result = f32[2] get-tuple-element(loop), index=1
}
)hlo"),
                    Arguments<int32_t>({0, 1}), /*expect_change=*/true));
  EXPECT_EQ(
      FindInstruction(module.get(), "known")->shape().tuple_shapes().size(), 2);
  EXPECT_EQ(Count(*module, HloOpcode::kCustomCall), 2);
}

// Once `staged` reads the operand of `known`, nothing reads `known`. Its true
// branch still has a side effect, which reads the old result of the branch.
TEST_F(ConditionalResidualElisionTest, KeepsSideEffectsOfUnreadProducers) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       RunAndCompare(R"hlo(
HloModule m

known_true {
  p = (f32[2]) parameter(0)
  x = f32[2] get-tuple-element(p), index=0
  ROOT t = (f32[2]) tuple(x)
  again = f32[2] get-tuple-element(t), index=0
  log = f32[2] custom-call(again), custom_call_target="Log",
      custom_call_has_side_effect=true
}

known_false {
  p = (f32[2]) parameter(0)
  zero = f32[] constant(0)
  zeros = f32[2] broadcast(zero), dimensions={}
  ROOT t = (f32[2]) tuple(zeros)
}

staged_true {
  p = (f32[2]) parameter(0)
  r = f32[2] get-tuple-element(p), index=0
  ROOT n = f32[2] negate(r)
}

staged_false {
  p = (f32[2]) parameter(0)
  r = f32[2] get-tuple-element(p), index=0
  ROOT a = f32[2] abs(r)
}

ENTRY entry {
  sel = pred[] parameter(0)
  x = f32[2] parameter(1)
  in = (f32[2]) tuple(x)
  known = (f32[2]) conditional(sel, in, in),
      true_computation=known_true, false_computation=known_false,
      origin={({"known" {0}})}
  res = f32[2] get-tuple-element(known), index=0
  args_true = (f32[2]) tuple(res)
  ROOT staged = f32[2] conditional(sel, args_true, in),
      true_computation=staged_true, false_computation=staged_false
}
)hlo",
                                     Arguments<bool>({true, false}),
                                     /*expect_change=*/true));
  EXPECT_EQ(
      FindInstruction(module.get(), "known")->shape().tuple_shapes().size(), 0);
}

// The origin of a call that the compiler made up has no elements to renumber
// when an output of `known` goes away.
TEST_F(ConditionalResidualElisionTest, KeepsOriginsOfSyntheticCalls) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      RunAndCompare(absl::StrCat(kBranches, R"hlo(
ENTRY entry {
  sel = pred[] parameter(0)
  x = f32[2] parameter(1)
  in = (f32[2]) tuple(x)
  known = (f32[2], f32[2], f32[2]) conditional(sel, in, in),
      true_computation=known_true, false_computation=known_false,
      origin={[synthetic_call]}
  out = f32[2] get-tuple-element(known), index=0
  res_true = f32[2] get-tuple-element(known), index=1
  res_false = f32[2] get-tuple-element(known), index=2
  args_true = (f32[2], f32[2]) tuple(res_true, x)
  args_false = (f32[2], f32[2]) tuple(res_false, out)
  ROOT staged = f32[2] conditional(sel, args_true, args_false),
      true_computation=staged_true, false_computation=staged_false
}
)hlo"),
                    Arguments<bool>({true, false}), /*expect_change=*/true));
  const HloInstruction* known = FindInstruction(module.get(), "known");
  EXPECT_EQ(known->shape().tuple_shapes().size(), 2);
  EXPECT_TRUE(known->original_value()->is_synthetic_call());
}

// `crossed` and `crossed_2` read the residuals of `known` like `staged`, but
// another instruction selects their branch.
TEST_F(ConditionalResidualElisionTest, KeepsResidualsOfOtherSelectors) {
  std::vector<std::vector<Literal>> arguments;
  for (bool sel : {true, false}) {
    for (bool other : {true, false}) {
      std::vector<Literal>& args = arguments.emplace_back();
      args.push_back(LiteralUtil::CreateR0<bool>(sel));
      args.push_back(LiteralUtil::CreateR1<float>({1.5f, -2.0f}));
      args.push_back(LiteralUtil::CreateR0<bool>(other));
    }
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       RunAndCompare(absl::StrCat(kBranches, kKnown, R"hlo(
  other = pred[] parameter(2)
  args_true = (f32[2], f32[2]) tuple(res_true, out)
  args_false = (f32[2], f32[2]) tuple(res_false, out)
  staged = f32[2] conditional(sel, args_true, args_false),
      true_computation=staged_true, false_computation=staged_false
  crossed = f32[2] conditional(other, args_true, args_false),
      true_computation=staged_true, false_computation=staged_false
  crossed_2 = f32[2] conditional(other, args_true, args_false),
      true_computation=staged_true, false_computation=staged_false
  sum = f32[2] add(staged, crossed)
  ROOT sum_2 = f32[2] add(sum, crossed_2)
}
)hlo"),
                                     arguments, /*expect_change=*/true));
  // Only `staged` reads the operand of `known` in place of res_true.
  EXPECT_EQ(Count(*module, HloOpcode::kCustomCall), 0);
}

// With index 2, `known` takes its branch 2 and `staged` its last branch 1.
TEST_F(ConditionalResidualElisionTest, KeepsResidualsOfOtherBranchCounts) {
  EXPECT_OK(RunAndCompare(R"hlo(
HloModule m

known_0 {
  p = (f32[2]) parameter(0)
  zero = f32[] constant(0)
  zeros = f32[2] broadcast(zero), dimensions={}
  ROOT t = (f32[2]) tuple(zeros)
}

known_1 {
  p = (f32[2]) parameter(0)
  x = f32[2] get-tuple-element(p), index=0
  ROOT t = (f32[2]) tuple(x)
}

known_2 {
  p = (f32[2]) parameter(0)
  x = f32[2] get-tuple-element(p), index=0
  e = f32[2] exponential(x)
  ROOT t = (f32[2]) tuple(e)
}

staged_0 {
  p = (f32[2]) parameter(0)
  ROOT c = f32[2] constant({1, 2})
}

staged_1 {
  p = (f32[2]) parameter(0)
  r = f32[2] get-tuple-element(p), index=0
  ROOT n = f32[2] negate(r)
}

ENTRY entry {
  i = s32[] parameter(0)
  x = f32[2] parameter(1)
  in = (f32[2]) tuple(x)
  known = (f32[2]) conditional(i, in, in, in),
      branch_computations={known_0, known_1, known_2}
  res = f32[2] get-tuple-element(known), index=0
  args = (f32[2]) tuple(res)
  ROOT staged = f32[2] conditional(i, args, args),
      branch_computations={staged_0, staged_1}
}
)hlo",
                          Arguments<int32_t>({0, 1, 2}),
                          /*expect_change=*/false));
}

// No other conditional takes the branch of `known`.
TEST_F(ConditionalResidualElisionTest, KeepsLoneConditionals) {
  EXPECT_OK(RunAndCompare(absl::StrCat(kBranches, kKnown, R"hlo(
  ROOT sum = f32[2] add(out, res_false)
}
)hlo"),
                          Arguments<bool>({true, false}),
                          /*expect_change=*/false));
}

// The operand of the true branch of `staged` is the result of a call, which
// holds the first operand res_true of the call in its second element.
TEST_F(ConditionalResidualElisionTest, KeepsOperandsThatAreNoTuples) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      RunAndCompare(absl::StrCat(kBranches, R"hlo(
swap {
  a = f32[2] parameter(0)
  b = f32[2] parameter(1)
  ROOT t = (f32[2], f32[2]) tuple(b, a)
}

second {
  p = (f32[2], f32[2]) parameter(0)
  ROOT r = f32[2] get-tuple-element(p), index=1
}
)hlo",
                                 kKnown, R"hlo(
  args_true = (f32[2], f32[2]) call(res_true, out), to_apply=swap
  args_false = (f32[2], f32[2]) tuple(res_false, out)
  ROOT staged = f32[2] conditional(sel, args_true, args_false),
      true_computation=second, false_computation=staged_false
}
)hlo"),
                    Arguments<bool>({true, false}), /*expect_change=*/true));
  const HloInstruction* call =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_EQ(call->operand(0)->opcode(), HloOpcode::kGetTupleElement);
}

// No branch of `known` returns a read of its parameter: the true branch
// returns an element of the result of a barrier of its whole operand, or a
// constant, and the false branch an element of the result of a call.
TEST_F(ConditionalResidualElisionTest, KeepsResidualsReadFromOtherTuples) {
  constexpr absl::string_view kBarrier = R"hlo(
  barrier = (f32[2]) opt-barrier(p)
  res = f32[2] get-tuple-element(barrier), index=0)hlo";
  constexpr absl::string_view kConstant = R"hlo(
  res = f32[2] constant({3, 4}))hlo";
  for (absl::string_view res_true : {kBarrier, kConstant}) {
    EXPECT_OK(RunAndCompare(absl::StrCat(R"hlo(
HloModule m

exp {
  q = (f32[2]) parameter(0)
  a = f32[2] get-tuple-element(q), index=0
  e = f32[2] exponential(a)
  ROOT t = (f32[2]) tuple(e)
}

known_true {
  p = (f32[2]) parameter(0))hlo",
                                         res_true, R"hlo(
  ROOT r = (f32[2]) tuple(res)
}

known_false {
  p = (f32[2]) parameter(0)
  x = f32[2] get-tuple-element(p), index=0
  t = (f32[2]) tuple(x)
  c = (f32[2]) call(t), to_apply=exp
  res = f32[2] get-tuple-element(c), index=0
  ROOT r = (f32[2]) tuple(res)
}

staged_true {
  p = (f32[2]) parameter(0)
  r = f32[2] get-tuple-element(p), index=0
  ROOT n = f32[2] negate(r)
}

staged_false {
  p = (f32[2]) parameter(0)
  r = f32[2] get-tuple-element(p), index=0
  ROOT a = f32[2] abs(r)
}

ENTRY entry {
  sel = pred[] parameter(0)
  x = f32[2] parameter(1)
  in = (f32[2]) tuple(x)
  known = (f32[2]) conditional(sel, in, in),
      true_computation=known_true, false_computation=known_false
  res = f32[2] get-tuple-element(known), index=0
  args = (f32[2]) tuple(res)
  ROOT staged = f32[2] conditional(sel, args, args),
      true_computation=staged_true, false_computation=staged_false
}
)hlo"),
                            Arguments<bool>({true, false}),
                            /*expect_change=*/false));
  }
}

// Both branches of `staged` read one tuple: only the true branch may read the
// operand of `known` in place of the residual.
TEST_F(ConditionalResidualElisionTest, KeepsTuplesOfOtherBranches) {
  EXPECT_OK(RunAndCompare(absl::StrCat(kBranches, kKnown, R"hlo(
  args = (f32[2], f32[2]) tuple(res_true, out)
  ROOT staged = f32[2] conditional(sel, args, args),
      true_computation=staged_true, false_computation=staged_false
}
)hlo"),
                          Arguments<bool>({true, false}),
                          /*expect_change=*/true));
}

// The operand of `staged` is also the result of the computation.
TEST_F(ConditionalResidualElisionTest, KeepsRootTuples) {
  EXPECT_OK(RunAndCompare(absl::StrCat(kBranches, kKnown, R"hlo(
  ROOT args_true = (f32[2], f32[2]) tuple(res_true, out)
  args_false = (f32[2], f32[2]) tuple(res_false, out)
  staged = f32[2] conditional(sel, args_true, args_false),
      true_computation=staged_true, false_computation=staged_false
}
)hlo"),
                          Arguments<bool>({true, false}),
                          /*expect_change=*/true));
}

// `again` is a residual and the result of the computation.
TEST_F(ConditionalResidualElisionTest, KeepsRootOutputs) {
  EXPECT_OK(RunAndCompare(absl::StrCat(kBranches, kKnown, R"hlo(
  ROOT again = f32[2] get-tuple-element(known), index=2
  args_true = (f32[2], f32[2]) tuple(res_true, out)
  args_false = (f32[2], f32[2]) tuple(again, out)
  staged = f32[2] conditional(sel, args_true, args_false),
      true_computation=staged_true, false_computation=staged_false
}
)hlo"),
                          Arguments<bool>({true, false}),
                          /*expect_change=*/true));
}

TEST_F(ConditionalResidualElisionTest, KeepsOutputsWithOtherReaders) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      RunAndCompare(absl::StrCat(kBranches, kKnown, R"hlo(
  args_true = (f32[2], f32[2]) tuple(res_true, out)
  args_false = (f32[2], f32[2]) tuple(res_false, out)
  staged = f32[2] conditional(sel, args_true, args_false),
      true_computation=staged_true, false_computation=staged_false
  ROOT sum = f32[2] add(staged, res_false)
}
)hlo"),
                    Arguments<bool>({true, false}), /*expect_change=*/true));
  // `sum` reads res_false whatever the branch.
  EXPECT_EQ(Count(*module, HloOpcode::kCustomCall), 0);
}

// The operand of the false branch of `staged` holds res_false twice, and the
// branch only reads the second.
TEST_F(ConditionalResidualElisionTest, KeepsOutputsReadInAnotherTupleSlot) {
  EXPECT_OK(RunAndCompare(absl::StrCat(kBranches, R"hlo(
second {
  p = (f32[2], f32[2]) parameter(0)
  ROOT r = f32[2] get-tuple-element(p), index=1
}
)hlo",
                                       kKnown, R"hlo(
  args_true = (f32[2], f32[2]) tuple(out, out)
  args_false = (f32[2], f32[2]) tuple(res_false, res_false)
  ROOT staged = f32[2] conditional(sel, args_true, args_false),
      true_computation=staged_true, false_computation=second
}
)hlo"),
                          Arguments<bool>({true, false}),
                          /*expect_change=*/true));
}

// Each branch of `staged` returns its operand, residual included.
TEST_F(ConditionalResidualElisionTest, KeepsOutputsThatBranchesReturn) {
  EXPECT_OK(RunAndCompare(absl::StrCat(kBranches, R"hlo(
through {
  p = (f32[2], f32[2]) parameter(0)
  ROOT barrier = (f32[2], f32[2]) opt-barrier(p)
}

same {
  ROOT p = (f32[2], f32[2]) parameter(0)
}
)hlo",
                                       kKnown, R"hlo(
  args_true = (f32[2], f32[2]) tuple(res_true, out)
  args_false = (f32[2], f32[2]) tuple(res_false, out)
  ROOT staged = (f32[2], f32[2]) conditional(sel, args_true, args_false),
      true_computation=through, false_computation=same
}
)hlo"),
                          Arguments<bool>({true, false}),
                          /*expect_change=*/true));
}

// The true branch of `known` returns its operand, which is no tuple
// instruction.
TEST_F(ConditionalResidualElisionTest, KeepsProducersWithOtherRoots) {
  EXPECT_OK(RunAndCompare(R"hlo(
HloModule m

known_true {
  ROOT p = (f32[2], f32[2]) parameter(0)
}

known_false {
  p = (f32[2], f32[2]) parameter(0)
  x = f32[2] get-tuple-element(p), index=0
  e = f32[2] exponential(x)
  zero = f32[] constant(0)
  zeros = f32[2] broadcast(zero), dimensions={}
  ROOT t = (f32[2], f32[2]) tuple(zeros, e)
}

staged_true {
  p = (f32[2]) parameter(0)
  r = f32[2] get-tuple-element(p), index=0
  ROOT n = f32[2] negate(r)
}

staged_false {
  p = (f32[2]) parameter(0)
  r = f32[2] get-tuple-element(p), index=0
  ROOT a = f32[2] abs(r)
}

ENTRY entry {
  sel = pred[] parameter(0)
  x = f32[2] parameter(1)
  in = (f32[2], f32[2]) tuple(x, x)
  known = (f32[2], f32[2]) conditional(sel, in, in),
      true_computation=known_true, false_computation=known_false
  res_true = f32[2] get-tuple-element(known), index=0
  res_false = f32[2] get-tuple-element(known), index=1
  args_true = (f32[2]) tuple(res_true)
  args_false = (f32[2]) tuple(res_false)
  ROOT staged = f32[2] conditional(sel, args_true, args_false),
      true_computation=staged_true, false_computation=staged_false
}
)hlo",
                          Arguments<bool>({true, false}),
                          /*expect_change=*/false));
}

// `other` calls the branches of `known` too, and its outputs are all read.
TEST_F(ConditionalResidualElisionTest, KeepsBranchesWithOtherCallers) {
  EXPECT_OK(RunAndCompare(absl::StrCat(kBranches, kKnown, R"hlo(
  other = (f32[2], f32[2], f32[2]) conditional(sel, in, in),
      true_computation=known_true, false_computation=known_false
  args_true = (f32[2], f32[2]) tuple(res_true, out)
  args_false = (f32[2], f32[2]) tuple(res_false, out)
  staged = f32[2] conditional(sel, args_true, args_false),
      true_computation=staged_true, false_computation=staged_false
  ROOT t = (f32[2], (f32[2], f32[2], f32[2])) tuple(staged, other)
}
)hlo"),
                          Arguments<bool>({true, false}),
                          /*expect_change=*/true));
}

// Both branches of `known` are one computation, and each branch of `staged`
// reads another output of it.
TEST_F(ConditionalResidualElisionTest, KeepsBranchesCalledTwice) {
  EXPECT_OK(RunAndCompare(R"hlo(
HloModule m

both {
  p = (f32[2]) parameter(0)
  x = f32[2] get-tuple-element(p), index=0
  n = f32[2] negate(x)
  e = f32[2] exponential(x)
  ROOT t = (f32[2], f32[2]) tuple(n, e)
}

staged_true {
  p = (f32[2]) parameter(0)
  r = f32[2] get-tuple-element(p), index=0
  ROOT n = f32[2] negate(r)
}

staged_false {
  p = (f32[2]) parameter(0)
  r = f32[2] get-tuple-element(p), index=0
  ROOT a = f32[2] abs(r)
}

ENTRY entry {
  sel = pred[] parameter(0)
  x = f32[2] parameter(1)
  in = (f32[2]) tuple(x)
  known = (f32[2], f32[2]) conditional(sel, in, in),
      true_computation=both, false_computation=both
  res_true = f32[2] get-tuple-element(known), index=0
  res_false = f32[2] get-tuple-element(known), index=1
  args_true = (f32[2]) tuple(res_true)
  args_false = (f32[2]) tuple(res_false)
  ROOT staged = f32[2] conditional(sel, args_true, args_false),
      true_computation=staged_true, false_computation=staged_false
}
)hlo",
                          Arguments<bool>({true, false}),
                          /*expect_change=*/false));
}

// The caller of the computation reads every output of `known`.
TEST_F(ConditionalResidualElisionTest, KeepsOutputsOfRootProducers) {
  EXPECT_OK(RunAndCompare(absl::StrCat(kBranches, R"hlo(
ENTRY entry {
  sel = pred[] parameter(0)
  x = f32[2] parameter(1)
  in = (f32[2]) tuple(x)
  ROOT known = (f32[2], f32[2], f32[2]) conditional(sel, in, in),
      true_computation=known_true, false_computation=known_false
  out = f32[2] get-tuple-element(known), index=0
  res_true = f32[2] get-tuple-element(known), index=1
  res_false = f32[2] get-tuple-element(known), index=2
  args_true = (f32[2], f32[2]) tuple(res_true, out)
  args_false = (f32[2], f32[2]) tuple(res_false, out)
  staged = f32[2] conditional(sel, args_true, args_false),
      true_computation=staged_true, false_computation=staged_false
}
)hlo"),
                          Arguments<bool>({true, false}),
                          /*expect_change=*/true));
}

// The true branch of `staged` reads all outputs of `known` at once.
TEST_F(ConditionalResidualElisionTest, KeepsOutputsOfProducersReadWhole) {
  EXPECT_OK(RunAndCompare(absl::StrCat(kBranches, R"hlo(
whole_true {
  p = ((f32[2], f32[2], f32[2]), f32[2]) parameter(0)
  whole = (f32[2], f32[2], f32[2]) get-tuple-element(p), index=0
  ROOT res = f32[2] get-tuple-element(whole), index=1
}
)hlo",
                                       kKnown, R"hlo(
  whole = (f32[2], f32[2], f32[2]) opt-barrier(known)
  args_true = ((f32[2], f32[2], f32[2]), f32[2]) tuple(whole, out)
  args_false = (f32[2], f32[2]) tuple(res_false, out)
  ROOT staged = f32[2] conditional(sel, args_true, args_false),
      true_computation=whole_true, false_computation=staged_false
}
)hlo"),
                          Arguments<bool>({true, false}),
                          /*expect_change=*/false));
}

// The residual of the false branch is a tuple, which no buffer can stand for.
TEST_F(ConditionalResidualElisionTest, KeepsTupleOutputs) {
  EXPECT_OK(RunAndCompare(R"hlo(
HloModule m

known_true {
  p = (f32[2]) parameter(0)
  x = f32[2] get-tuple-element(p), index=0
  out = f32[2] negate(x)
  nested = (f32[2]) tuple(out)
  ROOT t = (f32[2], (f32[2])) tuple(out, nested)
}

known_false {
  p = (f32[2]) parameter(0)
  x = f32[2] get-tuple-element(p), index=0
  res = f32[2] exponential(x)
  nested = (f32[2]) tuple(res)
  ROOT t = (f32[2], (f32[2])) tuple(res, nested)
}

staged_true {
  p = (f32[2]) parameter(0)
  g = f32[2] get-tuple-element(p), index=0
  ROOT n = f32[2] negate(g)
}

staged_false {
  p = ((f32[2]), f32[2]) parameter(0)
  nested = (f32[2]) get-tuple-element(p), index=0
  res = f32[2] get-tuple-element(nested), index=0
  g = f32[2] get-tuple-element(p), index=1
  ROOT d = f32[2] subtract(res, g)
}

ENTRY entry {
  sel = pred[] parameter(0)
  x = f32[2] parameter(1)
  in = (f32[2]) tuple(x)
  known = (f32[2], (f32[2])) conditional(sel, in, in),
      true_computation=known_true, false_computation=known_false
  out = f32[2] get-tuple-element(known), index=0
  res_false = (f32[2]) get-tuple-element(known), index=1
  args_true = (f32[2]) tuple(out)
  args_false = ((f32[2]), f32[2]) tuple(res_false, out)
  ROOT staged = f32[2] conditional(sel, args_true, args_false),
      true_computation=staged_true, false_computation=staged_false
}
)hlo",
                          Arguments<bool>({true, false}),
                          /*expect_change=*/false));
}

// The size of the residual of the false branch is part of its value.
TEST_F(ConditionalResidualElisionTest, KeepsDynamicOutputs) {
  EXPECT_OK(RunAndCompare(R"hlo(
HloModule m

known_true {
  p = (f32[<=2]) parameter(0)
  x = f32[<=2] get-tuple-element(p), index=0
  out = f32[<=2] negate(x)
  ROOT t = (f32[<=2], f32[<=2]) tuple(out, out)
}

known_false {
  p = (f32[<=2]) parameter(0)
  x = f32[<=2] get-tuple-element(p), index=0
  res = f32[<=2] exponential(x)
  ROOT t = (f32[<=2], f32[<=2]) tuple(res, res)
}

staged_true {
  p = (f32[<=2]) parameter(0)
  g = f32[<=2] get-tuple-element(p), index=0
  ROOT n = f32[<=2] negate(g)
}

staged_false {
  p = (f32[<=2], f32[<=2]) parameter(0)
  res = f32[<=2] get-tuple-element(p), index=0
  g = f32[<=2] get-tuple-element(p), index=1
  ROOT d = f32[<=2] subtract(res, g)
}

ENTRY entry {
  sel = pred[] parameter(0)
  x = f32[<=2] parameter(1)
  in = (f32[<=2]) tuple(x)
  known = (f32[<=2], f32[<=2]) conditional(sel, in, in),
      true_computation=known_true, false_computation=known_false
  out = f32[<=2] get-tuple-element(known), index=0
  res_false = f32[<=2] get-tuple-element(known), index=1
  args_true = (f32[<=2]) tuple(out)
  args_false = (f32[<=2], f32[<=2]) tuple(res_false, out)
  ROOT staged = f32[<=2] conditional(sel, args_true, args_false),
      true_computation=staged_true, false_computation=staged_false
}
)hlo",
                          /*arguments=*/{}, /*expect_change=*/false));
}

TEST_F(ConditionalResidualElisionTest, KeepsOutputsOfShardedProducers) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      RunAndCompare(absl::StrCat(kBranches, R"hlo(
ENTRY entry {
  sel = pred[] parameter(0)
  x = f32[2] parameter(1)
  in = (f32[2]) tuple(x)
  known = (f32[2], f32[2], f32[2]) conditional(sel, in, in),
      true_computation=known_true, false_computation=known_false,
      sharding={{replicated}, {replicated}, {replicated}}
  out = f32[2] get-tuple-element(known), index=0
  res_true = f32[2] get-tuple-element(known), index=1
  res_false = f32[2] get-tuple-element(known), index=2
  args_true = (f32[2], f32[2]) tuple(res_true, out)
  args_false = (f32[2], f32[2]) tuple(res_false, out)
  ROOT staged = f32[2] conditional(sel, args_true, args_false),
      true_computation=staged_true, false_computation=staged_false
}
)hlo"),
                    Arguments<bool>({true, false}), /*expect_change=*/true));
  EXPECT_EQ(
      FindInstruction(module.get(), "known")->shape().tuple_shapes().size(), 3);
  EXPECT_EQ(Count(*module, HloOpcode::kCustomCall), 0);
}

// `staged` would no longer wait for `out` once it stops reading an instruction
// that has `out` as control predecessor: the read of the residual, its bitcast
// or the tuple.
TEST_F(ConditionalResidualElisionTest, KeepsReadsWithControlDependencies) {
  for (int ordered = 0; ordered < 3; ++ordered) {
    const auto after_out = [&](int instruction) {
      return instruction == ordered ? ", control-predecessors={out}" : "";
    };
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<VerifiedHloModule> module,
        RunAndCompare(absl::StrCat(kBranches, R"hlo(
ENTRY entry {
  sel = pred[] parameter(0)
  x = f32[2] parameter(1)
  in = (f32[2]) tuple(x)
  known = (f32[2], f32[2], f32[2]) conditional(sel, in, in),
      true_computation=known_true, false_computation=known_false
  out = f32[2] get-tuple-element(known), index=0
  res_true = f32[2] get-tuple-element(known), index=1)hlo",
                                   after_out(0), R"hlo(
  cast = f32[2] bitcast(res_true))hlo",
                                   after_out(1), R"hlo(
  res_false = f32[2] get-tuple-element(known), index=2
  args_true = (f32[2], f32[2]) tuple(cast, out))hlo",
                                   after_out(2), R"hlo(
  args_false = (f32[2], f32[2]) tuple(res_false, out)
  ROOT staged = f32[2] conditional(sel, args_true, args_false),
      true_computation=staged_true, false_computation=staged_false
}
)hlo"),
                      Arguments<bool>({true, false}), /*expect_change=*/true));
    // `staged` still reads the residual through its tuple.
    EXPECT_EQ(module->entry_computation()->root_instruction()->operand(1),
              FindInstruction(module.get(), "args_true"));
  }
}

// A bitcast of a value in another layout holds other values. The layout of
// the operand of `known` differs from the one its branches expect.
TEST_F(ConditionalResidualElisionTest, KeepsBitcastsOfOtherLayouts) {
  EXPECT_OK(RunAndCompare(R"hlo(
HloModule m

known_true {
  p = (f32[2,2]{0,1}) parameter(0)
  x = f32[2,2]{0,1} get-tuple-element(p), index=0
  b = f32[4] bitcast(x)
  ROOT t = (f32[4]) tuple(b)
}

known_false {
  p = (f32[2,2]{0,1}) parameter(0)
  x = f32[2,2]{0,1} get-tuple-element(p), index=0
  b = f32[4] bitcast(x)
  e = f32[4] exponential(b)
  ROOT t = (f32[4]) tuple(e)
}

staged_true {
  p = (f32[4]) parameter(0)
  r = f32[4] get-tuple-element(p), index=0
  ROOT n = f32[4] negate(r)
}

staged_false {
  p = (f32[4]) parameter(0)
  r = f32[4] get-tuple-element(p), index=0
  ROOT a = f32[4] abs(r)
}

ENTRY entry {
  sel = pred[] parameter(0)
  x = f32[2,2]{1,0} parameter(1)
  in = (f32[2,2]{1,0}) tuple(x)
  known = (f32[4]) conditional(sel, in, in),
      true_computation=known_true, false_computation=known_false
  res = f32[4] get-tuple-element(known), index=0
  args = (f32[4]) tuple(res)
  ROOT staged = f32[4] conditional(sel, args, args),
      true_computation=staged_true, false_computation=staged_false
}
)hlo",
                          /*arguments=*/{}, /*expect_change=*/false));
}

// The operand of `staged` keeps its shape with an element of the same layout
// only. The layout of the output of `known` differs from the one of its
// operand, which its true branch returns.
TEST_F(ConditionalResidualElisionTest, KeepsResidualsOfOtherLayouts) {
  EXPECT_OK(RunAndCompare(R"hlo(
HloModule m

known_true {
  p = (f32[2,2]{0,1}) parameter(0)
  x = f32[2,2]{0,1} get-tuple-element(p), index=0
  ROOT t = (f32[2,2]{0,1}) tuple(x)
}

known_false {
  p = (f32[2,2]{0,1}) parameter(0)
  x = f32[2,2]{0,1} get-tuple-element(p), index=0
  e = f32[2,2]{0,1} exponential(x)
  ROOT t = (f32[2,2]{0,1}) tuple(e)
}

staged_true {
  p = (f32[2,2]{1,0}) parameter(0)
  r = f32[2,2]{1,0} get-tuple-element(p), index=0
  ROOT n = f32[2,2]{1,0} negate(r)
}

staged_false {
  p = (f32[2,2]{1,0}) parameter(0)
  r = f32[2,2]{1,0} get-tuple-element(p), index=0
  ROOT a = f32[2,2]{1,0} abs(r)
}

ENTRY entry {
  sel = pred[] parameter(0)
  x = f32[2,2]{0,1} parameter(1)
  in = (f32[2,2]{0,1}) tuple(x)
  known = (f32[2,2]{1,0}) conditional(sel, in, in),
      true_computation=known_true, false_computation=known_false
  res = f32[2,2]{1,0} get-tuple-element(known), index=0
  args = (f32[2,2]{1,0}) tuple(res)
  ROOT staged = f32[2,2]{1,0} conditional(sel, args, args),
      true_computation=staged_true, false_computation=staged_false
}
)hlo",
                          /*arguments=*/{}, /*expect_change=*/false));
}

}  // namespace
}  // namespace xla
