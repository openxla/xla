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

#include "xla/hlo/transforms/simplifiers/alias_copy_avoidance.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/copy_insertion.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;

class AliasCopyAvoidanceTest : public HloHardwareIndependentTestBase {
 protected:
  // Everything the module holds but its aliases.
  static std::string WithoutAliases(const HloModule& module) {
    HloModuleProto proto = module.ToProto();
    proto.clear_input_output_alias();
    return proto.ShortDebugString();
  }

  // Runs the pass on `module` and returns whether it removed an alias. Checks
  // that the pass did nothing else: the aliases that remain are those of
  // before, the removed ones were optional, and the rest of the module is
  // untouched.
  absl::StatusOr<bool> RunPass(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads = {}) {
    const HloInputOutputAliasConfig before =
        module->input_output_alias_config();
    const std::string rest = WithoutAliases(*module);
    ABSL_ASSIGN_OR_RETURN(const bool changed,
                          AliasCopyAvoidance().Run(module, execution_threads));
    const HloInputOutputAliasConfig& after =
        module->input_output_alias_config();
    EXPECT_EQ(WithoutAliases(*module), rest);
    bool removed = false;
    before.ForEachAlias([&](const ShapeIndex& output_index,
                            const HloInputOutputAliasConfig::Alias& alias) {
      if (!after.OutputHasAlias(output_index)) {
        removed = true;
        EXPECT_FALSE(alias.must_alias()) << alias.ToString();
      }
    });
    after.ForEachAlias([&](const ShapeIndex& output_index,
                           const HloInputOutputAliasConfig::Alias& alias) {
      EXPECT_TRUE(before.GetAliasedParameter(output_index) == alias)
          << alias.ToString();
    });
    EXPECT_EQ(changed, removed);
    return changed;
  }

  // Runs the pass on `hlo` and returns the aliases that the module has
  // afterwards.
  absl::StatusOr<std::string> AliasesAfterPass(
      absl::string_view hlo,
      const absl::flat_hash_set<absl::string_view>& execution_threads = {}) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
    ABSL_RETURN_IF_ERROR(RunPass(module.get(), execution_threads).status());
    return module->input_output_alias_config().ToShortString();
  }

  // Returns how many copies copy insertion leaves in the entry computation.
  absl::StatusOr<int64_t> CopiesAfterCopyInsertion(HloModule* module) {
    AliasInfo alias_info;
    ABSL_RETURN_IF_ERROR(CopyInsertion(&alias_info).Run(module).status());
    return absl::c_count_if(module->entry_computation()->instructions(),
                            [](const HloInstruction* instruction) {
                              return instruction->opcode() == HloOpcode::kCopy;
                            });
  }
};

// Output 0 aliases p0, which only its own update reads. Output 1 aliases p1,
// which `change` reads after `new1` is computed.
constexpr absl::string_view kReadAfterWrite = R"hlo(
HloModule m, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias) }

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  new0 = f32[8] negate(p0)
  new1 = f32[8] negate(p1)
  square = f32[8] multiply(new1, new1)
  change = f32[8] subtract(square, p1)
  ROOT out = (f32[8], f32[8], f32[8]) tuple(new0, new1, change)
}
)hlo";

TEST_F(AliasCopyAvoidanceTest, DropsAliasOfParameterReadAfterTheWrite) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kReadAfterWrite));
  EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(true));
  EXPECT_EQ(module->input_output_alias_config().ToShortString(),
            "{0}: (0, {}, may-alias)");
  EXPECT_EQ(module->buffer_donor_config().ToShortString(), "");
  EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(false));
  EXPECT_THAT(CopiesAfterCopyInsertion(module.get()), IsOkAndHolds(0));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> aliased,
                       ParseAndReturnVerifiedModule(kReadAfterWrite));
  EXPECT_THAT(CopiesAfterCopyInsertion(aliased.get()), IsOkAndHolds(1));
}

// Both parameters are read after their updates, and output 0 has to alias p0.
TEST_F(AliasCopyAvoidanceTest, KeepsRequiredAliases) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m, input_output_alias={ {0}: (0, {}, must-alias), {1}: (1, {}, may-alias) }

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  new0 = f32[8] negate(p0)
  new1 = f32[8] negate(p1)
  change0 = f32[8] subtract(new0, p0)
  change1 = f32[8] subtract(new1, p1)
  ROOT out = (f32[8], f32[8], f32[8], f32[8]) tuple(new0, new1, change0, change1)
}
)hlo";
  EXPECT_THAT(AliasesAfterPass(kHlo), IsOkAndHolds("{0}: (0, {}, must-alias)"));
}

// `first` reads p0 before the update that needs it. `aside` reads p1 and can
// run before or after the update of p1.
TEST_F(AliasCopyAvoidanceTest, KeepsAliasesWhoseReadersCanRunFirst) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias) }

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  first = f32[8] exponential(p0)
  new0 = f32[8] add(p0, first)
  new1 = f32[8] negate(p1)
  aside = f32[8] exponential(p1)
  ROOT out = (f32[8], f32[8], f32[8]) tuple(new0, new1, aside)
}
)hlo";
  EXPECT_THAT(AliasesAfterPass(kHlo),
              IsOkAndHolds("{0}: (0, {}, may-alias), {1}: (1, {}, may-alias)"));
}

// Output 0 is the parameter it aliases, which needs no copy as long as the
// alias stays. Output 1 is another parameter and output 2 is a constant. Copy
// insertion copies those two into their outputs, and no reader has to wait for
// the copies.
TEST_F(AliasCopyAvoidanceTest, KeepsAliasesOfReadOnlyOutputs) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias), {2}: (2, {}, may-alias) }

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  p2 = f32[8] parameter(2)
  p3 = f32[8] parameter(3)
  ones = f32[8] constant({1, 1, 1, 1, 1, 1, 1, 1})
  sum0 = f32[8] add(p0, ones)
  sum1 = f32[8] add(p1, p3)
  sum2 = f32[8] add(p2, ones)
  ROOT out = (f32[8], f32[8], f32[8], f32[8], f32[8], f32[8]) tuple(p0, p3, ones, sum0, sum1, sum2)
}
)hlo";
  EXPECT_THAT(AliasesAfterPass(kHlo),
              IsOkAndHolds("{0}: (0, {}, may-alias), {1}: (1, {}, may-alias), "
                           "{2}: (2, {}, may-alias)"));
}

// The root is an array, not a tuple. `total` reads p0 before the update that
// needs it, through a reduction.
TEST_F(AliasCopyAvoidanceTest, KeepsAliasOfArrayRootWhoseReadersRunFirst) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m, input_output_alias={ {}: (0, {}, may-alias) }

add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT sum = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  zero = f32[] constant(0)
  total = f32[] reduce(p0, zero), dimensions={0}, to_apply=add
  totals = f32[8] broadcast(total), dimensions={}
  ROOT new0 = f32[8] subtract(p0, totals)
}
)hlo";
  EXPECT_THAT(AliasesAfterPass(kHlo), IsOkAndHolds("{}: (0, {}, may-alias)"));
}

// Output 2 returns p0 through a call. It needs a copy of p0 with or without the
// alias, and `change` can read that copy.
TEST_F(AliasCopyAvoidanceTest, KeepsAliasOfParameterThatIsAlsoReturned) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m, input_output_alias={ {0}: (0, {}, may-alias) }

same {
  ROOT x = f32[8] parameter(0)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  new0 = f32[8] negate(p0)
  change = f32[8] subtract(new0, p0)
  returned = f32[8] call(p0), to_apply=same
  ROOT out = (f32[8], f32[8], f32[8]) tuple(new0, change, returned)
}
)hlo";
  EXPECT_THAT(AliasesAfterPass(kHlo), IsOkAndHolds("{0}: (0, {}, may-alias)"));
}

// Element 0 of the parameter is read after its update and element 1 is not.
TEST_F(AliasCopyAvoidanceTest, ReadsElementsOfTupleParameters) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m, input_output_alias={ {0}: (0, {0}, may-alias), {1}: (0, {1}, may-alias) }

ENTRY main {
  p = (f32[8], f32[8]) parameter(0)
  p0 = f32[8] get-tuple-element(p), index=0
  p1 = f32[8] get-tuple-element(p), index=1
  new0 = f32[8] negate(p0)
  change = f32[8] subtract(new0, p0)
  new1 = f32[8] add(p1, change)
  ROOT out = (f32[8], f32[8]) tuple(new0, new1)
}
)hlo";
  EXPECT_THAT(AliasesAfterPass(kHlo), IsOkAndHolds("{1}: (0, {1}, may-alias)"));
}

// `change` reads p0 through a bitcast, and output 0 is a bitcast of `new0`.
TEST_F(AliasCopyAvoidanceTest, SeesThroughBitcasts) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m, input_output_alias={ {0}: (0, {}, may-alias) }

ENTRY main {
  p0 = f32[8] parameter(0)
  new0 = f32[8] negate(p0)
  old = f32[2,4] bitcast(p0)
  folded = f32[2,4] bitcast(new0)
  change = f32[2,4] subtract(folded, old)
  ROOT out = (f32[2,4], f32[2,4]) tuple(folded, change)
}
)hlo";
  EXPECT_THAT(AliasesAfterPass(kHlo), IsOkAndHolds(""));
}

// The asynchronous copy reads p0 until `done`, which waits for `new0`.
TEST_F(AliasCopyAvoidanceTest, AsyncReaderReadsUntilItsDone) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m, input_output_alias={ {0}: (0, {}, may-alias) }

ENTRY main {
  p0 = f32[8] parameter(0)
  start = (f32[8], f32[8], u32[]) copy-start(p0)
  new0 = f32[8] negate(p0)
  done = f32[8] copy-done(start), control-predecessors={new0}
  ROOT out = (f32[8], f32[8]) tuple(new0, done)
}
)hlo";
  EXPECT_THAT(AliasesAfterPass(kHlo), IsOkAndHolds(""));
}

// The conditional runs after `new0` and its true branch reads p0, which
// reaches it through a tuple that does not depend on `new0`.
TEST_F(AliasCopyAvoidanceTest, FindsReadersInBranches) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m, input_output_alias={ {0}: (0, {}, may-alias) }

reads_old {
  args = (f32[8]) parameter(0)
  old = f32[8] get-tuple-element(args), index=0
  ROOT stat = f32[8] exponential(old)
}

returns_new {
  args = (f32[8]) parameter(0)
  ROOT new = f32[8] get-tuple-element(args), index=0
}

ENTRY main {
  p0 = f32[8] parameter(0)
  taken = pred[] parameter(1)
  new0 = f32[8] negate(p0)
  old_args = (f32[8]) tuple(p0)
  new_args = (f32[8]) tuple(new0)
  stat = f32[8] conditional(taken, old_args, new_args), true_computation=reads_old, false_computation=returns_new
  ROOT out = (f32[8], f32[8]) tuple(new0, stat)
}
)hlo";
  EXPECT_THAT(AliasesAfterPass(kHlo), IsOkAndHolds(""));
}

// The loop computes output 0 in the buffer of p0, and `change` reads p0 after
// the loop.
TEST_F(AliasCopyAvoidanceTest, TakesLoopAsWriterOfItsState) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m, input_output_alias={ {0}: (0, {}, may-alias) }

body {
  state = (f32[8], s32[]) parameter(0)
  value = f32[8] get-tuple-element(state), index=0
  count = s32[] get-tuple-element(state), index=1
  doubled = f32[8] add(value, value)
  one = s32[] constant(1)
  next = s32[] add(count, one)
  ROOT result = (f32[8], s32[]) tuple(doubled, next)
}

condition {
  state = (f32[8], s32[]) parameter(0)
  count = s32[] get-tuple-element(state), index=1
  limit = s32[] constant(3)
  ROOT less = pred[] compare(count, limit), direction=LT
}

ENTRY main {
  p0 = f32[8] parameter(0)
  zero = s32[] constant(0)
  init = (f32[8], s32[]) tuple(p0, zero)
  loop = (f32[8], s32[]) while(init), condition=condition, body=body
  new0 = f32[8] get-tuple-element(loop), index=0
  change = f32[8] subtract(new0, p0)
  ROOT out = (f32[8], f32[8]) tuple(new0, change)
}
)hlo";
  EXPECT_THAT(AliasesAfterPass(kHlo), IsOkAndHolds(""));
}

// `same` has two callers, so output 0 holds the value of `new0` or of
// `early`, and no reader of p0 has to wait for `early`.
TEST_F(AliasCopyAvoidanceTest, KeepsAliasIfOneValueOfTheOutputNeedsNoCopy) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule m, input_output_alias={ {0}: (0, {}, may-alias) }

same {
  ROOT x = f32[8] parameter(0)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  new0 = f32[8] negate(p0)
  change = f32[8] subtract(new0, p0)
  early = f32[8] exponential(p0)
  first = f32[8] call(new0), to_apply=same
  second = f32[8] call(early), to_apply=same
  ROOT out = (f32[8], f32[8], f32[8]) tuple(second, first, change)
}
)hlo";
  EXPECT_THAT(AliasesAfterPass(kHlo), IsOkAndHolds("{0}: (0, {}, may-alias)"));
}

TEST_F(AliasCopyAvoidanceTest, SkipsEntryComputationOfExcludedThread) {
  EXPECT_THAT(AliasesAfterPass(kReadAfterWrite, {"other"}),
              IsOkAndHolds("{0}: (0, {}, may-alias), {1}: (1, {}, may-alias)"));
}

}  // namespace
}  // namespace xla
