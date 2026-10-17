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

#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <optional>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;
using ::testing::Not;

// Test-only pass that forwards the operand of every `negate` to its users.
class RemoveNegatePass : public HloModulePass {
 public:
  absl::string_view name() const override { return "remove-negate"; }

 protected:
  absl::StatusOr<bool> RunImpl(HloModule* module,
                               const absl::flat_hash_set<absl::string_view>&
                                   execution_threads) override {
    bool changed = false;
    for (HloComputation* computation :
         module->MakeNonfusionComputations(execution_threads)) {
      for (HloInstruction* instruction :
           computation->MakeInstructionPostOrder()) {
        if (instruction->opcode() != HloOpcode::kNegate) {
          continue;
        }
        ABSL_RETURN_IF_ERROR(computation->ReplaceInstruction(
            instruction, instruction->mutable_operand(0)));
        changed = true;
      }
    }
    return changed;
  }
};

using HloHardwareIndependentTestBaseTest = HloHardwareIndependentTestBase;

constexpr absl::string_view kHloWithNegate = R"(
HloModule m

ENTRY main {
  p0 = f32[4] parameter(0)
  ROOT negate = f32[4] negate(p0)
})";

constexpr absl::string_view kHloWithoutNegate = R"(
HloModule m

ENTRY main {
  p0 = f32[4] parameter(0)
  ROOT abs = f32[4] abs(p0)
})";

TEST_F(HloHardwareIndependentTestBaseTest,
       RunAndCheckHloRewriteReturnsRewrittenModule) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      RunAndCheckHloRewrite(kHloWithNegate, RemoveNegatePass()));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kParameter);
}

TEST_F(HloHardwareIndependentTestBaseTest,
       RunAndCheckHloRewriteAcceptsPassPointer) {
  RemoveNegatePass pass;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       RunAndCheckHloRewrite(kHloWithNegate, &pass));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kParameter);
}

TEST_F(HloHardwareIndependentTestBaseTest,
       RunAndCheckHloRewriteReturnsUnchangedModule) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      RunAndCheckHloRewrite(kHloWithoutNegate, RemoveNegatePass(),
                            /*expect_change=*/false));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kAbs);
}

TEST_F(HloHardwareIndependentTestBaseTest,
       RunAndCheckHloRewriteSubstitutesTemplateParams) {
  constexpr absl::string_view kHloTemplate = R"(
HloModule m

ENTRY main {
  p0 = $type[4] parameter(0)
  ROOT op = $type[4] $op(p0)
})";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      RunAndCheckHloRewrite(kHloTemplate, RemoveNegatePass(),
                            /*expect_change=*/true,
                            {{"$type", "s32"}, {"$op", "negate"}}));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->shape().element_type(), S32);
}

TEST_F(HloHardwareIndependentTestBaseTest,
       RunAndCheckHloRewritePropagatesParseErrors) {
  EXPECT_THAT(RunAndCheckHloRewrite("not an hlo module", RemoveNegatePass()),
              Not(IsOk()));
}

TEST_F(HloHardwareIndependentTestBaseTest,
       RunAndFilecheckHloRewriteWithInterleavedChecks) {
  constexpr absl::string_view kHloWithChecks = R"(
HloModule m

// CHECK: ENTRY %main
ENTRY main {
  // CHECK: ROOT %p0 = f32[4]{0} parameter(0)
  p0 = f32[4] parameter(0)
  // CHECK-NOT: negate
  ROOT negate = f32[4] negate(p0)
})";
  bool after_pass_checks_ran = false;
  RunAndFilecheckHloRewrite(
      kHloWithChecks, RemoveNegatePass(), [&](HloModule* module) {
        after_pass_checks_ran = true;
        EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
                  HloOpcode::kParameter);
      });
  EXPECT_TRUE(after_pass_checks_ran);
}

TEST_F(HloHardwareIndependentTestBaseTest,
       RunAndFilecheckHloRewriteWithExpectedPattern) {
  RunAndFilecheckHloRewrite(kHloWithNegate, RemoveNegatePass(), R"(
    // CHECK: ROOT %p0 = f32[4]{0} parameter(0)
    // CHECK-NOT: negate
  )");
}

TEST_F(HloHardwareIndependentTestBaseTest,
       RunAndFilecheckHloRewriteSkipsChecksWhenUnchanged) {
  bool after_pass_checks_ran = false;
  RunAndFilecheckHloRewrite(kHloWithoutNegate, RemoveNegatePass(),
                            /*expected=*/std::nullopt,
                            [&](HloModule*) { after_pass_checks_ran = true; });
  EXPECT_FALSE(after_pass_checks_ran);
}

}  // namespace
}  // namespace xla
