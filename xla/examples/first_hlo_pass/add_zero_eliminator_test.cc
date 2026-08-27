#include "xla/examples/first_hlo_pass/add_zero_eliminator.h"

#include <memory>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla {
namespace {

using ::tsl::testing::IsOkAndHolds;

TEST(AddZeroEliminatorTest, EliminatesZeroOnRight) {
  const char* hlo = R"(
HloModule test

ENTRY main {
  value = s32[] parameter(0)
  zero = s32[] constant(0)
  ROOT add = s32[] add(value, zero)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo));

  AddZeroEliminator pass;
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kParameter);
}

TEST(AddZeroEliminatorTest, EliminatesZeroOnLeft) {
  const char* hlo = R"(
HloModule test

ENTRY main {
  value = s32[4]{0} parameter(0)
  zero = s32[4]{0} constant({0, 0, 0, 0})
  ROOT add = s32[4]{0} add(zero, value)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo));

  AddZeroEliminator pass;
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kParameter);
}

TEST(AddZeroEliminatorTest, KeepsNonzeroConstant) {
  const char* hlo = R"(
HloModule test

ENTRY main {
  value = s32[] parameter(0)
  one = s32[] constant(1)
  ROOT add = s32[] add(value, one)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo));

  AddZeroEliminator pass;
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kAdd);
}

TEST(AddZeroEliminatorTest, KeepsFloatingPointAddZero) {
  const char* hlo = R"(
HloModule test

ENTRY main {
  value = f32[] parameter(0)
  zero = f32[] constant(0)
  ROOT add = f32[] add(value, zero)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo));

  AddZeroEliminator pass;
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kAdd);
}

}  // namespace
}  // namespace xla
