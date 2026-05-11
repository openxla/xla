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

#include "xla/hlo/analysis/alias_info.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/analysis/hlo_operand_index.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

class AliasInfoTest : public HloHardwareIndependentTestBase {};

TEST_F(AliasInfoTest, AsyncStartDefaultAliasing) {
  const char* const kHlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  start = ((f32[2,3]), f32[2,3], s32[]) call-start(p0),
    to_apply=async_computation,
    output_to_operand_aliasing={{1}: (0, {})}
  ROOT done = f32[2,3] call-done(start)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* start = FindInstruction(module.get(), "start");

  AliasInfo alias_info;
  auto pairs = alias_info.GetInPlaceInputOutputPairs(start);

  // By default for forwarded operands: operand 0 maps to output {0, 0} for the
  // parameter subshape
  EXPECT_THAT(pairs, ElementsAre(std::pair<HloOperandIndex, ShapeIndex>{
                         HloOperandIndex{0, {}}, {1}}));
}

}  // namespace
}  // namespace xla
