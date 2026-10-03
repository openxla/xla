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

#include "xla/hlo/analysis/hlo_reachability_from_sources.h"

#include <memory>
#include <vector>

#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class HloReachabilityFromSourcesTest : public HloHardwareIndependentTestBase {
 protected:
  // Checks every (source, instruction) pair against HloReachabilityMap.
  void ExpectMatchesFullMap(const HloComputation* computation,
                            const std::vector<HloInstruction*>& sources) {
    std::unique_ptr<HloReachabilityMap> full =
        HloReachabilityMap::Build(computation);
    std::unique_ptr<HloReachabilityFromSources> from_sources =
        HloReachabilityFromSources::Build(computation, sources);
    for (const HloInstruction* source : sources) {
      for (const HloInstruction* instruction : computation->instructions()) {
        ASSERT_EQ(from_sources->IsReachable(source, instruction),
                  full->IsReachable(source, instruction))
            << source->name() << " -> " << instruction->name();
      }
    }
  }
};

TEST_F(HloReachabilityFromSourcesTest, DataAndControlEdges) {
  // const1    const2
  //    |         |
  //    | +-------+
  //    | |       |
  //    add ..   negate
  //     |   .     |
  //     |   .... exp
  //     |         |
  //     +---+   +-+---+
  //         |   |     |
  //       multiply   copy
  //             .......
  //
  // The dotted edges are control dependencies, from add to exp and from
  // copy to multiply. copy is created after multiply, so the order of
  // creation is not a dependency order and the build must use the post
  // order.
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  HloInstruction* constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f)));
  HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kAdd, constant1, constant2));
  HloInstruction* negate = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kNegate, constant2));
  HloInstruction* exp = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, negate));
  HloInstruction* mul = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kMultiply, add, exp));
  HloInstruction* copy = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kCopy, exp));
  auto module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputation(builder.Build(/*root_instruction=*/mul));
  ASSERT_OK(add->AddControlDependencyTo(exp));
  ASSERT_OK(copy->AddControlDependencyTo(mul));

  std::unique_ptr<HloReachabilityFromSources> reachability =
      HloReachabilityFromSources::Build(computation,
                                        {constant1, exp, mul, copy});
  // Through the control edges only.
  EXPECT_TRUE(reachability->IsReachable(constant1, exp));
  EXPECT_TRUE(reachability->IsReachable(copy, mul));
  EXPECT_TRUE(reachability->IsReachable(constant1, copy));
  EXPECT_FALSE(reachability->IsReachable(constant1, negate));
  EXPECT_TRUE(reachability->IsReachable(exp, exp));
  EXPECT_FALSE(reachability->IsReachable(exp, add));
  EXPECT_FALSE(reachability->IsReachable(mul, copy));
  EXPECT_TRUE(reachability->IsSource(exp));
  EXPECT_FALSE(reachability->IsSource(negate));
  EXPECT_TRUE(reachability->IsPresent(negate));

  ExpectMatchesFullMap(computation, {constant1, exp, mul, copy});
  ExpectMatchesFullMap(computation,
                       {constant1, constant2, add, negate, exp, mul, copy});
}

TEST_F(HloReachabilityFromSourcesTest, MoreSourcesThanBitsInAWord) {
  // Two chains of 70 instructions; chain_a[20] also feeds chain_b through
  // one add. Every instruction is a source, so rows span three words.
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  auto builder = HloComputation::Builder(TestName());
  std::vector<HloInstruction*> chain_a;
  std::vector<HloInstruction*> chain_b;
  chain_a.push_back(builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f))));
  chain_b.push_back(builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f))));
  for (int i = 1; i < 70; ++i) {
    chain_a.push_back(builder.AddInstruction(HloInstruction::CreateUnary(
        r0f32, HloOpcode::kNegate, chain_a.back())));
    HloInstruction* operand = chain_b.back();
    if (i == 35) {
      operand = builder.AddInstruction(HloInstruction::CreateBinary(
          r0f32, HloOpcode::kAdd, chain_a[20], chain_b.back()));
    }
    chain_b.push_back(builder.AddInstruction(
        HloInstruction::CreateUnary(r0f32, HloOpcode::kNegate, operand)));
  }
  HloInstruction* root = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kMultiply, chain_a.back(), chain_b.back()));
  auto module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputation(builder.Build(/*root_instruction=*/root));

  std::vector<HloInstruction*> sources(computation->instructions().begin(),
                                       computation->instructions().end());
  // A repeated source must be accepted.
  sources.push_back(chain_a[20]);
  ASSERT_GT(sources.size(), 128);
  ExpectMatchesFullMap(computation, sources);
  // Source counts around the word boundaries: the last bit of a word, one
  // bit in a new word.
  for (int count : {1, 63, 64, 65, 128, 129}) {
    SCOPED_TRACE(count);
    ExpectMatchesFullMap(
        computation,
        std::vector<HloInstruction*>(sources.begin(), sources.begin() + count));
  }
  // The bit position is the position among the sources, not the local id.
  ExpectMatchesFullMap(computation, {chain_a[60], chain_b[64], root});

  std::unique_ptr<HloReachabilityFromSources> reachability =
      HloReachabilityFromSources::Build(computation, sources);
  EXPECT_TRUE(reachability->IsReachable(chain_a[20], chain_b[50]));
  EXPECT_FALSE(reachability->IsReachable(chain_a[21], chain_b[50]));
  EXPECT_FALSE(reachability->IsReachable(chain_b[5], chain_a[69]));
}

TEST_F(HloReachabilityFromSourcesTest, ContractAndSnapshot) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  // Removed before Build, so the local ids of the computation have a gap.
  HloInstruction* dead = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kNegate, constant));
  HloInstruction* negate = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kNegate, constant));
  HloInstruction* root = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, negate));
  auto module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputation(builder.Build(/*root_instruction=*/root));
  ASSERT_OK(computation->RemoveInstruction(dead));
  ExpectMatchesFullMap(computation, {constant, root});

  std::unique_ptr<HloReachabilityFromSources> reachability =
      HloReachabilityFromSources::Build(computation, {constant});
  EXPECT_DEATH(reachability->IsReachable(negate, root), "is not a source");

  // Added after Build.
  HloInstruction* late = computation->AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kNegate, constant));
  EXPECT_FALSE(reachability->IsPresent(late));
  EXPECT_DEATH(reachability->IsReachable(constant, late),
               "is not in the analysis");

  // Another computation reuses the local ids.
  auto other_builder = HloComputation::Builder("other");
  HloInstruction* other_constant = other_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f)));
  module->AddEmbeddedComputation(other_builder.Build());
  ASSERT_EQ(other_constant->local_id(), constant->local_id());
  EXPECT_FALSE(reachability->IsPresent(other_constant));
  EXPECT_FALSE(reachability->IsSource(other_constant));
  EXPECT_FALSE(reachability->IsPresent(nullptr));
  EXPECT_DEATH(HloReachabilityFromSources::Build(computation, {other_constant}),
               "is not in");
}

TEST_F(HloReachabilityFromSourcesTest, NoSources) {
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  std::unique_ptr<HloReachabilityFromSources> reachability =
      HloReachabilityFromSources::Build(computation, {});
  EXPECT_FALSE(reachability->IsSource(constant));
  EXPECT_TRUE(reachability->IsPresent(constant));
}

}  // namespace
}  // namespace xla
