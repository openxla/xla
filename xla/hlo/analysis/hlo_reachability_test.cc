/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/analysis/hlo_reachability.h"

#include <gmock/gmock.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal_util.h"
#include "xla/service/device_assignment.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

class HloReachabilityMapTestPeer {
 public:
  static void SetMaxRowsPerEagerMerge(HloReachabilityMap& map,
                                      size_t max_rows) {
    map.max_rows_per_eager_merge_ = max_rows;
  }
  static size_t NumDeferredMerges(const HloReachabilityMap& map) {
    return map.deferred_merges_.size();
  }
  // Copies of the stored rows that have deferred merges pending, by index.
  static absl::flat_hash_map<size_t,
                             std::vector<HloReachabilityMap::BitSet::Word>>
  PendingRows(const HloReachabilityMap& map) {
    absl::flat_hash_map<size_t, std::vector<HloReachabilityMap::BitSet::Word>>
        rows;
    for (size_t i = 0; i < map.merges_applied_.size(); ++i) {
      if (map.merges_applied_[i] != map.deferred_merges_.size()) {
        std::vector<HloReachabilityMap::BitSet::Word>& words = rows[i];
        words.resize(map.words_per_bitset_);
        HloReachabilityMap::BitSet(words.data(), words.size())
            .CopyBitSet(map.BitSetFromIndex(i));
      }
    }
    return rows;
  }
};

namespace {

class HloReachabilityTest : public HloHardwareIndependentTestBase {};

TEST_F(HloReachabilityTest, Reachability) {
  // Construct and test a reachability graph of the following form:
  /*
       a
      / \
     b   c
      \ / \
       d   e
  */
  auto builder = HloComputation::Builder(TestName());
  auto a = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto b = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto c = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto d = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto e = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  HloReachabilityMap reachability({a, b, c, d, e});
  reachability.SetReachable(a, a);
  EXPECT_TRUE(reachability.SetReachabilityToUnion({a}, b));
  EXPECT_TRUE(reachability.SetReachabilityToUnion({a}, c));
  EXPECT_TRUE(reachability.SetReachabilityToUnion({b, c}, d));
  EXPECT_TRUE(reachability.SetReachabilityToUnion({c}, e));

  EXPECT_TRUE(reachability.IsReachable(a, a));
  EXPECT_TRUE(reachability.IsReachable(a, b));
  EXPECT_TRUE(reachability.IsReachable(a, c));
  EXPECT_TRUE(reachability.IsReachable(a, d));
  EXPECT_TRUE(reachability.IsReachable(a, e));

  EXPECT_FALSE(reachability.IsReachable(b, a));
  EXPECT_TRUE(reachability.IsReachable(b, b));
  EXPECT_FALSE(reachability.IsReachable(b, c));
  EXPECT_TRUE(reachability.IsReachable(b, d));
  EXPECT_FALSE(reachability.IsReachable(b, e));

  EXPECT_FALSE(reachability.IsReachable(e, a));
  EXPECT_FALSE(reachability.IsReachable(e, b));
  EXPECT_FALSE(reachability.IsReachable(e, c));
  EXPECT_FALSE(reachability.IsReachable(e, d));
  EXPECT_TRUE(reachability.IsReachable(e, e));

  // Recomputing the same reachability for a previously computed instruction
  // should return false (no change).
  EXPECT_FALSE(reachability.SetReachabilityToUnion({a}, b));
  EXPECT_FALSE(reachability.SetReachabilityToUnion({b, c}, d));
}

// The row of the instruction is written once: the first input row replaces
// it, unless the instruction is among its own inputs, in which case the row
// keeps its bits; without inputs only the diagonal bit remains.
TEST_F(HloReachabilityTest, SetReachabilityToUnionWritesTheRowOnce) {
  auto builder = HloComputation::Builder(TestName());
  auto a = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto b = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto c = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto d = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  HloReachabilityMap reachability({a, b, c, d});
  reachability.SetReachable(a, b);
  reachability.SetReachable(b, c);

  // The only input row replaces row d: b and c reach d, a does not.
  EXPECT_TRUE(reachability.SetReachabilityToUnion({c}, d));
  EXPECT_FALSE(reachability.IsReachable(a, d));
  EXPECT_TRUE(reachability.IsReachable(b, d));
  EXPECT_TRUE(reachability.IsReachable(c, d));

  // Listed among its own inputs, row d keeps its bits and takes the others.
  EXPECT_TRUE(reachability.SetReachabilityToUnion({d, a}, d));
  EXPECT_TRUE(reachability.IsReachable(a, d));
  EXPECT_TRUE(reachability.IsReachable(b, d));
  EXPECT_TRUE(reachability.IsReachable(c, d));
  EXPECT_FALSE(reachability.SetReachabilityToUnion({d, a}, d));

  // Not listed, the first input row replaces it again: only a reaches d.
  EXPECT_TRUE(reachability.SetReachabilityToUnion({a}, d));
  EXPECT_TRUE(reachability.IsReachable(a, d));
  EXPECT_FALSE(reachability.IsReachable(b, d));
  EXPECT_FALSE(reachability.IsReachable(c, d));
  EXPECT_TRUE(reachability.IsReachable(d, d));

  // Without inputs the row is cleared down to the diagonal bit.
  EXPECT_TRUE(reachability.SetReachabilityToUnion({}, d));
  EXPECT_FALSE(reachability.IsReachable(a, d));
  EXPECT_TRUE(reachability.IsReachable(d, d));
}

TEST_F(HloReachabilityTest, NonTrivialReachability) {
  // Test reachability of a non-trivial computation:
  //
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
  //
  // There is a control dependency from 'add' to 'exp'.
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f)));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kAdd, constant1, constant2));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kNegate, constant2));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, negate));
  auto mul = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kMultiply, add, exp));
  auto copy = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kCopy, exp));

  auto module = CreateNewVerifiedModule();
  auto computation =
      module->AddEntryComputation(builder.Build(/*root_instruction=*/mul));

  CHECK_OK(add->AddControlDependencyTo(exp));
  auto reachability = HloReachabilityMap::Build(computation);

  EXPECT_TRUE(reachability->IsReachable(constant1, constant1));
  EXPECT_FALSE(reachability->IsReachable(constant1, constant2));
  EXPECT_TRUE(reachability->IsReachable(constant1, add));
  EXPECT_FALSE(reachability->IsReachable(constant1, negate));
  EXPECT_TRUE(reachability->IsReachable(constant1, exp));
  EXPECT_TRUE(reachability->IsReachable(constant1, mul));
  EXPECT_TRUE(reachability->IsReachable(constant1, copy));

  EXPECT_FALSE(reachability->IsReachable(constant2, constant1));
  EXPECT_TRUE(reachability->IsReachable(constant2, constant2));
  EXPECT_TRUE(reachability->IsReachable(constant2, add));
  EXPECT_TRUE(reachability->IsReachable(constant2, negate));
  EXPECT_TRUE(reachability->IsReachable(constant2, exp));
  EXPECT_TRUE(reachability->IsReachable(constant2, mul));
  EXPECT_TRUE(reachability->IsReachable(constant2, copy));

  EXPECT_FALSE(reachability->IsReachable(exp, constant1));
  EXPECT_FALSE(reachability->IsReachable(exp, constant2));
  EXPECT_FALSE(reachability->IsReachable(exp, add));
  EXPECT_FALSE(reachability->IsReachable(exp, negate));
  EXPECT_TRUE(reachability->IsReachable(exp, exp));
  EXPECT_TRUE(reachability->IsReachable(exp, mul));
  EXPECT_TRUE(reachability->IsReachable(exp, copy));

  EXPECT_FALSE(reachability->IsReachable(mul, constant1));
  EXPECT_FALSE(reachability->IsReachable(mul, constant2));
  EXPECT_FALSE(reachability->IsReachable(mul, add));
  EXPECT_FALSE(reachability->IsReachable(mul, negate));
  EXPECT_FALSE(reachability->IsReachable(mul, exp));
  EXPECT_TRUE(reachability->IsReachable(mul, mul));
  EXPECT_FALSE(reachability->IsReachable(mul, copy));

  EXPECT_TRUE(reachability->IsConnected(constant1, copy));
  EXPECT_TRUE(reachability->IsConnected(copy, constant1));
  EXPECT_FALSE(reachability->IsConnected(negate, add));
  EXPECT_FALSE(reachability->IsConnected(add, negate));

  // Remove the control dependency then update and verify the reachability map
  ASSERT_IS_OK(add->RemoveControlDependencyTo(exp));
  reachability->UpdateReachabilityThroughInstruction(exp);

  EXPECT_TRUE(reachability->IsReachable(constant1, constant1));
  EXPECT_FALSE(reachability->IsReachable(constant1, constant2));
  EXPECT_TRUE(reachability->IsReachable(constant1, add));
  EXPECT_FALSE(reachability->IsReachable(constant1, negate));
  EXPECT_FALSE(reachability->IsReachable(constant1, exp));
  EXPECT_TRUE(reachability->IsReachable(constant1, mul));
  EXPECT_FALSE(reachability->IsReachable(constant1, copy));

  // Change a use within the graph then update and verify the reachability map
  ASSERT_IS_OK(constant2->ReplaceUseWith(negate, constant1));
  reachability->UpdateReachabilityThroughInstruction(negate);

  EXPECT_FALSE(reachability->IsReachable(constant2, constant1));
  EXPECT_TRUE(reachability->IsReachable(constant2, constant2));
  EXPECT_TRUE(reachability->IsReachable(constant2, add));
  EXPECT_FALSE(reachability->IsReachable(constant2, negate));
  EXPECT_FALSE(reachability->IsReachable(constant2, exp));
  EXPECT_TRUE(reachability->IsReachable(constant2, mul));
  EXPECT_FALSE(reachability->IsReachable(constant2, copy));
}

// Build writes each row of an unzeroed matrix exactly once. Check it against
// the incremental construction on a graph wide enough for multi word rows and
// tall enough for several row blocks, with input free instructions, repeated
// operands, wide fan in and control dependencies in the mix.
TEST_F(HloReachabilityTest, BuildMatchesIncrementalConstruction) {
  constexpr int kSize = 1200;
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  auto builder = HloComputation::Builder(TestName());
  std::vector<HloInstruction*> instructions;
  for (int i = 0; i < kSize; ++i) {
    HloInstruction* instruction;
    if (i % 97 == 0) {
      instruction = builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(i)));
    } else if (i % 5 == 0) {
      HloInstruction* x = instructions[i - 1];
      instruction = builder.AddInstruction(
          HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, x, x));
    } else if (i % 3 == 0) {
      instruction = builder.AddInstruction(HloInstruction::CreateBinary(
          r0f32, HloOpcode::kMultiply, instructions[i - 1],
          instructions[i - std::min(i, 7 + i % 11)]));
    } else {
      instruction = builder.AddInstruction(HloInstruction::CreateUnary(
          r0f32, HloOpcode::kExp,
          instructions[i - 1 - std::min(i - 1, (i % 2) * (i % 13))]));
    }
    instructions.push_back(instruction);
  }
  std::vector<HloInstruction*> roots;
  for (int i = kSize - 100; i < kSize; ++i) {
    if (instructions[i]->user_count() == 0) {
      roots.push_back(instructions[i]);
    }
  }
  HloInstruction* root =
      builder.AddInstruction(HloInstruction::CreateTuple(roots));
  auto module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputation(builder.Build(root));
  for (int i = 200; i + 300 < kSize; i += 250) {
    CHECK_OK(instructions[i]->AddControlDependencyTo(instructions[i + 300]));
  }

  std::vector<HloInstruction*> post_order =
      computation->MakeInstructionPostOrder();
  HloReachabilityMap expected(post_order);
  std::vector<HloInstruction*> inputs;
  for (HloInstruction* instruction : post_order) {
    inputs.assign(instruction->operands().begin(),
                  instruction->operands().end());
    inputs.insert(inputs.end(), instruction->control_predecessors().begin(),
                  instruction->control_predecessors().end());
    expected.FastSetReachabilityToUnion(inputs, instruction);
  }

  std::unique_ptr<HloReachabilityMap> built =
      HloReachabilityMap::Build(computation);
  int reachable_pairs = 0;
  for (HloInstruction* a : post_order) {
    for (HloInstruction* b : post_order) {
      ASSERT_EQ(built->IsReachable(a, b), expected.IsReachable(a, b))
          << a->name() << " -> " << b->name();
      reachable_pairs += built->IsReachable(a, b);
    }
  }
  // The graph is far from both extremes, so a matrix left all zero or all
  // one would fail above rather than pass by accident.
  EXPECT_GT(reachable_pairs, post_order.size() * 2);
  EXPECT_LT(reachable_pairs, post_order.size() * post_order.size() / 2);
  // The last two control edges connect instructions that no operand path
  // connects, so they show that Build reads control predecessors.
  std::unique_ptr<HloReachabilityMap> operands_only =
      HloReachabilityMap::BuildWithRestrictions(
          computation,
          [](const HloInstruction* hlo, std::vector<HloInstruction*>* inputs) {
            inputs->assign(hlo->operands().begin(), hlo->operands().end());
          });
  for (int i = 450; i + 300 < kSize; i += 250) {
    EXPECT_FALSE(
        operands_only->IsReachable(instructions[i], instructions[i + 300]));
    EXPECT_TRUE(built->IsReachable(instructions[i], instructions[i + 300]));
  }
}

TEST_F(HloReachabilityTest, ChannelReachability) {
  const Shape shape = ShapeUtil::MakeShape(F32, {5, 7});
  HloComputation::Builder builder("ChannelReachability");
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto token0 = builder.AddInstruction(HloInstruction::CreateToken());
  auto send = builder.AddInstruction(HloInstruction::CreateSend(
      param, token0, /*channel_id=*/1, /*is_host_transfer=*/false));
  auto send_done = builder.AddInstruction(HloInstruction::CreateSendDone(
      send, send->channel_id(), /*is_host_transfer=*/false));
  auto token1 = builder.AddInstruction(HloInstruction::CreateToken());
  auto recv = builder.AddInstruction(HloInstruction::CreateRecv(
      shape, token1, /*channel_id=*/1, /*is_host_transfer=*/false));
  auto recv_done = builder.AddInstruction(HloInstruction::CreateRecvDone(
      recv, recv->channel_id(), /*is_host_transfer=*/false));

  auto module = CreateNewVerifiedModule();
  module->mutable_config().set_use_spmd_partitioning(false);
  module->mutable_config().set_static_device_assignment(DeviceAssignment(1, 2));
  auto computation = module->AddEntryComputation(builder.Build(recv_done));
  auto reachability = HloReachabilityMap::Build(computation);
  EXPECT_FALSE(reachability->IsReachable(param, recv_done));
  EXPECT_FALSE(reachability->IsReachable(send, recv));
  EXPECT_FALSE(reachability->IsReachable(send_done, recv));
}

TEST_F(HloReachabilityTest, ReplaceInstructions) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY entry {
      p0 = f32[28,28]{1,0} parameter(0)
      ROOT add = f32[28,28]{1,0} add(p0, p0)
    })")
                    .value();
  auto computation = module->entry_computation();
  auto reachability = HloReachabilityMap::Build(computation);
  auto* add = module->entry_computation()->root_instruction();
  auto* p0 = add->operand(0);
  EXPECT_TRUE(reachability->IsReachable(p0, add));

  // Replacing an instruction with itself is a noop.
  reachability->Replace(add, add);
  EXPECT_TRUE(reachability->IsReachable(p0, add));

  // Introduce a fusion instruction taking the place of `add`.
  auto* fusion = computation->AddInstruction(HloInstruction::CreateFusion(
      add->shape(), HloInstruction::FusionKind::kLoop, add));
  EXPECT_FALSE(reachability->IsPresent(fusion));
  EXPECT_TRUE(reachability->IsReachable(p0, add));

  // Replace `add` with `fusion` in the readability map.
  reachability->Replace(add, fusion);
  EXPECT_FALSE(reachability->IsPresent(add));
  EXPECT_TRUE(reachability->IsReachable(p0, fusion));
}

TEST_F(HloReachabilityTest, UpdateMultipleInstructions) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  auto builder = HloComputation::Builder(TestName());
  auto a = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  auto b = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f)));
  auto c = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kNegate, a));
  auto d = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, b));
  auto e = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kCopy, c));
  auto f = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kCopy, d));

  auto module = CreateNewVerifiedModule();
  auto computation =
      module->AddEntryComputation(builder.Build(/*root_instruction=*/f));

  auto reachability = HloReachabilityMap::Build(computation);

  EXPECT_TRUE(reachability->IsReachable(a, c));
  EXPECT_TRUE(reachability->IsReachable(c, e));
  EXPECT_TRUE(reachability->IsReachable(a, e));

  EXPECT_FALSE(reachability->IsReachable(b, c));
  EXPECT_FALSE(reachability->IsReachable(b, e));
  EXPECT_FALSE(reachability->IsReachable(d, e));
  EXPECT_FALSE(reachability->IsReachable(a, d));
  EXPECT_FALSE(reachability->IsReachable(a, f));

  // Add a control dependency from b to c, and d to e.
  ASSERT_IS_OK(b->AddControlDependencyTo(c));
  ASSERT_IS_OK(d->AddControlDependencyTo(e));

  absl::flat_hash_map<const HloInstruction*,
                      absl::flat_hash_set<const HloInstruction*>>
      to_update;
  to_update[c].insert(b);
  to_update[e].insert(d);

  reachability->UpdateMultipleInstructions(to_update);

  // Now b should be reachable to c, e
  EXPECT_TRUE(reachability->IsReachable(b, c));
  EXPECT_TRUE(reachability->IsReachable(b, e));

  // d should be reachable to e
  EXPECT_TRUE(reachability->IsReachable(d, e));

  // a is still reachable to c, e
  EXPECT_TRUE(reachability->IsReachable(a, c));
  EXPECT_TRUE(reachability->IsReachable(a, e));

  // a is still not reachable to d, f
  EXPECT_FALSE(reachability->IsReachable(a, d));
  EXPECT_FALSE(reachability->IsReachable(a, f));
}

// Adds an entry computation of `size` scalar instructions in `lanes` lanes to
// `module`, with a tuple of the unused ones as its root. Operands come mostly
// from the last few instructions of the same lane, so instructions of
// different lanes tend to be independent. A few control edges join lanes.
// Returns the instructions other than the root.
std::vector<HloInstruction*> AddRandomLanes(HloModule* module, int size,
                                            int lanes, std::mt19937& rng) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  auto builder = HloComputation::Builder("random_lanes");
  std::vector<HloInstruction*> instructions;
  auto pick_operand = [&](int i) {
    if (i >= lanes && absl::Bernoulli(rng, 0.9)) {
      int back = absl::Uniform(rng, 1, 1 + std::min(4, i / lanes));
      return instructions[i - lanes * back];
    }
    return instructions[absl::Uniform(rng, 0, i)];
  };
  for (int i = 0; i < size; ++i) {
    if (i < lanes || absl::Bernoulli(rng, 0.02)) {
      instructions.push_back(builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(i))));
    } else if (absl::Bernoulli(rng, 0.5)) {
      instructions.push_back(builder.AddInstruction(HloInstruction::CreateUnary(
          r0f32, HloOpcode::kExp, pick_operand(i))));
    } else {
      instructions.push_back(
          builder.AddInstruction(HloInstruction::CreateBinary(
              r0f32, HloOpcode::kAdd, pick_operand(i), pick_operand(i))));
    }
  }
  std::vector<HloInstruction*> roots;
  for (HloInstruction* instruction : instructions) {
    if (instruction->user_count() == 0) {
      roots.push_back(instruction);
    }
  }
  builder.AddInstruction(HloInstruction::CreateTuple(roots));
  module->AddEntryComputation(builder.Build());
  for (int i = 0; i < size / 40; ++i) {
    int from = absl::Uniform(rng, 0, size - 1);
    int to = absl::Uniform(rng, from + 1, size);
    CHECK_OK(instructions[from]->AddControlDependencyTo(instructions[to]));
  }
  return instructions;
}

// Merges `absorbed` into `kept` in the computation, the way multi output fusion
// does: kept takes over the users of absorbed and comes to depend on all that
// absorbed depended on. With `relay_control_dependencies`, kept also takes over
// the control dependencies of absorbed, which then stays behind without users
// or is removed. Without, absorbed stays behind with its control dependencies.
void MergeInComputation(HloInstruction* absorbed, HloInstruction* kept,
                        bool relay_control_dependencies, bool remove) {
  for (HloInstruction* operand : absorbed->operands()) {
    CHECK_OK(operand->AddControlDependencyTo(kept));
  }
  if (relay_control_dependencies) {
    for (HloInstruction* predecessor : absorbed->control_predecessors()) {
      CHECK_OK(predecessor->AddControlDependencyTo(kept));
    }
    for (HloInstruction* successor : absorbed->control_successors()) {
      CHECK_OK(kept->AddControlDependencyTo(successor));
    }
    CHECK_OK(absorbed->DropAllControlDeps());
  }
  CHECK_OK(absorbed->ReplaceAllUsesWith(kept));
  if (remove) {
    CHECK_OK(absorbed->parent()->RemoveInstruction(absorbed));
  }
}

// The parameter is the number of rows a merge may change and still be applied
// at once by DeferReachabilityUpdateForMerge.
class DeferredMergeTest : public HloReachabilityTest,
                          public ::testing::WithParamInterface<size_t> {};

INSTANTIATE_TEST_SUITE_P(
    EagerMergeBudgets, DeferredMergeTest,
    ::testing::Values(0, 4, 16, std::numeric_limits<size_t>::max()),
    [](const ::testing::TestParamInfo<size_t>& info) -> std::string {
      return info.param == std::numeric_limits<size_t>::max()
                 ? "Unbounded"
                 : absl::StrCat("Budget", info.param);
    });

// Merges random pairs of independent instructions the way multi output fusion
// does: first the reachability update, then the merge in the computation, which
// either removes the absorbed instruction or, like a fused non-fusion, leaves
// it behind without users. One map defers its merges, the other applies them
// eagerly. Queries and new control dependencies are interleaved, and the
// deferring map ends with eager merges over its pending ones. Both maps must
// answer like a fresh Build for the live instructions.
TEST_P(DeferredMergeTest, DeferredMergesMatchEagerMerges) {
  constexpr int kMerges = 300;
  constexpr int kDeferredMerges = 200;
  const size_t budget = GetParam();
  std::mt19937 rng(1234);
  auto module = CreateNewVerifiedModule();
  std::vector<HloInstruction*> live =
      AddRandomLanes(module.get(), /*size=*/1100, /*lanes=*/40, rng);
  HloComputation* computation = module->entry_computation();
  std::unique_ptr<HloReachabilityMap> deferred =
      HloReachabilityMap::Build(computation);
  HloReachabilityMapTestPeer::SetMaxRowsPerEagerMerge(*deferred, budget);
  std::unique_ptr<HloReachabilityMap> eager =
      HloReachabilityMap::Build(computation);
  // Merges deferred, and rows that a merge applied at once changed while they
  // had deferred merges pending.
  int num_deferred = 0;
  int pending_rows_changed = 0;
  auto random_live = [&]() {
    return live[absl::Uniform<size_t>(rng, 0, live.size())];
  };
  auto expect_matches_build = [&]() {
    std::unique_ptr<HloReachabilityMap> built =
        HloReachabilityMap::Build(computation);
    for (HloInstruction* a : live) {
      for (HloInstruction* b : live) {
        ASSERT_EQ(eager->IsReachable(a, b), built->IsReachable(a, b))
            << a->name() << " -> " << b->name();
        ASSERT_EQ(deferred->IsReachable(a, b), built->IsReachable(a, b))
            << a->name() << " -> " << b->name();
      }
    }
  };
  // Instructions left behind, each with the live instruction that absorbed it.
  std::vector<std::pair<HloInstruction*, HloInstruction*>> left_behind;

  int merges = 0;
  for (int attempt = 0; merges < kMerges && attempt < 100 * kMerges;
       ++attempt) {
    HloInstruction* kept = random_live();
    HloInstruction* absorbed = random_live();
    if (kept == absorbed || eager->IsConnected(kept, absorbed)) {
      continue;
    }
    eager->UpdateReachabilityForMerge(absorbed, kept);
    if (merges < kDeferredMerges) {
      const size_t num_deferred_before =
          HloReachabilityMapTestPeer::NumDeferredMerges(*deferred);
      const auto pending_before =
          HloReachabilityMapTestPeer::PendingRows(*deferred);
      deferred->DeferReachabilityUpdateForMerge(absorbed, kept);
      if (HloReachabilityMapTestPeer::NumDeferredMerges(*deferred) !=
          num_deferred_before) {
        ++num_deferred;
      } else {
        const auto pending_after =
            HloReachabilityMapTestPeer::PendingRows(*deferred);
        for (const auto& [index, words] : pending_before) {
          auto it = pending_after.find(index);
          pending_rows_changed +=
              it != pending_after.end() && it->second != words;
        }
      }
    } else {
      deferred->UpdateReachabilityForMerge(absorbed, kept);
    }
    for (int q = 0; q < 4; ++q) {
      HloInstruction* a = random_live();
      HloInstruction* b = random_live();
      ASSERT_EQ(deferred->IsReachable(a, b), eager->IsReachable(a, b))
          << a->name() << " -> " << b->name() << " after merge " << merges;
    }
    const bool remove = absl::Bernoulli(rng, 0.5);
    MergeInComputation(absorbed, kept, /*relay_control_dependencies=*/true,
                       remove);
    if (!remove) {
      left_behind.push_back({absorbed, kept});
    }
    for (auto& [instruction, absorber] : left_behind) {
      if (absorber == absorbed) {
        absorber = kept;
      }
    }
    live.erase(absl::c_find(live, absorbed));
    ++merges;

    // Now and then a new control dependency, applied through either updater,
    // from a live user of the merged instruction: its row has the merge
    // pending.
    std::vector<HloInstruction*> live_users;
    for (HloInstruction* user : kept->users()) {
      if (absl::c_linear_search(live, user)) {
        live_users.push_back(user);
      }
    }
    if (merges % 4 == 0 && !live_users.empty()) {
      HloInstruction* from =
          live_users[absl::Uniform<size_t>(rng, 0, live_users.size())];
      HloInstruction* to = random_live();
      if (from != to && !eager->IsReachable(to, from)) {
        ASSERT_OK(from->AddControlDependencyTo(to));
        if (merges % 8 == 0) {
          eager->UpdateReachabilityThroughInstruction(to);
          deferred->UpdateReachabilityThroughInstruction(to);
        } else {
          eager->UpdateMultipleInstructions({{to, {from}}});
          deferred->UpdateMultipleInstructions({{to, {from}}});
        }
        for (HloInstruction* a : live) {
          ASSERT_EQ(deferred->IsReachable(a, to), eager->IsReachable(a, to))
              << a->name() << " -> " << to->name() << " after merge " << merges;
        }
      }
    }
    if (merges == kDeferredMerges / 2 || merges == kDeferredMerges) {
      expect_matches_build();
    }
  }
  EXPECT_EQ(merges, kMerges);
  expect_matches_build();
  if (budget == std::numeric_limits<size_t>::max()) {
    EXPECT_EQ(num_deferred, 0);
  } else {
    EXPECT_GT(num_deferred, 0);
  }
  if (budget == 4 || budget == 16) {
    EXPECT_GT(pending_rows_changed, 0);
  }
  // The row of an instruction left behind stays within the row of the
  // instruction that absorbed it, with either kind of merge.
  EXPECT_GT(left_behind.size(), kMerges / 4);
  for (const auto& [instruction, absorber] : left_behind) {
    for (HloInstruction* a : live) {
      EXPECT_TRUE(!deferred->IsReachable(a, instruction) ||
                  deferred->IsReachable(a, absorber))
          << a->name() << " -> " << instruction->name();
      EXPECT_TRUE(!eager->IsReachable(a, instruction) ||
                  eager->IsReachable(a, absorber))
          << a->name() << " -> " << instruction->name();
    }
  }
}

// Once a merge leaves the absorbed instruction behind with a control successor,
// the maps claim more dependencies than the computation has, and merges must
// be applied eagerly. Rows that still have deferred merges pending must then
// end up as if every merge had been applied eagerly.
TEST_P(DeferredMergeTest, DeferredMergesThenInexactMergesMatchEagerMerges) {
  constexpr int kMerges = 180;
  constexpr int kDeferredMerges = 120;
  std::mt19937 rng(4321);
  auto module = CreateNewVerifiedModule();
  std::vector<HloInstruction*> live =
      AddRandomLanes(module.get(), /*size=*/700, /*lanes=*/30, rng);
  HloComputation* computation = module->entry_computation();
  std::unique_ptr<HloReachabilityMap> deferred =
      HloReachabilityMap::Build(computation);
  HloReachabilityMapTestPeer::SetMaxRowsPerEagerMerge(*deferred, GetParam());
  std::unique_ptr<HloReachabilityMap> eager =
      HloReachabilityMap::Build(computation);
  auto random_live = [&]() {
    return live[absl::Uniform<size_t>(rng, 0, live.size())];
  };

  int merges = 0;
  for (int attempt = 0; merges < kMerges && attempt < 100 * kMerges;
       ++attempt) {
    HloInstruction* kept = random_live();
    HloInstruction* absorbed = random_live();
    if (kept == absorbed || eager->IsConnected(kept, absorbed)) {
      continue;
    }
    const bool exact = merges < kDeferredMerges;
    if (!exact) {
      // Give absorbed a control successor that it keeps after the merge.
      HloInstruction* successor = random_live();
      if (successor == kept || successor == absorbed ||
          eager->IsReachable(successor, absorbed)) {
        continue;
      }
      ASSERT_OK(absorbed->AddControlDependencyTo(successor));
      eager->UpdateReachabilityThroughInstruction(successor);
      deferred->UpdateReachabilityThroughInstruction(successor);
    }
    eager->UpdateReachabilityForMerge(absorbed, kept);
    if (exact) {
      deferred->DeferReachabilityUpdateForMerge(absorbed, kept);
    } else {
      deferred->UpdateReachabilityForMerge(absorbed, kept);
    }
    MergeInComputation(absorbed, kept, /*relay_control_dependencies=*/exact,
                       /*remove=*/exact && absl::Bernoulli(rng, 0.5));
    live.erase(absl::c_find(live, absorbed));
    ++merges;
  }
  EXPECT_EQ(merges, kMerges);
  for (HloInstruction* a : live) {
    for (HloInstruction* b : live) {
      ASSERT_EQ(deferred->IsReachable(a, b), eager->IsReachable(a, b))
          << a->name() << " -> " << b->name();
    }
  }
}

// d is fused into f but stays behind with its control successor s, so s claims
// the dependencies of f, among them a. c is then merged into a, which s does
// not see, and then b into a. s depends on b and claims a, so the merge of b
// must leave s alone, as on a map that never deferred. That holds only if the
// eager walk sees the merge still pending in the row of s.
TEST_F(HloReachabilityTest, EagerMergeAfterInexactMergeUsesCurrentRows) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = f32[] parameter(2)
      b = f32[] parameter(3)
      a = f32[] exponential(p1)
      c = f32[] exponential(p2)
      d = f32[] negate(p0)
      f = f32[] exponential(a)
      s = f32[] negate(b), control-predecessors={d}
      ROOT t = (f32[], f32[], f32[], f32[]) tuple(d, f, s, c)
    })")
                    .value();
  HloComputation* computation = module->entry_computation();
  auto get = [&](absl::string_view name) {
    return computation->GetInstructionWithName(name);
  };
  HloInstruction* a = get("a");
  HloInstruction* b = get("b");
  HloInstruction* c = get("c");
  HloInstruction* d = get("d");
  HloInstruction* f = get("f");
  HloInstruction* s = get("s");
  HloInstruction* p2 = get("p2");
  auto deferred = HloReachabilityMap::Build(computation);
  auto eager = HloReachabilityMap::Build(computation);
  deferred->DeferReachabilityUpdateForMerge(d, f);
  eager->UpdateReachabilityForMerge(d, f);
  MergeInComputation(d, f, /*relay_control_dependencies=*/false,
                     /*remove=*/false);
  deferred->UpdateReachabilityForMerge(c, a);
  eager->UpdateReachabilityForMerge(c, a);
  MergeInComputation(c, a, /*relay_control_dependencies=*/true,
                     /*remove=*/true);
  deferred->UpdateReachabilityForMerge(b, a);
  eager->UpdateReachabilityForMerge(b, a);
  EXPECT_FALSE(eager->IsReachable(p2, s));
  EXPECT_EQ(deferred->IsReachable(p2, s), eager->IsReachable(p2, s));
}

// SetReachabilityToUnion compares against the row with its deferred merges
// applied, so recomputing a row that a merge already covers is no change.
TEST_F(HloReachabilityTest, SetReachabilityToUnionSeesDeferredMerges) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY entry {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      c = f32[] negate(b)
      ROOT t = (f32[], f32[]) tuple(a, c)
    })")
                    .value();
  HloComputation* computation = module->entry_computation();
  HloInstruction* a = computation->parameter_instruction(0);
  HloInstruction* b = computation->parameter_instruction(1);
  HloInstruction* c = b->users()[0];
  auto reachability = HloReachabilityMap::Build(computation);
  reachability->DeferReachabilityUpdateForMerge(a, b);
  EXPECT_FALSE(reachability->SetReachabilityToUnion({b}, c));
  EXPECT_TRUE(reachability->IsReachable(a, c));
}

}  // namespace

class HloReachabilityMapBitSetBenchmark {
 public:
  explicit HloReachabilityMapBitSetBenchmark(int size) {
    size_t nwords = (size + 63) / 64;
    space_.resize(2 * nwords);
    a_ = HloReachabilityMap::BitSet(&space_[0], nwords);
    b_ = HloReachabilityMap::BitSet(&space_[nwords], nwords);
    // Initialize the bit sets to random inputs. Done out of caution -- note
    // that a sufficiently smart optimizer might realize that the bit sets
    // are otherwise initialized to 0.
    absl::BitGen gen;
    for (int i = 0; i < size; ++i) {
      if (absl::Bernoulli(gen, 0.5)) a_.Set(i);
      if (absl::Bernoulli(gen, 0.5)) b_.Set(i);
    }
  }
  void Union() { a_ |= b_; }

  void OrUpdatePartial(
      const std::vector<std::pair<size_t, HloReachabilityMap::BitSet::Word>>&
          diff) {
    a_.OrUpdatePartial(diff);
  }

  std::vector<std::pair<size_t, HloReachabilityMap::BitSet::Word>> GenerateDiff(
      int num_elements) {
    std::vector<std::pair<size_t, HloReachabilityMap::BitSet::Word>> diff;
    size_t nwords = a_.NumWords();
    if (nwords == 0) {
      return diff;
    }
    absl::BitGen gen;
    if (num_elements >= nwords) {
      for (size_t i = 0; i < nwords; ++i) {
        diff.push_back(
            {i, absl::Uniform<HloReachabilityMap::BitSet::Word>(gen)});
      }
    } else {
      absl::flat_hash_set<uint64_t> indices;
      while (indices.size() < num_elements) {
        indices.insert(absl::Uniform<size_t>(gen, 0, nwords));
      }
      std::vector<uint64_t> sorted_indices(indices.begin(), indices.end());
      std::sort(sorted_indices.begin(), sorted_indices.end());
      for (uint64_t idx : sorted_indices) {
        diff.push_back(
            {idx, absl::Uniform<HloReachabilityMap::BitSet::Word>(gen)});
      }
    }
    return diff;
  }

 private:
  std::vector<uint64_t> space_;
  HloReachabilityMap::BitSet a_;
  HloReachabilityMap::BitSet b_;
};

namespace {

void BM_HloReachabilityBitSetUnion(benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  for (auto s : state) {
    bm.Union();
  }
}
#define BM_ARGS Arg(1)->Arg(64)->Arg(128)->Arg(256)->Range(512, 256 * 1024)
BENCHMARK(BM_HloReachabilityBitSetUnion)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartialRandom2Diff(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  auto diff = bm.GenerateDiff(2);
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartialRandom2Diff)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartialRandom10Diff(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  auto diff = bm.GenerateDiff(10);
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartialRandom10Diff)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartial1Percent(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  int nwords = (state.range(0) + 63) / 64;
  auto diff = bm.GenerateDiff(std::max<int>(1, nwords / 100));
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartial1Percent)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartial10Percent(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  int nwords = (state.range(0) + 63) / 64;
  auto diff = bm.GenerateDiff(std::max<int>(1, nwords / 10));
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartial10Percent)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartial25Percent(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  int nwords = (state.range(0) + 63) / 64;
  auto diff = bm.GenerateDiff(std::max<int>(1, nwords / 4));
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartial25Percent)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartial50Percent(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  int nwords = (state.range(0) + 63) / 64;
  auto diff = bm.GenerateDiff(std::max<int>(1, nwords / 2));
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartial50Percent)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartial75Percent(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  int nwords = (state.range(0) + 63) / 64;
  auto diff = bm.GenerateDiff(std::max<int>(1, nwords * 3 / 4));
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartial75Percent)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartialDenseRandom(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  auto diff = bm.GenerateDiff((state.range(0) + 63) / 64);
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartialDenseRandom)->BM_ARGS;

class HloReachabilityBenchmark {
 public:
  HloReachabilityBenchmark(int size, absl::string_view name) : name_(name) {
    Shape r0f32 = ShapeUtil::MakeShape(F32, {});
    auto builder = HloComputation::Builder(name);

    // Build a graph of chained Exponentials, i.e. Exp(...(Exp(Input))...).
    HloInstruction* constant = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f)));
    HloInstruction* prev = constant;
    for (int i = 1; i < size; ++i) {
      prev = builder.AddInstruction(
          HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, prev));
    }

    HloModuleConfig hlo_config;
    module_ = std::make_unique<HloModule>(name_, hlo_config);
    computation_ =
        module_->AddEntryComputation(builder.Build(/*root_instruction=*/prev));
  }
  std::unique_ptr<HloReachabilityMap> Build() {
    return HloReachabilityMap::Build(computation_);
  }

 private:
  std::unique_ptr<HloModule> module_;
  HloComputation* computation_;
  const std::string name_;
};

void BM_HloReachabilityBuild(benchmark::State& state) {
  HloReachabilityBenchmark bm(state.range(0), state.name());
  for (auto s : state) {
    benchmark::DoNotOptimize(bm.Build());
  }
}
BENCHMARK(BM_HloReachabilityBuild)->BM_ARGS;

// Merges pairs of sibling negates of one parameter, each followed by a chain of
// range(1) exponentials, eagerly (range(0) = 0) or deferred (range(0) = 1). As
// in multi output fusion, each pair is checked for a connection first. With
// range(2) = 1, every row is read once afterwards, the worst case for deferred
// merges. Each iteration builds a new map untimed, so the iteration count is
// fixed.
void BM_MergeSiblingChains(benchmark::State& state) {
  const bool defer = state.range(0) != 0;
  const int chain_length = state.range(1);
  const bool read_all_rows = state.range(2) != 0;
  constexpr int kLanes = 16;
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  auto builder = HloComputation::Builder("sibling_chains");
  HloInstruction* parameter = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "parameter"));
  std::vector<HloInstruction*> heads;
  std::vector<HloInstruction*> tails;
  for (int lane = 0; lane < kLanes; ++lane) {
    heads.push_back(builder.AddInstruction(
        HloInstruction::CreateUnary(r0f32, HloOpcode::kNegate, parameter)));
    HloInstruction* tail = heads.back();
    for (int i = 0; i < chain_length; ++i) {
      tail = builder.AddInstruction(
          HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, tail));
    }
    tails.push_back(tail);
  }
  builder.AddInstruction(HloInstruction::CreateTuple(tails));
  HloModule module("sibling_chains", HloModuleConfig());
  HloComputation* computation = module.AddEntryComputation(builder.Build());
  std::unique_ptr<HloReachabilityMap> reachability;
  for (auto s : state) {
    state.PauseTiming();
    reachability.reset();
    reachability = HloReachabilityMap::Build(computation);
    state.ResumeTiming();
    for (int lane = 0; lane + 1 < kLanes; lane += 2) {
      benchmark::DoNotOptimize(
          reachability->IsConnected(heads[lane], heads[lane + 1]));
      if (defer) {
        reachability->DeferReachabilityUpdateForMerge(heads[lane],
                                                      heads[lane + 1]);
      } else {
        reachability->UpdateReachabilityForMerge(heads[lane], heads[lane + 1]);
      }
    }
    if (read_all_rows) {
      for (const HloInstruction* instruction : computation->instructions()) {
        benchmark::DoNotOptimize(
            reachability->IsReachable(heads[1], instruction));
      }
    }
  }
}
BENCHMARK(BM_MergeSiblingChains)
    ->ArgsProduct({{0, 1}, {256, 2048}, {0, 1}})
    ->Iterations(16);

// Merges range(1) sibling negates of one parameter one by one into the first,
// eagerly (range(0) = 0) or deferred (range(0) = 1), each after checking it for
// a connection, which reads its row. Like the get-tuple-elements that multi
// output fusion adds, 64 users of the first sibling are not in the map. Each
// merge changes only the rows of the pair, so DeferReachabilityUpdateForMerge
// must apply it at once rather than make every later row access test it.
void BM_MergeManySiblings(benchmark::State& state) {
  const bool defer = state.range(0) != 0;
  const int num_siblings = state.range(1);
  constexpr int kUsersNotInMap = 64;
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  auto builder = HloComputation::Builder("many_siblings");
  HloInstruction* parameter = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "parameter"));
  std::vector<HloInstruction*> siblings;
  for (int i = 0; i < num_siblings; ++i) {
    siblings.push_back(builder.AddInstruction(
        HloInstruction::CreateUnary(r0f32, HloOpcode::kNegate, parameter)));
  }
  builder.AddInstruction(HloInstruction::CreateTuple(siblings));
  HloModule module("many_siblings", HloModuleConfig());
  HloComputation* computation = module.AddEntryComputation(builder.Build());
  std::unique_ptr<HloReachabilityMap> reachability;
  std::vector<HloInstruction*> users_not_in_map;
  for (auto s : state) {
    state.PauseTiming();
    reachability.reset();
    reachability = HloReachabilityMap::Build(computation);
    for (int i = 0; i < kUsersNotInMap; ++i) {
      users_not_in_map.push_back(computation->AddInstruction(
          HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, siblings[0])));
    }
    state.ResumeTiming();
    for (int i = 1; i < num_siblings; ++i) {
      benchmark::DoNotOptimize(
          reachability->IsConnected(siblings[0], siblings[i]));
      if (defer) {
        reachability->DeferReachabilityUpdateForMerge(siblings[i], siblings[0]);
      } else {
        reachability->UpdateReachabilityForMerge(siblings[i], siblings[0]);
      }
    }
    state.PauseTiming();
    for (HloInstruction* user : users_not_in_map) {
      CHECK_OK(computation->RemoveInstruction(user));
    }
    users_not_in_map.clear();
    state.ResumeTiming();
  }
}
BENCHMARK(BM_MergeManySiblings)->ArgsProduct({{0, 1}, {20000}})->Iterations(4);

}  // namespace

}  // namespace xla
