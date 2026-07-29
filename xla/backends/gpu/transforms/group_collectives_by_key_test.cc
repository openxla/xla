/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/backends/gpu/transforms/group_collectives_by_key.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/transforms/explicit_collectives_group_async_wrapper.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;

using GroupCollectivesByKeyTest = HloHardwareIndependentTestBase;

int CountOpcode(const HloModule& module, HloOpcode opcode) {
  int count = 0;
  for (const HloInstruction* instr :
       module.entry_computation()->instructions()) {
    if (instr->opcode() == opcode) {
      ++count;
    }
  }
  return count;
}

// Input mimics combiner output: multi-operand AG + tuple output + GTE.
TEST_F(GroupCollectivesByKeyTest, GroupsCombinedAgAndRs) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    w0 = f32[8,8] parameter(0)
    w1 = f32[8,8] parameter(1)
    g0 = f32[32,8] parameter(2)
    g1 = f32[32,8] parameter(3)
    ag = (f32[32,8], f32[32,8]) all-gather(w0, w1), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    ag0 = f32[32,8] get-tuple-element(ag), index=0
    ag1 = f32[32,8] get-tuple-element(ag), index=1
    rs = (f32[8,8], f32[8,8]) reduce-scatter(g0, g1), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    rs0 = f32[8,8] get-tuple-element(rs), index=0
    rs1 = f32[8,8] get-tuple-element(rs), index=1
    ROOT result = (f32[32,8], f32[32,8], f32[8,8], f32[8,8])
        tuple(ag0, ag1, rs0, rs1)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  // A single async group replaces both top-level collectives.
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAsyncStart), 1);
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAllGather), 0);
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kReduceScatter), 0);

  // The async-start/done carry the marker and the preserved key.
  const HloInstruction* async_start = nullptr;
  for (HloInstruction* instr : module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kAsyncStart) {
      async_start = instr;
    }
  }
  ASSERT_NE(async_start, nullptr);
  EXPECT_EQ(async_start->get_frontend_attribute("_collectives_group"), "");
  EXPECT_EQ(async_start->get_frontend_attribute("collective_group_key"), "g0");

  // The cloned collectives inside the group computation retain the key.
  const HloComputation* group_comp = async_start->async_wrapped_computation();
  ASSERT_NE(group_comp, nullptr);
  int inner_collectives = 0;
  for (const HloInstruction* instr : group_comp->instructions()) {
    if (instr->opcode() == HloOpcode::kAllGather ||
        instr->opcode() == HloOpcode::kReduceScatter) {
      ++inner_collectives;
      EXPECT_EQ(instr->get_frontend_attribute("collective_group_key"), "g0");
    }
  }
  EXPECT_EQ(inner_collectives, 2);
}

TEST_F(GroupCollectivesByKeyTest, ErrorsWhenDependencyExists) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    weights = f32[8,8] parameter(0)
    ag = f32[32,8] all-gather(weights), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    rs = f32[8,8] reduce-scatter(ag), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    ROOT result = f32[8,8] copy(rs)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass;
  EXPECT_THAT(pass.Run(module.get()).status(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       ::testing::AllOf(::testing::HasSubstr("ag"),
                                        ::testing::HasSubstr("rs"),
                                        ::testing::HasSubstr("g0"))));
}

// A later key's invalid dependency path runs through an earlier key's members.
// Keys are processed in sorted order, so "g0" (valid, independent) would be
// grouped before "g1" (invalid) is validated. Validation must therefore happen
// for every key against a single mutation-free reachability map: otherwise the
// "g1" diagnostic would walk users() of instructions whose "g0" neighbors have
// already been replaced by async/GTE nodes absent from the stale map (crash),
// and the "g0" group would be left half-transformed behind the error.
TEST_F(GroupCollectivesByKeyTest, ErrorsOnLaterKeyWithoutMutatingEarlierKey) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    w1 = f32[8,8] parameter(0)
    g = f32[32,8] parameter(1)
    // g1 member; feeds an all-gather that carries key g0.
    rs1 = f32[8,8] reduce-scatter(g), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g1"}
    // g0 members: independent of each other, so g0 is valid on its own. But
    // ag0a depends on rs1 and feeds ag1, so it sits on the g1 dependency path.
    ag0a = f32[32,8] all-gather(rs1), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    ag0b = f32[32,8] all-gather(w1), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    // g1 member reachable from rs1 through ag0a: g1 is not independent.
    ag1 = f32[128,8] all-gather(ag0a), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g1"}
    ROOT result = (f32[32,8], f32[128,8]) tuple(ag0b, ag1)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass;
  EXPECT_THAT(pass.Run(module.get()).status(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       ::testing::AllOf(::testing::HasSubstr("rs1"),
                                        ::testing::HasSubstr("ag1"),
                                        ::testing::HasSubstr("g1"))));

  // The earlier, valid key g0 must be left untouched: no async group was
  // formed and all three all-gathers survive.
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAsyncStart), 0);
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAllGather), 3);
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kReduceScatter), 1);
}

TEST_F(GroupCollectivesByKeyTest, SkipsUnpairedKeys) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    w1 = f32[8,8] parameter(0)
    g1 = f32[32,8] parameter(1)
    w2 = f32[8,8] parameter(2)
    g2 = f32[32,8] parameter(3)
    ag1 = f32[32,8] all-gather(w1), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    rs1 = f32[8,8] reduce-scatter(g1), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g1"}
    ag2 = f32[32,8] all-gather(w2), dimensions={0},
        replica_groups={{0,1,2,3}}
    rs2 = f32[8,8] reduce-scatter(g2), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add
    ROOT result = (f32[32,8], f32[8,8], f32[32,8], f32[8,8])
        tuple(ag1, rs1, ag2, rs2)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

// Unannotated collectives are never grouped.
TEST_F(GroupCollectivesByKeyTest, UnannotatedCollectivesUnchanged) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    w = f32[8,8] parameter(0)
    g = f32[32,8] parameter(1)
    ag = f32[32,8] all-gather(w), dimensions={0},
        replica_groups={{0,1,2,3}}
    rs = f32[8,8] reduce-scatter(g), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add
    ROOT result = (f32[32,8], f32[8,8]) tuple(ag, rs)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

// AG and RS may belong to different communicators (different replica_groups).
// NCCL's group call fuses collectives across comms, so the pass should still
// group them when the key matches.
TEST_F(GroupCollectivesByKeyTest, GroupsAcrossDifferentReplicaGroups) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    weights = f32[8,8] parameter(0)
    grads = f32[32,8] parameter(1)
    ag = f32[16,8] all-gather(weights), dimensions={0},
        replica_groups={{0,1},{2,3}},
        frontend_attributes={collective_group_key="g0"}
    rs = f32[8,8] reduce-scatter(grads), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[16,8], f32[8,8]) tuple(ag, rs)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAsyncStart), 1);
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAllGather), 0);
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kReduceScatter), 0);
}

// Two AllGathers with the same key must be paired even though neither is a
// ReduceScatter.
TEST_F(GroupCollectivesByKeyTest, GroupsTwoAllGathers) {
  const absl::string_view hlo_string = R"(
  HloModule test

  ENTRY main {
    w1 = f32[8,8] parameter(0)
    w2 = f32[8,8] parameter(1)
    ag1 = f32[16,8] all-gather(w1), dimensions={0},
        replica_groups={{0,1},{2,3}},
        frontend_attributes={collective_group_key="g0"}
    ag2 = f32[32,8] all-gather(w2), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[16,8], f32[32,8]) tuple(ag1, ag2)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAsyncStart), 1);
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAllGather), 0);
}

// Two AllReduces with the same key group together (all-reduce support).
TEST_F(GroupCollectivesByKeyTest, GroupsTwoAllReduces) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    x = f32[8,8] parameter(0)
    y = f32[8,8] parameter(1)
    ar1 = f32[8,8] all-reduce(x), replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    ar2 = f32[8,8] all-reduce(y), replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[8,8], f32[8,8]) tuple(ar1, ar2)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAsyncStart), 1);
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAllReduce), 0);
}

// A custom predicate narrows the eligible opcode set: here only all-gathers
// are groupable, so a same-key AG+RS pair is left ungrouped (RS is not a
// candidate, leaving the AG as an unpaired singleton).
TEST_F(GroupCollectivesByKeyTest, CustomPredicateRestrictsToAllGather) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    w = f32[8,8] parameter(0)
    g = f32[32,8] parameter(1)
    ag = f32[32,8] all-gather(w), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    rs = f32[8,8] reduce-scatter(g), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[32,8], f32[8,8]) tuple(ag, rs)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass(HloPredicateIsOp<HloOpcode::kAllGather>);
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);

  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAsyncStart), 0);
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAllGather), 1);
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kReduceScatter), 1);
}

// With the same all-gather-only predicate, two same-key all-gathers still
// group; the restriction only excludes non-matching opcodes.
TEST_F(GroupCollectivesByKeyTest, CustomPredicateStillGroupsMatchingOpcodes) {
  const absl::string_view hlo_string = R"(
  HloModule test

  ENTRY main {
    w1 = f32[8,8] parameter(0)
    w2 = f32[8,8] parameter(1)
    ag1 = f32[16,8] all-gather(w1), dimensions={0},
        replica_groups={{0,1},{2,3}},
        frontend_attributes={collective_group_key="g0"}
    ag2 = f32[32,8] all-gather(w2), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[16,8], f32[32,8]) tuple(ag1, ag2)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass(HloPredicateIsOp<HloOpcode::kAllGather>);
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAsyncStart), 1);
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAllGather), 0);
}

// Three independent collectives sharing the same key fuse into a single group,
// not a pair plus a singleton.
TEST_F(GroupCollectivesByKeyTest, GroupsThreeCollectives) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    w1 = f32[8,8] parameter(0)
    w2 = f32[8,8] parameter(1)
    g1 = f32[32,8] parameter(2)
    ag1 = f32[16,8] all-gather(w1), dimensions={0},
        replica_groups={{0,1},{2,3}},
        frontend_attributes={collective_group_key="g0"}
    ag2 = f32[32,8] all-gather(w2), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    rs1 = f32[8,8] reduce-scatter(g1), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[16,8], f32[32,8], f32[8,8]) tuple(ag1, ag2, rs1)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAsyncStart), 1);
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAllGather), 0);
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kReduceScatter), 0);
}

TEST_F(GroupCollectivesByKeyTest, MultipleKeyPairsGroupedIndependently) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    w1 = f32[8,8] parameter(0)
    g1 = f32[32,8] parameter(1)
    w2 = f32[8,8] parameter(2)
    g2 = f32[32,8] parameter(3)
    ag1 = f32[32,8] all-gather(w1), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    rs1 = f32[8,8] reduce-scatter(g1), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    ag2 = f32[32,8] all-gather(w2), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g1"}
    rs2 = f32[8,8] reduce-scatter(g2), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g1"}
    ROOT result = (f32[32,8], f32[8,8], f32[32,8], f32[8,8])
        tuple(ag1, rs1, ag2, rs2)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAsyncStart), 2);
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAllGather), 0);
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kReduceScatter), 0);
}

// External control dependencies are relayed onto the async pair.
TEST_F(GroupCollectivesByKeyTest, PreservesExternalControlDeps) {
  const absl::string_view hlo_string = R"(
  HloModule test

  ENTRY main {
    w1 = f32[8,8] parameter(0)
    w2 = f32[8,8] parameter(1)
    barrier = f32[8,8] add(w1, w1)
    ag1 = f32[16,8] all-gather(w1), dimensions={0},
        replica_groups={{0,1},{2,3}}, control-predecessors={barrier},
        frontend_attributes={collective_group_key="g0"}
    ag2 = f32[32,8] all-gather(w2), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[16,8], f32[32,8]) tuple(ag1, ag2)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* async_start = nullptr;
  const HloInstruction* barrier = nullptr;
  for (HloInstruction* instr : module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kAsyncStart) {
      async_start = instr;
    }
    if (instr->name() == "barrier") {
      barrier = instr;
    }
  }
  ASSERT_NE(async_start, nullptr);
  ASSERT_NE(barrier, nullptr);
  EXPECT_THAT(async_start->control_predecessors(),
              ::testing::Contains(barrier));
}

// Running the pass twice is idempotent: the second run makes no change.
TEST_F(GroupCollectivesByKeyTest, Idempotent) {
  const absl::string_view hlo_string = R"(
  HloModule test

  ENTRY main {
    w1 = f32[8,8] parameter(0)
    w2 = f32[8,8] parameter(1)
    ag1 = f32[16,8] all-gather(w1), dimensions={0},
        replica_groups={{0,1},{2,3}},
        frontend_attributes={collective_group_key="g0"}
    ag2 = f32[32,8] all-gather(w2), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[16,8], f32[32,8]) tuple(ag1, ag2)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(bool changed_again, pass.Run(module.get()));
  EXPECT_FALSE(changed_again);
}

// The pass output is consumed cleanly: the wrapper does not re-wrap the groups
// this pass already formed, leaving exactly the async pairs it created.
TEST_F(GroupCollectivesByKeyTest, EndToEndWithWrapper) {
  const absl::string_view hlo_string = R"(
  HloModule test

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT sum = f32[] add(a, b)
  }

  ENTRY main {
    w = f32[8,8] parameter(0)
    g = f32[32,8] parameter(1)
    ag = f32[32,8] all-gather(w), dimensions={0},
        replica_groups={{0,1,2,3}},
        frontend_attributes={collective_group_key="g0"}
    rs = f32[8,8] reduce-scatter(g), dimensions={0},
        replica_groups={{0,1,2,3}}, to_apply=add,
        frontend_attributes={collective_group_key="g0"}
    ROOT result = (f32[32,8], f32[8,8]) tuple(ag, rs)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  GroupCollectivesByKey group_pass;
  ASSERT_OK_AND_ASSIGN(bool grouped, group_pass.Run(module.get()));
  EXPECT_TRUE(grouped);

  ExplicitCollectivesGroupAsyncWrapper wrapper;
  ASSERT_OK_AND_ASSIGN(bool wrapped, wrapper.Run(module.get()));
  // The group is already an async pair, so the wrapper (which only rewrites
  // _collectives_group calls) has nothing to do.
  EXPECT_FALSE(wrapped);
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kAsyncStart), 1);
}

}  // namespace
}  // namespace xla::gpu
