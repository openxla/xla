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

#include "xla/hlo/transforms/collectives/collective_permute_equivariant_eliminator.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

std::vector<const HloCollectivePermuteInstruction*> CollectPermutes(
    const HloComputation* comp) {
  std::vector<const HloCollectivePermuteInstruction*> out;
  for (const HloInstruction* inst : comp->instructions()) {
    if (inst->opcode() == HloOpcode::kCollectivePermute) {
      out.push_back(Cast<HloCollectivePermuteInstruction>(inst));
    }
  }
  return out;
}

int64_t TotalPermuteBytes(const HloComputation* comp) {
  int64_t total = 0;
  for (const HloCollectivePermuteInstruction* cp : CollectPermutes(comp)) {
    total += ShapeUtil::ByteSizeOfPrimitiveType(cp->shape().element_type()) *
             ShapeUtil::ElementsIn(cp->shape());
  }
  return total;
}

class CollectivePermuteEquivariantEliminatorTest
    : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<std::unique_ptr<HloModule>> ParsePartitioned(
      absl::string_view hlo, int64_t num_partitions,
      int64_t replica_count = 1) {
    HloModuleConfig config = GetModuleConfigForTest(
        /*replica_count=*/replica_count, /*num_partitions=*/num_partitions);
    return ParseAndReturnVerifiedModule(hlo, config);
  }
};

// Coupled forward + backward involution island with fixed points {(0,0),(1,2),
// (2,1),(3,3)}, replicated entry parameter weights recorded via
// `spmd_parameters_shardings`, tuple-returning approved custom calls, and small
// reduced/converted gradient side-exits: cancels all 4 large activation CPs,
// adds zero CPs on replicated weights, absorbs reduction + bf16 narrowing into
// the conjugated set, and inserts only the 2 tiny gradient exit CPs.
TEST_F(CollectivePermuteEquivariantEliminatorTest,
       EliminatesCoupledForwardBackwardInvolutionIslandAtMinCut) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule turn_island

add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

ENTRY main {
  x_raw = f32[2,64,256] parameter(0)
  dy_raw = f32[2,64,128] parameter(1)
  w_mix = f32[4,128] parameter(2)
  w_scale = bf16[128] parameter(3)

  a2a_in = f32[2,64,256] all-to-all(x_raw), channel_id=1,
      replica_groups={{0,1},{2,3}}, dimensions={0}
  x = f32[128,256] reshape(a2a_in)
  cp_in = f32[128,256] collective-permute(x), channel_id=2,
      source_target_pairs={{0,0},{2,1},{1,2},{3,3}}

  mix_fwd = f32[128,128] custom-call(cp_in, w_mix),
      custom_call_target="pure_local_op"
  w_scale_f32 = f32[128] convert(w_scale)
  norm_fwd = f32[128,128] custom-call(mix_fwd, w_scale_f32),
      custom_call_target="pure_local_op"
  y_bf16 = bf16[128,128] convert(norm_fwd)
  cp_fwd_out = bf16[128,128] collective-permute(y_bf16), channel_id=3,
      source_target_pairs={{0,0},{1,2},{2,1},{3,3}}
  ag_fwd = bf16[256,128] all-gather(cp_fwd_out), channel_id=4,
      replica_groups={{0,1},{2,3}}, use_global_device_ids=true, dimensions={0}

  a2a_dy = f32[2,64,128] all-to-all(dy_raw), channel_id=5,
      replica_groups={{0,1},{2,3}}, dimensions={0}
  dy = f32[128,128] reshape(a2a_dy)
  cp_bwd_in = f32[128,128] collective-permute(dy), channel_id=6,
      source_target_pairs={{0,0},{2,1},{1,2},{3,3}}

  norm_bwd = (f32[128,128], f32[16,128]) custom-call(
      mix_fwd, cp_bwd_in, w_scale_f32), custom_call_target="pure_local_op"
  dnorm = f32[128,128] get-tuple-element(norm_bwd), index=0
  dscale_part = f32[16,128] get-tuple-element(norm_bwd), index=1
  zero = f32[] constant(0)
  dscale_f32 = f32[128] reduce(dscale_part, zero), dimensions={0},
      to_apply=add_f32
  dscale_bf16 = bf16[128] convert(dscale_f32)
  ar_dscale = bf16[128] all-reduce(dscale_bf16), channel_id=7,
      replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=add_f32

  mix_bwd = (f32[128,256], f32[256]) custom-call(dnorm, cp_in, w_mix),
      custom_call_target="pure_local_op"
  dx = f32[128,256] get-tuple-element(mix_bwd), index=0
  dw = f32[256] get-tuple-element(mix_bwd), index=1
  ar_dw = f32[256] all-reduce(dw), channel_id=8,
      replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=add_f32

  cp_bwd_out = f32[128,256] collective-permute(dx), channel_id=9,
      source_target_pairs={{0,0},{1,2},{2,1},{3,3}}
  dx_2 = f32[2,64,256] reshape(cp_bwd_out)
  a2a_out = f32[2,64,256] all-to-all(dx_2), channel_id=10,
      replica_groups={{0,1},{2,3}}, dimensions={0}

  ROOT out = (bf16[256,128], f32[2,64,256], f32[256], bf16[128])
      tuple(ag_fwd, a2a_out, ar_dw, ar_dscale)
}
)hlo";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParsePartitioned(kHlo, /*num_partitions=*/4));
  // Record shardings via `spmd_parameters_shardings` (no per-param sharding
  // attribute), matching post-ShardingRemover SPMD modules.
  module->set_spmd_parameters_shardings({
      HloSharding::IotaTile({4, 1, 1}),
      HloSharding::IotaTile({4, 1, 1}),
      HloSharding::Replicate(),
      HloSharding::Replicate(),
  });

  const int64_t before_bytes = TotalPermuteBytes(module->entry_computation());
  // 128*256*4 + 128*128*2 + 128*128*4 + 128*256*4 = 360,448 B.
  EXPECT_EQ(before_bytes, 360448);

  CollectivePermuteEquivariantEliminator pass([](const HloInstruction* inst) {
    return inst->IsCustomCall("pure_local_op");
  });
  TF_ASSERT_OK_AND_ASSIGN(const bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  const auto cps = CollectPermutes(module->entry_computation());
  ASSERT_EQ(cps.size(), 2);
  // Only dw (f32[256] = 1024 B) and dscale_bf16 (bf16[128] = 256 B) remain!
  EXPECT_EQ(TotalPermuteBytes(module->entry_computation()), 1280);
  EXPECT_EQ(FindInstruction(module.get(), "ar_dw")->operand(0)->opcode(),
            HloOpcode::kCollectivePermute);
  EXPECT_EQ(FindInstruction(module.get(), "ar_dscale")->operand(0)->opcode(),
            HloOpcode::kCollectivePermute);
  EXPECT_EQ(FindInstruction(module.get(), "ag_fwd")->operand(0),
            FindInstruction(module.get(), "y_bf16"));

  // Fixed-point idempotence: re-running makes no further change.
  TF_ASSERT_OK_AND_ASSIGN(const bool changed_again,
                          RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed_again);
}

// Non-involution 4-cycle P = 0->1->2->3->0: cancels CP_{P^{-1}} entry and CP_P
// exit while emitting CP_{P^{-1}} (NOT CP_P) on an unconjugated side-output.
TEST_F(CollectivePermuteEquivariantEliminatorTest,
       NonInvolutionFourCycleInsertsExactInverseOnSideExit) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule four_cycle

add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[256,256] parameter(0)
  // CP_{P^{-1}}: 0->3, 1->0, 2->1, 3->2.
  cp_in = f32[256,256] collective-permute(p0),
      source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
  c = f32[] constant(2.0)
  b = f32[256,256] broadcast(c), dimensions={}
  scaled = f32[256,256] multiply(cp_in, b)
  zero = f32[] constant(0.0)
  side = f32[256] reduce(scaled, zero), dimensions={0}, to_apply=add_f32
  // CP_P: 0->1, 1->2, 2->3, 3->0.
  cp_out = f32[256,256] collective-permute(scaled),
      source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
  ROOT out = (f32[256,256], f32[256]) tuple(cp_out, side)
}
)hlo";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParsePartitioned(kHlo, /*num_partitions=*/1, /*replica_count=*/4));
  CollectivePermuteEquivariantEliminator pass;
  TF_ASSERT_OK_AND_ASSIGN(const bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  const auto cps = CollectPermutes(module->entry_computation());
  ASSERT_EQ(cps.size(), 1);
  EXPECT_EQ(cps[0]->operand(0), FindInstruction(module.get(), "side"));
  const std::vector<std::pair<int64_t, int64_t>> kExpectedPInv = {
      {0, 3}, {1, 0}, {2, 1}, {3, 2}};
  EXPECT_EQ(cps[0]->source_target_pairs(), kExpectedPInv);
  EXPECT_FALSE(cps[0]->channel_id().has_value());
}

// Partial-tile `ReplicateOnLastTileDim` sharding is treated as P-invariant iff
// P swaps only within replication groups; non-invariant partial shardings pay
// an entry CP and are never silently skipped.
TEST_F(CollectivePermuteEquivariantEliminatorTest,
       DistinguishesInvariantAndNonInvariantPartialTileShardings) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule partial_tile

ENTRY main {
  x = f32[128,128] parameter(0)
  w = f32[128] parameter(1)
  cp_in = f32[128,128] collective-permute(x), channel_id=1,
      source_target_pairs={{0,1},{1,0},{2,3},{3,2}}
  wb = f32[128,128] broadcast(w), dimensions={1}
  sum = f32[128,128] add(cp_in, wb)
  ROOT cp_out = f32[128,128] collective-permute(sum), channel_id=2,
      source_target_pairs={{0,1},{1,0},{2,3},{3,2}}
}
)hlo";

  // Case 1: w has {devices=[2,2]<=[4] last_tile_dim_replicate}: (0,1) share
  // tile 0 and (2,3) share tile 1 -> invariant under (0<->1)(2<->3) -> 0 CPs!
  {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParsePartitioned(kHlo, /*num_partitions=*/4));
    Array<int64_t> inv_tiles({2, 2}, {0, 1, 2, 3});
    module->set_spmd_parameters_shardings({
        HloSharding::IotaTile({4, 1}),
        HloSharding::PartialTile(inv_tiles),
    });
    CollectivePermuteEquivariantEliminator pass;
    TF_ASSERT_OK_AND_ASSIGN(const bool changed,
                            RunHloPass(&pass, module.get()));
    EXPECT_TRUE(changed);
    EXPECT_TRUE(CollectPermutes(module->entry_computation()).empty());
  }

  // Case 2: w has {devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}: (0,2)
  // share tile 0 and (1,3) share tile 1 -> NOT invariant under (0<->1)(2<->3)
  // -> must insert CP_P(w) (f32[128]) while still cancelling both f32[128,128]!
  {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParsePartitioned(kHlo, /*num_partitions=*/4));
    Array<int64_t> non_inv_tiles({2, 2}, {0, 2, 1, 3});
    module->set_spmd_parameters_shardings({
        HloSharding::IotaTile({4, 1}),
        HloSharding::PartialTile(non_inv_tiles),
    });
    CollectivePermuteEquivariantEliminator pass;
    TF_ASSERT_OK_AND_ASSIGN(const bool changed,
                            RunHloPass(&pass, module.get()));
    EXPECT_TRUE(changed);
    const auto cps = CollectPermutes(module->entry_computation());
    ASSERT_EQ(cps.size(), 1);
    EXPECT_EQ(cps[0]->operand(0), FindInstruction(module.get(), "w"));
    EXPECT_TRUE(cps[0]->channel_id().has_value());
  }
}

// Orthogonal collective (ordered groups closed under P) is admitted into E_P;
// non-orthogonal collective and kCrossReplicaAndPartition all-reduce are pinned
// outside E_P to preserve bitwise FP reduction order.
TEST_F(CollectivePermuteEquivariantEliminatorTest,
       HandlesOrthogonalAndNonOrthogonalCollectives) {
  // P swaps (0<->2)(1<->3): maps {0,1} -> {2,3} and {2,3} -> {0,1} in exact
  // element order -> orthogonal!
  constexpr absl::string_view kOrthogonalHlo = R"hlo(
HloModule orthogonal_ar

add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

ENTRY main {
  x = f32[64,64] parameter(0)
  cp1 = f32[64,64] collective-permute(x), channel_id=1,
      source_target_pairs={{0,2},{1,3},{2,0},{3,1}}
  ar = f32[64,64] all-reduce(cp1), channel_id=2,
      replica_groups={{0,1},{2,3}}, use_global_device_ids=true, to_apply=add_f32
  ROOT cp2 = f32[64,64] collective-permute(ar), channel_id=3,
      source_target_pairs={{0,2},{1,3},{2,0},{3,1}}
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> orth_mod,
      ParsePartitioned(kOrthogonalHlo, /*num_partitions=*/4));
  CollectivePermuteEquivariantEliminator pass;
  TF_ASSERT_OK_AND_ASSIGN(const bool orth_changed,
                          RunHloPass(&pass, orth_mod.get()));
  EXPECT_TRUE(orth_changed);
  EXPECT_TRUE(CollectPermutes(orth_mod->entry_computation()).empty());

  // Reversed element order {0,1} and {3,2}: P({0,1}) = {2,3} != {3,2} -> NOT
  // order-preserving -> must NOT eliminate!
  constexpr absl::string_view kReversedOrderHlo = R"hlo(
HloModule reversed_order_ar

add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

ENTRY main {
  x = f32[64,64] parameter(0)
  cp1 = f32[64,64] collective-permute(x), channel_id=1,
      source_target_pairs={{0,2},{1,3},{2,0},{3,1}}
  ar = f32[64,64] all-reduce(cp1), channel_id=2,
      replica_groups={{0,1},{3,2}}, use_global_device_ids=true, to_apply=add_f32
  ROOT cp2 = f32[64,64] collective-permute(ar), channel_id=3,
      source_target_pairs={{0,2},{1,3},{2,0},{3,1}}
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> rev_mod,
      ParsePartitioned(kReversedOrderHlo, /*num_partitions=*/4));
  TF_ASSERT_OK_AND_ASSIGN(const bool rev_changed,
                          RunHloPass(&pass, rev_mod.get()));
  EXPECT_FALSE(rev_changed);
  EXPECT_EQ(CollectPermutes(rev_mod->entry_computation()).size(), 2);

  // kCrossReplicaAndPartition (channel_id without use_global_device_ids):
  // reduces across all partitions in fixed partition order -> must NOT fold!
  constexpr absl::string_view kCrossRepAndPartHlo = R"hlo(
HloModule cross_rep_and_part_ar

add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

ENTRY main {
  x = f32[64,64] parameter(0)
  cp1 = f32[64,64] collective-permute(x), channel_id=1,
      source_target_pairs={{0,2},{1,3},{2,0},{3,1}}
  ar = f32[64,64] all-reduce(cp1), channel_id=2,
      replica_groups={{0}}, to_apply=add_f32
  ROOT cp2 = f32[64,64] collective-permute(ar), channel_id=3,
      source_target_pairs={{0,2},{1,3},{2,0},{3,1}}
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> crp_mod,
      ParsePartitioned(kCrossRepAndPartHlo, /*num_partitions=*/4));
  TF_ASSERT_OK_AND_ASSIGN(const bool crp_changed,
                          RunHloPass(&pass, crp_mod.get()));
  EXPECT_FALSE(crp_changed);
}

// Rejects partial (non-bijection) CPs, rank-dependent partition-id ops, and
// unapproved custom calls.
TEST_F(CollectivePermuteEquivariantEliminatorTest,
       RejectsPartialBijectionAndRankDependentOrOpaqueOps) {
  // Missing fixed points (0,0) and (3,3): zero-fills ranks 0 and 3!
  constexpr absl::string_view kPartialHlo = R"hlo(
HloModule partial_cp

ENTRY main {
  x = f32[64,64] parameter(0)
  cp1 = f32[64,64] collective-permute(x), channel_id=1,
      source_target_pairs={{1,2},{2,1}}
  neg = f32[64,64] negate(cp1)
  ROOT cp2 = f32[64,64] collective-permute(neg), channel_id=2,
      source_target_pairs={{1,2},{2,1}}
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> partial_mod,
                          ParsePartitioned(kPartialHlo, /*num_partitions=*/4));
  CollectivePermuteEquivariantEliminator pass;
  TF_ASSERT_OK_AND_ASSIGN(const bool c1, RunHloPass(&pass, partial_mod.get()));
  EXPECT_FALSE(c1);

  // partition-id is pinned outside S*: it must never be treated as P-invariant,
  // so conjugating across `sum` must explicitly permute `pid` with CP_P(pid)
  // rather than dropping all CPs (and an RNG op of full shape cannot be folded
  // to 0 CPs either).
  constexpr absl::string_view kPidHlo = R"hlo(
HloModule pid_cp

ENTRY main {
  x = f32[64,64] parameter(0)
  cp1 = f32[64,64] collective-permute(x), channel_id=1,
      source_target_pairs={{0,1},{1,0}}
  pid = u32[] partition-id()
  pid_f = f32[] convert(pid)
  pid_b = f32[64,64] broadcast(pid_f), dimensions={}
  sum = f32[64,64] add(cp1, pid_b)
  ROOT cp2 = f32[64,64] collective-permute(sum), channel_id=2,
      source_target_pairs={{0,1},{1,0}}
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> pid_mod,
                          ParsePartitioned(kPidHlo, /*num_partitions=*/2));
  TF_ASSERT_OK_AND_ASSIGN(const bool c2, RunHloPass(&pass, pid_mod.get()));
  EXPECT_TRUE(c2);
  const auto pid_cps = CollectPermutes(pid_mod->entry_computation());
  ASSERT_EQ(pid_cps.size(), 1);
  EXPECT_EQ(pid_cps[0]->operand(0), FindInstruction(pid_mod.get(), "pid_f"));

  // Non-deterministic RNG inside a fusion breaks rank equivariance.
  constexpr absl::string_view kRngHlo = R"hlo(
HloModule rng_cp

ENTRY main {
  x = f32[64,64] parameter(0)
  cp1 = f32[64,64] collective-permute(x), channel_id=1,
      source_target_pairs={{0,1},{1,0}}
  lo = f32[] constant(0.0)
  hi = f32[] constant(1.0)
  r = f32[64,64] rng(lo, hi), distribution=rng_uniform
  ROOT cp2 = f32[64,64] collective-permute(r), channel_id=2,
      source_target_pairs={{0,1},{1,0}}
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> rng_mod,
                          ParsePartitioned(kRngHlo, /*num_partitions=*/2));
  TF_ASSERT_OK_AND_ASSIGN(const bool c2b, RunHloPass(&pass, rng_mod.get()));
  EXPECT_FALSE(c2b);

  // Unapproved custom call breaks rank equivariance.
  constexpr absl::string_view kOpaqueCcHlo = R"hlo(
HloModule opaque_cc

ENTRY main {
  x = f32[64,64] parameter(0)
  cp1 = f32[64,64] collective-permute(x), channel_id=1,
      source_target_pairs={{0,1},{1,0}}
  cc = f32[64,64] custom-call(cp1), custom_call_target="unknown_op"
  ROOT cp2 = f32[64,64] collective-permute(cc), channel_id=2,
      source_target_pairs={{0,1},{1,0}}
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> cc_mod,
                          ParsePartitioned(kOpaqueCcHlo, /*num_partitions=*/2));
  TF_ASSERT_OK_AND_ASSIGN(const bool c3, RunHloPass(&pass, cc_mod.get()));
  EXPECT_FALSE(c3);
}

// Preserves explicit layouts on inserted boundary CPs and updates surviving
// tiled HloShardings on conjugated instructions by d -> P[d].
TEST_F(CollectivePermuteEquivariantEliminatorTest,
       PreservesExplicitLayoutsAndPermutesSurvivingShardings) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule layout_and_sharding

add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

ENTRY main {
  x = f32[64,128]{0,1} parameter(0)
  cp_in = f32[64,128]{0,1} collective-permute(x), channel_id=1,
      source_target_pairs={{0,0},{1,2},{2,1},{3,3}}
  act = f32[64,128]{0,1} tanh(cp_in), sharding={devices=[4,1]<=[4]}
  zero = f32[] constant(0.0)
  red = f32[64]{0} reduce(act, zero), dimensions={1}, to_apply=add_f32,
      sharding={devices=[4]<=[4]}
  cp_out = f32[64,128]{1,0} collective-permute(act), channel_id=2,
      source_target_pairs={{0,0},{1,2},{2,1},{3,3}}
  ROOT out = (f32[64,128]{1,0}, f32[64]{0}) tuple(cp_out, red)
}
)hlo";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParsePartitioned(kHlo, /*num_partitions=*/4));
  CollectivePermuteEquivariantEliminator pass;
  TF_ASSERT_OK_AND_ASSIGN(const bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  HloVerifier layout_verifier(/*layout_sensitive=*/true,
                              /*allow_mixed_precision=*/false);
  TF_ASSERT_OK(layout_verifier.Run(module.get()).status());

  const HloInstruction* act = FindInstruction(module.get(), "act");
  ASSERT_TRUE(act->has_sharding());
  Array<int64_t> expected_tiles({4, 1}, {0, 2, 1, 3});
  EXPECT_EQ(act->sharding(), HloSharding::Tile(expected_tiles));

  const auto cps = CollectPermutes(module->entry_computation());
  ASSERT_EQ(cps.size(), 1);
  EXPECT_EQ(cps[0]->operand(0), FindInstruction(module.get(), "red"));
  EXPECT_EQ(cps[0]->shape().layout(),
            FindInstruction(module.get(), "red")->shape().layout());
}

// One-sided byte shrinkage shrinks CP(broadcast(small)) -> broadcast(CP(small))
// and reduce(CP(large)) -> CP(reduce(large)), and leaves an isolated single CP
// untouched when no byte reduction exists.
TEST_F(CollectivePermuteEquivariantEliminatorTest,
       ShrinksOneSidedBroadcastAndReduceAndSkipsEqualCost) {
  constexpr absl::string_view kBroadcastHlo = R"hlo(
HloModule shrink_bcast

ENTRY main {
  small = f32[64] parameter(0)
  big = f32[256,64] broadcast(small), dimensions={1}
  ROOT cp = f32[256,64] collective-permute(big), channel_id=1,
      source_target_pairs={{0,1},{1,0}}
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> bcast_mod,
      ParsePartitioned(kBroadcastHlo, /*num_partitions=*/2));
  CollectivePermuteEquivariantEliminator pass;
  TF_ASSERT_OK_AND_ASSIGN(const bool c1, RunHloPass(&pass, bcast_mod.get()));
  EXPECT_TRUE(c1);
  const auto bcast_cps = CollectPermutes(bcast_mod->entry_computation());
  ASSERT_EQ(bcast_cps.size(), 1);
  EXPECT_EQ(bcast_cps[0]->shape(), ShapeUtil::MakeShape(F32, {64}));

  constexpr absl::string_view kReduceHlo = R"hlo(
HloModule shrink_reduce

add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

ENTRY main {
  big = f32[256,64] parameter(0)
  cp = f32[256,64] collective-permute(big), channel_id=1,
      source_target_pairs={{0,1},{1,0}}
  zero = f32[] constant(0.0)
  ROOT small = f32[64] reduce(cp, zero), dimensions={0}, to_apply=add_f32
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> red_mod,
                          ParsePartitioned(kReduceHlo, /*num_partitions=*/2));
  TF_ASSERT_OK_AND_ASSIGN(const bool c2, RunHloPass(&pass, red_mod.get()));
  EXPECT_TRUE(c2);
  const auto red_cps = CollectPermutes(red_mod->entry_computation());
  ASSERT_EQ(red_cps.size(), 1);
  EXPECT_EQ(red_cps[0]->shape(), ShapeUtil::MakeShape(F32, {64}));
  EXPECT_EQ(red_mod->entry_computation()->root_instruction(), red_cps[0]);

  constexpr absl::string_view kEqualCostHlo = R"hlo(
HloModule equal_cost

ENTRY main {
  x = f32[64,64] parameter(0)
  cp = f32[64,64] collective-permute(x), channel_id=1,
      source_target_pairs={{0,1},{1,0}}
  ROOT y = f32[64,64] tanh(cp)
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> eq_mod,
      ParsePartitioned(kEqualCostHlo, /*num_partitions=*/2));
  TF_ASSERT_OK_AND_ASSIGN(const bool c3, RunHloPass(&pass, eq_mod.get()));
  EXPECT_FALSE(c3);
}

}  // namespace
}  // namespace xla
