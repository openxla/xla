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

#include "xla/service/cpu/cpu_fusion_cost_model.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

class CpuFusionCostModelTest : public HloHardwareIndependentTestBase {
 protected:
  const HloInstruction* Find(const HloModule& module, absl::string_view name) {
    for (const HloComputation* computation : module.computations()) {
      for (const HloInstruction* instr : computation->instructions()) {
        if (instr->name() == name) return instr;
      }
    }
    return nullptr;
  }
};

TEST_F(CpuFusionCostModelTest, PerElementFlopsSeparatesCheapFromTranscendental) {
  constexpr absl::string_view kModule = R"(
HloModule m
ENTRY e {
  p = f32[8,8] parameter(0)
  cheap = f32[8,8] multiply(p, p)
  expensive = f32[8,8] exponential(p)
  ROOT r = f32[8,8] add(cheap, expensive)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  CpuFusionCostModel model{CpuFusionCostModel::Params{}};
  EXPECT_EQ(model.PerElementFlops(Find(*module, "cheap")), 1);
  EXPECT_GT(model.PerElementFlops(Find(*module, "expensive")),
            10 * model.PerElementFlops(Find(*module, "cheap")));
}

TEST_F(CpuFusionCostModelTest, PerElementFlopsOfReduceScalesWithReducedExtent) {
  constexpr absl::string_view kModule = R"(
HloModule m
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}
ENTRY e {
  p = f32[64,64,3] parameter(0)
  zero = f32[] constant(0)
  ROOT small = f32[64,64] reduce(p, zero), dimensions={2}, to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  CpuFusionCostModel model{CpuFusionCostModel::Params{}};
  // Three additions per output element, not one.
  EXPECT_EQ(model.PerElementFlops(Find(*module, "small")), 3);
}

// An intermediate that fits in cache never reaches DRAM, so there is no round
// trip to buy back and the model must decline to spend flops on it.
TEST_F(CpuFusionCostModelTest, SmallIntermediateIsNotWorthRecomputing) {
  constexpr absl::string_view kModule = R"(
HloModule m
ENTRY e {
  p = f32[64,64] parameter(0)
  s = f32[64,64] sqrt(p)
  b = f32[64,64,8] broadcast(s), dimensions={0,1}
  ROOT r = f32[64,64,8] multiply(b, b)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  CpuFusionCostModel model{CpuFusionCostModel::Params{}};
  EXPECT_FALSE(model.RecomputeBeatsMaterialize(Find(*module, "s")));
}

// The same graph above cache size: now the buffer is a real DRAM round trip,
// and one sqrt per extra read is far cheaper than paying it.
TEST_F(CpuFusionCostModelTest, LargeCheapIntermediateIsWorthRecomputing) {
  constexpr absl::string_view kModule = R"(
HloModule m
ENTRY e {
  p = f32[1024,1024] parameter(0)
  s = f32[1024,1024] sqrt(p)
  b = f32[1024,1024,8] broadcast(s), dimensions={0,1}
  ROOT r = f32[1024,1024,8] multiply(b, b)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  CpuFusionCostModel model{CpuFusionCostModel::Params{}};
  EXPECT_TRUE(model.RecomputeBeatsMaterialize(Find(*module, "s")));
}

// Same size, same reuse factor as the test above -- only the arithmetic per
// element differs. This is the pair no boolean "is this opcode expensive?"
// test and no byte threshold can separate, and the reason for the model.
TEST_F(CpuFusionCostModelTest, LargeExpensiveIntermediateIsWorthMaterializing) {
  constexpr absl::string_view kModule = R"(
HloModule m
expensive_body {
  x = f32[1024,1024] parameter(0)
  e0 = f32[1024,1024] exponential(x)
  e1 = f32[1024,1024] exponential(e0)
  e2 = f32[1024,1024] exponential(e1)
  e3 = f32[1024,1024] exponential(e2)
  e4 = f32[1024,1024] exponential(e3)
  e5 = f32[1024,1024] exponential(e4)
  e6 = f32[1024,1024] exponential(e5)
  ROOT e7 = f32[1024,1024] exponential(e6)
}
ENTRY e {
  p = f32[1024,1024] parameter(0)
  s = f32[1024,1024] fusion(p), kind=kLoop, calls=expensive_body
  b = f32[1024,1024,8] broadcast(s), dimensions={0,1}
  ROOT r = f32[1024,1024,8] multiply(b, b)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  CpuFusionCostModel model{CpuFusionCostModel::Params{}};
  EXPECT_GT(model.PerElementFlops(Find(*module, "s")), 100);
  EXPECT_FALSE(model.RecomputeBeatsMaterialize(Find(*module, "s")));
}

// The verdict is a property of the machine balance, so raising it flips the
// expensive case. This also pins that the tunable is actually consulted.
TEST_F(CpuFusionCostModelTest, RaisingMachineBalanceFlipsTheExpensiveCase) {
  constexpr absl::string_view kModule = R"(
HloModule m
expensive_body {
  x = f32[1024,1024] parameter(0)
  e0 = f32[1024,1024] exponential(x)
  e1 = f32[1024,1024] exponential(e0)
  e2 = f32[1024,1024] exponential(e1)
  e3 = f32[1024,1024] exponential(e2)
  e4 = f32[1024,1024] exponential(e3)
  e5 = f32[1024,1024] exponential(e4)
  e6 = f32[1024,1024] exponential(e5)
  ROOT e7 = f32[1024,1024] exponential(e6)
}
ENTRY e {
  p = f32[1024,1024] parameter(0)
  s = f32[1024,1024] fusion(p), kind=kLoop, calls=expensive_body
  b = f32[1024,1024,8] broadcast(s), dimensions={0,1}
  ROOT r = f32[1024,1024,8] multiply(b, b)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  CpuFusionCostModel::Params params;
  params.machine_balance_flops_per_byte = 1000;
  CpuFusionCostModel model{params};
  EXPECT_TRUE(model.RecomputeBeatsMaterialize(Find(*module, "s")));
}

// A single consumer duplicates nothing, so there is never anything to weigh:
// the answer must be yes regardless of how expensive the producer is.
TEST_F(CpuFusionCostModelTest, SingleConsumerIsAlwaysWorthRecomputing) {
  constexpr absl::string_view kModule = R"(
HloModule m
ENTRY e {
  p = f32[1024,1024] parameter(0)
  s = f32[1024,1024] exponential(p)
  ROOT r = f32[1024,1024] multiply(s, s)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  CpuFusionCostModel model{CpuFusionCostModel::Params{}};
  EXPECT_TRUE(model.RecomputeBeatsMaterialize(Find(*module, "s")));
}

// Contractions have no meaningful per-element cost; recomputing one per
// consumer element is never the intent.
TEST_F(CpuFusionCostModelTest, DotIsNeverWorthRecomputing) {
  constexpr absl::string_view kModule = R"(
HloModule m
ENTRY e {
  a = f32[1024,1024] parameter(0)
  b = f32[1024,1024] parameter(1)
  d = f32[1024,1024] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bc = f32[1024,1024,8] broadcast(d), dimensions={0,1}
  ROOT r = f32[1024,1024,8] multiply(bc, bc)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  CpuFusionCostModel model{CpuFusionCostModel::Params{}};
  EXPECT_FALSE(model.RecomputeBeatsMaterialize(Find(*module, "d")));
}

// The re-emission variant, used by FusionNodeIndexingEvaluation. One copy is
// no duplication at all, so it is always allowed; beyond that the same
// flops-against-bytes comparison decides. Compile-time blowup is not this
// model's concern -- kAllowedCodeDuplication caps the copy count before we
// are ever consulted.
TEST_F(CpuFusionCostModelTest, RematerializationRespectsCopyCount) {
  constexpr absl::string_view kModule = R"(
HloModule m
expensive_body {
  x = f32[1024,1024] parameter(0)
  e0 = f32[1024,1024] exponential(x)
  e1 = f32[1024,1024] exponential(e0)
  e2 = f32[1024,1024] exponential(e1)
  e3 = f32[1024,1024] exponential(e2)
  e4 = f32[1024,1024] exponential(e3)
  e5 = f32[1024,1024] exponential(e4)
  e6 = f32[1024,1024] exponential(e5)
  ROOT e7 = f32[1024,1024] exponential(e6)
}
ENTRY e {
  p = f32[1024,1024] parameter(0)
  cheap = f32[1024,1024] sqrt(p)
  costly = f32[1024,1024] fusion(p), kind=kLoop, calls=expensive_body
  ROOT r = f32[1024,1024] multiply(cheap, costly)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));
  CpuFusionCostModel model{CpuFusionCostModel::Params{}};
  const HloInstruction* cheap = Find(*module, "cheap");
  const HloInstruction* costly = Find(*module, "costly");
  // A single copy duplicates nothing.
  EXPECT_TRUE(model.RematerializationBeatsMaterialization(costly, 1));
  // A sqrt re-emitted four times is cheaper than a 4 MB round trip.
  EXPECT_TRUE(model.RematerializationBeatsMaterialization(cheap, 4));
  // Eight chained exponentials re-emitted four times are not.
  EXPECT_FALSE(model.RematerializationBeatsMaterialization(costly, 4));
}

}  // namespace
}  // namespace xla::cpu
