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

#include "xla/hlo/transforms/expanders/rng_expander.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/named_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_assignment.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

using RngExpanderTest = HloHardwareIndependentTestBase;

int CountOpcode(const HloModule& module, HloOpcode opcode) {
  int count = 0;
  for (const HloComputation* comp : module.computations()) {
    for (const HloInstruction* inst : comp->instructions()) {
      if (inst->opcode() == opcode) {
        ++count;
      }
    }
  }
  return count;
}

const HloRngGetAndUpdateStateInstruction* FindStateInst(
    const HloModule& module) {
  for (const HloComputation* comp : module.computations()) {
    for (const HloInstruction* inst : comp->instructions()) {
      if (inst->opcode() == HloOpcode::kRngGetAndUpdateState) {
        return static_cast<const HloRngGetAndUpdateStateInstruction*>(inst);
      }
    }
  }
  return nullptr;
}

absl::StatusOr<Literal> EvaluateExpandedRngModule(HloModule* module,
                                                  uint64_t seed = 0x12345678ULL,
                                                  uint64_t counter = 7) {
  std::unique_ptr<HloModule> clone = module->Clone();
  for (HloComputation* comp : clone->computations()) {
    std::vector<HloInstruction*> states;
    for (HloInstruction* inst : comp->instructions()) {
      if (inst->opcode() == HloOpcode::kRngGetAndUpdateState) {
        states.push_back(inst);
      }
    }
    for (HloInstruction* st : states) {
      HloInstruction* fixed = comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR1<uint64_t>(
              {counter, counter ^ 0x9E3779B9ULL})));
      counter += static_cast<uint64_t>(
          static_cast<HloRngGetAndUpdateStateInstruction*>(st)->delta());
      ABSL_RETURN_IF_ERROR(comp->ReplaceInstruction(st, fixed));
    }
  }
  HloEvaluator evaluator;
  evaluator.set_custom_call_handler(
      [seed](
          const HloInstruction* cc,
          absl::Span<const Literal*> /*operands*/) -> absl::StatusOr<Literal> {
        if (cc->custom_call_target() == "GetRngSeed") {
          return LiteralUtil::CreateR0<uint64_t>(seed);
        }
        return absl::UnimplementedError("Unexpected custom call.");
      });
  return evaluator.Evaluate(*clone, {});
}

TEST_F(RngExpanderTest, ReplacesRngWithGetRngSeedCustomCall) {
  const char* const hlo_string = R"(
    HloModule m
    ENTRY entry {
      min = f32[] constant(0)
      max = f32[] constant(1)
      ROOT result = f32[2,4,8] rng(min, max), distribution=rng_uniform
    })";

  const char* const expected = R"(
    // CHECK: %[[rng_comp:[^ ]+]] ({{.*}}) -> f32[2,4,8] {
    // CHECK-DAG:   %[[op_seed:[^ ]+]] = u64[] parameter(0)
    // CHECK-DAG:   %[[state:[^ ]+]] = u64[2]{0} parameter(1)
    // CHECK-DAG:   %[[seed:[^ ]+]] = u64[] custom-call(), custom_call_target="GetRngSeed", frontend_attributes={_xla_cse_safe_zero_operand="true"}
    // CHECK:   %[[xor:[^ ]+]] = u64[] xor(%[[seed]], %[[op_seed]])
    // CHECK:   ROOT %[[result:[^ ]+]] = f32[2,4,8]{2,1,0}
    // CHECK: }

    // CHECK-LABEL: ENTRY %entry
    // CHECK-DAG:   %[[min:[^ ]+]] = f32[] constant(0)
    // CHECK-DAG:   %[[max:[^ ]+]] = f32[] constant(1)
    // CHECK-DAG:   %[[op_seed_const:[^ ]+]] = u64[] constant({{[0-9]+}})
    // CHECK-DAG:   %[[state_inst:[^ ]+]] = u64[2]{0} rng-get-and-update-state(){{.*}}
    // CHECK:   ROOT %[[call:[^ ]+]] = f32[2,4,8]{2,1,0} call(%[[op_seed_const]], %[[state_inst]], %[[min]], %[[max]]), to_apply=%[[rng_comp]]
  )";

  RunAndFilecheckHloRewrite(hlo_string, RngExpander{}, expected);
}

TEST_F(RngExpanderTest, ReusingExpanderAcrossModulesConfigsAndErrorsIsSafe) {
  constexpr absl::string_view kSingleHlo = R"(
    HloModule m
    ENTRY main {
      lo = f32[] constant(0)
      hi = f32[] constant(1)
      ROOT r = f32[8] rng(lo, hi), distribution=rng_uniform
    })";
  RngExpander expander;

  // Destroy module 1 before running the same expander on module 2 (UAF repro).
  {
    ASSERT_OK_AND_ASSIGN(auto m1, ParseAndReturnVerifiedModule(kSingleHlo));
    EXPECT_THAT(expander.Run(m1.get()), IsOkAndHolds(true));
  }
  ASSERT_OK_AND_ASSIGN(auto m2, ParseAndReturnVerifiedModule(kSingleHlo));
  EXPECT_THAT(expander.Run(m2.get()), IsOkAndHolds(true));
  EXPECT_THAT(HloVerifier(HloVerifierOpts{}).Run(m2.get()), IsOk());
  const HloInstruction* m2_call = m2->entry_computation()->root_instruction();
  ASSERT_EQ(m2_call->opcode(), HloOpcode::kCall);
  EXPECT_EQ(m2_call->to_apply()->parent(), m2.get());

  // Reusing across a second live module must not point into m2, while identical
  // kRng ops within m3 still share one callee with distinct op_seed constants.
  ASSERT_OK_AND_ASSIGN(auto m3, ParseAndReturnVerifiedModule(R"(
    HloModule m3
    ENTRY main {
      lo = f32[] constant(0)
      hi = f32[] constant(1)
      a = f32[8] rng(lo, hi), distribution=rng_uniform
      b = f32[8] rng(lo, hi), distribution=rng_uniform
      ROOT t = (f32[8], f32[8]) tuple(a, b)
    })"));
  EXPECT_THAT(expander.Run(m3.get()), IsOkAndHolds(true));
  EXPECT_THAT(HloVerifier(HloVerifierOpts{}).Run(m3.get()), IsOk());
  const HloInstruction* call_a =
      m3->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* call_b =
      m3->entry_computation()->root_instruction()->operand(1);
  EXPECT_EQ(call_a->to_apply(), call_b->to_apply());
  EXPECT_EQ(call_a->to_apply()->parent(), m3.get());
  EXPECT_NE(call_a->to_apply(), m2_call->to_apply());
  EXPECT_NE(call_a->operand(0)->literal(), call_b->operand(0)->literal());

  // Replica and partition key mixing must follow each module's own config.
  HloModuleConfig rep_cfg = GetModuleConfigForTest(/*replica_count=*/4);
  ASSERT_OK_AND_ASSIGN(auto m_rep,
                       ParseAndReturnVerifiedModule(kSingleHlo, rep_cfg));
  EXPECT_THAT(expander.Run(m_rep.get()), IsOkAndHolds(true));
  EXPECT_EQ(CountOpcode(*m_rep, HloOpcode::kReplicaId), 1);
  EXPECT_EQ(CountOpcode(*m_rep, HloOpcode::kPartitionId), 0);

  HloModuleConfig part_cfg =
      GetModuleConfigForTest(/*replica_count=*/1, /*num_partitions=*/4);
  part_cfg.set_use_spmd_partitioning(false);
  ASSERT_OK_AND_ASSIGN(auto m_part,
                       ParseAndReturnVerifiedModule(kSingleHlo, part_cfg));
  EXPECT_THAT(expander.Run(m_part.get()), IsOkAndHolds(true));
  EXPECT_EQ(CountOpcode(*m_part, HloOpcode::kReplicaId), 0);
  EXPECT_EQ(CountOpcode(*m_part, HloOpcode::kPartitionId), 1);

  // A failed expansion must never leave a null cache entry behind.
  ASSERT_OK_AND_ASSIGN(auto m_bad, ParseAndReturnUnverifiedModule(R"(
    HloModule bad
    ENTRY main {
      lo = s32[] constant(0)
      hi = s32[] constant(1)
      ROOT r = s32[8] rng(lo, hi), distribution=rng_normal
    })"));
  EXPECT_THAT(expander.Run(m_bad.get()),
              StatusIs(absl::StatusCode::kInvalidArgument));
  ASSERT_OK_AND_ASSIGN(auto m_after, ParseAndReturnVerifiedModule(kSingleHlo));
  EXPECT_THAT(expander.Run(m_after.get()), IsOkAndHolds(true));
  EXPECT_THAT(HloVerifier(HloVerifierOpts{}).Run(m_after.get()), IsOk());
}

TEST_F(RngExpanderTest, NonMainExecutionThreadSetsCalleeThreadAndIsolates) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule threaded
    async_comp {
      lo = f32[] constant(0)
      hi = f32[] constant(1)
      ROOT r = f32[8] rng(lo, hi), distribution=rng_uniform
    }, execution_thread="async_thread"
    ENTRY main {
      start = ((), f32[8], u32[]) async-start(), calls=async_comp,
          async_execution_thread="async_thread"
      async_out = f32[8] async-done(start)
      lo = f32[] constant(0)
      hi = f32[] constant(1)
      main_out = f32[8] rng(lo, hi), distribution=rng_uniform
      ROOT sum = f32[8] add(async_out, main_out)
    })"));
  RngExpander expander;
  EXPECT_THAT(expander.Run(module.get(), {"main", "async_thread"}),
              IsOkAndHolds(true));
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kRng), 0);
  EXPECT_THAT(
      HloVerifier(HloVerifierOpts{}.VerifyCallNestedComputationThreadName())
          .Run(module.get()),
      IsOk());
  const HloComputation* async_comp =
      module->GetComputationWithName("async_comp");
  ASSERT_NE(async_comp, nullptr);
  const HloInstruction* async_call = async_comp->root_instruction();
  ASSERT_EQ(async_call->opcode(), HloOpcode::kCall);
  EXPECT_EQ(async_call->to_apply()->execution_thread(), "async_thread");
}

TEST_F(RngExpanderTest, ExpandsSub32BitIntegerUniformRngStrictlyInBounds) {
  struct IntCase {
    PrimitiveType ty;
    int64_t lo;
    int64_t hi;
  };
  const std::vector<IntCase> kCases = {
      {S1, -1, 0},    {U1, 0, 1},        {S2, -2, 1},    {U2, 0, 3},
      {S4, -7, 7},    {U4, 3, 15},       {S8, -120, 95}, {U8, 10, 250},
      {S16, -500, 5}, {U16, 100, 60000},
  };
  RngExpander expander;
  for (const IntCase& tc : kCases) {
    const std::string tname = primitive_util::LowercasePrimitiveTypeName(tc.ty);
    const std::string hlo =
        absl::StrCat("HloModule int_", tname, "\nENTRY main {\n  lo = ", tname,
                     "[] constant(", tc.lo, ")\n  hi = ", tname, "[] constant(",
                     tc.hi, ")\n  ROOT r = ", tname,
                     "[64] rng(lo, hi), distribution=rng_uniform\n}");
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    EXPECT_THAT(expander.Run(module.get()), IsOkAndHolds(true)) << tname;
    EXPECT_EQ(CountOpcode(*module, HloOpcode::kRng), 0) << tname;
    EXPECT_THAT(HloVerifier(HloVerifierOpts{}).Run(module.get()), IsOk());
    const auto* st = FindStateInst(*module);
    ASSERT_NE(st, nullptr);
    EXPECT_EQ(st->delta(), 64) << tname;

    ASSERT_OK_AND_ASSIGN(Literal out, EvaluateExpandedRngModule(module.get()));
    ASSERT_OK_AND_ASSIGN(Literal as_s64, out.Convert(S64));
    bool saw_distinct = false;
    const int64_t first = as_s64.data<int64_t>()[0];
    for (int64_t v : as_s64.data<int64_t>()) {
      EXPECT_GE(v, tc.lo) << tname;
      EXPECT_LT(v, tc.hi) << tname;
      saw_distinct |= (v != first);
    }
    if (tc.hi - tc.lo > 1) {
      EXPECT_TRUE(saw_distinct) << tname;
    }

    // Integer RNG_NORMAL must return clean InvalidArgument without crashing.
    const std::string bad_hlo = absl::StrCat(
        "HloModule bad_", tname, "\nENTRY main {\n  lo = ", tname,
        "[] constant(0)\n  hi = ", tname, "[] constant(0)\n  ROOT r = ", tname,
        "[8] rng(lo, hi), distribution=rng_normal\n}");
    ASSERT_OK_AND_ASSIGN(auto bad_mod, ParseAndReturnUnverifiedModule(bad_hlo));
    EXPECT_THAT(expander.Run(bad_mod.get()),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << tname;
  }
}

TEST_F(RngExpanderTest, ExpandsPredUniformRngAcrossAllBoundPairs) {
  RngExpander expander;
  for (bool lo : {false, true}) {
    for (bool hi : {false, true}) {
      const std::string hlo = absl::StrCat(
          "HloModule pred_rng\nENTRY main {\n  lo = pred[] constant(",
          lo ? "true" : "false", ")\n  hi = pred[] constant(",
          hi ? "true" : "false",
          ")\n  ROOT r = pred[128] rng(lo, hi), distribution=rng_uniform\n}");
      ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
      EXPECT_THAT(expander.Run(module.get()), IsOkAndHolds(true));
      EXPECT_EQ(CountOpcode(*module, HloOpcode::kRng), 0);
      EXPECT_THAT(HloVerifier(HloVerifierOpts{}).Run(module.get()), IsOk());
      ASSERT_OK_AND_ASSIGN(Literal out,
                           EvaluateExpandedRngModule(module.get()));
      int trues = 0;
      for (bool v : out.data<bool>()) {
        trues += v ? 1 : 0;
      }
      if (!lo && hi) {
        EXPECT_GT(trues, 20);
        EXPECT_LT(trues, 108);
      } else {
        EXPECT_EQ(trues, lo ? 128 : 0);
      }
    }
  }
  ASSERT_OK_AND_ASSIGN(auto bad_pred, ParseAndReturnUnverifiedModule(R"(
    HloModule bad_pred
    ENTRY main {
      lo = pred[] constant(false)
      hi = pred[] constant(true)
      ROOT r = pred[4] rng(lo, hi), distribution=rng_normal
    })"));
  EXPECT_THAT(expander.Run(bad_pred.get()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(RngExpanderTest, ExpandsAllSub32BitFloatingPointTypesUniformAndNormal) {
  const std::vector<PrimitiveType> kNarrowFps = {
      F4E2M1FN, F6E2M3FN,      F6E3M2FN,   F8E5M2,     F8E4M3,
      F8E4M3FN, F8E4M3B11FNUZ, F8E5M2FNUZ, F8E4M3FNUZ, F8E5M3FNU,
      F8E3M4,   F8E8M0FNU,     F16,        BF16,
  };
  RngExpander expander;
  for (PrimitiveType ty : kNarrowFps) {
    const std::string tname = primitive_util::LowercasePrimitiveTypeName(ty);
    const absl::string_view lo_str = (ty == F8E8M0FNU) ? "1" : "0";
    const absl::string_view hi_str = (ty == F8E8M0FNU) ? "4" : "2";
    for (absl::string_view dist : {"rng_uniform", "rng_normal"}) {
      const std::string hlo =
          absl::StrCat("HloModule fp_", tname, "\nENTRY main {\n  a = ", tname,
                       "[] constant(", lo_str, ")\n  b = ", tname,
                       "[] constant(", hi_str, ")\n  ROOT r = ", tname,
                       "[16] rng(a, b), distribution=", dist, "\n}");
      ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
      EXPECT_THAT(expander.Run(module.get()), IsOkAndHolds(true))
          << tname << " " << dist;
      EXPECT_EQ(CountOpcode(*module, HloOpcode::kRng), 0);
      EXPECT_THAT(HloVerifier(HloVerifierOpts{}).Run(module.get()), IsOk());
      // Keep direct F32 -> narrow convert (including round-to-nearest
      // F16/BF16).
      const HloInstruction* root =
          module->entry_computation()->root_instruction();
      EXPECT_EQ(root->opcode(), HloOpcode::kConvert) << tname;
      EXPECT_EQ(root->operand(0)->shape().element_type(), F32) << tname;

      if (dist == "rng_uniform") {
        ASSERT_OK_AND_ASSIGN(Literal out,
                             EvaluateExpandedRngModule(module.get()));
        ASSERT_OK_AND_ASSIGN(Literal as_f32, out.Convert(F32));
        const float min_v = (ty == F8E8M0FNU) ? 0.5f : 0.0f;
        const float max_v = (ty == F8E8M0FNU) ? 4.0f : 2.0f;
        for (float v : as_f32.data<float>()) {
          EXPECT_TRUE(std::isfinite(v)) << tname;
          EXPECT_GE(v, min_v) << tname;
          EXPECT_LE(v, max_v) << tname;
        }
      }
    }
  }
}

TEST_F(RngExpanderTest, ExpandsMixedPrecisionFloatingPointOperands) {
  constexpr absl::string_view kHlo = R"(
    HloModule mixed
    ENTRY main {
      lo_f32 = f32[] constant(0)
      hi_f32 = f32[] constant(1)
      lo_bf16 = bf16[] constant(0)
      hi_bf16 = bf16[] constant(2)
      sigma_f32 = f32[] constant(0.5)
      r_bf16 = bf16[16] rng(lo_f32, hi_f32), distribution=rng_uniform
      r_f8 = f8e4m3fn[16] rng(lo_f32, hi_f32), distribution=rng_uniform
      r_f32 = f32[16] rng(lo_bf16, hi_bf16), distribution=rng_uniform
      r_f64 = f64[16] rng(lo_f32, sigma_f32), distribution=rng_normal
      ROOT t = (bf16[16], f8e4m3fn[16], f32[16], f64[16])
          tuple(r_bf16, r_f8, r_f32, r_f64)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));
  ASSERT_THAT(HloVerifier(/*layout_sensitive=*/false,
                          /*allow_mixed_precision=*/true)
                  .Run(module.get()),
              IsOk());

  RngExpander expander;
  EXPECT_THAT(expander.Run(module.get()), IsOkAndHolds(true));
  EXPECT_EQ(CountOpcode(*module, HloOpcode::kRng), 0);
  // Expanded module must satisfy strict (non-mixed) verification as well.
  EXPECT_THAT(HloVerifier(HloVerifierOpts{}).Run(module.get()), IsOk());
  ASSERT_OK_AND_ASSIGN(Literal out, EvaluateExpandedRngModule(module.get()));
  std::vector<Literal> elems = out.DecomposeTuple();
  ASSERT_EQ(elems.size(), 4);
  for (float v : elems[2].data<float>()) {
    EXPECT_GE(v, 0.0f);
    EXPECT_LT(v, 2.0f);
  }
  for (double v : elems[3].data<double>()) {
    EXPECT_TRUE(std::isfinite(v));
  }
}

TEST_F(RngExpanderTest, Guards64BitPhiloxAndSplitOverflowAndAllowsZeroSized) {
  struct RejectCase {
    absl::string_view ty;
    std::vector<int64_t> dims;
    absl::string_view dist;
    bool verifier_valid;
  };
  const std::vector<RejectCase> kRejectCases = {
      {"f32", {2305843009213693951LL}, "rng_normal", true},
      {"f64", {1152921504606846975LL}, "rng_normal", true},
      {"f16", {2305843009213693952LL}, "rng_normal", true},
      {"f8e4m3fn", {2305843009213693951LL}, "rng_normal", true},
      {"s8", {9223372036854775807LL}, "rng_uniform", true},
      {"f32", {11, 209622091746708541LL}, "rng_normal", false},
      {"f32", {3, 768614336404564649LL}, "rng_normal", true},
      {"f32", {9223372036854775807LL, 0}, "rng_normal", false},
      {"f32", {2305843009213693949LL}, "rng_uniform", true},
      {"f64", {1152921504606846975LL}, "rng_uniform", true},
      {"f64", {4611686018427387904LL}, "rng_uniform", false},
      {"f32", {9223372036854775807LL}, "rng_uniform", false},
      {"f32", {5, 1844674407370955161LL}, "rng_normal", false},
  };
  RngExpander expander;
  for (const RejectCase& rc : kRejectCases) {
    // Build with small placeholder dimensions for `!verifier_valid` cases so
    // `HloInstruction`'s debug-build shape DCHECK does not fire before the
    // expander runs.
    std::string dims_str;
    for (size_t i = 0; i < rc.dims.size(); ++i) {
      absl::StrAppend(&dims_str, i == 0 ? "" : ",",
                      rc.verifier_valid ? rc.dims[i] : 2);
    }
    const std::string hlo = absl::StrCat(
        "HloModule ovf\nENTRY main {\n  a = ", rc.ty,
        "[] constant(0)\n  b = ", rc.ty, "[] constant(1)\n  ROOT r = ", rc.ty,
        "[", dims_str, "] rng(a, b), distribution=", rc.dist, "\n}");
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
    if (rc.verifier_valid) {
      ASSERT_THAT(HloVerifier(HloVerifierOpts{}).Run(module.get()), IsOk())
          << hlo;
    } else {
      Shape* s =
          module->entry_computation()->root_instruction()->mutable_shape();
      for (size_t i = 0; i < rc.dims.size(); ++i) {
        s->set_dimensions(static_cast<int64_t>(i), rc.dims[i]);
      }
    }
    const int64_t before = module->entry_computation()->instruction_count();
    EXPECT_THAT(expander.Run(module.get()),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << rc.ty << " " << rc.dist;
    // Must fail before any partial IR mutation.
    EXPECT_EQ(module->entry_computation()->instruction_count(), before);
  }

  struct AcceptCase {
    absl::string_view ty;
    absl::string_view shape;
    absl::string_view dist;
    int64_t expected_delta;
  };
  const std::vector<AcceptCase> kAcceptCases = {
      {"f32", "f32[2305843009213693948]", "rng_uniform", 2305843009213693948LL},
      {"f32", "f32[2305843009213693948]", "rng_normal", 2305843009213693948LL},
      {"f32", "f32[2305843009213693947]", "rng_normal", 2305843009213693947LL},
      {"f64", "f64[1152921504606846974]", "rng_uniform", 2305843009213693948LL},
      {"f64", "f64[1152921504606846974]", "rng_normal", 2305843009213693948LL},
      {"f32", "f32[3,768614336404564647]", "rng_normal", 2305843009213693941LL},
      {"f64", "f64[0,8]", "rng_uniform", 0},
      {"f32", "f32[0,9223372036854775807]", "rng_normal", 0},
  };
  for (const AcceptCase& ac : kAcceptCases) {
    const std::string hlo =
        absl::StrCat("HloModule ok\nENTRY main {\n  a = ", ac.ty,
                     "[] constant(0)\n  b = ", ac.ty,
                     "[] constant(1)\n  ROOT r = ", ac.shape,
                     " rng(a, b), distribution=", ac.dist, "\n}");
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    EXPECT_THAT(expander.Run(module.get()), IsOkAndHolds(true)) << ac.shape;
    EXPECT_EQ(CountOpcode(*module, HloOpcode::kRng), 0) << ac.shape;
    EXPECT_THAT(HloVerifier(HloVerifierOpts{}).Run(module.get()), IsOk())
        << ac.shape;
    const auto* st = FindStateInst(*module);
    ASSERT_NE(st, nullptr);
    EXPECT_EQ(st->delta(), ac.expected_delta) << ac.shape;
  }
}

TEST_F(RngExpanderTest, PreservesNonDefaultTiledSubByteAutoAndMemoryLayouts) {
  auto layout_verifier = []() {
    return HloVerifier(
        HloVerifierOpts{}.MakeLayoutSensitive().WithInstructionCanChangeLayout(
            LayoutAssignment::InstructionCanChangeLayout));
  };

  // Sub-byte :E(4) layout keeps :E(4) directly on kConvert with zero kCopy.
  ASSERT_OK_AND_ASSIGN(auto subbyte, ParseAndReturnVerifiedModule(R"(
    HloModule subbyte
    ENTRY main {
      lo = s4[]{:E(4)} constant(-4)
      hi = s4[]{:E(4)} constant(4)
      ROOT r = s4[4,8]{1,0:E(4)} rng(lo, hi), distribution=rng_uniform
    })"));
  EXPECT_THAT(RngExpander{}.Run(subbyte.get()), IsOkAndHolds(true));
  EXPECT_THAT(layout_verifier().Run(subbyte.get()), IsOk());
  EXPECT_EQ(CountOpcode(*subbyte, HloOpcode::kCopy), 0);
  EXPECT_EQ(subbyte->entry_computation()
                ->root_instruction()
                ->shape()
                .layout()
                .element_size_in_bits(),
            4);

  // Non-default minor-to-major, tiles, tail padding alignment, memory space,
  // AUTO layout, and alternate-memory-space scalar operands.
  ASSERT_OK_AND_ASSIGN(auto layouts, ParseAndReturnVerifiedModule(R"(
    HloModule layouts
    ENTRY main {
      p_f32_s4 = f32[]{:S(4)} parameter(0)
      hi_f32 = f32[] constant(1)
      p_f16_s4 = f16[]{:S(4)} parameter(1)
      hi_f16 = f16[] constant(1)
      p_pred_s4 = pred[]{:S(4)} parameter(2)
      hi_pred = pred[] constant(true)
      r0 = f32[2,4,8]{0,1,2} rng(p_f32_s4, hi_f32), distribution=rng_uniform
      r1 = f32[8,128]{1,0:T(8,128)L(2)S(1)} rng(p_f32_s4, hi_f32),
          distribution=rng_normal
      r2 = f16[8,128]{0,1:T(8,128)(2,1)L(2)S(1)} rng(p_f16_s4, hi_f16),
          distribution=rng_uniform
      r3 = pred[4,8]{0,1:T(2,4)S(1)} rng(p_pred_s4, hi_pred),
          distribution=rng_uniform
      ROOT t = (f32[2,4,8]{0,1,2}, f32[8,128]{1,0:T(8,128)L(2)S(1)},
                f16[8,128]{0,1:T(8,128)(2,1)L(2)S(1)},
                pred[4,8]{0,1:T(2,4)S(1)}) tuple(r0, r1, r2, r3)
    })"));
  EXPECT_THAT(RngExpander{}.Run(layouts.get()), IsOkAndHolds(true));
  EXPECT_EQ(CountOpcode(*layouts, HloOpcode::kRng), 0);
  EXPECT_THAT(layout_verifier().Run(layouts.get()), IsOk());

  // Pre-layout rank-2 AUTO layout with memory space (:S(1)) resets intermediate
  // layout cleanly and restores the exact AUTO result layout via kCopy.
  ASSERT_OK_AND_ASSIGN(auto auto_layout, ParseAndReturnVerifiedModule(R"(
    HloModule auto_layout
    ENTRY main {
      lo = f16[] constant(0)
      hi = f16[] constant(1)
      ROOT r = f16[4,8]{:S(1)} rng(lo, hi), distribution=rng_normal
    })"));
  EXPECT_THAT(RngExpander{}.Run(auto_layout.get()), IsOkAndHolds(true));
  EXPECT_EQ(CountOpcode(*auto_layout, HloOpcode::kRng), 0);
  EXPECT_THAT(HloVerifier(HloVerifierOpts{}).Run(auto_layout.get()), IsOk());
  EXPECT_TRUE(auto_layout->entry_computation()
                  ->root_instruction()
                  ->shape()
                  .layout()
                  .minor_to_major()
                  .empty());
  EXPECT_EQ(auto_layout->entry_computation()
                ->root_instruction()
                ->shape()
                .layout()
                .memory_space(),
            1);
}

TEST_F(RngExpanderTest,
       RejectsDynamicWithoutMutationAndPreservesMetadataShardingControlDeps) {
  // Dynamic narrow RNG must fail with Unimplemented before mutating IR.
  ASSERT_OK_AND_ASSIGN(auto dyn_mod, ParseAndReturnVerifiedModule(R"(
    HloModule dyn
    ENTRY main {
      n = s32[] parameter(0)
      lo = f16[] constant(0)
      hi = f16[] constant(1)
      base = f16[8] constant({0, 0, 0, 0, 0, 0, 0, 0})
      dyn_base = f16[<=8] set-dimension-size(base, n), dimensions={0}
      r = f16[<=8] rng(lo, hi), distribution=rng_uniform
      ROOT out = f16[<=8] add(dyn_base, r)
    })"));
  const int64_t dyn_count = dyn_mod->entry_computation()->instruction_count();
  EXPECT_THAT(RngExpander{}.Run(dyn_mod.get()),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_EQ(dyn_mod->entry_computation()->instruction_count(), dyn_count);

  // Metadata, frontend attributes, tiled + maximal + NamedSharding, and control
  // dependencies must survive both narrow and 32-bit expansion.
  ASSERT_OK_AND_ASSIGN(auto attr_mod, ParseAndReturnVerifiedModule(R"(
    HloModule attrs
    ENTRY main {
      lo = f16[] constant(0)
      hi = f16[] constant(1)
      lo32 = f32[] constant(0)
      hi32 = f32[] constant(1)
      pred_tok = token[] after-all()
      r_narrow = f16[4,4] rng(lo, hi), distribution=rng_uniform,
          control-predecessors={pred_tok},
          sharding={devices=[2,2]<=[4]},
          frontend_attributes={custom_attr="narrow"},
          metadata={op_name="narrow_rng" source_file="a.py" source_line=11}
      r_wide = f32[4] rng(lo32, hi32), distribution=rng_normal,
          sharding={maximal device=1},
          frontend_attributes={custom_attr="wide"},
          metadata={op_name="wide_rng" source_file="b.py" source_line=22}
      r_named = f32[4,4] rng(lo32, hi32), distribution=rng_uniform,
          metadata={op_name="named_rng"}
      succ_tok = token[] after-all(), control-predecessors={r_narrow, r_wide}
      ROOT t = (f16[4,4], f32[4], f32[4,4], token[])
          tuple(r_narrow, r_wide, r_named, succ_tok)
    })"));
  // Attach a rank-2 replicated NamedSharding (non-empty dim_shardings) to
  // verify it stays on the rank-2 call without polluting scalar/state ops.
  HloInstruction* r_named =
      attr_mod->entry_computation()->GetInstructionWithName("r_named");
  ASSERT_NE(r_named, nullptr);
  Mesh mesh({2, 2}, {"x", "y"});
  r_named->set_sharding(
      HloSharding(NamedSharding(mesh, {NamedSharding::DimensionSharding(),
                                       NamedSharding::DimensionSharding()})));

  EXPECT_THAT(RngExpander{}.Run(attr_mod.get()), IsOkAndHolds(true));
  EXPECT_EQ(CountOpcode(*attr_mod, HloOpcode::kRng), 0);
  EXPECT_THAT(HloVerifier(HloVerifierOpts{}).Run(attr_mod.get()), IsOk());

  const HloInstruction* root =
      attr_mod->entry_computation()->root_instruction();
  const HloInstruction* out_narrow = root->operand(0);
  const HloInstruction* out_wide = root->operand(1);
  const HloInstruction* out_named = root->operand(2);
  const HloInstruction* succ_tok = root->operand(3);
  EXPECT_EQ(out_narrow->metadata().op_name(), "narrow_rng");
  EXPECT_EQ(out_narrow->frontend_attributes().map().at("custom_attr"),
            "narrow");
  EXPECT_TRUE(out_narrow->has_sharding());
  EXPECT_FALSE(out_narrow->sharding().IsReplicatedOrSingleDevice());
  ASSERT_EQ(out_narrow->operand(0)->opcode(), HloOpcode::kCall);
  EXPECT_FALSE(out_narrow->operand(0)->operand(1)->has_sharding());
  EXPECT_EQ(out_narrow->operand(0)->control_predecessors().size(), 1);

  EXPECT_EQ(out_wide->metadata().op_name(), "wide_rng");
  EXPECT_EQ(out_wide->frontend_attributes().map().at("custom_attr"), "wide");
  EXPECT_TRUE(out_wide->has_sharding());
  EXPECT_TRUE(out_wide->operand(0)->has_sharding());
  EXPECT_TRUE(out_wide->operand(1)->has_sharding());
  EXPECT_EQ(succ_tok->control_predecessors().size(), 2);

  EXPECT_EQ(out_named->metadata().op_name(), "named_rng");
  EXPECT_TRUE(out_named->has_sharding());
  EXPECT_FALSE(out_named->operand(0)->has_sharding());
  EXPECT_FALSE(out_named->operand(1)->has_sharding());

  // Legacy 3-operand kRng (R1) continues to expand and widen array operands.
  ASSERT_OK_AND_ASSIGN(auto legacy, ParseAndReturnUnverifiedModule(R"(
    HloModule legacy
    ENTRY main {
      mean = f16[] constant(0)
      stdev = f16[] constant(1)
      shape = f16[3] constant({8, 4, 2})
      ROOT r = f16[] rng(mean, stdev, shape), distribution=rng_normal
    })"));
  EXPECT_THAT(RngExpander{}.Run(legacy.get()), IsOkAndHolds(true));
  EXPECT_EQ(CountOpcode(*legacy, HloOpcode::kRng), 0);
  int entry_converts = 0;
  for (const HloInstruction* inst :
       legacy->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kConvert) {
      ++entry_converts;
    }
  }
  EXPECT_EQ(entry_converts, 4);
}

}  // namespace
}  // namespace xla
