/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/tree_reduction_rewriter.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/transforms/dynamic_padder.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {

namespace {

using ::absl_testing::IsOkAndHolds;

class TreeReductionRewriterTest : public HloHardwareIndependentTestBase {
 public:
  void MatchTreeReducedHlo(absl::string_view hlo, absl::string_view pattern,
                           int64_t reduce_window_size) {
    auto optimized_module = ParseAndReturnVerifiedModule(hlo).value();
    ASSERT_OK(
        TreeReductionRewriter(reduce_window_size).Run(optimized_module.get()));
    EXPECT_TRUE(RunFileCheck(optimized_module->ToString(), pattern).value());
  }

  absl::StatusOr<Literal> RunCpuHloPassesAndEvaluate(
      absl::string_view hlo, const std::vector<const Literal*>& args) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
    CpuCompiler compiler;
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> optimized,
        compiler.RunHloPasses(std::move(module), /*stream_exec=*/nullptr,
                              /*options=*/{}));
    HloEvaluator evaluator;
    return evaluator.Evaluate(*optimized, args);
  }
};

TEST_F(TreeReductionRewriterTest, SimpleRewrite) {
  const char* hlo_text = R"(
HloModule SimpleReduction

add {
  acc = f32[] parameter(1)
  op = f32[] parameter(0)
  ROOT out = f32[] add(acc, op)
}

ENTRY main {
  input = f32[1000] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[] reduce(input, zero), dimensions={0}, to_apply=add
}
  )";

  MatchTreeReducedHlo(hlo_text,
                      R"(
; CHECK-LABEL: ENTRY %main (input: f32[1000]) -> f32[] {
; CHECK-NEXT:    [[INSTR_0:%[^ ]+]] = f32[1000]{0} parameter(0)
; CHECK-NEXT:    [[INSTR_1:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[INSTR_2:%[^ ]+]] = f32[32]{0} reduce-window([[INSTR_0]], [[INSTR_1]]), window={size=32 stride=32 pad=12_12}, to_apply=[[INSTR_3:%[^ ]+]]
; CHECK-NEXT:    ROOT [[INSTR_4:%[^ ]+]] = f32[] reduce([[INSTR_2]], [[INSTR_1]]), dimensions={0}, to_apply=[[INSTR_3]]
      )",
                      /*reduce_window_size=*/32);
}

TEST_F(TreeReductionRewriterTest, RewriteMultipleDimensions) {
  const char* hlo_text = R"(
HloModule SimpleReduction

add {
  acc = f32[] parameter(1)
  op = f32[] parameter(0)
  ROOT out = f32[] add(acc, op)
}

ENTRY main {
  input = f32[100,100] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[] reduce(input, zero), dimensions={0,1}, to_apply=add
}
  )";

  MatchTreeReducedHlo(hlo_text,
                      R"(
; CHECK:    [[INSTR_0:%[^ ]+]] = f32[4,4]{1,0} reduce-window([[INSTR_1:%[^ ]+]], [[INSTR_2:%[^ ]+]]), window={size=32x32 stride=32x32 pad=14_14x14_14}, to_apply=[[INSTR_3:%[^ ]+]]
; CHECK-NEXT: ROOT [[INSTR_4:%[^ ]+]] = f32[] reduce([[INSTR_0]], [[INSTR_2]]), dimensions={0,1}, to_apply=[[INSTR_3]]
      )",
                      /*reduce_window_size=*/32);
}

TEST_F(TreeReductionRewriterTest, RewriteMultipleDimensionsSingleSmaller) {
  const char* hlo_text = R"(
HloModule SimpleReduction

add {
  acc = f32[] parameter(1)
  op = f32[] parameter(0)
  ROOT out = f32[] add(acc, op)
}

ENTRY main {
  input = f32[1000,31] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[] reduce(input, zero), dimensions={0,1}, to_apply=add
}
  )";

  MatchTreeReducedHlo(hlo_text,
                      R"(
; CHECK:    [[INSTR_0:%[^ ]+]] = f32[32,1]{1,0} reduce-window([[INSTR_1:%[^ ]+]], [[INSTR_2:%[^ ]+]]), window={size=32x31 stride=32x31 pad=12_12x0_0}, to_apply=[[INSTR_3:%[^ ]+]]
; CHECK-NEXT: ROOT [[INSTR_4:%[^ ]+]] = f32[] reduce([[INSTR_0]], [[INSTR_2]]), dimensions={0,1}, to_apply=[[INSTR_3]]
      )",
                      /*reduce_window_size=*/32);
}

TEST_F(TreeReductionRewriterTest, NoRewriteRequired) {
  const char* hlo_text = R"(
HloModule SimpleReduction

add {
  acc = f32[] parameter(1)
  op = f32[] parameter(0)
  ROOT out = f32[] add(acc, op)
}

ENTRY main {
  input = f32[31,31] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[] reduce(input, zero), dimensions={0,1}, to_apply=add
}
  )";

  MatchTreeReducedHlo(hlo_text,
                      R"(
// CHECK: ROOT [[INSTR_0:%[^ ]+]] = f32[] reduce([[INSTR_1:%[^ ]+]], [[INSTR_2:%[^ ]+]]), dimensions={0,1}, to_apply=[[INSTR_3:%[^ ]+]]
      )",
                      /*reduce_window_size=*/32);
}

TEST_F(TreeReductionRewriterTest, NoRewriteRequiredZeroDim) {
  const char* hlo_text = R"(
HloModule SimpleReduction

add {
  acc = f32[] parameter(1)
  op = f32[] parameter(0)
  ROOT out = f32[] add(acc, op)
}

ENTRY main {
  input = f32[3000,0] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[] reduce(input, zero), dimensions={0,1}, to_apply=add
}
  )";

  MatchTreeReducedHlo(hlo_text,
                      R"(
// CHECK-NOT: reduce-window
// CHECK: ROOT {{.*}} = f32[] reduce({{.*}}), dimensions={0,1}, to_apply={{.*}}
      )",
                      /*reduce_window_size=*/32);
}

TEST_F(TreeReductionRewriterTest, NonZeroAndNonConstantAddInitNotRewritten) {
  constexpr absl::string_view kS32Init10 = R"(
    HloModule S32Init10
    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT r = s32[] add(a, b)
    }
    ENTRY main {
      x = s32[64] parameter(0)
      init = s32[] constant(10)
      ROOT out = s32[] reduce(x, init), dimensions={0}, to_apply=add
    }
  )";
  constexpr absl::string_view kS32PaddedInit10 = R"(
    HloModule S32PaddedInit10
    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT r = s32[] add(a, b)
    }
    ENTRY main {
      x = s32[65] parameter(0)
      init = s32[] constant(10)
      ROOT out = s32[] reduce(x, init), dimensions={0}, to_apply=add
    }
  )";
  constexpr absl::string_view kF32Init10 = R"(
    HloModule F32Init10
    add {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT r = f32[] add(a, b)
    }
    ENTRY main {
      x = f32[64] parameter(0)
      init = f32[] constant(10)
      ROOT out = f32[] reduce(x, init), dimensions={0}, to_apply=add
    }
  )";
  constexpr absl::string_view kParamInit = R"(
    HloModule ParamInit
    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT r = s32[] add(a, b)
    }
    ENTRY main {
      x = s32[65] parameter(0)
      init = s32[] parameter(1)
      ROOT out = s32[] reduce(x, init), dimensions={0}, to_apply=add
    }
  )";

  for (absl::string_view hlo :
       {kS32Init10, kS32PaddedInit10, kF32Init10, kParamInit}) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    EXPECT_THAT(TreeReductionRewriter(32).Run(module.get()),
                IsOkAndHolds(false));
  }

  Literal ones64 = LiteralUtil::CreateR1<int32_t>(std::vector<int32_t>(64, 1));
  Literal ones65 = LiteralUtil::CreateR1<int32_t>(std::vector<int32_t>(65, 1));
  Literal fones64 = LiteralUtil::CreateR1<float>(std::vector<float>(64, 1.0f));
  Literal init10 = LiteralUtil::CreateR0<int32_t>(10);

  ASSERT_OK_AND_ASSIGN(Literal r64,
                       RunCpuHloPassesAndEvaluate(kS32Init10, {&ones64}));
  EXPECT_EQ(r64.Get<int32_t>({}), 74);
  ASSERT_OK_AND_ASSIGN(Literal r65,
                       RunCpuHloPassesAndEvaluate(kS32PaddedInit10, {&ones65}));
  EXPECT_EQ(r65.Get<int32_t>({}), 75);
  ASSERT_OK_AND_ASSIGN(Literal rf64,
                       RunCpuHloPassesAndEvaluate(kF32Init10, {&fones64}));
  EXPECT_FLOAT_EQ(rf64.Get<float>({}), 74.0f);
  ASSERT_OK_AND_ASSIGN(Literal rparam, RunCpuHloPassesAndEvaluate(
                                           kParamInit, {&ones65, &init10}));
  EXPECT_EQ(rparam.Get<int32_t>({}), 75);
}

TEST_F(TreeReductionRewriterTest,
       NonIdentityMultiplyAndXorInitAndComplexMulRejected) {
  constexpr absl::string_view kS32MulInit2 = R"(
    HloModule S32MulInit2
    mul {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT r = s32[] multiply(a, b)
    }
    ENTRY main {
      x = s32[64] parameter(0)
      init = s32[] constant(2)
      ROOT out = s32[] reduce(x, init), dimensions={0}, to_apply=mul
    }
  )";
  constexpr absl::string_view kF32MulNeg1 = R"(
    HloModule F32MulNeg1
    mul {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT r = f32[] multiply(a, b)
    }
    ENTRY main {
      x = f32[96] parameter(0)
      init = f32[] constant(-1)
      ROOT out = f32[] reduce(x, init), dimensions={0}, to_apply=mul
    }
  )";
  constexpr absl::string_view kF32MulPos1 = R"(
    HloModule F32MulPos1
    mul {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT r = f32[] multiply(a, b)
    }
    ENTRY main {
      x = f32[96] parameter(0)
      init = f32[] constant(1)
      ROOT out = f32[] reduce(x, init), dimensions={0}, to_apply=mul
    }
  )";
  constexpr absl::string_view kS32XorInit1 = R"(
    HloModule S32XorInit1
    rxor {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT r = s32[] xor(a, b)
    }
    ENTRY main {
      x = s32[96] parameter(0)
      init = s32[] constant(1)
      ROOT out = s32[] reduce(x, init), dimensions={0}, to_apply=rxor
    }
  )";
  constexpr absl::string_view kPredXorTrue = R"(
    HloModule PredXorTrue
    rxor {
      a = pred[] parameter(0)
      b = pred[] parameter(1)
      ROOT r = pred[] xor(a, b)
    }
    ENTRY main {
      x = pred[96] parameter(0)
      init = pred[] constant(true)
      ROOT out = pred[] reduce(x, init), dimensions={0}, to_apply=rxor
    }
  )";
  constexpr absl::string_view kS32XorZero = R"(
    HloModule S32XorZero
    rxor {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT r = s32[] xor(a, b)
    }
    ENTRY main {
      x = s32[96] parameter(0)
      init = s32[] constant(0)
      ROOT out = s32[] reduce(x, init), dimensions={0}, to_apply=rxor
    }
  )";
  constexpr absl::string_view kPredXorFalse = R"(
    HloModule PredXorFalse
    rxor {
      a = pred[] parameter(0)
      b = pred[] parameter(1)
      ROOT r = pred[] xor(a, b)
    }
    ENTRY main {
      x = pred[96] parameter(0)
      init = pred[] constant(false)
      ROOT out = pred[] reduce(x, init), dimensions={0}, to_apply=rxor
    }
  )";
  constexpr absl::string_view kC64Mul = R"(
    HloModule C64Mul
    mul {
      a = c64[] parameter(0)
      b = c64[] parameter(1)
      ROOT r = c64[] multiply(a, b)
    }
    ENTRY main {
      x = c64[64] parameter(0)
      init = c64[] constant((1, 0))
      ROOT out = c64[] reduce(x, init), dimensions={0}, to_apply=mul
    }
  )";
  constexpr absl::string_view kC64Add = R"(
    HloModule C64Add
    add {
      a = c64[] parameter(0)
      b = c64[] parameter(1)
      ROOT r = c64[] add(a, b)
    }
    ENTRY main {
      x = c64[64] parameter(0)
      init = c64[] constant((0, 0))
      ROOT out = c64[] reduce(x, init), dimensions={0}, to_apply=add
    }
  )";
  constexpr absl::string_view kE8M0Add = R"(
    HloModule E8M0Add
    add {
      a = f8e8m0fnu[] parameter(0)
      b = f8e8m0fnu[] parameter(1)
      ROOT r = f8e8m0fnu[] add(a, b)
    }
    ENTRY main {
      x = f8e8m0fnu[64] parameter(0)
      init = f8e8m0fnu[] constant(1)
      ROOT out = f8e8m0fnu[] reduce(x, init), dimensions={0}, to_apply=add
    }
  )";
  constexpr absl::string_view kE8M0Mul = R"(
    HloModule E8M0Mul
    mul {
      a = f8e8m0fnu[] parameter(0)
      b = f8e8m0fnu[] parameter(1)
      ROOT r = f8e8m0fnu[] multiply(a, b)
    }
    ENTRY main {
      x = f8e8m0fnu[64] parameter(0)
      init = f8e8m0fnu[] constant(1)
      ROOT out = f8e8m0fnu[] reduce(x, init), dimensions={0}, to_apply=mul
    }
  )";

  for (absl::string_view rejected : {kS32MulInit2, kF32MulNeg1, kS32XorInit1,
                                     kPredXorTrue, kC64Mul, kE8M0Add}) {
    ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(rejected));
    EXPECT_THAT(TreeReductionRewriter(32).Run(m.get()), IsOkAndHolds(false));
  }
  for (absl::string_view allowed :
       {kF32MulPos1, kS32XorZero, kPredXorFalse, kC64Add, kE8M0Mul}) {
    ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(allowed));
    EXPECT_THAT(TreeReductionRewriter(32).Run(m.get()), IsOkAndHolds(true));
  }

  Literal ones64 = LiteralUtil::CreateR1<int32_t>(std::vector<int32_t>(64, 1));
  ASSERT_OK_AND_ASSIGN(Literal mul_res,
                       RunCpuHloPassesAndEvaluate(kS32MulInit2, {&ones64}));
  EXPECT_EQ(mul_res.Get<int32_t>({}), 2);

  Literal fones96 = LiteralUtil::CreateR1<float>(std::vector<float>(96, 1.0f));
  ASSERT_OK_AND_ASSIGN(Literal fmul_res,
                       RunCpuHloPassesAndEvaluate(kF32MulNeg1, {&fones96}));
  EXPECT_FLOAT_EQ(fmul_res.Get<float>({}), -1.0f);

  Literal zeros96 = LiteralUtil::CreateR1<int32_t>(std::vector<int32_t>(96, 0));
  ASSERT_OK_AND_ASSIGN(Literal xor_res,
                       RunCpuHloPassesAndEvaluate(kS32XorInit1, {&zeros96}));
  EXPECT_EQ(xor_res.Get<int32_t>({}), 1);

  std::array<bool, 96> pfalse{};
  Literal pzeros96 = LiteralUtil::CreateR1<bool>(pfalse);
  ASSERT_OK_AND_ASSIGN(Literal pxor_res,
                       RunCpuHloPassesAndEvaluate(kPredXorTrue, {&pzeros96}));
  EXPECT_TRUE(pxor_res.Get<bool>({}));

  std::vector<complex64> cvals(64, complex64(1.0f, 0.0f));
  cvals[62] = complex64(1e20f, 1e-20f);
  cvals[63] = complex64(0.0f, 1e20f);
  Literal clit = LiteralUtil::CreateR1<complex64>(cvals);
  ASSERT_OK_AND_ASSIGN(auto cmod, ParseAndReturnVerifiedModule(kC64Mul));
  ASSERT_OK(TreeReductionRewriter(32).Run(cmod.get()).status());
  HloEvaluator eval;
  ASSERT_OK_AND_ASSIGN(Literal cres, eval.Evaluate(*cmod, {&clit}));
  complex64 c = cres.Get<complex64>({});
  EXPECT_FLOAT_EQ(c.real(), -1.0f);
  EXPECT_TRUE(std::isinf(c.imag()) && c.imag() > 0.0f);
}

TEST_F(TreeReductionRewriterTest,
       SignedZeroAddAndIdempotentMinMaxAndOrRewritten) {
  constexpr absl::string_view kNegZeroAdd = R"(
    HloModule NegZeroAdd
    add {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT r = f32[] add(a, b)
    }
    ENTRY main {
      x = f32[65] parameter(0)
      init = f32[] constant(-0.0)
      ROOT out = f32[] reduce(x, init), dimensions={0}, to_apply=add
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto nz_mod, ParseAndReturnVerifiedModule(kNegZeroAdd));
  HloInstruction* nz_init = FindInstruction(nz_mod.get(), "init");
  ASSERT_OK(nz_mod->entry_computation()->ReplaceWithNewInstruction(
      nz_init,
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(-0.0f))));
  EXPECT_THAT(TreeReductionRewriter(32).Run(nz_mod.get()), IsOkAndHolds(true));
  Literal neg_zeros =
      LiteralUtil::CreateR1<float>(std::vector<float>(65, -0.0f));
  HloEvaluator nz_eval;
  nz_eval.set_reduce_use_fast_path(false);
  ASSERT_OK_AND_ASSIGN(Literal nz_res, nz_eval.Evaluate(*nz_mod, {&neg_zeros}));
  EXPECT_EQ(nz_res.Get<float>({}), 0.0f);
  EXPECT_TRUE(std::signbit(nz_res.Get<float>({})));

  constexpr absl::string_view kParamInitMinMaxAndOr = R"(
    HloModule IdempotentReducers
    rmin {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT r = f32[] minimum(a, b)
    }
    rmax {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT r = s32[] maximum(a, b)
    }
    rand {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT r = s32[] and(a, b)
    }
    ror {
      a = pred[] parameter(0)
      b = pred[] parameter(1)
      ROOT r = pred[] or(a, b)
    }
    ENTRY main {
      fx = f32[65] parameter(0)
      fi = f32[] parameter(1)
      ix = s32[65] parameter(2)
      ii = s32[] parameter(3)
      px = pred[65] parameter(4)
      pi = pred[] parameter(5)
      m0 = f32[] reduce(fx, fi), dimensions={0}, to_apply=rmin
      m1 = s32[] reduce(ix, ii), dimensions={0}, to_apply=rmax
      m2 = s32[] reduce(ix, ii), dimensions={0}, to_apply=rand
      m3 = pred[] reduce(px, pi), dimensions={0}, to_apply=ror
      ROOT out = (f32[], s32[], s32[], pred[]) tuple(m0, m1, m2, m3)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto ref_mod,
                       ParseAndReturnVerifiedModule(kParamInitMinMaxAndOr));
  ASSERT_OK_AND_ASSIGN(auto opt_mod,
                       ParseAndReturnVerifiedModule(kParamInitMinMaxAndOr));
  EXPECT_THAT(TreeReductionRewriter(32).Run(opt_mod.get()), IsOkAndHolds(true));
  EXPECT_EQ(absl::c_count_if(opt_mod->entry_computation()->instructions(),
                             [](const HloInstruction* inst) {
                               return inst->opcode() ==
                                      HloOpcode::kReduceWindow;
                             }),
            4);

  Literal fx = LiteralUtil::CreateR1<float>(std::vector<float>(65, 5.0f));
  Literal fi = LiteralUtil::CreateR0<float>(3.5f);
  Literal ix = LiteralUtil::CreateR1<int32_t>(std::vector<int32_t>(65, 7));
  Literal ii = LiteralUtil::CreateR0<int32_t>(14);
  std::array<bool, 65> pvec{};
  Literal px = LiteralUtil::CreateR1<bool>(pvec);
  Literal pi = LiteralUtil::CreateR0<bool>(true);
  std::vector<const Literal*> args = {&fx, &fi, &ix, &ii, &px, &pi};
  ASSERT_OK_AND_ASSIGN(Literal ref_out,
                       HloEvaluator().Evaluate(*ref_mod, args));
  ASSERT_OK_AND_ASSIGN(Literal opt_out,
                       HloEvaluator().Evaluate(*opt_mod, args));
  EXPECT_EQ(opt_out, ref_out);
}

TEST_F(TreeReductionRewriterTest,
       NonAssociativeMultiOpAndSelfOperandReducersRejected) {
  constexpr absl::string_view kSub = R"(
    HloModule SubReduce
    sub {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT r = s32[] subtract(a, b)
    }
    ENTRY main {
      x = s32[64] parameter(0)
      z = s32[] constant(0)
      ROOT out = s32[] reduce(x, z), dimensions={0}, to_apply=sub
    }
  )";
  constexpr absl::string_view kDiv = R"(
    HloModule DivReduce
    div {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT r = f32[] divide(a, b)
    }
    ENTRY main {
      x = f32[64] parameter(0)
      one = f32[] constant(1)
      ROOT out = f32[] reduce(x, one), dimensions={0}, to_apply=div
    }
  )";
  constexpr absl::string_view kSelfOp = R"(
    HloModule SelfOperandReduce
    self_add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT r = s32[] add(a, a)
    }
    ENTRY main {
      x = s32[64] parameter(0)
      z = s32[] constant(0)
      ROOT out = s32[] reduce(x, z), dimensions={0}, to_apply=self_add
    }
  )";
  constexpr absl::string_view kSumSq = R"(
    HloModule SumOfSquares
    sum_sq {
      acc = s32[] parameter(0)
      v = s32[] parameter(1)
      sq = s32[] multiply(v, v)
      ROOT r = s32[] add(acc, sq)
    }
    ENTRY main {
      x = s32[64] parameter(0)
      z = s32[] constant(0)
      ROOT out = s32[] reduce(x, z), dimensions={0}, to_apply=sum_sq
    }
  )";

  for (absl::string_view hlo : {kSub, kDiv, kSelfOp, kSumSq}) {
    ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
    EXPECT_THAT(TreeReductionRewriter(32).Run(m.get()), IsOkAndHolds(false));
  }

  Literal ones64 = LiteralUtil::CreateR1<int32_t>(std::vector<int32_t>(64, 1));
  ASSERT_OK_AND_ASSIGN(Literal sub_res,
                       RunCpuHloPassesAndEvaluate(kSub, {&ones64}));
  EXPECT_EQ(sub_res.Get<int32_t>({}), -64);

  Literal twos64 = LiteralUtil::CreateR1<int32_t>(std::vector<int32_t>(64, 2));
  ASSERT_OK_AND_ASSIGN(Literal sq_res,
                       RunCpuHloPassesAndEvaluate(kSumSq, {&twos64}));
  EXPECT_EQ(sq_res.Get<int32_t>({}), 256);
}

TEST_F(TreeReductionRewriterTest,
       DynamicReducedDimSkippedAndBoundedDynamicBatchAllowed) {
  auto eval_dynamic_sum = [&](absl::string_view hlo, int32_t bound,
                              int32_t dyn_size) -> absl::StatusOr<int32_t> {
    ABSL_ASSIGN_OR_RETURN(auto m, ParseAndReturnVerifiedModule(hlo));
    EXPECT_THAT(TreeReductionRewriter(32).Run(m.get()), IsOkAndHolds(false));
    ABSL_ASSIGN_OR_RETURN(bool padded, DynamicPadder().Run(m.get()));
    EXPECT_TRUE(padded);
    Literal x = LiteralUtil::CreateR1<int32_t>(std::vector<int32_t>(bound, 1));
    Literal d = LiteralUtil::CreateR0<int32_t>(dyn_size);
    HloEvaluator eval;
    ABSL_ASSIGN_OR_RETURN(Literal res, eval.Evaluate(*m, {&x, &d}));
    return res.Get<int32_t>({});
  };

  constexpr absl::string_view kDyn64 = R"(
    HloModule Dyn64
    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT r = s32[] add(a, b)
    }
    ENTRY main {
      x = s32[64] parameter(0)
      d = s32[] parameter(1)
      dx = s32[<=64] set-dimension-size(x, d), dimensions={0}
      z = s32[] constant(0)
      ROOT out = s32[] reduce(dx, z), dimensions={0}, to_apply=add
    }
  )";
  constexpr absl::string_view kDyn65 = R"(
    HloModule Dyn65
    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT r = s32[] add(a, b)
    }
    ENTRY main {
      x = s32[65] parameter(0)
      d = s32[] parameter(1)
      dx = s32[<=65] set-dimension-size(x, d), dimensions={0}
      z = s32[] constant(0)
      ROOT out = s32[] reduce(dx, z), dimensions={0}, to_apply=add
    }
  )";
  EXPECT_THAT(eval_dynamic_sum(kDyn64, 64, 10), IsOkAndHolds(10));
  EXPECT_THAT(eval_dynamic_sum(kDyn64, 64, 40), IsOkAndHolds(40));
  EXPECT_THAT(eval_dynamic_sum(kDyn65, 65, 20), IsOkAndHolds(20));

  constexpr absl::string_view kDynBatch = R"(
    HloModule DynBatch
    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT r = s32[] add(a, b)
    }
    ENTRY main {
      x = s32[4,64] parameter(0)
      d = s32[] parameter(1)
      dx = s32[<=4,64] set-dimension-size(x, d), dimensions={0}
      z = s32[] constant(0)
      ROOT out = s32[<=4] reduce(dx, z), dimensions={1}, to_apply=add
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto batch_mod, ParseAndReturnVerifiedModule(kDynBatch));
  EXPECT_THAT(TreeReductionRewriter(32).Run(batch_mod.get()),
              IsOkAndHolds(true));

  for (absl::string_view unbounded : {absl::string_view(R"(
             HloModule UnboundedBatch
             add {
               a = f32[] parameter(0)
               b = f32[] parameter(1)
               ROOT r = f32[] add(a, b)
             }
             ENTRY main {
               x = f32[?,64] parameter(0)
               z = f32[] constant(0)
               ROOT out = f32[?] reduce(x, z), dimensions={1}, to_apply=add
             }
           )"),
                                      absl::string_view(R"(
             HloModule UnboundedReduced
             add {
               a = f32[] parameter(0)
               b = f32[] parameter(1)
               ROOT r = f32[] add(a, b)
             }
             ENTRY main {
               x = f32[64,?] parameter(0)
               z = f32[] constant(0)
               ROOT out = f32[] reduce(x, z), dimensions={0,1}, to_apply=add
             }
           )")}) {
    ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnUnverifiedModule(unbounded));
    EXPECT_THAT(TreeReductionRewriter(32).Run(m.get()), IsOkAndHolds(false));
  }
}

TEST_F(TreeReductionRewriterTest, ZeroSizedReducedDimDoesNotCrash) {
  constexpr absl::string_view kZeroDim = R"(
    HloModule ZeroDim
    add {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT r = f32[] add(a, b)
    }
    ENTRY main {
      x = f32[3000,0] parameter(0)
      z = f32[] constant(0)
      ROOT out = f32[] reduce(x, z), dimensions={0,1}, to_apply=add
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kZeroDim));
  EXPECT_THAT(TreeReductionRewriter(32).Run(module.get()), IsOkAndHolds(false));
}

TEST_F(TreeReductionRewriterTest, MixedAccumulatorElementTypeSkipped) {
  constexpr absl::string_view kMixed = R"(
    HloModule MixedReduce
    add_f32 {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT r = f32[] add(a, b)
    }
    ENTRY main {
      x = f16[64] parameter(0)
      z = f32[] constant(0)
      ROOT out = f32[] reduce(x, z), dimensions={0}, to_apply=add_f32
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kMixed));
  ASSERT_OK(
      HloVerifier(HloVerifierOpts{}.AllowMixedPrecision()).Run(module.get()));
  EXPECT_THAT(TreeReductionRewriter(32).Run(module.get()), IsOkAndHolds(false));
}

TEST_F(TreeReductionRewriterTest, NonPositiveOrUnitWindowSizeSkipped) {
  constexpr absl::string_view kHlo = R"(
    HloModule WindowSizeGuards
    add {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT r = f32[] add(a, b)
    }
    ENTRY main {
      x = f32[64] parameter(0)
      z = f32[] constant(0)
      ROOT out = f32[] reduce(x, z), dimensions={0}, to_apply=add
    }
  )";
  for (int64_t window_size : {std::numeric_limits<int64_t>::min(), int64_t{-1},
                              int64_t{0}, int64_t{1}}) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
    EXPECT_THAT(TreeReductionRewriter(window_size).Run(module.get()),
                IsOkAndHolds(false));
  }
}

TEST_F(TreeReductionRewriterTest, ControlDependenciesAndMetadataPreserved) {
  constexpr absl::string_view kHlo = R"(
    HloModule ControlDepsAndMeta
    add {
      a = s4[] parameter(0)
      b = s4[] parameter(1)
      ROOT r = s4[] add(a, b)
    }
    ENTRY main {
      x = s4[2,64]{1,0:E(4)} parameter(0)
      pred_op = s4[] constant(1)
      z = s4[] constant(0)
      red = s4[2]{0:E(4)} reduce(x, z), dimensions={1}, to_apply=add,
        control-predecessors={pred_op}, sharding={devices=[2]0,1},
        metadata={op_name="tree_sum" source_file="f.py" source_line=7}
      succ_op = s4[] negate(pred_op), control-predecessors={red}
      ROOT out = (s4[2]{0:E(4)}, s4[]) tuple(red, succ_op)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(TreeReductionRewriter(32).Run(module.get()), IsOkAndHolds(true));
  ASSERT_OK(HloVerifier(/*layout_sensitive=*/true,
                        /*allow_mixed_precision=*/false)
                .Run(module.get()));

  const HloInstruction* pred_op = FindInstruction(module.get(), "pred_op");
  const HloInstruction* succ_op = FindInstruction(module.get(), "succ_op");
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* outer_reduce = root->operand(0);
  ASSERT_EQ(outer_reduce->opcode(), HloOpcode::kReduce);
  const HloInstruction* rw = outer_reduce->operand(0);
  ASSERT_EQ(rw->opcode(), HloOpcode::kReduceWindow);

  EXPECT_TRUE(absl::c_linear_search(rw->control_predecessors(), pred_op));
  EXPECT_TRUE(
      absl::c_linear_search(succ_op->control_predecessors(), outer_reduce));
  EXPECT_EQ(rw->metadata().op_name(), "tree_sum");
  EXPECT_EQ(outer_reduce->metadata().op_name(), "tree_sum");
  EXPECT_FALSE(rw->has_sharding());
  EXPECT_TRUE(outer_reduce->has_sharding());
  EXPECT_EQ(rw->shape().layout().element_size_in_bits(), 4);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
