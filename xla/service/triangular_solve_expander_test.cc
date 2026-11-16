/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/triangular_solve_expander.h"

#include <gmock/gmock.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xla/array2d.h"
#include "xla/error_spec.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/reference_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"

namespace xla {
namespace {

class TriangularExpanderTest : public HloTestBase,
                               public ::testing::WithParamInterface<int32_t> {};

TEST_P(TriangularExpanderTest, TestBlockSize) {
  auto block_size = GetParam();
  std::string hlo_string = R"(
    HloModule TensorFlowTriangularSolve

    ENTRY main {
      a = f32[256,256]{1,0} parameter(0)
      b = f32[256,192]{1,0} parameter(1)
      ROOT triangular-solve = f32[256,192]{1,0} triangular-solve(a, b),
                                    left_side=true, unit_diagonal=true,
                                    lower=true, transpose_a=NO_TRANSPOSE
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  {
    TriangularSolveExpander triangular_solve_expander(block_size);

    ASSERT_OK_AND_ASSIGN(bool result,
                         RunHloPass(&triangular_solve_expander, module.get()));
    EXPECT_TRUE(result);
  }

  // To test triangular solver expander we generate simple bidiagonal matrix:
  // Solve a * x = b.
  // Check that shape is still valid.
  // Use reference matrix multiplication to test validity of result.

  Array2D<float> a(256, 256);
  for (int64_t row = 0; row < a.dim(0); ++row) {
    a(row, row) = 1;
    if (row > 0) {
      a(row, row - 1) = 0.01;
    }
  }

  Array2D<float> b(256, 192);
  const float kMax = static_cast<float>(b.dim(0) * b.dim(1) + 1);
  for (int64_t row = 0; row < b.dim(0); ++row) {
    for (int64_t col = 0; col < b.dim(1); ++col) {
      b(row, col) = static_cast<float>(row + col + 1) / kMax;
    }
  }
  auto la = LiteralUtil::CreateR2FromArray2D(a);
  auto lb = LiteralUtil::CreateR2FromArray2D(b);

  ASSERT_OK_AND_ASSIGN(Literal lx, Execute(std::move(module), {&la, &lb}));

  auto x_shape = lx.shape();
  EXPECT_EQ(x_shape.dimensions().size(), 2);
  EXPECT_EQ(x_shape.dimensions(0), b.dim(0));
  EXPECT_EQ(x_shape.dimensions(1), b.dim(1));

  Array2D<float> x(x_shape.dimensions(0), x_shape.dimensions(1));
  x.SetValues(lx.data<float>());

  auto ref_b = ReferenceUtil::MatmulArray2D(a, x);
  auto ref_lb = LiteralUtil::CreateR2FromArray2D(*ref_b);

  EXPECT_TRUE(
      LiteralTestUtil::NearOrEqual(ref_lb, lb, ErrorSpec{0.001, 0.001}));
}

// block_size test limits based on the following considerations:
// - test at least twice the range of original value
// - try to test odd values unaligned with matrix dims
// - full 1-256 range test takes too long to run

INSTANTIATE_TEST_CASE_P(TriangularExpanderTestInstances, TriangularExpanderTest,
                        ::testing::Range(2, 256, 7));

using TriangularSolveExpanderNumericsTest = HloTestBase;

TEST_F(TriangularSolveExpanderNumericsTest, PreservesSubnormals) {
  constexpr const char* kHlo = R"(
    HloModule Subnormals
    ENTRY main {
      a_sub = f32[2,2]{1,0} parameter(0)
      b_sub = f32[2,1]{1,0} parameter(1)
      a_mix = f32[2,2]{1,0} parameter(2)
      b_mix = f32[2,1]{1,0} parameter(3)
      a_out = f32[2,2]{1,0} parameter(4)
      b_out = f32[2,1]{1,0} parameter(5)
      s_sub = f32[2,1]{1,0} triangular-solve(a_sub, b_sub),
          left_side=true, lower=true, unit_diagonal=false,
          transpose_a=NO_TRANSPOSE
      s_mix = f32[2,1]{1,0} triangular-solve(a_mix, b_mix),
          left_side=true, lower=true, unit_diagonal=false,
          transpose_a=NO_TRANSPOSE
      s_out = f32[2,1]{1,0} triangular-solve(a_out, b_out),
          left_side=true, lower=true, unit_diagonal=false,
          transpose_a=NO_TRANSPOSE
      ROOT out = (f32[2,1]{1,0}, f32[2,1]{1,0}, f32[2,1]{1,0})
          tuple(s_sub, s_mix, s_out)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  TriangularSolveExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&expander, module.get()));
  EXPECT_TRUE(changed);

  Literal a_sub =
      LiteralUtil::CreateR2<float>({{0x1p-130f, 0.0f}, {0x1p-131f, 0x1p-130f}});
  Literal b_sub = LiteralUtil::CreateR2<float>({{0x1p-130f}, {0x1p-130f}});
  Literal a_mix =
      LiteralUtil::CreateR2<float>({{0x1p20f, 0.0f}, {0.0f, 0x1p-130f}});
  Literal b_mix = LiteralUtil::CreateR2<float>({{0x1p20f}, {0x1p-130f}});
  Literal a_out = LiteralUtil::CreateR2<float>({{2.0f, 0.0f}, {0.0f, 2.0f}});
  Literal b_out = LiteralUtil::CreateR2<float>({{0x1p-130f}, {0x1p-129f}});

  // Evaluate with HloEvaluator (IEEE non-FTZ; XLA:CPU sets ScopedFlushDenormal
  // for F32).
  HloEvaluator evaluator;
  ASSERT_OK_AND_ASSIGN(Literal actual,
                       evaluator.Evaluate(*module, {&a_sub, &b_sub, &a_mix,
                                                    &b_mix, &a_out, &b_out}));
  Literal expected = LiteralUtil::MakeTupleOwned(
      LiteralUtil::CreateR2<float>({{1.0f}, {0.5f}}),
      LiteralUtil::CreateR2<float>({{1.0f}, {1.0f}}),
      LiteralUtil::CreateR2<float>({{0x1p-131f}, {0x1p-130f}}));
  EXPECT_TRUE(
      LiteralTestUtil::NearOrEqual(expected, actual, ErrorSpec{0.0f, 1e-3f}));

  // Also verify F16 subnormals directly through XLA:CPU execution.
  constexpr const char* kF16SubHlo = R"(
    HloModule F16Subnormals
    ENTRY main {
      a = f16[2,2]{1,0} parameter(0)
      b = f16[2,1]{1,0} parameter(1)
      a_mix = f16[2,2]{1,0} parameter(2)
      b_mix = f16[2,1]{1,0} parameter(3)
      s_sub = f16[2,1]{1,0} triangular-solve(a, b),
          left_side=true, lower=true, unit_diagonal=false,
          transpose_a=NO_TRANSPOSE
      s_mix = f16[2,1]{1,0} triangular-solve(a_mix, b_mix),
          left_side=true, lower=true, unit_diagonal=false,
          transpose_a=NO_TRANSPOSE
      c_sub = f32[2,1]{1,0} convert(s_sub)
      c_mix = f32[2,1]{1,0} convert(s_mix)
      ROOT out = (f32[2,1]{1,0}, f32[2,1]{1,0}) tuple(c_sub, c_mix)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto f16_module,
                       ParseAndReturnVerifiedModule(kF16SubHlo));
  ASSERT_OK_AND_ASSIGN(bool f16_changed,
                       RunHloPass(&expander, f16_module.get()));
  EXPECT_TRUE(f16_changed);

  Literal a_f16_sub = LiteralUtil::CreateR2<Eigen::half>(
      {{Eigen::half(0x1p-18f), Eigen::half(0.0f)},
       {Eigen::half(0x1p-19f), Eigen::half(0x1p-18f)}});
  Literal b_f16_sub = LiteralUtil::CreateR2<Eigen::half>(
      {{Eigen::half(0x1p-18f)}, {Eigen::half(0x1p-18f)}});
  Literal a_f16_mix = LiteralUtil::CreateR2<Eigen::half>(
      {{Eigen::half(0x1p10f), Eigen::half(0.0f)},
       {Eigen::half(0.0f), Eigen::half(0x1p-18f)}});
  Literal b_f16_mix = LiteralUtil::CreateR2<Eigen::half>(
      {{Eigen::half(0x1p10f)}, {Eigen::half(0x1p-18f)}});
  ASSERT_OK_AND_ASSIGN(Literal actual_f16, Execute(std::move(f16_module),
                                                   {&a_f16_sub, &b_f16_sub,
                                                    &a_f16_mix, &b_f16_mix}));
  Literal expected_f16 = LiteralUtil::MakeTupleOwned(
      LiteralUtil::CreateR2<float>({{1.0f}, {0.5f}}),
      LiteralUtil::CreateR2<float>({{1.0f}, {1.0f}}));
  EXPECT_TRUE(LiteralTestUtil::NearOrEqual(expected_f16, actual_f16,
                                           ErrorSpec{0.0f, 1e-2f}));
}

TEST_F(TriangularSolveExpanderNumericsTest,
       ComplexNonFiniteDiagonalsAndInfOutput) {
  constexpr const char* kHlo = R"(
    HloModule ComplexNonFinite
    ENTRY main {
      a_inf = c64[3,3]{1,0} parameter(0)
      b_l = c64[3,1]{1,0} parameter(1)
      b_r = c64[1,3]{1,0} parameter(2)
      a_ovf = c64[2,2]{1,0} parameter(3)
      b_ovf = c64[2,1]{1,0} parameter(4)
      s_lt = c64[3,1]{1,0} triangular-solve(a_inf, b_l),
          left_side=true, lower=true, unit_diagonal=false, transpose_a=TRANSPOSE
      s_la = c64[3,1]{1,0} triangular-solve(a_inf, b_l),
          left_side=true, lower=true, unit_diagonal=false, transpose_a=ADJOINT
      s_rn = c64[1,3]{1,0} triangular-solve(a_inf, b_r),
          left_side=false, lower=true, unit_diagonal=false,
          transpose_a=NO_TRANSPOSE
      s_ra = c64[1,3]{1,0} triangular-solve(a_inf, b_r),
          left_side=false, lower=true, unit_diagonal=false, transpose_a=ADJOINT
      s_ovf = c64[2,1]{1,0} triangular-solve(a_ovf, b_ovf),
          left_side=true, lower=true, unit_diagonal=false, transpose_a=ADJOINT
      ROOT out = (c64[3,1]{1,0}, c64[3,1]{1,0}, c64[1,3]{1,0}, c64[1,3]{1,0},
                  c64[2,1]{1,0}) tuple(s_lt, s_la, s_rn, s_ra, s_ovf)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  TriangularSolveExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&expander, module.get()));
  EXPECT_TRUE(changed);

  const float pinf = std::numeric_limits<float>::infinity();
  // A = i * [[2, 0, 0], [1, +Inf, 0], [1, 1, 2]], B = [4 - 2i, 4 - 2i, 4 - 2i]:
  //   A^T X = B and X A = B yield [(-0.5, -1), (0, 0), (-1, -2)].
  //   A^H X = B yields [(0.5, 1), (0, 0), (1, 2)].
  //   X A^H = B yields [(1, 2), (0, 0), (0.5, 1)].
  Literal a_inf = LiteralUtil::CreateR2<complex64>({
      {complex64(0.0f, 2.0f), complex64(0.0f, 0.0f), complex64(0.0f, 0.0f)},
      {complex64(0.0f, 1.0f), complex64(0.0f, pinf), complex64(0.0f, 0.0f)},
      {complex64(0.0f, 1.0f), complex64(0.0f, 1.0f), complex64(0.0f, 2.0f)},
  });
  Literal b_l = LiteralUtil::CreateR2<complex64>({{complex64(4.0f, -2.0f)},
                                                  {complex64(4.0f, -2.0f)},
                                                  {complex64(4.0f, -2.0f)}});
  Literal b_r = LiteralUtil::CreateR2<complex64>(
      {{complex64(4.0f, -2.0f), complex64(4.0f, -2.0f),
        complex64(4.0f, -2.0f)}});
  // Exercises the post-dot-divide (left_side=true, ADJOINT) path to ensure
  // overflowing complex outputs preserve (+Inf, 0) without introducing NaN.
  Literal a_ovf = LiteralUtil::CreateR2<complex64>(
      {{complex64(0x1p-100f, 0.0f), complex64(0.0f, 0.0f)},
       {complex64(0.0f, 0.0f), complex64(1.0f, 0.0f)}});
  Literal b_ovf = LiteralUtil::CreateR2<complex64>(
      {{complex64(0x1p100f, 0.0f)}, {complex64(1.0f, 0.0f)}});

  ASSERT_OK_AND_ASSIGN(
      Literal res,
      Execute(std::move(module), {&a_inf, &b_l, &b_r, &a_ovf, &b_ovf}));
  std::vector<Literal> out = res.DecomposeTuple();

  EXPECT_TRUE(LiteralTestUtil::NearOrEqual(
      LiteralUtil::CreateR2<complex64>({{complex64(-0.5f, -1.0f)},
                                        {complex64(0.0f, 0.0f)},
                                        {complex64(-1.0f, -2.0f)}}),
      out[0], ErrorSpec{1e-4f, 1e-4f}));
  EXPECT_TRUE(LiteralTestUtil::NearOrEqual(
      LiteralUtil::CreateR2<complex64>({{complex64(0.5f, 1.0f)},
                                        {complex64(0.0f, 0.0f)},
                                        {complex64(1.0f, 2.0f)}}),
      out[1], ErrorSpec{1e-4f, 1e-4f}));
  EXPECT_TRUE(LiteralTestUtil::NearOrEqual(
      LiteralUtil::CreateR2<complex64>(
          {{complex64(-0.5f, -1.0f), complex64(0.0f, 0.0f),
            complex64(-1.0f, -2.0f)}}),
      out[2], ErrorSpec{1e-4f, 1e-4f}));
  EXPECT_TRUE(LiteralTestUtil::NearOrEqual(
      LiteralUtil::CreateR2<complex64>(
          {{complex64(1.0f, 2.0f), complex64(0.0f, 0.0f),
            complex64(0.5f, 1.0f)}}),
      out[3], ErrorSpec{1e-4f, 1e-4f}));
  EXPECT_TRUE(std::isinf(out[4].Get<complex64>({0, 0}).real()));
  EXPECT_FALSE(std::isnan(out[4].Get<complex64>({0, 0}).imag()));
  EXPECT_EQ(out[4].Get<complex64>({1, 0}), complex64(1.0f, 0.0f));
}

TEST_F(TriangularSolveExpanderNumericsTest, F16ExtremeScaleAvoidsOverflow) {
  constexpr const char* kHlo = R"(
    HloModule F16ExtremeScale
    ENTRY main {
      a_big = f16[2,2]{1,0} parameter(0)
      b_big = f16[2,1]{1,0} parameter(1)
      a_tiny = f16[2,2]{1,0} parameter(2)
      b_tiny = f16[2,1]{1,0} parameter(3)
      b_tiny_r = f16[1,2]{1,0} parameter(4)
      s_big = f16[2,1]{1,0} triangular-solve(a_big, b_big),
          left_side=true, lower=true, unit_diagonal=false,
          transpose_a=NO_TRANSPOSE
      s_tiny = f16[2,1]{1,0} triangular-solve(a_tiny, b_tiny),
          left_side=true, lower=true, unit_diagonal=false,
          transpose_a=NO_TRANSPOSE
      s_tiny_r = f16[1,2]{1,0} triangular-solve(a_tiny, b_tiny_r),
          left_side=false, lower=true, unit_diagonal=false,
          transpose_a=NO_TRANSPOSE
      c_big = f32[2,1]{1,0} convert(s_big)
      c_tiny = f32[2,1]{1,0} convert(s_tiny)
      c_tiny_r = f32[1,2]{1,0} convert(s_tiny_r)
      ROOT out = (f32[2,1]{1,0}, f32[2,1]{1,0}, f32[1,2]{1,0})
          tuple(c_big, c_tiny, c_tiny_r)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  TriangularSolveExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&expander, module.get()));
  EXPECT_TRUE(changed);

  // Tiny F16 inputs overflow unfactored A^{-1}[1,0] = -16 / 2^-14 = -2^18 in
  // F16 (max 65504); large F16 inputs underflow A^{-1}[1,0] to subnormal.
  constexpr float kBig16 = 0x1p14f;
  constexpr float kTiny16 = 0x1p-14f;
  Literal a_big = LiteralUtil::CreateR2<Eigen::half>(
      {{Eigen::half(2 * kBig16), Eigen::half(0.0f)},
       {Eigen::half(0.25f * kBig16), Eigen::half(2 * kBig16)}});
  Literal b_big = LiteralUtil::CreateR2<Eigen::half>(
      {{Eigen::half(2 * kBig16)}, {Eigen::half(2.25f * kBig16)}});
  Literal a_tiny = LiteralUtil::CreateR2<Eigen::half>(
      {{Eigen::half(kTiny16), Eigen::half(0.0f)},
       {Eigen::half(16 * kTiny16), Eigen::half(kTiny16)}});
  Literal b_tiny = LiteralUtil::CreateR2<Eigen::half>(
      {{Eigen::half(kTiny16)}, {Eigen::half(17 * kTiny16)}});
  Literal b_tiny_r = LiteralUtil::CreateR2<Eigen::half>(
      {{Eigen::half(17 * kTiny16), Eigen::half(kTiny16)}});

  ASSERT_OK_AND_ASSIGN(Literal actual,
                       Execute(std::move(module),
                               {&a_big, &b_big, &a_tiny, &b_tiny, &b_tiny_r}));
  Literal expected = LiteralUtil::MakeTupleOwned(
      LiteralUtil::CreateR2<float>({{1.0f}, {1.0f}}),
      LiteralUtil::CreateR2<float>({{1.0f}, {1.0f}}),
      LiteralUtil::CreateR2<float>({{1.0f, 1.0f}}));
  EXPECT_TRUE(
      LiteralTestUtil::NearOrEqual(expected, actual, ErrorSpec{1e-2f, 1e-2f}));
}

}  // namespace
}  // namespace xla
