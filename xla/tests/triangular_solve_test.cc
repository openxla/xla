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

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/lib/matrix.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
constexpr complex64 kNanC64 = complex64(kNan, kNan);

using TriangularSolveTest =
    ClientLibraryTestRunnerMixin<HloPjRtInterpreterReferenceMixin<HloTestBase>>;
using TriangularSolveLeftLookingTest =
    ClientLibraryTestRunnerMixin<HloPjRtInterpreterReferenceMixin<HloTestBase>>;

Array2D<float> AValsLower() {
  return {{2, kNan, kNan, kNan},
          {3, 6, kNan, kNan},
          {4, 7, 9, kNan},
          {5, 8, 10, 11}};
}

Array2D<float> AValsUpper() {
  return {{2, 3, 4, 5},
          {kNan, 6, 7, 8},
          {kNan, kNan, 9, 10},
          {kNan, kNan, kNan, 11}};
}

Array2D<float> AValsLowerUnitDiagonal() {
  return {{kNan, kNan, kNan, kNan},
          {3, kNan, kNan, kNan},
          {4, 7, kNan, kNan},
          {5, 8, 10, kNan}};
}

Array2D<float> AValsUpperUnitDiagonal() {
  return {{kNan, 3, 4, 5},
          {kNan, kNan, 7, 8},
          {kNan, kNan, kNan, 10},
          {kNan, kNan, kNan, kNan}};
}

Array2D<float> BValsRight() {
  return {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
}

Array2D<float> BValsLeft() {
  return {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
}

Array2D<complex64> AValsLowerComplex() {
  return {{2, kNanC64, kNanC64, kNanC64},
          {complex64(3, 1), 6, kNanC64, kNanC64},
          {4, complex64(7, 2), 9, kNanC64},
          {5, 8, complex64(10, 3), 11}};
}

Array2D<complex64> AValsUpperComplex() {
  return {{2, 3, complex64(4, 3), 5},
          {kNanC64, 6, complex64(7, 2), 8},
          {kNanC64, kNanC64, complex64(9, 1), 10},
          {kNanC64, kNanC64, kNanC64, 11}};
}

Array2D<complex64> BValsRightComplex() {
  return {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
}

Array2D<complex64> BValsLeftComplex() {
  return {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
}

TEST_F(TriangularSolveTest, EmptyArrays) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data =
      CreateR2Parameter<float>(Array2D<float>(0, 0), 0, "a", &builder, &a);
  auto b_data =
      CreateR2Parameter<float>(Array2D<float>(0, 10), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/true, /*lower=*/true,
                  /*unit_diagonal=*/false,
                  /*transpose_a=*/TriangularSolveOptions::TRANSPOSE);

  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 10),
                             {&a_data, &b_data});
}

TEST_F(TriangularSolveTest, SimpleRightLowerTranspose) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data = CreateR2Parameter<float>(AValsLower(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsRight(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/false, /*lower=*/true,
                  /*unit_diagonal=*/false,
                  /*transpose_a=*/TriangularSolveOptions::TRANSPOSE);

  Array2D<float> expected({
      {0.5, 0.08333334, 0.04629629, 0.03367003},
      {2.5, -0.25, -0.1388889, -0.1010101},
      {4.5, -0.58333331, -0.32407406, -0.23569024},
  });

  ComputeAndCompareR2<float>(&builder, expected, {&a_data, &b_data},
                             ErrorSpec(1e-2, 1e-2));
}

TEST_F(TriangularSolveTest, SimpleRightLowerNotranspose) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data = CreateR2Parameter<float>(AValsLower(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsRight(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/false, /*lower=*/true,
                  /*unit_diagonal=*/false,
                  /*transpose_a=*/TriangularSolveOptions::NO_TRANSPOSE);

  Array2D<float> expected({
      {-0.16414141, -0.06902357, -0.07070707, 0.36363636},
      {0.64393939, 0.06565657, -0.03030303, 0.72727273},
      {1.4520202, 0.2003367, 0.01010101, 1.09090909},
  });

  ComputeAndCompareR2<float>(&builder, expected, {&a_data, &b_data},
                             ErrorSpec(1e-2, 1e-2));
}

TEST_F(TriangularSolveTest, SimpleRightUpperTranspose) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data = CreateR2Parameter<float>(AValsUpper(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsRight(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/false, /*lower=*/false,
                  /*unit_diagonal=*/false,
                  /*transpose_a=*/TriangularSolveOptions::TRANSPOSE);

  Array2D<float> expected({
      {-0.16414141, -0.06902357, -0.07070707, 0.36363636},
      {0.64393939, 0.06565657, -0.03030303, 0.72727273},
      {1.4520202, 0.2003367, 0.01010101, 1.09090909},
  });

  ComputeAndCompareR2<float>(&builder, expected, {&a_data, &b_data},
                             ErrorSpec(1e-2, 1e-2));
}

TEST_F(TriangularSolveTest, SimpleRightUpperNotranspose) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data = CreateR2Parameter<float>(AValsUpper(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsRight(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/false, /*lower=*/false,
                  /*unit_diagonal=*/false,
                  /*transpose_a=*/TriangularSolveOptions::NO_TRANSPOSE);

  Array2D<float> expected({
      {0.5, 0.08333334, 0.04629629, 0.03367003},
      {2.5, -0.25, -0.1388889, -0.1010101},
      {4.5, -0.58333331, -0.32407406, -0.23569024},
  });

  ComputeAndCompareR2<float>(&builder, expected, {&a_data, &b_data},
                             ErrorSpec(1e-2, 1e-2));
}

TEST_F(TriangularSolveTest, SimpleLeftLowerTranspose) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data = CreateR2Parameter<float>(AValsLower(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsLeft(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/true, /*lower=*/true,
                  /*unit_diagonal=*/false,
                  /*transpose_a=*/TriangularSolveOptions::TRANSPOSE);

  Array2D<float> expected({
      {-0.89646465, -0.69444444, -0.49242424},
      {-0.27441077, -0.24074074, -0.20707071},
      {-0.23232323, -0.22222222, -0.21212121},
      {0.90909091, 1., 1.09090909},
  });

  ComputeAndCompareR2<float>(&builder, expected, {&a_data, &b_data},
                             ErrorSpec(1e-2, 1e-2));
}

TEST_F(TriangularSolveTest, SimpleLeftLowerNotranspose) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data = CreateR2Parameter<float>(AValsLower(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsLeft(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/true, /*lower=*/true,
                  /*unit_diagonal=*/false,
                  /*transpose_a=*/TriangularSolveOptions::NO_TRANSPOSE);

  Array2D<float> expected({
      {0.5, 1.0, 1.5},
      {0.41666667, 0.33333333, 0.25},
      {0.23148148, 0.18518519, 0.13888889},
      {0.16835017, 0.13468013, 0.1010101},
  });

  ComputeAndCompareR2<float>(&builder, expected, {&a_data, &b_data},
                             ErrorSpec(1e-2, 1e-2));
}

TEST_F(TriangularSolveTest, SimpleLeftLowerNoTransposeUnitDiagonal) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data =
      CreateR2Parameter<float>(AValsLowerUnitDiagonal(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsLeft(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/true, /*lower=*/true,
                  /*unit_diagonal=*/true,
                  /*transpose_a=*/TriangularSolveOptions::NO_TRANSPOSE);

  Array2D<float> expected(
      {{1., 2., 3.}, {1., -1., -3.}, {-4., 7., 18.}, {37., -61., -159.}});

  ComputeAndCompareR2<float>(&builder, expected, {&a_data, &b_data},
                             ErrorSpec(1e-2, 1e-2));
}

TEST_F(TriangularSolveTest, SimpleLeftLowerNotransposeIrregularblock) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data = CreateR2Parameter<float>(AValsLower(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsLeft(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/true, /*lower=*/true,
                  /*unit_diagonal=*/false,
                  /*transpose_a=*/TriangularSolveOptions::NO_TRANSPOSE);

  Array2D<float> expected({
      {0.5, 1.0, 1.5},
      {0.41666667, 0.33333333, 0.25},
      {0.23148148, 0.18518519, 0.13888889},
      {0.16835017, 0.13468013, 0.1010101},
  });

  ComputeAndCompareR2<float>(&builder, expected, {&a_data, &b_data},
                             ErrorSpec(1e-2, 1e-2));
}

TEST_F(TriangularSolveTest, SimpleLeftUpperTranspose) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data = CreateR2Parameter<float>(AValsUpper(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsLeft(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/true, /*lower=*/false,
                  /*unit_diagonal=*/false,
                  /*transpose_a=*/TriangularSolveOptions::TRANSPOSE);

  Array2D<float> expected({
      {0.5, 1.0, 1.5},
      {0.41666667, 0.33333333, 0.25},
      {0.23148148, 0.18518519, 0.13888889},
      {0.16835017, 0.13468013, 0.1010101},
  });

  ComputeAndCompareR2<float>(&builder, expected, {&a_data, &b_data},
                             ErrorSpec(1e-2, 1e-2));
}

TEST_F(TriangularSolveTest, SimpleLeftUpperNotranspose) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data = CreateR2Parameter<float>(AValsUpper(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsLeft(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/true, /*lower=*/false,
                  /*unit_diagonal=*/false,
                  /*transpose_a=*/TriangularSolveOptions::NO_TRANSPOSE);

  Array2D<float> expected({
      {-0.89646465, -0.69444444, -0.49242424},
      {-0.27441077, -0.24074074, -0.20707071},
      {-0.23232323, -0.22222222, -0.21212121},
      {0.90909091, 1., 1.09090909},
  });

  ComputeAndCompareR2<float>(&builder, expected, {&a_data, &b_data},
                             ErrorSpec(1e-2, 1e-2));
}

TEST_F(TriangularSolveTest, SimpleLeftUpperNotransposeUnitDiagonal) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data =
      CreateR2Parameter<float>(AValsUpperUnitDiagonal(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsLeft(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/true, /*lower=*/false,
                  /*unit_diagonal=*/true,
                  /*transpose_a=*/TriangularSolveOptions::NO_TRANSPOSE);

  Array2D<float> expected({{-1402., -1538., -1674.},
                           {575., 631., 687.},
                           {-93., -102., -111.},
                           {10., 11., 12.}});

  ComputeAndCompareR2<float>(&builder, expected, {&a_data, &b_data},
                             ErrorSpec(1e-2, 1e-2));
}

TEST_F(TriangularSolveTest, SimpleRightLowerTransposeConjugate) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data =
      CreateR2Parameter<complex64>(AValsLowerComplex(), 0, "a", &builder, &a);
  auto b_data =
      CreateR2Parameter<complex64>(BValsRightComplex(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/false, /*lower=*/true,
                  /*unit_diagonal=*/false,
                  /*transpose_a=*/TriangularSolveOptions::ADJOINT);

  Array2D<complex64> expected({
      {0.5, complex64(0.08333333, 0.08333333),
       complex64(0.02777778, -0.0462963), complex64(0.06313131, -0.01094276)},
      {2.5, complex64(-0.25, 0.41666667), complex64(-0.23148148, -0.37962963),
       complex64(0.08670034, -0.02104377)},
      {4.5, complex64(-0.58333333, 0.75), complex64(-0.49074074, -0.71296296),
       complex64(0.11026936, -0.03114478)},
  });

  ComputeAndCompareR2<complex64>(&builder, expected, {&a_data, &b_data},
                                 ErrorSpec(1e-2, 1e-2));
}

TEST_F(TriangularSolveTest, SimpleLeftUpperTransposeNoconjugate) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data =
      CreateR2Parameter<complex64>(AValsUpperComplex(), 0, "a", &builder, &a);
  auto b_data =
      CreateR2Parameter<complex64>(BValsLeftComplex(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/true, /*lower=*/false,
                  /*unit_diagonal=*/false,
                  /*transpose_a=*/TriangularSolveOptions::TRANSPOSE);

  Array2D<complex64> expected({
      {0.5, 1., 1.5},
      {0.41666667, 0.33333333, 0.25},
      {complex64(0.20020325, -2.81504065e-01),
       complex64(0.13821138, -4.22764228e-01),
       complex64(0.07621951, -5.64024390e-01)},
      {complex64(0.19678492, 2.55912786e-01),
       complex64(0.17738359, 3.84331116e-01),
       complex64(0.15798226, 5.12749446e-01)},
  });

  ComputeAndCompareR2<complex64>(&builder, expected, {&a_data, &b_data},
                                 ErrorSpec(1e-2, 1e-2));
}

TEST_F(TriangularSolveTest, BatchedLeftUpper) {
  XlaBuilder builder(TestName());

  Array3D<float> bvals(7, 5, 5);
  bvals.FillIota(1.);

  // Set avals to the upper triangle of bvals.
  Array3D<float> avals = bvals;
  avals.Each([](absl::Span<const int64_t> indices, float* value) {
    if (indices[1] > indices[2]) {
      *value = 0;
    }
  });

  XlaOp a, b;
  auto a_data = CreateR3Parameter<float>(avals, 0, "a", &builder, &a);
  auto b_data = CreateR3Parameter<float>(bvals, 1, "b", &builder, &b);
  BatchDot(
      ConstantR3FromArray3D(&builder, avals),
      TriangularSolve(a, b,
                      /*left_side=*/true, /*lower=*/false,
                      /*unit_diagonal=*/false,
                      /*transpose_a=*/TriangularSolveOptions::NO_TRANSPOSE));

  ComputeAndCompareR3<float>(&builder, bvals, {&a_data, &b_data},
                             ErrorSpec(1e-2, 1e-2));
}

TEST_F(TriangularSolveTest, ExtremeScaleAvoidsBlockInversionFtzAndOverflow) {
  XlaBuilder builder(TestName());
  constexpr float kBig = 0x1p123f;
  constexpr float kTiny = 0x1p-123f;
  Array2D<float> a_lo({{2 * kBig, kNan, kNan},
                       {0.5f * kBig, 3 * kBig, kNan},
                       {0.0f, 0.5f * kBig, 4 * kBig}});
  Array2D<float> a_up({{2 * kBig, 0.5f * kBig, 0.0f},
                       {kNan, 3 * kBig, 0.5f * kBig},
                       {kNan, kNan, 4 * kBig}});
  Array2D<float> b_big({{kBig}, {kBig}, {kBig}});
  Array2D<float> a_tiny({{kTiny, 0.0f}, {32 * kTiny, kTiny}});
  Array2D<float> b_tiny({{kTiny}, {32 * kTiny}});

  // Right-side solves: X @ A = B and X @ A^T = B.
  Array2D<float> a_right({{4 * kBig, kNan, kNan},
                          {0.5f * kBig, 4 * kBig, kNan},
                          {0.0f, 0.5f * kBig, 4 * kBig}});
  Array2D<float> b_right({{4.5f * kBig, 4.5f * kBig, 4.0f * kBig}});
  Array2D<float> b_right_t({{4.0f * kBig, 4.5f * kBig, 4.5f * kBig}});
  Array2D<float> b_tiny_right({{33 * kTiny, kTiny}});

  // Mixed-scale batches of 32x32 matrices: batch=32, size=32 exercises both
  // SolveByInvertingDiagonalBlocks on CPU (size >= 32) and batched-in-lanes
  // block inversion on TPU (batch >= 32, size <= 100).
  constexpr int kBatch = 32;
  constexpr int kMatSize = 32;
  Array3D<float> a_bat(kBatch, kMatSize, kMatSize, 0.0f);
  Array3D<float> b_bat(kBatch, kMatSize, 1, 0.0f);
  Array3D<float> exp_bat(kBatch, kMatSize, 1, 0.0f);
  for (int i = 0; i < kBatch; ++i) {
    if (i % 2 == 0) {
      for (int r = 0; r < kMatSize; ++r) {
        a_bat(i, r, r) = 4 * kBig;
        if (r > 0) {
          a_bat(i, r, r - 1) = 0.5f * kBig;
        }
        b_bat(i, r, 0) = (r == 0 ? 4.0f : 4.5f) * kBig;
        exp_bat(i, r, 0) = 1.0f;
      }
    } else {
      for (int r = 0; r < kMatSize; ++r) {
        a_bat(i, r, r) = kTiny;
        if (r % 2 == 1) {
          a_bat(i, r, r - 1) = 32 * kTiny;
          b_bat(i, r, 0) = 32 * kTiny;
          exp_bat(i, r, 0) = 0.0f;
        } else {
          b_bat(i, r, 0) = kTiny;
          exp_bat(i, r, 0) = 1.0f;
        }
      }
    }
  }

  // Complex C64 extreme scale under ADJOINT (left_side=true and false).
  // Pre-fix A^{-1}[1,0] = -(2^-127, 2^-127) falls below FLT_MIN and hits FTZ.
  constexpr float kCBig = 0x1p121f;
  Array2D<complex64> a_c64({{complex64(2 * kCBig, 2 * kCBig), kNanC64},
                            {complex64(0.25f * kCBig, 0.25f * kCBig),
                             complex64(4 * kCBig, -4 * kCBig)}});
  Array2D<complex64> b_c64_left({{complex64(3.25f * kCBig, 0.75f * kCBig)},
                                 {complex64(4 * kCBig, 4 * kCBig)}});
  Array2D<complex64> b_c64_right({{complex64(2 * kCBig, -2 * kCBig),
                                   complex64(-3.75f * kCBig, 3.75f * kCBig)}});

  XlaOp p_alo, p_aup, p_bbig, p_atiny, p_btiny, p_aright, p_bright, p_bright_t,
      p_btiny_r, p_abat, p_bbat, p_ac64, p_bc64l, p_bc64r;
  Literal d_alo = CreateR2Parameter<float>(a_lo, 0, "a_lo", &builder, &p_alo);
  Literal d_aup = CreateR2Parameter<float>(a_up, 1, "a_up", &builder, &p_aup);
  Literal d_bbig =
      CreateR2Parameter<float>(b_big, 2, "b_big", &builder, &p_bbig);
  Literal d_atiny =
      CreateR2Parameter<float>(a_tiny, 3, "a_tiny", &builder, &p_atiny);
  Literal d_btiny =
      CreateR2Parameter<float>(b_tiny, 4, "b_tiny", &builder, &p_btiny);
  Literal d_aright =
      CreateR2Parameter<float>(a_right, 5, "a_right", &builder, &p_aright);
  Literal d_bright =
      CreateR2Parameter<float>(b_right, 6, "b_right", &builder, &p_bright);
  Literal d_bright_t = CreateR2Parameter<float>(b_right_t, 7, "b_right_t",
                                                &builder, &p_bright_t);
  Literal d_btiny_r = CreateR2Parameter<float>(b_tiny_right, 8, "b_tiny_r",
                                               &builder, &p_btiny_r);
  Literal d_abat =
      CreateR3Parameter<float>(a_bat, 9, "a_bat", &builder, &p_abat);
  Literal d_bbat =
      CreateR3Parameter<float>(b_bat, 10, "b_bat", &builder, &p_bbat);
  Literal d_ac64 =
      CreateR2Parameter<complex64>(a_c64, 11, "a_c64", &builder, &p_ac64);
  Literal d_bc64l = CreateR2Parameter<complex64>(b_c64_left, 12, "b_c64_l",
                                                 &builder, &p_bc64l);
  Literal d_bc64r = CreateR2Parameter<complex64>(b_c64_right, 13, "b_c64_r",
                                                 &builder, &p_bc64r);

  constexpr auto kNoTrans = TriangularSolveOptions::NO_TRANSPOSE;
  constexpr auto kTrans = TriangularSolveOptions::TRANSPOSE;
  constexpr auto kAdjoint = TriangularSolveOptions::ADJOINT;
  XlaOp r_lo =
      TriangularSolve(p_alo, p_bbig, /*left_side=*/true,
                      /*lower=*/true, /*unit_diagonal=*/false, kNoTrans);
  XlaOp r_up =
      TriangularSolve(p_aup, p_bbig, /*left_side=*/true,
                      /*lower=*/false, /*unit_diagonal=*/false, kNoTrans);
  XlaOp r_bat =
      TriangularSolve(p_abat, p_bbat, /*left_side=*/true,
                      /*lower=*/true, /*unit_diagonal=*/false, kNoTrans);
  XlaOp r_bf16 = ConvertElementType(
      TriangularSolve(ConvertElementType(p_alo, BF16),
                      ConvertElementType(p_bbig, BF16), /*left_side=*/true,
                      /*lower=*/true, /*unit_diagonal=*/false, kNoTrans),
      F32);
  XlaOp r_tiny =
      TriangularSolve(p_atiny, p_btiny, /*left_side=*/true,
                      /*lower=*/true, /*unit_diagonal=*/false, kNoTrans);
  XlaOp r_right =
      TriangularSolve(p_aright, p_bright, /*left_side=*/false,
                      /*lower=*/true, /*unit_diagonal=*/false, kNoTrans);
  XlaOp r_right_t =
      TriangularSolve(p_aright, p_bright_t, /*left_side=*/false,
                      /*lower=*/true, /*unit_diagonal=*/false, kTrans);
  XlaOp r_tiny_r =
      TriangularSolve(p_atiny, p_btiny_r, /*left_side=*/false,
                      /*lower=*/true, /*unit_diagonal=*/false, kNoTrans);
  XlaOp r_c64_l =
      TriangularSolve(p_ac64, p_bc64l, /*left_side=*/true,
                      /*lower=*/true, /*unit_diagonal=*/false, kAdjoint);
  XlaOp r_c64_r =
      TriangularSolve(p_ac64, p_bc64r, /*left_side=*/false,
                      /*lower=*/true, /*unit_diagonal=*/false, kAdjoint);
  Tuple(&builder, {r_lo, r_up, r_bat, r_bf16, r_tiny, r_right, r_right_t,
                   r_tiny_r, r_c64_l, r_c64_r});

  Literal expected = LiteralUtil::MakeTupleOwned(
      LiteralUtil::CreateR2<float>({{0.5f}, {0.25f}, {0.21875f}}),
      LiteralUtil::CreateR2<float>({{0.42708334f}, {0.29166666f}, {0.25f}}),
      LiteralUtil::CreateR3FromArray3D(exp_bat),
      LiteralUtil::CreateR2<float>({{0.5f}, {0.25f}, {0.21875f}}),
      LiteralUtil::CreateR2<float>({{1.0f}, {0.0f}}),
      LiteralUtil::CreateR2<float>({{1.0f, 1.0f, 1.0f}}),
      LiteralUtil::CreateR2<float>({{1.0f, 1.0f, 1.0f}}),
      LiteralUtil::CreateR2<float>({{1.0f, 1.0f}}),
      LiteralUtil::CreateR2<complex64>(
          {{complex64(0.5f, 1.0f)}, {complex64(1.0f, 0.0f)}}),
      LiteralUtil::CreateR2<complex64>(
          {{complex64(1.0f, 0.0f), complex64(0.0f, 1.0f)}}));
  ComputeAndCompareTuple(
      &builder, expected,
      {&d_alo, &d_aup, &d_bbig, &d_atiny, &d_btiny, &d_aright, &d_bright,
       &d_bright_t, &d_btiny_r, &d_abat, &d_bbat, &d_ac64, &d_bc64l, &d_bc64r},
      ErrorSpec(1e-3f, 1e-3f));
}

TEST_F(TriangularSolveTest, MixedScaleDiagonalsAndNearFltMinTransposed) {
  XlaBuilder builder(TestName());
  // Mixed diagonal scales that global prescaling would overflow or flush.
  Array2D<float> a_mix1(
      {{0x1p-90f, 0.0f, 0.0f}, {0.0f, 0x1p90f, 0.0f}, {0.0f, 0.0f, 0x1p-90f}});
  Array2D<float> b_mix1({{1.0f}, {1.0f}, {0x1p-90f}});

  Array2D<float> a_mix2(
      {{0x1p-100f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}});
  Array2D<float> b_mix2({{0x1p-100f}, {0x1p100f}, {0x1p-100f}});

  Array2D<float> a_mix3({{0x1p10f, 0.0f, 0.0f},
                         {0x1p120f, 0x1p100f, 0.0f},
                         {0.0f, 0.0f, 0x1p-120f}});
  Array2D<float> b_mix3({{0x1p10f}, {0x1p120f}, {0x1p-120f}});

  // Near-FLT_MIN transposed solve: A^T X = B where unfactored A^{-1}[1,0]
  // = -2^-127 flushes to zero on FTZ targets. Checked with abs=0 so the
  // relative tolerance is non-vacuous near FLT_MIN.
  constexpr float kBig = 0x1p123f;
  Array2D<float> a_near_min({{2 * kBig, 0.0f}, {0.5f * kBig, 4 * kBig}});
  Array2D<float> b_unit({{1.0f}, {1.0f}});

  XlaOp p_am1, p_bm1, p_am2, p_bm2, p_am3, p_bm3, p_amin, p_bunit;
  Literal d_am1 =
      CreateR2Parameter<float>(a_mix1, 0, "a_mix1", &builder, &p_am1);
  Literal d_bm1 =
      CreateR2Parameter<float>(b_mix1, 1, "b_mix1", &builder, &p_bm1);
  Literal d_am2 =
      CreateR2Parameter<float>(a_mix2, 2, "a_mix2", &builder, &p_am2);
  Literal d_bm2 =
      CreateR2Parameter<float>(b_mix2, 3, "b_mix2", &builder, &p_bm2);
  Literal d_am3 =
      CreateR2Parameter<float>(a_mix3, 4, "a_mix3", &builder, &p_am3);
  Literal d_bm3 =
      CreateR2Parameter<float>(b_mix3, 5, "b_mix3", &builder, &p_bm3);
  Literal d_amin =
      CreateR2Parameter<float>(a_near_min, 6, "a_min", &builder, &p_amin);
  Literal d_bunit =
      CreateR2Parameter<float>(b_unit, 7, "b_unit", &builder, &p_bunit);

  constexpr auto kNoTrans = TriangularSolveOptions::NO_TRANSPOSE;
  constexpr auto kTrans = TriangularSolveOptions::TRANSPOSE;
  XlaOp r_mix1 =
      TriangularSolve(p_am1, p_bm1, /*left_side=*/true,
                      /*lower=*/true, /*unit_diagonal=*/false, kNoTrans);
  XlaOp r_mix2 =
      TriangularSolve(p_am2, p_bm2, /*left_side=*/true,
                      /*lower=*/true, /*unit_diagonal=*/false, kNoTrans);
  XlaOp r_mix3 =
      TriangularSolve(p_am3, p_bm3, /*left_side=*/true,
                      /*lower=*/true, /*unit_diagonal=*/false, kNoTrans);
  XlaOp r_min_t =
      TriangularSolve(p_amin, p_bunit, /*left_side=*/true,
                      /*lower=*/true, /*unit_diagonal=*/false, kTrans);
  Tuple(&builder, {r_mix1, r_mix2, r_mix3, r_min_t});

  Literal expected = LiteralUtil::MakeTupleOwned(
      LiteralUtil::CreateR2<float>({{0x1p90f}, {0x1p-90f}, {1.0f}}),
      LiteralUtil::CreateR2<float>({{1.0f}, {0x1p100f}, {0x1p-100f}}),
      LiteralUtil::CreateR2<float>({{1.0f}, {0.0f}, {1.0f}}),
      LiteralUtil::CreateR2<float>({{1.75f * 0x1p-125f}, {0x1p-125f}}));
  ComputeAndCompareTuple(
      &builder, expected,
      {&d_am1, &d_bm1, &d_am2, &d_bm2, &d_am3, &d_bm3, &d_amin, &d_bunit},
      ErrorSpec(0.0f, 1e-3f));
}

TEST_F(TriangularSolveTest, MultiBlockExtremeAndMixedScale) {
  XlaBuilder builder(TestName());
  constexpr int kSize = 132;  // > block_size (128) to exercise multi-block.
  constexpr float kBig = 0x1p123f;

  Array2D<float> a_big(kSize, kSize, 0.0f);
  Array2D<float> b_left(kSize, 1, 0.0f);
  Array2D<float> b_left_t(kSize, 1, 0.0f);
  Array2D<float> b_right(1, kSize, 0.0f);
  Array2D<float> b_right_t(1, kSize, 0.0f);
  Array2D<float> exp_col(kSize, 1, 1.0f);
  Array2D<float> exp_row(1, kSize, 1.0f);
  for (int r = 0; r < kSize; ++r) {
    a_big(r, r) = 4 * kBig;
    if (r > 0) {
      a_big(r, r - 1) = 0.5f * kBig;
    }
  }
  a_big(130, 0) = 0.25f * kBig;
  for (int r = 0; r < kSize; ++r) {
    float row_sum = 4.0f * kBig + (r > 0 ? 0.5f * kBig : 0.0f) +
                    (r == 130 ? 0.25f * kBig : 0.0f);
    float col_sum = 4.0f * kBig + (r + 1 < kSize ? 0.5f * kBig : 0.0f) +
                    (r == 0 ? 0.25f * kBig : 0.0f);
    b_left(r, 0) = row_sum;
    b_left_t(r, 0) = col_sum;
    b_right(0, r) = col_sum;
    b_right_t(0, r) = row_sum;
  }

  // Multi-block with mixed block scales and an off-diagonal coupling block.
  Array2D<float> a_mixed(kSize, kSize, 0.0f);
  Array2D<float> b_mixed(kSize, 1, 0.0f);
  for (int r = 0; r < kSize; ++r) {
    float scale = (r < 128) ? 0x1p100f : 0x1p-100f;
    a_mixed(r, r) = scale;
    b_mixed(r, 0) = scale;
  }
  a_mixed(128, 0) = 0x1p-100f;
  b_mixed(128, 0) = 2 * 0x1p-100f;

  XlaOp p_abig, p_bl, p_blt, p_br, p_brt, p_amix, p_bmix;
  Literal d_abig =
      CreateR2Parameter<float>(a_big, 0, "a_big", &builder, &p_abig);
  Literal d_bl = CreateR2Parameter<float>(b_left, 1, "b_l", &builder, &p_bl);
  Literal d_blt =
      CreateR2Parameter<float>(b_left_t, 2, "b_lt", &builder, &p_blt);
  Literal d_br = CreateR2Parameter<float>(b_right, 3, "b_r", &builder, &p_br);
  Literal d_brt =
      CreateR2Parameter<float>(b_right_t, 4, "b_rt", &builder, &p_brt);
  Literal d_amix =
      CreateR2Parameter<float>(a_mixed, 5, "a_mix", &builder, &p_amix);
  Literal d_bmix =
      CreateR2Parameter<float>(b_mixed, 6, "b_mix", &builder, &p_bmix);

  constexpr auto kNoTrans = TriangularSolveOptions::NO_TRANSPOSE;
  constexpr auto kTrans = TriangularSolveOptions::TRANSPOSE;
  XlaOp r_l = TriangularSolve(p_abig, p_bl, /*left_side=*/true, /*lower=*/true,
                              /*unit_diagonal=*/false, kNoTrans);
  XlaOp r_lt = TriangularSolve(p_abig, p_blt, /*left_side=*/true,
                               /*lower=*/true, /*unit_diagonal=*/false, kTrans);
  XlaOp r_r = TriangularSolve(p_abig, p_br, /*left_side=*/false, /*lower=*/true,
                              /*unit_diagonal=*/false, kNoTrans);
  XlaOp r_rt = TriangularSolve(p_abig, p_brt, /*left_side=*/false,
                               /*lower=*/true, /*unit_diagonal=*/false, kTrans);
  XlaOp r_mix =
      TriangularSolve(p_amix, p_bmix, /*left_side=*/true,
                      /*lower=*/true, /*unit_diagonal=*/false, kNoTrans);
  Tuple(&builder, {r_l, r_lt, r_r, r_rt, r_mix});

  Literal expected =
      LiteralUtil::MakeTupleOwned(LiteralUtil::CreateR2FromArray2D(exp_col),
                                  LiteralUtil::CreateR2FromArray2D(exp_col),
                                  LiteralUtil::CreateR2FromArray2D(exp_row),
                                  LiteralUtil::CreateR2FromArray2D(exp_row),
                                  LiteralUtil::CreateR2FromArray2D(exp_col));
  ComputeAndCompareTuple(
      &builder, expected,
      {&d_abig, &d_bl, &d_blt, &d_br, &d_brt, &d_amix, &d_bmix},
      ErrorSpec(1e-3f, 1e-3f));
}

struct TriangularSolveTestSpec {
  std::vector<int64_t> dims;  // [..., m, n] A is mxm, B is mxn
  bool left_side;
  bool lower;
  TriangularSolveOptions::Transpose transpose_a;
};

class TriangularSolveParametricTest
    : public ClientLibraryTestRunnerMixin<
          HloPjRtInterpreterReferenceMixin<HloTestBase>>,
      public ::testing::WithParamInterface<TriangularSolveTestSpec> {};

TEST_P(TriangularSolveParametricTest, Random) {
  TriangularSolveTestSpec spec = GetParam();

  XlaBuilder builder(TestName());

  CHECK_GE(spec.dims.size(), 2);
  std::vector<int64_t> a_dims = spec.dims;
  a_dims.back() = a_dims.at(a_dims.size() - 2);
  Array<float> avals(a_dims);
  avals.FillRandom(1.0);
  avals.Each([](absl::Span<const int64_t> dims, float* v) {
    if (dims.back() == dims.at(dims.size() - 2)) {
      *v += 30;
    }
  });

  std::vector<int64_t> b_dims = spec.dims;
  if (!spec.left_side) {
    std::swap(b_dims.back(), b_dims.at(b_dims.size() - 2));
  }
  Array<float> bvals(b_dims);
  bvals.FillRandom(1.0);

  const Literal avals_literal = LiteralUtil::CreateFromArray(avals);
  const Literal bvals_literal = LiteralUtil::CreateFromArray(bvals);
  XlaOp a = Parameter(&builder, 0, avals_literal.shape(), "a");
  XlaOp b = Parameter(&builder, 1, bvals_literal.shape(), "b");
  XlaOp x = TriangularSolve(a, b, spec.left_side, spec.lower,
                            /*unit_diagonal=*/false, spec.transpose_a);
  XlaOp a_tri = Triangle(a, spec.lower);
  a_tri = MaybeTransposeInMinorDims(
      a_tri, spec.transpose_a != TriangularSolveOptions::NO_TRANSPOSE);
  if (spec.left_side) {
    BatchDot(a_tri, x, xla::PrecisionConfig::HIGHEST);
  } else {
    BatchDot(x, a_tri, xla::PrecisionConfig::HIGHEST);
  }

  ComputeAndCompareLiteral(&builder, LiteralUtil::CreateFromArray(bvals),
                           {&avals_literal, &bvals_literal},
                           ErrorSpec(3e-2, 3e-2));
}

std::vector<TriangularSolveTestSpec> TriangularSolveTests() {
  std::vector<TriangularSolveTestSpec> specs;
  for (auto batch : {std::vector<int64_t>{}, std::vector<int64_t>{1},
                     std::vector<int64_t>{5}, std::vector<int64_t>{65},
                     std::vector<int64_t>{129}}) {
    for (int m : {5, 10, 150}) {
      for (int n : {5, 150}) {
        for (bool left_side : {false, true}) {
          for (bool lower : {false, true}) {
            for (TriangularSolveOptions::Transpose transpose_a :
                 {TriangularSolveOptions::NO_TRANSPOSE,
                  TriangularSolveOptions::TRANSPOSE}) {
              std::vector<int64_t> dims(batch.begin(), batch.end());
              dims.push_back(m);
              dims.push_back(n);
              specs.push_back({dims, left_side, lower, transpose_a});
            }
          }
        }
      }
    }
  }
  return specs;
}

INSTANTIATE_TEST_SUITE_P(
    TriangularSolveParametricTestInstantiation, TriangularSolveParametricTest,
    ::testing::ValuesIn(TriangularSolveTests()),
    [](const ::testing::TestParamInfo<TriangularSolveTestSpec>& info) {
      const TriangularSolveTestSpec& spec = info.param;
      std::string name = absl::StrCat(
          absl::StrJoin(spec.dims, "_"), "_", spec.left_side ? "left" : "right",
          "_", spec.lower ? "lower" : "upper", "_",
          absl::AsciiStrToLower(
              TriangularSolveOptions_Transpose_Name(spec.transpose_a)));
      return name;
    });

}  // namespace
}  // namespace xla
