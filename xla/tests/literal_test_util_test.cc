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

// Tests that our utility functions for dealing with literals are correctly
// implemented.

#include "xla/tests/literal_test_util.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_join.h"
#include "xla/error_spec.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_comparison.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

TEST(LiteralTestUtilTest, ComparesEqualTuplesEqual) {
  Literal literal = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<int32_t>(42),
      LiteralUtil::CreateR0<int32_t>(64),
  });
  EXPECT_TRUE(LiteralTestUtil::Equal(literal, literal));
}

TEST(LiteralTestUtilTest, ComparesEqualComplex64TuplesEqual) {
  Literal literal = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex64>({42.0, 64.0}),
      LiteralUtil::CreateR0<complex64>({64.0, 42.0}),
  });
  EXPECT_TRUE(LiteralTestUtil::Equal(literal, literal));
}

TEST(LiteralTestUtilTest, ComparesEqualComplex128TuplesEqual) {
  Literal literal = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex128>({42.0, 64.0}),
      LiteralUtil::CreateR0<complex128>({64.0, 42.0}),
  });
  EXPECT_TRUE(LiteralTestUtil::Equal(literal, literal));
}

TEST(LiteralTestUtilTest, ComparesUnequalComplex64TuplesUnequal) {
  Literal literal0 = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex64>({42.0, 64.0}),
      LiteralUtil::CreateR0<complex64>({64.0, 42.0}),
  });
  Literal literal1 = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex64>({64.0, 42.0}),
      LiteralUtil::CreateR0<complex64>({42.0, 64.0}),
  });
  Literal literal2 = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex64>({42.42, 64.0}),
      LiteralUtil::CreateR0<complex64>({64.0, 42.0}),
  });
  Literal literal3 = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex64>({42.0, 64.0}),
      LiteralUtil::CreateR0<complex64>({64.0, 42.42}),
  });
  EXPECT_FALSE(LiteralTestUtil::Equal(literal0, literal1));
  EXPECT_FALSE(LiteralTestUtil::Equal(literal0, literal2));
  EXPECT_FALSE(LiteralTestUtil::Equal(literal0, literal3));
  EXPECT_FALSE(LiteralTestUtil::Equal(literal2, literal3));
}

TEST(LiteralTestUtilTest, ComparesUnequalComplex128TuplesUnequal) {
  Literal literal0 = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex128>({42.0, 64.0}),
      LiteralUtil::CreateR0<complex128>({64.0, 42.0}),
  });
  Literal literal1 = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex128>({64.0, 42.0}),
      LiteralUtil::CreateR0<complex128>({42.0, 64.0}),
  });
  Literal literal2 = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex128>({42.42, 64.0}),
      LiteralUtil::CreateR0<complex128>({64.0, 42.0}),
  });
  Literal literal3 = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex128>({42.0, 64.0}),
      LiteralUtil::CreateR0<complex128>({64.0, 42.42}),
  });
  EXPECT_FALSE(LiteralTestUtil::Equal(literal0, literal1));
  EXPECT_FALSE(LiteralTestUtil::Equal(literal0, literal2));
  EXPECT_FALSE(LiteralTestUtil::Equal(literal0, literal3));
  EXPECT_FALSE(LiteralTestUtil::Equal(literal2, literal3));
}

TEST(LiteralTestUtilTest, ComparesUnequalTuplesUnequal) {
  // Implementation note: we have to use a death test here, because you can't
  // un-fail an assertion failure. The CHECK-failure is death, so we can make a
  // death assertion.
  auto unequal_things_are_equal = [] {
    Literal lhs = LiteralUtil::MakeTupleFromSlices({
        LiteralUtil::CreateR0<int32_t>(42),
        LiteralUtil::CreateR0<int32_t>(64),
    });
    Literal rhs = LiteralUtil::MakeTupleFromSlices({
        LiteralUtil::CreateR0<int32_t>(64),
        LiteralUtil::CreateR0<int32_t>(42),
    });
    CHECK(LiteralTestUtil::Equal(lhs, rhs)) << "LHS and RHS are unequal";
  };
  ASSERT_DEATH(unequal_things_are_equal(), "LHS and RHS are unequal");
}

TEST(LiteralTestUtilTest, ExpectNearFailurePlacesResultsInTemporaryDirectory) {
  auto dummy_lambda = [] {
    auto two = LiteralUtil::CreateR0<float>(2);
    auto four = LiteralUtil::CreateR0<float>(4);
    ErrorSpec error(0.001);
    CHECK(LiteralTestUtil::Near(two, four, error)) << "two is not near four";
  };

  tsl::Env* env = tsl::Env::Default();

  std::string outdir;
  if (!tsl::io::GetTestUndeclaredOutputsDir(&outdir)) {
    outdir = tsl::testing::TmpDir();
  }
  std::string pattern = tsl::io::JoinPath(outdir, "tempfile-*.pb");
  std::vector<std::string> files;
  CHECK_OK(env->GetMatchingPaths(pattern, &files));
  for (const auto& f : files) {
    CHECK_OK(env->DeleteFile(f)) << f;
  }

  ASSERT_DEATH(dummy_lambda(), "two is not near four");

  // Now check we wrote temporary files to the temporary directory that we can
  // read.
  std::vector<std::string> results;
  CHECK_OK(env->GetMatchingPaths(pattern, &results));

  LOG(INFO) << "results: [" << absl::StrJoin(results, ", ") << "]";
  EXPECT_EQ(3, results.size());
  for (const std::string& result : results) {
    LiteralProto literal_proto;
    CHECK_OK(tsl::ReadBinaryProto(tsl::Env::Default(), result, &literal_proto));
    Literal literal = Literal::CreateFromProto(literal_proto).value();
    if (result.find("expected") != std::string::npos) {
      EXPECT_EQ("f32[] 2", literal.ToString());
    } else if (result.find("actual") != std::string::npos) {
      EXPECT_EQ("f32[] 4", literal.ToString());
    } else if (result.find("mismatches") != std::string::npos) {
      EXPECT_EQ("pred[] true", literal.ToString());
    } else {
      FAIL() << "unknown file in temporary directory: " << result;
    }
  }
}

TEST(LiteralTestUtilTest, NotEqualHasValuesInMessage) {
  auto expected = LiteralUtil::CreateR1<int32_t>({1, 2, 3});
  auto actual = LiteralUtil::CreateR1<int32_t>({4, 5, 6});
  ::testing::AssertionResult result = LiteralTestUtil::Equal(expected, actual);
  EXPECT_THAT(result.message(),
              ::testing::HasSubstr("Expected literal:\ns32[3] {1, 2, 3}"));
  EXPECT_THAT(result.message(),
              ::testing::HasSubstr("Actual literal:\ns32[3] {4, 5, 6}"));
}

TEST(LiteralTestUtilTest, NearComparatorR1) {
  auto a = LiteralUtil::CreateR1<float>(
      {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});
  auto b = LiteralUtil::CreateR1<float>(
      {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});
  EXPECT_TRUE(LiteralTestUtil::Near(a, b, ErrorSpec{0.0001}));
}

template <typename T>
class LiteralTestUtilComplexTest : public ::testing::Test {};

using ComplexTypes =
    ::testing::Types<complex32, bcomplex32, complex64, complex128>;
TYPED_TEST_SUITE(LiteralTestUtilComplexTest, ComplexTypes);

TYPED_TEST(LiteralTestUtilComplexTest, NearComparatorR1) {
  const Literal expected = LiteralUtil::CreateR1<TypeParam>(
      {TypeParam(0.0f, 1.0f), TypeParam(1.0f, 2.0f)});
  EXPECT_TRUE(
      LiteralTestUtil::Near(expected, expected.Clone(), ErrorSpec(0.0)));

  for (TypeParam value : {TypeParam(1.125f, 2.0f), TypeParam(1.0f, 2.125f),
                          TypeParam(1.125f, 2.125f)}) {
    Literal actual = expected.Clone();
    actual.Set<TypeParam>({1}, value);
    EXPECT_THAT(literal_comparison::Near(expected, actual, ErrorSpec(0.125),
                                         /*detailed_message=*/false,
                                         /*miscompare_callback=*/nullptr),
                absl_testing::IsOk());
    EXPECT_THAT(
        literal_comparison::Near(expected, actual, ErrorSpec(0.0625),
                                 /*detailed_message=*/true,
                                 /*miscompare_callback=*/nullptr),
        absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                               ::testing::HasSubstr("Mismatch count 1 ")));
  }

  Literal real_changed = expected.Clone();
  real_changed.Set<TypeParam>({1}, TypeParam(1.125f, 2.0f));
  Literal imag_changed = expected.Clone();
  imag_changed.Set<TypeParam>({1}, TypeParam(1.0f, 2.125f));
  EXPECT_FALSE(
      LiteralTestUtil::Near(real_changed, imag_changed, ErrorSpec(0.0001)));
}

TYPED_TEST(LiteralTestUtilComplexTest, NearComparatorRelativeTolerance) {
  const Literal expected = LiteralUtil::CreateR0(TypeParam(8.0f, 16.0f));
  const Literal actual = LiteralUtil::CreateR0(TypeParam(9.0f, 18.0f));
  EXPECT_THAT(literal_comparison::Near(expected, actual, ErrorSpec(0.0, 0.125),
                                       /*detailed_message=*/false,
                                       /*miscompare_callback=*/nullptr),
              absl_testing::IsOk());
  EXPECT_THAT(
      literal_comparison::Near(expected, actual, ErrorSpec(0.0, 0.0625),
                               /*detailed_message=*/false,
                               /*miscompare_callback=*/nullptr),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("Mismatch count 1 ")));
}

TYPED_TEST(LiteralTestUtilComplexTest, NearComparatorFloatDistance) {
  using ComponentT = typename TypeParam::value_type;
  const ComponentT one = ComponentT(1.0f);
  const ComponentT next =
      static_cast<ComponentT>(one + std::numeric_limits<ComponentT>::epsilon());
  const Literal expected =
      LiteralUtil::CreateR0(TypeParam(one, ComponentT(2.0f)));
  const Literal actual =
      LiteralUtil::CreateR0(TypeParam(next, ComponentT(2.0f)));
  ErrorSpec error(0.0, 0.0);
  error.low_precision_fp_error_spec.type =
      primitive_util::NativeToPrimitiveType<ComponentT>();
  error.low_precision_fp_error_spec.within_n_values = 0;
  EXPECT_THAT(
      literal_comparison::Near(expected, actual, error,
                               /*detailed_message=*/false,
                               /*miscompare_callback=*/nullptr),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("Mismatch count 1 ")));
  error.low_precision_fp_error_spec.within_n_values = 1;
  EXPECT_THAT(literal_comparison::Near(expected, actual, error,
                                       /*detailed_message=*/false,
                                       /*miscompare_callback=*/nullptr),
              absl_testing::IsOk());
}

TYPED_TEST(LiteralTestUtilComplexTest, NearComparatorNaNs) {
  using ComponentT = typename TypeParam::value_type;
  const ComponentT nan = std::numeric_limits<ComponentT>::quiet_NaN();
  const ComponentT other_nan = -nan;
  const Literal expected =
      LiteralUtil::CreateR0(TypeParam(nan, ComponentT(2.0f)));
  const Literal actual =
      LiteralUtil::CreateR0(TypeParam(other_nan, ComponentT(2.0f)));
  ErrorSpec error(0.0, 0.0);
  EXPECT_THAT(literal_comparison::Near(expected, actual, error,
                                       /*detailed_message=*/false,
                                       /*miscompare_callback=*/nullptr),
              absl_testing::IsOk());
  EXPECT_THAT(
      literal_comparison::Equal(expected, actual),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("not bitwise-equal")));

  const Literal finite = LiteralUtil::CreateR0(TypeParam(1.0f, 2.0f));
  EXPECT_THAT(
      literal_comparison::Near(expected, finite, error,
                               /*detailed_message=*/false,
                               /*miscompare_callback=*/nullptr),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("Mismatch count 1 ")));
  error.relaxed_nans = true;
  EXPECT_THAT(literal_comparison::Near(expected, finite, error,
                                       /*detailed_message=*/false,
                                       /*miscompare_callback=*/nullptr),
              absl_testing::IsOk());
  EXPECT_THAT(
      literal_comparison::Near(finite, expected, error,
                               /*detailed_message=*/false,
                               /*miscompare_callback=*/nullptr),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("Mismatch count 1 ")));
}

TYPED_TEST(LiteralTestUtilComplexTest, NearComparatorInfinities) {
  using ComponentT = typename TypeParam::value_type;
  const ComponentT inf = std::numeric_limits<ComponentT>::infinity();
  const ComponentT max = std::numeric_limits<ComponentT>::max();
  const Literal infinite =
      LiteralUtil::CreateR0(TypeParam(inf, ComponentT(2.0f)));
  const Literal finite =
      LiteralUtil::CreateR0(TypeParam(max, ComponentT(2.0f)));
  const Literal opposite =
      LiteralUtil::CreateR0(TypeParam(-inf, ComponentT(2.0f)));
  ErrorSpec error(0.0, 0.0);
  EXPECT_THAT(literal_comparison::Near(infinite, infinite, error,
                                       /*detailed_message=*/false,
                                       /*miscompare_callback=*/nullptr),
              absl_testing::IsOk());
  EXPECT_THAT(
      literal_comparison::Near(infinite, opposite, error,
                               /*detailed_message=*/false,
                               /*miscompare_callback=*/nullptr),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("Mismatch count 1 ")));
  EXPECT_THAT(
      literal_comparison::Near(finite, infinite, error,
                               /*detailed_message=*/false,
                               /*miscompare_callback=*/nullptr),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("Mismatch count 1 ")));
  error.fewer_infs_ok = true;
  EXPECT_THAT(literal_comparison::Near(finite, infinite, error,
                                       /*detailed_message=*/false,
                                       /*miscompare_callback=*/nullptr),
              absl_testing::IsOk());
  EXPECT_THAT(
      literal_comparison::Near(infinite, finite, error,
                               /*detailed_message=*/false,
                               /*miscompare_callback=*/nullptr),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("Mismatch count 1 ")));
}

TYPED_TEST(LiteralTestUtilComplexTest, NearComparatorSignedZeros) {
  const Literal positive = LiteralUtil::CreateR0(TypeParam(0.0f, 0.0f));
  const Literal negative = LiteralUtil::CreateR0(TypeParam(-0.0f, -0.0f));
  EXPECT_THAT(literal_comparison::Near(positive, negative, ErrorSpec(0.0, 0.0),
                                       /*detailed_message=*/false,
                                       /*miscompare_callback=*/nullptr),
              absl_testing::IsOk());
  EXPECT_THAT(
      literal_comparison::Equal(positive, negative),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("not bitwise-equal")));
}

TYPED_TEST(LiteralTestUtilComplexTest, NearComparatorDifferentLayouts) {
  const Literal expected = LiteralUtil::CreateR2<TypeParam>(
      {{TypeParam(1.0f, 2.0f), TypeParam(3.0f, 4.0f)},
       {TypeParam(5.0f, 6.0f), TypeParam(7.0f, 8.0f)}});
  Literal actual = expected.Relayout(LayoutUtil::MakeLayout({0, 1}));
  actual.Set<TypeParam>({1, 0}, TypeParam(5.125f, 6.0f));
  EXPECT_THAT(literal_comparison::Near(expected, actual, ErrorSpec(0.125),
                                       /*detailed_message=*/false,
                                       /*miscompare_callback=*/nullptr),
              absl_testing::IsOk());
  EXPECT_THAT(
      literal_comparison::Near(expected, actual, ErrorSpec(0.0625),
                               /*detailed_message=*/false,
                               /*miscompare_callback=*/nullptr),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("Mismatch count 1 ")));
}

TEST(LiteralTestUtilTest, NearComparatorR1Nan) {
  auto a = LiteralUtil::CreateR1<float>(
      {0.0, 0.1, 0.2, 0.3, NAN, 0.5, 0.6, 0.7, 0.8});
  auto b = LiteralUtil::CreateR1<float>(
      {0.0, 0.1, 0.2, 0.3, NAN, 0.5, 0.6, 0.7, 0.8});
  EXPECT_TRUE(LiteralTestUtil::Near(a, b, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtil, NearComparatorDifferentLengths) {
  auto a = LiteralUtil::CreateR1<float>(
      {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});
  auto b =
      LiteralUtil::CreateR1<float>({0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7});
  EXPECT_FALSE(LiteralTestUtil::Near(a, b, ErrorSpec{0.0001}));
  EXPECT_FALSE(LiteralTestUtil::Near(b, a, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, ExpectNearDoubleOutsideFloatValueRange) {
  auto two_times_float_max =
      LiteralUtil::CreateR0<double>(2.0 * std::numeric_limits<float>::max());
  ErrorSpec error(0.001);
  EXPECT_TRUE(
      LiteralTestUtil::Near(two_times_float_max, two_times_float_max, error));
}

TEST(LiteralTestUtilTest, DynamicEqualityR1) {
  auto literal1 = Literal(ShapeUtil::MakeShape(U32, {10}));
  literal1.PopulateR1<uint32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal1.SetDynamicSize(0, 5);
  auto literal2 = Literal(ShapeUtil::MakeShape(U32, {10}));
  literal2.PopulateR1<uint32_t>({1, 2, 3, 4, 5, 99, 99, 99, 99, 99});
  literal2.SetDynamicSize(0, 5);
  EXPECT_TRUE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, DynamicEqualityR2Dim) {
  auto literal1 = Literal(ShapeUtil::MakeShape(U32, {3, 3}));
  literal1.PopulateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(0, 2);
  auto literal2 = Literal(ShapeUtil::MakeShape(U32, {3, 3}));
  literal2.PopulateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {99, 99, 99}});
  literal2.SetDynamicSize(0, 2);
  EXPECT_TRUE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, DynamicEqualityR2Dim1) {
  auto literal1 = Literal(ShapeUtil::MakeShape(U32, {3, 3}));
  literal1.PopulateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(1, 2);
  auto literal2 = Literal(ShapeUtil::MakeShape(U32, {3, 3}));
  literal2.PopulateR2<uint32_t>({{1, 2, 99}, {4, 5, 99}, {7, 8, 99}});
  literal2.SetDynamicSize(1, 2);
  EXPECT_TRUE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, DynamicNearEqualityR1) {
  auto literal1 = Literal(ShapeUtil::MakeShape(F32, {10}));
  literal1.PopulateR1<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal1.SetDynamicSize(0, 5);
  auto literal2 = Literal(ShapeUtil::MakeShape(F32, {10}));
  literal2.PopulateR1<float>({1, 2, 3, 4, 5, 99, 99, 99, 99, 99});
  literal2.SetDynamicSize(0, 5);
  ErrorSpec error(0.001);
  EXPECT_TRUE(LiteralTestUtil::Near(literal1, literal2, error));
}

TEST(LiteralTestUtilTest, DynamicNearEqualityR2Dim) {
  auto literal1 = Literal(ShapeUtil::MakeShape(F32, {3, 3}));
  literal1.PopulateR2<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(0, 2);
  auto literal2 = Literal(ShapeUtil::MakeShape(F32, {3, 3}));
  literal2.PopulateR2<float>({{1, 2, 3}, {4, 5, 6}, {99, 99, 99}});
  literal2.SetDynamicSize(0, 2);
  ErrorSpec error(0.001);
  EXPECT_TRUE(LiteralTestUtil::Near(literal1, literal2, error));
}

TEST(LiteralTestUtilTest, DynamicNearEqualityR2Dim1) {
  auto literal1 = Literal(ShapeUtil::MakeShape(F32, {3, 3}));
  literal1.PopulateR2<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(1, 2);
  auto literal2 = Literal(ShapeUtil::MakeShape(F32, {3, 3}));
  literal2.PopulateR2<float>({{1, 2, 99}, {4, 5, 99}, {7, 8, 99}});
  literal2.SetDynamicSize(1, 2);
  ErrorSpec error(0.001);
  EXPECT_TRUE(LiteralTestUtil::Near(literal1, literal2, error));
}

TEST(LiteralTestUtilTest, UnequalDynamicDimensionsR1) {
  auto literal1 = Literal(ShapeUtil::MakeShape(U32, {10}));
  literal1.PopulateR1<uint32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal1.SetDynamicSize(0, 5);
  auto literal2 = Literal(ShapeUtil::MakeShape(U32, {10}));
  literal2.PopulateR1<uint32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal2.SetDynamicSize(0, 6);
  // Dynamic sizes do not match.
  EXPECT_FALSE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, UnequalDynamicDimensionsR1_F32) {
  auto literal1 = Literal(ShapeUtil::MakeShape(F32, {10}));
  literal1.PopulateR1<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal1.SetDynamicSize(0, 5);
  auto literal2 = Literal(ShapeUtil::MakeShape(F32, {10}));
  literal2.PopulateR1<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal2.SetDynamicSize(0, 6);
  // Dynamic sizes do not match.
  EXPECT_FALSE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, ExpectedIsDynamicActualIsNotR1) {
  auto literal1 = Literal(ShapeUtil::MakeShape(U32, {10}));
  literal1.PopulateR1<uint32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal1.SetDynamicSize(0, 5);
  auto literal2 = Literal(ShapeUtil::MakeShape(U32, {10}));
  literal2.PopulateR1<uint32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  // Only literal1 is dynamic.
  EXPECT_FALSE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, ExpectedIsDynamicActualIsNotR1_F32) {
  auto literal1 = Literal(ShapeUtil::MakeShape(F32, {10}));
  literal1.PopulateR1<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal1.SetDynamicSize(0, 5);
  auto literal2 = Literal(ShapeUtil::MakeShape(F32, {10}));
  literal2.PopulateR1<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  // Only literal1 is dynamic.
  EXPECT_FALSE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, ActualIsDynamicExpectedIsNotR1) {
  auto literal1 = Literal(ShapeUtil::MakeShape(U32, {10}));
  literal1.PopulateR1<uint32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  auto literal2 = Literal(ShapeUtil::MakeShape(U32, {10}));
  literal2.PopulateR1<uint32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal2.SetDynamicSize(0, 5);
  // Only literal2 is dynamic.
  EXPECT_FALSE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, ActualIsDynamicExpectedIsNotR1_F32) {
  auto literal1 = Literal(ShapeUtil::MakeShape(F32, {10}));
  literal1.PopulateR1<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  auto literal2 = Literal(ShapeUtil::MakeShape(F32, {10}));
  literal2.PopulateR1<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal2.SetDynamicSize(0, 5);
  // Only literal2 is dynamic.
  EXPECT_FALSE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, UnequalDynamicDimensionsR2Dim0) {
  auto literal1 = Literal(ShapeUtil::MakeShape(U32, {3, 3}));
  literal1.PopulateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(0, 2);
  auto literal2 = Literal(ShapeUtil::MakeShape(U32, {3, 3}));
  literal2.PopulateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal2.SetDynamicSize(0, 3);
  // Dynamic sizes do not match.
  EXPECT_FALSE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, UnequalDynamicDimensionsR2Dim0_F32) {
  auto literal1 = Literal(ShapeUtil::MakeShape(F32, {3, 3}));
  literal1.PopulateR2<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(0, 2);
  auto literal2 = Literal(ShapeUtil::MakeShape(F32, {3, 3}));
  literal2.PopulateR2<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal2.SetDynamicSize(0, 3);
  // Dynamic sizes do not match.
  EXPECT_FALSE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, UnequalDynamicDimensionsR2Dim1) {
  auto literal1 = Literal(ShapeUtil::MakeShape(U32, {3, 3}));
  literal1.PopulateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(1, 2);
  auto literal2 = Literal(ShapeUtil::MakeShape(U32, {3, 3}));
  literal2.PopulateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal2.SetDynamicSize(1, 3);
  // Dynamic sizes do not match.
  EXPECT_FALSE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, UnequalDynamicDimensionsR2Dim1_F32) {
  auto literal1 = Literal(ShapeUtil::MakeShape(F32, {3, 3}));
  literal1.PopulateR2<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(1, 2);
  auto literal2 = Literal(ShapeUtil::MakeShape(F32, {3, 3}));
  literal2.PopulateR2<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal2.SetDynamicSize(1, 3);
  // Dynamic sizes do not match.
  EXPECT_FALSE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, UnequalDynamicDimensionsR2DifferentDimensions) {
  auto literal1 = Literal(ShapeUtil::MakeShape(U32, {3, 3}));
  literal1.PopulateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(1, 2);
  auto literal2 = Literal(ShapeUtil::MakeShape(U32, {3, 3}));
  literal2.PopulateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal2.SetDynamicSize(0, 2);
  // Different dimensions were set as dynamic.
  EXPECT_FALSE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, UnequalDynamicDimensionsR2DifferentDimensions_F32) {
  auto literal1 = Literal(ShapeUtil::MakeShape(F32, {3, 3}));
  literal1.PopulateR2<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(1, 2);
  auto literal2 = Literal(ShapeUtil::MakeShape(F32, {3, 3}));
  literal2.PopulateR2<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal2.SetDynamicSize(0, 2);
  // Different dimensions were set as dynamic.
  EXPECT_FALSE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, DynamicTuplesAreEqual) {
  auto literal1 = Literal(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5}), ShapeUtil::MakeShape(U32, {5})}));
  auto literal2 = Literal(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5}), ShapeUtil::MakeShape(U32, {5})}));
  MutableBorrowingLiteral(&literal1, /*view_root=*/{0})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal1, /*view_root=*/{1})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  literal1.SetDynamicSize(0, {0}, 5);
  MutableBorrowingLiteral(&literal2, /*view_root=*/{0})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal2, /*view_root=*/{1})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  literal2.SetDynamicSize(0, {0}, 5);
  EXPECT_TRUE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, DynamicTuplesAreNear) {
  auto literal1 = Literal(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {5})}));
  auto literal2 = Literal(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {5})}));
  MutableBorrowingLiteral(&literal1, /*view_root=*/{0})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal1, /*view_root=*/{1})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  literal1.SetDynamicSize(0, {0}, 5);
  MutableBorrowingLiteral(&literal2, /*view_root=*/{0})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal2, /*view_root=*/{1})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  literal2.SetDynamicSize(0, {0}, 5);
  EXPECT_TRUE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, DynamicTuplesAreEqualWithinDynamicBounds) {
  auto literal1 = Literal(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5}), ShapeUtil::MakeShape(U32, {5})}));
  auto literal2 = Literal(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5}), ShapeUtil::MakeShape(U32, {5})}));
  MutableBorrowingLiteral(&literal1, /*view_root=*/{0})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal1, /*view_root=*/{1})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  literal1.SetDynamicSize(0, {0}, 3);
  MutableBorrowingLiteral(&literal2, /*view_root=*/{0})
      .PopulateR1<uint32_t>({1, 2, 3, 99, 99});
  MutableBorrowingLiteral(&literal2, /*view_root=*/{1})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  literal2.SetDynamicSize(0, {0}, 3);
  EXPECT_TRUE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, DynamicTuplesAreNearWithinDynamicBounds) {
  auto literal1 = Literal(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {5})}));
  auto literal2 = Literal(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {5})}));
  MutableBorrowingLiteral(&literal1, /*view_root=*/{0})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal1, /*view_root=*/{1})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  literal1.SetDynamicSize(0, {0}, 3);
  MutableBorrowingLiteral(&literal2, /*view_root=*/{0})
      .PopulateR1<float>({1, 2, 3, 99, 99});
  MutableBorrowingLiteral(&literal2, /*view_root=*/{1})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  literal2.SetDynamicSize(0, {0}, 3);
  EXPECT_TRUE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, DynamicTuplesHaveDifferentDynamicSizes) {
  auto literal1 = Literal(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5}), ShapeUtil::MakeShape(U32, {5})}));
  auto literal2 = Literal(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5}), ShapeUtil::MakeShape(U32, {5})}));
  MutableBorrowingLiteral(&literal1, /*view_root=*/{0})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal1, /*view_root=*/{1})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  literal1.SetDynamicSize(0, {0}, 5);
  MutableBorrowingLiteral(&literal2, /*view_root=*/{0})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal2, /*view_root=*/{1})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  literal2.SetDynamicSize(0, {0}, 4);
  // Dynamic sizes are not equal.
  EXPECT_FALSE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, DynamicTuplesHaveDifferentDynamicSizes_F32) {
  auto literal1 = Literal(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {5})}));
  auto literal2 = Literal(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {5})}));
  MutableBorrowingLiteral(&literal1, /*view_root=*/{0})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal1, /*view_root=*/{1})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  literal1.SetDynamicSize(0, {0}, 5);
  MutableBorrowingLiteral(&literal2, /*view_root=*/{0})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal2, /*view_root=*/{1})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  literal2.SetDynamicSize(0, {0}, 4);
  // Dynamic sizes are not equal.
  EXPECT_FALSE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, OneTupleDynamicOneIsNot) {
  auto literal1 = Literal(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5}), ShapeUtil::MakeShape(U32, {5})}));
  auto literal2 = Literal(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5}), ShapeUtil::MakeShape(U32, {5})}));
  MutableBorrowingLiteral(&literal1, /*view_root=*/{0})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal1, /*view_root=*/{1})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  literal1.SetDynamicSize(0, {0}, 5);
  MutableBorrowingLiteral(&literal2, /*view_root=*/{0})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal2, /*view_root=*/{1})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  // Only one of the tuples is dynamic.
  EXPECT_FALSE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, OneTupleDynamicOneIsNot_F32) {
  auto literal1 = Literal(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {5})}));
  auto literal2 = Literal(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {5})}));
  MutableBorrowingLiteral(&literal1, /*view_root=*/{0})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal1, /*view_root=*/{1})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  literal1.SetDynamicSize(0, {0}, 5);
  MutableBorrowingLiteral(&literal2, /*view_root=*/{0})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal2, /*view_root=*/{1})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  // Only one of the tuples is dynamic.
  EXPECT_FALSE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}
}  // namespace
}  // namespace xla
