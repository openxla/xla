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

#include "xla/codegen/intrinsic/cpp/log.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <limits>
#include <memory>
#include <string>

#include "absl/base/casts.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"
#include "xla/codegen/intrinsic/cpp/log_ll.h"
#include "xla/codegen/intrinsic/cpp/vector_ops.h"
#include "xla/codegen/intrinsic/test_matchers.h"

namespace xla::codegen {
namespace {

using ::testing::AllOf;
using ::testing::ContainsRegex;
using ::testing::Not;
using ::xla::codegen::intrinsic::NearUlps;

constexpr int kMaxUlps = 1;

TEST(LogTest, BitcodeIsVectorizedAndPortable) {
  ASSERT_TRUE(AreCppGenIntrinsicsAvailable());
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kLogLlIr);
  ASSERT_NE(module, nullptr);
  EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);

  EXPECT_THAT(
      ir, AllOf(ContainsRegex("define .*@xla\\.log\\.f64\\(double"),
                ContainsRegex("define .*@xla\\.log\\.v2f64\\(.*<2 x double>"),
                ContainsRegex("define .*@xla\\.log\\.v4f64\\(.*<4 x double>"),
                ContainsRegex("define .*@xla\\.log\\.v8f64\\(.*<8 x double>"),
                ContainsRegex("define .*@xla\\.log2\\.f64\\(double"),
                ContainsRegex("define .*@xla\\.log2\\.v2f64\\(.*<2 x double>"),
                ContainsRegex("define .*@xla\\.log2\\.v4f64\\(.*<4 x double>"),
                ContainsRegex("define .*@xla\\.log2\\.v8f64\\(.*<8 x double>"),
                ContainsRegex("fdiv <4 x double>"),
                ContainsRegex("llvm\\.fmuladd\\.v4f64"),
                Not(ContainsRegex("llvm\\.fma\\.")),
                Not(ContainsRegex("llvm\\.x86")),
                Not(ContainsRegex("llvm\\.aarch64"))));
}

TEST(LogTest, SpecialCasesAndFtzSubnormals) {
  using Limits = std::numeric_limits<double>;
  const double max_subnormal = absl::bit_cast<double>(0x000fffffffffffffULL);
  const double neg_max_subnormal =
      absl::bit_cast<double>(0x800fffffffffffffULL);

  for (double z : {0.0, -0.0, Limits::denorm_min(), -Limits::denorm_min(),
                   max_subnormal, neg_max_subnormal}) {
    EXPECT_EQ(log_f64(z), -Limits::infinity()) << "z = " << z;
    EXPECT_EQ(log2_f64(z), -Limits::infinity()) << "z = " << z;
  }

  for (double neg : {-Limits::min(), -1.0, -100.0, -Limits::infinity(),
                     Limits::quiet_NaN(), Limits::signaling_NaN()}) {
    EXPECT_TRUE(std::isnan(log_f64(neg))) << "neg = " << neg;
    EXPECT_TRUE(std::isnan(log2_f64(neg))) << "neg = " << neg;
  }

  EXPECT_EQ(log_f64(Limits::infinity()), Limits::infinity());
  EXPECT_EQ(log2_f64(Limits::infinity()), Limits::infinity());
  EXPECT_EQ(log_f64(1.0), 0.0);
  EXPECT_EQ(log2_f64(1.0), 0.0);
}

TEST(LogTest, PowersOfTwo) {
  for (int exp = -1022; exp <= 1023; ++exp) {
    double x = std::ldexp(1.0, exp);
    EXPECT_THAT(log_f64(x), NearUlps(std::log(x), kMaxUlps)) << "exp = " << exp;
    EXPECT_EQ(log2_f64(x), static_cast<double>(exp)) << "exp = " << exp;
  }
}

TEST(LogTest, NearOneAndRangeBoundaries) {
  const double sqrt2 = std::sqrt(2.0);
  const double half_sqrt2 = 0.5 * sqrt2;
  const double anchors[] = {
      1.0, sqrt2, half_sqrt2,           std::exp(1.0),
      0.5, 2.0,   0x1.6a09e667f345fp+0, 0x1.6a09e667f37afp+0};

  for (double anchor : anchors) {
    double cur = anchor;
    for (int i = 0; i < 2000; ++i) {
      EXPECT_THAT(log_f64(cur), NearUlps(std::log(cur), kMaxUlps))
          << "cur = " << cur;
      EXPECT_THAT(log2_f64(cur), NearUlps(std::log2(cur), kMaxUlps))
          << "cur = " << cur;
      cur = std::nextafter(cur, std::numeric_limits<double>::infinity());
    }
    cur = anchor;
    for (int i = 0; i < 2000; ++i) {
      EXPECT_THAT(log_f64(cur), NearUlps(std::log(cur), kMaxUlps))
          << "cur = " << cur;
      EXPECT_THAT(log2_f64(cur), NearUlps(std::log2(cur), kMaxUlps))
          << "cur = " << cur;
      cur = std::nextafter(cur, 0.0);
    }
  }
}

TEST(LogTest, FullExponentRangeSweepWithinOneUlp) {
  uint64_t state = 0x9e3779b97f4a7c15ULL;
  for (int i = 0; i < 50000; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint64_t exp_bits = (1ULL + (state % 2046ULL)) << 52;
    uint64_t mant_bits = (state >> 12) & 0x000fffffffffffffULL;
    double x = absl::bit_cast<double>(exp_bits | mant_bits);

    EXPECT_THAT(log_f64(x), NearUlps(std::log(x), kMaxUlps))
        << "x = " << x << " (0x" << std::hex << (exp_bits | mant_bits) << ")";
    EXPECT_THAT(log2_f64(x), NearUlps(std::log2(x), kMaxUlps))
        << "x = " << x << " (0x" << std::hex << (exp_bits | mant_bits) << ")";
  }
}

template <typename VecT, typename VecFn, typename ScalarFn>
void ExpectVectorMatchesScalar(VecFn vec_fn, ScalarFn scalar_fn,
                               const double* xs, size_t n) {
  constexpr size_t kLanes = NumLanes<VecT>;
  for (size_t i = 0; i < n; i += kLanes) {
    VecT v = Splat<VecT>(0.0);
    for (size_t lane = 0; lane < kLanes; ++lane) {
      SetLane(v, lane, xs[i + lane]);
    }
    VecT out = vec_fn(v);
    for (size_t lane = 0; lane < kLanes; ++lane) {
      double actual = GetLane(out, lane);
      double expected = scalar_fn(xs[i + lane]);
      if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(actual)) << "x = " << xs[i + lane];
      } else {
        EXPECT_EQ(absl::bit_cast<uint64_t>(actual),
                  absl::bit_cast<uint64_t>(expected))
            << "x = " << xs[i + lane];
      }
    }
  }
}

TEST(LogTest, VectorVariantsMatchScalar) {
  using Limits = std::numeric_limits<double>;
  alignas(64) const double kInputs[16] = {
      0.0,
      -0.0,
      Limits::denorm_min(),
      -Limits::denorm_min(),
      Limits::min(),
      -Limits::min(),
      0.5,
      1.0,
      1.0 + 1e-14,
      1.4142135623730951,
      2.718281828459045,
      1024.0,
      Limits::max(),
      Limits::infinity(),
      -Limits::infinity(),
      Limits::quiet_NaN(),
  };

  ExpectVectorMatchesScalar<Vec2d>(log_v2f64, log_f64, kInputs, 16);
  ExpectVectorMatchesScalar<Vec4d>(log_v4f64, log_f64, kInputs, 16);
  ExpectVectorMatchesScalar<Vec8d>(log_v8f64, log_f64, kInputs, 16);

  ExpectVectorMatchesScalar<Vec2d>(log2_v2f64, log2_f64, kInputs, 16);
  ExpectVectorMatchesScalar<Vec4d>(log2_v4f64, log2_f64, kInputs, 16);
  ExpectVectorMatchesScalar<Vec8d>(log2_v8f64, log2_f64, kInputs, 16);
}

}  // namespace
}  // namespace xla::codegen
