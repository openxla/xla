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

#include "xla/codegen/intrinsic/cpp/ynnpack_unary.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <string>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"
#include "xla/codegen/intrinsic/cpp/vector_ops.h"
#include "xla/codegen/intrinsic/cpp/ynnpack_unary_ll.h"
#include "xla/codegen/intrinsic/test_matchers.h"

namespace xla::codegen {
namespace {
using ::testing::HasSubstr;
using ::testing::Not;
using ::xla::codegen::intrinsic::NearUlps;

// The float32 result rounded from the double-precision libm result.
float RefExpF32(float x) {
  return static_cast<float>(std::exp(static_cast<double>(x)));
}

TEST(YnnpackUnaryTest, BitcodeIsTargetIndependentAndVectorized) {
  if (llvm_ir::kYnnpackUnaryLlIr.empty()) {
    GTEST_SKIP();
  }
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kYnnpackUnaryLlIr);
  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);

  EXPECT_THAT(ir, HasSubstr("xla.exp.v16f32"));
  EXPECT_THAT(ir, HasSubstr("xla.exp.v8f64"));
  EXPECT_THAT(ir, HasSubstr("xla.log.v16f32"));
  EXPECT_THAT(ir, HasSubstr("xla.log.v8f64"));
  EXPECT_THAT(ir, HasSubstr("xla.log1p.v8f64"));
  EXPECT_THAT(ir, Not(HasSubstr("llvm.x86")));
  EXPECT_THAT(ir, Not(HasSubstr("llvm.aarch64")));
}

TEST(YnnpackUnaryTest, ExpF32SpecialValues) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  EXPECT_EQ(xla_exp_f32(0.0f), 1.0f);
  EXPECT_EQ(xla_exp_f32(-0.0f), 1.0f);
  EXPECT_EQ(xla_exp_f32(kInf), kInf);
  EXPECT_EQ(xla_exp_f32(-kInf), 0.0f);
  EXPECT_TRUE(std::isnan(xla_exp_f32(std::numeric_limits<float>::quiet_NaN())));
  // ln(FLT_MAX) is between these two.
  EXPECT_THAT(xla_exp_f32(0x1.62e42ep+6f),
              NearUlps<float>(RefExpF32(0x1.62e42ep+6f), 1));
  EXPECT_EQ(xla_exp_f32(0x1.62e430p+6f), kInf);
  EXPECT_EQ(xla_exp_f32(1000.0f), kInf);
  // Below ln(FLT_TRUE_MIN).
  EXPECT_EQ(xla_exp_f32(-104.0f), 0.0f);
  EXPECT_EQ(xla_exp_f32(-1000.0f), 0.0f);
}

TEST(YnnpackUnaryTest, ExpF32IsWithinOneUlp) {
  // A grid over the inputs with a normal result: -87.3 > ln(FLT_MIN).
  for (int i = -8730; i <= 8872; ++i) {
    float x = static_cast<float>(i) * 0.01f;
    EXPECT_THAT(xla_exp_f32(x), NearUlps<float>(RefExpF32(x), 1))
        << "x = " << x;
  }
}

TEST(YnnpackUnaryTest, ExpF64SpecialValues) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(xla_exp_f64(0.0), 1.0);
  EXPECT_EQ(xla_exp_f64(-0.0), 1.0);
  EXPECT_EQ(xla_exp_f64(kInf), kInf);
  EXPECT_EQ(xla_exp_f64(-kInf), 0.0);
  EXPECT_TRUE(
      std::isnan(xla_exp_f64(std::numeric_limits<double>::quiet_NaN())));
  // ln(DBL_MAX) is between these two.
  EXPECT_THAT(xla_exp_f64(0x1.62e42fefa39efp+9),
              NearUlps<double>(std::exp(0x1.62e42fefa39efp+9), 1));
  EXPECT_EQ(xla_exp_f64(0x1.62e42fefa39f0p+9), kInf);
  EXPECT_EQ(xla_exp_f64(1.0e4), kInf);
  // Below ln(DBL_TRUE_MIN).
  EXPECT_EQ(xla_exp_f64(-746.0), 0.0);
  EXPECT_EQ(xla_exp_f64(-1.0e4), 0.0);
}

TEST(YnnpackUnaryTest, ExpF64IsWithinOneUlp) {
  for (int i = -10000; i <= 10000; ++i) {
    double x = static_cast<double>(i) * 0.07;
    EXPECT_THAT(xla_exp_f64(x), NearUlps<double>(std::exp(x), 1))
        << "x = " << x;
  }
}

TEST(YnnpackUnaryTest, ExpVectorsMatchScalar) {
  const Vec16f x32 = {-100.0f, -87.0f, -10.5f, -1.0f,   -0.25f, -0.0f,
                      0.0f,    1e-6f,  0.3f,   0.6931f, 1.0f,   5.2f,
                      17.0f,   59.3f,  88.7f,  89.0f};
  const Vec16f y16 = xla_exp_v16f32(x32);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(y16[i], xla_exp_f32(x32[i])) << "lane " << i;
  }
  const Vec8f y8 =
      xla_exp_v8f32(__builtin_shufflevector(x32, x32, 0, 1, 2, 3, 4, 5, 6, 7));
  const Vec4f y4 =
      xla_exp_v4f32(__builtin_shufflevector(x32, x32, 8, 9, 10, 11));
  const Vec2f y2 = xla_exp_v2f32(__builtin_shufflevector(x32, x32, 12, 13));
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(y8[i], y16[i]) << "lane " << i;
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(y4[i], y16[8 + i]) << "lane " << i;
  }
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(y2[i], y16[12 + i]) << "lane " << i;
  }

  const Vec8d x64 = {-745.0, -700.0, -1.0, -0.0, 0.3, 37.0, 709.7, 710.0};
  const Vec8d z8 = xla_exp_v8f64(x64);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(z8[i], xla_exp_f64(x64[i])) << "lane " << i;
  }
  const Vec4d z4 = xla_exp_v4f64(__builtin_shufflevector(x64, x64, 0, 1, 2, 3));
  const Vec2d z2 = xla_exp_v2f64(__builtin_shufflevector(x64, x64, 4, 5));
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(z4[i], z8[i]) << "lane " << i;
  }
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(z2[i], z8[4 + i]) << "lane " << i;
  }
}

}  // namespace
}  // namespace xla::codegen
