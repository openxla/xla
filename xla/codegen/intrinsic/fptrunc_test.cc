/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/intrinsic/fptrunc.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ios>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "xla/codegen/intrinsic/intrinsic_compiler_lib.h"
#include "xla/codegen/intrinsic/simple_jit_runner.h"
#include "xla/codegen/intrinsic/type.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {
using ::xla::codegen::intrinsic::JitRunner;

TEST(FpTruncTest, SclarIninsic) {
  EXPECT_EQ(FpTrunc::Name({Type::S(F32), Type::S(BF16)}),
            "xla.fptrunc.f32.to.bf16");
}

TEST(FpTruncTest, VectorIninsic) {
  EXPECT_EQ(FpTrunc::Name({Type::V(F32, 4), Type::V(BF16, 4)}),
            "xla.fptrunc.v4f32.to.v4bf16");
}

JitRunner CreateJitRunner(Type from, Type to) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("test_module", *context);

  llvm::Function* func =
      FpTrunc::CreateDefinition(module.get(), from, to).value();
  func->setLinkage(llvm::Function::ExternalLinkage);
  EXPECT_FALSE(llvm::verifyFunction(*func));

  llvm::Function* wrapper = intrinsic::CreateWrapperIntArgToFp(func);
  wrapper->setLinkage(llvm::Function::ExternalLinkage);
  EXPECT_FALSE(llvm::verifyFunction(*wrapper));

  llvm::Function* wrapper2 = intrinsic::CreateWrapperFpRetToInt(func);
  wrapper2->setLinkage(llvm::Function::ExternalLinkage);
  EXPECT_FALSE(llvm::verifyFunction(*wrapper2));

  return JitRunner(std::move(module), std::move(context));
}

TEST(FpTruncExecutionTest, F16ToF8e4m3fn) {
  JitRunner jit = CreateJitRunner(Type::S(F16), Type::S(F8E4M3FN));
  auto fptrunc = jit.GetScalarFn<int8_t(int16_t)>(
      FpTrunc::Name({Type::S(F16), Type::S(F8E4M3FN)}) + "_itofp");
  EXPECT_EQ(fptrunc(0x7FFF), 0x7F);  // overflows
  EXPECT_EQ(fptrunc(0x0), 0x0);
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0b1100000000000000)),
            static_cast<int8_t>(0b11000000));  // -2.0
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0b0011100000000000)),
            0b00110000);  // 0.5
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0b1011100000000000)),
            static_cast<int8_t>(0b10110000));  // -0.5

  // Test denormals (exponent all 0s) round to 0 in fp8e4m3fn.
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0b0000000100000000)),
            static_cast<int8_t>(0b00000000));
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0b0000000010000000)),
            static_cast<int8_t>(0b00000000));
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0b0000000001000000)),
            static_cast<int8_t>(0b00000000));
}

TEST(FpTruncExecutionTest, F16ToF8e4m3fn_Vector4) {
  JitRunner jit = CreateJitRunner(Type::V(F16, 4), Type::V(F8E4M3FN, 4));
  auto fptrunc = jit.GetVectorizedFn<4, int8_t, int16_t>(
      FpTrunc::Name({Type::V(F16, 4), Type::V(F8E4M3FN, 4)}) + "_itofp");
  std::array<int16_t, 4> vals = {0x7FFF, 0x0,
                                 static_cast<int16_t>(0b1100000000000000),
                                 static_cast<int16_t>(0b0011100000000000)};
  std::array<int8_t, 4> actuals = fptrunc(vals);
  EXPECT_EQ(actuals[0], 0x7F);
  EXPECT_EQ(actuals[1], 0x0);
  EXPECT_EQ(actuals[2], static_cast<int8_t>(0b11000000));
  EXPECT_EQ(actuals[3], static_cast<int8_t>(0b00110000));
}

TEST(FpTruncExecutionTest, F32ToF8e4m3fn) {
  JitRunner jit = CreateJitRunner(Type::S(F32), Type::S(F8E4M3FN));
  auto fptrunc = jit.GetScalarFn<int8_t(float)>(
      FpTrunc::Name({Type::S(F32), Type::S(F8E4M3FN)}));
  EXPECT_EQ(fptrunc(0x7FFFFFFF), 0x7F);  // overflows
  EXPECT_EQ(fptrunc(0x0), 0x0);

  EXPECT_EQ(fptrunc(-2.0f), static_cast<int8_t>(0b11000000));
  EXPECT_EQ(fptrunc(0.5f), static_cast<int8_t>(0b00110000));
  EXPECT_EQ(fptrunc(-0.5f), static_cast<int8_t>(0b10110000));
  EXPECT_EQ(fptrunc(0.125f), static_cast<int8_t>(0b00100000));

  // Test halfway points and double-rounding prevention.
  // 368.0f is the halfway point between 352.0f (0x7B) and 384.0f (0x7C).
  // RNE rounds 368.0f to even (384.0f / 0x7C).
  // 367.9f is slightly below halfway and should round down to 352.0f (0x7B).
  // 368.1f is slightly above halfway and should round up to 384.0f (0x7C).
  EXPECT_EQ(fptrunc(367.9f), static_cast<int8_t>(0x7B));
  EXPECT_EQ(fptrunc(368.1f), static_cast<int8_t>(0x7C));
  EXPECT_EQ(fptrunc(368.0f), static_cast<int8_t>(0x7C));

  // Test denormals (exponent all 0s) round to 0 in fp8e4m3fn.
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::denorm_min()), 0);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::min()), 0);
  // largest f32 denormal:
  EXPECT_EQ(fptrunc(1.17549e-38f), 0);
}

TEST(FpTruncExecutionTest, F32ToF8e3m4) {
  JitRunner jit = CreateJitRunner(Type::S(F32), Type::S(F8E3M4));
  auto fptrunc = jit.GetScalarFn<int8_t(float)>(
      FpTrunc::Name({Type::S(F32), Type::S(F8E3M4)}));
  EXPECT_EQ(fptrunc(0x0), 0x0);
  EXPECT_EQ(fptrunc(-2.0f), static_cast<int8_t>(0b11000000));
  EXPECT_EQ(fptrunc(0.5f), static_cast<int8_t>(0b00100000));
  EXPECT_EQ(fptrunc(-0.5f), static_cast<int8_t>(0b10100000));
  EXPECT_EQ(fptrunc(0.0156f), static_cast<int8_t>(0b00000001));

  // test underflow, denormals
  const float smallest_pos_subnormal_f8e3m4 = 0.015625;
  EXPECT_EQ(fptrunc(smallest_pos_subnormal_f8e3m4),
            static_cast<int8_t>(0b00000001));
  EXPECT_EQ(fptrunc(smallest_pos_subnormal_f8e3m4),
            static_cast<int8_t>(0b00000001));
  const float eps = std::numeric_limits<float>::epsilon();
  EXPECT_EQ(fptrunc(smallest_pos_subnormal_f8e3m4 / 2.0f - eps),
            static_cast<int8_t>(0b00000000));

  // test overflows and infinities
  const int8_t inf = static_cast<int8_t>(0b01110000);
  const int8_t neg_inf = static_cast<int8_t>(0b11110000);
  const float max_f8e3m4 = 15.5;
  EXPECT_EQ(fptrunc(max_f8e3m4 + 1.0f), inf);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::infinity()), inf);
  EXPECT_EQ(fptrunc(-std::numeric_limits<float>::infinity()), neg_inf);

  // test nan prop
  const int8_t nan = static_cast<int8_t>(0b01111000);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::quiet_NaN()), nan);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::signaling_NaN()), nan);
}
TEST(FpTruncExecutionTest, F32ToF8e4m3) {
  JitRunner jit = CreateJitRunner(Type::S(F32), Type::S(F8E4M3));
  auto fptrunc = jit.GetScalarFn<int8_t(float)>(
      FpTrunc::Name({Type::S(F32), Type::S(F8E4M3)}));

  // Test basic values
  EXPECT_EQ(fptrunc(0.0f), 0x0);
  EXPECT_EQ(fptrunc(16.0f), static_cast<int8_t>(0b01011000));
  EXPECT_EQ(fptrunc(-16.0f), static_cast<int8_t>(0b11011000));
  EXPECT_EQ(fptrunc(0.5f), static_cast<int8_t>(0b00110000));
  EXPECT_EQ(fptrunc(-0.5f), static_cast<int8_t>(0b10110000));

  // Test underflow and subnormals
  // Smallest positive subnormal for f8e4m3 is 2^-9
  const float smallest_pos_subnormal_f8e4m3 = 0.001953125f;
  EXPECT_EQ(fptrunc(smallest_pos_subnormal_f8e4m3),
            static_cast<int8_t>(0b00000001));

  const float eps = std::numeric_limits<float>::epsilon();
  EXPECT_EQ(fptrunc(smallest_pos_subnormal_f8e4m3 / 2.0f - eps),
            static_cast<int8_t>(0b00000000));

  // Test overflows and infinities
  const int8_t pos_inf = static_cast<int8_t>(0b01111000);
  const int8_t neg_inf = static_cast<int8_t>(0b11111000);
  const float max_f8e4m3 = 448.0f;
  EXPECT_EQ(fptrunc(max_f8e4m3 + 1.0f), pos_inf);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::infinity()), pos_inf);
  EXPECT_EQ(fptrunc(-std::numeric_limits<float>::infinity()), neg_inf);

  // Test NaN propagation
  const int8_t nan = static_cast<int8_t>(0x7C);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::quiet_NaN()), nan);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::signaling_NaN()), nan);
}

TEST(FpTruncExecutionTest, F32ToF8e4m3fnuz) {
  JitRunner jit = CreateJitRunner(Type::S(F32), Type::S(F8E4M3FNUZ));
  auto fptrunc = jit.GetScalarFn<int8_t(float)>(
      FpTrunc::Name({Type::S(F32), Type::S(F8E4M3FNUZ)}));

  EXPECT_EQ(fptrunc(0.0f), 0x0);
  EXPECT_EQ(fptrunc(16.0f), static_cast<int8_t>(0b01100000));
  EXPECT_EQ(fptrunc(-16.0f), static_cast<int8_t>(0b11100000));
  EXPECT_EQ(fptrunc(0.5f), static_cast<int8_t>(0b00111000));
  EXPECT_EQ(fptrunc(-0.5f), static_cast<int8_t>(0b10111000));

  // Test underflow and subnormals (FNUZ formats often support subnormals)
  const float smallest_pos_subnormal = 0.0009765625f;
  EXPECT_EQ(fptrunc(smallest_pos_subnormal), static_cast<int8_t>(0b00000001));
  EXPECT_EQ(fptrunc(smallest_pos_subnormal * 2.0f),
            static_cast<int8_t>(0b00000010));
  const float eps = std::numeric_limits<float>::epsilon();
  EXPECT_EQ(fptrunc(smallest_pos_subnormal / 2.0f - eps), 0x0);

  // Test overflows (clamps to nan, no infinity)
  const float max_val = 240.0f;
  EXPECT_EQ(fptrunc(max_val), static_cast<int8_t>(0b01111111));
  EXPECT_EQ(fptrunc(-max_val), static_cast<int8_t>(0b11111111));
  const int8_t nan = static_cast<int8_t>(0b10000000);
  EXPECT_EQ(fptrunc(max_val + 10.0f), nan);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::infinity()), nan);
  EXPECT_EQ(fptrunc(-(max_val + 10.0f)), nan);
  EXPECT_EQ(fptrunc(-std::numeric_limits<float>::infinity()), nan);

  // Test NaN propagation
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::quiet_NaN()), nan);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::signaling_NaN()), nan);
}

TEST(FpTruncExecutionTest, F32ToF8e4m3fnuz_Vector4) {
  JitRunner jit = CreateJitRunner(Type::V(F32, 4), Type::V(F8E4M3FNUZ, 4));
  auto fptrunc = jit.GetVectorizedFn<4, int8_t, float>(
      FpTrunc::Name({Type::V(F32, 4), Type::V(F8E4M3FNUZ, 4)}));
  std::array<float, 4> vals = {500, 16.0f, -0.5f, -240.0f};
  std::array<int8_t, 4> actuals = fptrunc(vals);
  EXPECT_EQ(actuals[0], static_cast<int8_t>(0b10000000));
  EXPECT_EQ(actuals[1], static_cast<int8_t>(0b01100000));
  EXPECT_EQ(actuals[2], static_cast<int8_t>(0b10111000));
  EXPECT_EQ(actuals[3], static_cast<int8_t>(0b11111111));
}

TEST(FpTruncExecutionTest, F32ToF8e4m3fnuz_Vector8) {
  JitRunner jit = CreateJitRunner(Type::V(F32, 8), Type::V(F8E4M3FNUZ, 8));
  auto fptrunc = jit.GetVectorizedFn<8, int8_t, float>(
      FpTrunc::Name({Type::V(F32, 8), Type::V(F8E4M3FNUZ, 8)}));
  std::array<float, 8> vals = {500,   16.0f,  -0.5f,  -240.0f,
                               -1.0f, -16.0f, 240.0f, 242.0f};
  std::array<int8_t, 8> actuals = fptrunc(vals);
  EXPECT_EQ(actuals[0], static_cast<int8_t>(0b10000000));
  EXPECT_EQ(actuals[1], static_cast<int8_t>(0b01100000));
  EXPECT_EQ(actuals[2], static_cast<int8_t>(0b10111000));
  EXPECT_EQ(actuals[3], static_cast<int8_t>(0b11111111));
  EXPECT_EQ(actuals[4], static_cast<int8_t>(0b11000000));
  EXPECT_EQ(actuals[5], static_cast<int8_t>(0b11100000));
  EXPECT_EQ(actuals[6], static_cast<int8_t>(0b01111111));

  // 242.0f rounds down to 240.0f:
  EXPECT_EQ(actuals[7], static_cast<int8_t>(0b01111111));
}

TEST(FpTruncExecutionTest, F32ToF8e4m3fn_Vector8) {
  JitRunner jit = CreateJitRunner(Type::V(F32, 8), Type::V(F8E4M3FN, 8));
  auto fptrunc = jit.GetVectorizedFn<8, int8_t, float>(
      FpTrunc::Name({Type::V(F32, 8), Type::V(F8E4M3FN, 8)}));
  std::array<float, 8> vals = {500,   16.0f,  -0.5f, -240.0f,
                               -1.0f, -16.0f, 0.5f,  242.0f};
  std::array<int8_t, 8> actuals = fptrunc(vals);
  EXPECT_EQ(actuals[0], static_cast<int8_t>(0x7F));        // 500 -> NaN
  EXPECT_EQ(actuals[1], static_cast<int8_t>(0b01011000));  // 16.0f
  EXPECT_EQ(actuals[2], static_cast<int8_t>(0b10110000));  // -0.5f
  EXPECT_EQ(actuals[3], static_cast<int8_t>(0b11110111));  // -240.0f
  EXPECT_EQ(actuals[4], static_cast<int8_t>(0b10111000));  // -1.0f
  EXPECT_EQ(actuals[5], static_cast<int8_t>(0b11011000));  // -16.0f
  EXPECT_EQ(actuals[6], static_cast<int8_t>(0b00110000));  // 0.5f

  // 242.0f rounds down to 240.0f:
  EXPECT_EQ(actuals[7], static_cast<int8_t>(0b01110111));
}

TEST(FpTruncExecutionTest, F32ToBf16_Vector16) {
  constexpr size_t kN = 16;
  const Type from = Type::V(F32, kN);
  const Type to = Type::V(BF16, kN);
  JitRunner jit = CreateJitRunner(from, to);
  auto fptrunc = jit.GetVectorizedFn<kN, uint16_t, float>(
      FpTrunc::Name({from, to}) + "_fptoi");
  constexpr float kEps = std::numeric_limits<float>::epsilon();
  std::array<float, kN> vals = {
      1.0f, -2.0f, 0.5f, 0.0f, -0.0f, 3.0f, -0.75f,
      // Rounds up to 65536.0f.
      65504.0f,
      // Ties round to even: 1 + 2^-8 rounds down, 1 + 3 * 2^-8 rounds up.
      1.00390625f, 1.01171875f,
      // Just above a tie rounds up.
      1.00390625f + kEps,
      // Rounds up to infinity.
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN()};
  std::array<uint16_t, kN> expected = {
      0x3F80, 0xC000, 0x3F00, 0x0000, 0x8000, 0x4040, 0xBF40, 0x4780,
      0x3F80, 0x3F82, 0x3F81, 0x7F80, 0x0000, 0x7F80, 0xFF80, 0x7FC0};
  EXPECT_EQ(fptrunc(vals), expected);
}

TEST(FpTruncExecutionTest, F32ToF8e4m3b11fnuz) {
  JitRunner jit = CreateJitRunner(Type::S(F32), Type::S(F8E4M3B11FNUZ));
  auto fptrunc = jit.GetScalarFn<int8_t(float)>(
      FpTrunc::Name({Type::S(F32), Type::S(F8E4M3B11FNUZ)}));

  // Test basic values (bias of 11 shifts the range)
  EXPECT_EQ(fptrunc(0.0f), 0x0);
  EXPECT_EQ(fptrunc(16.0f), static_cast<int8_t>(0b01111000));
  EXPECT_EQ(fptrunc(-16.0f), static_cast<int8_t>(0b11111000));
  EXPECT_EQ(fptrunc(0.5f), static_cast<int8_t>(0b01010000));
  EXPECT_EQ(fptrunc(-0.5f), static_cast<int8_t>(0b11010000));

  // Test underflow and subnormals
  // Smallest subnormal is 0.125 * 2^(1-11) = 2^-3 * 2^-10 = 2^-13
  constexpr float smallest_pos_subnormal = 0.0001220703125f;
  EXPECT_EQ(fptrunc(smallest_pos_subnormal), static_cast<int8_t>(0b00000001));
  EXPECT_EQ(fptrunc(smallest_pos_subnormal * 2.0f),
            static_cast<int8_t>(0b00000010));
  const float eps = std::numeric_limits<float>::epsilon();
  EXPECT_EQ(fptrunc(smallest_pos_subnormal / 2.0f - eps), 0x0);

  // Test overflows (clamps to nan, no infinity)
  // Max value is (1 + 7/8) * 2^(15-11) = 1.875 * 16 = 30
  const float max_val = 30.0f;
  EXPECT_EQ(fptrunc(max_val), static_cast<int8_t>(0b01111111));
  EXPECT_EQ(fptrunc(-max_val), static_cast<int8_t>(0b11111111));
  const int8_t nan = static_cast<int8_t>(0b10000000);
  EXPECT_EQ(fptrunc(max_val + 1.0f), nan);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::infinity()), nan);
  EXPECT_EQ(fptrunc(-(max_val + 1.0f)), nan);
  EXPECT_EQ(fptrunc(-std::numeric_limits<float>::infinity()), nan);
  EXPECT_EQ(fptrunc(166.0f), nan);

  // Test NaN propagation
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::quiet_NaN()), nan);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::signaling_NaN()), nan);
}

TEST(FpTruncExecutionTest, F32ToF8e5m2) {
  JitRunner jit = CreateJitRunner(Type::S(F32), Type::S(F8E5M2));
  auto fptrunc = jit.GetScalarFn<int8_t(float)>(
      FpTrunc::Name({Type::S(F32), Type::S(F8E5M2)}));

  // Test basic values (bias of 15)
  EXPECT_EQ(fptrunc(0.0f), 0x0);
  EXPECT_EQ(fptrunc(16.0f), static_cast<int8_t>(0b01001100));
  EXPECT_EQ(fptrunc(-16.0f), static_cast<int8_t>(0b11001100));
  EXPECT_EQ(fptrunc(0.5f), static_cast<int8_t>(0b00111000));
  EXPECT_EQ(fptrunc(-0.5f), static_cast<int8_t>(0b10111000));

  // Test halfway points and double-rounding prevention.
  // 1.875f is the halfway point between 1.75f (0x3F) and 2.0f (0x40).
  // RNE rounds 1.875f to even (2.0f / 0x40).
  // 1.8749f is slightly below halfway and should round down to 1.75f (0x3F).
  // 1.8751f is slightly above halfway and should round up to 2.0f (0x40).
  EXPECT_EQ(fptrunc(1.8749f), static_cast<int8_t>(0x3F));
  EXPECT_EQ(fptrunc(1.8751f), static_cast<int8_t>(0x40));
  EXPECT_EQ(fptrunc(1.8750f), static_cast<int8_t>(0x40));

  // Test underflow and subnormals
  // Smallest subnormal is 0.25 * 2^(1-15) = 2^-2 * 2^-14 = 2^-16
  constexpr float smallest_pos_subnormal = 0.0000152587890625f;
  EXPECT_EQ(fptrunc(smallest_pos_subnormal), static_cast<int8_t>(0b00000001));
  EXPECT_EQ(fptrunc(smallest_pos_subnormal * 2.0f),
            static_cast<int8_t>(0b00000010));
  const float eps = std::numeric_limits<float>::epsilon();
  EXPECT_EQ(fptrunc(smallest_pos_subnormal / 2.0f - eps), 0x0);

  // Test overflows (goes to infinity)
  // Max value is (1 + 3/4) * 2^(30-15) = 1.75 * 2^15 = 57344
  const float max_val = 57344.0f;
  EXPECT_EQ(fptrunc(max_val), static_cast<int8_t>(0b01111011));
  EXPECT_EQ(fptrunc(-max_val), static_cast<int8_t>(0b11111011));
  const int8_t pos_inf = static_cast<int8_t>(0b01111100);
  const int8_t neg_inf = static_cast<int8_t>(0b11111100);
  EXPECT_EQ(fptrunc(max_val * 2.0f), pos_inf);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::infinity()), pos_inf);
  EXPECT_EQ(fptrunc(-max_val * 2.0f), neg_inf);
  EXPECT_EQ(fptrunc(-std::numeric_limits<float>::infinity()), neg_inf);

  // Test NaN propagation
  const int8_t nan = static_cast<int8_t>(0b01111110);  // Canonical qNaN
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::quiet_NaN()), nan);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::signaling_NaN()), nan);
}

TEST(FpTruncExecutionTest, F32ToF8e5m2fnuz) {
  JitRunner jit = CreateJitRunner(Type::S(F32), Type::S(F8E5M2FNUZ));
  auto fptrunc = jit.GetScalarFn<int8_t(float)>(
      FpTrunc::Name({Type::S(F32), Type::S(F8E5M2FNUZ)}));

  // Test basic values (bias of 16 shifts the range)
  EXPECT_EQ(fptrunc(0.0f), 0x0);
  EXPECT_EQ(fptrunc(-0.0f), 0x0);
  EXPECT_EQ(fptrunc(16.0f), static_cast<int8_t>(0b01010000));
  EXPECT_EQ(fptrunc(-16.0f), static_cast<int8_t>(0b11010000));
  EXPECT_EQ(fptrunc(0.5f), static_cast<int8_t>(0b00111100));
  EXPECT_EQ(fptrunc(-0.5f), static_cast<int8_t>(0b10111100));

  // Test underflow and subnormals
  // Smallest subnormal is 0.25 * 2^(1-16) = 2^-2 * 2^-15 = 2^-17
  constexpr float smallest_pos_subnormal = 0.00000762939453125f;
  EXPECT_EQ(fptrunc(smallest_pos_subnormal), static_cast<int8_t>(0b00000001));
  EXPECT_EQ(fptrunc(smallest_pos_subnormal * 2.0f),
            static_cast<int8_t>(0b00000010));
  const float eps = std::numeric_limits<float>::epsilon();
  EXPECT_EQ(fptrunc(smallest_pos_subnormal / 2.0f - eps), 0x0);

  // Test overflows (clamps to nan, no infinity)
  // Max value is (1 + 3/4) * 2^(30-16) = 1.75 * 2^14 = 28672
  const float max_val = 57344.0f;
  EXPECT_EQ(fptrunc(max_val), static_cast<int8_t>(0b01111111));
  EXPECT_EQ(fptrunc(-max_val), static_cast<int8_t>(0b11111111));
  const int8_t nan = static_cast<int8_t>(0b10000000);
  EXPECT_EQ(fptrunc(max_val * 2.0f), nan);  // Overflow
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::infinity()), nan);
  EXPECT_EQ(fptrunc(-max_val * 2.0f), nan);  // Overflow
  EXPECT_EQ(fptrunc(-std::numeric_limits<float>::infinity()), nan);

  // Test NaN propagation
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::quiet_NaN()), nan);
  EXPECT_EQ(fptrunc(std::numeric_limits<float>::signaling_NaN()), nan);
}

TEST(FpTruncExecutionTest, F16ToF8e5m2fnuz) {
  JitRunner jit = CreateJitRunner(Type::S(F16), Type::S(F8E5M2FNUZ));
  auto fptrunc = jit.GetScalarFn<int8_t(int16_t)>(
      FpTrunc::Name({Type::S(F16), Type::S(F8E5M2FNUZ)}) + "_itofp");

  EXPECT_EQ(fptrunc(static_cast<int16_t>(0x00)),
            static_cast<int8_t>(0b00000000));
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0b1100000000000000)),
            static_cast<int8_t>(0b11000100));  // -2.0
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0b0011100000000000)),
            0b00111100);  // 0.5
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0b1011100000000000)),
            static_cast<int8_t>(0b10111100));  // -0.5

  // Test overflow behavior (should be NaN for FNUZ)
  // Max finite F8E5M2FNUZ is 57344.0.
  // 47616.0 in F16 is 0x79D0. It rounds to 49152.0 (0x7E).
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0x79D0)),
            static_cast<int8_t>(0b01111110));  // 47616.0 -> 0x7E

  // Max finite F16 is 65504.0 (0x7BFF).
  // 65504.0 > 61440.0 (cutoff for overflow in F8E5M2FNUZ), so it should become
  // NaN.
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0x7BFF)),
            static_cast<int8_t>(0b10000000));  // 65504.0 -> NaN

  EXPECT_EQ(fptrunc(static_cast<int16_t>(0x7C00)),
            static_cast<int8_t>(0b10000000));  // Inf -> NaN
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0xFC00)),
            static_cast<int8_t>(0b10000000));  // -Inf -> NaN

  // Test NaN
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0x7E00)),
            static_cast<int8_t>(0b10000000));  // NaN -> NaN
}

TEST(FpTruncExecutionTest, F64ToF8e5m2) {
  JitRunner jit = CreateJitRunner(Type::S(F64), Type::S(F8E5M2));
  // JitRunner doesn't have a direct helper for F64 input to int8_t output
  // wrapped as int64->int8 But GetScalarFn is a template. We need to pass
  // double. However, CreateWrapperIntArgToFp creates an integer argument
  // wrapper. For F64, the wrapper will take int64_t and bitcast to double.
  auto fptrunc = jit.GetScalarFn<int8_t(int64_t)>(
      FpTrunc::Name({Type::S(F64), Type::S(F8E5M2)}) + "_itofp");

  EXPECT_EQ(fptrunc(0), 0);
  // 1.0 in double: 0x3FF0000000000000
  // F8E5M2: 1 sign, 5 exp, 2 mantissa. Bias 15.
  // 1.0 = 1.0 * 2^0 -> Exp = 15 = 01111. Mantissa 00. Result 00111100 = 0x3C
  EXPECT_EQ(fptrunc(0x3FF0000000000000), 0x3C);

  // 1.5 in double: 0x3FF8000000000000
  // F8E5M2: 1.5 = 1.1 * 2^0 -> Exp 15. Mantissa 10. Result 00111110 = 0x3E
  EXPECT_EQ(fptrunc(0x3FF8000000000000), 0x3E);

  // Max finite F8E5M2: 57344.0
  // In double: 0x40EC000000000000
  // F8E5M2: 0x7B
  EXPECT_EQ(fptrunc(0x40EC000000000000), 0x7B);

  // 60000.0 in double: 0x40ED4C0000000000
  // 60000 < 61440 (overflow threshold), so it rounds to max finite (57344).
  EXPECT_EQ(fptrunc(0x40ED4C0000000000), 0x7B);

  // Overflow -> Inf
  // 62000.0 in double: 0x40EE460000000000
  // 62000 > 61440, so overflow.
  EXPECT_EQ(fptrunc(0x40EE460000000000), 0x7C);
}

TEST(FpTruncExecutionTest, F16ToF8e5m2) {
  JitRunner jit = CreateJitRunner(Type::S(F16), Type::S(F8E5M2));
  auto fptrunc = jit.GetScalarFn<int8_t(int16_t)>(
      FpTrunc::Name({Type::S(F16), Type::S(F8E5M2)}) + "_itofp");

  // Bias is 15 for both.
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0x00)), 0x00);
  // 1.0 (0 01111 0000000000) -> 1.0 (0 01111 00)
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0x3C00)), 0x3C);
  // 1.5 (0 01111 1000000000) -> 1.5 (0 01111 10)
  EXPECT_EQ(fptrunc(static_cast<int16_t>(0x3E00)), 0x3E);
}

namespace {

absl::Status AddFpTruncFunctions(llvm::Module* module, PrimitiveType from,
                                 PrimitiveType to, bool set_fast_math = false) {
  auto add_one = [&](Type f, Type t) -> absl::Status {
    ABSL_ASSIGN_OR_RETURN(llvm::Function * func,
                          FpTrunc::CreateDefinition(module, f, t));
    func->setLinkage(llvm::Function::ExternalLinkage);
    if (set_fast_math) {
      for (llvm::BasicBlock& bb : *func) {
        for (llvm::Instruction& inst : bb) {
          if (llvm::isa<llvm::FPMathOperator>(inst)) {
            inst.setFastMathFlags(llvm::FastMathFlags::getFast());
          }
        }
      }
    }
    if (llvm::verifyFunction(*func)) {
      return absl::InternalError(
          absl::StrCat("LLVM verification failed for ", func->getName().str()));
    }
    llvm::Function* wrapper = intrinsic::CreateWrapperIntArgToFp(func);
    wrapper->setLinkage(llvm::Function::ExternalLinkage);
    if (llvm::verifyFunction(*wrapper)) {
      return absl::InternalError(absl::StrCat("LLVM verification failed for ",
                                              wrapper->getName().str()));
    }
    return absl::OkStatus();
  };
  ABSL_RETURN_IF_ERROR(add_one(Type::S(from), Type::S(to)));
  for (size_t width : {4, 8, 16}) {
    ABSL_RETURN_IF_ERROR(add_one(Type::V(from, width), Type::V(to, width)));
  }
  return absl::OkStatus();
}

absl::StatusOr<JitRunner> CreateAllF8FromJitRunner(PrimitiveType from,
                                                   bool set_fast_math = false) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module =
      std::make_unique<llvm::Module>("fptrunc_all_f8_module", *context);
  for (PrimitiveType f8_type : {F8E4M3FN, F8E5M2, F8E4M3, F8E3M4, F8E4M3FNUZ,
                                F8E5M2FNUZ, F8E4M3B11FNUZ, F8E8M0FNU}) {
    ABSL_RETURN_IF_ERROR(
        AddFpTruncFunctions(module.get(), from, f8_type, set_fast_math));
  }
  if (set_fast_math) {
    intrinsic::RunInlineAndOptPasses(*module);
    if (llvm::verifyModule(*module)) {
      return absl::InternalError("LLVM verification failed for module");
    }
  }
  return JitRunner(std::move(module), std::move(context));
}

template <typename Src, typename SrcBits, typename F8>
uint8_t ExpectedTruncBits(SrcBits bits) {
  Src src;
  std::memcpy(&src, &bits, sizeof(Src));
  F8 expected;
  if constexpr (std::is_same_v<Src, half> || std::is_same_v<Src, bfloat16>) {
    expected = static_cast<F8>(static_cast<float>(src));
  } else {
    expected = static_cast<F8>(src);
  }
  uint8_t expected_bits = 0;
  std::memcpy(&expected_bits, &expected, 1);
  return expected_bits;
}

template <typename Src, typename SrcBits, typename F8, size_t Width>
void VerifyVectorTrunc(JitRunner& jit, PrimitiveType from_type,
                       PrimitiveType to_type,
                       const std::vector<SrcBits>& inputs) {
  auto fn = jit.GetVectorizedFn<Width, uint8_t, SrcBits>(
      FpTrunc::Name({Type::V(from_type, Width), Type::V(to_type, Width)}) +
      "_itofp");
  for (size_t base = 0; base < inputs.size(); base += Width) {
    std::array<SrcBits, Width> in{};
    for (size_t lane = 0; lane < Width; ++lane) {
      in[lane] = inputs[base + lane];
    }
    std::array<uint8_t, Width> out = fn(in);
    for (size_t lane = 0; lane < Width; ++lane) {
      EXPECT_EQ(out[lane], (ExpectedTruncBits<Src, SrcBits, F8>(in[lane])))
          << "width=" << Width << " input=0x" << std::hex << in[lane];
    }
  }
}

template <typename Src, typename SrcBits, typename F8>
void VerifyAllWidthsTrunc(JitRunner& jit, PrimitiveType from_type,
                          PrimitiveType to_type,
                          const std::vector<SrcBits>& inputs) {
  auto scalar_fn = jit.GetScalarFn<uint8_t(SrcBits)>(
      FpTrunc::Name({Type::S(from_type), Type::S(to_type)}) + "_itofp");
  for (SrcBits bits : inputs) {
    EXPECT_EQ(scalar_fn(bits), (ExpectedTruncBits<Src, SrcBits, F8>(bits)))
        << "scalar input=0x" << std::hex << bits;
  }
  VerifyVectorTrunc<Src, SrcBits, F8, 4>(jit, from_type, to_type, inputs);
  VerifyVectorTrunc<Src, SrcBits, F8, 8>(jit, from_type, to_type, inputs);
  VerifyVectorTrunc<Src, SrcBits, F8, 16>(jit, from_type, to_type, inputs);
}

template <typename Src, typename SrcBits>
void VerifyAllF8TypesTrunc(JitRunner& jit, PrimitiveType from_type,
                           const std::vector<SrcBits>& inputs) {
  VerifyAllWidthsTrunc<Src, SrcBits, float8_e4m3fn>(jit, from_type, F8E4M3FN,
                                                    inputs);
  VerifyAllWidthsTrunc<Src, SrcBits, float8_e5m2>(jit, from_type, F8E5M2,
                                                  inputs);
  VerifyAllWidthsTrunc<Src, SrcBits, float8_e4m3>(jit, from_type, F8E4M3,
                                                  inputs);
  VerifyAllWidthsTrunc<Src, SrcBits, float8_e3m4>(jit, from_type, F8E3M4,
                                                  inputs);
  VerifyAllWidthsTrunc<Src, SrcBits, float8_e4m3fnuz>(jit, from_type,
                                                      F8E4M3FNUZ, inputs);
  VerifyAllWidthsTrunc<Src, SrcBits, float8_e5m2fnuz>(jit, from_type,
                                                      F8E5M2FNUZ, inputs);
  VerifyAllWidthsTrunc<Src, SrcBits, float8_e4m3b11fnuz>(jit, from_type,
                                                         F8E4M3B11FNUZ, inputs);
  VerifyAllWidthsTrunc<Src, SrcBits, float8_e8m0fnu>(jit, from_type, F8E8M0FNU,
                                                     inputs);
}

std::vector<uint32_t> BuildDenseF32Inputs() {
  const std::array<uint32_t, 36> mant_samples = {
      0x000000u, 0x000001u, 0x07FFFFu, 0x080000u, 0x080001u, 0x0FFFFFu,
      0x100000u, 0x100001u, 0x17FFFFu, 0x180000u, 0x180001u, 0x1FFFFFu,
      0x200000u, 0x200001u, 0x27FFFFu, 0x280000u, 0x280001u, 0x2FFFFFu,
      0x300000u, 0x300001u, 0x380000u, 0x3FFFFFu, 0x400000u, 0x400001u,
      0x4FFFFFu, 0x500000u, 0x5FFFFFu, 0x600000u, 0x600001u, 0x680000u,
      0x6FFFFFu, 0x700000u, 0x700001u, 0x77FFFFu, 0x780000u, 0x7FFFFFu,
  };
  std::vector<uint32_t> inputs;
  inputs.reserve(2 * 256 * mant_samples.size() + 16);
  for (uint32_t sign : {0u, 1u}) {
    for (uint32_t exp = 0; exp < 256; ++exp) {
      for (uint32_t mant : mant_samples) {
        inputs.push_back((sign << 31) | (exp << 23) | mant);
      }
    }
  }
  while (inputs.size() % 16 != 0) {
    inputs.push_back(0);
  }
  return inputs;
}

std::vector<uint64_t> BuildDenseF64Inputs() {
  const std::array<uint64_t, 18> mant_samples = {
      0x0000000000000ULL, 0x0000000000001ULL, 0x07FFFFFFFFFFFULL,
      0x0800000000000ULL, 0x0800000000001ULL, 0x0FFFFFFFFFFFFULL,
      0x1000000000000ULL, 0x1000000000001ULL, 0x2000000000000ULL,
      0x3FFFFFFFFFFFFULL, 0x4000000000000ULL, 0x4000000000001ULL,
      0x7FFFFFFFFFFFFULL, 0x8000000000000ULL, 0x8000000000001ULL,
      0xC000000000000ULL, 0xFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFULL,
  };
  std::vector<uint64_t> exp_samples = {0, 1, 2, 894, 895, 896, 897, 898};
  for (uint64_t e = 1000; e <= 1048; ++e) {
    exp_samples.push_back(e);
  }
  for (uint64_t e :
       {1148ULL, 1149ULL, 1150ULL, 1151ULL, 1152ULL, 2046ULL, 2047ULL}) {
    exp_samples.push_back(e);
  }
  std::vector<uint64_t> inputs;
  inputs.reserve(2 * exp_samples.size() * mant_samples.size() + 16);
  for (uint64_t sign : {0ULL, 1ULL}) {
    for (uint64_t exp : exp_samples) {
      for (uint64_t mant : mant_samples) {
        inputs.push_back((sign << 63) | (exp << 52) | mant);
      }
    }
  }
  while (inputs.size() % 16 != 0) {
    inputs.push_back(0);
  }
  return inputs;
}

}  // namespace

TEST(FpTruncExecutionTest, ExhaustiveAllF16ToAllF8) {
  std::vector<uint16_t> all_u16(65536);
  for (int i = 0; i < 65536; ++i) {
    all_u16[i] = static_cast<uint16_t>(i);
  }
  ASSERT_OK_AND_ASSIGN(JitRunner jit, CreateAllF8FromJitRunner(F16));
  VerifyAllF8TypesTrunc<half, uint16_t>(jit, F16, all_u16);
}

TEST(FpTruncExecutionTest, ExhaustiveAllBf16ToAllF8) {
  std::vector<uint16_t> all_u16(65536);
  for (int i = 0; i < 65536; ++i) {
    all_u16[i] = static_cast<uint16_t>(i);
  }
  ASSERT_OK_AND_ASSIGN(JitRunner jit, CreateAllF8FromJitRunner(BF16));
  VerifyAllF8TypesTrunc<bfloat16, uint16_t>(jit, BF16, all_u16);
}

TEST(FpTruncExecutionTest, DenseF32ToAllF8) {
  ASSERT_OK_AND_ASSIGN(JitRunner jit, CreateAllF8FromJitRunner(F32));
  VerifyAllF8TypesTrunc<float, uint32_t>(jit, F32, BuildDenseF32Inputs());
}

TEST(FpTruncExecutionTest, DenseF64ToAllF8) {
  ASSERT_OK_AND_ASSIGN(JitRunner jit, CreateAllF8FromJitRunner(F64));
  VerifyAllF8TypesTrunc<double, uint64_t>(jit, F64, BuildDenseF64Inputs());
}

TEST(FpTruncExecutionTest, DenseF32ToAllF8FastMathSafe) {
  ASSERT_OK_AND_ASSIGN(JitRunner jit,
                       CreateAllF8FromJitRunner(F32, /*set_fast_math=*/true));
  VerifyAllF8TypesTrunc<float, uint32_t>(jit, F32, BuildDenseF32Inputs());
}

}  // namespace xla::codegen::intrinsics
