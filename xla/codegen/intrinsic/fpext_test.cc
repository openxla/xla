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

#include "xla/codegen/intrinsic/fpext.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ios>
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "xla/codegen/intrinsic/simple_jit_runner.h"
#include "xla/codegen/intrinsic/type.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {
namespace {

using ::xla::codegen::intrinsic::JitRunner;

TEST(FpExtTest, ScalarIntrinsic) {
  EXPECT_EQ(FpExt::Name({Type::S(F8E4M3FN), Type::S(F32)}),
            "xla.fpext.f8e4m3fn.to.f32");
}

TEST(FpExtTest, VectorIntrinsic) {
  EXPECT_EQ(FpExt::Name({Type::V(F8E5M2, 8), Type::V(F16, 8)}),
            "xla.fpext.v8f8e5m2.to.v8f16");
}

absl::Status AddFpExtFunctions(llvm::Module* module, PrimitiveType from,
                               PrimitiveType to) {
  auto add_one = [&](Type f, Type t) -> absl::Status {
    ABSL_ASSIGN_OR_RETURN(llvm::Function * func,
                          FpExt::CreateDefinition(module, f, t));
    func->setLinkage(llvm::Function::ExternalLinkage);
    if (llvm::verifyFunction(*func)) {
      return absl::InternalError(
          absl::StrCat("LLVM verification failed for ", func->getName().str()));
    }
    llvm::Function* wrapper = intrinsic::CreateWrapperFpRetToInt(func);
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

absl::StatusOr<JitRunner> CreateAllF8ToJitRunner(PrimitiveType to) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("fpext_test_module", *context);
  for (PrimitiveType f8_type : {F8E4M3FN, F8E5M2, F8E4M3, F8E3M4, F8E4M3FNUZ,
                                F8E5M2FNUZ, F8E4M3B11FNUZ, F8E8M0FNU}) {
    ABSL_RETURN_IF_ERROR(AddFpExtFunctions(module.get(), f8_type, to));
  }
  return JitRunner(std::move(module), std::move(context));
}

template <typename F8, typename Dst, typename DstBits>
void CheckExtMatch(uint8_t input_bits, DstBits actual_bits,
                   bool is_fnuz = false) {
  F8 val;
  std::memcpy(&val, &input_bits, 1);
  Dst expected;
  if constexpr (std::is_same_v<Dst, half>) {
    expected = static_cast<half>(val);
  } else if constexpr (std::is_same_v<Dst, bfloat16>) {
    expected = static_cast<bfloat16>(static_cast<float>(val));
  } else {
    expected = static_cast<Dst>(static_cast<float>(val));
  }
  DstBits expected_bits = 0;
  std::memcpy(&expected_bits, &expected, sizeof(DstBits));

  constexpr int kSignShift = sizeof(DstBits) * 8 - 1;
  const bool expected_is_nan = std::isnan(static_cast<float>(expected));
  if (expected_is_nan) {
    Dst actual;
    std::memcpy(&actual, &actual_bits, sizeof(Dst));
    EXPECT_TRUE(std::isnan(static_cast<float>(actual)))
        << "input=0x" << std::hex << static_cast<int>(input_bits);
    if (is_fnuz) {
      EXPECT_EQ(actual_bits >> kSignShift, 0u)
          << "FNUZ NaN must be positive, input=0x" << std::hex
          << static_cast<int>(input_bits);
    } else {
      EXPECT_EQ(actual_bits >> kSignShift, expected_bits >> kSignShift)
          << "NaN sign mismatch, input=0x" << std::hex
          << static_cast<int>(input_bits);
    }
  } else {
    EXPECT_EQ(actual_bits, expected_bits)
        << "input=0x" << std::hex << static_cast<int>(input_bits);
  }
}

template <typename F8, typename Dst, typename DstBits, size_t Width>
void VerifyVectorExt(JitRunner& jit, PrimitiveType from_type,
                     PrimitiveType to_type, bool is_fnuz) {
  auto fn = jit.GetVectorizedFn<Width, DstBits, uint8_t>(
      FpExt::Name({Type::V(from_type, Width), Type::V(to_type, Width)}) +
      "_fptoi");
  for (int base = 0; base < 256; base += Width) {
    std::array<uint8_t, Width> in{};
    for (size_t lane = 0; lane < Width; ++lane) {
      in[lane] = static_cast<uint8_t>(base + lane);
    }
    std::array<DstBits, Width> out = fn(in);
    for (size_t lane = 0; lane < Width; ++lane) {
      CheckExtMatch<F8, Dst, DstBits>(in[lane], out[lane], is_fnuz);
    }
  }
}

template <typename F8, typename Dst, typename DstBits>
void VerifyAllWidthsExt(JitRunner& jit, PrimitiveType from_type,
                        PrimitiveType to_type, bool is_fnuz = false) {
  auto scalar_fn = jit.GetScalarFn<DstBits(uint8_t)>(
      FpExt::Name({Type::S(from_type), Type::S(to_type)}) + "_fptoi");
  for (int i = 0; i < 256; ++i) {
    uint8_t bits = static_cast<uint8_t>(i);
    CheckExtMatch<F8, Dst, DstBits>(bits, scalar_fn(bits), is_fnuz);
  }
  VerifyVectorExt<F8, Dst, DstBits, 4>(jit, from_type, to_type, is_fnuz);
  VerifyVectorExt<F8, Dst, DstBits, 8>(jit, from_type, to_type, is_fnuz);
  VerifyVectorExt<F8, Dst, DstBits, 16>(jit, from_type, to_type, is_fnuz);
}

TEST(FpExtExecutionTest, F8e4m3fnToF16) {
  ASSERT_OK_AND_ASSIGN(JitRunner jit, CreateAllF8ToJitRunner(F16));
  auto fpext = jit.GetScalarFn<int16_t(int8_t)>(
      FpExt::Name({Type::S(F8E4M3FN), Type::S(F16)}) + "_fptoi");
  EXPECT_EQ(fpext(static_cast<int8_t>(0b11000000)),
            static_cast<int16_t>(0b1100000000000000));  // -2.0
  EXPECT_EQ(fpext(static_cast<int8_t>(0b00110000)),
            static_cast<int16_t>(0b0011100000000000));  // 0.5
  EXPECT_EQ(fpext(static_cast<int8_t>(0b10110000)),
            static_cast<int16_t>(0b1011100000000000));  // -0.5
}

TEST(FpExtExecutionTest, F8e4m3fnToF16_Vector4) {
  ASSERT_OK_AND_ASSIGN(JitRunner jit, CreateAllF8ToJitRunner(F16));
  auto fpext = jit.GetVectorizedFn<4, int16_t, int8_t>(
      FpExt::Name({Type::V(F8E4M3FN, 4), Type::V(F16, 4)}) + "_fptoi");
  std::array<int8_t, 4> vals = {static_cast<int8_t>(0b11000000), 0x0,
                                static_cast<int8_t>(0b10110000),
                                static_cast<int8_t>(0b00110000)};
  std::array<int16_t, 4> actuals = fpext(vals);
  EXPECT_EQ(actuals[0], static_cast<int16_t>(0b1100000000000000));
  EXPECT_EQ(actuals[1], static_cast<int16_t>(0b0000000000000000));
  EXPECT_EQ(actuals[2], static_cast<int16_t>(0b1011100000000000));
  EXPECT_EQ(actuals[3], static_cast<int16_t>(0b0011100000000000));
}

TEST(FpExtExecutionTest, ExhaustiveAllF8ToF32) {
  ASSERT_OK_AND_ASSIGN(JitRunner jit, CreateAllF8ToJitRunner(F32));
  VerifyAllWidthsExt<float8_e4m3fn, float, uint32_t>(jit, F8E4M3FN, F32);
  VerifyAllWidthsExt<float8_e5m2, float, uint32_t>(jit, F8E5M2, F32);
  VerifyAllWidthsExt<float8_e4m3, float, uint32_t>(jit, F8E4M3, F32);
  VerifyAllWidthsExt<float8_e3m4, float, uint32_t>(jit, F8E3M4, F32);
  VerifyAllWidthsExt<float8_e4m3fnuz, float, uint32_t>(jit, F8E4M3FNUZ, F32,
                                                       /*is_fnuz=*/true);
  VerifyAllWidthsExt<float8_e5m2fnuz, float, uint32_t>(jit, F8E5M2FNUZ, F32,
                                                       /*is_fnuz=*/true);
  VerifyAllWidthsExt<float8_e4m3b11fnuz, float, uint32_t>(jit, F8E4M3B11FNUZ,
                                                          F32,
                                                          /*is_fnuz=*/true);
  VerifyAllWidthsExt<float8_e8m0fnu, float, uint32_t>(jit, F8E8M0FNU, F32);
}

TEST(FpExtExecutionTest, ExhaustiveAllF8ToF16) {
  ASSERT_OK_AND_ASSIGN(JitRunner jit, CreateAllF8ToJitRunner(F16));
  VerifyAllWidthsExt<float8_e4m3fn, half, uint16_t>(jit, F8E4M3FN, F16);
  VerifyAllWidthsExt<float8_e5m2, half, uint16_t>(jit, F8E5M2, F16);
  VerifyAllWidthsExt<float8_e4m3, half, uint16_t>(jit, F8E4M3, F16);
  VerifyAllWidthsExt<float8_e3m4, half, uint16_t>(jit, F8E3M4, F16);
  VerifyAllWidthsExt<float8_e4m3fnuz, half, uint16_t>(jit, F8E4M3FNUZ, F16,
                                                      /*is_fnuz=*/true);
  VerifyAllWidthsExt<float8_e5m2fnuz, half, uint16_t>(jit, F8E5M2FNUZ, F16,
                                                      /*is_fnuz=*/true);
  VerifyAllWidthsExt<float8_e4m3b11fnuz, half, uint16_t>(jit, F8E4M3B11FNUZ,
                                                         F16,
                                                         /*is_fnuz=*/true);
  VerifyAllWidthsExt<float8_e8m0fnu, half, uint16_t>(jit, F8E8M0FNU, F16);
}

TEST(FpExtExecutionTest, ExhaustiveAllF8ToBf16) {
  ASSERT_OK_AND_ASSIGN(JitRunner jit, CreateAllF8ToJitRunner(BF16));
  VerifyAllWidthsExt<float8_e4m3fn, bfloat16, uint16_t>(jit, F8E4M3FN, BF16);
  VerifyAllWidthsExt<float8_e5m2, bfloat16, uint16_t>(jit, F8E5M2, BF16);
  VerifyAllWidthsExt<float8_e4m3, bfloat16, uint16_t>(jit, F8E4M3, BF16);
  VerifyAllWidthsExt<float8_e3m4, bfloat16, uint16_t>(jit, F8E3M4, BF16);
  VerifyAllWidthsExt<float8_e4m3fnuz, bfloat16, uint16_t>(jit, F8E4M3FNUZ, BF16,
                                                          /*is_fnuz=*/true);
  VerifyAllWidthsExt<float8_e5m2fnuz, bfloat16, uint16_t>(jit, F8E5M2FNUZ, BF16,
                                                          /*is_fnuz=*/true);
  VerifyAllWidthsExt<float8_e4m3b11fnuz, bfloat16, uint16_t>(
      jit, F8E4M3B11FNUZ, BF16, /*is_fnuz=*/true);
  VerifyAllWidthsExt<float8_e8m0fnu, bfloat16, uint16_t>(jit, F8E8M0FNU, BF16);
}

TEST(FpExtExecutionTest, ExhaustiveAllF8ToF64) {
  ASSERT_OK_AND_ASSIGN(JitRunner jit, CreateAllF8ToJitRunner(F64));
  VerifyAllWidthsExt<float8_e4m3fn, double, uint64_t>(jit, F8E4M3FN, F64);
  VerifyAllWidthsExt<float8_e5m2, double, uint64_t>(jit, F8E5M2, F64);
  VerifyAllWidthsExt<float8_e4m3, double, uint64_t>(jit, F8E4M3, F64);
  VerifyAllWidthsExt<float8_e3m4, double, uint64_t>(jit, F8E3M4, F64);
  VerifyAllWidthsExt<float8_e4m3fnuz, double, uint64_t>(jit, F8E4M3FNUZ, F64,
                                                        /*is_fnuz=*/true);
  VerifyAllWidthsExt<float8_e5m2fnuz, double, uint64_t>(jit, F8E5M2FNUZ, F64,
                                                        /*is_fnuz=*/true);
  VerifyAllWidthsExt<float8_e4m3b11fnuz, double, uint64_t>(jit, F8E4M3B11FNUZ,
                                                           F64,
                                                           /*is_fnuz=*/true);
  VerifyAllWidthsExt<float8_e8m0fnu, double, uint64_t>(jit, F8E8M0FNU, F64);
}

}  // namespace
}  // namespace xla::codegen::intrinsics
