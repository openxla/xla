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

#include <cmath>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "xla/codegen/intrinsic/accuracy/accuracy_budget.h"
#include "xla/codegen/intrinsic/accuracy/accuracy_test_framework.h"
#include "xla/codegen/intrinsic/accuracy/golden_baselines.h"
#include "xla/codegen/intrinsic/exp.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/codegen/intrinsic/log1p.h"
#include "xla/codegen/intrinsic/rsqrt.h"
#include "xla/codegen/intrinsic/simple_jit_runner.h"
#include "xla/codegen/intrinsic/tanh.h"
#include "xla/codegen/intrinsic/type.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsic::accuracy {
namespace {

using ::xla::codegen::intrinsic::JitRunner;
using ::xla::codegen::intrinsics::Exp;
using ::xla::codegen::intrinsics::Log1p;
using ::xla::codegen::intrinsics::Rsqrt;
using ::xla::codegen::intrinsics::Tanh;
using ::xla::codegen::intrinsics::Type;

// Shortcuts for types
constexpr PrimitiveType kF32 = xla::F32;
constexpr PrimitiveType kF64 = xla::F64;

template <typename IntrinsicT>
JitRunner CreateRunner(Type type) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("accuracy_test", *context);
  llvm::Function* fn;
  if constexpr (std::is_same_v<IntrinsicT, Rsqrt>) {
    xla::codegen::intrinsics::IntrinsicOptions options;
    fn = IntrinsicT::CreateDefinition(module.get(), options, type).value();
  } else {
    fn = IntrinsicT::CreateDefinition(module.get(), type).value();
  }
  fn->setLinkage(llvm::Function::ExternalLinkage);
  EXPECT_FALSE(llvm::verifyFunction(*fn));
  return JitRunner(std::move(module), std::move(context));
}

// ------ Tanh ------

TEST(IntrinsicAccuracy, TanhF32) {
  auto runner = CreateRunner<Tanh>(Type::S(kF32));
  auto* fn = runner.GetScalarFn<float(float)>(Tanh::Name(Type::S(kF32)));
  ASSERT_TRUE(fn != nullptr);

  auto report = RunAccuracyTest<float>(
      ToVector(kGoldenTanh), std::function<float(float)>(fn), "TanhF32");
  LogReport<float>(report, "TanhF32");
  AssertWithinBudget<float>(report, kTanhF32MaxUlp);
}

TEST(LibmBaseline, TanhF32) {
  auto report = RunAccuracyTest<float>(
      ToVector(kGoldenTanh),
      std::function<float(float)>([](float x) { return std::tanh(x); }),
      "std::tanh(f32)");
  LogReport<float>(report, "std::tanh(f32)");
}

TEST(IntrinsicAccuracy, TanhF64) {
  auto runner = CreateRunner<Tanh>(Type::S(kF64));
  auto* fn = runner.GetScalarFn<double(double)>(Tanh::Name(Type::S(kF64)));
  ASSERT_TRUE(fn != nullptr);

  auto report = RunAccuracyTest<double>(
      ToVector(kGoldenTanh), std::function<double(double)>(fn), "TanhF64");
  LogReport<double>(report, "TanhF64");
  AssertWithinBudget<double>(report, kTanhF64MaxUlp);
}

TEST(LibmBaseline, TanhF64) {
  auto report = RunAccuracyTest<double>(
      ToVector(kGoldenTanh),
      std::function<double(double)>([](double x) { return std::tanh(x); }),
      "std::tanh(f64)");
  LogReport<double>(report, "std::tanh(f64)");
}

// ------ Exp ------

TEST(IntrinsicAccuracy, ExpF64) {
  auto runner = CreateRunner<Exp>(Type::S(kF64));
  auto* fn = runner.GetScalarFn<double(double)>(Exp::Name(Type::S(kF64)));
  ASSERT_TRUE(fn != nullptr);

  auto report = RunAccuracyTest<double>(
      ToVector(kGoldenExp), std::function<double(double)>(fn), "ExpF64");
  LogReport<double>(report, "ExpF64");
  AssertWithinBudget<double>(report, kExpF64MaxUlp);
}

TEST(LibmBaseline, ExpF64) {
  auto report = RunAccuracyTest<double>(
      ToVector(kGoldenExp),
      std::function<double(double)>([](double x) { return std::exp(x); }),
      "std::exp(f64)");
  LogReport<double>(report, "std::exp(f64)");
}

// ------ Log1p ------

TEST(IntrinsicAccuracy, Log1pF32) {
  auto runner = CreateRunner<Log1p>(Type::S(kF32));
  auto* fn = runner.GetScalarFn<float(float)>(Log1p::Name(Type::S(kF32)));
  ASSERT_TRUE(fn != nullptr);

  auto report = RunAccuracyTest<float>(
      ToVector(kGoldenLog1p), std::function<float(float)>(fn), "Log1pF32");
  LogReport<float>(report, "Log1pF32");
  AssertWithinBudget<float>(report, kLog1pF32MaxUlp);
}

TEST(LibmBaseline, Log1pF32) {
  auto report = RunAccuracyTest<float>(
      ToVector(kGoldenLog1p),
      std::function<float(float)>([](float x) { return std::log1p(x); }),
      "std::log1p(f32)");
  LogReport<float>(report, "std::log1p(f32)");
}

TEST(IntrinsicAccuracy, Log1pF64) {
  auto runner = CreateRunner<Log1p>(Type::S(kF64));
  auto* fn = runner.GetScalarFn<double(double)>(Log1p::Name(Type::S(kF64)));
  ASSERT_TRUE(fn != nullptr);

  auto report = RunAccuracyTest<double>(
      ToVector(kGoldenLog1p), std::function<double(double)>(fn), "Log1pF64");
  LogReport<double>(report, "Log1pF64");
  AssertWithinBudget<double>(report, kLog1pF64MaxUlp);
}

TEST(LibmBaseline, Log1pF64) {
  auto report = RunAccuracyTest<double>(
      ToVector(kGoldenLog1p),
      std::function<double(double)>([](double x) { return std::log1p(x); }),
      "std::log1p(f64)");
  LogReport<double>(report, "std::log1p(f64)");
}

// ------ Rsqrt ------

TEST(IntrinsicAccuracy, RsqrtF32) {
  auto runner = CreateRunner<Rsqrt>(Type::S(kF32));
  auto* fn = runner.GetScalarFn<float(float)>(Rsqrt::Name(Type::S(kF32)));
  ASSERT_TRUE(fn != nullptr);

  auto report = RunAccuracyTest<float>(
      ToVector(kGoldenRsqrt), std::function<float(float)>(fn), "RsqrtF32");
  LogReport<float>(report, "RsqrtF32");
  AssertWithinBudget<float>(report, kRsqrtF32MaxUlp);
}

TEST(LibmBaseline, RsqrtF32) {
  auto report = RunAccuracyTest<float>(
      ToVector(kGoldenRsqrt),
      std::function<float(float)>([](float x) { return 1.0f / std::sqrt(x); }),
      "1/sqrt(f32)");
  LogReport<float>(report, "1/sqrt(f32)");
}

TEST(IntrinsicAccuracy, RsqrtF64) {
  auto runner = CreateRunner<Rsqrt>(Type::S(kF64));
  auto* fn = runner.GetScalarFn<double(double)>(Rsqrt::Name(Type::S(kF64)));
  ASSERT_TRUE(fn != nullptr);

  auto report = RunAccuracyTest<double>(
      ToVector(kGoldenRsqrt), std::function<double(double)>(fn), "RsqrtF64");
  LogReport<double>(report, "RsqrtF64");
  AssertWithinBudget<double>(report, kRsqrtF64MaxUlp);
}

TEST(LibmBaseline, RsqrtF64) {
  auto report = RunAccuracyTest<double>(
      ToVector(kGoldenRsqrt), std::function<double(double)>([](double x) {
        return 1.0 / std::sqrt(x);
      }),
      "1/sqrt(f64)");
  LogReport<double>(report, "1/sqrt(f64)");
}

}  // namespace
}  // namespace xla::codegen::intrinsic::accuracy
