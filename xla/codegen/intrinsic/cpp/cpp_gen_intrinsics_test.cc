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

#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_16_ll.h"
#include "xla/codegen/intrinsic/simple_jit_runner.h"

namespace xla::codegen {
namespace {

using ::xla::codegen::intrinsic::JitRunner;

llvm::FunctionType* UnaryType(llvm::Type* type) {
  return llvm::FunctionType::get(type, {type}, /*isVarArg=*/false);
}

llvm::Type* VecF32(llvm::LLVMContext& context, int width) {
  return llvm::VectorType::get(llvm::Type::getFloatTy(context),
                               llvm::ElementCount::getFixed(width));
}

void StripHostAttrs(llvm::Module& module) {
  for (llvm::Function& f : module) {
    f.removeFnAttr("probe-stack");
    f.removeFnAttr("target-cpu");
    f.removeFnAttr("target-features");
  }
}

TEST(CppGenIntrinsicsTest, AtanV8F32HasDirectSignature) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary16LlIr);
  llvm::FunctionType* type = UnaryType(VecF32(context, 8));

  llvm::Function* fn = GetCppGenFunction(module.get(), "xla.atan.v8f32", type);
  ASSERT_NE(fn, nullptr);
  EXPECT_EQ(fn->getFunctionType(), type);
  EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
}

TEST(CppGenIntrinsicsTest, AtanV8F32IsIdempotent) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary16LlIr);
  llvm::FunctionType* type = UnaryType(VecF32(context, 8));

  llvm::Function* first =
      GetCppGenFunction(module.get(), "xla.atan.v8f32", type);
  llvm::Function* second =
      GetCppGenFunction(module.get(), "xla.atan.v8f32", type);
  EXPECT_EQ(first, second);
  EXPECT_EQ(second->getFunctionType(), type);
}

TEST(CppGenIntrinsicsTest, DirectSignaturesAreUntouched) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary16LlIr);

  llvm::Function* scalar = GetCppGenFunction(
      module.get(), "xla.atan.f32", UnaryType(llvm::Type::getFloatTy(context)));
  EXPECT_EQ(scalar, FindCppGenFunction(*module, "xla.atan.f32"));
  EXPECT_EQ(FindCppGenFunction(*module, "xla.atan.f32.body"), nullptr);

  llvm::Function* v4 = GetCppGenFunction(module.get(), "xla.atan.v4f32",
                                         UnaryType(VecF32(context, 4)));
  EXPECT_EQ(v4, FindCppGenFunction(*module, "xla.atan.v4f32"));
  EXPECT_EQ(FindCppGenFunction(*module, "xla.atan.v4f32.body"), nullptr);
}

TEST(CppGenIntrinsicsTest, AtanV8F32ComputesThroughJit) {
  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(*context, llvm_ir::kEigenUnary16LlIr);
  StripHostAttrs(*module);

  llvm::Function* fn = GetCppGenFunction(module.get(), "xla.atan.v8f32",
                                         UnaryType(VecF32(*context, 8)));
  fn->setLinkage(llvm::Function::ExternalLinkage);
  std::string name = fn->getName().str();

  JitRunner jit(std::move(module), std::move(context));
  auto atan = jit.GetVectorizedFn<8, float, float>(name);

  std::array<float, 8> x = {0.0f, 0.5f, -1.5f, 3.0f, 0.4f, 2.0f, -4.0f, 5.5f};
  std::array<float, 8> y = atan(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_NEAR(y[i], std::atan(x[i]), 1e-6f) << "lane " << i;
  }
}

// x86-64 without AVX passes 256-bit vectors as byval pointers.
TEST(CppGenIntrinsicsTest, ByvalArgumentIsAdapted) {
  constexpr absl::string_view kIr = R"(
    define <8 x float> @xla.neg.v8f32(ptr byval(<8 x float>) align 16 %p) {
      %x = load <8 x float>, ptr %p, align 16
      %r = fneg <8 x float> %x
      ret <8 x float> %r
    }
  )";
  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(*context, std::string(kIr));
  llvm::FunctionType* type = UnaryType(VecF32(*context, 8));

  llvm::Function* fn = GetCppGenFunction(module.get(), "xla.neg.v8f32", type);
  EXPECT_EQ(fn->getFunctionType(), type);
  EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
  fn->setLinkage(llvm::Function::ExternalLinkage);

  JitRunner jit(std::move(module), std::move(context));
  auto neg = jit.GetVectorizedFn<8, float, float>("xla.neg.v8f32");
  std::array<float, 8> x = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
  std::array<float, 8> y = neg(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(y[i], -x[i]) << "lane " << i;
  }
}

// AArch64 returns vectors wider than 128 bits through sret and passes them
// through an unannotated pointer.
TEST(CppGenIntrinsicsTest, SretReturnIsAdapted) {
  constexpr absl::string_view kIr = R"(
    define void @xla.neg.v8f32(ptr sret(<8 x float>) align 16 %out, ptr %p) {
      %x = load <8 x float>, ptr %p, align 16
      %r = fneg <8 x float> %x
      store <8 x float> %r, ptr %out, align 16
      ret void
    }
  )";
  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(*context, std::string(kIr));
  llvm::FunctionType* type = UnaryType(VecF32(*context, 8));

  llvm::Function* fn = GetCppGenFunction(module.get(), "xla.neg.v8f32", type);
  EXPECT_EQ(fn->getFunctionType(), type);
  EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
  fn->setLinkage(llvm::Function::ExternalLinkage);

  JitRunner jit(std::move(module), std::move(context));
  auto neg = jit.GetVectorizedFn<8, float, float>("xla.neg.v8f32");
  std::array<float, 8> x = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
  std::array<float, 8> y = neg(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(y[i], -x[i]) << "lane " << i;
  }
}

TEST(CppGenIntrinsicsTest, LinkAdaptsMismatchedDeclaration) {
  constexpr absl::string_view kLib = R"(
    define void @xla.neg.v8f32(ptr sret(<8 x float>) align 16 %out, ptr %p) {
      %x = load <8 x float>, ptr %p, align 16
      %r = fneg <8 x float> %x
      store <8 x float> %r, ptr %out, align 16
      ret void
    }
  )";
  constexpr absl::string_view kKernel = R"(
    declare <8 x float> @xla.neg.v8f32(<8 x float>)
    define <8 x float> @kernel(<8 x float> %x) {
      %r = call <8 x float> @xla.neg.v8f32(<8 x float> %x)
      ret <8 x float> %r
    }
  )";
  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(*context, std::string(kKernel));
  CppGenIntrinsicLibrary(std::string(kLib), "lib").LinkIntoModule(*module);
  EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));

  llvm::Function* kernel = module->getFunction("kernel");
  auto* call = llvm::cast<llvm::CallInst>(&kernel->getEntryBlock().front());
  llvm::Function* callee = call->getCalledFunction();
  ASSERT_NE(callee, nullptr);
  EXPECT_FALSE(callee->isDeclaration());
  EXPECT_EQ(callee->getName(), "xla.neg.v8f32");
  EXPECT_EQ(module->getFunction("xla.neg.v8f32.old_decl"), nullptr);

  JitRunner jit(std::move(module), std::move(context));
  auto neg = jit.GetVectorizedFn<8, float, float>("kernel");
  std::array<float, 8> x = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
  std::array<float, 8> y = neg(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(y[i], -x[i]) << "lane " << i;
  }
}

}  // namespace
}  // namespace xla::codegen
