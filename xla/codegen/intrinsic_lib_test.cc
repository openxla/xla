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

#include "xla/codegen/intrinsic_lib.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/codegen/intrinsic/intrinsic_compiler_lib.h"
#include "xla/codegen/intrinsic/simple_jit_runner.h"

namespace xla::codegen::intrinsics {
namespace {

using ::testing::Contains;
using ::testing::IsSupersetOf;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;

std::string ToString(const llvm::VecDesc& vec_desc) {
  return absl::StrJoin(
      {vec_desc.getScalarFnName().str(), vec_desc.getVectorFnName().str(),
       absl::StrCat(vec_desc.getVectorizationFactor().getKnownMinValue()),
       vec_desc.getVABIPrefix()},
      ":");
}

TEST(IntrinsicLibTest, ExpVectorizations) {
  if (!AreCppGenIntrinsicsAvailable()) {
    GTEST_SKIP();
  }
  IntrinsicOptions options;
  auto lib = IntrinsicFunctionLib(options);
  std::vector<llvm::VecDesc> vec_descs = lib.Vectorizations();
  std::vector<std::string> vec_descs_str;
  for (const auto& vec_desc : vec_descs) {
    if (vec_desc.getScalarFnName().starts_with("xla.exp.")) {
      vec_descs_str.push_back(ToString(vec_desc));
    }
  }

  EXPECT_THAT(vec_descs_str, UnorderedElementsAre(
                                 "xla.exp.f32:xla.exp.v2f32:2:_ZGV_LLVM_N2v",
                                 "xla.exp.f32:xla.exp.v4f32:4:_ZGV_LLVM_N4v",
                                 "xla.exp.f32:xla.exp.v8f32:8:_ZGV_LLVM_N8v",
                                 "xla.exp.f32:xla.exp.v16f32:16:_ZGV_LLVM_N16v",
                                 "xla.exp.f64:xla.exp.v2f64:2:_ZGV_LLVM_N2v",
                                 "xla.exp.f64:xla.exp.v4f64:4:_ZGV_LLVM_N4v",
                                 "xla.exp.f64:xla.exp.v8f64:8:_ZGV_LLVM_N8v"));
}

TEST(IntrinsicLibTest, AtanVectorizations) {
  IntrinsicOptions options;
  auto lib = IntrinsicFunctionLib(options);
  std::vector<llvm::VecDesc> vec_descs = lib.Vectorizations();
  std::vector<std::string> vec_descs_str;
  for (const auto& vec_desc : vec_descs) {
    if (vec_desc.getScalarFnName().starts_with("xla.atan")) {
      vec_descs_str.push_back(ToString(vec_desc));
    }
  }

  EXPECT_THAT(
      vec_descs_str,
      UnorderedElementsAre("xla.atan.f32:xla.atan.v4f32:4:_ZGV_LLVM_N4v",
                           "xla.atan.f32:xla.atan.v8f32:8:_ZGV_LLVM_N8v",
                           "xla.atan.f32:xla.atan.v16f32:16:_ZGV_LLVM_N16v",
                           "xla.atan.f64:xla.atan.v2f64:2:_ZGV_LLVM_N2v",
                           "xla.atan.f64:xla.atan.v4f64:4:_ZGV_LLVM_N4v",
                           "xla.atan.f64:xla.atan.v8f64:8:_ZGV_LLVM_N8v"));
}

TEST(IntrinsicLibTest, AtanVectorizationsOnNeon) {
  IntrinsicOptions options;
  options.features = "+neon,+fp-armv8";
  auto lib = IntrinsicFunctionLib(options);
  std::vector<std::string> vec_descs_str;
  for (const auto& vec_desc : lib.Vectorizations()) {
    if (vec_desc.getScalarFnName().starts_with("xla.atan")) {
      vec_descs_str.push_back(ToString(vec_desc));
    }
  }

  EXPECT_THAT(vec_descs_str,
              IsSupersetOf({"xla.atan.f32:xla.atan.v4f32:4:_ZGV_LLVM_N4v",
                            "xla.atan.f64:xla.atan.v2f64:2:_ZGV_LLVM_N2v"}));
}

TEST(IntrinsicLibTest, DefinesWideAtan) {
  if (!AreCppGenIntrinsicsAvailable()) {
    GTEST_SKIP();
  }
  constexpr absl::string_view kKernel = R"(
    declare <8 x float> @xla.atan.v8f32(<8 x float>)
    define <8 x float> @kernel(<8 x float> %x) {
      %r = call <8 x float> @xla.atan.v8f32(<8 x float> %x)
      ret <8 x float> %r
    }
  )";
  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(*context, std::string(kKernel));
  IntrinsicFunctionLib lib((IntrinsicOptions()));
  lib.Vectorizations();
  EXPECT_THAT(lib.DefineIntrinsicFunctions(*module),
              UnorderedElementsAre("xla.atan.v8f32"));

  intrinsic::JitRunner jit(std::move(module), std::move(context));
  auto atan = jit.GetVectorizedFn<8, float, float>("kernel");
  std::array<float, 8> x = {0.0f, 0.5f, -1.5f, 3.0f, 0.4f, 2.0f, -4.0f, 5.5f};
  std::array<float, 8> y = atan(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_NEAR(y[i], std::atan(x[i]), 1e-6f) << "lane " << i;
  }
}

TEST(IntrinsicLibTest, AtanHasNoLibmCallsAfterInlining) {
  if (!AreCppGenIntrinsicsAvailable()) {
    GTEST_SKIP();
  }
  std::string kernels;
  std::vector<std::string> kernel_names;
  std::vector<std::string> atan_names;
  struct Width {
    absl::string_view ir_type;
    absl::string_view suffix;
  };
  for (const auto& [type, suffix] :
       {Width{"float", "f32"}, Width{"<4 x float>", "v4f32"},
        Width{"<8 x float>", "v8f32"}, Width{"<16 x float>", "v16f32"},
        Width{"double", "f64"}, Width{"<2 x double>", "v2f64"},
        Width{"<4 x double>", "v4f64"}, Width{"<8 x double>", "v8f64"}}) {
    std::string atan = absl::StrCat("xla.atan.", suffix);
    std::string kernel = absl::StrCat("kernel_", suffix);
    absl::StrAppend(&kernels, absl::Substitute(R"(
      declare $0 @$1($0)
      define $0 @$2($0 %x) {
        %r = call $0 @$1($0 %x)
        ret $0 %r
      }
    )",
                                               type, atan, kernel));
    atan_names.push_back(atan);
    kernel_names.push_back(kernel);
  }

  struct Config {
    absl::string_view features;
    int prefer_vector_width;
  };
  for (const Config& cfg :
       {Config{"+avx,+avx2", 256}, Config{"+avx,+avx512f", 512}}) {
    SCOPED_TRACE(cfg.features);
    IntrinsicOptions options;
    options.features = std::string(cfg.features);
    options.prefer_vector_width = cfg.prefer_vector_width;

    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module =
        ParseEmbeddedBitcode(context, kernels);
    IntrinsicFunctionLib lib(options);
    lib.Vectorizations();
    EXPECT_THAT(lib.DefineIntrinsicFunctions(*module),
                UnorderedElementsAreArray(atan_names));
    intrinsic::RunInlineAndOptPasses(*module);
    EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));

    for (const llvm::Function& f : *module) {
      if (f.isDeclaration()) {
        EXPECT_TRUE(f.isIntrinsic()) << f.getName().str();
      } else {
        EXPECT_THAT(kernel_names, Contains(f.getName().str()));
      }
    }
  }
}

TEST(IntrinsicLibTest, Expm1Vectorizations) {
  for (absl::string_view features : {"", "+avx512f"}) {
    IntrinsicOptions options;
    options.features = features;
    auto lib = IntrinsicFunctionLib(options);
    std::vector<std::string> vec_descs_str;
    for (const auto& vec_desc : lib.Vectorizations()) {
      if (vec_desc.getScalarFnName().starts_with("xla.expm1")) {
        vec_descs_str.push_back(ToString(vec_desc));
      }
    }

    EXPECT_THAT(
        vec_descs_str,
        UnorderedElementsAre("xla.expm1.f32:xla.expm1.v2f32:2:_ZGV_LLVM_N2v",
                             "xla.expm1.f32:xla.expm1.v4f32:4:_ZGV_LLVM_N4v",
                             "xla.expm1.f32:xla.expm1.v8f32:8:_ZGV_LLVM_N8v",
                             "xla.expm1.f32:xla.expm1.v16f32:16:_ZGV_LLVM_N16v",
                             "xla.expm1.f64:xla.expm1.v2f64:2:_ZGV_LLVM_N2v",
                             "xla.expm1.f64:xla.expm1.v4f64:4:_ZGV_LLVM_N4v",
                             "xla.expm1.f64:xla.expm1.v8f64:8:_ZGV_LLVM_N8v"))
        << "features: " << features;
  }
}

// Returns a JIT of a function `kernel` that calls the intrinsic `name` on a
// value of the LLVM type `type`.
std::unique_ptr<intrinsic::JitRunner> JitCalling(const std::string& name,
                                                 const std::string& type) {
  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(*context, absl::Substitute(R"(
        declare $0 @$1($0)
        define $0 @kernel($0 %x) {
          %r = call $0 @$1($0 %x)
          ret $0 %r
        }
      )",
                                                      type, name));
  IntrinsicFunctionLib lib((IntrinsicOptions()));
  lib.Vectorizations();
  EXPECT_THAT(lib.DefineIntrinsicFunctions(*module),
              UnorderedElementsAre(name));
  return std::make_unique<intrinsic::JitRunner>(std::move(module),
                                                std::move(context));
}

// Expects the N-lane definition of `op` to give the bits of the scalar one.
template <size_t N, typename T, size_t Size>
void ExpectWideDefinitionMatchesScalar(const std::string& op,
                                       const std::array<T, Size>& x) {
  const std::string type = sizeof(T) == 4 ? "float" : "double";
  const std::string suffix = absl::StrCat("f", 8 * sizeof(T));
  auto scalar_jit = JitCalling(absl::StrCat("xla.", op, ".", suffix), type);
  auto scalar = scalar_jit->template GetScalarFn<T(T)>("kernel");
  auto wide_jit = JitCalling(absl::StrCat("xla.", op, ".v", N, suffix),
                             absl::Substitute("<$0 x $1>", N, type));
  auto wide = wide_jit->template GetVectorizedFn<N, T, T>("kernel");
  for (size_t i = 0; i < Size; i += N) {
    std::array<T, N> chunk;
    std::copy_n(x.begin() + i, N, chunk.begin());
    const std::array<T, N> y = wide(chunk);
    for (size_t j = 0; j < N; ++j) {
      const T expected = scalar(chunk[j]);
      EXPECT_EQ(std::memcmp(&y[j], &expected, sizeof(T)), 0)
          << op << " with " << N << " lanes: x = " << chunk[j]
          << ", wide = " << y[j] << ", scalar = " << expected;
    }
  }
}

// Runs the definitions of `op` at every width, also the ones that the
// vectorizer does not pick on the machine that runs the tests.
void ExpectWideDefinitionsMatchScalar(const std::string& op) {
  constexpr float kInfF = std::numeric_limits<float>::infinity();
  constexpr double kInfD = std::numeric_limits<double>::infinity();
  // Zeros, infinities, NaN, overflow, underflow and ordinary values.
  const std::array<float, 16> x32 = {
      0.0f,  -0.0f,   kInfF,   -kInfF, std::numeric_limits<float>::quiet_NaN(),
      89.0f, -104.0f, 88.7f,   -87.0f, 1e-6f,
      0.35f, -0.4f,   0.6931f, 5.2f,   17.1f,
      -1.0f};
  const std::array<double, 8> x64 = {
      0.0,   -0.0,   kInfD, -kInfD, std::numeric_limits<double>::quiet_NaN(),
      710.0, -746.0, 0.35};
  ExpectWideDefinitionMatchesScalar<2>(op, x32);
  ExpectWideDefinitionMatchesScalar<4>(op, x32);
  ExpectWideDefinitionMatchesScalar<8>(op, x32);
  ExpectWideDefinitionMatchesScalar<16>(op, x32);
  ExpectWideDefinitionMatchesScalar<2>(op, x64);
  ExpectWideDefinitionMatchesScalar<4>(op, x64);
  ExpectWideDefinitionMatchesScalar<8>(op, x64);
}

TEST(IntrinsicLibTest, WideExpDefinitionsMatchScalar) {
  if (!AreCppGenIntrinsicsAvailable()) {
    GTEST_SKIP();
  }
  ExpectWideDefinitionsMatchScalar("exp");
}

TEST(IntrinsicLibTest, WideExpm1DefinitionsMatchScalar) {
  if (!AreCppGenIntrinsicsAvailable()) {
    GTEST_SKIP();
  }
  ExpectWideDefinitionsMatchScalar("expm1");
}

TEST(IntrinsicLibTest, SinAndCosVectorizations) {
  auto descs_with_prefix = [](absl::string_view features,
                              absl::string_view prefix) {
    IntrinsicOptions options;
    options.features = features;
    auto lib = IntrinsicFunctionLib(options);
    std::vector<std::string> descs;
    for (const auto& vec_desc : lib.Vectorizations()) {
      if (vec_desc.getScalarFnName().starts_with(prefix)) {
        descs.push_back(ToString(vec_desc));
      }
    }
    return descs;
  };

  EXPECT_THAT(
      descs_with_prefix("", "xla.sin."),
      UnorderedElementsAre("xla.sin.f32:xla.sin.v4f32:4:_ZGV_LLVM_N4v",
                           "xla.sin.f32:xla.sin.v8f32:8:_ZGV_LLVM_N8v",
                           "xla.sin.f64:xla.sin.v2f64:2:_ZGV_LLVM_N2v",
                           "xla.sin.f64:xla.sin.v4f64:4:_ZGV_LLVM_N4v"));
  EXPECT_THAT(
      descs_with_prefix("", "xla.cos."),
      UnorderedElementsAre("xla.cos.f32:xla.cos.v4f32:4:_ZGV_LLVM_N4v",
                           "xla.cos.f32:xla.cos.v8f32:8:_ZGV_LLVM_N8v",
                           "xla.cos.f64:xla.cos.v2f64:2:_ZGV_LLVM_N2v",
                           "xla.cos.f64:xla.cos.v4f64:4:_ZGV_LLVM_N4v"));

  EXPECT_THAT(
      descs_with_prefix("+avx512f", "xla.sin."),
      UnorderedElementsAre("xla.sin.f32:xla.sin.v4f32:4:_ZGV_LLVM_N4v",
                           "xla.sin.f32:xla.sin.v8f32:8:_ZGV_LLVM_N8v",
                           "xla.sin.f32:xla.sin.v16f32:16:_ZGV_LLVM_N16v",
                           "xla.sin.f64:xla.sin.v2f64:2:_ZGV_LLVM_N2v",
                           "xla.sin.f64:xla.sin.v4f64:4:_ZGV_LLVM_N4v",
                           "xla.sin.f64:xla.sin.v8f64:8:_ZGV_LLVM_N8v"));
  EXPECT_THAT(
      descs_with_prefix("+avx512f", "xla.cos."),
      UnorderedElementsAre("xla.cos.f32:xla.cos.v4f32:4:_ZGV_LLVM_N4v",
                           "xla.cos.f32:xla.cos.v8f32:8:_ZGV_LLVM_N8v",
                           "xla.cos.f32:xla.cos.v16f32:16:_ZGV_LLVM_N16v",
                           "xla.cos.f64:xla.cos.v2f64:2:_ZGV_LLVM_N2v",
                           "xla.cos.f64:xla.cos.v4f64:4:_ZGV_LLVM_N4v",
                           "xla.cos.f64:xla.cos.v8f64:8:_ZGV_LLVM_N8v"));
}

TEST(IntrinsicLibTest, SinhVectorizations) {
  IntrinsicOptions options;
  auto lib = IntrinsicFunctionLib(options);
  std::vector<llvm::VecDesc> vec_descs = lib.Vectorizations();
  std::vector<std::string> vec_descs_str;
  for (const auto& vec_desc : vec_descs) {
    if (vec_desc.getScalarFnName().starts_with("xla.sinh")) {
      vec_descs_str.push_back(ToString(vec_desc));
    }
  }

  EXPECT_THAT(
      vec_descs_str,
      UnorderedElementsAre("xla.sinh.f32:xla.sinh.v4f32:4:_ZGV_LLVM_N4v",
                           "xla.sinh.f32:xla.sinh.v8f32:8:_ZGV_LLVM_N8v",
                           "xla.sinh.f32:xla.sinh.v16f32:16:_ZGV_LLVM_N16v"));
}

TEST(IntrinsicLibTest, CoshVectorizations) {
  IntrinsicOptions options;
  auto lib = IntrinsicFunctionLib(options);
  std::vector<llvm::VecDesc> vec_descs = lib.Vectorizations();
  std::vector<std::string> vec_descs_str;
  for (const auto& vec_desc : vec_descs) {
    if (vec_desc.getScalarFnName().starts_with("xla.cosh")) {
      vec_descs_str.push_back(ToString(vec_desc));
    }
  }

  EXPECT_THAT(
      vec_descs_str,
      UnorderedElementsAre("xla.cosh.f32:xla.cosh.v4f32:4:_ZGV_LLVM_N4v",
                           "xla.cosh.f32:xla.cosh.v8f32:8:_ZGV_LLVM_N8v",
                           "xla.cosh.f32:xla.cosh.v16f32:16:_ZGV_LLVM_N16v"));
}

TEST(IntrinsicLibTest, AsinAndAcosVectorizations) {
  IntrinsicOptions options;
  auto lib = IntrinsicFunctionLib(options);
  std::vector<llvm::VecDesc> vec_descs = lib.Vectorizations();
  std::vector<std::string> asin_descs;
  std::vector<std::string> acos_descs;
  for (const auto& vec_desc : vec_descs) {
    if (vec_desc.getScalarFnName().starts_with("xla.asin")) {
      asin_descs.push_back(ToString(vec_desc));
    } else if (vec_desc.getScalarFnName().starts_with("xla.acos")) {
      acos_descs.push_back(ToString(vec_desc));
    }
  }

  EXPECT_THAT(
      asin_descs,
      UnorderedElementsAre("xla.asin.f32:xla.asin.v4f32:4:_ZGV_LLVM_N4v",
                           "xla.asin.f32:xla.asin.v8f32:8:_ZGV_LLVM_N8v",
                           "xla.asin.f32:xla.asin.v16f32:16:_ZGV_LLVM_N16v"));
  EXPECT_THAT(
      acos_descs,
      UnorderedElementsAre("xla.acos.f32:xla.acos.v4f32:4:_ZGV_LLVM_N4v",
                           "xla.acos.f32:xla.acos.v8f32:8:_ZGV_LLVM_N8v",
                           "xla.acos.f32:xla.acos.v16f32:16:_ZGV_LLVM_N16v"));
}
TEST(IntrinsicLibTest, CppGenIntrinsicLibraryPreservesNoInline) {
  llvm::LLVMContext context;
  llvm::Module dst_module("dst_module", context);

  std::string gcov_ir = R"(
    define void @__llvm_gcov_init() noinline {
      ret void
    }
  )";

  CppGenIntrinsicLibrary lib(gcov_ir, "gcov_test");
  lib.LinkIntoModule(dst_module);

  // A linked module containing noinline functions must remain valid after
  // LinkIntoModule and not have alwaysinline attached to them.
  EXPECT_FALSE(llvm::verifyModule(dst_module, &llvm::errs()));
}

}  // namespace

}  // namespace xla::codegen::intrinsics
