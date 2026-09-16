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

#include "xla/backends/cpu/codegen/polynomial_approximations.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/util.h"

namespace xla::cpu {
namespace {

absl::StatusOr<std::unique_ptr<llvm::Module>> ParseModule(
    llvm::LLVMContext& context, absl::string_view ir, absl::string_view name) {
  llvm::SMDiagnostic diagnostic;
  llvm::MemoryBufferRef ir_buffer(ir, name);
  std::unique_ptr<llvm::Module> m =
      llvm::parseAssembly(ir_buffer, diagnostic, context);
  if (m == nullptr) {
    return Internal("Failed to parse LLVM IR: %s",
                    diagnostic.getMessage().str());
  }
  return m;
}

// The rewritten bodies below are anchored on the range reduction constant that
// is unique to each approximation: 1/log(2) (f0x3FB8AA3B) for exp and sqrt(1/2)
// (f0x3F3504F3) for log. Matching those proves that the Cephes polynomial was
// actually inlined, and not merely that the original call disappeared.
class PolynomialApproximationsTest : public ::testing::Test {
 protected:
  // Rewrites `ir` to polynomial approximations, then checks that `callee` is
  // erased from the module and that the resulting IR matches `pattern`.
  void RewriteAndCheck(absl::string_view ir, absl::string_view callee,
                       absl::string_view pattern) {
    llvm::LLVMContext context;
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<llvm::Module> module,
                         ParseModule(context, ir, "test_module"));

    llvm::FastMathFlags fast_math_flags;
    fast_math_flags.setFast();
    RewriteToPolynomialApproximations(module.get(), fast_math_flags);

    std::string error;
    llvm::raw_string_ostream error_stream(error);
    ASSERT_FALSE(llvm::verifyModule(*module, &error_stream)) << error;

    // The rewritten function must be erased from the module, so that no
    // declaration is left behind for the linker to resolve.
    EXPECT_EQ(module->getFunction(llvm::StringRef(callee)), nullptr);

    ASSERT_OK_AND_ASSIGN(
        bool filecheck_matched,
        RunFileCheck(llvm_ir::DumpToString(module.get()), pattern));
    EXPECT_TRUE(filecheck_matched);
  }
};

TEST_F(PolynomialApproximationsTest, VectorizationIncludesWidth32) {
  std::vector<llvm::VecDesc> vec_descs =
      PolynomialApproximationsVectorization();

  auto has_vec_desc = [&](absl::string_view scalar_fn,
                          absl::string_view vector_fn, unsigned width) {
    for (const llvm::VecDesc& desc : vec_descs) {
      if (desc.getScalarFnName() == llvm::StringRef(scalar_fn) &&
          desc.getVectorFnName() == llvm::StringRef(vector_fn) &&
          desc.getVectorizationFactor().isFixed() &&
          desc.getVectorizationFactor().getFixedValue() == width) {
        return true;
      }
    }
    return false;
  };

  // Exp vectorization mappings for 32 elements.
  EXPECT_TRUE(has_vec_desc("expf", "__xla_cpu_ExpV32F32", 32));
  EXPECT_TRUE(has_vec_desc("llvm.exp.f32", "__xla_cpu_ExpV32F32", 32));
  EXPECT_TRUE(has_vec_desc("expf", "__xla_cpu_ExpV32F16", 32));
  EXPECT_TRUE(has_vec_desc("llvm.exp.f16", "__xla_cpu_ExpV32F16", 32));

  // Log vectorization mappings for 32 elements.
  EXPECT_TRUE(has_vec_desc("logf", "__xla_cpu_LogV32F32", 32));
  EXPECT_TRUE(has_vec_desc("llvm.log.f32", "__xla_cpu_LogV32F32", 32));
  EXPECT_TRUE(has_vec_desc("logf", "__xla_cpu_LogV32F16", 32));
  EXPECT_TRUE(has_vec_desc("llvm.log.f16", "__xla_cpu_LogV32F16", 32));
}

TEST_F(PolynomialApproximationsTest, RewritesLlvmExpV32F32) {
  constexpr absl::string_view kIr = R"(
    declare <32 x float> @llvm.exp.v32f32(<32 x float>)

    define <32 x float> @exp_v32f32(<32 x float> %x) {
      %res = call <32 x float> @llvm.exp.v32f32(<32 x float> %x)
      ret <32 x float> %res
    }
  )";
  constexpr absl::string_view kExpected = R"(
    CHECK-LABEL: define <32 x float> @exp_v32f32
    CHECK-NOT:     @llvm.exp
    CHECK:         fmul fast <32 x float> %{{.*}}, splat (float f0x3FB8AA3B)
    CHECK:         call fast <32 x float> @llvm.floor.v32f32
    CHECK:         ret <32 x float>
  )";
  RewriteAndCheck(kIr, "llvm.exp.v32f32", kExpected);
}

TEST_F(PolynomialApproximationsTest, RewritesXlaExpV32F32) {
  constexpr absl::string_view kIr = R"(
    declare <32 x float> @__xla_cpu_ExpV32F32(<32 x float>)

    define <32 x float> @exp_v32f32(<32 x float> %x) {
      %res = call <32 x float> @__xla_cpu_ExpV32F32(<32 x float> %x)
      ret <32 x float> %res
    }
  )";

  constexpr absl::string_view kExpected = R"(
    CHECK-LABEL: define <32 x float> @exp_v32f32
    CHECK-NOT:     @__xla_cpu_Exp
    CHECK:         fmul fast <32 x float> %{{.*}}, splat (float f0x3FB8AA3B)
    CHECK:         call fast <32 x float> @llvm.floor.v32f32
    CHECK:         ret <32 x float>
  )";
  RewriteAndCheck(kIr, "__xla_cpu_ExpV32F32", kExpected);
}

TEST_F(PolynomialApproximationsTest, RewritesXlaExpV32F16) {
  constexpr absl::string_view kIr = R"(
    declare <32 x half> @__xla_cpu_ExpV32F16(<32 x half>)

    define <32 x half> @exp_v32f16(<32 x half> %x) {
      %res = call <32 x half> @__xla_cpu_ExpV32F16(<32 x half> %x)
      ret <32 x half> %res
    }
  )";
  // F16 is approximated by upcasting to F32 and truncating the result back.
  constexpr absl::string_view kExpected = R"(
    CHECK-LABEL: define <32 x half> @exp_v32f16
    CHECK-NOT:     @__xla_cpu_Exp
    CHECK:         fpext fast <32 x half> %{{.*}} to <32 x float>
    CHECK:         fmul fast <32 x float> %{{.*}}, splat (float f0x3FB8AA3B)
    CHECK:         fptrunc fast <32 x float> %{{.*}} to <32 x half>
    CHECK:         ret <32 x half>
  )";
  RewriteAndCheck(kIr, "__xla_cpu_ExpV32F16", kExpected);
}

TEST_F(PolynomialApproximationsTest, RewritesLlvmLogV32F32) {
  constexpr absl::string_view kIr = R"(
    declare <32 x float> @llvm.log.v32f32(<32 x float>)

    define <32 x float> @log_v32f32(<32 x float> %x) {
      %res = call <32 x float> @llvm.log.v32f32(<32 x float> %x)
      ret <32 x float> %res
    }
  )";
  constexpr absl::string_view kExpected = R"(
    CHECK-LABEL: define <32 x float> @log_v32f32
    CHECK-NOT:     @llvm.log
    CHECK:         fcmp fast olt <32 x float> %{{.*}}, splat (float f0x3F3504F3)
    CHECK:         ret <32 x float>
  )";
  RewriteAndCheck(kIr, "llvm.log.v32f32", kExpected);
}

TEST_F(PolynomialApproximationsTest, RewritesXlaLogV32F32) {
  constexpr absl::string_view kIr = R"(
    declare <32 x float> @__xla_cpu_LogV32F32(<32 x float>)

    define <32 x float> @log_v32f32(<32 x float> %x) {
      %res = call <32 x float> @__xla_cpu_LogV32F32(<32 x float> %x)
      ret <32 x float> %res
    }
  )";
  constexpr absl::string_view kExpected = R"(
    CHECK-LABEL: define <32 x float> @log_v32f32
    CHECK-NOT:     @__xla_cpu_Log
    CHECK:         fcmp fast olt <32 x float> %{{.*}}, splat (float f0x3F3504F3)
    CHECK:         ret <32 x float>
  )";
  RewriteAndCheck(kIr, "__xla_cpu_LogV32F32", kExpected);
}

TEST_F(PolynomialApproximationsTest, RewritesXlaLogV32F16) {
  constexpr absl::string_view kIr = R"(
    declare <32 x half> @__xla_cpu_LogV32F16(<32 x half>)

    define <32 x half> @log_v32f16(<32 x half> %x) {
      %res = call <32 x half> @__xla_cpu_LogV32F16(<32 x half> %x)
      ret <32 x half> %res
    }
  )";
  // F16 is approximated by upcasting to F32 and truncating the result back.
  constexpr absl::string_view kExpected = R"(
    CHECK-LABEL: define <32 x half> @log_v32f16
    CHECK-NOT:     @__xla_cpu_Log
    CHECK:         fpext fast <32 x half> %{{.*}} to <32 x float>
    CHECK:         fcmp fast olt <32 x float> %{{.*}}, splat (float f0x3F3504F3)
    CHECK:         fptrunc fast <32 x float> %{{.*}} to <32 x half>
    CHECK:         ret <32 x half>
  )";
  RewriteAndCheck(kIr, "__xla_cpu_LogV32F16", kExpected);
}

}  // namespace
}  // namespace xla::cpu
