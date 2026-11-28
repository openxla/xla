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

#include "xla/python/ifrt/mlir/fingerprint_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "xla/pjrt/mlir_to_hlo.h"

namespace xla {
namespace ifrt {
namespace {

TEST(FingerprintUtilsTest, IdenticalModulesHaveSameFingerprint) {
  static constexpr absl::string_view kModule = R"mlir(
module {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = stablehlo.add %arg0, %0 : tensor<f32>
    return %1 : tensor<f32>
  }
}
)mlir";
  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       xla::ParseMlirModuleString(kModule, context));
  ASSERT_OK_AND_ASSIGN(uint64_t fp1, FingerprintModuleOp(*module));
  ASSERT_OK_AND_ASSIGN(uint64_t fp2, FingerprintModuleOp(*module));
  EXPECT_EQ(fp1, fp2);
}

TEST(FingerprintUtilsTest, DistinctStablehloModulesHaveDifferentFingerprints) {
  static constexpr absl::string_view kModule1 = R"mlir(
module {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = stablehlo.add %arg0, %0 : tensor<f32>
    return %1 : tensor<f32>
  }
}
)mlir";
  static constexpr absl::string_view kModule2 = R"mlir(
module {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1 = stablehlo.add %arg0, %0 : tensor<f32>
    return %1 : tensor<f32>
  }
}
)mlir";
  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module1,
                       xla::ParseMlirModuleString(kModule1, context));
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module2,
                       xla::ParseMlirModuleString(kModule2, context));
  ASSERT_OK_AND_ASSIGN(uint64_t fp1, FingerprintModuleOp(*module1));
  ASSERT_OK_AND_ASSIGN(uint64_t fp2, FingerprintModuleOp(*module2));
  EXPECT_NE(fp1, fp2);
}

TEST(FingerprintUtilsTest, IgnoresDebugLocations) {
  static constexpr absl::string_view kModule1 = R"mlir(
module @foo {
  func.func @main(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
    return %arg0 : tensor<2x3xi32> loc("foo")
  }
}
)mlir";
  static constexpr absl::string_view kModule2 = R"mlir(
module @foo {
  func.func @main(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
    return %arg0 : tensor<2x3xi32> loc("bar")
  }
}
)mlir";
  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module1,
                       xla::ParseMlirModuleString(kModule1, context));
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module2,
                       xla::ParseMlirModuleString(kModule2, context));
  ASSERT_OK_AND_ASSIGN(uint64_t fp1, FingerprintModuleOp(*module1));
  ASSERT_OK_AND_ASSIGN(uint64_t fp2, FingerprintModuleOp(*module2));
  EXPECT_EQ(fp1, fp2);
}

TEST(FingerprintUtilsTest, IgnoresDebugLocationStructure) {
  static constexpr absl::string_view kModule1 = R"mlir(
module @foo {
  func.func @main(%arg0: tensor<2x3xi32> loc("foo")) -> tensor<2x3xi32> {
    return %arg0 : tensor<2x3xi32> loc("foo")
  } loc("foo")
} loc("foo")
)mlir";
  static constexpr absl::string_view kModule2 = R"mlir(
module @foo {
  func.func @main(%arg0: tensor<2x3xi32> loc("bar")) -> tensor<2x3xi32> {
    return %arg0 : tensor<2x3xi32> loc("baz")
  } loc("qux")
} loc("quux")
)mlir";
  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module1,
                       xla::ParseMlirModuleString(kModule1, context));
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module2,
                       xla::ParseMlirModuleString(kModule2, context));
  ASSERT_OK_AND_ASSIGN(uint64_t fp1, FingerprintModuleOp(*module1));
  ASSERT_OK_AND_ASSIGN(uint64_t fp2, FingerprintModuleOp(*module2));
  EXPECT_EQ(fp1, fp2);
}

// Parses a module that calls the Mosaic kernel `kernel` (serialized with its
// locations, as JAX does) with `extra_config` prepended to the kernel config.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseModuleWithKernel(
    absl::string_view kernel, mlir::MLIRContext& context,
    absl::string_view extra_config = "") {
  mlir::MLIRContext kernel_context;
  kernel_context.allowUnregisteredDialects();
  mlir::OwningOpRef<mlir::ModuleOp> kernel_module =
      mlir::parseSourceString<mlir::ModuleOp>(kernel, &kernel_context);
  if (!kernel_module) {
    return absl::InvalidArgumentError("Failed to parse the kernel");
  }
  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  if (mlir::failed(mlir::writeBytecodeToFile(*kernel_module, os))) {
    return absl::InternalError("Failed to serialize the kernel");
  }
  return xla::ParseMlirModuleString(
      absl::Substitute(R"mlir(
module {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.custom_call @tpu_custom_call(%arg0) {backend_config = "{\22custom_call_config\22: {$1\22body\22: \22$0\22}}"} : (tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}
)mlir",
                       absl::Base64Escape(bytecode), extra_config),
      context);
}

TEST(FingerprintUtilsTest, IgnoresMosaicKernelLocations) {
  static constexpr absl::string_view kKernel1 = R"mlir(
module {
  "tpu.add"() : () -> () loc("/tmp/a/repro.py":1:2)
}
)mlir";
  static constexpr absl::string_view kKernel2 = R"mlir(
module {
  "tpu.add"() : () -> () loc("/tmp/bb/repro.py":3:4)
}
)mlir";
  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module1,
                       ParseModuleWithKernel(kKernel1, context));
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module2,
                       ParseModuleWithKernel(kKernel2, context));
  ASSERT_OK_AND_ASSIGN(uint64_t fp1, FingerprintModuleOp(*module1));
  ASSERT_OK_AND_ASSIGN(uint64_t fp2, FingerprintModuleOp(*module2));
  EXPECT_EQ(fp1, fp2);
}

TEST(FingerprintUtilsTest, IgnoresMosaicKernelDebugLocations) {
  static constexpr absl::string_view kKernel = R"mlir(
module {
  "tpu.add"() : () -> ()
}
)mlir";
  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module1,
                       ParseModuleWithKernel(kKernel, context));
  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> module2,
      ParseModuleWithKernel(kKernel, context,
                            R"(\22debug_locations\22:\22bG9j\22, )"));
  ASSERT_OK_AND_ASSIGN(uint64_t fp1, FingerprintModuleOp(*module1));
  ASSERT_OK_AND_ASSIGN(uint64_t fp2, FingerprintModuleOp(*module2));
  EXPECT_EQ(fp1, fp2);
}

TEST(FingerprintUtilsTest, DistinctMosaicKernelsHaveDifferentFingerprints) {
  static constexpr absl::string_view kKernel1 = R"mlir(
module {
  "tpu.add"() : () -> ()
}
)mlir";
  static constexpr absl::string_view kKernel2 = R"mlir(
module {
  "tpu.mul"() : () -> ()
}
)mlir";
  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module1,
                       ParseModuleWithKernel(kKernel1, context));
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module2,
                       ParseModuleWithKernel(kKernel2, context));
  ASSERT_OK_AND_ASSIGN(uint64_t fp1, FingerprintModuleOp(*module1));
  ASSERT_OK_AND_ASSIGN(uint64_t fp2, FingerprintModuleOp(*module2));
  EXPECT_NE(fp1, fp2);
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
