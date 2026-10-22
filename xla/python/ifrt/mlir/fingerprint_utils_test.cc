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

#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/pjrt/mlir_to_hlo.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::Not;

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
  ASSERT_OK_AND_ASSIGN(uint64_t fp1,
                       FingerprintModuleOp(*module, FingerprintOptions()));
  ASSERT_OK_AND_ASSIGN(uint64_t fp2,
                       FingerprintModuleOp(*module, FingerprintOptions()));
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
  ASSERT_OK_AND_ASSIGN(uint64_t fp1,
                       FingerprintModuleOp(*module1, FingerprintOptions()));
  ASSERT_OK_AND_ASSIGN(uint64_t fp2,
                       FingerprintModuleOp(*module2, FingerprintOptions()));
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
  ASSERT_OK_AND_ASSIGN(uint64_t fp1,
                       FingerprintModuleOp(*module1, FingerprintOptions()));
  ASSERT_OK_AND_ASSIGN(uint64_t fp2,
                       FingerprintModuleOp(*module2, FingerprintOptions()));
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
  ASSERT_OK_AND_ASSIGN(uint64_t fp1,
                       FingerprintModuleOp(*module1, FingerprintOptions()));
  ASSERT_OK_AND_ASSIGN(uint64_t fp2,
                       FingerprintModuleOp(*module2, FingerprintOptions()));
  EXPECT_EQ(fp1, fp2);
}

constexpr FingerprintOptions kCanonicalizeNames = {
    .canonicalize_module_name = true, .canonicalize_sdy_mesh_names = true};

// Like a stage of an MPMD program: the module name, the name of the stage mesh
// and the locations differ between stages.
constexpr absl::string_view kStageModule = R"mlir(
module @"$0" attributes {mhlo.num_partitions = 2 : i32} {
  sdy.mesh @$1 = <["seq"=2]>
  sdy.mesh @mesh = <["seq"=2]>
  func.func @main(
      %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@$1, [{"seq"}, {}]>} loc("$2/x"),
      %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>} loc("$2/y"))
      -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@$1, [{"seq"}, {}]>}) {
    %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@$1, [{"seq"}, {}]>]>} : tensor<8x16xf32> loc("$2/add")
    return %0 : tensor<8x16xf32> loc("$2/return")
  }
}
)mlir";

// Two meshes, of 2 and 4 devices, one of which shards the argument.
constexpr absl::string_view kTwoMeshModule = R"mlir(
module @m {
  sdy.mesh @$0 = <["x"=2]>
  sdy.mesh @$1 = <["x"=4]>
  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@$2, [{"x"}]>}) -> tensor<8xf32> {
    return %arg0 : tensor<8xf32>
  }
}
)mlir";

absl::StatusOr<uint64_t> Fingerprint(absl::string_view module_str,
                                     const FingerprintOptions& options,
                                     mlir::MLIRContext& context) {
  ABSL_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                        xla::ParseMlirModuleString(module_str, context));
  return FingerprintModuleOp(*module, options);
}

absl::StatusOr<uint64_t> Fingerprint(absl::string_view module_str,
                                     const FingerprintOptions& options) {
  mlir::MLIRContext context;
  return Fingerprint(module_str, options, context);
}

TEST(FingerprintUtilsTest, CanonicalizesModuleAndMeshNames) {
  const std::string stage1 =
      absl::Substitute(kStageModule, "p1_layer_5:9.step", "mesh1", "layer_5");
  const std::string stage2 = absl::Substitute(
      kStageModule, "p2_layer_10:14.step", "mesh2", "layer_10");
  ASSERT_OK_AND_ASSIGN(uint64_t fingerprint,
                       Fingerprint(stage1, kCanonicalizeNames));
  EXPECT_THAT(Fingerprint(stage2, kCanonicalizeNames),
              IsOkAndHolds(fingerprint));

  // Modules may share an MLIR context, like the stages of an MPMD program.
  mlir::MLIRContext shared_context(mlir::MLIRContext::Threading::DISABLED);
  EXPECT_THAT(Fingerprint(stage1, kCanonicalizeNames, shared_context),
              IsOkAndHolds(fingerprint));
  EXPECT_THAT(Fingerprint(stage2, kCanonicalizeNames, shared_context),
              IsOkAndHolds(fingerprint));
}

TEST(FingerprintUtilsTest, EachOptionCanonicalizesOnlyItsNames) {
  const std::string p1_mesh1 =
      absl::Substitute(kStageModule, "p1", "mesh1", "layer_5");
  const std::string p1_mesh2 =
      absl::Substitute(kStageModule, "p1", "mesh2", "layer_5");
  const std::string p2_mesh1 =
      absl::Substitute(kStageModule, "p2", "mesh1", "layer_5");

  const FingerprintOptions canonicalize_module_name = {
      .canonicalize_module_name = true};
  ASSERT_OK_AND_ASSIGN(uint64_t fingerprint,
                       Fingerprint(p1_mesh1, canonicalize_module_name));
  EXPECT_THAT(Fingerprint(p2_mesh1, canonicalize_module_name),
              IsOkAndHolds(fingerprint));
  EXPECT_THAT(Fingerprint(p1_mesh2, canonicalize_module_name),
              IsOkAndHolds(Not(fingerprint)));

  const FingerprintOptions canonicalize_mesh_names = {
      .canonicalize_sdy_mesh_names = true};
  ASSERT_OK_AND_ASSIGN(fingerprint,
                       Fingerprint(p1_mesh1, canonicalize_mesh_names));
  EXPECT_THAT(Fingerprint(p1_mesh2, canonicalize_mesh_names),
              IsOkAndHolds(fingerprint));
  EXPECT_THAT(Fingerprint(p2_mesh1, canonicalize_mesh_names),
              IsOkAndHolds(Not(fingerprint)));

  ASSERT_OK_AND_ASSIGN(fingerprint, Fingerprint(p1_mesh1, {}));
  EXPECT_THAT(Fingerprint(p1_mesh2, {}), IsOkAndHolds(Not(fingerprint)));
  EXPECT_THAT(Fingerprint(p2_mesh1, {}), IsOkAndHolds(Not(fingerprint)));
}

TEST(FingerprintUtilsTest,
     CanonicalizingNamesKeepsFingerprintsOfUnnamedModules) {
  static constexpr absl::string_view kModule = R"mlir(
module {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.negate %arg0 : tensor<f32> loc("negate")
    return %0 : tensor<f32>
  }
}
)mlir";
  ASSERT_OK_AND_ASSIGN(uint64_t fingerprint, Fingerprint(kModule, {}));
  EXPECT_THAT(Fingerprint(kModule, kCanonicalizeNames),
              IsOkAndHolds(fingerprint));
}

TEST(FingerprintUtilsTest, CanonicalizingMeshNamesKeepsMeshDefinitions) {
  ASSERT_OK_AND_ASSIGN(
      uint64_t fingerprint,
      Fingerprint(absl::Substitute(kTwoMeshModule, "a", "b", "a"),
                  kCanonicalizeNames));
  const std::string resized = absl::Substitute(
      absl::StrReplaceAll(kTwoMeshModule, {{"\"x\"=2", "\"x\"=8"}}), "a", "b",
      "a");
  EXPECT_THAT(Fingerprint(resized, kCanonicalizeNames),
              IsOkAndHolds(Not(fingerprint)));
}

TEST(FingerprintUtilsTest, CanonicalizingMeshNamesKeepsMeshUses) {
  ASSERT_OK_AND_ASSIGN(
      uint64_t fingerprint,
      Fingerprint(absl::Substitute(kTwoMeshModule, "a", "b", "a"),
                  kCanonicalizeNames));
  EXPECT_THAT(Fingerprint(absl::Substitute(kTwoMeshModule, "a", "b", "b"),
                          kCanonicalizeNames),
              IsOkAndHolds(Not(fingerprint)));
}

TEST(FingerprintUtilsTest, CanonicalizingMeshNamesKeepsTheirOrder) {
  ASSERT_OK_AND_ASSIGN(
      uint64_t fingerprint,
      Fingerprint(absl::Substitute(kTwoMeshModule, "a", "b", "a"),
                  kCanonicalizeNames));
  EXPECT_THAT(Fingerprint(absl::Substitute(kTwoMeshModule, "c", "d", "c"),
                          kCanonicalizeNames),
              IsOkAndHolds(fingerprint));
  EXPECT_THAT(Fingerprint(absl::Substitute(kTwoMeshModule, "b", "a", "b"),
                          kCanonicalizeNames),
              IsOkAndHolds(Not(fingerprint)));
}

TEST(FingerprintUtilsTest, CanonicalizingNamesKeepsOps) {
  ASSERT_OK_AND_ASSIGN(
      uint64_t fingerprint,
      Fingerprint(absl::Substitute(kStageModule, "p1", "mesh1", "layer_5"),
                  kCanonicalizeNames));
  const std::string multiply = absl::StrReplaceAll(
      absl::Substitute(kStageModule, "p2", "mesh2", "layer_10"),
      {{"stablehlo.add", "stablehlo.multiply"}});
  EXPECT_THAT(Fingerprint(multiply, kCanonicalizeNames),
              IsOkAndHolds(Not(fingerprint)));
}

TEST(FingerprintUtilsTest, CanonicalizingMeshNamesKeepsThemInStrings) {
  static constexpr absl::string_view kModule = R"mlir(
module @m {
  sdy.mesh @$0 = <["x"=2]>
  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@$0, [{"x"}]>}) -> tensor<8xf32>
      attributes {mhlo.frontend_attributes = {tag = "$0"}} {
    return %arg0 : tensor<8xf32>
  }
}
)mlir";
  ASSERT_OK_AND_ASSIGN(
      uint64_t fingerprint,
      Fingerprint(absl::Substitute(kModule, "mesh1"), kCanonicalizeNames));
  EXPECT_THAT(
      Fingerprint(absl::Substitute(kModule, "mesh2"), kCanonicalizeNames),
      IsOkAndHolds(Not(fingerprint)));
}

TEST(FingerprintUtilsTest, CanonicalizingMeshNamesKeepsNamesOfNestedSymbols) {
  // The reference resolves to the nested function if it is named like it.
  static constexpr absl::string_view kModule = R"mlir(
module @m {
  sdy.mesh @$0 = <["x"=2]>
  module @nested {
    func.func private @mesh1()
    func.func @main() attributes {test.ref = @$0} {
      return
    }
  }
}
)mlir";
  ASSERT_OK_AND_ASSIGN(
      uint64_t fingerprint,
      Fingerprint(absl::Substitute(kModule, "mesh1"), kCanonicalizeNames));
  EXPECT_THAT(
      Fingerprint(absl::Substitute(kModule, "mesh2"), kCanonicalizeNames),
      IsOkAndHolds(Not(fingerprint)));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
