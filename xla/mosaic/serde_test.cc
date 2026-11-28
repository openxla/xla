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

#include "xla/mosaic/serde.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <initializer_list>
#include <limits>
#include <memory>
#include <string>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/ArithDialect.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/transforms/serde.h"

namespace {

using ::testing::HasSubstr;

constexpr llvm::StringLiteral kPrefix = "stable_mosaic.";
constexpr llvm::StringLiteral kVersionAttr = "stable_mosaic.version";

void RegisterDialects(mlir::MLIRContext& context, bool allow_unregistered) {
  context.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                      mlir::tpu::TPUDialect>();
  context.allowUnregisteredDialects(allow_unregistered);
}

mlir::LogicalResult RunEmptySerde(mlir::ModuleOp module, bool serialize,
                                  int highest_version, int serialize_version,
                                  std::string* diag_out) {
  diag_out->clear();
  mlir::ScopedDiagnosticHandler handler(module.getContext(),
                                        [diag_out](mlir::Diagnostic& diag) {
                                          if (!diag_out->empty()) {
                                            diag_out->push_back('\n');
                                          }
                                          diag_out->append(diag.str());
                                          return mlir::success();
                                        });
  llvm::StringMap<jaxlib::mosaic::UpgradeRuleType> upgrades;
  llvm::StringMap<jaxlib::mosaic::DowngradeRuleType> downgrades;
  return jaxlib::mosaic::RunSerde(module, upgrades, downgrades, serialize,
                                  {.dialect_prefix = kPrefix,
                                   .highest_version = highest_version,
                                   .version_attr_name = kVersionAttr,
                                   .serialize_version = serialize_version});
}

TEST(MosaicSerdeTest, RejectsInvalidAndTruncatedVersions) {
  mlir::MLIRContext context;
  RegisterDialects(context, /*allow_unregistered=*/true);
  mlir::OpBuilder builder(&context);
  std::string diag;

  // Invalid highest_version and serialize_version bounds.
  for (int bad_highest : {0, -1, std::numeric_limits<int>::min()}) {
    mlir::OwningOpRef<mlir::ModuleOp> m =
        mlir::parseSourceString<mlir::ModuleOp>("module {}", &context);
    EXPECT_TRUE(mlir::failed(RunEmptySerde(*m, /*serialize=*/true, bad_highest,
                                           /*serialize_version=*/1, &diag)));
  }
  for (int bad_target : {0, -1, -20, std::numeric_limits<int>::min()}) {
    mlir::OwningOpRef<mlir::ModuleOp> m =
        mlir::parseSourceString<mlir::ModuleOp>("module {}", &context);
    EXPECT_TRUE(mlir::failed(RunEmptySerde(
        *m, /*serialize=*/true, /*highest_version=*/20, bad_target, &diag)));
  }
  {
    mlir::OwningOpRef<mlir::ModuleOp> m =
        mlir::parseSourceString<mlir::ModuleOp>("module {}", &context);
    EXPECT_TRUE(mlir::failed(RunEmptySerde(*m, /*serialize=*/true,
                                           /*highest_version=*/20,
                                           /*serialize_version=*/21, &diag)));
    EXPECT_THAT(diag, HasSubstr("The highest supported version is 20"));
  }
  {
    mlir::OwningOpRef<mlir::ModuleOp> m =
        mlir::parseSourceString<mlir::ModuleOp>("module {}", &context);
    EXPECT_TRUE(mlir::failed(RunEmptySerde(*m, /*serialize=*/false,
                                           /*highest_version=*/20,
                                           /*serialize_version=*/5, &diag)));
    EXPECT_THAT(diag, HasSubstr("Cannot deserialize to a specific version"));
  }

  // Invalid deserialize version attributes, including 64->32 truncation (-2^32
  // + 20 = -4294967276), zero/negative, i0, 1:i1 (signed -1), non-signless,
  // index, and negative 128-bit APInt.
  auto expect_invalid_attr = [&](mlir::Attribute attr) {
    mlir::OwningOpRef<mlir::ModuleOp> m =
        mlir::parseSourceString<mlir::ModuleOp>("module {}", &context);
    (*m)->setAttr(kVersionAttr, attr);
    EXPECT_TRUE(mlir::failed(RunEmptySerde(*m, /*serialize=*/false,
                                           /*highest_version=*/20,
                                           /*serialize_version=*/-1, &diag)));
    EXPECT_THAT(diag, HasSubstr("Missing or invalid version attribute"));
  };
  expect_invalid_attr(builder.getI64IntegerAttr(0));
  expect_invalid_attr(builder.getI64IntegerAttr(-1));
  expect_invalid_attr(builder.getI64IntegerAttr(-4294967276LL));
  expect_invalid_attr(builder.getI64IntegerAttr(-4294967275LL));
  expect_invalid_attr(builder.getI64IntegerAttr(-4294967281LL));
  expect_invalid_attr(
      mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 1), 1));
  expect_invalid_attr(mlir::IntegerAttr::get(
      mlir::IntegerType::get(&context, 0), llvm::APInt::getZero(0)));
  expect_invalid_attr(mlir::IntegerAttr::get(
      mlir::IntegerType::get(&context, 64, mlir::IntegerType::Unsigned), 5));
  expect_invalid_attr(mlir::IntegerAttr::get(
      mlir::IntegerType::get(&context, 64, mlir::IntegerType::Signed), 5));
  expect_invalid_attr(builder.getIndexAttr(5));
  llvm::APInt neg_i128 = -(llvm::APInt(128, 1) << 100);
  expect_invalid_attr(
      mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 128), neg_i128));

  // Versions above highest_version, including +2^100 : i128.
  llvm::APInt pos_i128 = llvm::APInt(128, 1) << 100;
  {
    mlir::OwningOpRef<mlir::ModuleOp> m =
        mlir::parseSourceString<mlir::ModuleOp>("module {}", &context);
    (*m)->setAttr(kVersionAttr,
                  mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 128),
                                         pos_i128));
    EXPECT_TRUE(mlir::failed(RunEmptySerde(*m, /*serialize=*/false,
                                           /*highest_version=*/20,
                                           /*serialize_version=*/-1, &diag)));
    EXPECT_THAT(diag, HasSubstr("Unsupported version: expected <= 20 but got "
                                "1267650600228229401496703205376"));
  }

  // Valid versions across signless widths (1:i2, 1:i32, 15:i64, 20:i128).
  for (mlir::Attribute good : std::initializer_list<mlir::Attribute>{
           mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 2), 1),
           builder.getI32IntegerAttr(1),
           builder.getI64IntegerAttr(15),
           mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 128),
                                  llvm::APInt(128, 20)),
       }) {
    mlir::OwningOpRef<mlir::ModuleOp> m =
        mlir::parseSourceString<mlir::ModuleOp>("module {}", &context);
    (*m)->setAttr(kVersionAttr, good);
    EXPECT_TRUE(mlir::succeeded(
        RunEmptySerde(*m, /*serialize=*/false, /*highest_version=*/20,
                      /*serialize_version=*/-1, &diag)));
  }
}

TEST(MosaicSerdeTest,
     StrictContextHandlesUnregisteredErasedDotlessAndSuccessorOps) {
  // 1) Strict context rejects unregistered op and preserves erased tpu.repeat.
  constexpr llvm::StringLiteral kUnregisteredIr = R"mlir(
    module attributes {stable_mosaic.version = 20 : i64} {
      "stable_mosaic.unknown_dialect.op"() : () -> ()
    }
  )mlir";
  constexpr llvm::StringLiteral kRepeatIr = R"mlir(
    module attributes {stable_mosaic.version = 18 : i64} {
      "stable_mosaic.func.func"() ({
      ^bb0(%arg0: vector<8x128xf32>):
        %0 = "stable_mosaic.tpu.repeat"(%arg0) {dimension = 1 : i32, times = 2 : i32} : (vector<8x128xf32>) -> vector<8x256xf32>
        "stable_mosaic.func.return"(%0) : (vector<8x256xf32>) -> ()
      }) {function_type = (vector<8x128xf32>) -> vector<8x256xf32>, sym_name = "main"} : () -> ()
    }
  )mlir";
  {
    mlir::MLIRContext ctx;
    RegisterDialects(ctx, /*allow_unregistered=*/true);
    mlir::OwningOpRef<mlir::ModuleOp> bad =
        mlir::parseSourceString<mlir::ModuleOp>(kUnregisteredIr, &ctx);
    ASSERT_TRUE(bad);
    mlir::OwningOpRef<mlir::ModuleOp> rep =
        mlir::parseSourceString<mlir::ModuleOp>(kRepeatIr, &ctx);
    ASSERT_TRUE(rep);
    ctx.allowUnregisteredDialects(false);

    mlir::PassManager pm(&ctx);
    pm.addPass(mlir::tpu::createMosaicSerdePass(
        mlir::tpu::MosaicSerdePassOptions{.serialize = false}));
    EXPECT_TRUE(mlir::failed(pm.run(*bad)));
    EXPECT_TRUE(mlir::succeeded(pm.run(*rep)));
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*rep)));
  }

  // Dotless/empty demangled names and non-terminator with block successors.
  {
    mlir::MLIRContext context;
    RegisterDialects(context, /*allow_unregistered=*/true);
    mlir::OpBuilder builder(&context);
    for (llvm::StringRef bad_name :
         {"stable_mosaic.", "stable_mosaic.nodot", "stable_mosaic..op",
          "stable_mosaic.tpu."}) {
      mlir::OwningOpRef<mlir::ModuleOp> m =
          mlir::parseSourceString<mlir::ModuleOp>("module {}", &context);
      (*m)->setAttr(kVersionAttr, builder.getI64IntegerAttr(20));
      builder.setInsertionPointToStart(m->getBody());
      mlir::OperationState st(builder.getUnknownLoc(), bad_name);
      builder.create(st);
      std::string diag;
      EXPECT_TRUE(
          mlir::failed(RunEmptySerde(*m, /*serialize=*/false, 20, -1, &diag)));
      EXPECT_THAT(diag, HasSubstr("Operation not in a serialized form"));
    }

    // Registered non-terminator (arith.addi) carrying a successor block.
    mlir::OwningOpRef<mlir::ModuleOp> m =
        mlir::parseSourceString<mlir::ModuleOp>("module {}", &context);
    (*m)->setAttr(kVersionAttr, builder.getI64IntegerAttr(20));
    mlir::Region* holder_region = nullptr;
    {
      builder.setInsertionPointToStart(m->getBody());
      mlir::OperationState fn(builder.getUnknownLoc(),
                              "stable_mosaic.func.func");
      fn.addRegion();
      holder_region = &builder.create(fn)->getRegion(0);
    }
    mlir::Block* b0 = &holder_region->emplaceBlock();
    mlir::Block* b1 = &holder_region->emplaceBlock();
    mlir::Value arg =
        b0->addArgument(builder.getI32Type(), builder.getUnknownLoc());
    builder.setInsertionPointToStart(b0);
    mlir::OperationState bad_succ(builder.getUnknownLoc(),
                                  "stable_mosaic.arith.addi");
    bad_succ.addOperands({arg, arg});
    bad_succ.addTypes(builder.getI32Type());
    bad_succ.addSuccessors(b1);
    builder.create(bad_succ);
    builder.setInsertionPointToStart(b1);
    mlir::OperationState ret(builder.getUnknownLoc(),
                             "stable_mosaic.func.return");
    builder.create(ret);

    std::string diag;
    EXPECT_TRUE(
        mlir::failed(RunEmptySerde(*m, /*serialize=*/false, 20, -1, &diag)));
    EXPECT_THAT(diag, HasSubstr("non-terminator"));
  }
}

TEST(MosaicSerdeTest, PassOptionsDefaultsAndCopyPreserveOptionState) {
  mlir::MLIRContext ctx;
  RegisterDialects(ctx, /*allow_unregistered=*/true);
  constexpr llvm::StringLiteral kSimple = R"mlir(
    module {
      func.func @main() {
        return
      }
    }
  )mlir";

  // Default target_version = -1 serializes at kVersion rather than 0.
  mlir::OwningOpRef<mlir::ModuleOp> m =
      mlir::parseSourceString<mlir::ModuleOp>(kSimple, &ctx);
  ASSERT_TRUE(m);
  mlir::tpu::MosaicSerdePassOptions ser_opts;
  EXPECT_FALSE(ser_opts.serialize);
  EXPECT_EQ(ser_opts.target_version, -1);
  ser_opts.serialize = true;
  {
    mlir::PassManager pm(&ctx);
    pm.addPass(mlir::tpu::createMosaicSerdePass(ser_opts));
    ASSERT_TRUE(mlir::succeeded(pm.run(*m)));
  }
  auto ver = (*m)->getAttrOfType<mlir::IntegerAttr>(kVersionAttr);
  ASSERT_TRUE(ver);
  EXPECT_EQ(ver.getValue().getSExtValue(),
            mlir::tpu::MosaicSerdePass::kVersion);

  // Explicit {.serialize = false, .target_version = 0} remains accepted.
  {
    mlir::OwningOpRef<mlir::ModuleOp> clone = m->clone();
    mlir::PassManager pm(&ctx);
    pm.addPass(
        mlir::tpu::createMosaicSerdePass(mlir::tpu::MosaicSerdePassOptions{
            .serialize = false, .target_version = 0}));
    EXPECT_TRUE(mlir::succeeded(pm.run(*clone)));
  }

  // Copy-ctor and copy-assignment preserve keep-version-attr=false and unset
  // serialize occurrence state.
  struct TestablePass : public mlir::tpu::MosaicSerdePass {
    TestablePass() = default;
    explicit TestablePass(mlir::tpu::MosaicSerdePassOptions opts)
        : mlir::tpu::MosaicSerdePass(opts) {}
    void SetKeepVersionAttr(bool v) { keep_version_attr = v; }
    bool HasSerialize() const { return serialize.hasValue(); }
    bool HasTargetVersion() const { return target_version.hasValue(); }
    bool KeepVersionAttr() const { return keep_version_attr; }
  };
  TestablePass unset_pass;
  mlir::tpu::MosaicSerdePass copied_unset(unset_pass);
  {
    mlir::OwningOpRef<mlir::ModuleOp> clone = m->clone();
    mlir::PassManager pm(&ctx);
    pm.addPass(std::make_unique<mlir::tpu::MosaicSerdePass>(copied_unset));
    EXPECT_TRUE(mlir::failed(pm.run(*clone)));
  }

  TestablePass deser_strip(
      mlir::tpu::MosaicSerdePassOptions{.serialize = false});
  deser_strip.SetKeepVersionAttr(false);
  EXPECT_FALSE(deser_strip.HasTargetVersion());
  EXPECT_FALSE(deser_strip.KeepVersionAttr());
  mlir::tpu::MosaicSerdePass copy_constructed(deser_strip);
  mlir::tpu::MosaicSerdePass copy_assigned;
  copy_assigned = copy_constructed;
  {
    mlir::OwningOpRef<mlir::ModuleOp> clone = m->clone();
    mlir::PassManager pm(&ctx);
    pm.addPass(std::make_unique<mlir::tpu::MosaicSerdePass>(copy_assigned));
    ASSERT_TRUE(mlir::succeeded(pm.run(*clone)));
    EXPECT_FALSE((*clone)->hasAttr(kVersionAttr));
  }
}

}  // namespace
