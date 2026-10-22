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

#include <string>
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace jaxlib::mosaic {
namespace {

using ::testing::ElementsAre;

class SerdeTest : public ::testing::Test {
 protected:
  SerdeTest() : builder_(&context_) {
    context_.allowUnregisteredDialects();
    module_ = xla::llvm_ir::CreateMlirModuleOp(builder_.getUnknownLoc());
    builder_.setInsertionPointToEnd(module_->getBody());
  }

  mlir::Operation* Create(llvm::StringRef name,
                          mlir::ValueRange operands = {}) {
    mlir::OperationState state(builder_.getUnknownLoc(), name);
    state.addOperands(operands);
    state.addTypes(builder_.getI32Type());
    return builder_.create(state);
  }

  mlir::LogicalResult Serialize(
      const llvm::StringMap<SerdeRuleType>& downgrade_rules) {
    SerdeOptions options;
    options.dialect_prefix = "mangled.";
    options.highest_version = 2;
    options.version_attr_name = "mangled.version";
    options.serialize_version = 1;
    return RunSerde(*module_, /*upgrade_rules=*/{}, downgrade_rules,
                    /*serialize=*/true, options);
  }

  std::vector<std::string> OpNames() {
    std::vector<std::string> names;
    for (mlir::Operation& op : *module_->getBody()) {
      names.push_back(op.getName().getStringRef().str());
    }
    return names;
  }

  mlir::MLIRContext context_;
  mlir::OpBuilder builder_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
};

// A downgrade rule can route the uses of the serialized op through a new op.
TEST_F(SerdeTest, DowngradeRuleSeesUsesOfSerializedOp) {
  mlir::Operation* producer = Create("test.producer");
  Create("test.consumer", producer->getResults());
  ASSERT_TRUE(mlir::succeeded(Serialize(
      {{"test.producer", [this](mlir::Operation* op, int, bool&) {
          builder_.setInsertionPointAfter(op);
          mlir::Operation* wrap = Create("mangled.test.wrap", op->getResults());
          op->getResult(0).replaceAllUsesExcept(wrap->getResult(0), wrap);
          return mlir::success();
        }}})));

  EXPECT_THAT(OpNames(),
              ElementsAre("mangled.test.producer", "mangled.test.wrap",
                          "mangled.test.consumer"));
  mlir::Operation& consumer = module_->getBody()->back();
  EXPECT_EQ(consumer.getOperand(0).getDefiningOp()->getName().getStringRef(),
            "mangled.test.wrap");
}

// A downgrade rule can replace and erase the serialized op.
TEST_F(SerdeTest, DowngradeRuleCanEraseSerializedOp) {
  mlir::Operation* producer = Create("test.producer");
  Create("test.consumer", producer->getResults());
  ASSERT_TRUE(mlir::succeeded(Serialize(
      {{"test.producer", [this](mlir::Operation* op, int, bool& erased) {
          builder_.setInsertionPoint(op);
          op->replaceAllUsesWith(Create("mangled.test.old_producer"));
          op->erase();
          erased = true;
          return mlir::success();
        }}})));

  EXPECT_THAT(OpNames(), ElementsAre("mangled.test.old_producer",
                                     "mangled.test.consumer"));
  mlir::Operation& consumer = module_->getBody()->back();
  EXPECT_EQ(consumer.getOperand(0).getDefiningOp(),
            &module_->getBody()->front());
}

}  // namespace
}  // namespace jaxlib::mosaic
