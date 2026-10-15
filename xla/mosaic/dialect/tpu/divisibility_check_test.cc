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

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {
namespace {

class DivisibilityCheckTest : public ::testing::Test {
 protected:
  DivisibilityCheckTest()
      : context_([]() {
          DialectRegistry registry;
          registry.insert<arith::ArithDialect, func::FuncDialect, TPUDialect>();
          return registry;
        }()),
        builder_(&context_) {
    context_.loadAllAvailableDialects();
  }

  ~DivisibilityCheckTest() override {
    for (int i = ops_.size() - 1; i >= 0; --i) {
      ops_[i]->erase();
    }
  }

  template <typename OpTy, typename... Args>
  OpTy Create(Args&&... args) {
    OpTy op = OpTy::create(builder_, std::forward<Args>(args)...);
    ops_.push_back(op.getOperation());
    return op;
  }

  MLIRContext context_;
  OpBuilder builder_;
  std::vector<Operation*> ops_;
};

TEST_F(DivisibilityCheckTest, ConstantAddIOpIsDivisible) {
  Location loc = builder_.getUnknownLoc();
  auto c1_a = Create<arith::ConstantIndexOp>(loc, 1);
  auto c1_b = Create<arith::ConstantIndexOp>(loc, 1);

  // add = 1 + 1 = 2
  auto add = Create<arith::AddIOp>(loc, c1_a, c1_b);

  std::optional<bool> result = isDivisible(add, /*divisor=*/2);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

// Tests that getRemainder directly returns non-zero remainders for sums
// ((1 + 1) mod 3 = 2).
TEST_F(DivisibilityCheckTest, ConstantAddIOpGetRemainder) {
  Location loc = builder_.getUnknownLoc();
  auto c1_a = Create<arith::ConstantIndexOp>(loc, 1);
  auto c1_b = Create<arith::ConstantIndexOp>(loc, 1);

  // add = 1 + 1 = 2
  auto add = Create<arith::AddIOp>(loc, c1_a, c1_b);

  std::optional<int64_t> rem = getRemainder(add, /*divisor=*/3);
  ASSERT_TRUE(rem.has_value());
  EXPECT_EQ(*rem, 2);
}

// Tests that dynamic terms like (x * 8) + (y * 8) are proven divisible by 8.
TEST_F(DivisibilityCheckTest, DynamicMulIOpAddIsDivisible) {
  Location loc = builder_.getUnknownLoc();
  Type index_type = builder_.getIndexType();
  FunctionType func_type =
      builder_.getFunctionType({index_type, index_type}, {});
  auto func = Create<func::FuncOp>(loc, "test_func", func_type);
  Block* entry = func.addEntryBlock();

  Value x = entry->getArgument(0);
  Value y = entry->getArgument(1);

  builder_.setInsertionPointToStart(entry);
  auto c8_a = Create<arith::ConstantIndexOp>(loc, 8);
  auto c8_b = Create<arith::ConstantIndexOp>(loc, 8);

  auto mul_x = Create<arith::MulIOp>(loc, x, c8_a);
  auto mul_y = Create<arith::MulIOp>(loc, y, c8_b);
  auto add = Create<arith::AddIOp>(loc, mul_x, mul_y);

  std::optional<bool> result = isDivisible(add, /*divisor=*/8);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

TEST_F(DivisibilityCheckTest, NegativeConstantGetRemainder) {
  Location loc = builder_.getUnknownLoc();
  auto neg = Create<arith::ConstantIndexOp>(loc, -1);
  auto pos = Create<arith::ConstantIndexOp>(loc, 5);
  auto add = Create<arith::AddIOp>(loc, neg, pos);

  std::optional<int64_t> rem = getRemainder(neg, /*divisor=*/4);
  ASSERT_TRUE(rem.has_value());
  EXPECT_EQ(*rem, 3);

  std::optional<bool> result = isDivisible(add, /*divisor=*/4);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

TEST_F(DivisibilityCheckTest, MinMaxOpsAreDivisible) {
  Location loc = builder_.getUnknownLoc();
  Type index_type = builder_.getIndexType();
  FunctionType func_type =
      builder_.getFunctionType({index_type, index_type}, {});
  auto func = Create<func::FuncOp>(loc, "test_minmax", func_type);
  Block* entry = func.addEntryBlock();
  Value x = entry->getArgument(0);
  Value y = entry->getArgument(1);

  builder_.setInsertionPointToStart(entry);
  auto c8 = Create<arith::ConstantIndexOp>(loc, 8);
  auto mul_x = Create<arith::MulIOp>(loc, x, c8);
  auto mul_y = Create<arith::MulIOp>(loc, y, c8);

  auto min_si = Create<arith::MinSIOp>(loc, mul_x, mul_y);
  auto min_ui = Create<arith::MinUIOp>(loc, mul_x, mul_y);
  auto max_si = Create<arith::MaxSIOp>(loc, mul_x, mul_y);
  auto max_ui = Create<arith::MaxUIOp>(loc, mul_x, mul_y);

  EXPECT_EQ(isDivisible(min_si, /*divisor=*/8), std::optional<bool>(true));
  EXPECT_EQ(isDivisible(min_ui, /*divisor=*/8), std::optional<bool>(true));
  EXPECT_EQ(isDivisible(max_si, /*divisor=*/8), std::optional<bool>(true));
  EXPECT_EQ(isDivisible(max_ui, /*divisor=*/8), std::optional<bool>(true));
}

TEST_F(DivisibilityCheckTest, OverflowGuardsInMulAndDiv) {
  Location loc = builder_.getUnknownLoc();
  Type index_type = builder_.getIndexType();
  FunctionType func_type = builder_.getFunctionType({index_type}, {});
  auto func = Create<func::FuncOp>(loc, "test_overflow", func_type);
  Block* entry = func.addEntryBlock();
  Value x = entry->getArgument(0);

  builder_.setInsertionPointToStart(entry);
  auto int64_min =
      Create<arith::ConstantIndexOp>(loc, std::numeric_limits<int64_t>::min());
  auto mul_min = Create<arith::MulIOp>(loc, x, int64_min);
  EXPECT_EQ(isDivisible(mul_min, /*divisor=*/8), std::optional<bool>(true));

  auto large =
      Create<arith::ConstantIndexOp>(loc, std::numeric_limits<int64_t>::max());
  auto div_large = Create<arith::DivSIOp>(loc, x, large);
  EXPECT_EQ(isDivisible(div_large, /*divisor=*/8), std::nullopt);
}

}  // namespace
}  // namespace mlir::tpu
