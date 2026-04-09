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

#include "xla/codegen/tiling/experimental/tile.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/constraint_expression.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla::gpu::experimental {
namespace {

class TileTest : public HloHardwareIndependentTestBase {
 public:
  TileTest() { RegisterSymbolicExprStorage(&mlir_context_); }
  mlir::MLIRContext mlir_context_;
};

TilingSpace GetFakeTilingSpace(int64_t num_dims, int64_t num_rt_vars) {
  TilingSpace tiling_space;
  for (int64_t i = 0; i < num_dims; ++i) {
    tiling_space.AppendDimension(nullptr, i, 1,
                                 TilingSpace::DimensionSemantics::kParallel);
  }
  for (int64_t i = 0; i < num_rt_vars; ++i) {
    tiling_space.AppendRTVar(nullptr, i, nullptr, 1);
  }
  return tiling_space;
}

TEST_F(TileTest, StringFormat) {
  TilingSpace tiling_space =
      GetFakeTilingSpace(/*num_dims=*/2, /*num_rt_vars=*/1);

  SymbolicExpr tid0 = CreateDimExpr(0, &mlir_context_);
  SymbolicExpr tid1 = CreateDimExpr(1, &mlir_context_);
  SymbolicExpr ts0 = CreateSymbolExpr(0, /*num_dims=*/2, &mlir_context_);
  SymbolicExpr ts1 = CreateSymbolExpr(1, /*num_dims=*/2, &mlir_context_);
  SymbolicExpr rt = CreateSymbolExpr(2, /*num_dims=*/2, &mlir_context_);
  auto c1 = CreateSymbolicConstant(1, &mlir_context_);
  auto c16 = CreateSymbolicConstant(16, &mlir_context_);
  auto c32 = CreateSymbolicConstant(32, &mlir_context_);

  Tile tile{tiling_space,
            {DimTile{tid0 * ts0, ts0, c1, c16},
             DimTile{rt + tid1 * ts1, ts1, c1, c32}}};

  EXPECT_THAT(tile.ToString(), MatchIndexingString(R"(
    (tid_0, tid_1){rt_0} ->
      offsets [tid_0 * ts_0, rt_0 + tid_1 * ts_1]
      sizes [ts_0, ts_1]
      strides [1, 1]
      upper bounds [16, 32]
  )"));
}

TEST_F(TileTest, SimplifiesAlwaysSatisfiedDisjunction) {
  TilingSpace tiling_space =
      GetFakeTilingSpace(/*num_dims=*/1, /*num_rt_vars=*/0);

  // (d0 in [0, 5]) || (1 in [1, 1]) should simplify to "always satisfied"
  ConstraintExpression c_sym{{ParseAffineExpr("d0", &mlir_context_), {0, 5}}};
  ConstraintExpression c_true{{ParseAffineExpr("1", &mlir_context_), {1, 1}}};

  ConstraintExpression constraints = c_sym || c_true;
  constraints.Simplify();
  tiling_space.mutable_constraint() = constraints;

  Tile tile{tiling_space, {GetFullDimTile(10, &mlir_context_)}};

  EXPECT_TRUE(tiling_space.constraint().IsAlwaysSatisfied());
  EXPECT_THAT(tile.ToString(), MatchIndexingString(R"(
    (tid_0) ->
      offsets [0]
      sizes [16]
      strides [1]
      upper bounds [10]
  )"));
}

TEST_F(TileTest, KeepsNecessaryConstraints) {
  TilingSpace tiling_space =
      GetFakeTilingSpace(/*num_dims=*/1, /*num_rt_vars=*/0);

  // (d0 in [0, 5]) || (d0 in [10, 15])
  ConstraintExpression c1{{ParseAffineExpr("d0", &mlir_context_), {0, 5}}};
  ConstraintExpression c2{{ParseAffineExpr("d0", &mlir_context_), {10, 15}}};

  ConstraintExpression constraints = c1 || c2;
  constraints.Simplify();
  tiling_space.mutable_constraint() = constraints;

  Tile tile{tiling_space, {GetFullDimTile(10, &mlir_context_)}};

  EXPECT_FALSE(tiling_space.constraint().IsAlwaysSatisfied());
  EXPECT_THAT(tile.ToString(), MatchIndexingString(R"(
    (tid_0) ->
      offsets [0]
      sizes [16]
      strides [1]
      upper bounds [10]
  )"));
  EXPECT_THAT(tiling_space.ToString(),
              ::testing::HasSubstr("d0 in [0, 5] || d0 in [10, 15]"));
}

}  // namespace
}  // namespace xla::gpu::experimental
