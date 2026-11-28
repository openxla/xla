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

#include "xla/mosaic/tiled_layout_util.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/layout.h"

namespace jaxlib::mosaic {
namespace {

std::optional<bool> CombineDivisibility(mlir::Value lhs, mlir::Value rhs,
                                        int64_t divisor, int64_t fuel,
                                        bool short_circuit_value,
                                        AssumeMultipleExtractor extract) {
  const std::optional<bool> lhs_divisible =
      IsDivisible(lhs, divisor, fuel / 2, extract);
  if (lhs_divisible == short_circuit_value) {
    return short_circuit_value;
  }
  const std::optional<bool> rhs_divisible =
      IsDivisible(rhs, divisor, llvm::divideCeil(fuel, 2), extract);
  if (rhs_divisible == short_circuit_value) {
    return short_circuit_value;
  }
  if (lhs_divisible.has_value() && rhs_divisible.has_value()) {
    return !short_circuit_value;
  }
  return std::nullopt;
}

std::optional<bool> SameDivisibility(mlir::Value lhs, mlir::Value rhs,
                                     int64_t divisor, int64_t fuel,
                                     AssumeMultipleExtractor extract) {
  const std::optional<bool> lhs_divisible =
      IsDivisible(lhs, divisor, fuel / 2, extract);
  if (!lhs_divisible.has_value()) {
    return std::nullopt;
  }
  const std::optional<bool> rhs_divisible =
      IsDivisible(rhs, divisor, llvm::divideCeil(fuel, 2), extract);
  return lhs_divisible == rhs_divisible ? lhs_divisible : std::nullopt;
}

std::optional<bool> IsMulDivisible(mlir::arith::MulIOp mul_op, int64_t divisor,
                                   int64_t fuel,
                                   AssumeMultipleExtractor extract) {
  for (auto [const_operand, other_operand] :
       {std::pair{mul_op.getRhs(), mul_op.getLhs()},
        std::pair{mul_op.getLhs(), mul_op.getRhs()}}) {
    const std::optional<int64_t> const_int =
        mlir::getConstantIntValue(const_operand);
    if (!const_int || *const_int == std::numeric_limits<int64_t>::min()) {
      continue;
    }
    if (const int64_t shared_factor = std::gcd(*const_int, divisor);
        shared_factor > 1) {
      return IsDivisible(other_operand, divisor / shared_factor, fuel - 1,
                         extract);
    }
  }
  return CombineDivisibility(mul_op.getRhs(), mul_op.getLhs(), divisor, fuel,
                             /*short_circuit_value=*/true, extract);
}

std::optional<int64_t> GetExplicitRemainder(mlir::Value value, int64_t divisor,
                                            int64_t fuel,
                                            AssumeMultipleExtractor extract) {
  if (const std::optional<int64_t> const_int =
          mlir::getConstantIntValue(value)) {
    const int64_t rem = *const_int % divisor;
    return rem < 0 ? rem + divisor : rem;
  }
  auto add_op = value.getDefiningOp<mlir::arith::AddIOp>();
  if (!add_op) {
    return std::nullopt;
  }
  const std::optional<int64_t> lhs_rem =
      GetRemainder(add_op.getLhs(), divisor, fuel / 2, extract);
  if (!lhs_rem.has_value()) {
    return std::nullopt;
  }
  const std::optional<int64_t> rhs_rem = GetRemainder(
      add_op.getRhs(), divisor, llvm::divideCeil(fuel, 2), extract);
  if (!rhs_rem.has_value()) {
    return std::nullopt;
  }
  // Subtract `(divisor - *rhs_rem)` before adding to prevent `int64_t`
  // overflow when `divisor` is large.
  return *lhs_rem >= divisor - *rhs_rem ? *lhs_rem - (divisor - *rhs_rem)
                                        : *lhs_rem + *rhs_rem;
}

std::optional<bool> IsStructurallyDivisible(mlir::Value value, int64_t divisor,
                                            int64_t fuel,
                                            AssumeMultipleExtractor extract) {
  if (auto block_arg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    auto for_op = llvm::dyn_cast_if_present<mlir::scf::ForOp>(
        block_arg.getOwner()->getParentOp());
    if (for_op && for_op.getInductionVar() == value) {
      return CombineDivisibility(for_op.getLowerBound(), for_op.getStep(),
                                 divisor, fuel, /*short_circuit_value=*/false,
                                 extract);
    }
    return std::nullopt;
  }
  mlir::Operation* def = value.getDefiningOp();
  if (!def) {
    return std::nullopt;
  }
  if (auto assumed = extract(def)) {
    return assumed->second % divisor == 0
               ? std::optional<bool>(true)
               : IsDivisible(assumed->first, divisor, fuel - 1, extract);
  }
  const int64_t lhs_fuel = fuel / 2;
  const int64_t rhs_fuel = llvm::divideCeil(fuel, 2);
  auto true_or_unknown = [](bool cond) -> std::optional<bool> {
    return cond ? std::optional<bool>(true) : std::nullopt;
  };
  return llvm::TypeSwitch<mlir::Operation*, std::optional<bool>>(def)
      .Case([&](mlir::arith::MulIOp mul_op) {
        return IsMulDivisible(mul_op, divisor, fuel, extract);
      })
      .Case([&](mlir::arith::IndexCastOp cast_op) {
        return IsDivisible(cast_op.getOperand(), divisor, fuel - 1, extract);
      })
      .Case<mlir::arith::DivUIOp, mlir::arith::DivSIOp>([&](auto div_op) {
        const std::optional<int64_t> rhs =
            mlir::getConstantIntValue(div_op.getRhs());
        return true_or_unknown(
            rhs && *rhs > 0 &&
            divisor <= std::numeric_limits<int64_t>::max() / *rhs &&
            IsGuaranteedDivisible(div_op.getLhs(), divisor * *rhs, fuel - 1,
                                  extract));
      })
      .Case([&](mlir::arith::SubIOp sub_op) -> std::optional<bool> {
        const std::optional<bool> lhs_divisible =
            IsDivisible(sub_op.getLhs(), divisor, lhs_fuel, extract);
        if (!lhs_divisible.has_value()) {
          return std::nullopt;
        }
        const std::optional<bool> rhs_divisible =
            IsDivisible(sub_op.getRhs(), divisor, rhs_fuel, extract);
        // If neither side is divisible, their remainders could still cancel.
        return (rhs_divisible.has_value() && (*lhs_divisible || *rhs_divisible))
                   ? std::optional<bool>(*lhs_divisible && *rhs_divisible)
                   : std::nullopt;
      })
      .Case<mlir::arith::RemSIOp, mlir::arith::RemUIOp>([&](auto rem_op) {
        return true_or_unknown(
            IsGuaranteedDivisible(rem_op.getLhs(), divisor, lhs_fuel,
                                  extract) &&
            IsGuaranteedDivisible(rem_op.getRhs(), divisor, rhs_fuel, extract));
      })
      .Case([&](mlir::arith::AndIOp and_op) {
        return true_or_unknown(llvm::isPowerOf2_64(divisor) &&
                               (IsGuaranteedDivisible(and_op.getRhs(), divisor,
                                                      lhs_fuel, extract) ||
                                IsGuaranteedDivisible(and_op.getLhs(), divisor,
                                                      rhs_fuel, extract)));
      })
      .Case<mlir::arith::MinSIOp, mlir::arith::MinUIOp, mlir::arith::MaxSIOp,
            mlir::arith::MaxUIOp>([&](auto minmax_op) {
        return SameDivisibility(minmax_op.getLhs(), minmax_op.getRhs(), divisor,
                                fuel, extract);
      })
      .Case([&](mlir::arith::SelectOp select_op) {
        return SameDivisibility(select_op.getTrueValue(),
                                select_op.getFalseValue(), divisor, fuel,
                                extract);
      })
      .Default(std::nullopt);
}

int64_t GetPackingSubtileDim(llvm::ArrayRef<xla::Tile> tiles) {
  return (tiles.size() > 1 && !tiles[1].dimensions().empty())
             ? tiles[1].dimensions()[0]
             : 1;
}

// Finds the outermost dimension packed contiguously inside each lane chunk,
// verifying that all dimensions above it step across full lane chunks.
std::optional<int64_t> FindInputChunkSplit(llvm::ArrayRef<int64_t> shape,
                                           llvm::ArrayRef<int64_t> strides,
                                           int64_t tile_rows,
                                           int64_t lane_chunks) {
  const int64_t rank = shape.size();
  const int64_t lane_stride = strides.back();
  const int64_t input_second_minor = shape[rank - 2];
  int64_t inner_tiles = llvm::divideCeil(input_second_minor, tile_rows);
  if (inner_tiles > lane_stride) {
    return std::nullopt;
  }
  int64_t split_dim = rank - 2;
  for (int64_t dim = rank - 3; dim >= 0; --dim) {
    if (shape[dim] > 1 && strides[dim] == inner_tiles &&
        inner_tiles * shape[dim] <= lane_stride) {
      // Folding an outer dimension requires unpadded second minor tiles.
      if (input_second_minor % tile_rows != 0) {
        return std::nullopt;
      }
      inner_tiles *= shape[dim];
      split_dim = dim;
    } else if (shape[dim] == 1 && strides[dim] <= lane_stride) {
      split_dim = dim;
    } else {
      break;
    }
  }
  int64_t expected_outer_stride = lane_stride * lane_chunks;
  for (int64_t dim = split_dim - 1; dim >= 0; --dim) {
    if (shape[dim] != 1 && strides[dim] != expected_outer_stride) {
      return std::nullopt;
    }
    if (dim > 0) {
      if (mlir::ShapedType::isDynamic(shape[dim])) {
        return std::nullopt;
      }
      expected_outer_stride *= shape[dim];
    }
  }
  return split_dim;
}

// Finds the dimension in `result_shape` that partitions inner sublane elements
// from outer batch slices identically to `input_split` in `input_shape`.
std::optional<int64_t> FindMatchingResultChunkSplit(
    llvm::ArrayRef<int64_t> input_shape, int64_t input_split,
    llvm::ArrayRef<int64_t> result_shape) {
  const int64_t inner_elements =
      llvm::product_of(input_shape.drop_back().drop_front(input_split));
  int64_t result_inner_elements = 1;
  int64_t result_split = static_cast<int64_t>(result_shape.size()) - 1;
  do {
    --result_split;
    if (result_shape[result_split] <= 0) {
      return std::nullopt;
    }
    result_inner_elements *= result_shape[result_split];
  } while (result_inner_elements < inner_elements && result_split > 0);
  if (result_inner_elements != inner_elements) {
    return std::nullopt;
  }
  const llvm::ArrayRef<int64_t> outer_input =
      input_shape.take_front(input_split);
  const llvm::ArrayRef<int64_t> outer_result =
      result_shape.take_front(result_split);
  const bool input_has_dynamic_outer =
      !outer_input.empty() && mlir::ShapedType::isDynamic(outer_input[0]);
  const bool result_has_dynamic_outer =
      !outer_result.empty() && mlir::ShapedType::isDynamic(outer_result[0]);
  if (input_has_dynamic_outer != result_has_dynamic_outer) {
    return std::nullopt;
  }
  const llvm::ArrayRef<int64_t> static_outer_input =
      outer_input.drop_front(input_has_dynamic_outer);
  const llvm::ArrayRef<int64_t> static_outer_result =
      outer_result.drop_front(result_has_dynamic_outer);
  if (llvm::any_of(static_outer_result, mlir::ShapedType::isDynamic) ||
      llvm::product_of(static_outer_input) !=
          llvm::product_of(static_outer_result)) {
    return std::nullopt;
  }
  return result_split;
}

std::optional<InferredTiledLayout> InferSecondMinorReshapeLayout(
    llvm::ArrayRef<int64_t> input_shape, llvm::ArrayRef<int64_t> result_shape,
    llvm::ArrayRef<xla::Tile> tiles, llvm::ArrayRef<int64_t> strides,
    bool is_contiguous) {
  const int64_t input_rank = input_shape.size();
  const int64_t result_rank = result_shape.size();
  const int64_t packing = GetPackingSubtileDim(tiles);
  if (input_rank < 2 || result_rank < 2 ||
      !HasValidPacked2DTiling(tiles, packing) ||
      mlir::ShapedType::isDynamic(input_shape.back()) ||
      result_shape.back() != input_shape.back()) {
    return std::nullopt;
  }
  const llvm::ArrayRef<int64_t> first_tile = tiles.front().dimensions();
  const int64_t lane_chunks =
      llvm::divideCeil(input_shape.back(), first_tile[1]);
  const int64_t input_second_minor = input_shape[input_rank - 2];
  const int64_t result_second_minor = result_shape[result_rank - 2];
  auto is_unit_dim = [](int64_t dim) { return dim == 1; };
  if (is_contiguous) {
    const bool has_valid_sublane_sizes =
        first_tile[0] > 1 && mlir::ShapedType::isStatic(input_second_minor) &&
        input_second_minor <= first_tile[0] && result_second_minor > 0 &&
        input_second_minor % packing == 0 && result_second_minor % packing == 0;
    // Partial tiles leave padding between outer batch slices when any outer
    // dim exceeds 1, so reshaping across them would shift slice offsets.
    const bool has_padded_outer_slices =
        input_second_minor < first_tile[0] &&
        !llvm::all_of(input_shape.drop_back(2), is_unit_dim);
    if (!has_valid_sublane_sizes || has_padded_outer_slices) {
      return std::nullopt;
    }
    if (lane_chunks <= 1) {
      if (result_second_minor > first_tile[0] &&
          input_second_minor != first_tile[0]) {
        return std::nullopt;
      }
      llvm::SmallVector<xla::Tile> result_tiles(tiles.begin(), tiles.end());
      result_tiles[0] = xla::Tile({result_second_minor, first_tile[1]});
      return InferredTiledLayout{std::move(result_tiles), std::nullopt};
    }
    if (first_tile[0] % result_second_minor != 0) {
      return std::nullopt;
    }
  } else if (strides[input_rank - 2] != 1 || strides.back() < 1 ||
             lane_chunks <= 1) {
    return std::nullopt;
  }
  if (llvm::any_of(strides, mlir::ShapedType::isDynamic)) {
    return std::nullopt;
  }
  const int64_t lane_stride = strides.back();
  if (mlir::ShapedType::isDynamic(input_second_minor)) {
    if (!mlir::ShapedType::isDynamic(result_second_minor) ||
        !llvm::all_of(input_shape.drop_back(2), is_unit_dim) ||
        !llvm::all_of(result_shape.drop_back(2), is_unit_dim)) {
      return std::nullopt;
    }
    llvm::SmallVector<int64_t> result_strides(result_rank, 1);
    result_strides.back() = lane_stride;
    return InferredTiledLayout{
        llvm::SmallVector<xla::Tile>(tiles.begin(), tiles.end()),
        std::move(result_strides)};
  }
  const std::optional<int64_t> input_split =
      FindInputChunkSplit(input_shape, strides, first_tile[0], lane_chunks);
  if (!input_split.has_value()) {
    return std::nullopt;
  }
  const std::optional<int64_t> result_split =
      FindMatchingResultChunkSplit(input_shape, *input_split, result_shape);
  if (!result_split.has_value()) {
    return std::nullopt;
  }
  const llvm::ArrayRef<int64_t> inner_input =
      input_shape.drop_back().drop_front(*input_split);
  const llvm::ArrayRef<int64_t> inner_result =
      result_shape.drop_back().drop_front(*result_split);
  const int64_t inner_elements = llvm::product_of(inner_input);
  int64_t result_tile_rows = first_tile[0];
  if (inner_input != inner_result && (inner_elements > input_second_minor ||
                                      inner_elements > result_second_minor)) {
    if (input_second_minor % packing != 0 ||
        result_second_minor % packing != 0) {
      return std::nullopt;
    }
    result_tile_rows = std::gcd(first_tile[0], result_second_minor);
  }
  const int64_t result_lane_stride =
      lane_stride * (first_tile[0] / result_tile_rows);
  llvm::SmallVector<xla::Tile> result_tiles(tiles.begin(), tiles.end());
  result_tiles[0] = xla::Tile({result_tile_rows, first_tile[1]});
  const int64_t result_match_dims = result_rank - 1 - *result_split;
  return InferredTiledLayout{
      std::move(result_tiles),
      ComputeMatchDimensionTileStrides(result_shape, result_match_dims,
                                       result_tile_rows, first_tile[1],
                                       result_lane_stride)};
}

std::optional<InferredTiledLayout> InferPreservedTileReshapeLayout(
    llvm::ArrayRef<int64_t> input_shape, llvm::ArrayRef<int64_t> result_shape,
    llvm::ArrayRef<xla::Tile> input_tiles) {
  const llvm::ArrayRef<int64_t> first_tile = input_tiles.front().dimensions();
  const int64_t input_rank = input_shape.size();
  const int64_t result_rank = result_shape.size();
  if (result_rank < static_cast<int64_t>(first_tile.size())) {
    return std::nullopt;
  }
  const llvm::ArrayRef<int64_t> active_tile =
      first_tile.drop_while([](int64_t dim) { return dim == 1; });
  if (active_tile.empty()) {
    return InferredTiledLayout{{}, std::nullopt};
  }
  const int64_t active_rank = active_tile.size();
  if (input_shape.take_back(active_rank - 1) !=
      result_shape.take_back(active_rank - 1)) {
    return std::nullopt;
  }
  const int64_t input_leading = input_shape[input_rank - active_rank];
  const int64_t result_leading = result_shape[result_rank - active_rank];
  const bool both_tile_aligned = mlir::ShapedType::isStatic(input_leading) &&
                                 mlir::ShapedType::isStatic(result_leading) &&
                                 input_leading % active_tile.front() == 0 &&
                                 result_leading % active_tile.front() == 0;
  if (input_leading != result_leading && !both_tile_aligned) {
    return std::nullopt;
  }
  return InferredTiledLayout{
      llvm::SmallVector<xla::Tile>(input_tiles.begin(), input_tiles.end()),
      std::nullopt};
}

// Expands one collapsed `(non_tile_elements, tile_elements)` group into
// consecutive static dimensions of `result_shape` starting at `result_dim`,
// appending corresponding tile extents to `result_tile`.
bool ExpandCollapsedDimsIntoResult(
    llvm::ArrayRef<int64_t> result_shape, int64_t non_tile_elements,
    int64_t tile_elements, int64_t& result_dim,
    llvm::SmallVectorImpl<int64_t>& result_tile) {
  const int64_t result_rank = result_shape.size();
  while (result_dim < result_rank &&
         mlir::ShapedType::isStatic(result_shape[result_dim])) {
    const int64_t dim = result_shape[result_dim];
    if (non_tile_elements % dim == 0) {
      if (!result_tile.empty()) {
        result_tile.push_back(1);
      }
      non_tile_elements /= dim;
      ++result_dim;
    } else if (dim % non_tile_elements == 0) {
      break;
    } else {
      return false;
    }
  }
  while (result_dim < result_rank &&
         mlir::ShapedType::isStatic(result_shape[result_dim])) {
    CHECK_EQ(result_shape[result_dim] % non_tile_elements, 0);
    const int64_t tile_dim = result_shape[result_dim] / non_tile_elements;
    non_tile_elements = 1;
    if (tile_elements % tile_dim != 0) {
      break;
    }
    if (!result_tile.empty() || tile_dim != 1) {
      result_tile.push_back(tile_dim);
    }
    tile_elements /= tile_dim;
    ++result_dim;
  }
  return non_tile_elements == 1 && tile_elements == 1;
}

// Collapses adjacent compatible outer and tile factors in `input_shape` and
// expands them into `result_shape` while preserving any packing subtile.
std::optional<InferredTiledLayout> InferCollapsedTileReshapeLayout(
    llvm::ArrayRef<int64_t> input_shape, llvm::ArrayRef<int64_t> result_shape,
    llvm::ArrayRef<xla::Tile> input_tiles) {
  const llvm::ArrayRef<int64_t> first_tile = input_tiles.front().dimensions();
  const int64_t first_tile_rank = first_tile.size();
  const int64_t input_rank = input_shape.size();
  const int64_t result_rank = result_shape.size();
  const int64_t untiled_rank = input_rank - first_tile_rank;
  const int64_t packing = GetPackingSubtileDim(input_tiles);
  const bool has_packed_subtile =
      input_tiles.size() == 2 && first_tile_rank >= 2 && packing > 1 &&
      HasValidPackingSubtile(input_tiles, packing) &&
      first_tile[first_tile_rank - 2] % packing == 0;
  if (input_tiles.size() != 1 && !has_packed_subtile) {
    return std::nullopt;
  }
  auto tile_size = [&](int64_t dim) {
    return untiled_rank <= dim ? first_tile[dim - untiled_rank] : 1;
  };
  auto is_dynamic_or_padded = [&](int64_t dim) {
    return mlir::ShapedType::isDynamic(input_shape[dim]) ||
           input_shape[dim] % tile_size(dim) != 0;
  };
  int64_t input_dim = 0;
  int64_t result_dim = 0;
  llvm::SmallVector<int64_t> result_tile;
  while (input_dim < input_rank || result_dim < result_rank) {
    if (input_dim < input_rank && is_dynamic_or_padded(input_dim)) {
      if (result_dim == result_rank ||
          input_shape[input_dim] != result_shape[result_dim]) {
        return std::nullopt;
      }
      if (untiled_rank <= input_dim) {
        result_tile.push_back(first_tile[input_dim - untiled_rank]);
      }
      ++input_dim;
      ++result_dim;
      continue;
    }
    int64_t non_tile_elements = 1;
    while (input_dim < input_rank && !is_dynamic_or_padded(input_dim)) {
      const int64_t dim_tile_size = tile_size(input_dim);
      non_tile_elements *= input_shape[input_dim] / dim_tile_size;
      if (dim_tile_size != 1) {
        break;
      }
      ++input_dim;
    }
    int64_t tile_elements = 1;
    while (input_dim < input_rank && !is_dynamic_or_padded(input_dim)) {
      tile_elements *= tile_size(input_dim);
      ++input_dim;
      // Keep the lane dimension intact when a `(packing, 1)` subtile exists.
      if (input_dim == input_rank ||
          (has_packed_subtile && input_dim == input_rank - 1) ||
          input_shape[input_dim] != tile_size(input_dim)) {
        break;
      }
    }
    if (!ExpandCollapsedDimsIntoResult(result_shape, non_tile_elements,
                                       tile_elements, result_dim,
                                       result_tile)) {
      return std::nullopt;
    }
  }
  llvm::SmallVector<xla::Tile> result_tiles{xla::Tile(result_tile)};
  if (has_packed_subtile) {
    if (result_tile.size() < 2 ||
        result_tile[result_tile.size() - 2] % packing != 0 ||
        result_tile.back() != first_tile.back()) {
      return std::nullopt;
    }
    result_tiles.push_back(input_tiles[1]);
  }
  return InferredTiledLayout{std::move(result_tiles), std::nullopt};
}

bool MeetsExpectedStrideFloor(llvm::ArrayRef<int64_t> dims,
                              int64_t match_dim_count, int64_t tile_rows,
                              int64_t lane_tile,
                              llvm::ArrayRef<int64_t> tile_strides) {
  const llvm::SmallVector<int64_t> expected = ComputeMatchDimensionTileStrides(
      dims, match_dim_count, tile_rows, lane_tile);
  return llvm::all_of(llvm::zip(tile_strides, expected), [](auto pair) {
    auto [actual, floor] = pair;
    return mlir::ShapedType::isDynamic(actual) || actual >= floor;
  });
}

}  // namespace

llvm::SmallVector<int64_t> GetUnitSubtile(llvm::ArrayRef<xla::Tile> tiles) {
  CHECK(!tiles.empty());
  const int64_t first_tile_rank = tiles.front().dimensions().size();
  llvm::SmallVector<int64_t> unit_subtile(first_tile_rank, 1);
  size_t current_tiled_rank = first_tile_rank;
  for (const xla::Tile& tile : tiles.drop_front()) {
    const size_t tile_rank = tile.dimensions().size();
    CHECK_LE(tile_rank, current_tiled_rank);
    for (size_t i = 0; i < tile_rank; ++i) {
      const size_t tiled_dim = current_tiled_rank - tile_rank + i;
      if (tiled_dim < static_cast<size_t>(first_tile_rank)) {
        unit_subtile[tiled_dim] *= tile.dimension(i);
      }
    }
    current_tiled_rank += tile_rank;
  }
  return unit_subtile;
}

bool HasValidPackingSubtile(llvm::ArrayRef<xla::Tile> tiles, int64_t packing) {
  return packing > 0 &&
         ((tiles.size() == 1 && packing == 1) ||
          (tiles.size() == 2 && tiles[1] == xla::Tile({packing, 1})));
}

bool HasValidPacked2DTiling(llvm::ArrayRef<xla::Tile> tiles, int64_t packing,
                            std::optional<int64_t> target_lanes) {
  if (tiles.empty() || tiles.front().dimensions().size() != 2 ||
      !HasValidPackingSubtile(tiles, packing)) {
    return false;
  }
  const llvm::ArrayRef<int64_t> first = tiles.front().dimensions();
  return first[0] >= packing && first[0] % packing == 0 &&
         (target_lanes ? first[1] == *target_lanes : first[1] > 0);
}

bool StepsAcrossSublane(llvm::ArrayRef<int64_t> dims, int64_t match_dim_count,
                        int64_t lane_tile) {
  return dims.size() >= 2 &&
         (match_dim_count >= 2 || mlir::ShapedType::isDynamic(dims.back()) ||
          dims.back() > lane_tile ||
          llvm::any_of(dims.drop_back(2), [](int64_t dim) {
            return mlir::ShapedType::isDynamic(dim) || dim > 1;
          }));
}

llvm::SmallVector<int64_t> ComputeMatchDimensionTileStrides(
    llvm::ArrayRef<int64_t> shape, int64_t match_dim_count, int64_t tile_rows,
    int64_t lane_tile, std::optional<int64_t> lane_stride_override) {
  const int64_t rank = shape.size();
  CHECK_GE(match_dim_count, 1);
  CHECK_GE(rank, match_dim_count + 1);
  CHECK_GT(tile_rows, 0);
  CHECK_GT(lane_tile, 0);
  llvm::SmallVector<int64_t> tile_strides(rank, 1);
  int64_t stride = 1;
  auto assign_stride_and_advance = [&](int64_t dim, int64_t tile_size = 1) {
    tile_strides[dim] = stride;
    if (mlir::ShapedType::isDynamic(stride) ||
        mlir::ShapedType::isDynamic(shape[dim])) {
      stride = mlir::ShapedType::kDynamic;
    } else if (lane_stride_override && shape[dim] == 0) {
      stride = 0;
    } else {
      stride *= llvm::divideCeil(std::max<int64_t>(shape[dim], 1), tile_size);
    }
  };
  // Match dimension order places all `match_dim_count` sublane dimensions
  // contiguous within each lane chunk, followed by lane chunks, then outer
  // batch dimensions.
  const int64_t outermost_match_dim = rank - 1 - match_dim_count;
  for (int64_t dim = rank - 2; dim >= outermost_match_dim; --dim) {
    assign_stride_and_advance(dim, dim == rank - 2 ? tile_rows : 1);
  }
  if (lane_stride_override) {
    stride = *lane_stride_override;
    assign_stride_and_advance(rank - 1, lane_tile);
  } else if (mlir::ShapedType::isDynamic(shape[rank - 1]) ||
             shape[rank - 1] > lane_tile) {
    assign_stride_and_advance(rank - 1, lane_tile);
  }
  for (int64_t dim = outermost_match_dim - 1; dim >= 0; --dim) {
    assign_stride_and_advance(dim);
  }
  return tile_strides;
}

bool AreMatchDimensionTileStridesValid(llvm::ArrayRef<int64_t> dims,
                                       int64_t match_dim_count,
                                       int64_t tile_rows, int64_t lane_tile,
                                       llvm::ArrayRef<int64_t> tile_strides,
                                       bool require_expected_floor) {
  using mlir::ShapedType;
  const int64_t rank = dims.size();
  const bool has_non_positive_stride =
      llvm::any_of(tile_strides, [](int64_t stride) {
        return ShapedType::isStatic(stride) && stride < 1;
      });
  if (rank < 2 || match_dim_count < 1 || rank < match_dim_count + 1 ||
      tile_rows <= 0 || lane_tile <= 0 ||
      static_cast<int64_t>(tile_strides.size()) != rank ||
      has_non_positive_stride) {
    return false;
  }
  if (require_expected_floor &&
      !MeetsExpectedStrideFloor(dims, match_dim_count, tile_rows, lane_tile,
                                tile_strides)) {
    return false;
  }
  const int64_t lane_stride = tile_strides[rank - 1];
  const bool has_multiple_lanes =
      (require_expected_floor && ShapedType::isDynamic(dims[rank - 1])) ||
      dims[rank - 1] > lane_tile;
  const bool steps_across_lanes = has_multiple_lanes || lane_stride > 1;
  const bool has_multiple_sublane_tiles =
      ShapedType::isDynamic(dims[rank - 2]) || dims[rank - 2] > tile_rows;
  const bool requires_unit_second_minor_stride =
      !require_expected_floor || match_dim_count >= 2 ||
      (has_multiple_lanes && has_multiple_sublane_tiles);
  if (requires_unit_second_minor_stride && tile_strides[rank - 2] != 1) {
    return false;
  }
  auto count_dim_tiles = [&](int64_t dim) -> int64_t {
    if (ShapedType::isDynamic(dims[dim])) {
      return 0;
    }
    const int64_t tile_size =
        dim == rank - 2 ? tile_rows : (dim == rank - 1 ? lane_tile : 1);
    return llvm::divideCeil(std::max<int64_t>(dims[dim], 1), tile_size);
  };
  bool seen_dynamic_dim = false;
  auto is_valid_stride_step = [&](int64_t prev_stride, int64_t next_stride,
                                  int64_t min_factor = 1,
                                  int64_t max_factor = 0) {
    if (!require_expected_floor && seen_dynamic_dim &&
        ShapedType::isStatic(next_stride)) {
      return false;
    }
    if (ShapedType::isDynamic(prev_stride) ||
        ShapedType::isDynamic(next_stride)) {
      return true;
    }
    return next_stride >= prev_stride * std::max<int64_t>(1, min_factor) &&
           (max_factor <= 0 || next_stride <= prev_stride * max_factor);
  };
  const int64_t outermost_match_dim = rank - 1 - match_dim_count;
  const bool require_full_inner_step =
      require_expected_floor ? steps_across_lanes : has_multiple_lanes;
  for (int64_t dim = rank - 3; dim >= outermost_match_dim; --dim) {
    seen_dynamic_dim |= ShapedType::isDynamic(dims[dim + 1]);
    const int64_t tiles = count_dim_tiles(dim + 1);
    if ((tiles > 0 || !require_expected_floor) &&
        !is_valid_stride_step(tile_strides[dim + 1], tile_strides[dim],
                              require_full_inner_step ? tiles : 1, tiles)) {
      return false;
    }
  }
  seen_dynamic_dim |= ShapedType::isDynamic(dims[outermost_match_dim]);
  if (steps_across_lanes &&
      !is_valid_stride_step(
          tile_strides[outermost_match_dim], lane_stride,
          has_multiple_lanes ? count_dim_tiles(outermost_match_dim) : 1)) {
    return false;
  }
  if (ShapedType::isDynamic(dims[rank - 1])) {
    if (!require_expected_floor && seen_dynamic_dim &&
        ShapedType::isStatic(lane_stride)) {
      return false;
    }
    seen_dynamic_dim = true;
  }
  for (int64_t dim = outermost_match_dim - 1; dim >= 0; --dim) {
    const bool is_first_outer = dim == outermost_match_dim - 1;
    const int64_t prev_stride =
        is_first_outer ? lane_stride : tile_strides[dim + 1];
    const int64_t prev_dim = is_first_outer ? rank - 1 : dim + 1;
    const bool require_full_outer_step =
        has_multiple_lanes && (is_first_outer || require_expected_floor);
    if (!is_valid_stride_step(
            prev_stride, tile_strides[dim],
            require_full_outer_step ? count_dim_tiles(prev_dim) : 1) ||
        !is_valid_stride_step(lane_stride, tile_strides[dim])) {
      return false;
    }
    seen_dynamic_dim |= ShapedType::isDynamic(dims[dim]);
  }
  return true;
}

std::optional<int64_t> LargestDivisorMatching(
    int64_t max_align, int64_t unit,
    llvm::function_ref<bool(int64_t)> predicate) {
  if (unit <= 0 || max_align < unit) {
    return std::nullopt;
  }
  for (int64_t candidate = (max_align / unit) * unit; candidate >= unit;
       candidate -= unit) {
    if (max_align % candidate == 0 && predicate(candidate)) {
      return candidate;
    }
  }
  return std::nullopt;
}

std::optional<int64_t> GetRemainder(
    mlir::Value value, int64_t divisor, int64_t fuel,
    AssumeMultipleExtractor extract_assume_multiple) {
  CHECK_GT(divisor, 0);
  if (divisor == 1) {
    return 0;
  }
  if (fuel <= 0) {
    return std::nullopt;
  }
  if (auto rem =
          GetExplicitRemainder(value, divisor, fuel, extract_assume_multiple)) {
    return rem;
  }
  if (IsStructurallyDivisible(value, divisor, fuel, extract_assume_multiple) ==
      true) {
    return 0;
  }
  return std::nullopt;
}

std::optional<bool> IsDivisible(
    mlir::Value value, int64_t divisor, int64_t fuel,
    AssumeMultipleExtractor extract_assume_multiple) {
  CHECK_GT(divisor, 0);
  if (divisor == 1) {
    return true;
  }
  if (fuel <= 0) {
    return std::nullopt;
  }
  if (auto rem =
          GetExplicitRemainder(value, divisor, fuel, extract_assume_multiple)) {
    return *rem == 0;
  }
  return IsStructurallyDivisible(value, divisor, fuel, extract_assume_multiple);
}

mlir::LogicalResult VerifyTiledMemRefSlice(
    mlir::Operation* op, llvm::ArrayRef<int64_t> source_shape,
    llvm::ArrayRef<int64_t> slice_shape, llvm::ArrayRef<xla::Tile> tiles,
    mlir::ValueRange indices, mlir::ValueRange dynamic_sizes,
    AssumeMultipleExtractor extract_assume_multiple, int64_t fuel) {
  if (tiles.empty()) {
    return mlir::success();
  }
  const int64_t tile_rank = tiles.front().dimensions().size();
  const int64_t rank = source_shape.size();
  if (rank < tile_rank) {
    return op->emitOpError("Source memref has ")
           << rank << " dimensions, but the tiling has " << tile_rank
           << " dimensions. The memref must have at least as many dimensions "
              "as the tiling.";
  }
  const int64_t untiled_dims = rank - tile_rank;
  int64_t dynamic_size_idx = llvm::count_if(
      slice_shape.take_front(untiled_dims), mlir::ShapedType::isDynamic);
  const llvm::SmallVector<int64_t> unit_subtile = GetUnitSubtile(tiles);
  for (int64_t i = 0; i < tile_rank; ++i) {
    const int64_t dim = untiled_dims + i;
    const int64_t alignment = unit_subtile[i];
    if (!IsGuaranteedDivisible(indices[dim], alignment, fuel,
                               extract_assume_multiple)) {
      return op->emitOpError("Index in dimension ")
             << dim << " is not divisible by " << alignment << ".";
    }
    if (mlir::ShapedType::isDynamic(slice_shape[dim])) {
      if (!IsGuaranteedDivisible(dynamic_sizes[dynamic_size_idx++], alignment,
                                 fuel, extract_assume_multiple)) {
        return op->emitOpError("Dynamic slice size in dimension ")
               << dim << " is not divisible by " << alignment << ".";
      }
    } else if (slice_shape[dim] != source_shape[dim] &&
               slice_shape[dim] % alignment != 0) {
      return op->emitOpError("Static slice size in dimension ")
             << dim << " (" << slice_shape[dim] << ") is not divisible by "
             << alignment << ".";
    }
  }
  return mlir::success();
}

mlir::FailureOr<InferredTiledLayout> InferTiledMemRefReshapeLayout(
    llvm::ArrayRef<int64_t> input_shape, llvm::ArrayRef<int64_t> result_shape,
    llvm::ArrayRef<xla::Tile> input_tiles,
    llvm::ArrayRef<int64_t> input_strides, bool is_contiguous,
    llvm::function_ref<mlir::InFlightDiagnostic()> emit_error) {
  if (!is_contiguous) {
    if (auto layout = InferSecondMinorReshapeLayout(input_shape, result_shape,
                                                    input_tiles, input_strides,
                                                    /*is_contiguous=*/false)) {
      return *std::move(layout);
    }
    return emit_error()
           << "Not implemented: Reshape on a non-contiguous memref.";
  }
  if (input_tiles.empty()) {
    return InferredTiledLayout{{}, std::nullopt};
  }
  if (auto preserved = InferPreservedTileReshapeLayout(
          input_shape, result_shape, input_tiles)) {
    return *std::move(preserved);
  }
  // Power of two leading tiles preserve collapsed tile layouts first; other
  // leading tiles prefer reshaping the single second minor tile directly.
  const bool prefer_collapsed =
      llvm::isPowerOf2_64(input_tiles.front().dimensions().front());
  for (bool try_collapsed : {prefer_collapsed, !prefer_collapsed}) {
    std::optional<InferredTiledLayout> layout =
        try_collapsed ? InferCollapsedTileReshapeLayout(
                            input_shape, result_shape, input_tiles)
                      : InferSecondMinorReshapeLayout(
                            input_shape, result_shape, input_tiles,
                            input_strides, /*is_contiguous=*/true);
    if (layout.has_value()) {
      return *std::move(layout);
    }
  }
  return emit_error()
         << "Not implemented: Supported only when (a) tiled dimensions "
            "remain unchanged, except maybe for tile-aligned "
            "folding/unfolding of the first tiled dimension, or (b) the "
            "input layout has a single tile, under specific conditions.";
}

}  // namespace jaxlib::mosaic
