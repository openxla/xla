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

#include "xla/mosaic/dialect/tpu/tpu_dialect_impl.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/layout.h"
#include "xla/mosaic/util.h"

namespace mlir::tpu::impl {

void printTiledLayout(ArrayRef<xla::Tile> tiles, ArrayRef<int64_t> tile_strides,
                      AsmPrinter& printer) {
  printer << '<';
  for (const xla::Tile& tile : tiles) {
    printer << tile.ToString();
  }
  printer << ",[";
  for (int i = 0; i < tile_strides.size(); ++i) {
    if (i > 0) {
      printer << ',';
    }
    int64_t stride = tile_strides[i];
    if (ShapedType::isDynamic(stride)) {
      printer << "?";
    } else {
      printer << stride;
    }
  }
  printer << "]>";
}

LogicalResult parseTiledLayout(AsmParser& parser,
                               SmallVectorImpl<xla::Tile>& tiles,
                               SmallVectorImpl<int64_t>& tile_strides) {
  if (failed(parser.parseLess())) {
    return failure();
  }
  int64_t size;
  while (succeeded(parser.parseOptionalLParen())) {
    xla::Tile& tile = tiles.emplace_back();
    bool first = true;
    while (!succeeded(parser.parseOptionalRParen())) {
      if (!first) {
        if (failed(parser.parseComma())) {
          return failure();
        }
      }
      first = false;
      if (failed(parser.parseInteger(size))) {
        return failure();
      }
      tile.add_dimensions(size);
    }
  }
  int64_t stride;
  if (failed(parser.parseComma())) {
    return failure();
  }
  if (succeeded(parser.parseOptionalLSquare())) {
    bool first = true;
    while (!succeeded(parser.parseOptionalRSquare())) {
      if (!first) {
        if (failed(parser.parseComma())) {
          return failure();
        }
      }
      first = false;
      if (succeeded(parser.parseOptionalQuestion())) {
        stride = ShapedType::kDynamic;
      } else if (failed(parser.parseInteger(stride))) {
        return failure();
      }
      tile_strides.push_back(stride);
    }
  } else {
    return failure();
  }
  if (failed(parser.parseGreater())) {
    return failure();
  }
  return success();
}

AffineMap getTiledLayoutAffineMap(MLIRContext* context,
                                  ArrayRef<xla::Tile> tiles,
                                  ArrayRef<int64_t> tile_strides) {
  const int64_t rank = tile_strides.size();
  SmallVector<AffineExpr, 8> exprs;
  for (int64_t i = 0; i < rank; ++i) {
    exprs.push_back(getAffineDimExpr(i, context));
  }
  for (const xla::Tile& tile : tiles) {
    SmallVector<AffineExpr, 8> new_exprs;
    auto dimensions = tile.dimensions();
    int64_t untiled_rank = exprs.size() - dimensions.size();
    CHECK_GE(untiled_rank, 0);
    for (int64_t i = 0; i < untiled_rank; ++i) {
      new_exprs.push_back(exprs[i]);
    }
    for (int64_t i = 0; i < dimensions.size(); ++i) {
      new_exprs.push_back(exprs[untiled_rank + i].floorDiv(dimensions[i]));
    }
    for (int64_t i = 0; i < dimensions.size(); ++i) {
      new_exprs.push_back(exprs[untiled_rank + i] % dimensions[i]);
    }
    exprs = std::move(new_exprs);
  }
  int64_t num_symbols = 0;
  AffineExpr result = getAffineConstantExpr(0, context);
  FailureOr<SmallVector<int64_t>> expanded_strides =
      getExpandedStrides(tiles, tile_strides);
  CHECK(succeeded(expanded_strides));
  SmallVector<int64_t> strides = *std::move(expanded_strides);
  CHECK_EQ(strides.size(), exprs.size());
  for (int64_t i = 0; i < exprs.size(); ++i) {
    AffineExpr stride_expr = ShapedType::isDynamic(strides[i])
                                 ? getAffineSymbolExpr(num_symbols++, context)
                                 : getAffineConstantExpr(strides[i], context);
    result = result + exprs[i] * stride_expr;
  }
  return AffineMap::get(rank, num_symbols, result);
}

int64_t getUntiledRank(ArrayRef<xla::Tile> tiles, const int64_t rank) {
  // Note: This implementation does not assume there is no nested tiling across
  // the first level of tiling, though this is enforced by the verifier.
  int64_t untiled_rank = rank;
  int64_t tiled_rank = rank;
  for (const xla::Tile& tile : tiles) {
    const int64_t tile_ndims = tile.dimensions().size();
    untiled_rank = std::min(untiled_rank, tiled_rank - tile_ndims);
    tiled_rank += tile_ndims;
  }
  return untiled_rank;
}

FailureOr<SmallVector<int64_t>> getExpandedShape(
    const ArrayRef<int64_t> untiled_shape, const ArrayRef<xla::Tile> tiles,
    const bool require_alignment) {
  SmallVector<int64_t> shape(untiled_shape);
  for (const xla::Tile& tile : tiles) {
    const int64_t tile_ndims = tile.dimensions().size();
    const llvm::ArrayRef<int64_t> tiled_shape =
        llvm::ArrayRef(shape).take_back(tile_ndims);
    llvm::SmallVector<int64_t> new_tiled_shape(2 * tile_ndims);
    for (int64_t i = 0; i < tile_ndims; ++i) {
      if (require_alignment && (ShapedType::isDynamic(tiled_shape[i]) ||
                                tiled_shape[i] % tile.dimension(i) != 0)) {
        return failure();
      }
      if (ShapedType::isDynamic(tiled_shape[i])) {
        new_tiled_shape[i] = ShapedType::kDynamic;
      } else {
        new_tiled_shape[i] =
            llvm::divideCeil(tiled_shape[i], tile.dimension(i));
      }
      new_tiled_shape[tile_ndims + i] = tile.dimension(i);
    }
    shape.pop_back_n(tile_ndims);
    shape.append(new_tiled_shape);
  }
  return shape;
}

SmallVector<int64_t> getContiguousTileStrides(const ArrayRef<xla::Tile> tiles,
                                              const ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size());
  int64_t stride = 1;
  const xla::Tile* const first_tile = tiles.empty() ? nullptr : &tiles.front();
  const int64_t first_tile_rank =
      first_tile == nullptr ? 0 : first_tile->dimensions().size();
  for (int64_t d = shape.size() - 1; d >= 0; --d) {
    strides[d] = stride;
    if (ShapedType::isDynamic(shape[d]) || ShapedType::isDynamic(stride)) {
      stride = ShapedType::kDynamic;
      continue;
    }
    if (d >= shape.size() - first_tile_rank) {
      assert(first_tile != nullptr);
      const int64_t tile_d = d - (shape.size() - first_tile_rank);
      stride *= llvm::divideCeil(shape[d], first_tile->dimension(tile_d));
    } else {
      stride *= shape[d];
    }
  }
  return strides;
}

int64_t getNumTrailingDimsWithContiguousTiles(
    const ArrayRef<xla::Tile> tiles, const ArrayRef<int64_t> tile_strides,
    const ArrayRef<int64_t> shape) {
  int64_t stride = 1;
  const xla::Tile* const first_tile = tiles.empty() ? nullptr : &tiles.front();
  const int64_t first_tile_rank =
      first_tile == nullptr ? 0 : first_tile->dimensions().size();
  int64_t d = shape.size() - 1;
  for (; d >= 0; --d) {
    int64_t size_tiles;
    if (d >= shape.size() - first_tile_rank &&
        shape[d] != ShapedType::kDynamic) {
      assert(first_tile != nullptr);
      const int64_t tile_d = d - (shape.size() - first_tile_rank);
      size_tiles = llvm::divideCeil(shape[d], first_tile->dimension(tile_d));
    } else {
      size_tiles = shape[d];
    }

    // Dynamic strides are always considered to be a mismatch as they could
    // potentially not be contiguous.
    bool mismatch_stride =
        stride != tile_strides[d] || ShapedType::isDynamic(tile_strides[d]);
    // Dimensions with only one element/tile can have any stride.
    if (mismatch_stride && size_tiles != 1) {
      break;
    }
    if (stride == ShapedType::kDynamic || size_tiles == ShapedType::kDynamic) {
      stride = ShapedType::kDynamic;
    } else {
      stride *= size_tiles;
    }
  }
  return shape.size() - 1 - d;
}

FailureOr<SmallVector<int64_t>> getExpandedStrides(
    ArrayRef<xla::Tile> tiles, ArrayRef<int64_t> tile_strides) {
  if (tiles.empty()) {
    return SmallVector<int64_t>(tile_strides);
  }
  SmallVector<int64_t> strides(tile_strides);
  // Expand front tile. A nested padding tile (if any) pads the first tile.
  const xla::Tile& first_tile = tiles.front();
  FAILUREOR_ASSIGN_OR_RETURN(
      SmallVector<int64_t> expanded_tile,
      getExpandedShape(first_tile.dimensions(), tiles.drop_front(),
                       /*require_alignment=*/!hasNestedPadding(tiles)));
  const int64_t rank = tile_strides.size();
  strides.resize_for_overwrite(rank + expanded_tile.size());
  // Tile strides are in units of the physical (possibly padded) tile size.
  const int64_t physical_tile_size = llvm::product_of(expanded_tile);
  int64_t tile_size = 1;
  for (int64_t d = strides.size() - 1; d >= 0; --d) {
    if (d >= rank) {
      const int64_t new_stride = tile_size;
      tile_size *= expanded_tile[d - rank];
      strides[d] = new_stride;
    } else {
      if (ShapedType::isStatic(strides[d])) {
        strides[d] *= physical_tile_size;
      }
    }
  }
  return strides;
}

bool isPaddingTile(const xla::Tile& prev_tile, const xla::Tile& tile) {
  const ArrayRef<int64_t> prev_dims(prev_tile.dimensions().begin(),
                                    prev_tile.dimensions().end());
  const ArrayRef<int64_t> dims(tile.dimensions().begin(),
                               tile.dimensions().end());
  if (prev_dims.empty() || prev_dims.size() != dims.size()) {
    return false;
  }
  return prev_dims.drop_back() == dims.drop_back() &&
         dims.back() == prev_dims.back() + 1;
}

bool hasNestedPadding(const ArrayRef<xla::Tile> tiles) {
  return tiles.size() >= 2 && isPaddingTile(tiles[0], tiles[1]);
}

LogicalResult verifyTiledLayout(function_ref<InFlightDiagnostic()> emitError,
                                const llvm::ArrayRef<xla::Tile> tiles,
                                const llvm::ArrayRef<int64_t> tile_strides,
                                const bool allow_nested_padding) {
  if (tiles.empty()) {
    return success();
  }
  const int64_t rank = tile_strides.size();
  const xla::Tile& first_tile = tiles.front();
  const int64_t first_tile_rank = first_tile.dimensions().size();
  if (first_tile_rank > rank) {
    return emitError() << "First tile rank cannot exceed the length of strides "
                          "(the memref rank)";
  }
  // The interpretation of tile strides is unclear if there is nested tiling
  // across first tiles (e.g. T(8, 128)(2, 4, 64)), and this has no applications
  // anyway.
  if (getUntiledRank(tiles, rank) != rank - first_tile_rank) {
    return emitError() << "Not implemented: Nested tiling across first tiles";
  }
  // If allowed, the second tile may pad the first tile. All other nested tiles
  // must evenly divide the (padded) tile before them (so they don't add any
  // padding or change the tile size).
  if (allow_nested_padding && hasNestedPadding(tiles)) {
    if (failed(getExpandedShape(tiles[1].dimensions(), tiles.drop_front(2),
                                /*require_alignment=*/true))) {
      return emitError() << "Not implemented: Nested tiles must evenly divide "
                         << "the padding tile " << tiles[1].ToString()
                         << " but they do not (would add padding)";
    }
    return success();
  }
  if (failed(getExpandedShape(first_tile.dimensions(), tiles.drop_front(),
                              /*require_alignment=*/true))) {
    InFlightDiagnostic diag = emitError();
    diag << "Not implemented: Nested tiles must evenly divide the first tile "
         << first_tile.ToString() << " but they do not (would add padding)";
    if (allow_nested_padding) {
      diag << ". Only a second tile that pads the minor dimension of the "
              "first tile by one element is supported";
    }
    return diag;
  }
  return success();
}

LogicalResult verifyTiledLayoutRank(
    const int64_t layout_rank, ArrayRef<int64_t> shape,
    function_ref<InFlightDiagnostic()> emitError) {
  if (layout_rank != shape.size()) {
    return emitError() << "Layout rank does not match shape rank";
  }
  return success();
}

}  // namespace mlir::tpu::impl
