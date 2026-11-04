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

#include "xla/mosaic/dialect/tpu/tiled_layout_util.h"

#include <cstdint>
#include <numeric>

#include "absl/log/check.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "xla/layout.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

FailureOr<tpu::TiledLayoutAttr> getTiledLayout(Location loc,
                                               MemRefType memref_type) {
  if (auto tiled_layout =
          dyn_cast<tpu::TiledLayoutAttr>(memref_type.getLayout());
      tiled_layout != nullptr) {
    return tiled_layout;
  }
  if (memref_type.getLayout().isIdentity()) {
    return tpu::TiledLayoutAttr::getContiguous(
        memref_type.getContext(), /*tiles=*/{}, memref_type.getShape());
  }
  return emitError(loc, "Not implemented: Unsupported layout");
}

bool keepsMinorDimSequentialAndContiguous(ArrayRef<int64_t> shape,
                                          tpu::TiledLayoutAttr layout) {
  if (shape.empty()) {
    return true;
  }
  const ArrayRef<xla::Tile> tiles = layout.getTiles();
  const ArrayRef<int64_t> tile_strides = layout.getTileStrides();
  CHECK(!tile_strides.empty());
  const bool is_untiled = llvm::all_of(
      tiles, [](xla::Tile tile) { return tile.dimensions().empty(); });
  // The memref is untiled and has unit minor stride (or minor shape <= 1).
  if (is_untiled) {
    return tile_strides.back() == 1 ||
           (!ShapedType::isDynamic(shape.back()) && shape.back() <= 1);
  }
  if (tiles.size() == 1 && !tiles[0].dimensions().empty()) {
    const int64_t minor_tile_dim = tiles[0].dimensions().back();
    const int64_t minor_shape_dim = shape.back();
    // If the memref does not span more than one tile along the minormost
    // dimension, the minormost dimension is contiguous within that single tile
    // regardless of tile strides or other tile dimensions (e.g. 2D tiles).
    if (!ShapedType::isDynamic(minor_shape_dim) &&
        minor_shape_dim <= minor_tile_dim) {
      return true;
    }
    // The memref is tiled along the minormost dimension only with consecutive
    // tiles laid out adjacently (unit minor tile stride).
    const ArrayRef<int64_t> tile_dims(tiles[0].dimensions().data(),
                                      tiles[0].dimensions().size());
    const bool is_single_1d_tile =
        llvm::all_of(tile_dims.drop_back(), llvm::equal_to<int64_t>(1));
    return is_single_1d_tile && tile_strides.back() == 1;
  }
  return false;
}

void tryScaleDownToAlignedSubtile(SmallVectorImpl<int64_t>& tile,
                                  const ArrayRef<int64_t> unit_subtile,
                                  const int64_t alignment_elements) {
  CHECK_GT(alignment_elements, 0);

  const int64_t tile_rank = tile.size();
  CHECK_EQ(tile_rank, unit_subtile.size());

  const int64_t tile_elements = llvm::product_of(tile);
  if (tile_elements % alignment_elements != 0) {
    return;
  }

  int64_t curr_alignment_units = tile_elements / alignment_elements;

  for (int64_t i = 0; i < tile_rank; ++i) {
    CHECK_EQ(tile[i] % unit_subtile[i], 0);
    const int64_t ratio = tile[i] / unit_subtile[i];
    const int64_t gcd = std::gcd(ratio, curr_alignment_units);
    tile[i] /= gcd;
    curr_alignment_units /= gcd;
    if (gcd != ratio) {
      break;
    }
  }
}

}  // namespace mlir::tpu
