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

#ifndef XLA_MOSAIC_TILED_LAYOUT_UTIL_H_
#define XLA_MOSAIC_TILED_LAYOUT_UTIL_H_

#include <cstdint>
#include <optional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/layout.h"

namespace jaxlib::mosaic {

// Extracts `(operand, multiple)` when `op` is a dialect assume_multiple op.
using AssumeMultipleExtractor =
    llvm::function_ref<std::optional<std::pair<::mlir::Value, int64_t>>(
        ::mlir::Operation*)>;

struct InferredTiledLayout {
  llvm::SmallVector<xla::Tile> tiles;
  // Empty when the inferred layout uses default contiguous tile strides.
  std::optional<llvm::SmallVector<int64_t>> tile_strides;
};

// Returns the minimum divisible subtile size across each dimension of the
// leading tile after accounting for all nested subtiles.
llvm::SmallVector<int64_t> GetUnitSubtile(llvm::ArrayRef<xla::Tile> tiles);

// Returns true if `tiles` ends with the expected `(packing, 1)` subtile (or no
// subtile when `packing == 1`).
bool HasValidPackingSubtile(llvm::ArrayRef<xla::Tile> tiles, int64_t packing);

// Returns true if `tiles` describes a packed 2D tile whose rows are divisible
// by `packing` and whose lane dimension matches `target_lanes` (if provided).
bool HasValidPacked2DTiling(llvm::ArrayRef<xla::Tile> tiles, int64_t packing,
                            std::optional<int64_t> target_lanes = std::nullopt);

// Returns true if `dims` spans multiple sublane blocks via folded match
// dimensions, multiple lane chunks, or outer batch dimensions larger than 1.
bool StepsAcrossSublane(llvm::ArrayRef<int64_t> dims, int64_t match_dim_count,
                        int64_t lane_tile);

// Computes tile strides in units of `(tile_rows, lane_tile)` for a shape whose
// `match_dim_count` sublane dimensions precede lane chunks in memory.
llvm::SmallVector<int64_t> ComputeMatchDimensionTileStrides(
    llvm::ArrayRef<int64_t> shape, int64_t match_dim_count, int64_t tile_rows,
    int64_t lane_tile,
    std::optional<int64_t> lane_stride_override = std::nullopt);

// Returns true if `tile_strides` satisfies monotonic stride ordering for a
// match dimension layout on `dims`. When `require_expected_floor` is true, each
// static stride must also meet the canonical contiguous stride.
bool AreMatchDimensionTileStridesValid(llvm::ArrayRef<int64_t> dims,
                                       int64_t match_dim_count,
                                       int64_t tile_rows, int64_t lane_tile,
                                       llvm::ArrayRef<int64_t> tile_strides,
                                       bool require_expected_floor = false);

// Returns the largest multiple of `unit` dividing `max_align` that satisfies
// `predicate`, or `std::nullopt` if none matches.
std::optional<int64_t> LargestDivisorMatching(
    int64_t max_align, int64_t unit,
    llvm::function_ref<bool(int64_t)> predicate);

// Returns `value mod divisor` in `[0, divisor)` when statically provable within
// `fuel` recursive analysis steps.
std::optional<int64_t> GetRemainder(
    ::mlir::Value value, int64_t divisor, int64_t fuel,
    AssumeMultipleExtractor extract_assume_multiple);

// Returns whether `value` is divisible by `divisor`, or `std::nullopt` when
// divisibility cannot be determined within `fuel` steps.
std::optional<bool> IsDivisible(
    ::mlir::Value value, int64_t divisor, int64_t fuel,
    AssumeMultipleExtractor extract_assume_multiple);

inline bool IsGuaranteedDivisible(
    ::mlir::Value value, int64_t divisor, int64_t fuel,
    AssumeMultipleExtractor extract_assume_multiple) {
  return IsDivisible(value, divisor, fuel, extract_assume_multiple)
      .value_or(false);
}

// Verifies that all tiled slice indices and sizes align to `tiles`'s unit
// subtile.
::mlir::LogicalResult VerifyTiledMemRefSlice(
    ::mlir::Operation* op, llvm::ArrayRef<int64_t> source_shape,
    llvm::ArrayRef<int64_t> slice_shape, llvm::ArrayRef<xla::Tile> tiles,
    ::mlir::ValueRange indices, ::mlir::ValueRange dynamic_sizes,
    AssumeMultipleExtractor extract_assume_multiple, int64_t fuel = 128);

// Infers the result tiled layout for a zero copy memref reshape, emitting a
// diagnostic via `emit_error` when the reshape cannot be represented.
::mlir::FailureOr<InferredTiledLayout> InferTiledMemRefReshapeLayout(
    llvm::ArrayRef<int64_t> input_shape, llvm::ArrayRef<int64_t> result_shape,
    llvm::ArrayRef<xla::Tile> input_tiles,
    llvm::ArrayRef<int64_t> input_strides, bool is_contiguous,
    llvm::function_ref<::mlir::InFlightDiagnostic()> emit_error);

}  // namespace jaxlib::mosaic

#endif  // XLA_MOSAIC_TILED_LAYOUT_UTIL_H_
