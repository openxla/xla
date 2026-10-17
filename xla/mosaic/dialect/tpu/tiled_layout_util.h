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

#ifndef XLA_MOSAIC_DIALECT_TPU_TILED_LAYOUT_UTIL_H_
#define XLA_MOSAIC_DIALECT_TPU_TILED_LAYOUT_UTIL_H_

#include <cstdint>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

// Returns the tiled layout of `memref_type`. Memrefs with an identity layout
// are described by an equivalent untiled layout. Returns an error if the layout
// is neither identity nor tiled.
FailureOr<tpu::TiledLayoutAttr> getTiledLayout(Location loc,
                                               MemRefType memref_type);

// Returns true if the minormost dimension of a memref with the given shape and
// tiled layout is both sequential (successive elements along that dimension
// have a physical stride of 1) and contiguous (no intervening padding or
// non-adjacent runs).
//
// This is best-effort and might return false negatives.
bool keepsMinorDimSequentialAndContiguous(ArrayRef<int64_t> shape,
                                          tpu::TiledLayoutAttr layout);

// Scales down a tile until its number of elements is the smallest possible
// multiple of `alignment_elements`. If the tile's number of elements is not
// divisible by `alignment_elements` in the first place, it is left unchanged.
void tryScaleDownToAlignedSubtile(SmallVectorImpl<int64_t>& tile,
                                  ArrayRef<int64_t> unit_subtile,
                                  int64_t alignment_elements);

}  // namespace mlir::tpu

#endif  // XLA_MOSAIC_DIALECT_TPU_TILED_LAYOUT_UTIL_H_
