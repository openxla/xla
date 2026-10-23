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

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "triton/Dialect/Triton/IR/Dialect.h"
// Above header needs to be included first to avoid 'major' macro collision.

#include "absl/algorithm/container.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLACOLLAPSECONTIGUOUSMINORDIMSPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// Return offset_m1 rescaled by the size of m0 dimension.
//
// `minor` is the minormost dimension. Its offset must be 0.
// `major` is the next-to-minormost dimension. Its offset needs to be scaled by
// `minor`'s size.
OpFoldResult ComputeMergedOffset(PatternRewriter& rewriter, Location loc,
                                 OpFoldResult offset_major,
                                 int64_t src_shape_minor) {
  if (std::optional<int64_t> c_offset_major =
          mlir::getConstantIntValue(offset_major);
      c_offset_major.has_value() && *c_offset_major >= 0) {
    return rewriter.getIndexAttr(*c_offset_major * src_shape_minor);
  }

  if (auto val = llvm::dyn_cast_if_present<Value>(offset_major)) {
    if (auto apply_indexing = val.getDefiningOp<::xla::ApplyIndexingOp>()) {
      OpResult op_result = mlir::cast<OpResult>(val);
      unsigned result_idx = op_result.getResultNumber();
      ::xla::IndexingMap indexing_map = apply_indexing.getIndexingMap();
      const ::xla::SymbolicMap& symbolic_map = indexing_map.GetSymbolicMap();

      ::xla::SymbolicExpr scaled_result =
          symbolic_map.GetResult(result_idx) * src_shape_minor;

      ::xla::SymbolicMap new_symbolic_map = ::xla::SymbolicMap::Get(
          rewriter.getContext(), symbolic_map.GetNumDims(),
          symbolic_map.GetNumSymbols(), {scaled_result});
      ::xla::IndexingMap new_indexing_map(
          new_symbolic_map, indexing_map.GetDimVars(),
          indexing_map.GetRangeVars(), indexing_map.GetRTVars(),
          indexing_map.GetSymbolicConstraints());
      new_indexing_map.Simplify();
      ::xla::ApplyIndexingOp new_indexing_op = ::xla::ApplyIndexingOp::create(
          rewriter, loc, apply_indexing.getOperands(), new_indexing_map);
      return new_indexing_op.getResult(0);
    }
  }
  Value val_major =
      getValueOrCreateConstantIndexOp(rewriter, loc, offset_major);
  Value scale = arith::ConstantIndexOp::create(rewriter, loc, src_shape_minor);
  return arith::MulIOp::create(rewriter, loc, val_major, scale).getResult();
}

struct CollapsedOpData {
  SmallVector<int64_t> shape;
  SmallVector<int64_t> layout;
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> strides;
  SmallVector<OpFoldResult> offsets;
  // For each dimension of the collapsed op, stores the original logical
  // dimensions that were merged into it in major-to-minor order.
  SmallVector<SmallVector<int64_t>> dim_groups;
};

// If the minor-most dimensions are contiguous in memory, returns the shape with
// minor dimensions collapsed into one. Otherwise, returns nullopt.
//
// Triton doesn't implement emitting vectorized loads/stores that span across
// dimensions. This prevents tensors with small minor-most dimensions from using
// fully vectorized loads/stores. For example, a f32[1024, 2]{1, 0} tensor will
// only use 64-bit wide loads/stores even though memory layout would allow for
// 128-bit ones.
std::optional<CollapsedOpData> CollapseContiguousMinorDims(
    PatternRewriter& rewriter, Location loc, ArrayRef<int64_t> orig_shape,
    ArrayRef<int64_t> orig_layout, ArrayRef<int64_t> orig_sizes,
    ArrayRef<int64_t> orig_strides, ArrayRef<OpFoldResult> orig_offsets,
    RankedTensorType tile_type) {
  Type elem_type = tile_type.getElementType();
  if (!elem_type.isIntOrFloat()) {
    return std::nullopt;
  }

  if (!llvm::isPowerOf2_64(tile_type.getNumElements())) {
    return std::nullopt;
  }
  int64_t bit_width = elem_type.getIntOrFloatBitWidth();

  SmallVector<int64_t> shape = to_vector(orig_shape);
  SmallVector<int64_t> layout = to_vector(orig_layout);
  SmallVector<int64_t> sizes = to_vector(orig_sizes);
  SmallVector<int64_t> strides = to_vector(orig_strides);
  SmallVector<OpFoldResult> offsets = to_vector(orig_offsets);
  SmallVector<SmallVector<int64_t>> dim_groups(orig_shape.size());
  for (int64_t i = 0; i < static_cast<int64_t>(orig_shape.size()); ++i) {
    dim_groups[i] = {i};
  }

  bool collapsed_any = false;
  for (int64_t current_rank = shape.size(); current_rank >= 2; --current_rank) {
    // We mutate layout, shape, sizes, strides, and offsets in place, so [0] and
    // [1] are the two minormost dimensions that are not collapsed yet.
    int64_t m0 = layout[0];
    int64_t m1 = layout[1];

    // Only collapse if doing so allows for wider load/store. SASS has
    // vectorized load/store instructions up to 128 bits.
    static constexpr int64_t kMaxVectorizedOpBitWidth = 128;
    if (sizes[m0] * bit_width >= kMaxVectorizedOpBitWidth) {
      break;
    }

    // We can only collapse if the elements are contiguous in memory.
    // * The minor-most dimension must be fully covered by a single tile.
    if (sizes[m0] != shape[m0]) {
      break;
    }
    // * The offset must be zero.
    if (!mlir::isZeroInteger(offsets[m0])) {
      break;
    }
    // * Strides must be 1.
    if (strides[m0] != 1 || strides[m1] != 1) {
      break;
    }

    // Overflow checks for merged dimension and offset: bail if either exceeds
    // INT32_MAX.
    if (shape[m0] <= 0 || shape[m1] <= 0 ||
        shape[m1] > std::numeric_limits<int32_t>::max() / shape[m0]) {
      break;
    }
    int64_t merged_shape = shape[m1] * shape[m0];
    if (merged_shape > std::numeric_limits<int32_t>::max()) {
      break;
    }
    if (std::optional<int64_t> c1 = mlir::getConstantIntValue(offsets[m1])) {
      if (*c1 < 0 || *c1 > std::numeric_limits<int32_t>::max() / shape[m0]) {
        break;
      }
    }
    if (auto val = llvm::dyn_cast_if_present<Value>(offsets[m1])) {
      if (auto apply_indexing = val.getDefiningOp<::xla::ApplyIndexingOp>()) {
        OpResult op_result = mlir::cast<OpResult>(val);
        unsigned result_idx = op_result.getResultNumber();
        const ::xla::IndexingMap& indexing_map =
            apply_indexing.getIndexingMap();
        ::xla::Interval range =
            indexing_map.GetRangeEvaluator().ComputeExpressionRange(
                indexing_map.GetSymbolicMap().GetResult(result_idx));
        if (range.lower < 0 ||
            range.upper > std::numeric_limits<int32_t>::max() / shape[m0]) {
          break;
        }
      }
    }

    // Collapse m1 (major) and m0 (minor) into m1's position, then erase m0.
    int64_t merged_size = sizes[m1] * sizes[m0];
    OpFoldResult merged_offset =
        ComputeMergedOffset(rewriter, loc, offsets[m1], shape[m0]);

    shape[m1] = merged_shape;
    sizes[m1] = merged_size;
    offsets[m1] = merged_offset;
    dim_groups[m1].append(dim_groups[m0].begin(), dim_groups[m0].end());

    shape.erase(shape.begin() + m0);
    sizes.erase(sizes.begin() + m0);
    strides.erase(strides.begin() + m0);
    offsets.erase(offsets.begin() + m0);
    dim_groups.erase(dim_groups.begin() + m0);

    layout.erase(layout.begin());
    for (int64_t& d : layout) {
      if (d > m0) {
        --d;
      }
    }

    collapsed_any = true;
  }

  if (!collapsed_any) {
    return std::nullopt;
  }

  return CollapsedOpData{
      std::move(shape),   std::move(layout),  std::move(sizes),
      std::move(strides), std::move(offsets), std::move(dim_groups),
  };
}

struct CollapsedTensorMapping {
  // Axes of `orig_tensor_shape` permuted so that collapsed dimensions are
  // adjacent and in major-to-minor order.
  SmallVector<int64_t> reshaped_shape;
  // `reshaped_shape` with contiguous minor-most dimensions collapsed.
  SmallVector<int64_t> collapsed_tensor_shape;
  // Permutation mapping axes of `reshaped_shape` to `orig_tensor_shape`.
  SmallVector<int32_t> reshaped_to_orig_order;
};

// Returns the layout and shape information needed to emit the collapse ops.
//
// In case of non-default layout, we need to transpose the original tensor
// (orig_tensor_shape -> reshaped_shape) so that collapsed dimensions are
// adjacent and in major-to-minor order to enable collapsing reshape
// (reshaped_shape -> collapsed_tensor_shape). When the collapse merges a
// dimension into a rank-reduced one, we may need a transpose without the
// reshape to restore the original dimension order.
//
// Preserves rank-reduced unit dimensions from the original op.
std::optional<CollapsedTensorMapping> GetCollapsedTensorMapping(
    ArrayRef<int64_t> orig_sizes, ArrayRef<int64_t> collapsed_sizes,
    ArrayRef<SmallVector<int64_t>> collapsed_dim_groups,
    ArrayRef<int64_t> orig_tensor_shape) {
  std::optional<llvm::SmallDenseSet<unsigned>> reduced_dims =
      mlir::computeRankReductionMask(orig_sizes, orig_tensor_shape);
  if (!reduced_dims.has_value()) {
    return std::nullopt;
  }

  SmallVector<int64_t> collapsed_tensor_shape;
  SmallVector<int64_t> reshaped_shape;
  SmallVector<int64_t> reshaped_dims;
  for (int64_t g = 0; g < static_cast<int64_t>(collapsed_sizes.size()); ++g) {
    // Exclude reduced dimensions from the
    // reshaped_shape/collapsed_tensor_shape.
    bool group_retained = false;
    for (int64_t orig_dim : collapsed_dim_groups[g]) {
      if (!reduced_dims->contains(orig_dim)) {
        reshaped_shape.push_back(orig_sizes[orig_dim]);
        reshaped_dims.push_back(orig_dim);
        group_retained = true;
      }
    }
    if (group_retained) {
      collapsed_tensor_shape.push_back(collapsed_sizes[g]);
    }
  }

  if (collapsed_tensor_shape.empty() && !collapsed_sizes.empty()) {
    collapsed_tensor_shape.push_back(collapsed_sizes.back());
    reshaped_shape.push_back(collapsed_sizes.back());
    reshaped_dims.push_back(collapsed_dim_groups.back().front());
  }

  SmallVector<int32_t> reshaped_to_orig_order(reshaped_dims.size());
  absl::c_iota(reshaped_to_orig_order, 0);
  absl::c_sort(reshaped_to_orig_order, [&](int32_t a, int32_t b) {
    return reshaped_dims[a] < reshaped_dims[b];
  });

  return CollapsedTensorMapping{
      /*reshaped_shape=*/std::move(reshaped_shape),
      /*collapsed_tensor_shape=*/std::move(collapsed_tensor_shape),
      /*reshaped_to_orig_order=*/std::move(reshaped_to_orig_order),
  };
}

class CollapseExtractOp : public mlir::OpRewritePattern<ExtractOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractOp op,
                                PatternRewriter& rewriter) const override {
    std::optional<CollapsedOpData> collapsed_data = CollapseContiguousMinorDims(
        rewriter, op.getLoc(), op.getSrcShape(), op.getSrcLayout(),
        op.getStaticSizes(), op.getStaticStrides(), op.getMixedOffsets(),
        op.getType());
    if (!collapsed_data.has_value()) {
      return failure();
    }

    std::optional<CollapsedTensorMapping> mapping = GetCollapsedTensorMapping(
        op.getStaticSizes(), collapsed_data->sizes, collapsed_data->dim_groups,
        op.getType().getShape());
    if (!mapping.has_value()) {
      return failure();
    }
    RankedTensorType collapsed_type = RankedTensorType::get(
        mapping->collapsed_tensor_shape, op.getType().getElementType());

    ExtractOp new_extract = ExtractOp::create(
        rewriter, op.getLoc(), collapsed_type, op.getSrc(),
        collapsed_data->offsets, collapsed_data->sizes, collapsed_data->strides,
        collapsed_data->shape, collapsed_data->layout);
    new_extract->setDiscardableAttrs(op->getDiscardableAttrDictionary());

    Value result = new_extract.getResult();
    if (mapping->collapsed_tensor_shape != mapping->reshaped_shape) {
      result = triton::ReshapeOp::create(rewriter, op.getLoc(),
                                         mapping->reshaped_shape, result,
                                         /*allowReorder=*/false);
    }
    if (!absl::c_is_sorted(mapping->reshaped_to_orig_order)) {
      result = triton::TransOp::create(rewriter, op.getLoc(), result,
                                       mapping->reshaped_to_orig_order);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class CollapseInsertOp : public mlir::OpRewritePattern<InsertOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp op,
                                PatternRewriter& rewriter) const override {
    RankedTensorType src_type =
        mlir::cast<RankedTensorType>(op.getSrc().getType());
    std::optional<CollapsedOpData> collapsed_data = CollapseContiguousMinorDims(
        rewriter, op.getLoc(), op.getDstShape(), op.getDstLayout(),
        op.getStaticSizes(), op.getStaticStrides(), op.getMixedOffsets(),
        src_type);
    if (!collapsed_data.has_value()) {
      return failure();
    }

    std::optional<CollapsedTensorMapping> mapping = GetCollapsedTensorMapping(
        op.getStaticSizes(), collapsed_data->sizes, collapsed_data->dim_groups,
        src_type.getShape());
    if (!mapping.has_value()) {
      return failure();
    }
    Value new_src = op.getSrc();
    if (!absl::c_is_sorted(mapping->reshaped_to_orig_order)) {
      SmallVector<int32_t> orig_to_reshaped_order(
          mapping->reshaped_to_orig_order.size());
      for (int32_t i = 0;
           i < static_cast<int32_t>(mapping->reshaped_to_orig_order.size());
           ++i) {
        orig_to_reshaped_order[mapping->reshaped_to_orig_order[i]] = i;
      }
      new_src = triton::TransOp::create(rewriter, op.getLoc(), new_src,
                                        orig_to_reshaped_order);
    }
    if (mapping->collapsed_tensor_shape != mapping->reshaped_shape) {
      new_src = triton::ReshapeOp::create(
          rewriter, op.getLoc(), mapping->collapsed_tensor_shape, new_src,
          /*allowReorder=*/false);
    }

    InsertOp new_insert = InsertOp::create(
        rewriter, op.getLoc(), new_src, op.getDst(), collapsed_data->offsets,
        collapsed_data->sizes, collapsed_data->strides, collapsed_data->shape,
        collapsed_data->layout);
    new_insert->setDiscardableAttrs(op->getDiscardableAttrDictionary());

    rewriter.eraseOp(op);
    return success();
  }
};

class TritonXLACollapseContiguousMinorDimsPass
    : public impl::TritonXLACollapseContiguousMinorDimsPassBase<
          TritonXLACollapseContiguousMinorDimsPass> {
 public:
  using TritonXLACollapseContiguousMinorDimsPassBase::
      TritonXLACollapseContiguousMinorDimsPassBase;

  void runOnOperation() override {
    mlir::MLIRContext* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<CollapseExtractOp, CollapseInsertOp>(ctx);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::triton::xla
