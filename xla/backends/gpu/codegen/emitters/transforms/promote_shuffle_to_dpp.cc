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

#include <cstdint>
#include <memory>
#include <optional>

#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_PROMOTESHUFFLETODPPPASS
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h.inc"

namespace {

namespace arith = ::mlir::arith;

std::optional<int64_t> getConstantIntValue(mlir::Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIntOp>()) {
    return cst.value();
  }
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>()) {
    return cst.value();
  }
  return std::nullopt;
}

// Promotes gpu.shuffle down ops with constant offsets in [1, 15] to
// amdgpu.dpp row_shr ops. DPP row_shr shifts data right within 16-lane rows,
// which is correct for the reduction use case where only lane 0's result
// matters and all relevant source lanes are within the same row.
//
// The pattern only fires when the validity predicate (second result) has no
// users, which is always the case for XLA's shuffle-reduce lowering.
struct PromoteShuffleDownToDPP
    : public mlir::OpRewritePattern<mlir::gpu::ShuffleOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::gpu::ShuffleOp op, mlir::PatternRewriter& rewriter) const override {
    if (op.getMode() != mlir::gpu::ShuffleMode::DOWN) {
      return rewriter.notifyMatchFailure(op,
                                         "only down shuffle mode is supported");
    }

    if (!op.getValid().use_empty()) {
      return rewriter.notifyMatchFailure(
          op, "validity predicate has users; cannot safely replace");
    }

    std::optional<int64_t> offset = getConstantIntValue(op.getOffset());
    if (!offset) {
      return rewriter.notifyMatchFailure(op,
                                         "offset must be a constant integer");
    }
    int64_t offset_value = *offset;
    if (offset_value < 1 || offset_value > 15) {
      return rewriter.notifyMatchFailure(
          op, "offset must be in the range [1, 15] for DPP row_shr");
    }

    std::optional<int64_t> width = getConstantIntValue(op.getWidth());
    if (!width) {
      return rewriter.notifyMatchFailure(op,
                                         "width must be a constant integer");
    }

    mlir::Location loc = op.getLoc();
    mlir::Type result_type = op.getShuffleResult().getType();

    mlir::Value dpp = mlir::amdgpu::DPPOp::create(
        rewriter, loc, result_type,
        /*old=*/op.getValue(),
        /*src=*/op.getValue(),
        /*kind=*/mlir::amdgpu::DPPPerm::row_shr,
        /*permArgument=*/rewriter.getI32IntegerAttr(offset_value),
        /*row_mask=*/0xF,
        /*bank_mask=*/0xF,
        /*bound_ctrl=*/true);

    mlir::Value valid =
        arith::ConstantIntOp::create(rewriter, loc, /*value=*/1, /*width=*/1);
    rewriter.replaceOp(op, {dpp, valid});
    return mlir::success();
  }
};

// Promotes gpu.shuffle down with offset 16 to amdgpu.swizzle_bitmode (which
// lowers to ds_swizzle). ds_swizzle operates within 32-lane groups using
// bitwise operations on lane IDs.
//
// For power-of-2 offset d, lane+d == lane XOR d when bit log2(d) of lane is 0.
// In the reduction tree, all lanes whose results matter at step d are multiples
// of 2d (bits 0..log2(d) all zero), so XOR gives the correct source lane.
//
// ds_swizzle is ~4x faster than ds_bpermute on GCN and ~2x on RDNA.
// Offset 32 cannot use swizzle because it crosses the 32-lane group boundary.
struct PromoteShuffleDownToSwizzle
    : public mlir::OpRewritePattern<mlir::gpu::ShuffleOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::gpu::ShuffleOp op, mlir::PatternRewriter& rewriter) const override {
    if (op.getMode() != mlir::gpu::ShuffleMode::DOWN) {
      return rewriter.notifyMatchFailure(op,
                                         "only down shuffle mode is supported");
    }

    if (!op.getValid().use_empty()) {
      return rewriter.notifyMatchFailure(
          op, "validity predicate has users; cannot safely replace");
    }

    std::optional<int64_t> offset = getConstantIntValue(op.getOffset());
    if (!offset || *offset != 16) {
      return rewriter.notifyMatchFailure(
          op, "only offset 16 is supported for swizzle promotion");
    }

    mlir::Location loc = op.getLoc();
    mlir::Type result_type = op.getShuffleResult().getType();

    mlir::Value swizzle = mlir::amdgpu::SwizzleBitModeOp::create(
        rewriter, loc, result_type, op.getValue(),
        /*and_mask=*/0x1F, /*or_mask=*/0x00, /*xor_mask=*/16);

    mlir::Value valid =
        arith::ConstantIntOp::create(rewriter, loc, /*value=*/1, /*width=*/1);
    rewriter.replaceOp(op, {swizzle, valid});
    return mlir::success();
  }
};

class PromoteShuffleToDPPPass
    : public impl::PromoteShuffleToDPPPassBase<PromoteShuffleToDPPPass> {
 public:
  using PromoteShuffleToDPPPassBase::PromoteShuffleToDPPPassBase;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<PromoteShuffleDownToDPP, PromoteShuffleDownToSwizzle>(
        &getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreatePromoteShuffleToDPPPass() {
  return std::make_unique<PromoteShuffleToDPPPass>();
}

}  // namespace gpu
}  // namespace xla
