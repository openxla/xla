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
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_CONVERTINDEXTYPEPASS
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h.inc"

namespace {

using mlir::ImplicitLocOpBuilder;
using mlir::IndexType;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::Type;

namespace arith = mlir::arith;

// Rewrites a binary elementwise op on 'index' types to a binary elementwise
// op on integers with the specified bit width.
template <typename BinaryElementwiseOp>
class RewriteIndexBinaryElementwiseOp
    : public OpRewritePattern<BinaryElementwiseOp> {
 public:
  using OpRewritePattern<BinaryElementwiseOp>::OpRewritePattern;
  RewriteIndexBinaryElementwiseOp(MLIRContext* context, uint64_t index_bitwidth)
      : OpRewritePattern<BinaryElementwiseOp>(context, 2),
        index_bitwidth_(index_bitwidth) {}

  LogicalResult matchAndRewrite(BinaryElementwiseOp op,
                                PatternRewriter& rewriter) const override {
    CHECK_EQ(op->getNumOperands(), 2);
    CHECK_EQ(op->getNumResults(), 1);

    auto is_index = [](Type type) { return llvm::isa<IndexType>(type); };

    bool operands_are_indices = absl::c_all_of(op->getOperandTypes(), is_index);
    bool result_is_index = is_index(op->getResultTypes().front());

    if (!operands_are_indices || !result_is_index) {
      return rewriter.notifyMatchFailure(
          op, "operation has non-index operands or results");
    }

    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    b.setInsertionPoint(op);

    Type index_type = IndexType::get(op->getContext());
    Type dst_type = b.getIntegerType(index_bitwidth_);
    auto lhs = arith::IndexCastOp::create(b, dst_type, op->getOperand(0));
    auto rhs = arith::IndexCastOp::create(b, dst_type, op->getOperand(1));
    auto new_op = BinaryElementwiseOp::create(b, lhs, rhs);

    rewriter.replaceAllUsesWith(
        op.getResult(),
        arith::IndexCastOp::create(b, index_type, new_op.getResult()));

    return mlir::success();
  }

 private:
  uint64_t index_bitwidth_;
};

// Folds `arith.index_cast(arith.index_cast(x : iN) : index) : iM` into direct
// integer casts (`arith.extsi` / `arith.trunci` / identity) using the known
// `index` bitwidth from the module's DataLayout. MLIR's built-in
// canonicalizer only folds when N == M because the index width is unknown.
// Eliminating the intermediate `index` casts lets downstream analyses (such as
// Triton's AxisInfoAnalysis) preserve offset divisibility and vectorize loads.
class FoldIndexCastChain : public OpRewritePattern<arith::IndexCastOp> {
 public:
  FoldIndexCastChain(MLIRContext* context, uint64_t index_bitwidth)
      : OpRewritePattern<arith::IndexCastOp>(context, 2),
        index_bitwidth_(index_bitwidth) {}

  LogicalResult matchAndRewrite(arith::IndexCastOp outer,
                                PatternRewriter& rewriter) const override {
    auto inner = outer.getIn().getDefiningOp<arith::IndexCastOp>();
    if (!inner) {
      return rewriter.notifyMatchFailure(outer, "input is not an index_cast");
    }
    mlir::Value src = inner.getIn();
    Type src_elem = mlir::getElementTypeOrSelf(src.getType());
    Type dst_elem = mlir::getElementTypeOrSelf(outer.getType());
    auto src_int = llvm::dyn_cast<mlir::IntegerType>(src_elem);
    auto dst_int = llvm::dyn_cast<mlir::IntegerType>(dst_elem);
    if (!src_int || !dst_int || !src_int.isSignless() ||
        !dst_int.isSignless()) {
      return rewriter.notifyMatchFailure(
          outer, "endpoints are not signless integer types");
    }

    uint64_t src_w = src_int.getWidth();
    uint64_t dst_w = dst_int.getWidth();
    mlir::Location loc = outer.getLoc();
    mlir::Value cur = src;
    if (src_w > index_bitwidth_ && dst_w > index_bitwidth_) {
      Type idx_int_type =
          src.getType() == src_elem
              ? Type(rewriter.getIntegerType(index_bitwidth_))
              : llvm::cast<mlir::ShapedType>(src.getType())
                    .clone(rewriter.getIntegerType(index_bitwidth_));
      cur = arith::TruncIOp::create(rewriter, loc, idx_int_type, cur);
      cur = arith::ExtSIOp::create(rewriter, loc, outer.getType(), cur);
      rewriter.replaceOp(outer, cur);
      return mlir::success();
    }
    if (src_w == dst_w) {
      rewriter.replaceOp(outer, cur);
    } else if (src_w < dst_w) {
      rewriter.replaceOpWithNewOp<arith::ExtSIOp>(outer, outer.getType(), cur);
    } else {
      rewriter.replaceOpWithNewOp<arith::TruncIOp>(outer, outer.getType(), cur);
    }
    return mlir::success();
  }

 private:
  uint64_t index_bitwidth_;
};

struct ConvertIndexTypePass
    : public impl::ConvertIndexTypePassBase<ConvertIndexTypePass> {
 public:
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    mlir::DataLayout layout(getOperation());
    std::optional<uint64_t> index_bitwidth =
        layout.getTypeIndexBitwidth(IndexType::get(ctx));
    CHECK(index_bitwidth.has_value());
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<RewriteIndexBinaryElementwiseOp<arith::AddIOp>,
                 RewriteIndexBinaryElementwiseOp<arith::DivUIOp>,
                 RewriteIndexBinaryElementwiseOp<arith::DivSIOp>,
                 RewriteIndexBinaryElementwiseOp<arith::MulIOp>,
                 RewriteIndexBinaryElementwiseOp<arith::RemUIOp>,
                 RewriteIndexBinaryElementwiseOp<arith::RemSIOp>,
                 RewriteIndexBinaryElementwiseOp<arith::SubIOp>,
                 RewriteIndexBinaryElementwiseOp<arith::XOrIOp>,
                 FoldIndexCastChain>(ctx, *index_bitwidth);
    arith::IndexCastOp::getCanonicalizationPatterns(patterns, ctx);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace gpu
}  // namespace xla
