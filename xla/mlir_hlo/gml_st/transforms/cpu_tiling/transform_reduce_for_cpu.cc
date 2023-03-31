/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <utility>

#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/peeling/peeling.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMREDUCEFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

SmallVector<int64_t> getParallelDimTileSizes(int64_t reductionDim,
                                             int64_t parallelDimTileSize) {
  return reductionDim ? SmallVector<int64_t>{parallelDimTileSize, 0}
                      : SmallVector<int64_t>{0, parallelDimTileSize};
}

SmallVector<int64_t> getReductionDimTileSizes(int64_t reductionDim,
                                              int64_t reductionDimTileSize) {
  return reductionDim ? SmallVector<int64_t>{0, reductionDimTileSize}
                      : SmallVector<int64_t>{reductionDimTileSize, 0};
}

LogicalResult validateOp(linalg::ReduceOp reduceOp, PatternRewriter &rewriter,
                         int64_t expectedRank) {
  ArrayRef<int64_t> reduceDimensions = reduceOp.getDimensions();
  if (reduceDimensions.size() != 1) {
    return rewriter.notifyMatchFailure(
        reduceOp, "expects 1 reduction dimension element. 0 or > 1 received.");
  }
  OpOperandVector operands = reduceOp.getDpsInputOperands();
  if (operands.size() != 1) {
    return rewriter.notifyMatchFailure(reduceOp,
                                       "expects 1 operand. 0 or > 1 received.");
  }
  const int64_t operandRank =
      operands[0]->get().getType().cast<RankedTensorType>().getRank();
  if (operandRank != expectedRank) {
    return rewriter.notifyMatchFailure(reduceOp, [&](Diagnostic &diag) {
      diag << "expects rank " << expectedRank << ". " << operandRank
           << "received.";
    });
  }
  return success();
}

// Tiles, pads and reshapes every input argument of type tensor<?xELEM_TYPE>
// into tensor<(TILE_SIZE/SPLIT_RATIO)xSPLIT_RATIOxELEM_TYPE>.
Value tileAndReshapeInputTensors(OpBuilder &b, Location loc, Value iv,
                                 Value input, OpFoldResult inputSizeOfr,
                                 int64_t tileSize, int64_t vectorSize,
                                 Value neutralValue) {
  SmallVector<ReassociationIndices> indices = {{0, 1}};
  auto identityMap1D = b.getMultiDimIdentityMap(1);

  OpFoldResult tileSizeOfr(b.getIndexAttr(tileSize));
  SmallVector<OpFoldResult> partialTileSizes =
      linalg::computeTileSizes(b, loc, tileSizeOfr, inputSizeOfr);

  // Extract slice of input.
  Value slice = linalg::makeTiledShape(b, loc, input, tileSizeOfr,
                                       identityMap1D, OpFoldResult{iv},
                                       inputSizeOfr, partialTileSizes, false);
  auto elementType = slice.getType().cast<ShapedType>().getElementType();

  // Pad input tile.
  Value pad =
      tensor::createPadHighOp(RankedTensorType::get({tileSize}, elementType),
                              slice, neutralValue, false, loc, b);

  // Reshape input tile to
  // tensor<(TILE_SIZE/SPLIT_RATIO)xSPLIT_RATIOxELEM_TYPE>.
  Value expandShape = b.create<tensor::ExpandShapeOp>(
      loc,
      RankedTensorType::get({tileSize / vectorSize, vectorSize}, elementType),
      pad, indices);
  return expandShape;
}

FailureOr<scf::ForOp> splitReduction1D(PatternRewriter &rewriter, Location loc,
                                       linalg::ReduceOp reduceOp,
                                       int64_t tileSize, int64_t vectorSize) {
  if (failed(validateOp(reduceOp, rewriter, /*expectedRank=*/1)))
    return failure();

  // 0-d tensor with the neutral elements.
  auto fillOp = reduceOp.getInits().front().getDefiningOp<linalg::FillOp>();
  if (!fillOp)
    return rewriter.notifyMatchFailure(reduceOp, "init not defined by fill op");
  auto neutralValue = fillOp.value();

  // Constants.
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value tileSizeValue = rewriter.create<arith::ConstantIndexOp>(loc, tileSize);

  // Input.
  Value input = reduceOp.getInputs().front();
  FailureOr<OpFoldResult> inputSizeOfr =
      tensor::createDimValue(rewriter, loc, input, 0);
  if (failed(inputSizeOfr)) {
    return rewriter.notifyMatchFailure(
        reduceOp, "cannot get the size of the input tensor");
  }

  // Create tensor<SPLIT_RATIOxELEM_TYPE> with neutral elements for tile loop
  // init.
  Type elementType = neutralValue.getType();
  Value emptyVector = rewriter.create<tensor::EmptyOp>(
      loc, llvm::ArrayRef({vectorSize}), elementType);
  Value filledVector =
      rewriter.create<linalg::FillOp>(loc, neutralValue, emptyVector)
          .getResult(0);

  auto tiledLoopBodyBuilder = [&](OpBuilder &b, Location loc, Value iv,
                                  ValueRange inits) {
    // Tile input, pad it to tensor<TILE_SIZExELEM_TYPE> and reshape into
    // tensor<(TILE_SIZE/SPLIT_RATIO)xSPLIT_RATIOxELEM_TYPE>.
    Value inputSlice = tileAndReshapeInputTensors(
        b, loc, iv, input, *inputSizeOfr, tileSize, vectorSize, neutralValue);

    // Create `linalg.reduce` to combine
    // `tensor<(TILE_SIZE/SPLIT_RATIO)xSPLIT_RATIOxELEM_TYPE> input with the
    // `tensor<SPLIT_RATIOxELEM_TYPE>` accumulator.
    auto tiledReduceOp = b.create<linalg::ReduceOp>(
        loc, ValueRange{inputSlice}, inits,
        /*dimensions=*/SmallVector<int64_t>{0},
        /*bodyBuilder=*/nullptr, linalg::getPrunedAttributeList(reduceOp));
    OpBuilder::InsertionGuard g(rewriter);
    Region &region = tiledReduceOp.getRegion();
    rewriter.cloneRegionBefore(reduceOp.getRegion(), region, region.end());
    setLabel(tiledReduceOp, kTransformedLabel);

    b.create<scf::YieldOp>(loc, tiledReduceOp.getResults());
  };

  // Create a tiled loop
  Value inputSize =
      getValueOrCreateConstantIndexOp(rewriter, loc, *inputSizeOfr);
  auto tiledLoop = rewriter.create<scf::ForOp>(
      loc, zero, inputSize, tileSizeValue, filledVector, tiledLoopBodyBuilder);
  setLabel(tiledLoop, kPerfectlyTiledLoopLabel);

  // Create `linalg.reduce` from tensor<SPLIT_RATIOxELEM_TYPE> to
  // tensor<ELEM_TYPE>.
  auto *horizontalReduce =
      clone(rewriter, reduceOp, reduceOp.getType(0),
            {tiledLoop.getResult(0), reduceOp.getInits().front()});
  setLabel(horizontalReduce, kTransformedLabel);

  rewriter.replaceOp(reduceOp, horizontalReduce->getResults());
  return tiledLoop;
}

struct Reduce1DTransformPattern : public OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  explicit Reduce1DTransformPattern(MLIRContext *context, int64_t tileSize,
                                    int64_t splitRatio,
                                    PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::ReduceOp>(context, benefit),
        tileSize(tileSize),
        splitRatio(splitRatio) {}

  LogicalResult matchAndRewrite(linalg::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(reduceOp, kTransformedLabel)) {
      return rewriter.notifyMatchFailure(reduceOp,
                                         "has already been transformed.");
    }
    Location loc = reduceOp.getLoc();

    // Rewrite 1D reduction as a loop followed by a horizontal reduce.
    auto splitReductionLoopOr =
        splitReduction1D(rewriter, loc, reduceOp, tileSize, splitRatio);
    if (failed(splitReductionLoopOr)) {
      return rewriter.notifyMatchFailure(
          reduceOp, "failed to split the reduction dimension");
    }

    // Fuse elementwise ops.
    fuseGreedily(rewriter, *splitReductionLoopOr->getBody(),
                 [&](Operation *op) { return isa<linalg::MapOp>(op); });

    // Peel the loop.
    peelSCFForOp(rewriter, *splitReductionLoopOr);
    return success();
  }

 private:
  int64_t tileSize;
  int64_t splitRatio;
};

/// Pattern to tile `linalg.reduce` and fuse `linalg.fill` into generated
/// `scf.forall`.
struct Reduce2DTransformPattern : public OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  explicit Reduce2DTransformPattern(MLIRContext *context,
                                    int64_t parallelDimTileSize = 4,
                                    int64_t reductionDimTileSize = 2,
                                    PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::ReduceOp>(context, benefit),
        parallelDimTileSize(parallelDimTileSize),
        reductionDimTileSize(reductionDimTileSize) {}

  LogicalResult matchAndRewrite(linalg::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (reduceOp.getDimensions().size() != 1) return failure();
    int64_t reductionDim = reduceOp.getDimensions()[0];

    if (hasLabel(reduceOp, kTransformedLabel)) {
      return rewriter.notifyMatchFailure(reduceOp,
                                         "has already been transformed.");
    }
    if (isa<scf::ForallOp, scf::ForOp>(reduceOp->getParentOp())) {
      return rewriter.notifyMatchFailure(
          reduceOp, "has already been tiled by another pass.");
    }
    if (failed(validateOp(reduceOp, rewriter, /*expectedRank=*/2)))
      return failure();

    auto producerFilterFn = [](Operation *op) {
      return isa<linalg::BroadcastOp, linalg::FillOp, linalg::MapOp,
                 linalg::TransposeOp, tensor::CastOp>(op);
    };
    auto consumerFilterFn = [](Operation *op) {
      return isa<linalg::MapOp, thlo::ReverseOp>(op);
    };
    auto fusionClusterFn = [&](Operation *op) {
      return producerFilterFn(op) || isa<linalg::ReduceOp>(op);
    };
    auto cluster =
        getFusionCluster(reduceOp, producerFilterFn, consumerFilterFn);
    auto fusionCluster = cluster.operations;
    auto *tilingRoot = cluster.root;

    // First level tiling: parallel dimension.
    auto parallelDimsTileSizes =
        isa<linalg::ReduceOp>(tilingRoot)
            ? getParallelDimTileSizes(reduceOp.getDimensions()[0],
                                      parallelDimTileSize)
            : SmallVector<int64_t>{parallelDimTileSize};
    auto tilingParallelDimsResult = tileUsingSCFForallOpAndFuseGreedily(
        rewriter, tilingRoot, getSCFTilingOptions(parallelDimsTileSizes),
        [&](Operation *op) { return fusionCluster.contains(op); });
    if (failed(tilingParallelDimsResult)) return failure();

    auto peeledParallelLoop =
        peelAllLoops(tilingParallelDimsResult->loop, rewriter);

    // Process main parallel loop.
    scf::ForallOp mainParallelLoop = peeledParallelLoop.mainLoop;
    if (mainParallelLoop) {
      auto tiledReduceOp =
          *mainParallelLoop.getBody()->getOps<linalg::ReduceOp>().begin();
      if (failed(tileAndPeelReductionDim(rewriter, tiledReduceOp, reductionDim,
                                         producerFilterFn))) {
        return failure();
      }
    }

    // Process tail parallel loop.
    scf::ForallOp tailParallelLoop = peeledParallelLoop.tailLoops.size() == 1
                                         ? peeledParallelLoop.tailLoops.front()
                                         : nullptr;
    if (tailParallelLoop) {
      Value yieldedTensor =
          getYieldedValues(tailParallelLoop.getTerminator()).front();
      auto *definingOp = yieldedTensor.getDefiningOp();
      if (!definingOp) return failure();

      auto opts = getSCFTilingOptions(SmallVector<int64_t>(
          definingOp->getResult(0).getType().cast<RankedTensorType>().getRank(),
          1));
      auto parallelDimTilingOpts =
          isa<linalg::ReduceOp>(definingOp)
              ? getSCFTilingOptions(getParallelDimTileSizes(reductionDim, 1))
              : getSCFTilingOptions({1});
      auto parallelDimTilingResult = tileUsingSCFForallOpAndFuseGreedily(
          rewriter, definingOp, parallelDimTilingOpts, fusionClusterFn);
      if (failed(parallelDimTilingResult)) return failure();

      for (auto tiledReduceOp :
           llvm::to_vector(parallelDimTilingResult->loop.getBody()
                               ->getOps<linalg::ReduceOp>())) {
        auto reductionDimTilingResult = tileUsingSCFForOpAndFuseGreedily(
            rewriter, tiledReduceOp,
            getSCFTilingOptions(getReductionDimTileSizes(reductionDim, 1)),
            producerFilterFn);
        if (failed(reductionDimTilingResult)) return failure();
      }
    }

    return success();
  }

 private:
  LogicalResult tileAndPeelReductionDim(
      PatternRewriter &rewriter, linalg::ReduceOp reduceOp,
      int64_t reductionDim,
      llvm::function_ref<bool(Operation *)> producerFilterFn) const {
    FailureOr<scf::SCFTilingResult> reductionDimTilingResult =
        tileUsingSCFForOpAndFuseGreedily(
            rewriter, reduceOp,
            getSCFTilingOptions(
                getReductionDimTileSizes(reductionDim, reductionDimTileSize)),
            producerFilterFn);
    if (failed(reductionDimTilingResult)) return failure();

    SCFForPeelingResult reductionDimPeelingResult =
        peelSCFForOp(rewriter, reductionDimTilingResult->loops.front());
    if (reductionDimPeelingResult.mainLoop) {
      setLabel(reductionDimPeelingResult.mainLoop, kPerfectlyTiledLoopLabel);
    }
    if (reductionDimPeelingResult.tailLoop) {
      for (auto reduOp :
           llvm::to_vector(reductionDimPeelingResult.tailLoop.getBody()
                               ->getOps<linalg::ReduceOp>())) {
        // Column reductions have to be tiled even further, otherwise we
        // would get vector.multi_reduction 4x1 -> 1, which is expensive.
        // Potentially, we could lower it to a horizontal add.
        if (reductionDim == 0) {
          auto parallelDimSizeOneTilingResult =
              tileUsingSCFForOpAndFuseGreedily(
                  rewriter, reduOp,
                  getSCFTilingOptions(getParallelDimTileSizes(reductionDim, 1)),
                  producerFilterFn);
          if (failed(parallelDimSizeOneTilingResult)) return failure();

          reduOp = cast<linalg::ReduceOp>(
              parallelDimSizeOneTilingResult->tiledOps.front());
        }
        if (failed(tileUsingSCFForOpAndFuseGreedily(
                rewriter, reduOp,
                getSCFTilingOptions(getReductionDimTileSizes(reductionDim, 1)),
                producerFilterFn))) {
          return failure();
        }
      }
    }
    return success();
  }

  int64_t parallelDimTileSize;
  int64_t reductionDimTileSize;
};

struct TransformReduceForCpuPass
    : public impl::TransformReduceForCpuPassBase<TransformReduceForCpuPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<Reduce1DTransformPattern>(ctx, tileSize1D, splitRatio1D);
    patterns.add<Reduce2DTransformPattern>(ctx, parallelDimTileSize2D,
                                           reductionDimTileSize2D);
    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createTransformReduceForCpuPass(
    const TransformReduceForCpuPassOptions &opts) {
  return std::make_unique<TransformReduceForCpuPass>(opts);
}

}  // namespace mlir::gml_st
