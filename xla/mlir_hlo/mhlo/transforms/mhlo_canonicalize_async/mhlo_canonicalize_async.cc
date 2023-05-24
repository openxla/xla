/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

#define GEN_PASS_DEF_HLOCANONICALIZEASYNCPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

struct CanonicalizeAsyncPattern : OpRewritePattern<AsyncStartOp> {
  using OpRewritePattern<AsyncStartOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AsyncStartOp asyncStartOp,
                                PatternRewriter& rewriter) const override {
    SmallVector<AsyncDoneOp> asyncDoneOps;
    for (mlir::Operation* user : asyncStartOp.getResult().getUsers()) {
      auto asyncDoneOp = llvm::dyn_cast<AsyncDoneOp>(user);
      if (asyncDoneOp == nullptr) {
        return rewriter.notifyMatchFailure(user->getLoc(),
                                           "has users that are not async-done");
      }
      if (asyncStartOp.getCalledComputation() !=
              asyncDoneOp.getCalledComputation() ||
          asyncStartOp.getExecutionThread() !=
              asyncDoneOp.getExecutionThread() ||
          asyncStartOp.getGroupId() != asyncDoneOp.getGroupId()) {
        return rewriter.notifyMatchFailure(
            user->getLoc(),
            "async-start and async-done have different attributes");
      }
      asyncDoneOps.push_back(asyncDoneOp);
    }

    auto func = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
        asyncStartOp,
        asyncStartOp.getCalledComputationAttr().cast<SymbolRefAttr>());
    if (func == nullptr) {
      return rewriter.notifyMatchFailure(
          asyncStartOp.getLoc(), "refers to an unknown called computation");
    }

    WalkResult result = func.walk([](Operation* op) {
      return isa<func::FuncOp, func::ReturnOp, SendOp, RecvOp>(op)
                 ? WalkResult::advance()
                 : WalkResult::interrupt();
    });
    if (result.wasInterrupted()) {
      return rewriter.notifyMatchFailure(
          func.getLoc(), "only async ops calling send/recv ops can be removed");
    }

    auto call = rewriter.create<func::CallOp>(asyncStartOp.getLoc(), func,
                                              asyncStartOp.getInputs());
    for (AsyncDoneOp asyncDoneOp : asyncDoneOps) {
      rewriter.replaceOp(asyncDoneOp, call.getResults());
    }
    rewriter.eraseOp(asyncStartOp);

    return success();
  }

  mutable SymbolTableCollection symbolTable;
};

struct HloCanonicalizeAsyncPass
    : impl::HloCanonicalizeAsyncPassBase<HloCanonicalizeAsyncPass> {
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<CanonicalizeAsyncPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createHloCanonicalizeAsyncPass() {
  return std::make_unique<HloCanonicalizeAsyncPass>();
}

}  // namespace mhlo
}  // namespace mlir
