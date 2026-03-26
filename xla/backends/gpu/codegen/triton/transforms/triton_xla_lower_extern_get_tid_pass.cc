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

// Generic lowering of GetTidOp using tt.extern_elementwise.
// This implementation uses tt.extern_elementwise to call a custom function
// that will be implemented in platform-specific passes in the Triton pipeline.

#include <memory>
#include <utility>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLALOWEREXTERNGETTIDPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

LogicalResult LowerGetTidOp(GetTidOp get_flat_tid, PatternRewriter& rewriter) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(get_flat_tid);
  const Location loc = get_flat_tid.getLoc();

  const mlir::Type i32_type = rewriter.getI32Type();

  // Use tt.extern_elementwise to call a custom function that returns thread ID
  // This function will be implemented in platform-specific passes
  auto tid_op = rewriter.create<triton::ExternElementwiseOp>(
      loc,
      /*resultType=*/i32_type,
      /*srcs=*/mlir::ValueRange{},  // No inputs needed
      /*libname=*/"",
      /*libpath=*/"",
      /*symbol=*/"xla_get_thread_id",
      /*pure=*/true);  // Thread ID is pure (deterministic for a given thread)

  rewriter.replaceOp(get_flat_tid, tid_op->getResults());
  return success();
}

class TritonXLALowerExternGetTidPass
    : public impl::TritonXLALowerExternGetTidPassBase<
          TritonXLALowerExternGetTidPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(LowerGetTidOp);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLALowerExternGetTidPass() {
  return std::make_unique<TritonXLALowerExternGetTidPass>();
}

}  // namespace mlir::triton::xla
