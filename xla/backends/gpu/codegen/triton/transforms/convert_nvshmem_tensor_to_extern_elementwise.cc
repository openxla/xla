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

#include "xla/backends/gpu/codegen/triton/transforms/convert_nvshmem_tensor_to_extern_elementwise.h"
#include <memory>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"

namespace mlir {
namespace triton {
namespace xla {

namespace ttir = ::mlir::triton;

namespace {

// Convert nvshmem_float_sum_reduce_tensor to standard tt.extern_elementwise format
class ConvertNvshmemTensorToExternElementwisePattern 
    : public mlir::OpRewritePattern<ttir::ExternElementwiseOp> {
public:
  using OpRewritePattern<ttir::ExternElementwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ttir::ExternElementwiseOp op, 
      mlir::PatternRewriter& rewriter) const override {
    
    // Handle both new placeholder and existing NVSHMEM symbols
    auto symbol = op.getSymbol();
    bool is_nvshmem_placeholder = (symbol == "__xla_nvshmem_allreduce_placeholder");
    bool is_existing_nvshmem = symbol.starts_with("nvshmem_") && symbol.ends_with("_reduce");
    
    if (!is_nvshmem_placeholder && !is_existing_nvshmem) {
      return failure();
    }
    
    // Converting NVSHMEM symbol to standard format
    
    auto loc = op.getLoc();
    auto operands = op.getOperands();
    
    // Current operation should have single tensor operand
    if (operands.size() != 1) {
      return rewriter.notifyMatchFailure(op, "Expected single tensor operand for NVSHMEM placeholder");
    }
    
    auto tensor_operand = operands[0];
    auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(tensor_operand.getType());
    if (!tensor_type) {
      return rewriter.notifyMatchFailure(op, "Expected ranked tensor type");
    }
    
    // Get element type (should be f32)
    auto element_type = tensor_type.getElementType();
    if (!element_type.isF32()) {
      return rewriter.notifyMatchFailure(op, "Currently only f32 tensors are supported");
    }
    
    // Calculate total number of elements
    int64_t total_elements = 1;
    for (auto dim : tensor_type.getShape()) {
      total_elements *= dim;
    }
    
    // Create constants for NVSHMEM call (team=0, count=total_elements)
    auto team_const = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    
    // count: total number of elements
    auto count_const = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(total_elements));
    
    // Convert tensor to pointer using UnrealizedConversionCastOp
    auto ptr_type = ttir::PointerType::get(element_type, 1); // address space 1 (global)
    auto cast_to_ptr = rewriter.create<mlir::UnrealizedConversionCastOp>(
        loc, ptr_type, tensor_operand);
    auto ptr_value = cast_to_ptr.getResult(0);
    
    // Create standard tt.extern_elementwise operation for NVSHMEM API call
    auto new_extern_op = rewriter.create<ttir::ExternElementwiseOp>(
        loc,
        rewriter.getI32Type(),  // return type: i32 (NVSHMEM return code)
        mlir::ValueRange{team_const.getResult(), ptr_value, ptr_value, count_const.getResult()},
        "nvshmem",              // libname
        "",                     // libpath (empty)
        "nvshmemx_float_sum_reduce_block",  // symbol
        /*pure=*/false          // not a pure function
    );
    
    rewriter.replaceOp(op, new_extern_op.getResult());
    
    return success();
  }
};

}  // namespace

class ConvertNvshmemTensorToExternElementwisePass 
    : public PassWrapper<ConvertNvshmemTensorToExternElementwisePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertNvshmemTensorToExternElementwisePass)

  void runOnOperation() override {
    auto module = getOperation();
    MLIRContext* context = &getContext();
    
    // Set up patterns for rewrite
    RewritePatternSet patterns(context);
    patterns.add<ConvertNvshmemTensorToExternElementwisePattern>(context);
    
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const final { return "convert-nvshmem-tensor-to-extern-elementwise"; }
  StringRef getDescription() const final {
    return "Convert NVSHMEM tensor placeholder to standard tt.extern_elementwise format";
  }
};

std::unique_ptr<mlir::Pass> CreateConvertNvshmemTensorToExternElementwisePass() {
  return std::make_unique<ConvertNvshmemTensorToExternElementwisePass>();
}

}  // namespace xla
}  // namespace triton
}  // namespace mlir 