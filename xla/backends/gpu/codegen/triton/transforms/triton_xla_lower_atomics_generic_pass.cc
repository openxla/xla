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

// Generic lowering of atomic operations for Triton XLA using only Triton ops.
//
// This implementation uses pure Triton atomic operations (tt.atomic_rmw and
// tt.atomic_cas) instead of platform-specific inline assembly.
//
// Key transformations:
// 1. AtomicWriteOp → tt.atomic_rmw with XCHG (exchange) operation
// 2. AtomicSpinWaitOp → scf.while loop with tt.atomic_cas for polling
//
// Supports both scalar and tensor (vectorized) operations:
// - Scalar: Single pointer (!tt.ptr<T>) and value
// - Tensor: Tensor of pointers (tensor<N x !tt.ptr<T>>) with scalar or tensor
// values
//   In tensor mode, each element is processed by a separate thread in Triton's
//   SPMD model, enabling vectorized atomic operations.
//
// This approach makes it suitable for ROCm and other targets that don't
// support CUDA-specific PTX inline assembly.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
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

#define GEN_PASS_DEF_TRITONXLALOWERATOMICSGENERICPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// Lower AtomicWriteOp to Triton atomic_rmw XCHG operation.
//
// Handles both scalar and tensor pointer types:
// - Scalar: !tt.ptr<T> → single atomic operation
// - Tensor: tensor<N x !tt.ptr<T>> → N parallel atomic operations
//
// When the pointer is a tensor, creates a vectorized atomic operation where
// each tensor element (processed by a separate thread) performs an atomic
// exchange on its corresponding pointer.
LogicalResult LowerAtomicWriteOpROCm(AtomicWriteOp atomic_write,
                                     PatternRewriter& rewriter) {
  VLOG(2) << "LowerAtomicWriteOpROCm: Starting";
  mlir::ImplicitLocOpBuilder builder(atomic_write.getLoc(), rewriter);

  mlir::Value ptr = atomic_write.getPtr();
  mlir::Value value = atomic_write.getValue();
  mlir::Value mask = atomic_write.getMask();

  triton::MemSemantic semantic = atomic_write.getMemSyncSemantic();
  if (semantic != triton::MemSemantic::RELAXED &&
      semantic != triton::MemSemantic::RELEASE) {
    return rewriter.notifyMatchFailure(
        atomic_write, absl::StrFormat("Unsupported memory semantic: %s",
                                      stringifyMemSemantic(semantic)));
  }

  // Check if ptr is a tensor type (vectorized operation)
  auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(ptr.getType());

  if (tensor_type) {
    // Vectorized path: ptr is tensor<N x !tt.ptr<T>>
    VLOG(3) << "LowerAtomicWriteOpROCm: Vectorized atomic write";

    // Get element type from tensor of pointers
    auto elem_ptr_type =
        mlir::cast<triton::PointerType>(tensor_type.getElementType());
    auto elem_type = elem_ptr_type.getPointeeType();

    // Result type is tensor<N x T>
    auto result_tensor_type =
        mlir::RankedTensorType::get(tensor_type.getShape(), elem_type);

    // Create value tensor (splat scalar to tensor)
    auto value_tensor =
        triton::SplatOp::create(builder, result_tensor_type, value);

    // Use vectorized atomic_rmw XCHG
    triton::AtomicRMWOp::create(builder,
                                /*result_type=*/result_tensor_type,
                                triton::RMWOp::XCHG,
                                /*ptr=*/ptr,
                                /*val=*/value_tensor,
                                /*mask=*/mask,
                                /*sem=*/semantic,
                                /*scope=*/atomic_write.getMemSyncScope());
  } else {
    // Scalar path: ptr is !tt.ptr<T>
    VLOG(3) << "LowerAtomicWriteOpROCm: Scalar atomic write";

    // Use scalar atomic_rmw XCHG
    triton::AtomicRMWOp::create(builder,
                                /*result_type=*/value.getType(),
                                triton::RMWOp::XCHG,
                                /*ptr=*/ptr,
                                /*val=*/value,
                                /*mask=*/mask,
                                /*sem=*/semantic,
                                /*scope=*/atomic_write.getMemSyncScope());
  }

  rewriter.eraseOp(atomic_write);
  return success();
}

// Lower AtomicSpinWaitOp to scf.while loop with tensor atomic CAS.
//
// Creates a spin-wait loop that polls memory locations using atomic
// compare-and-swap operations until a condition is met.
//
// Currently only supports tensor pointers (tensor<N x !tt.ptr<T>>):
// - Each tensor element represents a memory location to poll
// - Each element is processed by a separate thread in Triton's SPMD model
// - Uses reduction to check if all elements are ready
LogicalResult LowerAtomicSpinWaitOpROCm(AtomicSpinWaitOp atomic_wait,
                                        PatternRewriter& rewriter) {
  VLOG(2) << "LowerAtomicSpinWaitOpROCm: Starting";
  mlir::ImplicitLocOpBuilder builder(atomic_wait.getLoc(), rewriter);

  mlir::Value ptr = atomic_wait.getPtr();
  mlir::Value expected = atomic_wait.getExpected();
  mlir::Value mask = atomic_wait.getMask();

  triton::MemSemantic semantic = atomic_wait.getMemSyncSemantic();
  Comparator comparator = atomic_wait.getComparator();

  // Get tensor type info
  auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(ptr.getType());
  if (!tensor_type) {
    return rewriter.notifyMatchFailure(
        atomic_wait, "AtomicSpinWaitOp requires tensor pointer type");
  }

  auto elem_ptr_type =
      mlir::cast<triton::PointerType>(tensor_type.getElementType());
  auto elem_type = elem_ptr_type.getPointeeType();
  auto result_tensor_type =
      mlir::RankedTensorType::get(tensor_type.getShape(), elem_type);

  // Create expected value tensor (splat scalar to tensor)
  auto expected_tensor =
      triton::SplatOp::create(builder, result_tensor_type, expected);

  // Map Comparator to arith::CmpIPredicate before creating the loop
  mlir::arith::CmpIPredicate predicate;
  switch (comparator) {
    case Comparator::LT:
      predicate = mlir::arith::CmpIPredicate::slt;  // signed less than
      break;
    case Comparator::EQ:
      predicate = mlir::arith::CmpIPredicate::eq;  // equal
      break;
    default:
      return rewriter.notifyMatchFailure(
          atomic_wait, "Unsupported comparator for AtomicSpinWaitOp");
  }

  // Create initial loop condition (true)
  auto true_val =
      mlir::arith::ConstantOp::create(builder, builder.getBoolAttr(true));

  mlir::scf::WhileOp::create(
      builder,
      /*resultTypes=*/mlir::TypeRange{},
      /*operands=*/mlir::ValueRange{true_val},
      /*beforeBuilder=*/
      [&](mlir::OpBuilder& op_builder, mlir::Location location,
          mlir::ValueRange args) {
        mlir::ImplicitLocOpBuilder loop_builder(location, op_builder);
        // args[0] is the continue_loop condition
        mlir::scf::ConditionOp::create(loop_builder, args[0],
                                       mlir::ValueRange{});
      },
      /*afterBuilder=*/
      [&](mlir::OpBuilder& op_builder, mlir::Location location,
          mlir::ValueRange args) {
        mlir::ImplicitLocOpBuilder loop_builder(location, op_builder);

        // Atomic CAS with tensor operands
        mlir::Value cas_results = triton::AtomicCASOp::create(
            loop_builder,
            /*result=*/result_tensor_type,
            /*ptr=*/ptr,
            /*cmp=*/expected_tensor,
            /*val=*/expected_tensor,
            /*sem=*/semantic,
            /*scope=*/atomic_wait.getMemSyncScope());

        // Compare results with expected value
        auto bool_tensor_type = mlir::RankedTensorType::get(
            tensor_type.getShape(), loop_builder.getI1Type());

        // Use the predicate that was determined outside the lambda
        mlir::Value comparison_result = mlir::arith::CmpIOp::create(
            loop_builder, predicate, cas_results, expected_tensor);

        // Apply mask if provided: arith.select %mask, %comparison_result,
        // %false
        mlir::Value masked_comparison = comparison_result;
        if (mask) {
          auto false_tensor = mlir::arith::ConstantOp::create(
              loop_builder,
              mlir::DenseElementsAttr::get(bool_tensor_type, false));
          masked_comparison = mlir::arith::SelectOp::create(
              loop_builder, mask, comparison_result, false_tensor);
        }

        // Extend booleans to i32 for reduction (matching Triton Python)
        // arith.extui %masked_comparison : tensor<Nxi1> to tensor<Nxi32>
        auto i32_tensor_type = mlir::RankedTensorType::get(
            tensor_type.getShape(), loop_builder.getI32Type());
        mlir::Value comparison_i32 = mlir::arith::ExtUIOp::create(
            loop_builder, i32_tensor_type, masked_comparison);

        // Reduce with ADD (matching Triton Python, not OR)
        // tt.reduce %comparison_i32 {axis = 0} : tensor<Nxi32> -> i32
        auto reduce_op = triton::ReduceOp::create(
            loop_builder, mlir::ValueRange{comparison_i32},
            /*axis=*/0);

        // Build ADD combiner region
        mlir::Region& combine_region = reduce_op.getCombineOp();
        mlir::Block* combine_block = new mlir::Block();
        combine_region.push_back(combine_block);

        combine_block->addArgument(loop_builder.getI32Type(), location);
        combine_block->addArgument(loop_builder.getI32Type(), location);

        {
          mlir::OpBuilder::InsertionGuard guard(loop_builder);
          loop_builder.setInsertionPointToStart(combine_block);

          mlir::Value add_result = mlir::arith::AddIOp::create(
              loop_builder, combine_block->getArgument(0),
              combine_block->getArgument(1));
          triton::ReduceReturnOp::create(loop_builder,
                                         mlir::ValueRange{add_result});
        }

        mlir::Value sum = *reduce_op.getResult().begin();

        // Check if any element is not ready: arith.cmpi sgt, %sum, %c0
        auto zero = mlir::arith::ConstantOp::create(
            loop_builder, loop_builder.getI32IntegerAttr(0));
        mlir::Value any_not_ready = mlir::arith::CmpIOp::create(
            loop_builder, mlir::arith::CmpIPredicate::sgt, sum, zero);

        // Yield the continue condition
        mlir::scf::YieldOp::create(loop_builder,
                                   mlir::ValueRange{any_not_ready});
      });

  rewriter.eraseOp(atomic_wait);
  return success();
}

class TritonXLALowerAtomicsGenericPass
    : public impl::TritonXLALowerAtomicsGenericPassBase<
          TritonXLALowerAtomicsGenericPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(LowerAtomicWriteOpROCm);
    patterns.add(LowerAtomicSpinWaitOpROCm);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLALowerAtomicsGenericPass() {
  return std::make_unique<TritonXLALowerAtomicsGenericPass>();
}

}  // namespace mlir::triton::xla
