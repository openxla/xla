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

// Generic lowering of atomic operations for Triton XLA using
// tt.extern_elementwise. This implementation uses tt.extern_elementwise to call
// custom atomic functions that will be implemented in platform-specific passes
// (CUDA and ROCm) later in the Triton pipeline.

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

#define GEN_PASS_DEF_TRITONXLALOWEREXTERNATOMICSPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// Convert MemSyncScope to string for function naming
std::string MemSyncScopeToString(triton::MemSyncScope scope) {
  switch (scope) {
    case triton::MemSyncScope::GPU:
      return "gpu";
    case triton::MemSyncScope::CTA:
      return "cta";
    case triton::MemSyncScope::SYSTEM:
      return "system";
  }
  LOG(FATAL) << "Unknown MemSyncScope: " << static_cast<int>(scope);
}

// Convert MemSemantic to string for function naming
std::string MemSemanticToString(triton::MemSemantic semantic) {
  switch (semantic) {
    case triton::MemSemantic::RELAXED:
      return "relaxed";
    case triton::MemSemantic::ACQUIRE:
      return "acquire";
    case triton::MemSemantic::RELEASE:
      return "release";
    case triton::MemSemantic::ACQUIRE_RELEASE:
      return "acq_rel";
  }
  LOG(FATAL) << "Unknown MemSemantic: " << static_cast<int>(semantic);
}

// Convert Comparator to string for function naming
std::string ComparatorToString(Comparator comparator) {
  switch (comparator) {
    case Comparator::LT:
      return "lt";
    case Comparator::EQ:
      return "eq";
  }
  LOG(FATAL) << "Unknown Comparator: " << static_cast<int>(comparator);
}

// Helper to get result type based on pointer type.
// For tensor<N x !tt.ptr<T>>, returns tensor<N x T>.
// For !tt.ptr<T>, returns T.
mlir::Type GetResultType(mlir::Type ptr_type) {
  auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(ptr_type);
  if (tensor_type) {
    // Tensor of pointers: tensor<N x !tt.ptr<T>> -> tensor<N x T>
    auto elem_ptr_type =
        mlir::cast<triton::PointerType>(tensor_type.getElementType());
    return mlir::RankedTensorType::get(tensor_type.getShape(),
                                       elem_ptr_type.getPointeeType());
  } else {
    // Scalar pointer: !tt.ptr<T> -> T
    auto ptr_type_cast = mlir::cast<triton::PointerType>(ptr_type);
    return ptr_type_cast.getPointeeType();
  }
}

// Lower AtomicWriteOp to tt.extern_elementwise.
// This creates extern calls that will be implemented in a separate ROCm pass.
LogicalResult LowerAtomicWriteOp(AtomicWriteOp atomic_write,
                                 PatternRewriter& rewriter) {
  VLOG(2) << "LowerAtomicWriteOp: Starting tt.extern_elementwise lowering";
  mlir::ImplicitLocOpBuilder builder(atomic_write.getLoc(), rewriter);

  mlir::Value ptr = atomic_write.getPtr();
  mlir::Value value = atomic_write.getValue();
  mlir::Value mask = atomic_write.getMask();

  triton::MemSemantic semantic = atomic_write.getMemSyncSemantic();
  triton::MemSyncScope scope = atomic_write.getMemSyncScope();

  // Validate memory semantics
  if (semantic != triton::MemSemantic::RELAXED &&
      semantic != triton::MemSemantic::RELEASE) {
    return rewriter.notifyMatchFailure(
        atomic_write,
        "AtomicWriteOp only supports RELAXED or RELEASE semantics");
  }

  // Build function name: xla_atomic_write_<semantic>_<scope>
  std::string func_name =
      absl::StrFormat("xla_atomic_write_%s_%s", MemSemanticToString(semantic),
                      MemSyncScopeToString(scope));

  VLOG(3) << "LowerAtomicWriteOp: Creating extern_elementwise call to "
          << func_name;

  // Get result type (handles both tensor and scalar pointers)
  mlir::Type result_type = GetResultType(ptr.getType());

  // Prepare operands: ptr (tensor or scalar), value (always scalar)
  llvm::SmallVector<mlir::Value> operands = {ptr, value};

  // If mask is provided, pass it as third argument
  if (mask) {
    operands.push_back(mask);
  }

  // Create tt.extern_elementwise call
  // The function will perform atomic exchange and return the old value
  // Note: extern_elementwise handles broadcasting scalar value to tensor
  // automatically
  builder.create<triton::ExternElementwiseOp>(
      /*resultType=*/result_type,
      /*srcs=*/operands,
      /*libname=*/"",
      /*libpath=*/"",
      /*symbol=*/func_name,
      /*pure=*/false);

  rewriter.eraseOp(atomic_write);
  return success();
}

// Lower AtomicSpinWaitOp to tt.extern_elementwise.
// This creates extern calls that will be implemented in a separate ROCm pass.
LogicalResult LowerAtomicSpinWaitOp(AtomicSpinWaitOp atomic_wait,
                                    PatternRewriter& rewriter) {
  VLOG(2) << "LowerAtomicSpinWaitOp: Starting tt.extern_elementwise lowering";
  mlir::ImplicitLocOpBuilder builder(atomic_wait.getLoc(), rewriter);

  mlir::Value ptr = atomic_wait.getPtr();
  mlir::Value expected = atomic_wait.getExpected();
  mlir::Value mask = atomic_wait.getMask();

  triton::MemSemantic semantic = atomic_wait.getMemSyncSemantic();
  triton::MemSyncScope scope = atomic_wait.getMemSyncScope();
  Comparator comparator = atomic_wait.getComparator();

  // Validate memory semantics
  if (semantic != triton::MemSemantic::RELAXED &&
      semantic != triton::MemSemantic::ACQUIRE) {
    return rewriter.notifyMatchFailure(
        atomic_wait,
        "AtomicSpinWaitOp only supports RELAXED or ACQUIRE semantics");
  }

  // Build function name: xla_atomic_spin_wait_<semantic>_<scope>_<comparator>
  std::string func_name = absl::StrFormat(
      "xla_atomic_spin_wait_%s_%s_%s", MemSemanticToString(semantic),
      MemSyncScopeToString(scope), ComparatorToString(comparator));

  VLOG(3) << "LowerAtomicSpinWaitOp: Creating extern_elementwise call to "
          << func_name;

  // Get result type (handles both tensor and scalar pointers)
  mlir::Type result_type = GetResultType(ptr.getType());

  // Prepare operands: ptr (tensor or scalar), expected (always scalar)
  llvm::SmallVector<mlir::Value> operands = {ptr, expected};

  // If mask is provided, pass it as third argument
  if (mask) {
    operands.push_back(mask);
  }

  // Create tt.extern_elementwise call
  // The function will spin-wait until the condition is met
  // Note: extern_elementwise handles broadcasting scalar expected to tensor
  // automatically
  builder.create<triton::ExternElementwiseOp>(
      /*resultType=*/result_type,
      /*srcs=*/operands,
      /*libname=*/"",
      /*libpath=*/"",
      /*symbol=*/func_name,
      /*pure=*/false);

  rewriter.eraseOp(atomic_wait);
  return success();
}

class TritonXLALowerExternAtomicsPass
    : public impl::TritonXLALowerExternAtomicsPassBase<
          TritonXLALowerExternAtomicsPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(LowerAtomicWriteOp);
    patterns.add(LowerAtomicSpinWaitOp);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLALowerExternAtomicsPass() {
  return std::make_unique<TritonXLALowerExternAtomicsPass>();
}

}  // namespace mlir::triton::xla
