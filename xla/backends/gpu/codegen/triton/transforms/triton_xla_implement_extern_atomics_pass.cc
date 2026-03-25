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

// Unified implementation of extern_elementwise atomic functions for both
// CUDA and ROCm backends. This pass runs in the Triton pipeline and inlines
// the implementations of custom atomic functions by replacing llvm.call
// operations with actual LLVM dialect operations (intrinsics, atomics, loops).

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLAIMPLEMENTEXTERNATOMICSPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// Helper to parse syncscope from function name for the target backend
// Function names follow pattern: xla_atomic_*_<semantic>_<scope>_<comparator>
llvm::StringRef ParseSyncScope(llvm::StringRef func_name,
                               TargetBackend target) {
  if (target == TargetBackend::CUDA) {
    // Per NVPTX memory model:
    // - "" (empty) = system scope (cross-device visibility)
    // - "gpu" = GPU scope (single device)
    // - "cta" = CTA/block scope
    if (func_name.contains("_system")) {
      return "";  // System scope for cross-GPU visibility
    } else if (func_name.contains("_gpu")) {
      return "gpu";
    } else if (func_name.contains("_cta")) {
      return "cta";
    }
  } else {  // ROCM
    // Per AMDGPU memory model:
    // - "" (empty) = system scope (cross-device visibility)
    // - "agent" = GPU scope (single device)
    // - "workgroup" = CTA/block scope
    if (func_name.contains("_system")) {
      return "";  // System scope for cross-GPU visibility
    } else if (func_name.contains("_gpu")) {
      return "agent";
    } else if (func_name.contains("_cta")) {
      return "workgroup";
    }
  }

  LOG(FATAL) << "Unable to parse syncscope from function name: "
             << func_name.str();
}

// Helper to parse memory ordering from function name
// Function names follow pattern: xla_atomic_*_<semantic>_<scope>_<comparator>
LLVM::AtomicOrdering ParseAtomicOrdering(llvm::StringRef func_name) {
  // Map Triton semantics to LLVM atomic ordering
  if (func_name.contains("_relaxed_")) {
    return LLVM::AtomicOrdering::monotonic;  // LLVM's "relaxed"
  } else if (func_name.contains("_acquire_")) {
    return LLVM::AtomicOrdering::acquire;
  } else if (func_name.contains("_release_")) {
    return LLVM::AtomicOrdering::release;
  } else if (func_name.contains("_acq_rel_")) {
    return LLVM::AtomicOrdering::acq_rel;
  }
  LOG(FATAL) << "Unable to parse memory ordering from function name: "
             << func_name.str();
}

// MLIR pass that inlines extern function calls with actual implementations
class TritonXLAImplementExternAtomicsPass
    : public impl::TritonXLAImplementExternAtomicsPassBase<
          TritonXLAImplementExternAtomicsPass> {
 public:
  using Base::Base;

  explicit TritonXLAImplementExternAtomicsPass(TargetBackend target_backend) {
    target_ = target_backend == TargetBackend::CUDA ? "cuda" : "rocm";
  }

 private:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module.getContext());

    // Parse target backend from option
    TargetBackend target;
    if (target_ == "cuda") {
      target = TargetBackend::CUDA;
    } else if (target_ == "rocm") {
      target = TargetBackend::ROCM;
    } else {
      LOG(FATAL) << "Invalid target backend: " << target_
                 << ". Must be 'cuda' or 'rocm'";
    }

    // Find all llvm.call operations to our extern functions
    llvm::SmallVector<LLVM::CallOp> calls_to_replace;
    module.walk([&](LLVM::CallOp call_op) {
      if (auto callee = call_op.getCallee()) {
        llvm::StringRef name = *callee;
        if (name.starts_with("xla_atomic_write_") ||
            name.starts_with("xla_atomic_spin_wait_") ||
            name.starts_with("xla_get_thread_id")) {
          calls_to_replace.push_back(call_op);
        }
      }
    });

    // Replace each call inline
    for (auto call_op : calls_to_replace) {
      llvm::StringRef callee_name = *call_op.getCallee();
      builder.setInsertionPoint(call_op);
      auto loc = call_op.getLoc();
      auto i32_type = builder.getI32Type();

      if (absl::StartsWith(callee_name, "xla_get_thread_id")) {
        // Replace with direct intrinsic call (backend-specific)
        auto intrinsic_name = builder.getStringAttr(
            target == TargetBackend::CUDA ? "llvm.nvvm.read.ptx.sreg.tid.x"
                                          : "llvm.amdgcn.workitem.id.x");
        auto intrinsic_call = builder.create<LLVM::CallIntrinsicOp>(
            loc, i32_type, intrinsic_name, mlir::ValueRange{});
        call_op.replaceAllUsesWith(intrinsic_call->getResults());
        call_op.erase();

      } else if (absl::StartsWith(callee_name, "xla_atomic_write_")) {
        // Expected operand layout: [ptr, value, mask?]
        auto operands = call_op.getOperands();
        auto addr = operands[0];
        auto value = operands[1];
        mlir::Value mask = operands.size() > 2 ? operands[2] : mlir::Value{};

        llvm::StringRef syncscope = ParseSyncScope(callee_name, target);
        LLVM::AtomicOrdering ordering = ParseAtomicOrdering(callee_name);

        // Prepare atomic store location
        if (mask) {
          // Masked atomic: if (mask != 0) { atomic_store } else { nop }
          auto current_block = call_op->getBlock();
          auto atomic_block = current_block->splitBlock(call_op);
          auto exit_block = atomic_block->splitBlock(call_op);

          // Check mask and branch
          builder.setInsertionPointToEnd(current_block);
          auto zero = builder.create<LLVM::ConstantOp>(
              loc, i32_type, builder.getI32IntegerAttr(0));
          auto mask_nonzero = builder.create<LLVM::ICmpOp>(
              loc, LLVM::ICmpPredicate::ne, mask, zero);
          LLVM::CondBrOp::create(builder, loc, mask_nonzero, atomic_block,
                                 exit_block);

          // Set insertion point for atomic store
          builder.setInsertionPointToStart(atomic_block);
        }

        // Perform atomic store (atomic due to ordering parameter)
        builder.create<LLVM::StoreOp>(
            loc, value, addr, /*alignment=*/4, /*isVolatile=*/false,
            /*isNonTemporal=*/false, /*isInvariantGroup=*/false, ordering,
            builder.getStringAttr(syncscope));

        if (mask) {
          // Complete masked path: branch to exit
          auto exit_block = builder.getBlock()->getNextNode();
          builder.create<LLVM::BrOp>(loc, exit_block);
          builder.setInsertionPointToStart(exit_block);
        }

        // Return poison value (result not expected to be used)
        auto poison = builder.create<LLVM::PoisonOp>(loc, i32_type);
        call_op.replaceAllUsesWith(mlir::ValueRange{poison});
        call_op.erase();

      } else if (absl::StartsWith(callee_name, "xla_atomic_spin_wait_")) {
        // Expected operand layout: [ptr, expected, mask?]
        auto operands = call_op.getOperands();
        auto addr = operands[0];
        auto expected = operands[1];
        mlir::Value mask = operands.size() > 2 ? operands[2] : mlir::Value{};

        llvm::StringRef syncscope = ParseSyncScope(callee_name, target);
        LLVM::AtomicOrdering ordering = ParseAtomicOrdering(callee_name);
        // acq_rel is not valid for loads (only for RMW operations)
        if (ordering == LLVM::AtomicOrdering::acq_rel) {
          LOG(FATAL) << "acq_rel ordering is not supported for atomic loads in "
                     << callee_name.str() << ". Use acquire ordering instead.";
        }
        bool is_lt = callee_name.ends_with("_lt");

        // Create block structure (common for both masked and unmasked)
        auto current_block = call_op->getBlock();
        auto loop_block = current_block->splitBlock(call_op);
        auto exit_block = loop_block->splitBlock(call_op);
        exit_block->addArgument(i32_type, loc);

        builder.setInsertionPointToEnd(current_block);

        if (mask) {
          // Masked: conditional branch based on mask (if mask==0, skip loop)
          auto zero = builder.create<LLVM::ConstantOp>(
              loc, i32_type, builder.getI32IntegerAttr(0));
          auto mask_nonzero = builder.create<LLVM::ICmpOp>(
              loc, LLVM::ICmpPredicate::ne, mask, zero);
          LLVM::CondBrOp::create(builder, loc, mask_nonzero, loop_block,
                                 mlir::ValueRange{}, exit_block,
                                 mlir::ValueRange{zero}, std::nullopt);
        } else {
          // Unmasked: unconditional branch to loop (required terminator)
          LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loop_block);
        }

        // Loop: atomic load + compare + conditional branch (atomic due to
        // ordering parameter)
        builder.setInsertionPointToStart(loop_block);
        auto loaded = builder.create<LLVM::LoadOp>(
            loc, i32_type, addr, /*alignment=*/4, /*isVolatile=*/false,
            /*isNonTemporal=*/false, /*isInvariant=*/false,
            /*isInvariantGroup=*/false, ordering,
            builder.getStringAttr(syncscope));

        auto condition =
            is_lt ? builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ult,
                                                 loaded, expected)
                  : builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne,
                                                 loaded, expected);

        LLVM::CondBrOp::create(builder, loc, condition, loop_block,
                               mlir::ValueRange{}, exit_block,
                               mlir::ValueRange{loaded}, std::nullopt);

        // Replace call with exit block argument
        call_op.replaceAllUsesWith(
            mlir::ValueRange{exit_block->getArgument(0)});
        call_op.erase();
      }
    }

    // Clean up unused extern function declarations
    llvm::SmallVector<LLVM::LLVMFuncOp> to_erase;
    module.walk([&](LLVM::LLVMFuncOp func) {
      if (func.isExternal()) {
        llvm::StringRef name = func.getName();
        if (name.starts_with("xla_atomic_write_") ||
            name.starts_with("xla_atomic_spin_wait_") ||
            name.starts_with("xla_get_thread_id")) {
          to_erase.push_back(func);
        }
      }
    });

    for (auto func : to_erase) {
      func.erase();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateTritonXLAImplementExternAtomicsPass(
    TargetBackend target) {
  return std::make_unique<TritonXLAImplementExternAtomicsPass>(target);
}

}  // namespace mlir::triton::xla
