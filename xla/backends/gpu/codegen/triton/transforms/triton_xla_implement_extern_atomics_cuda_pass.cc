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

// CUDA-specific implementation of extern_elementwise atomic functions.
// This pass runs in the Triton CUDA pipeline and inlines the implementations
// of custom atomic functions by replacing llvm.call operations with PTX
// inline assembly operations.

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLAIMPLEMENTEXTERNATOMICSCUDAPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

absl::string_view GetMemorySemanticStr(llvm::StringRef func_name) {
  if (func_name.contains("_relaxed_")) {
    return "relaxed";
  } else if (func_name.contains("_acquire_")) {
    return "acquire";
  } else if (func_name.contains("_release_")) {
    return "release";
  } else if (func_name.contains("_acq_rel_")) {
    return "acq_rel";
  }
  LOG(FATAL) << "Unable to parse memory semantic from function name: "
             << func_name.str();
}

absl::string_view GetMemSyncScopeStr(llvm::StringRef func_name) {
  if (func_name.contains("_system")) {
    return "sys";
  } else if (func_name.contains("_gpu")) {
    return "gpu";
  } else if (func_name.contains("_cta")) {
    return "cta";
  }
  LOG(FATAL) << "Unable to parse sync scope from function name: "
             << func_name.str();
}

absl::string_view GetComparatorStr(llvm::StringRef func_name) {
  if (func_name.ends_with("_eq")) {
    return "eq";
  } else if (func_name.ends_with("_lt")) {
    return "lt";
  }
  LOG(FATAL) << "Unable to parse comparator from function name: "
             << func_name.str();
}

// MLIR pass that inlines extern function calls with PTX inline assembly
class TritonXLAImplementExternAtomicsCudaPass
    : public impl::TritonXLAImplementExternAtomicsCudaPassBase<
          TritonXLAImplementExternAtomicsCudaPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module.getContext());

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

    // Replace each call with inline PTX assembly
    for (auto call_op : calls_to_replace) {
      llvm::StringRef callee_name = *call_op.getCallee();
      builder.setInsertionPoint(call_op);
      auto loc = call_op.getLoc();
      auto i32_type = builder.getI32Type();

      if (absl::StartsWith(callee_name, "xla_get_thread_id")) {
        // Replace with PTX inline assembly to get thread ID
        const absl::string_view get_tid_asm = R"(
    mov.u32 $0, %tid.x;
  )";
        auto asm_op = LLVM::InlineAsmOp::create(
            builder, loc,
            /*outputType=*/i32_type,
            /*operands=*/mlir::ValueRange{},
            /*asm_string=*/get_tid_asm,
            /*constraints=*/"=r",
            /*has_side_effects=*/false,
            /*is_align_stack=*/false,
            /*tail_call_kind=*/LLVM::TailCallKind::None,
            /*asm_dialect=*/LLVM::AsmDialectAttr{},
            /*operand_attrs=*/mlir::ArrayAttr{});
        call_op.replaceAllUsesWith(asm_op->getResults());
        call_op.erase();

      } else if (absl::StartsWith(callee_name, "xla_atomic_write_")) {
        // Expected operand layout: [ptr, value, mask?]
        auto operands = call_op.getOperands();
        auto addr = operands[0];
        auto value = operands[1];
        mlir::Value mask = operands.size() > 2 ? operands[2] : mlir::Value{};

        absl::string_view memory_semantic = GetMemorySemanticStr(callee_name);
        absl::string_view scope = GetMemSyncScopeStr(callee_name);

        // Build PTX inline assembly based on whether mask is present
        std::string atomic_write_asm;
        std::string constraints;
        mlir::ValueRange asm_operands;

        if (mask) {
          constexpr absl::string_view kAtomicWriteAsmWithMaskTemplate = R"(
    {
    .reg .pred %%p<>;
    setp.ne.u32 %%p<>, $2, 0;
    @%%p st.global.%s.%s.u32 [$0], $1;
    }
  )";
          atomic_write_asm = absl::StrFormat(kAtomicWriteAsmWithMaskTemplate,
                                             scope, memory_semantic);
          constraints = "l,r,r";
          asm_operands = mlir::ValueRange{addr, value, mask};
        } else {
          constexpr absl::string_view kAtomicWriteAsmTemplate = R"(
    st.global.%s.%s.u32 [$0], $1;
  )";
          atomic_write_asm =
              absl::StrFormat(kAtomicWriteAsmTemplate, scope, memory_semantic);
          constraints = "l,r";
          asm_operands = mlir::ValueRange{addr, value};
        }

        auto asm_op = LLVM::InlineAsmOp::create(
            builder, loc,
            /*outputType=*/i32_type,
            /*operands=*/asm_operands,
            /*asm_string=*/atomic_write_asm,
            /*constraints=*/constraints,
            /*has_side_effects=*/true,
            /*is_align_stack=*/false,
            /*tail_call_kind=*/LLVM::TailCallKind::None,
            /*asm_dialect=*/LLVM::AsmDialectAttr{},
            /*operand_attrs=*/mlir::ArrayAttr{});
        call_op.replaceAllUsesWith(asm_op->getResults());
        call_op.erase();

      } else if (absl::StartsWith(callee_name, "xla_atomic_spin_wait_")) {
        // Expected operand layout: [ptr, expected, mask?]
        auto operands = call_op.getOperands();
        auto addr = operands[0];
        auto expected = operands[1];
        mlir::Value mask = operands.size() > 2 ? operands[2] : mlir::Value{};

        absl::string_view memory_semantic = GetMemorySemanticStr(callee_name);
        absl::string_view scope = GetMemSyncScopeStr(callee_name);
        absl::string_view comparator = GetComparatorStr(callee_name);

        // Build PTX inline assembly based on whether mask is present
        std::string atomic_wait_asm;
        std::string constraints;
        mlir::ValueRange asm_operands;

        if (mask) {
          constexpr absl::string_view kAtomicSpinWaitAsmWithMaskTemplate = R"(
    {
    .reg .pred %%p<2>;
    .reg .b32 %%r<1>;
    setp.ne.u32 %%p0, $2, 0;
    @%%!p0 bra done;
    wait:
      ld.global.%s.%s.u32 %%r0, [$0];
      setp.%s.u32 %%p1, %%r0, $1;
      @%%p1 bra wait;
    done:
    }
  )";
          atomic_wait_asm = absl::StrFormat(kAtomicSpinWaitAsmWithMaskTemplate,
                                            scope, memory_semantic, comparator);
          constraints = "l,r,r";
          asm_operands = mlir::ValueRange{addr, expected, mask};
        } else {
          constexpr absl::string_view kAtomicSpinWaitAsmTemplate = R"(
    {
    .reg .pred %%p<1>;
    .reg .b32 %%r<1>;
    wait:
      ld.global.%s.%s.u32 %%r0, [$0];
      setp.%s.u32 %%p0, %%r0, $1;
      @%%p0 bra wait;
    }
  )";
          atomic_wait_asm = absl::StrFormat(kAtomicSpinWaitAsmTemplate, scope,
                                            memory_semantic, comparator);
          constraints = "l,r";
          asm_operands = mlir::ValueRange{addr, expected};
        }

        auto asm_op = LLVM::InlineAsmOp::create(
            builder, loc,
            /*outputType=*/i32_type,
            /*operands=*/asm_operands,
            /*asm_string=*/atomic_wait_asm,
            /*constraints=*/constraints,
            /*has_side_effects=*/true,
            /*is_align_stack=*/false,
            /*tail_call_kind=*/LLVM::TailCallKind::None,
            /*asm_dialect=*/LLVM::AsmDialectAttr{},
            /*operand_attrs=*/mlir::ArrayAttr{});
        call_op.replaceAllUsesWith(asm_op->getResults());
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

    VLOG(2) << "TritonXLAImplementExternAtomicsCudaPass: Replaced "
            << calls_to_replace.size() << " calls, removed " << to_erase.size()
            << " declarations";
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateTritonXLAImplementExternAtomicsCudaPass() {
  return std::make_unique<TritonXLAImplementExternAtomicsCudaPass>();
}

}  // namespace mlir::triton::xla
