/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/triton/extern_function_helper.h"

#include <string>
#include <variant>
#include <vector>

#include "absl/functional/overload.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/tsl/platform/statusor.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

namespace {

using ::mlir::triton::MemSemantic;
using ::mlir::triton::MemSyncScope;

// Helper to parse MemSemantic from string
absl::StatusOr<MemSemantic> ParseMemSemantic(absl::string_view semantic_str) {
  if (semantic_str == "relaxed") {
    return MemSemantic::RELAXED;
  }
  if (semantic_str == "acquire") {
    return MemSemantic::ACQUIRE;
  }
  if (semantic_str == "release") {
    return MemSemantic::RELEASE;
  }
  if (semantic_str == "acqrel") {
    return MemSemantic::ACQUIRE_RELEASE;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Unknown memory semantic: %s", semantic_str));
}

// Helper to parse MemSyncScope from string
absl::StatusOr<MemSyncScope> ParseMemSyncScope(absl::string_view scope_str) {
  if (scope_str == "system") {
    return MemSyncScope::SYSTEM;
  }
  if (scope_str == "gpu") {
    return MemSyncScope::GPU;
  }
  if (scope_str == "cta") {
    return MemSyncScope::CTA;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Unknown memory sync scope: %s", scope_str));
}

// Helper to parse Comparator from string
absl::StatusOr<Comparator> ParseComparator(absl::string_view comparator_str) {
  if (comparator_str == "eq") {
    return Comparator::EQ;
  }
  if (comparator_str == "lt") {
    return Comparator::LT;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Unknown comparator: %s", comparator_str));
}

// Helper to convert MemSemantic to string
absl::string_view MemSemanticToString(MemSemantic semantic) {
  switch (semantic) {
    case MemSemantic::RELAXED:
      return "relaxed";
    case MemSemantic::ACQUIRE:
      return "acquire";
    case MemSemantic::RELEASE:
      return "release";
    case MemSemantic::ACQUIRE_RELEASE:
      return "acqrel";
  }
  LOG(FATAL) << "Unknown MemSemantic value";
}

// Helper to convert MemSyncScope to string
absl::string_view MemSyncScopeToString(MemSyncScope scope) {
  switch (scope) {
    case MemSyncScope::SYSTEM:
      return "system";
    case MemSyncScope::GPU:
      return "gpu";
    case MemSyncScope::CTA:
      return "cta";
  }
  LOG(FATAL) << "Unknown MemSyncScope value";
}

// Helper to convert Comparator to string
absl::string_view ComparatorToString(Comparator comparator) {
  switch (comparator) {
    case Comparator::EQ:
      return "eq";
    case Comparator::LT:
      return "lt";
  }
  LOG(FATAL) << "Unknown Comparator value";
}

// Helper to convert MemSyncScope to PTX scope string
absl::string_view MemSyncScopeToPTXScope(MemSyncScope scope) {
  switch (scope) {
    case MemSyncScope::SYSTEM:
      return "sys";
    case MemSyncScope::GPU:
      return "gpu";
    case MemSyncScope::CTA:
      return "cta";
  }
  LOG(FATAL) << "Unknown MemSyncScope value";
}

}  // namespace

absl::StatusOr<ExternFunctionInstruction> ParseExternFunctionName(
    absl::string_view func_name) {
  // Function name format: xla_<functionname>_<arg1>_<arg2>_...
  // Split by underscore to get tokens
  std::vector<absl::string_view> tokens = absl::StrSplit(func_name, '_');

  // Must have at least 2 tokens: "xla" and function name
  if (tokens.size() < 2 || tokens[0] != "xla") {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid extern function name: %s", func_name));
  }

  absl::string_view fn_name = tokens[1];

  // xla_getthreadid (2 tokens total)
  if (fn_name == "getthreadid") {
    if (tokens.size() != 2) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "getthreadid expects no arguments, got %d", tokens.size() - 2));
    }
    return GetThreadIdInstruction{};
  }

  // xla_atomicwrite_<semantic>_<scope> (4 tokens total)
  if (fn_name == "atomicwrite") {
    if (tokens.size() != 4) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "atomicwrite expects 2 arguments (semantic, scope), got %d",
          tokens.size() - 2));
    }

    TF_ASSIGN_OR_RETURN(auto semantic, ParseMemSemantic(tokens[2]));
    TF_ASSIGN_OR_RETURN(auto scope, ParseMemSyncScope(tokens[3]));

    return AtomicWriteInstruction{semantic, scope};
  }

  // xla_atomicspinwait_<semantic>_<scope>_<comparator> (5 tokens total)
  if (fn_name == "atomicspinwait") {
    if (tokens.size() != 5) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "atomicspinwait expects 3 arguments (semantic, scope, comparator), "
          "got %d",
          tokens.size() - 2));
    }

    TF_ASSIGN_OR_RETURN(auto semantic, ParseMemSemantic(tokens[2]));
    TF_ASSIGN_OR_RETURN(auto scope, ParseMemSyncScope(tokens[3]));
    TF_ASSIGN_OR_RETURN(auto comparator, ParseComparator(tokens[4]));

    return AtomicSpinWaitInstruction{semantic, scope, comparator};
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Unknown extern function name: %s", func_name));
}

std::string SerializeExternFunctionName(
    const ExternFunctionInstruction& instruction) {
  return std::visit(
      absl::Overload{
          [](const GetThreadIdInstruction&) -> std::string {
            return "xla_getthreadid";
          },
          [](const AtomicWriteInstruction& arg) -> std::string {
            return absl::StrJoin(
                {"xla", "atomicwrite", MemSemanticToString(arg.semantic),
                 MemSyncScopeToString(arg.scope)},
                "_");
          },
          [](const AtomicSpinWaitInstruction& arg) -> std::string {
            return absl::StrJoin(
                {"xla", "atomicspinwait", MemSemanticToString(arg.semantic),
                 MemSyncScopeToString(arg.scope),
                 ComparatorToString(arg.comparator)},
                "_");
          },
      },
      instruction);
}

absl::Status ValidateMemorySemantic(
    const ExternFunctionInstruction& instruction) {
  return std::visit(
      absl::Overload{
          [](const GetThreadIdInstruction&) -> absl::Status {
            // No memory semantic validation needed for GetThreadId
            return absl::OkStatus();
          },
          [](const AtomicWriteInstruction& arg) -> absl::Status {
            // AtomicWrite only supports RELAXED or RELEASE semantics
            if (arg.semantic != MemSemantic::RELAXED &&
                arg.semantic != MemSemantic::RELEASE) {
              return absl::InvalidArgumentError(
                  "AtomicWriteOp only supports RELAXED or RELEASE semantics");
            }
            return absl::OkStatus();
          },
          [](const AtomicSpinWaitInstruction& arg) -> absl::Status {
            // AtomicSpinWait supports RELAXED, ACQUIRE, or ACQUIRE_RELEASE
            // semantics
            if (arg.semantic != MemSemantic::RELAXED &&
                arg.semantic != MemSemantic::ACQUIRE) {
              return absl::InvalidArgumentError(
                  "AtomicSpinWaitOp only supports RELAXED or ACQUIRE "
                  "semantics");
            }
            return absl::OkStatus();
          },
      },
      instruction);
}

namespace {

// Create LLVM ops for GetThreadIdInstruction
mlir::Value CreateGetThreadIdOps(const LLVMOpCreationParams& params) {
  auto& builder = params.builder;
  auto i32_type = builder.getI32Type();

  // Use inline PTX assembly for CUDA
  const absl::string_view get_tid_asm = R"(
    mov.u32 $0, %tid.x;
  )";
  auto asm_op = LLVM::InlineAsmOp::create(
      builder, params.loc, i32_type, mlir::ValueRange{},
      builder.getStringAttr(get_tid_asm), builder.getStringAttr("=r"),
      /*has_side_effects=*/mlir::UnitAttr(),
      /*is_align_stack=*/mlir::UnitAttr(),
      LLVM::TailCallKindAttr::get(builder.getContext(),
                                  LLVM::TailCallKind::None),
      /*asm_dialect=*/LLVM::AsmDialectAttr(),
      /*operand_attrs=*/mlir::ArrayAttr());
  return asm_op.getResult(0);
}

// Create LLVM ops for AtomicWriteInstruction
mlir::Value CreateAtomicWriteOps(const AtomicWriteInstruction& instruction,
                                 const LLVMOpCreationParams& params) {
  auto& builder = params.builder;
  auto operands = params.operands;
  auto i32_type = builder.getI32Type();

  // Expected operand layout: [ptr, value, mask?]
  auto addr = operands[0];
  auto value = operands[1];
  mlir::Value mask = operands.size() > 2 ? operands[2] : mlir::Value{};

  absl::string_view memory_semantic = MemSemanticToString(instruction.semantic);
  absl::string_view scope = MemSyncScopeToPTXScope(instruction.scope);

  // Build PTX inline assembly based on whether mask is present
  if (mask) {
    constexpr absl::string_view kAtomicWriteAsmWithMaskTemplate = R"(
    {
    .reg .pred %%p<>;
    setp.ne.u32 %%p<>, $2, 0;
    @%%p<> st.global.%s.%s.u32 [$0], $1;
    }
  )";
    std::string atomic_write_asm = absl::StrFormat(
        kAtomicWriteAsmWithMaskTemplate, scope, memory_semantic);
    auto asm_op = LLVM::InlineAsmOp::create(
        builder, params.loc, i32_type, mlir::ValueRange{addr, value, mask},
        builder.getStringAttr(atomic_write_asm), builder.getStringAttr("l,r,r"),
        /*has_side_effects=*/builder.getUnitAttr(),
        /*is_align_stack=*/nullptr,
        LLVM::TailCallKindAttr::get(builder.getContext(),
                                    LLVM::TailCallKind::None),
        /*asm_dialect=*/nullptr,
        /*operand_attrs=*/nullptr);
    return asm_op.getResult(0);
  }
  constexpr absl::string_view kAtomicWriteAsmTemplate = R"(
    st.global.%s.%s.u32 [$0], $1;
  )";
  std::string atomic_write_asm =
      absl::StrFormat(kAtomicWriteAsmTemplate, scope, memory_semantic);
  auto asm_op = LLVM::InlineAsmOp::create(
      builder, params.loc, i32_type, mlir::ValueRange{addr, value},
      builder.getStringAttr(atomic_write_asm), builder.getStringAttr("l,r"),
      /*has_side_effects=*/builder.getUnitAttr(),
      /*is_align_stack=*/nullptr,
      LLVM::TailCallKindAttr::get(builder.getContext(),
                                  LLVM::TailCallKind::None),
      /*asm_dialect=*/nullptr,
      /*operand_attrs=*/nullptr);
  return asm_op.getResult(0);
}

// Create LLVM ops for AtomicSpinWaitInstruction
mlir::Value CreateAtomicSpinWaitOps(
    const AtomicSpinWaitInstruction& instruction,
    const LLVMOpCreationParams& params) {
  auto& builder = params.builder;
  auto operands = params.operands;
  auto i32_type = builder.getI32Type();

  // Expected operand layout: [ptr, expected, mask?]
  auto addr = operands[0];
  auto expected = operands[1];
  mlir::Value mask = operands.size() > 2 ? operands[2] : mlir::Value{};

  absl::string_view memory_semantic = MemSemanticToString(instruction.semantic);
  absl::string_view scope = MemSyncScopeToPTXScope(instruction.scope);
  absl::string_view comparator = ComparatorToString(instruction.comparator);

  // Build PTX inline assembly based on whether mask is present
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
    std::string atomic_wait_asm = absl::StrFormat(
        kAtomicSpinWaitAsmWithMaskTemplate, scope, memory_semantic, comparator);
    auto asm_op = LLVM::InlineAsmOp::create(
        builder, params.loc, i32_type, mlir::ValueRange{addr, expected, mask},
        builder.getStringAttr(atomic_wait_asm), builder.getStringAttr("l,r,r"),
        /*has_side_effects=*/builder.getUnitAttr(),
        /*is_align_stack=*/nullptr,
        LLVM::TailCallKindAttr::get(builder.getContext(),
                                    LLVM::TailCallKind::None),
        /*asm_dialect=*/nullptr,
        /*operand_attrs=*/nullptr);
    return asm_op.getResult(0);
  }
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
  std::string atomic_wait_asm = absl::StrFormat(
      kAtomicSpinWaitAsmTemplate, scope, memory_semantic, comparator);
  auto asm_op = LLVM::InlineAsmOp::create(
      builder, params.loc, i32_type, mlir::ValueRange{addr, expected},
      builder.getStringAttr(atomic_wait_asm), builder.getStringAttr("l,r"),
      /*has_side_effects=*/builder.getUnitAttr(),
      /*is_align_stack=*/nullptr,
      LLVM::TailCallKindAttr::get(builder.getContext(),
                                  LLVM::TailCallKind::None),
      /*asm_dialect=*/nullptr,
      /*operand_attrs=*/nullptr);
  return asm_op.getResult(0);
}

}  // namespace

mlir::Value CreateLLVMOpsForInstruction(
    const ExternFunctionInstruction& instruction,
    const LLVMOpCreationParams& params) {
  return std::visit(
      absl::Overload{
          [&params](const GetThreadIdInstruction&) -> mlir::Value {
            return CreateGetThreadIdOps(params);
          },
          [&params](const AtomicWriteInstruction& arg) -> mlir::Value {
            return CreateAtomicWriteOps(arg, params);
          },
          [&params](const AtomicSpinWaitInstruction& arg) -> mlir::Value {
            return CreateAtomicSpinWaitOps(arg, params);
          },
      },
      instruction);
}

}  // namespace mlir::triton::xla
