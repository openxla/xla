/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/elemental_ir_emitter.h"

#include <stddef.h>

#include <utility>
#include <vector>

#include "llvm/IR/IntrinsicsNVPTX.h"
#include "tsl/platform/logging.h"
// IWYU pragma: no_include "llvm/IR/Attributes.gen.inc"
// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APInt.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/ModRef.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_loop.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/math_ops.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

using absl::StrAppend;
using llvm_ir::IrArray;
using llvm_ir::IrName;
using llvm_ir::SetToFirstInsertPoint;

namespace {
// Returns whether operand is a floating-point literal with the given value.
bool IsFPLiteralWithValue(const HloInstruction* operand, float value) {
  if (operand->opcode() == HloOpcode::kConstant &&
      operand->literal().IsAllFloat(value)) {
    return true;
  }
  return operand->opcode() == HloOpcode::kBroadcast &&
         IsFPLiteralWithValue(operand->operand(0), value);
}
}  // namespace

GpuElementalIrEmitter::GpuElementalIrEmitter(
    const HloModuleConfig& hlo_module_config, llvm::Module* module,
    llvm::IRBuilder<>* b, NestedComputer compute_nested,
    IrEmitterContext* ir_emitter_context)
    : ElementalIrEmitter(module, b),
      hlo_module_config_(hlo_module_config),
      compute_nested_(std::move(compute_nested)),
      ir_emitter_context_(ir_emitter_context) {}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitDeviceMathCall(
    TargetDeviceFunctionID funcid, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type,
    absl::string_view name) {
  // Device functions dont have f16 math functions, so we convert the operands
  // to f32 before calling the function and then convert the result back to f16.
  bool cast_result_to_fp16 = false;
  std::vector<llvm::Value*> converted_operands(operands.begin(),
                                               operands.end());
  std::vector<PrimitiveType> converted_input_types(input_types.begin(),
                                                   input_types.end());
  switch (output_type) {
    case F16:
      cast_result_to_fp16 = true;
      for (int64_t i = 0; i < operands.size(); ++i) {
        if (input_types[i] == F16) {
          converted_operands[i] =
              FPCast(converted_operands[i], b()->getFloatTy());
          converted_input_types[i] = F32;
        }
      }
      output_type = F32;
      [[fallthrough]];
    case F32:
      break;
    case F64:
      break;
    default:
      return Unimplemented("Bad type for device math call: %s",
                           PrimitiveType_Name(output_type));
  }
  const std::string& munged_callee =
      ObtainDeviceFunctionName(funcid, output_type, b());
  llvm::Value* result = EmitMathCall(munged_callee, converted_operands,
                                     converted_input_types, output_type, name)
                            .value();
  if (cast_result_to_fp16) {
    result = FPCast(result, b()->getHalfTy());
  }
  return result;
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitLlvmIntrinsicMathCall(
    const std::string& callee_name, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type) {
  // llvm intrinsics differentiate between half/float/double functions via
  // the suffixes ".f16", ".f32" and ".f64".
  std::string munged_callee = callee_name;
  switch (output_type) {
    case F16:
      StrAppend(&munged_callee, ".f16");
      break;
    case F32:
      StrAppend(&munged_callee, ".f32");
      break;
    case F64:
      StrAppend(&munged_callee, ".f64");
      break;
    default:
      return Unimplemented("Bad type for llvm intrinsic math call: %s",
                           PrimitiveType_Name(output_type));
  }
  return EmitMathCall(munged_callee, operands, input_types, output_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitMathCall(
    const std::string& callee_name, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type,
    absl::string_view name) {
  // Binary math functions transform are of type [T] -> T.
  for (PrimitiveType input_type : input_types) {
    if (output_type != input_type) {
      return Unimplemented("Input type != output type: %s != %s",
                           PrimitiveType_Name(input_type),
                           PrimitiveType_Name(output_type));
    }
  }

  return EmitDeviceFunctionCall(callee_name, operands, input_types, output_type,
                                llvm::AttrBuilder(b()->getContext())
                                    .addMemoryAttr(llvm::MemoryEffects::none())
                                    .addAttribute(llvm::Attribute::NoUnwind),
                                b(), name);
}

llvm_ir::IrArray::Index GpuElementalIrEmitter::GetSourceIndexOfBitcast(
    const llvm_ir::IrArray::Index& index, const HloInstruction* hlo) {
  Shape shape = hlo->shape();
  Shape operand_shape = hlo->operand(0)->shape();

  // Decode the layout of the shape from the Protobugs attached to
  // backend_config_.
  BitcastBackendConfig bitcast_config;
  CHECK(bitcast_config.ParseFromString(hlo->raw_backend_config_string()));

  *shape.mutable_layout() =
      xla::Layout::CreateFromProto(bitcast_config.result_layout());
  *operand_shape.mutable_layout() =
      xla::Layout::CreateFromProto(bitcast_config.source_layout());
  return index.SourceIndexOfBitcast(shape, operand_shape, b());
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitFloatBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
  PrimitiveType lhs_input_type = op->operand(0)->shape().element_type();
  PrimitiveType rhs_input_type = op->operand(1)->shape().element_type();
  PrimitiveType output_type = op->shape().element_type();
  HloOpcode opcode = op->opcode();

  if (hlo_module_config_.debug_options().xla_gpu_enable_fast_min_max() &&
      (opcode == HloOpcode::kMaximum || opcode == HloOpcode::kMinimum)) {
    return llvm_ir::EmitCallToIntrinsic(
        opcode == HloOpcode::kMaximum ? llvm::Intrinsic::maxnum
                                      : llvm::Intrinsic::minnum,
        {lhs_value, rhs_value}, {lhs_value->getType()}, b());
  }

  switch (op->opcode()) {
    case HloOpcode::kRemainder: {
      return EmitDeviceMathCall(TargetDeviceFunctionID::kFmod,
                                {lhs_value, rhs_value},
                                {lhs_input_type, rhs_input_type}, output_type);
    }
    case HloOpcode::kPower: {
      return EmitPowerOp(op, lhs_value, rhs_value);
    }
    default:
      return ElementalIrEmitter::EmitFloatBinaryOp(op, lhs_value, rhs_value);
  }
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitPowerOp(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
  CHECK_EQ(op->opcode(), HloOpcode::kPower);
  PrimitiveType lhs_input_type = op->operand(0)->shape().element_type();
  PrimitiveType rhs_input_type = op->operand(1)->shape().element_type();
  PrimitiveType output_type = op->shape().element_type();
  return EmitDeviceMathCall(TargetDeviceFunctionID::kPow,
                            {lhs_value, rhs_value},
                            {lhs_input_type, rhs_input_type}, output_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitLog(PrimitiveType prim_type,
                                                      llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kLog, {value}, {prim_type},
                            prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitLog1p(PrimitiveType prim_type,
                                                        llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kLog1p, {value},
                            {prim_type}, prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitSin(PrimitiveType prim_type,
                                                      llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kSin, {value}, {prim_type},
                            prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitCos(PrimitiveType prim_type,
                                                      llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kCos, {value}, {prim_type},
                            prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitTan(PrimitiveType prim_type,
                                                      llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kTan, {value}, {prim_type},
                            prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitExp(
    PrimitiveType prim_type, llvm::Value* value, absl::string_view /*name*/) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kExp, {value}, {prim_type},
                            prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitExpm1(PrimitiveType prim_type,
                                                        llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kExpm1, {value},
                            {prim_type}, prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitPow(PrimitiveType prim_type,
                                                      llvm::Value* lhs,
                                                      llvm::Value* rhs,
                                                      absl::string_view name) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kPow, {lhs, rhs},
                            {prim_type, prim_type}, prim_type, name);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitSqrt(PrimitiveType prim_type,
                                                       llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kSqrt, {value}, {prim_type},
                            prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitRsqrt(PrimitiveType prim_type,
                                                        llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kRsqrt, {value},
                            {prim_type}, prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitAtan2(
    PrimitiveType prim_type, llvm::Value* lhs, llvm::Value* rhs,
    absl::string_view name) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kAtan2, {lhs, rhs},
                            {prim_type, prim_type}, prim_type, name);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitTanh(PrimitiveType prim_type,
                                                       llvm::Value* value) {
  // When F64 is being requested, assume performance is less important and use
  // the more numerically precise tanh function.
  if (prim_type == F64) {
    return EmitDeviceMathCall(TargetDeviceFunctionID::kTanh, {value},
                              {prim_type}, prim_type);
  }

  // Emit a fast approximation of tanh instead of calling __nv_tanh.
  // __nv_tanh is particularly bad because it contains branches, thus
  // preventing LLVM's load-store vectorizer from working its magic across a
  // function which contains tanh calls.
  //
  // This routine isn't numerically precise, but it's good enough for ML.

  // Upcast F16 to F32 if necessary.
  llvm::Type* type = prim_type == F16 ? b()->getFloatTy() : value->getType();
  llvm::Value* input = FPCast(value, type);

  // If |value| >= kMaxValue, tanh() is set to -1.0 or 1.0.
  constexpr double kMaxValue = 20.0;
  auto max_value = llvm::ConstantFP::get(type, kMaxValue);
  llvm::Value* abs_value =
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {input}, {type}, b());

  llvm::Value* fast_tanh = llvm_ir::EmitFastTanh(b(), input);
  auto one = llvm::ConstantFP::get(type, 1.0);
  auto one_with_sign = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::copysign,
                                                    {one, input}, {type}, b());
  return FPCast(Select(FCmpULT(abs_value, max_value), fast_tanh, one_with_sign),
                value->getType(), "tanh");
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitComplexAbs(
    PrimitiveType prim_type, llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kHypot,
                            {EmitExtractReal(value), EmitExtractImag(value)},
                            {prim_type, prim_type}, prim_type);
}

llvm::Value* GpuElementalIrEmitter::EmitThreadId() {
  llvm::Value* block_id = IntCast(
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kBlockIdx, {}, {}, b()),
      b()->getIntNTy(128), /*isSigned=*/true, "block.id");
  llvm::Value* thread_id_in_block = IntCast(
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kThreadIdx, {}, {}, b()),
      b()->getIntNTy(128), /*isSigned=*/true, "thread.id");
  llvm::Value* threads_per_block = IntCast(
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kBlockDimx, {}, {}, b()),
      b()->getIntNTy(128), /*isSigned=*/true, "threads_per_block");
  return NSWAdd(NSWMul(block_id, threads_per_block), thread_id_in_block);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitF32ToBF16(
    llvm::Value* f32_value) {
  if (ir_emitter_context_->cuda_compute_capability().IsAtLeast(8)) {
    return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::nvvm_f2bf16_rn,
                                        {f32_value}, {}, b());
  } else {
    // More complex fallback solution.
    return ElementalIrEmitter::EmitF32ToBF16(f32_value);
  }
}

namespace {

llvm::Value* EmitF8ToF16InlineAsm(llvm::Value* f8_value, PrimitiveType f8_type,
                                  llvm::IRBuilder<>* b) {
  // LLVM IR does not yet have an FP8 support, so we represent FP8 values using
  // int8 and use inline PTX assembly to convert to a wider type.
  // TODO(b/259609697): Once LLVM IR supports FP8, use LLVM conversions instead
  // of inline assembly
  //
  // PTX only supports converting from two packed FP8 values, so we extend the
  // input width and truncate the output.
  llvm::Value* as_int16 = b->CreateZExt(f8_value, b->getInt16Ty());
  std::string ptx_packed_f8_type;
  if (f8_type == F8E5M2) {
    ptx_packed_f8_type = "e5m2x2";
  } else {
    CHECK(f8_type == F8E4M3FN);  // Crash OK
    ptx_packed_f8_type = "e4m3x2";
  }
  llvm::FunctionType* func_type =
      llvm::FunctionType::get(b->getInt32Ty(), {b->getInt16Ty()},
                              /*isVarArg=*/false);
  llvm::InlineAsm* inline_asm = llvm::InlineAsm::get(
      func_type,
      absl::StrCat("{ cvt.rn.f16x2.", ptx_packed_f8_type, " $0, $1; }"), "=r,h",
      /*hasSideEffects=*/false);
  llvm::Value* asm_output = b->CreateCall(inline_asm, {as_int16});
  llvm::Value* truncated = b->CreateTrunc(asm_output, b->getInt16Ty());
  return b->CreateBitCast(truncated, b->getHalfTy());
}

}  // namespace

llvm::Value* GpuElementalIrEmitter::EmitF8e5m2ToF16(llvm::Value* f8_value) {
  if (ir_emitter_context_->cuda_compute_capability().IsAtLeast(9)) {
    return EmitF8ToF16InlineAsm(f8_value, F8E5M2, b());
  } else {
    // More complex fallback solution.
    return ElementalIrEmitter::EmitF8e5m2ToF16(f8_value);
  }
}

llvm::Value* GpuElementalIrEmitter::EmitF8e4m3fnToF16(llvm::Value* f8_value) {
  if (ir_emitter_context_->cuda_compute_capability().IsAtLeast(9)) {
    return EmitF8ToF16InlineAsm(f8_value, F8E4M3FN, b());
  } else {
    // More complex fallback solution.
    return ElementalIrEmitter::EmitF8e4m3fnToF16(f8_value);
  }
}

}  // namespace gpu
}  // namespace xla
