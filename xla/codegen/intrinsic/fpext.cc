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

#include "xla/codegen/intrinsic/fpext.h"

#include <cstddef>
#include <cstdint>
#include <optional>

#include "absl/log/check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "xla/codegen/intrinsic/type.h"
#include "xla/primitive_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {
namespace {

llvm::Function* CreateFunction(llvm::Module* module, Type from, Type to) {
  DCHECK_OK(Type::VerifySameWidth(from, to));
  llvm::LLVMContext& context = module->getContext();
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      to.to_ir_type(context), {from.to_ir_type(context)},
      /*isVarArg=*/false);
  llvm::Function* func = llvm::dyn_cast<llvm::Function>(
      module->getOrInsertFunction(FpExt::Name({from, to}), function_type)
          .getCallee());
  func->getArg(0)->setName("arg");
  func->setDoesNotAccessMemory();
  func->setDoesNotThrow();
  func->setWillReturn();
  return func;
}

// Branchless integer bit conversion from an 8-bit float (`f8e4m3fn`, `f8e5m2`,
// `f8e4m3`, `f8e3m4`, `f8e4m3fnuz`, `f8e5m2fnuz`, `f8e4m3b11fnuz`,
// `f8e8m0fnu`) to `f32`, supporting both scalar and vector types.
llvm::Value* EmitByteFloatToF32(llvm::IRBuilder<>& b, PrimitiveType src_elem,
                                std::optional<size_t> vector_width,
                                llvm::Value* in) {
  llvm::LLVMContext& context = b.getContext();
  const llvm::fltSemantics* sem = GetByteFloatSemantics(src_elem);
  CHECK_NE(sem, nullptr);

  llvm::Type* i32_type = Type(U32, vector_width).to_ir_type(context);
  llvm::Type* f32_type = Type(F32, vector_width).to_ir_type(context);
  auto c32 = [&](uint32_t v) { return llvm::ConstantInt::get(i32_type, v); };

  llvm::Value* u = b.CreateZExt(in, i32_type);

  if (src_elem == F8E8M0FNU) {
    // 0x00 -> 2^-127 (0x00400000), 0x01..0xFE -> u << 23, 0xFF -> qNaN.
    llvm::Value* res = b.CreateShl(u, c32(23));
    llvm::Value* is_min = b.CreateICmpEQ(u, c32(0));
    res = b.CreateSelect(is_min, c32(0x00400000u), res);
    llvm::Value* is_nan = b.CreateICmpEQ(u, c32(0xFFu));
    res = b.CreateSelect(is_nan, c32(0x7FC00000u), res);
    return b.CreateBitCast(res, f32_type);
  }

  const int mant_bits = llvm::APFloat::semanticsPrecision(*sem) - 1;
  const int bias = 1 - llvm::APFloat::semanticsMinExponent(*sem);
  const int mant_shift = 23 - mant_bits;
  const uint32_t bias_diff = static_cast<uint32_t>(127 - bias) << 23;
  const bool has_inf = primitive_util::HasInfinity(src_elem);
  const bool is_fnuz = !has_inf && !primitive_util::HasNegativeZero(src_elem);

  llvm::Value* abs_u = b.CreateAnd(u, c32(0x7Fu));
  llvm::Value* sign = b.CreateShl(b.CreateAnd(u, c32(0x80u)), c32(24));

  // Normals: re-bias exponent and shift mantissa into position.
  llvm::Value* norm =
      b.CreateAdd(b.CreateShl(abs_u, c32(mant_shift)), c32(bias_diff));

  // Subnormals and zero: abs_u in [0, 2^mant_bits - 1] multiplied by the exact
  // power-of-two subnormal quantum 2^(1 - bias - mant_bits) produces exact f32
  // values (including +0.0) in two vector instructions with no clz or spills.
  const uint32_t sub_scale_bits = static_cast<uint32_t>(128 - bias - mant_bits)
                                  << 23;
  llvm::Value* sub_scale = b.CreateBitCast(c32(sub_scale_bits), f32_type);
  llvm::Value* sub_f32 =
      b.CreateFMul(b.CreateUIToFP(abs_u, f32_type), sub_scale);
  llvm::Value* sub = b.CreateBitCast(sub_f32, i32_type);

  llvm::Value* is_normal = b.CreateICmpUGE(abs_u, c32(1u << mant_bits));
  llvm::Value* val = b.CreateSelect(is_normal, norm, sub);

  if (has_inf) {
    const uint32_t inf_abs =
        llvm::APFloat::getInf(*sem).bitcastToAPInt().getZExtValue();
    llvm::Value* is_inf = b.CreateICmpEQ(abs_u, c32(inf_abs));
    val = b.CreateSelect(is_inf, c32(0x7F800000u), val);
    llvm::Value* is_nan = b.CreateICmpUGT(abs_u, c32(inf_abs));
    val = b.CreateSelect(is_nan, c32(0x7FC00000u), val);
    return b.CreateBitCast(b.CreateOr(val, sign), f32_type);
  }

  if (!is_fnuz) {
    llvm::Value* is_nan = b.CreateICmpEQ(abs_u, c32(0x7Fu));
    val = b.CreateSelect(is_nan, c32(0x7FC00000u), val);
    return b.CreateBitCast(b.CreateOr(val, sign), f32_type);
  }

  // FNUZ: 0x80 is NaN (with sign cleared in f32), 0x00 is +0.0, 0x81..0xFF are
  // negative finite values.
  llvm::Value* res = b.CreateOr(val, sign);
  llvm::Value* is_nan = b.CreateICmpEQ(u, c32(0x80u));
  res = b.CreateSelect(is_nan, c32(0x7FC00000u), res);
  return b.CreateBitCast(res, f32_type);
}

}  // namespace

absl::StatusOr<llvm::Function*> FpExt::CreateDefinition(llvm::Module* module,
                                                        Type from, Type to) {
  ABSL_RETURN_IF_ERROR(Type::VerifySameWidth(from, to));

  const PrimitiveType src_elem = from.element_type();
  const PrimitiveType dst_elem = to.element_type();
  if (GetByteFloatSemantics(src_elem) == nullptr) {
    return Internal("Unsupported fpext conversion: from=%s to=%s", from.name(),
                    to.name());
  }

  llvm::LLVMContext& context = module->getContext();
  llvm::IRBuilder<> b(context);
  llvm::Function* func = CreateFunction(module, from, to);
  llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(context, "entry", func);
  b.SetInsertPoint(entry_bb);
  llvm::Value* in = func->getArg(0);

  if (src_elem == F8E5M2 && dst_elem == F16) {
    llvm::Type* i16_type = Type(U16, from.vector_width()).to_ir_type(context);
    llvm::Value* as_i16 = b.CreateZExt(in, i16_type);
    llvm::Value* shifted = b.CreateShl(as_i16, 8);
    b.CreateRet(b.CreateBitCast(shifted, to.to_ir_type(context)));
    return func;
  }

  llvm::Value* f32 = EmitByteFloatToF32(b, src_elem, from.vector_width(), in);

  if (dst_elem == F32) {
    b.CreateRet(f32);
    return func;
  }
  if (dst_elem == F64) {
    b.CreateRet(b.CreateFPExt(f32, to.to_ir_type(context)));
    return func;
  }
  if (dst_elem == BF16) {
    // Every value of all 8-bit float formats (finite, Inf, and canonical qNaN
    // 0x7FC00000) is exact in bf16, so the low 16 bits of the f32 result are
    // always zero and truncating is a simple right shift.
    llvm::Type* i32_type = Type(U32, from.vector_width()).to_ir_type(context);
    llvm::Type* i16_type = Type(U16, from.vector_width()).to_ir_type(context);
    llvm::Value* as_i32 = b.CreateBitCast(f32, i32_type);
    llvm::Value* hi16 = b.CreateTrunc(b.CreateLShr(as_i32, 16), i16_type);
    b.CreateRet(b.CreateBitCast(hi16, to.to_ir_type(context)));
    return func;
  }
  if (dst_elem == F16) {
    b.CreateRet(b.CreateFPTrunc(f32, to.to_ir_type(context)));
    return func;
  }

  func->eraseFromParent();
  return Internal("Unsupported fpext conversion: from=%s to=%s", from.name(),
                  to.name());
}

}  // namespace xla::codegen::intrinsics
