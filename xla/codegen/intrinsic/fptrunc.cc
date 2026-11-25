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

#include "xla/codegen/intrinsic/fptrunc.h"

#include <cstddef>
#include <cstdint>
#include <optional>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
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
      module->getOrInsertFunction(FpTrunc::Name({from, to}), function_type)
          .getCallee());
  func->getArg(0)->setName("arg");
  func->setDoesNotAccessMemory();
  func->setDoesNotThrow();
  func->setWillReturn();
  return func;
}

// Truncates an f32 value (scalar or vector) to bf16 with correct rounding.
llvm::Function* TruncateF32ToBf16(llvm::Module* module, Type from, Type to) {
  llvm::LLVMContext& context = module->getContext();
  llvm::IRBuilder<> builder(context);
  llvm::Function* func = CreateFunction(module, from, to);

  // Wraps a scalar type into a vector type if we are building a vector
  // intrinsic declaration.
  auto vec = [&](llvm::Type* scalar_type) -> llvm::Type* {
    if (from.vector_width().has_value()) {
      return llvm::VectorType::get(scalar_type, *from.vector_width(), false);
    }
    return scalar_type;
  };

  llvm::Type* i16_type = vec(builder.getInt16Ty());
  llvm::Type* i32_type = vec(builder.getInt32Ty());
  llvm::Type* bf16_type = vec(builder.getBFloatTy());

  llvm::Argument* arg = func->getArg(0);
  llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(context, "entry", func);
  builder.SetInsertPoint(entry_bb);

  auto* i32 = builder.CreateBitCast(arg, i32_type);

  // Rounding bias for non-nan values.
  auto* lsb = builder.CreateAnd(builder.CreateLShr(i32, 16),
                                llvm::ConstantInt::get(i32_type, 1));
  auto* rounding_bias =
      builder.CreateAdd(llvm::ConstantInt::get(i32_type, 0x7fff), lsb);

  // For NaNs, we set all of them to quiet NaNs by masking the mantissa
  // so that only the MSB is 1, then simply truncate the original value
  // to retain the sign.
  auto* is_nan = builder.createIsFPClass(arg, llvm::FPClassTest::fcNan);
  auto* nan_mask = llvm::ConstantInt::get(i32_type, 0xFFC00000);
  auto* msb = llvm::ConstantInt::get(i32_type, 0x00400000);
  auto* quiet_nan = builder.CreateOr(builder.CreateAnd(i32, nan_mask), msb);
  auto* i16 = builder.CreateTrunc(
      builder.CreateLShr(
          builder.CreateSelect(is_nan, quiet_nan,
                               builder.CreateAdd(i32, rounding_bias)),
          16),
      i16_type);

  llvm::Value* result = builder.CreateBitCast(i16, bf16_type);
  builder.CreateRet(result);

  return func;
}

// Branchless integer bit conversion (RTNE) from `f32` or `f64` to an 8-bit
// float, supporting both scalar and vector types.
llvm::Value* EmitWideFloatToByteFloat(llvm::IRBuilder<>& b,
                                      PrimitiveType src_type,
                                      PrimitiveType dst_type,
                                      std::optional<size_t> vector_width,
                                      llvm::Value* in) {
  DCHECK(src_type == F32 || src_type == F64);
  llvm::LLVMContext& context = b.getContext();
  const llvm::fltSemantics* sem = GetByteFloatSemantics(dst_type);
  CHECK_NE(sem, nullptr);

  const bool is_f64 = (src_type == F64);
  const int fx_bits = is_f64 ? 64 : 32;
  const int fx_mant = is_f64 ? 52 : 23;
  const int fx_bias = is_f64 ? 1023 : 127;
  const uint64_t abs_mask = is_f64 ? 0x7FFFFFFFFFFFFFFFULL : 0x7FFFFFFFULL;
  const uint64_t inf_bits = is_f64 ? 0x7FF0000000000000ULL : 0x7F800000ULL;

  llvm::Type* ix_type =
      Type(is_f64 ? U64 : U32, vector_width).to_ir_type(context);
  llvm::Type* fx_type = Type(src_type, vector_width).to_ir_type(context);
  llvm::Type* i8_type = Type(U8, vector_width).to_ir_type(context);
  auto cx = [&](uint64_t v) { return llvm::ConstantInt::get(ix_type, v); };

  llvm::Value* u = b.CreateBitCast(in, ix_type);
  llvm::Value* sign_bit = b.CreateLShr(u, cx(fx_bits - 1));
  llvm::Value* abs_u = b.CreateAnd(u, cx(abs_mask));

  const bool has_inf = primitive_util::HasInfinity(dst_type);
  const bool is_fnuz = !has_inf && !primitive_util::HasNegativeZero(dst_type);
  const uint64_t max_finite =
      llvm::APFloat::getLargest(*sem).bitcastToAPInt().getZExtValue();
  const uint64_t pos_inf =
      llvm::APFloat::getInf(*sem).bitcastToAPInt().getZExtValue();
  const uint64_t pos_nan =
      llvm::APFloat::getNaN(*sem).bitcastToAPInt().getZExtValue();

  if (dst_type == F8E8M0FNU) {
    llvm::Value* res;
    if (is_f64) {
      // In f64, 2^-127 is normal (biased exponent 896). Any positive value
      // below 2^-127 clamps to 0x00 (the minimum representable value).
      const uint64_t min_e8m0_bits = uint64_t{896} << 52;
      const uint64_t half_ulp = uint64_t{1} << 51;
      llvm::Value* normal_res = b.CreateLShr(
          b.CreateAdd(b.CreateSub(abs_u, cx(min_e8m0_bits)), cx(half_ulp)),
          cx(52));
      res = b.CreateSelect(b.CreateICmpUGE(abs_u, cx(min_e8m0_bits)),
                           normal_res, cx(0));
    } else {
      // In f32, 2^-127 is subnormal (0x00400000); 1.5 * 2^-127 is 0x00600000.
      llvm::Value* normal_res =
          b.CreateLShr(b.CreateAdd(abs_u, cx(0x00400000u)), cx(23));
      llvm::Value* subnorm_res =
          b.CreateSelect(b.CreateICmpUGE(abs_u, cx(0x00600000u)), cx(1), cx(0));
      res = b.CreateSelect(b.CreateICmpUGE(abs_u, cx(0x00800000u)), normal_res,
                           subnorm_res);
    }
    res =
        b.CreateSelect(b.CreateICmpUGT(res, cx(max_finite)), cx(pos_nan), res);
    llvm::Value* is_neg_or_zero = b.CreateOr(b.CreateICmpNE(sign_bit, cx(0)),
                                             b.CreateICmpEQ(abs_u, cx(0)));
    res = b.CreateSelect(is_neg_or_zero, cx(pos_nan), res);
    return b.CreateTrunc(res, i8_type);
  }

  const int mant_bits = llvm::APFloat::semanticsPrecision(*sem) - 1;
  const int bias = 1 - llvm::APFloat::semanticsMinExponent(*sem);
  const int mant_shift = fx_mant - mant_bits;
  const uint64_t bias_diff = static_cast<uint64_t>(fx_bias - bias) << fx_mant;
  const uint64_t half_m1 = (uint64_t{1} << (mant_shift - 1)) - 1;
  const uint64_t min_normal = static_cast<uint64_t>(fx_bias + 1 - bias)
                              << fx_mant;
  // Subnormal RTNE magic constant: 2^(fx_mant + 1 - bias - mant_bits), whose
  // ULP is the exact subnormal quantum 2^(1 - bias - mant_bits). For any
  // 0 <= |x| < 2^(1 - bias), fadd_rn(|x|, magic) lies in [magic, 2*magic) and
  // performs a single exact RTNE rounding, yielding the subnormal code (or
  // smallest normal code 2^mant_bits on carry) in the low bits.
  const uint64_t magic_bits =
      static_cast<uint64_t>(fx_bias + fx_mant + 1 - bias - mant_bits)
      << fx_mant;

  // Normal path: abs_u >= min_normal >= bias_diff, and max abs_u minus
  // bias_diff + half_m1 + 1 never overflows ix_type.
  llvm::Value* n_sub = b.CreateSub(abs_u, cx(bias_diff));
  llvm::Value* n_lsb = b.CreateAnd(b.CreateLShr(n_sub, cx(mant_shift)), cx(1));
  llvm::Value* n_raw = b.CreateLShr(
      b.CreateAdd(b.CreateAdd(n_sub, cx(half_m1)), n_lsb), cx(mant_shift));
  llvm::Value* n_res =
      (pos_inf == max_finite + 1)
          ? static_cast<llvm::Value*>(b.CreateIntrinsic(
                llvm::Intrinsic::umin, {ix_type}, {n_raw, cx(pos_inf)}))
          : b.CreateSelect(b.CreateICmpUGT(n_raw, cx(max_finite)), cx(pos_inf),
                           n_raw);

  // Subnormal path: clamp to min_normal so the input to fadd is always finite
  // and non-negative (avoiding poison under fast-math nnan/ninf and blocking
  // FMA contraction with upstream ops), then do a single hardware fadd_rn.
  llvm::Value* s_abs_u = b.CreateIntrinsic(llvm::Intrinsic::umin, {ix_type},
                                           {abs_u, cx(min_normal)});
  llvm::Value* abs_fp = b.CreateBitCast(s_abs_u, fx_type);
  llvm::Value* magic_fp = b.CreateBitCast(cx(magic_bits), fx_type);
  llvm::Value* s_res = b.CreateSub(
      b.CreateBitCast(b.CreateFAdd(abs_fp, magic_fp), ix_type), cx(magic_bits));

  llvm::Value* res =
      b.CreateSelect(b.CreateICmpUGE(abs_u, cx(min_normal)), n_res, s_res);

  llvm::Value* sign = b.CreateShl(sign_bit, cx(7));
  llvm::Value* signed_res = b.CreateOr(sign, res);
  if (is_fnuz) {
    res = b.CreateSelect(b.CreateICmpEQ(res, cx(0)), cx(0), signed_res);
  } else {
    res = signed_res;
  }

  if (has_inf) {
    // Inf already maps through n_res -> pos_inf | sign. Only NaN needs an
    // explicit override because pos_nan != pos_inf.
    llvm::Value* nan_val = b.CreateOr(sign, cx(pos_nan));
    res = b.CreateSelect(b.CreateICmpUGT(abs_u, cx(inf_bits)), nan_val, res);
  }
  // For non-Inf formats (f8e4m3fn and FNUZ), pos_inf == pos_nan and
  // (pos_nan | sign) == neg_nan, so n_res already produces the exact Inf and
  // NaN encoding without an extra select.

  return b.CreateTrunc(res, i8_type);
}

absl::StatusOr<llvm::Function*> EmitFxxToF8E(llvm::Module* module,
                                             const Type& from, const Type& to) {
  if (!from.is_scalar() && !from.is_vector()) {
    return absl::InvalidArgumentError("from_type must be a scalar or vector.");
  }
  ABSL_RETURN_IF_ERROR(Type::VerifySameWidth(from, to));

  const PrimitiveType fx_type = from.element_type();
  const PrimitiveType f8_type = to.element_type();
  if (GetByteFloatSemantics(f8_type) == nullptr) {
    return Internal("Unsupported fptrunc conversion: from=%s to=%s",
                    from.name(), to.name());
  }

  llvm::LLVMContext& context = module->getContext();
  llvm::IRBuilder<> b(context);
  llvm::Function* func = CreateFunction(module, from, to);
  llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(context, "entry", func);
  b.SetInsertPoint(entry_bb);
  llvm::Value* from_value = func->getArg(0);

  if (fx_type == F32 || fx_type == F64) {
    b.CreateRet(EmitWideFloatToByteFloat(b, fx_type, f8_type,
                                         from.vector_width(), from_value));
    return func;
  }
  if (fx_type == BF16) {
    // bf16 -> f32 is lossless, so truncating via f32 avoids double rounding.
    llvm::Type* i16_type = Type(U16, from.vector_width()).to_ir_type(context);
    llvm::Type* i32_type = Type(U32, from.vector_width()).to_ir_type(context);
    llvm::Type* f32_type = Type(F32, from.vector_width()).to_ir_type(context);
    llvm::Value* as_i32 =
        b.CreateZExt(b.CreateBitCast(from_value, i16_type), i32_type);
    llvm::Value* f32 = b.CreateBitCast(b.CreateShl(as_i32, 16), f32_type);
    b.CreateRet(
        EmitWideFloatToByteFloat(b, F32, f8_type, from.vector_width(), f32));
    return func;
  }
  if (fx_type == F16) {
    // f16 -> f32 is lossless, so truncating via f32 avoids double rounding.
    llvm::Type* f32_type = Type(F32, from.vector_width()).to_ir_type(context);
    llvm::Value* f32 = b.CreateFPExt(from_value, f32_type);
    b.CreateRet(
        EmitWideFloatToByteFloat(b, F32, f8_type, from.vector_width(), f32));
    return func;
  }

  func->eraseFromParent();
  return absl::InvalidArgumentError(
      "from_type must be F16, BF16, F32, or F64.");
}

}  // namespace

absl::StatusOr<llvm::Function*> FpTrunc::CreateDefinition(llvm::Module* module,
                                                          Type from, Type to) {
  ABSL_RETURN_IF_ERROR(Type::VerifySameWidth(from, to));

  if (primitive_util::IsF8Type(to.element_type()) &&
      (from.element_type() == F16 || from.element_type() == BF16 ||
       from.element_type() == F32 || from.element_type() == F64)) {
    return EmitFxxToF8E(module, from, to);
  }
  if (from.element_type() == F32 && to.element_type() == BF16) {
    return TruncateF32ToBf16(module, from, to);
  }

  return Internal("Unsupported fptrunc conversion: from=%s to=%s", from.name(),
                  to.name());
}

}  // namespace xla::codegen::intrinsics
