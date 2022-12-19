/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project

#include "llvm/Support/ErrorHandling.h"
#include "xla/tools/mlir_interpreter/dialects/comparators.h"
#include "xla/tools/mlir_interpreter/dialects/util.h"
#include "xla/tools/mlir_interpreter/interpreter.h"
#include "xla/tools/mlir_interpreter/interpreter_value_util.h"

namespace mlir {
namespace interpreter {
namespace {

llvm::SmallVector<InterpreterValue> Constant(MutableArrayRef<InterpreterValue>,
                                             mlir::Operation* op,
                                             InterpreterState&) {
  auto constant = llvm::cast<arith::ConstantOp>(op);

  auto ty = constant->getResultTypes()[0];
  auto shaped_type = ty.dyn_cast<ShapedType>();
  auto elem_ty = shaped_type ? shaped_type.getElementType() : ty;
  return {DispatchScalarType(elem_ty, [&](auto dummy) -> InterpreterValue {
    using T = decltype(dummy);
    if (shaped_type) {
      auto values =
          constant.getValue().cast<DenseElementsAttr>().getValues<T>();
      auto result = TensorOrMemref<T>::Empty(shaped_type.getShape());
      llvm::copy(values, result.buffer->storage.begin());
      return {result};
    }

    auto value = constant.getValue();
    if (auto integer = value.dyn_cast<IntegerAttr>()) {
      return {static_cast<T>(integer.getInt())};
    }
    if (auto float_ = value.dyn_cast<FloatAttr>()) {
      return {static_cast<T>(float_.getValueAsDouble())};
    }

    llvm_unreachable("unsupported constant type");
  })};
}

// TODO(jreiffers): Support all cases, e.g. narrowing casts from index.
struct IndexCast {
  static int64_t apply(int64_t in) { return in; }
  static int64_t apply(int32_t in) { return in; }

  template <typename T>
  static T apply(T in) {
    llvm_unreachable("unsupported index_cast");
  }
};

llvm::SmallVector<InterpreterValue> CmpI(MutableArrayRef<InterpreterValue> args,
                                         mlir::Operation* op,
                                         InterpreterState&) {
  auto compare = llvm::cast<arith::CmpIOp>(op);
  switch (compare.getPredicate()) {
    case arith::CmpIPredicate::eq:
      return {ApplyCWiseBinaryMap<eq>(args[0], args[1])};
    case arith::CmpIPredicate::ne:
      return {ApplyCWiseBinaryMap<ne>(args[0], args[1])};
    case arith::CmpIPredicate::slt:
      return {ApplyCWiseBinaryMap<lt>(args[0], args[1])};
    case arith::CmpIPredicate::sle:
      return {ApplyCWiseBinaryMap<le>(args[0], args[1])};
    case arith::CmpIPredicate::sgt:
      return {ApplyCWiseBinaryMap<gt>(args[0], args[1])};
    case arith::CmpIPredicate::sge:
      return {ApplyCWiseBinaryMap<ge>(args[0], args[1])};
    case arith::CmpIPredicate::ult:
      return {ApplyCWiseBinaryMap<ult>(args[0], args[1])};
    case arith::CmpIPredicate::ule:
      return {ApplyCWiseBinaryMap<ule>(args[0], args[1])};
    case arith::CmpIPredicate::ugt:
      return {ApplyCWiseBinaryMap<ugt>(args[0], args[1])};
    case arith::CmpIPredicate::uge:
      return {ApplyCWiseBinaryMap<uge>(args[0], args[1])};
  }
}

template <typename T>
T AndI(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return a & b;
  }
  llvm_unreachable("unsupported operator&");
}

template <typename T>
T OrI(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return a | b;
  }
  llvm_unreachable("unsupported operator&");
}

llvm::SmallVector<InterpreterValue> Select(
    MutableArrayRef<InterpreterValue> args, mlir::Operation*,
    InterpreterState&) {
  return {std::get<bool>(args[0].storage) ? args[1] : args[2]};
}

REGISTER_MLIR_INTERPRETER_BINARY_CWISE(arith, andi, AndI);
REGISTER_MLIR_INTERPRETER_BINARY_CWISE(arith, ori, OrI);
REGISTER_MLIR_INTERPRETER_OP(arith, cmpi, CmpI);
REGISTER_MLIR_INTERPRETER_OP(arith, constant, Constant);
REGISTER_MLIR_INTERPRETER_OP(arith, index_cast, ApplyCWiseMap<IndexCast>);
REGISTER_MLIR_INTERPRETER_OP(arith, maxsi, ApplyCWiseBinaryMap<max>);
REGISTER_MLIR_INTERPRETER_OP(arith, minsi, ApplyCWiseBinaryMap<min>);
REGISTER_MLIR_INTERPRETER_OP(arith, select, Select);
REGISTER_MLIR_INTERPRETER_OP_ALIAS(arith, addf, mhlo, add);
REGISTER_MLIR_INTERPRETER_OP_ALIAS(arith, addi, mhlo, add);
REGISTER_MLIR_INTERPRETER_OP_ALIAS(arith, mulf, mhlo, multiply);
REGISTER_MLIR_INTERPRETER_OP_ALIAS(arith, muli, mhlo, multiply);
REGISTER_MLIR_INTERPRETER_OP_ALIAS(arith, subf, mhlo, sub);
REGISTER_MLIR_INTERPRETER_OP_ALIAS(arith, subi, mhlo, sub);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
