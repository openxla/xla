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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_INTERPRETER_VALUE_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_INTERPRETER_VALUE_H_

#include <complex>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <variant>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/tools/mlir_interpreter/tensor_or_memref.h"

namespace mlir {
namespace interpreter {

struct InterpreterValue;

struct Tuple {
  Tuple() = default;

  bool operator==(const Tuple& other) const;

  SmallVector<std::shared_ptr<InterpreterValue>> values;
};

struct InterpreterValue {
  void Print(llvm::raw_ostream& os) const;
  std::string ToString() const;

  InterpreterValue ExtractElement(llvm::ArrayRef<int64_t> indices) const;
  void InsertElement(llvm::ArrayRef<int64_t> indices,
                     const InterpreterValue& value);
  void Fill(
      const std::function<InterpreterValue(llvm::ArrayRef<int64_t> indices)>&
          f);

  // Converts a scalar to a unit tensor.
  InterpreterValue AsUnitTensor() const;
  int64_t AsInt() {
    if (std::holds_alternative<int64_t>(storage)) {
      return std::get<int64_t>(storage);
    }
    if (std::holds_alternative<int32_t>(storage)) {
      return std::get<int32_t>(storage);
    }
    if (std::holds_alternative<int16_t>(storage)) {
      return std::get<int16_t>(storage);
    }
    llvm_unreachable("invalid type");
  }

  // Creates a new tensor InterpreterValue (backed a new buffer) with the same
  // elementtype as this, but a different shape. If this is not a tensor, it is
  // used as the element type.
  InterpreterValue Clone() const;
  InterpreterValue TypedAlike(llvm::ArrayRef<int64_t> shape) const;

  // Creates a tensor with the given element type and shape.
  static InterpreterValue MakeTensor(mlir::Type element_type,
                                     llvm::ArrayRef<int64_t> shape);

  // Returns the underlying tensor's view. Must be a tensor.
  BufferView& view();

  bool IsTensor() const;

  bool operator==(const InterpreterValue& other) const {
    return storage == other.storage;
  }

  std::variant<
      Tuple, bool, float, double, uint16_t, int16_t, int32_t, int64_t,
      std::complex<float>, std::complex<double>, TensorOrMemref<bool>,
      TensorOrMemref<float>, TensorOrMemref<double>, TensorOrMemref<uint16_t>,
      TensorOrMemref<int16_t>, TensorOrMemref<int32_t>, TensorOrMemref<int64_t>,
      TensorOrMemref<std::complex<float>>, TensorOrMemref<std::complex<double>>>
      storage;
};

template <class Fn>
auto DispatchScalarType(mlir::Type ty, Fn&& functor) {
  if (ty.isF32()) {
    return functor((float)0);
  } else if (ty.isF64()) {
    return functor((double)0);
  } else if (ty.isInteger(64) || ty.isIndex()) {
    return functor((int64_t)0);
  } else if (ty.isInteger(32)) {
    return functor((int32_t)0);
  } else if (ty.isInteger(16)) {
    if (ty.isUnsignedInteger(16)) {
      return functor((uint16_t)0);
    }
    return functor((int16_t)0);
  } else if (ty.isInteger(1)) {
    return functor(false);
  }

  llvm::errs() << "DispatchScalarType unimplemented for ";
  ty.print(llvm::errs());
  llvm::errs() << "\n";
  llvm_unreachable("unimplemented");
}

}  // namespace interpreter
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_INTERPRETER_VALUE_H_
