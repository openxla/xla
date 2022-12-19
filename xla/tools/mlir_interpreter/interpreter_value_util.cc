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

#include "xla/tools/mlir_interpreter/interpreter_value_util.h"

namespace mlir {
namespace interpreter {

template <typename T, template <typename _> class rng_t>
mlir::interpreter::InterpreterValue RandomTensor(
    absl::BitGenRef bitgen, llvm::ArrayRef<int64_t> shape) {
  auto rng = rng_t<T>{};
  auto result = mlir::interpreter::TensorOrMemref<T>::Empty(shape);
  for (auto& elem : result.buffer->storage) {
    elem = rng(bitgen);
    // Ints are typically indices, so scale them down to a more reasonable
    // range.
    if constexpr (std::is_same_v<T, int64_t>) {
      elem >>= 60;
    }
  }
  return {result};
}

mlir::FailureOr<mlir::interpreter::InterpreterValue> MakeRandomInput(
    absl::BitGenRef bitgen, mlir::Type type) {
  if (auto ty = type.dyn_cast<mlir::ShapedType>()) {
    auto elemTy = ty.getElementType();
    if (elemTy.isF32()) {
      return RandomTensor<float, absl::gaussian_distribution>(bitgen,
                                                              ty.getShape());
    }
    if (elemTy.isF64()) {
      return RandomTensor<double, absl::gaussian_distribution>(bitgen,
                                                               ty.getShape());
    }
    if (elemTy.isInteger(16)) {
      return RandomTensor<int16_t, absl::uniform_int_distribution>(
          bitgen, ty.getShape());
    }
    if (elemTy.isInteger(64)) {
      return RandomTensor<int64_t, absl::uniform_int_distribution>(
          bitgen, ty.getShape());
    }
  }
  llvm::errs() << "Unsupported type: ";
  type.print(llvm::errs());
  llvm::errs() << "\n";
  return failure();
}

}  // namespace interpreter
}  // namespace mlir
