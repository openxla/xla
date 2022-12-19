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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_DIALECTS_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_DIALECTS_UTIL_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/Interfaces/ViewLikeInterface.h"  // from @llvm-project
#include "xla/tools/mlir_interpreter/interpreter_value.h"

namespace mlir {
namespace interpreter {

struct OffsetsSizesStrides {
  llvm::SmallVector<int64_t> offsets;
  llvm::SmallVector<int64_t> sizes;
  llvm::SmallVector<int64_t> strides;
};

// Replaces dynamic placeholders in static_vals using elements from the front
// of args, which are removed.
SmallVector<int64_t> ReplaceDynamicVals(llvm::ArrayRef<int64_t> static_vals,
                                        ArrayRef<InterpreterValue>& args);

OffsetsSizesStrides ExtractOffsetsSizesStrides(
    ArrayRef<InterpreterValue> args, OffsetSizeAndStrideOpInterface op);

InterpreterValue ReshapeTensor(const InterpreterValue& in,
                               ArrayRef<int64_t> shape);

int64_t EvalAffineExpr(AffineExpr expr, ArrayRef<int64_t> dims);
llvm::SmallVector<int64_t> EvalAffineMap(AffineMap map, ArrayRef<int64_t> dims);

// Gets the given operand, cloning its storage if it is a tensor.
InterpreterValue GetInitOperand(mlir::Operation* op, int64_t index,
                                MutableArrayRef<InterpreterValue> args);

}  // namespace interpreter
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_DIALECTS_UTIL_H_
