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

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project

#include "xla/tools/mlir_interpreter/dialects/util.h"
#include "xla/tools/mlir_interpreter/interpreter.h"

namespace mlir {
namespace interpreter {
namespace {

InterpreterValue ToTensor(InterpreterValue& in) { return in.Clone(); }

InterpreterValue ToMemref(InterpreterValue& in) { return in; }

llvm::SmallVector<InterpreterValue> AllocTensor(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState& state) {
  auto alloc = llvm::cast<bufferization::AllocTensorOp>(op);
  auto ty = op->getResultTypes().front().cast<mlir::ShapedType>();
  auto shape = ReplaceDynamicVals(ty.getShape(), args);

  if (alloc.getCopy()) {
    return {args[0].Clone()};
  }
  return {InterpreterValue::MakeTensor(ty.getElementType(), shape)};
}

REGISTER_MLIR_INTERPRETER_OP(bufferization, to_tensor, ToTensor);
REGISTER_MLIR_INTERPRETER_OP(bufferization, to_memref, ToMemref);
REGISTER_MLIR_INTERPRETER_OP(bufferization, alloc_tensor, AllocTensor);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
