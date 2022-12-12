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

#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project

// clang-format erroneously puts the MemRef header above.
#include <algorithm>  // NOLINT
#include <limits>     // NOLINT
#include <variant>    // NOLINT

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "xla/tools/mlir_interpreter/dialects/util.h"
#include "xla/tools/mlir_interpreter/interpreter.h"
#include "xla/tools/mlir_interpreter/interpreter_value_util.h"
#include "tsl/platform/logging.h"

namespace mlir {
namespace interpreter {
namespace {

InterpreterValue Load(MutableArrayRef<InterpreterValue> args) {
  return args[0].ExtractElement(
      UnpackInterpreterValues<int64_t>(args.drop_front(1)));
}

void Store(MutableArrayRef<InterpreterValue> args) {
  args[1].InsertElement(UnpackInterpreterValues<int64_t>(args.drop_front(2)),
                        args[0]);
}

llvm::SmallVector<InterpreterValue> Alloc(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState& state) {
  auto ty = op->getResultTypes().front().cast<mlir::ShapedType>();
  auto shape = ReplaceDynamicVals(ty.getShape(), args);
  // TODO(jreiffers): Layout map support.
  return {DispatchScalarType(
      ty.getElementType(), [&](auto dummy) -> InterpreterValue {
        return {TensorOrMemref<decltype(dummy)>::Empty(shape)};
      })};
}

void Dealloc(MutableArrayRef<InterpreterValue> args) {}

void Copy(MutableArrayRef<InterpreterValue> args) {
  args[1].Fill([&](llvm::ArrayRef<int64_t> indices) {
    return args[0].ExtractElement(indices);
  });
}

llvm::SmallVector<InterpreterValue> Subview(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState& state) {
  auto& in = args.front();
  auto v = ExtractOffsetsSizesStrides(args.drop_front(1), op);

  const auto& in_view = in.view();
  llvm::SmallVector<int64_t> strides;
  for (auto [in_stride, subview_stride] :
       llvm::zip(in_view.strides, v.strides)) {
    strides.push_back(in_stride * subview_stride);
  }

  auto out = in;
  out.view() = {in_view.physical_index(v.offsets), v.sizes, strides};
  return {out};
}

llvm::SmallVector<InterpreterValue> CollapseShape(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState& state) {
  auto collapse = llvm::cast<memref::CollapseShapeOp>(op);

  BufferView input_view = args[0].view();

  InterpreterValue out = args[0];
  auto& out_view = out.view();
  out_view.sizes.clear();
  out_view.strides.clear();

  llvm::SmallVector<int64_t> strides;
  for (const auto& group : collapse.getReassociationIndices()) {
    int64_t& size = out_view.sizes.emplace_back(1);
    int64_t& stride =
        out_view.strides.emplace_back(std::numeric_limits<int64_t>::max());
    for (int64_t dim : group) {
      size *= input_view.sizes[dim];
      stride = std::min(stride, input_view.strides[dim]);
    }
  }

  return {out};
}

llvm::SmallVector<InterpreterValue> GetGlobal(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState& state) {
  auto get_global = llvm::cast<memref::GetGlobalOp>(op);
  auto global = llvm::cast<memref::GlobalOp>(
      state.GetSymbols().lookup(get_global.getName()));

  auto value = global.getConstantInitValue();
  assert(value && "mutable globals are not implemented");

  auto ty = get_global->getResultTypes()[0].cast<ShapedType>();
  return {DispatchScalarType(
      ty.getElementType(), [&](auto dummy) -> InterpreterValue {
        auto values = value.getValues<decltype(dummy)>();
        auto result = TensorOrMemref<decltype(dummy)>::Empty(ty.getShape());
        llvm::copy(values, result.buffer->storage.begin());
        return {result};
      })};
}

REGISTER_MLIR_INTERPRETER_OP(memref, alloc, Alloc);
REGISTER_MLIR_INTERPRETER_OP(memref, collapse_shape, CollapseShape);
REGISTER_MLIR_INTERPRETER_OP(memref, copy, Copy);
REGISTER_MLIR_INTERPRETER_OP(memref, dealloc, Dealloc);
REGISTER_MLIR_INTERPRETER_OP(memref, get_global, GetGlobal);
REGISTER_MLIR_INTERPRETER_OP(memref, load, Load);
REGISTER_MLIR_INTERPRETER_OP(memref, store, Store);
REGISTER_MLIR_INTERPRETER_OP(memref, subview, Subview);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
