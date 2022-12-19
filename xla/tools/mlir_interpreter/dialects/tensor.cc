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

#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project

// clang-format erroneously puts the Tensor.h header above.
#include <variant>  // NOLINT

#include "llvm/ADT/STLExtras.h"
#include "xla/tools/mlir_interpreter/dialects/util.h"
#include "xla/tools/mlir_interpreter/interpreter.h"
#include "xla/tools/mlir_interpreter/interpreter_value_util.h"

namespace mlir {
namespace interpreter {
namespace {

llvm::SmallVector<InterpreterValue> Empty(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState& state) {
  auto ty = op->getResultTypes().front().cast<mlir::ShapedType>();
  auto shape = ReplaceDynamicVals(ty.getShape(), args);
  return {DispatchScalarType(
      ty.getElementType(), [&](auto dummy) -> InterpreterValue {
        return {TensorOrMemref<decltype(dummy)>::Empty(shape)};
      })};
}

InterpreterValue Extract(MutableArrayRef<InterpreterValue> args) {
  return args[0].ExtractElement(
      UnpackInterpreterValues<int64_t>(args.drop_front(1)));
}

llvm::SmallVector<InterpreterValue> FromElements(
    MutableArrayRef<InterpreterValue> elements, mlir::Operation* op,
    InterpreterState& state) {
  auto ty = op->getResultTypes().front().cast<mlir::ShapedType>();
  auto shape = ty.getShape();
  return {DispatchScalarType(
      ty.getElementType(), [&](auto dummy) -> InterpreterValue {
        auto tensor = TensorOrMemref<decltype(dummy)>::Empty(shape);
        for (int64_t i = 0; i < tensor.buffer->storage.size(); ++i) {
          tensor.buffer->storage[i] =
              std::get<decltype(dummy)>(elements[i].storage);
        }
        return {tensor};
      })};
}

llvm::SmallVector<InterpreterValue> TensorReshape(
    MutableArrayRef<InterpreterValue> elements, mlir::Operation* op,
    InterpreterState& state) {
  auto ty = op->getResultTypes().front().cast<mlir::ShapedType>();
  return {ReshapeTensor(elements.front(), ty.getShape())};
}

llvm::SmallVector<InterpreterValue> ExtractSlice(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState& state) {
  auto extract = llvm::cast<tensor::ExtractSliceOp>(op);
  auto& in = args.front();
  auto v = ExtractOffsetsSizesStrides(args.drop_front(1), op);
  int64_t rank = v.offsets.size();
  auto out = in.TypedAlike(v.sizes);
  out.Fill([&](llvm::ArrayRef<int64_t> indices) {
    llvm::SmallVector<int64_t> src_indices;
    for (int64_t i = 0; i < rank; ++i) {
      src_indices.push_back(indices[i] * v.strides[i] + v.offsets[i]);
    }
    return in.ExtractElement(src_indices);
  });

  int64_t num_dropped = 0;
  auto& out_view = out.view();
  for (int64_t bit : extract.getDroppedDims().set_bits()) {
    assert(out_view.sizes[bit - num_dropped] == 1 && "Can only drop unit dims");
    out_view.sizes.erase(out_view.sizes.begin() + (bit - num_dropped));
    out_view.strides.erase(out_view.strides.begin() + (bit - num_dropped));
    ++num_dropped;
  }
  return {out};
}

llvm::SmallVector<InterpreterValue> InsertSlice(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState& state) {
  auto insert = llvm::cast<tensor::InsertSliceOp>(op);
  auto& src = args[0];
  auto dest = args[1].Clone();
  auto v = ExtractOffsetsSizesStrides(args.drop_front(2), op);

  auto static_sizes = insert.getStaticSizes();
  llvm::SmallVector<int64_t> inserted_dims;
  auto src_size_it = src.view().sizes.begin();
  for (auto [dim, size] : llvm::enumerate(static_sizes)) {
    if (*src_size_it != size) {
      assert(size == 1 && "Can only insert unit dims");
      inserted_dims.push_back(dim);
    } else {
      ++src_size_it;
    }
  }

  for (const auto& src_indices : src.view().indices()) {
    llvm::SmallVector<int64_t> src_with_inserted_dims = src_indices;
    for (int64_t dim : inserted_dims) {
      src_with_inserted_dims.insert(src_with_inserted_dims.begin() + dim, 0);
    }
    llvm::SmallVector<int64_t> dst_indices;
    for (auto [src_index, stride, offset] :
         llvm::zip(src_with_inserted_dims, v.strides, v.offsets)) {
      dst_indices.push_back(src_index * stride + offset);
    }
    dest.InsertElement(dst_indices, src.ExtractElement(src_indices));
  }
  return {dest};
}

llvm::SmallVector<InterpreterValue> Generate(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState& state) {
  auto ty = op->getResultTypes().front().cast<ShapedType>();
  auto sizes = ReplaceDynamicVals(ty.getShape(), args);

  auto result = InterpreterValue::MakeTensor(ty.getElementType(), sizes);
  result.Fill([&](ArrayRef<int64_t> indices) {
    return Interpret(state, op->getRegion(0),
                     PackInterpreterValues<int64_t>(indices))
        .front();
  });
  return {result};
}

llvm::SmallVector<InterpreterValue> Insert(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState& state) {
  auto value = args[0];
  auto result = args[1].Clone();
  auto indices = UnpackInterpreterValues<int64_t>(args.drop_front(2));
  result.InsertElement(indices, value);
  return {result};
}

REGISTER_MLIR_INTERPRETER_OP(tensor, collapse_shape, TensorReshape);
REGISTER_MLIR_INTERPRETER_OP(tensor, empty, Empty);
REGISTER_MLIR_INTERPRETER_OP(tensor, expand_shape, TensorReshape);
REGISTER_MLIR_INTERPRETER_OP(tensor, extract, Extract);
REGISTER_MLIR_INTERPRETER_OP(tensor, extract_slice, ExtractSlice);
REGISTER_MLIR_INTERPRETER_OP(tensor, from_elements, FromElements);
REGISTER_MLIR_INTERPRETER_OP(tensor, generate, Generate);
REGISTER_MLIR_INTERPRETER_OP(tensor, insert, Insert);
REGISTER_MLIR_INTERPRETER_OP(tensor, insert_slice, InsertSlice);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
