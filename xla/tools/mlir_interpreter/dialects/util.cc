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

#include "xla/tools/mlir_interpreter/dialects/util.h"

#include <variant>

#include "mlir/Support/MathExtras.h"  // from @llvm-project

namespace mlir {
namespace interpreter {

SmallVector<int64_t> ReplaceDynamicVals(llvm::ArrayRef<int64_t> static_vals,
                                        ArrayRef<InterpreterValue>& args) {
  llvm::SmallVector<int64_t> out;
  for (int64_t val : static_vals) {
    if (ShapedType::isDynamic(val)) {
      out.push_back(std::get<int64_t>(args.front().storage));
      args = args.drop_front(1);
    } else {
      out.push_back(val);
    }
  }
  return out;
}

OffsetsSizesStrides ExtractOffsetsSizesStrides(
    ArrayRef<InterpreterValue> args, OffsetSizeAndStrideOpInterface op) {
  auto offsets = ReplaceDynamicVals(op.static_offsets(), args);
  auto sizes = ReplaceDynamicVals(op.static_sizes(), args);
  auto strides = ReplaceDynamicVals(op.static_strides(), args);
  return {offsets, sizes, strides};
}

struct ReshapeVisitor {
  ArrayRef<int64_t> shape;

  template <typename T>
  InterpreterValue operator()(const TensorOrMemref<T>& in) {
    // This doesn't need a copy in many cases, but it's easier that way.
    auto out = TensorOrMemref<T>::Empty(shape);
    for (auto [in_index, out] :
         llvm::zip(in.view.physical_indices(), out.buffer->storage)) {
      out = in.buffer->storage[in_index];
    }
    return {out};
  }

  template <typename T>
  InterpreterValue operator()(const T& t) {
    llvm_unreachable("reshape is only defined for tensors");
  }
};

InterpreterValue ReshapeTensor(const InterpreterValue& in,
                               ArrayRef<int64_t> shape) {
  return std::visit(ReshapeVisitor{shape}, in.storage);
}

int64_t EvalAffineExpr(AffineExpr expr, ArrayRef<int64_t> dims) {
  switch (expr.getKind()) {
    case AffineExprKind::Add:
      return EvalAffineExpr(expr.cast<AffineBinaryOpExpr>().getLHS(), dims) +
             EvalAffineExpr(expr.cast<AffineBinaryOpExpr>().getRHS(), dims);
    case AffineExprKind::Mul:
      return EvalAffineExpr(expr.cast<AffineBinaryOpExpr>().getLHS(), dims) *
             EvalAffineExpr(expr.cast<AffineBinaryOpExpr>().getRHS(), dims);
    case AffineExprKind::Mod:
      return mod(
          EvalAffineExpr(expr.cast<AffineBinaryOpExpr>().getLHS(), dims),
          EvalAffineExpr(expr.cast<AffineBinaryOpExpr>().getRHS(), dims));
    case AffineExprKind::FloorDiv:
      return floorDiv(
          EvalAffineExpr(expr.cast<AffineBinaryOpExpr>().getLHS(), dims),
          EvalAffineExpr(expr.cast<AffineBinaryOpExpr>().getRHS(), dims));
    case AffineExprKind::CeilDiv:
      return ceilDiv(
          EvalAffineExpr(expr.cast<AffineBinaryOpExpr>().getLHS(), dims),
          EvalAffineExpr(expr.cast<AffineBinaryOpExpr>().getRHS(), dims));
    case AffineExprKind::Constant:
      return expr.cast<AffineConstantExpr>().getValue();
    case AffineExprKind::DimId:
      return dims[expr.cast<AffineDimExpr>().getPosition()];
    case AffineExprKind::SymbolId:
      llvm_unreachable("Symbol is unsupported");
  }
}

llvm::SmallVector<int64_t> EvalAffineMap(AffineMap map,
                                         ArrayRef<int64_t> dims) {
  llvm::SmallVector<int64_t> result;
  for (auto expr : map.getResults()) {
    result.push_back(EvalAffineExpr(expr, dims));
  }
  return result;
}

InterpreterValue GetInitOperand(mlir::Operation* op, int64_t index,
                                MutableArrayRef<InterpreterValue> args) {
  return op->getOperand(index).getType().isa<TensorType>() ? args[index].Clone()
                                                           : args[index];
}

}  // namespace interpreter
}  // namespace mlir
