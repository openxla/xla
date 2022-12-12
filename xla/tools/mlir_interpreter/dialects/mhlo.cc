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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/tools/mlir_interpreter/dialects/comparators.h"
#include "xla/tools/mlir_interpreter/dialects/util.h"
#include "xla/tools/mlir_interpreter/interpreter.h"
#include "xla/tools/mlir_interpreter/interpreter_value_util.h"
#include "tsl/platform/logging.h"

namespace mlir {
namespace interpreter {
namespace {

// std::plus, etc. are weird with int16_t (returning int for some reason).
template <typename T>
T div(T a, T b) {
  return a / b;
}
template <typename T>
T add(T a, T b) {
  return a + b;
}
template <typename T>
T mul(T a, T b) {
  return a * b;
}
template <typename T>
T sub(T a, T b) {
  return a - b;
}
template <typename T>
T neg(T a) {
  return -a;
}
template <typename T>
T bool_and(T a, T b) {
  if constexpr (std::is_same_v<T, bool>) {
    return a && b;
  }
  llvm_unreachable("and is only supported for bool");
}
template <typename T>
T bool_or(T a, T b) {
  if constexpr (std::is_same_v<T, bool>) {
    return a || b;
  }
  llvm_unreachable("and is only supported for bool");
}

InterpreterValue MakeTuple(MutableArrayRef<InterpreterValue> values) {
  Tuple result;
  for (auto& value : values) {
    result.values.push_back(
        std::make_shared<InterpreterValue>(std::move(value)));
  }
  return {result};
}

llvm::SmallVector<InterpreterValue> BroadcastInDim(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState&) {
  auto broadcast = llvm::cast<mhlo::BroadcastInDimOp>(op);
  auto broadcast_dims = broadcast.getBroadcastDimensions().getValues<int64_t>();

  auto& in = args[0];
  const auto& in_sizes = in.view().sizes;
  auto out =
      in.TypedAlike(op->getResult(0).getType().cast<ShapedType>().getShape());
  out.Fill([&](llvm::ArrayRef<int64_t> out_indices) {
    llvm::SmallVector<int64_t> in_indices;
    for (auto [in_dim, out_dim] : llvm::enumerate(broadcast_dims)) {
      in_indices.push_back(in_sizes[in_dim] == 1 ? 0 : out_indices[out_dim]);
    }
    return in.ExtractElement(in_indices);
  });
  return {out};
}

llvm::SmallVector<InterpreterValue> Reshape(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState& state) {
  auto ty = op->getResultTypes().front().cast<mlir::ShapedType>();
  return {ReshapeTensor(args.front(), ty.getShape())};
}

llvm::SmallVector<InterpreterValue> Slice(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState& state) {
  auto slice = llvm::cast<mhlo::SliceOp>(op);
  auto& in = args.front();
  auto starts = slice.getStartIndices().getValues<int64_t>();
  auto limits = slice.getLimitIndices().getValues<int64_t>();
  auto strides = slice.getStrides().getValues<int64_t>();

  llvm::SmallVector<int64_t> sizes;
  for (auto [start, limit, stride] : llvm::zip(starts, limits, strides)) {
    sizes.push_back(((limit - start) + (stride - 1)) / stride);
  }
  auto result = in.TypedAlike(sizes);
  result.Fill([&](llvm::ArrayRef<int64_t> out_indices) {
    llvm::SmallVector<int64_t> in_indices;
    for (auto [start, stride, index] :
         llvm::zip(starts, strides, out_indices)) {
      in_indices.push_back(start + stride * index);
    }
    return in.ExtractElement(in_indices);
  });
  return {result};
}

llvm::SmallVector<InterpreterValue> Constant(MutableArrayRef<InterpreterValue>,
                                             mlir::Operation* op,
                                             InterpreterState&) {
  auto constant = llvm::cast<mhlo::ConstantOp>(op);
  auto ty = constant->getResultTypes()[0].cast<ShapedType>();
  return {DispatchScalarType(
      ty.getElementType(), [&](auto dummy) -> InterpreterValue {
        auto values = constant.getValue().getValues<decltype(dummy)>();
        auto result = TensorOrMemref<decltype(dummy)>::Empty(ty.getShape());
        llvm::copy(values, result.buffer->storage.begin());
        return {result};
      })};
}

llvm::SmallVector<InterpreterValue> Pad(MutableArrayRef<InterpreterValue> args,
                                        mlir::Operation* op,
                                        InterpreterState&) {
  auto pad = llvm::cast<mhlo::PadOp>(op);
  auto& arg = args[0];
  auto padding_value = args[1].ExtractElement({});

  // TODO(jreiffers): support negative padding

  auto his = pad.getEdgePaddingHigh().getValues<int64_t>();
  auto los = pad.getEdgePaddingLow().getValues<int64_t>();
  auto ins = pad.getInteriorPadding().getValues<int64_t>();

  llvm::SmallVector<int64_t> sizes;
  for (auto [size, lo, in, hi] : llvm::zip(arg.view().sizes, los, ins, his)) {
    sizes.push_back(size + lo + hi + (size - 1) * in);
  }

  auto result = arg.TypedAlike(sizes);
  result.Fill([&](llvm::ArrayRef<int64_t>) { return padding_value; });

  for (const auto& in_indices : arg.view().indices()) {
    llvm::SmallVector<int64_t> out_indices;
    for (auto [in_index, in, lo] : llvm::zip(in_indices, ins, los)) {
      out_indices.push_back(in_index * (in + 1) + lo);
    }
    result.InsertElement(out_indices, arg.ExtractElement(in_indices));
  }

  return {result};
}

llvm::SmallVector<InterpreterValue> Compare(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState&) {
  auto compare = llvm::cast<mhlo::CompareOp>(op);
  switch (compare.getComparisonDirection()) {
    case mlir::mhlo::ComparisonDirection::EQ:
      return {ApplyCWiseBinaryMap<eq>(args[0], args[1])};
    case mlir::mhlo::ComparisonDirection::NE:
      return {ApplyCWiseBinaryMap<ne>(args[0], args[1])};
    case mlir::mhlo::ComparisonDirection::GE:
      return {ApplyCWiseBinaryMap<ge>(args[0], args[1])};
    case mlir::mhlo::ComparisonDirection::GT:
      return {ApplyCWiseBinaryMap<gt>(args[0], args[1])};
    case mlir::mhlo::ComparisonDirection::LE:
      return {ApplyCWiseBinaryMap<le>(args[0], args[1])};
    case mlir::mhlo::ComparisonDirection::LT:
      return {ApplyCWiseBinaryMap<lt>(args[0], args[1])};
  }
}

llvm::SmallVector<InterpreterValue> Gather(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState&) {
  auto gather = llvm::cast<mhlo::GatherOp>(op);

  auto index_vector_dim = gather.getDimensionNumbers().getIndexVectorDim();
  auto start_index_map = gather.getDimensionNumbers().getStartIndexMap();
  auto offset_dims = gather.getDimensionNumbers().getOffsetDims();
  auto collapsed_slice_dims =
      gather.getDimensionNumbers().getCollapsedSliceDims();
  auto slice_sizes = gather.getSliceSizes().getValues<int64_t>();

  auto& operand = args[0];
  auto& start_indices = args[1];
  const auto& operand_view = operand.view();
  int64_t operand_rank = operand_view.rank();

  // Make a fake BufferView for the start indices.
  BufferView start_indices_view = start_indices.view();
  int64_t output_rank = start_indices_view.rank() + offset_dims.size();
  if (index_vector_dim < start_indices_view.rank()) {
    --output_rank;
    start_indices_view.sizes[index_vector_dim] = 1;
  }

  SmallVector<int64_t> batch_dims;
  for (int64_t i = 0; i < output_rank; ++i) {
    if (!llvm::is_contained(offset_dims, i)) {
      batch_dims.push_back(i);
    }
  }

  if (VLOG_IS_ON(10)) {
    llvm::errs() << "batch dims: ";
    llvm::interleaveComma(batch_dims, llvm::errs());
    llvm::errs() << "\n";
  }

  // Make a fake BufferView for the slice indices.
  BufferView slice_indices_view{0, SmallVector<int64_t>{slice_sizes}, {}};

  SmallVector<int64_t> non_collapsed_slice_dims;
  for (int64_t i = 0; i < operand_rank; ++i) {
    if (!llvm::is_contained(collapsed_slice_dims, i)) {
      non_collapsed_slice_dims.push_back(i);
    }
  }

  SmallVector<int64_t> output_sizes(output_rank);
  for (auto [output_dim, slice_dim] :
       llvm::zip(offset_dims, non_collapsed_slice_dims)) {
    output_sizes[output_dim] = slice_sizes[slice_dim];
  }
  for (auto [batch_index, output_dim] : llvm::enumerate(batch_dims)) {
    if (batch_index >= index_vector_dim) {
      ++batch_index;
    }
    output_sizes[output_dim] = start_indices_view.sizes[batch_index];
  }

  if (VLOG_IS_ON(10)) {
    llvm::errs() << "output shape: ";
    llvm::interleaveComma(output_sizes, llvm::errs());
    llvm::errs() << "\n";
  }

  auto output = operand.TypedAlike(output_sizes);
  for (auto start_indices_index : start_indices_view.indices()) {
    SmallVector<int64_t> operand_base_indices(operand_rank);
    for (auto [i, dim] : llvm::enumerate(start_index_map)) {
      if (index_vector_dim < start_indices_view.rank()) {
        start_indices_index[index_vector_dim] = i;
      }
      operand_base_indices[dim] = std::max<int64_t>(
          0, std::min(start_indices.ExtractElement(start_indices_index).AsInt(),
                      operand_view.sizes[dim] - slice_sizes[dim]));
    }

    if (VLOG_IS_ON(10)) {
      llvm::errs() << "base indices: ";
      llvm::interleaveComma(operand_base_indices, llvm::errs());
      llvm::errs() << "\n";
    }

    for (const auto& slice_indices : slice_indices_view.indices()) {
      SmallVector<int64_t> operand_indices;
      for (int64_t i = 0; i < operand_rank; ++i) {
        operand_indices.push_back(operand_base_indices[i] + slice_indices[i]);
      }

      SmallVector<int64_t> output_indices(output_rank);
      for (auto [output_dim, slice_dim] :
           llvm::zip(offset_dims, non_collapsed_slice_dims)) {
        output_indices[output_dim] = slice_indices[slice_dim];
      }
      for (auto [batch_index, output_dim] : llvm::enumerate(batch_dims)) {
        output_indices[output_dim] =
            start_indices_index[batch_index >= index_vector_dim
                                    ? batch_index + 1
                                    : batch_index];
      }

      if (VLOG_IS_ON(10)) {
        llvm::errs() << "source indices: ";
        llvm::interleaveComma(operand_indices, llvm::errs());
        llvm::errs() << "\ndst indices: ";
        llvm::interleaveComma(output_indices, llvm::errs());
        llvm::errs() << "\n";
      }

      auto value = operand.ExtractElement(operand_indices);
      output.InsertElement(output_indices, value);
    }
  }

  return {output};
}

llvm::SmallVector<InterpreterValue> Scatter(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState& state) {
  auto scatter = llvm::cast<mhlo::ScatterOp>(op);
  auto index_vector_dim =
      scatter.getScatterDimensionNumbers().getIndexVectorDim();
  auto scatter_dims_to_operand_dims =
      scatter.getScatterDimensionNumbers().getScatterDimsToOperandDims();
  auto inserted_window_dims =
      scatter.getScatterDimensionNumbers().getInsertedWindowDims();
  auto update_window_dims =
      scatter.getScatterDimensionNumbers().getUpdateWindowDims();

  int64_t n = (args.size() - 1) / 2;
  auto n_inputs = args.take_front(n);
  auto n_updates = args.take_back(n);
  auto& scatter_indices = args[n];
  auto input_view = n_inputs.front().view();
  int64_t operand_rank = input_view.rank();
  int64_t updates_rank = n_updates.front().view().rank();
  int64_t indices_rank = scatter_indices.view().rank();

  llvm::SmallVector<int64_t> batch_dims;
  for (int64_t dim = 0; dim < operand_rank; ++dim) {
    if (!llvm::is_contained(inserted_window_dims, dim)) {
      batch_dims.push_back(dim);
    }
  }

  llvm::SmallVector<int64_t> update_scatter_dims;
  for (int64_t dim = 0; dim < updates_rank; ++dim) {
    if (!llvm::is_contained(update_window_dims, dim)) {
      update_scatter_dims.push_back(dim);
    }
  }

  llvm::SmallVector<InterpreterValue> n_results;
  for (auto& inputs : n_inputs) {
    n_results.push_back(inputs.Clone());
  }

  for (auto [inputs, updates, results] :
       llvm::zip(n_results, n_updates, n_results)) {
    const auto& updates_view = updates.view();
    for (const auto& update_indices : updates_view.indices()) {
      llvm::SmallVector<int64_t> input_indices(operand_rank);
      llvm::SmallVector<int64_t> max_indices(operand_rank);
      llvm::SmallVector<int64_t> min_indices(operand_rank);
      llvm::SmallVector<int64_t> scatter_indices_index(indices_rank);

      for (auto [index, dim] : llvm::enumerate(update_scatter_dims)) {
        scatter_indices_index[index >= index_vector_dim ? index + 1 : index] +=
            update_indices[dim];
      }

      for (auto [update_dim, operand_dim] :
           llvm::zip(update_window_dims, batch_dims)) {
        input_indices[operand_dim] = update_indices[update_dim];
        max_indices[operand_dim] = updates_view.sizes[update_dim];
      }

      for (auto [index, dim] : llvm::enumerate(scatter_dims_to_operand_dims)) {
        if (index_vector_dim < indices_rank) {
          scatter_indices_index[index_vector_dim] = index;
        }

        if (VLOG_IS_ON(10)) {
          llvm::errs() << "scatter indices index: ";
          llvm::interleaveComma(scatter_indices_index, llvm::errs());
          llvm::errs() << "\n";
        }

        int64_t scatter_index =
            scatter_indices.ExtractElement(scatter_indices_index).AsInt();
        input_indices[dim] += scatter_index;
        min_indices[dim] += scatter_index;
        max_indices[dim] += scatter_index;
      }

      if (!input_view.InBounds(min_indices)) continue;
      if (!input_view.InBounds(max_indices)) continue;

      if (VLOG_IS_ON(10)) {
        llvm::errs() << "input indices: ";
        llvm::interleaveComma(input_indices, llvm::errs());
        llvm::errs() << "\nupdate indices: ";
        llvm::interleaveComma(update_indices, llvm::errs());
        llvm::errs() << "\n";
      }

      auto current_value = inputs.ExtractElement(input_indices).AsUnitTensor();
      auto update = updates.ExtractElement(update_indices).AsUnitTensor();

      auto result = Interpret(state, scatter.getUpdateComputation(),
                              {current_value, update})
                        .front()
                        .ExtractElement({});
      if (state.HasFailure()) {
        return n_results;
      }
      inputs.InsertElement(input_indices, result);
    }
  }

  return n_results;
}

llvm::SmallVector<InterpreterValue> Select(
    MutableArrayRef<InterpreterValue> args, mlir::Operation*,
    InterpreterState&) {
  auto& condition = args[0];
  auto& true_vals = args[1];
  auto& false_vals = args[2];

  auto result = true_vals.Clone();
  for (const auto& indices : condition.view().indices()) {
    if (!std::get<bool>(condition.ExtractElement(indices).storage)) {
      result.InsertElement(indices, false_vals.ExtractElement(indices));
    }
  }
  return {result};
}

REGISTER_MLIR_INTERPRETER_BINARY_CWISE(mhlo, add, add);
REGISTER_MLIR_INTERPRETER_BINARY_CWISE(mhlo, and, bool_and);
REGISTER_MLIR_INTERPRETER_BINARY_CWISE(mhlo, or, bool_or);
REGISTER_MLIR_INTERPRETER_BINARY_CWISE(mhlo, divide, div);
REGISTER_MLIR_INTERPRETER_BINARY_CWISE(mhlo, multiply, mul);
REGISTER_MLIR_INTERPRETER_BINARY_CWISE(mhlo, subtract, sub);
REGISTER_MLIR_INTERPRETER_OP(mhlo, broadcast_in_dim, BroadcastInDim);
REGISTER_MLIR_INTERPRETER_OP(mhlo, compare, Compare);
REGISTER_MLIR_INTERPRETER_OP(mhlo, constant, Constant);
REGISTER_MLIR_INTERPRETER_OP(mhlo, pad, Pad);
REGISTER_MLIR_INTERPRETER_OP(mhlo, reshape, Reshape);
REGISTER_MLIR_INTERPRETER_OP(mhlo, gather, Gather);
REGISTER_MLIR_INTERPRETER_OP(mhlo, scatter, Scatter);
REGISTER_MLIR_INTERPRETER_OP(mhlo, select, Select);
REGISTER_MLIR_INTERPRETER_OP(mhlo, slice, Slice);
REGISTER_MLIR_INTERPRETER_OP(mhlo, tuple, MakeTuple);
REGISTER_MLIR_INTERPRETER_UNARY_CWISE(mhlo, cosine, std::cos);
REGISTER_MLIR_INTERPRETER_UNARY_CWISE(mhlo, negate, neg);
REGISTER_MLIR_INTERPRETER_UNARY_CWISE(mhlo, sine, std::sin);
REGISTER_MLIR_INTERPRETER_UNARY_CWISE(mhlo, sqrt, std::sqrt);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
