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

#include <iterator>

#include "mlir/IR/Operation.h"  // from @llvm-project
#include "xla/mlir_hlo/gml_st/IR/gml_st_ops.h"
#include "xla/tools/mlir_interpreter/dialects/util.h"
#include "xla/tools/mlir_interpreter/interpreter.h"
#include "xla/tools/mlir_interpreter/interpreter_value.h"
#include "xla/tools/mlir_interpreter/interpreter_value_util.h"

namespace mlir {
namespace interpreter {
namespace {

llvm::SmallVector<InterpreterValue> For(MutableArrayRef<InterpreterValue> args,
                                        mlir::Operation* op,
                                        InterpreterState& state) {
  bool is_bufferized = op->getNumResults() == 0;
  auto gmlst_for = llvm::cast<gml_st::ForOp>(op);

  int64_t num_outputs = gmlst_for.getOutputs().size();
  assert((args.size() - num_outputs) % 3 == 0 &&
         "expected uniform sizes for lbs, ubs and steps");

  int64_t num_loops = (args.size() - num_outputs) / 3;
  auto bound_args = args.take_front(num_loops * 3);
  auto lbs = UnpackInterpreterValues<int64_t>(bound_args.take_front(num_loops));
  auto ubs =
      UnpackInterpreterValues<int64_t>(bound_args.slice(num_loops, num_loops));
  auto steps =
      UnpackInterpreterValues<int64_t>(bound_args.take_back(num_loops));

  SmallVector<InterpreterValue> outputs;
  for (int64_t i = args.size() - num_outputs; i < args.size(); ++i) {
    outputs.push_back(GetInitOperand(op, i, args));
  }

  SmallVector<int64_t> iter_sizes;
  for (auto [lb, ub, step] : llvm::zip(lbs, ubs, steps)) {
    iter_sizes.push_back((ub - lb + (step - 1)) / step);
  }

  // Make a fake buffer view to abuse its index iterator.
  BufferView view{0, iter_sizes, {}};

  for (const auto& indices : view.indices()) {
    SmallVector<InterpreterValue> args;
    for (auto [i, lb, step] : llvm::zip(indices, lbs, steps)) {
      args.push_back(InterpreterValue{i * step + lb});
    }
    llvm::copy(outputs, std::back_inserter(args));

    auto yielded = Interpret(state, op->getRegion(0), args);
    if (state.HasFailure()) break;

    assert(yielded.size() == 3 * num_outputs &&
           "expected equal number of srcs, dsts and sets");

    MutableArrayRef<InterpreterValue> yielded_ref = yielded;

    // The dsts of set yield are always the outputs, so we can ignore them.
    auto srcs = yielded_ref.take_front(num_outputs);
    auto tiles = yielded_ref.take_back(num_outputs);

    for (auto [src, tile, output] : llvm::zip(srcs, tiles, outputs)) {
      ArrayRef<int64_t> tile_args =
          std::get<TensorOrMemref<int64_t>>(tile.storage).buffer->storage;
      if (!src.IsTensor()) {
        output.InsertElement(tile_args.take_front(tile_args.size() / 3), src);
      } else {
        for (const auto& src_indices : src.view().indices()) {
          assert(src_indices.size() * 3 == tile_args.size() &&
                 "mismatched tile/src rank");
          // The sizes of the tile must match the sizes of the src, so we can
          // ignore them.
          auto offsets = tile_args.take_front(src_indices.size());
          auto strides = tile_args.take_back(src_indices.size());

          SmallVector<int64_t> dst_indices;
          for (auto [src_index, offset, stride] :
               llvm::zip(src_indices, offsets, strides)) {
            dst_indices.push_back(src_index * stride + offset);
          }
          output.InsertElement(dst_indices, src.ExtractElement(src_indices));
        }
      }
    }
  }

  if (is_bufferized) return {};
  return outputs;
}

llvm::SmallVector<InterpreterValue> Tile(MutableArrayRef<InterpreterValue> args,
                                         mlir::Operation* op,
                                         InterpreterState& state) {
  auto values = ExtractOffsetsSizesStrides(args, op);
  int64_t rank = static_cast<int64_t>(values.offsets.size());

  auto result = TensorOrMemref<int64_t>::Empty({rank * 3});
  llvm::copy(values.offsets, result.buffer->storage.begin());
  llvm::copy(values.sizes, result.buffer->storage.begin() + rank);
  llvm::copy(values.strides, result.buffer->storage.begin() + 2 * rank);

  return {{result}};
}

llvm::SmallVector<InterpreterValue> Materialize(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState&) {
  auto& src = args[0];
  auto& set = args[1];
  ArrayRef<int64_t> tile_vals =
      std::get<TensorOrMemref<int64_t>>(set.storage).buffer->storage;
  int64_t rank = static_cast<int64_t>(tile_vals.size()) / 3;

  auto offsets = tile_vals.take_front(rank);
  auto sizes = tile_vals.slice(rank, rank);
  auto strides = tile_vals.take_back(rank);

  auto out = src.TypedAlike(sizes);
  out.Fill([&](llvm::ArrayRef<int64_t> indices) {
    llvm::SmallVector<int64_t> src_indices;
    for (int64_t i = 0; i < rank; ++i) {
      src_indices.push_back(indices[i] * strides[i] + offsets[i]);
    }
    return src.ExtractElement(src_indices);
  });

  if (op->getResultTypes().front().isa<ShapedType>()) {
    return {out};
  }
  return {out.ExtractElement({})};
}

REGISTER_MODULE_INITIALIZER(init_gmlst, {
  mlir::interpreter::detail::RegisterInterpreterOp("gml_st.for", For);
  mlir::interpreter::detail::RegisterInterpreterOp("gml_st.tile", Tile);
  mlir::interpreter::detail::RegisterInterpreterOp("gml_st.materialize",
                                                   Materialize);
});

}  // namespace
}  // namespace interpreter
}  // namespace mlir
