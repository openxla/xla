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

#include "mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project

// clang-format erroneously puts the Linalg header above.
#include <functional>  // NOLINT

#include "llvm/ADT/STLExtras.h"
#include "xla/tools/mlir_interpreter/dialects/util.h"
#include "xla/tools/mlir_interpreter/interpreter.h"

namespace mlir {
namespace interpreter {
namespace {

llvm::SmallVector<InterpreterValue> Generic(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState& state) {
  bool is_bufferized = op->getNumResults() == 0;
  auto generic = llvm::cast<linalg::GenericOp>(*op);

  auto ranges = generic.getStaticLoopRanges();
  auto indexing_maps = generic.getIndexingMapsArray();

  for (auto range : ranges) {
    (void)range;
    // TODO(jreiffers): Support this.
    assert(!mlir::ShapedType::isDynamic(range) &&
           "Dynamic ranges not supported yet.");
  }

  int64_t num_inputs = generic.getInputs().size();
  int64_t num_outputs = generic.getOutputs().size();

  llvm::SmallVector<InterpreterValue> outputs;
  for (int64_t output = 0; output < generic.getOutputs().size(); ++output) {
    outputs.push_back(GetInitOperand(op, num_inputs + output, args));
  }

  llvm::SmallVector<int64_t> ivs(ranges.size());
  llvm::SmallVector<InterpreterValue> inputs;

  InterpreterScope scope(state);

  std::function<void(int64_t)> run;
  run = [&](int64_t loop_index) {
    // Abort recursion if we encountered some error previously.s
    if (state.HasFailure()) return;

    if (loop_index < ranges.size()) {
      for (int64_t index = 0; index < ranges[loop_index]; ++index) {
        scope.SetIterationIndex(loop_index, index);
        ivs[loop_index] = index;
        run(loop_index + 1);
      }
    } else {
      llvm::SmallVector<InterpreterValue> bbargs;
      // Build bbargs: 1. inputs, 2. outputs.
      for (int64_t input = 0; input < num_inputs; ++input) {
        auto indices = EvalAffineMap(indexing_maps[input], ivs);
        bbargs.push_back(args[input].ExtractElement(indices));
      }
      for (int64_t output = 0; output < num_outputs; ++output) {
        auto indices = EvalAffineMap(indexing_maps[num_inputs + output], ivs);
        bbargs.push_back(outputs[output].ExtractElement(indices));
      }
      // Evaluate region.
      auto yielded = Interpret(state, generic.getRegion(), bbargs);
      if (state.HasFailure()) return;
      // Insert yielded values in the outputs.
      for (int64_t output = 0; output < generic.getOutputs().size(); ++output) {
        auto indices = EvalAffineMap(indexing_maps[num_inputs + output], ivs);
        outputs[output].InsertElement(indices, yielded[output]);
      }
    }
  };
  run(0);

  if (is_bufferized) return {};
  return outputs;
}

llvm::SmallVector<InterpreterValue> Map(MutableArrayRef<InterpreterValue> args,
                                        mlir::Operation* op,
                                        InterpreterState& state) {
  bool is_bufferized = op->getNumResults() == 0;
  InterpreterValue output = GetInitOperand(op, args.size() - 1, args);

  InterpreterScope scope(state);
  for (const auto& indices : args.front().view().indices()) {
    for (auto [dim, index] : llvm::enumerate(indices)) {
      scope.SetIterationIndex(dim, index);
    }
    llvm::SmallVector<InterpreterValue> inputs;
    for (auto& arg : args.drop_back(1)) {
      inputs.push_back(arg.ExtractElement(indices));
    }
    auto yielded = Interpret(state, op->getRegion(0), inputs);
    if (state.HasFailure()) break;
    output.InsertElement(indices, yielded[0]);
  }

  if (is_bufferized) return {};
  return {output};
}

llvm::SmallVector<InterpreterValue> Fill(MutableArrayRef<InterpreterValue> args,
                                         mlir::Operation* op,
                                         InterpreterState& state) {
  bool is_bufferized = op->getNumResults() == 0;
  InterpreterValue output = GetInitOperand(op, 1, args);
  output.Fill([&](llvm::ArrayRef<int64_t>) { return args[0]; });
  if (is_bufferized) return {};
  return {output};
}

llvm::SmallVector<InterpreterValue> Index(MutableArrayRef<InterpreterValue>,
                                          mlir::Operation* op,
                                          InterpreterState& state) {
  auto index = llvm::cast<linalg::IndexOp>(*op);
  return {{state.GetTopScope()->GetIterationIndex(index.getDim())}};
}

REGISTER_MLIR_INTERPRETER_OP(linalg, generic, Generic);
REGISTER_MLIR_INTERPRETER_OP(linalg, map, Map);
REGISTER_MLIR_INTERPRETER_OP(linalg, fill, Fill);
REGISTER_MLIR_INTERPRETER_OP(linalg, index, Index);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
