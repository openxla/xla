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

#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project

#include "xla/tools/mlir_interpreter/interpreter.h"

namespace mlir {
namespace interpreter {
namespace {

llvm::SmallVector<InterpreterValue> For(MutableArrayRef<InterpreterValue> args,
                                        mlir::Operation* op,
                                        InterpreterState& state) {
  auto lb = std::get<int64_t>(args[0].storage);
  auto ub = std::get<int64_t>(args[1].storage);
  auto step = std::get<int64_t>(args[2].storage);

  auto init_args = args.drop_front(3);
  (void)init_args;
  assert(init_args.empty() && "init args are TODO");

  auto& region = op->getRegion(0);
  for (; lb < ub; lb += step) {
    SmallVector<InterpreterValue> inputs;
    inputs.push_back({lb});
    Interpret(state, region, inputs);
    if (state.HasFailure()) break;
  }
  return {};
}

llvm::SmallVector<InterpreterValue> If(MutableArrayRef<InterpreterValue> args,
                                       mlir::Operation* op,
                                       InterpreterState& state) {
  auto if_op = llvm::cast<scf::IfOp>(op);
  if (std::get<bool>(args[0].storage)) {
    return Interpret(state, if_op.getThenRegion(), {});
  }
  if (!if_op.getElseRegion().hasOneBlock()) {
    return {};
  }
  return Interpret(state, if_op.getElseRegion(), {});
}

REGISTER_MODULE_INITIALIZER(init_scf, {
  mlir::interpreter::detail::RegisterInterpreterOp("scf.for", For);
  mlir::interpreter::detail::RegisterInterpreterOp("scf.if", If);
});

}  // namespace
}  // namespace interpreter
}  // namespace mlir
