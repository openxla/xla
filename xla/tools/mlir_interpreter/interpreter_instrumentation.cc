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

#include "xla/tools/mlir_interpreter/interpreter_instrumentation.h"

#include <random>
#include <string>
#include <utility>

#include "absl/random/random.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "xla/tools/mlir_interpreter/interpreter.h"
#include "xla/tools/mlir_interpreter/interpreter_value_util.h"
#include "tsl/platform/logging.h"

namespace mlir {
namespace interpreter {

void MlirInterpreterInstrumentation::runAfterPass(Pass* pass, Operation* op) {
  ModuleOp module = llvm::dyn_cast<ModuleOp>(op);
  if (!module) {
    module = op->getParentOfType<mlir::ModuleOp>();
  }
  if (!module) {
    LOG(ERROR) << "Failed to find a ModuleOp after " << pass->getName().str()
               << ".";
    return;
  }

  SymbolTable symbols(module);
  auto main = symbols.lookup("main");
  if (!main) {
    LOG(ERROR) << "Failed to find main function after " << pass->getName().str()
               << ".";
    return;
  }

  auto main_function = llvm::dyn_cast<func::FuncOp>(main);
  if (!main_function) {
    LOG(ERROR) << "Main is not a func::FuncOp after " << pass->getName().str()
               << ".";
    return;
  }

  llvm::SmallVector<mlir::interpreter::InterpreterValue> args;
  std::seed_seq my_seed_seq({0});
  absl::BitGen bitgen(my_seed_seq);

  LOG(INFO) << "Running interpreter after " << pass->getName().str() << ".";
  module.dump();

  for (auto arg : main_function.getBody().getBlocks().front().getArguments()) {
    auto arg_or = mlir::interpreter::MakeRandomInput(bitgen, arg.getType());
    if (!succeeded(arg_or)) {
      LOG(ERROR) << "failed to convert argument";
      return;
    }
    args.push_back(*arg_or);
  }

  LOG(INFO) << "Inputs:";
  for (const auto& arg : args) {
    LOG(INFO) << arg.ToString();
  }
  auto maybe_result = RunInterpreter(symbols, main_function, args);
  if (!succeeded(maybe_result)) {
    LOG(ERROR) << "Interpreter failed after " << pass->getName().str() << ".";
    return;
  }

  LOG(INFO) << "Results:";
  for (const auto& result : *maybe_result) {
    LOG(INFO) << result.ToString();
  }

  if (reference_results_.empty()) {
    reference_results_ = std::move(*maybe_result);
  } else {
    if (maybe_result->empty()) {
      // BufferResultsToOutParams already ran.
      ArrayRef<InterpreterValue> results = args;
      results = results.take_back(reference_results_.size());
      for (const auto& result : results) {
        LOG(INFO) << result.ToString();
      }

      if (!llvm::equal(results, reference_results_)) {
        LOG(ERROR) << "Results changed";
      }
    } else {
      if (*maybe_result != reference_results_) {
        LOG(ERROR) << "Results changed";
      }
    }
  }
}

}  // namespace interpreter
}  // namespace mlir
