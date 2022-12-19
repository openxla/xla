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

#include "xla/tools/mlir_interpreter/interpreter.h"

#include <functional>
#include <string>
#include <utility>
#include <variant>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tsl/platform/logging.h"

namespace mlir {
namespace interpreter {
namespace {

auto& op_aliases = *new DenseMap<llvm::StringRef, llvm::StringRef>();
auto& interpreter_functions =
    *new DenseMap<llvm::StringRef, std::function<SmallVector<InterpreterValue>(
                                       MutableArrayRef<InterpreterValue>,
                                       mlir::Operation*, InterpreterState&)>>();

}  // namespace

SmallVector<InterpreterValue> Interpret(InterpreterState& state,
                                        Operation& op) {
  static int log_depth = 0;
  auto fn = interpreter_functions.find(op.getName().getStringRef());
  if (fn == interpreter_functions.end()) {
    auto alias = op_aliases.find(op.getName().getStringRef());
    if (alias != op_aliases.end()) {
      fn = interpreter_functions.find(alias->second);
    }
  }
  if (fn == interpreter_functions.end()) {
    llvm::errs() << "unsupported op: " << op.getName().getStringRef() << "\n";
    state.AddFailure("unsupported op");
    return {};
  }
  SmallVector<InterpreterValue> operands;
  for (auto operand : op.getOperands()) {
    operands.push_back(state.GetTopScope()->Get(operand));
  }
  auto make_call_log = [&]() {
    std::string log_line(log_depth, ' ');
    absl::StrAppend(&log_line, op.getName().getStringRef().str(), "(");
    bool first = true;
    for (auto& operand : operands) {
      if (!first) absl::StrAppend(&log_line, ", ");
      first = false;
      absl::StrAppend(&log_line, operand.ToString());
    }
    absl::StrAppend(&log_line, ") ");
    return log_line;
  };
  std::string call_log = VLOG_IS_ON(6) ? make_call_log() : "";
  if (VLOG_IS_ON(6) && !op.getRegions().empty()) {
    llvm::errs() << call_log << "\n";
  }
  log_depth += 2;
  auto results = fn->second(operands, &op, state);
  log_depth -= 2;
  if (VLOG_IS_ON(6)) {
    std::string log_line;
    if (op.getRegions().empty())
      log_line = call_log;
    else
      log_line = std::string(log_depth + 2, ' ');

    if (!results.empty()) {
      absl::StrAppend(&log_line, "-> ");
      bool first = true;
      for (const auto& result : results) {
        if (!first) absl::StrAppend(&log_line, ", ");
        first = false;
        absl::StrAppend(&log_line, result.ToString());
      }
    }
    if (log_line.size() > log_depth + 2) {
      llvm::errs() << log_line << "\n";
    }
  }
  return results;
}

SmallVector<InterpreterValue> Interpret(InterpreterState& state, Region& region,
                                        ArrayRef<InterpreterValue> bbargs) {
  assert(region.hasOneBlock() && "expected region to have one block");
  InterpreterScope scope(state);

  auto& block = region.getBlocks().front();
  for (auto [value, interpreter_value] :
       llvm::zip(block.getArguments(), bbargs)) {
    scope.Set(value, interpreter_value);
  }

  for (mlir::Operation& op : block.without_terminator()) {
    auto results = Interpret(state, op);
    if (state.HasFailure()) return {};
    if (results.size() != op.getNumResults()) {
      llvm::errs() << "Unexpected number of results while interpreting "
                   << op.getName().getStringRef() << ". Interpreter bug?\n";
      llvm_unreachable("uenxpected number of results");
    }
    for (auto [v, iv] : llvm::zip(op.getResults(), results)) {
      scope.Set(v, iv);
    }
  }

  SmallVector<InterpreterValue> result;
  for (auto v : block.getTerminator()->getOperands()) {
    result.push_back(scope.Get(v));
  }
  return result;
}

void InterpreterState::AddFailure(llvm::StringRef failure) {
  failed_ = true;
  LOG(ERROR) << "Interpreter failure: " << failure.str();
}

mlir::FailureOr<SmallVector<InterpreterValue>> RunInterpreter(
    const mlir::SymbolTable& symbols, mlir::func::FuncOp function,
    ArrayRef<InterpreterValue> args) {
  InterpreterState state{symbols};
  auto results = Interpret(state, function.getBody(), args);
  if (state.HasFailure()) {
    return failure();
  }
  return results;
}

namespace detail {

void RegisterInterpreterOp(llvm::StringRef name,
                           InterpreterValue (*fn)(InterpreterValue&)) {
  RegisterInterpreterOp(
      name,
      [fn](MutableArrayRef<InterpreterValue> operands, mlir::Operation*,
           InterpreterState&) -> SmallVector<InterpreterValue> {
        assert(operands.size() == 1 && "unexpected number of operands");
        return {fn(operands[0])};
      });
}

void RegisterInterpreterOp(llvm::StringRef name,
                           InterpreterValue (*fn)(InterpreterValue&,
                                                  InterpreterValue&)) {
  RegisterInterpreterOp(
      name,
      [fn](MutableArrayRef<InterpreterValue> operands, mlir::Operation*,
           InterpreterState&) -> SmallVector<InterpreterValue> {
        assert(operands.size() == 2 && "unexpected number of operands");
        return {fn(operands[0], operands[1])};
      });
}

void RegisterInterpreterOp(
    llvm::StringRef name,
    InterpreterValue (*fn)(MutableArrayRef<InterpreterValue>)) {
  RegisterInterpreterOp(
      name,
      [fn](MutableArrayRef<InterpreterValue> operands, mlir::Operation*,
           InterpreterState&) -> SmallVector<InterpreterValue> {
        return {fn(operands)};
      });
}

void RegisterInterpreterOp(llvm::StringRef name,
                           void (*fn)(MutableArrayRef<InterpreterValue>)) {
  RegisterInterpreterOp(
      name,
      [fn](MutableArrayRef<InterpreterValue> operands, mlir::Operation*,
           InterpreterState&) -> SmallVector<InterpreterValue> {
        fn(operands);
        return {};
      });
}

void RegisterInterpreterOp(
    llvm::StringRef name,
    std::function<llvm::SmallVector<InterpreterValue>(
        MutableArrayRef<InterpreterValue>, mlir::Operation*, InterpreterState&)>
        fn) {
  VLOG(5) << "Registering interpreter op " << interpreter_functions.size()
          << ": " << name.str() << "\n";
  interpreter_functions[name] = fn;
}

void RegisterInterpreterOpAlias(llvm::StringRef name,
                                llvm::StringRef original) {
  op_aliases[name] = original;
}

}  // namespace detail
}  // namespace interpreter
}  // namespace mlir
