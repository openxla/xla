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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_INTERPRETER_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_INTERPRETER_H_

#include <complex>
#include <functional>
#include <iterator>
#include <memory>
#include <utility>
#include <variant>

#include "base/googleinit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/tools/mlir_interpreter/interpreter_value.h"

#define REGISTER_MLIR_INTERPRETER_OP(dialect, op, fn)                       \
  REGISTER_MODULE_INITIALIZER(init_##dialect##op, {                         \
    mlir::interpreter::detail::RegisterInterpreterOp(#dialect "." #op, fn); \
  })

#define REGISTER_MLIR_INTERPRETER_OP_ALIAS(dialect, op, original_dialect, \
                                           original_op)                   \
  REGISTER_MODULE_INITIALIZER(init_##dialect##op, {                       \
    mlir::interpreter::detail::RegisterInterpreterOpAlias(                \
        #dialect "." #op, #original_dialect "." #original_op);            \
  })

#define REGISTER_MLIR_INTERPRETER_UNARY_CWISE(dialect, op, fn) \
  struct dialect##op##__functor {                              \
    template <typename T>                                      \
    static auto apply(T v) {                                   \
      return fn(v);                                            \
    }                                                          \
  };                                                           \
  REGISTER_MLIR_INTERPRETER_OP(dialect, op,                    \
                               ApplyCWiseMap<dialect##op##__functor>);

#define REGISTER_MLIR_INTERPRETER_BINARY_CWISE(dialect, op, fn) \
  struct dialect##op##__functor {                               \
    template <typename T>                                       \
    static auto apply(T lhs, T rhs) {                           \
      return fn(lhs, rhs);                                      \
    }                                                           \
  };                                                            \
  REGISTER_MLIR_INTERPRETER_OP(dialect, op,                     \
                               ApplyCWiseBinaryMap<dialect##op##__functor>);

namespace mlir {
namespace interpreter {

class InterpreterScope;

class InterpreterState {
 public:
  explicit InterpreterState(const mlir::SymbolTable& symbols)
      : symbols_(symbols) {}

  void AddFailure(llvm::StringRef failure);
  bool HasFailure() const { return failed_; }
  InterpreterScope* GetTopScope() { return top_; }
  const mlir::SymbolTable& GetSymbols() const { return symbols_; }

 private:
  const mlir::SymbolTable& symbols_;
  InterpreterScope* top_ = nullptr;
  bool failed_ = false;

  friend class InterpreterScope;
};

class InterpreterScope {
 public:
  InterpreterScope(InterpreterScope&&) = delete;
  explicit InterpreterScope(InterpreterState& state)
      : state_(state), parent_(state_.top_) {
    state_.top_ = this;
  }
  ~InterpreterScope() { state_.top_ = parent_; }

  void Set(Value v, InterpreterValue iv) { values_[v] = std::move(iv); }

  const InterpreterValue& Get(Value v) {
    if (values_.find(v) == values_.end()) {
      assert(parent_ && "value not found");
      return parent_->Get(v);
    }
    return values_[v];
  }

  // Stores the given iteration index for retrieval by other ops. This is only
  // for supporting linalg.index.
  void SetIterationIndex(int64_t dim, int64_t index) {
    iteration_indices_[dim] = index;
  }

  int64_t GetIterationIndex(int64_t dim) {
    if (iteration_indices_.find(dim) == iteration_indices_.end()) {
      assert(parent_ && "iteration index not found");
      return parent_->GetIterationIndex(dim);
    }
    return iteration_indices_[dim];
  }

 private:
  DenseMap<Value, InterpreterValue> values_;
  DenseMap<int64_t, int64_t> iteration_indices_;

  InterpreterState& state_;
  InterpreterScope* parent_;
};

// Interprets the given region and returns the terminator's arguments. The
// region must have a single block.
SmallVector<InterpreterValue> Interpret(InterpreterState& state, Region& region,
                                        ArrayRef<InterpreterValue> bbargs);

// Interprets the given function.
mlir::FailureOr<SmallVector<InterpreterValue>> RunInterpreter(
    const mlir::SymbolTable& symbols, mlir::func::FuncOp function,
    ArrayRef<InterpreterValue> args);

namespace detail {

// Simple unary ops.
void RegisterInterpreterOp(llvm::StringRef name,
                           InterpreterValue (*fn)(InterpreterValue&));

// Simple binary ops.
void RegisterInterpreterOp(llvm::StringRef name,
                           InterpreterValue (*fn)(InterpreterValue&,
                                                  InterpreterValue&));

// Simple variadic ops (single output).
void RegisterInterpreterOp(
    llvm::StringRef name,
    InterpreterValue (*fn)(MutableArrayRef<InterpreterValue>));

// Simple variadic ops (no output).
void RegisterInterpreterOp(llvm::StringRef name,
                           void (*fn)(MutableArrayRef<InterpreterValue>));

// Generic ops.
void RegisterInterpreterOp(
    llvm::StringRef name,
    std::function<llvm::SmallVector<InterpreterValue>(
        MutableArrayRef<InterpreterValue>, mlir::Operation*, InterpreterState&)>
        fn);

void RegisterInterpreterOpAlias(llvm::StringRef name, llvm::StringRef original);

}  // namespace detail

}  // namespace interpreter
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_INTERPRETER_H_
