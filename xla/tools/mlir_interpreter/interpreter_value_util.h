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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_INTERPRETER_VALUE_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_INTERPRETER_VALUE_UTIL_H_

#include <utility>

#include "absl/random/bit_gen_ref.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "xla/tools/mlir_interpreter/interpreter_value.h"

namespace mlir {
namespace interpreter {
namespace detail {

template <typename Fn>
struct InterpreterValueMapVisitor {
  template <typename T>
  InterpreterValue operator()(const TensorOrMemref<T>& t) {
    using out_elem_t = decltype(Fn::apply(T()));
    auto out = TensorOrMemref<out_elem_t>::EmptyLike(t.view);
    for (auto [src_index, dst_index] :
         llvm::zip(t.view.physical_indices(), out.view.physical_indices())) {
      out.buffer->storage[dst_index] = Fn::apply(t.buffer->storage[src_index]);
    }
    return {out};
  }

  InterpreterValue operator()(const Tuple& t) {
    Tuple out;
    for (const auto& value : t.values) {
      out.values.push_back(std::make_unique<InterpreterValue>(
          std::move(std::visit(*this, value->storage))));
    }
    return {out};
  }

  template <typename T>
  InterpreterValue operator()(const T& t) {
    return {Fn::apply(t)};
  }
};

template <typename Fn>
struct InterpreterValueBiMapVisitor {
  const InterpreterValue& rhs;

  template <typename T>
  InterpreterValue operator()(const TensorOrMemref<T>& lhs_t) {
    using out_elem_t = decltype(Fn::apply(T(), T()));
    auto out = TensorOrMemref<out_elem_t>::EmptyLike(lhs_t.view);
    const auto& rhs_t = std::get<TensorOrMemref<T>>(rhs.storage);
    for (auto [lhs_index, rhs_index, dst_index] :
         llvm::zip(lhs_t.view.physical_indices(), rhs_t.view.physical_indices(),
                   out.view.physical_indices())) {
      out.buffer->storage[dst_index] = Fn::apply(
          lhs_t.buffer->storage[lhs_index], rhs_t.buffer->storage[rhs_index]);
    }
    return {out};
  }

  InterpreterValue operator()(const Tuple& lhs_t) {
    const auto& rhs_t = std::get<Tuple>(rhs.storage);
    Tuple out;
    for (const auto& [lhs_v, rhs_v] : llvm::zip(lhs_t.values, rhs_t.values)) {
      out.values.push_back(std::make_unique<InterpreterValue>(std::move(
          std::visit(InterpreterValueBiMapVisitor{*rhs_v}, lhs_v->storage))));
    }
    return {std::move(out)};
  }

  template <typename T>
  InterpreterValue operator()(const T& t) {
    return {Fn::apply(t, std::get<T>(rhs.storage))};
  }
};

}  // namespace detail

template <typename Fn>
InterpreterValue ApplyCWiseMap(InterpreterValue& value) {
  return std::visit(detail::InterpreterValueMapVisitor<Fn>{}, value.storage);
}

template <typename Fn>
InterpreterValue ApplyCWiseBinaryMap(InterpreterValue& lhs,
                                     InterpreterValue& rhs) {
  assert(lhs.storage.index() == rhs.storage.index());
  return std::visit(detail::InterpreterValueBiMapVisitor<Fn>{rhs}, lhs.storage);
}

template <typename T>
SmallVector<T> UnpackInterpreterValues(ArrayRef<InterpreterValue> values) {
  SmallVector<T> result;
  for (const auto& value : values) {
    result.push_back(std::get<T>(value.storage));
  }
  return result;
}

template <typename T>
SmallVector<InterpreterValue> PackInterpreterValues(ArrayRef<T> values) {
  SmallVector<InterpreterValue> result;
  for (const auto& value : values) {
    result.push_back({value});
  }
  return result;
}

mlir::FailureOr<mlir::interpreter::InterpreterValue> MakeRandomInput(
    absl::BitGenRef bitgen, mlir::Type type);

}  // namespace interpreter
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_INTERPRETER_VALUE_UTIL_H_
