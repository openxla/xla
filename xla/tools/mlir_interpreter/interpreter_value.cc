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

#include "xla/tools/mlir_interpreter/interpreter_value.h"

#include <complex>
#include <functional>
#include <string>
#include <string_view>
#include <variant>

namespace mlir {
namespace interpreter {

struct TypeStr {
  static std::string_view get(bool) { return "i1"; }
  static std::string_view get(int64_t) { return "i64"; }
  static std::string_view get(int32_t) { return "i32"; }
  static std::string_view get(int16_t) { return "i16"; }
  static std::string_view get(uint64_t) { return "ui64"; }
  static std::string_view get(float) { return "f32"; }
  static std::string_view get(double) { return "f64"; }
  static std::string_view get(std::complex<float>) { return "complex<f32>"; }
  static std::string_view get(std::complex<double>) { return "complex<f64>"; }
};

struct InterpreterValuePrinter {
  llvm::raw_ostream& os;

  template <typename T>
  void operator()(const TensorOrMemref<T>& t) {
    os << "TensorOrMemref<";
    for (int64_t size : t.view.sizes) {
      os << size << "x";
    }
    os << TypeStr::get(T{}) << ">: ";
    SmallVector<int64_t> indices(t.view.rank());
    std::function<void(int64_t)> print;
    print = [&](int64_t dim) {
      if (dim == indices.size()) {
        PrintScalar(t.buffer->storage[t.view.physical_index(indices)]);
      } else {
        os << "[";
        for (int64_t i = 0; i < t.view.sizes[dim]; ++i) {
          if (i > 0) os << ", ";
          indices[dim] = i;
          print(dim + 1);
        }
        os << "]";
      }
    };
    print(0);
  }

  void operator()(const Tuple& t) {
    os << "(";
    bool first = true;
    for (const auto& v : t.values) {
      if (!first) os << ", ";
      first = false;
      v->Print(os);
    }
    os << ")";
  }

  template <typename T>
  void operator()(const T& t) {
    os << TypeStr::get(t) << ": ";
    PrintScalar(t);
  }

  template <typename T>
  void PrintScalar(const T& v) {
    os << v;
  }

  template <typename T>
  void PrintScalar(const std::complex<T>& v) {
    os << v.real() << (v.imag() >= 0 ? "+" : "") << v.imag() << "i";
  }

  void PrintScalar(bool v) { os << "i1: " << (v ? "true" : "false"); }
};

void InterpreterValue::Print(llvm::raw_ostream& os) const {
  std::visit(InterpreterValuePrinter{os}, storage);
}

std::string InterpreterValue::ToString() const {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  Print(os);
  return buf;
}

struct ExtractElementVisitor {
  llvm::ArrayRef<int64_t> indices;

  template <typename T>
  InterpreterValue operator()(const TensorOrMemref<T>& t) {
    return {t.at(indices)};
  }

  InterpreterValue operator()(const Tuple& t) {
    llvm_unreachable("extracting elements from Tuples is unsupported");
  }

  template <typename T>
  InterpreterValue operator()(const T& t) {
    return {t};
  }
};

InterpreterValue InterpreterValue::ExtractElement(
    llvm::ArrayRef<int64_t> indices) const {
  return std::visit(ExtractElementVisitor{indices}, storage);
}

struct InsertElementVisitor {
  llvm::ArrayRef<int64_t> indices;
  const InterpreterValue& value;

  template <typename T>
  void operator()(TensorOrMemref<T>& t) {
    assert(std::holds_alternative<T>(value.storage) && "mismatched value");
    t.at(indices) = std::get<T>(value.storage);
  }

  void operator()(const Tuple& t) {
    llvm_unreachable("inserting elements into Tuples is unsupported");
  }

  template <typename T>
  void operator()(T& t) {
    assert(std::holds_alternative<T>(value.storage) && "mismatched value");
    t = std::get<T>(value.storage);
  }
};

void InterpreterValue::InsertElement(llvm::ArrayRef<int64_t> indices,
                                     const InterpreterValue& value) {
  std::visit(InsertElementVisitor{indices, value}, storage);
}

struct FillVisitor {
  const std::function<InterpreterValue(llvm::ArrayRef<int64_t> indices)>& f;

  template <typename T>
  void operator()(TensorOrMemref<T>& t) {
    for (const auto& indices : t.view.indices()) {
      auto value = f(indices);
      assert(std::holds_alternative<T>(value.storage) && "mismatched value");
      t.at(indices) = std::get<T>(value.storage);
    }
  }

  void operator()(const Tuple& t) {
    llvm_unreachable("filling Tuples is unsupported");
  }

  template <typename T>
  void operator()(T& t) {
    InterpreterValue value = f({});
    assert(std::holds_alternative<T>(value.storage) && "mismatched value");
    t = std::get<T>(value.storage);
  }
};

void InterpreterValue::Fill(
    const std::function<InterpreterValue(llvm::ArrayRef<int64_t> indices)>& f) {
  std::visit(FillVisitor{f}, storage);
}

struct InterpreterValueCloneVisitor {
  template <typename T>
  InterpreterValue operator()(const TensorOrMemref<T>& t) {
    return {t.Clone()};
  }

  InterpreterValue operator()(const Tuple& t) {
    llvm_unreachable("cloning tuples is unsupported");
  }

  template <typename T>
  InterpreterValue operator()(const T& t) {
    return {t};
  }
};

InterpreterValue InterpreterValue::Clone() const {
  return std::visit(InterpreterValueCloneVisitor{}, storage);
}

struct InterpreterValueTypedAlikeVisitor {
  llvm::ArrayRef<int64_t> shape;

  template <typename T>
  InterpreterValue operator()(const TensorOrMemref<T>& t) {
    return {TensorOrMemref<T>::Empty(shape)};
  }

  InterpreterValue operator()(const Tuple& t) {
    llvm_unreachable("TypedAlike for tuples is unsupported");
  }

  template <typename T>
  InterpreterValue operator()(const T& t) {
    return {TensorOrMemref<T>::Empty(shape)};
  }
};

InterpreterValue InterpreterValue::TypedAlike(
    llvm::ArrayRef<int64_t> shape) const {
  return std::visit(InterpreterValueTypedAlikeVisitor{shape}, storage);
}

InterpreterValue InterpreterValue::MakeTensor(mlir::Type element_type,
                                              llvm::ArrayRef<int64_t> shape) {
  return DispatchScalarType(element_type, [&](auto dummy) {
    return InterpreterValue{TensorOrMemref<decltype(dummy)>::Empty(shape)};
  });
}

struct InterpreterValueGetViewVisitor {
  template <typename T>
  BufferView& operator()(TensorOrMemref<T>& t) {
    return t.view;
  }

  template <typename T>
  BufferView& operator()(const T& t) {
    llvm_unreachable("view is only supported for tensors");
  }
};

BufferView& InterpreterValue::view() {
  return std::visit(InterpreterValueGetViewVisitor{}, storage);
}

struct IsTensorVisitor {
  template <typename T>
  bool operator()(const TensorOrMemref<T>& t) {
    return true;
  }

  template <typename T>
  bool operator()(const T& t) {
    return false;
  }
};

bool InterpreterValue::IsTensor() const {
  return std::visit(IsTensorVisitor{}, storage);
}

InterpreterValue InterpreterValue::AsUnitTensor() const {
  auto result = TypedAlike({});
  result.InsertElement({}, *this);
  return result;
}

bool Tuple::operator==(const Tuple& other) const {
  if (other.values.size() != values.size()) return false;
  for (const auto& [lhs, rhs] : llvm::zip(values, other.values)) {
    if (!(*lhs == *rhs)) return false;
  }
  return true;
}

}  // namespace interpreter
}  // namespace mlir
