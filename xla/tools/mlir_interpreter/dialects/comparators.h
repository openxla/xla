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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_DIALECTS_COMPARATORS_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_DIALECTS_COMPARATORS_H_

#include <complex>

#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace interpreter {

struct eq {
  template <typename T>
  static bool apply(T lhs, T rhs) {
    return lhs == rhs;
  }
};

struct ne {
  template <typename T>
  static bool apply(T lhs, T rhs) {
    return lhs != rhs;
  }
};

struct lt {
  template <typename T>
  static bool apply(std::complex<T> lhs, std::complex<T> rhs) {
    llvm_unreachable("lt not supported for complex types");
  }
  template <typename T>
  static bool apply(T lhs, T rhs) {
    return lhs < rhs;
  }
};

struct gt {
  template <typename T>
  static bool apply(std::complex<T> lhs, std::complex<T> rhs) {
    llvm_unreachable("gt not supported for complex types");
  }
  template <typename T>
  static bool apply(T lhs, T rhs) {
    return lhs > rhs;
  }
};

struct le {
  template <typename T>
  static bool apply(std::complex<T> lhs, std::complex<T> rhs) {
    llvm_unreachable("le not supported for complex types");
  }
  template <typename T>
  static bool apply(T lhs, T rhs) {
    return lhs <= rhs;
  }
};

struct ge {
  template <typename T>
  static bool apply(std::complex<T> lhs, std::complex<T> rhs) {
    llvm_unreachable("ge not supported for complex types");
  }
  template <typename T>
  static bool apply(T lhs, T rhs) {
    return lhs >= rhs;
  }
};

struct uge {
  static bool apply(int64_t lhs, int64_t rhs) {
    return static_cast<uint64_t>(lhs) >= static_cast<uint64_t>(rhs);
  }
  static bool apply(int32_t lhs, int32_t rhs) {
    return static_cast<uint32_t>(lhs) >= static_cast<uint32_t>(rhs);
  }
  static bool apply(int16_t lhs, int16_t rhs) {
    return static_cast<uint16_t>(lhs) >= static_cast<uint16_t>(rhs);
  }

  template <typename T>
  static bool apply(T lhs, T rhs) {
    llvm_unreachable("uge not supported for this typre");
  }
};

struct ule {
  static bool apply(int64_t lhs, int64_t rhs) {
    return static_cast<uint64_t>(lhs) <= static_cast<uint64_t>(rhs);
  }
  static bool apply(int32_t lhs, int32_t rhs) {
    return static_cast<uint32_t>(lhs) <= static_cast<uint32_t>(rhs);
  }
  static bool apply(int16_t lhs, int16_t rhs) {
    return static_cast<uint16_t>(lhs) <= static_cast<uint16_t>(rhs);
  }

  template <typename T>
  static bool apply(T lhs, T rhs) {
    llvm_unreachable("uge not supported for this typre");
  }
};

struct ult {
  static bool apply(int64_t lhs, int64_t rhs) {
    return static_cast<uint64_t>(lhs) < static_cast<uint64_t>(rhs);
  }
  static bool apply(int32_t lhs, int32_t rhs) {
    return static_cast<uint32_t>(lhs) < static_cast<uint32_t>(rhs);
  }
  static bool apply(int16_t lhs, int16_t rhs) {
    return static_cast<uint16_t>(lhs) < static_cast<uint16_t>(rhs);
  }

  template <typename T>
  static T apply(T lhs, T rhs) {
    llvm_unreachable("uge not supported for this typre");
  }
};

struct ugt {
  static bool apply(int64_t lhs, int64_t rhs) {
    return static_cast<uint64_t>(lhs) > static_cast<uint64_t>(rhs);
  }
  static bool apply(int32_t lhs, int32_t rhs) {
    return static_cast<uint32_t>(lhs) > static_cast<uint32_t>(rhs);
  }
  static bool apply(int16_t lhs, int16_t rhs) {
    return static_cast<uint16_t>(lhs) > static_cast<uint16_t>(rhs);
  }

  template <typename T>
  static T apply(T lhs, T rhs) {
    llvm_unreachable("uge not supported for this typre");
  }
};

struct max {
  template <typename T>
  static T apply(T lhs, T rhs) {
    return std::max(lhs, rhs);
  }

  template <typename T>
  static std::complex<T> apply(std::complex<T> lhs, std::complex<T> rhs) {
    llvm_unreachable("max not supported for complex types");
  }
};

struct min {
  template <typename T>
  static T apply(T lhs, T rhs) {
    return std::min(lhs, rhs);
  }

  template <typename T>
  static std::complex<T> apply(std::complex<T> lhs, std::complex<T> rhs) {
    llvm_unreachable("min not supported for complex types");
  }
};

}  // namespace interpreter
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_DIALECTS_COMPARATORS_H_
