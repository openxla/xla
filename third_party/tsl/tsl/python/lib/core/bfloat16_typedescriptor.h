/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PYTHON_LIB_CORE_BFLOAT16_TYPEDESCRIPTOR_H_
#define TENSORFLOW_TSL_PYTHON_LIB_CORE_BFLOAT16_TYPEDESCRIPTOR_H_

#include "tsl/platform/bfloat16.h"
#include "tsl/python/lib/core/bfloat16.h"
#include "tsl/python/lib/core/custom_float.h"

namespace tsl {
namespace custom_float_internal {

template <>
struct TypeDescriptor<bfloat16>
    : custom_float_internal::CustomFloatTypeDescriptor<bfloat16> {
  typedef bfloat16 T;
  static constexpr const char* kTypeName = "bfloat16";
  static constexpr const char* kTpDoc = "bfloat16 floating-point values";
  // We must register bfloat16 with a kind other than "f", because numpy
  // considers two types with the same kind and size to be equal, but
  // float16 != bfloat16.
  // The downside of this is that NumPy scalar promotion does not work with
  // bfloat16 values.
  static constexpr char kNpyDescrKind = 'V';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'E';
  static constexpr char kNpyDescrByteorder = '=';
};

}  // namespace custom_float_internal
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PYTHON_LIB_CORE_BFLOAT16_TYPEDESCRIPTOR_H_
