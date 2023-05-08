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

#ifndef TENSORFLOW_TSL_PYTHON_LIB_CORE_FLOAT8_TYPEDESCRIPTOR_H_
#define TENSORFLOW_TSL_PYTHON_LIB_CORE_FLOAT8_TYPEDESCRIPTOR_H_

#include "tsl/platform/float8.h"
#include "tsl/python/lib/core/custom_float.h"
#include "tsl/python/lib/core/float8.h"

namespace tsl {
namespace custom_float_internal {

template <>
struct TypeDescriptor<float8_e4m3fn>
    : custom_float_internal::CustomFloatTypeDescriptor<float8_e4m3fn> {
  typedef float8_e4m3fn T;
  static constexpr const char* kTypeName = "float8_e4m3fn";
  static constexpr const char* kTpDoc = "float8_e4m3fn floating-point values";
  // We must register float8_e4m3fn with a unique kind, because numpy
  // considers two types with the same kind and size to be equal.
  // The downside of this is that NumPy scalar promotion does not work with
  // float8 values.  Using 'V' to mirror bfloat16 vs float16.
  static constexpr char kNpyDescrKind = 'V';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = '4';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e4m3b11>
    : custom_float_internal::CustomFloatTypeDescriptor<float8_e4m3b11> {
  typedef float8_e4m3b11 T;
  static constexpr const char* kTypeName = "float8_e4m3b11";
  static constexpr const char* kTpDoc = "float8_e4m3b11 floating-point values";
  // We must register float8_e4m3b11 with a unique kind, because numpy
  // considers two types with the same kind and size to be equal.
  // The downside of this is that NumPy scalar promotion does not work with
  // float8 values.
  static constexpr char kNpyDescrKind = 'V';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'L';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e5m2>
    : custom_float_internal::CustomFloatTypeDescriptor<float8_e5m2> {
  typedef float8_e5m2 T;
  static constexpr const char* kTypeName = "float8_e5m2";
  static constexpr const char* kTpDoc = "float8_e5m2 floating-point values";
  // Treating e5m2 as the natural "float" type since it is IEEE-754 compliant.
  static constexpr char kNpyDescrKind = 'f';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = '5';
  static constexpr char kNpyDescrByteorder = '=';
};

}  // namespace custom_float_internal
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PYTHON_LIB_CORE_FLOAT8_TYPEDESCRIPTOR_H_
