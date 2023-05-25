/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
// Must be included first
// clang-format off
#include "tsl/python/lib/core/numpy.h" //NOLINT
// clang-format on

#include "tsl/python/lib/core/bfloat16.h"

#include <array>   // NOLINT
#include <cmath>   // NOLINT
#include <limits>  // NOLINT
#include <locale>  // NOLINT

// Place `<locale>` before <Python.h> to avoid a build failure in macOS.
#include <Python.h>

#include "Eigen/Core"  // from @eigen_archive
#include "tsl/platform/types.h"
#include "tsl/python/lib/core/custom_float.h"

namespace tsl {
namespace custom_float_internal {

template <>
struct TypeDescriptor<bfloat16>
    : custom_float_internal::CustomFloatTypeDescriptor<bfloat16> {
  using T = bfloat16;
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

namespace {

// Initializes the module.
bool Initialize() {
  tsl::ImportNumpy();
  import_umath1(false);

  custom_float_internal::Safe_PyObjectPtr numpy_str =
      custom_float_internal::make_safe(PyUnicode_FromString("numpy"));
  if (!numpy_str) {
    return false;
  }
  custom_float_internal::Safe_PyObjectPtr numpy =
      custom_float_internal::make_safe(PyImport_Import(numpy_str.get()));
  if (!numpy) {
    return false;
  }

  if (!custom_float_internal::RegisterNumpyDtype<bfloat16>(numpy.get())) {
    return false;
  }

  return true;
}

}  // namespace

bool RegisterNumpyBfloat16() {
  if (custom_float_internal::TypeDescriptor<bfloat16>::Dtype() != NPY_NOTYPE) {
    // Already initialized.
    return true;
  }
  if (!Initialize()) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "cannot load bfloat16 module.");
    }
    PyErr_Print();
    return false;
  }
  return true;
}

PyObject* Bfloat16Dtype() {
  return reinterpret_cast<PyObject*>(
      custom_float_internal::TypeDescriptor<bfloat16>::type_ptr);
}

int Bfloat16NumpyType() {
  return custom_float_internal::TypeDescriptor<bfloat16>::Dtype();
}

}  // namespace tsl
