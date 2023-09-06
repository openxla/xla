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

#ifndef XLA_TYPES_H_
#define XLA_TYPES_H_

#include <complex>
#include <istream>
#include <limits>
#include <ostream>
#include <string>
#include <type_traits>

#include "absl/strings/str_format.h"
#include "Eigen/Core"  // from @eigen_archive
#include "include/int4.h"  // from @ml_dtypes

namespace xla {

using ::Eigen::bfloat16;  // NOLINT(misc-unused-using-decls)
using ::Eigen::half;      // NOLINT(misc-unused-using-decls)
using s4 = ml_dtypes::int4;
using u4 = ml_dtypes::uint4;

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

template <class T>
struct is_complex : std::false_type {};
template <class T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

template <typename T>
constexpr bool is_specialized_floating_point_v =
    std::numeric_limits<T>::is_specialized &&
    !std::numeric_limits<T>::is_integer;

// LINT.ThenChange(//tensorflow/compiler/xla/literal.cc)

}  // namespace xla

// Alias namespace ::stream_executor as ::xla::se.
namespace stream_executor {}
namespace xla {
namespace se = ::stream_executor;  // NOLINT(misc-unused-alias-decls)

template <typename T>
struct make_specialized_signed {
  using type = std::make_signed_t<T>;
};

template <>
struct make_specialized_signed<xla::s4> {
  using type = xla::s4;
};

template <>
struct make_specialized_signed<xla::u4> {
  using type = xla::s4;
};

template <typename T>
using make_specialized_signed_t = typename make_specialized_signed<T>::type;

}  // namespace xla

#endif  // XLA_TYPES_H_
