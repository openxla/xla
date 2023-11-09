/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_LIB_RANDOM_RANDOM_DISTRIBUTIONS_UTILS_H_
#define TENSORFLOW_TSL_LIB_RANDOM_RANDOM_DISTRIBUTIONS_UTILS_H_

#include <string.h>

#include <cstdint>
#include <limits>
#include <utility>

#include "Eigen/Core"  // from @eigen_archive
#include "tsl/lib/random/philox_random.h"

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

namespace tsl {
namespace random {

// Helper function to convert an unsigned integer to a float between [0..1).
template <typename FloatOut, typename UintIn>
PHILOX_DEVICE_INLINE FloatOut UintToFloat(UintIn x) {
  static_assert(std::numeric_limits<UintIn>::is_specialized);
  static_assert(std::numeric_limits<UintIn>::is_integer);
  static_assert(!std::numeric_limits<UintIn>::is_signed);
  static_assert(std::numeric_limits<FloatOut>::is_specialized);
  static_assert(!std::numeric_limits<FloatOut>::is_integer);
  static_assert(sizeof(UintIn) >= sizeof(FloatOut));
  constexpr int kBias = (1 - std::numeric_limits<FloatOut>::min_exponent) + 1;
  constexpr int kTrailingSignificandFieldWidth =
      std::numeric_limits<FloatOut>::digits - 1;
  // IEEE754 floats are formatted as follows (MSB first):
  //    sign exponent mantissa
  // Conceptually construct the following:
  //    sign == 0
  //    exponent == an excess representation of a zero exponent
  //    mantissa == random bits
  const UintIn man = x & ((UintIn(1) << kTrailingSignificandFieldWidth) - 1);
  const UintIn exp = static_cast<UintIn>(kBias);
  const UintIn val = (exp << kTrailingSignificandFieldWidth) | man;

  // Assumes that endian-ness is same for float and UintIn.
  FloatOut result = Eigen::numext::bit_cast<FloatOut>(
      static_cast<typename Eigen::numext::get_integer_by_size<sizeof(
          FloatOut)>::unsigned_type>(val));
  return result - static_cast<FloatOut>(1.0);
}

// Helper function to convert two 32-bit integers to a double between [0..1).
PHILOX_DEVICE_INLINE double Uint64ToDouble(uint32_t x0, uint32_t x1) {
  return UintToFloat<double>((static_cast<uint64_t>(x0) << 32) | x1);
}

// Helper function to convert two uniform integers to two floats
// under the unit normal distribution.
template <typename FloatOut, typename UintIn>
PHILOX_DEVICE_INLINE std::pair<FloatOut, FloatOut> BoxMullerFloat(UintIn x0,
                                                                  UintIn x1) {
  // This function implements the Box-Muller transform:
  // http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Basic_form
  // Do not send a really small number to log().
  // We cannot mark "epsilon" as "static const" because NVCC would complain
  const FloatOut epsilon = 1.0e-7;
  FloatOut u1 = UintToFloat<FloatOut>(x0);
  if (u1 < epsilon) {
    u1 = epsilon;
  }
  const FloatOut v1 = 2.0 * M_PI * UintToFloat<FloatOut>(x1);
  const FloatOut u2 = sqrt(-2.0 * log(u1));
  FloatOut f0, f1;
#if !defined(__linux__)
  f0 = Eigen::numext::sin(v1);
  f1 = Eigen::numext::cos(v1);
#else
  if constexpr (std::is_same_v<FloatOut, double>) {
    sincos(v1, &f0, &f1);
  } else {
    sincosf(v1, &f0, &f1);
  }
#endif
  f0 *= u2;
  f1 *= u2;
  return std::make_pair(f0, f1);
}

}  // namespace random
}  // namespace tsl

#endif  // TENSORFLOW_TSL_LIB_RANDOM_RANDOM_DISTRIBUTIONS_UTILS_H_
