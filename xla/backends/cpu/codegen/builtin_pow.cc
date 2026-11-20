/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/builtin_pow.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "absl/base/attributes.h"
#include "absl/base/casts.h"
#include "absl/base/optimization.h"
#include "tsl/platform/denormal.h"

template <typename T>
static T Powi(T a, int32_t b) {
  const bool recip = b < 0;
  T r = 1;
  while (true) {
    if (b & 1) {
      r *= a;
    }
    b /= 2;
    if (b == 0) {
      break;
    }
    a *= a;
  }
  return recip ? 1 / r : r;
}

float ABSL_ATTRIBUTE_WEAK __powisf2(float a, int32_t b) { return Powi(a, b); }

double ABSL_ATTRIBUTE_WEAK __powidf2(double a, int32_t b) { return Powi(a, b); }

namespace xla::cpu {
namespace {

// Tests the bits: under DAZ a float comparison already reads a subnormal as 0.
template <typename Float>
bool IsSubnormal(Float value) {
  static_assert(std::numeric_limits<Float>::is_iec559);
  using Bits = std::conditional_t<sizeof(Float) == 4, uint32_t, uint64_t>;
  constexpr Bits kSignMask = Bits{1} << (sizeof(Bits) * 8 - 1);
  // The smallest normal magnitude: only the lowest exponent bit is set.
  constexpr Bits kMinNormal = Bits{1}
                              << (std::numeric_limits<Float>::digits - 1);
  const Bits magnitude = absl::bit_cast<Bits>(value) & ~kSignMask;
  return magnitude != 0 && magnitude < kMinNormal;
}

// Out of line, so that a call without subnormal operands needs no stack frame.
template <typename Float>
ABSL_ATTRIBUTE_NOINLINE Float PowOfSubnormal(Float x, Float y) {
  // If the CPU reads subnormals as zero (DAZ), pass libm a signed zero.
  if (tsl::port::GetDenormalState().denormals_are_zero()) {
    if (IsSubnormal(x)) {
      x = std::copysign(Float{0}, x);
    }
    if (IsSubnormal(y)) {
      y = std::copysign(Float{0}, y);
    }
  }
  return std::pow(x, y);
}

template <typename Float>
Float PowImpl(Float x, Float y) {
  if (ABSL_PREDICT_FALSE(IsSubnormal(x) || IsSubnormal(y))) {
    return PowOfSubnormal(x, y);
  }
  return std::pow(x, y);
}

}  // namespace

double BuiltinPow(double x, double y) { return PowImpl(x, y); }

float BuiltinPowf(float x, float y) { return PowImpl(x, y); }

}  // namespace xla::cpu
