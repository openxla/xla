/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/builtin_atan2.h"

#include <cmath>
#include <cstdint>

#include "absl/base/casts.h"

namespace xla::cpu {

double BuiltinAtan2(double y, double x) {
  // Floating-point comparisons can already treat subnormal operands as zero
  // under DAZ, so inspect their bits instead. Exclude signed zeros,
  // infinities, and NaNs, which must pass through unchanged. Flushed inputs
  // keep their sign bit so the quadrant of the result is still correct.
  constexpr uint64_t kSignMask = 0x8000000000000000;
  constexpr uint64_t kMagnitudeMask = 0x7fffffffffffffff;
  constexpr uint64_t kMinNormalBits = 0x0010000000000000;
  const uint64_t y_bits = absl::bit_cast<uint64_t>(y);
  const uint64_t x_bits = absl::bit_cast<uint64_t>(x);
  const uint64_t abs_y = y_bits & kMagnitudeMask;
  const uint64_t abs_x = x_bits & kMagnitudeMask;
  if (abs_y != 0 && abs_y < kMinNormalBits) {
    y = absl::bit_cast<double>(y_bits & kSignMask);
  }
  if (abs_x != 0 && abs_x < kMinNormalBits) {
    x = absl::bit_cast<double>(x_bits & kSignMask);
  }
  return std::atan2(y, x);
}

float BuiltinAtan2f(float y, float x) {
  // Same as above for f32 inputs. glibc's atan2f can return NaN for subnormal
  // inputs (even atan2f(denorm_min, 1)), so flush them explicitly as well.
  constexpr uint32_t kSignMask = 0x80000000;
  constexpr uint32_t kMagnitudeMask = 0x7fffffff;
  constexpr uint32_t kMinNormalBits = 0x00800000;
  const uint32_t y_bits = absl::bit_cast<uint32_t>(y);
  const uint32_t x_bits = absl::bit_cast<uint32_t>(x);
  const uint32_t abs_y = y_bits & kMagnitudeMask;
  const uint32_t abs_x = x_bits & kMagnitudeMask;
  if (abs_y != 0 && abs_y < kMinNormalBits) {
    y = absl::bit_cast<float>(y_bits & kSignMask);
  }
  if (abs_x != 0 && abs_x < kMinNormalBits) {
    x = absl::bit_cast<float>(x_bits & kSignMask);
  }
  return ::atan2f(y, x);
}

}  // namespace xla::cpu
