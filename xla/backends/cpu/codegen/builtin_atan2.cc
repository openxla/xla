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

#include "absl/base/attributes.h"
#include "absl/base/casts.h"
#include "tsl/platform/denormal.h"

namespace xla::cpu {
namespace {

// Keep the state-changing call out of the normal-input fast path.
ABSL_ATTRIBUTE_NOINLINE double Atan2WithSubnormalInputs(double y, double x) {
  // XLA's compute threads flush denormals by default. libm needs the original
  // operands to distinguish their ratio and quadrant. Restore the caller's
  // state after this call without changing the execution mode of other ops.
  tsl::port::ScopedDontFlushDenormal dont_flush;
  // Prevent the compiler from treating atan2 as a pure builtin and moving
  // the call past the restoration of the floating-point environment.
  double (*volatile atan2_fn)(double, double) = std::atan2;
  return atan2_fn(y, x);
}

}  // namespace

double BuiltinAtan2(double y, double x) {
  // Floating-point comparisons can already treat subnormal operands as zero
  // under DAZ, so inspect their bits instead. Exclude signed zeros, infinities,
  // and NaNs from the slow path.
  constexpr uint64_t kMagnitudeMask = 0x7fffffffffffffff;
  constexpr uint64_t kMinNormalBits = 0x0010000000000000;
  const uint64_t abs_y = absl::bit_cast<uint64_t>(y) & kMagnitudeMask;
  const uint64_t abs_x = absl::bit_cast<uint64_t>(x) & kMagnitudeMask;
  if ((abs_y != 0 && abs_y < kMinNormalBits) ||
      (abs_x != 0 && abs_x < kMinNormalBits)) {
    return Atan2WithSubnormalInputs(y, x);
  }
  return std::atan2(y, x);
}

}  // namespace xla::cpu
