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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "tsl/platform/denormal.h"

namespace xla::cpu {
namespace {

// Flushes a subnormal input to signed zero, mirroring BuiltinAtan2, so
// expectations can be computed with libm on flushed operands.
double FlushF64Subnormal(double value) {
  constexpr uint64_t kSignMask = 0x8000000000000000;
  constexpr uint64_t kMagnitudeMask = 0x7fffffffffffffff;
  constexpr uint64_t kMinNormalBits = 0x0010000000000000;
  const uint64_t bits = absl::bit_cast<uint64_t>(value);
  const uint64_t magnitude = bits & kMagnitudeMask;
  if (magnitude != 0 && magnitude < kMinNormalBits) {
    return absl::bit_cast<double>(bits & kSignMask);
  }
  return value;
}

// Flushes a subnormal input to signed zero, mirroring BuiltinAtan2f.
float FlushF32Subnormal(float value) {
  constexpr uint32_t kSignMask = 0x80000000;
  constexpr uint32_t kMagnitudeMask = 0x7fffffff;
  constexpr uint32_t kMinNormalBits = 0x00800000;
  const uint32_t bits = absl::bit_cast<uint32_t>(value);
  const uint32_t magnitude = bits & kMagnitudeMask;
  if (magnitude != 0 && magnitude < kMinNormalBits) {
    return absl::bit_cast<float>(bits & kSignMask);
  }
  return value;
}

TEST(BuiltinAtan2Test, FlushesF64SubnormalInputs) {
  const double tiny = std::numeric_limits<double>::denorm_min();
  const double largest_subnormal =
      absl::bit_cast<double>(uint64_t{0x000fffffffffffff});
  const double min_normal = std::numeric_limits<double>::min();
  const double infinity = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> values = {tiny,
                                      -tiny,
                                      largest_subnormal,
                                      -largest_subnormal,
                                      min_normal / 2,
                                      -min_normal / 2,
                                      min_normal,
                                      -min_normal,
                                      1.0,
                                      -1.0,
                                      0.0,
                                      -0.0,
                                      infinity,
                                      -infinity,
                                      nan};
  // Expectations use flushed inputs, so they hold under any FTZ/DAZ state and
  // match what XLA:CPU computes at runtime.
  std::vector<double> expected;
  for (double y : values) {
    for (double x : values) {
      expected.push_back(
          std::atan2(FlushF64Subnormal(y), FlushF64Subnormal(x)));
    }
  }

  const tsl::port::DenormalState states[] = {
      {true, true}, {true, false}, {false, true}, {false, false}};
  for (const tsl::port::DenormalState& state : states) {
    tsl::port::ScopedRestoreFlushDenormalState restore;
    if (!tsl::port::SetDenormalState(state)) continue;
    size_t index = 0;
    for (double y : values) {
      for (double x : values) {
        SCOPED_TRACE(index);
        const double actual = BuiltinAtan2(y, x);
        if (std::isnan(expected[index])) {
          EXPECT_TRUE(std::isnan(actual));
        } else {
          // Bitwise comparison also checks signed zeros.
          EXPECT_EQ(absl::bit_cast<uint64_t>(actual),
                    absl::bit_cast<uint64_t>(expected[index]));
        }
        EXPECT_EQ(tsl::port::GetDenormalState(), state);
        ++index;
      }
    }
  }
}

TEST(BuiltinAtan2Test, FlushesF32SubnormalInputs) {
  const float tiny = std::numeric_limits<float>::denorm_min();
  const float largest_subnormal =
      absl::bit_cast<float>(uint32_t{0x007fffff});
  const float min_normal = std::numeric_limits<float>::min();
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> values = {tiny,
                                     -tiny,
                                     largest_subnormal,
                                     -largest_subnormal,
                                     min_normal / 2,
                                     -min_normal / 2,
                                     min_normal,
                                     -min_normal,
                                     1.0f,
                                     -1.0f,
                                     0.0f,
                                     -0.0f,
                                     infinity,
                                     -infinity,
                                     nan};
  std::vector<float> expected;
  for (float y : values) {
    for (float x : values) {
      expected.push_back(
          std::atan2(FlushF32Subnormal(y), FlushF32Subnormal(x)));
    }
  }

  const tsl::port::DenormalState states[] = {
      {true, true}, {true, false}, {false, true}, {false, false}};
  for (const tsl::port::DenormalState& state : states) {
    tsl::port::ScopedRestoreFlushDenormalState restore;
    if (!tsl::port::SetDenormalState(state)) continue;
    size_t index = 0;
    for (float y : values) {
      for (float x : values) {
        SCOPED_TRACE(index);
        const float actual = BuiltinAtan2f(y, x);
        if (std::isnan(expected[index])) {
          EXPECT_TRUE(std::isnan(actual));
        } else {
          EXPECT_EQ(absl::bit_cast<uint32_t>(actual),
                    absl::bit_cast<uint32_t>(expected[index]));
        }
        EXPECT_EQ(tsl::port::GetDenormalState(), state);
        ++index;
      }
    }
  }
}

TEST(BuiltinAtan2Test, GlibcDenormalMisbehaviors) {
  // Under DAZ, glibc computes atan2(min_normal / 2, -1) as if y were zero
  // but picks the wrong quadrant, returning -pi. The flushed result is +pi.
  const double min_normal = std::numeric_limits<double>::min();
  EXPECT_EQ(absl::bit_cast<uint64_t>(BuiltinAtan2(min_normal / 2, -1.0)),
            absl::bit_cast<uint64_t>(std::atan2(0.0, -1.0)));
  // Under DAZ, glibc returns NaN for atan2f(denorm_min, 1). The flushed
  // result is +0.
  const float tiny_f = std::numeric_limits<float>::denorm_min();
  const float f32_actual = BuiltinAtan2f(tiny_f, 1.0f);
  EXPECT_FALSE(std::isnan(f32_actual));
  EXPECT_EQ(absl::bit_cast<uint32_t>(f32_actual),
            absl::bit_cast<uint32_t>(0.0f));
  // Under DAZ, glibc returns +/-pi/2 for subnormal pairs. The flushed
  // (+0, +0) result is +0.
  const double tiny = std::numeric_limits<double>::denorm_min();
  EXPECT_EQ(absl::bit_cast<uint64_t>(BuiltinAtan2(tiny, tiny)),
            absl::bit_cast<uint64_t>(0.0));
  // Signs of flushed inputs still select the quadrant: (-0, -0) gives -pi.
  EXPECT_EQ(absl::bit_cast<uint64_t>(BuiltinAtan2(-tiny, -tiny)),
            absl::bit_cast<uint64_t>(std::atan2(-0.0, -0.0)));
}

}  // namespace
}  // namespace xla::cpu
