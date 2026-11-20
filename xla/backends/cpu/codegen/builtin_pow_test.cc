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

#include "xla/backends/cpu/codegen/builtin_pow.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <ios>
#include <limits>
#include <vector>

#include "absl/base/casts.h"
#include "tsl/platform/denormal.h"

namespace xla::cpu {
namespace {

using tsl::port::DenormalState;

uint32_t Bits(float value) { return absl::bit_cast<uint32_t>(value); }
uint64_t Bits(double value) { return absl::bit_cast<uint64_t>(value); }

// Reads the bits: under DAZ a float comparison cannot tell a subnormal from 0.
bool IsSubnormal(float value) {
  const uint32_t magnitude = Bits(value) & 0x7fffffffu;
  return magnitude != 0 && magnitude < 0x00800000u;
}
bool IsSubnormal(double value) {
  const uint64_t magnitude = Bits(value) & 0x7fffffffffffffffull;
  return magnitude != 0 && magnitude < 0x0010000000000000ull;
}

// Calls `fn` in every combination of flush-to-zero and denormals-are-zero the
// platform supports, and checks that `fn` leaves the mode unchanged.
template <typename Fn>
void ForEachDenormalState(Fn fn) {
  const DenormalState states[] = {
      DenormalState(false, false), DenormalState(true, false),
      DenormalState(false, true), DenormalState(true, true)};
  tsl::port::ScopedRestoreFlushDenormalState restore;
  for (const DenormalState& state : states) {
    // A platform without control over the mode still runs its ambient state.
    if (!tsl::port::SetDenormalState(state) &&
        tsl::port::GetDenormalState() != state) {
      continue;
    }
    fn(state);
    EXPECT_TRUE(tsl::port::GetDenormalState() == state);
  }
}

template <typename T>
struct PowCase {
  T x;
  T y;
  T expected;
};

template <typename T, typename PowFn>
void ExpectFlushedUnderDaz(PowFn pow_fn, const std::vector<PowCase<T>>& cases) {
  ForEachDenormalState([&](const DenormalState& state) {
    if (!state.denormals_are_zero()) {
      return;
    }
    for (const PowCase<T>& c : cases) {
      // Volatile operands keep the compiler from folding the call.
      volatile T x = c.x;
      volatile T y = c.y;
      const T actual = pow_fn(x, y);
      EXPECT_EQ(Bits(actual), Bits(c.expected))
          << std::hex << "x 0x" << Bits(c.x) << ", y 0x" << Bits(c.y);
    }
  });
}

TEST(BuiltinPowTest, F32SubnormalOperandsAreFlushedUnderDaz) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kSub = std::numeric_limits<float>::min() / 2;
  constexpr float kTiny = std::numeric_limits<float>::denorm_min();
  const float max_sub = absl::bit_cast<float>(uint32_t{0x007fffff});
  ExpectFlushedUnderDaz<float>(
      BuiltinPowf, {
                       // glibc returns 2^-75 here.
                       {kSub, 0.5f, 0.0f},
                       {kTiny, 0.3f, 0.0f},
                       {max_sub, 0.5f, 0.0f},
                       {kSub, -0.5f, kInf},
                       {kSub, 0.0f, 1.0f},
                       {-kSub, 0.5f, 0.0f},
                       {-kSub, 2.0f, 0.0f},
                       {-kSub, -2.0f, kInf},
                       // An odd integer exponent keeps the sign of the base.
                       {-kSub, 3.0f, -0.0f},
                       {-kSub, -3.0f, -kInf},
                       // A subnormal exponent is zero, and pow(x, 0) is 1.
                       {2.0f, kSub, 1.0f},
                       {-2.0f, kSub, 1.0f},
                       {0.0f, -kTiny, 1.0f},
                       {kInf, kTiny, 1.0f},
                       {kNaN, kSub, 1.0f},
                       {kSub, kTiny, 1.0f},
                   });
}

TEST(BuiltinPowTest, F64SubnormalOperandsAreFlushedUnderDaz) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  constexpr double kSub = std::numeric_limits<double>::min() / 2;
  constexpr double kTiny = std::numeric_limits<double>::denorm_min();
  const double max_sub = absl::bit_cast<double>(uint64_t{0x000fffffffffffff});
  ExpectFlushedUnderDaz<double>(BuiltinPow, {
                                                // glibc returns 2^-537.5 here.
                                                {kSub, 0.5, 0.0},
                                                {kTiny, 0.3, 0.0},
                                                {max_sub, 0.5, 0.0},
                                                {kSub, -0.5, kInf},
                                                {kSub, 0.0, 1.0},
                                                {-kSub, 0.5, 0.0},
                                                {-kSub, 2.0, 0.0},
                                                {-kSub, -2.0, kInf},
                                                {-kSub, 3.0, -0.0},
                                                {-kSub, -3.0, -kInf},
                                                {2.0, kSub, 1.0},
                                                {-2.0, kSub, 1.0},
                                                {0.0, -kTiny, 1.0},
                                                {kInf, kTiny, 1.0},
                                                {kNaN, kSub, 1.0},
                                                {kSub, kTiny, 1.0},
                                            });
}

// Every operand must reach libm unchanged, except subnormal ones under DAZ.
template <typename T, typename PowFn, typename LibmFn>
void ExpectSameAsLibm(PowFn pow_fn, LibmFn libm_fn) {
  constexpr T kInf = std::numeric_limits<T>::infinity();
  const std::vector<T> values = {T{0},
                                 -T{0},
                                 std::numeric_limits<T>::denorm_min(),
                                 -std::numeric_limits<T>::min() / 2,
                                 std::numeric_limits<T>::min(),
                                 -std::numeric_limits<T>::min(),
                                 T{0.3},
                                 T{0.5},
                                 T{1},
                                 -T{1},
                                 T{2},
                                 T{-2.5},
                                 T{3},
                                 T{1e-20},
                                 T{1e20},
                                 std::numeric_limits<T>::max(),
                                 kInf,
                                 -kInf,
                                 std::numeric_limits<T>::quiet_NaN()};
  ForEachDenormalState([&](const DenormalState& state) {
    for (T x_value : values) {
      for (T y_value : values) {
        if (state.denormals_are_zero() &&
            (IsSubnormal(x_value) || IsSubnormal(y_value))) {
          continue;
        }
        volatile T x = x_value;
        volatile T y = y_value;
        const T expected = libm_fn(x, y);
        const T actual = pow_fn(x, y);
        if (expected != expected) {
          EXPECT_TRUE(actual != actual) << std::hex << "x 0x" << Bits(x_value)
                                        << ", y 0x" << Bits(y_value);
        } else {
          EXPECT_EQ(Bits(actual), Bits(expected))
              << std::hex << "x 0x" << Bits(x_value) << ", y 0x"
              << Bits(y_value);
        }
      }
    }
  });
}

TEST(BuiltinPowTest, F32OtherOperandsMatchLibm) {
  ExpectSameAsLibm<float>(BuiltinPowf,
                          [](float x, float y) { return std::pow(x, y); });
}

TEST(BuiltinPowTest, F64OtherOperandsMatchLibm) {
  ExpectSameAsLibm<double>(BuiltinPow,
                           [](double x, double y) { return std::pow(x, y); });
}

}  // namespace
}  // namespace xla::cpu
