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

TEST(BuiltinAtan2Test, SubnormalInputsAndSpecialValues) {
  tsl::port::ScopedDontFlushDenormal dont_flush;
  const double tiny = std::numeric_limits<double>::denorm_min();
  const double largest_subnormal =
      absl::bit_cast<double>(uint64_t{0x000fffffffffffff});
  const double min_normal = std::numeric_limits<double>::min();
  const double infinity = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> values = {tiny,
                                      -tiny,
                                      2 * tiny,
                                      -2 * tiny,
                                      largest_subnormal,
                                      -largest_subnormal,
                                      min_normal,
                                      -min_normal,
                                      1.0,
                                      -1.0,
                                      0.0,
                                      -0.0,
                                      infinity,
                                      -infinity,
                                      nan};
  std::vector<double> expected;
  for (double y : values) {
    for (double x : values) {
      expected.push_back(std::atan2(y, x));
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
          // Bitwise comparison also checks signed zeros and subnormal outputs.
          EXPECT_EQ(absl::bit_cast<uint64_t>(actual),
                    absl::bit_cast<uint64_t>(expected[index]));
        }
        EXPECT_EQ(tsl::port::GetDenormalState(), state);
        ++index;
      }
    }
  }
}

}  // namespace
}  // namespace xla::cpu
