/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/types.h"

#include <array>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "xla/hlo/testlib/test.h"

namespace xla {
namespace {

TEST(U4Test, NumericLimits) {
  EXPECT_EQ(static_cast<int64_t>(std::numeric_limits<u4>::min()), 0);
  EXPECT_EQ(static_cast<int64_t>(std::numeric_limits<u4>::max()), 15);
  EXPECT_EQ(static_cast<int64_t>(std::numeric_limits<u4>::lowest()), 0);
}

TEST(S4Test, NumericLimits) {
  EXPECT_EQ(static_cast<int64_t>(std::numeric_limits<s4>::min()), -8);
  EXPECT_EQ(static_cast<int64_t>(std::numeric_limits<s4>::max()), 7);
  EXPECT_EQ(static_cast<int64_t>(std::numeric_limits<s4>::lowest()), -8);
}

TEST(C32Test, ComponentLayout) {
  complex32 c32(half(1.0f), half(-1.0f));
  const auto c32_halves = absl::bit_cast<std::array<uint16_t, 2>>(c32);
  EXPECT_EQ(c32_halves[0], absl::bit_cast<uint16_t>(half(1.0f)));
  EXPECT_EQ(c32_halves[1], absl::bit_cast<uint16_t>(half(-1.0f)));
}

TEST(BC32Test, ComponentLayout) {
  bcomplex32 bc32(bfloat16(1.0f), bfloat16(-1.0f));
  const auto bc32_halves = absl::bit_cast<std::array<uint16_t, 2>>(bc32);
  EXPECT_EQ(bc32_halves[0], absl::bit_cast<uint16_t>(bfloat16(1.0f)));
  EXPECT_EQ(bc32_halves[1], absl::bit_cast<uint16_t>(bfloat16(-1.0f)));
}

}  // namespace
}  // namespace xla
