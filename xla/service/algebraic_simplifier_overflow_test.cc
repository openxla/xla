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

#include <gtest/gtest.h>

#include <optional>
#include <string>

#include "xla/error_spec.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"

namespace xla {
namespace {

class AlgebraicSimplifierOverflowTest
    : public HloInterpreterReferenceMixin<HloTestBase> {};

// Test that the algebraic simplifier does not generate integer overflows
// by moving the subtraction to the other side of the comparison
TEST_F(AlgebraicSimplifierOverflowTest, CompareOptOverflow) {
  const std::string hlo_text = R"(
    HloModule m
    ENTRY test {
      a = s32[2] parameter(0)
      b = s32[2] constant({1, 1})
      diff = s32[2] subtract(a, b)
      c = s32[2] constant({2147483647, 2147483647})
      ROOT ret = pred[2] compare(diff, c), direction=GT
    }
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{0}));
}

TEST_F(AlgebraicSimplifierOverflowTest, DivideToInverseMultiplyExecution) {
  const std::string hlo_text = R"(
    HloModule m
    ENTRY test {
      x = f32[2,4] parameter(0)
      v = f32[2] parameter(1)
      abs_v = f32[2] abs(v)
      eps = f32[2] broadcast(f32[] constant(1.0)), dimensions={}
      pos_v = f32[2] add(abs_v, eps)
      s = f32[2] sqrt(pos_v)
      bs = f32[2,4] broadcast(s), dimensions={0}
      d1 = f32[2,4] divide(x, bs)
      c_int = s32[] constant(4)
      c_f32 = f32[] convert(c_int)
      bc = f32[2,4] broadcast(c_f32), dimensions={}
      d2 = f32[2,4] divide(d1, bc)
      c_neg = f32[] negate(f32[] constant(2.0))
      b_neg = f32[2,4] broadcast(c_neg), dimensions={}
      ROOT d3 = f32[2,4] divide(d2, b_neg)
    }
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace xla
