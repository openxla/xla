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

#include "xla/backends/cpu/transforms/reduction_dimension_normalizer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/str_replace.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"

namespace xla::cpu {
namespace {

using ReductionDimensionNormalizerTest = HloHardwareIndependentTestBase;
using ::absl_testing::IsOkAndHolds;

TEST_F(ReductionDimensionNormalizerTest, ElementwiseChain) {
  for (const char* type : {"f32", "f64"}) {
    RunAndFilecheckHloRewrite(
        absl::StrReplaceAll(R"(
HloModule test
add {
  a = $type[] parameter(0)
  b = $type[] parameter(1)
  ROOT sum = $type[] add(a, b)
}
ENTRY main {
  p = $type[1024,1,1] parameter(0)
  a = $type[1024,1,1] abs(p)
  n = $type[1024,1,1] negate(a)
  e = $type[1024,1,1] exponential(n)
  l = $type[1024,1,1] log-plus-one(e)
  z = $type[] constant(0)
  ROOT r = $type[] reduce(l, z), dimensions={0,1,2}, to_apply=add
}
// CHECK: $type[1024]{0} abs(
// CHECK: $type[1024]{0} negate(
// CHECK: $type[1024]{0} exponential(
// CHECK: $type[1024]{0} log-plus-one(
// CHECK: ROOT {{.*}} = $type[] reduce({{.*}}), dimensions={0}
)",
                            {{"$type", type}}),
        ReductionDimensionNormalizer{}, [](HloModule* module) {
          EXPECT_THAT(ReductionDimensionNormalizer{}.Run(module),
                      IsOkAndHolds(false));
        });
  }
}

TEST_F(ReductionDimensionNormalizerTest, PreservesOtherUsersAndOutputShape) {
  RunAndFilecheckHloRewrite(R"(
HloModule test
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT sum = f32[] add(a, b)
}
ENTRY main {
  p = f32[1,8,1,16,1] parameter(0)
  e = f32[1,8,1,16,1] exponential(p)
  z = f32[] constant(0)
  r = f32[1,8,1] reduce(e, z), dimensions={2,3}, to_apply=add
  ROOT t = (f32[1,8,1,16,1], f32[1,8,1]) tuple(e, r)
}
// CHECK: %[[EXP:[^ ]+]] = f32[8,16]{1,0} exponential(
// CHECK: %[[RESTORED:[^ ]+]] = f32[1,8,1,16,1]{4,3,2,1,0} reshape(%[[EXP]])
// CHECK: %[[REDUCE:[^ ]+]] = f32[8]{0} reduce(%[[EXP]], {{.*}}), dimensions={1}
// CHECK: %[[RESULT:[^ ]+]] = f32[1,8,1]{2,1,0} reshape(%[[REDUCE]])
// CHECK: ROOT {{.*}} tuple(%[[RESTORED]], %[[RESULT]])
)",
                            ReductionDimensionNormalizer{});
}

TEST_F(ReductionDimensionNormalizerTest, TreeReductionWindow) {
  RunAndFilecheckHloRewrite(R"(
HloModule test
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT sum = f32[] add(a, b)
}
ENTRY main {
  p = f32[1025,1] parameter(0)
  e = f32[1025,1] exponential(p)
  z = f32[] constant(0)
  w = f32[33,1] reduce-window(e, z), window={size=32x1 stride=32x1 pad=15_16x0_0}, to_apply=add
  ROOT r = f32[] reduce(w, z), dimensions={0,1}, to_apply=add
}
// CHECK: %[[EXP:[^ ]+]] = f32[1025]{0} exponential(
// CHECK: %[[WINDOW:[^ ]+]] = f32[33]{0} reduce-window(%[[EXP]], {{.*}}), window={size=32 stride=32 pad=15_16}
// CHECK: ROOT {{.*}} = f32[] reduce(%[[WINDOW]], {{.*}}), dimensions={0}
)",
                            ReductionDimensionNormalizer{});
}

TEST_F(ReductionDimensionNormalizerTest, KeepsWindowOutputDimensions) {
  // The first dimension becomes size one because of the window. It must not
  // be removed along with the input's size-one second dimension.
  RunAndFilecheckHloRewrite(R"(
HloModule test
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT sum = f32[] add(a, b)
}
ENTRY main {
  p = f32[32,1] parameter(0)
  z = f32[] constant(0)
  ROOT w = f32[1,1] reduce-window(p, z), window={size=32x1}, to_apply=add
}
// CHECK: %[[WINDOW:[^ ]+]] = f32[1]{0} reduce-window({{.*}}), window={size=32}
// CHECK: ROOT {{.*}} = f32[1,1]{1,0} reshape(%[[WINDOW]])
)",
                            ReductionDimensionNormalizer{});
}

TEST_F(ReductionDimensionNormalizerTest, DoesNotDropNontrivialWindow) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT sum = f32[] add(a, b)
}
ENTRY main {
  p = f32[8,1] parameter(0)
  z = f32[] constant(0)
  ROOT w = f32[8,2] reduce-window(p, z), window={size=1x2 pad=0_0x1_1}, to_apply=add
})");
  ASSERT_TRUE(module.ok()) << module.status();
  EXPECT_THAT(ReductionDimensionNormalizer{}.Run(module->get()),
              IsOkAndHolds(false));
}

TEST_F(ReductionDimensionNormalizerTest, DynamicDimensionIsNotRemoved) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT sum = f32[] add(a, b)
}
ENTRY main {
  p = f32[8,<=1] parameter(0)
  z = f32[] constant(0)
  ROOT r = f32[] reduce(p, z), dimensions={0,1}, to_apply=add
})");
  ASSERT_TRUE(module.ok()) << module.status();
  EXPECT_THAT(ReductionDimensionNormalizer{}.Run(module->get()),
              IsOkAndHolds(false));
}

TEST_F(ReductionDimensionNormalizerTest, EmptyInputKeepsZeroExtent) {
  RunAndFilecheckHloRewrite(R"(
HloModule test
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT sum = f32[] add(a, b)
}
ENTRY main {
  p = f32[0,1] parameter(0)
  z = f32[] constant(7)
  ROOT r = f32[] reduce(p, z), dimensions={0,1}, to_apply=add
}
// CHECK: %[[INPUT:[^ ]+]] = f32[0]{0} reshape(
// CHECK: ROOT {{.*}} = f32[] reduce(%[[INPUT]], {{.*}}), dimensions={0}
)",
                            ReductionDimensionNormalizer{});
}

TEST_F(ReductionDimensionNormalizerTest, ScalarReductionKeepsInitialValue) {
  RunAndFilecheckHloRewrite(R"(
HloModule test
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT sum = f32[] add(a, b)
}
ENTRY main {
  p = f32[1,1] parameter(0)
  z = f32[] constant(7)
  ROOT r = f32[] reduce(p, z), dimensions={0,1}, to_apply=add
}
// CHECK: %[[INPUT:[^ ]+]] = f32[] reshape(
// CHECK: ROOT {{.*}} = f32[] reduce(%[[INPUT]], %z), dimensions={}
)",
                            ReductionDimensionNormalizer{});
}

TEST_F(ReductionDimensionNormalizerTest, VariadicReductionIsUnchanged) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  c = f32[] parameter(2)
  d = f32[] parameter(3)
  ac = f32[] add(a, c)
  bd = f32[] add(b, d)
  ROOT t = (f32[], f32[]) tuple(ac, bd)
}
ENTRY main {
  p = f32[8,1] parameter(0)
  q = f32[8,1] parameter(1)
  z = f32[] constant(0)
  ROOT r = (f32[], f32[]) reduce(p, q, z, z), dimensions={0,1}, to_apply=add
})");
  ASSERT_TRUE(module.ok()) << module.status();
  EXPECT_THAT(ReductionDimensionNormalizer{}.Run(module->get()),
              IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla::cpu
