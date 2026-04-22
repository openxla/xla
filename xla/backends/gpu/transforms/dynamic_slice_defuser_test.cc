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

#include "xla/backends/gpu/transforms/dynamic_slice_defuser.h"

#include <optional>

#include <gtest/gtest.h>
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla::gpu {
namespace {

class DynamicSliceDefuserTest : public HloHardwareIndependentTestBase {};

TEST_F(DynamicSliceDefuserTest, DefusesTrivialDUSFusion) {
  const char* hlo = R"(
    HloModule test

    %fused_dus (p0: f32[4,64], p1: s32[], p2: f32[64]) -> f32[4,64] {
      %p0 = f32[4,64]{1,0} parameter(0)
      %p2 = f32[64]{0} parameter(2)
      %bitcast = f32[1,64]{1,0} bitcast(%p2)
      %p1 = s32[] parameter(1)
      %zero = s32[] constant(0)
      ROOT %dus = f32[4,64]{1,0} dynamic-update-slice(%p0, %bitcast, %p1, %zero)
    }

    ENTRY main {
      %buf = f32[4,64]{1,0} parameter(0)
      %idx = s32[] parameter(1)
      %update = f32[64]{0} parameter(2)
      ROOT %fusion = f32[4,64]{1,0} fusion(%buf, %idx, %update),
          kind=kLoop, calls=%fused_dus
    }
  )";

  RunAndFilecheckHloRewrite(hlo, DynamicSliceDefuser(), R"(
    CHECK: ENTRY
    CHECK-NOT: fusion
    CHECK: bitcast(
    CHECK: ROOT {{.*}} dynamic-update-slice(
  )");
}

TEST_F(DynamicSliceDefuserTest, DefusesTrivialDSFusion) {
  const char* hlo = R"(
    HloModule test

    %fused_ds (p0: f32[4,64], p1: s32[]) -> f32[1,64] {
      %p0 = f32[4,64]{1,0} parameter(0)
      %p1 = s32[] parameter(1)
      %zero = s32[] constant(0)
      ROOT %ds = f32[1,64]{1,0} dynamic-slice(%p0, %p1, %zero),
          dynamic_slice_sizes={1,64}
    }

    ENTRY main {
      %buf = f32[4,64]{1,0} parameter(0)
      %idx = s32[] parameter(1)
      ROOT %fusion = f32[1,64]{1,0} fusion(%buf, %idx),
          kind=kLoop, calls=%fused_ds
    }
  )";

  RunAndFilecheckHloRewrite(hlo, DynamicSliceDefuser(), R"(
    CHECK: ENTRY
    CHECK-NOT: fusion
    CHECK: ROOT {{.*}} dynamic-slice(
  )");
}

TEST_F(DynamicSliceDefuserTest, DoesNotDefuseNonTrivialFusion) {
  const char* hlo = R"(
    HloModule test

    %fused_comp (p0: f32[4,64], p1: f32[64]) -> f32[4,64] {
      %p0 = f32[4,64]{1,0} parameter(0)
      %p1 = f32[64]{0} parameter(1)
      %bitcast = f32[1,64]{1,0} bitcast(%p1)
      %zero = s32[] constant(0)
      %dus = f32[4,64]{1,0} dynamic-update-slice(%p0, %bitcast, %zero, %zero)
      ROOT %add = f32[4,64]{1,0} add(%dus, %p0)
    }

    ENTRY main {
      %buf = f32[4,64]{1,0} parameter(0)
      %update = f32[64]{0} parameter(1)
      ROOT %fusion = f32[4,64]{1,0} fusion(%buf, %update),
          kind=kLoop, calls=%fused_comp
    }
  )";

  RunAndFilecheckHloRewrite(hlo, DynamicSliceDefuser(), std::nullopt);
}

TEST_F(DynamicSliceDefuserTest, DoesNotDefuseCustomFusion) {
  const char* hlo = R"(
    HloModule test

    %fused_dus (p0: f32[4,64], p1: s32[], p2: f32[64]) -> f32[4,64] {
      %p0 = f32[4,64]{1,0} parameter(0)
      %p2 = f32[64]{0} parameter(2)
      %bitcast = f32[1,64]{1,0} bitcast(%p2)
      %p1 = s32[] parameter(1)
      %zero = s32[] constant(0)
      ROOT %dus = f32[4,64]{1,0} dynamic-update-slice(%p0, %bitcast, %p1, %zero)
    }

    ENTRY main {
      %buf = f32[4,64]{1,0} parameter(0)
      %idx = s32[] parameter(1)
      %update = f32[64]{0} parameter(2)
      ROOT %fusion = f32[4,64]{1,0} fusion(%buf, %idx, %update),
          kind=kCustom, calls=%fused_dus
    }
  )";

  RunAndFilecheckHloRewrite(hlo, DynamicSliceDefuser(), std::nullopt);
}

}  // namespace
}  // namespace xla::gpu
