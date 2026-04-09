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

#include <gtest/gtest.h>
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/xla.pb.h"

namespace xla::cpu {
namespace {

class YnnE2eTest : public HloPjRtTestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloPjRtTestBase::GetDebugOptionsForTest();
    debug_options.add_xla_cpu_experimental_ynn_fusion_type(
        DebugOptions::LIBRARY_FUSION_TYPE_INDIVIDUAL_CONVOLUTION);
    debug_options.clear_xla_cpu_experimental_ynn_fusion_type();
    return debug_options;
  }
};

TEST_F(YnnE2eTest, DoNotDegroupConvolutionFeatures) {
  const char* matmul_module_str = R"(
  HloModule convolution

  ENTRY %main {
    %lhs = f32[1,7,8,9] parameter(0)
    %rhs = f32[1,5,3,9] parameter(1)
    ROOT %conv = f32[1,4,8,9] convolution(%lhs, %rhs),
        window={size=1x5 stride=2x1 pad=0_0x2_2}, dim_labels=b01f_01io->b01f,
        feature_group_count=3
  })";

  // If the convolution feature group is de-grouped, the shape will change to:
  //   f32[1,4,8,3,3]{4,3,2,1,0}
  // This convolution is supported by YNNPACK, so the shape should not change.
  MatchOptimizedHlo(matmul_module_str,
                    "CHECK: f32[1,4,8,9]{3,2,1,0} convolution");
}

class YnnReduceTest : public HloPjRtTestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloPjRtTestBase::GetDebugOptionsForTest();
    debug_options.add_xla_cpu_experimental_ynn_fusion_type(
        DebugOptions::LIBRARY_FUSION_TYPE_REDUCE);
    return debug_options;
  }
};

TEST_F(YnnReduceTest, ReduceWindowFollowedByReduce) {
  const char* hlo_text = R"(
  HloModule reduce_window_reduce

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY main {
    input = f32[512,512] parameter(0)
    init = f32[] constant(0)
    rw = f32[256,256] reduce-window(input, init), window={size=2x2 stride=2x2}, to_apply=add
    ROOT result = f32[] reduce(rw, init), dimensions={0,1}, to_apply=add
  }
  )";

  MatchOptimizedHlo(hlo_text, R"(
    CHECK: reduce-window
    CHECK: reduce
    CHECK: ENTRY
    CHECK: kind=kCustom
    CHECK: "kind":"__ynn_fusion"
  )");
}

TEST_F(YnnReduceTest, ReduceReshape) {
  const char* hlo_text = R"(
  HloModule reduce_reshape

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY main {
    input = f32[512,512] parameter(0)
    init = f32[] constant(0)
    reduced = f32[512] reduce(input, init), dimensions={1}, to_apply=add
    ROOT result = f32[1,512] reshape(reduced)
  }
  )";

  MatchOptimizedHlo(hlo_text, R"(
    CHECK: reduce
    CHECK: reshape
    CHECK: ENTRY
    CHECK: kind=kCustom
    CHECK: "kind":"__ynn_fusion"
  )");
}

TEST_F(YnnReduceTest, ReshapeReduce) {
  const char* hlo_text = R"(
  HloModule reshape_reduce

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY main {
    input = f32[512,512] parameter(0)
    init = f32[] constant(0)
    reshaped = f32[262144] reshape(input)
    ROOT result = f32[] reduce(reshaped, init), dimensions={0}, to_apply=add
  }
  )";

  MatchOptimizedHlo(hlo_text, R"(
    CHECK: reshape
    CHECK: reduce
    CHECK: ENTRY
    CHECK: kind=kCustom
    CHECK: "kind":"__ynn_fusion"
  )");
}

TEST_F(YnnReduceTest, ReduceConvert) {
  const char* hlo_text = R"(
  HloModule reduce_convert

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY main {
    input = f32[512,512] parameter(0)
    init = f32[] constant(0)
    reduced = f32[512] reduce(input, init), dimensions={1}, to_apply=add
    ROOT result = bf16[512] convert(reduced)
  }
  )";

  MatchOptimizedHlo(hlo_text, R"(
    CHECK: %[[reduce:.+]] = {{.+}} reduce({{.+}})
    CHECK: ROOT {{.+}} = {{.+}} convert(%[[reduce]])
    CHECK: ENTRY
    CHECK: kind=kCustom
    CHECK: "kind":"__ynn_fusion"
  )");
}

TEST_F(YnnReduceTest, ConvertReduce) {
  const char* hlo_text = R"(
  HloModule convert_reduce

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY main {
    input = bf16[512,512] parameter(0)
    init = f32[] constant(0)
    converted = f32[512,512] convert(input)
    ROOT result = f32[] reduce(converted, init), dimensions={0,1}, to_apply=add
  }
  )";

  MatchOptimizedHlo(hlo_text, R"(
    CHECK: %[[convert:.+]] = {{.+}} convert({{.+}})
    CHECK: ROOT {{.+}} = {{.+}} reduce-window(%[[convert]], {{.+}})
    CHECK: ENTRY
    CHECK: kind=kCustom
    CHECK: "kind":"__ynn_fusion"
  )");
}

class YnnEltwiseTest : public HloPjRtTestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloPjRtTestBase::GetDebugOptionsForTest();
    debug_options.add_xla_cpu_experimental_ynn_fusion_type(
        DebugOptions::LIBRARY_FUSION_TYPE_ELTWISE);
    return debug_options;
  }
};

TEST_F(YnnEltwiseTest, BroadcastAdd) {
  const char* hlo_text = R"(
  HloModule broadcast_add

  ENTRY main {
    input = f32[512] parameter(0)
    bias = f32[512,512] parameter(1)
    broadcasted = f32[512,512] broadcast(input), dimensions={1}
    ROOT result = f32[512,512] add(broadcasted, bias)
  }
  )";

  MatchOptimizedHlo(hlo_text, R"(
    CHECK: %[[fused_fn:[^ ]+]] (
    CHECK: broadcast
    CHECK: add
    CHECK: }
    CHECK: "kind":"__ynn_fusion"
  )");
}

TEST_F(YnnEltwiseTest, BroadcastMultiply) {
  const char* hlo_text = R"(
  HloModule broadcast_mul

  ENTRY main {
    input = f32[512] parameter(0)
    arg1 = f32[512,512] parameter(1)
    broadcasted = f32[512,512] broadcast(input), dimensions={0}
    ROOT result = f32[512,512] multiply(broadcasted, arg1)
  }
  )";

  MatchOptimizedHlo(hlo_text, R"(
    CHECK: %[[fused_fn:[^ ]+]] (
    CHECK: broadcast
    CHECK: multiply
    CHECK: }
    CHECK: "kind":"__ynn_fusion"
  )");
}

TEST_F(YnnEltwiseTest, Broadcast3DAdd) {
  const char* hlo_text = R"(
  HloModule broadcast_3d_add

  ENTRY main {
    input = f32[128,256] parameter(0)
    bias = f32[128,512,256] parameter(1)
    broadcasted = f32[128,512,256] broadcast(input), dimensions={0,2}
    ROOT result = f32[128,512,256] add(broadcasted, bias)
  }
  )";

  MatchOptimizedHlo(hlo_text, R"(
    CHECK: %[[fused_fn:[^ ]+]] (
    CHECK: broadcast
    CHECK: add
    CHECK: }
    CHECK: "kind":"__ynn_fusion"
  )");
}

TEST_F(YnnEltwiseTest, NonMonotonicBroadcastAdd) {
  const char* hlo_text = R"(
  HloModule non_monotonic_broadcast_add

  ENTRY main {
    input = f32[512,256] parameter(0)
    bias = f32[256,512,1024] parameter(1)
    broadcasted = f32[256,512,1024] broadcast(input), dimensions={1,0}
    ROOT result = f32[256,512,1024] add(broadcasted, bias)
  }
  )";

  // The broadcast is not supported because dimensions={1,0} is not monotonic.
  // It should NOT be fused into the ynn_fusion.
  MatchOptimizedHlo(hlo_text, R"(
    CHECK: %[[fused_fn:[^ ]+]] (
    CHECK-NOT: broadcast
    CHECK: add
    CHECK: }
    CHECK: "kind":"__ynn_fusion"
  )");
}

}  // namespace
}  // namespace xla::cpu
