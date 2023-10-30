/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/priority_fusion.h"

#include <stdint.h>

#include <memory>

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/verified_hlo_module.h"

namespace m = ::xla::match;

namespace xla {
namespace gpu {

class PriorityFusionTest : public HloTestBase {
  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const {
    return [&](const Shape& shape) {
      constexpr int64_t kPointerSize = 8;
      return ShapeUtil::ByteSizeOf(shape, kPointerSize);
    };
  }

 public:
  std::vector<HloFusionAnalysis::EmitterFusionKind> RunAndGetFusionKinds(
      absl::string_view hlo) {
    auto module = ParseAndReturnVerifiedModule(hlo).value();
    EXPECT_THAT(priority_fusion_.Run(module.get()),
                ::testing::status::IsOkAndHolds(true));
    TF_CHECK_OK(module->RemoveUnusedComputations());
    std::vector<HloFusionAnalysis::EmitterFusionKind> kinds;
    for (auto computation : module->computations()) {
      if (!computation->FusionInstruction()) continue;

      auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();
      auto analysis =
          HloFusionAnalysis::Create(
              Cast<HloFusionInstruction>(computation->FusionInstruction()),
              &device_info)
              .value();
      kinds.push_back(analysis.GetEmitterFusionKind());
    }
    return kinds;
  }

  GpuPriorityFusion priority_fusion_{
      TestGpuDeviceInfo::RTXA6000DeviceInfo(),
      GpuHloCostAnalysis::Options{ShapeSizeBytesFunction(),
                                  /*per_second_rates=*/{},
                                  /*count_multiple_input_accesses=*/true}};
};

TEST_F(PriorityFusionTest, FuseWithSharedArgument) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY main {
      %p0 = f32[] parameter(0)
      %p1 = f32[] parameter(1)
      %subtract = f32[] subtract(%p0, %p1)
      %compare = pred[] compare(%subtract, %subtract), direction=NE
      %add = f32[] add(%p0, %p1)
      %abs = f32[] abs(%subtract)
      ROOT %select = f32[] select(%compare, %add, %abs)
    })")
                    .value();

  EXPECT_TRUE(priority_fusion_.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Fusion()));
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kLoop);
}

TEST_F(PriorityFusionTest, FusionFusionWithDuplication) {
  absl::string_view kHlo = R"(
    HloModule test_module

    square {
      p = f32[16384]{0} parameter(0)
      ROOT m = f32[16384]{0} multiply(p, p)
    }

    exp {
      p = f32[16384]{0} parameter(0)
      ROOT e = f32[16384]{0} exponential(p)
    }

    log {
      p = f32[16384]{0} parameter(0)
      ROOT l = f32[16384]{0} log(p)
    }

    ENTRY main {
      p = f32[16384]{0} parameter(0)
      s = f32[16384]{0} fusion(p), kind=kLoop, calls=square
      e = f32[16384]{0} fusion(s), kind=kLoop, calls=exp
      l = f32[16384]{0} fusion(s), kind=kInput, calls=log
      ROOT t = (f32[16384], f32[16384]) tuple(l, e)
    })";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK: ENTRY
CHECK-NEXT: %[[PARAM:.*]] = f32[16384]{0} parameter(0)
CHECK-NEXT: %[[FUSION_0:.*]] = f32[16384]{0} fusion(%[[PARAM]])
CHECK-NEXT: %[[FUSION_1:.*]] = f32[16384]{0} fusion(%[[PARAM]])
CHECK-NEXT: ROOT {{.*}} tuple(%[[FUSION_0]], %[[FUSION_1]])
  )");
}

TEST_F(PriorityFusionTest, FuseWideningConvertIntoConsumers) {
  absl::string_view kHlo = R"(
    HloModule test_module

    ENTRY main {
      p = f16[512]{0} parameter(0)
      a = f16[512]{0} add(p, p)
      c = f32[512]{0} convert(a)
      s = f32[512]{0} multiply(c, c)
      bc = s32[512]{0} bitcast(c)
      ROOT t = (f32[512], s32[512]) tuple(s, bc)
    })";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK:      ENTRY
CHECK-NEXT: %[[PARAM:.*]] = f16[512]{0} parameter(0)
CHECK-NEXT: %[[FUSION_F32:.*]] = f32[512]{0} fusion(%[[PARAM]])
CHECK-NEXT: %[[FUSION_S32:.*]] = s32[512]{0} fusion(%[[PARAM]])
CHECK-NEXT: ROOT %{{.*}} = (f32[512]{0}, s32[512]{0}) tuple(%[[FUSION_F32]], %[[FUSION_S32]])
  )");
}

TEST_F(PriorityFusionTest, FuseConvertIntoReduce) {
  absl::string_view kHlo = R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add.13235 = f32[] add(p0, p1)
    }

    ENTRY main {
      param_0_0.79 = bf16[1024,8192]{1,0} parameter(0)
      param_1_0.79 = bf16[1024,8192]{1,0} parameter(1)
      param_2.483 = f32[8192]{0} parameter(2)
      param_4.2892 = bf16[1024,8192]{1,0} parameter(3)
      convert.21854 = f32[1024,8192]{1,0} convert(param_0_0.79)
      convert.21855 = f32[1024,8192]{1,0} convert(param_1_0.79)
      constant_7773 = f32[] constant(0)
      broadcast.14555 = f32[1024,8192]{1,0} broadcast(param_2.483), dimensions={1}
      multiply.6906 = f32[1024,8192]{1,0} multiply(broadcast.14555, convert.21854)
      reduce.4813 = f32[1024]{0} reduce(multiply.6906, constant_7773), dimensions={1}, to_apply=add
      convert.13970 = bf16[1024]{0} convert(reduce.4813)
      convert.21534 = f32[1024,8192]{1,0} convert(param_4.2892)
      multiply.6910.clone.1 = f32[1024,8192]{1,0} multiply(broadcast.14555, convert.21534)
      reduce.4811.clone.1 = f32[1024]{0} reduce(multiply.6910.clone.1, constant_7773), dimensions={1}, to_apply=add
      convert.13967.clone.1 = bf16[1024]{0} convert(reduce.4811.clone.1)
      multiply.6908.clone.1 = f32[1024,8192]{1,0} multiply(broadcast.14555, convert.21855)
      reduce.4812.clone.1 = f32[1024]{0} reduce(multiply.6908.clone.1, constant_7773), dimensions={1}, to_apply=add
      convert.13969.clone.1 = bf16[1024]{0} convert(reduce.4812.clone.1)
      ROOT fusion.241 = (bf16[1024]{0}, bf16[1024]{0}, bf16[1024]{0}) tuple(convert.13970, convert.13967.clone.1, convert.13969.clone.1)
    })";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK-COUNT-3: ROOT {{.*}} convert(
CHECK: ENTRY %main
CHECK-COUNT-3: fusion
  )");
}

TEST_F(PriorityFusionTest, ReductionEpilogueFusionRegressionTest) {
  // Regression test for epilogue fusion of convert+bitcast into a reduction.
  absl::string_view kHlo = R"(
    HloModule test_module

    add {
      rhs.407 = f32[] parameter(1)
      lhs.407 = f32[] parameter(0)
      ROOT add.24451 = f32[] add(lhs.407, rhs.407)
    }

    ENTRY main {
      param_1.15162 = f32[2752]{0} parameter(1)
      convert.44829 = bf16[2752]{0} convert(param_1.15162)
      bitcast.24686 = bf16[1,1,2752]{2,1,0} bitcast(convert.44829)
      convert.44468 = f32[1,1,2752]{2,1,0} convert(bitcast.24686)
      constant_13722 = bf16[] constant(1)
      convert.17451 = f32[] convert(constant_13722)
      broadcast.17565 = f32[1,1,2752]{2,1,0} broadcast(convert.17451), dimensions={}
      negate.167 = f32[1,1,2752]{2,1,0} negate(convert.44468)
      exponential.569 = f32[1,1,2752]{2,1,0} exponential(negate.167)
      add.1850 = f32[1,1,2752]{2,1,0} add(broadcast.17565, exponential.569)
      divide.1376 = f32[1,1,2752]{2,1,0} divide(broadcast.17565, add.1850)
      multiply.9709 = f32[1,1,2752]{2,1,0} multiply(convert.44468, divide.1376)
      param_0.15005 = f32[2752]{0} parameter(0)
      convert.44826 = bf16[2752]{0} convert(param_0.15005)
      bitcast.24683 = bf16[1,1,2752]{2,1,0} bitcast(convert.44826)
      convert.44467 = f32[1,1,2752]{2,1,0} convert(bitcast.24683)
      multiply.9708 = f32[1,1,2752]{2,1,0} multiply(multiply.9709, convert.44467)
      convert.16959 = bf16[1,1,2752]{2,1,0} convert(multiply.9708)
      fusion.3203 = bf16[2752]{0} bitcast(convert.16959)
      convert.15093 = f32[2752]{0} convert(fusion.3203)
      broadcast.13841 = f32[8192,2752]{1,0} broadcast(convert.15093), dimensions={1}
      param_0.15525 = bf16[8192,2752]{1,0} parameter(2)
      convert.13738 = f32[8192,2752]{1,0} convert(param_0.15525)
      multiply.6422 = f32[8192,2752]{1,0} multiply(broadcast.13841, convert.13738)
      constant_14382 = f32[] constant(0)
      fusion.339 = f32[8192]{0} reduce(multiply.6422, constant_14382), dimensions={1}, to_apply=add
      convert.44633 = bf16[8192]{0} convert(fusion.339)
      ROOT bitcast.24487 = bf16[1,1,8192]{2,1,0} bitcast(convert.44633)
    }
                                                     )";

  EXPECT_THAT(
      RunAndGetFusionKinds(kHlo),
      ::testing::ElementsAre(HloFusionAnalysis::EmitterFusionKind::kReduction,
                             HloFusionAnalysis::EmitterFusionKind::kLoop));

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK: ENTRY
CHECK: ROOT {{.*}} fusion(
  )");
}

TEST_F(PriorityFusionTest, DoNotFuseReduceIntoTransposeSimple) {
  absl::string_view kHlo = R"(
    HloModule jit_apply_fn, is_scheduled=true,
    entry_computation_layout={(f32[256,50]{1,0})->f32[50,256]{1,0}}

    region_sum {
      param0 = f32[] parameter(0)
      param1 = f32[] parameter(1)
      ROOT add = f32[] add(param0, param1)
    }

    reduce {
      param0 = f32[256,50]{1,0} parameter(0)
      zero = f32[] constant(0)
      ROOT reduce = f32[256]{0} reduce(param0, zero), dimensions={1},
    to_apply=region_sum
    }

    fused_transpose {
      param0 = f32[256,50]{1,0} parameter(0)
      transpose = f32[50,256]{1,0} transpose(param0), dimensions={1,0}
      param1 = f32[256]{0} parameter(1)
      broadcast = f32[50,256]{1,0} broadcast(param1), dimensions={1}
      ROOT add = f32[50,256]{1,0} add(transpose, broadcast)
    }

    ENTRY main {
      param0 = f32[256,50]{1,0} parameter(0), sharding={replicated}
      reduce = f32[256]{0} fusion(param0), kind=kInput, calls=reduce
      ROOT fusion = f32[50,256]{1,0} fusion(param0, reduce), kind=kInput,
    calls=fused_transpose
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(priority_fusion_.Run(module.get()),
              ::testing::status::IsOkAndHolds(false));
}

TEST_F(PriorityFusionTest, DoNotFuseReduceIntoTransposeResnet200) {
  absl::string_view kHlo = R"(
    HloModule jit_apply_fn, is_scheduled=true, entry_computation_layout={
      (f16[256,1,50,1]{3,2,1,0}, f16[256]{0}, f16[1,1,1,256]{3,2,1,0})->f16[1,1,50,256]{3,2,1,0}
    }

    region_sum {
      Arg_1.657 = f32[] parameter(1)
      Arg_0.656 = f32[] parameter(0)
      ROOT add.205 = f32[] add(Arg_0.656, Arg_1.657)
    }

    %fused_transpose (param_0.10044: f16[256,1,50,1], param_1.11032: f16[256], param_2.4666: f16[1,1,1,256], param_3.3799: f16[1,1,1,256], param_4.2502: f16[256]) -> f16[1,1,50,256] {
      %param_0.10044 = f16[256,1,50,1]{3,2,1,0} parameter(0)
      %transpose.410 = f16[1,1,50,256]{3,2,1,0} transpose(f16[256,1,50,1]{3,2,1,0} %param_0.10044), dimensions={3,1,2,0}
      %param_1.11032 = f16[256]{0} parameter(1)
      %constant.2525 = f16[] constant(0.020004)
      %broadcast.5429 = f16[256]{0} broadcast(f16[] %constant.2525), dimensions={}
      %multiply.6117 = f16[256]{0} multiply(f16[256]{0} %param_1.11032, f16[256]{0} %broadcast.5429)
      %broadcast.2476 = f16[1,1,50,256]{3,2,1,0} broadcast(f16[256]{0} %multiply.6117), dimensions={3}
      %subtract.827 = f16[1,1,50,256]{3,2,1,0} subtract(f16[1,1,50,256]{3,2,1,0} %transpose.410, f16[1,1,50,256]{3,2,1,0} %broadcast.2476)
      %param_3.3799 = f16[1,1,1,256]{3,2,1,0} parameter(3)
      %param_4.2502 = f16[256]{0} parameter(4)
      %constant.4175 = f16[] constant(0.020004)
      %broadcast.6031 = f16[256]{0} broadcast(f16[] %constant.4175), dimensions={}
      %multiply.6798 = f16[256]{0} multiply(f16[256]{0} %param_4.2502, f16[256]{0} %broadcast.6031)
      %broadcast.6030 = f16[256]{0} broadcast(f16[] %constant.4175), dimensions={}
      %multiply.6797 = f16[256]{0} multiply(f16[256]{0} %param_1.11032, f16[256]{0} %broadcast.6030)
      %multiply.6796 = f16[256]{0} multiply(f16[256]{0} %multiply.6797, f16[256]{0} %multiply.6797)
      %subtract.2011 = f16[256]{0} subtract(f16[256]{0} %multiply.6798, f16[256]{0} %multiply.6796)
      %constant.4174 = f16[] constant(1.0014e-05)
      %broadcast.6029 = f16[256]{0} broadcast(f16[] %constant.4174), dimensions={}
      %add.2304 = f16[256]{0} add(f16[256]{0} %subtract.2011, f16[256]{0} %broadcast.6029)
      %rsqrt.1349 = f16[256]{0} rsqrt(f16[256]{0} %add.2304)
      %bitcast.3364 = f16[1,1,1,256]{3,2,1,0} bitcast(f16[256]{0} %rsqrt.1349)
      %multiply.6795 = f16[1,1,1,256]{3,2,1,0} multiply(f16[1,1,1,256]{3,2,1,0} %param_3.3799, f16[1,1,1,256]{3,2,1,0} %bitcast.3364)
      %bitcast.3363 = f16[256]{0} bitcast(f16[1,1,1,256]{3,2,1,0} %multiply.6795)
      %broadcast.2475 = f16[1,1,50,256]{3,2,1,0} broadcast(f16[256]{0} %bitcast.3363), dimensions={3}
      %multiply.3108 = f16[1,1,50,256]{3,2,1,0} multiply(f16[1,1,50,256]{3,2,1,0} %subtract.827, f16[1,1,50,256]{3,2,1,0} %broadcast.2475)
      %param_2.4666 = f16[1,1,1,256]{3,2,1,0} parameter(2)
      %bitcast.3042 = f16[256]{0} bitcast(f16[1,1,1,256]{3,2,1,0} %param_2.4666)
      %broadcast.2474 = f16[1,1,50,256]{3,2,1,0} broadcast(f16[256]{0} %bitcast.3042), dimensions={3}
      %add.1064 = f16[1,1,50,256]{3,2,1,0} add(f16[1,1,50,256]{3,2,1,0} %multiply.3108, f16[1,1,50,256]{3,2,1,0} %broadcast.2474)
      %constant.3263 = f16[] constant(0)
      %broadcast.141 = f16[1,1,50,256]{3,2,1,0} broadcast(f16[] %constant.3263), dimensions={}
      ROOT %maximum.67 = f16[1,1,50,256]{3,2,1,0} maximum(f16[1,1,50,256]{3,2,1,0} %add.1064, f16[1,1,50,256]{3,2,1,0} %broadcast.141)
    }

    %fused_reduce (param_0.9689: f16[256,1,50,1]) -> f16[256] {
      %param_0.9689 = f16[256,1,50,1]{3,2,1,0} parameter(0)
      %multiply.3994 = f16[256,1,50,1]{3,2,1,0} multiply(f16[256,1,50,1]{3,2,1,0} %param_0.9689, f16[256,1,50,1]{3,2,1,0} %param_0.9689)
      %convert.3057 = f32[256,1,50,1]{3,2,1,0} convert(f16[256,1,50,1]{3,2,1,0} %multiply.3994)
      %bitcast.1598 = f32[256,50]{1,0} bitcast(f32[256,1,50,1]{3,2,1,0} %convert.3057)
      %constant.3669 = f32[] constant(0)
      %reduce.593 = f32[256]{0} reduce(f32[256,50]{1,0} %bitcast.1598, f32[] %constant.3669), dimensions={1}, to_apply=region_sum
      ROOT %convert.3055 = f16[256]{0} convert(f32[256]{0} %reduce.593)
    }

    ENTRY main {
      parameter0 = f16[256,1,50,1]{3,2,1,0} parameter(0), sharding={replicated}
      parameter1 = f16[256]{0} parameter(1), sharding={replicated}
      parameter2 = f16[1,1,1,256]{3,2,1,0} parameter(2), sharding={replicated}
      fused_reduce = f16[256]{0} fusion(parameter0), kind=kInput, calls=fused_reduce
      ROOT fused_transpose = f16[1,1,50,256]{3,2,1,0} fusion(parameter0, parameter1, parameter2, parameter2, fused_reduce), kind=kInput, calls=fused_transpose
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(priority_fusion_.Run(module.get()),
              ::testing::status::IsOkAndHolds(false));
}

TEST_F(PriorityFusionTest, DoNotFuseReduceIntoReduce) {
  absl::string_view kHlo = R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add.13235 = f32[] add(p0, p1)
    }

    ENTRY main {
      p0 = f32[8,4,128,226]{3,2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r0 = f32[8,4,128]{2,1,0} reduce(p0, c0), dimensions={3}, to_apply=add
      ROOT r1 = f32[8,4]{1,0} reduce(r0, c0), dimensions={2}, to_apply=add
    })";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK: ROOT {{.*}} reduce(
CHECK: ROOT {{.*}} reduce(
  )");
}

TEST_F(PriorityFusionTest, ConvertFusedIntoReduce) {
  absl::string_view kHlo = R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add.13235 = f32[] add(p0, p1)
    }

    ENTRY main {
      param_0_0.79 = bf16[1024,8192]{1,0} parameter(0)
      param_1_0.79 = bf16[1024,8192]{1,0} parameter(1)
      param_2.483 = f32[8192]{0} parameter(2)
      param_4.2892 = bf16[1024,8192]{1,0} parameter(3)
      convert.21854 = f32[1024,8192]{1,0} convert(param_0_0.79)
      convert.21855 = f32[1024,8192]{1,0} convert(param_1_0.79)
      constant_7773 = f32[] constant(0)
      broadcast.14555 = f32[1024,8192]{1,0} broadcast(param_2.483), dimensions={1}
      multiply.6906 = f32[1024,8192]{1,0} multiply(broadcast.14555, convert.21854)
      reduce.4813 = f32[1024]{0} reduce(multiply.6906, constant_7773), dimensions={1}, to_apply=add
      convert.13970 = bf16[1024]{0} convert(reduce.4813)
      convert.21534 = f32[1024,8192]{1,0} convert(param_4.2892)
      multiply.6910.clone.1 = f32[1024,8192]{1,0} multiply(broadcast.14555, convert.21534)
      reduce.4811.clone.1 = f32[1024]{0} reduce(multiply.6910.clone.1, constant_7773), dimensions={1}, to_apply=add
      convert.13967.clone.1 = bf16[1024]{0} convert(reduce.4811.clone.1)
      multiply.6908.clone.1 = f32[1024,8192]{1,0} multiply(broadcast.14555, convert.21855)
      reduce.4812.clone.1 = f32[1024]{0} reduce(multiply.6908.clone.1, constant_7773), dimensions={1}, to_apply=add
      convert.13969.clone.1 = bf16[1024]{0} convert(reduce.4812.clone.1)
      ROOT fusion.241 = (bf16[1024]{0}, bf16[1024]{0}, bf16[1024]{0}) tuple(convert.13970, convert.13967.clone.1, convert.13969.clone.1)
    })";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK-COUNT-3: ROOT {{.*}} convert(
CHECK: ENTRY %main
CHECK-COUNT-3: fusion(
CHECK-NOT: fusion(
  )");
}

TEST_F(PriorityFusionTest, DoNotFuseDynamicUpdateSliceIntoReduce) {
  GTEST_SKIP() << "b/294198633";
  absl::string_view kHlo = R"(
    HloModule test_module

add {
  Arg_1.1046 = f32[] parameter(1)
  Arg_0.1045 = f32[] parameter(0)
  ROOT add.3303 = f32[] add(Arg_0.1045, Arg_1.1046)
}

ENTRY main {
  param_0.10549 = f32[4,2112]{1,0} parameter(0)
  param_5.2561 = pred[] parameter(5)
  broadcast.19725 = pred[4,1]{1,0} broadcast(param_5.2561), dimensions={}
  param_1.11587 = pred[4]{0} parameter(1)
  constant_5837 = f32[] constant(1)
  broadcast.19723 = f32[4]{0} broadcast(constant_5837), dimensions={}
  param_2.5952 = f32[4,8000]{1,0} parameter(2)
  param_3.4004 = f32[4]{0} parameter(3)
  broadcast.19718 = f32[4,8000]{1,0} broadcast(param_3.4004), dimensions={0}
  subtract.1112 = f32[4,8000]{1,0} subtract(param_2.5952, broadcast.19718)
  exponential.418 = f32[4,8000]{1,0} exponential(subtract.1112)
  constant_6254 = f32[] constant(0)
  reduce.1154 = f32[4]{0} reduce(exponential.418, constant_6254), dimensions={1}, to_apply=add
  log.38 = f32[4]{0} log(reduce.1154)
  broadcast.19717 = f32[4,8000]{1,0} broadcast(log.38), dimensions={0}
  subtract.1111 = f32[4,8000]{1,0} subtract(subtract.1112, broadcast.19717)
  iota.170 = s32[4,1]{1,0} iota(), iota_dimension=0
  constant_6281 = s32[] constant(0)
  broadcast.19735 = s32[4]{0} broadcast(constant_6281), dimensions={}
  param_4.3400 = s32[4,8000]{1,0} parameter(4)
  slice.3186 = s32[4,40]{1,0} slice(param_4.3400), slice={[0:4], [0:40]}
  iota.168 = s32[4,1]{1,0} iota(), iota_dimension=0
  param_7.1596 = s32[4]{0} parameter(7)
  compare.341 = pred[4]{0} compare(param_7.1596, broadcast.19735), direction=LT
  constant_5833 = s32[] constant(40)
  broadcast.19731 = s32[4]{0} broadcast(constant_5833), dimensions={}
  add.8348 = s32[4]{0} add(param_7.1596, broadcast.19731)
  select.418 = s32[4]{0} select(compare.341, add.8348, param_7.1596)
  bitcast.20942 = s32[4,1]{1,0} bitcast(select.418)
  concatenate.1337 = s32[4,2]{1,0} concatenate(iota.168, bitcast.20942), dimensions={1}
  gather.43 = s32[4,1,1]{2,1,0} gather(slice.3186, concatenate.1337), offset_dims={1,2}, collapsed_slice_dims={}, start_index_map={0,1}, index_vector_dim=1, slice_sizes={1,1}
  bitcast.20941 = s32[4]{0} bitcast(gather.43)
  select.398 = s32[4]{0} select(param_1.11587, broadcast.19735, bitcast.20941)
  compare.334 = pred[4]{0} compare(select.398, broadcast.19735), direction=LT
  constant_6260 = s32[] constant(8000)
  broadcast.19720 = s32[4]{0} broadcast(constant_6260), dimensions={}
  add.8336 = s32[4]{0} add(select.398, broadcast.19720)
  select.396 = s32[4]{0} select(compare.334, add.8336, select.398)
  bitcast.20830 = s32[4,1]{1,0} bitcast(select.396)
  concatenate.1308 = s32[4,2]{1,0} concatenate(iota.170, bitcast.20830), dimensions={1}
  gather.41 = f32[4,1,1]{2,1,0} gather(subtract.1111, concatenate.1308), offset_dims={1,2}, collapsed_slice_dims={}, start_index_map={0,1}, index_vector_dim=1, slice_sizes={1,1}
  bitcast.20824 = f32[4]{0} bitcast(gather.41)
  select.389 = f32[4]{0} select(param_1.11587, broadcast.19723, bitcast.20824)
  bitcast.20823 = f32[4,1]{1,0} bitcast(select.389)
  param_6.1719 = s32[] parameter(6)
  constant_6323 = s32[] constant(2048)
  add.8549 = s32[] add(param_6.1719, constant_6323)
  compare.388 = pred[] compare(add.8549, constant_6281), direction=LT
  constant_5436 = s32[] constant(4160)
  add.8339 = s32[] add(param_6.1719, constant_5436)
  select.409 = s32[] select(compare.388, add.8339, add.8549)
  dynamic-slice.36 = f32[4,1]{1,0} dynamic-slice(param_0.10549, constant_6281, select.409), dynamic_slice_sizes={4,1}
  select.388 = f32[4,1]{1,0} select(broadcast.19725, bitcast.20823, dynamic-slice.36)
  ROOT dynamic-update-slice.307 = f32[4,2112]{1,0} dynamic-update-slice(param_0.10549, select.388, constant_6281, select.409)
})";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK: ROOT {{.*}} dynamic-update-slice(
CHECK: %[[REDUCE:.*]] = {{.*}} reduce(
CHECK: ROOT {{.*}} log(%[[REDUCE]])
CHECK: ENTRY
CHECK-COUNT-2: fusion(
  )");
}

TEST_F(PriorityFusionTest, SingleTransposeFusion) {
  // A regression test that verifies the given HLO fuses into a single fusion.
  absl::string_view kHlo = R"(
    HloModule test_module

  ENTRY main {
    param_0.14390 = bf16[2048,24576]{1,0} parameter(0)
    convert.34192 = f32[2048,24576]{1,0} convert(param_0.14390)
    constant_11107 = bf16[] constant(0.02002)
    convert.35472 = f32[] convert(constant_11107)
    broadcast.21886 = f32[2048,24576]{1,0} broadcast(convert.35472), dimensions={}
    multiply.14420 = f32[2048,24576]{1,0} multiply(convert.34192, broadcast.21886)
    fusion.3520 = f32[2048,24576]{1,0} tanh(multiply.14420)

    constant_11286 = bf16[] constant(50)
    convert.42562 = f32[] convert(constant_11286)
    broadcast.22230 = f32[2048,24576]{1,0} broadcast(convert.42562), dimensions={}
    multiply.14798 = f32[2048,24576]{1,0} multiply(fusion.3520, broadcast.22230)
    convert.34603 = bf16[2048,24576]{1,0} convert(multiply.14798)
    bitcast.21354 = bf16[1,2048,2048,12]{3,2,1,0} bitcast(convert.34603)
    ROOT transpose.6502 = bf16[1,12,2048,2048]{3,2,1,0} transpose(bitcast.21354), dimensions={0,3,2,1}
  })";

  using Kind = HloFusionAnalysis::EmitterFusionKind;
  EXPECT_THAT(RunAndGetFusionKinds(kHlo),
              ::testing::ElementsAre(Kind::kTranspose));
}

TEST_F(PriorityFusionTest, DontFuseIntoFirstOperandOfScatter) {
  auto module = *ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY FuseIntoScatter {
      p0 = s32[3,3] parameter(0)
      operand = s32[3,3] add(p0, p0)
      p1 = s32[2] parameter(1)
      indices = s32[2] add(p1, p1)
      p2 = s32[2,3] parameter(2)
      updates = s32[2,3] add(p2, p2)
      scatter = s32[3,3] scatter(operand, indices, updates),
          to_apply=add,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
      ROOT add = s32[3,3] add(scatter, scatter)
    })");

  EXPECT_TRUE(priority_fusion_.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root, GmockMatch(m::Add(m::Fusion(&fusion), m::Fusion())));
  EXPECT_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kInput);
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Scatter(m::Parameter(), m::Add(), m::Add())));
}

TEST_F(PriorityFusionTest, DoNotFuseReduceIntoReduceEvenIfOccupancyIsHigh) {
  constexpr absl::string_view kHlo = R"(
    HloModule test_module

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY main {
      p0 = f32[4,3584,128,168]{3,2,1,0} parameter(0)
      c = f32[] constant(0)
      r1 = f32[4,3584,128]{2,1,0} reduce(p0, c), dimensions={3}, to_apply=add
      ROOT r2 = f32[4,3584]{1,0} reduce(r1, c), dimensions={2}, to_apply=add
    })";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK: ROOT {{.*}} reduce(
CHECK: ROOT {{.*}} reduce(
  )");
}

TEST_F(PriorityFusionTest, FuseReductionEpilogueWithMultipleUsers) {
  // Regression test that verifies we correctly fuse the `log` into the reduce.
  constexpr absl::string_view kHlo = R"(
    HloModule test_module

    add {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT add = f32[] add(x, y)
    }

    fused_computation {
      p0 = f32[64,16384]{1,0} parameter(0)
      c0 = f32[] constant(0)
      ROOT reduce.858 = f32[64]{0} reduce(p0, c0), dimensions={1}, to_apply=add
    }

    ENTRY main {
      p0 = f32[64,16384]{1,0} parameter(0)
      fusion = f32[64]{0} fusion(p0), kind=kInput, calls=fused_computation
      log = f32[64]{0} log(fusion)
      negate = f32[64]{0} custom-call(log), custom_call_target="negate"
      ROOT add = f32[64]{0} add(negate, log)
    }
  )";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
    CHECK: ENTRY
    CHECK: %[[PARAM:.*]] = {{.*}} parameter(0)
    CHECK: %[[FUSION:.*]] = {{.*}} fusion(%[[PARAM]])
    CHECK: custom-call(%[[FUSION]])
  )");
}

TEST_F(PriorityFusionTest, EpilogueFusion) {
  absl::string_view kHlo = R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add.13235 = f32[] add(p0, p1)
    }

    fused_computation.1 {
      p0 = f32[8,4,128,226]{3,2,1,0} parameter(0)
      c0 = f32[] constant(0)
      ROOT r0 = f32[8,4,128]{2,1,0} reduce(p0, c0), dimensions={3}, to_apply=add
    }

    fused_computation.2 {
      p0 = f32[8,4,128]{2,1,0} parameter(0)
      r1 = f32[8,4,128]{2,1,0} log(p0)
      ROOT r2 = f32[8,4,128]{2,1,0} log(r1)
    }

    ENTRY main {
      p0 = f32[8,4,128,226]{3,2,1,0} parameter(0)
      f1 = f32[8,4,128]{2,1,0} fusion(p0), kind=kInput, calls=%fused_computation.1
      ROOT fusion = f32[8,4,128]{2,1,0} fusion(f1), kind=kLoop, calls=%fused_computation.2
    })";

  RunAndFilecheckHloRewrite(kHlo, std::move(priority_fusion_), R"(
CHECK: ROOT %fusion = f32[8,4,128]{2,1,0} fusion(%p{{.*}}), kind=kInput, calls=%fused_computation
  )");
}

}  // namespace gpu
}  // namespace xla
