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

#include "xla/backends/gpu/transforms/hoist_fusion_boundary_bitcasts.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

#include "absl/algorithm/container.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/xtile/xtile_config.pb.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;

class HoistFusionBoundaryBitcastsTest : public HloHardwareIndependentTestBase {
 protected:
  HoistFusionBoundaryBitcastsTest() {
    RegisterSymbolicExprStorage(&mlir_context_);
  }

  se::DeviceDescription device_info_{TestGpuDeviceInfo::RTXA6000DeviceInfo()};
  mlir::MLIRContext mlir_context_;
};

TEST_F(HoistFusionBoundaryBitcastsTest,
       HoistsParameterAndRootBitcastsOnSingleOutputFusion) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule m, is_scheduled=true

fused_comp {
  p0 = f32[128]{0} parameter(0)
  bc_in = f32[8,16]{1,0} bitcast(p0)
  neg = f32[8,16]{1,0} negate(bc_in)
  ROOT bc_out = f32[128]{0} bitcast(neg)
}

ENTRY main {
  p0 = f32[128]{0} parameter(0)
  ROOT fusion = f32[128]{0} fusion(p0), kind=kLoop, calls=fused_comp
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  ASSERT_TRUE(module->has_schedule());

  EXPECT_THAT(
      HoistFusionBoundaryBitcasts(
          device_info_, HloCostAnalysis::DefaultShapeSize, &mlir_context_)
          .Run(module.get()),
      IsOkAndHolds(true));
  EXPECT_THAT(module->schedule().Verify(), IsOk());

  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kBitcast);
  EXPECT_EQ(root->shape(), ShapeUtil::MakeShape(F32, {128}));

  const HloInstruction* fusion = root->operand(0);
  ASSERT_EQ(fusion->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(fusion->shape(), ShapeUtil::MakeShape(F32, {8, 16}));
  EXPECT_EQ(fusion->operand(0)->opcode(), HloOpcode::kBitcast);
  EXPECT_EQ(
      fusion->fused_instructions_computation()->root_instruction()->opcode(),
      HloOpcode::kNegate);
}

TEST_F(HoistFusionBoundaryBitcastsTest,
       ClearsIncompatibleShardingWhenHoistingChangesRank) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule m

fused_comp {
  p0 = f32[8,16]{1,0} parameter(0)
  neg = f32[8,16]{1,0} negate(p0)
  ROOT bc_out = f32[128]{0} bitcast(neg)
}

ENTRY main {
  p0 = f32[8,16]{1,0} parameter(0)
  ROOT fusion = f32[128]{0} fusion(p0), kind=kLoop, calls=fused_comp,
    sharding={devices=[2]<=[2]}
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  EXPECT_THAT(
      HoistFusionBoundaryBitcasts(
          device_info_, HloCostAnalysis::DefaultShapeSize, &mlir_context_)
          .Run(module.get()),
      IsOkAndHolds(true));

  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_FALSE(fusion->has_sharding());
}

TEST_F(HoistFusionBoundaryBitcastsTest,
       HoistsParameterAndRootBitcastsOnMultiOutputFusion) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule m

fused_comp {
  p0 = f32[128]{0} parameter(0)
  bc_in = f32[8,16]{1,0} bitcast(p0)
  neg = f32[8,16]{1,0} negate(bc_in)
  abs_val = f32[8,16]{1,0} abs(bc_in)
  bc_out0 = f32[128]{0} bitcast(neg)
  bc_out1 = f32[128]{0} bitcast(abs_val)
  ROOT out = (f32[128]{0}, f32[128]{0}) tuple(bc_out0, bc_out1)
}

ENTRY main {
  p0 = f32[128]{0} parameter(0)
  fusion = (f32[128]{0}, f32[128]{0}) fusion(p0), kind=kLoop, calls=fused_comp
  gte0 = f32[128]{0} get-tuple-element(fusion), index=0
  gte1 = f32[128]{0} get-tuple-element(fusion), index=1
  ROOT sum = f32[128]{0} add(gte0, gte1)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  EXPECT_THAT(
      HoistFusionBoundaryBitcasts(
          device_info_, HloCostAnalysis::DefaultShapeSize, &mlir_context_)
          .Run(module.get()),
      IsOkAndHolds(true));

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kBitcast);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kBitcast);

  const HloInstruction* gte0 = root->operand(0)->operand(0);
  EXPECT_EQ(gte0->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(gte0->shape(), ShapeUtil::MakeShape(F32, {8, 16}));

  const HloInstruction* fusion = gte0->operand(0);
  EXPECT_EQ(fusion->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(fusion->operand(0)->opcode(), HloOpcode::kBitcast);
}

TEST_F(HoistFusionBoundaryBitcastsTest,
       DoesNotHoistMultiOutputRootBitcastWhenResultingShapesDiffer) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule m

fused_comp {
  p0 = f32[8,16]{1,0} parameter(0)
  neg = f32[8,16]{1,0} negate(p0)
  bc_out0 = f32[128]{0} bitcast(neg)
  p1 = f32[128]{0} parameter(1)
  abs_val = f32[128]{0} abs(p1)
  ROOT out = (f32[128]{0}, f32[128]{0}) tuple(bc_out0, abs_val)
}

ENTRY main {
  p0 = f32[8,16]{1,0} parameter(0)
  p1 = f32[128]{0} parameter(1)
  fusion = (f32[128]{0}, f32[128]{0}) fusion(p0, p1), kind=kLoop,
    calls=fused_comp
  gte0 = f32[128]{0} get-tuple-element(fusion), index=0
  gte1 = f32[128]{0} get-tuple-element(fusion), index=1
  ROOT sum = f32[128]{0} add(gte0, gte1)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  EXPECT_THAT(
      HoistFusionBoundaryBitcasts(
          device_info_, HloCostAnalysis::DefaultShapeSize, &mlir_context_)
          .Run(module.get()),
      IsOkAndHolds(false));
}

TEST_F(HoistFusionBoundaryBitcastsTest,
       HoistsRootBitcastsOnNonSameShapeMultiOutputFusion) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule m

reducer {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

fused_comp {
  p0 = f32[8,16]{1,0} parameter(0)
  neg = f32[8,16]{1,0} negate(p0)
  bc_out0 = f32[128]{0} bitcast(neg)
  c0 = f32[] constant(0)
  red = f32[8]{0} reduce(neg, c0), dimensions={1}, to_apply=reducer
  bc_out1 = f32[2,4]{1,0} bitcast(red)
  ROOT out = (f32[128]{0}, f32[2,4]{1,0}) tuple(bc_out0, bc_out1)
}

ENTRY main {
  p0 = f32[8,16]{1,0} parameter(0)
  fusion = (f32[128]{0}, f32[2,4]{1,0}) fusion(p0), kind=kInput,
    calls=fused_comp
  gte0 = f32[128]{0} get-tuple-element(fusion), index=0
  gte1 = f32[2,4]{1,0} get-tuple-element(fusion), index=1
  ROOT out = (f32[128]{0}, f32[2,4]{1,0}) tuple(gte0, gte1)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  EXPECT_THAT(
      HoistFusionBoundaryBitcasts(
          device_info_, HloCostAnalysis::DefaultShapeSize, &mlir_context_)
          .Run(module.get()),
      IsOkAndHolds(true));

  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kBitcast);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kBitcast);

  const HloInstruction* gte0 = root->operand(0)->operand(0);
  const HloInstruction* gte1 = root->operand(1)->operand(0);
  EXPECT_EQ(gte0->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(gte0->shape(), ShapeUtil::MakeShape(F32, {8, 16}));
  EXPECT_EQ(gte1->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(gte1->shape(), ShapeUtil::MakeShape(F32, {8}));

  const HloInstruction* fusion = gte0->operand(0);
  ASSERT_EQ(fusion->opcode(), HloOpcode::kFusion);
  const HloInstruction* fused_root =
      fusion->fused_instructions_computation()->root_instruction();
  ASSERT_EQ(fused_root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(fused_root->operand(0)->opcode(), HloOpcode::kNegate);
  EXPECT_EQ(fused_root->operand(1)->opcode(), HloOpcode::kReduce);
}

TEST_F(HoistFusionBoundaryBitcastsTest,
       DoesNotHoistBitcastsWhenItWouldDuplicateMultiOutputRootOperands) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule m

fused_comp {
  p0 = f32[128]{0} parameter(0)
  bc0 = f32[8,16]{1,0} bitcast(p0)
  bc1 = f32[8,16]{1,0} bitcast(p0)
  ROOT out = (f32[8,16]{1,0}, f32[8,16]{1,0}) tuple(bc0, bc1)
}

ENTRY main {
  p0 = f32[128]{0} parameter(0)
  ROOT fusion = (f32[8,16]{1,0}, f32[8,16]{1,0}) fusion(p0), kind=kLoop,
    calls=fused_comp
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  EXPECT_THAT(
      HoistFusionBoundaryBitcasts(
          device_info_, HloCostAnalysis::DefaultShapeSize, &mlir_context_)
          .Run(module.get()),
      IsOkAndHolds(false));
}

TEST_F(HoistFusionBoundaryBitcastsTest,
       HoistsBitcastThatLeavesBareParameterRoot) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule m

fused_comp {
  p0 = f32[128]{0} parameter(0)
  ROOT bc = f32[8,16]{1,0} bitcast(p0)
}

ENTRY main {
  p0 = f32[128]{0} parameter(0)
  ROOT fusion = f32[8,16]{1,0} fusion(p0), kind=kLoop, calls=fused_comp
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  EXPECT_THAT(
      HoistFusionBoundaryBitcasts(
          device_info_, HloCostAnalysis::DefaultShapeSize, &mlir_context_)
          .Run(module.get()),
      IsOkAndHolds(true));

  const HloInstruction* fusion =
      module->entry_computation()->root_instruction();
  EXPECT_EQ(fusion->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(fusion->operand(0)->opcode(), HloOpcode::kBitcast);
  EXPECT_EQ(
      fusion->fused_instructions_computation()->root_instruction()->opcode(),
      HloOpcode::kParameter);
}

TEST_F(HoistFusionBoundaryBitcastsTest,
       HoistsMultiOutputRootBitcastLeavingBareParameterInTupleRoot) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule m

fused_comp {
  p0 = f32[8,16]{1,0} parameter(0)
  neg = f32[8,16]{1,0} negate(p0)
  bc_out0 = f32[128]{0} bitcast(neg)
  bc_out1 = f32[128]{0} bitcast(p0)
  ROOT out = (f32[128]{0}, f32[128]{0}) tuple(bc_out0, bc_out1)
}

ENTRY main {
  p0 = f32[8,16]{1,0} parameter(0)
  fusion = (f32[128]{0}, f32[128]{0}) fusion(p0), kind=kLoop, calls=fused_comp
  gte0 = f32[128]{0} get-tuple-element(fusion), index=0
  gte1 = f32[128]{0} get-tuple-element(fusion), index=1
  ROOT sum = f32[128]{0} add(gte0, gte1)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  EXPECT_THAT(
      HoistFusionBoundaryBitcasts(
          device_info_, HloCostAnalysis::DefaultShapeSize, &mlir_context_)
          .Run(module.get()),
      IsOkAndHolds(true));

  const HloInstruction* fusion =
      module->entry_computation()->GetInstructionWithName("fusion");
  ASSERT_NE(fusion, nullptr);
  const HloInstruction* fused_root =
      fusion->fused_instructions_computation()->root_instruction();
  ASSERT_EQ(fused_root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(fused_root->operand(0)->opcode(), HloOpcode::kNegate);
  EXPECT_EQ(fused_root->operand(1)->opcode(), HloOpcode::kParameter);
}

TEST_F(HoistFusionBoundaryBitcastsTest,
       RecalculatesOptimalTilingForModifiedTiledTritonFusion) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule m, is_scheduled=true

fused_transpose {
  p0 = f32[9011200]{0} parameter(0)
  bc = f32[5,1408,1280]{2,1,0} bitcast(p0)
  neg = f32[5,1408,1280]{2,1,0} negate(bc)
  t0 = f32[1280,5,1408]{2,1,0} transpose(neg), dimensions={2,0,1}
  abs_val = f32[5,1408,1280]{2,1,0} abs(bc)
  t1 = f32[1280,5,1408]{2,1,0} transpose(abs_val), dimensions={2,0,1}
  ROOT out = (f32[1280,5,1408]{2,1,0}, f32[1280,5,1408]{2,1,0}) tuple(t0, t1)
}

ENTRY main {
  p0 = f32[9011200]{0} parameter(0)
  fusion = (f32[1280,5,1408]{2,1,0}, f32[1280,5,1408]{2,1,0}) fusion(p0),
    kind=kCustom, calls=fused_transpose,
    backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"num_warps":"4","output_tiles":[{"sizes":["1","1","512"]},{"sizes":["1","1","512"]}],"num_ctas":1,"num_stages":1}}}
  gte0 = f32[1280,5,1408]{2,1,0} get-tuple-element(fusion), index=0
  gte1 = f32[1280,5,1408]{2,1,0} get-tuple-element(fusion), index=1
  ROOT sum = f32[1280,5,1408]{2,1,0} add(gte0, gte1)
})hlo";

  DebugOptions debug_options =
      HloHardwareIndependentTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_enable_same_shape_multi_output_fusion(
      true);

  HloModuleConfig config;
  config.set_debug_options(debug_options);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText, config));

  EXPECT_THAT(
      HoistFusionBoundaryBitcasts(
          device_info_, HloCostAnalysis::DefaultShapeSize, &mlir_context_)
          .Run(module.get()),
      IsOkAndHolds(true));
  EXPECT_THAT(module->schedule().Verify(), IsOk());

  const HloInstruction* fusion =
      module->entry_computation()->GetInstructionWithName("fusion");
  ASSERT_NE(fusion, nullptr);
  EXPECT_EQ(fusion->operand(0)->opcode(), HloOpcode::kBitcast);

  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  const xtile::BlockLevelFusionConfig& block_level_config =
      gpu_config.fusion_backend_config().block_level_fusion_config();
  ASSERT_GT(block_level_config.output_tiles_size(), 0);
  const auto& sizes = block_level_config.output_tiles(0).sizes();
  EXPECT_GE(absl::c_count_if(sizes, [](int64_t s) { return s > 1; }), 2);
}

TEST_F(HoistFusionBoundaryBitcastsTest,
       ReusesOuterBitcastAcrossSiblingFusions) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule m

fused_comp0 {
  p0 = f32[128]{0} parameter(0)
  bc = f32[8,16]{1,0} bitcast(p0)
  ROOT neg = f32[8,16]{1,0} negate(bc)
}

fused_comp1 {
  p0 = f32[128]{0} parameter(0)
  bc = f32[8,16]{1,0} bitcast(p0)
  ROOT abs_val = f32[8,16]{1,0} abs(bc)
}

ENTRY main {
  p0 = f32[128]{0} parameter(0)
  f0 = f32[8,16]{1,0} fusion(p0), kind=kLoop, calls=fused_comp0
  f1 = f32[8,16]{1,0} fusion(p0), kind=kLoop, calls=fused_comp1
  ROOT sum = f32[8,16]{1,0} add(f0, f1)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  EXPECT_THAT(
      HoistFusionBoundaryBitcasts(
          device_info_, HloCostAnalysis::DefaultShapeSize, &mlir_context_)
          .Run(module.get()),
      IsOkAndHolds(true));

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* f0 = root->operand(0);
  const HloInstruction* f1 = root->operand(1);
  EXPECT_EQ(f0->operand(0)->opcode(), HloOpcode::kBitcast);
  EXPECT_EQ(f0->operand(0), f1->operand(0));
}

TEST_F(HoistFusionBoundaryBitcastsTest, SkipsNonGenericCustomFusion) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule m

fused_comp {
  p0 = f32[128]{0} parameter(0)
  bc_in = f32[8,16]{1,0} bitcast(p0)
  p1 = f32[16,4]{1,0} parameter(1)
  dot = f32[8,4]{1,0} dot(bc_in, p1), lhs_contracting_dims={1},
    rhs_contracting_dims={0}
  ROOT bc_out = f32[32]{0} bitcast(dot)
}

ENTRY main {
  p0 = f32[128]{0} parameter(0)
  p1 = f32[16,4]{1,0} parameter(1)
  ROOT fusion = f32[32]{0} fusion(p0, p1), kind=kCustom, calls=fused_comp,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  EXPECT_THAT(
      HoistFusionBoundaryBitcasts(
          device_info_, HloCostAnalysis::DefaultShapeSize, &mlir_context_)
          .Run(module.get()),
      IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla::gpu
