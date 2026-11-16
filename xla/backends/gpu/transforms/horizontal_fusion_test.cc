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

#include "xla/backends/gpu/transforms/horizontal_fusion.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;

class HorizontalFusionTest : public HloHardwareIndependentTestBase {
 protected:
  const se::DeviceDescription device_info_{
      TestGpuDeviceInfo::RTXA6000DeviceInfo()};
};

TEST_F(HorizontalFusionTest, FusesConsecutiveIndependentLoopFusions) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_add {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  ROOT add = f32[1024]{0} add(p0, p1)
}

fused_mul {
  p0 = f32[512]{0} parameter(0)
  ROOT mul = f32[512]{0} multiply(p0, p0)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  p2 = f32[512]{0} parameter(2)
  f0 = f32[1024]{0} fusion(p0, p1), kind=kLoop, calls=fused_add
  f1 = f32[512]{0} fusion(p2), kind=kLoop, calls=fused_mul
  ROOT out = (f32[1024]{0}, f32[512]{0}) tuple(f0, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: horizontal_fusion_computation
// CHECK-DAG: %[[P0:.*]] = f32[1024]{0} parameter(0)
// CHECK-DAG: %[[P1:.*]] = f32[1024]{0} parameter(1)
// CHECK-DAG: %[[P2:.*]] = f32[512]{0} parameter(2)
// CHECK-DAG: %[[NESTED0:.*]] = f32[1024]{0} fusion(%[[P0]], %[[P1]]), kind=kLoop, calls=%fused_add
// CHECK-DAG: %[[NESTED1:.*]] = f32[512]{0} fusion(%[[P2]]), kind=kLoop, calls=%fused_mul
// CHECK: ROOT %{{.*}} = (f32[1024]{0}, f32[512]{0}) tuple(%[[NESTED0]], %[[NESTED1]])
// CHECK: ENTRY %main
// CHECK: %[[HF:.*]] = (f32[1024]{0}, f32[512]{0}) fusion({{.*}}), kind=kCustom, calls=%horizontal_fusion_computation, backend_config={{.*}}"kind":"__thread_block_specialized"
// CHECK: ROOT %{{.*}} = (f32[1024]{0}, f32[512]{0}) tuple(%[[HF]]#0, %[[HF]]#1)
)"));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(HorizontalFusionTest, DeduplicatesSharedExternalOperands) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_neg {
  p0 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p0)
}

fused_exp {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  f0 = f32[1024]{0} fusion(p0), kind=kLoop, calls=fused_neg
  f1 = f32[1024]{0} fusion(p0), kind=kLoop, calls=fused_exp
  ROOT out = (f32[1024]{0}, f32[1024]{0}) tuple(f0, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kTuple);
  const HloInstruction* gte0 = root->operand(0);
  ASSERT_EQ(gte0->opcode(), HloOpcode::kGetTupleElement);
  const HloInstruction* outer_fusion = gte0->operand(0);
  ASSERT_EQ(outer_fusion->opcode(), HloOpcode::kFusion);
  // Shared operand `p0` should be deduplicated to a single parameter.
  EXPECT_EQ(outer_fusion->operand_count(), 1);
}

TEST_F(HorizontalFusionTest, SkipsInterveningIndependentBitcast) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_neg {
  p0 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p0)
}

fused_exp {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[1024]{0} fusion(p0), kind=kLoop, calls=fused_neg
  bc = f32[32,32]{1,0} bitcast(f0)
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_exp
  ROOT out = (f32[32,32]{1,0}, f32[1024]{0}) tuple(bc, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(HorizontalFusionTest, DoesNotFuseDependentFusionsThroughBitcast) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_neg {
  p0 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p0)
}

fused_exp {
  p0 = f32[32,32]{1,0} parameter(0)
  ROOT exp = f32[32,32]{1,0} exponential(p0)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  f0 = f32[1024]{0} fusion(p0), kind=kLoop, calls=fused_neg
  bc = f32[32,32]{1,0} bitcast(f0)
  ROOT f1 = f32[32,32]{1,0} fusion(bc), kind=kLoop, calls=fused_exp
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest, DoesNotFuseControlDependentFusions) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_neg {
  p0 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p0)
}

fused_exp {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[1024]{0} fusion(p0), kind=kLoop, calls=fused_neg
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_exp, control-predecessors={f0}
  ROOT out = (f32[1024]{0}, f32[1024]{0}) tuple(f0, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest, FlushesBatchAtNonSkippableInstruction) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_neg {
  p0 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p0)
}

fused_exp {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[1024]{0} fusion(p0), kind=kLoop, calls=fused_neg
  cc = f32[1024]{0} custom-call(p0), custom_call_target="foo"
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_exp
  ROOT out = (f32[1024]{0}, f32[1024]{0}, f32[1024]{0}) tuple(f0, cc, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest, SkipsLargeGridFusionsExceedingSmThreshold) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_large {
  p0 = f32[16777216]{0} parameter(0)
  ROOT neg = f32[16777216]{0} negate(p0)
}

fused_small {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[16777216]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[16777216]{0} fusion(p0), kind=kLoop, calls=fused_large
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_small
  ROOT out = (f32[16777216]{0}, f32[1024]{0}) tuple(f0, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest, FusesLoopAndReductionWithWarpAlignedThreads) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

fused_loop {
  p0 = f32[512]{0} parameter(0)
  ROOT neg = f32[512]{0} negate(p0)
}

fused_reduce {
  p1 = f32[8,512]{1,0} parameter(0)
  c0 = f32[] constant(0)
  ROOT red = f32[8]{0} reduce(p1, c0), dimensions={1}, to_apply=add_f32
}

ENTRY main {
  p0 = f32[512]{0} parameter(0)
  p1 = f32[8,512]{1,0} parameter(1)
  f0 = f32[512]{0} fusion(p0), kind=kLoop, calls=fused_loop
  f1 = f32[8]{0} fusion(p1), kind=kInput, calls=fused_reduce
  ROOT out = (f32[512]{0}, f32[8]{0}) tuple(f0, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(HorizontalFusionTest, FusesMultiOutputNestedFusion) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_mof {
  p0 = f32[1024]{0} parameter(0)
  n0 = f32[1024]{0} negate(p0)
  e0 = f32[1024]{0} exponential(p0)
  ROOT t = (f32[1024]{0}, f32[1024]{0}) tuple(n0, e0)
}

fused_single {
  p1 = f32[512]{0} parameter(0)
  ROOT c0 = f32[512]{0} cosine(p1)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  f0 = (f32[1024]{0}, f32[1024]{0}) fusion(p0), kind=kLoop, calls=fused_mof
  gte0 = f32[1024]{0} get-tuple-element(f0), index=0
  gte1 = f32[1024]{0} get-tuple-element(f0), index=1
  f1 = f32[512]{0} fusion(p1), kind=kLoop, calls=fused_single
  ROOT out = (f32[1024]{0}, f32[1024]{0}, f32[512]{0}) tuple(gte0, gte1, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: horizontal_fusion_computation
// CHECK-DAG: %[[MOF:.*]] = (f32[1024]{0}, f32[1024]{0}) fusion({{.*}}), kind=kLoop, calls=%fused_mof
// CHECK-DAG: %[[SINGLE:.*]] = f32[512]{0} fusion({{.*}}), kind=kLoop, calls=%fused_single
// CHECK: ROOT %{{.*}} = (f32[1024]{0}, f32[1024]{0}, f32[512]{0}) tuple(%[[MOF]]#0, %[[MOF]]#1, %[[SINGLE]])
)"));
  EXPECT_TRUE(filecheck_matched) << module->ToString();
}

TEST_F(HorizontalFusionTest,
       FusesMultiOutputNestedFusionWithControlDependentGte) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_mof {
  p0 = f32[1024]{0} parameter(0)
  n0 = f32[1024]{0} negate(p0)
  e0 = f32[1024]{0} exponential(p0)
  ROOT t = (f32[1024]{0}, f32[1024]{0}) tuple(n0, e0)
}

fused_single {
  p1 = f32[512]{0} parameter(0)
  ROOT c0 = f32[512]{0} cosine(p1)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  f0 = (f32[1024]{0}, f32[1024]{0}) fusion(p0), kind=kLoop, calls=fused_mof
  f1 = f32[512]{0} fusion(p1), kind=kLoop, calls=fused_single
  gte0 = f32[1024]{0} get-tuple-element(f0), index=0
  gte1 = f32[1024]{0} get-tuple-element(f0), index=1, control-predecessors={f1}
  ROOT out = (f32[1024]{0}, f32[1024]{0}, f32[512]{0}) tuple(gte0, gte1, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* gte1 = root->operand(1);
  ASSERT_EQ(gte1->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_THAT(gte1->control_predecessors(),
              ::testing::ElementsAre(gte1->operand(0)));
}

TEST_F(HorizontalFusionTest, FusesMultiOutputNestedFusionUsedAsWholeTuple) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

fused_mof {
  p0 = f32[1024]{0} parameter(0)
  n0 = f32[1024]{0} negate(p0)
  e0 = f32[1024]{0} exponential(p0)
  ROOT t = (f32[1024]{0}, f32[1024]{0}) tuple(n0, e0)
}

fused_single {
  p1 = f32[512]{0} parameter(0)
  ROOT c0 = f32[512]{0} cosine(p1)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  f0 = (f32[1024]{0}, f32[1024]{0}) fusion(p0), kind=kLoop, calls=fused_mof
  gte1 = f32[1024]{0} get-tuple-element(f0), index=1
  f1 = f32[512]{0} fusion(p1), kind=kLoop, calls=fused_single
  ROOT out = ((f32[1024]{0}, f32[1024]{0}), f32[1024]{0}, f32[512]{0}) tuple(f0, gte1, f1)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: ENTRY %main
// CHECK: %[[HF:.*]] = (f32[1024]{0}, f32[1024]{0}, f32[512]{0}) fusion({{.*}}), kind=kCustom, calls=%horizontal_fusion_computation
// CHECK: %[[TUPLE:.*]] = (f32[1024]{0}, f32[1024]{0}) tuple(%[[HF]]#0, %[[HF]]#1)
// CHECK: ROOT %{{.*}} = ((f32[1024]{0}, f32[1024]{0}), f32[1024]{0}, f32[512]{0}) tuple(%[[TUPLE]], %[[HF]]#1, %[[HF]]#2)
)"));
  EXPECT_TRUE(filecheck_matched) << module->ToString();
}

TEST_F(HorizontalFusionTest, SkipsNestedTupleOutputFusions) {
  // `f0` returns a nested tuple because one of its roots is a variadic reduce.
  // It must stay outside the horizontal fusion, while `f1` and `f2` fuse.
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

argmax {
  lhs_value = f32[] parameter(0)
  lhs_index = s32[] parameter(1)
  rhs_value = f32[] parameter(2)
  rhs_index = s32[] parameter(3)
  ge = pred[] compare(lhs_value, rhs_value), direction=GE
  value = f32[] select(ge, lhs_value, rhs_value)
  index = s32[] select(ge, lhs_index, rhs_index)
  ROOT t = (f32[], s32[]) tuple(value, index)
}

fused_nested_tuple {
  p0 = f32[16,256]{1,0} parameter(0)
  idx = s32[16,256]{1,0} iota(), iota_dimension=1
  init_value = f32[] constant(-inf)
  init_index = s32[] constant(0)
  red = (f32[16]{0}, s32[16]{0}) reduce(p0, idx, init_value, init_index), dimensions={1}, to_apply=argmax
  neg = f32[16,256]{1,0} negate(p0)
  ROOT t = ((f32[16]{0}, s32[16]{0}), f32[16,256]{1,0}) tuple(red, neg)
}

fused_cos {
  p0 = f32[512]{0} parameter(0)
  ROOT c0 = f32[512]{0} cosine(p0)
}

fused_exp {
  p0 = f32[512]{0} parameter(0)
  ROOT e0 = f32[512]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[16,256]{1,0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  f0 = ((f32[16]{0}, s32[16]{0}), f32[16,256]{1,0}) fusion(p0), kind=kInput, calls=fused_nested_tuple
  gte0 = (f32[16]{0}, s32[16]{0}) get-tuple-element(f0), index=0
  gte1 = f32[16,256]{1,0} get-tuple-element(f0), index=1
  f1 = f32[512]{0} fusion(p1), kind=kLoop, calls=fused_cos
  f2 = f32[512]{0} fusion(p1), kind=kLoop, calls=fused_exp
  ROOT out = ((f32[16]{0}, s32[16]{0}), f32[16,256]{1,0}, f32[512]{0}, f32[512]{0}) tuple(gte0, gte1, f1, f2)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: ENTRY %main
// CHECK-DAG: %[[F0:.*]] = ((f32[16]{0}, s32[16]{0}), f32[16,256]{1,0}) fusion({{.*}}), kind=kInput, calls=%fused_nested_tuple
// CHECK-DAG: %[[HF:.*]] = (f32[512]{0}, f32[512]{0}) fusion({{.*}}), kind=kCustom, calls=%horizontal_fusion_computation
// CHECK: ROOT %{{.*}} = ((f32[16]{0}, s32[16]{0}), f32[16,256]{1,0}, f32[512]{0}, f32[512]{0}) tuple(%[[F0]]#0, %[[F0]]#1, %[[HF]]#0, %[[HF]]#1)
)"));
  EXPECT_TRUE(filecheck_matched) << module->ToString();
}

TEST_F(HorizontalFusionTest, FusesNestedTritonAndLoopFusions) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule triton_plus_loop, is_scheduled=true

add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT res = f32[] add(lhs, rhs)
}

triton_norm {
  p0 = f32[8,512]{1,0} parameter(0)
  sq = f32[8,512]{1,0} multiply(p0, p0)
  c0 = f32[] constant(0)
  red = f32[8]{0} reduce(sq, c0), dimensions={1}, to_apply=add_f32
  bcast = f32[8,512]{1,0} broadcast(red), dimensions={0}
  ROOT out = f32[8,512]{1,0} multiply(p0, bcast)
}

fused_loop {
  p1 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p1)
}

ENTRY main {
  p0 = f32[8,512]{1,0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[8,512]{1,0} fusion(p0), kind=kCustom, calls=triton_norm,
    backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["1","512"]}],"num_warps":"4","num_ctas":1,"num_stages":1}}}
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_loop
  ROOT out = (f32[8,512]{1,0}, f32[1024]{0}) tuple(f0, f1)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: %horizontal_fusion_computation
// CHECK-DAG: %[[TRITON:.*]] = f32[8,512]{1,0} fusion({{.*}}), kind=kCustom, calls=%triton_norm
// CHECK-DAG: %[[LOOP:.*]] = f32[1024]{0} fusion({{.*}}), kind=kLoop, calls=%fused_loop
// CHECK: ROOT %{{.*}} = (f32[8,512]{1,0}, f32[1024]{0}) tuple(%[[TRITON]], %[[LOOP]])
)"));
  EXPECT_TRUE(filecheck_matched) << module->ToString();
}

TEST_F(HorizontalFusionTest, SkipsFusionsExceedingMaxLeafExecTime) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule heavy_reduce, is_scheduled=true

add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT res = f32[] add(lhs, rhs)
}

fused_heavy_reduce {
  p0 = f32[32,262144]{1,0} parameter(0)
  c0 = f32[] constant(0)
  ROOT red = f32[32]{0} reduce(p0, c0), dimensions={1}, to_apply=add_f32
}

fused_small_loop {
  p1 = f32[512]{0} parameter(0)
  ROOT neg = f32[512]{0} negate(p1)
}

ENTRY main {
  p0 = f32[32,262144]{1,0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  f0 = f32[32]{0} fusion(p0), kind=kCustom, calls=fused_heavy_reduce,
    backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],"sequential_tile_sizes":["4096"],"num_warps":"8","num_ctas":1,"num_stages":1}}}
  f1 = f32[512]{0} fusion(p1), kind=kLoop, calls=fused_small_loop
  ROOT out = (f32[32]{0}, f32[512]{0}) tuple(f0, f1)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest, FusesHeavyLeafWhenMaxLeafExecTimeDisabled) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule heavy_reduce_unlimited_time, is_scheduled=true

add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT res = f32[] add(lhs, rhs)
}

fused_heavy_reduce {
  p0 = f32[32,262144]{1,0} parameter(0)
  c0 = f32[] constant(0)
  ROOT red = f32[32]{0} reduce(p0, c0), dimensions={1}, to_apply=add_f32
}

fused_small_loop {
  p1 = f32[512]{0} parameter(0)
  ROOT neg = f32[512]{0} negate(p1)
}

ENTRY main {
  p0 = f32[32,262144]{1,0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  f0 = f32[32]{0} fusion(p0), kind=kCustom, calls=fused_heavy_reduce,
    backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],"sequential_tile_sizes":["4096"],"num_warps":"8","num_ctas":1,"num_stages":1}}}
  f1 = f32[512]{0} fusion(p1), kind=kLoop, calls=fused_small_loop
  ROOT out = (f32[32]{0}, f32[512]{0}) tuple(f0, f1)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion::Options options;
  options.max_leaf_exec_time = absl::ZeroDuration();
  HorizontalFusion pass(device_info_, options);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(HorizontalFusionTest, SkipsStreamAnnotatedFusion) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule stream_annotated_fusion, is_scheduled=true

fused_neg {
  p0 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p0)
}

fused_exp {
  p1 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p1)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[1024]{0} fusion(p0), kind=kLoop, calls=fused_neg,
    frontend_attributes={_xla_stream_annotation="1"}
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_exp
  ROOT out = (f32[1024]{0}, f32[1024]{0}) tuple(f0, f1)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest, SkipsNonZeroOperationQueueIdFusion) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule non_zero_queue_id_fusion, is_scheduled=true

fused_cos {
  p0 = f32[1024]{0} parameter(0)
  ROOT cos = f32[1024]{0} cosine(p0)
}

fused_exp {
  p1 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p1)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[1024]{0} fusion(p0), kind=kLoop, calls=fused_cos,
    backend_config={"operation_queue_id":"1"}
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_exp
  ROOT out = (f32[1024]{0}, f32[1024]{0}) tuple(f0, f1)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest, SkipsDynamicShapeFusions) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule dynamic_shape, is_scheduled=true

fused_dyn {
  p0 = f32[<=1024]{0} parameter(0)
  ROOT neg = f32[<=1024]{0} negate(p0)
}

fused_static {
  p1 = f32[512]{0} parameter(0)
  ROOT exp = f32[512]{0} exponential(p1)
}

ENTRY main {
  p0 = f32[<=1024]{0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  f0 = f32[<=1024]{0} fusion(p0), kind=kLoop, calls=fused_dyn
  f1 = f32[512]{0} fusion(p1), kind=kLoop, calls=fused_static
  ROOT out = (f32[<=1024]{0}, f32[512]{0}) tuple(f0, f1)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest, SkipsTupleOperandFusions) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule tuple_operand_fusion, is_scheduled=true

fused_tuple_param {
  p0 = (f32[16]{0}, f32[16]{0}) parameter(0)
  lhs = f32[16]{0} get-tuple-element(p0), index=0
  rhs = f32[16]{0} get-tuple-element(p0), index=1
  ROOT sum = f32[16]{0} add(lhs, rhs)
}

fused_neg {
  p1 = f32[16]{0} parameter(0)
  ROOT neg = f32[16]{0} negate(p1)
}

ENTRY main {
  p0 = (f32[16]{0}, f32[16]{0}) parameter(0)
  p1 = f32[16]{0} parameter(1)
  f0 = f32[16]{0} fusion(p0), kind=kLoop, calls=fused_tuple_param
  f1 = f32[16]{0} fusion(p1), kind=kLoop, calls=fused_neg
  ROOT out = (f32[16]{0}, f32[16]{0}) tuple(f0, f1)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest, SkipsPreAmpereDevices) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule pre_ampere, is_scheduled=true

fused_neg {
  p0 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p0)
}

fused_exp {
  p1 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p1)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[1024]{0} fusion(p0), kind=kLoop, calls=fused_neg
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_exp
  ROOT out = (f32[1024]{0}, f32[1024]{0}) tuple(f0, f1)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  const se::DeviceDescription turing_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(
          se::GpuComputeCapability{se::CudaComputeCapability(7, 5)});
  HorizontalFusion turing_pass(turing_info);
  EXPECT_THAT(turing_pass.Run(module.get()), IsOkAndHolds(false));
}

struct UnsupportedNestedTritonCase {
  absl::string_view test_name;
  absl::string_view hlo_text;
};

class UnsupportedNestedTritonFusionTest
    : public HorizontalFusionTest,
      public testing::WithParamInterface<UnsupportedNestedTritonCase> {};

TEST_P(UnsupportedNestedTritonFusionTest, SkipsUnsupportedTritonLeaf) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(GetParam().hlo_text));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

INSTANTIATE_TEST_SUITE_P(
    UnsupportedNestedTritonFusions, UnsupportedNestedTritonFusionTest,
    testing::Values(UnsupportedNestedTritonCase{"Dot",
                                                R"hlo(
HloModule triton_dot_plus_loop, is_scheduled=true

triton_dot {
  p0 = f32[16,16]{1,0} parameter(0)
  p1 = f32[16,16]{1,0} parameter(1)
  ROOT dot = f32[16,16]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

fused_loop {
  p0 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p0)
}

ENTRY main {
  p0 = f32[16,16]{1,0} parameter(0)
  p1 = f32[16,16]{1,0} parameter(1)
  p2 = f32[1024]{0} parameter(2)
  f0 = f32[16,16]{1,0} fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["16","16"]}],"num_warps":"4","num_ctas":1,"num_stages":1}}}
  f1 = f32[1024]{0} fusion(p2), kind=kLoop, calls=fused_loop
  ROOT out = (f32[16,16]{1,0}, f32[1024]{0}) tuple(f0, f1)
})hlo"},
                    UnsupportedNestedTritonCase{"ScaledDot",
                                                R"hlo(
HloModule triton_scaled_dot_plus_loop, is_scheduled=true

triton_scaled_dot {
  lhs = f8e4m3fn[16,32]{1,0} parameter(0)
  lhs_scale = f8e8m0fnu[16,1]{1,0} parameter(1)
  rhs = f8e4m3fn[32,16]{1,0} parameter(2)
  rhs_scale = f8e8m0fnu[1,16]{1,0} parameter(3)
  ROOT dot = f32[16,16]{1,0} scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

fused_loop {
  p0 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p0)
}

ENTRY main {
  lhs = f8e4m3fn[16,32]{1,0} parameter(0)
  lhs_scale = f8e8m0fnu[16,1]{1,0} parameter(1)
  rhs = f8e4m3fn[32,16]{1,0} parameter(2)
  rhs_scale = f8e8m0fnu[1,16]{1,0} parameter(3)
  p_loop = f32[1024]{0} parameter(4)
  f0 = f32[16,16]{1,0} fusion(lhs, lhs_scale, rhs, rhs_scale), kind=kCustom, calls=triton_scaled_dot,
    backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["16","16"]}],"num_warps":"4","num_ctas":1,"num_stages":1}}}
  f1 = f32[1024]{0} fusion(p_loop), kind=kLoop, calls=fused_loop
  ROOT out = (f32[16,16]{1,0}, f32[1024]{0}) tuple(f0, f1)
})hlo"},
                    UnsupportedNestedTritonCase{"Transpose",
                                                R"hlo(
HloModule triton_transpose_plus_loop, is_scheduled=true

triton_transpose {
  p0 = f32[16,32]{1,0} parameter(0)
  ROOT t = f32[32,16]{1,0} transpose(p0), dimensions={1,0}
}

fused_loop {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[16,32]{1,0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[32,16]{1,0} fusion(p0), kind=kCustom, calls=triton_transpose,
    backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["16","16"]}],"num_warps":"4","num_ctas":1,"num_stages":1}}}
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_loop
  ROOT out = (f32[32,16]{1,0}, f32[1024]{0}) tuple(f0, f1)
})hlo"},
                    UnsupportedNestedTritonCase{"NonGenericTritonKind",
                                                R"hlo(
HloModule triton_non_generic_plus_loop, is_scheduled=true

triton_non_generic {
  p0 = f32[8,512]{1,0} parameter(0)
  ROOT neg = f32[8,512]{1,0} negate(p0)
}

fused_loop {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[8,512]{1,0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[8,512]{1,0} fusion(p0), kind=kCustom, calls=triton_non_generic,
    backend_config={"fusion_backend_config":{"kind":"__triton_nested_gemm_fusion","block_level_fusion_config":{"output_tiles":[{"sizes":["1","512"]}],"num_warps":"4","num_ctas":1,"num_stages":1}}}
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_loop
  ROOT out = (f32[8,512]{1,0}, f32[1024]{0}) tuple(f0, f1)
})hlo"},
                    UnsupportedNestedTritonCase{"MissingBlockLevelConfig",
                                                R"hlo(
HloModule triton_missing_cfg_plus_loop, is_scheduled=true

triton_missing_cfg {
  p0 = f32[8,512]{1,0} parameter(0)
  ROOT neg = f32[8,512]{1,0} negate(p0)
}

fused_loop {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[8,512]{1,0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[8,512]{1,0} fusion(p0), kind=kCustom, calls=triton_missing_cfg,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_loop
  ROOT out = (f32[8,512]{1,0}, f32[1024]{0}) tuple(f0, f1)
})hlo"},
                    UnsupportedNestedTritonCase{"MultiCta",
                                                R"hlo(
HloModule triton_multi_cta_plus_loop, is_scheduled=true

triton_multi_cta {
  p0 = f32[8,512]{1,0} parameter(0)
  ROOT neg = f32[8,512]{1,0} negate(p0)
}

fused_loop {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[8,512]{1,0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[8,512]{1,0} fusion(p0), kind=kCustom, calls=triton_multi_cta,
    backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["1","512"]}],"num_warps":"4","num_ctas":2,"num_stages":1}}}
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_loop
  ROOT out = (f32[8,512]{1,0}, f32[1024]{0}) tuple(f0, f1)
})hlo"},
                    UnsupportedNestedTritonCase{"TmaAllowed",
                                                R"hlo(
HloModule triton_tma_plus_loop, is_scheduled=true

triton_tma {
  p0 = f32[8,512]{1,0} parameter(0)
  ROOT neg = f32[8,512]{1,0} negate(p0)
}

fused_loop {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[8,512]{1,0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[8,512]{1,0} fusion(p0), kind=kCustom, calls=triton_tma,
    backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["1","512"]}],"num_warps":"4","num_ctas":1,"num_stages":1,"is_tma_allowed":true}}}
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_loop
  ROOT out = (f32[8,512]{1,0}, f32[1024]{0}) tuple(f0, f1)
})hlo"},
                    UnsupportedNestedTritonCase{"WarpSpecializationAllowed",
                                                R"hlo(
HloModule triton_warp_spec_plus_loop, is_scheduled=true

triton_warp_spec {
  p0 = f32[8,512]{1,0} parameter(0)
  ROOT neg = f32[8,512]{1,0} negate(p0)
}

fused_loop {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[8,512]{1,0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[8,512]{1,0} fusion(p0), kind=kCustom, calls=triton_warp_spec,
    backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["1","512"]}],"num_warps":"4","num_ctas":1,"num_stages":1,"is_warp_specialization_allowed":true}}}
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_loop
  ROOT out = (f32[8,512]{1,0}, f32[1024]{0}) tuple(f0, f1)
})hlo"},
                    UnsupportedNestedTritonCase{"ExceedsThreadsPerBlockLimit",
                                                R"hlo(
HloModule triton_too_many_warps_plus_loop, is_scheduled=true

triton_too_many_warps {
  p0 = f32[8,512]{1,0} parameter(0)
  ROOT neg = f32[8,512]{1,0} negate(p0)
}

fused_loop {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[8,512]{1,0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[8,512]{1,0} fusion(p0), kind=kCustom, calls=triton_too_many_warps,
    backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["1","512"]}],"num_warps":"64","num_ctas":1,"num_stages":1}}}
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_loop
  ROOT out = (f32[8,512]{1,0}, f32[1024]{0}) tuple(f0, f1)
})hlo"},
                    UnsupportedNestedTritonCase{"NonPositiveNumStages",
                                                R"hlo(
HloModule triton_zero_stages_plus_loop, is_scheduled=true

triton_zero_stages {
  p0 = f32[8,512]{1,0} parameter(0)
  ROOT neg = f32[8,512]{1,0} negate(p0)
}

fused_loop {
  p0 = f32[1024]{0} parameter(0)
  ROOT exp = f32[1024]{0} exponential(p0)
}

ENTRY main {
  p0 = f32[8,512]{1,0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  f0 = f32[8,512]{1,0} fusion(p0), kind=kCustom, calls=triton_zero_stages,
    backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["1","512"]}],"num_warps":"4","num_ctas":1,"num_stages":0}}}
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_loop
  ROOT out = (f32[8,512]{1,0}, f32[1024]{0}) tuple(f0, f1)
})hlo"}),
    [](const testing::TestParamInfo<UnsupportedNestedTritonCase>& info) {
      return std::string(info.param.test_name);
    });

TEST_F(HorizontalFusionTest, FusesLargeBatchLoopQkvReductionsOnBlackwell) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule qkv_loop_reductions, is_scheduled=true

add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

fused_q {
  p0 = f32[3,128,4096]{2,1,0} parameter(0)
  c0 = f32[] constant(0)
  red = f32[128,4096]{1,0} reduce(p0, c0), dimensions={0}, to_apply=add_f32
  ROOT out = bf16[128,4096]{1,0} convert(red)
}

fused_kv {
  p1 = f32[3,128,2048]{2,1,0} parameter(0)
  c0 = f32[] constant(0)
  s_k = f32[3,128,1024]{2,1,0} slice(p1), slice={[0:3], [0:128], [0:1024]}
  s_v = f32[3,128,1024]{2,1,0} slice(p1), slice={[0:3], [0:128], [1024:2048]}
  red_k = f32[128,1024]{1,0} reduce(s_k, c0), dimensions={0}, to_apply=add_f32
  red_v = f32[128,1024]{1,0} reduce(s_v, c0), dimensions={0}, to_apply=add_f32
  k = bf16[128,1024]{1,0} convert(red_k)
  v = bf16[128,1024]{1,0} convert(red_v)
  ROOT out = (bf16[128,1024]{1,0}, bf16[128,1024]{1,0}) tuple(k, v)
}

triton_norm {
  p2 = f32[8,512]{1,0} parameter(0)
  ROOT neg = f32[8,512]{1,0} negate(p2)
}

ENTRY main {
  p0 = f32[3,128,4096]{2,1,0} parameter(0)
  p1 = f32[3,128,2048]{2,1,0} parameter(1)
  p2 = f32[8,512]{1,0} parameter(2)
  f_q = bf16[128,4096]{1,0} fusion(p0), kind=kLoop, calls=fused_q
  f_kv = (bf16[128,1024]{1,0}, bf16[128,1024]{1,0}) fusion(p1), kind=kLoop, calls=fused_kv
  gte_k = bf16[128,1024]{1,0} get-tuple-element(f_kv), index=0
  gte_v = bf16[128,1024]{1,0} get-tuple-element(f_kv), index=1
  f_triton = f32[8,512]{1,0} fusion(p2), kind=kCustom, calls=triton_norm,
    backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["1","512"]}],"num_warps":"4","num_ctas":1,"num_stages":1}}}
  ROOT out = (bf16[128,4096]{1,0}, bf16[128,1024]{1,0}, bf16[128,1024]{1,0}, f32[8,512]{1,0}) tuple(f_q, gte_k, gte_v, f_triton)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion pass(TestGpuDeviceInfo::B200SXMDeviceInfo());
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  // `f_q` (131,072 threads) and `f_kv` (65,536 threads) fit in a single wave
  // as pure `kLoop` fusions (196,608 <= 148 * 2048 = 303,104), while `f_triton`
  // remains separate because adding a `kCustom` fusion tightens the budget to
  // 148 * 1024 = 151,552 threads.
  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: %horizontal_fusion_computation
// CHECK-DAG: bf16[128,4096]{1,0} fusion({{.*}}), kind=kLoop, calls=%fused_q
// CHECK-DAG: (bf16[128,1024]{1,0}, bf16[128,1024]{1,0}) fusion({{.*}}), kind=kLoop, calls=%fused_kv
// CHECK: ENTRY %main
// CHECK-DAG: fusion({{.*}}), kind=kCustom, calls=%horizontal_fusion_computation
// CHECK-DAG: f32[8,512]{1,0} fusion({{.*}}), kind=kCustom, calls=%triton_norm
)"));
  EXPECT_TRUE(filecheck_matched) << module->ToString();
}

TEST_F(HorizontalFusionTest,
       FusesTinyIsolatedFusionsAcrossIndependentCustomCall) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule tiny_cross_barrier, is_scheduled=true

fused_s0 {
  p0 = s32[1]{0} parameter(0)
  const1 = s32[1]{0} constant({1})
  ROOT add = s32[1]{0} add(p0, const1)
}

fused_s1 {
  p1 = u32[4]{0} parameter(0)
  slice0 = u32[2]{0} slice(p1), slice={[0:2]}
  slice1 = u32[2]{0} slice(p1), slice={[2:4]}
  ROOT out = (u32[2]{0}, u32[2]{0}) tuple(slice0, slice1)
}

ENTRY main {
  p0 = s32[1]{0} parameter(0)
  p_heavy = f32[4096]{0} parameter(1)
  f0 = s32[1]{0} fusion(p0), kind=kLoop, calls=fused_s0
  cc = f32[4096]{0} custom-call(p_heavy, f0), custom_call_target="foo"
  p1 = u32[2,2]{1,0} parameter(2)
  bc1 = u32[4]{0} bitcast(p1)
  f1 = (u32[2]{0}, u32[2]{0}) fusion(bc1), kind=kLoop, calls=fused_s1
  gte0 = u32[2]{0} get-tuple-element(f1), index=0
  gte1 = u32[2]{0} get-tuple-element(f1), index=1
  ROOT out = (f32[4096]{0}, u32[2]{0}, u32[2]{0}) tuple(cc, gte0, gte1)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: %horizontal_fusion_computation
// CHECK-DAG: s32[1]{0} fusion({{.*}}), kind=kLoop, calls=%fused_s0
// CHECK-DAG: (u32[2]{0}, u32[2]{0}) fusion({{.*}}), kind=kLoop, calls=%fused_s1
// CHECK: ENTRY %main
// CHECK: %[[HF:.*]] = (s32[1]{0}, u32[2]{0}, u32[2]{0}) fusion({{.*}}), kind=kCustom, calls=%horizontal_fusion_computation
// CHECK: custom-call({{.*}}, %[[HF]]#0)
)"));
  EXPECT_TRUE(filecheck_matched) << module->ToString();
}

TEST_F(HorizontalFusionTest, DoesNotHoistTinyFusionDependentOnBarrier) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule tiny_dependent_on_barrier, is_scheduled=true

fused_s0 {
  p0 = s32[1]{0} parameter(0)
  c1 = s32[1]{0} constant({1})
  ROOT add = s32[1]{0} add(p0, c1)
}

fused_s1 {
  p1 = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p1)
}

ENTRY main {
  p0 = s32[1]{0} parameter(0)
  f0 = s32[1]{0} fusion(p0), kind=kLoop, calls=fused_s0
  cc = s32[1,1]{1,0} custom-call(p0), custom_call_target="foo"
  bc = s32[1]{0} bitcast(cc)
  f1 = s32[1]{0} fusion(bc), kind=kLoop, calls=fused_s1
  ROOT out = (s32[1]{0}, s32[1]{0}) tuple(f0, f1)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest,
       DoesNotBatchFusionExceedingCrossBarrierOutputLimit) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule cross_barrier_exceeds_limit, is_scheduled=true

fused_small {
  p0 = s32[1]{0} parameter(0)
  c1 = s32[1]{0} constant({1})
  ROOT add = s32[1]{0} add(p0, c1)
}

fused_4kib {
  p1 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p1)
}

ENTRY main {
  p0 = s32[1]{0} parameter(0)
  p_heavy = f32[4096]{0} parameter(1)
  p1 = f32[1024]{0} parameter(2)
  f0 = s32[1]{0} fusion(p0), kind=kLoop, calls=fused_small
  cc = f32[4096]{0} custom-call(p_heavy, f0), custom_call_target="foo"
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_4kib
  ROOT out = (f32[4096]{0}, f32[1024]{0}) tuple(cc, f1)
})hlo";

  // `f1` writes `f32[1024]` (4096 bytes), above the default 2048-byte limit.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest,
       BatchesLargerFusionAcrossBarrierWhenLimitIncreased) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule cross_barrier_increased_limit, is_scheduled=true

fused_small {
  p0 = s32[1]{0} parameter(0)
  c1 = s32[1]{0} constant({1})
  ROOT add = s32[1]{0} add(p0, c1)
}

fused_4kib {
  p1 = f32[1024]{0} parameter(0)
  ROOT neg = f32[1024]{0} negate(p1)
}

ENTRY main {
  p0 = s32[1]{0} parameter(0)
  p_heavy = f32[4096]{0} parameter(1)
  p1 = f32[1024]{0} parameter(2)
  f0 = s32[1]{0} fusion(p0), kind=kLoop, calls=fused_small
  cc = f32[4096]{0} custom-call(p_heavy, f0), custom_call_target="foo"
  f1 = f32[1024]{0} fusion(p1), kind=kLoop, calls=fused_4kib
  ROOT out = (f32[4096]{0}, f32[1024]{0}) tuple(cc, f1)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion::Options options;
  options.max_cross_barrier_output_bytes = 4096;
  HorizontalFusion pass(device_info_, options);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(HorizontalFusionTest, DoesNotBatchAcrossBarrierWhenDisabled) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule cross_barrier_disabled, is_scheduled=true

fused_s0 {
  p0 = s32[1]{0} parameter(0)
  c1 = s32[1]{0} constant({1})
  ROOT add = s32[1]{0} add(p0, c1)
}

fused_s1 {
  p1 = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p1)
}

ENTRY main {
  p0 = s32[1]{0} parameter(0)
  p_heavy = f32[4096]{0} parameter(1)
  p1 = s32[1]{0} parameter(2)
  f0 = s32[1]{0} fusion(p0), kind=kLoop, calls=fused_s0
  cc = f32[4096]{0} custom-call(p_heavy, f0), custom_call_target="foo"
  f1 = s32[1]{0} fusion(p1), kind=kLoop, calls=fused_s1
  ROOT out = (f32[4096]{0}, s32[1]{0}) tuple(cc, f1)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion::Options options;
  options.max_cross_barrier_output_bytes = 0;
  HorizontalFusion pass(device_info_, options);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest, FlushesCrossBarrierSmallBatchAtMaxNestedKernels) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule flush_small_batch_at_max_kernels, is_scheduled=true

fused_0 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}

fused_1 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}

fused_2 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}

fused_3 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}

ENTRY main {
  p0 = s32[1]{0} parameter(0)
  p1 = s32[1]{0} parameter(1)
  f0 = s32[1]{0} fusion(p0), kind=kLoop, calls=fused_0
  cc0 = s32[1]{0} custom-call(f0), custom_call_target="bar0"
  f1 = s32[1]{0} fusion(p1), kind=kLoop, calls=fused_1
  cc1 = s32[1]{0} custom-call(f1), custom_call_target="bar1"
  f2 = s32[1]{0} fusion(cc0), kind=kLoop, calls=fused_2
  cc2 = s32[1]{0} custom-call(f2), custom_call_target="bar2"
  f3 = s32[1]{0} fusion(cc0), kind=kLoop, calls=fused_3
  ROOT out = (s32[1]{0}, s32[1]{0}, s32[1]{0}) tuple(cc1, cc2, f3)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion::Options options;
  options.max_nested_kernels = 2;
  HorizontalFusion pass(device_info_, options);
  ASSERT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: %horizontal_fusion_computation
// CHECK-DAG: calls=%fused_0
// CHECK-DAG: calls=%fused_1
// CHECK: %horizontal_fusion_computation.1
// CHECK-DAG: calls=%fused_2
// CHECK-DAG: calls=%fused_3
)"));
  EXPECT_TRUE(filecheck_matched) << module->ToString();
}

TEST_F(HorizontalFusionTest, EvictsAndFusesOldestMultiMemberOpenSmallBatch) {
  // Opens 8 small batches (`kMaxOpenSmallBatches = 8`) where the oldest open
  // batch already holds 2 fusions (`f0a`, `f0b`). Opening a 9th independent
  // batch (`f8`) evicts and fuses `{f0a, f0b}` as a 2-way horizontal fusion, so
  // the trailing parameter-only `f0c` joins the new oldest batch (`f1`) instead
  // of `f0a`.
  constexpr absl::string_view kHloText = R"hlo(
HloModule evict_multi_member_small_batch, is_scheduled=true

fused_0a {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_0b {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_1 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_2 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_3 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_4 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_5 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_6 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_7 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_8 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_0c {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}

ENTRY main {
  p0 = s32[1]{0} parameter(0)
  p1 = s32[1]{0} parameter(1)
  p2 = s32[1]{0} parameter(2)
  f0a = s32[1]{0} fusion(p0), kind=kLoop, calls=fused_0a
  b0 = s32[1]{0} custom-call(p0), custom_call_target="bar"
  f0b = s32[1]{0} fusion(p1), kind=kLoop, calls=fused_0b
  b1 = s32[1]{0} custom-call(b0), custom_call_target="bar"
  f1 = s32[1]{0} fusion(b1), kind=kLoop, calls=fused_1
  b2 = s32[1]{0} custom-call(b1), custom_call_target="bar"
  f2 = s32[1]{0} fusion(b2), kind=kLoop, calls=fused_2
  b3 = s32[1]{0} custom-call(b2), custom_call_target="bar"
  f3 = s32[1]{0} fusion(b3), kind=kLoop, calls=fused_3
  b4 = s32[1]{0} custom-call(b3), custom_call_target="bar"
  f4 = s32[1]{0} fusion(b4), kind=kLoop, calls=fused_4
  b5 = s32[1]{0} custom-call(b4), custom_call_target="bar"
  f5 = s32[1]{0} fusion(b5), kind=kLoop, calls=fused_5
  b6 = s32[1]{0} custom-call(b5), custom_call_target="bar"
  f6 = s32[1]{0} fusion(b6), kind=kLoop, calls=fused_6
  b7 = s32[1]{0} custom-call(b6), custom_call_target="bar"
  f7 = s32[1]{0} fusion(b7), kind=kLoop, calls=fused_7
  b8 = s32[1]{0} custom-call(b7), custom_call_target="bar"
  f8 = s32[1]{0} fusion(b8), kind=kLoop, calls=fused_8
  b9 = s32[1]{0} custom-call(b8), custom_call_target="bar"
  f0c = s32[1]{0} fusion(p2), kind=kLoop, calls=fused_0c
  ROOT out = (s32[1]{0}, s32[1]{0}, s32[1]{0}, s32[1]{0}, s32[1]{0}, s32[1]{0}, s32[1]{0}, s32[1]{0}, s32[1]{0}, s32[1]{0}, s32[1]{0}, s32[1]{0})
    tuple(f0a, f0b, f1, f2, f3, f4, f5, f6, f7, f8, b9, f0c)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion pass(device_info_);
  ASSERT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: %horizontal_fusion_computation
// CHECK-DAG: calls=%fused_0a
// CHECK-DAG: calls=%fused_0b
// CHECK: ROOT %{{.*}} = (s32[1]{0}, s32[1]{0}) tuple(
// CHECK: %horizontal_fusion_computation.1
// CHECK-DAG: calls=%fused_1
// CHECK-DAG: calls=%fused_0c
// CHECK: ROOT %{{.*}} = (s32[1]{0}, s32[1]{0}) tuple(
)"));
  EXPECT_TRUE(filecheck_matched) << module->ToString();
}

TEST_F(HorizontalFusionTest, DropsOldestSingletonOpenSmallBatchOnEviction) {
  // Opens 8 singleton small batches (`f0`..`f7`). Opening a 9th batch (`f8`)
  // evicts and drops singleton `f0`, so `f0` stays unfused while trailing
  // `f_late` joins `f1`.
  constexpr absl::string_view kHloText = R"hlo(
HloModule evict_singleton_small_batch, is_scheduled=true

fused_0 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_1 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_2 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_3 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_4 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_5 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_6 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_7 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_8 {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}
fused_late {
  p = s32[1]{0} parameter(0)
  ROOT neg = s32[1]{0} negate(p)
}

ENTRY main {
  p0 = s32[1]{0} parameter(0)
  p1 = s32[1]{0} parameter(1)
  f0 = s32[1]{0} fusion(p0), kind=kLoop, calls=fused_0
  b1 = s32[1]{0} custom-call(p0), custom_call_target="bar"
  f1 = s32[1]{0} fusion(b1), kind=kLoop, calls=fused_1
  b2 = s32[1]{0} custom-call(b1), custom_call_target="bar"
  f2 = s32[1]{0} fusion(b2), kind=kLoop, calls=fused_2
  b3 = s32[1]{0} custom-call(b2), custom_call_target="bar"
  f3 = s32[1]{0} fusion(b3), kind=kLoop, calls=fused_3
  b4 = s32[1]{0} custom-call(b3), custom_call_target="bar"
  f4 = s32[1]{0} fusion(b4), kind=kLoop, calls=fused_4
  b5 = s32[1]{0} custom-call(b4), custom_call_target="bar"
  f5 = s32[1]{0} fusion(b5), kind=kLoop, calls=fused_5
  b6 = s32[1]{0} custom-call(b5), custom_call_target="bar"
  f6 = s32[1]{0} fusion(b6), kind=kLoop, calls=fused_6
  b7 = s32[1]{0} custom-call(b6), custom_call_target="bar"
  f7 = s32[1]{0} fusion(b7), kind=kLoop, calls=fused_7
  b8 = s32[1]{0} custom-call(b7), custom_call_target="bar"
  f8 = s32[1]{0} fusion(b8), kind=kLoop, calls=fused_8
  b9 = s32[1]{0} custom-call(b8), custom_call_target="bar"
  f_late = s32[1]{0} fusion(p1), kind=kLoop, calls=fused_late
  ROOT out = (s32[1]{0}, s32[1]{0}, s32[1]{0}, s32[1]{0}, s32[1]{0}, s32[1]{0}, s32[1]{0}, s32[1]{0}, s32[1]{0}, s32[1]{0}, s32[1]{0})
    tuple(f0, f1, f2, f3, f4, f5, f6, f7, f8, b9, f_late)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion pass(device_info_);
  ASSERT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: %horizontal_fusion_computation
// CHECK-DAG: calls=%fused_1
// CHECK-DAG: calls=%fused_late
// CHECK: ENTRY %main
// CHECK: s32[1]{0} fusion({{.*}}), kind=kLoop, calls=%fused_0
)"));
  EXPECT_TRUE(filecheck_matched) << module->ToString();
}

TEST_F(HorizontalFusionTest, DoesNotHoistAcrossWhileCarryBufferAlias) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule while_carry_shift, is_scheduled=true

fused_inc {
  i = s64[] parameter(0)
  one = s64[] constant(1)
  ROOT next_i = s64[] add(i, one)
}

fused_zero {
  z = f32[] constant(0)
  ROOT bcast = f32[3]{0} broadcast(z), dimensions={}
}

cond_comp {
  arg = (s64[], f32[3]{0}, f32[3]{0}) parameter(0)
  i = s64[] get-tuple-element(arg), index=0
  two = s64[] constant(2)
  ROOT cmp = pred[] compare(i, two), direction=LT
}

body_comp {
  arg = (s64[], f32[3]{0}, f32[3]{0}) parameter(0)
  i = s64[] get-tuple-element(arg), index=0
  c2 = f32[3]{0} get-tuple-element(arg), index=2
  next_i = s64[] fusion(i), kind=kLoop, calls=fused_inc
  shifted_c1 = f32[3]{0} copy(c2)
  zero_c2 = f32[3]{0} fusion(), kind=kLoop, calls=fused_zero
  ROOT out = (s64[], f32[3]{0}, f32[3]{0}) tuple(next_i, shifted_c1, zero_c2)
}

ENTRY main {
  i0 = s64[] parameter(0)
  a = f32[3]{0} parameter(1)
  b = f32[3]{0} parameter(2)
  init = (s64[], f32[3]{0}, f32[3]{0}) tuple(i0, a, b)
  ROOT res = (s64[], f32[3]{0}, f32[3]{0}) while(init), condition=cond_comp, body=body_comp
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion pass(device_info_);
  // `zero_c2` writes into while carry slot 2, which shares its buffer with `c2`
  // read by `shifted_c1`. Hoisting `zero_c2` into a horizontal batch at
  // `next_i` before `shifted_c1` would clobber `c2` before the copy reads it.
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest, DoesNotFuseAdjacentWhileCarryBufferAlias) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule while_carry_adjacent, is_scheduled=true

fused_read_c1 {
  c1 = f32[3]{0} parameter(0)
  ROOT neg = f32[3]{0} negate(c1)
}

fused_zero {
  z = f32[] constant(0)
  ROOT bcast = f32[3]{0} broadcast(z), dimensions={}
}

cond_comp {
  arg = (f32[3]{0}, f32[3]{0}) parameter(0)
  ROOT p = pred[] constant(false)
}

body_comp {
  arg = (f32[3]{0}, f32[3]{0}) parameter(0)
  c1 = f32[3]{0} get-tuple-element(arg), index=1
  next_c0 = f32[3]{0} fusion(c1), kind=kLoop, calls=fused_read_c1
  next_c1 = f32[3]{0} fusion(), kind=kLoop, calls=fused_zero
  ROOT out = (f32[3]{0}, f32[3]{0}) tuple(next_c0, next_c1)
}

ENTRY main {
  a = f32[3]{0} parameter(0)
  b = f32[3]{0} parameter(1)
  init = (f32[3]{0}, f32[3]{0}) tuple(a, b)
  ROOT res = (f32[3]{0}, f32[3]{0}) while(init), condition=cond_comp, body=body_comp
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest, PreservesScheduleAfterWarPredecessor) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule while_war_schedule, is_scheduled=true

fused_zero {
  z = f32[] constant(0)
  ROOT bcast = f32[3]{0} broadcast(z), dimensions={}
}

fused_inc {
  i = s64[] parameter(0)
  one = s64[] constant(1)
  ROOT next_i = s64[] add(i, one)
}

cond_comp {
  arg = (s64[], f32[3]{0}, f32[3]{0}) parameter(0)
  i = s64[] get-tuple-element(arg), index=0
  one = s64[] constant(1)
  ROOT cmp = pred[] compare(i, one), direction=LT
}

body_comp {
  arg = (s64[], f32[3]{0}, f32[3]{0}) parameter(0)
  i = s64[] get-tuple-element(arg), index=0
  c2 = f32[3]{0} get-tuple-element(arg), index=2
  shifted_c1 = f32[3]{0} copy(c2)
  zero_c2 = f32[3]{0} fusion(), kind=kLoop, calls=fused_zero
  next_i = s64[] fusion(i), kind=kLoop, calls=fused_inc
  ROOT out = (s64[], f32[3]{0}, f32[3]{0}) tuple(next_i, shifted_c1, zero_c2)
}

ENTRY main {
  i0 = s64[] parameter(0)
  a = f32[3]{0} parameter(1)
  b = f32[3]{0} parameter(2)
  init = (s64[], f32[3]{0}, f32[3]{0}) tuple(i0, a, b)
  ROOT res = (s64[], f32[3]{0}, f32[3]{0}) while(init), condition=cond_comp, body=body_comp
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion pass(device_info_);
  ASSERT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  const HloComputation* body_comp = module->GetComputationWithName("body_comp");
  ASSERT_NE(body_comp, nullptr);
  const std::vector<HloInstruction*>& seq =
      module->schedule().sequence(body_comp).instructions();
  std::vector<HloInstruction*>::const_iterator copy_it =
      absl::c_find_if(seq, [](const HloInstruction* inst) {
        return inst->opcode() == HloOpcode::kCopy;
      });
  std::vector<HloInstruction*>::const_iterator hf_it =
      absl::c_find_if(seq, [](const HloInstruction* inst) {
        return inst->opcode() == HloOpcode::kFusion;
      });
  ASSERT_NE(copy_it, seq.end());
  ASSERT_NE(hf_it, seq.end());
  EXPECT_LT(copy_it, hf_it) << module->schedule().ToString();
}

TEST_F(HorizontalFusionTest, DoesNotHoistOrFuseAcrossEntryInputOutputAlias) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule entry_io_alias, is_scheduled=true, input_output_alias={ {1}: (1, {}) }

fused_s0 {
  p0 = s32[1]{0} parameter(0)
  c1 = s32[1]{0} constant({1})
  ROOT add = s32[1]{0} add(p0, c1)
}

fused_zero {
  z = f32[] constant(0)
  ROOT bcast = f32[4]{0} broadcast(z), dimensions={}
}

ENTRY main {
  p0 = s32[1]{0} parameter(0)
  p1 = f32[4]{0} parameter(1)
  f0 = s32[1]{0} fusion(p0), kind=kLoop, calls=fused_s0
  cc = f32[4]{0} custom-call(p1, f0), custom_call_target="read_p1"
  f1 = f32[4]{0} fusion(), kind=kLoop, calls=fused_zero
  ROOT out = (f32[4]{0}, f32[4]{0}) tuple(cc, f1)
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion pass(device_info_);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(HorizontalFusionTest,
       DoesNotHoistAcrossConditionalCrossBranchBufferAlias) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule cond_cross_branch_alias, is_scheduled=true

fused_neg_y {
  p = f32[2]{0} parameter(0)
  ROOT neg = f32[2]{0} negate(p)
}

fused_neg_x {
  p = f32[2]{0} parameter(0)
  ROOT neg = f32[2]{0} negate(p)
}

branch_0 {
  arg0 = (f32[2]{0}, f32[2]{0}) parameter(0)
  x0 = f32[2]{0} get-tuple-element(arg0), index=0
  y0 = f32[2]{0} get-tuple-element(arg0), index=1
  // Returning `x0` directly at tuple index 1 causes `HloAliasAnalysis` to share
  // the buffer of `x` with `conditional` output {1} across all branches.
  ROOT out0 = (f32[2]{0}, f32[2]{0}, f32[2]{0}) tuple(y0, x0, y0)
}

branch_1 {
  arg1 = (f32[2]{0}, f32[2]{0}) parameter(0)
  x1 = f32[2]{0} get-tuple-element(arg1), index=0
  y1 = f32[2]{0} get-tuple-element(arg1), index=1
  neg_y = f32[2]{0} fusion(y1), kind=kLoop, calls=fused_neg_y
  saved_x = f32[2]{0} copy(x1)
  neg_x = f32[2]{0} fusion(x1), kind=kLoop, calls=fused_neg_x
  ROOT out1 = (f32[2]{0}, f32[2]{0}, f32[2]{0}) tuple(saved_x, neg_x, neg_y)
}

ENTRY main {
  pred_idx = s32[] parameter(0)
  x = f32[2]{0} parameter(1)
  y = f32[2]{0} parameter(2)
  branch_arg = (f32[2]{0}, f32[2]{0}) tuple(x, y)
  ROOT res = (f32[2]{0}, f32[2]{0}, f32[2]{0}) conditional(pred_idx, branch_arg, branch_arg), branch_computations={branch_0, branch_1}
})hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion pass(device_info_);
  // In `branch_1`, `neg_x` writes `out1{1}`, which shares a buffer with `x1`
  // because `branch_0` forwards `x0` (`x`) to `out0{1}`. Hoisting `neg_x` into
  // a horizontal batch at `neg_y` before `saved_x = copy(x1)` would overwrite
  // `x1` before `saved_x` reads it.
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}
}  // namespace
}  // namespace xla::gpu
