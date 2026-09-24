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

#include "xla/backends/gpu/transforms/constant_fill_copy_rewriter.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/transforms/copy_fusion.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "xla/literal.h"
#include "xla/service/buffer_value.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

constexpr char kHlo[] = R"(
HloModule test

fill_body {
  value = f32[] constant(7)
  ROOT broadcast = f32[512,512]{1,0} broadcast(value), dimensions={}
}

ENTRY main {
  predecessor = f32[] constant(1)
  fill = f32[512,512]{1,0} fusion(), kind=kLoop, calls=fill_body
  ROOT copy = f32[512,512]{1,0} copy(fill)
}
)";

using ConstantFillCopyRewriterTest = HloHardwareIndependentTestBase;

TEST_F(ConstantFillCopyRewriterTest, ReplacesCopyAndRemovesUnusedSource) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ConstantFillCopyRewriter pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->operand_count(), 0);
  EXPECT_EQ(root->fused_expression_root()
                ->operand(0)
                ->literal()
                .GetFirstElement<float>(),
            7);
  EXPECT_EQ(module->entry_computation()->GetInstructionWithName("fill"),
            nullptr);
  // The removed fill's fused computation must not linger in the module.
  EXPECT_EQ(module->GetComputationWithName("fill_body"), nullptr);
  EXPECT_EQ(module->computation_count(), 2);
  ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ConstantFillCopyRewriterTest, TransfersCopyAnnotationsToFill) {
  std::string hlo = absl::StrReplaceAll(
      kHlo,
      {{"calls=fill_body",
        R"(calls=fill_body, frontend_attributes={_scheduling_group_id="1",from_fill="yes"}, backend_config={"fusion_backend_config":{"kind":"__fill"}}, metadata={op_name="fill_op"})"},
       {"copy(fill)",
        R"(copy(fill), frontend_attributes={_scheduling_group_id="2",from_copy="yes"}, backend_config={"operation_queue_id":"3"}, metadata={op_name="copy_op" source_file="copy.py" source_line=7})"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ConstantFillCopyRewriter pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion);
  // The copy's annotations win on conflict; the fill's unrelated ones remain.
  const auto& attributes = root->frontend_attributes().map();
  EXPECT_EQ(attributes.at("_scheduling_group_id"), "2");
  EXPECT_EQ(attributes.at("from_copy"), "yes");
  EXPECT_EQ(attributes.at("from_fill"), "yes");
  EXPECT_EQ(root->metadata().op_name(), "copy_op");
  EXPECT_EQ(root->metadata().source_file(), "copy.py");
  EXPECT_EQ(root->metadata().source_line(), 7);
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(config.operation_queue_id(), 3);
  EXPECT_EQ(config.fusion_backend_config().kind(), "__fill");
}

TEST_F(ConstantFillCopyRewriterTest,
       CopiesAreIndependentEvenWithSharedConsumer) {
  std::string hlo = absl::StrReplaceAll(
      kHlo, {{"ROOT copy = f32[512,512]{1,0} copy(fill)", R"(
  copy0 = f32[512,512]{1,0} copy(fill)
  copy1 = f32[512,512]{1,0} copy(fill)
  ROOT result = (f32[512,512], f32[512,512], f32[512,512])
      tuple(fill, copy0, copy1))"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  HloInstruction* original =
      module->entry_computation()->GetInstructionWithName("fill");
  ConstantFillCopyRewriter pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* result =
      module->entry_computation()->root_instruction();
  EXPECT_EQ(result->operand(0), original);
  for (const HloInstruction* fill : result->operands()) {
    ASSERT_EQ(fill->opcode(), HloOpcode::kFusion);
    EXPECT_EQ(fill->operand_count(), 0);
    EXPECT_EQ(fill->shape(), original->shape());
    EXPECT_EQ(*fill->fused_instructions_computation(),
              *original->fused_instructions_computation());
  }
  EXPECT_NE(result->operand(0), result->operand(1));
  EXPECT_NE(result->operand(0), result->operand(2));
  EXPECT_NE(result->operand(1), result->operand(2));

  // The following pipeline pass must leave the independent fills intact.
  const auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  CopyFusion copy_fusion(device);
  ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&copy_fusion, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ConstantFillCopyRewriterTest, RespectsExecutionThreads) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(bool changed,
                       ConstantFillCopyRewriter().Run(module.get(), {"other"}));
  EXPECT_FALSE(changed);
}

TEST_F(ConstantFillCopyRewriterTest, ReducesScheduledPeakMemory) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule test
sum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}
fill_body {
  zero = f32[] constant(0)
  ROOT broadcast = f32[512,512]{1,0} broadcast(zero), dimensions={}
}
ENTRY main {
  update = f32[1,512]{1,0} parameter(0)
  zero = s32[] constant(0)
  one = s32[] constant(1)
  initial_sum = f32[] constant(0)
  fill = f32[512,512]{1,0} fusion(), kind=kLoop, calls=fill_body
  copy = f32[512,512]{1,0} copy(fill)
  early = f32[512,512]{1,0} dynamic-update-slice(copy, update, zero, zero)
  early_sum = f32[] reduce(early, initial_sum), dimensions={0,1}, to_apply=sum
  middle = f32[1024,1024]{1,0} broadcast(early_sum), dimensions={}
  middle_sum = f32[] reduce(middle, initial_sum), dimensions={0,1}, to_apply=sum
  late_update = f32[1,512]{1,0} broadcast(middle_sum), dimensions={}
  ROOT late = f32[512,512]{1,0}
      dynamic-update-slice(fill, late_update, one, zero)
})"));
  auto baseline = module->Clone();
  const auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  GpuAliasInfo alias_info(device);
  CopyFusion copy_fusion(device);
  ASSERT_OK_AND_ASSIGN(bool baseline_changed,
                       RunHloPass(&copy_fusion, baseline.get()));
  EXPECT_TRUE(baseline_changed);
  auto size = [](const BufferValue& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
  };
  int64_t baseline_peak;
  ASSERT_OK_AND_ASSIGN(
      auto baseline_schedule,
      ScheduleModule(baseline.get(), &alias_info, size, {}, &baseline_peak));
  ASSERT_OK(baseline_schedule.Verify());

  ConstantFillCopyRewriter pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&copy_fusion, module.get()));
  EXPECT_FALSE(changed);
  int64_t peak;
  ASSERT_OK_AND_ASSIGN(auto schedule, ScheduleModule(module.get(), &alias_info,
                                                     size, {}, &peak));
  ASSERT_OK(schedule.Verify());
  // The original 1 MiB fill no longer needs to span the 4 MiB middle buffer.
  EXPECT_LT(peak, baseline_peak);
}

TEST_F(ConstantFillCopyRewriterTest, LeavesCopiesInsideWhileBodyAlone) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule test
fill_body {
  zero = f32[] constant(0)
  ROOT broadcast = f32[512,512]{1,0} broadcast(zero), dimensions={}
}
condition {
  p = f32[512,512]{1,0} parameter(0)
  ROOT stop = pred[] constant(false)
}
body {
  p = f32[512,512]{1,0} parameter(0)
  fill = f32[512,512]{1,0} fusion(), kind=kLoop, calls=fill_body
  ROOT copy = f32[512,512]{1,0} copy(fill)
}
ENTRY main {
  p = f32[512,512]{1,0} parameter(0)
  ROOT loop = f32[512,512]{1,0} while(p), condition=condition, body=body
})"));
  ConstantFillCopyRewriter pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

struct RejectedCopy {
  const char* name;
  std::vector<std::pair<absl::string_view, absl::string_view>> replacements;
};

class RejectedConstantFillCopyTest
    : public ConstantFillCopyRewriterTest,
      public ::testing::WithParamInterface<RejectedCopy> {};

TEST_P(RejectedConstantFillCopyTest, LeavesCopyUnchanged) {
  const RejectedCopy& test = GetParam();
  std::string hlo = absl::StrReplaceAll(kHlo, test.replacements);
  ASSERT_NE(hlo, kHlo);
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  std::string before = module->ToString();
  ConstantFillCopyRewriter pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(module->ToString(), before);
}

INSTANTIATE_TEST_SUITE_P(
    Safeguards, RejectedConstantFillCopyTest,
    ::testing::Values(
        RejectedCopy{"SmallFill", {{"512", "256"}}},
        RejectedCopy{"LayoutChange",
                     {{"ROOT copy = f32[512,512]{1,0}",
                       "ROOT copy = f32[512,512]{0,1}"}}},
        RejectedCopy{
            "ParameterFill",
            {{"value = f32[] constant(7)", "value = f32[] parameter(0)"},
             {"fusion()", "fusion(predecessor)"}}},
        RejectedCopy{"ComputedFill",
                     {{"ROOT broadcast",
                       "negated = f32[] negate(value)\n"
                       "  ROOT broadcast"},
                      {"broadcast(value)", "broadcast(negated)"}}},
        RejectedCopy{"NonscalarLiteral",
                     {{"f32[] constant(7)", "f32[1] constant({7})"},
                      {"512,512", "1,262144"},
                      {"dimensions={}", "dimensions={0}"}}},
        RejectedCopy{
            "CopyControlDependency",
            {{"copy(fill)", "copy(fill), control-predecessors={predecessor}"}}},
        RejectedCopy{"FillControlDependency",
                     {{"calls=fill_body",
                       "calls=fill_body, control-predecessors={predecessor}"}}},
        RejectedCopy{"CopySharding",
                     {{"copy(fill)", "copy(fill), sharding={replicated}"}}},
        RejectedCopy{
            "FillSharding",
            {{"calls=fill_body", "calls=fill_body, sharding={replicated}"}}},
        RejectedCopy{"BitcastSource",
                     {{"ROOT copy",
                       "bitcast = f32[512,512]{1,0} bitcast(fill)\n"
                       "  ROOT copy"},
                      {"copy(fill)", "copy(bitcast)"}}}),
    [](const ::testing::TestParamInfo<RejectedCopy>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace xla::gpu
