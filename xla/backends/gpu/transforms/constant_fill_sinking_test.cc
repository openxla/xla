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

#include "xla/backends/gpu/transforms/constant_fill_sinking.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/buffer_value.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

constexpr absl::string_view kHlo = R"(
HloModule test, is_scheduled=true

fill_body {
  value = f32[] constant(0)
  ROOT broadcast = f32[512,512]{1,0} broadcast(value), dimensions={}
}
condition {
  p = s32[] parameter(0)
  limit = s32[] constant(2)
  ROOT test = pred[] compare(p, limit), direction=LT
}
body {
  p = s32[] parameter(0)
  one = s32[] constant(1)
  ROOT next = s32[] add(p, one)
}
ENTRY main {
  initial = s32[] parameter(0)
  fill = f32[512,512]{1,0} fusion(), kind=kLoop, calls=fill_body
  loop = s32[] while(initial), condition=condition, body=body
  after = s32[] negate(loop)
  ROOT result = (s32[], f32[512,512]{1,0}) tuple(after, fill)
}
)";

constexpr absl::string_view kRoot =
    "ROOT result = (s32[], f32[512,512]{1,0}) tuple(after, fill)";

std::vector<std::string> ScheduleNames(const HloModule& module) {
  std::vector<std::string> names;
  for (const HloInstruction* instruction :
       module.schedule().sequence(module.entry_computation()).instructions()) {
    names.emplace_back(instruction->name());
  }
  return names;
}

std::string GraphWithoutSchedule(const HloModule& module) {
  HloModuleProto proto = module.ToProto();
  proto.clear_schedule();
  return proto.SerializeAsString();
}

using ConstantFillSinkingTest = HloHardwareIndependentTestBase;

TEST_F(ConstantFillSinkingTest, SinksOnlyTheFillAndIsIdempotent) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  std::string graph = GraphWithoutSchedule(*module);
  ConstantFillSinking pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(ScheduleNames(*module),
              ElementsAre("initial", "loop", "after", "fill", "result"));
  EXPECT_EQ(GraphWithoutSchedule(*module), graph);
  ASSERT_OK(module->schedule().Verify());
  ASSERT_OK_AND_ASSIGN(changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ConstantFillSinkingTest, SinksNonzeroScalarConstant) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(absl::StrReplaceAll(
                           kHlo, {{"constant(0)", "constant(7)"}})));
  std::string graph = GraphWithoutSchedule(*module);
  ASSERT_OK_AND_ASSIGN(bool changed, ConstantFillSinking().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(GraphWithoutSchedule(*module), graph);
}

TEST_F(ConstantFillSinkingTest, StopsAtFirstOfMultipleUsers) {
  std::string hlo = absl::StrReplaceAll(kHlo, {{kRoot, R"(
  first = f32[512,512]{1,0} negate(fill)
  second = f32[512,512]{1,0} add(first, fill)
  ROOT result = (s32[], f32[512,512]{1,0}) tuple(after, second))"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(bool changed, ConstantFillSinking().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(ScheduleNames(*module),
              ElementsAre("initial", "loop", "after", "fill", "first", "second",
                          "result"));
  ASSERT_OK(module->schedule().Verify());
}

TEST_F(ConstantFillSinkingTest, PreservesOrderOfFillsSharingFirstUser) {
  std::string hlo = absl::StrReplaceAll(
      kHlo, {{"ENTRY main", R"(other_fill_body {
  value = f32[] constant(0)
  ROOT broadcast = f32[512,512]{1,0} broadcast(value), dimensions={}
}
ENTRY main)"},
             {"  loop =",
              "  other = f32[512,512]{1,0} fusion(), kind=kLoop, "
              "calls=other_fill_body\n  loop ="},
             {kRoot, R"(sum = f32[512,512]{1,0} add(other, fill)
  ROOT result = (s32[], f32[512,512]{1,0}) tuple(after, sum))"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(bool changed, ConstantFillSinking().Run(module.get()));
  EXPECT_TRUE(changed);
  // Preserve the original fill order, not the consumer's operand order.
  EXPECT_THAT(ScheduleNames(*module),
              ElementsAre("initial", "loop", "after", "fill", "other", "sum",
                          "result"));
  ASSERT_OK(module->schedule().Verify());
}

TEST_F(ConstantFillSinkingTest, StopsBeforeAsyncStart) {
  std::string hlo = absl::StrReplaceAll(
      kHlo,
      {{"ENTRY main", R"(async_body {
  p = f32[512,512]{1,0} parameter(0)
  ROOT negated = f32[512,512]{1,0} negate(p)
}
ENTRY main)"},
       {kRoot,
        R"(start = ((f32[512,512]{1,0}), f32[512,512]{1,0}, u32[]) async-start(fill), calls=async_body
  done = f32[512,512]{1,0} async-done(start)
  ROOT result = (s32[], f32[512,512]{1,0}) tuple(after, done))"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(bool changed, ConstantFillSinking().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(ScheduleNames(*module),
              ElementsAre("initial", "loop", "after", "fill", "start", "done",
                          "result"));
  ASSERT_OK(module->schedule().Verify());
}

TEST_F(ConstantFillSinkingTest, StopsAtAnAliasingBitcast) {
  std::string hlo = absl::StrReplaceAll(kHlo, {{kRoot, R"(
  view = f32[262144]{0} bitcast(fill)
  second_loop = s32[] while(after), condition=condition, body=body
  ROOT result = (s32[], f32[262144]{0}) tuple(second_loop, view))"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(bool changed, ConstantFillSinking().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(ScheduleNames(*module),
              ElementsAre("initial", "loop", "after", "fill", "view",
                          "second_loop", "result"));
}

TEST_F(ConstantFillSinkingTest, CanCrossMultipleWhiles) {
  std::string hlo =
      absl::StrReplaceAll(kHlo, {{"  after =",
                                  "  second_loop = s32[] while(loop), "
                                  "condition=condition, body=body\n  after ="},
                                 {"negate(loop)", "negate(second_loop)"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(bool changed, ConstantFillSinking().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      ScheduleNames(*module),
      ElementsAre("initial", "loop", "second_loop", "after", "fill", "result"));
}

TEST_F(ConstantFillSinkingTest, RequiresSchedule) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  module->clear_schedule();
  EXPECT_THAT(ConstantFillSinking().Run(module.get()),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(ConstantFillSinkingTest, RespectsExecutionThreads) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  std::string before = module->ToString();
  ASSERT_OK_AND_ASSIGN(bool changed,
                       ConstantFillSinking().Run(module.get(), {"other"}));
  EXPECT_FALSE(changed);
  EXPECT_EQ(module->ToString(), before);
}

TEST_F(ConstantFillSinkingTest, ReducesPeakMemoryAcrossWhile) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule memory, is_scheduled=true
sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}
small_fill {
  zero = f32[] constant(0)
  ROOT broadcast = f32[512,512]{1,0} broadcast(zero), dimensions={}
}
large_fill {
  one = f32[] constant(1)
  ROOT broadcast = f32[1024,1024]{1,0} broadcast(one), dimensions={}
}
condition {
  p = f32[1024,1024]{1,0} parameter(0)
  ROOT stop = pred[] constant(false)
}
body {
  p = f32[1024,1024]{1,0} parameter(0)
  ROOT negated = f32[1024,1024]{1,0} negate(p)
}
ENTRY main {
  late = f32[512,512]{1,0} fusion(), kind=kLoop, calls=small_fill
  early = f32[1024,1024]{1,0} fusion(), kind=kLoop, calls=large_fill
  loop = f32[1024,1024]{1,0} while(early), condition=condition, body=body
  zero = f32[] constant(0)
  reduced = f32[] reduce(loop, zero), dimensions={0,1}, to_apply=sum
  ROOT result = (f32[], f32[512,512]{1,0}) tuple(reduced, late)
})"));
  GpuAliasInfo alias_info(TestGpuDeviceInfo::RTXA6000DeviceInfo());
  ASSERT_OK_AND_ASSIGN(auto aliases,
                       HloAliasAnalysis::Run(module.get(), &alias_info));
  BufferValue::SizeFunction size = [](const BufferValue& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
  };
  ASSERT_OK_AND_ASSIGN(int64_t before,
                       HeapSimulator::MinimumMemoryForModule(
                           module->schedule(), *aliases, &alias_info, &size));
  ASSERT_OK_AND_ASSIGN(bool changed, ConstantFillSinking().Run(module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(int64_t after,
                       HeapSimulator::MinimumMemoryForModule(
                           module->schedule(), *aliases, &alias_info, &size));
  EXPECT_GE(before - after, 1024 * 1024);
}

struct RejectedFill {
  const char* name;
  std::vector<std::pair<absl::string_view, absl::string_view>> replacements;
};

class RejectedConstantFillSinkingTest
    : public ConstantFillSinkingTest,
      public ::testing::WithParamInterface<RejectedFill> {};

TEST_P(RejectedConstantFillSinkingTest, LeavesScheduleUnchanged) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(
                           absl::StrReplaceAll(kHlo, GetParam().replacements)));
  std::string before = module->ToString();
  ASSERT_OK_AND_ASSIGN(bool changed, ConstantFillSinking().Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(module->ToString(), before);
}

INSTANTIATE_TEST_SUITE_P(
    Safeguards, RejectedConstantFillSinkingTest,
    ::testing::Values(
        RejectedFill{"SmallFill", {{"512", "256"}}},
        RejectedFill{"NoWhile",
                     {{"while(initial), condition=condition, body=body",
                       "negate(initial)"}}},
        RejectedFill{"Operand",
                     {{"f32[] constant(0)", "s32[] parameter(0)"},
                      {"broadcast(value)", "broadcast(converted)"},
                      {"  ROOT broadcast",
                       "  converted = f32[] convert(value)\n  ROOT broadcast"},
                      {"fusion()", "fusion(initial)"}}},
        RejectedFill{"ComputedValue",
                     {{"  ROOT broadcast",
                       "  negated = f32[] negate(value)\n  ROOT broadcast"},
                      {"broadcast(value)", "broadcast(negated)"}}},
        RejectedFill{"NonScalarLiteral",
                     {{"f32[] constant(0)", "f32[1] constant({0})"},
                      {"512,512", "1,262144"},
                      {"dimensions={}", "dimensions={0}"}}},
        RejectedFill{"ControlPredecessor",
                     {{"calls=fill_body",
                       "calls=fill_body, control-predecessors={initial}"}}},
        RejectedFill{
            "ControlSuccessor",
            {{"negate(loop)", "negate(loop), control-predecessors={fill}"}}},
        RejectedFill{
            "Sharding",
            {{"calls=fill_body", "calls=fill_body, sharding={replicated}"}}},
        RejectedFill{"SchedulingGroup",
                     {{"calls=fill_body",
                       "calls=fill_body, "
                       "frontend_attributes={scheduling_group_id=\"1\"}"}}},
        RejectedFill{
            "ForceEarly",
            {{"calls=fill_body",
              R"(calls=fill_body, backend_config={"force_earliest_schedule":true})"}}},
        RejectedFill{
            "Stream",
            {{"calls=fill_body",
              R"(calls=fill_body, backend_config={"operation_queue_id":"1"})"}}},
        RejectedFill{
            "Host",
            {{"calls=fill_body",
              R"(calls=fill_body, backend_config={"device_type":"DEVICE_TYPE_HOST"})"}}},
        RejectedFill{
            "EarlyUser",
            {{"  loop =", "  early = f32[512,512]{1,0} copy(fill)\n  loop ="},
             {kRoot,
              "ROOT result = (s32[], f32[512,512]{1,0}, f32[512,512]{1,0}) "
              "tuple(after, fill, early)"}}},
        RejectedFill{"TupleBeforeLoop",
                     {{"  loop =",
                       "  early = (s32[], f32[512,512]{1,0}) tuple(initial, "
                       "fill)\n  loop ="}}}),
    [](const ::testing::TestParamInfo<RejectedFill>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace xla::gpu
