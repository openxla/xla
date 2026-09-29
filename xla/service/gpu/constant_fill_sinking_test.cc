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

#include "xla/service/gpu/constant_fill_sinking.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
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
#include "xla/shape_util.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

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

std::vector<std::string> Names(const std::vector<HloInstruction*>& sequence) {
  std::vector<std::string> names;
  names.reserve(sequence.size());
  for (const HloInstruction* instruction : sequence) {
    names.emplace_back(instruction->name());
  }
  return names;
}

class ConstantFillSinkingTest : public HloHardwareIndependentTestBase {
 protected:
  // Runs the sink on the entry sequence, writes the result back so that the
  // schedule can be verified, and returns whether the sequence changed.
  bool Sink(HloModule& module) {
    HloComputation* entry = module.entry_computation();
    std::vector<HloInstruction*> sequence =
        module.schedule().sequence(entry).instructions();
    bool changed = SinkConstantFills(sequence);
    module.schedule().set_sequence(entry, sequence);
    return changed;
  }

  std::vector<std::string> EntryNames(const HloModule& module) {
    return Names(
        module.schedule().sequence(module.entry_computation()).instructions());
  }
};

TEST_F(ConstantFillSinkingTest, SinksOnlyTheFillAndIsIdempotent) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(Sink(*module));
  EXPECT_THAT(EntryNames(*module),
              ElementsAre("initial", "loop", "after", "fill", "result"));
  ASSERT_OK(module->schedule().Verify());
  EXPECT_FALSE(Sink(*module));
}

TEST_F(ConstantFillSinkingTest, SinksNonzeroScalarConstant) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(absl::StrReplaceAll(
                           kHlo, {{"constant(0)", "constant(7)"}})));
  EXPECT_TRUE(Sink(*module));
  EXPECT_THAT(EntryNames(*module),
              ElementsAre("initial", "loop", "after", "fill", "result"));
}

TEST_F(ConstantFillSinkingTest, SinksWithoutWhile) {
  std::string hlo = absl::StrReplaceAll(
      kHlo,
      {{"while(initial), condition=condition, body=body", "negate(initial)"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_TRUE(Sink(*module));
  EXPECT_THAT(EntryNames(*module),
              ElementsAre("initial", "loop", "after", "fill", "result"));
  ASSERT_OK(module->schedule().Verify());
}

TEST_F(ConstantFillSinkingTest, StopsAtFirstOfMultipleUsers) {
  std::string hlo = absl::StrReplaceAll(kHlo, {{kRoot, R"(
  first = f32[512,512]{1,0} negate(fill)
  second = f32[512,512]{1,0} add(first, fill)
  ROOT result = (s32[], f32[512,512]{1,0}) tuple(after, second))"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EXPECT_TRUE(Sink(*module));
  EXPECT_THAT(EntryNames(*module),
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
  EXPECT_TRUE(Sink(*module));
  // Preserve the original fill order, not the consumer's operand order.
  EXPECT_THAT(EntryNames(*module),
              ElementsAre("initial", "loop", "after", "fill", "other", "sum",
                          "result"));
  ASSERT_OK(module->schedule().Verify());
}

TEST_F(ConstantFillSinkingTest, DoesNotReorderAdjacentFills) {
  // Two fills already sit directly before their shared first user. Neither
  // should move, otherwise repeated runs would swap them back and forth.
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule adjacent, is_scheduled=true

fill_body {
  value = f32[] constant(0)
  ROOT broadcast = f32[512,512]{1,0} broadcast(value), dimensions={}
}
other_fill_body {
  value = f32[] constant(1)
  ROOT broadcast = f32[512,512]{1,0} broadcast(value), dimensions={}
}
ENTRY main {
  p = f32[512,512]{1,0} parameter(0)
  negated = f32[512,512]{1,0} negate(p)
  fill = f32[512,512]{1,0} fusion(), kind=kLoop, calls=fill_body
  other = f32[512,512]{1,0} fusion(), kind=kLoop, calls=other_fill_body
  sum = f32[512,512]{1,0} add(fill, other)
  ROOT result = f32[512,512]{1,0} add(sum, negated)
}
)"));
  std::vector<std::string> before = EntryNames(*module);
  EXPECT_FALSE(Sink(*module));
  EXPECT_EQ(EntryNames(*module), before);
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
  EXPECT_TRUE(Sink(*module));
  EXPECT_THAT(EntryNames(*module),
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
  EXPECT_TRUE(Sink(*module));
  EXPECT_THAT(EntryNames(*module),
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
  EXPECT_TRUE(Sink(*module));
  EXPECT_THAT(EntryNames(*module), ElementsAre("initial", "loop", "second_loop",
                                               "after", "fill", "result"));
}

TEST_F(ConstantFillSinkingTest, LeavesFillWhoseUserIsOutsideTheSequence) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  std::vector<HloInstruction*> sequence =
      module->schedule().sequence(module->entry_computation()).instructions();
  // Drop the fill's user from the sequence handed to the sink.
  sequence.pop_back();
  std::vector<std::string> before = Names(sequence);
  EXPECT_FALSE(SinkConstantFills(sequence));
  EXPECT_EQ(Names(sequence), before);
}

// Entry template for async-window tests. $0 is the entry body after `input`.
// In the bodies below, $0 opens an async window on `input` as `start` and $1
// closes it as `done`.
constexpr absl::string_view kAsyncHlo = R"(
HloModule async, is_scheduled=true

fill_body {
  value = f32[] constant(0)
  ROOT broadcast = f32[512,512]{1,0} broadcast(value), dimensions={}
}
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT sum = f32[] add(a, b)
}
async_body {
  p = f32[512,512]{1,0} parameter(0)
  ROOT negated = f32[512,512]{1,0} negate(p)
}
ENTRY main {
  input = f32[512,512]{1,0} parameter(0)
$0
}
)";

struct AsyncPair {
  const char* name;
  absl::string_view start;
  absl::string_view done;
};

std::string AsyncHlo(const AsyncPair& pair, absl::string_view body) {
  return absl::Substitute(kAsyncHlo,
                          absl::Substitute(body, pair.start, pair.done));
}

class AsyncWindowConstantFillSinkingTest
    : public ConstantFillSinkingTest,
      public ::testing::WithParamInterface<AsyncPair> {};

TEST_P(AsyncWindowConstantFillSinkingTest, LeavesFillWhoseUserFollowsTheDone) {
  // The fill hides under the async operation. Its user runs after the done,
  // so sinking would put the fill back on the critical path.
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(AsyncHlo(GetParam(), R"(  $0
  fill = f32[512,512]{1,0} fusion(), kind=kLoop, calls=fill_body
  other = f32[512,512]{1,0} negate(input)
  $1
  ROOT result = f32[512,512]{1,0} add(done, fill))")));
  std::vector<std::string> before = EntryNames(*module);
  EXPECT_FALSE(Sink(*module));
  EXPECT_EQ(EntryNames(*module), before);
}

TEST_P(AsyncWindowConstantFillSinkingTest, SinksFillWhoseUserSharesTheWindow) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(AsyncHlo(GetParam(), R"(  $0
  fill = f32[512,512]{1,0} fusion(), kind=kLoop, calls=fill_body
  other = f32[512,512]{1,0} negate(input)
  use = f32[512,512]{1,0} add(other, fill)
  $1
  ROOT result = (f32[512,512]{1,0}, f32[512,512]{1,0}) tuple(done, use))")));
  EXPECT_TRUE(Sink(*module));
  EXPECT_THAT(
      EntryNames(*module),
      ElementsAre("input", "start", "other", "fill", "use", "done", "result"));
  ASSERT_OK(module->schedule().Verify());
}

TEST_P(AsyncWindowConstantFillSinkingTest, SinksFillScheduledBeforeTheWindow) {
  // The fill is outside the window, so crossing the window is free.
  ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(AsyncHlo(
          GetParam(),
          R"(  fill = f32[512,512]{1,0} fusion(), kind=kLoop, calls=fill_body
  $0
  other = f32[512,512]{1,0} negate(input)
  $1
  ROOT result = f32[512,512]{1,0} add(done, fill))")));
  EXPECT_TRUE(Sink(*module));
  EXPECT_THAT(EntryNames(*module),
              ElementsAre("input", "start", "other", "done", "fill", "result"));
  ASSERT_OK(module->schedule().Verify());
}

INSTANTIATE_TEST_SUITE_P(
    AsyncPairs, AsyncWindowConstantFillSinkingTest,
    ::testing::Values(
        AsyncPair{"AllReduce",
                  "start = f32[512,512]{1,0} all-reduce-start(input), "
                  "to_apply=add",
                  "done = f32[512,512]{1,0} all-reduce-done(start)"},
        AsyncPair{"AsyncCompute",
                  "start = ((f32[512,512]{1,0}), f32[512,512]{1,0}, u32[]) "
                  "async-start(input), calls=async_body",
                  "done = f32[512,512]{1,0} async-done(start)"},
        AsyncPair{"Copy",
                  "start = (f32[512,512]{1,0}, f32[512,512]{1,0}, u32[]) "
                  "copy-start(input)",
                  "done = f32[512,512]{1,0} copy-done(start)"}),
    [](const ::testing::TestParamInfo<AsyncPair>& info) {
      return info.param.name;
    });

TEST_F(ConstantFillSinkingTest, LeavesFillWhenAnyEnclosingWindowClosesFirst) {
  // The fill sits in two nested windows. Its user is still inside the outer
  // one, but the inner window closes first, so the fill stays.
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule nested, is_scheduled=true

fill_body {
  value = f32[] constant(0)
  ROOT broadcast = f32[512,512]{1,0} broadcast(value), dimensions={}
}
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT sum = f32[] add(a, b)
}
ENTRY main {
  p = f32[512,512]{1,0} parameter(0)
  outer_start = f32[512,512]{1,0} all-reduce-start(p), to_apply=add
  inner_start = f32[512,512]{1,0} all-reduce-start(p), to_apply=add
  fill = f32[512,512]{1,0} fusion(), kind=kLoop, calls=fill_body
  inner_done = f32[512,512]{1,0} all-reduce-done(inner_start)
  use = f32[512,512]{1,0} add(inner_done, fill)
  outer_done = f32[512,512]{1,0} all-reduce-done(outer_start)
  ROOT result = (f32[512,512]{1,0}, f32[512,512]{1,0}) tuple(outer_done, use)
}
)"));
  std::vector<std::string> before = EntryNames(*module);
  EXPECT_FALSE(Sink(*module));
  EXPECT_EQ(EntryNames(*module), before);
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
  EXPECT_TRUE(Sink(*module));
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

TEST_P(RejectedConstantFillSinkingTest, LeavesSequenceUnchanged) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(
                           absl::StrReplaceAll(kHlo, GetParam().replacements)));
  std::vector<std::string> before = EntryNames(*module);
  EXPECT_FALSE(Sink(*module));
  EXPECT_EQ(EntryNames(*module), before);
}

INSTANTIATE_TEST_SUITE_P(
    Safeguards, RejectedConstantFillSinkingTest,
    ::testing::Values(
        RejectedFill{"SmallFill", {{"512", "256"}}},
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
