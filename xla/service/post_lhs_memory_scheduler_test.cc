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

#include "xla/service/post_lhs_memory_scheduler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/buffer_value.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

int64_t ShapeSize(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, /*pointer_size=*/8);
}

class FixedLatencyEstimator : public LatencyEstimator {
 public:
  TimeCost GetLatencyBetween(const HloGraphNode& from,
                             const HloGraphNode& target) const override {
    return IsAsyncPair(from, target) ? 100 : 1;
  }
  TimeCost NodeCost(const HloInstruction*) const override { return 1; }
  int CyclesPerMicrosecond() const override { return 1; }
};

class PostLhsMemorySchedulerTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<bool> RunScheduler(HloModule* module, int64_t budget = 64) {
    auto context = std::make_shared<SchedulingContext>(
        module, std::make_shared<FixedLatencyEstimator>(),
        std::make_shared<AsyncTracker>(SchedulerConfig()), &alias_info_,
        ShapeSize);
    return PostLhsMemoryScheduler(context, budget).Run(module);
  }

  absl::StatusOr<int64_t> Peak(HloModule* module) {
    ABSL_ASSIGN_OR_RETURN(auto aliases,
                          HloAliasAnalysis::Run(module, &alias_info_));
    BufferValue::SizeFunction size = [](const BufferValue& buffer) {
      return ShapeSize(buffer.shape());
    };
    return HeapSimulator::MinimumMemoryForModule(module->schedule(), *aliases,
                                                 &alias_info_, &size);
  }

  std::vector<std::string> Names(const HloModule& module) {
    std::vector<std::string> names;
    for (const HloInstruction* instruction :
         module.schedule()
             .sequence(module.entry_computation())
             .instructions()) {
      names.emplace_back(instruction->name());
    }
    return names;
  }

  int64_t Position(const HloModule& module, absl::string_view name) {
    const auto& instructions =
        module.schedule().sequence(module.entry_computation()).instructions();
    for (int64_t i = 0; i < instructions.size(); ++i) {
      if (instructions[i]->name() == name) {
        return i;
      }
    }
    return -1;
  }

  AliasInfo alias_info_;
};

// The last consumer b is a parent-level sibling, so it can move across an
// entire async region. The scratch allocation makes the long lifetime costly.
constexpr absl::string_view kParentConsumer = R"hlo(
HloModule m, is_scheduled=true
sum {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}
ENTRY entry {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  zero = f32[] constant(0)
  big = f32[1024]{0} exponential(p0)
  a = f32[] reduce(big, zero), dimensions={0}, to_apply=sum
  cp1s = (f32[1024]{0}, f32[1024]{0}) collective-permute-start(p1), source_target_pairs={{0,1},{1,0}}, control-predecessors={a}
  scratch = f32[1024]{0} exponential(p1)
  c = f32[] reduce(scratch, zero), dimensions={0}, to_apply=sum
  cp1d = f32[1024]{0} collective-permute-done(cp1s)
  b = f32[] reduce(big, zero), dimensions={0}, to_apply=sum
  ROOT root = (f32[], f32[], f32[], f32[1024]{0}) tuple(a, b, c, cp1d)
})hlo";

TEST_F(PostLhsMemorySchedulerTest, MovesParentConsumerAcrossWholeAsyncBlock) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kParentConsumer));
  ASSERT_OK_AND_ASSIGN(int64_t before, Peak(module.get()));
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_LT(Position(*module, "b"), Position(*module, "cp1s"));
  EXPECT_LT(Position(*module, "cp1s"), Position(*module, "scratch"));
  EXPECT_LT(Position(*module, "c"), Position(*module, "cp1d"));
  ASSERT_OK_AND_ASSIGN(int64_t after, Peak(module.get()));
  EXPECT_LT(after, before);
  RecordProperty("baseline_peak_bytes", std::to_string(before));
  RecordProperty("optimized_peak_bytes", std::to_string(after));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(PostLhsMemorySchedulerTest, DataAndControlEdgesPreventConsumerMotion) {
  std::string hlo = absl::StrReplaceAll(
      kParentConsumer,
      {{"b = f32[] reduce(big, zero), dimensions={0}, to_apply=sum",
        "b = f32[] reduce(big, zero), dimensions={0}, to_apply=sum, "
        "control-predecessors={cp1d}"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  auto before = Names(*module);
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(Names(*module), before);
}

TEST_F(PostLhsMemorySchedulerTest, ZeroBudgetPreservesBaseline) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kParentConsumer));
  auto before = Names(*module);
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get(), 0));
  EXPECT_FALSE(changed);
  EXPECT_EQ(Names(*module), before);
}

TEST_F(PostLhsMemorySchedulerTest, RequiresSchedule) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(absl::StrReplaceAll(
                           kParentConsumer, {{", is_scheduled=true", ""}})));
  EXPECT_THAT(RunScheduler(module.get()),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("requires an LHS schedule")));
}

TEST_F(PostLhsMemorySchedulerTest, KeepsConsumerInsideLaterDependentBlock) {
  std::string hlo = absl::StrReplaceAll(
      kParentConsumer,
      {{"  b = f32[] reduce(big, zero), dimensions={0}, to_apply=sum",
        "  cp2s = (f32[1024]{0}, f32[1024]{0}) collective-permute-start(cp1d), "
        "source_target_pairs={{0,1},{1,0}}\n"
        "  b = f32[] reduce(big, zero), dimensions={0}, to_apply=sum\n"
        "  cp2d = f32[1024]{0} collective-permute-done(cp2s)"},
       {"tuple(a, b, c, cp1d)", "tuple(a, b, c, cp2d)"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK(RunScheduler(module.get()).status());
  EXPECT_LT(Position(*module, "cp1d"), Position(*module, "cp2s"));
  EXPECT_LT(Position(*module, "cp2s"), Position(*module, "b"));
  EXPECT_LT(Position(*module, "b"), Position(*module, "cp2d"));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(PostLhsMemorySchedulerTest, CrossingIntervalsRemainOpaqueAndMovable) {
  std::string hlo = absl::StrReplaceAll(
      kParentConsumer,
      {{"  scratch =",
        "  cp2s = (f32[1024]{0}, f32[1024]{0}) collective-permute-start(p0), "
        "source_target_pairs={{0,1},{1,0}}\n  scratch ="},
       {"  b =",
        "  cp2d = f32[1024]{0} collective-permute-done(cp2s)\n  b ="}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_LT(Position(*module, "b"), Position(*module, "cp1s"));
  std::vector<std::string> names = Names(*module);
  int64_t begin = Position(*module, "cp1s");
  EXPECT_THAT(std::vector<std::string>(names.begin() + begin,
                                       names.begin() + begin + 6),
              ElementsAre("cp1s", "cp2s", "scratch", "c", "cp1d", "cp2d"));
}

TEST_F(PostLhsMemorySchedulerTest, NestedIntervalsRetainMembership) {
  std::string hlo = absl::StrReplaceAll(
      kParentConsumer,
      {{"  scratch =",
        "  cp2s = (f32[1024]{0}, f32[1024]{0}) collective-permute-start(p0), "
        "source_target_pairs={{0,1},{1,0}}\n  scratch ="},
       {"  cp1d =",
        "  cp2d = f32[1024]{0} collective-permute-done(cp2s)\n  cp1d ="}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_LT(Position(*module, "b"), Position(*module, "cp1s"));
  EXPECT_LT(Position(*module, "cp1s"), Position(*module, "cp2s"));
  EXPECT_LT(Position(*module, "cp2s"), Position(*module, "scratch"));
  EXPECT_LT(Position(*module, "c"), Position(*module, "cp2d"));
  EXPECT_LT(Position(*module, "cp2d"), Position(*module, "cp1d"));
}

TEST_F(PostLhsMemorySchedulerTest, AliasConsumerKeepsBufferLive) {
  std::string hlo = absl::StrReplaceAll(
      kParentConsumer,
      {{"  a =", "  alias = f32[1024]{0} bitcast(big)\n  a ="},
       {"b = f32[] reduce(big, zero)", "b = f32[] reduce(alias, zero)"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(int64_t before, Peak(module.get()));
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_LT(Position(*module, "alias"), Position(*module, "b"));
  EXPECT_LT(Position(*module, "b"), Position(*module, "cp1s"));
  ASSERT_OK_AND_ASSIGN(int64_t after, Peak(module.get()));
  EXPECT_LT(after, before);
}

TEST_F(PostLhsMemorySchedulerTest, OptimizesSiblingsInsideAnOuterAsyncRegion) {
  std::string hlo = absl::StrReplaceAll(
      kParentConsumer,
      {{"  big =",
        "  outer_start = (f32[1024]{0}, f32[1024]{0}) "
        "collective-permute-start(p0), source_target_pairs={{0,1},{1,0}}\n  "
        "big ="},
       {"  ROOT root =",
        "  outer_done = f32[1024]{0} collective-permute-done(outer_start)\n  "
        "ROOT root ="}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(int64_t before, Peak(module.get()));
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_LT(Position(*module, "outer_start"), Position(*module, "big"));
  EXPECT_LT(Position(*module, "b"), Position(*module, "cp1s"));
  EXPECT_LT(Position(*module, "cp1d"), Position(*module, "outer_done"));
  ASSERT_OK_AND_ASSIGN(int64_t after, Peak(module.get()));
  EXPECT_LT(after, before);
}

TEST_F(PostLhsMemorySchedulerTest, SinksEarlyProducerPastAnAsyncRegion) {
  constexpr absl::string_view hlo = R"hlo(
HloModule sink, is_scheduled=true
sum {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}
ENTRY entry {
  p = f32[] parameter(0)
  q = f32[1024]{0} parameter(1)
  zero = f32[] constant(0)
  big = f32[4096]{0} broadcast(p), dimensions={}
  start = (f32[1024]{0}, f32[1024]{0}) collective-permute-start(q), source_target_pairs={{0,1},{1,0}}
  scratch = f32[1024]{0} exponential(q)
  small = f32[] reduce(scratch, zero), dimensions={0}, to_apply=sum
  done = f32[1024]{0} collective-permute-done(start)
  consume = f32[] reduce(big, zero), dimensions={0}, to_apply=sum
  ROOT root = (f32[], f32[], f32[1024]{0}) tuple(consume, small, done)
})hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(int64_t before, Peak(module.get()));
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_LT(Position(*module, "done"), Position(*module, "big"));
  ASSERT_OK_AND_ASSIGN(int64_t after, Peak(module.get()));
  EXPECT_LT(after, before);
}

TEST_F(PostLhsMemorySchedulerTest, SchedulingAnnotationsPreserveBaseline) {
  std::string hlo = absl::StrReplaceAll(
      kParentConsumer,
      {{"exponential(p0)",
        "exponential(p0), frontend_attributes={_scheduling_group_id=\"0\"}"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  auto before = Names(*module);
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(Names(*module), before);
}

TEST_F(PostLhsMemorySchedulerTest, CustomCallBoundaryPreventsConsumerMotion) {
  std::string hlo = absl::StrReplaceAll(
      kParentConsumer, {{"  b =",
                         "  boundary = f32[] custom-call(zero), "
                         "custom_call_target=\"boundary\", "
                         "schedule=SCHEDULE_LATEST\n  b ="}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  auto before = Names(*module);
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(Names(*module), before);
}

TEST_F(PostLhsMemorySchedulerTest, OptimizesCalledComputation) {
  std::string hlo =
      absl::StrReplaceAll(kParentConsumer, {{"ENTRY entry", "worker"}});
  hlo += R"hlo(
ENTRY entry {
  p = f32[1024]{0} parameter(0)
  q = f32[1024]{0} parameter(1)
  ROOT call = (f32[], f32[], f32[], f32[1024]{0}) call(p, q), to_apply=worker
})hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(int64_t before, Peak(module.get()));
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(int64_t after, Peak(module.get()));
  EXPECT_LT(after, before);
  EXPECT_THAT(Names(*module), ElementsAre("p", "q", "call"));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(PostLhsMemorySchedulerTest, RejectsLocalGainWithoutWholeModuleGain) {
  std::string hlo =
      absl::StrReplaceAll(kParentConsumer, {{"ENTRY entry", "worker"}});
  hlo += R"hlo(
ENTRY entry {
  p = f32[1024]{0} parameter(0)
  q = f32[1024]{0} parameter(1)
  z = f32[] constant(0)
  call = (f32[], f32[], f32[], f32[1024]{0}) call(p, q), to_apply=worker
  large = f32[65536]{0} broadcast(z), dimensions={}, control-predecessors={call}
  reduced = f32[] reduce(large, z), dimensions={0}, to_apply=sum
  ROOT root = ((f32[], f32[], f32[], f32[1024]{0}), f32[]) tuple(call, reduced)
})hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  std::string before = module->schedule().ToString();
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(module->schedule().ToString(), before);
}

TEST_F(PostLhsMemorySchedulerTest, ASecondRunRetainsAnEqualPeakSchedule) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kParentConsumer));
  ASSERT_OK_AND_ASSIGN(bool first, RunScheduler(module.get()));
  ASSERT_TRUE(first);
  auto before = Names(*module);
  ASSERT_OK_AND_ASSIGN(bool second, RunScheduler(module.get()));
  EXPECT_FALSE(second);
  EXPECT_EQ(Names(*module), before);
}

// A single-array version of the worker can serve as either a loop body or
// a conditional branch. Its enclosing operator stays atomic in the caller.
std::string ControlFlowWorker() {
  return absl::StrReplaceAll(
      kParentConsumer,
      {{"ENTRY entry", "worker"},
       {"p1 = f32[1024]{0} parameter(1)", "p1 = f32[1024]{0} copy(p0)"},
       {"ROOT root = (f32[], f32[], f32[], f32[1024]{0}) tuple(a, b, c, cp1d)",
        "ab = f32[] add(a, b)\n"
        "abc = f32[] add(ab, c)\n"
        "broadcast = f32[1024]{0} broadcast(abc), dimensions={}\n"
        "ROOT root = f32[1024]{0} add(cp1d, broadcast)"}});
}

TEST_F(PostLhsMemorySchedulerTest, OptimizesWhileBodyWithAliasedLoopState) {
  std::string hlo = ControlFlowWorker() + R"hlo(
condition {
  p = f32[1024]{0} parameter(0)
  ROOT condition_result = pred[] constant(false)
}
ENTRY entry {
  initial_state = f32[1024]{0} parameter(0)
  ROOT loop = f32[1024]{0} while(initial_state), body=worker, condition=condition
})hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(int64_t before, Peak(module.get()));
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(int64_t after, Peak(module.get()));
  EXPECT_LT(after, before);
  EXPECT_THAT(Names(*module), ElementsAre("initial_state", "loop"));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(PostLhsMemorySchedulerTest, OptimizesConditionalBranch) {
  std::string hlo = ControlFlowWorker() + R"hlo(
identity {
  ROOT p = f32[1024]{0} parameter(0)
}
ENTRY entry {
  predicate = pred[] parameter(0)
  p = f32[1024]{0} parameter(1)
  ROOT branch = f32[1024]{0} conditional(predicate, p, p), true_computation=worker, false_computation=identity
})hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(int64_t before, Peak(module.get()));
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(int64_t after, Peak(module.get()));
  EXPECT_LT(after, before);
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(PostLhsMemorySchedulerTest, PreservesSynchronousCollectiveOrder) {
  std::string hlo = absl::StrReplaceAll(
      kParentConsumer, {{"  b =",
                         "  sync = f32[1024]{0} all-reduce(p0), "
                         "replica_groups={{0,1}}, to_apply=sum\n  b ="}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_LT(Position(*module, "b"), Position(*module, "cp1s"));
  EXPECT_LT(Position(*module, "cp1d"), Position(*module, "sync"));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(PostLhsMemorySchedulerTest, PreservesTokenAndEffectOrder) {
  std::string hlo = absl::StrReplaceAll(
      kParentConsumer,
      {{"  cp1s =",
        "  token0 = token[] after-all()\n"
        "  first = token[] outfeed(a, token0), outfeed_shape=f32[]\n  cp1s ="},
       {"  b =",
        "  second = token[] outfeed(c, first), outfeed_shape=f32[]\n  b ="}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_LT(Position(*module, "token0"), Position(*module, "first"));
  EXPECT_LT(Position(*module, "first"), Position(*module, "cp1s"));
  EXPECT_LT(Position(*module, "cp1d"), Position(*module, "second"));
  EXPECT_LT(Position(*module, "b"), Position(*module, "cp1s"));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(PostLhsMemorySchedulerTest, HandlesTokensAndEmptyArrays) {
  constexpr absl::string_view hlo = R"hlo(
HloModule empty, is_scheduled=true
ENTRY entry {
  p = f32[0]{0} parameter(0)
  token0 = token[] after-all()
  copy = f32[0]{0} copy(p)
  ROOT root = (f32[0]{0}, token[]) tuple(copy, token0)
})hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  auto before = Names(*module);
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(Names(*module), before);
}

std::string GenericAsyncHlo() {
  return absl::StrReplaceAll(
      kParentConsumer,
      {{"ENTRY entry {", R"hlo(
async_computation {
  p = f32[1024]{0} parameter(0)
  ROOT cp = f32[1024]{0} collective-permute(p), source_target_pairs={{0,1},{1,0}}
}
ENTRY entry {)hlo"},
       {"(f32[1024]{0}, f32[1024]{0}) collective-permute-start(p1), "
        "source_target_pairs={{0,1},{1,0}}",
        "((f32[1024]{0}), f32[1024]{0}, u32[]) async-start(p1), "
        "calls=async_computation"},
       {"collective-permute-done(cp1s)", "async-done(cp1s)"}});
}

TEST_F(PostLhsMemorySchedulerTest, GenericAsyncPairsRetainMembership) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(GenericAsyncHlo()));
  ASSERT_OK_AND_ASSIGN(int64_t before, Peak(module.get()));
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_LT(Position(*module, "b"), Position(*module, "cp1s"));
  EXPECT_LT(Position(*module, "cp1s"), Position(*module, "scratch"));
  EXPECT_LT(Position(*module, "c"), Position(*module, "cp1d"));
  ASSERT_OK_AND_ASSIGN(int64_t after, Peak(module.get()));
  EXPECT_LT(after, before);
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(PostLhsMemorySchedulerTest, UnsupportedAsyncUpdatesPreserveBaseline) {
  std::string hlo = absl::StrReplaceAll(
      GenericAsyncHlo(),
      {{"cp1d = f32[1024]{0} async-done(cp1s)",
        "update = ((f32[1024]{0}), f32[1024]{0}, u32[]) async-update(cp1s)\n"
        "cp1d = f32[1024]{0} async-done(update)"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  auto before = Names(*module);
  ASSERT_OK_AND_ASSIGN(bool changed, RunScheduler(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(Names(*module), before);
}

}  // namespace
}  // namespace xla
