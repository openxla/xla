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

#include "xla/service/hlo_early_buffer_release.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/buffer_value.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"

namespace xla {
namespace {

// dlogits feeds a critical-path GEMM and a delayed weight-gradient GEMM.
// expand/middle stand in for unrelated backward work retaining scratch memory.
constexpr absl::string_view kSplitConsumers = R"(
HloModule split_consumers, is_scheduled=true

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT out = f32[] add(a, b)
}

ENTRY main {
  p = f32[64,128] parameter(0)
  w = f32[128,4] parameter(1)
  h = f32[64,4] parameter(2)
  zero = f32[] constant(0)
  dlogits = f32[64,128] add(p, p)
  dhidden = f32[64,4] dot(dlogits, w), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  expand = f32[64,4,32] broadcast(dhidden), dimensions={0,1}
  middle = f32[64,4,32] tanh(expand)
  dweights = f32[4,128] dot(h, dlogits), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  result = f32[] reduce(middle, zero), dimensions={0,1,2}, to_apply=sum
  ROOT out = (f32[4,128], f32[]) tuple(dweights, result)
})";

int64_t Bytes(const BufferValue& value) {
  return ShapeUtil::ByteSizeOf(value.shape(), 8);
}

int64_t Position(absl::Span<const HloInstruction* const> sequence,
                 absl::string_view name) {
  for (int64_t i = 0; i < sequence.size(); ++i) {
    if (sequence[i]->name() == name) {
      return i;
    }
  }
  return -1;
}

class HloEarlyBufferReleaseTest : public HloHardwareIndependentTestBase {
 protected:
  HloEarlyBufferRelease::Options Options() {
    HloEarlyBufferRelease::Options options;
    options.min_buffer_bytes = 1;
    options.max_iterations = 1;
    return options;
  }

  absl::StatusOr<bool> RunRelease(HloModule* module,
                                  HloEarlyBufferRelease::Options options) {
    HloEarlyBufferRelease pass(
        &alias_info_, Bytes,
        [](const HloComputation*, absl::Span<const HloInstruction* const>) {
          return 1.0;
        },
        options);
    return pass.Run(module);
  }

  absl::StatusOr<int64_t> Peak(HloModule* module) {
    auto alias_analysis = HloAliasAnalysis::Run(module, &alias_info_);
    if (!alias_analysis.ok()) {
      return alias_analysis.status();
    }
    BufferValue::SizeFunction size = Bytes;
    return HeapSimulator::MinimumMemoryForModule(
        module->schedule(), **alias_analysis, &alias_info_, &size);
  }

  int64_t Pos(HloModule* module, absl::string_view name) {
    return Position(
        module->schedule().sequence(module->entry_computation()).instructions(),
        name);
  }

  AliasInfo alias_info_;
};

TEST_F(HloEarlyBufferReleaseTest, HoistsFinalConsumerAndReducesHeapPeak) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kSplitConsumers));
  ASSERT_OK_AND_ASSIGN(int64_t before, Peak(module.get()));
  ASSERT_OK_AND_ASSIGN(bool changed, RunRelease(module.get(), Options()));
  EXPECT_TRUE(changed);
  EXPECT_LT(Pos(module.get(), "dhidden"), Pos(module.get(), "dweights"));
  EXPECT_LT(Pos(module.get(), "dweights"), Pos(module.get(), "expand"));
  EXPECT_OK(module->schedule().Verify());
  ASSERT_OK_AND_ASSIGN(int64_t after, Peak(module.get()));
  EXPECT_LT(after, before);
}

TEST_F(HloEarlyBufferReleaseTest, HonorsThreshold) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kSplitConsumers));
  auto options = Options();
  options.min_buffer_bytes = 1 << 30;
  ASSERT_OK_AND_ASSIGN(bool changed, RunRelease(module.get(), options));
  EXPECT_FALSE(changed);
}

TEST_F(HloEarlyBufferReleaseTest, RejectsEstimatedSlowdown) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kSplitConsumers));
  HloEarlyBufferRelease pass(
      &alias_info_, Bytes,
      [](const HloComputation*,
         absl::Span<const HloInstruction* const> sequence) {
        return Position(sequence, "dweights") < Position(sequence, "middle")
                   ? 2.0
                   : 1.0;
      },
      Options());
  ASSERT_OK(pass.Run(module.get()).status());
  EXPECT_GT(Pos(module.get(), "dweights"), Pos(module.get(), "middle"));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(HloEarlyBufferReleaseTest, PreservesControlDependencies) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kSplitConsumers));
  HloComputation* computation = module->entry_computation();
  ASSERT_OK(
      computation->GetInstructionWithName("middle")->AddControlDependencyTo(
          computation->GetInstructionWithName("dweights")));
  ASSERT_OK(RunRelease(module.get(), Options()).status());
  EXPECT_LT(Pos(module.get(), "middle"), Pos(module.get(), "dweights"));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(HloEarlyBufferReleaseTest, LeavesSchedulingGroupsIntact) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kSplitConsumers));
  module->entry_computation()
      ->GetInstructionWithName("middle")
      ->set_frontend_attribute(kXlaSchedulingGroupIdAttr, "1");
  ASSERT_OK_AND_ASSIGN(bool changed, RunRelease(module.get(), Options()));
  EXPECT_FALSE(changed);
}

TEST_F(HloEarlyBufferReleaseTest, MovesPrerequisitesAndMultipleConsumers) {
  std::string hlo(kSplitConsumers);
  const std::string original = "  dweights = f32[4,128] dot(h, dlogits)";
  hlo.replace(hlo.find(original), original.size(),
              "  extra = f32[64,4] negate(h)\n"
              "  dweights = f32[4,128] dot(extra, dlogits)");
  const std::string root =
      "  ROOT out = (f32[4,128], f32[]) tuple(dweights, result)";
  hlo.replace(
      hlo.find(root), root.size(),
      "  other = f32[] reduce(dlogits, zero), dimensions={0,1}, to_apply=sum\n"
      "  ROOT out = (f32[4,128], f32[], f32[]) tuple(dweights, result, other)");
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(int64_t before, Peak(module.get()));
  ASSERT_OK_AND_ASSIGN(bool changed, RunRelease(module.get(), Options()));
  EXPECT_TRUE(changed);
  EXPECT_LT(Pos(module.get(), "extra"), Pos(module.get(), "dweights"));
  EXPECT_LT(Pos(module.get(), "dweights"), Pos(module.get(), "expand"));
  EXPECT_LT(Pos(module.get(), "other"), Pos(module.get(), "expand"));
  ASSERT_OK_AND_ASSIGN(int64_t after, Peak(module.get()));
  EXPECT_LT(after, before);
}

TEST_F(HloEarlyBufferReleaseTest, LooksThroughExpandingConversion) {
  std::string hlo(kSplitConsumers);
  const std::string original = "  dweights = f32[4,128] dot(h, dlogits)";
  hlo.replace(hlo.find(original), original.size(),
              "  wide = f64[64,128] convert(dlogits)\n"
              "  narrow = f32[64,128] convert(wide)\n"
              "  dweights = f32[4,128] dot(h, narrow)");
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  auto options = Options();
  options.max_iterations = 4;
  ASSERT_OK_AND_ASSIGN(bool changed, RunRelease(module.get(), options));
  EXPECT_TRUE(changed);
  EXPECT_LT(Pos(module.get(), "dweights"), Pos(module.get(), "expand"));
  EXPECT_LT(Pos(module.get(), "wide"), Pos(module.get(), "narrow"));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(HloEarlyBufferReleaseTest, RejectsNewTransientPeak) {
  std::string hlo(kSplitConsumers);
  // middle has no longer-lived result at the proposed insertion point, but
  // making the final consumer expand dlogits would overlap expand and middle.
  const std::string original =
      "  dweights = f32[4,128] dot(h, dlogits), lhs_contracting_dims={0}, "
      "rhs_contracting_dims={0}";
  hlo.replace(
      hlo.find(original), original.size(),
      "  dweights = f32[4,64,128] broadcast(dlogits), dimensions={1,2}");
  hlo.replace(hlo.find("ROOT out = (f32[4,128]"),
              std::string("ROOT out = (f32[4,128]").size(),
              "ROOT out = (f32[4,64,128]");
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(int64_t before, Peak(module.get()));
  ASSERT_OK(RunRelease(module.get(), Options()).status());
  EXPECT_GT(Pos(module.get(), "dweights"), Pos(module.get(), "middle"));
  ASSERT_OK_AND_ASSIGN(int64_t after, Peak(module.get()));
  EXPECT_LE(after, before);
}

TEST_F(HloEarlyBufferReleaseTest, DoesNotTreatLiveOutAsReleasable) {
  std::string hlo(kSplitConsumers);
  const std::string root =
      "  ROOT out = (f32[4,128], f32[]) tuple(dweights, result)";
  hlo.replace(hlo.find(root), root.size(),
              "  ROOT out = (f32[4,128], f32[], f32[64,128]) tuple(dweights, "
              "result, dlogits)");
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK(RunRelease(module.get(), Options()).status());
  EXPECT_GT(Pos(module.get(), "dweights"), Pos(module.get(), "middle"));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(HloEarlyBufferReleaseTest, DoesNotTreatParameterAliasAsReleasable) {
  std::string hlo(kSplitConsumers);
  const std::string definition = "dlogits = f32[64,128] add(p, p)";
  hlo.replace(hlo.find(definition), definition.size(),
              "dlogits = f32[64,128] bitcast(p)");
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK(RunRelease(module.get(), Options()).status());
  EXPECT_GT(Pos(module.get(), "dweights"), Pos(module.get(), "middle"));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(HloEarlyBufferReleaseTest, KeepsAsyncConsumersInPlace) {
  std::string hlo(kSplitConsumers);
  const std::string marker = "  dweights =";
  hlo.insert(hlo.find(marker),
             "  start = (f32[64,128], f32[64,128]) "
             "collective-permute-start(dlogits), source_target_pairs={{0,0}}\n"
             "  done = f32[64,128] collective-permute-done(start)\n");
  const std::string root =
      "  ROOT out = (f32[4,128], f32[]) tuple(dweights, result)";
  hlo.replace(hlo.find(root), root.size(),
              "  ROOT out = (f32[4,128], f32[], f32[64,128]) tuple(dweights, "
              "result, done)");
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK(RunRelease(module.get(), Options()).status());
  EXPECT_GT(Pos(module.get(), "start"), Pos(module.get(), "middle"));
  EXPECT_GT(Pos(module.get(), "done"), Pos(module.get(), "start"));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(HloEarlyBufferReleaseTest, DoesNotCrossExplicitlyEarlyCustomCall) {
  std::string hlo(kSplitConsumers);
  const std::string original = "middle = f32[64,4,32] tanh(expand)";
  hlo.replace(hlo.find(original), original.size(),
              "middle = f32[64,4,32] custom-call(expand), "
              "custom_call_target=\"test_middle\", schedule=SCHEDULE_EARLIEST");
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK(RunRelease(module.get(), Options()).status());
  EXPECT_GT(Pos(module.get(), "dweights"), Pos(module.get(), "middle"));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(HloEarlyBufferReleaseTest, DoesNotHoistExplicitlyLateCustomCall) {
  std::string hlo(kSplitConsumers);
  const std::string original =
      "dweights = f32[4,128] dot(h, dlogits), lhs_contracting_dims={0}, "
      "rhs_contracting_dims={0}";
  hlo.replace(hlo.find(original), original.size(),
              "dweights = f32[4,128] custom-call(h, dlogits), "
              "custom_call_target=\"test_gemm\", schedule=SCHEDULE_LATEST");
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK(RunRelease(module.get(), Options()).status());
  EXPECT_GT(Pos(module.get(), "dweights"), Pos(module.get(), "middle"));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(HloEarlyBufferReleaseTest, BoundsPrerequisiteClosure) {
  std::string hlo(kSplitConsumers);
  const std::string original = "  dweights = f32[4,128] dot(h, dlogits)";
  hlo.replace(hlo.find(original), original.size(),
              "  extra = f32[64,4] negate(h)\n"
              "  dweights = f32[4,128] dot(extra, dlogits)");
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  auto options = Options();
  options.max_group_size = 1;
  ASSERT_OK(RunRelease(module.get(), options).status());
  EXPECT_GT(Pos(module.get(), "dweights"), Pos(module.get(), "middle"));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(HloEarlyBufferReleaseTest, RequiresScheduledModule) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kSplitConsumers));
  module->clear_schedule();
  EXPECT_THAT(RunRelease(module.get(), Options()),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace xla
