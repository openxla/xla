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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_replace.h"
#include "xla/backends/gpu/transforms/recompute_fusion_side_outputs.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/decision.h"
#include "xla/service/hlo_module_config.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

constexpr char kHlo[] = R"(
HloModule recomputation_policy
producer_body {
  x = f32[256]{0} parameter(0)
  side = f32[256]{0} exponential(x)
  other = f32[256]{0} negate(x)
  ROOT outputs = (f32[256]{0}, f32[256]{0}) tuple(other, side)
}
consumer_body {
  saved = f32[256]{0} parameter(0)
  x = f32[256]{0} parameter(1)
  ROOT output = f32[256]{0} add(saved, x)
}
ENTRY main {
  x = f32[256]{0} parameter(0)
  producer = (f32[256]{0}, f32[256]{0}) fusion(x), kind=kLoop, calls=producer_body
  other = f32[256]{0} get-tuple-element(producer), index=0
  saved = f32[256]{0} get-tuple-element(producer), index=1
  consumer = f32[256]{0} fusion(saved, x), kind=kLoop, calls=consumer_body
  ROOT result = (f32[256]{0}, f32[256]{0}) tuple(other, consumer)
}
)";

class RecomputeFusionSideOutputsPolicyTest
    : public HloHardwareIndependentTestBase,
      public ::testing::WithParamInterface<bool> {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug =
        HloHardwareIndependentTestBase::GetDebugOptionsForTest();
    debug.set_xla_gpu_experimental_fusion_recomputation(GetParam() ? 1 : 2);
    debug.set_xla_gpu_autotune_level(4);
    debug.set_xla_gpu_deterministic_ops(false);
    debug.set_xla_gpu_exclude_nondeterministic_ops(false);
    debug.set_xla_gpu_require_complete_aot_autotune_results(false);
    return debug;
  }

  void ExpectEvaluation(const DebugOptions& debug, bool expect_evaluation) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
    module->mutable_config().set_debug_options(debug);
    const std::string original = module->ToString();
    int evaluated = 0;
    RecomputeFusionSideOutputs pass(
        [&](const FusionRecomputeVariants&) {
          ++evaluated;
          return Decision::Allow();
        },
        /*min_bytes=*/0, /*max_candidates=*/1, /*analyze_only=*/GetParam());
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
    EXPECT_EQ(evaluated, expect_evaluation ? 1 : 0);
    EXPECT_EQ(changed, expect_evaluation && !GetParam());
    if (!expect_evaluation || GetParam()) {
      EXPECT_EQ(module->ToString(), original);
    }
  }
};

TEST_P(RecomputeFusionSideOutputsPolicyTest,
       EvaluatesWhenLiveAutotuningAllowed) {
  ExpectEvaluation(GetDebugOptionsForTest(), /*expect_evaluation=*/true);
}

TEST_P(RecomputeFusionSideOutputsPolicyTest,
       PreservesDeadCodeInExcludedExecutionThread) {
  const std::string hlo = absl::StrReplaceAll(
      kHlo,
      {{"ENTRY main", R"(
other_thread {
  x = f32[256]{0} parameter(0)
  dead = f32[256]{0} negate(x)
  ROOT output = f32[256]{0} copy(x)
}
ENTRY main)"},
       {"  ROOT result = (f32[256]{0}, f32[256]{0}) tuple(other, consumer)",
        R"(
  start = ((f32[256]{0}), f32[256]{0}, s32[]) async-start(x), calls=other_thread, async_execution_thread="other"
  done = f32[256]{0} async-done(start)
  ROOT result = (f32[256]{0}, f32[256]{0}, f32[256]{0}) tuple(other, consumer, done))"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  HloComputation* other_thread = module->GetComputationWithName("other_thread");
  ASSERT_NE(other_thread, nullptr);
  ASSERT_EQ(other_thread->execution_thread(), "other");
  ASSERT_NE(other_thread->GetInstructionWithName("dead"), nullptr);
  const std::string original = other_thread->ToString();
  int evaluated = 0;
  RecomputeFusionSideOutputs pass(
      [&](const FusionRecomputeVariants&) {
        ++evaluated;
        return Decision::Allow();
      },
      /*min_bytes=*/0, /*max_candidates=*/1, /*analyze_only=*/GetParam());
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      pass.Run(module.get(), {HloInstruction::kMainExecutionThread}));
  EXPECT_EQ(evaluated, 1);
  EXPECT_EQ(changed, !GetParam());
  EXPECT_EQ(other_thread->ToString(), original);
  EXPECT_EQ(
      module->entry_computation()->GetInstructionWithName("saved") == nullptr,
      !GetParam());
}

TEST_P(RecomputeFusionSideOutputsPolicyTest, SkipsWhenAutotuningDisabled) {
  for (int level : {0, -1}) {
    SCOPED_TRACE(level);
    DebugOptions debug = GetDebugOptionsForTest();
    debug.set_xla_gpu_autotune_level(level);
    ExpectEvaluation(debug, /*expect_evaluation=*/false);
  }
}

TEST_P(RecomputeFusionSideOutputsPolicyTest,
       SkipsWhenDeterministicOpsRequested) {
  DebugOptions debug = GetDebugOptionsForTest();
  debug.set_xla_gpu_deterministic_ops(true);
  ExpectEvaluation(debug, /*expect_evaluation=*/false);
}

TEST_P(RecomputeFusionSideOutputsPolicyTest,
       SkipsWhenNondeterministicOpsExcluded) {
  DebugOptions debug = GetDebugOptionsForTest();
  debug.set_xla_gpu_exclude_nondeterministic_ops(true);
  ExpectEvaluation(debug, /*expect_evaluation=*/false);
}

TEST_P(RecomputeFusionSideOutputsPolicyTest,
       SkipsWhenCompleteCachedResultsRequired) {
  DebugOptions debug = GetDebugOptionsForTest();
  debug.set_xla_gpu_require_complete_aot_autotune_results(true);
  ExpectEvaluation(debug, /*expect_evaluation=*/false);
}

INSTANTIATE_TEST_SUITE_P(AnalyzeAndApply, RecomputeFusionSideOutputsPolicyTest,
                         ::testing::Bool());

}  // namespace
}  // namespace xla::gpu
