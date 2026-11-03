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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/autotuner/fusion_recomputation.h"
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

// Use a non-splat, non-integer constant large enough that abbreviated literal
// printing would hide changes to it.
std::string RegionHlo() {
  std::vector<std::string> values(32, "0.125");
  values.back() = "0.25";
  return absl::StrReplaceAll(R"(
HloModule layer
layer_producer {
  x = f32[2,32]{1,0} parameter(0)
  y = f32[2,32]{1,0} exponential(x)
  values = f32[2,32]{1,0} constant({{VALUES}, {VALUES}})
  z = f32[2,32]{1,0} add(x, values)
  ROOT outputs = (f32[2,32]{1,0}, f32[2,32]{1,0}) tuple(y, z)
}
layer_consumer {
  y = f32[2,32]{1,0} parameter(0)
  x = f32[2,32]{1,0} parameter(1)
  ROOT result = f32[2,32]{1,0} multiply(y, x)
}
ENTRY main {
  x = f32[2,32]{1,0} parameter(0)
  p = (f32[2,32]{1,0}, f32[2,32]{1,0}) fusion(x), kind=kLoop, calls=layer_producer
  y = f32[2,32]{1,0} get-tuple-element(p), index=0
  z = f32[2,32]{1,0} get-tuple-element(p), index=1
  c = f32[2,32]{1,0} fusion(y, x), kind=kLoop, calls=layer_consumer
  ROOT result = (f32[2,32]{1,0}, f32[2,32]{1,0}) tuple(c, z)
}
)",
                             {{"VALUES", absl::StrJoin(values, ", ")}});
}

class FusionRecomputationCacheTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<FusionRecomputeVariants> Extract(absl::string_view hlo) {
    ABSL_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo));
    HloComputation* entry = module->entry_computation();
    HloInstruction* producer = entry->GetInstructionWithName("p");
    HloInstruction* consumer = entry->GetInstructionWithName("c");
    return ExtractFusionRecomputeVariants(
        {producer,
         consumer,
         consumer->mutable_operand(0),
         producer->fused_expression_root()->mutable_operand(0),
         {consumer->fused_parameter(1)}});
  }
};

TEST_F(FusionRecomputationCacheTest,
       ReusesSeparatelyExtractedEquivalentRegions) {
  ASSERT_OK_AND_ASSIGN(auto first, Extract(RegionHlo()));
  ASSERT_OK_AND_ASSIGN(
      auto renamed,
      Extract(absl::StrReplaceAll(RegionHlo(), {{"layer", "different_layer.17"},
                                                {"main", "entry.42"}})));
  ASSERT_NE(first.before->ToString(), renamed.before->ToString());
  int evaluations = 0;
  auto evaluate =
      CacheFusionRecomputationEvaluator([&](const FusionRecomputeVariants&) {
        ++evaluations;
        return Decision::Allow();
      });
  ASSERT_OK_AND_ASSIGN(auto first_decision, evaluate(first));
  ASSERT_OK_AND_ASSIGN(auto renamed_decision, evaluate(renamed));
  EXPECT_TRUE(first_decision.IsAllowed());
  EXPECT_TRUE(renamed_decision.IsAllowed());
  EXPECT_EQ(evaluations, 1);
}

TEST_F(FusionRecomputationCacheTest, DistinguishesConstantsLayoutsAndConfigs) {
  const std::string original = RegionHlo();
  const std::vector<std::string> variants = {
      original, absl::StrReplaceAll(original, {{"0.25", "0.375"}}),
      absl::StrReplaceAll(original, {{"{1,0}", "{0,1}"}}),
      absl::StrReplaceAll(
          original,
          {{"calls=layer_consumer",
            R"(calls=layer_consumer, backend_config={"native_emitter_backend_config":{"unroll_factor":2}})"}})};
  int evaluations = 0;
  auto evaluate =
      CacheFusionRecomputationEvaluator([&](const FusionRecomputeVariants&) {
        ++evaluations;
        return Decision::Forbid("measured regression");
      });
  for (const std::string& hlo : variants) {
    ASSERT_OK_AND_ASSIGN(auto region, Extract(hlo));
    ASSERT_OK_AND_ASSIGN(auto decision, evaluate(region));
    EXPECT_TRUE(decision.IsForbidden());
    EXPECT_THAT(decision.Explain(),
                ::testing::HasSubstr("measured regression"));
  }
  EXPECT_EQ(evaluations, variants.size());
  for (const std::string& hlo : variants) {
    ASSERT_OK_AND_ASSIGN(auto region, Extract(hlo));
    ASSERT_OK_AND_ASSIGN(auto decision, evaluate(region));
    EXPECT_TRUE(decision.IsForbidden());
  }
  EXPECT_EQ(evaluations, variants.size());
}

TEST_F(FusionRecomputationCacheTest, DistinguishesBothGraphsAndDebugOptions) {
  ASSERT_OK_AND_ASSIGN(auto original, Extract(RegionHlo()));
  ASSERT_OK_AND_ASSIGN(auto different, Extract(absl::StrReplaceAll(
                                           RegionHlo(), {{"0.25", "0.375"}})));
  int evaluations = 0;
  auto evaluate =
      CacheFusionRecomputationEvaluator([&](const FusionRecomputeVariants&) {
        ++evaluations;
        return Decision::Allow();
      });
  EXPECT_OK(evaluate(original).status());
  FusionRecomputeVariants changed_before{different.before->Clone(),
                                         original.after->Clone(),
                                         original.eliminated_bytes};
  EXPECT_OK(evaluate(changed_before).status());
  FusionRecomputeVariants changed_after{original.before->Clone(),
                                        different.after->Clone(),
                                        original.eliminated_bytes};
  EXPECT_OK(evaluate(changed_after).status());
  EXPECT_EQ(evaluations, 3);
  for (auto* module : {original.before.get(), original.after.get()}) {
    auto& debug = module->mutable_config().mutable_debug_options();
    debug.set_xla_gpu_enable_fast_min_max(!debug.xla_gpu_enable_fast_min_max());
    EXPECT_OK(evaluate(original).status());
  }
  EXPECT_EQ(evaluations, 5);
}

TEST_F(FusionRecomputationCacheTest, IgnoresDebugOptionMapInsertionOrder) {
  ASSERT_OK_AND_ASSIGN(auto first, Extract(RegionHlo()));
  ASSERT_OK_AND_ASSIGN(auto reordered, Extract(RegionHlo()));
  auto& first_options = *first.before->mutable_config()
                             .mutable_debug_options()
                             .mutable_xla_backend_extra_options();
  first_options.clear();
  first_options["a"] = "1";
  first_options["b"] = "2";
  auto& reordered_options = *reordered.before->mutable_config()
                                 .mutable_debug_options()
                                 .mutable_xla_backend_extra_options();
  reordered_options.clear();
  reordered_options["b"] = "2";
  reordered_options["a"] = "1";
  int evaluations = 0;
  auto evaluate =
      CacheFusionRecomputationEvaluator([&](const FusionRecomputeVariants&) {
        ++evaluations;
        return Decision::Allow();
      });
  EXPECT_OK(evaluate(first).status());
  EXPECT_OK(evaluate(reordered).status());
  EXPECT_EQ(evaluations, 1);
  reordered_options["a"] = "3";
  EXPECT_OK(evaluate(reordered).status());
  EXPECT_EQ(evaluations, 2);
}

TEST_F(FusionRecomputationCacheTest, PreservesExecutionThreads) {
  ASSERT_OK_AND_ASSIGN(auto region, Extract(RegionHlo()));
  int evaluations = 0;
  auto evaluate =
      CacheFusionRecomputationEvaluator([&](const FusionRecomputeVariants&) {
        ++evaluations;
        return Decision::Allow();
      });
  EXPECT_OK(evaluate(region).status());
  region.before->entry_computation()->SetExecutionThread("other");
  EXPECT_OK(evaluate(region).status());
  EXPECT_EQ(evaluations, 2);
}

TEST_F(FusionRecomputationCacheTest, DoesNotCacheEvaluationErrors) {
  ASSERT_OK_AND_ASSIGN(auto region, Extract(RegionHlo()));
  int evaluations = 0;
  auto evaluate = CacheFusionRecomputationEvaluator(
      [&](const FusionRecomputeVariants&) -> absl::StatusOr<Decision> {
        if (++evaluations == 1) {
          return absl::ResourceExhaustedError("profiling allocation failed");
        }
        return Decision::Allow();
      });
  EXPECT_EQ(evaluate(region).status().code(),
            absl::StatusCode::kResourceExhausted);
  ASSERT_OK_AND_ASSIGN(auto decision, evaluate(region));
  EXPECT_TRUE(decision.IsAllowed());
  EXPECT_EQ(evaluations, 2);
  EXPECT_OK(evaluate(region).status());
  EXPECT_EQ(evaluations, 2);
}

}  // namespace
}  // namespace xla::gpu
