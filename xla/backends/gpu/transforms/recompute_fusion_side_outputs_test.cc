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

#include "xla/backends/gpu/transforms/recompute_fusion_side_outputs.h"

#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/decision.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

// Model the loss fusions with a split reduction and a BF16 logits gradient.
// The loss output already needs the shifted logits inside the consumer.
constexpr char kHlo[] = R"(
HloModule softmax_loss

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT sum = f32[] add(x, y)
}

exponential_reduction {
  logits = bf16[512]{0} parameter(0)
  row_max = f32[2]{0} parameter(1)
  converted = f32[512]{0} convert(logits)
  reshaped = f32[2,256]{1,0} bitcast(converted)
  max_broadcast = f32[2,256]{1,0} broadcast(row_max), dimensions={0}
  shifted = f32[2,256]{1,0} subtract(reshaped, max_broadcast)
  exponential = f32[2,256]{1,0} exponential(shifted)
  tiles = f32[2,2,128]{2,1,0} bitcast(exponential)
  zero = f32[] constant(0)
  partial_sum = f32[2,2]{1,0} reduce(tiles, zero), dimensions={2}, to_apply=add
  ROOT output = (f32[2,2]{1,0}, f32[2,256]{1,0}) tuple(partial_sum, exponential)
}

finish_reduction {
  partial = f32[2,2]{1,0} parameter(0)
  zero = f32[] constant(0)
  ROOT sum = f32[2]{0} reduce(partial, zero), dimensions={1}, to_apply=add
}

gradient_and_loss {
  saved_exponential = f32[2,256]{1,0} parameter(0)
  row_sum = f32[2]{0} parameter(1)
  logits = bf16[512]{0} parameter(2)
  row_max = f32[2]{0} parameter(3)
  converted = f32[512]{0} convert(logits)
  reshaped = f32[2,256]{1,0} bitcast(converted)
  max_broadcast = f32[2,256]{1,0} broadcast(row_max), dimensions={0}
  consumer_shifted = f32[2,256]{1,0} subtract(reshaped, max_broadcast)
  sum_broadcast = f32[2,256]{1,0} broadcast(row_sum), dimensions={0}
  probabilities = f32[2,256]{1,0} divide(saved_exponential, sum_broadcast)
  one = f32[] constant(1)
  target = f32[2,256]{1,0} broadcast(one), dimensions={}
  derivative = f32[2,256]{1,0} subtract(probabilities, target)
  gradient = bf16[2,256]{1,0} convert(derivative)
  zero = f32[] constant(0)
  loss = f32[2]{0} reduce(consumer_shifted, zero), dimensions={1}, to_apply=add
  ROOT output = (bf16[2,256]{1,0}, f32[2]{0}) tuple(gradient, loss)
}

ENTRY main {
  logits = bf16[512]{0} parameter(0)
  row_max = f32[2]{0} parameter(1)
  producer = (f32[2,2]{1,0}, f32[2,256]{1,0}) fusion(logits, row_max), kind=kInput, calls=exponential_reduction
  partial = f32[2,2]{1,0} get-tuple-element(producer), index=0
  saved = f32[2,256]{1,0} get-tuple-element(producer), index=1
  sum = f32[2]{0} fusion(partial), kind=kInput, calls=finish_reduction
  ROOT consumer = (bf16[2,256]{1,0}, f32[2]{0}) fusion(saved, sum, logits, row_max), kind=kInput, calls=gradient_and_loss
}
)";

// Structural/execution tests force the transform independently of hardware
// timings. Production always uses measured profitability.
Decision AllowForTest(const FusionRecomputeVariants&) {
  return Decision::Allow();
}

std::string WithTritonReduction() {
  return absl::StrReplaceAll(
      kHlo,
      {{"kind=kInput, calls=finish_reduction",
        R"(kind=kCustom, calls=finish_reduction, backend_config={"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],"num_warps":"4","num_ctas":1,"num_stages":1}}})"}});
}

using RecomputeFusionSideOutputsTest = HloHardwareIndependentTestBase;

TEST_F(RecomputeFusionSideOutputsTest, ReusesShiftedLogitsAndKeepsReduction) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloComputation* entry = module->entry_computation();
  HloInstruction* producer = entry->GetInstructionWithName("producer");
  HloInstruction* reduction =
      producer->fused_expression_root()->mutable_operand(0);
  HloInstruction* consumer = entry->root_instruction();
  HloComputation* body = consumer->fused_instructions_computation();
  HloInstruction* shifted = body->GetInstructionWithName("consumer_shifted");
  RecomputeFusionSideOutputs pass(AllowForTest, /*min_bytes=*/0);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(producer->fused_expression_root(), reduction);
  EXPECT_EQ(producer->shape(), reduction->shape());
  EXPECT_EQ(entry->GetInstructionWithName("saved"), nullptr);
  EXPECT_EQ(consumer->operand_count(), 3);
  HloInstruction* probabilities = body->GetInstructionWithName("probabilities");
  ASSERT_EQ(probabilities->operand(0)->opcode(), HloOpcode::kExp);
  EXPECT_EQ(probabilities->operand(0)->operand(0), shifted);
  ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RecomputeFusionSideOutputsTest, PreservesTritonIntermediate) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(WithTritonReduction()));
  int evaluated = 0;
  RecomputeFusionSideOutputs pass(
      [&](const FusionRecomputeVariants& v) {
        ++evaluated;
        for (auto* region : {v.before.get(), v.after.get()}) {
          EXPECT_EQ(
              absl::c_count_if(region->entry_computation()->instructions(),
                               [](HloInstruction* i) {
                                 return i->opcode() == HloOpcode::kFusion &&
                                        i->fusion_kind() ==
                                            HloInstruction::FusionKind::kCustom;
                               }),
              1);
        }
        return Decision::Allow();
      },
      0);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(evaluated, 1);
}

TEST_F(RecomputeFusionSideOutputsTest, KeepsSmallSideOutput) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  RecomputeFusionSideOutputs pass(AllowForTest);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RecomputeFusionSideOutputsTest, RequiresIdenticalInputExpression) {
  std::string hlo = absl::StrReplaceAll(
      kHlo,
      {{"consumer_shifted = f32[2,256]{1,0} subtract(reshaped, max_broadcast)",
        "consumer_shifted = f32[2,256]{1,0} add(reshaped, max_broadcast)"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  RecomputeFusionSideOutputs pass(AllowForTest, /*min_bytes=*/0);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RecomputeFusionSideOutputsTest, RequiresSameOuterOperands) {
  std::string hlo = absl::StrReplaceAll(
      kHlo, {{"ROOT consumer =",
              "other_logits = bf16[512]{0} negate(logits)\n  ROOT consumer ="},
             {"fusion(saved, sum, logits, row_max)",
              "fusion(saved, sum, other_logits, row_max)"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  RecomputeFusionSideOutputs pass(AllowForTest, /*min_bytes=*/0);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RecomputeFusionSideOutputsTest, KeepsSharedSideOutput) {
  std::string hlo = absl::StrReplaceAll(
      kHlo, {{"ROOT consumer =",
              "extra = f32[2,256]{1,0} negate(saved)\n  ROOT consumer ="}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  RecomputeFusionSideOutputs pass(AllowForTest, /*min_bytes=*/0);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RecomputeFusionSideOutputsTest, KeepsDuplicateTupleExtraction) {
  std::string hlo = absl::StrReplaceAll(
      kHlo, {{"ROOT consumer =",
              "extra = f32[2,256]{1,0} get-tuple-element(producer), index=1\n  "
              "ROOT consumer ="}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  RecomputeFusionSideOutputs pass(AllowForTest, /*min_bytes=*/0);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RecomputeFusionSideOutputsTest, KeepsNonElementwiseConsumerUse) {
  std::string hlo = absl::StrReplaceAll(
      kHlo, {{"probabilities =",
              "reversed = f32[2,256]{1,0} reverse(saved_exponential), "
              "dimensions={1}\n  probabilities ="},
             {"divide(saved_exponential, sum_broadcast)",
              "divide(reversed, sum_broadcast)"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  RecomputeFusionSideOutputs pass(AllowForTest, /*min_bytes=*/0);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RecomputeFusionSideOutputsTest, KeepsControlDependencies) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloComputation* entry = module->entry_computation();
  ASSERT_OK(entry->GetInstructionWithName("producer")
                ->AddControlDependencyTo(entry->root_instruction()));
  RecomputeFusionSideOutputs pass(AllowForTest, /*min_bytes=*/0);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RecomputeFusionSideOutputsTest, MissingOrNegativeCostKeepsGraph) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  RecomputeFusionSideOutputs missing({}, /*min_bytes=*/0);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&missing, module.get()));
  EXPECT_FALSE(changed);
  RecomputeFusionSideOutputs negative(
      [](const auto&) { return Decision::Forbid("measured regression"); }, 0);
  ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&negative, module.get()));
  EXPECT_FALSE(changed);
  EXPECT_NE(module->entry_computation()->GetInstructionWithName("saved"),
            nullptr);
}

TEST_F(RecomputeFusionSideOutputsTest, EvaluationFailureKeepsGraph) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  RecomputeFusionSideOutputs pass(
      [](const auto&) -> absl::StatusOr<Decision> {
        return absl::ResourceExhaustedError("profiling allocation failed");
      },
      0);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RecomputeFusionSideOutputsTest, AnalyzeOnlyAndBudgetDoNotRewrite) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  int evaluated = 0;
  auto evaluate = [&](const auto&) {
    ++evaluated;
    return Decision::Allow();
  };
  RecomputeFusionSideOutputs analyze(evaluate, 0, 16, /*analyze_only=*/true);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&analyze, module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(evaluated, 1);
  RecomputeFusionSideOutputs no_budget(evaluate, 0, 0);
  ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&no_budget, module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(evaluated, 1);
}

TEST_F(RecomputeFusionSideOutputsTest, RegionPreservesDependenciesAndOutputs) {
  // Keep an additional use of the partial reduction outside the region.
  std::string hlo =
      absl::StrReplaceAll(kHlo, {{"ROOT consumer =", "consumer ="},
                                 {"calls=gradient_and_loss",
                                  "calls=gradient_and_loss\n ROOT result = "
                                  "((bf16[2,256]{1,0}, f32[2]{0}), "
                                  "f32[2,2]{1,0}) tuple(consumer, partial)"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  int evaluated = 0;
  RecomputeFusionSideOutputs pass(
      [&](const FusionRecomputeVariants& v) {
        ++evaluated;
        EXPECT_EQ(v.eliminated_bytes, 2 * 256 * sizeof(float));
        EXPECT_EQ(v.before->entry_computation()->num_parameters(), 2);
        EXPECT_EQ(v.after->entry_computation()->num_parameters(), 2);
        EXPECT_TRUE(ShapeUtil::Equal(v.before->result_shape(),
                                     v.after->result_shape()));
        for (auto* region : {v.before.get(), v.after.get()}) {
          EXPECT_EQ(
              absl::c_count_if(region->entry_computation()->instructions(),
                               [](HloInstruction* i) {
                                 return i->opcode() == HloOpcode::kFusion;
                               }),
              3);
        }
        EXPECT_EQ(
            v.before->entry_computation()->root_instruction()->operand_count(),
            2);
        EXPECT_NE(
            v.before->entry_computation()->GetInstructionWithName("saved"),
            nullptr);
        EXPECT_EQ(v.after->entry_computation()->GetInstructionWithName("saved"),
                  nullptr);
        return Decision::Allow();
      },
      0);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(evaluated, 1);
}

constexpr char kLoopHlo[] = R"(
HloModule loop_side_output
producer_body {
  x = f32[256]{0} parameter(0)
  y = f32[256]{0} parameter(1)
  side = f32[256]{0} add(x, y)
  other = f32[256]{0} negate(x)
  ROOT outputs = (f32[256]{0}, f32[256]{0}) tuple(other, side)
}
consumer_body {
  saved = f32[256]{0} parameter(0)
  x = f32[256]{0} parameter(1)
  y = f32[256]{0} parameter(2)
  product = f32[256]{0} multiply(x, y)
  ROOT output = f32[256]{0} add(saved, product)
}
ENTRY main {
  x = f32[256]{0} parameter(0)
  y = f32[256]{0} parameter(1)
  producer = (f32[256]{0}, f32[256]{0}) fusion(x, y), kind=kLoop, calls=producer_body
  other = f32[256]{0} get-tuple-element(producer), index=0
  saved = f32[256]{0} get-tuple-element(producer), index=1
  consumer = f32[256]{0} fusion(saved, x, y), kind=kLoop, calls=consumer_body
  ROOT result = (f32[256]{0}, f32[256]{0}) tuple(other, consumer)
}
)";

TEST_F(RecomputeFusionSideOutputsTest, BinaryLoopSideOutputsAcrossDtypes) {
  for (const char* dtype : {"f16", "bf16", "f32", "f64", "s32"}) {
    SCOPED_TRACE(dtype);
    ASSERT_OK_AND_ASSIGN(auto module,
                         ParseAndReturnVerifiedModule(
                             absl::StrReplaceAll(kLoopHlo, {{"f32", dtype}})));
    RecomputeFusionSideOutputs pass(AllowForTest, 0);
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
    ASSERT_TRUE(changed);
    auto* consumer =
        module->entry_computation()->GetInstructionWithName("consumer");
    EXPECT_EQ(consumer->operand_count(), 2);
    EXPECT_EQ(consumer->fused_expression_root()->operand(0)->opcode(),
              HloOpcode::kAdd);
  }
}

TEST_F(RecomputeFusionSideOutputsTest, UnaryOperationsBeyondExponential) {
  for (const char* operation : {"negate", "sine", "sqrt", "tanh"}) {
    SCOPED_TRACE(operation);
    ASSERT_OK_AND_ASSIGN(
        auto module,
        ParseAndReturnVerifiedModule(absl::StrReplaceAll(
            kLoopHlo, {{"add(x, y)", std::string(operation) + "(x)"},
                       {"other = f32[256]{0} negate(x)",
                        "other = f32[256]{0} negate(y)"}})));
    RecomputeFusionSideOutputs pass(AllowForTest, 0);
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
    EXPECT_TRUE(changed);
  }
}

using RecomputeFusionSideOutputsExecutionTest = HloPjRtGpuTestBase;

TEST_F(RecomputeFusionSideOutputsExecutionTest, PreservesResults) {
  ASSERT_OK_AND_ASSIGN(auto before, ParseAndReturnVerifiedModule(kHlo));
  auto after = before->Clone();
  RecomputeFusionSideOutputs pass(AllowForTest, /*min_bytes=*/0);
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(after.get()));
  ASSERT_TRUE(changed);
  // Bypass optimization so both sides execute their intended fusion boundaries.
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(before), std::move(after),
                                      ErrorSpec{1e-5, 1e-5},
                                      /*run_hlo_passes=*/false));
}

TEST_F(RecomputeFusionSideOutputsExecutionTest,
       PreservesResultsWithTritonIntermediate) {
  ASSERT_OK_AND_ASSIGN(auto before,
                       ParseAndReturnVerifiedModule(WithTritonReduction()));
  auto after = before->Clone();
  RecomputeFusionSideOutputs pass(AllowForTest, 0);
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(after.get()));
  ASSERT_TRUE(changed);
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(before), std::move(after),
                                      ErrorSpec{1e-5, 1e-5}, false));
}

TEST_F(RecomputeFusionSideOutputsExecutionTest,
       PreservesBinaryResultsAcrossDtypes) {
  for (const char* dtype : {"f16", "bf16", "f32", "f64", "s32"}) {
    SCOPED_TRACE(dtype);
    ASSERT_OK_AND_ASSIGN(auto before,
                         ParseAndReturnVerifiedModule(
                             absl::StrReplaceAll(kLoopHlo, {{"f32", dtype}})));
    auto after = before->Clone();
    RecomputeFusionSideOutputs pass(AllowForTest, 0);
    ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(after.get()));
    ASSERT_TRUE(changed);
    EXPECT_TRUE(RunAndCompareTwoModules(std::move(before), std::move(after),
                                        ErrorSpec{0, 0}, false));
  }
}

}  // namespace
}  // namespace xla::gpu
