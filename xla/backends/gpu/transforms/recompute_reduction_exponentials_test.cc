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

#include "xla/backends/gpu/transforms/recompute_reduction_exponentials.h"

#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_replace.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

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

using RecomputeReductionExponentialsTest = HloHardwareIndependentTestBase;

TEST_F(RecomputeReductionExponentialsTest,
       ReusesShiftedLogitsAndKeepsReduction) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloComputation* entry = module->entry_computation();
  HloInstruction* producer = entry->GetInstructionWithName("producer");
  HloInstruction* reduction =
      producer->fused_expression_root()->mutable_operand(0);
  HloInstruction* consumer = entry->root_instruction();
  HloComputation* body = consumer->fused_instructions_computation();
  HloInstruction* shifted = body->GetInstructionWithName("consumer_shifted");
  RecomputeReductionExponentials pass(/*min_bytes=*/0);
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

TEST_F(RecomputeReductionExponentialsTest, KeepsSmallSideOutput) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  RecomputeReductionExponentials pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RecomputeReductionExponentialsTest, RequiresIdenticalInputExpression) {
  std::string hlo = absl::StrReplaceAll(
      kHlo,
      {{"consumer_shifted = f32[2,256]{1,0} subtract(reshaped, max_broadcast)",
        "consumer_shifted = f32[2,256]{1,0} add(reshaped, max_broadcast)"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  RecomputeReductionExponentials pass(/*min_bytes=*/0);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RecomputeReductionExponentialsTest, RequiresSameOuterOperands) {
  std::string hlo = absl::StrReplaceAll(
      kHlo, {{"ROOT consumer =",
              "other_logits = bf16[512]{0} negate(logits)\n  ROOT consumer ="},
             {"fusion(saved, sum, logits, row_max)",
              "fusion(saved, sum, other_logits, row_max)"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  RecomputeReductionExponentials pass(/*min_bytes=*/0);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RecomputeReductionExponentialsTest, KeepsSharedSideOutput) {
  std::string hlo = absl::StrReplaceAll(
      kHlo, {{"ROOT consumer =",
              "extra = f32[2,256]{1,0} negate(saved)\n  ROOT consumer ="}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  RecomputeReductionExponentials pass(/*min_bytes=*/0);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RecomputeReductionExponentialsTest, KeepsDuplicateTupleExtraction) {
  std::string hlo = absl::StrReplaceAll(
      kHlo, {{"ROOT consumer =",
              "extra = f32[2,256]{1,0} get-tuple-element(producer), index=1\n  "
              "ROOT consumer ="}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  RecomputeReductionExponentials pass(/*min_bytes=*/0);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RecomputeReductionExponentialsTest, KeepsNonElementwiseConsumerUse) {
  std::string hlo = absl::StrReplaceAll(
      kHlo, {{"probabilities =",
              "reversed = f32[2,256]{1,0} reverse(saved_exponential), "
              "dimensions={1}\n  probabilities ="},
             {"divide(saved_exponential, sum_broadcast)",
              "divide(reversed, sum_broadcast)"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  RecomputeReductionExponentials pass(/*min_bytes=*/0);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RecomputeReductionExponentialsTest, KeepsControlDependencies) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloComputation* entry = module->entry_computation();
  ASSERT_OK(entry->GetInstructionWithName("producer")
                ->AddControlDependencyTo(entry->root_instruction()));
  RecomputeReductionExponentials pass(/*min_bytes=*/0);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

using RecomputeReductionExponentialsExecutionTest = HloPjRtGpuTestBase;

TEST_F(RecomputeReductionExponentialsExecutionTest, PreservesResults) {
  ASSERT_OK_AND_ASSIGN(auto before, ParseAndReturnVerifiedModule(kHlo));
  auto after = before->Clone();
  RecomputeReductionExponentials pass(/*min_bytes=*/0);
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(after.get()));
  ASSERT_TRUE(changed);
  // Bypass optimization so both sides execute their intended fusion boundaries.
  EXPECT_TRUE(RunAndCompareTwoModules(std::move(before), std::move(after),
                                      ErrorSpec{1e-5, 1e-5},
                                      /*run_hlo_passes=*/false));
}

}  // namespace
}  // namespace xla::gpu
