/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/allreduce_softmax_fusion.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/pattern_matcher.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::_;
namespace m = ::xla::match;

class AllReduceSoftmaxFusionTest : public HloTestBase {
 protected:
  AllReduceSoftmaxFusion fusion_pass_;
};

TEST_F(AllReduceSoftmaxFusionTest, PassRunsWithoutError) {
  const std::string hlo_string = R"(
HloModule test_module

add_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(arg_0, arg_1)
}

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

ENTRY main {
  param = f32[32]{0} parameter(0)
  all_reduce = f32[32]{0} all-reduce(param), replica_groups={{0,1,2,3}}, to_apply=add_computation
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[] reduce(all_reduce, constant_neg_inf), dimensions={0}, to_apply=max_computation
  broadcast = f32[32]{0} broadcast(reduce), dimensions={}
  ROOT subtract = f32[32]{0} subtract(all_reduce, broadcast)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  
  // The pass should run without error
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fusion_pass_.Run(module.get()));
  
  // For this simple case without Triton softmax, no fusion should occur
  EXPECT_FALSE(changed);
  
  // Verify the module is still valid
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
}

TEST_F(AllReduceSoftmaxFusionTest, PassRunsOnEmptyModule) {
  const std::string hlo_string = R"(
HloModule empty_module

ENTRY main {
  ROOT constant = f32[] constant(1.0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  
  // The pass should run without error on an empty module
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fusion_pass_.Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
}

TEST_F(AllReduceSoftmaxFusionTest, NoFusionWithoutTritonSoftmax) {
  const std::string hlo_string = R"(
HloModule test_module

add_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(arg_0, arg_1)
}

ENTRY main {
  param = f32[32]{0} parameter(0)
  all_reduce = f32[32]{0} all-reduce(param), replica_groups={{0,1,2,3}}, to_apply=add_computation
  ROOT exponential = f32[32]{0} exponential(all_reduce)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  
  // Should not fuse when there's no Triton softmax fusion
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fusion_pass_.Run(module.get()));
  EXPECT_FALSE(changed);
  
  // Verify all-reduce is still present
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Exp(m::AllReduce(m::Parameter(0)))));
}

TEST_F(AllReduceSoftmaxFusionTest, NoFusionWithMultipleUsers) {
  const std::string hlo_string = R"(
HloModule test_module

add_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(arg_0, arg_1)
}

ENTRY main {
  param = f32[32]{0} parameter(0)
  all_reduce = f32[32]{0} all-reduce(param), replica_groups={{0,1,2,3}}, to_apply=add_computation
  exponential = f32[32]{0} exponential(all_reduce)
  ROOT add_user = f32[32]{0} add(all_reduce, exponential)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  
  // Should not fuse when all-reduce has multiple users
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fusion_pass_.Run(module.get()));
  EXPECT_FALSE(changed);
  
  // Verify all-reduce is still present with multiple users
  const HloInstruction* all_reduce = FindInstruction(module.get(), "all_reduce");
  ASSERT_NE(all_reduce, nullptr);
  EXPECT_EQ(all_reduce->user_count(), 2);
}

TEST_F(AllReduceSoftmaxFusionTest, FusionPassRegistration) {
  // Test that the pass has the correct name
  EXPECT_EQ(fusion_pass_.name(), "allreduce-softmax-fusion");
}

// Test with a simple mock Triton softmax-like structure
TEST_F(AllReduceSoftmaxFusionTest, SimpleAllReducePattern) {
  const std::string hlo_string = R"(
HloModule test_module

add_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(arg_0, arg_1)
}

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

ENTRY main {
  param = f32[32]{0} parameter(0)
  all_reduce = f32[32]{0} all-reduce(param), replica_groups={{0,1,2,3}}, to_apply=add_computation
  constant_neg_inf = f32[] constant(-inf)
  reduce_max = f32[] reduce(all_reduce, constant_neg_inf), dimensions={0}, to_apply=max_computation
  broadcast_max = f32[32]{0} broadcast(reduce_max), dimensions={}
  subtract = f32[32]{0} subtract(all_reduce, broadcast_max)
  exponential = f32[32]{0} exponential(subtract)
  constant_zero = f32[] constant(0)
  reduce_sum = f32[] reduce(exponential, constant_zero), dimensions={0}, to_apply=add_computation
  broadcast_sum = f32[32]{0} broadcast(reduce_sum), dimensions={}
  ROOT divide = f32[32]{0} divide(exponential, broadcast_sum)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  
  // Count instructions before transformation
  int original_instruction_count = module->entry_computation()->instruction_count();
  
  // Run the pass
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fusion_pass_.Run(module.get()));
  
  // For this test, we expect no change since there's no actual Triton fusion
  // The pass should only work with actual Triton softmax fusions
  EXPECT_FALSE(changed);
  
  // Verify module is still valid
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  
  // Instruction count should remain the same
  EXPECT_EQ(module->entry_computation()->instruction_count(), original_instruction_count);
}

}  // namespace
}  // namespace gpu
}  // namespace xla 