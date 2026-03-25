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

#include "xla/hlo/transforms/collectives/collective_permute_cse.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

using CollectivePermuteCseTest = HloHardwareIndependentTestBase;

TEST_F(CollectivePermuteCseTest, RemovesIdenticalPermutes) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[100] parameter(0)
  c1 = f32[100] collective-permute(p0), channel_id=1, source_target_pairs={{0,1}}
  c2 = f32[100] collective-permute(p0), channel_id=1, source_target_pairs={{0,1}}
  ROOT root = (f32[100], f32[100]) tuple(c1, c2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  CollectivePermuteCse pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::CollectivePermute(op::Parameter(0)),
                              op::CollectivePermute(op::Parameter(0))));
  EXPECT_EQ(root->operand(0), root->operand(1));
}

TEST_F(CollectivePermuteCseTest, SortedPairsMatch) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[100] parameter(0)
  c1 = f32[100] collective-permute(p0), channel_id=1, source_target_pairs={{0,1}, {1,2}}
  c2 = f32[100] collective-permute(p0), channel_id=1, source_target_pairs={{1,2}, {0,1}}
  ROOT root = (f32[100], f32[100]) tuple(c1, c2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  CollectivePermuteCse pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::CollectivePermute(op::Parameter(0)),
                              op::CollectivePermute(op::Parameter(0))));
  EXPECT_EQ(root->operand(0), root->operand(1));
}

TEST_F(CollectivePermuteCseTest, SubsetPermuteSlices) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[100] parameter(0)
  slice0 = f32[50] slice(p0), slice={[0:50]}
  c1 = f32[100] collective-permute(p0), channel_id=1, source_target_pairs={{0,1}}
  c2 = f32[50] collective-permute(slice0), channel_id=1, source_target_pairs={{0,1}}
  ROOT root = (f32[100], f32[50]) tuple(c1, c2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  CollectivePermuteCse pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              op::Tuple(op::CollectivePermute(op::Parameter(0)),
                        op::Slice(op::CollectivePermute(op::Parameter(0)))));
  EXPECT_EQ(root->operand(1)->operand(0), root->operand(0));
}

TEST_F(CollectivePermuteCseTest, SubsetPermuteSlicesSmallBeforeLarge) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[100] parameter(0)
  slice0 = f32[50] slice(p0), slice={[0:50]}
  c2 = f32[50] collective-permute(slice0), channel_id=1, source_target_pairs={{0,1}}
  c1 = f32[100] collective-permute(p0), channel_id=1, source_target_pairs={{0,1}}
  ROOT root = (f32[100], f32[50]) tuple(c1, c2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  CollectivePermuteCse pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              op::Tuple(op::CollectivePermute(op::Parameter(0)),
                        op::Slice(op::CollectivePermute(op::Parameter(0)))));
  EXPECT_EQ(root->operand(1)->operand(0), root->operand(0));
}

TEST_F(CollectivePermuteCseTest,
       SubsetPermuteSlicesSmallBeforeLargeReachability) {
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[100] parameter(0)
  slice0 = f32[50] slice(p0), slice={[0:50]}
  c2 = f32[50] collective-permute(slice0), channel_id=1, source_target_pairs={{0,1}}
  c1 = f32[100] collective-permute(p0), channel_id=1, source_target_pairs={{0,1}}, control-predecessors={c2}
  ROOT root = (f32[100], f32[50]) tuple(c1, c2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  CollectivePermuteCse pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              op::Tuple(op::CollectivePermute(op::Parameter(0)),
                        op::Slice(op::CollectivePermute(op::Parameter(0)))));
  EXPECT_EQ(root->operand(1)->operand(0), root->operand(0));
}

}  // namespace
}  // namespace xla
