/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/buffer_sharing.h"

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class BufferSharingTest : public HloTestBase {};

TEST_F(BufferSharingTest, ShareBufferExtraOutputReduction) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%maximum {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %res = f32[] maximum(%lhs, %rhs)
}

%fused_computation {
  %lhs = f32[3,40] parameter(0)
  %rhs = f32[3,40] parameter(1)
  %add = f32[3,40] add(%lhs, %rhs)
  %bc = f32[120] bitcast(%add)
  %init = f32[] constant(-inf)
  %max = f32[] reduce(%bc, %init), dimensions={0}, to_apply=%maximum
  ROOT %result = (f32[], f32[3,40]) tuple(%max, %add)
}

ENTRY %main {
  %lhs = f32[3,40] parameter(0)
  %rhs = f32[3,40] parameter(1)
  ROOT %fusion = (f32[], f32[3,40]) fusion(%lhs, %rhs),
      kind=kLoop, calls=%fused_computation
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(*FusionCanShareBufferHint(root, root->operand(0), {1}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
