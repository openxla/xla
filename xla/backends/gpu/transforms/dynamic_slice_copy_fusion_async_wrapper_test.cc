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

#include "xla/backends/gpu/transforms/dynamic_slice_copy_fusion_async_wrapper.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/transforms/fusion_dynamic_memcpy_rewriter.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla::gpu {
namespace {

using DynamicSliceCopyFusionAsyncWrapperTest = HloHardwareIndependentTestBase;

TEST_F(DynamicSliceCopyFusionAsyncWrapperTest, WrapsDynamicMemcpyFusion) {
  constexpr char kHlo[] = R"(
    dynamic_slice {
      p0 = s32[4] parameter(0)
      c1 = s32[] constant(1)

      ROOT slice = s32[1] dynamic-slice(p0, c1), dynamic_slice_sizes={1},
          backend_config={"dynamic_slice_config":
              {"byte_offset":"4","byte_stride":"0"}}
    }

    ENTRY main {
      p0 = s32[4] parameter(0)
      ROOT fusion = s32[1] fusion(p0), kind=kLoop, calls=dynamic_slice
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  EXPECT_THAT(FusionDynamicMemcpyRewriter().Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  DynamicSliceCopyFusionAsyncWrapper wrapper;
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(true));

  HloInstruction* async_done = module->entry_computation()->root_instruction();
  ASSERT_EQ(async_done->opcode(), HloOpcode::kAsyncDone);
  HloInstruction* async_start = async_done->mutable_operand(0);
  ASSERT_EQ(async_start->opcode(), HloOpcode::kAsyncStart);

  HloInstruction* wrapped = async_start->async_wrapped_instruction();
  ASSERT_EQ(wrapped->opcode(), HloOpcode::kFusion);
  EXPECT_TRUE(IsCopyHeroDynamicSliceFusion(wrapped));

  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(false));
}

TEST_F(DynamicSliceCopyFusionAsyncWrapperTest, DoesNotWrapNonCopyHeroFusion) {
  constexpr char kHlo[] = R"(
    dsf_computation {
      p0 = s32[4] parameter(0)
      c1 = s32[] constant(1)
      ds = s32[1] dynamic-slice(p0, c1), dynamic_slice_sizes={1},
          backend_config={"dynamic_slice_config":
              {"byte_offset":"4","byte_stride":"0"}}
      ROOT add = s32[1] add(ds, ds)
    }

    ENTRY main {
      p0 = s32[4] parameter(0)
      ROOT fusion = s32[1] fusion(p0), kind=kCustom, calls=dsf_computation,
          backend_config={"fusion_backend_config":{"kind":"__custom_fusion",
              "custom_fusion_config":{"name":"dynamic_slice_fusion"}}}
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  DynamicSliceCopyFusionAsyncWrapper wrapper;
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(false));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kFusion);
}

}  // namespace
}  // namespace xla::gpu
