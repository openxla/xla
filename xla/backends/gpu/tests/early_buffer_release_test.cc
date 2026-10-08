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

#include <algorithm>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/buffer_value.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/hlo_early_buffer_release.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_interpreter_reference_mixin.h"

namespace xla::gpu {
namespace {

using EarlyBufferReleaseTest = HloInterpreterReferenceMixin<GpuPjRtCodegenTest>;

TEST_F(EarlyBufferReleaseTest, MovedConsumerPreservesResults) {
  const char* hlo = R"(
HloModule release, is_scheduled=true
producer {
  p = f32[256,256]{1,0} parameter(0)
  ROOT out = f32[256,256]{1,0} add(p, p)
}
first {
  p = f32[256,256]{1,0} parameter(0)
  ROOT out = f32[4,256]{1,0} slice(p), slice={[0:4],[0:256]}
}
expand_fn {
  p = f32[4,256]{1,0} parameter(0)
  ROOT out = f32[4,256,256]{2,1,0} broadcast(p), dimensions={0,1}
}
middle_fn {
  p = f32[4,256,256]{2,1,0} parameter(0)
  ROOT out = f32[4,256,256]{2,1,0} tanh(p)
}
last {
  p = f32[256,256]{1,0} parameter(0)
  ROOT out = f32[1,256]{1,0} slice(p), slice={[0:1],[0:256]}
}
result_fn {
  p = f32[4,256,256]{2,1,0} parameter(0)
  ROOT out = f32[1,1,1]{2,1,0} slice(p), slice={[0:1],[0:1],[0:1]}
}
ENTRY main {
  p = f32[256,256]{1,0} parameter(0)
  big = f32[256,256]{1,0} fusion(p), kind=kLoop, calls=producer
  early = f32[4,256]{1,0} fusion(big), kind=kLoop, calls=first
  expand = f32[4,256,256]{2,1,0} fusion(early), kind=kLoop, calls=expand_fn
  middle = f32[4,256,256]{2,1,0} fusion(expand), kind=kLoop, calls=middle_fn
  late = f32[1,256]{1,0} fusion(big), kind=kLoop, calls=last
  result = f32[1,1,1]{2,1,0} fusion(middle), kind=kLoop, calls=result_fn
  ROOT out = (f32[1,256]{1,0}, f32[1,1,1]{2,1,0}) tuple(late, result)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(hlo));
  GpuAliasInfo alias_info(device_description());
  HloEarlyBufferRelease::Options options;
  options.min_buffer_bytes = 1;
  options.max_iterations = 1;
  HloEarlyBufferRelease pass(
      &alias_info,
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 8);
      },
      [](const HloComputation*, absl::Span<const HloInstruction* const>) {
        return 1.0;
      },
      options);
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  ASSERT_TRUE(changed);
  const auto& sequence =
      module->schedule().sequence(module->entry_computation()).instructions();
  EXPECT_LT(
      std::find(sequence.begin(), sequence.end(),
                module->entry_computation()->GetInstructionWithName("late")),
      std::find(sequence.begin(), sequence.end(),
                module->entry_computation()->GetInstructionWithName("expand")));
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(std::move(module), ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace xla::gpu
