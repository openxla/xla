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

#include "xla/backends/gpu/transforms/constant_fill_sinking.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/substitute.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

class ConstantFillSinkingExecutionTest
    : public HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>,
      public ::testing::WithParamInterface<int> {};

TEST_P(ConstantFillSinkingExecutionTest,
       PreservesInitializedAndUpdatedElements) {
  // Two loop iterations transform an input row, which then overwrites part of
  // the filled array. Check both the updated row and the untouched elements
  // against the interpreter. This test needs only one GPU.
  std::string hlo = absl::Substitute(R"(
HloModule fill_after_loop, is_scheduled=true
fill_body {
  value = f32[] constant($0)
  ROOT broadcast = f32[512,512]{1,0} broadcast(value), dimensions={}
}
condition {
  p = (s32[], f32[1,512]{1,0}) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  limit = s32[] constant(2)
  ROOT test = pred[] compare(i, limit), direction=LT
}
negate_row {
  p = f32[1,512]{1,0} parameter(0)
  ROOT negated = f32[1,512]{1,0} negate(p)
}
body {
  p = (s32[], f32[1,512]{1,0}) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  body_row = f32[1,512]{1,0} get-tuple-element(p), index=1
  one = s32[] constant(1)
  next = s32[] add(i, one)
  negated = f32[1,512]{1,0} fusion(body_row), kind=kLoop, calls=negate_row
  ROOT body_result = (s32[], f32[1,512]{1,0}) tuple(next, negated)
}
update_body {
  buffer = f32[512,512]{1,0} parameter(0)
  update_row = f32[1,512]{1,0} parameter(1)
  update_zero = s32[] constant(0)
  ROOT update = f32[512,512]{1,0} dynamic-update-slice(buffer, update_row, update_zero, update_zero)
}
ENTRY main {
  row = f32[1,512]{1,0} parameter(0)
  zero = s32[] constant(0)
  initial = (s32[], f32[1,512]{1,0}) tuple(zero, row)
  fill = f32[512,512]{1,0} fusion(), kind=kLoop, calls=fill_body
  loop = (s32[], f32[1,512]{1,0}) while(initial), condition=condition, body=body
  updated_row = f32[1,512]{1,0} get-tuple-element(loop), index=1
  ROOT result = f32[512,512]{1,0} fusion(fill, updated_row), kind=kLoop, calls=update_body
}
)",
                                     GetParam());
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  auto original = module->Clone();
  ASSERT_OK_AND_ASSIGN(bool changed, ConstantFillSinking().Run(module.get()));
  ASSERT_TRUE(changed);
  ASSERT_OK(module->schedule().Verify());
  std::vector<std::string> names;
  for (const HloInstruction* instruction :
       module->schedule()
           .sequence(module->entry_computation())
           .instructions()) {
    names.emplace_back(instruction->name());
  }
  EXPECT_THAT(names, ::testing::ElementsAre("row", "zero", "initial", "loop",
                                            "updated_row", "fill", "result"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec{0, 0}));

  // Compile the original schedule as well, to exercise the post-scheduling
  // pipeline hookup rather than only invoking the pass directly.
  ASSERT_OK_AND_ASSIGN(
      auto executable,
      CreateExecutable(std::move(original), /*run_hlo_passes=*/false));
  ASSERT_OK_AND_ASSIGN(const HloModule* compiled,
                       test_runner().HloModuleFromWrapped(executable.get()));
  ASSERT_TRUE(compiled->has_schedule());
  names.clear();
  for (const HloInstruction* instruction :
       compiled->schedule()
           .sequence(compiled->entry_computation())
           .instructions()) {
    names.emplace_back(instruction->name());
  }
  auto loop = std::find(names.begin(), names.end(), "loop");
  auto fill = std::find(names.begin(), names.end(), "fill");
  ASSERT_NE(loop, names.end());
  ASSERT_NE(fill, names.end());
  EXPECT_LT(loop, fill);
}

INSTANTIATE_TEST_SUITE_P(ZeroAndNonzero, ConstantFillSinkingExecutionTest,
                         ::testing::Values(0, 7));

}  // namespace
}  // namespace xla::gpu
