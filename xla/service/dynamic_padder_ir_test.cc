/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/tests/xla_test_backend_predicates.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/llvm_irgen_test_base.h"

namespace xla {
namespace {

TEST_F(LlvmIrGenTestBase, LargeDynamicInput) {
  if (!test::DeviceTypeIs(test::kGpu)) {
    GTEST_SKIP();
  }
  const std::string hlo_text = R"( // NOLINT: Will be executed for GPU.
HloModule LargeDynamicInput

add (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY main {
  param = f32[<=20,<=20,<=20,<=20,<=20,<=20,<=20,<=20] parameter(0)
  zero = f32[] constant(0)
  ROOT out = reduce(param, zero), to_apply=add, dimensions={0,1,2,3,4,5,6,7}
}
)";
  HloModuleConfig config;
  CompileAndVerifyIr(hlo_text, R"(
; CHECK-LABEL: @input_reduce_fusion
; CHECK: }
)",
                     /*match_optimized_ir=*/true);
}

}  // namespace
}  // namespace xla
