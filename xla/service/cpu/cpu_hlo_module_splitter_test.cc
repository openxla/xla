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

#include "xla/service/cpu/cpu_hlo_module_splitter.h"

#include <memory>
#include <vector>

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::cpu {
namespace {

using CpuHloModuleSplitterTest = HloHardwareIndependentTestBase;

TEST_F(CpuHloModuleSplitterTest, SplitNonInlineableComputation) {
  const char* hlo_string = R"(
HloModule module
callee {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}
ENTRY entry {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT call = f32[] call(p0, p1), to_apply=callee, frontend_attributes={inlineable="false"}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  CpuHloModuleSplitter splitter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  EXPECT_TRUE(changed);

  for (auto& submodule : splitter.submodules()) {
    EXPECT_TRUE(
        HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
            .Run(submodule.get())
            .ok());
  }

  const HloInstruction* root = module->entry_computation()->root_instruction();
  // Expecting: custom-call
  EXPECT_EQ(root->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(root->custom_call_target(), "__xla_cpu_multi_module_call");
  EXPECT_EQ(root->raw_backend_config_string(), "callee");

  EXPECT_EQ(splitter.submodules().size(), 1);
  EXPECT_EQ(splitter.submodules()[0]->name(), "callee");
  EXPECT_EQ(splitter.submodules()[0]
                ->entry_computation()
                ->root_instruction()
                ->opcode(),
            HloOpcode::kAdd);

  EXPECT_TRUE(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .ok());
}

TEST_F(CpuHloModuleSplitterTest, NestedNonInlineableComputations) {
  const char* hlo_string = R"(
HloModule module
inner {
  p0 = f32[] parameter(0)
  ROOT neg = f32[] negate(p0)
}
outer {
  p0 = f32[] parameter(0)
  ROOT call = f32[] call(p0), to_apply=inner, frontend_attributes={inlineable="false"}
}
ENTRY entry {
  p0 = f32[] parameter(0)
  ROOT call = f32[] call(p0), to_apply=outer, frontend_attributes={inlineable="false"}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  CpuHloModuleSplitter splitter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  EXPECT_TRUE(changed);

  for (auto& submodule : splitter.submodules()) {
    EXPECT_TRUE(
        HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
            .Run(submodule.get())
            .ok());
  }

  // entry should call outer (CustomCall)
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(root->custom_call_target(), "__xla_cpu_multi_module_call");
  EXPECT_EQ(root->raw_backend_config_string(), "outer");

  // We should have 2 submodules: inner and outer.
  EXPECT_EQ(splitter.submodules().size(), 2);

  // Find outer module
  HloModule* outer_mod = nullptr;
  for (const auto& m : splitter.submodules()) {
    if (m->name() == "outer") {
      outer_mod = m.get();
    }
  }
  ASSERT_NE(outer_mod, nullptr);

  // outer module should call inner (CustomCall)
  const HloInstruction* outer_root =
      outer_mod->entry_computation()->root_instruction();
  EXPECT_EQ(outer_root->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(outer_root->custom_call_target(), "__xla_cpu_multi_module_call");
  EXPECT_EQ(outer_root->raw_backend_config_string(), "inner");

  EXPECT_TRUE(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .ok());
}

TEST_F(CpuHloModuleSplitterTest, SideEffectComputation) {
  const char* hlo_string = R"(
HloModule module
callee {
  p0 = f32[] parameter(0)
  ROOT outfeed = token[] outfeed(p0, token[] after-all()), outfeed_config="abc"
}
ENTRY entry {
  p0 = f32[] parameter(0)
  ROOT call = token[] call(p0), to_apply=callee, frontend_attributes={inlineable="false"}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  CpuHloModuleSplitter splitter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  EXPECT_TRUE(changed);

  for (auto& submodule : splitter.submodules()) {
    EXPECT_TRUE(
        HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
            .Run(submodule.get())
            .ok());
  }

  const HloInstruction* root = module->entry_computation()->root_instruction();
  // Expecting NO copy because it's a token.
  EXPECT_EQ(root->opcode(), HloOpcode::kCustomCall);
  const auto* custom_call = Cast<HloCustomCallInstruction>(root);
  EXPECT_TRUE(custom_call->custom_call_has_side_effect());

  EXPECT_TRUE(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .ok());
}

TEST_F(CpuHloModuleSplitterTest, ReduceComputation) {
  const char* hlo_string = R"(
HloModule module
reducer {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}
callee {
  p0 = f32[10] parameter(0)
  init = f32[] constant(0.0)
  ROOT reduce = f32[] reduce(p0, init), dimensions={0}, to_apply=reducer
}
ENTRY entry {
  p0 = f32[10] parameter(0)
  ROOT call = f32[] call(p0), to_apply=callee, frontend_attributes={inlineable="false"}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  CpuHloModuleSplitter splitter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  EXPECT_TRUE(changed);

  for (auto& submodule : splitter.submodules()) {
    EXPECT_TRUE(
        HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
            .Run(submodule.get())
            .ok());
  }
}

}  // namespace
}  // namespace xla::cpu
