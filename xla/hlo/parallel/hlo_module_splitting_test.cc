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

#include "xla/hlo/parallel/hlo_module_splitting.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parallel/hlo_module_linking.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"

namespace xla::parallel {
namespace {

using ::testing::UnorderedElementsAreArray;

class SplittingTest : public HloHardwareIndependentTestBase {};

TEST_F(SplittingTest, AllComputationsInBuckets) {
  constexpr absl::string_view module_text = R"(
HloModule simple_module


// Simple alpha equivalence examples

%add.0 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

%add.1 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

%add.2 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

%add.3 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

ENTRY %main() -> f32[] {

  %p = f32[] constant(3.3)
  %q = f32[] constant(-1.0)
  %r = f32[] constant(7.1)
  %s = f32[] constant(0.2)

  %res.0 = call(%p, %q), to_apply=%add.0
  %res.1 = call(%p, %q), to_apply=%add.1
  %res.2 = call(%r, %s), to_apply=%add.2
  %res.3 = call(%p, %s), to_apply=%add.3

  %res.01 = f32[] add(f32[] %res.0, f32[] %res.1)
  %res.23 = f32[] add(f32[] %res.2, f32[] %res.3)

  ROOT %result = f32[] add(f32[] %res.01, f32[] %res.23)
})";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(module_text));
  ASSERT_OK_AND_ASSIGN(auto splits, GroupComputationsForSplitting(*module));

  absl::flat_hash_set<const HloComputation*> all_computations_in_splits;
  for (const auto& bucket : splits) {
    all_computations_in_splits.insert(bucket.begin(), bucket.end());
  }
  EXPECT_THAT(all_computations_in_splits,
              UnorderedElementsAreArray(module->computations()));
}

TEST_F(SplittingTest, CreateModule) {
  constexpr absl::string_view module_text = R"(
HloModule simple_module


// Simple alpha equivalence examples

%add.0 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

%add.1 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

%add.2 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

%add.3 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

ENTRY %main() -> f32[] {

  %p = f32[] constant(3.3)
  %q = f32[] constant(-1.0)
  %r = f32[] constant(7.1)
  %s = f32[] constant(0.2)

  %res.0 = call(%p, %q), to_apply=%add.0
  %res.1 = call(%p, %q), to_apply=%add.1
  %res.2 = call(%r, %s), to_apply=%add.2
  %res.3 = call(%p, %s), to_apply=%add.3

  %res.01 = f32[] add(f32[] %res.0, f32[] %res.1)
  %res.23 = f32[] add(f32[] %res.2, f32[] %res.3)

  ROOT %result = f32[] add(f32[] %res.01, f32[] %res.23)
})";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(module_text));
  auto* main = FindComputation(module.get(), "main");

  ASSERT_OK_AND_ASSIGN(auto module_split,
                       CreateHloModuleSplit(*module, {main}));

  const int kMainModuleComputationCount = 5;  // the main + 4 stubs
  // const int kMainModuleClonedComputations = 1;
  const int kMainModuleCallSitesCount = 4;
  EXPECT_EQ(module_split->module.computation_count(),
            kMainModuleComputationCount);
  EXPECT_EQ(module_split->stub_map.size(), kMainModuleCallSitesCount);
  EXPECT_EQ(module_split->call_sites.size(), kMainModuleCallSitesCount);
}

TEST_F(SplittingTest, SplitEntryNameDoesNotCollideWithClonedComputation) {
  // A fusion computation is cloned into the split with its name intact, so it
  // can already hold the name the split gives its entry.
  constexpr absl::string_view module_text = R"(
HloModule entry_name_clash

entry {
  x = f32[] parameter(0)
  ROOT neg = f32[] negate(x)
}

ENTRY main {
  p = f32[] parameter(0)
  ROOT f = f32[] fusion(p), kind=kLoop, calls=entry
})";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(module_text));
  auto* main = FindComputation(module.get(), "main");

  ASSERT_OK_AND_ASSIGN(auto module_split,
                       CreateHloModuleSplit(*module, {main}));

  EXPECT_EQ(module_split->submodule->computation_count(), 2);
  EXPECT_OK(module_split->submodule
                ->CheckUniqueNamesAndIdsForComputationsAndInstructions());
}

TEST_F(SplittingTest, SplitModule) {
  constexpr absl::string_view module_text = R"(
HloModule simple_module


// Simple alpha equivalence examples

%fusion.1 (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %add = f32[] negate(f32[] %x)
}

%add.0 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

%add.1 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

%add.2 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

%fusion.2 (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %add = f32[] negate(f32[] %x)
}

ENTRY %main() -> f32[] {

  %p = f32[] constant(3.3)
  %q = f32[] constant(-1.0)
  %r = f32[] constant(7.1)
  %s = f32[] constant(0.2)

  %res.0 = call(%p, %q), to_apply=%add.0
  %res.1 = call(%p, %q), to_apply=%add.1
  %res.2 = call(%r, %s), to_apply=%add.2
  %res.3 = call(%p, %s), to_apply=%add.2
  %fusion = f32[] fusion(%res.3), kind=kLoop, calls=%fusion.2

  %res.01 = f32[] add(f32[] %res.0, f32[] %res.1)
  %res.23 = f32[] add(f32[] %res.2, f32[] %res.3)
  %res.fu23 = f32[] add(f32[] %fusion, f32[] %res.23)

  ROOT %result = f32[] add(f32[] %res.01, f32[] %res.fu23)
})";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(module_text));
  ASSERT_OK_AND_ASSIGN(auto module_split_group,
                       CreateHloModuleSplitGroup(*module));

  ASSERT_OK_AND_ASSIGN(stream_executor::Platform * platform,
                       PlatformUtil::GetPlatform("cpu"));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Compiler> compiler,
                       Compiler::GetForPlatform(platform->id()));
  ASSERT_OK(compiler->RunHloPasses(module->Clone(), /*executor=*/nullptr,
                                   Compiler::CompileOptions{}));
  for (auto& split : module_split_group->module_splits) {
    EXPECT_OK(compiler->RunHloPasses(split->submodule->Clone(),
                                     /*executor=*/nullptr,
                                     Compiler::CompileOptions{}));
  }
}

TEST_F(SplittingTest, SplitDiamondGraphModule) {
  constexpr absl::string_view module_text = R"(
    HloModule shared_callee_module

    %y {
      %p = f32[] parameter(0)
      ROOT %result = f32[] exponential(%p)
    }

    %x {
      %p = f32[] parameter(0)
      %call_y = f32[] call(%p), to_apply=%y
      ROOT %result = f32[] cosine(%call_y)
    }

    %a {
      %p = f32[] parameter(0)
      %c = f32[] constant(5.0)
      %call_x = f32[] call(%p), to_apply=%x
      ROOT %result = f32[] add(%call_x, %c)
    }

    %b {
      %p = f32[] parameter(0)
      %c = f32[] constant(10.0)
      %call_x = f32[] call(%p), to_apply=%x
      ROOT %result = f32[] subtract(%call_x, %c)
    }

    ENTRY %entry {
      %p_entry = f32[] parameter(0)
      %call_a = f32[] call(%p_entry), to_apply=%a
      %call_b = f32[] call(%p_entry), to_apply=%b
      ROOT %result = f32[] add(%call_a, %call_b)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(module_text));
  ASSERT_OK_AND_ASSIGN(auto module_split_group,
                       CreateHloModuleSplitGroup(*module));

  // Expect all computations to be assigned somewhere.
  for (auto original_comp : module->computations()) {
    EXPECT_TRUE(module_split_group->address_book.contains(original_comp));
  }

  EXPECT_EQ(module_split_group->module_splits.size(),
            module->computation_count());

  ASSERT_OK_AND_ASSIGN(stream_executor::Platform * platform,
                       PlatformUtil::GetPlatform("cpu"));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Compiler> compiler,
                       Compiler::GetForPlatform(platform->id()));
  ASSERT_OK(compiler->RunHloPasses(module->Clone(), /*executor=*/nullptr,
                                   Compiler::CompileOptions{}));
  for (auto& split : module_split_group->module_splits) {
    EXPECT_OK(compiler->RunHloPasses(split->submodule->Clone(),
                                     /*executor=*/nullptr,
                                     Compiler::CompileOptions{}));
  }
}

TEST_F(SplittingTest, SplitModuleWithSharedComputations) {
  constexpr absl::string_view module_text = R"(
HloModule simple_module


// Simple alpha equivalence examples

%fusion.1 (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %add = f32[] negate(f32[] %x)
}

%fusion.2 (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %add = f32[] negate(f32[] %x)
}

%add.0 (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

%add.1 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

%add.2 (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

ENTRY %main() -> f32[] {

  %p = f32[] constant(3.3)
  %q = f32[] constant(-1.0)
  %r = f32[] constant(7.1)
  %s = f32[] constant(0.2)

  %res.0 = call(%p, %q), to_apply=%add.0
  %res.1 = call(%p, %q), to_apply=%add.1
  %res.1f = f32[] fusion(%res.1), kind=kLoop, calls=%fusion.1
  %res.2 = call(%r, %s), to_apply=%add.2
  %res.3 = call(%p, %s), to_apply=%add.2
  %fusion = f32[] fusion(%res.3), kind=kLoop, calls=%fusion.2

  %res.01 = f32[] add(f32[] %res.0, f32[] %res.1f)
  %res.23 = f32[] add(f32[] %res.2, f32[] %res.3)
  %res.f = f32[] add(f32[] %fusion, f32[], %res.23)

  ROOT %result = f32[] add(f32[] %res.01, f32[] %res.f)
})";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(module_text));
  ASSERT_OK_AND_ASSIGN(auto module_split_group,
                       CreateHloModuleSplitGroup(*module));

  ASSERT_OK_AND_ASSIGN(stream_executor::Platform * platform,
                       PlatformUtil::GetPlatform("cpu"));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Compiler> compiler,
                       Compiler::GetForPlatform(platform->id()));
  ASSERT_OK(compiler->RunHloPasses(module->Clone(), /*executor=*/nullptr,
                                   Compiler::CompileOptions{}));
  for (auto& split : module_split_group->module_splits) {
    EXPECT_OK(compiler->RunHloPasses(split->submodule->Clone(),
                                     /*executor=*/nullptr,
                                     Compiler::CompileOptions{}));
  }
}

TEST_F(SplittingTest, SplitWithCallPredicateAndLinkComputationInto) {
  constexpr absl::string_view module_text = R"(
HloModule split_with_predicate_module, input_output_alias={ {}: (0, {}, may-alias) }

%helper (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %sq = f32[] multiply(f32[] %x, f32[] %x)
}

%late_callee (p: f32[]) -> f32[] {
  %p = f32[] parameter(0)
  %h = f32[] call(f32[] %p), to_apply=%helper
  ROOT %out = f32[] exponential(f32[] %h)
}

ENTRY %main (arg: f32[]) -> f32[] {
  %arg = f32[] parameter(0)
  %c0 = f32[] call(f32[] %arg), to_apply=%late_callee, frontend_attributes={inlineable="xla_late"}
  %c1 = f32[] call(f32[] %c0), to_apply=%late_callee, frontend_attributes={inlineable="xla_late"}
  ROOT %res = f32[] call(f32[] %c1), to_apply=%helper
})";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(module_text));
  auto is_late_call = [](const HloInstruction* inst) {
    if (inst->opcode() != HloOpcode::kCall) {
      return false;
    }
    auto it = inst->frontend_attributes().map().find("inlineable");
    return it != inst->frontend_attributes().map().end() &&
           it->second == "xla_late";
  };

  ASSERT_OK_AND_ASSIGN(auto buckets,
                       GroupComputationsForSplitting(*module, is_late_call));
  // Only ENTRY %main and %late_callee are split boundaries; %helper is cloned
  // into each split that calls it.
  EXPECT_EQ(buckets.size(), 2);

  ASSERT_OK_AND_ASSIGN(auto split_group,
                       CreateHloModuleSplitGroup(*module, is_late_call));
  EXPECT_EQ(split_group->module_splits.size(), 2);

  ASSERT_OK_AND_ASSIGN(
      const HloComputation* split_root,
      split_group->GetClonedComputation(module->entry_computation()));
  ASSERT_NE(split_root, nullptr);
  EXPECT_TRUE(
      split_root->parent()->input_output_alias_config().ParameterHasAlias(0,
                                                                          {}));

  ASSERT_OK(LinkComputationInto(module.get(), split_group->linking_manifest,
                                split_root));
  EXPECT_TRUE(module->input_output_alias_config().ParameterHasAlias(0, {}));
  // Linked module has main, late_callee, and the cloned helper computations.
  EXPECT_GE(module->computation_count(), 3);
}

}  // namespace
}  // namespace xla::parallel
