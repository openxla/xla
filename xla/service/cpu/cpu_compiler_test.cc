/* Copyright 2024 The OpenXLA Authors.

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/test.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/monitoring/collected_metrics.h"
#include "xla/tsl/lib/monitoring/collection_registry.h"

namespace xla {
namespace cpu {
namespace {

using CpuCompilerTest = HloTestBase;

constexpr absl::string_view kCpuCompilerStacktraceMetricName =
    "/xla/service/cpu/compiler_stacktrace_count";

TEST_F(CpuCompilerTest, RecordsStreamzStackTrace) {
  if (tsl::kIsOpenSource) {
    GTEST_SKIP() << "Streamz is not supported in OSS.";
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
    HloModule test
    ENTRY main {
      p = f32[10]{0} parameter(0)
      ROOT neg = f32[10]{0} negate(p)
    }
  )"));
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));

  tsl::monitoring::CollectionRegistry::CollectMetricsOptions options;
  std::unique_ptr<tsl::monitoring::CollectedMetrics> metrics =
      tsl::monitoring::CollectionRegistry::Default()->CollectMetrics(options);

  const auto it = metrics->point_set_map.find(
      std::string(kCpuCompilerStacktraceMetricName));
  ASSERT_TRUE(it != metrics->point_set_map.end());

  // Since Streamz is recorded every call, we expect at least one point.
  // All other callers may increment the counter as well.
  EXPECT_GT(it->second->points.size(), 0);
}

TEST_F(CpuCompilerTest, CompilationWithLargeConstants) {
  absl::string_view module_string = R"(
HloModule module

ENTRY main {
  a = f32[1000,1000]{1,0} parameter(0)
  b = f32[1000,1000]{1,0} constant({...})
  a_plus_b = f32[1000,1000]{1,0} add(a, b)
  c = f32[1000,1000]{1,0} constant({...})
  ROOT result = f32[1000,1000]{1,0} add(a_plus_b, c)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(module_string));

  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));
}

TEST_F(CpuCompilerTest, NoLibraryCallsInHostOffload) {
  absl::string_view hlo = R"(
HloModule module
ENTRY main {
  a = f32[128,128]{1,0} parameter(0)
  b = f32[128,128]{1,0} parameter(1)
  ROOT dot = f32[128,128]{1,0} dot(a, b), lhs_contracting_dims={1},
                                          rhs_contracting_dims={0}
}
)";
  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.add_xla_cpu_experimental_ynn_fusion_type(
      xla::DebugOptions::LIBRARY_FUSION_TYPE_INDIVIDUAL_DOT);
  debug_options.set_xla_cpu_experimental_onednn_custom_call(false);
  debug_options.mutable_xla_backend_extra_options()->insert(
      {"xla_is_host_offload", "true"});
  config.set_debug_options(debug_options);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                       GetOptimizedModule(hlo, config));

  for (auto* computation : optimized_module->computations()) {
    for (auto* instruction : computation->instructions()) {
      // A single dot is only in a custom fusion when it is sent to libraries.
      if (instruction->opcode() == HloOpcode::kFusion) {
        EXPECT_NE(instruction->fusion_kind(),
                  HloInstruction::FusionKind::kCustom);
      }
    }
  }
}

TEST_F(CpuCompilerTest, PermutationSortConvertedToScatter) {
  absl::string_view hlo = R"(
    HloModule test

    compare {
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, order=TOTAL
    }

    compare2 {
      p.0.lhs = s32[] parameter(0)
      p.0.rhs = s32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
    }

    ENTRY sort_computation {
      keys = f32[4,8]{1,0} parameter(0)
      values = s32[4,8]{1,0} iota(), iota_dimension=1
      sort = (f32[4,8]{1,0}, s32[4,8]{1,0}) sort(keys, values), dimensions={1}, to_apply=compare
      gte = s32[4,8]{1,0} get-tuple-element(sort), index=1
      ROOT sort2 = (s32[4,8]{1,0}, s32[4,8]{1,0}) sort(gte, values), dimensions={1}, to_apply=compare2
    }
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                       GetOptimizedModule(hlo, config));

  int64_t sort_count = 0;
  for (auto* computation : optimized_module->computations()) {
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kSort) {
        sort_count++;
      }
    }
  }
  EXPECT_EQ(sort_count, 1);

  EXPECT_TRUE(Run(hlo, /*run_hlo_passes=*/true));
}

TEST_F(CpuCompilerTest, CompilesElementalOpsWithFusionWrapperPassDisabled) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
    HloModule test
    ENTRY main {
      p0 = f32[8]{0} parameter(0)
      p1 = f32[8]{0} parameter(1)
      ROOT add = f32[8]{0} add(p0, p1)
    }
  )"));
  module->mutable_config().mutable_debug_options().add_xla_disable_hlo_passes(
      "fusion-wrapper");

  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));
}

TEST_F(CpuCompilerTest, SimplifiesReshapesAroundCustomCallBeforeFusion) {
  constexpr absl::string_view kHlo = R"(
    HloModule module, entry_computation_layout={(f32[13,13]{1,0}, f32[13]{0})->f32[13]{0}}

    ENTRY main {
      %a = f32[13,13]{1,0} parameter(0)
      %b = f32[13]{0} parameter(1)
      %b_col = f32[13,1]{1,0} reshape(%b)
      %trsm = f32[13,1]{0,1} custom-call(%a, %b_col),
          custom_call_target="lapack_strsm_ffi",
          operand_layout_constraints={f32[13,13]{0,1}, f32[13,1]{0,1}},
          output_to_operand_aliasing={{}: (1, {})},
          api_version=API_VERSION_TYPED_FFI
      ROOT %out = f32[13]{0} reshape(%trsm)
    }
  )";
  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_cpu_mock_custom_calls(true);
  config.set_debug_options(debug_options);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                       GetOptimizedModule(kHlo, config));

  std::vector<const HloInstruction*> fusions;
  for (const HloComputation* computation : optimized_module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      EXPECT_NE(instruction->opcode(), HloOpcode::kReshape)
          << instruction->ToString();
      if (instruction->opcode() == HloOpcode::kFusion) {
        fusions.push_back(instruction);
      }
    }
  }
  // Only the [13,13]{1,0} -> [13,13]{0,1} transpose of parameter 0 requires a
  // copy fusion; the reshapes become bitcasts.
  ASSERT_EQ(fusions.size(), 1) << optimized_module->ToString();
  const HloInstruction* fusion = fusions[0];
  EXPECT_EQ(fusion->operand(0),
            optimized_module->entry_computation()->parameter_instruction(0));
  EXPECT_EQ(fusion->fused_expression_root()->opcode(), HloOpcode::kCopy);
}

// Regression test for emitting a fused bitcast that changes element type.
TEST_F(CpuCompilerTest, CompilesBitcastConvertInSmallWhileLoop) {
  constexpr absl::string_view kHlo = R"(
    HloModule module

    body {
      tuple = (s32[], f32[4]) parameter(0)
      i = s32[] get-tuple-element(tuple), index=0
      x = f32[4] get-tuple-element(tuple), index=1
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      x_bits = s32[4] bitcast-convert(x)
      zero = s32[] constant(0)
      zero_bcast = s32[4] broadcast(zero), dimensions={}
      is_neg = pred[4] compare(x_bits, zero_bcast), direction=LT
      neg_x = f32[4] negate(x)
      next_x = f32[4] select(is_neg, neg_x, x)
      ROOT out = (s32[], f32[4]) tuple(next_i, next_x)
    }

    cond {
      tuple = (s32[], f32[4]) parameter(0)
      i = s32[] get-tuple-element(tuple), index=0
      limit = s32[] constant(2)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      init_i = s32[] constant(0)
      init_x = f32[4] parameter(0)
      init = (s32[], f32[4]) tuple(init_i, init_x)
      loop = (s32[], f32[4]) while(init), condition=cond, body=body
      ROOT res = f32[4] get-tuple-element(loop), index=1
    }
  )";
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                       GetOptimizedModule(kHlo, config));
  bool has_fused_type_changing_bitcast = false;
  for (const HloComputation* computation : optimized_module->computations()) {
    if (!computation->IsFusionComputation()) {
      continue;
    }
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kBitcast &&
          instruction->shape().element_type() !=
              instruction->operand(0)->shape().element_type()) {
        has_fused_type_changing_bitcast = true;
      }
    }
  }
  EXPECT_TRUE(has_fused_type_changing_bitcast) << optimized_module->ToString();

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  Literal input = LiteralUtil::CreateR1<float>({-1.5f, 2.0f, -3.0f, 4.0f});
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<float>({1.5f, 2.0f, 3.0f, 4.0f}), result));
}

}  // namespace
}  // namespace cpu
}  // namespace xla
