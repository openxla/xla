/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/hlo/transforms/memory_space_propagation.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::UnorderedElementsAreArray;

class MemorySpacePropagationTest : public HloHardwareIndependentTestBase {
 public:
  MemorySpacePropagationTest()
      : HloHardwareIndependentTestBase(),
        verifier_(/*layout_sensitive=*/false, /*allow_mixed_precision*/ false) {
  }

  absl::Status Verify(HloModule* module) {
    return verifier_.Run(module).status();
  }

 protected:
  // Returns a dataflow analysis for the given module.
  std::unique_ptr<HloDataflowAnalysis> GetDataflowAnalysis(
      const HloModule& module) {
    if (auto status_or =
            HloDataflowAnalysis::Run(module, /*ssa_form=*/false,
                                     /*bitcast_defines_value=*/true);
        status_or.ok()) {
      return std::move(status_or.value());
    }
    return nullptr;
  }

  // Checks, for every (instruction, index) of every fusion computation of
  // module, that the values Run() computes from the fusion computation are
  // the values of the dataflow analysis it would otherwise build.
  void ExpectLocalDataflowMatchesAnalysis(HloModule* module) {
    ASSERT_TRUE(MemorySpacePropagation::HasLocalFusionDataflow(*module));
    std::unique_ptr<HloDataflowAnalysis> analysis =
        GetDataflowAnalysis(*module);
    ASSERT_NE(analysis, nullptr);
    int64_t checked = 0;
    for (HloComputation* computation : module->computations()) {
      if (!computation->IsFusionComputation()) {
        continue;
      }
      for (HloInstruction* instruction : computation->instructions()) {
        ShapeUtil::ForEachSubshape(
            instruction->shape(),
            [&](const Shape& /*subshape*/, const ShapeIndex& index) {
              const HloValue& value =
                  analysis->GetUniqueValueAt(instruction, index);
              const HloPosition defining =
                  MemorySpacePropagation::DefiningPosition(
                      HloPosition{instruction, index});
              EXPECT_EQ(defining, value.defining_position());
              const absl::InlinedVector<HloPosition, 4> positions =
                  MemorySpacePropagation::Positions(defining);
              EXPECT_EQ(positions.front(), defining);
              EXPECT_THAT(positions,
                          UnorderedElementsAreArray(value.positions()));
              std::vector<HloUse> fusion_uses;
              for (const HloUse& use : value.GetUses()) {
                if (use.instruction->opcode() == HloOpcode::kFusion) {
                  fusion_uses.push_back(use);
                }
              }
              EXPECT_THAT(MemorySpacePropagation::FusionUses(positions),
                          UnorderedElementsAreArray(fusion_uses));
              ++checked;
            });
      }
    }
    EXPECT_GT(checked, 0);
  }

 private:
  HloVerifier verifier_;
};

TEST_F(MemorySpacePropagationTest, NoMemorySpace) {
  absl::string_view hlo_string = R"(
  HloModule NoMemorySpace

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)} parameter(0)
    ROOT %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)} copy(%param2)
    %fusion = s32[6]{0:T(128)} fusion(s32[6]{0:T(128)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[6]{0:T(128)} copy(%fusion)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ExpectLocalDataflowMatchesAnalysis(module.get());
  MemorySpacePropagation memory_space_propagation;
  EXPECT_FALSE(memory_space_propagation.Run(module.get()).value());
  TF_ASSERT_OK_AND_ASSIGN(auto ref, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

TEST_F(MemorySpacePropagationTest, NonTupleOutput) {
  absl::string_view hlo_string = R"(
  HloModule NonTupleOutput

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)} parameter(0)
    ROOT %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)SC(0:3)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[6]{0:T(128)S(1)} fusion(s32[6]{0:T(128)S(1)SC(0:3)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[6]{0:T(128)} copy(%fusion)
  }
  )";
  absl::string_view expected_hlo_string = R"(
  HloModule NonTupleOutput

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)S(1)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)S(1)SC(0:3)} parameter(0)
    ROOT %add.0 = s32[6]{0:T(128)S(1)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)SC(0:3)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[6]{0:T(128)S(1)} fusion(s32[6]{0:T(128)S(1)SC(0:3)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[6]{0:T(128)} copy(%fusion)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  ExpectLocalDataflowMatchesAnalysis(module.get());
  MemorySpacePropagation memory_space_propagation;
  EXPECT_TRUE(memory_space_propagation.Run(module.get()).value());
  TF_EXPECT_OK(Verify(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

TEST_F(MemorySpacePropagationTest, TupleOutput) {
  absl::string_view hlo_string = R"(
  HloModule TupleOutput

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)} parameter(0)
    %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    %multiply.0 = s32[6]{0:T(128)} multiply(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    ROOT %tuple = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%add.0, %multiply.0)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = (s32[6]{0:T(128)S(1)SC(0:3)}, s32[6]{0:T(128)}) fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    %gte0 = s32[6]{0:T(128)S(1)SC(0:3)} get-tuple-element(%fusion), index=0
    %gte1 = s32[6]{0:T(128)} get-tuple-element(%fusion), index=1
    ROOT %root = s32[6]{0:T(128)} add(%gte0, %gte1)
  }
  )";
  absl::string_view expected_hlo_string = R"(
  HloModule TupleOutput

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)S(1)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)S(1)} parameter(0)
    %add.0 = s32[6]{0:T(128)S(1)SC(0:3)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    %multiply.0 = s32[6]{0:T(128)} multiply(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    ROOT %tuple = (s32[6]{0:T(128)S(1)SC(0:3)}, s32[6]{0:T(128)}) tuple(%add.0, %multiply.0)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = (s32[6]{0:T(128)S(1)SC(0:3)}, s32[6]{0:T(128)}) fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    %gte0 = s32[6]{0:T(128)S(1)SC(0:3)} get-tuple-element(%fusion), index=0
    %gte1 = s32[6]{0:T(128)} get-tuple-element(%fusion), index=1
    ROOT %root = s32[6]{0:T(128)} add(%gte0, %gte1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  ExpectLocalDataflowMatchesAnalysis(module.get());
  MemorySpacePropagation memory_space_propagation;
  EXPECT_TRUE(memory_space_propagation.Run(module.get()).value());
  TF_EXPECT_OK(Verify(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

TEST_F(MemorySpacePropagationTest, NestedInputFusion) {
  // Tests propagating the memory space to nested fusions on the input side.
  absl::string_view hlo_string = R"(
  HloModule NestedFusion

  %bitcast_fusion {
    %bf_param = s32[3,2]{0,1:T(128)} parameter(0)
    ROOT %bitcast = s32[6]{0:T(128)} bitcast(%bf_param)
  }

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[3,2]{0,1:T(128)} parameter(0)
    %fusion.1 = s32[6]{0:T(128)} fusion(%param_0.1), kind=kLoop, calls=bitcast_fusion
    ROOT %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %fusion.1)
  }

  ENTRY %entry {
    %param0 = s32[3,2]{0,1:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[3,2]{0,1:T(128)S(1)SC(1:1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[6]{0:T(128)S(1)} fusion(s32[3,2]{0,1:T(128)S(1)SC(1:1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[6]{0:T(128)} copy(%fusion)
  }
  )";
  absl::string_view expected_hlo_string = R"(
  HloModule NestedFusion

  %bitcast_fusion {
    %bf_param = s32[3,2]{0,1:T(128)S(1)SC(1:1)} parameter(0)
    ROOT %bitcast = s32[6]{0:T(128)} bitcast(%bf_param)
  }

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)S(1)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[3,2]{0,1:T(128)S(1)SC(1:1)} parameter(0)
    %fusion.1 = s32[6]{0:T(128)} fusion(%param_0.1), kind=kLoop, calls=bitcast_fusion
    ROOT %add.0 = s32[6]{0:T(128)S(1)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %fusion.1)
  }

  ENTRY %entry {
    %param0 = s32[3,2]{0,1:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[3,2]{0,1:T(128)S(1)SC(1:1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[6]{0:T(128)S(1)} fusion(s32[3,2]{0,1:T(128)S(1)SC(1:1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[6]{0:T(128)} copy(%fusion)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  ExpectLocalDataflowMatchesAnalysis(module.get());
  MemorySpacePropagation memory_space_propagation;
  EXPECT_TRUE(memory_space_propagation.Run(module.get()).value());
  TF_EXPECT_OK(Verify(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

TEST_F(MemorySpacePropagationTest, NestedOutputFusion) {
  // Tests propagating the memory space to nested fusions on the output side.
  absl::string_view hlo_string = R"(
  HloModule NestedFusion

  %bitcast_fusion {
    %bf_param = s32[6]{0:T(128)} parameter(0)
    ROOT %bitcast = s32[3,2]{0,1:T(128)} bitcast(%bf_param)
  }

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)} parameter(0)
    %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    ROOT %fusion.1 = s32[3,2]{0,1:T(128)} fusion(%add.0), kind=kLoop, calls=bitcast_fusion
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[3,2]{0,1:T(128)S(1)SC(1:1)} fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[3,2]{0,1:T(128)} copy(%fusion)
  }
  )";
  absl::string_view expected_hlo_string = R"(
  HloModule NestedFusion

  %bitcast_fusion {
    %bf_param = s32[6]{0:T(128)} parameter(0)
    ROOT %bitcast = s32[3,2]{0,1:T(128)S(1)SC(1:1)} bitcast(%bf_param)
  }

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)S(1)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)S(1)} parameter(0)
    %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)S(1)} %param_0.1)
    ROOT %fusion.1 = s32[3,2]{0,1:T(128)S(1)SC(1:1)} fusion(%add.0), kind=kLoop, calls=bitcast_fusion
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[3,2]{0,1:T(128)S(1)SC(1:1)} fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[3,2]{0,1:T(128)} copy(%fusion)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  ExpectLocalDataflowMatchesAnalysis(module.get());
  MemorySpacePropagation memory_space_propagation;
  EXPECT_TRUE(memory_space_propagation.Run(module.get()).value());
  TF_EXPECT_OK(Verify(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

TEST_F(MemorySpacePropagationTest, BitcastInFusion) {
  absl::string_view hlo_string = R"(
  HloModule TupleOutput

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)} parameter(0)
    %bitcast.0 = s32[6]{0:T(128)} bitcast(s32[6]{0:T(128)} %param_0.1)
    %multiply.0 = s32[6]{0:T(128)} multiply(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    ROOT %tuple = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%bitcast.0, %multiply.0)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)SC(0:3)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    ROOT %fusion = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) fusion(s32[6]{0:T(128)S(1)SC(0:3)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
  }
  )";
  absl::string_view expected_hlo_string = R"(
  HloModule TupleOutput

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)S(1)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)S(1)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)S(1)SC(0:3)} parameter(0)
    %bitcast.0 = s32[6]{0:T(128)} bitcast(s32[6]{0:T(128)S(1)SC(0:3)} %param_0.1)
    %multiply.0 = s32[6]{0:T(128)} multiply(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)S(1)SC(0:3)} %param_0.1)
    ROOT %tuple = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%bitcast.0, %multiply.0)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)SC(0:3)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    ROOT %fusion = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) fusion(s32[6]{0:T(128)S(1)SC(0:3)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  ExpectLocalDataflowMatchesAnalysis(module.get());
  MemorySpacePropagation memory_space_propagation;
  EXPECT_TRUE(memory_space_propagation.Run(module.get()).value());
  TF_EXPECT_OK(Verify(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

// This test tests RunOnComputation. The parameters do _not_ get the memory
// space propagated from the operands. The operations in the fusion get the
// memory space propagated from the parameters.
TEST_F(MemorySpacePropagationTest, RunOnComputationPropagateFromParameters) {
  absl::string_view hlo_string = R"(
    HloModule NoMemorySpace

    %fused_computation {
      %param_1.3 = s32[6]{0:T(128)S(1)} parameter(0)
      %param_2.3 = s32[6]{0:T(128)} parameter(1)
      %tuple = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%param_1.3, %param_2.3)
      %gte_1.3 = s32[6]{0:T(128)} get-tuple-element(%tuple), index=0
      %neg_1.3 = s32[6]{0:T(128)} negate(%gte_1.3)
      ROOT %root = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%neg_1.3, %param_2.3)
    }
    ENTRY %entry {
      %param0 = s32[6]{0:T(128)} parameter(0)
      %param1 = s32[6]{0:T(128)} parameter(1)
      %param1_copy = s32[6]{0:T(128)S(1)} copy(%param1)
      ROOT %fusion = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) fusion(%param0, %param1_copy), kind=kLoop, calls=%fused_computation
    }
  )";
  absl::string_view expected_hlo_string = R"(
    HloModule NoMemorySpace

    %fused_computation {
      %param_1.3 = s32[6]{0:T(128)S(1)} parameter(0)
      %param_2.3 = s32[6]{0:T(128)} parameter(1)
      %tuple = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) tuple(%param_1.3, %param_2.3)
      %gte_1.3 = s32[6]{0:T(128)S(1)} get-tuple-element(%tuple), index=0
      %neg_1.3 = s32[6]{0:T(128)} negate(%gte_1.3)
      ROOT %root = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%neg_1.3, %param_2.3)
    }
    ENTRY %entry {
      %param0 = s32[6]{0:T(128)} parameter(0)
      %param1 = s32[6]{0:T(128)} parameter(1)
      %param1_copy = s32[6]{0:T(128)S(1)} copy(%param1)
      ROOT %fusion = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) fusion(%param0, %param1_copy), kind=kLoop, calls=%fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ExpectLocalDataflowMatchesAnalysis(module.get());
  auto dataflow_analysis = GetDataflowAnalysis(*module);
  MemorySpacePropagation memory_space_propagation(std::move(dataflow_analysis));
  HloComputation* computation =
      module->GetComputationWithName("fused_computation");
  ASSERT_NE(computation, nullptr);
  EXPECT_TRUE(memory_space_propagation.RunOnComputation(computation));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

// This test tests that the parameters in nested fusions get the memory space
// propagated from the operands.
TEST_F(MemorySpacePropagationTest, RunOnComputationFromParametersNestedFusion) {
  absl::string_view hlo_string = R"(
    HloModule NoMemorySpace

    %nested_fusion {
      %param_1.3 = s32[6]{0:T(128)} parameter(0)
      ROOT %neg_1.3 = s32[6]{0:T(128)} negate(%param_1.3)
    }

    %fused_computation {
      %param_1.3 = s32[6]{0:T(128)S(1)} parameter(0)
      %param_2.3 = s32[6]{0:T(128)} parameter(1)
      %tuple = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%param_1.3, %param_2.3)
      %gte_1.3 = s32[6]{0:T(128)} get-tuple-element(%tuple), index=0
      %neg_1.3 = s32[6]{0:T(128)} fusion(%gte_1.3), kind=kLoop, calls=%nested_fusion
      ROOT %root = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%neg_1.3, %param_2.3)
    }

    ENTRY %entry {
      %param0 = s32[6]{0:T(128)} parameter(0)
      %param1 = s32[6]{0:T(128)} parameter(1)
      ROOT %fusion = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) fusion(%param0, %param1), kind=kLoop, calls=%fused_computation
    }
  )";
  absl::string_view expected_hlo_string = R"(
    HloModule NoMemorySpace

    %nested_fusion {
      %param_1.3 = s32[6]{0:T(128)S(1)} parameter(0)
      ROOT %neg_1.3 = s32[6]{0:T(128)} negate(%param_1.3)
    }

    %fused_computation {
      %param_1.3 = s32[6]{0:T(128)S(1)} parameter(0)
      %param_2.3 = s32[6]{0:T(128)} parameter(1)
      %tuple = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) tuple(%param_1.3, %param_2.3)
      %gte_1.3 = s32[6]{0:T(128)S(1)} get-tuple-element(%tuple), index=0
      %neg_1.3 = s32[6]{0:T(128)} fusion(%gte_1.3), kind=kLoop, calls=%nested_fusion
      ROOT %root = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%neg_1.3, %param_2.3)
    }

    ENTRY %entry {
      %param0 = s32[6]{0:T(128)} parameter(0)
      %param1 = s32[6]{0:T(128)} parameter(1)
      ROOT %fusion = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) fusion(%param0, %param1), kind=kLoop, calls=%fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ExpectLocalDataflowMatchesAnalysis(module.get());
  auto dataflow_analysis = GetDataflowAnalysis(*module);
  MemorySpacePropagation memory_space_propagation(std::move(dataflow_analysis));
  HloComputation* computation =
      module->GetComputationWithName("fused_computation");
  ASSERT_NE(computation, nullptr);
  EXPECT_TRUE(memory_space_propagation.RunOnComputation(computation));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

// This test tests that the operations in the fusion get the memory space
// propagated from the output.
TEST_F(MemorySpacePropagationTest, RunOnComputationPropagateFromOutput) {
  absl::string_view hlo_string = R"(
    HloModule NoMemorySpace

    %fused_computation {
      %param_1.3 = s32[6]{0:T(128)} parameter(0)
      %param_2.3 = s32[6]{0:T(128)} parameter(1)
      %neg_1.3 = s32[6]{0:T(128)} negate(%param_1.3)
      ROOT %root = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) tuple(%neg_1.3, %param_2.3)
    }
    ENTRY %entry {
      %param0 = s32[6]{0:T(128)} parameter(0)
      %param1 = s32[6]{0:T(128)} parameter(1)
      ROOT %fusion = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) fusion(%param0, %param1), kind=kLoop, calls=%fused_computation
    }
  )";
  absl::string_view expected_hlo_string = R"(
    HloModule NoMemorySpace

    %fused_computation {
      %param_1.3 = s32[6]{0:T(128)} parameter(0)
      %param_2.3 = s32[6]{0:T(128)} parameter(1)
      %neg_1.3 = s32[6]{0:T(128)S(1)} negate(%param_1.3)
      ROOT %root = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) tuple(%neg_1.3, %param_2.3)
    }
    ENTRY %entry {
      %param0 = s32[6]{0:T(128)} parameter(0)
      %param1 = s32[6]{0:T(128)} parameter(1)
      ROOT %fusion = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) fusion(%param0, %param1), kind=kLoop, calls=%fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ExpectLocalDataflowMatchesAnalysis(module.get());
  auto dataflow_analysis = GetDataflowAnalysis(*module);
  MemorySpacePropagation memory_space_propagation(std::move(dataflow_analysis));
  HloComputation* computation =
      module->GetComputationWithName("fused_computation");
  ASSERT_NE(computation, nullptr);
  EXPECT_TRUE(memory_space_propagation.RunOnComputation(computation));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

// This test tests that the memory space propagation works correctly when there
// is a nested fusion with a shape mismatch. In this test, S(1) must propagate
// from the parameter of the nested fusion fusion.505 to its output shape, and
// from there to the root of the nested fusion %copy.4014.
TEST_F(MemorySpacePropagationTest, NestedFusionShapeMismatchBug) {
  absl::string_view hlo_string =
      R"(HloModule jit_insert.fusion.21.isolated, is_scheduled=true

%copy_fusion.20.clone {
  %input.20 = s4[8,32768,1,256]{3,1,0,2:T(64,128)(8,1)E(4)S(1)} parameter(0)
  ROOT %copy.4014 = s4[8,32768,1,256]{3,1,0,2:T(8,128)(8,1)E(4)} copy(%input.20)
}

%fused_computation.434.clone {
  %param_0.777 = s4[8,32768,1,256]{3,1,0,2:T(64,128)(8,1)E(4)S(1)} parameter(0)
  %fusion.505 = s4[8,32768,1,256]{3,1,0,2:T(8,128)(8,1)E(4)} fusion(%param_0.777), kind=kLoop, output_to_operand_aliasing={{}: (0, {})}, calls=%copy_fusion.20.clone
  %param_3.751 = pred[]{:T(512)} parameter(3)
  %broadcast.1846 = pred[1,16384,1,256]{3,1,0,2:T(8,128)(4,1)} broadcast(%param_3.751), dimensions={}, metadata={op_name="jit(insert)/jit(main)/pjit/jit(insert)/jit(main)/jit(insert)/pjit/jit(insert)/jit(main)/jit(insert)/jit(insert)/dynamic_update_slice" stack_frame_id=146}
  %param_2.1768 = s4[1,16384,1,256]{3,1,0,2:T(8,128)(8,1)E(4)S(1)} parameter(2)
  %param_1.980 = s32[]{:T(128)S(6)} parameter(1)
  %constant.9791 = s32[]{:T(128)} constant(0), metadata={op_name="jit(insert)/jit(main)/pjit/jit(insert)/jit(main)/jit(insert)/pjit"}
  %dynamic-slice.2042 = s4[1,16384,1,256]{3,1,0,2:T(8,128)(8,1)E(4)} dynamic-slice(%fusion.505, %param_1.980, %constant.9791, %constant.9791, %constant.9791), dynamic_slice_sizes={1,16384,1,256}, metadata={op_name="jit(insert)/jit(main)/pjit/jit(insert)/jit(main)/jit(insert)/pjit/jit(insert)/jit(main)/jit(insert)/jit(insert)/dynamic_update_slice" stack_frame_id=146}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"indices_config":{"index_known_bits":[{"zeroes":"0","ones":"0","bitwidth":"32"},{"zeroes":"4294967295","ones":"0","bitwidth":"32"},{"zeroes":"4294967295","ones":"0","bitwidth":"32"},{"zeroes":"4294967295","ones":"0","bitwidth":"32"}],"is_index_aligned":[true,true,true,true]},"used_scoped_memory_configs":[]}
  %select.912 = s4[1,16384,1,256]{3,1,0,2:T(8,128)(8,1)E(4)} select(%broadcast.1846, %param_2.1768, %dynamic-slice.2042), metadata={op_name="jit(insert)/jit(main)/pjit/jit(insert)/jit(main)/jit(insert)/pjit/jit(insert)/jit(main)/jit(insert)/jit(insert)/dynamic_update_slice" stack_frame_id=146}
  ROOT %dynamic-update-slice.455 = s4[8,32768,1,256]{3,1,0,2:T(64,128)(8,1)E(4)S(1)} dynamic-update-slice(%fusion.505, %select.912, %param_1.980, %constant.9791, %constant.9791, /*index=5*/%constant.9791), metadata={op_name="jit(insert)/jit(main)/pjit/jit(insert)/jit(main)/jit(insert)/pjit/jit(insert)/jit(main)/jit(insert)/jit(insert)/dynamic_update_slice" stack_frame_id=146}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"indices_config":{"index_known_bits":[{"zeroes":"0","ones":"0","bitwidth":"32"},{"zeroes":"4294967295","ones":"0","bitwidth":"32"},{"zeroes":"4294967295","ones":"0","bitwidth":"32"},{"zeroes":"4294967295","ones":"0","bitwidth":"32"}],"is_index_aligned":[true,true,true,true]},"used_scoped_memory_configs":[]}
}

ENTRY %jit_insert.fusion.21.isolated.root {
  %bitcast.1556.hbm = s4[8,32768,1,256]{3,1,0,2:T(64,128)(8,1)E(4)} parameter(0)
  %select.32 = s32[]{:T(128)S(6)} parameter(1)
  %collective-permute.56.hbm = s4[1,16384,1,256]{3,1,0,2:T(8,128)(8,1)E(4)} parameter(2)
  %and.74 = pred[]{:T(512)} parameter(3)
  %copy = s4[8,32768,1,256]{3,1,0,2:T(64,128)(8,1)E(4)S(1)} copy(%bitcast.1556.hbm)
  %copy.1 = s4[1,16384,1,256]{3,1,0,2:T(8,128)(8,1)E(4)S(1)} copy(%collective-permute.56.hbm)
  %fusion.21 = s4[8,32768,1,256]{3,1,0,2:T(64,128)(8,1)E(4)S(1)} fusion(%copy, %select.32, %copy.1, %and.74), kind=kLoop, calls=%fused_computation.434.clone, metadata={op_name="jit(insert)/jit(main)/pjit/jit(insert)/jit(main)/jit(insert)/pjit/jit(insert)/jit(main)/jit(insert)/jit(insert)/dynamic_update_slice" stack_frame_id=146}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"used_scoped_memory_configs":[],"aliasing_operands":{"lists":[{"indices":["0","4"]}]}}
  ROOT %copy.2 = s4[8,32768,1,256]{3,1,0,2:T(64,128)(8,1)E(4)} copy(%fusion.21)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  ExpectLocalDataflowMatchesAnalysis(module.get());
  MemorySpacePropagation memory_space_propagation;
  // %copy.4014 output memory space must get modified to match %fusion.505
  // output shape.
  EXPECT_TRUE(memory_space_propagation.Run(module.get()).value());
  HloComputation* computation =
      module->GetComputationWithName("copy_fusion.20.clone");
  ASSERT_NE(computation, nullptr);
  const HloInstruction* copy = computation->GetInstructionWithName("copy.4014");
  ASSERT_NE(copy, nullptr);
  EXPECT_EQ(copy->shape().layout().memory_space(), 1);
  computation = module->GetComputationWithName("fused_computation.434.clone");
  ASSERT_NE(computation, nullptr);
  const HloInstruction* fusion =
      computation->GetInstructionWithName("fusion.505");
  ASSERT_NE(fusion, nullptr);
  EXPECT_EQ(fusion->shape().layout().memory_space(), 1);
  TF_EXPECT_OK(Verify(module.get()));
}

// Fusion computations that forward values through every rule Run() models
// without a dataflow analysis: a nested tuple parameter read through
// get-tuple-elements, tuples re wrapped at depth one and two, add-dependency,
// opt-barrier, domain, a tuple copy, a root tuple returning a parameter leaf
// twice, nested fusions on the input side (one using a value at two
// operands) and on the output side, a bitcast chain and a dynamic-update-slice
// root. One value (p{0,0}) is reached from operand leaf {0,0} and from output
// leaves {0}, {1} and {3}, with different memory spaces.
constexpr absl::string_view kForwardingShapesHlo = R"(
  HloModule ForwardingShapes

  %nested_bitcast {
    %q = s32[6]{0:T(128)} parameter(0)
    ROOT %bc = s32[3,2]{0,1:T(128)} bitcast(%q)
  }

  %nested_negate {
    %r = s32[3,2]{0,1:T(128)} parameter(0)
    ROOT %neg = s32[3,2]{0,1:T(128)} negate(%r)
  }

  %nested_add {
    %a = s32[6]{0:T(128)} parameter(0)
    %b = s32[6]{0:T(128)} parameter(1)
    ROOT %sum = s32[6]{0:T(128)} add(%a, %b)
  }

  %tuple_forwarding {
    %p = ((s32[6]{0:T(128)}, s32[6]{0:T(128)}), s32[6]{0:T(128)}) parameter(0)
    %inner = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) get-tuple-element(%p), index=0
    %g0 = s32[6]{0:T(128)} get-tuple-element(%inner), index=0
    %g1 = s32[6]{0:T(128)} get-tuple-element(%inner), index=1
    %g2 = s32[6]{0:T(128)} get-tuple-element(%p), index=1
    %t = (s32[6]{0:T(128)}, (s32[6]{0:T(128)}, s32[6]{0:T(128)})) tuple(%g2, %inner)
    %tok = token[] after-all()
    %ad = (s32[6]{0:T(128)}, (s32[6]{0:T(128)}, s32[6]{0:T(128)})) add-dependency(%t, %tok)
    %ob = (s32[6]{0:T(128)}, (s32[6]{0:T(128)}, s32[6]{0:T(128)})) opt-barrier(%ad)
    %cp = (s32[6]{0:T(128)}, (s32[6]{0:T(128)}, s32[6]{0:T(128)})) copy(%ob)
    %inner2 = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) get-tuple-element(%cp), index=1
    %g3 = s32[6]{0:T(128)} get-tuple-element(%inner2), index=0
    %g4 = s32[6]{0:T(128)} get-tuple-element(%ob), index=0
    %dom = s32[6]{0:T(128)} domain(%g1), domain={kind="sharding", entry={maximal device=0}, exit={maximal device=1}}
    %n = s32[6]{0:T(128)} negate(%dom)
    %f1 = s32[3,2]{0,1:T(128)} fusion(%n), kind=kLoop, calls=%nested_bitcast
    %f2 = s32[3,2]{0,1:T(128)} fusion(%f1), kind=kLoop, calls=%nested_negate
    %f3 = s32[6]{0:T(128)} fusion(%g4, %g4), kind=kLoop, calls=%nested_add
    ROOT %root = (s32[6]{0:T(128)}, s32[6]{0:T(128)}, s32[3,2]{0,1:T(128)}, s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%g3, %g0, %f2, %g0, %f3)
  }

  %dus_root {
    %base = s32[2,3]{1,0:T(128)} parameter(0)
    %update = s32[1,3]{1,0:T(128)} parameter(1)
    %idx = s32[]{:T(128)} parameter(2)
    %bcb = s32[2,3]{1,0:T(128)} bitcast(%base)
    ROOT %dus = s32[2,3]{1,0:T(128)} dynamic-update-slice(%bcb, %update, %idx, %idx)
  }

  ENTRY %entry {
    %p0 = s32[6]{0:T(128)} parameter(0)
    %p1 = s32[6]{0:T(128)} parameter(1)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%p0)
    %inner_pair = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) tuple(%arg0, %p1)
    %triple = ((s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}), s32[6]{0:T(128)}) tuple(%inner_pair, %p1)
    %fwd = (s32[6]{0:T(128)}, s32[6]{0:T(128)S(1)}, s32[3,2]{0,1:T(128)S(1)}, s32[6]{0:T(128)}, s32[6]{0:T(128)S(1)}) fusion(%triple), kind=kLoop, calls=%tuple_forwarding
    %base = s32[2,3]{1,0:T(128)S(1)} parameter(2)
    %update = s32[1,3]{1,0:T(128)} parameter(3)
    %idx = s32[]{:T(128)} parameter(4)
    %dus_out = s32[2,3]{1,0:T(128)} fusion(%base, %update, %idx), kind=kLoop, calls=%dus_root
    ROOT %r = ((s32[6]{0:T(128)}, s32[6]{0:T(128)S(1)}, s32[3,2]{0,1:T(128)S(1)}, s32[6]{0:T(128)}, s32[6]{0:T(128)S(1)}), s32[2,3]{1,0:T(128)}) tuple(%fwd, %dus_out)
  }
  )";

// Adds to the entry computation of module a fusion whose computation calls
// another computation, one of the instruction kinds Run() does not model, so
// Run() falls back to the dataflow analysis on this module.
void AddFusionWithCall(HloModule* module) {
  const Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {}, {});
  HloComputation::Builder callee_builder("callee");
  HloInstruction* callee_param = callee_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "callee_param"));
  callee_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, callee_param));
  HloComputation* callee =
      module->AddEmbeddedComputation(callee_builder.Build());
  HloComputation::Builder fused_builder("fused_call");
  HloInstruction* fused_param = fused_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "fused_param"));
  fused_builder.AddInstruction(
      HloInstruction::CreateCall(shape, {fused_param}, callee));
  HloComputation* fused = module->AddEmbeddedComputation(fused_builder.Build());
  HloComputation* entry = module->entry_computation();
  HloInstruction* constant = entry->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
  entry->AddInstruction(HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kLoop, {constant}, fused));
}

TEST_F(MemorySpacePropagationTest, LocalDataflowMatchesAnalysisOnForwarding) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kForwardingShapesHlo));
  ExpectLocalDataflowMatchesAnalysis(module.get());

  // The same module propagated through the dataflow analysis, forced by an
  // unrelated fusion with a call, ends with the same layouts everywhere.
  std::unique_ptr<HloModule> reference = module->Clone(/*suffix=*/"");
  AddFusionWithCall(reference.get());
  EXPECT_FALSE(MemorySpacePropagation::HasLocalFusionDataflow(*reference));
  EXPECT_TRUE(MemorySpacePropagation().Run(module.get()).value());
  EXPECT_TRUE(MemorySpacePropagation().Run(reference.get()).value());
  int64_t compared = 0;
  for (HloComputation* computation : module->computations()) {
    HloComputation* reference_computation =
        reference->GetComputationWithName(computation->name());
    ASSERT_NE(reference_computation, nullptr);
    for (const HloInstruction* instruction : computation->instructions()) {
      const HloInstruction* reference_instruction =
          reference_computation->GetInstructionWithName(instruction->name());
      ASSERT_NE(reference_instruction, nullptr);
      EXPECT_TRUE(
          Shape::Equal()(instruction->shape(), reference_instruction->shape()))
          << instruction->ToString() << " vs "
          << reference_instruction->ToString();
      ++compared;
    }
  }
  EXPECT_EQ(compared, module->instruction_count());

  // p{0,0} is reached from operand leaf {0,0} with S(1), then from output
  // leaves {0} with S(0), {1} with S(1) and {3} with S(0); the last leaf wins
  // on every position of that value, the depth two positions inside the re
  // wrapped tuples included.
  HloComputation* fused = module->GetComputationWithName("tuple_forwarding");
  ASSERT_NE(fused, nullptr);
  for (absl::string_view name : {"g0", "g3"}) {
    const HloInstruction* instruction = fused->GetInstructionWithName(name);
    ASSERT_NE(instruction, nullptr) << name;
    EXPECT_EQ(instruction->shape().layout().memory_space(), 0) << name;
  }
  for (absl::string_view name : {"t", "ad", "ob", "cp"}) {
    const HloInstruction* instruction = fused->GetInstructionWithName(name);
    ASSERT_NE(instruction, nullptr) << name;
    EXPECT_EQ(ShapeUtil::GetSubshape(instruction->shape(), {1, 0})
                  .layout()
                  .memory_space(),
              0)
        << name;
  }
  const HloInstruction* root = fused->root_instruction();
  EXPECT_EQ(root->shape().tuple_shapes(1).layout().memory_space(), 0);
  EXPECT_EQ(root->shape().tuple_shapes(2).layout().memory_space(), 1);
  // Output leaf {4} colors f3's value; its two uses of g4 recolor nothing
  // since g4 holds p{1}, which the output leaves never color.
  EXPECT_EQ(module->GetComputationWithName("nested_add")
                ->parameter_instruction(1)
                ->shape()
                .layout()
                .memory_space(),
            0);
  const HloInstruction* neg =
      module->GetComputationWithName("nested_negate")->root_instruction();
  EXPECT_EQ(neg->shape().layout().memory_space(), 1);
  HloComputation* dus_root = module->GetComputationWithName("dus_root");
  ASSERT_NE(dus_root, nullptr);
  EXPECT_EQ(dus_root->parameter_instruction(0)->shape().layout().memory_space(),
            1);
  EXPECT_EQ(
      dus_root->GetInstructionWithName("bcb")->shape().layout().memory_space(),
      0);
}

TEST_F(MemorySpacePropagationTest, FallsBackToAnalysisForCallInsideFusion) {
  absl::string_view hlo_string = R"(
  HloModule CallInsideFusion

  %callee {
    %cp = s32[6]{0:T(128)} parameter(0)
    ROOT %cn = s32[6]{0:T(128)} negate(%cp)
  }

  %fused_call {
    %fp = s32[6]{0:T(128)} parameter(0)
    ROOT %call = s32[6]{0:T(128)} call(%fp), to_apply=%callee
  }

  ENTRY %entry {
    %p0 = s32[6]{0:T(128)} parameter(0)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%p0)
    ROOT %fusion = s32[6]{0:T(128)S(1)} fusion(%arg0), kind=kLoop, calls=%fused_call
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo_string));
  EXPECT_FALSE(MemorySpacePropagation::HasLocalFusionDataflow(*module));
  EXPECT_TRUE(MemorySpacePropagation().Run(module.get()).value());
  EXPECT_EQ(module->GetComputationWithName("fused_call")
                ->parameter_instruction(0)
                ->shape()
                .layout()
                .memory_space(),
            1);
  // A computation called from a fusion computation is embedded too, so its
  // parameter defines its own value and keeps its memory space, while the
  // call forwards the callee's root value: only the analysis reaches %cn.
  HloComputation* callee = module->GetComputationWithName("callee");
  EXPECT_EQ(callee->parameter_instruction(0)->shape().layout().memory_space(),
            0);
  EXPECT_EQ(callee->root_instruction()->shape().layout().memory_space(), 1);
}

// Control flow and calls outside fusion computations do not disturb the
// values inside them, so Run() stays on the local path, the production case.
TEST_F(MemorySpacePropagationTest, ControlFlowOutsideFusionsStaysLocal) {
  absl::string_view hlo_string = R"(
  HloModule ControlFlowOutsideFusions

  %fused_add {
    %fp = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) parameter(0)
    %fa = s32[6]{0:T(128)} get-tuple-element(%fp), index=0
    %fb = s32[6]{0:T(128)} get-tuple-element(%fp), index=1
    %fsum = s32[6]{0:T(128)} add(%fa, %fb)
    ROOT %ft = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%fsum, %fb)
  }

  %body {
    %bp = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) parameter(0)
    %ba = s32[6]{0:T(128)} get-tuple-element(%bp), index=0
    %bb = s32[6]{0:T(128)} get-tuple-element(%bp), index=1
    %bcopy = s32[6]{0:T(128)S(1)} copy(%bb)
    %bpair = (s32[6]{0:T(128)}, s32[6]{0:T(128)S(1)}) tuple(%ba, %bcopy)
    ROOT %bf = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)S(1)}) fusion(%bpair), kind=kLoop, calls=%fused_add
  }

  %cond {
    %cp = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) parameter(0)
    ROOT %lt = pred[] constant(false)
  }

  %callee {
    %kp = s32[6]{0:T(128)} parameter(0)
    ROOT %kn = s32[6]{0:T(128)} negate(%kp)
  }

  ENTRY %entry {
    %p0 = s32[6]{0:T(128)} parameter(0)
    %p1 = s32[6]{0:T(128)} parameter(1)
    %called = s32[6]{0:T(128)} call(%p0), to_apply=%callee
    %init = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%called, %p1)
    ROOT %loop = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) while(%init), condition=%cond, body=%body
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo_string));
  ExpectLocalDataflowMatchesAnalysis(module.get());
  std::unique_ptr<HloModule> reference = module->Clone(/*suffix=*/"");
  AddFusionWithCall(reference.get());
  EXPECT_TRUE(MemorySpacePropagation().Run(module.get()).value());
  EXPECT_TRUE(MemorySpacePropagation().Run(reference.get()).value());
  HloComputation* fused = module->GetComputationWithName("fused_add");
  ASSERT_NE(fused, nullptr);
  HloComputation* reference_fused =
      reference->GetComputationWithName("fused_add");
  ASSERT_NE(reference_fused, nullptr);
  for (const HloInstruction* instruction : fused->instructions()) {
    const HloInstruction* reference_instruction =
        reference_fused->GetInstructionWithName(instruction->name());
    ASSERT_NE(reference_instruction, nullptr);
    EXPECT_TRUE(
        Shape::Equal()(instruction->shape(), reference_instruction->shape()))
        << instruction->ToString() << " vs "
        << reference_instruction->ToString();
  }
  // Operand leaf {1} and output leaf {1} color the parameter's second element
  // and the root position holding it; output leaf {0} colors the sum.
  const HloInstruction* fp = fused->parameter_instruction(0);
  EXPECT_EQ(fp->shape().tuple_shapes(0).layout().memory_space(), 0);
  EXPECT_EQ(fp->shape().tuple_shapes(1).layout().memory_space(), 1);
  EXPECT_EQ(
      fused->GetInstructionWithName("fa")->shape().layout().memory_space(), 0);
  EXPECT_EQ(
      fused->GetInstructionWithName("fb")->shape().layout().memory_space(), 1);
  EXPECT_EQ(
      fused->GetInstructionWithName("fsum")->shape().layout().memory_space(),
      1);
}

// The scan covers the fusion computations of the run's threads only: a call
// inside a fusion computation of another thread leaves a main thread run on
// the local path, with the same result as the analysis.
TEST_F(MemorySpacePropagationTest, ScanCoversTheRunsThreadsOnly) {
  absl::string_view hlo_string = R"(
  HloModule ThreadScopedScan

  %callee {
    %cp = s32[6]{0:T(128)} parameter(0)
    ROOT %cn = s32[6]{0:T(128)} negate(%cp)
  }, execution_thread="other"

  %fused_call {
    %fp = s32[6]{0:T(128)} parameter(0)
    ROOT %call = s32[6]{0:T(128)} call(%fp), to_apply=%callee
  }, execution_thread="other"

  %other_computation {
    %op = s32[6]{0:T(128)} parameter(0)
    ROOT %of = s32[6]{0:T(128)} fusion(%op), kind=kLoop, calls=%fused_call
  }, execution_thread="other"

  %fused_add {
    %ap = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) parameter(0)
    %a0 = s32[6]{0:T(128)} get-tuple-element(%ap), index=0
    %a1 = s32[6]{0:T(128)} get-tuple-element(%ap), index=1
    ROOT %as = s32[6]{0:T(128)} add(%a0, %a1)
  }

  ENTRY %entry {
    %p0 = s32[6]{0:T(128)} parameter(0)
    %p1 = s32[6]{0:T(128)} parameter(1)
    %start = ((s32[6]{0:T(128)}), s32[6]{0:T(128)}, u32[]) async-start(%p0), calls=%other_computation, async_execution_thread="other"
    %done = s32[6]{0:T(128)} async-done(%start)
    %arg1 = s32[6]{0:T(128)S(1)} copy(%p1)
    %pair = (s32[6]{0:T(128)}, s32[6]{0:T(128)S(1)}) tuple(%done, %arg1)
    ROOT %fusion = s32[6]{0:T(128)S(1)} fusion(%pair), kind=kLoop, calls=%fused_add
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  const absl::flat_hash_set<absl::string_view> main_thread = {
      HloInstruction::kMainExecutionThread};
  EXPECT_FALSE(MemorySpacePropagation::HasLocalFusionDataflow(*module));
  EXPECT_TRUE(
      MemorySpacePropagation::HasLocalFusionDataflow(*module, main_thread));

  std::unique_ptr<HloModule> reference = module->Clone(/*suffix=*/"");
  AddFusionWithCall(reference.get());
  EXPECT_FALSE(
      MemorySpacePropagation::HasLocalFusionDataflow(*reference, main_thread));
  EXPECT_TRUE(MemorySpacePropagation().Run(module.get(), main_thread).value());
  EXPECT_TRUE(
      MemorySpacePropagation().Run(reference.get(), main_thread).value());
  for (absl::string_view name : {"fused_add", "fused_call"}) {
    HloComputation* computation = module->GetComputationWithName(name);
    ASSERT_NE(computation, nullptr) << name;
    HloComputation* reference_computation =
        reference->GetComputationWithName(name);
    ASSERT_NE(reference_computation, nullptr) << name;
    for (const HloInstruction* instruction : computation->instructions()) {
      const HloInstruction* reference_instruction =
          reference_computation->GetInstructionWithName(instruction->name());
      ASSERT_NE(reference_instruction, nullptr);
      EXPECT_TRUE(
          Shape::Equal()(instruction->shape(), reference_instruction->shape()))
          << instruction->ToString() << " vs "
          << reference_instruction->ToString();
    }
  }
  HloComputation* fused = module->GetComputationWithName("fused_add");
  EXPECT_EQ(fused->parameter_instruction(0)
                ->shape()
                .tuple_shapes(1)
                .layout()
                .memory_space(),
            1);
  EXPECT_EQ(
      fused->GetInstructionWithName("a1")->shape().layout().memory_space(), 1);
  EXPECT_EQ(fused->root_instruction()->shape().layout().memory_space(), 1);
}

// Every opcode Run() leaves to the dataflow analysis, inside a fusion
// computation, and the same instructions in the entry computation, where they
// do not matter. Async-update and async-done follow an async-start in the same
// computation, so one chain covers the three.
TEST_F(MemorySpacePropagationTest, InstructionsWithOtherDataflowNeedAnalysis) {
  constexpr absl::string_view kComputations = R"(
  HloModule NeedsAnalysis

  %reducer {
    %x = s32[] parameter(0)
    %y = s32[] parameter(1)
    ROOT %z = s32[] add(%x, %y)
  }

  %wrapped {
    %wp = s32[6]{0:T(128)} parameter(0)
    ROOT %wn = s32[6]{0:T(128)} negate(%wp)
  }

  %while_body {
    %lp = s32[6]{0:T(128)} parameter(0)
    ROOT %ln = s32[6]{0:T(128)} negate(%lp)
  }

  %while_cond {
    %cp = s32[6]{0:T(128)} parameter(0)
    ROOT %ct = pred[] constant(false)
  }

  %branch {
    %brp = s32[6]{0:T(128)} parameter(0)
    ROOT %brn = s32[6]{0:T(128)} negate(%brp)
  }
  )";
  // The instructions consume %fp of the given shape and end in a ROOT.
  struct Body {
    absl::string_view parameter_shape;
    absl::string_view text;
  };
  constexpr absl::string_view kArray = "s32[6]{0:T(128)}";
  const std::vector<Body> bodies = {
      {kArray,
       R"(ROOT %w = s32[6]{0:T(128)} while(%fp), condition=%while_cond, body=%while_body)"},
      {kArray, R"(%pr = pred[] constant(true)
    ROOT %c = s32[6]{0:T(128)} conditional(%pr, %fp, %fp), true_computation=%branch, false_computation=%wrapped)"},
      {kArray, R"(ROOT %call = s32[6]{0:T(128)} call(%fp), to_apply=%wrapped)"},
      {kArray,
       R"(%cs = (s32[6]{0:T(128)}, s32[6]{0:T(128)}, u32[]) copy-start(%fp)
    ROOT %g = s32[6]{0:T(128)} get-tuple-element(%cs), index=0)"},
      {"(s32[6]{0:T(128)}, s32[6]{0:T(128)}, u32[])",
       R"(ROOT %cd = s32[6]{0:T(128)} copy-done(%fp))"},
      {kArray, R"(ROOT %ard = s32[6]{0:T(128)} all-reduce-done(%fp))"},
      {kArray,
       R"(%ags = (s32[6]{0:T(128)}, s32[12]{0:T(128)}) all-gather-start(%fp), dimensions={0}
    ROOT %g = s32[6]{0:T(128)} get-tuple-element(%ags), index=0)"},
      {"(s32[6]{0:T(128)}, s32[12]{0:T(128)})",
       R"(%agd = s32[12]{0:T(128)} all-gather-done(%fp)
    ROOT %sl = s32[6]{0:T(128)} slice(%agd), slice={[0:6]})"},
      {kArray,
       R"(%cps = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) collective-permute-start(%fp), source_target_pairs={{0,1},{1,0}}
    ROOT %g = s32[6]{0:T(128)} get-tuple-element(%cps), index=0)"},
      {"(s32[6]{0:T(128)}, s32[6]{0:T(128)})",
       R"(ROOT %cpd = s32[6]{0:T(128)} collective-permute-done(%fp))"},
      {kArray,
       R"(%as = ((s32[6]{0:T(128)}), s32[6]{0:T(128)}, u32[]) async-start(%fp), calls=%wrapped
    %au = ((s32[6]{0:T(128)}), s32[6]{0:T(128)}, u32[]) async-update(%as)
    ROOT %ad = s32[6]{0:T(128)} async-done(%au))"},
      {kArray, R"(%tok = token[] after-all()
    %send = (s32[6]{0:T(128)}, u32[], token[]) send(%fp, %tok), channel_id=1
    ROOT %n = s32[6]{0:T(128)} negate(%fp))"},
      {"(s32[6]{0:T(128)}, u32[], token[])",
       R"(%rd = (s32[6]{0:T(128)}, token[]) recv-done(%fp), channel_id=2
    ROOT %got = s32[6]{0:T(128)} get-tuple-element(%rd), index=0)"},
  };
  // $0 is the parameter shape, $1 the body.
  constexpr absl::string_view kInsideFusion = R"(
  %fused {
    %fp = $0 parameter(0)
    $1
  }

  ENTRY %entry {
    %p0 = $0 parameter(0)
    ROOT %fusion = s32[6]{0:T(128)} fusion(%p0), kind=kLoop, calls=%fused
  }
  )";
  constexpr absl::string_view kInEntry = R"(
  %fused {
    %fq = $0 parameter(0)
    ROOT %fc = $0 copy(%fq)
  }

  ENTRY %entry {
    %p0 = $0 parameter(0)
    %fp = $0 fusion(%p0), kind=kLoop, calls=%fused
    $1
  }
  )";
  for (const Body& body : bodies) {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        ParseAndReturnUnverifiedModule(absl::StrCat(
            kComputations,
            absl::Substitute(kInsideFusion, body.parameter_shape, body.text))));
    EXPECT_FALSE(MemorySpacePropagation::HasLocalFusionDataflow(*module))
        << body.text;
    ASSERT_OK_AND_ASSIGN(
        module,
        ParseAndReturnUnverifiedModule(absl::StrCat(
            kComputations,
            absl::Substitute(kInEntry, body.parameter_shape, body.text))));
    EXPECT_TRUE(MemorySpacePropagation::HasLocalFusionDataflow(*module))
        << body.text;
  }
}

}  // namespace
}  // namespace xla
