/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/gather_simplifier.h"

#include <gtest/gtest.h>

#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/layout_util.h"
#include "xla/service/gather_scatter_utils.h"

namespace xla {
namespace {

class GatherSimplifierTest : public HloHardwareIndependentTestBase {};

TEST_F(GatherSimplifierTest, TransformsStartIndices) {
  // Verifies that GatherSimplifier
  // - Makes the index_vector_dim dimensions explicit
  // - Flattens start_indices into a 2d tensor.
  // - Undoes the flattening in the output.
  constexpr absl::string_view kModuleStr = R"(
    HloModule gather_simplifier

    ENTRY kernel_entry {
      operand = f32[33,34] parameter(0)
      indices = s32[42,43] parameter(1)
      ROOT gather = f32[42,43,7,8] gather(operand, indices),
          offset_dims={2,3},
          collapsed_slice_dims={},
          start_index_map={0},
          index_vector_dim=2,
          slice_sizes={7,8}
    })";

  RunAndFilecheckHloRewrite(kModuleStr, GatherSimplifier(), R"(
         CHECK: %[[VECTOR_DIM:.*]] = s32[42,43,1]{2,1,0} reshape(%indices)
         CHECK: %[[INDICES_2D:.*]] = s32[1806,1]{1,0} reshape(%[[VECTOR_DIM]])
         CHECK: %[[GATHER:.*]] = f32[1806,7,8]{{.*}} gather(
    CHECK-SAME:     %operand, %[[INDICES_2D]]),
    CHECK-SAME:     offset_dims={1,2},
    CHECK-SAME:     collapsed_slice_dims={},
    CHECK-SAME:     start_index_map={0},
    CHECK-SAME:     index_vector_dim=1,
    CHECK-SAME:     slice_sizes={7,8}
         CHECK: ROOT %{{.*}} = f32[42,43,7,8]{3,2,1,0} reshape(%[[GATHER]])
  )");
}

TEST_F(GatherSimplifierTest, RewritesScalarOperandGatherToBroadcast) {
  // A gather from a scalar (rank-0) operand selects that scalar for every
  // index, so it is rewritten to a broadcast of the operand. Previously
  // IsSimplifiedGather dereferenced the empty offset_dims out of bounds.
  constexpr absl::string_view kModuleStr = R"(
    HloModule gather_simplifier

    ENTRY kernel_entry {
      operand = f32[] parameter(0)
      indices = s32[7,0] parameter(1)
      ROOT gather = f32[7] gather(operand, indices),
          offset_dims={},
          collapsed_slice_dims={},
          start_index_map={},
          index_vector_dim=1,
          slice_sizes={}
    })";

  RunAndFilecheckHloRewrite(kModuleStr, GatherSimplifier(), R"(
      CHECK: ROOT {{.*}} = f32[7]{0} broadcast(%operand), dimensions={}
  )");
}

TEST_F(GatherSimplifierTest, RemovesCollapsedSliceDims) {
  // Verifies that GatherSimplifier sets the collapsed_slice_dims parameter to
  // the empty list.
  constexpr absl::string_view kModuleStr = R"(
    HloModule gather_simplifier

    ENTRY kernel_entry {
      operand = f32[33,34] parameter(0)
      indices = s32[42,1] parameter(1)
      ROOT gather = f32[42] gather(operand, indices),
          offset_dims={},
          collapsed_slice_dims={0,1},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={1,1}
    })";

  RunAndFilecheckHloRewrite(kModuleStr, GatherSimplifier(), R"(
           CHECK: %[[GATHER:.*]] = f32[42,1,1]{2,1,0} gather(%operand, %indices)
      CHECK-SAME:     offset_dims={1,2},
      CHECK-SAME:     collapsed_slice_dims={},
           CHECK: ROOT %{{.*}} = f32[42]{0} reshape(%[[GATHER]])
  )");
}

TEST_F(GatherSimplifierTest, KeepsStartIndexIntact) {
  // Verifies that GatherSimplifier does not change the start_index_map.
  constexpr absl::string_view kModuleStr = R"(
    HloModule gather_simplifier

    ENTRY kernel_entry {
      operand = f32[33,34,35] parameter(0)
      indices = s32[42,3] parameter(1)
      ROOT gather = f32[42,1,2,3] gather(operand, indices),
          offset_dims={1,2,3},
          collapsed_slice_dims={},
          start_index_map={2,0,1},
          index_vector_dim=1,
          slice_sizes={1,2,3}
    })";

  // Expect unchanged.
  RunAndFilecheckHloRewrite(kModuleStr, GatherSimplifier(), std::nullopt);
}

TEST_F(GatherSimplifierTest, CollapsesSomeDims) {
  // Verifies that GatherSimplifier can collapse only some dimensions.
  constexpr absl::string_view kModuleStr = R"(
    HloModule gather_simplifier

    ENTRY kernel_entry {
      operand = f32[33,34,35] parameter(0)
      indices = s32[42,1] parameter(1)
      ROOT gather = f32[7,42] gather(operand, indices),
          offset_dims={0},
          collapsed_slice_dims={0,2},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={1,7,1}
    })";

  RunAndFilecheckHloRewrite(kModuleStr, GatherSimplifier(), R"(
           CHECK: %[[GATHER:.*]] = f32[42,1,7,1]{3,2,1,0} gather(
           CHECK: %[[COLLAPSED:.*]] = f32[42,7]{1,0} reshape(%[[GATHER]])
           CHECK: ROOT {{.*}} = f32[7,42]{1,0} transpose(%[[COLLAPSED]]),
      CHECK-SAME: dimensions={1,0}
  )");
}

TEST_F(GatherSimplifierTest, ZeroDimStartIndices) {
  // Verifies that a zero-dimensional start indices tensor doesn't contribute
  // any dimensions to the output.
  constexpr absl::string_view kModuleStr = R"(
    HloModule gather_simplifier

    ENTRY kernel_entry {
      operand = f32[8,16] parameter(0)
      indices = s32[2] parameter(1)

      ROOT gather = f32[8,16] gather(f32[8,16] operand, s32[2] indices),
          offset_dims={0,1},
          collapsed_slice_dims={},
          start_index_map={0,1},
          index_vector_dim=0,
          slice_sizes={8,16}
    })";

  // The shape check is sufficient.
  RunAndFilecheckHloRewrite(kModuleStr, GatherSimplifier(), R"(
      CHECK: gather(
  )");
}

TEST_F(GatherSimplifierTest, ZeroSizeSlice) {
  // Verifies that slices of size zero result in a constant result.
  constexpr absl::string_view kModuleStr = R"(
    HloModule gather_simplifier

    ENTRY kernel_entry {
      operand = f32[0,2] parameter(0)
      indices = s32[3] parameter(1)

      ROOT gather = f32[3,2] gather(f32[0,2] operand, s32[3]{0} indices),
          offset_dims={1},
          collapsed_slice_dims={0},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={0,2}
    })";

  // The shape check is sufficient.
  RunAndFilecheckHloRewrite(kModuleStr, GatherSimplifier(), R"(
      CHECK: %[[ZERO:.*]] = f32[] constant(0)
      CHECK: ROOT {{.*}} = f32[3,2]{1,0} broadcast(%[[ZERO]]), dimensions={}
  )");
}

TEST_F(GatherSimplifierTest, SkipsExplicitBatchingAndFoldsSimplifiedZeroSlice) {
  for (absl::string_view batched : {absl::string_view(R"hlo(
HloModule BatchedRank2
ENTRY main {
  ROOT g = f32[4,3] gather(f32[4,3] parameter(0), s32[4,1] parameter(1)),
    offset_dims={1}, collapsed_slice_dims={}, operand_batching_dims={0},
    start_indices_batching_dims={0}, start_index_map={0},
    index_vector_dim=1, slice_sizes={1,3}
})hlo"),
                                    absl::string_view(R"hlo(
HloModule AllBatchedEmptyOffset
ENTRY main {
  ROOT g = f32[4] gather(f32[4] parameter(0), s32[4,0] parameter(1)),
    offset_dims={}, collapsed_slice_dims={}, operand_batching_dims={0},
    start_indices_batching_dims={0}, start_index_map={},
    index_vector_dim=1, slice_sizes={1}
})hlo"),
                                    absl::string_view(R"hlo(
HloModule BatchedWithCollapsed
ENTRY main {
  ROOT g = f32[4] gather(f32[4,5] parameter(0), s32[4,1] parameter(1)),
    offset_dims={}, collapsed_slice_dims={1}, operand_batching_dims={0},
    start_indices_batching_dims={0}, start_index_map={1},
    index_vector_dim=1, slice_sizes={1,1}
})hlo")}) {
    ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(batched));
    auto* g =
        Cast<HloGatherInstruction>(m->entry_computation()->root_instruction());
    EXPECT_FALSE(GatherSimplifier::IsSimplifiedGather(g));
    ASSERT_OK_AND_ASSIGN(bool changed, GatherSimplifier().Run(m.get()));
    EXPECT_FALSE(changed);
    struct DirectGatherSimplifier : GatherSimplifier {
      using GatherSimplifier::ExpandInstruction;
    };
    EXPECT_EQ(DirectGatherSimplifier().ExpandInstruction(g).status().code(),
              absl::StatusCode::kInvalidArgument);
  }

  // Simplified-layout gather with a zero slice still folds to zero broadcast.
  constexpr absl::string_view kSimpZero = R"hlo(
HloModule SimplifiedZeroSlice
ENTRY main {
  ROOT g = f32[3,0,2] gather(f32[4,2] parameter(0), s32[3,1] parameter(1)),
    offset_dims={1,2}, collapsed_slice_dims={}, start_index_map={0},
    index_vector_dim=1, slice_sizes={0,2}
})hlo";
  ASSERT_OK_AND_ASSIGN(auto zm, ParseAndReturnVerifiedModule(kSimpZero));
  EXPECT_FALSE(GatherSimplifier::IsSimplifiedGather(
      Cast<HloGatherInstruction>(zm->entry_computation()->root_instruction())));
  ASSERT_OK_AND_ASSIGN(bool changed, GatherSimplifier().Run(zm.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(zm->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kBroadcast);
}

TEST_F(GatherSimplifierTest, PreservesAndGuardsDynamicDimensions) {
  // Single dynamic batch dim + full-slice dynamic operand dim are preserved.
  constexpr absl::string_view kPreserved = R"hlo(
HloModule DynGatherSimplify
ENTRY main {
  op = f32[<=5,4] set-dimension-size(f32[5,4] parameter(0), s32[] parameter(2)), dimensions={0}
  idx = s32[<=2,3,1] set-dimension-size(s32[2,3,1] parameter(1), s32[] parameter(3)), dimensions={0}
  ROOT g = f32[<=2,3,<=5] gather(op, idx), offset_dims={2},
    collapsed_slice_dims={1}, start_index_map={1},
    index_vector_dim=2, slice_sizes={5,1}
})hlo";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kPreserved));
  ASSERT_OK_AND_ASSIGN(bool changed, GatherSimplifier().Run(m.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_TRUE(root->shape().is_dynamic_dimension(0));
  EXPECT_FALSE(root->shape().is_dynamic_dimension(1));
  EXPECT_TRUE(root->shape().is_dynamic_dimension(2));
  // Idempotent on second run.
  ASSERT_OK_AND_ASSIGN(changed, GatherSimplifier().Run(m.get()));
  EXPECT_FALSE(changed);

  // Dual dynamic batch dims and dynamic collapsed <=1 dim are skipped cleanly.
  for (absl::string_view skip_hlo : {absl::string_view(R"hlo(
HloModule DualDynBatch
ENTRY main {
  d0 = s32[] parameter(2)
  t = s32[<=2,3,1] set-dimension-size(s32[2,3,1] parameter(1), d0), dimensions={0}
  idx = s32[<=2,<=3,1] set-dimension-size(t, d0), dimensions={1}
  ROOT g = f32[<=2,<=3,4] gather(f32[5,4] parameter(0), idx), offset_dims={2},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=2, slice_sizes={1,4}
})hlo"),
                                     absl::string_view(R"hlo(
HloModule DynCollapsedUnit
ENTRY main {
  op = f32[<=1,4] set-dimension-size(f32[1,4] parameter(0), s32[] parameter(2)), dimensions={0}
  ROOT g = f32[3,4] gather(op, s32[3,1] parameter(1)), offset_dims={1},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,4}
})hlo")}) {
    ASSERT_OK_AND_ASSIGN(auto sm, ParseAndReturnVerifiedModule(skip_hlo));
    ASSERT_OK_AND_ASSIGN(changed, GatherSimplifier().Run(sm.get()));
    EXPECT_FALSE(changed);
  }
}

TEST_F(GatherSimplifierTest,
       PreservesSubByteMemorySpaceUnlayoutedTransposeAndControlDeps) {
  constexpr absl::string_view kSubByte = R"hlo(
HloModule SubByteSimplify
ENTRY main {
  op = s4[6,4]{1,0:E(4)S(1)} parameter(0)
  idx = s4[2,3]{1,0:E(4)S(1)} parameter(1)
  tok = token[] after-all()
  g = s4[4,2,3]{2,1,0:E(4)S(1)} gather(op, idx), offset_dims={0},
    collapsed_slice_dims={0}, start_index_map={0},
    index_vector_dim=2, slice_sizes={1,4}, control-predecessors={tok}
  sink = token[] after-all(), control-predecessors={g}
  ROOT out = (s4[4,2,3]{2,1,0:E(4)S(1)}, token[]) tuple(g, sink)
})hlo";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kSubByte));
  ASSERT_OK_AND_ASSIGN(bool changed, GatherSimplifier().Run(m.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* tok = FindInstruction(m.get(), "tok");
  const HloInstruction* sink = FindInstruction(m.get(), "sink");
  const HloInstruction* res =
      m->entry_computation()->root_instruction()->operand(0);
  EXPECT_EQ(sink->control_predecessors().size(), 1);
  EXPECT_EQ(sink->control_predecessors()[0], res);
  int gather_count = 0;
  for (const HloInstruction* inst : m->entry_computation()->instructions()) {
    if (inst->shape().IsArray() && inst->shape().element_type() == S4) {
      EXPECT_EQ(inst->shape().layout().element_size_in_bits(), 4);
      EXPECT_EQ(inst->shape().layout().memory_space(), 1);
    }
    if (inst->opcode() == HloOpcode::kGather) {
      ++gather_count;
      EXPECT_EQ(inst->control_predecessors().size(), 1);
      EXPECT_EQ(inst->control_predecessors()[0], tok);
    }
  }
  EXPECT_EQ(gather_count, 1);

  // Unlayouted MaybeTranspose does not crash and preserves unlayouted status.
  HloInstruction* p = m->entry_computation()->parameter_instruction(0);
  p->mutable_shape()->clear_layout();
  ASSERT_OK_AND_ASSIGN(HloInstruction * tr, MaybeTranspose(p, {1, 0}));
  EXPECT_FALSE(LayoutUtil::HasLayout(tr->shape()));
}

}  // namespace
}  // namespace xla
