/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/gather_expander.h"

#include <gmock/gmock.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/transforms/expanders/scatter_expander.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/types.h"

namespace xla {
namespace {

class GatherExpanderTest : public HloHardwareIndependentTestBase {
 protected:
  void CheckWhileBody(HloModule* module, absl::string_view expected) {
    std::vector<HloInstruction*> while_instructions =
        FindInstructions(module, HloOpcode::kWhile);
    EXPECT_EQ(while_instructions.size(), 1);
    HloComputation* while_body = while_instructions[0]->while_body();
    EXPECT_TRUE(*RunFileCheck(
        while_body->ToString(
            HloPrintOptions{}.set_include_layout_in_shapes(false)),
        expected));
  }
};

TEST_F(GatherExpanderTest, ErrorStatusOnTooManyIndices) {
  const std::string hlo_text = R"(
HloModule TensorFlowGatherMultipleBatchDims

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2147483647,5] parameter(1)
  ROOT gather = s32[2147483647,3,5] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=2,
      slice_sizes={3, 1}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  absl::Status status = GatherExpander{GatherExpander::kEliminateAllGathers}
                            .Run(module.get())
                            .status();
  EXPECT_EQ(status.code(), tsl::error::UNIMPLEMENTED);

  ASSERT_THAT(
      status.message(),
      ::testing::HasSubstr("Gather operations with more than 2147483647 gather "
                           "indices are not supported."));
}

TEST_F(GatherExpanderTest, AvoidDegenerateDims) {
  const std::string hlo_text = R"(
HloModule TensorFlowGatherV2

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[3,2] gather(operand, indices),
      offset_dims={0},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=1,
      slice_sizes={3, 1}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      GatherExpander{GatherExpander::kEliminateAllGathers}.Run(module.get()));
  ASSERT_TRUE(changed);

  HloInstruction* while_instr = nullptr;
  for (auto* instr : module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kWhile) {
      ASSERT_EQ(while_instr, nullptr)
          << "Expected exactly one while instruction in the entry computation "
             "after gather expansion";
      while_instr = instr;
    }
  }

  ASSERT_NE(while_instr, nullptr)
      << "Expected exactly one while instruction in the entry computation "
         "after gather expansion";

  // We want to avoid create while loop with shapes that have degenerate
  // dimensions for TF gather.  In this case we expect the loop state to be of
  // the shape (sNN[], s32[3,3]{1,0}, s32[2]{0}, s32[2,3]{1,0}).  The leading
  // sNN is an implementation detail from WhileUtil::MakeCountedLoop so we don't
  // check it here (though in theory the form of the while loop state is itself
  // an implementation detail from WhileUtil::MakeCountedLoop).

  const Shape& while_shape = while_instr->shape();
  ASSERT_TRUE(while_shape.IsTuple());
  ASSERT_EQ(ShapeUtil::TupleElementCount(while_shape), 4);

  EXPECT_TRUE(ShapeUtil::SameDimensions(
      ShapeUtil::MakeShape(S32, {3, 3}),
      ShapeUtil::GetTupleElementShape(while_shape, 1)));

  EXPECT_TRUE(ShapeUtil::SameDimensions(
      ShapeUtil::MakeShape(S32, {2}),
      ShapeUtil::GetTupleElementShape(while_shape, 2)));

  EXPECT_TRUE(ShapeUtil::SameDimensions(
      ShapeUtil::MakeShape(S32, {2, 3}),
      ShapeUtil::GetTupleElementShape(while_shape, 3)));
}

TEST_F(GatherExpanderTest, CheckOpMetadata) {
  const std::string hlo_text = R"(
HloModule TensorFlowGatherV2

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[3,2] gather(operand, indices),
      offset_dims={0},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=1,
      slice_sizes={3, 1}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  OpMetadata metadata;
  metadata.set_op_name("Gather");
  module->entry_computation()->root_instruction()->set_metadata(metadata);
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      GatherExpander{GatherExpander::kEliminateAllGathers}.Run(module.get()));
  ASSERT_TRUE(changed);

  HloInstruction* while_instr = nullptr;
  for (auto* instr : module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kWhile) {
      ASSERT_EQ(while_instr, nullptr)
          << "Expected exactly one while instruction in the entry computation "
             "after gather expansion";
      while_instr = instr;
    }
  }

  ASSERT_NE(while_instr, nullptr)
      << "Expected exactly one while instruction in the entry computation "
         "after gather expansion";
  EXPECT_EQ(while_instr->metadata().op_name(), "Gather");
}

TEST_F(GatherExpanderTest, EliminateSimpleGathersSkipsNontrivialGather) {
  const std::string hlo_text = R"(
HloModule TensorFlowGatherV1

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[2,3] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1, 3}
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  GatherExpander pass(GatherExpander::kEliminateSimpleGathers);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  ASSERT_FALSE(changed);
}

TEST_F(GatherExpanderTest, EliminateSimpleGathersRewritesTrivialGather) {
  const std::string hlo_text = R"(
HloModule test

ENTRY main {
  operand = s32[100] parameter(0)
  indices = s32[1] parameter(1)
  ROOT gather = s32[10] gather(operand, indices),
      offset_dims={0},
      collapsed_slice_dims={},
      start_index_map={0},
      index_vector_dim=0,
      slice_sizes={10}
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  GatherExpander pass(GatherExpander::kEliminateAllGathers);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  ASSERT_TRUE(changed);
  ASSERT_FALSE(hlo_query::ContainsInstrWithOpcode(module->entry_computation(),
                                                  {HloOpcode::kGather}));
}

TEST_F(GatherExpanderTest, GatherIsBroadcast) {
  const std::string hlo_text = R"(
HloModule test

ENTRY main {
  operand = s32[1,3] parameter(0)
  indices = s32[7,5] parameter(1)
  ROOT gather = s32[7,3,5] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=2,
      slice_sizes={1,3}
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  GatherExpander pass(GatherExpander::kEliminateSimpleGathers);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  ASSERT_TRUE(changed);
  ASSERT_FALSE(hlo_query::ContainsInstrWithOpcode(module->entry_computation(),
                                                  {HloOpcode::kGather}));
  ASSERT_TRUE(hlo_query::ContainsInstrWithOpcode(module->entry_computation(),
                                                 {HloOpcode::kBroadcast}));
  module->VerifyOrAddFailure("after-gather-expander.");
}

TEST_F(GatherExpanderTest, GatherIsBroadcastBatchDim) {
  const std::string hlo_text = R"(
HloModule test

ENTRY main {
  operand = s32[1,3,1] parameter(0)
  indices = s32[1,5] parameter(1)
  ROOT gather = s32[1,3,5] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={2},
      start_index_map={0},
      index_vector_dim=2,
      slice_sizes={1,3,1},
      operand_batching_dims={0},
      start_indices_batching_dims={0}
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  GatherExpander pass(GatherExpander::kEliminateSimpleGathers);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  ASSERT_TRUE(changed);
  ASSERT_FALSE(hlo_query::ContainsInstrWithOpcode(module->entry_computation(),
                                                  {HloOpcode::kGather}));
  ASSERT_TRUE(hlo_query::ContainsInstrWithOpcode(module->entry_computation(),
                                                 {HloOpcode::kBroadcast}));
  module->VerifyOrAddFailure("after-gather-expander.");
}

TEST_F(GatherExpanderTest, GatherToLoopWithBatchDims) {
  const std::string hlo_text = R"(
HloModule GatherWithBatchDims

ENTRY main {
  operand = s32[5,2] parameter(0)
  indices = s64[5,1] parameter(1)
  ROOT gather = s32[5,1] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={},
      start_index_map={1},
      index_vector_dim=1,
      operand_batching_dims={0},
      start_indices_batching_dims={0},
      slice_sizes={1,1}
}
)";
  const std::string expected = R"(
  //CHECK: (s32[], s32[5,2], s64[5,1], s32[5,1])) -> (s32[], s32[5,2], s64[5,1], s32[5,1]) {
  //CHECK: %[[PARAM:.*]] = (s32[], s32[5,2], s64[5,1], s32[5,1]) parameter(0)
  //CHECK: %[[I_0:.*]] = s32[] get-tuple-element(%[[PARAM]]), index=0
  //CHECK: %[[CONSTANT1:.*]] = s32[] constant(1)
  //CHECK: %[[I_PLUS_1:.*]] = s32[] add(%[[I_0]], %[[CONSTANT1]])
  //CHECK: %[[OPERAND:.*]] = s32[5,2] get-tuple-element(%[[PARAM]]), index=1
  //CHECK: %[[START_INDICES:.*]] = s64[5,1] get-tuple-element(%[[PARAM]]), index=2
  //CHECK: %[[RESULT:.*]] = s32[5,1] get-tuple-element(%[[PARAM]]), index=3
  //CHECK: %[[CONVERT:.*]] = s64[] convert(%[[I_0]])
  //CHECK: %[[I_1D_1:.*]] = s64[1] broadcast(%[[CONVERT]])
  //CHECK: %[[I_1D_2:.*]] = s32[1] broadcast(%[[I_0]])

  //CHECK: %[[START_INDICES_INDEX_D1_PAD:.*]] = s32[] constant(0)
  //CHECK: %[[START_INDICES_INDEX_VECTOR:.*]] = s32[2] pad(%[[I_1D_2]], %[[START_INDICES_INDEX_D1_PAD]]), padding=0_1
  //CHECK: %[[START_INDICES_INDEX_D0_SLICE:.*]] = s32[1] slice(%[[START_INDICES_INDEX_VECTOR]]), slice={[0:1]}
  //CHECK: %[[START_INDICES_INDEX_D0:.*]] = s32[] reshape(%[[START_INDICES_INDEX_D0_SLICE]])
  //CHECK: %[[START_INDICES_INDEX_D1_SLICE:.*]] = s32[1] slice(%[[START_INDICES_INDEX_VECTOR]]), slice={[1:2]}
  //CHECK: %[[START_INDICES_INDEX_D1:.*]] = s32[] reshape(%[[START_INDICES_INDEX_D1_SLICE]])
  //CHECK: %[[INDEX_VECTOR:.*]] = s64[1,1] dynamic-slice(%[[START_INDICES]], %[[START_INDICES_INDEX_D0]], %[[START_INDICES_INDEX_D1]])

  //CHECK: %[[OFFSET_RAW:.*]] = s64[1] reshape(%[[INDEX_VECTOR]])
  //CHECK: %[[OFFSET:.*]] = s64[1] slice(%[[OFFSET_RAW]])
  //CHECK: %[[OPERAND_INDEX:.*]] = s64[2] concatenate(%[[I_1D_1]], %[[OFFSET]])
  //CHECK: %[[OPERAND_INDEX_D0_RAW:.*]] = s64[1] slice(%[[OPERAND_INDEX]]), slice={[0:1]}
  //CHECK: %[[OPERAND_INDEX_D0:.*]] = s64[] reshape(%[[OPERAND_INDEX_D0_RAW]])
  //CHECK: %[[OPERAND_INDEX_D1_RAW:.*]] = s64[1] slice(%[[OPERAND_INDEX]]), slice={[1:2]}
  //CHECK: %[[OPERAND_INDEX_D1:.*]] = s64[] reshape(%[[OPERAND_INDEX_D1_RAW]])
  //CHECK: %[[RESULT_SLICE_RAW0:.*]] = s32[1,1] dynamic-slice(%[[OPERAND]], %[[OPERAND_INDEX_D0]], %[[OPERAND_INDEX_D1]])

  //CHECK: %[[RESULT_SLICE_RAW1:.*]] = s32[1] reshape(%[[RESULT_SLICE_RAW0]])
  //CHECK: %[[RESULT_SLICE:.*]] = s32[1,1] reshape(%[[RESULT_SLICE_RAW1]])
  //CHECK: %[[RESULT_INDEX_D1_PAD:.*]] = s32[] constant(0)
  //CHECK: %[[RESULT_INDEX_VECTOR:.*]] = s32[2] pad(%[[I_1D_2]], %[[RESULT_INDEX_D1_PAD]]), padding=0_1
  //CHECK: %[[RESULT_INDEX_D0_SLICE:.*]] = s32[1] slice(%[[RESULT_INDEX_VECTOR]]), slice={[0:1]}
  //CHECK: %[[RESULT_INDEX_D0:.*]] = s32[] reshape(%[[RESULT_INDEX_D0_SLICE]])
  //CHECK: %[[RESULT_INDEX_D1_SLICE:.*]] = s32[1] slice(%[[RESULT_INDEX_VECTOR]]), slice={[1:2]}
  //CHECK: %[[RESULT_INDEX_D1:.*]] = s32[] reshape(%[[RESULT_INDEX_D1_SLICE]])
  //CHECK: %[[UPDATED_RESULT:.*]] = s32[5,1] dynamic-update-slice(%[[RESULT]], %[[RESULT_SLICE]], %[[RESULT_INDEX_D0]], %[[RESULT_INDEX_D1]])

  //CHECK: ROOT %{{.*}} = (s32[], s32[5,2], s64[5,1], s32[5,1]) tuple(%[[I_PLUS_1]], %[[OPERAND]], %[[START_INDICES]], %[[UPDATED_RESULT]])
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      GatherExpander{GatherExpander::kEliminateAllGathers}.Run(module.get()));
  ASSERT_TRUE(changed);
  CheckWhileBody(module.get(), expected);
}

TEST_F(GatherExpanderTest, GatherToLoopWithBatchDimsAndCollapsedDims) {
  const std::string hlo_text = R"(
HloModule GatherWithBatchAndCollapsedDims

ENTRY main {
  operand = s32[7,3,4,5] parameter(0)
  indices = s32[5,2,7] parameter(1)
  ROOT gather = s32[5,3,2,7] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={2},
      start_index_map={2},
      index_vector_dim=3,
      operand_batching_dims={0,3},
      start_indices_batching_dims={2,0},
      slice_sizes={1,3,1,1}
}
)";
  // Compared with the previous test, this test adds complexity in calculating
  // the indices for the operand. As such, we mostly check the operand indices
  // here.
  const std::string expected = R"(
  //CHECK: (s32[], s32[7,3,4,5], s32[70], s32[70,3])) -> (s32[], s32[7,3,4,5], s32[70], s32[70,3]) {
  //CHECK: %[[PARAM:.*]] = (s32[], s32[7,3,4,5], s32[70], s32[70,3]) parameter(0)
  //CHECK: %[[I:.*]] = s32[] get-tuple-element(%[[PARAM]]), index=0

  //CHECK: %[[CONSTANT1:.*]] = s32[] constant(1)
  //CHECK: %[[I_PLUS_1:.*]] = s32[] add(%[[I]], %[[CONSTANT1]])
  //CHECK: %[[OPERAND:.*]] = s32[7,3,4,5] get-tuple-element(%[[PARAM]]), index=1
  //CHECK: %[[START_INDICES:.*]] = s32[70] get-tuple-element(%[[PARAM]]), index=2

  //CHECK: %[[CONSTANT7:.*]] = s32[] constant(7)
  //CHECK: %[[BD0_RAW:.*]] = s32[] remainder(%[[I]], %[[CONSTANT7]])
  //CHECK: %[[BD0:.*]] = s32[1] broadcast(%[[BD0_RAW]])
  //CHECK: %[[CONSTANT0:.*]] = s32[1] constant({0})
  //CHECK: %[[I_1D_1:.*]] = s32[1] broadcast(%[[I]])
  //CHECK: %[[START_INDICES_INDEX_RAW:.*]] = s32[1] slice(%[[I_1D_1]])
  //CHECK: %[[START_INDICES_INDEX:.*]] = s32[] reshape(%[[START_INDICES_INDEX_RAW]])
  //CHECK: %[[INDEX_VECTOR:.*]] = s32[1] dynamic-slice(%[[START_INDICES]], %[[START_INDICES_INDEX]])

  //CHECK: %[[OFFSET:.*]] = s32[1] slice(%[[INDEX_VECTOR]])
  //CHECK: %[[BD1:.*]] = s32[] divide(%[[I]], %[[CONSTANT7]])
  //CHECK: %[[CONSTANT2:.*]] = s32[] constant(2)
  //CHECK: %[[BD2_RAW:.*]] = s32[] divide(%[[BD1]], %[[CONSTANT2]])
  //CHECK: %[[BD2:.*]] = s32[1] broadcast(%[[BD2_RAW]])
  //CHECK: %[[OPERAND_INDEX:.*]] = s32[4] concatenate(%[[BD0]], %[[CONSTANT0]], %[[OFFSET]], %[[BD2]])

  //CHECK: %[[OPERAND_INDEX_D0_RAW:.*]] = s32[1] slice(%[[OPERAND_INDEX]]), slice={[0:1]}
  //CHECK: %[[OPERAND_INDEX_D0:.*]] = s32[] reshape(%[[OPERAND_INDEX_D0_RAW]])
  //CHECK: %[[OPERAND_INDEX_D1_RAW:.*]] = s32[1] slice(%[[OPERAND_INDEX]]), slice={[1:2]}
  //CHECK: %[[OPERAND_INDEX_D1:.*]] = s32[] reshape(%[[OPERAND_INDEX_D1_RAW]])
  //CHECK: %[[OPERAND_INDEX_D2_RAW:.*]] = s32[1] slice(%[[OPERAND_INDEX]]), slice={[2:3]}
  //CHECK: %[[OPERAND_INDEX_D2:.*]] = s32[] reshape(%[[OPERAND_INDEX_D2_RAW]])
  //CHECK: %[[OPERAND_INDEX_D3_RAW:.*]] = s32[1] slice(%[[OPERAND_INDEX]]), slice={[3:4]}
  //CHECK: %[[OPERAND_INDEX_D3:.*]] = s32[] reshape(%[[OPERAND_INDEX_D3_RAW]])
  //CHECK: %{{.*}} = s32[1,3,1,1] dynamic-slice(%[[OPERAND]], %[[OPERAND_INDEX_D0]], %[[OPERAND_INDEX_D1]], %[[OPERAND_INDEX_D2]], %[[OPERAND_INDEX_D3]])
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      GatherExpander{GatherExpander::kEliminateAllGathers}.Run(module.get()));
  ASSERT_TRUE(changed);
  CheckWhileBody(module.get(), expected);
}

TEST_F(GatherExpanderTest, NarrowSignedAndUnsignedBatchedGatherAndScatter) {
  HloEvaluator eval;
  // Batched s8[200,1] (with negative index -3 and rows >= 128), s8[16,16,1],
  // u8[300,1], zero-divisor u8[2,256,1], and sub-byte s4[20,1]{1,0:E(4)}.
  constexpr absl::string_view kS8Gather = R"hlo(
HloModule NarrowS8BatchedGather
ENTRY main {
  op = s32[200,4] parameter(0)
  idx = s8[200,1] parameter(1)
  ROOT g = s32[200,1] gather(op, idx), offset_dims={1},
    collapsed_slice_dims={}, operand_batching_dims={0},
    start_indices_batching_dims={0}, start_index_map={1},
    index_vector_dim=1, slice_sizes={1,1}
})hlo";
  Literal op200(ShapeUtil::MakeShape(S32, {200, 4}));
  Literal idx200(ShapeUtil::MakeShape(S8, {200, 1}));
  for (int i = 0; i < 200; ++i) {
    for (int j = 0; j < 4; ++j) op200.Set<int32_t>({i, j}, i * 10 + j);
    idx200.Set<int8_t>({i, 0},
                       i % 5 == 0 ? int8_t{-3} : static_cast<int8_t>(i % 4));
  }
  ASSERT_OK_AND_ASSIGN(auto m_s8, ParseAndReturnVerifiedModule(kS8Gather));
  ASSERT_OK_AND_ASSIGN(Literal ref_s8, eval.Evaluate(*m_s8, {&op200, &idx200}));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      GatherExpander(GatherExpander::kEliminateAllGathers).Run(m_s8.get()));
  ASSERT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(Literal got_s8, eval.Evaluate(*m_s8, {&op200, &idx200}));
  EXPECT_EQ(got_s8, ref_s8);

  // Zero-divisor u8[2,256,1], wraparound u8[300,1], and s4[20,1]{1,0:E(4)}.
  constexpr absl::string_view kU8ZeroDiv = R"hlo(
HloModule NarrowU8ZeroDivGather
ENTRY main {
  b0 = s32[2,256,3] iota(), iota_dimension=0
  b1 = s32[2,256,3] iota(), iota_dimension=1
  b2 = s32[2,256,3] iota(), iota_dimension=2
  k1000 = s32[2,256,3] broadcast(s32[] constant(1000)), dimensions={}
  k10 = s32[2,256,3] broadcast(s32[] constant(10)), dimensions={}
  m0 = s32[2,256,3] multiply(b0, k1000)
  m1 = s32[2,256,3] multiply(b1, k10)
  m01 = s32[2,256,3] add(m0, m1)
  op = s32[2,256,3] add(m01, b2)
  idx = u8[2,256,1] broadcast(u8[] constant(1)), dimensions={}
  ROOT g = s32[2,256,1] gather(op, idx), offset_dims={2},
    collapsed_slice_dims={}, operand_batching_dims={0,1},
    start_indices_batching_dims={0,1}, start_index_map={2},
    index_vector_dim=2, slice_sizes={1,1,1}
})hlo";
  ASSERT_OK_AND_ASSIGN(auto m_u8, ParseAndReturnVerifiedModule(kU8ZeroDiv));
  ASSERT_OK_AND_ASSIGN(
      changed,
      GatherExpander(GatherExpander::kEliminateAllGathers).Run(m_u8.get()));
  ASSERT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(Literal got_u8, HloEvaluator().Evaluate(*m_u8, {}));
  for (int b = 0; b < 2; ++b) {
    for (int j = 0; j < 256; ++j) {
      EXPECT_EQ(got_u8.Get<int32_t>({b, j, 0}), b * 1000 + j * 10 + 1);
    }
  }

  constexpr absl::string_view kU8Wrap = R"hlo(
HloModule NarrowU8WrapGather
ENTRY main {
  op = s32[300,3] iota(), iota_dimension=0
  idx = u8[300,1] broadcast(u8[] constant(2)), dimensions={}
  ROOT g = s32[300,1] gather(op, idx), offset_dims={1},
    collapsed_slice_dims={}, operand_batching_dims={0},
    start_indices_batching_dims={0}, start_index_map={1},
    index_vector_dim=1, slice_sizes={1,1}
})hlo";
  ASSERT_OK_AND_ASSIGN(auto m_u8w, ParseAndReturnVerifiedModule(kU8Wrap));
  ASSERT_OK_AND_ASSIGN(
      changed,
      GatherExpander(GatherExpander::kEliminateAllGathers).Run(m_u8w.get()));
  ASSERT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(Literal got_u8w, HloEvaluator().Evaluate(*m_u8w, {}));
  for (int i = 0; i < 300; ++i) {
    EXPECT_EQ(got_u8w.Get<int32_t>({i, 0}), i);
  }

  // Sub-byte s4[20,1]{1,0:E(4)} batched gather + matching s8[200,1] scatter.
  constexpr absl::string_view kS4Gather = R"hlo(
HloModule NarrowS4BatchedGather
ENTRY main {
  r0 = s32[20,3] iota(), iota_dimension=0
  r1 = s32[20,3] iota(), iota_dimension=1
  c10 = s32[20,3] broadcast(s32[] constant(10)), dimensions={}
  r0_scaled = s32[20,3] multiply(r0, c10)
  op = s32[20,3] add(r0_scaled, r1)
  idx = s4[20,1]{1,0:E(4)} broadcast(s4[] constant(-1)), dimensions={}
  ROOT g = s32[20,1] gather(op, idx), offset_dims={1},
    collapsed_slice_dims={}, operand_batching_dims={0},
    start_indices_batching_dims={0}, start_index_map={1},
    index_vector_dim=1, slice_sizes={1,1}
})hlo";
  ASSERT_OK_AND_ASSIGN(auto m_s4, ParseAndReturnVerifiedModule(kS4Gather));
  ASSERT_OK_AND_ASSIGN(
      changed,
      GatherExpander(GatherExpander::kEliminateAllGathers).Run(m_s4.get()));
  ASSERT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(Literal got_s4, HloEvaluator().Evaluate(*m_s4, {}));
  for (int i = 0; i < 20; ++i) {
    EXPECT_EQ(got_s4.Get<int32_t>({i, 0}), i * 10);
  }

  constexpr absl::string_view kS8Scatter = R"hlo(
HloModule NarrowS8BatchedScatter
add_s32 {
  ROOT r = s32[] add(s32[] parameter(0), s32[] parameter(1))
}
ENTRY main {
  op = s32[200,2] broadcast(s32[] constant(0)), dimensions={}
  idx = s8[200,1] broadcast(s8[] constant(1)), dimensions={}
  upd = s32[200] iota(), iota_dimension=0
  ROOT sc = s32[200,2] scatter(op, idx, upd), update_window_dims={},
    inserted_window_dims={1}, input_batching_dims={0},
    scatter_indices_batching_dims={0}, scatter_dims_to_operand_dims={1},
    index_vector_dim=1, to_apply=add_s32
})hlo";
  ASSERT_OK_AND_ASSIGN(auto m_sc, ParseAndReturnVerifiedModule(kS8Scatter));
  ASSERT_OK_AND_ASSIGN(Literal ref_sc, HloEvaluator().Evaluate(*m_sc, {}));
  ASSERT_OK_AND_ASSIGN(
      changed,
      ScatterExpander(ScatterExpander::kEliminateAllScatters).Run(m_sc.get()));
  ASSERT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(Literal got_sc, HloEvaluator().Evaluate(*m_sc, {}));
  EXPECT_EQ(got_sc, ref_sc);
}

TEST_F(GatherExpanderTest, OverlappingBatchingDimsAndStartIndexMap) {
  HloEvaluator eval;
  constexpr absl::string_view kOverlap = R"hlo(
HloModule OverlapBatchGather
ENTRY main {
  op = s32[4,3,5] parameter(0)
  idx = s32[4,5,1] parameter(1)
  ROOT g = s32[4,5,3] gather(op, idx), offset_dims={2},
    collapsed_slice_dims={}, operand_batching_dims={0,2},
    start_indices_batching_dims={0,1}, start_index_map={0},
    index_vector_dim=2, slice_sizes={1,3,1}
})hlo";
  Literal op(ShapeUtil::MakeShape(S32, {4, 3, 5}));
  Literal idx(ShapeUtil::MakeShape(S32, {4, 5, 1}));
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 5; ++c)
        op.Set<int32_t>({a, b, c}, a * 100 + b * 10 + c);
    }
    for (int c = 0; c < 5; ++c) idx.Set<int32_t>({a, c, 0}, 3 - a);
  }
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kOverlap));
  ASSERT_OK_AND_ASSIGN(Literal ref, eval.Evaluate(*m, {&op, &idx}));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      GatherExpander(GatherExpander::kEliminateAllGathers).Run(m.get()));
  ASSERT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(Literal got, eval.Evaluate(*m, {&op, &idx}));
  EXPECT_EQ(got, ref);
}

TEST_F(GatherExpanderTest, ZeroOperandF8E8M0FNUProducesZeroLikeNotNan) {
  HloEvaluator eval;
  constexpr absl::string_view kE8M0 = R"hlo(
HloModule ZeroOperandE8M0
ENTRY main {
  op = f8e8m0fnu[0,3] parameter(0)
  idx = s32[2,1] constant({{0}, {0}})
  ROOT g = f8e8m0fnu[2,3] gather(op, idx), offset_dims={1},
    collapsed_slice_dims={0}, start_index_map={0},
    index_vector_dim=1, slice_sizes={0,3}
})hlo";
  Literal op(ShapeUtil::MakeShape(F8E8M0FNU, {0, 3}));
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kE8M0));
  ASSERT_OK_AND_ASSIGN(Literal ref, eval.Evaluate(*m, {&op}));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      GatherExpander(GatherExpander::kEliminateSimpleGathers).Run(m.get()));
  ASSERT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(Literal got, eval.Evaluate(*m, {&op}));
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_FALSE(Eigen::numext::isnan(got.Get<tsl::float8_e8m0fnu>({i, j})));
      EXPECT_EQ(got.Get<tsl::float8_e8m0fnu>({i, j}),
                ref.Get<tsl::float8_e8m0fnu>({i, j}));
    }
  }
}

TEST_F(GatherExpanderTest, UnboundedOverflowAndDynamicGathersHandledSafely) {
  // Unbounded s32[?,2,1] must not UBSan-overflow or wrap trip_count to 0.
  ASSERT_OK_AND_ASSIGN(auto unb, ParseAndReturnUnverifiedModule(R"hlo(
HloModule UnboundedGather
ENTRY main {
  op = f32[4,3] parameter(0)
  idx = s32[?,2,1] parameter(1)
  ROOT g = f32[?,2,3] gather(op, idx), offset_dims={2},
    collapsed_slice_dims={0}, start_index_map={0},
    index_vector_dim=2, slice_sizes={1,3}
})hlo"));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      GatherExpander(GatherExpander::kEliminateSimpleGathers).Run(unb.get()));
  EXPECT_FALSE(changed);
  ASSERT_OK_AND_ASSIGN(
      changed,
      GatherExpander(GatherExpander::kEliminateAllGathers).Run(unb.get()));
  EXPECT_FALSE(changed);

  // Static overflowing batch products (wrapping to -1 and +1 in int64).
  for (int64_t d1 : {int64_t{3}, int64_t{7}}) {
    const int64_t d2 =
        d1 == 3 ? int64_t{6148914691236517205} : int64_t{7905747460161236407};
    ASSERT_OK_AND_ASSIGN(auto ov, ParseAndReturnUnverifiedModule(R"hlo(
HloModule OverflowTripGather
ENTRY main {
  op = f32[2] parameter(0)
  idx = s32[0,2,2] parameter(1)
  ROOT g = f32[2,2,1] gather(op, idx), offset_dims={2},
    collapsed_slice_dims={}, start_index_map={},
    index_vector_dim=0, slice_sizes={1}
})hlo"));
    HloInstruction* idx = ov->entry_computation()->parameter_instruction(1);
    HloInstruction* g = ov->entry_computation()->root_instruction();
    idx->mutable_shape()->set_dimensions(1, d1);
    idx->mutable_shape()->set_dimensions(2, d2);
    g->mutable_shape()->set_dimensions(0, d1);
    g->mutable_shape()->set_dimensions(1, d2);
    ASSERT_OK_AND_ASSIGN(
        changed,
        GatherExpander(GatherExpander::kEliminateSimpleGathers).Run(ov.get()));
    EXPECT_FALSE(changed);
    EXPECT_EQ(GatherExpander(GatherExpander::kEliminateAllGathers)
                  .Run(ov.get())
                  .status()
                  .code(),
              absl::StatusCode::kUnimplemented);
  }

  // Bounded-dynamic s32[<=1,1] and dynamic-batch/collapsed GatherIsBroadcast
  // must be skipped rather than silently erasing dynamic bounds.
  for (absl::string_view dyn_hlo : {absl::string_view(R"hlo(
HloModule DynUnitBatch
ENTRY main {
  idx = s32[<=1,1] set-dimension-size(s32[1,1] parameter(1), s32[] parameter(2)), dimensions={0}
  ROOT g = f32[<=1,3] gather(f32[4,3] parameter(0), idx), offset_dims={1},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,3}
})hlo"),
                                    absl::string_view(R"hlo(
HloModule DynBcastBatch
ENTRY main {
  idx = s32[<=4,1] set-dimension-size(s32[4,1] parameter(1), s32[] parameter(2)), dimensions={0}
  ROOT g = f32[<=4,3] gather(f32[1,3] parameter(0), idx), offset_dims={1},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,3}
})hlo"),
                                    absl::string_view(R"hlo(
HloModule DynCollapsedBcast
ENTRY main {
  op = f32[<=1,3] set-dimension-size(f32[1,3] parameter(0), s32[] parameter(2)), dimensions={0}
  ROOT g = f32[2,3] gather(op, s32[2,1] parameter(1)), offset_dims={1},
    collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,3}
})hlo")}) {
    ASSERT_OK_AND_ASSIGN(auto dm, ParseAndReturnVerifiedModule(dyn_hlo));
    ASSERT_OK_AND_ASSIGN(
        changed,
        GatherExpander(GatherExpander::kEliminateSimpleGathers).Run(dm.get()));
    EXPECT_FALSE(changed);
  }
}

TEST_F(GatherExpanderTest, PreservesSubByteMemorySpaceAndControlDeps) {
  constexpr absl::string_view kSubByteCtrl = R"hlo(
HloModule SubByteMemSpaceCtrl
ENTRY main {
  op = s4[4,2]{1,0:E(4)S(1)} parameter(0)
  idx = s32[2,1]{1,0:S(1)} parameter(1)
  tok = token[] after-all()
  g = s4[2,2]{1,0:E(4)S(1)} gather(op, idx), offset_dims={1},
    collapsed_slice_dims={0}, start_index_map={0},
    index_vector_dim=1, slice_sizes={1,2}, control-predecessors={tok}
  sink = token[] after-all(), control-predecessors={g}
  ROOT out = (s4[2,2]{1,0:E(4)S(1)}, token[]) tuple(g, sink)
})hlo";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kSubByteCtrl));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      GatherExpander(GatherExpander::kEliminateAllGathers).Run(m.get()));
  ASSERT_TRUE(changed);
  EXPECT_FALSE(hlo_query::ContainsInstrWithOpcode(m->entry_computation(),
                                                  {HloOpcode::kGather}));
  const HloInstruction* res =
      m->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* tok = FindInstruction(m.get(), "tok");
  const HloInstruction* sink = FindInstruction(m.get(), "sink");
  EXPECT_EQ(res->shape().layout().element_size_in_bits(), 4);
  EXPECT_EQ(res->shape().layout().memory_space(), 1);
  EXPECT_EQ(sink->control_predecessors().size(), 1);
  EXPECT_EQ(sink->control_predecessors()[0], res);
  std::vector<HloInstruction*> whiles =
      FindInstructions(m.get(), HloOpcode::kWhile);
  ASSERT_EQ(whiles.size(), 1);
  EXPECT_THAT(whiles[0]->control_predecessors(), ::testing::Contains(tok));
  for (const HloInstruction* bi : whiles[0]->while_body()->instructions()) {
    if (bi->shape().IsArray() && bi->shape().element_type() == S4 &&
        !bi->IsConstant() && bi->user_count() > 0) {
      EXPECT_EQ(bi->shape().layout().element_size_in_bits(), 4);
      EXPECT_EQ(bi->shape().layout().memory_space(), 1);
    }
  }
}

}  // namespace
}  // namespace xla
