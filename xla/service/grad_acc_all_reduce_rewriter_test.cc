/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/grad_acc_all_reduce_rewriter.h"

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;
using ::testing::NotNull;

class GradAccAllReduceRewriterTest : public HloTestBase {
 public:
  template <HloOpcode op>
  HloInstruction* find_op(HloComputation* computation) {
    return *std::find_if(computation->instructions().begin(),
                         computation->instructions().end(),
                         HloPredicateIsOp<op>);
  }
};

TEST_F(GradAccAllReduceRewriterTest, SkipBackwardAllReduce) {
  constexpr absl::string_view kHloModule = R"(
    HloModule backward_all_reduce_output

    %add (x: bf16[], y: bf16[]) -> bf16[] {
      %x = bf16[] parameter(0)
      %y = bf16[] parameter(1)
      ROOT %add.1 = bf16[] add(bf16[] %x, bf16[] %y)
    }

    ENTRY backward_all_reduce_output {
      %constant = bf16[] constant(0.1)
      %constant.1 = bf16[] constant(0.2)
      %broadcast =  bf16[16,1024]{1,0} broadcast(bf16[] %constant)
      %broadcast.1 =  bf16[16,1024]{1,0} broadcast(bf16[] %constant.1)
      %dot = bf16[1024,1024]{1,0} dot(bf16[16,1024]{1,0} %broadcast, bf16[16,1024]{1,0} %broadcast.1), lhs_contracting_dims={0}, rhs_contracting_dims={0}, metadata={op_type="xla__einsum_backward" op_name="xla__einsum_backward"}
      %all-reduce = bf16[1024,1024]{1,0} all-reduce(bf16[1024,1024]{1,0} %dot), channel_id=3, replica_groups={{0}}, to_apply=%add, metadata={op_type="xla__einsum_backward" op_name="xla__einsum_backward"}
      %convert = f32[1024,1024]{1,0} convert(bf16[1024,1024]{1,0} %all-reduce)
      %transpose = f32[1024,1024]{0,1} transpose(f32[1024,1024]{1,0} %convert), dimensions={1,0}
      ROOT %tuple.1 = (f32[1024,1024]{1,0}, f32[1024,1024]{1,0}) tuple(f32[1024,1024]{1,0} %convert, f32[1024,1024]{0,1} %transpose)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool skip_allreduce,
                          GradAccAllReduceRewriter().Run(module.get()));
  ASSERT_TRUE(skip_allreduce);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());
  auto entry = module->entry_computation();
  EXPECT_THAT(entry->instructions(), Each(Not(op::AllReduce())));
  auto dot_instr =
      entry->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(dot_instr, op::Dot());
}

TEST_F(GradAccAllReduceRewriterTest, SkipBackwardInSelectAllReduce) {
  constexpr absl::string_view kHloModule = R"(
    HloModule backward_all_reduce_output

    %AddComputation.1579 (x.1580: f32[], y.1581: f32[]) -> f32[] {
      %x.1580 = f32[] parameter(0)
      %y.1581 = f32[] parameter(1)
      ROOT %add.1582 = f32[] add(f32[] %x.1580, f32[] %y.1581)
    }

    ENTRY backward_all_reduce_output {
      %param.1 = s64[4096,1]{1,0} parameter(0)
      %param.2 = f32[4096,1,4096]{2,1,0} parameter(1)
      %constant.826 = u32[1]{0} constant({0})
      %convert.296 = pred[1]{0} convert(u32[1]{0} %constant.826), metadata={op_type="aten__index_put" op_name="aten__index_put"}
      %reshape.2038 = pred[] reshape(pred[1]{0} %convert.296), metadata={op_type="aten__index_put" op_name="aten__index_put"}
      %broadcast.583 = pred[32001,4096]{1,0} broadcast(pred[] %reshape.2038), dimensions={}, metadata={op_type="aten__index_put" op_name="aten__index_put"}
      %constant.482 = f32[] constant(0), metadata={op_type="aten__nll_loss_backward" op_name="aten__nll_loss_backward"}
      %broadcast.582 = f32[32001,4096]{1,0} broadcast(f32[] %constant.482), dimensions={}, metadata={op_type="aten__index_put" op_name="aten__index_put"}
      %param.3 = f32[32001,4096]{1,0} parameter(2), metadata={op_type="xla__device_data" op_name="xla__device_data"}
      %select.25 = f32[32001,4096]{1,0} select(pred[32001,4096]{1,0} %broadcast.583, f32[32001,4096]{1,0} %broadcast.582, f32[32001,4096]{1,0} %param.3), metadata={op_type="aten__index_put" op_name="aten__index_put"}
      %scatter.1 = f32[32001,4096]{1,0} scatter(f32[32001,4096]{1,0} %select.25, s64[4096,1]{1,0} %param.1, f32[4096,1,4096]{2,1,0} %param.2), update_window_dims={1,2}, inserted_window_dims={}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=%AddComputation.1579, metadata={op_type="aten__index_put" op_name="aten__index_put"}
      %all-reduce.29 = f32[32001,4096]{1,0} all-reduce(f32[32001,4096]{1,0} %scatter.1), channel_id=48, replica_groups={{0,4},{1,5},{2,6},{3,7}}, use_global_device_ids=true, to_apply=%AddComputation.1579, metadata={op_type="aten__index_put" op_name="aten__index_put"}
      ROOT %tuple.1 = (f32[32001,4096]{1,0}, f32[32001,4096]{1,0}) tuple(f32[32001,4096]{1,0} %param.3, f32[32001,4096]{1,0} %all-reduce.29)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool skip_allreduce,
                          GradAccAllReduceRewriter().Run(module.get()));
  ASSERT_TRUE(skip_allreduce);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());
  auto entry = module->entry_computation();
  EXPECT_THAT(entry->instructions(), Each(Not(op::AllReduce())));
  auto scatter_instr = entry->root_instruction()->operand(1);
  EXPECT_THAT(scatter_instr, op::Scatter());
}

TEST_F(GradAccAllReduceRewriterTest, SkipBackwardAllReduceAdd) {
  constexpr absl::string_view kHloModule = R"(
    HloModule backward_all_reduce_output

    %add (x: bf16[], y: bf16[]) -> bf16[] {
      %x = bf16[] parameter(0)
      %y = bf16[] parameter(1)
      ROOT %add.1 = bf16[] add(bf16[] %x, bf16[] %y)
    }

    ENTRY backward_all_reduce_output {
      %param.1 = f32[1024,1024]{1,0} parameter(0)
      %constant = bf16[] constant(0.1)
      %constant.1 = bf16[] constant(0.2)
      %broadcast =  bf16[16,1024]{1,0} broadcast(bf16[] %constant)
      %broadcast.1 =  bf16[16,1024]{1,0} broadcast(bf16[] %constant.1)
      %dot = bf16[1024,1024]{1,0} dot(bf16[16,1024]{1,0} %broadcast, bf16[16,1024]{1,0} %broadcast.1), lhs_contracting_dims={0}, rhs_contracting_dims={0}, metadata={op_type="xla__einsum_backward" op_name="xla__einsum_backward"}
      %all-reduce = bf16[1024,1024]{1,0} all-reduce(bf16[1024,1024]{1,0} %dot), channel_id=3, replica_groups={{0}}, to_apply=%add, metadata={op_type="xla__einsum_backward" op_name="xla__einsum_backward"}
      %transpose = bf16[1024,1024]{0,1} transpose(bf16[1024,1024]{1,0} %all-reduce), dimensions={1,0}, metadata={op_type="aten__mul" op_name="aten__mul"}
      %convert = f32[1024,1024]{0,1} convert(bf16[1024,1024]{0,1} %transpose), metadata={op_type="aten__mul" op_name="aten__mul"}
      %add = f32[1024,1024]{1,0} add(f32[1024,1024]{1,0} %param.1, f32[1024,1024]{0,1} %convert), metadata={op_type="aten__add" op_name="aten__add"}
      ROOT %tuple.1 = (f32[1024,1024]{1,0}) tuple(f32[1024,1024]{1,0} %add)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool skip_allreduce,
                          GradAccAllReduceRewriter().Run(module.get()));
  ASSERT_TRUE(skip_allreduce);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());
  auto entry = module->entry_computation();
  EXPECT_THAT(entry->instructions(), Each(Not(op::AllReduce())));
  auto add_instr = entry->root_instruction()->operand(0);
  auto dot_instr = add_instr->operand(1)->operand(0)->operand(0);
  EXPECT_THAT(dot_instr, op::Dot());
}

TEST_F(GradAccAllReduceRewriterTest, MoveBackwardAllReduceAdd) {
  constexpr absl::string_view kHloModule = R"(
    HloModule backward_all_reduce_add

    %add (x: bf16[], y: bf16[]) -> bf16[] {
      %x = bf16[] parameter(0)
      %y = bf16[] parameter(1)
      ROOT %add.1 = bf16[] add(bf16[] %x, bf16[] %y)
    }

    ENTRY backward_all_reduce_add {
      %param.1 = f32[1024,1024]{1,0} parameter(0)
      %param.2 = f32[1024,1024]{1,0} parameter(1)
      %param.3 = f32[] parameter(2)
      %param.4 = f32[1024,1024]{1,0} parameter(3)
      %broadcast.9 = f32[1024,1024]{1,0} broadcast(f32[] %param.3), dimensions={}, metadata={op_type="aten__mul" op_name="aten__lerp.1/aten__mul"}
      %constant = bf16[] constant(0.1)
      %constant.1 = bf16[] constant(0.2)
      %broadcast =  bf16[16,1024]{1,0} broadcast(bf16[] %constant)
      %broadcast.1 =  bf16[16,1024]{1,0} broadcast(bf16[] %constant.1)
      %dot = bf16[1024,1024]{1,0} dot(bf16[16,1024]{1,0} %broadcast, bf16[16,1024]{1,0} %broadcast.1), lhs_contracting_dims={0}, rhs_contracting_dims={0}, metadata={op_type="xla__einsum_backward" op_name="xla__einsum_backward"}
      %all-reduce = bf16[1024,1024]{1,0} all-reduce(bf16[1024,1024]{1,0} %dot), channel_id=3, replica_groups={{0}}, to_apply=%add, metadata={op_type="xla__einsum_backward" op_name="xla__einsum_backward"}
      %transpose = bf16[1024,1024]{0,1} transpose(bf16[1024,1024]{1,0} %all-reduce), dimensions={1,0}, metadata={op_type="aten__mul" op_name="aten__mul"}
      %convert = f32[1024,1024]{0,1} convert(bf16[1024,1024]{0,1} %transpose), metadata={op_type="aten__mul" op_name="aten__mul"}
      %add.10 = f32[1024,1024]{1,0} add(f32[1024,1024]{1,0} %param.1, f32[1024,1024]{0,1} %convert), metadata={op_type="aten__add" op_name="aten__add"}
      %subtract.1 = f32[1024,1024]{1,0} subtract(f32[1024,1024]{1,0} %add.10, f32[1024,1024]{1,0} %param.2), metadata={op_type="aten__sub" op_name="aten__lerp.2/aten__sub"}
      %multiply.10 = f32[1024,1024]{1,0} multiply(f32[1024,1024]{1,0} %broadcast.9, f32[1024,1024]{1,0} %subtract.1), metadata={op_type="aten__mul" op_name="aten__lerp.2/aten__mul"}
      %add.11 = f32[1024,1024]{1,0} add(f32[1024,1024]{1,0} %param.4, f32[1024,1024]{1,0} %multiply.10), metadata={op_type="aten__add" op_name="aten__lerp.2/aten__add"}
      ROOT %tuple.1 = (f32[1024,1024]{1,0}) tuple(f32[1024,1024]{1,0} %add.11)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool skip_allreduce,
                          GradAccAllReduceRewriter().Run(module.get()));
  ASSERT_TRUE(skip_allreduce);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());

  auto entry = module->entry_computation();
  auto moved_all_reduce =
      DynCast<HloAllReduceInstruction>(find_op<HloOpcode::kAllReduce>(entry));
  ASSERT_THAT(moved_all_reduce, NotNull());
  EXPECT_THAT(moved_all_reduce, op::ReplicaGroups({{0}}));
  EXPECT_THAT(moved_all_reduce, op::Shape("f32[1024, 1024]"));

  EXPECT_THAT(moved_all_reduce->operand(0), op::Add());
  EXPECT_THAT(moved_all_reduce->users()[0], op::Subtract());
}

TEST_F(GradAccAllReduceRewriterTest, SkipBackwardAllReduceDynamicSlice) {
  constexpr absl::string_view kHloModule = R"(
    HloModule backward_all_reduce_output

    %add (x: bf16[], y: bf16[]) -> bf16[] {
      %x = bf16[] parameter(0)
      %y = bf16[] parameter(1)
      ROOT %add.1 = bf16[] add(bf16[] %x, bf16[] %y)
    }

    ENTRY backward_all_reduce_output {
      %constant.398 = s32[] constant(0)
      %constant = bf16[] constant(0.1)
      %constant.1 = bf16[] constant(0.2)
      %constant.1580 = s32[] constant(2)
      %broadcast =  bf16[1024,1024]{1,0} broadcast(bf16[] %constant)
      %broadcast.1 =  bf16[1024,4096]{1,0} broadcast(bf16[] %constant.1)
      %dot = bf16[1024,4096]{1,0} dot(bf16[1024,1024]{1,0} %broadcast, bf16[1024,4096]{1,0} %broadcast.1), lhs_contracting_dims={0}, rhs_contracting_dims={0}, metadata={op_type="xla__einsum_backward" op_name="xla__einsum_backward"}
      %all-reduce.35 = bf16[1024,4096]{1,0} all-reduce(bf16[1024,4096]{1,0} %dot), channel_id=3, replica_groups={{0}}, to_apply=%add, metadata={op_type="xla__einsum_backward" op_name="xla__einsum_backward"}
      %dynamic-slice.1 = bf16[1024,2048]{1,0} dynamic-slice(bf16[1024,4096]{1,0} %all-reduce.35, s32[] %constant.398, s32[] %constant.1580), dynamic_slice_sizes={1024,2048}, metadata={op_type="xla__einsum_backward" op_name="xla__einsum_backward"}
      %transpose.1 = bf16[2048,1024]{0,1} transpose(bf16[1024,2048]{1,0} %dynamic-slice.1), dimensions={1,0}, metadata={op_type="aten__permute" op_name="aten__permute"}
      %convert.1 = f32[2048,1024]{0,1} convert(bf16[2048,1024]{0,1} %transpose.1), metadata={op_type="aten__permute" op_name="aten__permute"}
      ROOT %tuple.1 = (f32[2048,1024]{1,0}) tuple(f32[2048,1024]{1,0} %convert.1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool skip_allreduce,
                          GradAccAllReduceRewriter().Run(module.get()));
  ASSERT_TRUE(skip_allreduce);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());
  auto entry = module->entry_computation();
  EXPECT_THAT(entry->instructions(), Each(Not(op::AllReduce())));
  auto dyna_slice_instr =
      entry->root_instruction()->operand(0)->operand(0)->operand(0);
  EXPECT_THAT(dyna_slice_instr, op::DynamicSlice());
  auto dot_instr = dyna_slice_instr->operand(0);
  EXPECT_THAT(dot_instr, op::Dot());
}

TEST_F(GradAccAllReduceRewriterTest, SkipBackwardAllReduceDynamicSliceAdd) {
  constexpr absl::string_view kHloModule = R"(
    HloModule backward_all_reduce_output

    %add (x: bf16[], y: bf16[]) -> bf16[] {
      %x = bf16[] parameter(0)
      %y = bf16[] parameter(1)
      ROOT %add.1 = bf16[] add(bf16[] %x, bf16[] %y)
    }

    ENTRY backward_all_reduce_output {
      %param.1 = f32[1024,2048]{1,0} parameter(0)
      %constant.398 = s32[] constant(0)
      %constant = bf16[] constant(0.1)
      %constant.1 = bf16[] constant(0.2)
      %constant.1580 = s32[] constant(2)
      %broadcast =  bf16[1024,1024]{1,0} broadcast(bf16[] %constant)
      %broadcast.1 =  bf16[1024,4096]{1,0} broadcast(bf16[] %constant.1)
      %dot = bf16[1024,4096]{1,0} dot(bf16[1024,1024]{1,0} %broadcast, bf16[1024,4096]{1,0} %broadcast.1), lhs_contracting_dims={0}, rhs_contracting_dims={0}, metadata={op_type="xla__einsum_backward" op_name="xla__einsum_backward"}
      %all-reduce.35 = bf16[1024,4096]{1,0} all-reduce(bf16[1024,4096]{1,0} %dot), channel_id=3, replica_groups={{0}}, to_apply=%add, metadata={op_type="xla__einsum_backward" op_name="xla__einsum_backward"}
      %dynamic-slice.1 = bf16[1024,2048]{1,0} dynamic-slice(bf16[1024,4096]{1,0} %all-reduce.35, s32[] %constant.398, s32[] %constant.1580), dynamic_slice_sizes={1024,2048}, metadata={op_type="xla__einsum_backward" op_name="xla__einsum_backward"}
      %convert.1 = f32[1024,2048]{0,1} convert(bf16[1024,2048]{0,1} %dynamic-slice.1), metadata={op_type="xla__cast" op_name="xla__cast"}
      %add.153 = f32[1024,2048]{1,0} add(f32[1024,2048]{1,0} %param.1, f32[1024,2048]{1,0} %convert.1), metadata={op_type="aten__add" op_name="aten__add"}
      %transpose.557 = f32[2048,1024]{0,1} transpose(f32[1024,2048]{1,0} %add.153), dimensions={1,0}, metadata={op_type="aten__add" op_name="aten__add"}
      ROOT %tuple.1 = (f32[2048,1024]{1,0}) tuple(f32[2048,1024]{1,0} %transpose.557)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool skip_allreduce,
                          GradAccAllReduceRewriter().Run(module.get()));
  ASSERT_TRUE(skip_allreduce);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());
  auto entry = module->entry_computation();
  EXPECT_THAT(entry->instructions(), Each(Not(op::AllReduce())));
  auto dyna_slice_instr =
      entry->root_instruction()->operand(0)->operand(0)->operand(1)->operand(0);
  EXPECT_THAT(dyna_slice_instr, op::DynamicSlice());
  auto dot_instr = dyna_slice_instr->operand(0);
  EXPECT_THAT(dot_instr, op::Dot());
}

TEST_F(GradAccAllReduceRewriterTest, MoveBackwardAllReduceDynamicSliceAdd) {
  constexpr absl::string_view kHloModule = R"(
    HloModule backward_all_reduce_add

    %add (x: bf16[], y: bf16[]) -> bf16[] {
      %x = bf16[] parameter(0)
      %y = bf16[] parameter(1)
      ROOT %add.1 = bf16[] add(bf16[] %x, bf16[] %y)
    }

    ENTRY backward_all_reduce_add {
      %param.1 = f32[2048,1024]{1,0} parameter(0)
      %param.2 = f32[2048,1024]{1,0} parameter(1)
      %param.3 = f32[] parameter(2)
      %param.4 = f32[2048,1024]{1,0} parameter(3)
      %broadcast.9 = f32[2048,1024]{1,0} broadcast(f32[] %param.3), dimensions={}
      %constant = bf16[] constant(0.1)
      %constant.1 = bf16[] constant(0.2)
      %broadcast =  bf16[1024,1024]{1,0} broadcast(bf16[] %constant)
      %broadcast.1 =  bf16[1024,4096]{1,0} broadcast(bf16[] %constant.1)
      %constant.398 = s32[] constant(0)
      %constant.1580 = s32[] constant(2)
      %dot = bf16[1024,4096]{1,0} dot(bf16[1024,1024]{1,0} %broadcast, bf16[1024,4096]{1,0} %broadcast.1), lhs_contracting_dims={0}, rhs_contracting_dims={0}, metadata={op_type="xla__einsum_backward" op_name="xla__einsum_backward"}
      %all-reduce.35 = bf16[1024,4096]{1,0} all-reduce(bf16[1024,4096]{1,0} %dot), channel_id=3, replica_groups={{0}}, to_apply=%add, metadata={op_type="xla__einsum_backward" op_name="xla__einsum_backward"}
      %dynamic-slice.1 = bf16[1024,2048]{1,0} dynamic-slice(bf16[1024,4096]{1,0} %all-reduce.35, s32[] %constant.398, s32[] %constant.1580), dynamic_slice_sizes={1024,2048}, metadata={op_type="xla__einsum_backward" op_name="xla__einsum_backward"}
      %transpose = bf16[2048,1024]{0,1} transpose(bf16[1024,2048]{1,0} %dynamic-slice.1), dimensions={1,0}, metadata={op_type="aten__mul" op_name="aten__mul"}
      %convert = f32[2048,1024]{0,1} convert(bf16[2048,1024]{0,1} %transpose), metadata={op_type="aten__mul" op_name="aten__mul"}
      %add.10 = f32[2048,1024]{1,0} add(f32[2048,1024]{1,0} %param.1, f32[2048,1024]{0,1} %convert), metadata={op_type="aten__add" op_name="aten__add"}
      %subtract.1 = f32[2048,1024]{1,0} subtract(f32[2048,1024]{1,0} %add.10, f32[2048,1024]{1,0} %param.2), metadata={op_type="aten__sub" op_name="aten__lerp.2/aten__sub"}
      %multiply.10 = f32[2048,1024]{1,0} multiply(f32[2048,1024]{1,0} %broadcast.9, f32[2048,1024]{1,0} %subtract.1), metadata={op_type="aten__mul" op_name="aten__lerp.2/aten__mul"}
      %add.11 = f32[2048,1024]{1,0} add(f32[2048,1024]{1,0} %param.4, f32[2048,1024]{1,0} %multiply.10), metadata={op_type="aten__add" op_name="aten__lerp.2/aten__add"}
      ROOT %tuple.1 = (f32[2048,1024]{1,0}) tuple(f32[2048,1024]{1,0} %add.11)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(bool skip_allreduce,
                          GradAccAllReduceRewriter().Run(module.get()));
  ASSERT_TRUE(skip_allreduce);
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/true)
          .Run(module.get())
          .status());

  auto entry = module->entry_computation();
  auto moved_all_reduce =
      DynCast<HloAllReduceInstruction>(find_op<HloOpcode::kAllReduce>(entry));
  ASSERT_THAT(moved_all_reduce, NotNull());
  EXPECT_THAT(moved_all_reduce, op::ReplicaGroups({{0}}));
  EXPECT_THAT(moved_all_reduce, op::Shape("f32[2048, 1024]"));

  EXPECT_THAT(moved_all_reduce->operand(0), op::Add());
  EXPECT_THAT(moved_all_reduce->users()[0], op::Subtract());
}

}  // namespace
}  // namespace xla
