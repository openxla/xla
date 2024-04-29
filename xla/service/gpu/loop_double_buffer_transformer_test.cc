/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/loop_double_buffer_transformer.h"

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using tsl::testing::IsOkAndHolds;

int64_t CountInstructions(const HloComputation& computation, HloOpcode opcode) {
  int64_t count = 0;
  for (const auto& instruction : computation.instructions()) {
    if (instruction->opcode() == opcode) {
      count++;
    }
  }
  return count;
}

int64_t CountInstructions(const HloModule& module, HloOpcode opcode) {
  int64_t count = 0;
  for (const auto& computation : module.computations()) {
    count += CountInstructions((*computation), opcode);
  }
  return count;
}

class GpuLoopDoubleBufferTransformerTest : public HloTestBase {
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_while_loop_double_buffering(true);
    return debug_options;
  }
};

TEST_F(GpuLoopDoubleBufferTransformerTest, UnrolledLoopEvenTripCount) {
  const char* const kModuleString = R"(
HloModule all_gather_overlapping
condition {
  input_tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=3
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
 input_tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) parameter(0)
 param_0 = f32[1,128] get-tuple-element(input_tuple), index=0
 param_1 = f32[2,128] get-tuple-element(input_tuple), index=2
 cond = s32[] get-tuple-element(input_tuple), index=3
 c0 = f32[] constant(0)
 splat_c0 = f32[1,128] broadcast(c0), dimensions={}
 add = f32[1,128] add(splat_c0, param_0)
 // Start all-gather communication
 all-gather-start = (f32[1,128], f32[2,128]) all-gather-start(add), channel_id=1337, replica_groups={{0,1}}, dimensions={0}, use_global_device_ids=true
 // Intertwined with the all-gather communication, an operation happens which
 // depends on param_1, but crucially has a different output shape (which
 // excludes reusing param_1's buffer for its output).
 c1_s32 = s32[] constant(1)
 c0_s32 = s32[] constant(0)
 one = s32[] constant(1)
 cond_plus_1 = s32[] add(cond, one)
 dynamic-slice = f32[1,128] dynamic-slice(param_1, c1_s32, c0_s32), dynamic_slice_sizes={1,128}
 // The all-gather communication finishes
 all-gather-done = f32[2,128] all-gather-done(all-gather-start)
 ROOT output_tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) tuple(param_0, dynamic-slice, all-gather-done, cond_plus_1)
}

ENTRY main {
 param_0 = f32[1,128] parameter(0)
 param_1 = f32[2,128] parameter(1)
 param_2 = s32[] constant(0)
 tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) tuple(param_0, param_0, param_1, param_2)
 ROOT while = (f32[1,128], f32[1,128], f32[2,128], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  LoopDoubleBufferTransformer double_buffer;
  TupleSimplifier tuple_simp;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, double_buffer.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(changed, tuple_simp.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* while_instruction;
  for (auto instr : module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kWhile) {
      while_instruction = instr;
    }
  }
  TF_ASSERT_OK_AND_ASSIGN(
      WhileLoopBackendConfig config,
      while_instruction->backend_config<WhileLoopBackendConfig>());
  int64_t exact_trip_count = config.known_trip_count().n();
  // We expect that after unrolling, the total trip count is half of original
  // count.
  EXPECT_EQ(exact_trip_count, 5);
  // We expect that after unrolling, there should be 2 allgather starts,
  // both in while body.
  EXPECT_EQ(CountInstructions((*while_instruction->while_body()),
                              HloOpcode::kAllGatherStart),
            2);
  EXPECT_EQ(CountInstructions((*module), HloOpcode::kAllGatherStart), 2);
}

TEST_F(GpuLoopDoubleBufferTransformerTest, UnrolledLoopOddTripCount) {
  const char* const kModuleString = R"(
HloModule all_gather_overlapping
condition {
  input_tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=3
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
 input_tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) parameter(0)
 param_0 = f32[1,128] get-tuple-element(input_tuple), index=0
 param_1 = f32[2,128] get-tuple-element(input_tuple), index=2
 cond = s32[] get-tuple-element(input_tuple), index=3
 c0 = f32[] constant(0)
 splat_c0 = f32[1,128] broadcast(c0), dimensions={}
 add = f32[1,128] add(splat_c0, param_0)
 // Start all-gather communication
 all-gather-start = (f32[1,128], f32[2,128]) all-gather-start(add), channel_id=1337, replica_groups={{0,1}}, dimensions={0}, use_global_device_ids=true
 // Intertwined with the all-gather communication, an operation happens which
 // depends on param_1, but crucially has a different output shape (which
 // excludes reusing param_1's buffer for its output).
 c1_s32 = s32[] constant(1)
 c0_s32 = s32[] constant(0)
 one = s32[] constant(1)
 cond_plus_1 = s32[] add(cond, one)
 dynamic-slice = f32[1,128] dynamic-slice(param_1, c1_s32, c0_s32), dynamic_slice_sizes={1,128}
 // The all-gather communication finishes
 all-gather-done = f32[2,128] all-gather-done(all-gather-start)
 ROOT output_tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) tuple(param_0, dynamic-slice, all-gather-done, cond_plus_1)
}

ENTRY main {
 param_0 = f32[1,128] parameter(0)
 param_1 = f32[2,128] parameter(1)
 param_2 = s32[] constant(0)
 tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) tuple(param_0, param_0, param_1, param_2)
 ROOT while = (f32[1,128], f32[1,128], f32[2,128], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"11"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  LoopDoubleBufferTransformer double_buffer;
  TupleSimplifier tuple_simp;
  EXPECT_THAT(double_buffer.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(tuple_simp.Run(module.get()), IsOkAndHolds(true));

  // We expect that for the while loop, no further copy needs to be added to the
  // module.
  HloInstruction* while_instruction;
  for (auto instr : module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kWhile) {
      while_instruction = instr;
    }
  }
  TF_ASSERT_OK_AND_ASSIGN(
      WhileLoopBackendConfig config,
      while_instruction->backend_config<WhileLoopBackendConfig>());
  int64_t exact_trip_count = config.known_trip_count().n();
  // We expect that after unrolling, the total trip count is half of original
  // count.
  EXPECT_EQ(exact_trip_count, 5);

  // We expect that after unrolling, there should be 3 allgather starts,
  // 1 in parent computation, 2 in while body.
  EXPECT_EQ(CountInstructions((*while_instruction->while_body()),
                              HloOpcode::kAllGatherStart),
            2);
  EXPECT_EQ(CountInstructions((*module), HloOpcode::kAllGatherStart), 3);

  // We expect that after unrolling, the third operand of the input tuple should
  // be the peeled allgather done.
  EXPECT_EQ(while_instruction->operand(0)->operand(2)->opcode(),
            HloOpcode::kAllGatherDone);
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       UnrolledLoopNoControlDepsForConstantAdd) {
  const char* const kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
 input_tuple = (f32[], s32[]) parameter(0)
 param_0 = f32[] get-tuple-element(input_tuple), index=0
 cond = s32[] get-tuple-element(input_tuple), index=1
 c2 = f32[] constant(2)
 add = f32[] add(c2, param_0)
 one = s32[] constant(1)
 cond_plus_1 = s32[] add(cond, one)
 ROOT output_tuple = (f32[], s32[]) tuple(add, cond_plus_1)
}

ENTRY main {
 param_0 = f32[] parameter(0)
 param_2 = s32[] constant(0)
 tuple = (f32[], s32[]) tuple(param_0, param_2)
 ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"11"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  LoopDoubleBufferTransformer double_buffer;
  TupleSimplifier tuple_simp;
  EXPECT_THAT(double_buffer.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(tuple_simp.Run(module.get()), IsOkAndHolds(true));

  HloInstruction* while_instruction;
  for (auto instr : module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kWhile) {
      while_instruction = instr;
    }
  }
  TF_ASSERT_OK_AND_ASSIGN(
      WhileLoopBackendConfig config,
      while_instruction->backend_config<WhileLoopBackendConfig>());
  int64_t exact_trip_count = config.known_trip_count().n();
  // We expect that after unrolling, the total trip count is half of original
  // count.
  EXPECT_EQ(exact_trip_count, 5);

  // We expect that after unrolling, there should be 4 adds
  EXPECT_EQ(
      CountInstructions((*while_instruction->while_body()), HloOpcode::kAdd),
      4);

  // We expect that after unrolling, the first operand of the output tuple
  // should not have any control dependency since it's a elementwise add with a
  // constant operand.
  EXPECT_EQ(while_instruction->while_body()
                ->root_instruction()
                ->operand(0)
                ->control_predecessors()
                .size(),
            0);
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       UnrolledLoopNoControlDepsForCollective) {
  const char* const kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

ar_add {
  Arg_1 = f32[] parameter(1)
  Arg_0 = f32[] parameter(0)
  ROOT add_ar = f32[] add(Arg_1, Arg_0)
}

body {
 input_tuple = (f32[], s32[]) parameter(0)
 param_0 = f32[] get-tuple-element(input_tuple), index=0
 cond = s32[] get-tuple-element(input_tuple), index=1
 all-reduce-start = f32[] all-reduce-start(param_0), channel_id=8, replica_groups={{0}}, to_apply=ar_add, backend_config="{\"is_sync\":false}"
 one = s32[] constant(1)
 all-reduce-done = f32[] all-reduce-done(all-reduce-start)
 cond_plus_1 = s32[] add(cond, one)
 ROOT output_tuple = (f32[], s32[]) tuple(all-reduce-done, cond_plus_1)
}

ENTRY main {
 param_0 = f32[] parameter(0)
 param_2 = s32[] constant(0)
 tuple = (f32[], s32[]) tuple(param_0, param_2)
 ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  LoopDoubleBufferTransformer double_buffer;
  TupleSimplifier tuple_simp;
  EXPECT_THAT(double_buffer.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(tuple_simp.Run(module.get()), IsOkAndHolds(true));

  HloInstruction* while_instruction;
  for (auto instr : module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kWhile) {
      while_instruction = instr;
    }
  }
  TF_ASSERT_OK_AND_ASSIGN(
      WhileLoopBackendConfig config,
      while_instruction->backend_config<WhileLoopBackendConfig>());
  int64_t exact_trip_count = config.known_trip_count().n();
  // We expect that after unrolling, the total trip count is half of original
  // count.
  EXPECT_EQ(exact_trip_count, 5);

  // We expect that after unrolling, there should be 2 all-reduce-starts
  EXPECT_EQ(CountInstructions((*while_instruction->while_body()),
                              HloOpcode::kAllReduceStart),
            2);
  absl::flat_hash_set<int64_t> channel_ids;
  for (HloInstruction* ar : while_instruction->while_body()->instructions()) {
    if (ar->opcode() == HloOpcode::kAllReduceStart) {
      // We expect that after unrolling, all-reduces should not have any control
      // deps.
      EXPECT_EQ(ar->control_predecessors().size(), 0);
      channel_ids.insert(*(ar->channel_id()));
    }
  }
  // we expect that all 2 all-reduces will have different channel ids.
  EXPECT_EQ(channel_ids.size(), 2);
}

// The following 2 tests also address the regression described here:
// https://github.com/openxla/xla/issues/6353
TEST_F(GpuLoopDoubleBufferTransformerTest, NestedWhileLoopRemainsFlattened) {
  const char* const kModuleString = R"(
HloModule loop_unrolling_nested_while_loop_remains_flattened

condition_nested {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body_nested {
 input_tuple = (s32[]) parameter(0)
 cond = s32[] get-tuple-element(input_tuple), index=0
 one = s32[] constant(1)
 cond_plus_1 = s32[] add(cond, one)
 ROOT output = (s32[]) tuple(cond_plus_1)
}

condition {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
  input_tuple = (s32[]) parameter(0)
  ROOT output = (s32[]) while(input_tuple), condition=condition_nested, body=body_nested
}

ENTRY main {
 param_0 = (s32[]) parameter(0)
 ROOT while = (s32[]) while(param_0), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  LoopDoubleBufferTransformer double_buffer;
  EXPECT_THAT(double_buffer.Run(module.get()), IsOkAndHolds(true));

  absl::flat_hash_set<const HloComputation*> while_loops_callees;

  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kWhile) {
        EXPECT_TRUE(
            while_loops_callees.insert(instr->while_condition()).second);
        EXPECT_TRUE(while_loops_callees.insert(instr->while_body()).second);
      }
    }
  }

  // We expect that the nested while loop has been duplicated, along with its
  // associated computations.
  EXPECT_EQ(while_loops_callees.size(), 6);
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       NestedWhileLoopRemainsFlattenedOddTripCount) {
  const char* const kModuleString = R"(
HloModule loop_unrolling_nested_while_loop_remains_flattened

condition_nested {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body_nested {
 input_tuple = (s32[]) parameter(0)
 cond = s32[] get-tuple-element(input_tuple), index=0
 one = s32[] constant(1)
 cond_plus_1 = s32[] add(cond, one)
 ROOT output = (s32[]) tuple(cond_plus_1)
}

condition {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
  input_tuple = (s32[]) parameter(0)
  ROOT output = (s32[]) while(input_tuple), condition=condition_nested, body=body_nested
}

ENTRY main {
 param_0 = (s32[]) parameter(0)
 ROOT while = (s32[]) while(param_0), condition=condition, body=body, backend_config={"known_trip_count":{"n":"11"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  LoopDoubleBufferTransformer double_buffer;
  EXPECT_THAT(double_buffer.Run(module.get()), IsOkAndHolds(true));

  absl::flat_hash_set<const HloComputation*> while_loops_callees;

  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kWhile) {
        EXPECT_TRUE(
            while_loops_callees.insert(instr->while_condition()).second);
        EXPECT_TRUE(while_loops_callees.insert(instr->while_body()).second);
      }
    }
  }

  // We expect that the nested while loop has been duplicated, along with its
  // associated computations.
  EXPECT_EQ(while_loops_callees.size(), 8);
}

TEST_F(GpuLoopDoubleBufferTransformerTest, NestedWhileLoopAreUnrolled) {
  const char* const kModuleString = R"(
HloModule loop_unrolling_nested_are_unrolled
condition_nested {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}
body_nested {
 input_tuple = (s32[]) parameter(0)
 cond = s32[] get-tuple-element(input_tuple), index=0
 one = s32[] constant(1)
 cond_plus_1 = s32[] add(cond, one)
 ROOT output = (s32[]) tuple(cond_plus_1)
}
condition {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}
body {
  input_tuple = (s32[]) parameter(0)
  ROOT output = (s32[]) while(input_tuple), condition=condition_nested, body=body_nested, backend_config={"known_trip_count":{"n":"11"}}
}
ENTRY main {
 param_0 = (s32[]) parameter(0)
 ROOT while = (s32[]) while(param_0), condition=condition, body=body, backend_config={"known_trip_count":{"n":"11"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  LoopDoubleBufferTransformer double_buffer;
  EXPECT_THAT(double_buffer.Run(module.get()), IsOkAndHolds(true));

  int64_t num_whiles = 0;
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kWhile) {
        // All loops in the module should be unrolled now and have trip count
        // of 5.
        EXPECT_EQ(instr->backend_config<WhileLoopBackendConfig>()
                      ->known_trip_count()
                      .n(),
                  5);
        ++num_whiles;
      }
    }
  }
  // We expect the number of while loops to be 4 in total after unrolling.
  EXPECT_EQ(num_whiles, 4);
}

TEST_F(GpuLoopDoubleBufferTransformerTest, WhileLoopWithCollectivePermute) {
  const char* kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

ar_add {
  Arg_1 = f32[] parameter(1)
  Arg_0 = f32[] parameter(0)
  ROOT add_ar = f32[] add(Arg_1, Arg_0)
}

body {
  input_tuple = (f32[], s32[]) parameter(0)
  param_0 = f32[] get-tuple-element(input_tuple), index=0
  cond = s32[] get-tuple-element(input_tuple), index=1
  collective-permute = f32[] collective-permute(param_0), channel_id=1, source_target_pairs={{0,1},{1,2},{2,3},{3,0}},
                             frontend_attributes={_xla_send_recv_validation="{{0,7},{1,8},{2,9},{3,10}}"}
  one = s32[] constant(1)
  cond_plus_1 = s32[] add(cond, one)
  ROOT output_tuple = (f32[], s32[]) tuple(collective-permute, cond_plus_1)
}

ENTRY main {
  param_0 = f32[] parameter(0)
  param_2 = s32[] constant(0)
  tuple = (f32[], s32[]) tuple(param_0, param_2)
  ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"}}
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  LoopDoubleBufferTransformer double_buffer;
  EXPECT_THAT(double_buffer.Run(module.get()), IsOkAndHolds(true));

  // Get the collective permute instructions
  HloComputation* while_body =
      module->entry_computation()->root_instruction()->while_body();
  std::vector<HloInstruction*> collective_permutes;
  for (HloInstruction* inst : while_body->instructions()) {
    if (inst->opcode() != HloOpcode::kCollectivePermute) continue;
    collective_permutes.push_back(inst);
  }
  ASSERT_EQ(collective_permutes.size(), 2);
  ASSERT_TRUE(collective_permutes[0]->frontend_attributes().map().contains(
      kSendRecvValidationAttr));
  ASSERT_TRUE(collective_permutes[1]->frontend_attributes().map().contains(
      kSendRecvValidationAttr));
  EXPECT_EQ(collective_permutes[0]->frontend_attributes().map().at(
                kSendRecvValidationAttr),
            "{{0,4},{1,4},{1,5},{2,5}}");
  EXPECT_EQ(collective_permutes[1]->frontend_attributes().map().at(
                kSendRecvValidationAttr),
            "{{0,3},{0,4},{1,4},{1,5}}");
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       WhileLoopWithCollectivePermutePeeled) {
  const char* kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(15)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

ar_add {
  Arg_1 = f32[] parameter(1)
  Arg_0 = f32[] parameter(0)
  ROOT add_ar = f32[] add(Arg_1, Arg_0)
}

body {
  input_tuple = (f32[], s32[]) parameter(0)
  param_0 = f32[] get-tuple-element(input_tuple), index=0
  cond = s32[] get-tuple-element(input_tuple), index=1
  collective-permute = f32[] collective-permute(param_0), channel_id=1, source_target_pairs={{0,1},{1,2},{2,3},{3,4},{4,5},{5,6},{6,7},{7,0}},
                             frontend_attributes={_xla_send_recv_validation="{{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15}}"}
  one = s32[] constant(1)
  cond_plus_1 = s32[] add(cond, one)
  ROOT output_tuple = (f32[], s32[]) tuple(collective-permute, cond_plus_1)
}

ENTRY main {
  param_0 = f32[] parameter(0)
  param_2 = s32[] constant(0)
  tuple = (f32[], s32[]) tuple(param_0, param_2)
  ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"15"}}
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  LoopDoubleBufferTransformer double_buffer;
  EXPECT_THAT(double_buffer.Run(module.get()), IsOkAndHolds(true));

  // Get the collective permute instructions outside while body
  for (HloInstruction* instr : module->entry_computation()->instructions()) {
    if (instr->opcode() != HloOpcode::kCollectivePermute) continue;
    ASSERT_TRUE(
        instr->frontend_attributes().map().contains(kSendRecvValidationAttr));
    EXPECT_EQ(instr->frontend_attributes().map().at(kSendRecvValidationAttr),
              "{{0,1},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0}}");
  }

  // Get the collective permute instructions in while body
  HloComputation* while_body =
      module->entry_computation()->root_instruction()->while_body();
  std::vector<HloInstruction*> collective_permutes;
  for (HloInstruction* inst : while_body->instructions()) {
    if (inst->opcode() != HloOpcode::kCollectivePermute) continue;
    collective_permutes.push_back(inst);
  }
  ASSERT_EQ(collective_permutes.size(), 2);
  ASSERT_TRUE(collective_permutes[0]->frontend_attributes().map().contains(
      kSendRecvValidationAttr));
  ASSERT_TRUE(collective_permutes[1]->frontend_attributes().map().contains(
      kSendRecvValidationAttr));
  EXPECT_EQ(collective_permutes[0]->frontend_attributes().map().at(
                kSendRecvValidationAttr),
            "{{0,4},{0,4},{1,5},{1,5},{2,6},{2,6},{3,7},{3,7}}");
  EXPECT_EQ(collective_permutes[1]->frontend_attributes().map().at(
                kSendRecvValidationAttr),
            "{{0,3},{0,4},{0,4},{1,5},{1,5},{2,6},{2,6},{3,7}}");
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       WhileLoopWithCollectivePermuteBackwardCycle) {
  const char* kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(14)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

ar_add {
  Arg_1 = f32[] parameter(1)
  Arg_0 = f32[] parameter(0)
  ROOT add_ar = f32[] add(Arg_1, Arg_0)
}

body {
  input_tuple = (f32[], s32[]) parameter(0)
  param_0 = f32[] get-tuple-element(input_tuple), index=0
  cond = s32[] get-tuple-element(input_tuple), index=1
  collective-permute = f32[] collective-permute(param_0), channel_id=1, source_target_pairs={{0,7},{1,0},{2,1},{3,2},{4,3},{5,4},{6,5},{7,6}},
                             frontend_attributes={_xla_send_recv_validation="{{7,14},{6,13},{5,12},{4,11},{3,10},{2,9},{1,8},{0,7}}"}
  one = s32[] constant(1)
  cond_plus_1 = s32[] add(cond, one)
  ROOT output_tuple = (f32[], s32[]) tuple(collective-permute, cond_plus_1)
}

ENTRY main {
  param_0 = f32[] parameter(0)
  param_2 = s32[] constant(0)
  tuple = (f32[], s32[]) tuple(param_0, param_2)
  ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"14"}}
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  LoopDoubleBufferTransformer double_buffer;
  EXPECT_THAT(double_buffer.Run(module.get()), IsOkAndHolds(true));
  // Get the collective permute instructions
  HloComputation* while_body =
      module->entry_computation()->root_instruction()->while_body();
  std::vector<HloInstruction*> collective_permutes;
  for (HloInstruction* inst : while_body->instructions()) {
    if (inst->opcode() != HloOpcode::kCollectivePermute) continue;
    collective_permutes.push_back(inst);
  }
  ASSERT_EQ(collective_permutes.size(), 2);
  ASSERT_TRUE(collective_permutes[0]->frontend_attributes().map().contains(
      kSendRecvValidationAttr));
  ASSERT_TRUE(collective_permutes[1]->frontend_attributes().map().contains(
      kSendRecvValidationAttr));
  EXPECT_EQ(collective_permutes[0]->frontend_attributes().map().at(
                kSendRecvValidationAttr),
            "{{4,7},{3,7},{3,6},{2,6},{2,5},{1,5},{1,4},{0,4}}");
  EXPECT_EQ(collective_permutes[1]->frontend_attributes().map().at(
                kSendRecvValidationAttr),
            "{{3,7},{3,6},{2,6},{2,5},{1,5},{1,4},{0,4},{0,3}}");
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       WhileLoopWithCollectivePermuteBackwardCyclePeeled) {
  const char* kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(15)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

ar_add {
  Arg_1 = f32[] parameter(1)
  Arg_0 = f32[] parameter(0)
  ROOT add_ar = f32[] add(Arg_1, Arg_0)
}

body {
  input_tuple = (f32[], s32[]) parameter(0)
  param_0 = f32[] get-tuple-element(input_tuple), index=0
  cond = s32[] get-tuple-element(input_tuple), index=1
  collective-permute = f32[] collective-permute(param_0), channel_id=1, source_target_pairs={{0,7},{1,0},{2,1},{3,2},{4,3},{5,4},{6,5},{7,6}},
                             frontend_attributes={_xla_send_recv_validation="{{7,15},{6,14},{5,13},{4,12},{3,11},{2,10},{1,9},{0,8}}"}
  one = s32[] constant(1)
  cond_plus_1 = s32[] add(cond, one)
  ROOT output_tuple = (f32[], s32[]) tuple(collective-permute, cond_plus_1)
}

ENTRY main {
  param_0 = f32[] parameter(0)
  param_2 = s32[] constant(0)
  tuple = (f32[], s32[]) tuple(param_0, param_2)
  ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"15"}}
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  LoopDoubleBufferTransformer double_buffer;
  EXPECT_THAT(double_buffer.Run(module.get()), IsOkAndHolds(true));

  // Get the collective permute instructions outside while body
  for (HloInstruction* instr : module->entry_computation()->instructions()) {
    if (instr->opcode() != HloOpcode::kCollectivePermute) continue;
    ASSERT_TRUE(
        instr->frontend_attributes().map().contains(kSendRecvValidationAttr));
    EXPECT_EQ(instr->frontend_attributes().map().at(kSendRecvValidationAttr),
              "{{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,1}}");
  }

  // Get the collective permute instructions
  HloComputation* while_body =
      module->entry_computation()->root_instruction()->while_body();
  std::vector<HloInstruction*> collective_permutes;
  for (HloInstruction* inst : while_body->instructions()) {
    if (inst->opcode() != HloOpcode::kCollectivePermute) continue;
    collective_permutes.push_back(inst);
  }
  ASSERT_EQ(collective_permutes.size(), 2);
  ASSERT_TRUE(collective_permutes[0]->frontend_attributes().map().contains(
      kSendRecvValidationAttr));
  ASSERT_TRUE(collective_permutes[1]->frontend_attributes().map().contains(
      kSendRecvValidationAttr));
  EXPECT_EQ(collective_permutes[0]->frontend_attributes().map().at(
                kSendRecvValidationAttr),
            "{{3,7},{3,7},{2,6},{2,6},{1,5},{1,5},{0,4},{0,4}}");
  EXPECT_EQ(collective_permutes[1]->frontend_attributes().map().at(
                kSendRecvValidationAttr),
            "{{3,7},{2,6},{2,6},{1,5},{1,5},{0,4},{0,4},{0,3}}");
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       WhileLoopWithCollectivePermuteStartDone) {
  const char* kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

ar_add {
  Arg_1 = f32[] parameter(1)
  Arg_0 = f32[] parameter(0)
  ROOT add_ar = f32[] add(Arg_1, Arg_0)
}

body {
  input_tuple = (f32[], s32[]) parameter(0)
  param_0 = f32[] get-tuple-element(input_tuple), index=0
  cond = s32[] get-tuple-element(input_tuple), index=1
  collective-permute-start = (f32[], f32[], u32[], u32[]) collective-permute-start(param_0), channel_id=1, source_target_pairs={{0,1},{1,2},{2,3},{3,0}},
                             frontend_attributes={_xla_send_recv_validation="{{0,7},{1,8},{2,9},{3,10}}"}
  collective-permute = f32[] collective-permute-done(collective-permute-start)
  one = s32[] constant(1)
  cond_plus_1 = s32[] add(cond, one)
  ROOT output_tuple = (f32[], s32[]) tuple(collective-permute, cond_plus_1)
}

ENTRY main {
  param_0 = f32[] parameter(0)
  param_2 = s32[] constant(0)
  tuple = (f32[], s32[]) tuple(param_0, param_2)
  ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"}}
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  LoopDoubleBufferTransformer double_buffer;
  EXPECT_THAT(double_buffer.Run(module.get()), IsOkAndHolds(true));

  HloComputation* while_body =
      module->entry_computation()->root_instruction()->while_body();
  std::vector<HloInstruction*> collective_permutes;
  for (HloInstruction* inst : while_body->instructions()) {
    if (inst->opcode() != HloOpcode::kCollectivePermuteStart) continue;
    collective_permutes.push_back(inst);
  }
  ASSERT_EQ(collective_permutes.size(), 2);
  ASSERT_TRUE(collective_permutes[0]->frontend_attributes().map().contains(
      kSendRecvValidationAttr));
  ASSERT_TRUE(collective_permutes[1]->frontend_attributes().map().contains(
      kSendRecvValidationAttr));
  EXPECT_EQ(collective_permutes[0]->frontend_attributes().map().at(
                kSendRecvValidationAttr),
            "{{0,4},{1,4},{1,5},{2,5}}");
  EXPECT_EQ(collective_permutes[1]->frontend_attributes().map().at(
                kSendRecvValidationAttr),
            "{{0,3},{0,4},{1,4},{1,5}}");
}

TEST_F(GpuLoopDoubleBufferTransformerTest, WhileLoopWithRecvDone) {
  const char* kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

ar_add {
  Arg_1 = f32[] parameter(1)
  Arg_0 = f32[] parameter(0)
  ROOT add_ar = f32[] add(Arg_1, Arg_0)
}

body {
  input_tuple = (f32[], s32[]) parameter(0)
  param_0 = f32[] get-tuple-element(input_tuple), index=0
  cond = s32[] get-tuple-element(input_tuple), index=1
  after-all.0 = token[] after-all()
  recv.0 = (f32[], u32[], token[]) recv(after-all.0), channel_id=1,
        frontend_attributes={
          _xla_send_recv_source_target_pairs="{{0,1},{1,2},{2,3},{3,0}}",
          _xla_send_recv_pipeline="0",
          _xla_send_recv_validation="{{0,7},{1,8},{2,9},{3,10}}"
        }
  recv-done.0 = (f32[], token[]) recv-done(recv.0), channel_id=1,
        frontend_attributes={
          _xla_send_recv_pipeline="0"
        }
  recv-data = f32[] get-tuple-element(recv-done.0), index=0
  one = s32[] constant(1)
  cond_plus_1 = s32[] add(cond, one)
  ROOT output_tuple = (f32[], s32[]) tuple(recv-data, cond_plus_1)
}

ENTRY main {
  param_0 = f32[] parameter(0)
  param_2 = s32[] constant(0)
  tuple = (f32[], s32[]) tuple(param_0, param_2)
  ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"}}
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  LoopDoubleBufferTransformer double_buffer;
  EXPECT_THAT(double_buffer.Run(module.get()), IsOkAndHolds(true));

  HloComputation* while_body =
      module->entry_computation()->root_instruction()->while_body();
  std::vector<HloInstruction*> collective_permutes;
  for (HloInstruction* inst : while_body->instructions()) {
    if (inst->opcode() != HloOpcode::kRecv) continue;
    collective_permutes.push_back(inst);
  }
  ASSERT_EQ(collective_permutes.size(), 2);
  ASSERT_TRUE(collective_permutes[0]->frontend_attributes().map().contains(
      kSendRecvValidationAttr));
  ASSERT_TRUE(collective_permutes[1]->frontend_attributes().map().contains(
      kSendRecvValidationAttr));
  EXPECT_EQ(collective_permutes[0]->frontend_attributes().map().at(
                kSendRecvValidationAttr),
            "{{0,4},{1,4},{1,5},{2,5}}");
  EXPECT_EQ(collective_permutes[1]->frontend_attributes().map().at(
                kSendRecvValidationAttr),
            "{{0,3},{0,4},{1,4},{1,5}}");
}

TEST_F(GpuLoopDoubleBufferTransformerTest, WhileLoopWithSendDone) {
  const char* kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

ar_add {
  Arg_1 = f32[] parameter(1)
  Arg_0 = f32[] parameter(0)
  ROOT add_ar = f32[] add(Arg_1, Arg_0)
}

body {
  input_tuple = (f32[], s32[]) parameter(0)
  param_0 = f32[] get-tuple-element(input_tuple), index=0
  cond = s32[] get-tuple-element(input_tuple), index=1
  after-all.0 = token[] after-all()
  send.0 = (f32[], u32[], token[]) send(param_0, after-all.0), channel_id=1,
        frontend_attributes={
          _xla_send_recv_source_target_pairs="{{0,1},{1,2},{2,3},{3,0}}",
          _xla_send_recv_pipeline="0",
          _xla_send_recv_validation="{{0,7},{1,8},{2,9},{3,10}}"
        }
  send-done.0 = token[] send-done(send.0), channel_id=1,
        frontend_attributes={
          _xla_send_recv_pipeline="0"
        }
  one = s32[] constant(1)
  cond_plus_1 = s32[] add(cond, one)
  ROOT output_tuple = (f32[], s32[]) tuple(param_0, cond_plus_1)
}

ENTRY main {
  param_0 = f32[] parameter(0)
  param_2 = s32[] constant(0)
  tuple = (f32[], s32[]) tuple(param_0, param_2)
  ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"}}
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  LoopDoubleBufferTransformer double_buffer;
  EXPECT_THAT(double_buffer.Run(module.get()), IsOkAndHolds(true));

  HloComputation* while_body =
      module->entry_computation()->root_instruction()->while_body();
  std::vector<HloInstruction*> collective_permutes;
  for (HloInstruction* inst : while_body->instructions()) {
    if (inst->opcode() != HloOpcode::kSend) continue;
    collective_permutes.push_back(inst);
  }
  ASSERT_EQ(collective_permutes.size(), 2);
  ASSERT_TRUE(collective_permutes[0]->frontend_attributes().map().contains(
      kSendRecvValidationAttr));
  ASSERT_TRUE(collective_permutes[1]->frontend_attributes().map().contains(
      kSendRecvValidationAttr));
  EXPECT_EQ(collective_permutes[0]->frontend_attributes().map().at(
                kSendRecvValidationAttr),
            "{{0,4},{1,4},{1,5},{2,5}}");
  EXPECT_EQ(collective_permutes[1]->frontend_attributes().map().at(
                kSendRecvValidationAttr),
            "{{0,3},{0,4},{1,4},{1,5}}");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
