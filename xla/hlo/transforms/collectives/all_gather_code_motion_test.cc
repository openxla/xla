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

#include "xla/hlo/transforms/collectives/all_gather_code_motion.h"

#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "tsl/platform/statusor.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

class AllGatherCodeMotionTest : public HloHardwareIndependentTestBase {
 protected:
  AllGatherCodeMotion pass_;
};

TEST_F(AllGatherCodeMotionTest, MovesAllGatherOutOfWhileLoop) {
  absl::string_view hlo_string = R"(
    HloModule module

    while_cond {
      param = (bf16[], bf16[4], bf16[4]) parameter(0)
      count = bf16[] get-tuple-element(param), index=0
      limit = bf16[] constant(10)
      ROOT cmp = pred[] compare(count, limit), direction=LT
    }

    while_body {
      param = (bf16[], bf16[4], bf16[4]) parameter(0)
      count = bf16[] get-tuple-element(param), index=0
      data = bf16[4] get-tuple-element(param), index=1
      output = bf16[4] get-tuple-element(param), index=2
      all-gather = bf16[8] all-gather(data), dimensions={0}, replica_groups={{0,1}}
      const = bf16[] constant(1)
      inc = bf16[] add(count, const)
      ROOT tuple = (bf16[], bf16[4], bf16[4]) tuple(inc, data, output)
    }

    ENTRY main {
      data = bf16[4] parameter(0)
      output = bf16[4] parameter(1)
      init = bf16[] constant(0)
      tuple = (bf16[], bf16[4], bf16[4]) tuple(init, data, output)
      while = (bf16[], bf16[4], bf16[4]) while(tuple), condition=while_cond, body=while_body
      ROOT result = bf16[4] get-tuple-element(while), index=2
    }
  )";

  HloModuleConfig config;
  config.set_num_partitions(2);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  EXPECT_TRUE(changed);
  HloDCE dce;
  TF_RETURN_IF_ERROR(dce.Run(module.get()).status());

  auto* while_op = module->entry_computation()->root_instruction()->operand(0);
  EXPECT_THAT(while_op, op::While(op::Tuple(op::Constant(), op::AllGather(),
                                            op::Parameter(1))));
}

TEST_F(AllGatherCodeMotionTest, DoesNotMoveNonParameterAllGather) {
  absl::string_view hlo_string = R"(
    HloModule module

    while_cond {
      param = (bf16[], bf16[4], bf16[4]) parameter(0)
      count = bf16[] get-tuple-element(param), index=0
      limit = bf16[] constant(10)
      ROOT cmp = pred[] compare(count, limit), direction=LT
    }

    while_body {
      param = (bf16[], bf16[4], bf16[4]) parameter(0)
      count = bf16[] get-tuple-element(param), index=0
      data = bf16[4] get-tuple-element(param), index=1
      output = bf16[4] get-tuple-element(param), index=2
      neg = bf16[4] negate(data)
      all-gather = bf16[8] all-gather(neg), dimensions={0}, replica_groups={{0,1}}
      const = bf16[] constant(1)
      inc = bf16[] add(count, const)
      ROOT tuple = (bf16[], bf16[4], bf16[4]) tuple(inc, data, output)
    }

    ENTRY main {
      data = bf16[4] parameter(0)
      output = bf16[4] parameter(1)
      init = bf16[] constant(0)
      tuple = (bf16[], bf16[4], bf16[4]) tuple(init, data, output)
      while = (bf16[], bf16[4], bf16[4]) while(tuple), condition=while_cond, body=while_body
      ROOT result = bf16[4] get-tuple-element(while), index=2
    }
  )";

  HloModuleConfig config;
  config.set_num_partitions(2);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(AllGatherCodeMotionTest, HandlesMultipleAllGathers) {
  absl::string_view hlo_string = R"(
    HloModule module

    while_cond {
      param = (bf16[], bf16[4], bf16[4], bf16[4]) parameter(0)
      count = bf16[] get-tuple-element(param), index=0
      limit = bf16[] constant(10)
      ROOT cmp = pred[] compare(count, limit), direction=LT
    }

    while_body {
      param = (bf16[], bf16[4], bf16[4], bf16[4]) parameter(0)
      count = bf16[] get-tuple-element(param), index=0
      data1 = bf16[4] get-tuple-element(param), index=1
      data2 = bf16[4] get-tuple-element(param), index=2
      output = bf16[4] get-tuple-element(param), index=3
      all-gather.1 = bf16[8] all-gather(data1), dimensions={0}, replica_groups={{0,1}}
      all-gather.2 = bf16[8] all-gather(data2), dimensions={0}, replica_groups={{0,1}}
      const = bf16[] constant(1)
      inc = bf16[] add(count, const)
      ROOT tuple = (bf16[], bf16[4], bf16[4], bf16[4]) tuple(inc, data1, data2, output)
    }

    ENTRY main {
      data1 = bf16[4] parameter(0)
      data2 = bf16[4] parameter(1)
      output = bf16[4] parameter(2)
      init = bf16[] constant(0)
      tuple = (bf16[], bf16[4], bf16[4], bf16[4]) tuple(init, data1, data2, output)
      while = (bf16[], bf16[4], bf16[4], bf16[4]) while(tuple), condition=while_cond, body=while_body
      ROOT result = bf16[4] get-tuple-element(while), index=3
    }
  )";

  HloModuleConfig config;
  config.set_num_partitions(2);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  EXPECT_TRUE(changed);
  HloDCE dce;
  TF_RETURN_IF_ERROR(dce.Run(module.get()).status());

  auto* while_op = module->entry_computation()->root_instruction()->operand(0);
  EXPECT_THAT(while_op,
              op::While(op::Tuple(op::Constant(), op::AllGather(),
                                  op::AllGather(), op::Parameter(2))));
}

TEST_F(AllGatherCodeMotionTest, MovesAllGatherWithIntervingConvert) {
  absl::string_view hlo_string = R"(
    HloModule module

    while_cond {
      param = (bf16[], bf16[4], bf16[4]) parameter(0)
      count = bf16[] get-tuple-element(param), index=0
      limit = bf16[] constant(10)
      ROOT cmp = pred[] compare(count, limit), direction=LT
    }

    while_body {
      param = (bf16[], bf16[4], bf16[4]) parameter(0)
      count = bf16[] get-tuple-element(param), index=0
      data = bf16[4] get-tuple-element(param), index=1
      output = bf16[4] get-tuple-element(param), index=2
      convert = f32[4] convert(data)
      all-gather = f32[8] all-gather(convert), dimensions={0}, replica_groups={{0,1}}
      const = bf16[] constant(1)
      inc = bf16[] add(count, const)
      ROOT tuple = (bf16[], bf16[4], bf16[4]) tuple(inc, data, output)
    }

    ENTRY main {
      data = bf16[4] parameter(0)
      output = bf16[4] parameter(1)
      init = bf16[] constant(0)
      tuple = (bf16[], bf16[4], bf16[4]) tuple(init, data, output)
      while = (bf16[], bf16[4], bf16[4]) while(tuple), condition=while_cond, body=while_body
      ROOT result = bf16[4] get-tuple-element(while), index=2
    }
  )";

  HloModuleConfig config;
  config.set_num_partitions(2);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  EXPECT_TRUE(changed);
  HloDCE dce;
  TF_RETURN_IF_ERROR(dce.Run(module.get()).status());

  auto* while_op = module->entry_computation()->root_instruction()->operand(0);
  EXPECT_THAT(while_op,
              op::While(op::Tuple(op::Constant(), op::AllGather(op::Convert()),
                                  op::Parameter(1))));
}

TEST_F(AllGatherCodeMotionTest, MovesMultipleAllGathersWithConverts) {
  absl::string_view hlo_string = R"(
    HloModule module

    while_cond {
      param = (bf16[], bf16[4], bf16[4], bf16[4]) parameter(0)
      count = bf16[] get-tuple-element(param), index=0
      limit = bf16[] constant(10)
      ROOT cmp = pred[] compare(count, limit), direction=LT
    }

    while_body {
      param = (bf16[], bf16[4], bf16[4], bf16[4]) parameter(0)
      count = bf16[] get-tuple-element(param), index=0
      data1 = bf16[4] get-tuple-element(param), index=1
      data2 = bf16[4] get-tuple-element(param), index=2
      output = bf16[4] get-tuple-element(param), index=3
      convert1 = f32[4] convert(data1)
      convert2 = f32[4] convert(data2)
      all-gather.1 = f32[8] all-gather(convert1), dimensions={0}, replica_groups={{0,1}}
      all-gather.2 = f32[8] all-gather(convert2), dimensions={0}, replica_groups={{0,1}}
      const = bf16[] constant(1)
      inc = bf16[] add(count, const)
      ROOT tuple = (bf16[], bf16[4], bf16[4], bf16[4]) tuple(inc, data1, data2, output)
    }

    ENTRY main {
      data1 = bf16[4] parameter(0)
      data2 = bf16[4] parameter(1)
      output = bf16[4] parameter(2)
      init = bf16[] constant(0)
      tuple = (bf16[], bf16[4], bf16[4], bf16[4]) tuple(init, data1, data2, output)
      while = (bf16[], bf16[4], bf16[4], bf16[4]) while(tuple), condition=while_cond, body=while_body
      ROOT result = bf16[4] get-tuple-element(while), index=3
    }
  )";

  HloModuleConfig config;
  config.set_num_partitions(2);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  EXPECT_TRUE(changed);
  HloDCE dce;
  TF_RETURN_IF_ERROR(dce.Run(module.get()).status());

  auto* while_op = module->entry_computation()->root_instruction()->operand(0);
  EXPECT_THAT(while_op, op::While(op::Tuple(
                            op::Constant(), op::AllGather(op::Convert()),
                            op::AllGather(op::Convert()), op::Parameter(2))));
}

TEST_F(AllGatherCodeMotionTest, OnlyMovesSpecificReplicaGroup) {
  absl::string_view hlo_string = R"(
    HloModule module

    while_cond {
      param = (bf16[], bf16[4], bf16[4], bf16[4]) parameter(0)
      count = bf16[] get-tuple-element(param), index=0
      limit = bf16[] constant(10)
      ROOT cmp = pred[] compare(count, limit), direction=LT
    }

    while_body {
      param = (bf16[], bf16[4], bf16[4], bf16[4]) parameter(0)
      count = bf16[] get-tuple-element(param), index=0
      data1 = bf16[4] get-tuple-element(param), index=1
      data2 = bf16[4] get-tuple-element(param), index=2
      output = bf16[4] get-tuple-element(param), index=3
      
      // This is the all-gather we want to move (all 8 workers)
      all-gather.1 = bf16[32] all-gather(data1), dimensions={0}, replica_groups={{0,1,2,3,4,5,6,7}}
      
      // This should stay in the loop (pairs of workers)
      all-gather.2 = bf16[8] all-gather(data2), dimensions={0}, replica_groups={{0,1},{2,3},{4,5},{6,7}}
      
      const = bf16[] constant(1)
      inc = bf16[] add(count, const)
      ROOT tuple = (bf16[], bf16[4], bf16[4], bf16[4]) tuple(inc, data1, data2, output)
    }

    ENTRY main {
      data1 = bf16[4] parameter(0)
      data2 = bf16[4] parameter(1)
      output = bf16[4] parameter(2)
      init = bf16[] constant(0)
      tuple = (bf16[], bf16[4], bf16[4], bf16[4]) tuple(init, data1, data2, output)
      while = (bf16[], bf16[4], bf16[4], bf16[4]) while(tuple), condition=while_cond, body=while_body
      ROOT result = bf16[4] get-tuple-element(while), index=3
    }
  )";

  HloModuleConfig config;
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  // Create replica group we want to move - all 8 workers
  std::vector<ReplicaGroup> moveable_group;
  ReplicaGroup group;
  for (int64_t i = 0; i < 8; ++i) {
    group.add_replica_ids(i);
  }
  moveable_group.push_back(group);

  // Initialize pass with specific replica group
  AllGatherCodeMotion pass(&moveable_group);

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);
  HloDCE dce;
  TF_RETURN_IF_ERROR(dce.Run(module.get()).status());

  // Verify:
  // 1. The 8-worker all-gather.1 was moved out (should see AllGather in tuple)
  // 2. The pairwise all-gather.2 remained inside the loop (shouldn't see it in
  // tuple)
  auto* while_op = module->entry_computation()->root_instruction()->operand(0);
  EXPECT_THAT(while_op, op::While(op::Tuple(
                            op::Constant(),
                            op::Parameter(0),  // all-gather.1 moved out
                            op::AllGather(),   // all-gather.2 still in loop
                            op::Parameter(2))));
}

TEST_F(AllGatherCodeMotionTest, MovesAllGatherBeforeOptBarrier) {
  absl::string_view hlo_string = R"(
    HloModule module

    while_cond {
      param = (bf16[], bf16[4], bf16[4]) parameter(0)
      count = bf16[] get-tuple-element(param), index=0
      limit = bf16[] constant(10)
      ROOT cmp = pred[] compare(count, limit), direction=LT
    }

    while_body {
      param = (bf16[], bf16[4], bf16[4]) parameter(0)
      count = bf16[] get-tuple-element(param), index=0
      data = bf16[4] get-tuple-element(param), index=1
      output = bf16[4] get-tuple-element(param), index=2
      all-gather = bf16[8] all-gather(data), dimensions={0}, replica_groups={{0,1}}
      opt = (bf16[8]) opt-barrier((bf16[8]) tuple(all-gather))
      result = bf16[8] get-tuple-element(opt), index=0
      const = bf16[] constant(1)
      inc = bf16[] add(count, const)
      ROOT tuple = (bf16[], bf16[4], bf16[4]) tuple(inc, data, output)
    }

    ENTRY main {
      data = bf16[4] parameter(0)
      output = bf16[4] parameter(1)
      init = bf16[] constant(0)
      tuple = (bf16[], bf16[4], bf16[4]) tuple(init, data, output)
      while = (bf16[], bf16[4], bf16[4]) while(tuple), condition=while_cond, body=while_body
      ROOT result = bf16[4] get-tuple-element(while), index=2
    }
  )";

  HloModuleConfig config;
  config.set_num_partitions(2);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass_.Run(module.get()));
  EXPECT_TRUE(changed);
  HloDCE dce;
  TF_RETURN_IF_ERROR(dce.Run(module.get()).status());

  auto* while_op = module->entry_computation()->root_instruction()->operand(0);
  EXPECT_THAT(while_op, op::While(op::Tuple(op::Constant(), op::AllGather(),
                                            op::Parameter(1))));
}

}  // namespace
}  // namespace xla