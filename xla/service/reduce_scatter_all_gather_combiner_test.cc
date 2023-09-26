/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/reduce_scatter_all_gather_combiner.h"

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class ReduceScatterAllGatherCombinerTest : public HloTestBase {
 public:
  StatusOr<std::unique_ptr<HloModule>> RunPass(absl::string_view hlo_module,
                                               int64_t num_replicas,
                                               int64_t num_partitions,
                                               bool expect_change) {
    HloModuleConfig config = GetModuleConfigForTest(
        /*replica_count=*/num_replicas,
        /*num_partitions=*/num_partitions);
    config.set_use_spmd_partitioning(num_partitions > 1);
    TF_ASSIGN_OR_RETURN(auto module,
                        ParseAndReturnVerifiedModule(hlo_module, config));

    auto changed = ReduceScatterAllGatherCombiner().Run(module.get());
    if (!changed.ok()) {
      return changed.status();
    }
    EXPECT_EQ(changed.value(), expect_change);
    return StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }

  template <HloOpcode op>
  size_t CollectiveCount(std::unique_ptr<HloModule> &module) {
    return absl::c_count_if(module->entry_computation()->instructions(),
                            HloPredicateIsOp<op>);
  }
};

TEST_F(ReduceScatterAllGatherCombinerTest, SimpleCombine) {
  absl::string_view hlo_string = R"(
HloModule ReduceScatter

add {
  x = bf16[] parameter(0)
  y = bf16[] parameter(1)
  ROOT add = bf16[] add(x, y)
}

ENTRY main {
param.1 = bf16[8,128,1024] parameter(0)
reduce-scatter.1 = bf16[8,64,1024] reduce-scatter(param.1), channel_id=8, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
all-gather.1 = bf16[8,128,1024] all-gather(reduce-scatter.1), channel_id=5, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={1}, use_global_device_ids=true
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*expect_change=*/true));
  EXPECT_EQ(CollectiveCount<HloOpcode::kAllReduce>(module), 1);
}

TEST_F(ReduceScatterAllGatherCombinerTest, IncompatibleProperties) {
  absl::string_view hlo_string = R"(
HloModule ReduceScatter

add {
  x = bf16[] parameter(0)
  y = bf16[] parameter(1)
  ROOT add = bf16[] add(x, y)
}

ENTRY main {
param.1 = bf16[8,128,1024] parameter(0)
reduce-scatter.1 = bf16[8,64,1024] reduce-scatter(param.1), channel_id=8, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
all-gather.1 = bf16[16,64,1024] all-gather(reduce-scatter.1), channel_id=5, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={0}, use_global_device_ids=true
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*expect_change=*/false));
}

TEST_F(ReduceScatterAllGatherCombinerTest, IncompatibleReplicaGroups) {
  absl::string_view hlo_string = R"(
HloModule ReduceScatter

add {
  x = bf16[] parameter(0)
  y = bf16[] parameter(1)
  ROOT add = bf16[] add(x, y)
}

ENTRY main {
param.1 = bf16[8,128,1024] parameter(0)
reduce-scatter.1 = bf16[8,64,1024] reduce-scatter(param.1), channel_id=8, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
all-gather.1 = bf16[8,128,1024] all-gather(reduce-scatter.1), channel_id=5, replica_groups={{0,2},{1,3},{4,5},{6,7}}, dimensions={1}, use_global_device_ids=true
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*expect_change=*/false));
}

TEST_F(ReduceScatterAllGatherCombinerTest, IncompatibleDeviceIds) {
  absl::string_view hlo_string = R"(
HloModule ReduceScatter

add {
  x = bf16[] parameter(0)
  y = bf16[] parameter(1)
  ROOT add = bf16[] add(x, y)
}

ENTRY main {
param.1 = bf16[8,128,1024] parameter(0)
reduce-scatter.1 = bf16[8,64,1024] reduce-scatter(param.1), channel_id=8, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
all-gather.1 = bf16[8,128,1024] all-gather(reduce-scatter.1), channel_id=5, replica_groups={{0,2},{1,3},{4,5},{6,7}}, dimensions={1}, use_global_device_ids=false
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*expect_change=*/false));
}

TEST_F(ReduceScatterAllGatherCombinerTest, MultiplePatterns) {
  absl::string_view hlo_string = R"(
HloModule ReduceScatter

add {
  x = bf16[] parameter(0)
  y = bf16[] parameter(1)
  ROOT add = bf16[] add(x, y)
}

ENTRY main {
param.1 = bf16[8,128,1024] parameter(0)
param.2 = bf16[8,128,1024] parameter(1)
reduce-scatter.1 = bf16[8,64,1024] reduce-scatter(param.1), channel_id=8, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
all-gather.1 = bf16[8,128,1024] all-gather(reduce-scatter.1), channel_id=5, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={1}, use_global_device_ids=true
reduce-scatter.2 = bf16[8,64,1024] reduce-scatter(param.2), channel_id=9, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
all-gather.2 = bf16[8,128,1024] all-gather(reduce-scatter.2), channel_id=5, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={1}, use_global_device_ids=true
add.1 = bf16[8,128,1024] add(all-gather.1, all-gather.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*expect_change=*/true));
  EXPECT_EQ(CollectiveCount<HloOpcode::kAllReduce>(module), 2);
}

TEST_F(ReduceScatterAllGatherCombinerTest, ReduceScatterWithMoreThanOneUser) {
  /* The following pattern should stay untouched as the the ReduceScatters
     have two users

      Param            Param
        |                |
    ReduceScatter   ReduceScatter
    |          \     /        |
    AllGather    Add      AllGather

  */

  absl::string_view hlo_string = R"(
HloModule ReduceScatter

add {
  x = bf16[] parameter(0)
  y = bf16[] parameter(1)
  ROOT add = bf16[] add(x, y)
}

ENTRY main {
param.1 = bf16[8,128,1024] parameter(0)
param.2 = bf16[8,128,1024] parameter(1)
reduce-scatter.1 = bf16[8,64,1024] reduce-scatter(param.1), channel_id=8, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
all-gather.1 = bf16[8,128,1024] all-gather(reduce-scatter.1), channel_id=5, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={1}, use_global_device_ids=true
reduce-scatter.2 = bf16[8,64,1024] reduce-scatter(param.2), channel_id=9, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, dimensions={1}, to_apply=add
all-gather.2 = bf16[8,128,1024] all-gather(reduce-scatter.2), channel_id=5, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={1}, use_global_device_ids=true

add.1 = bf16[8,128,1024] add(all-gather.1, all-gather.2)
add.2 = bf16[8,64,1024] add(reduce-scatter.1, reduce-scatter.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/8,
                                               /*num_partitions=*/1,
                                               /*expect_change=*/false));
}

}  // namespace
}  // namespace xla
