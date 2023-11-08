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

#include "xla/service/gpu/gpu_replica_reduce_splitter.h"

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
namespace gpu {
namespace {

namespace op = xla::testing::opcode_matchers;

class GpuReplicaReduceSplitterTest : public HloTestBase {
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

    auto changed = ReplicaReduceSplitter().Run(module.get());
    if (!changed.ok()) {
      return changed.status();
    }
    EXPECT_EQ(changed.value(), expect_change);
    return StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }

  template <HloOpcode oc>
  size_t CollectiveCount(std::unique_ptr<HloModule> &module) {
    return absl::c_count_if(module->entry_computation()->instructions(),
                            HloPredicateIsOp<oc>);
  }
};

TEST_F(GpuReplicaReduceSplitterTest, RewriteTest) {
const std::string hlo_string = R"(
HloModule ModulePassChanged
max {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT c = f32[] maximum(a, b)
}

ENTRY main {
  pid = u32[] partition-id()
  param0 = f32[256,128] parameter(0), sharding={replicated}
  abs.0 = f32[256,128] abs(param0)
  c0 = f32[] constant(-inf)
  gtable = u32[8]{0} constant({0,1,0,1,0,1,0,1})
  dynamic-slice.18 = u32[1] dynamic-slice(gtable, pid), dynamic_slice_sizes={1}
  reshape.146 = u32[] reshape(dynamic-slice.18)
  c1 = s32[2]{0} constant({0,64})
  dynamic-slice.19 = s32[1] dynamic-slice(c1, reshape.146), dynamic_slice_sizes={1}
  reshape.147 = s32[] reshape(dynamic-slice.19)
  c2 = s32[] constant(0)
  dynamic-slice.20 = f32[256,64] dynamic-slice(param0, c2, reshape.147), dynamic_slice_sizes={256,64}
  ROOT amax = f32[] reduce(abs.0, c0), dimensions={0,1}, to_apply=max
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/true));
  // graph should contain 1 all-reduce
  EXPECT_EQ(CollectiveCount<HloOpcode::kAllReduce>(module), 1);
}

}  // namespace
}  // namespace gpu
}  // namespace xla