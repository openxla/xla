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

#include "xla/hlo/transforms/simplifiers/all_gather_pad_ds_simplifier.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <complex>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal_util.h"
#include "xla/service/collective_opt_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;
using ::testing::ElementsAre;
using ::testing::Pair;

class AllGatherPadDsSimplifierTest : public HloHardwareIndependentTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> RunPass(
      absl::string_view hlo_module, int64_t num_replicas,
      int64_t num_partitions, bool expect_change) {
    HloModuleConfig config = GetModuleConfigForTest(
        /*replica_count=*/num_replicas,
        /*num_partitions=*/num_partitions);
    config.set_use_spmd_partitioning(num_partitions > 1);
    ABSL_ASSIGN_OR_RETURN(auto module,
                          ParseAndReturnVerifiedModule(hlo_module, config));
    ABSL_ASSIGN_OR_RETURN(bool changed,
                          AllGatherPadDsSimplifier().Run(module.get(), {}));
    EXPECT_EQ(changed, expect_change);
    LOG(INFO) << "new module: " << module->ToString();
    return module;
  }
};

TEST_F(AllGatherPadDsSimplifierTest, MultiReplicaGenericCaseLowPad) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)

      const.24 = s32[1]{0} constant({24})

      all-gather = f64[1,8,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[2,4]<=[8], dimensions={1}, use_global_device_ids=true,
        frontend_attributes={collective_group_key="g0"}

      // Pad the all-gather result.
      pad = f64[1,96,40]{2,1,0} pad(all-gather, zero), padding=0_0x0_88x0_0

      // Get the ds offset on large pad.
      const.list = s32[8]{0} constant({0, 24, 48, 72, 0, 24, 48, 72})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)

      ROOT ds = f64[1,24,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,24,40},
        frontend_attributes={collective_group_key="stale_group"}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/true));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->get_frontend_attribute(kCollectiveGroupKeyAttr),
            std::nullopt);
  EXPECT_THAT(
      root, op::Select(op::Compare(op::Reshape(op::DynamicSlice(
                                       op::Constant(), op::PartitionId())),
                                   op::Constant()),
                       op::Concatenate(op::Parameter(0),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::Broadcast(op::Constant())),
                       op::Broadcast(op::Constant())));

  const HloInstruction* const_list = root               // select
                                         ->operand(0)   // compare
                                         ->operand(0)   // reshape
                                         ->operand(0)   // dynamic-slice
                                         ->operand(0);  // constant
  ASSERT_NE(const_list, nullptr);
  EXPECT_THAT(const_list, op::Constant());
  EXPECT_EQ(const_list->literal(),
            LiteralUtil::CreateR1<int64_t>({1, 0, 0, 0, 1, 0, 0, 0}));

  const HloInstruction* concate = root->operand(1);
  ASSERT_NE(concate, nullptr);
  EXPECT_THAT(concate, op::Concatenate());
  EXPECT_THAT(concate->operand(1)->source_target_pairs(),
              ElementsAre(Pair(5, 4), Pair(1, 0)));
  EXPECT_THAT(concate->operand(2)->source_target_pairs(),
              ElementsAre(Pair(6, 4), Pair(2, 0)));
  EXPECT_THAT(concate->operand(3)->source_target_pairs(),
              ElementsAre(Pair(7, 4), Pair(3, 0)));
  for (int64_t i = 1; i <= 3; ++i) {
    EXPECT_EQ(
        concate->operand(i)->get_frontend_attribute(kCollectiveGroupKeyAttr),
        "g0");
    EXPECT_EQ(concate->operand(i)->channel_id(), 4);
  }
}

TEST_F(AllGatherPadDsSimplifierTest,
       MultiReplicaGenericCaseLowPadMultiplyIndex) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)

      all-gather = f64[1,8,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[2,4]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,96,40]{2,1,0} pad(all-gather, zero), padding=0_0x0_88x0_0

      // Get the ds offset on large pad.
      // const.list = s32[8]{0} constant({0, 24, 48, 72, 0, 24, 48, 72})
      const.list = s32[8]{0} constant({0, 1, 2, 3, 0, 1, 2, 3})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      const.24 = s32[1]{0} constant({24})
      multiply = s32[1]{0} multiply(dynamic-slice, const.24)
      reshape.15 = s32[] reshape(multiply)

      ROOT ds = f64[1,24,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,24,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/true));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, op::Select(op::Compare(op::Reshape(op::DynamicSlice(
                                       op::Constant(), op::PartitionId())),
                                   op::Constant()),
                       op::Concatenate(op::Parameter(0),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::Broadcast(op::Constant())),
                       op::Broadcast(op::Constant())));

  const HloInstruction* const_list = root               // select
                                         ->operand(0)   // compare
                                         ->operand(0)   // reshape
                                         ->operand(0)   // dynamic-slice
                                         ->operand(0);  // constant
  ASSERT_NE(const_list, nullptr);
  EXPECT_THAT(const_list, op::Constant());
  EXPECT_EQ(const_list->literal(),
            LiteralUtil::CreateR1<int64_t>({1, 0, 0, 0, 1, 0, 0, 0}));

  const HloInstruction* concate = root->operand(1);
  ASSERT_NE(concate, nullptr);
  EXPECT_THAT(concate, op::Concatenate());
  EXPECT_THAT(concate->operand(1)->source_target_pairs(),
              ElementsAre(Pair(5, 4), Pair(1, 0)));
  EXPECT_THAT(concate->operand(2)->source_target_pairs(),
              ElementsAre(Pair(6, 4), Pair(2, 0)));
  EXPECT_THAT(concate->operand(3)->source_target_pairs(),
              ElementsAre(Pair(7, 4), Pair(3, 0)));
}

TEST_F(AllGatherPadDsSimplifierTest, 4ReplicaGenericCaseLowPad) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)

      const.24 = s32[1]{0} constant({24})

      all-gather = f64[1,4,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[4,2]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,24,40]{2,1,0} pad(all-gather, zero), padding=0_0x0_20x0_0

      // Get the ds offset on large pad.
      const.list = s32[8]{0} constant({0, 6, 0, 6, 0, 6, 0, 6})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)

      ROOT ds = f64[1,6,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,6,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/true));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, op::Select(op::Compare(op::Reshape(op::DynamicSlice(
                                       op::Constant(), op::PartitionId())),
                                   op::Constant()),
                       op::Concatenate(op::Parameter(0),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::Broadcast(op::Constant())),
                       op::Broadcast(op::Constant())));

  const HloInstruction* const_list = root               // select
                                         ->operand(0)   // compare
                                         ->operand(0)   // reshape
                                         ->operand(0)   // dynamic-slice
                                         ->operand(0);  // constant
  ASSERT_NE(const_list, nullptr);
  EXPECT_THAT(const_list, op::Constant());
  EXPECT_EQ(const_list->literal(),
            LiteralUtil::CreateR1<int64_t>({1, 0, 1, 0, 1, 0, 1, 0}));

  const HloInstruction* concate = root->operand(1);
  ASSERT_NE(concate, nullptr);
  EXPECT_THAT(concate, op::Concatenate());
  EXPECT_THAT(concate->operand(1)->source_target_pairs(),
              ElementsAre(Pair(7, 6), Pair(5, 4), Pair(3, 2), Pair(1, 0)));
}

TEST_F(AllGatherPadDsSimplifierTest, MultiReplicaGenericCaseHighPad) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
            zero_idx = s32[] constant(0)

      const.24 = s32[1]{0} constant({24})

      all-gather = f64[1,8,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[2,4]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,96,40]{2,1,0} pad(all-gather, zero), padding=0_0x88_0x0_0

      // Get the ds offset on large pad.
      const.list = s32[8]{0} constant({0, 24, 48, 72, 0, 24, 48, 72})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)

      ROOT ds = f64[1,24,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,24,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/true));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      op::Select(op::Compare(op::Reshape(op::DynamicSlice(op::Constant(),
                                                          op::PartitionId())),
                             op::Constant()),
                 op::Concatenate(
                     op::Broadcast(op::Constant()),
                     op::CollectivePermute(op::Parameter(0)),  // {{4,7},{0,3}}
                     op::CollectivePermute(op::Parameter(0)),  // {{5,7},{1,3}}
                     op::CollectivePermute(op::Parameter(0)),  // {{6,7},{2,3}}
                     op::Parameter(0)),
                 op::Broadcast(op::Constant())));

  const HloInstruction* const_list = root->operand(0)   // compare
                                         ->operand(0)   // reshape
                                         ->operand(0)   // dynamic-slice
                                         ->operand(0);  // constant
  ASSERT_NE(const_list, nullptr);
  EXPECT_THAT(const_list, op::Constant());
  EXPECT_EQ(const_list->literal(),
            LiteralUtil::CreateR1<int64_t>({0, 0, 0, 1, 0, 0, 0, 1}));

  const HloInstruction* concate = root->operand(1);
  ASSERT_NE(concate, nullptr);
  EXPECT_THAT(concate, op::Concatenate());
  EXPECT_THAT(concate->operand(1)->source_target_pairs(),
              ElementsAre(Pair(4, 7), Pair(0, 3)));
  EXPECT_THAT(concate->operand(2)->source_target_pairs(),
              ElementsAre(Pair(5, 7), Pair(1, 3)));
  EXPECT_THAT(concate->operand(3)->source_target_pairs(),
              ElementsAre(Pair(6, 7), Pair(2, 3)));
}

TEST_F(AllGatherPadDsSimplifierTest, MultiReplicaGenericCaseLowPadNoGap) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)

      const.24 = s32[1]{0} constant({24})

      all-gather = f64[1,8,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[2,4]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,32,40]{2,1,0} pad(all-gather, zero), padding=0_0x0_24x0_0

      // Get the ds offset on large pad.
      const.list = s32[8]{0} constant({0, 8, 16, 24, 0, 8, 16, 24})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)

      ROOT ds = f64[1,8,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,8,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/true));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, op::Select(op::Compare(op::Reshape(op::DynamicSlice(
                                       op::Constant(), op::PartitionId())),
                                   op::Constant()),
                       op::Concatenate(op::Parameter(0),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0))),
                       op::Broadcast(op::Constant())));

  const HloInstruction* const_list = root->operand(0)   // compare
                                         ->operand(0)   // reshape
                                         ->operand(0)   // dynamic-slice
                                         ->operand(0);  // constant
  ASSERT_NE(const_list, nullptr);
  EXPECT_THAT(const_list, op::Constant());
  EXPECT_EQ(const_list->literal(),
            LiteralUtil::CreateR1<int64_t>({1, 0, 0, 0, 1, 0, 0, 0}));

  const HloInstruction* concate = root->operand(1);
  ASSERT_NE(concate, nullptr);
  EXPECT_THAT(concate, op::Concatenate());
  EXPECT_THAT(concate->operand(1)->source_target_pairs(),
              ElementsAre(Pair(5, 4), Pair(1, 0)));
  EXPECT_THAT(concate->operand(2)->source_target_pairs(),
              ElementsAre(Pair(6, 4), Pair(2, 0)));
  EXPECT_THAT(concate->operand(3)->source_target_pairs(),
              ElementsAre(Pair(7, 4), Pair(3, 0)));
}

TEST_F(AllGatherPadDsSimplifierTest, SingleReplicaGenericCaseLowPad) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)

      const.24 = s32[1]{0} constant({24})

      all-gather = f64[1,16,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[1,8]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,128,40]{2,1,0} pad(all-gather, zero), padding=0_0x112_0x0_0

      // Get the ds offset on large pad.
      const.list = s32[8]{0} constant({0, 16, 32, 48, 64, 80 , 96, 112})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)

      ROOT ds = f64[1,16,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,16,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/true));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, op::Select(op::Compare(op::Reshape(op::DynamicSlice(
                                       op::Constant(), op::PartitionId())),
                                   op::Constant()),
                       op::Concatenate(op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::CollectivePermute(op::Parameter(0)),
                                       op::Parameter(0)),
                       op::Broadcast(op::Constant())));

  const HloInstruction* const_list = root->operand(0)   // compare
                                         ->operand(0)   // reshape
                                         ->operand(0)   // dynamic-slice
                                         ->operand(0);  // constant
  ASSERT_NE(const_list, nullptr);
  EXPECT_THAT(const_list, op::Constant());
  EXPECT_EQ(const_list->literal(),
            LiteralUtil::CreateR1<int64_t>({0, 0, 0, 0, 0, 0, 0, 1}));

  const HloInstruction* concate = root->operand(1);
  ASSERT_NE(concate, nullptr);
  EXPECT_THAT(concate, op::Concatenate());
  EXPECT_THAT(concate->operand(0)->source_target_pairs(),
              ElementsAre(Pair(0, 7)));
  EXPECT_THAT(concate->operand(1)->source_target_pairs(),
              ElementsAre(Pair(1, 7)));
  EXPECT_THAT(concate->operand(2)->source_target_pairs(),
              ElementsAre(Pair(2, 7)));
  EXPECT_THAT(concate->operand(3)->source_target_pairs(),
              ElementsAre(Pair(3, 7)));
  EXPECT_THAT(concate->operand(4)->source_target_pairs(),
              ElementsAre(Pair(4, 7)));
  EXPECT_THAT(concate->operand(5)->source_target_pairs(),
              ElementsAre(Pair(5, 7)));
  EXPECT_THAT(concate->operand(6)->source_target_pairs(),
              ElementsAre(Pair(6, 7)));
}

TEST_F(AllGatherPadDsSimplifierTest, SingleReplicaGenericCaseLowPadUnordered) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)

      const.24 = s32[1]{0} constant({24})

      all-gather = f64[1,16,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[1,8]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,128,40]{2,1,0} pad(all-gather, zero), padding=0_0x112_0x0_0

      // Get the ds offset on large pad.
      const.list = s32[8]{0} constant({16, 0, 32, 48, 64, 112 , 96, 80})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)

      ROOT ds = f64[1,16,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,16,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/false));
}

TEST_F(AllGatherPadDsSimplifierTest, NotMatchInValidPadSmallPad) {
  absl::string_view hlo_string_small_pad = R"(
   HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)

      const.24 = s32[1]{0} constant({24})

      all-gather = f64[1,16,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[1,8]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,24,40]{2,1,0} pad(all-gather, zero), padding=0_0x8_0x0_0

      // Get the ds offset on large pad.
      const.list = s32[8]{0} constant({0, 3, 6, 9, 12, 15 , 18, 21})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)

      ROOT ds = f64[1,3,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,3,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string_small_pad,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/false));
}

TEST_F(AllGatherPadDsSimplifierTest, NotMatchInValidPadInteriorPad) {
  absl::string_view hlo_string_interior_padding = R"(
   HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)

      const.24 = s32[1]{0} constant({24})

      all-gather = f64[1,16,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[1,8]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,76,40]{2,1,0} pad(all-gather, zero), padding=0_0x0_0_4x0_0

      // Get the ds offset on large pad.
      const.list = s32[8]{0} constant({0, 16, 32, 48, 64, 80 , 96, 112})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)

      ROOT ds = f64[1,16,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,16,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string_interior_padding,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/false));
}

TEST_F(AllGatherPadDsSimplifierTest, NotMatchShortSlice) {
  absl::string_view hlo_string = R"(
   HloModule module

    ENTRY entry {

      param = f64[1,2,40]{2,1,0} parameter(0)

      zero = f64[] constant(0)
      zero_idx = s32[] constant(0)

      const.24 = s32[1]{0} constant({24})

      all-gather = f64[1,16,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[1,8]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result.
      pad = f64[1,128,40]{2,1,0} pad(all-gather, zero), padding=0_0x112_0x0_0

      // Slice on two all-gather result.
      const.list = s32[8]{0} constant({0, 8, 16, 24, 32, 40 , 48, 56})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)

      ROOT ds = f64[1,8,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,8,40}
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string,
                                               /*num_replicas=*/1,
                                               /*num_partitions=*/8,
                                               /*expect_change=*/false));
}

class ExtractValidPadSpecTest : public HloHardwareIndependentTestBase {};

// Test case for ExtractValidPadSpec where the padding is valid.
// The padding is on the high side of the split dimension.
TEST_F(ExtractValidPadSpecTest, SuccessHighPad) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
      param = f64[1,2,40]{2,1,0} parameter(0)
      zero = f64[] constant(0)
      all-gather = f64[1,8,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[2,4]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result on the high side of dimension 1.
      pad = f64[1,96,40]{2,1,0} pad(all-gather, zero), padding=0_0x0_88x0_0

      zero_idx = s32[] constant(0)
      const.list = s32[8]{0} constant({0, 24, 48, 72, 0, 24, 48, 72})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)
      ROOT ds = f64[1,24,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,24,40}
    }
  )";
  HloModuleConfig config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  HloInstruction* pad =
      module->entry_computation()->GetInstructionWithName("pad");
  ASSERT_NE(pad, nullptr);
  HloInstruction* ds =
      module->entry_computation()->GetInstructionWithName("ds");
  ASSERT_NE(ds, nullptr);
  HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("all-gather");
  ASSERT_NE(ag, nullptr);

  auto pad_spec = ExtractValidPadSpec(*Cast<HloPadInstruction>(pad),
                                      ds->shape(), ag->shape(),
                                      /*split_dim=*/1);
  ASSERT_TRUE(pad_spec.has_value());
  EXPECT_EQ(pad_spec->split_dim, 1);
  EXPECT_EQ(pad_spec->start_offset, 8);
  EXPECT_EQ(pad_spec->end_offset, 96);
}

// Test case for ExtractValidPadSpec where the padding is on multiple
// dimensions. This is not a valid padding spec.
TEST_F(ExtractValidPadSpecTest, PadOnMultipleDims) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
      param = f64[1,2,40]{2,1,0} parameter(0)
      zero = f64[] constant(0)
      all-gather = f64[1,8,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[2,4]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result on dims 1 and 2.
      pad = f64[1,96,41]{2,1,0} pad(all-gather, zero), padding=0_0x0_88x0_1

      zero_idx = s32[] constant(0)
      const.list = s32[8]{0} constant({0, 24, 48, 72, 0, 24, 48, 72})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)
      ROOT ds = f64[1,24,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,24,40}
    }
  )";
  HloModuleConfig config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  HloInstruction* pad =
      module->entry_computation()->GetInstructionWithName("pad");
  ASSERT_NE(pad, nullptr);
  HloInstruction* ds =
      module->entry_computation()->GetInstructionWithName("ds");
  ASSERT_NE(ds, nullptr);
  HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("all-gather");
  ASSERT_NE(ag, nullptr);

  auto pad_spec = ExtractValidPadSpec(*Cast<HloPadInstruction>(pad),
                                      ds->shape(), ag->shape(),
                                      /*split_dim=*/1);
  EXPECT_FALSE(pad_spec.has_value());
}

// Test case for ExtractValidPadSpec where the padding is on both sides of the
// split dimension. This is not a valid padding spec.
TEST_F(ExtractValidPadSpecTest, PadOnBothSides) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
      param = f64[1,2,40]{2,1,0} parameter(0)
      zero = f64[] constant(0)
      all-gather = f64[1,8,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[2,4]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result on both sides of dim 1.
      pad = f64[1,96,40]{2,1,0} pad(all-gather, zero), padding=0_0x1_87x0_0

      zero_idx = s32[] constant(0)
      const.list = s32[8]{0} constant({0, 24, 48, 72, 0, 24, 48, 72})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)
      ROOT ds = f64[1,24,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,24,40}
    }
  )";
  HloModuleConfig config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  HloInstruction* pad =
      module->entry_computation()->GetInstructionWithName("pad");
  ASSERT_NE(pad, nullptr);
  HloInstruction* ds =
      module->entry_computation()->GetInstructionWithName("ds");
  ASSERT_NE(ds, nullptr);
  HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("all-gather");
  ASSERT_NE(ag, nullptr);

  auto pad_spec = ExtractValidPadSpec(*Cast<HloPadInstruction>(pad),
                                      ds->shape(), ag->shape(),
                                      /*split_dim=*/1);
  EXPECT_FALSE(pad_spec.has_value());
}

// Test case for ExtractValidPadSpec where the padding has interior padding.
// This is not a valid padding spec.
TEST_F(ExtractValidPadSpecTest, InteriorPadding) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY entry {
      param = f64[1,2,40]{2,1,0} parameter(0)
      zero = f64[] constant(0)
      all-gather = f64[1,8,40]{2,1,0} all-gather(param), channel_id=4,
        replica_groups=[2,4]<=[8], dimensions={1}, use_global_device_ids=true

      // Pad the all-gather result with interior padding on dim 1.
      pad = f64[1,103,40]{2,1,0} pad(all-gather, zero), padding=0_0x0_88_1x0_0

      zero_idx = s32[] constant(0)
      const.list = s32[8]{0} constant({0, 24, 48, 72, 0, 24, 48, 72})
      partition-id = u32[] partition-id()
      dynamic-slice = s32[1]{0} dynamic-slice(const.list, partition-id),
        dynamic_slice_sizes={1}
      reshape.15 = s32[] reshape(dynamic-slice)
      ROOT ds = f64[1,24,40]{2,1,0} dynamic-slice(pad, zero_idx, reshape.15,
        zero_idx), dynamic_slice_sizes={1,24,40}
    }
  )";
  HloModuleConfig config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  HloInstruction* pad =
      module->entry_computation()->GetInstructionWithName("pad");
  ASSERT_NE(pad, nullptr);
  HloInstruction* ds =
      module->entry_computation()->GetInstructionWithName("ds");
  ASSERT_NE(ds, nullptr);
  HloInstruction* ag =
      module->entry_computation()->GetInstructionWithName("all-gather");
  ASSERT_NE(ag, nullptr);

  auto pad_spec = ExtractValidPadSpec(*Cast<HloPadInstruction>(pad),
                                      ds->shape(), ag->shape(),
                                      /*split_dim=*/1);
  EXPECT_FALSE(pad_spec.has_value());
}
class GetPartitionIdForOffsetTest : public HloHardwareIndependentTestBase {};

TEST_F(GetPartitionIdForOffsetTest, InRangeFunctionality) {
  OffsetToIdMap offset_to_partition_map = {{0, 0}, {24, 1}, {48, 2}, {72, 3}};

  auto result = GetPartitionIdForOffset(offset_to_partition_map, 0);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->first, 0);
  EXPECT_EQ(result.value()->second, 0);

  result = GetPartitionIdForOffset(offset_to_partition_map, -1);
  ASSERT_FALSE(result.has_value());

  result = GetPartitionIdForOffset(offset_to_partition_map, 23);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->first, 0);
  EXPECT_EQ(result.value()->second, 0);

  result = GetPartitionIdForOffset(offset_to_partition_map, 24);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->first, 24);
  EXPECT_EQ(result.value()->second, 1);

  result = GetPartitionIdForOffset(offset_to_partition_map, 47);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->first, 24);
  EXPECT_EQ(result.value()->second, 1);

  result = GetPartitionIdForOffset(offset_to_partition_map, 48);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->first, 48);
  EXPECT_EQ(result.value()->second, 2);

  result = GetPartitionIdForOffset(offset_to_partition_map, 72);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->first, 72);
  EXPECT_EQ(result.value()->second, 3);

  result = GetPartitionIdForOffset(offset_to_partition_map, 75);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->first, 72);
  EXPECT_EQ(result.value()->second, 3);
}

TEST_F(GetPartitionIdForOffsetTest, EmptyMap) {
  OffsetToIdMap offset_to_partition_map = {};
  EXPECT_EQ(GetPartitionIdForOffset(offset_to_partition_map, 24), std::nullopt);
}

TEST_F(GetPartitionIdForOffsetTest, OutOfRange) {
  OffsetToIdMap offset_to_partition_map = {{0, 0}, {24, 1}, {48, 2}, {72, 3}};
  EXPECT_EQ(GetPartitionIdForOffset(offset_to_partition_map, -1), std::nullopt);

  auto result = GetPartitionIdForOffset(offset_to_partition_map, 100);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->first, 72);
  EXPECT_EQ(result.value()->second, 3);
}

class AddPredInstrBasedOnPartitionIdAndListTest
    : public HloHardwareIndependentTestBase {};

TEST_F(AddPredInstrBasedOnPartitionIdAndListTest, Basic) {
  HloModuleConfig config = GetModuleConfigForTest();
  auto module = std::make_unique<HloModule>(TestName(), config);
  HloComputation::Builder builder(TestName());
  builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(U32, {}), "dumb"));
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  std::vector<int64_t> select_list = {0, 0, 0, 1, 0, 0, 0, 1};
  HloInstruction* pred =
      AddPredInstrBasedOnPartitionIdAndList(computation, select_list);

  EXPECT_THAT(pred, op::Compare(op::Reshape(op::DynamicSlice(
                                    op::Constant(), op::PartitionId())),
                                op::Constant()));
  EXPECT_EQ(pred->shape().element_type(), PRED);
}

// Evaluates a rank-1 F32 multi-partition computation and returns raw IEEE-754
// bit patterns per partition (distinguishing +0.0f, -0.0f, subnormals, sNaN).
std::vector<std::vector<uint32_t>> Eval1DF32PartitionsBitwise(
    const HloModule& module,
    const std::vector<std::vector<uint32_t>>& param0_bits) {
  const int64_t num_partitions = param0_bits.size();
  const HloComputation* comp = module.entry_computation();
  std::map<const HloInstruction*, std::vector<std::vector<uint32_t>>> f32_vals;
  std::map<const HloInstruction*, std::vector<std::vector<int64_t>>> i64_vals;

  auto get_i64_scalar = [&](const HloInstruction* hlo, int64_t p) -> int64_t {
    return i64_vals.at(hlo)[p][0];
  };

  for (const HloInstruction* inst : comp->MakeInstructionPostOrder()) {
    const PrimitiveType ty = inst->shape().element_type();
    if (inst->opcode() == HloOpcode::kParameter) {
      f32_vals[inst] = param0_bits;
    } else if (inst->opcode() == HloOpcode::kPartitionId) {
      std::vector<std::vector<int64_t>> out(num_partitions);
      for (int64_t p = 0; p < num_partitions; ++p) {
        out[p] = {p};
      }
      i64_vals[inst] = std::move(out);
    } else if (inst->opcode() == HloOpcode::kConstant) {
      const int64_t n =
          inst->shape().dimensions().empty() ? 1 : inst->shape().dimensions(0);
      if (ty == F32) {
        uint32_t bits =
            std::bit_cast<uint32_t>(inst->literal().GetFirstElement<float>());
        f32_vals[inst].assign(num_partitions, std::vector<uint32_t>(n, bits));
      } else {
        std::vector<int64_t> v(n, 0);
        for (int64_t i = 0; i < n; ++i) {
          v[i] = inst->shape().dimensions().empty()
                     ? *inst->literal().GetIntegralAsS64({})
                     : *inst->literal().GetIntegralAsS64({i});
        }
        i64_vals[inst].assign(num_partitions, v);
      }
    } else if (inst->opcode() == HloOpcode::kIota) {
      const int64_t n = inst->shape().dimensions(0);
      std::vector<int64_t> v(n, 0);
      for (int64_t i = 0; i < n; ++i) {
        v[i] = i;
      }
      i64_vals[inst].assign(num_partitions, v);
    } else if (inst->opcode() == HloOpcode::kReshape ||
               inst->opcode() == HloOpcode::kCopy) {
      i64_vals[inst] = i64_vals.at(inst->operand(0));
    } else if (inst->opcode() == HloOpcode::kMultiply) {
      std::vector<std::vector<int64_t>> out(num_partitions);
      for (int64_t p = 0; p < num_partitions; ++p) {
        const auto& a = i64_vals.at(inst->operand(0))[p];
        const auto& b = i64_vals.at(inst->operand(1))[p];
        if (ty == U32) {
          out[p] = {static_cast<int64_t>(static_cast<uint32_t>(a[0]) *
                                         static_cast<uint32_t>(b[0]))};
        } else {
          out[p] = {static_cast<int64_t>(static_cast<int32_t>(
              static_cast<uint32_t>(a[0]) * static_cast<uint32_t>(b[0])))};
        }
      }
      i64_vals[inst] = std::move(out);
    } else if (inst->opcode() == HloOpcode::kBroadcast) {
      const int64_t n = inst->shape().dimensions(0);
      std::vector<std::vector<uint32_t>> out(num_partitions);
      for (int64_t p = 0; p < num_partitions; ++p) {
        out[p].assign(n, f32_vals.at(inst->operand(0))[p][0]);
      }
      f32_vals[inst] = std::move(out);
    } else if (inst->opcode() == HloOpcode::kAllGather) {
      const int64_t out_n = inst->shape().dimensions(0);
      const auto& in = f32_vals.at(inst->operand(0));
      std::vector<std::vector<uint32_t>> out(
          num_partitions, std::vector<uint32_t>(out_n, uint32_t{0}));
      for (const ReplicaGroup& rg : inst->replica_groups()) {
        std::vector<uint32_t> gathered;
        for (int64_t id : rg.replica_ids()) {
          gathered.insert(gathered.end(), in[id].begin(), in[id].end());
        }
        for (int64_t id : rg.replica_ids()) {
          out[id] = gathered;
        }
      }
      f32_vals[inst] = std::move(out);
    } else if (inst->opcode() == HloOpcode::kCollectivePermute) {
      const int64_t out_n = inst->shape().dimensions(0);
      const auto& in = f32_vals.at(inst->operand(0));
      std::vector<std::vector<uint32_t>> out(
          num_partitions, std::vector<uint32_t>(out_n, uint32_t{0}));
      for (const auto& [src, dst] : inst->source_target_pairs()) {
        out[dst] = in[src];
      }
      f32_vals[inst] = std::move(out);
    } else if (inst->opcode() == HloOpcode::kPad) {
      const auto& in = f32_vals.at(inst->operand(0));
      const auto& pv = f32_vals.at(inst->operand(1));
      const int64_t low =
          inst->padding_config().dimensions(0).edge_padding_low();
      const int64_t high =
          inst->padding_config().dimensions(0).edge_padding_high();
      std::vector<std::vector<uint32_t>> out(num_partitions);
      for (int64_t p = 0; p < num_partitions; ++p) {
        out[p].assign(low, pv[p][0]);
        out[p].insert(out[p].end(), in[p].begin(), in[p].end());
        out[p].insert(out[p].end(), high, pv[p][0]);
      }
      f32_vals[inst] = std::move(out);
    } else if (inst->opcode() == HloOpcode::kDynamicSlice) {
      const int64_t sz = inst->dynamic_slice_sizes()[0];
      const int64_t in_n = inst->operand(0)->shape().dimensions(0);
      if (ty == F32) {
        const auto& in = f32_vals.at(inst->operand(0));
        std::vector<std::vector<uint32_t>> out(num_partitions);
        for (int64_t p = 0; p < num_partitions; ++p) {
          const int64_t s = std::clamp<int64_t>(
              get_i64_scalar(inst->operand(1), p), 0, in_n - sz);
          out[p].assign(in[p].begin() + s, in[p].begin() + s + sz);
        }
        f32_vals[inst] = std::move(out);
      } else {
        const auto& in = i64_vals.at(inst->operand(0));
        std::vector<std::vector<int64_t>> out(num_partitions);
        for (int64_t p = 0; p < num_partitions; ++p) {
          const int64_t s = std::clamp<int64_t>(
              get_i64_scalar(inst->operand(1), p), 0, in_n - sz);
          out[p].assign(in[p].begin() + s, in[p].begin() + s + sz);
        }
        i64_vals[inst] = std::move(out);
      }
    } else if (inst->opcode() == HloOpcode::kConcatenate) {
      std::vector<std::vector<uint32_t>> out(num_partitions);
      for (const HloInstruction* op : inst->operands()) {
        const auto& in = f32_vals.at(op);
        for (int64_t p = 0; p < num_partitions; ++p) {
          out[p].insert(out[p].end(), in[p].begin(), in[p].end());
        }
      }
      f32_vals[inst] = std::move(out);
    } else if (inst->opcode() == HloOpcode::kCompare) {
      std::vector<std::vector<int64_t>> out(num_partitions);
      for (int64_t p = 0; p < num_partitions; ++p) {
        out[p] = {get_i64_scalar(inst->operand(0), p) ==
                          get_i64_scalar(inst->operand(1), p)
                      ? 1
                      : 0};
      }
      i64_vals[inst] = std::move(out);
    } else if (inst->opcode() == HloOpcode::kSelect) {
      std::vector<std::vector<uint32_t>> out(num_partitions);
      for (int64_t p = 0; p < num_partitions; ++p) {
        out[p] = get_i64_scalar(inst->operand(0), p) != 0
                     ? f32_vals.at(inst->operand(1))[p]
                     : f32_vals.at(inst->operand(2))[p];
      }
      f32_vals[inst] = std::move(out);
    } else {
      LOG(FATAL) << "Unhandled opcode in test evaluator: " << inst->ToString();
    }
  }
  return f32_vals.at(comp->root_instruction());
}

std::vector<std::vector<uint32_t>> MakeTestInputs(int64_t num_partitions,
                                                  int64_t shard_size) {
  // Includes -0.0f, min subnormal, sNaN, and distinct finite values.
  const uint32_t kSpecials[] = {0x80000000u, 0x00000001u, 0x7f800001u,
                                0x3f800000u};
  std::vector<std::vector<uint32_t>> inputs(num_partitions);
  for (int64_t p = 0; p < num_partitions; ++p) {
    inputs[p].resize(shard_size);
    for (int64_t i = 0; i < shard_size; ++i) {
      inputs[p][i] =
          (i == 0 && p < 4)
              ? kSpecials[p]
              : std::bit_cast<uint32_t>(static_cast<float>((p + 1) * 10 + i));
    }
  }
  return inputs;
}

TEST_F(AllGatherPadDsSimplifierTest,
       PreservesSignedZeroPadBitsAndLayoutElementSize) {
  constexpr absl::string_view kF32NegZero = R"(
    HloModule f32_neg_zero
    ENTRY entry {
      param = f32[2]{0} parameter(0)
      neg_zero = f32[] constant(-0.0)
      ag = f32[8]{0} all-gather(param), channel_id=1,
        replica_groups={{0,1,2,3}}, dimensions={0}, use_global_device_ids=true
      pad = f32[96]{0} pad(ag, neg_zero), padding=88_0
      offsets = s32[4]{0} constant({0, 24, 48, 72})
      pid = u32[] partition-id()
      idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
      idx = s32[] reshape(idx1)
      ROOT ds = f32[24]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
    }
  )";
  const auto inputs = MakeTestInputs(4, 2);
  TF_ASSERT_OK_AND_ASSIGN(auto before,
                          ParseAndReturnVerifiedModule(kF32NegZero, 1, 4));
  const auto expected = Eval1DF32PartitionsBitwise(*before, inputs);
  TF_ASSERT_OK_AND_ASSIGN(auto after, RunPass(kF32NegZero, 1, 4, true));
  EXPECT_EQ(Eval1DF32PartitionsBitwise(*after, inputs), expected);
  EXPECT_EQ(expected[0][0], 0x80000000u);
  EXPECT_EQ(expected[3][0], 0x80000000u);

  constexpr absl::string_view kC64NegImagZero = R"(
    HloModule c64_neg_zero
    ENTRY entry {
      param = c64[2]{0} parameter(0)
      neg_zero = c64[] constant((0.0, -0.0))
      ag = c64[8]{0} all-gather(param), channel_id=1,
        replica_groups={{0,1,2,3}}, dimensions={0}, use_global_device_ids=true
      pad = c64[96]{0} pad(ag, neg_zero), padding=88_0
      offsets = s32[4]{0} constant({0, 24, 48, 72})
      pid = u32[] partition-id()
      idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
      idx = s32[] reshape(idx1)
      ROOT ds = c64[24]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto c64_mod, RunPass(kC64NegImagZero, 1, 4, true));
  const HloInstruction* root = c64_mod->entry_computation()->root_instruction();
  const auto check_c64_neg_imag = [](const HloInstruction* bcast) {
    ASSERT_EQ(bcast->opcode(), HloOpcode::kBroadcast);
    const std::complex<float> v =
        bcast->operand(0)->literal().GetFirstElement<std::complex<float>>();
    EXPECT_EQ(std::bit_cast<uint32_t>(v.real()), 0x00000000u);
    EXPECT_EQ(std::bit_cast<uint32_t>(v.imag()), 0x80000000u);
  };
  check_c64_neg_imag(root->operand(1)->operand(0));
  check_c64_neg_imag(root->operand(2));

  constexpr absl::string_view kSubByteLayout = R"(
    HloModule s4_elem_bits
    ENTRY entry {
      param = s4[2]{0:E(4)} parameter(0)
      zero = s4[]{:E(4)} constant(0)
      ag = s4[8]{0:E(4)} all-gather(param), channel_id=1,
        replica_groups={{0,1,2,3}}, dimensions={0}, use_global_device_ids=true
      pad = s4[96]{0:E(4)} pad(ag, zero), padding=88_0
      offsets = s32[4]{0} constant({0, 24, 48, 72})
      pid = u32[] partition-id()
      idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
      idx = s32[] reshape(idx1)
      ROOT ds = s4[24]{0:E(4)} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto s4_mod, RunPass(kSubByteLayout, 1, 4, true));
  const HloInstruction* s4_root =
      s4_mod->entry_computation()->root_instruction();
  EXPECT_EQ(
      s4_root->operand(1)->operand(0)->shape().layout().element_size_in_bits(),
      4);
  EXPECT_EQ(s4_root->operand(2)->shape().layout().element_size_in_bits(), 4);
}

TEST_F(AllGatherPadDsSimplifierTest, ClampsPastEndAndNegativeTargetOffsets) {
  constexpr absl::string_view kPastEnd = R"(
    HloModule past_end
    ENTRY entry {
      param = f32[2]{0} parameter(0)
      zero = f32[] constant(0)
      ag = f32[8]{0} all-gather(param), channel_id=1,
        replica_groups={{0,1,2,3}}, dimensions={0}, use_global_device_ids=true
      pad = f32[96]{0} pad(ag, zero), padding=88_0
      offsets = s32[4]{0} constant({0, 24, 48, 80})
      pid = u32[] partition-id()
      idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
      idx = s32[] reshape(idx1)
      ROOT ds = f32[24]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
    }
  )";
  const auto inputs = MakeTestInputs(4, 2);
  TF_ASSERT_OK_AND_ASSIGN(auto before_hi,
                          ParseAndReturnVerifiedModule(kPastEnd, 1, 4));
  const auto exp_hi = Eval1DF32PartitionsBitwise(*before_hi, inputs);
  TF_ASSERT_OK_AND_ASSIGN(auto after_hi, RunPass(kPastEnd, 1, 4, true));
  EXPECT_EQ(Eval1DF32PartitionsBitwise(*after_hi, inputs), exp_hi);
  EXPECT_EQ(after_hi->entry_computation()
                ->root_instruction()
                ->operand(1)
                ->operand(0)
                ->shape()
                .dimensions(0),
            16);

  constexpr absl::string_view kNegative = R"(
    HloModule neg_start
    ENTRY entry {
      param = f32[2]{0} parameter(0)
      zero = f32[] constant(0)
      ag = f32[8]{0} all-gather(param), channel_id=1,
        replica_groups={{0,1,2,3}}, dimensions={0}, use_global_device_ids=true
      pad = f32[96]{0} pad(ag, zero), padding=0_88
      offsets = s32[4]{0} constant({-4, 24, 48, 72})
      pid = u32[] partition-id()
      idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
      idx = s32[] reshape(idx1)
      ROOT ds = f32[24]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto before_lo,
                          ParseAndReturnVerifiedModule(kNegative, 1, 4));
  const auto exp_lo = Eval1DF32PartitionsBitwise(*before_lo, inputs);
  TF_ASSERT_OK_AND_ASSIGN(auto after_lo, RunPass(kNegative, 1, 4, true));
  EXPECT_EQ(Eval1DF32PartitionsBitwise(*after_lo, inputs), exp_lo);
  const HloInstruction* concat_lo =
      after_lo->entry_computation()->root_instruction()->operand(1);
  EXPECT_EQ(concat_lo->operand_count(), 5);
  EXPECT_EQ(concat_lo->operand(4)->shape().dimensions(0), 16);
}

TEST_F(AllGatherPadDsSimplifierTest, RejectsOverlappingNonTargetSliceWindows) {
  const absl::string_view kCases[] = {
      // High-pad: p1 [7, 31) overlaps ag [0, 8).
      R"(
        HloModule overlap_hi
        ENTRY entry {
          param = f32[2]{0} parameter(0)
          zero = f32[] constant(0)
          ag = f32[8]{0} all-gather(param), channel_id=1,
            replica_groups={{0,1,2,3}}, dimensions={0},
            use_global_device_ids=true
          pad = f32[96]{0} pad(ag, zero), padding=0_88
          offsets = s32[4]{0} constant({0, 7, 48, 72})
          pid = u32[] partition-id()
          idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
          idx = s32[] reshape(idx1)
          ROOT ds = f32[24]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
        }
      )",
      // Low-pad: p2 [70, 94) overlaps ag [88, 96).
      R"(
        HloModule overlap_lo
        ENTRY entry {
          param = f32[2]{0} parameter(0)
          zero = f32[] constant(0)
          ag = f32[8]{0} all-gather(param), channel_id=1,
            replica_groups={{0,1,2,3}}, dimensions={0},
            use_global_device_ids=true
          pad = f32[96]{0} pad(ag, zero), padding=88_0
          offsets = s32[4]{0} constant({0, 24, 70, 72})
          pid = u32[] partition-id()
          idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
          idx = s32[] reshape(idx1)
          ROOT ds = f32[24]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
        }
      )",
      // Low-pad: both p2 (80) and p3 (85) clamp to [72, 96) covering ag.
      R"(
        HloModule dual_clamped
        ENTRY entry {
          param = f32[2]{0} parameter(0)
          zero = f32[] constant(0)
          ag = f32[8]{0} all-gather(param), channel_id=1,
            replica_groups={{0,1,2,3}}, dimensions={0},
            use_global_device_ids=true
          pad = f32[96]{0} pad(ag, zero), padding=88_0
          offsets = s32[4]{0} constant({0, 24, 80, 85})
          pid = u32[] partition-id()
          idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
          idx = s32[] reshape(idx1)
          ROOT ds = f32[24]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
        }
      )",
      // Low-pad: p2 [65, 89) partially overlaps ag [88, 96) by 1 element.
      R"(
        HloModule partial_overlap
        ENTRY entry {
          param = f32[2]{0} parameter(0)
          zero = f32[] constant(0)
          ag = f32[8]{0} all-gather(param), channel_id=1,
            replica_groups={{0,1,2,3}}, dimensions={0},
            use_global_device_ids=true
          pad = f32[96]{0} pad(ag, zero), padding=88_0
          offsets = s32[4]{0} constant({0, 24, 65, 72})
          pid = u32[] partition-id()
          idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
          idx = s32[] reshape(idx1)
          ROOT ds = f32[24]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
        }
      )",
  };
  for (absl::string_view hlo : kCases) {
    TF_ASSERT_OK_AND_ASSIGN(auto before,
                            ParseAndReturnVerifiedModule(hlo, 1, 4));
    const int64_t before_count =
        before->entry_computation()->instruction_count();
    TF_ASSERT_OK_AND_ASSIGN(bool changed,
                            AllGatherPadDsSimplifier().Run(before.get()));
    EXPECT_FALSE(changed);
    EXPECT_EQ(before->entry_computation()->instruction_count(), before_count);
  }
}

TEST_F(AllGatherPadDsSimplifierTest,
       MultiGroupUpstreamPermuteAndZeroDeadInstructions) {
  const auto inputs = MakeTestInputs(4, 2);

  // Compatible groups with upstream kCollectivePermute: must keep cp_in intact.
  constexpr absl::string_view kUpstreamCpCompatible = R"(
    HloModule upstream_cp_ok
    ENTRY entry {
      param = f32[2]{0} parameter(0)
      cp_in = f32[2]{0} collective-permute(param), channel_id=1,
        source_target_pairs={{0,3},{1,2},{2,1},{3,0}}
      zero = f32[] constant(0)
      ag = f32[4]{0} all-gather(cp_in), channel_id=2,
        replica_groups={{0,1},{2,3}}, dimensions={0}, use_global_device_ids=true
      pad = f32[20]{0} pad(ag, zero), padding=16_0
      offsets = s32[4]{0} constant({0, 10, 0, 10})
      pid = u32[] partition-id()
      idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
      idx = s32[] reshape(idx1)
      ROOT ds = f32[10]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={10}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto before_ok, ParseAndReturnVerifiedModule(
                                              kUpstreamCpCompatible, 1, 4));
  const auto exp_ok = Eval1DF32PartitionsBitwise(*before_ok, inputs);
  TF_ASSERT_OK_AND_ASSIGN(auto after_ok,
                          RunPass(kUpstreamCpCompatible, 1, 4, true));
  EXPECT_EQ(Eval1DF32PartitionsBitwise(*after_ok, inputs), exp_ok);

  // Incompatible reversed second group with upstream kCollectivePermute: reject
  // cleanly without mutating cp_in or leaking dead instructions.
  constexpr absl::string_view kUpstreamCpReversed = R"(
    HloModule upstream_cp_rev
    ENTRY entry {
      param = f32[2]{0} parameter(0)
      cp_in = f32[2]{0} collective-permute(param), channel_id=1,
        source_target_pairs={{0,3},{1,2},{2,1},{3,0}}
      zero = f32[] constant(0)
      ag = f32[4]{0} all-gather(cp_in), channel_id=2,
        replica_groups={{0,1},{3,2}}, dimensions={0}, use_global_device_ids=true
      pad = f32[20]{0} pad(ag, zero), padding=16_0
      offsets = s32[4]{0} constant({0, 10, 0, 10})
      pid = u32[] partition-id()
      idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
      idx = s32[] reshape(idx1)
      ROOT ds = f32[10]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={10}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(
      auto rev_mod, ParseAndReturnVerifiedModule(kUpstreamCpReversed, 1, 4));
  const int64_t rev_count = rev_mod->entry_computation()->instruction_count();
  TF_ASSERT_OK_AND_ASSIGN(bool rev_changed,
                          AllGatherPadDsSimplifier().Run(rev_mod.get()));
  EXPECT_FALSE(rev_changed);
  EXPECT_EQ(rev_mod->entry_computation()->instruction_count(), rev_count);

  // Distinct pad widths across groups after clamping (low_pad=18, pad=22,
  // ds=10, max_start=12: group 0 starts at 11 -> pad 7, group 1 at 12 -> pad
  // 6).
  constexpr absl::string_view kPadWidthMismatch = R"(
    HloModule pad_width_mismatch
    ENTRY entry {
      param = f32[2]{0} parameter(0)
      zero = f32[] constant(0)
      ag = f32[4]{0} all-gather(param), channel_id=2,
        replica_groups={{0,1},{2,3}}, dimensions={0}, use_global_device_ids=true
      pad = f32[22]{0} pad(ag, zero), padding=18_0
      offsets = s32[4]{0} constant({0, 11, 0, 12})
      pid = u32[] partition-id()
      idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
      idx = s32[] reshape(idx1)
      ROOT ds = f32[10]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={10}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(
      auto mm_mod, ParseAndReturnVerifiedModule(kPadWidthMismatch, 1, 4));
  const int64_t mm_count = mm_mod->entry_computation()->instruction_count();
  TF_ASSERT_OK_AND_ASSIGN(bool mm_changed,
                          AllGatherPadDsSimplifier().Run(mm_mod.get()));
  EXPECT_FALSE(mm_changed);
  EXPECT_EQ(mm_mod->entry_computation()->instruction_count(), mm_count);

  // Asymmetric raw starts {0, 11, 0, 12} that both clamp to 10 (pad=20, ds=10)
  // yield identical width-6 leading pad and succeed bitwise.
  constexpr absl::string_view kClampedEqual = R"(
    HloModule clamped_equal
    ENTRY entry {
      param = f32[2]{0} parameter(0)
      zero = f32[] constant(0)
      ag = f32[4]{0} all-gather(param), channel_id=2,
        replica_groups={{0,1},{2,3}}, dimensions={0}, use_global_device_ids=true
      pad = f32[20]{0} pad(ag, zero), padding=16_0
      offsets = s32[4]{0} constant({0, 11, 0, 12})
      pid = u32[] partition-id()
      idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
      idx = s32[] reshape(idx1)
      ROOT ds = f32[10]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={10}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto before_eq,
                          ParseAndReturnVerifiedModule(kClampedEqual, 1, 4));
  const auto exp_eq = Eval1DF32PartitionsBitwise(*before_eq, inputs);
  TF_ASSERT_OK_AND_ASSIGN(auto after_eq, RunPass(kClampedEqual, 1, 4, true));
  EXPECT_EQ(Eval1DF32PartitionsBitwise(*after_eq, inputs), exp_eq);

  // A 2x4 multi-group rewrite emits exactly 3 live kCollectivePermutes (0
  // dead).
  constexpr absl::string_view kTwoByFour = R"(
    HloModule two_by_four
    ENTRY entry {
      param = f32[2]{0} parameter(0)
      zero = f32[] constant(0)
      ag = f32[8]{0} all-gather(param), channel_id=4,
        replica_groups=[2,4]<=[8], dimensions={0}, use_global_device_ids=true
      pad = f32[96]{0} pad(ag, zero), padding=0_88
      offsets = s32[8]{0} constant({0, 24, 48, 72, 0, 24, 48, 72})
      pid = u32[] partition-id()
      idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
      idx = s32[] reshape(idx1)
      ROOT ds = f32[24]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto mod_2x4, RunPass(kTwoByFour, 1, 8, true));
  int64_t cp_count = 0;
  for (const HloInstruction* inst :
       mod_2x4->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kCollectivePermute) {
      ++cp_count;
      EXPECT_GT(inst->user_count(), 0);
    }
  }
  EXPECT_EQ(cp_count, 3);
}

TEST_F(AllGatherPadDsSimplifierTest,
       ValidatesIndexMultiplyWrapZeroBitcastAndIotaBounds) {
  // Zero multiplier, S32 multiply wrap (1073741824 * 4 == 0 mod 2^32), U32
  // multiply wrap, U32->S32 bitcast, and short iota[2] on 4 partitions reject.
  const absl::string_view kRejected[] = {
      R"(
        HloModule zero_mult
        ENTRY entry {
          param = f32[2]{0} parameter(0)
          zero = f32[] constant(0)
          ag = f32[8]{0} all-gather(param), channel_id=1,
            replica_groups={{0,1,2,3}}, dimensions={0},
            use_global_device_ids=true
          pad = f32[96]{0} pad(ag, zero), padding=0_88
          offsets = s32[4]{0} constant({0, 1, 2, 3})
          pid = u32[] partition-id()
          idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
          z = s32[1]{0} constant({0})
          mul = s32[1]{0} multiply(idx1, z)
          idx = s32[] reshape(mul)
          ROOT ds = f32[24]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
        }
      )",
      R"(
        HloModule s32_mul_wrap
        ENTRY entry {
          param = f32[2]{0} parameter(0)
          zero = f32[] constant(0)
          ag = f32[8]{0} all-gather(param), channel_id=1,
            replica_groups={{0,1,2,3}}, dimensions={0},
            use_global_device_ids=true
          pad = f32[96]{0} pad(ag, zero), padding=88_0
          offsets = s32[4]{0} constant({0, 6, 12, 1073741824})
          pid = u32[] partition-id()
          idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
          four = s32[1]{0} constant({4})
          mul = s32[1]{0} multiply(idx1, four)
          idx = s32[] reshape(mul)
          ROOT ds = f32[24]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
        }
      )",
      R"(
        HloModule u32_mul_wrap
        ENTRY entry {
          param = f32[2]{0} parameter(0)
          zero = f32[] constant(0)
          ag = f32[8]{0} all-gather(param), channel_id=1,
            replica_groups={{0,1,2,3}}, dimensions={0},
            use_global_device_ids=true
          pad = f32[96]{0} pad(ag, zero), padding=88_0
          offsets = u32[4]{0} constant({0, 6, 12, 1073741824})
          pid = u32[] partition-id()
          idx1 = u32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
          four = u32[1]{0} constant({4})
          mul = u32[1]{0} multiply(idx1, four)
          idx = u32[] reshape(mul)
          ROOT ds = f32[24]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
        }
      )",
      R"(
        HloModule u32_to_s32_bitcast
        ENTRY entry {
          param = f32[2]{0} parameter(0)
          zero = f32[] constant(0)
          ag = f32[8]{0} all-gather(param), channel_id=1,
            replica_groups={{0,1,2,3}}, dimensions={0},
            use_global_device_ids=true
          pad = f32[96]{0} pad(ag, zero), padding=88_0
          offsets = u32[4]{0} constant({0, 24, 48, 2147483648})
          pid = u32[] partition-id()
          idx1 = u32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
          bc = s32[1]{0} bitcast(idx1)
          idx = s32[] reshape(bc)
          ROOT ds = f32[24]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
        }
      )",
      R"(
        HloModule short_iota
        ENTRY entry {
          param = f32[2]{0} parameter(0)
          zero = f32[] constant(0)
          ag = f32[8]{0} all-gather(param), channel_id=1,
            replica_groups={{0,1,2,3}}, dimensions={0},
            use_global_device_ids=true
          pad = f32[96]{0} pad(ag, zero), padding=88_0
          tbl = s32[2]{0} iota(), iota_dimension=0
          pid = u32[] partition-id()
          idx1 = s32[1]{0} dynamic-slice(tbl, pid), dynamic_slice_sizes={1}
          step = s32[1]{0} constant({24})
          mul = s32[1]{0} multiply(idx1, step)
          idx = s32[] reshape(mul)
          ROOT ds = f32[24]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
        }
      )",
  };
  for (absl::string_view hlo : kRejected) {
    TF_ASSERT_OK_AND_ASSIGN(auto m, RunPass(hlo, 1, 4, false));
  }

  // Full-length iota[4] * 24 succeeds bitwise.
  constexpr absl::string_view kFullIota = R"(
    HloModule full_iota
    ENTRY entry {
      param = f32[2]{0} parameter(0)
      zero = f32[] constant(0)
      ag = f32[8]{0} all-gather(param), channel_id=1,
        replica_groups={{0,1,2,3}}, dimensions={0}, use_global_device_ids=true
      pad = f32[96]{0} pad(ag, zero), padding=88_0
      tbl = s32[4]{0} iota(), iota_dimension=0
      pid = u32[] partition-id()
      idx1 = s32[1]{0} dynamic-slice(tbl, pid), dynamic_slice_sizes={1}
      step = s32[1]{0} constant({24})
      mul = s32[1]{0} multiply(idx1, step)
      idx = s32[] reshape(mul)
      ROOT ds = f32[24]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
    }
  )";
  const auto inputs = MakeTestInputs(4, 2);
  TF_ASSERT_OK_AND_ASSIGN(auto before,
                          ParseAndReturnVerifiedModule(kFullIota, 1, 4));
  const auto expected = Eval1DF32PartitionsBitwise(*before, inputs);
  TF_ASSERT_OK_AND_ASSIGN(auto after, RunPass(kFullIota, 1, 4, true));
  EXPECT_EQ(Eval1DF32PartitionsBitwise(*after, inputs), expected);
}

TEST_F(AllGatherPadDsSimplifierTest,
       RejectsNonSplitSliceDynamicShapeControlDepsAndLayoutConstraints) {
  const absl::string_view kRejected[] = {
      // Non-split dim 0 sliced from 4 down to 2.
      R"(
        HloModule nonsplit_slice
        ENTRY entry {
          param = f64[4,2,40]{2,1,0} parameter(0)
          zero = f64[] constant(0)
          z = s32[] constant(0)
          ag = f64[4,8,40]{2,1,0} all-gather(param), channel_id=1,
            replica_groups={{0,1,2,3}}, dimensions={1},
            use_global_device_ids=true
          pad = f64[4,96,40]{2,1,0} pad(ag, zero), padding=0_0x88_0x0_0
          offsets = s32[4]{0} constant({0, 24, 48, 72})
          pid = u32[] partition-id()
          idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
          idx = s32[] reshape(idx1)
          ROOT ds = f64[2,24,40]{2,1,0} dynamic-slice(pad, z, idx, z),
            dynamic_slice_sizes={2,24,40}
        }
      )",
      // Bounded dynamic dimension on non-split axis.
      R"(
        HloModule bounded_dynamic
        ENTRY entry {
          param = f32[<=4,2]{1,0} parameter(0)
          zero = f32[] constant(0)
          z = s32[] constant(0)
          ag = f32[<=4,8]{1,0} all-gather(param), channel_id=1,
            replica_groups={{0,1,2,3}}, dimensions={1},
            use_global_device_ids=true
          pad = f32[<=4,96]{1,0} pad(ag, zero), padding=0_0x88_0
          offsets = s32[4]{0} constant({0, 24, 48, 72})
          pid = u32[] partition-id()
          idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
          idx = s32[] reshape(idx1)
          ROOT ds = f32[<=4,24]{1,0} dynamic-slice(pad, z, idx),
            dynamic_slice_sizes={4,24}
        }
      )",
      // Control dependency on all-gather.
      R"(
        HloModule control_dep
        ENTRY entry {
          param = f32[2]{0} parameter(0)
          tok = token[] after-all()
          zero = f32[] constant(0)
          ag = f32[8]{0} all-gather(param), channel_id=1,
            replica_groups={{0,1,2,3}}, dimensions={0},
            use_global_device_ids=true, control-predecessors={tok}
          pad = f32[96]{0} pad(ag, zero), padding=88_0
          offsets = s32[4]{0} constant({0, 24, 48, 72})
          pid = u32[] partition-id()
          idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
          idx = s32[] reshape(idx1)
          ROOT ds = f32[24]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
        }
      )",
      // Layout-constrained all-gather.
      R"(
        HloModule constrain_layout
        ENTRY entry {
          param = f32[2]{0} parameter(0)
          zero = f32[] constant(0)
          ag = f32[8]{0} all-gather(param), channel_id=1,
            replica_groups={{0,1,2,3}}, dimensions={0},
            use_global_device_ids=true, constrain_layout=true
          pad = f32[96]{0} pad(ag, zero), padding=88_0
          offsets = s32[4]{0} constant({0, 24, 48, 72})
          pid = u32[] partition-id()
          idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
          idx = s32[] reshape(idx1)
          ROOT ds = f32[24]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
        }
      )",
      // Memory space mismatch between ag operand and ds result.
      R"(
        HloModule mem_space_mismatch
        ENTRY entry {
          param = f32[2]{0:S(1)} parameter(0)
          zero = f32[] constant(0)
          ag = f32[8]{0} all-gather(param), channel_id=1,
            replica_groups={{0,1,2,3}}, dimensions={0},
            use_global_device_ids=true
          pad = f32[96]{0} pad(ag, zero), padding=88_0
          offsets = s32[4]{0} constant({0, 24, 48, 72})
          pid = u32[] partition-id()
          idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
          idx = s32[] reshape(idx1)
          ROOT ds = f32[24]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={24}
        }
      )",
  };
  for (absl::string_view hlo : kRejected) {
    TF_ASSERT_OK_AND_ASSIGN(auto m, RunPass(hlo, 1, 4, false));
  }
}

TEST_F(AllGatherPadDsSimplifierTest,
       HandlesPartialParticipationAndRejectsMalformedGroups) {
  const auto inputs = MakeTestInputs(4, 2);

  // Partial participation {{0, 1}} and non-contiguous {{2, 3}} on 4 partitions:
  // selection_list must have full length 4 and evaluate bitwise identically.
  constexpr absl::string_view kSubset01 = R"(
    HloModule subset_01
    ENTRY entry {
      param = f32[2]{0} parameter(0)
      zero = f32[] constant(0)
      ag = f32[4]{0} all-gather(param), channel_id=1,
        replica_groups={{0,1}}, dimensions={0}, use_global_device_ids=true
      pad = f32[20]{0} pad(ag, zero), padding=16_0
      offsets = s32[4]{0} constant({0, 10, 10, 10})
      pid = u32[] partition-id()
      idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
      idx = s32[] reshape(idx1)
      ROOT ds = f32[10]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={10}
    }
  )";
  HloModuleConfig cfg = GetModuleConfigForTest(1, 4);
  TF_ASSERT_OK_AND_ASSIGN(auto b01,
                          ParseAndReturnUnverifiedModule(kSubset01, cfg));
  const auto exp01 = Eval1DF32PartitionsBitwise(*b01, inputs);
  TF_ASSERT_OK_AND_ASSIGN(bool ch01, AllGatherPadDsSimplifier().Run(b01.get()));
  EXPECT_TRUE(ch01);
  EXPECT_EQ(Eval1DF32PartitionsBitwise(*b01, inputs), exp01);
  EXPECT_EQ(b01->entry_computation()
                ->root_instruction()
                ->operand(0)
                ->operand(0)
                ->operand(0)
                ->operand(0)
                ->literal(),
            LiteralUtil::CreateR1<int64_t>({0, 1, 0, 0}));

  constexpr absl::string_view kSubset23 = R"(
    HloModule subset_23
    ENTRY entry {
      param = f32[2]{0} parameter(0)
      zero = f32[] constant(0)
      ag = f32[4]{0} all-gather(param), channel_id=1,
        replica_groups={{2,3}}, dimensions={0}, use_global_device_ids=true
      pad = f32[20]{0} pad(ag, zero), padding=16_0
      offsets = s32[4]{0} constant({0, 0, 0, 10})
      pid = u32[] partition-id()
      idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
      idx = s32[] reshape(idx1)
      ROOT ds = f32[10]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={10}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto b23,
                          ParseAndReturnUnverifiedModule(kSubset23, cfg));
  const auto exp23 = Eval1DF32PartitionsBitwise(*b23, inputs);
  TF_ASSERT_OK_AND_ASSIGN(bool ch23, AllGatherPadDsSimplifier().Run(b23.get()));
  EXPECT_TRUE(ch23);
  EXPECT_EQ(Eval1DF32PartitionsBitwise(*b23, inputs), exp23);

  // Partial participation with -0.0 pad must reject because non-participants
  // observe +0.0 from AllGather inside the [16, 20) slice window.
  const absl::string_view kUnverifiedRejected[] = {
      R"(
        HloModule subset_neg_zero
        ENTRY entry {
          param = f32[2]{0} parameter(0)
          neg_zero = f32[] constant(-0.0)
          ag = f32[4]{0} all-gather(param), channel_id=1,
            replica_groups={{0,1}}, dimensions={0}, use_global_device_ids=true
          pad = f32[20]{0} pad(ag, neg_zero), padding=16_0
          offsets = s32[4]{0} constant({0, 10, 10, 10})
          pid = u32[] partition-id()
          idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
          idx = s32[] reshape(idx1)
          ROOT ds = f32[10]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={10}
        }
      )",
      R"(
        HloModule dup_pid
        ENTRY entry {
          param = f32[2]{0} parameter(0)
          zero = f32[] constant(0)
          ag = f32[4]{0} all-gather(param), channel_id=1,
            replica_groups={{0,1},{1,2}}, dimensions={0},
            use_global_device_ids=true
          pad = f32[20]{0} pad(ag, zero), padding=16_0
          offsets = s32[4]{0} constant({0, 5, 10, 10})
          pid = u32[] partition-id()
          idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
          idx = s32[] reshape(idx1)
          ROOT ds = f32[10]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={10}
        }
      )",
      R"(
        HloModule oob_pid
        ENTRY entry {
          param = f32[2]{0} parameter(0)
          zero = f32[] constant(0)
          ag = f32[4]{0} all-gather(param), channel_id=1,
            replica_groups={{0,7}}, dimensions={0}, use_global_device_ids=true
          pad = f32[20]{0} pad(ag, zero), padding=16_0
          offsets = s32[8]{0} constant({0, 0, 0, 0, 0, 0, 0, 10})
          pid = u32[] partition-id()
          idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
          idx = s32[] reshape(idx1)
          ROOT ds = f32[10]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={10}
        }
      )",
      R"(
        HloModule ragged_groups
        ENTRY entry {
          param = f32[2]{0} parameter(0)
          zero = f32[] constant(0)
          ag = f32[4]{0} all-gather(param), channel_id=1,
            replica_groups={{0,1},{2}}, dimensions={0},
            use_global_device_ids=true
          pad = f32[20]{0} pad(ag, zero), padding=16_0
          offsets = s32[4]{0} constant({0, 10, 10, 10})
          pid = u32[] partition-id()
          idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
          idx = s32[] reshape(idx1)
          ROOT ds = f32[10]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={10}
        }
      )",
      R"(
        HloModule empty_group
        ENTRY entry {
          param = f32[2]{0} parameter(0)
          zero = f32[] constant(0)
          ag = f32[4]{0} all-gather(param), channel_id=1,
            replica_groups={{}}, dimensions={0}, use_global_device_ids=true
          pad = f32[20]{0} pad(ag, zero), padding=16_0
          offsets = s32[4]{0} constant({0, 10, 10, 10})
          pid = u32[] partition-id()
          idx1 = s32[1]{0} dynamic-slice(offsets, pid), dynamic_slice_sizes={1}
          idx = s32[] reshape(idx1)
          ROOT ds = f32[10]{0} dynamic-slice(pad, idx), dynamic_slice_sizes={10}
        }
      )",
  };
  for (absl::string_view hlo : kUnverifiedRejected) {
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnUnverifiedModule(hlo, cfg));
    TF_ASSERT_OK_AND_ASSIGN(bool ch, AllGatherPadDsSimplifier().Run(m.get()));
    EXPECT_FALSE(ch);
  }

  // replica_count == 0 must also reject.
  HloModuleConfig zero_rep_cfg = GetModuleConfigForTest(1, 4);
  zero_rep_cfg.set_replica_count(0);
  TF_ASSERT_OK_AND_ASSIGN(
      auto zr, ParseAndReturnUnverifiedModule(kSubset01, zero_rep_cfg));
  TF_ASSERT_OK_AND_ASSIGN(bool zr_ch, AllGatherPadDsSimplifier().Run(zr.get()));
  EXPECT_FALSE(zr_ch);
}

TEST_F(ExtractValidPadSpecTest,
       RejectsNegativePaddingInvalidSplitDimOverflowAndBadPadValues) {
  // Negative edge padding on low or high side.
  for (absl::string_view hlo : {absl::string_view(R"(
        HloModule neg_low
        ENTRY entry {
          p = f32[8]{0} parameter(0)
          z = f32[] constant(0)
          ROOT pad = f32[20]{0} pad(p, z), padding=-2_14
        }
      )"),
                                absl::string_view(R"(
        HloModule neg_high
        ENTRY entry {
          p = f32[8]{0} parameter(0)
          z = f32[] constant(0)
          ROOT pad = f32[20]{0} pad(p, z), padding=14_-2
        }
      )")}) {
    TF_ASSERT_OK_AND_ASSIGN(auto m_neg, ParseAndReturnVerifiedModule(hlo));
    const auto* pad_neg =
        Cast<HloPadInstruction>(m_neg->entry_computation()->root_instruction());
    EXPECT_FALSE(ExtractValidPadSpec(*pad_neg, ShapeUtil::MakeShape(F32, {10}),
                                     ShapeUtil::MakeShape(F32, {8}), 0)
                     .has_value());
  }

  // Out-of-range split_dim and overflowed padding extent.
  HloComputation::Builder b(TestName());
  HloInstruction* p = b.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {8}), "p"));
  HloInstruction* z = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  PaddingConfig good_cfg;
  auto* gd = good_cfg.add_dimensions();
  gd->set_edge_padding_low(0);
  gd->set_edge_padding_high(12);
  gd->set_interior_padding(0);
  auto* good_pad =
      Cast<HloPadInstruction>(b.AddInstruction(HloInstruction::CreatePad(
          ShapeUtil::MakeShape(F32, {20}), p, z, good_cfg)));

  PaddingConfig ovf_cfg;
  auto* od = ovf_cfg.add_dimensions();
  od->set_edge_padding_low(0);
  od->set_edge_padding_high(std::numeric_limits<int64_t>::max());
  od->set_interior_padding(0);
  auto* ovf_pad =
      Cast<HloPadInstruction>(b.AddInstruction(HloInstruction::CreatePad(
          ShapeUtil::MakeShape(F32, {20}), p, z, ovf_cfg)));

  HloInstruction* vec_z = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({0.0f})));
  auto* nonscalar_pad =
      Cast<HloPadInstruction>(b.AddInstruction(HloInstruction::CreatePad(
          ShapeUtil::MakeShape(F32, {20}), p, vec_z, good_cfg)));

  HloInstruction* f16_z =
      b.AddInstruction(HloInstruction::CreateConstant(LiteralUtil::Zero(F16)));
  auto* mixed_pad =
      Cast<HloPadInstruction>(b.AddInstruction(HloInstruction::CreatePad(
          ShapeUtil::MakeShape(F32, {20}), p, f16_z, good_cfg)));

  auto mod = CreateNewUnverifiedModule();
  HloComputation* comp = mod->AddEntryComputation(b.Build());

  const Shape ds_s = ShapeUtil::MakeShape(F32, {10});
  const Shape ag_s = ShapeUtil::MakeShape(F32, {8});
  EXPECT_TRUE(ExtractValidPadSpec(*good_pad, ds_s, ag_s, 0).has_value());
  EXPECT_FALSE(ExtractValidPadSpec(*good_pad, ds_s, ag_s, -1).has_value());
  EXPECT_FALSE(ExtractValidPadSpec(*good_pad, ds_s, ag_s, 1).has_value());
  EXPECT_FALSE(ExtractValidPadSpec(*ovf_pad, ds_s, ag_s, 0).has_value());
  EXPECT_FALSE(ExtractValidPadSpec(*nonscalar_pad, ds_s, ag_s, 0).has_value());
  EXPECT_FALSE(ExtractValidPadSpec(*mixed_pad, ds_s, ag_s, 0).has_value());
  EXPECT_EQ(AddPredInstrBasedOnPartitionIdAndList(comp, {}), nullptr);
  EXPECT_EQ(AddPredInstrBasedOnPartitionIdAndList(nullptr, {1}), nullptr);
}

}  // namespace
}  // namespace xla
