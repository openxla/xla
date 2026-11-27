/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/all_reduce_folder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/ir/replica_group.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace matcher = xla::testing::opcode_matchers;
using ::absl_testing::IsOkAndHolds;
using ::testing::HasSubstr;

class AllReduceFolderTest : public HloHardwareIndependentTestBase {};

constexpr absl::string_view k2AllReduce = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add.2 = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups=$group_0, to_apply=sum
      ROOT ar1 = f32[8] all-reduce(ar0), replica_groups=$group_1, to_apply=sum
    }
  )";

size_t AllReduceCount(HloModule* module) {
  return absl::c_count_if(module->entry_computation()->instructions(),
                          HloPredicateIsOp<HloOpcode::kAllReduce>);
}

void ExpectOneAllReduce(HloModule* module,
                        absl::string_view target_replica_groups) {
  EXPECT_EQ(AllReduceCount(module), 1);
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, matcher::AllReduce(matcher::Parameter(0)));
  EXPECT_THAT(root->ToString(), HasSubstr(target_replica_groups));
}

TEST_F(AllReduceFolderTest, Simple) {
  ASSERT_OK_AND_ASSIGN(
      auto module, RunAndCheckHloRewrite(k2AllReduce, AllReduceFolder(), true,
                                         {{"$group_0", "{{0,1},{2,3}}"},
                                          {"$group_1", "{{0,2},{1,3}}"}}));
  ExpectOneAllReduce(module.get(), "replica_groups={{0,1,2,3}}");
}

// Same as Simple, but groups for the 2 all-reduce's are swapped.
TEST_F(AllReduceFolderTest, SimpleSwap) {
  ASSERT_OK_AND_ASSIGN(
      auto module, RunAndCheckHloRewrite(k2AllReduce, AllReduceFolder(), true,
                                         {{"$group_1", "{{0,1},{2,3}}"},
                                          {"$group_0", "{{0,2},{1,3}}"}}));
  ExpectOneAllReduce(module.get(), "replica_groups={{0,1,2,3}}");
}

TEST_F(AllReduceFolderTest, PreservesFrontendAttributes) {
  absl::string_view hlo_string = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add.2 = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups={{0,1},{2,3}},
        to_apply=sum, frontend_attributes={pipeline="stage0", shared="value"}
      ROOT ar1 = f32[8] all-reduce(ar0), replica_groups={{0,2},{1,3}},
        to_apply=sum, frontend_attributes={pipeline="stage1", shared="value"}
    }
  )";

  ASSERT_OK_AND_ASSIGN(
      auto module, RunAndCheckHloRewrite(hlo_string, AllReduceFolder(), true));
  ExpectOneAllReduce(module.get(), "replica_groups={{0,1,2,3}}");
  const auto& attributes =
      module->entry_computation()->root_instruction()->frontend_attributes();
  EXPECT_EQ(attributes.map().at("pipeline"), "stage0,stage1");
  EXPECT_EQ(attributes.map().at("shared"), "value");
}

TEST_F(AllReduceFolderTest, KeyedAndUnkeyedCollectivesAreFolded) {
  constexpr absl::string_view kHloModule = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add.2 = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups={{0,1},{2,3}},
        to_apply=sum, frontend_attributes={collective_group_key="g0"}
      ROOT ar1 = f32[8] all-reduce(ar0),
        replica_groups={{0,2},{1,3}}, to_apply=sum
    }
  )";

  ASSERT_OK_AND_ASSIGN(
      auto module, RunAndCheckHloRewrite(kHloModule, AllReduceFolder(), true));
  ExpectOneAllReduce(module.get(), "replica_groups={{0,1,2,3}}");
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->get_frontend_attribute("collective_group_key"), "g0");
}

TEST_F(AllReduceFolderTest, MatchingCollectiveGroupKeysAllowFolding) {
  constexpr absl::string_view kHloModule = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add.2 = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups={{0,1},{2,3}},
        to_apply=sum, frontend_attributes={collective_group_key="g0"}
      ROOT ar1 = f32[8] all-reduce(ar0), replica_groups={{0,2},{1,3}},
        to_apply=sum, frontend_attributes={collective_group_key="g0"}
    }
  )";

  ASSERT_OK_AND_ASSIGN(
      auto module, RunAndCheckHloRewrite(kHloModule, AllReduceFolder(), true));
  ExpectOneAllReduce(module.get(), "replica_groups={{0,1,2,3}}");
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->get_frontend_attribute("collective_group_key"), "g0");
}

TEST_F(AllReduceFolderTest, ConflictingCollectiveGroupKeysAreDropped) {
  constexpr absl::string_view kHloModule = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add.2 = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups={{0,1},{2,3}},
        to_apply=sum, frontend_attributes={collective_group_key="g0"}
      ROOT ar1 = f32[8] all-reduce(ar0), replica_groups={{0,2},{1,3}},
        to_apply=sum, frontend_attributes={collective_group_key="g1"}
    }
  )";

  ASSERT_OK_AND_ASSIGN(
      auto module, RunAndCheckHloRewrite(kHloModule, AllReduceFolder(), true));
  ExpectOneAllReduce(module.get(), "replica_groups={{0,1,2,3}}");
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->get_frontend_attribute("collective_group_key"), std::nullopt);
}

TEST_F(AllReduceFolderTest, BothEmptyReplicaGroups_NotTransformed) {
  ASSERT_OK(RunAndCheckHloRewrite(k2AllReduce, AllReduceFolder(), false,
                                  {{"$group_0", "{}"}, {"$group_1", "{}"}}));
}

TEST_F(AllReduceFolderTest, EmptyReplicaGroups_NotTransformed) {
  ASSERT_OK(RunAndCheckHloRewrite(
      k2AllReduce, AllReduceFolder(), false,
      {{"$group_0", "{}"}, {"$group_1", "{{0,2},{1,3}}"}}));
}

TEST_F(AllReduceFolderTest, MismatchOtherProperties0_NotTransformed) {
  absl::string_view hlo_string = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add.2 = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups={{0,1},{2,3}}, channel_id=1, to_apply=sum
      ROOT ar1 = f32[8] all-reduce(ar0), replica_groups={{0,2},{1,3}}, to_apply=sum
    }
    )";
  ASSERT_OK(RunAndCheckHloRewrite(hlo_string, AllReduceFolder(), false));
}

TEST_F(AllReduceFolderTest, MismatchOtherProperties1_NotTransformed) {
  absl::string_view hlo_string = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add.2 = f32[] add(a, b)
    }

    mul {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT mul = f32[] multiply(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups={{0,1},{2,3}}, to_apply=sum
      ROOT ar1 = f32[8] all-reduce(ar0), replica_groups={{0,2},{1,3}}, to_apply=mul
    }
    )";
  ASSERT_OK(RunAndCheckHloRewrite(hlo_string, AllReduceFolder(), false));
}

TEST_F(AllReduceFolderTest, NotFoldable_NotTransformed) {
  absl::string_view hlo_string = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add.2 = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups={{0,1},{2,3}}, to_apply=sum
      ROOT ar1 = f32[8] all-reduce(ar0), replica_groups={{0,1},{2,3}}, to_apply=sum
    }
    )";
  ASSERT_OK(RunAndCheckHloRewrite(hlo_string, AllReduceFolder(), false));
}

TEST_F(AllReduceFolderTest, Foldable0) {
  absl::string_view hlo_string = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add.2 = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups={{0,4},{1,5},{2,3},{6,7}}, to_apply=sum
      ROOT ar1 = f32[8] all-reduce(ar0), replica_groups={{0,5},{4,1},{2,7},{3,6}}, to_apply=sum
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module,
                       RunAndCheckHloRewrite(hlo_string, AllReduceFolder()));
  ExpectOneAllReduce(module.get(), "replica_groups={{0,1,4,5},{2,3,6,7}}");
}

// Verify that a chain of foldable all-reduce's folds in a single pass
// invocation.
TEST_F(AllReduceFolderTest, FoldableChain) {
  absl::string_view hlo_string = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add.2 = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups={{0,1},{2,3},{4,5},{6,7}}, to_apply=sum
      ar1 = f32[8] all-reduce(ar0), replica_groups={{0,2},{1,3},{4,6},{5,7}}, to_apply=sum
      ROOT ar2 = f32[8] all-reduce(ar1), replica_groups={{0,4},{1,5},{2,6},{3,7}}, to_apply=sum
    }
    )";
  ASSERT_OK_AND_ASSIGN(auto module,
                       RunAndCheckHloRewrite(hlo_string, AllReduceFolder()));
  ExpectOneAllReduce(module.get(), "replica_groups={{0,1,2,3,4,5,6,7}}");
}

TEST_F(AllReduceFolderTest,
       CrossReplicaAndPartitionMultiPartitionNonIdempotentNotFolded) {
  constexpr absl::string_view kNonIdempotent = R"(
    HloModule m, replica_count=4, num_partitions=2

    red {
      a = $type[] parameter(0)
      b = $type[] parameter(1)
      ROOT r = $type[] $op(a, b)
    }

    ENTRY main {
      p0 = $type[8] parameter(0)
      ar0 = $type[8] all-reduce(p0), channel_id=1,
        replica_groups={{0,1},{2,3}}, to_apply=red
      ROOT ar1 = $type[8] all-reduce(ar0), channel_id=2,
        replica_groups=$group_1, to_apply=red
    }
  )";
  ASSERT_OK(RunAndCheckHloRewrite(
      kNonIdempotent, AllReduceFolder(), false,
      {{"$type", "f32"}, {"$op", "add"}, {"$group_1", "{{0,2},{1,3}}"}}));
  ASSERT_OK(RunAndCheckHloRewrite(
      kNonIdempotent, AllReduceFolder(), false,
      {{"$type", "f32"}, {"$op", "add"}, {"$group_1", "{{0},{1},{2},{3}}"}}));
  ASSERT_OK(RunAndCheckHloRewrite(
      kNonIdempotent, AllReduceFolder(), false,
      {{"$type", "f32"}, {"$op", "multiply"}, {"$group_1", "{{0,2},{1,3}}"}}));
  ASSERT_OK(RunAndCheckHloRewrite(
      kNonIdempotent, AllReduceFolder(), false,
      {{"$type", "s32"}, {"$op", "xor"}, {"$group_1", "{{0,2},{1,3}}"}}));
}

TEST_F(AllReduceFolderTest,
       CrossReplicaAndPartitionIdempotentAndOtherModesFold) {
  constexpr absl::string_view kMultiPartition = R"(
    HloModule m, replica_count=4, num_partitions=$partitions

    red {
      a = $type[] parameter(0)
      b = $type[] parameter(1)
      ROOT r = $type[] $op(a, b)
    }

    ENTRY main {
      p0 = $type[8] parameter(0)
      ar0 = $type[8] all-reduce(p0), channel_id=1,
        replica_groups={{0,1},{2,3}}, to_apply=red
      ROOT ar1 = $type[8] all-reduce(ar0), channel_id=2,
        replica_groups={{0,2},{1,3}}, to_apply=red
    }
  )";
  for (const auto& [type, op] :
       std::vector<std::pair<absl::string_view, absl::string_view>>{
           {"f32", "maximum"},
           {"f32", "minimum"},
           {"s32", "and"},
           {"pred", "or"}}) {
    ASSERT_OK_AND_ASSIGN(
        auto module, RunAndCheckHloRewrite(
                         kMultiPartition, AllReduceFolder(), true,
                         {{"$partitions", "2"}, {"$type", type}, {"$op", op}}));
    ExpectOneAllReduce(module.get(), "replica_groups={{0,1,2,3}}");
  }

  // Single-partition CROSS_REPLICA_AND_PARTITION and multi-partition
  // FLATTENED_ID both list every participating device once and still fold sum.
  ASSERT_OK_AND_ASSIGN(
      auto single_partition,
      RunAndCheckHloRewrite(
          kMultiPartition, AllReduceFolder(), true,
          {{"$partitions", "1"}, {"$type", "f32"}, {"$op", "add"}}));
  ExpectOneAllReduce(single_partition.get(), "replica_groups={{0,1,2,3}}");

  constexpr absl::string_view kFlattenedId = R"(
    HloModule m, replica_count=2, num_partitions=2

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT r = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), channel_id=1, use_global_device_ids=true,
        replica_groups={{0,1},{2,3}}, to_apply=sum
      ROOT ar1 = f32[8] all-reduce(ar0), channel_id=2,
        use_global_device_ids=true, replica_groups={{0,2},{1,3}}, to_apply=sum
    }
  )";
  ASSERT_OK_AND_ASSIGN(
      auto flattened,
      RunAndCheckHloRewrite(kFlattenedId, AllReduceFolder(), true));
  ExpectOneAllReduce(flattened.get(), "replica_groups={{0,1,2,3}}");
}

TEST_F(AllReduceFolderTest, NonAssociativeAndDegenerateReducersNotFolded) {
  constexpr absl::string_view kCustomReducer = R"(
    HloModule m

    red0 {
      a = $type[] parameter(0)
      b = $type[] parameter(1)
      ROOT r = $type[] $expr0
    }

    red1 {
      a = $type[] parameter(0)
      b = $type[] parameter(1)
      ROOT r = $type[] $expr1
    }

    ENTRY main {
      p0 = $type[8] parameter(0)
      ar0 = $type[8] all-reduce(p0), replica_groups={{0,1},{2,3}}, to_apply=red0
      ROOT ar1 = $type[8] all-reduce(ar0), replica_groups={{0,2},{1,3}},
        to_apply=red1
    }
  )";
  for (absl::string_view bad_expr :
       {"subtract(a, b)", "divide(a, b)", "add(a, a)", "add(b, b)"}) {
    ASSERT_OK(RunAndCheckHloRewrite(
        kCustomReducer, AllReduceFolder(), false,
        {{"$type", "f32"}, {"$expr0", bad_expr}, {"$expr1", bad_expr}}));
  }
  // Mixed valid sum followed by degenerate add(a, a) must also be rejected.
  ASSERT_OK(RunAndCheckHloRewrite(
      kCustomReducer, AllReduceFolder(), false,
      {{"$type", "f32"}, {"$expr0", "add(a, b)"}, {"$expr1", "add(a, a)"}}));

  // Commuted add(b, a) and integral bitwise xor/and/or are associative and
  // commutative and still fold.
  ASSERT_OK_AND_ASSIGN(
      auto commuted,
      RunAndCheckHloRewrite(kCustomReducer, AllReduceFolder(), true,
                            {{"$type", "f32"},
                             {"$expr0", "add(a, b)"},
                             {"$expr1", "add(b, a)"}}));
  ExpectOneAllReduce(commuted.get(), "replica_groups={{0,1,2,3}}");
  ASSERT_OK_AND_ASSIGN(
      auto bitwise_xor,
      RunAndCheckHloRewrite(kCustomReducer, AllReduceFolder(), true,
                            {{"$type", "s32"},
                             {"$expr0", "xor(a, b)"},
                             {"$expr1", "xor(a, b)"}}));
  ExpectOneAllReduce(bitwise_xor.get(), "replica_groups={{0,1,2,3}}");
}

TEST_F(AllReduceFolderTest, StrictSubsetReplicaGroupsNotFolded) {
  ASSERT_OK(RunAndCheckHloRewrite(
      k2AllReduce, AllReduceFolder(), false,
      {{"$group_0", "{{0,1},{2,3},{4,5}}"}, {"$group_1", "{{0,2},{1,3}}"}}));
}

TEST_F(AllReduceFolderTest, SupersetSparseAndMalformedReplicaGroupsHandled) {
  // Superset ar1 (overflowed replica_group_no and contributing_replicas_set_id
  // under ASan previously).
  ASSERT_OK(RunAndCheckHloRewrite(
      k2AllReduce, AllReduceFolder(), false,
      {{"$group_0", "{{0,1},{2,3}}"}, {"$group_1", "{{0,2},{1,3},{4,5}}"}}));

  // Unverified C++ API inputs: valid large sparse replica IDs fold in
  // O(listed replicas) memory, while missing sparse IDs, negative IDs,
  // duplicate IDs in ar0 or ar1, and empty subgroups are all rejected safely.
  auto run_raw = [](const std::vector<std::vector<int64_t>>& groups0,
                    const std::vector<std::vector<int64_t>>& groups1) {
    HloModule module("raw", HloModuleConfig());
    HloComputation::Builder rb("sum");
    auto* a = rb.AddInstruction(
        HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "a"));
    auto* b = rb.AddInstruction(
        HloInstruction::CreateParameter(1, ShapeUtil::MakeShape(F32, {}), "b"));
    rb.AddInstruction(
        HloInstruction::CreateBinary(a->shape(), HloOpcode::kAdd, a, b));
    HloComputation* sum = module.AddEmbeddedComputation(rb.Build());

    auto make_list = [](const std::vector<std::vector<int64_t>>& gs) {
      std::vector<ReplicaGroup> out;
      for (const auto& g : gs) {
        ReplicaGroup& rg = out.emplace_back();
        for (int64_t id : g) {
          rg.add_replica_ids(id);
        }
      }
      return std::make_shared<CollectiveDeviceList>(out);
    };
    const Shape s = ShapeUtil::MakeShape(F32, {8});
    HloComputation::Builder eb("main");
    auto* p0 = eb.AddInstruction(HloInstruction::CreateParameter(0, s, "p0"));
    auto* ar0 = eb.AddInstruction(HloInstruction::CreateAllReduce(
        s, {p0}, sum, make_list(groups0), false, std::nullopt, false));
    eb.AddInstruction(HloInstruction::CreateAllReduce(
        s, {ar0}, sum, make_list(groups1), false, std::nullopt, false));
    module.AddEntryComputation(eb.Build());
    return AllReduceFolder().Run(&module);
  };
  EXPECT_THAT(run_raw({{0, 100000000}, {200000000, 300000000}},
                      {{0, 200000000}, {100000000, 300000000}}),
              IsOkAndHolds(true));
  EXPECT_THAT(run_raw({{0, 2}}, {{0, 1}}), IsOkAndHolds(false));
  EXPECT_THAT(run_raw({{-1, 0}}, {{-1, 0}}), IsOkAndHolds(false));
  EXPECT_THAT(run_raw({{0, 1}}, {{0, -1}}), IsOkAndHolds(false));
  EXPECT_THAT(run_raw({{0, 0}, {1, 2}}, {{0, 1}, {0, 2}}), IsOkAndHolds(false));
  EXPECT_THAT(run_raw({{0, 1}, {2, 3}}, {{0, 2}, {0, 3}, {1, 3}}),
              IsOkAndHolds(false));
  EXPECT_THAT(run_raw({{}, {0, 1}}, {{0}, {1}}), IsOkAndHolds(false));
  EXPECT_THAT(run_raw({{0, 1}}, {{}, {0, 1}}), IsOkAndHolds(false));
}

TEST_F(AllReduceFolderTest, VariadicAndLiveRootAllReducesNotFolded) {
  constexpr absl::string_view kVariadicDistinct = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT r = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      p1 = f32[8] parameter(1)
      ar0 = f32[8] all-reduce(p0), channel_id=1,
        replica_groups={{0,1},{2,3}}, to_apply=sum
      ROOT ar1 = (f32[8], f32[8]) all-reduce(ar0, p1), channel_id=2,
        replica_groups={{0,2},{1,3}}, to_apply=sum
    }
  )";
  ASSERT_OK_AND_ASSIGN(
      auto mod1,
      RunAndCheckHloRewrite(kVariadicDistinct, AllReduceFolder(), false));
  EXPECT_EQ(AllReduceCount(mod1.get()), 2);

  constexpr absl::string_view kVariadicRepeated = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT r = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups={{0,1},{2,3}}, to_apply=sum
      ROOT ar1 = (f32[8], f32[8]) all-reduce(ar0, ar0),
        replica_groups={{0,2},{1,3}}, to_apply=sum
    }
  )";
  ASSERT_OK(RunAndCheckHloRewrite(kVariadicRepeated, AllReduceFolder(), false));

  constexpr absl::string_view kLiveRootWithDeadUser = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT r = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ROOT ar0 = f32[8] all-reduce(p0), replica_groups={{0,1},{2,3}},
        to_apply=sum
      dead_ar1 = f32[8] all-reduce(ar0), replica_groups={{0,2},{1,3}},
        to_apply=sum
    }
  )";
  ASSERT_OK(
      RunAndCheckHloRewrite(kLiveRootWithDeadUser, AllReduceFolder(), false));
}

TEST_F(AllReduceFolderTest, PreservesShardingMetadataBackendConfigAndSchedule) {
  constexpr absl::string_view kScheduledSharded = R"(
    HloModule m, is_scheduled=true

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT r = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups={{0,1},{2,3}}, to_apply=sum,
        sharding={replicated}, metadata={op_name="step/ar0"},
        frontend_attributes={collective_group_key="g0", stage="s0"},
        backend_config="{\"flag\":true}"
      ROOT ar1 = f32[8] all-reduce(ar0), replica_groups={{0,2},{1,3}},
        to_apply=sum, sharding={replicated}, metadata={op_name="step/ar1"},
        frontend_attributes={collective_group_key="g0", stage="s1"},
        backend_config="{\"flag\":true}"
    }
  )";
  ASSERT_OK_AND_ASSIGN(
      auto module,
      RunAndCheckHloRewrite(kScheduledSharded, AllReduceFolder(), true));
  ASSERT_OK(module->schedule().Verify());
  ExpectOneAllReduce(module.get(), "replica_groups={{0,1,2,3}}");
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(root->has_sharding());
  EXPECT_TRUE(root->sharding().IsReplicated());
  EXPECT_EQ(root->metadata().op_name(), "step/ar1");
  EXPECT_EQ(root->raw_backend_config_string(), "{\"flag\":true}");
  EXPECT_EQ(root->get_frontend_attribute("collective_group_key"), "g0");
  EXPECT_EQ(root->get_frontend_attribute("stage"), "s0,s1");

  // Mismatched backend_config declines folding rather than silently dropping
  // one instruction's configuration.
  constexpr absl::string_view kMismatchedBackendConfig = R"(
    HloModule m

    sum {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT r = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      ar0 = f32[8] all-reduce(p0), replica_groups={{0,1},{2,3}}, to_apply=sum,
        backend_config="{\"flag\":true}"
      ROOT ar1 = f32[8] all-reduce(ar0), replica_groups={{0,2},{1,3}},
        to_apply=sum, backend_config="{\"flag\":false}"
    }
  )";
  ASSERT_OK(RunAndCheckHloRewrite(kMismatchedBackendConfig, AllReduceFolder(),
                                  false));
}

}  // namespace
}  // namespace xla
