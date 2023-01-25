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

#include <array>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xla/tests/literal_test_util.h"
#include "xla/tools/multihost_hlo_runner/hlo_runner.h"

// This test exercises collective communication ops in multi-host context.
// Note that that does not necessarily mean that the test has to run on
// multiple hosts, just that the HLO uses channel-id to link collectives
// from different execution instances.
namespace xla {

struct TestCase {
  std::string name;
  std::string hlo_text;
  std::pair<int, int> execution_size;  // num_replicas, num_partitions.
  std::function<void(int, int, const Literal &)> verifier;

  std::string ToString() const {
    return name + std::to_string(execution_size.first) + "x" +
           std::to_string(execution_size.second);
  }
};

std::ostream &operator<<(std::ostream &os, const TestCase &tc) {
  os << tc.ToString();
  return os;
}

std::vector<TestCase> GetTestCases() {
  std::vector<TestCase> test_cases;

  const std::string main_with_instance_id = R"(
      ENTRY main {
        // compute instance_id = replica_id * 100 + partition_id.
        partition_id = u32[] partition-id()
        replica_id = u32[] replica-id()
        constant100 = u32[] constant(100)
        temp0 = u32[] multiply(replica_id, constant100)
        instance_id = u32[] add(temp0, partition_id))";

  const std::vector<TestCase> all_reduce_test_cases = {
      {// cross replica all-reduce (with num_partitions > 1)
       "AllReduce_CrossReplica",
       // clang-format off
       R"(
      HloModule AllReduce_CrossReplica

      apply_op {
         x = u32[] parameter(0)
         y = u32[] parameter(1)
         ROOT apply_op = u32[] add(x, y)
      })" + main_with_instance_id + R"(
        ROOT r = u32[] all-reduce(instance_id), replica_groups={{0, 1}},
                                                to_apply=apply_op
      })",
       // clang-format on
       {2, 2},
       [](int replica_id, int partition_id, const Literal &output) {
         // instance_id:    r0p0 = 0, r0p1 = 1, r1p0 = 100, r1p1 = 101
         // for p0, both replicas participate, so output = r0p0+r1p0 = 100
         // for p1:, output = r0p1 + r1p1 = 102
         uint32_t expected = partition_id == 0 ? 100 : 102;
         LiteralTestUtil::ExpectR0Equal<uint32_t>(expected, output);
       }},

      {// all-reduce with channel_id, use_global_device_ids = false
       // reduce happens across all partitions (mode kCrossReplicaAndPartition)
       "AllReduce_CrossReplicaAndPartition",
       // clang-format off
       R"(
      HloModule AllReduce_CrossReplicaAndPartition

      apply_op {
         x = u32[] parameter(0)
         y = u32[] parameter(1)
         ROOT apply_op = u32[] add(x, y)
      })" + main_with_instance_id + R"(
        ROOT r = u32[] all-reduce(instance_id), channel_id=1,
                                                replica_groups={{0, 1}},
                                                to_apply=apply_op
      })",
       // clang-format on
       {2, 2},
       [](int replica_id, int partition_id, const Literal &output) {
         // instance_id:    r0p0 = 0, r0p1 = 1, r1p0 = 100, r1p1 = 101
         // all 4 instances participate, so we expect output = 202.
         LiteralTestUtil::ExpectR0Equal<uint32_t>(202, output);
       }},

      {// all-reduce with channel_id, use_global_device_ids = true
       // (Flattened ID mode) Flattened IDs are:
       //   r0p0 = 0, r0p1 = 1, r1p0 = 2, r1p1 = 3.
       "AllReduce_FlattenedID",
       // clang-format off
       R"(
      HloModule AllReduce_FlattenedID

      apply_op {
         x = u32[] parameter(0)
         y = u32[] parameter(1)
         ROOT apply_op = u32[] add(x, y)
      })" + main_with_instance_id + R"(
        ROOT r = u32[] all-reduce(instance_id), channel_id=1,
             replica_groups={{0}, {1, 2, 3}}, to_apply=apply_op,
                                              use_global_device_ids=true
      })",
       // clang-format on
       {2, 2},
       [](int replica_id, int partition_id, const Literal &output) {
         // instance_id:    r0p0 = 0, r0p1 = 1, r1p0 = 100, r1p1 = 101
         // reduction is among: {r0p0} = 0, {r0p1, r1p0, r1p1} = 202
         uint32_t expected = replica_id == 0 && partition_id == 0 ? 0 : 202;
         LiteralTestUtil::ExpectR0Equal<uint32_t>(expected, output);
       }},
  };

  const std::vector<TestCase> all_gather_test_cases = {
      {// all-gather with no channel-id, > 1 partitions (kCrossReplica)
       "AllGather_CrossReplica",
       // clang-format off
      "HloModule AllGather_CrossReplica " + main_with_instance_id + R"(
         bcast = u32[1] broadcast(instance_id), dimensions={}
         ROOT r = u32[2] all-gather(bcast), replica_groups={{0,1}},
                           dimensions={0}
      })",
       // clang-format on
       {2, 2},
       [](int replica_id, int partition_id, const xla::Literal &output) {
         // instance_id:    r0p0 = 0, r0p1 = 1, r1p0 = 100, r1p1 = 101
         // for p0, both replicas participate, so output = [0, 100]
         // for p1, output = r0p1 + r1p1 = [1, 101]
         if (partition_id == 0) {
           LiteralTestUtil::ExpectR1Equal<uint32_t>({0, 100}, output);
         } else {
           LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 101}, output);
         }
       }},

      {// all-gather with channel-id, use_global_device_ids = false
       // gather happens across all partitions and replicas
       // (kCrossReplicaAndPartition)
       "AllGather_CrossReplicaAndPartition",
       // clang-format off
      "HloModule AllGather_CrossReplicaAndPartition " + main_with_instance_id +
      R"(bcast = u32[1] broadcast(instance_id), dimensions={}
         ROOT r = u32[4] all-gather(bcast), replica_groups={{0,1}},
                           dimensions={0}, channel_id=1
      })",
       // clang-format on
       {2, 2},
       [](int replica_id, int partition_id, const xla::Literal &output) {
         // instance_id:    r0p0 = 0, r0p1 = 1, r1p0 = 100, r1p1 = 101
         // all 4 instances participate in the gather to we expect the
         // output to be {0, 1, 100, 101} (all r0's, followed by all r1's)
         LiteralTestUtil::ExpectR1Equal<uint32_t>({0, 1, 100, 101}, output);
       }},

      {// all-gather with channel-id, use_global_device_ids = true
       // (Flattened ID mode) Flattened IDs are:
       //   r0p0 = 0, r0p1 = 1, r1p0 = 2, r1p1 = 3.
       "AllGather_FlattenedID",
       // clang-format off
      "HloModule AllGather_FlattenedID " + main_with_instance_id +
      R"(bcast = u32[1] broadcast(instance_id), dimensions={}
         ROOT r = u32[2] all-gather(bcast), replica_groups={{0,3}, {1, 2}},
                           dimensions={0}, use_global_device_ids=true,
                           channel_id=1
      })",
       // clang-format on
       {2, 2},
       [](int replica_id, int partition_id, const xla::Literal &output) {
         // instance_id:    r0p0 = 0, r0p1 = 1, r1p0 = 100, r1p1 = 101
         // We gather from flattened_ids [0, 3] = {r0p0, r1p1} = {0, 101}
         //                flattened_ids [1, 2] = {r0p1, r1p0} = {1, 100}
         int flattened_id = replica_id * 2 + partition_id;
         if (flattened_id == 0 || flattened_id == 3) {
           LiteralTestUtil::ExpectR1Equal<uint32_t>({0, 101}, output);
         } else {
           LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 100}, output);
         }
       }},
  };

  const std::vector<TestCase> all_to_all_test_cases = {
      {// all-to-all with no-channel id (cross replica) (mode 4)
       "AllToAll_CrossReplica",
       "HloModule AllToAll_CrossReplica " + main_with_instance_id + R"(
        // generate a per instance 2x1 input unique to that input.
        bcast = u32[2] broadcast(instance_id), dimensions={}
        ids = u32[2] iota(), iota_dimension=0
        input = u32[2] add(bcast, ids)
        ROOT r = u32[2] all-to-all(input), replica_groups={{0,1},{2,3}},
                                           dimensions={0}
      })",
       {4, 1},
       [](int replica_id, int partition_id, const Literal &output) {
         // There are 4 replicas and their inputs to all-to-all are:
         // r0 = [0, 1], r1 = [100, 101], r2 = [200, 201], r3 = [300, 301]
         // r0 and r1 for one all-to-all group, so the output is:
         //   r0 = [0, 100], r1 = [1, 101]
         // Similarly, r2 and r3 for a group, so the output is:
         //   r2 = [200, 300], r3 = [201, 301]
         const std::array<std::pair<uint32_t, uint32_t>, 4> expected = {
             {{0, 100}, {1, 101}, {200, 300}, {201, 301}}};
         LiteralTestUtil::ExpectR1Equal<uint32_t>(
             {expected[replica_id].first, expected[replica_id].second}, output);
       }},

      {// all-to-all with channel id (cross partition) (mode 5)
       "AllToAll_CrossPartition",
       "HloModule AllToAll_CrossPartition " + main_with_instance_id + R"(
        // generate a per instance 2x1 input unique to that input.
        // = instance_id * 2 + <0, 1>
        constant2 = u32[] constant(2)
        temp1 = u32[] multiply(instance_id, constant2)
        bcast = u32[2] broadcast(temp1), dimensions={}
        ids = u32[2] iota(), iota_dimension=0
        input = u32[2] add(bcast, ids)
        ROOT r = u32[2] all-to-all(input), replica_groups={{0,1},{2,3}},
                                           dimensions={0}, channel_id=1
      })",
       {1, 4},
       [](int replica_id, int partition_id, const Literal &output) {
         // There are 4 partitions and their inputs to all-to-all are:
         // p0 = [0, 1], p1 = [2, 3], p2 = [4, 5], p3 = [6, 7]
         // p0 and p1 for one all-to-all group, so the output is:
         //   p0 = [0, 2], p1 = [1, 3]
         // Similarly, p2 and p3 for a group, so the output is:
         //   p2 = [4, 6], p3 = [5, 7]
         const std::array<std::pair<uint32_t, uint32_t>, 4> expected = {
             {{0, 2}, {1, 3}, {4, 6}, {5, 7}}};
         LiteralTestUtil::ExpectR1Equal<uint32_t>(
             {expected[partition_id].first, expected[partition_id].second},
             output);
       }},
  };

  const std::vector<TestCase> collective_permute_test_cases = {
      {// cross replica permute with > 1 partitions.
       "CollectivePermute_CrossReplica",
       // clang-format off
        "HloModule CollectivePermute_CrossReplica " + main_with_instance_id +
        R"(
          ROOT r = u32[] collective-permute(instance_id),
                                          source_target_pairs={{0,1},{1,0}}
        })",
       // clang-format on
       {2, 2},
       [](int replica_id, int partition_id, const Literal &output) {
         // There are 2 replicas and 2 partitions, with instance id
         // instance_id:    r0p0 = 0, r0p1 = 1, r1p0 = 100, r1p1 = 101
         // expected: r0p0 = 100, r0p1 = 101, r1p0 = 0, r1p1 = 1.
         const uint32_t expected[2][2] = {{100, 101}, {0, 1}};
         LiteralTestUtil::ExpectR0Equal<uint32_t>(
             expected[replica_id][partition_id], output);
       }},

      {// cross-partition permute with 1 replica.
       "CollectivePermute_CrossPartition",
       // clang-format off
       "HloModule CollectivePermute_CrossPartition " + main_with_instance_id +
       R"(
          ROOT r = u32[] collective-permute(instance_id), channel_id=1,
                                          source_target_pairs={{0,1},{1,0}}
       })",
       // clang-format on
       {1, 2},
       [](int replica_id, int partition_id, const Literal &output) {
         LiteralTestUtil::ExpectR0Equal(partition_id == 0 ? 1U : 0U, output);
       }},

      {// cross-partition permute with >1 replica.
       "CollectivePermute_CrossPartition",
       // clang-format off
       "HloModule CollectivePermute_CrossPartition " + main_with_instance_id +
       R"(
          ROOT r = u32[] collective-permute(instance_id), channel_id=1,
                                          source_target_pairs={{0,1},{1,0}}
       })",
       // clang-format on
       {2, 2},
       [](int replica_id, int partition_id, const Literal &output) {
         // instance_id:    r0p0 = 0, r0p1 = 1, r1p0 = 100, r1p1 = 101
         // expected: r0p0 = 1, r0p1 = 0, r1p0 = 101, r1p1 = 100
         const uint32_t expected[2][2] = {{1, 0}, {101, 100}};
         LiteralTestUtil::ExpectR0Equal(expected[replica_id][partition_id],
                                        output);
       }},
  };

  test_cases.insert(test_cases.end(), all_reduce_test_cases.begin(),
                    all_reduce_test_cases.end());
  test_cases.insert(test_cases.end(), all_gather_test_cases.begin(),
                    all_gather_test_cases.end());
  test_cases.insert(test_cases.end(), all_to_all_test_cases.begin(),
                    all_to_all_test_cases.end());
  test_cases.insert(test_cases.end(), collective_permute_test_cases.begin(),
                    collective_permute_test_cases.end());
  return test_cases;
}

class MultiHostCollectiveOpsTest : public ::testing::TestWithParam<TestCase> {
 public:
  static void SetUpTestSuite() {
    setenv("NCCL_LAUNCH_MODE", "PARALLEL", /*replace=*/1);
    ::testing::TestWithParam<TestCase>::SetUpTestSuite();
  }
};

TEST_P(MultiHostCollectiveOpsTest, TestCollective) {
  const TestCase &tc = GetParam();
  int num_replicas = tc.execution_size.first;
  int num_partitions = tc.execution_size.second;
  int required_devices = num_replicas * num_partitions;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      MultiHostHloRunner::ReadModuleFromString(tc.hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          MultiHostHloRunner::GetDeviceClient(
                              xla::MultiHostHloRunner::DeviceType::kGpu));
  const int num_devices = client->device_count();
  if (required_devices > num_devices) {
    GTEST_SKIP() << "Requires " << required_devices
                 << " devices but found only " << num_devices;
  }
  MultiHostHloRunner::Options options{
      .num_replicas = static_cast<size_t>(tc.execution_size.first),
      .num_partitions = static_cast<size_t>(tc.execution_size.second),
      .log_output_mode = MultiHostHloRunner::LogOutputMode::kLogOutput,
      .hlo_passes_mode =
          MultiHostHloRunner::HloPassesMode::kDisableAllHloPasses,
      .spmd_mode = (num_partitions > 1)
                       ? MultiHostHloRunner::SpmdMode::kUseSpmdPartitioning
                       : MultiHostHloRunner::SpmdMode::kNotUseSpmdPartitioning,
      .spmd_partitioned_mode =
          MultiHostHloRunner::SpmdPartitionedMode::kIsNotSpmdPartitionedModule};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiHostHloRunner> hlo_runner,
      MultiHostHloRunner::CreateMultiHostHloRunner(options, std::move(client)));
  TF_ASSERT_OK_AND_ASSIGN(auto output,
                          hlo_runner->CompileAndRun(hlo_module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto logical_id_to_device_id,
                          hlo_runner->CreateLogicalIdToDeviceIdMap());
  EXPECT_EQ(logical_id_to_device_id.n1(), tc.execution_size.first);
  EXPECT_EQ(logical_id_to_device_id.n2(), tc.execution_size.second);
  for (int replica_id = 0; replica_id < num_replicas; ++replica_id) {
    for (int partition_id = 0; partition_id < num_partitions; ++partition_id) {
      int device_id = logical_id_to_device_id(replica_id, partition_id);
      std::vector<Literal> &output_slice = output[device_id];
      EXPECT_EQ(output_slice.size(), 1);
      tc.verifier(replica_id, partition_id, output_slice[0]);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(MultiHostCollectiveOpsTest, MultiHostCollectiveOpsTest,
                         ::testing::ValuesIn(GetTestCases()));

}  // namespace xla
