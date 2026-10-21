/* Copyright 2026 The OpenXLA Authors.

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

#include <gtest/gtest.h>

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/array.h"
#include "xla/backends/gpu/tests/ragged_all_to_all_test_utils.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/literal_test_util.h"
#include "xla/xla.pb.h"

namespace xla {
namespace {

class RaggedAllToAllSymmetricTest : public RaggedAllToAllTest {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions opts = RaggedAllToAllTest::GetDebugOptionsForTest();
    opts.set_xla_gpu_experimental_nccl_window_symmetric(true);
    return opts;
  }
};

TEST_P(RaggedAllToAllSymmetricTest, RaggedAllToAll_2GPUs) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[4] parameter(0)
    output = f32[4] parameter(1)
    input_offsets = s32[2] parameter(2)
    send_sizes = s32[2] parameter(3)
    output_offsets = s32[2] parameter(4)
    recv_sizes = s32[2] parameter(5)
    ROOT ra2a = f32[4] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleReplicatedStr, kNumReplicas));

  ASSERT_OK(CreateRandomTestData(module.get(),
                                 /*input_sizes=*/{/*replica_0=*/{1, 1},
                                                  /*replica_1=*/{3, 1}}));

  ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;

  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllSymmetricTest, RaggedAllToAll_SeveralOps_2GPUs) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[4] parameter(0)
    output = f32[4] parameter(1)
    input_offsets = s32[2] parameter(2)
    send_sizes = s32[2] parameter(3)
    output_offsets = s32[2] parameter(4)
    recv_sizes = s32[2] parameter(5)
    ra2a_0 = f32[4] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
    ROOT ra2a = f32[4] ragged-all-to-all(ra2a_0, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleReplicatedStr, kNumReplicas));

  inputs_.clear();
  inputs_.reserve(kNumReplicas);
  inputs_.push_back(LiteralUtil::CreateR1<float>({45.0f, 38.0f, 0.0f, 0.0f}));
  inputs_.push_back(
      LiteralUtil::CreateR1<float>({41.0f, 16.0f, 110.0f, 107.0f}));

  output_init_ = LiteralUtil::CreateR1<float>({-1.0f, -1.0f, -1.0f, -1.0f});

  input_offsets_.clear();
  input_offsets_.reserve(kNumReplicas);
  input_offsets_.push_back(LiteralUtil::CreateR1<int32_t>({0, 1}));
  input_offsets_.push_back(LiteralUtil::CreateR1<int32_t>({0, 3}));
  input_sizes_.clear();
  input_sizes_.reserve(kNumReplicas);
  input_sizes_.push_back(LiteralUtil::CreateR1<int32_t>({1, 1}));
  input_sizes_.push_back(LiteralUtil::CreateR1<int32_t>({3, 1}));

  output_offsets_.clear();
  output_offsets_.reserve(kNumReplicas);
  output_offsets_.push_back(LiteralUtil::CreateR1<int32_t>({0, 0}));
  output_offsets_.push_back(LiteralUtil::CreateR1<int32_t>({1, 1}));
  output_sizes_.clear();
  output_sizes_.reserve(kNumReplicas);
  output_sizes_.push_back(LiteralUtil::CreateR1<int32_t>({1, 3}));
  output_sizes_.push_back(LiteralUtil::CreateR1<int32_t>({1, 1}));

  expected_outputs_.clear();
  expected_outputs_.reserve(kNumReplicas);
  expected_outputs_.push_back(
      LiteralUtil::CreateR1<float>({45.0f, 38.0f, 107.0f, -1.0f}));
  expected_outputs_.push_back(
      LiteralUtil::CreateR1<float>({41.0f, -1.0f, -1.0f, -1.0f}));

  ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;

  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllSymmetricTest, RaggedAllToAll_2GPUs_CommandBuffer) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[4] parameter(0)
    output = f32[4] parameter(1)
    input_offsets = s32[2] parameter(2)
    send_sizes = s32[2] parameter(3)
    output_offsets = s32[2] parameter(4)
    recv_sizes = s32[2] parameter(5)
    ROOT ra2a = f32[4] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleReplicatedStr, kNumReplicas));

  DebugOptions& debug_options =
      module->mutable_config().mutable_debug_options();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);
  debug_options.set_xla_gpu_graph_min_graph_size(1);

  ASSERT_OK(CreateRandomTestData(module.get(),
                                 /*input_sizes=*/{/*replica_0=*/{1, 1},
                                                  /*replica_1=*/{3, 1}}));

  ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;

  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllSymmetricTest,
       RaggedAllToAll_2GPUs_InputBufferLargerThanOutput) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[32] parameter(0)
    output = f32[16] parameter(1)
    input_offsets = s32[2] parameter(2)
    send_sizes = s32[2] parameter(3)
    output_offsets = s32[2] parameter(4)
    recv_sizes = s32[2] parameter(5)
    ROOT ra2a = f32[16] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleReplicatedStr, kNumReplicas));

  ASSERT_OK(CreateRandomTestData(module.get(),
                                 /*input_sizes=*/{/*replica_0=*/{8, 5},
                                                  /*replica_1=*/{4, 3}}));

  ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllSymmetricTest,
       RaggedAllToAll_2GPUs_OutputBufferLargerThanInput) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[16] parameter(0)
    output = f32[32] parameter(1)
    input_offsets = s32[2] parameter(2)
    send_sizes = s32[2] parameter(3)
    output_offsets = s32[2] parameter(4)
    recv_sizes = s32[2] parameter(5)
    ROOT ra2a = f32[32] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleReplicatedStr, kNumReplicas));

  ASSERT_OK(CreateRandomTestData(module.get(),
                                 /*input_sizes=*/{/*replica_0=*/{4, 12},
                                                  /*replica_1=*/{5, 11}}));

  ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllSymmetricTest, RaggedAllToAll_2GPUs_MultipleUpdates) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[8] parameter(0)
    output = f32[8] parameter(1)
    input_offsets = s32[4] parameter(2)
    send_sizes = s32[4] parameter(3)
    output_offsets = s32[4] parameter(4)
    recv_sizes = s32[4] parameter(5)
    ROOT ra2a = f32[8] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);

  ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleReplicatedStr, config));

  ASSERT_OK(CreateRandomTestData(
      module.get(), /*input_sizes=*/{/*replica_0=*/{{1, 2}, {2, 1}},
                                     /*replica_1=*/{{3, 1}, {1, 1}}}));

  ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllSymmetricTest, RaggedAllToAll_2GPUs_MultiDimData) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = bf16[16, 5, 32] parameter(0)
    output = bf16[16, 5, 32] parameter(1)
    input_offsets = s64[2] parameter(2)
    send_sizes = s64[2] parameter(3)
    output_offsets = s64[2] parameter(4)
    recv_sizes = s64[2] parameter(5)
    ROOT ra2a = bf16[16, 5, 32] ragged-all-to-all(input, output,
      input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleReplicatedStr, kNumReplicas));

  ASSERT_OK(CreateRandomTestData(module.get(),
                                 /*input_sizes=*/{/*replica_0=*/{4, 7},
                                                  /*replica_1=*/{2, 5}}));

  ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllSymmetricTest, RaggedAllToAll_2GPUs_Degenerate) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module

  ENTRY entry {
    input = f32[4] parameter(0)
    output = f32[4] parameter(1)
    input_offsets = s32[1] parameter(2)
    send_sizes = s32[1] parameter(3)
    output_offsets = s32[1] parameter(4)
    recv_sizes = s32[1] parameter(5)
    ROOT ra2a = f32[4] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{0},{1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleReplicatedStr, kNumReplicas));

  ASSERT_OK(CreateRandomTestData(module.get(),
                                 /*input_sizes=*/{/*replica_0=*/{1},
                                                  /*replica_1=*/{3}}));

  ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllSymmetricTest,
       RaggedAllToAll_2GPUs_DevicesInReplicaGroupInReverseOrder) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[4] parameter(0)
    output = f32[4] parameter(1)
    input_offsets = s32[2] parameter(2)
    send_sizes = s32[2] parameter(3)
    output_offsets = s32[2] parameter(4)
    recv_sizes = s32[2] parameter(5)
    ROOT ra2a = f32[4] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{1,0}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleReplicatedStr, kNumReplicas));

  ASSERT_OK(CreateRandomTestData(module.get(),
                                 /*input_sizes=*/{/*replica_0=*/{1, 1},
                                                  /*replica_1=*/{3, 1}}));

  ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllSymmetricTest, RaggedAllToAll_2GPUs_PreservesInitialData) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1, replica_count=2

  ENTRY entry {
    io = f32[4] iota(), iota_dimension=0
    id = u32[] replica-id()
    ten = u32[] constant(10)
    id2 = u32[] multiply(id, ten)
    id3 = f32[] convert(id2)
    id4 = f32[4] broadcast(id3)
    input = f32[4] add(io, id4)
    output = f32[4] constant({-1,-1,-1,-1})
    send_sizes = s32[2] constant({1,1})
    recv_sizes = s32[2] constant({1,1})
    input_offsets = s32[2] constant({0, 2})
    step = u32[] constant(2)
    oof = u32[] multiply(id, step)
    oof2 = s32[] convert(oof)
    output_offsets = s32[2] broadcast(oof2)

    ROOT ra2a = f32[4] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleReplicatedStr, kNumReplicas));

  std::vector<std::vector<Literal*>> arguments(kNumReplicas);

  expected_outputs_.clear();
  expected_outputs_.reserve(kNumReplicas);
  expected_outputs_.push_back(
      LiteralUtil::CreateR1<float>({0.0f, -1.0f, 10.0f, -1.0f}));
  expected_outputs_.push_back(
      LiteralUtil::CreateR1<float>({2.0f, -1.0f, 12.0f, -1.0f}));

  ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                       ExecuteReplicated(std::move(module), arguments));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllSymmetricTest, RaggedAllToAll_2GPUs_SparseNeighbors) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[4] parameter(0)
    output = f32[4] parameter(1)
    input_offsets = s32[2] parameter(2)
    send_sizes = s32[2] parameter(3)
    output_offsets = s32[2] parameter(4)
    recv_sizes = s32[2] parameter(5)
    ROOT ra2a = f32[4] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleReplicatedStr, kNumReplicas));

  ASSERT_OK(CreateRandomTestData(module.get(),
                                 /*input_sizes=*/{/*replica_0=*/{1, 3},
                                                  /*replica_1=*/{0, 1}}));

  ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllSymmetricTest, RaggedAllToAll_8GPUs) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[512, 5, 32] parameter(0)
    output = f32[512, 5, 32] parameter(1)
    input_offsets = s32[32] parameter(2)
    send_sizes = s32[32] parameter(3)
    output_offsets = s32[32] parameter(4)
    recv_sizes = s32[32] parameter(5)
    ROOT ra2a = f32[512, 5, 32] ragged-all-to-all(input, output,
      input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,1,2,3,4,5,6,7}}
  })";

  const int64_t kNumReplicas = 8;
  const int64_t kNumUpdatesPerReplica = 4;
  if (device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << device_count() << " available)";
  }

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleReplicatedStr, kNumReplicas));

  Array<int64_t> input_sizes(
      {kNumReplicas, kNumReplicas, kNumUpdatesPerReplica});
  input_sizes.FillRandomUniform(0, 10);

  ASSERT_OK(CreateRandomTestData(module.get(), input_sizes));

  ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[i], results[i]));
  }
}

TEST_P(RaggedAllToAllSymmetricTest, RaggedAllToAll_8GPUs_2ReplicasPerGroups) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[512, 5, 32] parameter(0)
    output = f32[512, 5, 32] parameter(1)
    input_offsets = s32[32] parameter(2)
    send_sizes = s32[32] parameter(3)
    output_offsets = s32[32] parameter(4)
    recv_sizes = s32[32] parameter(5)
    ROOT ra2a = f32[512, 5, 32] ragged-all-to-all(input, output,
      input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,4},{1,5},{2,6},{3,7}}
  })";

  const int64_t kNumReplicas = 8;
  const int64_t kNumReplicasPerGroup = 2;
  const int64_t kNumUpdatesPerReplica = 16;
  if (device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << device_count() << " available)";
  }

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleReplicatedStr, kNumReplicas));

  Array<int64_t> input_sizes(
      {kNumReplicas, kNumReplicasPerGroup, kNumUpdatesPerReplica});
  input_sizes.FillRandomUniform(0, 10);

  ASSERT_OK(CreateRandomTestData(module.get(), input_sizes));

  ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[i], results[i]));
  }
}

TEST_P(RaggedAllToAllSymmetricTest, RaggedAllToAll_8GPUs_4ReplicasPerGroups) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[512, 5, 32] parameter(0)
    output = f32[512, 5, 32] parameter(1)
    input_offsets = s32[32] parameter(2)
    send_sizes = s32[32] parameter(3)
    output_offsets = s32[32] parameter(4)
    recv_sizes = s32[32] parameter(5)
    ROOT ra2a = f32[512, 5, 32] ragged-all-to-all(input, output,
      input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,1,2,3},{4,5,6,7}}
  })";

  const int64_t kNumReplicas = 8;
  const int64_t kNumReplicasPerGroup = 4;
  const int64_t kNumUpdatesPerReplica = 8;
  if (device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << device_count() << " available)";
  }

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleReplicatedStr, kNumReplicas));

  Array<int64_t> input_sizes(
      {kNumReplicas, kNumReplicasPerGroup, kNumUpdatesPerReplica});
  input_sizes.FillRandomUniform(0, 10);

  ASSERT_OK(CreateRandomTestData(module.get(), input_sizes));

  ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[i], results[i]));
  }
}

INSTANTIATE_TEST_SUITE_P(
    RaggedAllToAllTest, RaggedAllToAllSymmetricTest,
    ::testing::Combine(
        ::testing::Bool(),
        ::testing::Values(RaggedAllToAllImplType::kNcclFallback),
        ::testing::Values(DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY)),
    [](const ::testing::TestParamInfo<std::tuple<
           bool, RaggedAllToAllImplType, DebugOptions::CollectivesMode>>&
           info) {
      return absl::StrCat(std::get<0>(info.param) ? "async" : "sync",
                          "_nccl_fallback_symmetric");
    });

}  // namespace
}  // namespace xla
