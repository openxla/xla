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

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/backends/gpu/tests/ragged_all_to_all_test_utils.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::NotNull;

TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs) {
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

TEST_P(RaggedAllToAllTest, RaggedAllToAll_SeveralOps_2GPUs) {
  // TODO(patrios): Make this a loop.
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

TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs_CommandBuffer) {
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

  // Verify correctness of ragged-all-to-all when command buffers for
  // collectives are enabled.
  // As of Dec 2025, ragged-all-to-all command is not implemented, so this test
  // verifies that we don't try to accidentally create a command buffer and
  // crash.
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

TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs_S4) {
  if (IsSymmetricNcclPath()) {
    GTEST_SKIP() << "S4 buffers do not meet NCCL symmetric window alignment "
                    "requirements.";
  }
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = s4[4,2]{1,0:E(4)} parameter(0)
    output = s4[4,2]{1,0:E(4)} parameter(1)
    input_offsets = s32[2] parameter(2)
    send_sizes = s32[2] parameter(3)
    output_offsets = s32[2] parameter(4)
    recv_sizes = s32[2] parameter(5)
    ROOT ra2a = s4[4,2]{1,0:E(4)} ragged-all-to-all(input, output, 
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
                                 /*input_sizes=*/{/*replica_0=*/{1, 1},
                                                  /*replica_1=*/{3, 1}}));

  // `CreateRandomTestData` calculates sizes and offsets metadata, but fill
  // input and expected output literals with random floats. We need to manually
  // override them with s4 values.
  inputs_[0] = LiteralUtil::CreateR2<s4>(
      {{s4(1), s4(1)}, {s4(2), s4(2)}, {s4(0), s4(0)}, {s4(0), s4(0)}});
  inputs_[1] = LiteralUtil::CreateR2<s4>(
      {{s4(3), s4(3)}, {s4(4), s4(4)}, {s4(5), s4(5)}, {s4(6), s4(6)}});

  expected_outputs_[0] = LiteralUtil::CreateR2<s4>(
      {{s4(1), s4(1)}, {s4(3), s4(3)}, {s4(4), s4(4)}, {s4(5), s4(5)}});
  expected_outputs_[1] = LiteralUtil::CreateR2<s4>(
      {{s4(2), s4(2)}, {s4(6), s4(6)}, {s4(-1), s4(-1)}, {s4(-1), s4(-1)}});

  ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  // Check that ragged-all-to-all on S4 was converted to S8.
  // Skip this check for decomposer test, because there ragged-all-to-all was
  // lowered to all-to-all.
  if (std::get<1>(GetParam()) != RaggedAllToAllImplType::kDecomposer) {
    HloInstruction* ragged_all_to_all = FindInstruction(
        execution_result.optimized_module, HloOpcode::kRaggedAllToAll);
    ASSERT_NE(ragged_all_to_all, nullptr);
    EXPECT_EQ(ragged_all_to_all->shape().element_type(), S8);
  }

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs_InputBufferLargerThanOutput) {
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

TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs_OutputBufferLargerThanInput) {
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

TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs_MultipleUpdates) {
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

TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs_MultiDimData) {
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

TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs_Degenerate) {
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

TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs_NonDefaultLayout) {
  if (IsSymmetricNcclPath()) {
    GTEST_SKIP() << "Non-default layout buffers do not meet NCCL symmetric "
                    "window alignment requirements.";
  }
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module

  ENTRY entry {
    input = f32[16,4,8]{0,2,1} parameter(0)
    output = f32[16,4,8]{0,1,2} parameter(1)
    input_offsets = s32[2] parameter(2)
    send_sizes = s32[2] parameter(3)
    output_offsets = s32[2] parameter(4)
    recv_sizes = s32[2] parameter(5)
    ROOT ra2a = f32[16,4,8]{0,1,2} ragged-all-to-all(input, output,
      input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleReplicatedStr, kNumReplicas));

  auto ragged_all_to_all =
      FindInstruction(module.get(), HloOpcode::kRaggedAllToAll);
  EXPECT_THAT(ragged_all_to_all, NotNull());

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

TEST_P(RaggedAllToAllTest,
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

// This test checks if RaggedAllToAll kernel preserves the initial output data.
// It updates every second element of the output with the input data, and the
// other elements are preserved.
TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs_PreservesInitialData) {
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

  // No input arguments are needed
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

// Regression test for issue #46982: zero-size peer slices must be skipped so
// that the collective's cost is O(actual_neighbors) not O(n_devices).
//
// Pattern: replica_0 sends 3 rows to replica_1; replica_1 sends 0 rows to
// replica_0. This creates one direction with all-zero sizes per peer, which
// previously caused ~12 us of unnecessary NCCL overhead for every zero-size
// slot in the Send/Recv or Put+Signal paths.
TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs_SparseNeighbors) {
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

  // Sparse pattern: replica_0 sends 3 rows to replica_1, and 1 row to itself.
  // replica_1 sends 0 rows to replica_0 (zero-size slot), and 1 row to itself.
  // The zero-size slot for replica_1->replica_0 must be skipped without hangs.
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

TEST_P(RaggedAllToAllTest, RaggedAllToAll_8GPUs) {
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

TEST_P(RaggedAllToAllTest, RaggedAllToAll_8GPUs_2ReplicasPerGroups) {
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

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);

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

TEST_P(RaggedAllToAllTest, RaggedAllToAll_8GPUs_4ReplicasPerGroups) {
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

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);

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

std::string RaggedAllToAllImplTypeName(
    RaggedAllToAllImplType ragged_all_to_all_impl_type) {
  switch (ragged_all_to_all_impl_type) {
    case RaggedAllToAllImplType::kNcclFallback:
      return "nccl_fallback";
    case RaggedAllToAllImplType::kDecomposer:
      return "decomposer";
    case RaggedAllToAllImplType::kOneShotWithMultiGpuBarrier:
      return "one_shot_with_multi_gpu_barrier";
    case RaggedAllToAllImplType::kOneShotWithMultiGpuBarrierWithNccl:
      return "one_shot_with_multi_gpu_barrier_with_nccl";
    case RaggedAllToAllImplType::kDeviceKernel:
      return "device_kernel";
    default:
      LOG(FATAL) << "Unknown ragged all-to-all implementation type.";
  }
}

std::string CollectivesModeName(DebugOptions::CollectivesMode mode) {
  switch (mode) {
    case DebugOptions::COLLECTIVES_PRIVATE_MEMORY:
      return "private";
    case DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY:
      return "symmetric";
    case DebugOptions::COLLECTIVES_PEER_MEMORY:
      return "peer";
    default:
      return absl::StrCat("mode_", static_cast<int>(mode));
  }
}

// Builds the test parameters: NCCL impl is exercised against private/peer
// collectives modes (symmetric mode is tested in
// ragged_all_to_all_e2e_symmetric_test); the device-kernel impl
// requires symmetric memory; other impls only need PRIVATE since they don't
// dispatch on the mode.
std::vector<
    std::tuple<bool, RaggedAllToAllImplType, DebugOptions::CollectivesMode>>
BuildRaggedAllToAllTestParams() {
  std::vector<
      std::tuple<bool, RaggedAllToAllImplType, DebugOptions::CollectivesMode>>
      params;
  for (bool enable_async : {false, true}) {
    for (DebugOptions::CollectivesMode mode :
         {DebugOptions::COLLECTIVES_PRIVATE_MEMORY,
          DebugOptions::COLLECTIVES_PEER_MEMORY}) {
      params.emplace_back(enable_async, RaggedAllToAllImplType::kNcclFallback,
                          mode);
    }
    for (RaggedAllToAllImplType impl_type :
         {RaggedAllToAllImplType::kDecomposer,
          RaggedAllToAllImplType::kOneShotWithMultiGpuBarrier,
          RaggedAllToAllImplType::kOneShotWithMultiGpuBarrierWithNccl}) {
      params.emplace_back(enable_async, impl_type,
                          DebugOptions::COLLECTIVES_PRIVATE_MEMORY);
    }
    params.emplace_back(enable_async, RaggedAllToAllImplType::kDeviceKernel,
                        DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY);
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(
    RaggedAllToAllTest, RaggedAllToAllTest,
    ::testing::ValuesIn(BuildRaggedAllToAllTestParams()),
    [](const ::testing::TestParamInfo<std::tuple<
           bool, RaggedAllToAllImplType, DebugOptions::CollectivesMode>>&
           info) {
      return absl::StrCat(std::get<0>(info.param) ? "async" : "sync", "_",
                          RaggedAllToAllImplTypeName(std::get<1>(info.param)),
                          "_", CollectivesModeName(std::get<2>(info.param)));
    });

struct RaggedAllToAllMultiHostDecomposerTestConfig {
  int64_t num_input_rows;
  int64_t num_output_rows;
  int64_t slice_size;
  std::vector<std::vector<int64_t>> replica_groups;

  int64_t num_replicas() const {
    int64_t num_replicas = 0;
    for (const auto& group : replica_groups) {
      num_replicas += group.size();
    }
    return num_replicas;
  }

  std::string replica_groups_str() const {
    std::vector<std::string> rg_strings;
    for (const auto& group : replica_groups) {
      rg_strings.push_back(absl::StrCat("{", absl::StrJoin(group, ","), "}"));
    }
    return absl::StrCat("{", absl::StrJoin(rg_strings, ","), "}");
  }
};

class RaggedAllToAllMultiHostDecomposerTest
    : public RaggedAllToAllTestBase,
      public ::testing::WithParamInterface<
          RaggedAllToAllMultiHostDecomposerTestConfig> {
 public:
  RaggedAllToAllMultiHostDecomposerTest()
      : RaggedAllToAllTestBase(
            /*enable_async=*/false,
            /*impl_type=*/RaggedAllToAllImplType::kOneShotWithMultiGpuBarrier) {
  }

  int64_t num_input_rows() const { return GetParam().num_input_rows; }
  int64_t num_output_rows() const { return GetParam().num_output_rows; }
  const std::vector<std::vector<int64_t>>& replica_groups() const {
    return GetParam().replica_groups;
  }

  bool IsLocalRaggedAllToAll(const HloInstruction* ragged_all_to_all) {
    return gpu::IsAllReplicasLocal(GetParam().slice_size,
                                   ragged_all_to_all->replica_groups(),
                                   COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA);
  }

 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        RaggedAllToAllTestBase::GetDebugOptionsForTest();
    debug_options
        .set_xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer(
            true);
    return debug_options;
  }
};

TEST_P(RaggedAllToAllMultiHostDecomposerTest, DecomposesCorrectly) {
  const int64_t kNumUpdates = 32;

  std::string kModuleReplicatedStr = absl::Substitute(
      R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[$0,5,32] parameter(0)
    output = f32[$1,5,32] parameter(1)
    input_offsets = s32[$2] parameter(2)
    send_sizes = s32[$2] parameter(3)
    output_offsets = s32[$2] parameter(4)
    recv_sizes = s32[$2] parameter(5)
    ROOT ra2a = f32[$1,5,32] ragged-all-to-all(input, output,
      input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups=$3
  })",
      num_input_rows(), num_output_rows(), kNumUpdates,
      GetParam().replica_groups_str());

  const int64_t num_replicas = GetParam().num_replicas();
  const int64_t num_replicas_per_group = GetParam().replica_groups[0].size();
  const int64_t num_updates_per_replica = kNumUpdates / num_replicas_per_group;

  if (device_count() < num_replicas) {
    GTEST_SKIP() << "Test requires at least " << num_replicas << " devices ("
                 << device_count() << " available)";
  }

  HloModuleConfig config = GetModuleConfigForTest(num_replicas);
  config.mutable_debug_options()
      .set_xla_gpu_unsupported_override_fast_interconnect_slice_size(
          GetParam().slice_size);

  ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleReplicatedStr, config));

  Array<int64_t> input_sizes(
      {num_replicas, num_replicas_per_group, num_updates_per_replica});
  input_sizes.FillRandomUniform(0, 16);

  ASSERT_OK(CreateRandomTestData(module.get(), input_sizes));

  ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  auto ragged_all_to_all_instructions = FindInstructions(
      execution_result.optimized_module, HloOpcode::kRaggedAllToAll);

  // Verify that ragged-all-to-all was decomposed.
  for (const auto& ragged_all_to_all : ragged_all_to_all_instructions) {
    EXPECT_TRUE(IsLocalRaggedAllToAll(ragged_all_to_all));
  }

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), num_replicas);
  for (int i = 0; i < num_replicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[i], results[i]));
  }
}

std::vector<RaggedAllToAllMultiHostDecomposerTestConfig>
GetRaggedAllToAllMultiHostDecomposerTestConfigs() {
  std::vector<RaggedAllToAllMultiHostDecomposerTestConfig> configs;

  auto add_configs_for_input_sizes = [&](int64_t num_input_rows,
                                         int64_t num_output_rows) {
    configs.push_back({num_input_rows,
                       num_output_rows,
                       /*slice_size=*/1,
                       {{0, 1}}});

    for (int64_t slice_size : {1, 2, 4}) {
      configs.push_back({num_input_rows,
                         num_output_rows,
                         slice_size,
                         {{0, 1, 2, 3, 4, 5, 6, 7}}});

      configs.push_back({num_input_rows,
                         num_output_rows,
                         slice_size,
                         {{0, 2, 4, 6, 1, 3, 5, 7}}});

      configs.push_back({num_input_rows,
                         num_output_rows,
                         slice_size,
                         {{0, 2, 4, 6}, {1, 3, 5, 7}}});
    }
  };

  add_configs_for_input_sizes(512, 4096);
  add_configs_for_input_sizes(4096, 512);

  return configs;
}

INSTANTIATE_TEST_SUITE_P(
    RaggedAllToAllMultiHostDecomposerTest,
    RaggedAllToAllMultiHostDecomposerTest,
    ::testing::ValuesIn(GetRaggedAllToAllMultiHostDecomposerTestConfigs()),
    [](const ::testing::TestParamInfo<
        RaggedAllToAllMultiHostDecomposerTestConfig>& info) {
      std::string num_replica_groups =
          info.param.replica_groups.size() == 1
              ? "1ReplicaGroup"
              : absl::StrCat(info.param.replica_groups.size(), "ReplicaGroups");

      bool is_combine = info.param.num_input_rows > info.param.num_output_rows;
      bool is_shuffled = absl::c_any_of(info.param.replica_groups,
                                        [](const std::vector<int64_t>& group) {
                                          return !absl::c_is_sorted(group);
                                        });

      return absl::StrCat(is_combine ? "Combine" : "Dispatch", "_",
                          info.param.num_replicas(), "GPUs_SliceSize",
                          info.param.slice_size, "_", num_replica_groups,
                          is_shuffled ? "_ShuffledReplicaGroups" : "");
    });

class RaggedAllToAllFallbackTest : public RaggedAllToAllTestBase {
 public:
  RaggedAllToAllFallbackTest()
      : RaggedAllToAllTestBase(
            /*enable_async=*/false,
            /*impl_type=*/RaggedAllToAllImplType::kNcclFallback,
            /*collectives_mode=*/DebugOptions::COLLECTIVES_PRIVATE_MEMORY) {}

 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions opts = RaggedAllToAllTestBase::GetDebugOptionsForTest();
    opts.set_xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel(false);
    opts.set_xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback(false);
    return opts;
  }
};

TEST_F(RaggedAllToAllFallbackTest, FailsWhenFallbackIsDisabled) {
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

  absl::StatusOr<ExecutionResult> execution_result =
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs());
  EXPECT_FALSE(execution_result.ok());
  EXPECT_THAT(
      execution_result.status().message(),
      ::testing::HasSubstr("RaggedAllToAll fallback to NCCL is not allowed"));
}

}  // namespace
}  // namespace xla
