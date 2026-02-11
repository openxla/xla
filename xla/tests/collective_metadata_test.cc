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
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

class CollectiveMetadataTest : public CollectiveOpsE2ETestBase {
 protected:
  CollectiveMetadataTest()
      : CollectiveOpsE2ETestBase(/*memory_size=*/32 * kMB,
                                 /*collectives_memory_size=*/1 * kMB) {}

  void SetUp() override {
    CollectiveOpsE2ETestBase::SetUp();
    if (!IsHopperAndHigher()) {
      GTEST_SKIP() << "Test requires Hopper or newer architecture since it's "
                      "using a multicast.";
    }
  }
};

TEST_F(CollectiveMetadataTest, ConstructCollectiveMetadata) {
  const absl::string_view kModuleStr = R"(
  HloModule test, replica_count=2

  ENTRY test_computation {
    param_0 = f32[4] parameter(0)
    param_1 = f32[4] parameter(1)
    copy_1 = f32[4]{0:S(1)} copy(param_1)

    const_0 = f32[1] constant({10})

    result_tuple = (f32[4], f32[4]{0:S(1)}, f32[1], u64[12]) custom-call(param_0, copy_1, const_0), custom_call_target="CollectiveMetadata", output_to_operand_aliasing={{0}: (0, {}), {1}: (1, {})}
    ROOT get_tuple_element = u64[12] get-tuple-element(result_tuple), index=3
  })";

  constexpr int kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto unoptimized_module,
      ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  Literal input_0 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  Literal input_1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(unoptimized_module),
                        /*arguments=*/std::vector<Literal*>{&input_0, &input_1},
                        /*run_hlo_passes=*/false));
  const std::vector<Literal>& result = execution_result.results;
  ASSERT_EQ(result.size(), kNumReplicas);

  absl::Span<const uint64_t> first_result_data = result[0].data<uint64_t>();
  absl::Span<const uint64_t> second_result_data = result[1].data<uint64_t>();
  constexpr int kNumElements = 12;
  ASSERT_EQ(first_result_data.size(), kNumElements);
  ASSERT_EQ(second_result_data.size(), kNumElements);

  EXPECT_EQ(first_result_data[0], 0) << "First result rank is not 0.";
  EXPECT_EQ(second_result_data[0], 1) << "Second result rank is not 1.";

  EXPECT_NE(first_result_data[1], 0)
      << "First result pointer to peers is NULL.";
  EXPECT_NE(second_result_data[1], 0)
      << "Second result pointer to peers is NULL.";

  constexpr int kParamToPeersEnd = 9;
  for (int i = 3; i < kParamToPeersEnd; ++i) {
    EXPECT_NE(first_result_data[i], 0)
        << "First result param_to_peers is NULL.";
    EXPECT_EQ(second_result_data[i], first_result_data[i])
        << "Param_to_peers mismatch at index " << i
        << " in the first result: " << first_result_data[i]
        << " and in the second result: " << second_result_data[i];
  }

  for (int i = kParamToPeersEnd; i < kNumElements; ++i) {
    EXPECT_EQ(first_result_data[i], 0)
        << "First result multimem metadata is not NULL.";
    EXPECT_EQ(second_result_data[i], 0)
        << "Second result multimem metadata is not NULL.";
  }
}

TEST_F(CollectiveMetadataTest, ConstructCollectiveMetadataForPartitions) {
  const absl::string_view kModuleStr = R"(
  HloModule test, allow_spmd_sharding_propagation_to_parameters={true}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=2

  ENTRY test_computation {
    param_0 = f32[4] parameter(0)
    param_1 = f32[4] parameter(1)

    const_0 = f32[1] constant({10})

    result_tuple = (f32[4], f32[4]{0}, f32[1], u64[12]) custom-call(param_0, param_1, const_0), custom_call_target="CollectiveMetadata", output_to_operand_aliasing={{0}: (0, {}), {1}: (1, {})}
    ROOT get_tuple_element = u64[12] get-tuple-element(result_tuple), index=3
  })";

  constexpr int kNumPartitions = 2;
  ASSERT_GE(device_count(), kNumPartitions)
      << "Test requires at least " << kNumPartitions << " devices ("
      << device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto unoptimized_module,
      ParseAndReturnVerifiedModule(kModuleStr, /*replica_count=*/1,
                                   /*num_partitions=*/kNumPartitions));

  Literal input_0 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  Literal input_1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(unoptimized_module),
                        /*arguments=*/std::vector<Literal*>{&input_0, &input_1},
                        /*run_hlo_passes=*/false));
  const std::vector<Literal>& result = execution_result.results;
  ASSERT_EQ(result.size(), kNumPartitions);

  absl::Span<const uint64_t> first_result_data = result[0].data<uint64_t>();
  absl::Span<const uint64_t> second_result_data = result[1].data<uint64_t>();
  constexpr int kNumElements = 12;
  ASSERT_EQ(first_result_data.size(), kNumElements);
  ASSERT_EQ(second_result_data.size(), kNumElements);
}

TEST_F(CollectiveMetadataTest, ConstructCollectiveMetadataWithReplicaGroup) {
  const absl::string_view kModuleStr = R"(
  HloModule test, replica_count=4

  ENTRY test_computation {
    param_0 = f32[4] parameter(0)
    param_1 = f32[4] parameter(1)
    copy_1 = f32[4]{0:S(1)} copy(param_1)

    result_tuple = (f32[4], f32[4]{0:S(1)}, u64[9]) custom-call(param_0, copy_1), custom_call_target="CollectiveMetadata", output_to_operand_aliasing={{0}: (0, {}), {1}: (1, {})}, backend_config="{\"collective_metadata_backend_config\":{\"collective_devices\": { \"replica_groups\": [{\"replica_ids\": [0,1]}, {\"replica_ids\": [2,3]}]}}}"
    ROOT get_tuple_element = u64[9] get-tuple-element(result_tuple), index=2
  })";

  constexpr int kNumReplicas = 4;
  if (device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << device_count() << " available)";
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  Literal input_0 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  Literal input_1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/std::vector<Literal*>{&input_0, &input_1},
                        /*run_hlo_passes=*/false));
  const std::vector<Literal>& result = execution_result.results;
  ASSERT_EQ(result.size(), kNumReplicas);
  absl::Span<const uint64_t> replica_0_result_0_data =
      result[0].data<uint64_t>();
  absl::Span<const uint64_t> replica_0_result_1_data =
      result[1].data<uint64_t>();
  absl::Span<const uint64_t> replica_1_result_0_data =
      result[2].data<uint64_t>();
  absl::Span<const uint64_t> replica_1_result_1_data =
      result[3].data<uint64_t>();

  // Check the rank in the first position.
  constexpr int kNumElements = 9;
  ASSERT_EQ(replica_0_result_0_data.size(), kNumElements);
  ASSERT_EQ(replica_0_result_1_data.size(), kNumElements);
  ASSERT_EQ(replica_1_result_0_data.size(), kNumElements);
  ASSERT_EQ(replica_1_result_1_data.size(), kNumElements);

  EXPECT_EQ(replica_0_result_0_data[0], 0);
  EXPECT_EQ(replica_0_result_1_data[0], 1);
  EXPECT_EQ(replica_1_result_0_data[0], 0);
  EXPECT_EQ(replica_1_result_1_data[0], 1);

  // Check pointer to peers in the second position.
  EXPECT_NE(replica_0_result_0_data[1], 0);
  EXPECT_NE(replica_0_result_1_data[1], 0);
  EXPECT_NE(replica_1_result_0_data[1], 0);
  EXPECT_NE(replica_1_result_1_data[1], 0);

  // Check pointer to multimem metadata in the third position.
  EXPECT_NE(replica_0_result_0_data[2], 0);
  EXPECT_NE(replica_0_result_1_data[2], 0);
  EXPECT_NE(replica_1_result_0_data[2], 0);
  EXPECT_NE(replica_1_result_1_data[2], 0);

  constexpr int kParamToPeersEnd = 7;
  // Check param_to_peers structure.
  for (int i = 3; i < kParamToPeersEnd; ++i) {
    EXPECT_NE(replica_0_result_0_data[i], 0);
    EXPECT_EQ(replica_0_result_1_data[i], replica_0_result_0_data[i]);
    EXPECT_NE(replica_1_result_0_data[i], 0);
    EXPECT_EQ(replica_1_result_1_data[i], replica_1_result_0_data[i]);
  }

  // Check that multimem metadata is zeroed.
  for (int i = kParamToPeersEnd; i < kNumElements; ++i) {
    EXPECT_EQ(replica_0_result_0_data[i], 0);
    EXPECT_EQ(replica_0_result_1_data[i], 0);
    EXPECT_EQ(replica_1_result_0_data[i], 0);
    EXPECT_EQ(replica_1_result_1_data[i], 0);
  }
}

}  // namespace
}  // namespace xla
