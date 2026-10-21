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

#include "xla/backends/gpu/tests/ragged_all_to_all_test_utils.h"

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu::ragged_all_to_all {

Array<int64_t> CalculateOffsetsFromSizes(const Array<int64_t>& sizes,
                                         int64_t gap) {
  int64_t num_replicas = sizes.dim(0);
  int64_t num_peers = sizes.dim(1);
  int64_t num_updates = sizes.dim(2);
  Array<int64_t> offsets(sizes.dimensions());
  for (int i = 0; i < num_replicas; ++i) {
    int64_t cur_offset = 0;
    for (int j = 0; j < num_peers; ++j) {
      for (int k = 0; k < num_updates; ++k) {
        offsets(i, j, k) = cur_offset;
        cur_offset += sizes(i, j, k) + gap;
      }
    }
  }
  return offsets;
}

void FillWithRandomData(std::vector<Array<float>>& input_data,
                        std::vector<Array<float>>& output_data,
                        const Array<int64_t>& input_offsets,
                        const Array<int64_t>& output_offsets,
                        const Array<int64_t>& input_sizes) {
  int64_t num_replicas = input_sizes.dim(0);
  int64_t num_updates_per_replica = input_sizes.dim(2);
  std::vector<int64_t> start_indices(input_data[0].num_dimensions());
  std::vector<int64_t> chunk_sizes{input_data[0].dimensions().begin(),
                                   input_data[0].dimensions().end()};

  for (int i = 0; i < num_replicas; ++i) {
    for (int j = 0; j < num_replicas; ++j) {
      for (int k = 0; k < num_updates_per_replica; ++k) {
        chunk_sizes[0] = input_sizes(i, j, k);

        Array<float> chunk_data(chunk_sizes);
        chunk_data.FillRandomUniform(
            1.0f, 127.0f,
            /*seed=*/(i * num_replicas + j) * num_updates_per_replica + k);

        start_indices[0] = input_offsets(i, j, k);
        input_data[i].UpdateSlice(chunk_data, start_indices);

        start_indices[0] = output_offsets(i, j, k);
        output_data[j].UpdateSlice(chunk_data, start_indices);
      }
    }
  }
}

}  // namespace gpu::ragged_all_to_all

absl::Status RaggedAllToAllTestBase::CreateRandomTestData(
    HloModule* module, Array<int64_t> input_sizes) {
  CHECK(inputs_.empty());
  if (input_sizes.num_dimensions() == 2) {
    input_sizes.Reshape({input_sizes.dim(0), input_sizes.dim(1), 1});
  }
  auto ragged_all_to_all = FindInstruction(module, HloOpcode::kRaggedAllToAll);
  EXPECT_THAT(ragged_all_to_all, ::testing::NotNull());

  const std::vector<ReplicaGroup>& replica_groups =
      ragged_all_to_all->replica_groups();
  EXPECT_FALSE(replica_groups.empty());

  int64_t num_total_replicas = input_sizes.dim(0);
  int64_t num_replicas = replica_groups[0].replica_ids_size();

  EXPECT_TRUE(
      absl::c_all_of(replica_groups, [&](const ReplicaGroup& replica_group) {
        return replica_group.replica_ids_size() == num_replicas;
      }));

  inputs_.resize(num_total_replicas);
  expected_outputs_.resize(num_total_replicas);
  input_offsets_.resize(num_total_replicas);
  input_sizes_.resize(num_total_replicas);
  output_offsets_.resize(num_total_replicas);
  output_sizes_.resize(num_total_replicas);

  HloInstruction* output_param =
      module->entry_computation()->parameter_instruction(1);

  // The ragged-all-to-all accepts an output tensor as a parameter to allow
  // buffer reuse. We initialize the output tensor with -1 to make sure that
  // we don't accidentally overwrite data that is not part of the
  // ragged-all-to-all update.
  Array<float> output_init_data(output_param->shape().dimensions());
  output_init_data.Fill(-1);

  // Iterate over all replica groups and create random test data for each
  // group.
  for (const ReplicaGroup& replica_group : replica_groups) {
    Array<int64_t> input_sizes_per_replica_group(
        {num_replicas, input_sizes.dim(1), input_sizes.dim(2)});

    for (int64_t i = 0; i < num_replicas; ++i) {
      int64_t replica_id = replica_group.replica_ids(i);
      input_sizes_per_replica_group.UpdateSlice(
          input_sizes.Slice(
              {replica_id, 0, 0},
              {replica_id + 1, input_sizes.dim(1), input_sizes.dim(2)}),
          {i, 0, 0});
    }

    ABSL_RETURN_IF_ERROR(CreateRandomTestDataForReplicaGroup(
        module, input_sizes_per_replica_group, output_init_data,
        replica_group));
  }

  ABSL_ASSIGN_OR_RETURN(output_init_,
                        LiteralUtil::CreateFromArrayWithLayout(
                            output_init_data, output_param->shape().layout())
                            .Convert(output_param->shape().element_type()));
  return absl::OkStatus();
}

absl::Status RaggedAllToAllTestBase::CreateRandomTestDataForReplicaGroup(
    HloModule* module, Array<int64_t> input_sizes,
    const Array<float>& output_init_data, const ReplicaGroup& replica_group) {
  HloInstruction* input_param =
      module->entry_computation()->parameter_instruction(0);
  HloInstruction* output_param =
      module->entry_computation()->parameter_instruction(1);
  int64_t num_replicas = replica_group.replica_ids_size();

  Array<int64_t> output_sizes = input_sizes;
  output_sizes.TransposeDimensions({1, 0, 2});

  Array<int64_t> input_offsets =
      gpu::ragged_all_to_all::CalculateOffsetsFromSizes(input_sizes);
  Array<int64_t> output_offsets =
      gpu::ragged_all_to_all::CalculateOffsetsFromSizes(output_sizes);
  output_offsets.TransposeDimensions({1, 0, 2});

  std::vector<Array<float>> input_data(
      num_replicas, Array<float>(input_param->shape().dimensions()));
  std::vector<Array<float>> output_data(num_replicas, output_init_data);
  gpu::ragged_all_to_all::FillWithRandomData(
      input_data, output_data, input_offsets, output_offsets, input_sizes);

  // Create literals from array data.
  for (int64_t i = 0; i < num_replicas; ++i) {
    int64_t replica_id = replica_group.replica_ids(i);
    ABSL_ASSIGN_OR_RETURN(inputs_[replica_id],
                          LiteralUtil::CreateFromArrayWithLayout(
                              input_data[i], input_param->shape().layout())
                              .Convert(input_param->shape().element_type()));

    ABSL_ASSIGN_OR_RETURN(expected_outputs_[replica_id],
                          LiteralUtil::CreateFromArrayWithLayout(
                              output_data[i], output_param->shape().layout())
                              .Convert(output_param->shape().element_type()));

    ABSL_ASSIGN_OR_RETURN(
        input_offsets_[replica_id],
        GetParameterLiteral(module, /*parameter_index=*/2, i, input_offsets));

    ABSL_ASSIGN_OR_RETURN(
        input_sizes_[replica_id],
        GetParameterLiteral(module, /*parameter_index=*/3, i, input_sizes));

    ABSL_ASSIGN_OR_RETURN(
        output_offsets_[replica_id],
        GetParameterLiteral(module, /*parameter_index=*/4, i, output_offsets));
    ABSL_ASSIGN_OR_RETURN(
        output_sizes_[replica_id],
        GetParameterLiteral(module, /*parameter_index=*/5, i, output_sizes));
  }
  return absl::OkStatus();
}

std::vector<std::vector<Literal*>>
RaggedAllToAllTestBase::GetInputLiteralPtrs() {
  std::vector<std::vector<Literal*>> input_literal_ptrs;
  input_literal_ptrs.reserve(inputs_.size());
  for (int i = 0; i < inputs_.size(); ++i) {
    input_literal_ptrs.push_back({&inputs_[i], &output_init_,
                                  &input_offsets_[i], &input_sizes_[i],
                                  &output_offsets_[i], &output_sizes_[i]});
  }
  return input_literal_ptrs;
}

void RaggedAllToAllTestBase::SetUp() {
  if (device_count() < 2) {
    GTEST_SKIP() << "Test requires at least 2 devices.";
  }
  if (RequiresSymmetricMemory() && !IsHopperAndHigher()) {
    GTEST_SKIP() << "Symmetric memory is only supported on Hopper "
                    "architecture and above.";
  }
  if (impl_type_ == RaggedAllToAllImplType::kDeviceKernel &&
      Capability().IsRocm()) {
    GTEST_SKIP() << "Device initiated communication not supported on ROCm.";
  }
}

DebugOptions RaggedAllToAllTestBase::GetDebugOptionsForTest() const {
  DebugOptions opts = CollectiveOpsWithFlagsBase::GetDebugOptionsForTest();
  opts.set_xla_gpu_unsupported_enable_ragged_all_to_all_decomposer(
      impl_type_ == RaggedAllToAllImplType::kDecomposer);
  opts.set_xla_gpu_ragged_all_to_all_mode(collectives_mode_);
  // Requires symmetric memory.
  opts.set_xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl(false);
  if (impl_type_ == RaggedAllToAllImplType::kNcclFallback) {
    opts.set_xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback(true);
  }
  if (IsSymmetricNcclPath()) {
    opts.set_xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel(false);
  }
  if (impl_type_ == RaggedAllToAllImplType::kOneShotWithMultiGpuBarrier) {
    opts.set_xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel(true);
  }
  if (impl_type_ ==
          RaggedAllToAllImplType::kOneShotWithMultiGpuBarrierWithNccl &&
      // Disable for pre-Hopper architectures for now, until we can check if
      // devices are actually connected via fast interconnect.
      // TODO: b/493935137 - Enable for all architectures once we get
      // NCCL 2.29 integrated with host API to check LSA connectivity.
      Capability().IsCuda() &&
      Capability().cuda_compute_capability()->IsAtLeastHopper()) {
    opts.set_xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel(true);
    opts.set_xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl(true);
  }
  if (impl_type_ == RaggedAllToAllImplType::kDeviceKernel) {
    opts.set_xla_gpu_experimental_ragged_all_to_all_use_device_kernel(true);
  }
  return opts;
}

absl::StatusOr<Literal> RaggedAllToAllTestBase::GetParameterLiteral(
    HloModule* module, int64_t parameter_index, int64_t replica_id,
    const Array<int64_t>& data) {
  HloInstruction* param =
      module->entry_computation()->parameter_instruction(parameter_index);

  int64_t num_replicas = data.dim(0);
  int64_t num_updates_per_replica = data.dim(2);
  Array<int64_t> replica_slice =
      data.Slice({replica_id, 0, 0},
                 {replica_id + 1, num_replicas, num_updates_per_replica});
  replica_slice.Reshape({num_replicas * num_updates_per_replica});
  return LiteralUtil::CreateFromArray(replica_slice)
      .Convert(param->shape().element_type());
}

}  // namespace xla
