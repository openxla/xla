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

#ifndef XLA_BACKENDS_GPU_TESTS_RAGGED_ALL_TO_ALL_TEST_UTILS_H_
#define XLA_BACKENDS_GPU_TESTS_RAGGED_ALL_TO_ALL_TEST_UTILS_H_

#include <cstdint>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/array.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {

enum class RaggedAllToAllImplType {
  kNcclFallback,
  kDecomposer,
  kOneShotWithMultiGpuBarrier,
  kOneShotWithMultiGpuBarrierWithNccl,
  kDeviceKernel,
};

class RaggedAllToAllTestBase : public CollectiveOpsWithFlagsBase {
 public:
  RaggedAllToAllTestBase(bool enable_async, RaggedAllToAllImplType impl_type,
                         DebugOptions::CollectivesMode collectives_mode =
                             DebugOptions::COLLECTIVES_PRIVATE_MEMORY)
      : CollectiveOpsWithFlagsBase(
            enable_async, /*enable_p2p_memcpy=*/false,
            /*enable_symmetric_buffer=*/
            collectives_mode == DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY,
            /*memory_size=*/64 * kMB,
            /*collectives_memory_size=*/
            collectives_mode == DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY
                ? 64 * kMB
                : 0),
        impl_type_(impl_type),
        collectives_mode_(collectives_mode) {}

  bool IsSymmetricNcclPath() const {
    return impl_type_ == RaggedAllToAllImplType::kNcclFallback &&
           collectives_mode_ == DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY;
  }

  bool RequiresSymmetricMemory() const {
    return IsSymmetricNcclPath() ||
           impl_type_ == RaggedAllToAllImplType::kDeviceKernel ||
           impl_type_ ==
               RaggedAllToAllImplType::kOneShotWithMultiGpuBarrierWithNccl ||
           collectives_mode_ == DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY;
  }

  // Creates random test data for a ragged-all-to-all.
  //
  // Ragged tensors which are ragged (have various size) along the second most
  // changing dimension only, i.e. shape such as [8, (4), 3]. In memory those
  // tensors are flattened out the outermost dimension.
  //
  // A ragged tensor is represented by three arrays: data, offsets, and sizes.
  //   * The data array holds the elements of the ragged tensor.
  //   * The offsets array holds the starting offset of each ragged row.
  //   * The sizes array holds the number of elements in each ragged row.
  //
  // A ragged-all-to-all of N replicas performance a collective transpose of the
  // ragged tensors. Each pair of replicas exchanges one ragged row. To generate
  // the test data we need to know the sizes of all ragged rows for each
  // replica.
  //
  // `input_sizes` is an array of shape [num_replicas, num_replicas,
  // num_updates_per_replica]. For concenivence, `input_sizes` can be a 2D
  // array, in that case `num_updates_per_replica` is assumed to be 1.
  absl::Status CreateRandomTestData(HloModule* module,
                                    Array<int64_t> input_sizes);

  // Create random test data for a ragged-all-to-all for a single replica group.
  absl::Status CreateRandomTestDataForReplicaGroup(
      HloModule* module, Array<int64_t> input_sizes,
      const Array<float>& output_init_data, const ReplicaGroup& replica_group);

  // Returns a vector of pointers to the literals in the format needed for
  // ExecuteReplicated.
  std::vector<std::vector<Literal*>> GetInputLiteralPtrs();

 protected:
  void SetUp() override;

  DebugOptions GetDebugOptionsForTest() const override;

  // Returns a literal for the given parameter of the given replica.
  absl::StatusOr<Literal> GetParameterLiteral(HloModule* module,
                                              int64_t parameter_index,
                                              int64_t replica_id,
                                              const Array<int64_t>& data);

  // Literates for the input and output data, offset, and size parameters of
  // the ragged-all-to-all. Each vector contains one literal per replica.
  std::vector<Literal> inputs_;
  std::vector<Literal> input_offsets_;
  std::vector<Literal> input_sizes_;

  std::vector<Literal> expected_outputs_;
  std::vector<Literal> output_offsets_;
  std::vector<Literal> output_sizes_;

  Literal output_init_;

  RaggedAllToAllImplType impl_type_;
  DebugOptions::CollectivesMode collectives_mode_;
};

class RaggedAllToAllTest
    : public RaggedAllToAllTestBase,
      public ::testing::WithParamInterface<std::tuple<
          bool, RaggedAllToAllImplType, DebugOptions::CollectivesMode>> {
 public:
  RaggedAllToAllTest()
      : RaggedAllToAllTestBase(std::get<0>(GetParam()), std::get<1>(GetParam()),
                               std::get<2>(GetParam())) {}
};

namespace gpu::ragged_all_to_all {

// Computes ragged tensor offsets based on the sizes of the ragged rows.
// sizes shape: [num_replicas, num_peers, num_updates]
Array<int64_t> CalculateOffsetsFromSizes(const Array<int64_t>& sizes,
                                         int64_t gap = 0);

// Fill the input and output tensors with random data. An all-to-all is
// effectively a transpose. We generate a chunk of random data for each update
// of each pair of replicas and write the chunk starting from the (i, j, k)
// offset of the input tensor and starting from the (j, i, k) offset of the
// output tensor.
void FillWithRandomData(std::vector<Array<float>>& input_data,
                        std::vector<Array<float>>& output_data,
                        const Array<int64_t>& input_offsets,
                        const Array<int64_t>& output_offsets,
                        const Array<int64_t>& input_sizes);

}  // namespace gpu::ragged_all_to_all
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TESTS_RAGGED_ALL_TO_ALL_TEST_UTILS_H_
