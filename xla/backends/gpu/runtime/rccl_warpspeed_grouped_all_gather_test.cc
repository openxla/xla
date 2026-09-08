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

// ROCm integration test for XLA's grouped multi-buffer AllGather through RCCL.

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

constexpr int kNumDevices = 8;
constexpr int kNumBuffers = 2;
constexpr int64_t kElementsPerRank = 524'288;

class RcclWarpSpeedGroupedAllGatherTest : public CollectiveOpsE2ETestBase {
 public:
  RcclWarpSpeedGroupedAllGatherTest()
      : CollectiveOpsE2ETestBase(/*memory_size=*/128 * kMB,
                                 /*collectives_memory_size=*/128 * kMB) {}

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions options = CollectiveOpsE2ETestBase::GetDebugOptionsForTest();
    // Exercise ordinary RCCL group submission, without graph capture or
    // alternative collective kernels.
    options.add_xla_gpu_disable_async_collectives(DebugOptions::ALLGATHER);
    options.set_xla_gpu_all_gather_mode(
        DebugOptions::COLLECTIVES_PRIVATE_MEMORY);
    options.clear_xla_gpu_experimental_use_collective_kernels();
    options.clear_xla_enable_nccl_symmetric_buffers_for_collectives();
    options.set_xla_gpu_experimental_enable_nccl_symmetric_buffers(false);
    options.clear_xla_gpu_enable_command_buffer();
    options.clear_xla_gpu_enable_collectives_command_buffer_filter();
    return options;
  }
};

Literal MakeInput(int buffer, int rank) {
  Literal input =
      Literal::CreateFromShape(ShapeUtil::MakeShape(BF16, {kElementsPerRank}));
  for (int64_t i = 0; i < kElementsPerRank; ++i) {
    uint32_t mixed = static_cast<uint32_t>(i) * 2654435761u;
    mixed ^= mixed >> 16;
    // Distinct rank/buffer ranges and an index-dependent pattern. All values
    // are integers in [0, 255], exactly representable in BF16.
    input.data<bfloat16>()[i] = bfloat16(
        static_cast<float>((buffer * kNumDevices + rank) * 16 + (mixed & 15)));
  }
  return input;
}

TEST_F(RcclWarpSpeedGroupedAllGatherTest, TwoBuffersProduceExactResults) {
  ASSERT_GE(device_count(), kNumDevices);
  ASSERT_TRUE(Capability().IsRocm());
  ASSERT_EQ(Capability().rocm_compute_capability()->gfx_version(), "gfx950");
  ASSERT_GT(device_description().core_count(), 128);
  ASSERT_EQ(GpuCollectives::Resolve("ROCM"),
            GpuCollectives::Resolve("ROCM", "rccl"));

  // One variadic AllGather lowers to two RCCL calls in the same group.
  constexpr absl::string_view kHlo = R"(
HloModule grouped_all_gather

ENTRY main {
  a = bf16[524288]{0} parameter(0)
  b = bf16[524288]{0} parameter(1)
  ROOT gathered = (bf16[4194304]{0}, bf16[4194304]{0})
    all-gather(a, b), dimensions={0},
    replica_groups={{0,1,2,3,4,5,6,7}}
}
)";
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kHlo, kNumDevices));

  // Prepare two inputs per rank and their expected rank-ordered concatenation.
  std::vector<std::vector<Literal>> inputs(kNumDevices);
  std::vector<Literal> expected_buffers;
  for (int buffer = 0; buffer < kNumBuffers; ++buffer) {
    Literal gathered = Literal::CreateFromShape(
        ShapeUtil::MakeShape(BF16, {kNumDevices * kElementsPerRank}));
    for (int rank = 0; rank < kNumDevices; ++rank) {
      inputs[rank].push_back(MakeInput(buffer, rank));
      const auto data = inputs[rank].back().data<bfloat16>();
      std::copy(data.begin(), data.end(),
                gathered.data<bfloat16>().begin() + rank * kElementsPerRank);
    }
    expected_buffers.push_back(std::move(gathered));
  }
  Literal expected = LiteralUtil::MakeTupleOwned(std::move(expected_buffers));
  std::vector<std::vector<Literal*>> arguments(kNumDevices);
  for (int rank = 0; rank < kNumDevices; ++rank) {
    arguments[rank] = {&inputs[rank][0], &inputs[rank][1]};
  }

  ASSERT_OK_AND_ASSIGN(ExecutionResult execution,
                       ExecuteReplicated(std::move(module), arguments));

  // Reject a passing result if compilation removed the two-buffer collective.
  int all_gather_count = 0;
  for (const HloComputation* computation :
       execution.optimized_module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kAllGather) {
        ++all_gather_count;
        EXPECT_EQ(instruction->operand_count(), kNumBuffers);
        EXPECT_TRUE(
            ShapeUtil::Compatible(instruction->shape(), expected.shape()));
      }
    }
  }
  EXPECT_EQ(all_gather_count, 1);

  ASSERT_EQ(execution.results.size(), kNumDevices);
  for (int rank = 0; rank < kNumDevices; ++rank) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, execution.results[rank]))
        << "destination rank " << rank;
  }
}

}  // namespace
}  // namespace xla::gpu
