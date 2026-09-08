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
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

constexpr int kNumDevices = 8;
constexpr int kNumBuffers = 2;
constexpr int64_t kElementsPerRank = 524288;

class RcclWarpSpeedGroupedAllGatherTest : public CollectiveOpsE2ETestBase {
 public:
  RcclWarpSpeedGroupedAllGatherTest()
      : CollectiveOpsE2ETestBase(/*memory_size=*/128 * kMB,
                                 /*collectives_memory_size=*/128 * kMB) {}

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions options = CollectiveOpsE2ETestBase::GetDebugOptionsForTest();
    // Use synchronous AllGather with private memory and no graph capture.
    options.add_xla_gpu_disable_async_collectives(DebugOptions::ALLGATHER);
    options.set_xla_gpu_all_gather_mode(
        DebugOptions::COLLECTIVES_PRIVATE_MEMORY);
    options.clear_xla_gpu_enable_command_buffer();
    return options;
  }
};

Literal MakeInput(int buffer, int rank) {
  Literal input =
      Literal::CreateFromShape(ShapeUtil::MakeShape(BF16, {kElementsPerRank}));
  // All values are integers in [0, 255], exactly representable in BF16.
  std::mt19937 rng(buffer * kNumDevices + rank);
  std::uniform_int_distribution<int> distribution(0, 15);
  const int base = (buffer * kNumDevices + rank) * 16;
  for (int64_t i = 0; i < kElementsPerRank; ++i) {
    input.data<bfloat16>()[i] =
        bfloat16(static_cast<float>(base + distribution(rng)));
  }
  return input;
}

TEST_F(RcclWarpSpeedGroupedAllGatherTest, TwoBuffersProduceExactResults) {
  if (!Capability().IsRocm()) {
    GTEST_SKIP() << "This test requires the ROCm platform";
  }
  ASSERT_GE(device_count(), kNumDevices);

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

  // Prepare inputs and expected outputs.
  std::vector<std::vector<Literal>> inputs(kNumDevices);
  Literal expected = Literal::CreateFromShape(
      module->entry_computation()->root_instruction()->shape());
  for (int buffer = 0; buffer < kNumBuffers; ++buffer) {
    for (int rank = 0; rank < kNumDevices; ++rank) {
      inputs[rank].push_back(MakeInput(buffer, rank));
      const auto data = inputs[rank].back().data<bfloat16>();
      std::copy(
          data.begin(), data.end(),
          expected.data<bfloat16>({buffer}).begin() + rank * kElementsPerRank);
    }
  }
  std::vector<std::vector<Literal*>> arguments(kNumDevices);
  for (int rank = 0; rank < kNumDevices; ++rank) {
    arguments[rank] = {&inputs[rank][0], &inputs[rank][1]};
  }

  ASSERT_OK_AND_ASSIGN(ExecutionResult execution,
                       ExecuteReplicated(std::move(module), arguments));

  // Check that one AllGather with two inputs remains after compilation.
  int all_gather_count = 0;
  for (const HloComputation* computation :
       execution.optimized_module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kAllGather) {
        ++all_gather_count;
        EXPECT_EQ(instruction->operand_count(), kNumBuffers);
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
