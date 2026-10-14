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

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/all_to_all.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class AllToAllTest : public CollectiveOpsWithFlagsBase {
 public:
  AllToAllTest()
      : CollectiveOpsWithFlagsBase(
            /*enable_async=*/true,
            /*enable_p2p_memcpy=*/false,
            /*enable_symmetric_buffer=*/true,
            /*memory_size=*/32 * kMB,
            /*collectives_memory_size=*/32 * kMB) {}

 protected:
  void SetUp() override {
    CollectiveOpsE2ETestBase::SetUp();
    if (Capability().IsCuda() && !IsHopperAndHigher()) {
      GTEST_SKIP() << "Test requires Hopper or newer architecture.";
    }
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions opts = CollectiveOpsWithFlagsBase::GetDebugOptionsForTest();
    opts.clear_xla_gpu_experimental_use_collective_kernels();
    opts.add_xla_gpu_experimental_use_collective_kernels(
        DebugOptions::COLLECTIVE_KERNEL_ALL_TO_ALL);
    return opts;
  }

  bool CheckDeviceCount(int32_t required_device_count) {
    [&]() -> void {
      const int32_t current_device_count = device_count();
      if (current_device_count < required_device_count) {
        ASSERT_GE(current_device_count, 2)
            << "Test requires at least 2 devices but only "
            << current_device_count << " available";
        if (current_device_count < required_device_count) {
          GTEST_SKIP() << "Test requires at least " << required_device_count
                       << " devices but only " << current_device_count
                       << " available.";
        }
      }
    }();
    return !IsSkipped() && !HasFatalFailure();
  }

  void VerifyOneShotAllToAll(const HloModule* optimized_module) {
    ASSERT_NE(optimized_module, nullptr);
    bool found_all_to_all = false;
    for (const HloComputation* comp : optimized_module->computations()) {
      for (const HloInstruction* instr : comp->instructions()) {
        if (instr->opcode() == HloOpcode::kAllToAll) {
          found_all_to_all = true;
          ASSERT_OK_AND_ASSIGN(gpu::GpuBackendConfig gpu_config,
                               instr->backend_config<gpu::GpuBackendConfig>());
          EXPECT_EQ(
              gpu_config.collective_backend_config().kernel_strategy(),
              gpu::CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT)
              << "Expected AllToAll instruction " << instr->name()
              << " to use KERNEL_STRATEGY_TRITON_ONE_SHOT, but got: "
              << gpu::CollectiveBackendConfig::CollectiveKernelStrategy_Name(
                     gpu_config.collective_backend_config().kernel_strategy());
        }
      }
    }
    EXPECT_TRUE(found_all_to_all)
        << "Expected to find an AllToAll instruction in optimized HLO.";
  }
};

// Basic 2-GPU all-to-all of f32[256] -> f32[256].
TEST_F(AllToAllTest, Basic2GpuF32) {
  constexpr int32_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = f32[256] parameter(0)
    ROOT all-to-all = f32[256] all-to-all(param_0), dimensions={0},
      replica_groups={{0,1}}
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  // Rank 0 gets [1.0 x 128, 2.0 x 128], rank 1 gets [3.0 x 128, 4.0 x 128].
  std::vector<float> data_r0(256), data_r1(256);
  for (int i = 0; i < 128; ++i) {
    data_r0[i] = 1.0f;
    data_r0[i + 128] = 2.0f;
    data_r1[i] = 3.0f;
    data_r1[i + 128] = 4.0f;
  }
  Literal input_r0 = LiteralUtil::CreateR1<float>(data_r0);
  Literal input_r1 = LiteralUtil::CreateR1<float>(data_r1);

  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllToAll(result.optimized_module);

  ASSERT_EQ(result.results.size(), kNumReplicas);

  // Expected output:
  // Rank 0: [1.0 x 128, 3.0 x 128]
  // Rank 1: [2.0 x 128, 4.0 x 128]
  std::vector<float> expected_data_r0(256), expected_data_r1(256);
  for (int i = 0; i < 128; ++i) {
    expected_data_r0[i] = 1.0f;
    expected_data_r0[i + 128] = 3.0f;
    expected_data_r1[i] = 2.0f;
    expected_data_r1[i + 128] = 4.0f;
  }
  Literal expected_r0 = LiteralUtil::CreateR1<float>(expected_data_r0);
  Literal expected_r1 = LiteralUtil::CreateR1<float>(expected_data_r1);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r0, result.results[0]))
      << "Mismatch at replica 0";
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r1, result.results[1]))
      << "Mismatch at replica 1";
}

// Cross-partition 2-GPU all-to-all (channel_id set, replica_count=1,
// num_partitions=2) of f32[256] -> f32[256].
TEST_F(AllToAllTest, CrossPartition2GpuF32) {
  constexpr int32_t kNumReplicas = 1;
  constexpr int32_t kNumPartitions = 2;
  if (!CheckDeviceCount(kNumPartitions)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test, replica_count=1, num_partitions=2
  ENTRY test_computation {
    param_0 = f32[256] parameter(0)
    ROOT all-to-all = f32[256] all-to-all(param_0), dimensions={0},
      channel_id=1, replica_groups={{0,1}}
  }
  )";

  ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas, kNumPartitions));

  std::vector<float> data_p0(256), data_p1(256);
  for (int i = 0; i < 128; ++i) {
    data_p0[i] = 1.0f;
    data_p0[i + 128] = 2.0f;
    data_p1[i] = 3.0f;
    data_p1[i + 128] = 4.0f;
  }
  Literal input_p0 = LiteralUtil::CreateR1<float>(data_p0);
  Literal input_p1 = LiteralUtil::CreateR1<float>(data_p1);

  std::vector<std::vector<Literal*>> args = {{&input_p0}, {&input_p1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllToAll(result.optimized_module);

  ASSERT_EQ(result.results.size(), kNumPartitions);

  std::vector<float> expected_data_p0(256), expected_data_p1(256);
  for (int i = 0; i < 128; ++i) {
    expected_data_p0[i] = 1.0f;
    expected_data_p0[i + 128] = 3.0f;
    expected_data_p1[i] = 2.0f;
    expected_data_p1[i + 128] = 4.0f;
  }
  Literal expected_p0 = LiteralUtil::CreateR1<float>(expected_data_p0);
  Literal expected_p1 = LiteralUtil::CreateR1<float>(expected_data_p1);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected_p0, result.results[0]))
      << "Mismatch at partition 0";
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_p1, result.results[1]))
      << "Mismatch at partition 1";
}

// Larger 2-GPU all-to-all to test multi-tile behavior: f32[8192] -> f32[8192].
TEST_F(AllToAllTest, Large2GpuF32) {
  constexpr int32_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = f32[8192] parameter(0)
    ROOT all-to-all = f32[8192] all-to-all(param_0), dimensions={0},
      replica_groups={{0,1}}
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  // rank 0: incrementing values [0, 1, ..., 8191]
  // rank 1: incrementing values [8192, 8193, ..., 16383]
  std::vector<float> data_r0(8192), data_r1(8192);
  for (int i = 0; i < 8192; ++i) {
    data_r0[i] = static_cast<float>(i);
    data_r1[i] = static_cast<float>(i + 8192);
  }
  Literal input_r0 = LiteralUtil::CreateR1<float>(data_r0);
  Literal input_r1 = LiteralUtil::CreateR1<float>(data_r1);

  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllToAll(result.optimized_module);

  ASSERT_EQ(result.results.size(), kNumReplicas);

  std::vector<float> expected_data_r0(8192), expected_data_r1(8192);
  for (int i = 0; i < 4096; ++i) {
    expected_data_r0[i] = static_cast<float>(i);
    expected_data_r0[i + 4096] = static_cast<float>(i + 8192);
    expected_data_r1[i] = static_cast<float>(i + 4096);
    expected_data_r1[i + 4096] = static_cast<float>(i + 12288);
  }
  Literal expected_r0 = LiteralUtil::CreateR1<float>(expected_data_r0);
  Literal expected_r1 = LiteralUtil::CreateR1<float>(expected_data_r1);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r0, result.results[0]))
      << "Mismatch at replica 0";
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r1, result.results[1]))
      << "Mismatch at replica 1";
}

// 2D shape: f32[32, 32] -> f32[32, 32] (split along dim 0).
TEST_F(AllToAllTest, TwoDimensional2Gpu) {
  constexpr int32_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = f32[32,32] parameter(0)
    ROOT all-to-all = f32[32,32] all-to-all(param_0), dimensions={0},
      replica_groups={{0,1}}
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  Literal input_r0 = LiteralUtil::CreateFull<float>({32, 32}, 0.0f);
  Literal input_r1 = LiteralUtil::CreateFull<float>({32, 32}, 0.0f);
  for (int64_t row = 0; row < 32; ++row) {
    for (int64_t col = 0; col < 32; ++col) {
      input_r0.Set<float>({row, col}, (row < 16) ? 1.0f : 2.0f);
      input_r1.Set<float>({row, col}, (row < 16) ? 3.0f : 4.0f);
    }
  }

  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllToAll(result.optimized_module);

  ASSERT_EQ(result.results.size(), kNumReplicas);

  Literal expected_r0 = LiteralUtil::CreateFull<float>({32, 32}, 0.0f);
  Literal expected_r1 = LiteralUtil::CreateFull<float>({32, 32}, 0.0f);
  for (int64_t row = 0; row < 32; ++row) {
    for (int64_t col = 0; col < 32; ++col) {
      expected_r0.Set<float>({row, col}, (row < 16) ? 1.0f : 3.0f);
      expected_r1.Set<float>({row, col}, (row < 16) ? 2.0f : 4.0f);
    }
  }

  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r0, result.results[0]))
      << "Mismatch at replica 0";
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r1, result.results[1]))
      << "Mismatch at replica 1";
}

// Runs 9 consequent all-to-alls and verifies that by default all of them use
// XLA's one-shot all-to-all kernel and are placed into command buffers, and
// that excluding COLLECTIVES_KERNEL from --xla_gpu_enable_command_buffer
// disables command buffer capture for them.
TEST_F(AllToAllTest, NineConsequentAllToAllsWithCudaGraphs) {
  constexpr int32_t kNumReplicas = 2;
  constexpr int kNumAllToAlls = 9;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    p0 = f32[256] parameter(0)
    a0 = f32[256] all-to-all(p0), dimensions={0}, replica_groups={{0,1}}
    a1 = f32[256] all-to-all(a0), dimensions={0}, replica_groups={{0,1}}
    a2 = f32[256] all-to-all(a1), dimensions={0}, replica_groups={{0,1}}
    a3 = f32[256] all-to-all(a2), dimensions={0}, replica_groups={{0,1}}
    a4 = f32[256] all-to-all(a3), dimensions={0}, replica_groups={{0,1}}
    a5 = f32[256] all-to-all(a4), dimensions={0}, replica_groups={{0,1}}
    a6 = f32[256] all-to-all(a5), dimensions={0}, replica_groups={{0,1}}
    a7 = f32[256] all-to-all(a6), dimensions={0}, replica_groups={{0,1}}
    ROOT a8 = f32[256] all-to-all(a7), dimensions={0}, replica_groups={{0,1}}
  }
  )";

  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory dump_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  const std::string default_dump_dir =
      absl::StrCat(dump_dir.path(), "/default");
  const std::string no_graph_dump_dir =
      absl::StrCat(dump_dir.path(), "/no_graphs");

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  DebugOptions& debug_options =
      module->mutable_config().mutable_debug_options();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.set_xla_dump_to(default_dump_dir);

  std::vector<float> data_r0(256), data_r1(256);
  for (int i = 0; i < 128; ++i) {
    data_r0[i] = 1.0f;
    data_r0[i + 128] = 2.0f;
    data_r1[i] = 3.0f;
    data_r1[i + 128] = 4.0f;
  }
  Literal input_r0 = LiteralUtil::CreateR1<float>(data_r0);
  Literal input_r1 = LiteralUtil::CreateR1<float>(data_r1);
  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllToAll(result.optimized_module);

  ASSERT_OK_AND_ASSIGN(
      CommandBufferThunkCounts one_shot,
      CountThunksInDump(default_dump_dir, "kCollectiveKernel"));
  EXPECT_EQ(one_shot.in_command_buffer, kNumAllToAlls);
  EXPECT_EQ(one_shot.outside_command_buffer, 0);
  ASSERT_OK_AND_ASSIGN(CommandBufferThunkCounts nccl,
                       CountThunksInDump(default_dump_dir, "kAllToAll"));
  EXPECT_EQ(nccl.in_command_buffer + nccl.outside_command_buffer, 0);

  // An odd number (9) of all-to-alls produces the transposed result.
  ASSERT_EQ(result.results.size(), kNumReplicas);
  std::vector<float> expected_data_r0(256), expected_data_r1(256);
  for (int i = 0; i < 128; ++i) {
    expected_data_r0[i] = 1.0f;
    expected_data_r0[i + 128] = 3.0f;
    expected_data_r1[i] = 2.0f;
    expected_data_r1[i + 128] = 4.0f;
  }
  Literal expected_r0 = LiteralUtil::CreateR1<float>(expected_data_r0);
  Literal expected_r1 = LiteralUtil::CreateR1<float>(expected_data_r1);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r0, result.results[0]))
      << "Mismatch at replica 0";
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r1, result.results[1]))
      << "Mismatch at replica 1";

  // Excluding COLLECTIVES_KERNEL from --xla_gpu_enable_command_buffer must
  // disable command buffer capture for the collective kernels.
  ASSERT_OK_AND_ASSIGN(auto no_graph_module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  DebugOptions& no_graph_debug_options =
      no_graph_module->mutable_config().mutable_debug_options();
  auto* enabled_commands =
      no_graph_debug_options.mutable_xla_gpu_enable_command_buffer();
  enabled_commands->erase(
      std::remove(enabled_commands->begin(), enabled_commands->end(),
                  DebugOptions::COLLECTIVES_KERNEL),
      enabled_commands->end());
  no_graph_debug_options.set_xla_gpu_graph_min_graph_size(1);
  no_graph_debug_options.set_xla_dump_to(no_graph_dump_dir);

  ASSERT_OK_AND_ASSIGN(ExecutionResult no_graph_result,
                       ExecuteReplicated(std::move(no_graph_module), args));
  VerifyOneShotAllToAll(no_graph_result.optimized_module);

  ASSERT_OK_AND_ASSIGN(
      CommandBufferThunkCounts one_shot_no_graph,
      CountThunksInDump(no_graph_dump_dir, "kCollectiveKernel"));
  EXPECT_EQ(one_shot_no_graph.in_command_buffer, 0);
  EXPECT_EQ(one_shot_no_graph.outside_command_buffer, kNumAllToAlls);
  ASSERT_OK_AND_ASSIGN(CommandBufferThunkCounts nccl_no_graph,
                       CountThunksInDump(no_graph_dump_dir, "kAllToAll"));
  EXPECT_EQ(
      nccl_no_graph.in_command_buffer + nccl_no_graph.outside_command_buffer,
      0);
  ASSERT_EQ(no_graph_result.results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r0, no_graph_result.results[0]))
      << "Mismatch at replica 0 without graphs";
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r1, no_graph_result.results[1]))
      << "Mismatch at replica 1 without graphs";
}

// Runs 20 one-shot all-to-alls inside a 20-iteration while loop with WHILE
// command buffers enabled, and verifies that the while thunk and all 20
// collective kernels are placed into command buffers and produce the expected
// result across repeated executions.
TEST_F(AllToAllTest, TwentyAllToAllsInWhileLoopWithCudaGraphs) {
  constexpr int32_t kNumReplicas = 2;
  constexpr int kNumAllToAlls = 20;
  constexpr int kNumIterations = 20;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  std::string body_ops;
  std::string prev = "acc";
  for (int k = 0; k < kNumAllToAlls; ++k) {
    absl::StrAppendFormat(
        &body_ops,
        "    scaled%d = f32[256] multiply(%s, half)\n"
        "    in%d = f32[256] add(scaled%d, p0)\n"
        "    a2a%d = f32[256] all-to-all(in%d), dimensions={0}, "
        "replica_groups={{0,1}}\n"
        "    s0_%d = f32[128] slice(a2a%d), slice={[0:128]}\n"
        "    s1_%d = f32[128] slice(a2a%d), slice={[128:256]}\n"
        "    half_sum%d = f32[128] add(s0_%d, s1_%d)\n"
        "    sum%d = f32[256] concatenate(half_sum%d, half_sum%d), "
        "dimensions={0}\n",
        k, prev, k, k, k, k, k, k, k, k, k, k, k, k, k, k);
    prev = absl::StrCat("sum", k);
  }

  const std::string module_str = absl::StrFormat(
      R"(
  HloModule test

  cond {
    state = (s32[], f32[256], f32[256]) parameter(0)
    i = s32[] get-tuple-element(state), index=0
    limit = s32[] constant(%d)
    ROOT cmp = pred[] compare(i, limit), direction=LT
  }

  body {
    state = (s32[], f32[256], f32[256]) parameter(0)
    i = s32[] get-tuple-element(state), index=0
    acc = f32[256] get-tuple-element(state), index=1
    p0 = f32[256] get-tuple-element(state), index=2
    one = s32[] constant(1)
    next_i = s32[] add(i, one)
    half_scalar = f32[] constant(0.5)
    half = f32[256] broadcast(half_scalar), dimensions={}
%s    ROOT next_state = (s32[], f32[256], f32[256]) tuple(next_i, %s, p0)
  }

  ENTRY test_computation {
    p0 = f32[256] parameter(0)
    zero_i = s32[] constant(0)
    zero_f = f32[] constant(0.0)
    init_acc = f32[256] broadcast(zero_f), dimensions={}
    init = (s32[], f32[256], f32[256]) tuple(zero_i, init_acc, p0)
    loop = (s32[], f32[256], f32[256]) while(init), condition=cond, body=body
    ROOT out = f32[256] get-tuple-element(loop), index=1
  }
  )",
      kNumIterations, body_ops, prev);

  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory dump_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(module_str, kNumReplicas));

  DebugOptions& debug_options =
      module->mutable_config().mutable_debug_options();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::WHILE);
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.set_xla_gpu_enable_while_loop_unrolling(
      DebugOptions::WHILE_LOOP_UNROLLING_NO_UNROLL);
  debug_options.set_xla_dump_to(dump_dir.path());

  Literal input_r0 =
      LiteralUtil::CreateR1<float>(std::vector<float>(256, 1.0f));
  Literal input_r1 =
      LiteralUtil::CreateR1<float>(std::vector<float>(256, 2.0f));
  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllToAll(result.optimized_module);

  ASSERT_OK_AND_ASSIGN(CommandBufferThunkCounts while_counts,
                       CountThunksInDump(dump_dir.path(), "kWhile"));
  EXPECT_EQ(while_counts.in_command_buffer, 1);
  EXPECT_EQ(while_counts.outside_command_buffer, 0);
  ASSERT_OK_AND_ASSIGN(CommandBufferThunkCounts one_shot,
                       CountThunksInDump(dump_dir.path(), "kCollectiveKernel"));
  EXPECT_EQ(one_shot.in_command_buffer, kNumAllToAlls);
  EXPECT_EQ(one_shot.outside_command_buffer, 0);
  ASSERT_OK_AND_ASSIGN(CommandBufferThunkCounts nccl,
                       CountThunksInDump(dump_dir.path(), "kAllToAll"));
  EXPECT_EQ(nccl.in_command_buffer + nccl.outside_command_buffer, 0);

  // 20 iterations * 20 all-to-alls * 3.0 per all-to-all = 1200.0.
  const float expected_val = 3.0f * kNumAllToAlls * kNumIterations;
  Literal expected =
      LiteralUtil::CreateR1<float>(std::vector<float>(256, expected_val));
  ASSERT_EQ(result.results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result.results[i]))
        << "Mismatch at replica " << i << " on first execution";
  }

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> second_results,
                       ExecuteReplicated(result.executable.get(), args));
  ASSERT_EQ(second_results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, second_results[i]))
        << "Mismatch at replica " << i << " on second execution";
  }
}

// 2D shape: f32[16, 64] -> f32[16, 64] (split along dim 1).
TEST_F(AllToAllTest, TwoDimensionalSplitDim1) {
  constexpr int32_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = f32[16,64] parameter(0)
    ROOT all-to-all = f32[16,64] all-to-all(param_0), dimensions={1},
      replica_groups={{0,1}}
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  Literal input_r0 = LiteralUtil::CreateFull<float>({16, 64}, 0.0f);
  Literal input_r1 = LiteralUtil::CreateFull<float>({16, 64}, 0.0f);
  for (int64_t row = 0; row < 16; ++row) {
    for (int64_t col = 0; col < 64; ++col) {
      input_r0.Set<float>({row, col}, (col < 32) ? 10.0f : 20.0f);
      input_r1.Set<float>({row, col}, (col < 32) ? 30.0f : 40.0f);
    }
  }

  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllToAll(result.optimized_module);

  ASSERT_EQ(result.results.size(), kNumReplicas);

  Literal expected_r0 = LiteralUtil::CreateFull<float>({16, 64}, 0.0f);
  Literal expected_r1 = LiteralUtil::CreateFull<float>({16, 64}, 0.0f);
  for (int64_t row = 0; row < 16; ++row) {
    for (int64_t col = 0; col < 64; ++col) {
      expected_r0.Set<float>({row, col}, (col < 32) ? 10.0f : 30.0f);
      expected_r1.Set<float>({row, col}, (col < 32) ? 20.0f : 40.0f);
    }
  }

  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r0, result.results[0]))
      << "Mismatch at replica 0";
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r1, result.results[1]))
      << "Mismatch at replica 1";
}

// 2D shape where the split dimension per rank is 1 (not divisible by
// world_size = 2), so tile_sizes = [1, 64] and staging split_dim falls back to
// dim 1.
TEST_F(AllToAllTest, TwoDimensionalSplitDimNotDivisibleByWorldSize) {
  constexpr int32_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = f32[2,64] parameter(0)
    ROOT all-to-all = f32[2,64] all-to-all(param_0), dimensions={0},
      replica_groups={{0,1}}
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  Literal input_r0 = LiteralUtil::CreateFull<float>({2, 64}, 0.0f);
  Literal input_r1 = LiteralUtil::CreateFull<float>({2, 64}, 0.0f);
  for (int64_t col = 0; col < 64; ++col) {
    input_r0.Set<float>({0, col}, 10.0f);
    input_r0.Set<float>({1, col}, 20.0f);
    input_r1.Set<float>({0, col}, 30.0f);
    input_r1.Set<float>({1, col}, 40.0f);
  }

  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllToAll(result.optimized_module);

  ASSERT_EQ(result.results.size(), kNumReplicas);

  Literal expected_r0 = LiteralUtil::CreateFull<float>({2, 64}, 0.0f);
  Literal expected_r1 = LiteralUtil::CreateFull<float>({2, 64}, 0.0f);
  for (int64_t col = 0; col < 64; ++col) {
    expected_r0.Set<float>({0, col}, 10.0f);
    expected_r0.Set<float>({1, col}, 30.0f);
    expected_r1.Set<float>({0, col}, 20.0f);
    expected_r1.Set<float>({1, col}, 40.0f);
  }

  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r0, result.results[0]))
      << "Mismatch at replica 0";
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r1, result.results[1]))
      << "Mismatch at replica 1";
}

// 1D sub-128-bit transfer (8 bytes per chunk): s32[4] -> s32[4].
TEST_F(AllToAllTest, SubTransactionSize1DS32) {
  constexpr int32_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = s32[4] parameter(0)
    ROOT all-to-all = s32[4] all-to-all(param_0), dimensions={0},
      replica_groups={{0,1}}
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  Literal input_r0 = LiteralUtil::CreateR1<int32_t>({10, 11, 20, 21});
  Literal input_r1 = LiteralUtil::CreateR1<int32_t>({30, 31, 40, 41});
  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllToAll(result.optimized_module);

  Literal expected_r0 = LiteralUtil::CreateR1<int32_t>({10, 11, 30, 31});
  Literal expected_r1 = LiteralUtil::CreateR1<int32_t>({20, 21, 40, 41});
  ASSERT_EQ(result.results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r0, result.results[0]))
      << "Mismatch at replica 0";
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r1, result.results[1]))
      << "Mismatch at replica 1";
}

// 2D shape with unaligned non-split dim: s32[4, 7] -> s32[4, 7].
TEST_F(AllToAllTest, UnalignedNonSplitDim2DS32_4x7) {
  constexpr int32_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = s32[4,7] parameter(0)
    ROOT all-to-all = s32[4,7] all-to-all(param_0), dimensions={0},
      replica_groups={{0,1}}
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  ASSERT_OK_AND_ASSIGN(Literal full_data,
                       MakeFakeLiteral(ShapeUtil::MakeShape(S32, {8, 7})));
  Literal input_r0 = full_data.Slice({0, 0}, {4, 7});
  Literal input_r1 = full_data.Slice({4, 0}, {8, 7});

  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllToAll(result.optimized_module);

  Literal expected_r0(ShapeUtil::MakeShape(S32, {4, 7}));
  ASSERT_OK(expected_r0.CopySliceFrom(full_data, {0, 0}, {0, 0}, {2, 7}));
  ASSERT_OK(expected_r0.CopySliceFrom(full_data, {4, 0}, {2, 0}, {2, 7}));
  Literal expected_r1(ShapeUtil::MakeShape(S32, {4, 7}));
  ASSERT_OK(expected_r1.CopySliceFrom(full_data, {2, 0}, {0, 0}, {2, 7}));
  ASSERT_OK(expected_r1.CopySliceFrom(full_data, {6, 0}, {2, 0}, {2, 7}));

  ASSERT_EQ(result.results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r0, result.results[0]))
      << "Mismatch at replica 0";
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r1, result.results[1]))
      << "Mismatch at replica 1";
}

// 2D shape with larger odd non-split dim: s32[4, 529] -> s32[4, 529].
TEST_F(AllToAllTest, UnalignedNonSplitDim2DS32_4x529) {
  constexpr int32_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = s32[4,529] parameter(0)
    ROOT all-to-all = s32[4,529] all-to-all(param_0), dimensions={0},
      replica_groups={{0,1}}
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  ASSERT_OK_AND_ASSIGN(Literal full_data,
                       MakeFakeLiteral(ShapeUtil::MakeShape(S32, {8, 529})));
  Literal input_r0 = full_data.Slice({0, 0}, {4, 529});
  Literal input_r1 = full_data.Slice({4, 0}, {8, 529});

  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllToAll(result.optimized_module);

  Literal expected_r0(ShapeUtil::MakeShape(S32, {4, 529}));
  ASSERT_OK(expected_r0.CopySliceFrom(full_data, {0, 0}, {0, 0}, {2, 529}));
  ASSERT_OK(expected_r0.CopySliceFrom(full_data, {4, 0}, {2, 0}, {2, 529}));
  Literal expected_r1(ShapeUtil::MakeShape(S32, {4, 529}));
  ASSERT_OK(expected_r1.CopySliceFrom(full_data, {2, 0}, {0, 0}, {2, 529}));
  ASSERT_OK(expected_r1.CopySliceFrom(full_data, {6, 0}, {2, 0}, {2, 529}));

  ASSERT_EQ(result.results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r0, result.results[0]))
      << "Mismatch at replica 0";
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r1, result.results[1]))
      << "Mismatch at replica 1";
}

// Repeated invocations on the same executable to verify double-buffering
// across alternating slots (signal_value & 1).
TEST_F(AllToAllTest, RepeatedInvocationsDoubleBuffering) {
  constexpr int32_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = f32[256] parameter(0)
    ROOT all-to-all = f32[256] all-to-all(param_0), dimensions={0},
      replica_groups={{0,1}}
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  Literal input1_r0 =
      LiteralUtil::CreateR1<float>(std::vector<float>(256, 1.0f));
  Literal input1_r1 =
      LiteralUtil::CreateR1<float>(std::vector<float>(256, 2.0f));
  std::vector<std::vector<Literal*>> args1 = {{&input1_r0}, {&input1_r1}};

  ASSERT_OK_AND_ASSIGN(ExecutionResult result1,
                       ExecuteReplicated(std::move(module), args1));
  VerifyOneShotAllToAll(result1.optimized_module);

  for (int iter = 2; iter <= 4; ++iter) {
    float v00 = static_cast<float>(iter * 10 + 1);
    float v01 = static_cast<float>(iter * 10 + 2);
    float v10 = static_cast<float>(iter * 10 + 3);
    float v11 = static_cast<float>(iter * 10 + 4);
    std::vector<float> d_r0(256), d_r1(256);
    std::vector<float> exp_r0(256), exp_r1(256);
    for (int i = 0; i < 128; ++i) {
      d_r0[i] = v00;
      d_r0[i + 128] = v01;
      d_r1[i] = v10;
      d_r1[i + 128] = v11;
      exp_r0[i] = v00;
      exp_r0[i + 128] = v10;
      exp_r1[i] = v01;
      exp_r1[i + 128] = v11;
    }
    Literal in_r0 = LiteralUtil::CreateR1<float>(d_r0);
    Literal in_r1 = LiteralUtil::CreateR1<float>(d_r1);
    std::vector<std::vector<Literal*>> iter_args = {{&in_r0}, {&in_r1}};

    ASSERT_OK_AND_ASSIGN(
        std::vector<Literal> iter_results,
        ExecuteReplicated(result1.executable.get(), iter_args));
    ASSERT_EQ(iter_results.size(), kNumReplicas);

    Literal expected_r0 = LiteralUtil::CreateR1<float>(exp_r0);
    Literal expected_r1 = LiteralUtil::CreateR1<float>(exp_r1);
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_r0, iter_results[0]))
        << "Mismatch at replica 0 on iteration " << iter;
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_r1, iter_results[1]))
        << "Mismatch at replica 1 on iteration " << iter;
  }
}

class AllToAllTypesTest : public AllToAllTest,
                          public ::testing::WithParamInterface<PrimitiveType> {
};

// Same shape as Large2GpuF32, for every element type supported by the
// one-shot kernel.
TEST_P(AllToAllTypesTest, Large2Gpu) {
  constexpr int32_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = %1$s[8192] parameter(0)
    ROOT all-to-all = %1$s[8192] all-to-all(param_0), dimensions={0},
      replica_groups={{0,1}}
  }
  )";

  const PrimitiveType type = GetParam();
  const std::string module_str = absl::StrFormat(
      kModuleStr, primitive_util::LowercasePrimitiveTypeName(type));
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(module_str, kNumReplicas));

  ASSERT_OK_AND_ASSIGN(Literal full_data,
                       MakeFakeLiteral(ShapeUtil::MakeShape(type, {16384})));
  Literal input_r0 = full_data.Slice({0}, {8192});
  Literal input_r1 = full_data.Slice({8192}, {16384});

  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllToAll(result.optimized_module);

  Literal expected_r0(ShapeUtil::MakeShape(type, {8192}));
  ASSERT_OK(expected_r0.CopySliceFrom(full_data, {0}, {0}, {4096}));
  ASSERT_OK(expected_r0.CopySliceFrom(full_data, {8192}, {4096}, {4096}));
  Literal expected_r1(ShapeUtil::MakeShape(type, {8192}));
  ASSERT_OK(expected_r1.CopySliceFrom(full_data, {4096}, {0}, {4096}));
  ASSERT_OK(expected_r1.CopySliceFrom(full_data, {12288}, {4096}, {4096}));

  ASSERT_EQ(result.results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r0, result.results[0]))
      << "Mismatch at replica 0";
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_r1, result.results[1]))
      << "Mismatch at replica 1";
}

INSTANTIATE_TEST_SUITE_P(
    AllToAllTypes, AllToAllTypesTest,
    ::testing::ValuesIn(gpu::kSupportedAllToAllTypes),
    [](const ::testing::TestParamInfo<PrimitiveType>& info) {
      return std::string(
          primitive_util::LowercasePrimitiveTypeName(info.param));
    });

}  // namespace
}  // namespace xla
