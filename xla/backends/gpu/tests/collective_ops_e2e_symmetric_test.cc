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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::NotNull;

const HloInstruction* FindCollectiveStart(const HloModule* module,
                                          HloOpcode collective_opcode) {
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kAsyncStart &&
          instr->async_wrapped_instruction()->opcode() == collective_opcode) {
        return instr;
      }
    }
  }
  return nullptr;
}

const HloInstruction* FindCollectiveDone(const HloModule* module,
                                         HloOpcode collective_opcode) {
  const HloInstruction* start = FindCollectiveStart(module, collective_opcode);
  if (start == nullptr) {
    return nullptr;
  }
  for (const HloInstruction* user : start->users()) {
    if (user->opcode() == HloOpcode::kAsyncDone) {
      return user;
    }
  }
  return nullptr;
}

std::vector<const HloInstruction*> FindCollectiveStarts(
    const HloModule* module, HloOpcode collective_opcode) {
  std::vector<const HloInstruction*> result;
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kAsyncStart &&
          instr->async_wrapped_instruction()->opcode() == collective_opcode) {
        result.push_back(instr);
      }
    }
  }
  return result;
}

std::string GetAsyncTestName(bool is_async) {
  return is_async ? "async" : "sync";
}

// Parameterized test class for collectives with symmetric memory mode
// requiring NCCL_WIN_COLL_SYMMETRIC registration (ncclPutSignal).
class CollectivesModeOps
    : public CollectiveOpsE2ETestBase,
      public ::testing::WithParamInterface<
          std::tuple<bool, DebugOptions::CollectivesMode>> {
 public:
  CollectivesModeOps()
      : CollectiveOpsE2ETestBase(
            /*memory_size=*/32 * kMB,
            /*collectives_memory_size=*/32 * kMB),
        enable_async_(std::get<0>(GetParam())),
        collectives_mode_(std::get<1>(GetParam())) {}

 protected:
  void SetUp() override {
    CollectiveOpsE2ETestBase::SetUp();
    if (!IsHopperAndHigher()) {
      GTEST_SKIP() << "Test requires Hopper or higher";
    }
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        CollectiveOpsE2ETestBase::GetDebugOptionsForTest();
    if (enable_async_) {
      debug_options.add_xla_disable_hlo_passes(
          "gpu-convert-async-collectives-to-sync");
    } else {
      debug_options.add_xla_gpu_disable_async_collectives(
          DebugOptions::COLLECTIVEPERMUTE);
      debug_options.add_xla_gpu_disable_async_collectives(
          DebugOptions::ALLGATHER);
    }
    debug_options.set_xla_gpu_collective_permute_mode(collectives_mode_);
    debug_options.set_xla_gpu_all_gather_mode(collectives_mode_);
    debug_options.set_xla_gpu_experimental_nccl_window_symmetric(true);
    return debug_options;
  }

  bool enable_async() const { return enable_async_; }
  DebugOptions::CollectivesMode collectives_mode() { return collectives_mode_; }

 private:
  bool enable_async_;
  DebugOptions::CollectivesMode collectives_mode_;
};

std::string GetCollectivesModeTestSuiteName(
    const ::testing::TestParamInfo<
        std::tuple<bool, DebugOptions::CollectivesMode>>& info) {
  return absl::StrCat(GetAsyncTestName(std::get<0>(info.param)), "_symmetric");
}

TEST_P(CollectivesModeOps, AllGather) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[1, 2] broadcast(id), dimensions={}
    a0 = u32[1, 2] constant({{10, 15}})
    a1 = u32[1, 2] add(id2, a0)
    allgather = u32[2, 2] all-gather(a1), dimensions={0}
    ROOT out = u32[4] reshape(allgather)
  }
  )";
  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                       ExecuteReplicated(std::move(module)));

  const HloModule* hlo_module = execution_result.optimized_module;
  if (enable_async()) {
    const HloInstruction* all_gather_start =
        FindCollectiveStart(hlo_module, HloOpcode::kAllGather);
    const HloInstruction* all_gather_done =
        FindCollectiveDone(hlo_module, HloOpcode::kAllGather);
    EXPECT_THAT(all_gather_start, NotNull());
    EXPECT_THAT(all_gather_done, NotNull());
  } else {
    EXPECT_THAT(FindInstruction(hlo_module, HloOpcode::kAllGather), NotNull());
  }

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (const Literal& result : results) {
    LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 15, 11, 16}, result);
  }
}

TEST_P(CollectivesModeOps, AllGatherMixedTypes) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[1, 2] broadcast(id), dimensions={}
    a0 = u32[1, 2] constant({{10, 15}})
    a1 = u32[1, 2] add(id2, a0)
    a2 = f32[1, 2] convert(a1)
    allgather = (u32[2, 2], f32[2,2]) all-gather(a1, a2), dimensions={0}
    gte0 = u32[2,2] get-tuple-element(allgather), index=0
    gte1 = f32[2,2] get-tuple-element(allgather), index=1
    out0 = u32[4] reshape(gte0)
    out1 = f32[4] reshape(gte1)
    ROOT out = (u32[4], f32[4]) tuple(out0, out1)
  }
  )";
  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                       ExecuteReplicated(std::move(module)));

  const HloModule* hlo_module = execution_result.optimized_module;
  if (enable_async()) {
    const HloInstruction* all_gather_start =
        FindCollectiveStart(hlo_module, HloOpcode::kAllGather);
    const HloInstruction* all_gather_done =
        FindCollectiveDone(hlo_module, HloOpcode::kAllGather);
    EXPECT_THAT(all_gather_start, NotNull());
    EXPECT_THAT(all_gather_done, NotNull());
  } else {
    EXPECT_THAT(FindInstruction(hlo_module, HloOpcode::kAllGather), NotNull());
  }

  std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (Literal& result : results) {
    std::vector<Literal> tuple_results = result.DecomposeTuple();
    LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 15, 11, 16},
                                             tuple_results[0]);
    LiteralTestUtil::ExpectR1Equal<float>({10.0, 15.0, 11.0, 16.0},
                                          tuple_results[1]);
  }
}

TEST_P(CollectivesModeOps, CollectivePermute) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    replica = u32[] replica-id()
    ten = u32[] constant(10)
    sum = u32[] add(replica, ten)
    p = u32[2] broadcast(sum), dimensions={}
    permute = u32[2] collective-permute(p), source_target_pairs={{1,0}, {0,1}}
    ROOT copy = u32[2] copy(permute)
  }
  )";
  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                       ExecuteReplicated(std::move(module)));

  const HloModule* hlo_module = execution_result.optimized_module;
  if (enable_async()) {
    const HloInstruction* cp_start =
        FindCollectiveStart(hlo_module, HloOpcode::kCollectivePermute);
    const HloInstruction* cp_done =
        FindCollectiveDone(hlo_module, HloOpcode::kCollectivePermute);
    ASSERT_THAT(cp_start, NotNull());
    ASSERT_THAT(cp_done, NotNull());
  } else {
    EXPECT_THAT(FindInstruction(hlo_module, HloOpcode::kCollectivePermute),
                NotNull());
  }

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 11}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 10}, results[1]);
}

TEST_P(CollectivesModeOps, CollectivePermuteOnParameters) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    p = u32[2] parameter(0)
    ROOT permute = u32[2] collective-permute(p), source_target_pairs={{1,0}, {0,1}}
  }
  )";
  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  auto arg0 = LiteralUtil::CreateR1<uint32_t>({10, 10});
  auto arg1 = LiteralUtil::CreateR1<uint32_t>({11, 11});
  std::vector<std::vector<Literal*>> args = {{&arg0}, {&arg1}};

  ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                       ExecuteReplicated(std::move(module), args));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 11}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 10}, results[1]);
}

TEST_P(CollectivesModeOps, CombinedCollectivePermute) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    replica = u32[] replica-id()
    ten = u32[] constant(10)
    sum = u32[] add(replica, ten)
    replica.1 = u32[2] broadcast(replica), dimensions={}
    sum.1 = u32[2] broadcast(sum), dimensions={}
    permute = (u32[2], u32[2]) collective-permute(replica.1, sum.1), source_target_pairs={{1,0}, {0,1}}
    gte0 = get-tuple-element(permute), index=0
    gte1 = get-tuple-element(permute), index=1
    ROOT concat = u32[4] concatenate(gte0, gte1), dimensions={0}
  }
  )";
  const int64_t kNumReplicas = 2;

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                       ExecuteReplicated(std::move(module)));

  const HloModule* hlo_module = execution_result.optimized_module;
  if (enable_async()) {
    const HloInstruction* cp_start =
        FindCollectiveStart(hlo_module, HloOpcode::kCollectivePermute);
    const HloInstruction* cp_done =
        FindCollectiveDone(hlo_module, HloOpcode::kCollectivePermute);
    EXPECT_THAT(cp_start, NotNull());
    EXPECT_THAT(cp_done, NotNull());
  } else {
    EXPECT_THAT(FindInstruction(hlo_module, HloOpcode::kCollectivePermute),
                NotNull());
  }

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1, 11, 11}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({0, 0, 10, 10}, results[1]);
}

TEST_P(CollectivesModeOps, CollectivePermuteCombiner) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    replica = u32[] replica-id()
    ten = u32[] constant(10)
    sum = u32[] add(replica, ten)
    replica.1 = u32[2] broadcast(replica), dimensions={}
    sum.1 = u32[2] broadcast(sum), dimensions={}
    replica.2 = u32[2] add(replica.1, replica.1)
    permute.0 = u32[2] collective-permute(replica.1), source_target_pairs={{0,1}, {1, 2}, {2, 3}, {3, 0}}
    permute.1 = u32[2] collective-permute(replica.2), source_target_pairs={{0,1}, {1, 2}, {2, 3}, {3, 0}}
    permute.2 = u32[2] collective-permute(sum.1), source_target_pairs={{0,1}, {1, 2}, {2, 3}, {3, 0}}
    ROOT concat = u32[6] concatenate(permute.0, permute.1, permute.2), dimensions={0}
  }
  )";
  const int64_t kNumReplicas = 4;
  if (device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << device_count() << " available)";
  }

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                       ExecuteReplicated(std::move(module)));

  const HloModule* hlo_module = execution_result.optimized_module;
  if (enable_async()) {
    const HloInstruction* cp_start =
        FindCollectiveStart(hlo_module, HloOpcode::kCollectivePermute);
    const HloInstruction* cp_done =
        FindCollectiveDone(hlo_module, HloOpcode::kCollectivePermute);

    EXPECT_THAT(cp_start, NotNull());
    const int cp_start_count =
        FindCollectiveStarts(hlo_module, HloOpcode::kCollectivePermute).size();
    EXPECT_EQ(cp_start_count, 1)
        << "Expected exactly one CollectivePermuteStart instruction";

    EXPECT_EQ(cp_start->operand_count(), 3);
    EXPECT_THAT(cp_done, NotNull());
  } else {
    const HloInstruction* cp =
        FindInstruction(hlo_module, HloOpcode::kCollectivePermute);
    EXPECT_THAT(cp, NotNull());
    EXPECT_EQ(cp->operand_count(), 3);
  }

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({3, 3, 6, 6, 13, 13}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({0, 0, 0, 0, 10, 10}, results[1]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1, 2, 2, 11, 11}, results[2]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({2, 2, 4, 4, 12, 12}, results[3]);
}

TEST_P(CollectivesModeOps, CollectivePermuteInWhileLoop) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  body {
    param = (u32[], u32[2]) parameter(0)
    i = u32[] get-tuple-element(param), index=0
    data = u32[2] get-tuple-element(param), index=1
    replica = u32[] replica-id()
    replica_bcast = u32[2] broadcast(replica), dimensions={}
    data_plus_rid = u32[2] add(data, replica_bcast)
    permuted = u32[2] collective-permute(data_plus_rid), source_target_pairs={{0,1},{1,0}}
    one = u32[] constant(1)
    i_next = u32[] add(i, one)
    ROOT tuple = (u32[], u32[2]) tuple(i_next, permuted)
  }

  cond {
    param = (u32[], u32[2]) parameter(0)
    i = u32[] get-tuple-element(param), index=0
    limit = u32[] constant(4)
    ROOT lt = pred[] compare(i, limit), direction=LT
  }

  ENTRY test_computation {
    replica = u32[] replica-id()
    ten = u32[] constant(10)
    init_data = u32[] add(replica, ten)
    bcast = u32[2] broadcast(init_data), dimensions={}
    zero = u32[] constant(0)
    init = (u32[], u32[2]) tuple(zero, bcast)
    loop = (u32[], u32[2]) while(init), condition=cond, body=body
    ROOT result = u32[2] get-tuple-element(loop), index=1
  }
  )";
  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                       ExecuteReplicated(std::move(module)));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({12, 12}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({13, 13}, results[1]);
}

TEST_P(CollectivesModeOps, CombinedCollectivePermuteInWhileLoop) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  body {
    param = (u32[], u32[2], f32[2]) parameter(0)
    i = u32[] get-tuple-element(param), index=0
    data0 = u32[2] get-tuple-element(param), index=1
    data1 = f32[2] get-tuple-element(param), index=2
    permuted = (u32[2], f32[2]) collective-permute(data0, data1),
      source_target_pairs={{0,1},{1,0}}
    perm0 = u32[2] get-tuple-element(permuted), index=0
    perm1 = f32[2] get-tuple-element(permuted), index=1
    one = u32[] constant(1)
    i_next = u32[] add(i, one)
    ROOT tuple = (u32[], u32[2], f32[2]) tuple(i_next, perm0, perm1)
  }

  cond {
    param = (u32[], u32[2], f32[2]) parameter(0)
    i = u32[] get-tuple-element(param), index=0
    limit = u32[] constant(4)
    ROOT lt = pred[] compare(i, limit), direction=LT
  }

  ENTRY test_computation {
    replica = u32[] replica-id()
    ten = u32[] constant(10)
    init_u = u32[] add(replica, ten)
    bcast_u = u32[2] broadcast(init_u), dimensions={}
    init_f = f32[] convert(init_u)
    bcast_f = f32[2] broadcast(init_f), dimensions={}
    zero = u32[] constant(0)
    init = (u32[], u32[2], f32[2]) tuple(zero, bcast_u, bcast_f)
    loop = (u32[], u32[2], f32[2]) while(init), condition=cond, body=body
    result0 = u32[2] get-tuple-element(loop), index=1
    result1 = f32[2] get-tuple-element(loop), index=2
    ROOT out = (u32[2], f32[2]) tuple(result0, result1)
  }
  )";
  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                       ExecuteReplicated(std::move(module)));

  std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  std::vector<Literal> r0 = results[0].DecomposeTuple();
  std::vector<Literal> r1 = results[1].DecomposeTuple();
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 10}, r0[0]);
  LiteralTestUtil::ExpectR1Equal<float>({10.0, 10.0}, r0[1]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 11}, r1[0]);
  LiteralTestUtil::ExpectR1Equal<float>({11.0, 11.0}, r1[1]);
}

INSTANTIATE_TEST_SUITE_P(
    CollectivesModeOps, CollectivesModeOps,
    ::testing::Combine(
        ::testing::Bool(),
        ::testing::Values(DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY)),
    GetCollectivesModeTestSuiteName);

}  // namespace
}  // namespace xla
