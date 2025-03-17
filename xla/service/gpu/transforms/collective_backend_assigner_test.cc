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
#include <stdlib.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/transforms/collective_backend_assigner.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/test.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/mem.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace gpu {
namespace {
using ::testing::ElementsAreArray;
class NvshmemGpuCollectivesTest : public ::testing::Test {};

static const char* test_binary_name;

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileExecutable(
    absl::string_view program, xla::PjRtClient& client,
    xla::CompileOptions compile_options = xla::CompileOptions()) {
  TF_ASSIGN_OR_RETURN(auto hlo_module,
                      ParseAndReturnUnverifiedModule(program, {}));

  xla::XlaComputation xla_computation(hlo_module->ToProto());
  return client.Compile(xla_computation, compile_options);
}

void RunCollectiveTest(const std::string& hlo_module, int num_ranks) {
  VLOG(1) << "Starting test with " << num_ranks << " ranks";
  std::vector<tsl::SubProcess> child(num_ranks);
  for (int rank_id = 0; rank_id < num_ranks; ++rank_id) {
    std::vector<std::string> argv;
    argv.push_back(test_binary_name);
    argv.push_back(absl::StrFormat("--rank_id=%d", rank_id));
    argv.push_back(absl::StrFormat("--num_ranks=%d", num_ranks));
    argv.push_back("--hlo_module=" + hlo_module);
    child[rank_id].SetProgram(test_binary_name, argv);
    child[rank_id].SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
    child[rank_id].SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
    VLOG(1) << "Starting child process for rank " << rank_id;
    ASSERT_TRUE(child[rank_id].Start()) << "rank " << rank_id;
  }
  for (int rank_id = 0; rank_id < num_ranks; ++rank_id) {
    std::string stdout_str;
    std::string stderr_str;
    int child_status =
        child[rank_id].Communicate(nullptr, &stdout_str, &stderr_str);
    EXPECT_EQ(child_status, 0) << " rank " << rank_id << "\nstdout:\n"
                               << stdout_str << "\nstderr:\n"
                               << stderr_str;
  }
  VLOG(1) << "Test completed";
}

TEST(CollectiveBackendAssignerTest, AllReduceSmallMessage) {
  const int num_ranks = 2;

  const std::string kProgram = R"(
    HloModule module

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY main {
      p0 = f32[1024,1024] parameter(0)
      ROOT all-reduce = f32[1024,1024] all-reduce(p0), to_apply=add, replica_groups={{0,1}}, channel_id=1
    })";

  RunCollectiveTest(kProgram, num_ranks);
}

TEST(CollectiveBackendAssignerTest, AllReduceLargeMessage) {
  const int num_ranks = 2;

  const std::string kProgram = R"(
    HloModule module

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY main {
      p0 = f32[8192,8192] parameter(0)
      ROOT all-reduce = f32[8192,8192] all-reduce(p0), to_apply=add, replica_groups={{0,1}}, channel_id=1
    })";

  RunCollectiveTest(kProgram, num_ranks);
}

TEST(CollectiveBackendAssignerTest, CollectivePermuteSmallMessage) {
  const int num_ranks = 2;

  const std::string kProgram = R"(
    HloModule module

    ENTRY main {
      p = u32[1024,1024] parameter(0)
      ROOT permute = u32[1024,1024] collective-permute(p), channel_id=1,
        source_target_pairs={{0,1},{1,0}}
    })";

  RunCollectiveTest(kProgram, num_ranks);
}

TEST(CollectiveBackendAssignerTest, CollectivePermuteLargeMessage) {
  const int num_ranks = 2;

  const std::string kProgram = R"(
    HloModule module

    ENTRY main {
      p = u32[8192,8192] parameter(0)
      ROOT permute = u32[8192,8192] collective-permute(p), channel_id=1,
        source_target_pairs={{0,1},{1,0}}
    })";

  RunCollectiveTest(kProgram, num_ranks);
}

TEST(CollectiveBackendAssignerTest, PreexistingBackendConfig) {
  const int num_ranks = 2;

  const std::string kProgram = R"(
    HloModule module

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY main {
      p0 = f32[1024,1024] parameter(0)
      ROOT all-reduce = f32[1024,1024] all-reduce(p0), to_apply=add, replica_groups={{0,1}}, channel_id=1, backend_config={"collective_backend_config":{"backend":"NCCL"}}
    })";

  RunCollectiveTest(kProgram, num_ranks);
}

TEST(CollectiveBackendAssignerTest, DetectCommunicationType) {
  // Test intranode communication (all devices on same node)
  std::vector<ReplicaGroup> intranode_groups;
  ReplicaGroup group1;
  group1.add_replica_ids(0);
  group1.add_replica_ids(1);
  group1.add_replica_ids(2);
  intranode_groups.push_back(group1);

  EXPECT_FALSE(CollectiveBackendAssigner::HasInternodeCommunication(
      intranode_groups, 4));

  // Test internode communication (devices across nodes)
  std::vector<ReplicaGroup> internode_groups;
  ReplicaGroup group2;
  group2.add_replica_ids(0);
  group2.add_replica_ids(4);
  group2.add_replica_ids(8);
  internode_groups.push_back(group2);

  EXPECT_TRUE(CollectiveBackendAssigner::HasInternodeCommunication(
      internode_groups, 4));

  // Test mixed intra/inter node communication
  std::vector<ReplicaGroup> mixed_groups;
  ReplicaGroup group3;
  group3.add_replica_ids(0);
  group3.add_replica_ids(1);
  group3.add_replica_ids(4);
  mixed_groups.push_back(group3);

  EXPECT_TRUE(
      CollectiveBackendAssigner::HasInternodeCommunication(mixed_groups, 4));
}

absl::Status CollectiveBackendAssignerTestBody(int rank_id, int num_ranks,
                                               absl::string_view hlo_module) {
  VLOG(1) << "Starting CollectiveBackendAssignerTestBody";

  std::unique_ptr<xla::DistributedRuntimeService> service;
  if (rank_id == 0) {
    TF_ASSIGN_OR_RETURN(service,
                        xla::GetDistributedRuntimeService(
                            "[::]:12345", xla::CoordinationServiceImpl::Options{
                                              .num_nodes = num_ranks}));
  }

  xla::DistributedRuntimeClient::Options distributed_options;
  distributed_options.node_id = rank_id;
  distributed_options.init_timeout = absl::Seconds(120);
  auto distributed_client =
      GetDistributedRuntimeClient("127.0.0.1:12345", distributed_options);
  TF_QCHECK_OK(distributed_client->Connect());

  GpuClientOptions client_options;
  client_options.node_id = rank_id;
  client_options.allowed_devices = {rank_id};
  client_options.num_nodes = num_ranks;
  client_options.kv_store = GetDistributedKeyValueStore(distributed_client,
                                                        /*key_prefix=*/"gpu:");
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                      GetStreamExecutorGpuClient(client_options));

  xla::CompileOptions options;
  options.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_experimental_enable_nvshmem(true);
  options.executable_build_options.set_use_spmd_partitioning(false);
  options.executable_build_options.set_num_replicas(num_ranks);

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      ParseAndReturnUnverifiedModule(hlo_module, {}));

  TF_ASSIGN_OR_RETURN(auto executable,
                      CompileExecutable(hlo_module, *client, options));

  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> hlo_modules,
                      executable->GetHloModules());
  const auto* entry = hlo_modules[0]->entry_computation();
  VLOG(1) << "Entry computation: " << entry->ToString();
  for (const auto* instr : entry->instructions()) {
    if (instr->opcode() == HloOpcode::kAllReduceStart ||
        instr->opcode() == HloOpcode::kCollectivePermuteStart) {
      if (instr->has_backend_config()) {
        TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                            instr->backend_config<GpuBackendConfig>());
        const auto& collective_backend_config =
            gpu_config.collective_backend_config();

        // For small tensors, expect NVSHMEM, for large tensors expect NCCL
        const Shape& shape_to_check =
            instr->opcode() == HloOpcode::kCollectivePermuteStart
                ? instr->operand(0)->shape()
                : instr->shape();
        const bool is_large = shape_to_check.dimensions(0) >= 8192;
        const auto expected_backend = is_large
                                          ? CollectiveBackendConfig::NCCL
                                          : CollectiveBackendConfig::NVSHMEM;
        if (collective_backend_config.backend() != expected_backend) {
          return absl::InvalidArgumentError(
              absl::StrFormat("backend config does not specify expected "
                              "backend. Got: %s, Expected: %s",
                              CollectiveBackendConfig_CollectiveBackend_Name(
                                  collective_backend_config.backend()),
                              CollectiveBackendConfig_CollectiveBackend_Name(
                                  expected_backend)));
        }
      } else {
        return absl::InvalidArgumentError("backend config is missing");
      }
      break;
    }
  }

  return absl::OkStatus();
}
}  // namespace
}  // namespace gpu
}  // namespace xla

int main(int argc, char* argv[]) {
  // Save name of binary so that it may invoke itself.
  xla::gpu::test_binary_name = argv[0];
  int rank_id = -1;
  int num_ranks = -1;
  std::string hlo_module;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("rank_id", &rank_id, "Rank ID for nvshmem collective test."),
      tsl::Flag("num_ranks", &num_ranks,
                "Total number of ranks for nvshmem collective test."),
      tsl::Flag("hlo_module", &hlo_module, "HLO module to compile and run."),
  };
  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);
  testing::InitGoogleTest(&argc, argv);
  if (rank_id >= 0) {
    return xla::gpu::CollectiveBackendAssignerTestBody(rank_id, num_ranks,
                                                       hlo_module)
        .raw_code();
  }
  return RUN_ALL_TESTS();
}