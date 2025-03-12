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

#include "xla/backends/gpu/collectives/nvshmem_collectives.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/status_macros.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/util/command_line_flags.h"

static constexpr int kTestInitialization = 0;
static constexpr int kTestUserBufferWithNvshmemMalloc = 1;

namespace xla::gpu {
namespace {

using ::testing::NotNull;
using ::testing::SizeIs;

void RunMultiprocessTest(int test_id) {
  const int num_nodes = 2;
  tsl::SubProcess child[num_nodes];
  for (int node_id = 0; node_id < num_nodes; ++node_id) {
    std::vector<std::string> argv;
    argv.push_back("nvshmem_test");
    argv.push_back(absl::StrFormat("--test_id=%d", test_id));
    argv.push_back(absl::StrFormat("--node_id=%d", node_id));
    argv.push_back(absl::StrFormat("--num_nodes=%d", num_nodes));
    child[node_id].SetProgram("/proc/self/exe", argv);
    child[node_id].SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
    child[node_id].SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
    ASSERT_TRUE(child[node_id].Start()) << "node " << node_id;
  }
  for (int node_id = 0; node_id < num_nodes; ++node_id) {
    std::string stdout_str;
    std::string stderr_str;
    int child_status =
        child[node_id].Communicate(nullptr, &stdout_str, &stderr_str);
    EXPECT_EQ(child_status, 0) << " node " << node_id << "\nstdout:\n"
                               << stdout_str << "\nstderr:\n"
                               << stderr_str;
  }
}

HloInstruction* FindInstruction(const HloModule* module, HloOpcode opcode) {
  for (const HloComputation* computation : module->computations()) {
    if (HloInstruction* instruction =
            hlo_query::FindInstruction(computation, opcode)) {
      return instruction;
    }
  }
  return nullptr;
}

// Tests that NVSHMEM library can be loaded and initialized.
TEST(NvshmemTest, Initialization) { RunMultiprocessTest(kTestInitialization); }

absl::Status InitializationTestBody(const int node_id, const int num_nodes) {
  std::unique_ptr<xla::DistributedRuntimeService> service;
  if (node_id == 0) {
    xla::CoordinationServiceImpl::Options service_options;
    service_options.num_nodes = num_nodes;
    TF_ASSIGN_OR_RETURN(service, xla::GetDistributedRuntimeService(
                                     "[::]:12345", service_options));
  }

  xla::DistributedRuntimeClient::Options distributed_options;
  distributed_options.node_id = node_id;
  distributed_options.init_timeout = absl::Seconds(120);
  auto distributed_client =
      GetDistributedRuntimeClient("127.0.0.1:12345", distributed_options);
  TF_QCHECK_OK(distributed_client->Connect());
  auto kv_store =
      GetDistributedKeyValueStore(distributed_client, /*key_prefix=*/"gpu:");

  NvshmemCollectives::Default()->SetEnvInfo(node_id, num_nodes, 1, kv_store);
  cudaSetDevice(node_id);
  TF_ASSIGN_OR_RETURN(void* ptr, NvshmemCollectives::Default()->Allocate(1024));
  TF_RET_CHECK(ptr != nullptr);
  TF_RETURN_IF_ERROR(NvshmemCollectives::Default()->Deallocate(ptr));
  return absl::OkStatus();
}

// Tests that NCCL user buffer collectives can use nvshmem_malloc memory.
TEST(NvshmemTest, UserBufferWithNvshmemMalloc) {
  RunMultiprocessTest(kTestUserBufferWithNvshmemMalloc);
}

absl::Status UserBufferWithNvshmemMallocTestBody(const int node_id,
                                                 const int num_nodes) {
  tsl::setenv("XLA_FLAGS", "--xla_gpu_experimental_enable_nvshmem=true",
              /*overwrite=*/true);
  const absl::string_view kModuleStr = R"(
      HloModule test

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        ROOT all-reduce = u32[] all-reduce(id), to_apply=apply_op
      }
    )";
  std::unique_ptr<xla::DistributedRuntimeService> service;
  if (node_id == 0) {
    xla::CoordinationServiceImpl::Options service_options;
    service_options.num_nodes = num_nodes;
    TF_ASSIGN_OR_RETURN(service, xla::GetDistributedRuntimeService(
                                     "[::]:12346", service_options));
  }

  xla::DistributedRuntimeClient::Options distributed_options;
  distributed_options.node_id = node_id;
  distributed_options.init_timeout = absl::Seconds(120);
  auto distributed_client =
      GetDistributedRuntimeClient("127.0.0.1:12346", distributed_options);
  TF_QCHECK_OK(distributed_client->Connect());
  GpuClientOptions client_options;
  client_options.node_id = node_id;
  client_options.allowed_devices = {node_id};
  client_options.num_nodes = num_nodes;
  client_options.kv_store =
      GetDistributedKeyValueStore(distributed_client, /*key_prefix=*/"gpu:");
  ;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                      GetStreamExecutorGpuClient(client_options));
  xla::CompileOptions options;
  options.executable_build_options.set_num_replicas(num_nodes);
  options.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_experimental_enable_nvshmem(true);
  options.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_enable_nccl_user_buffers(true);

  TF_ASSIGN_OR_RETURN(auto hlo_module,
                      ParseAndReturnUnverifiedModule(kModuleStr, {}));
  xla::XlaComputation xla_computation(hlo_module->ToProto());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtLoadedExecutable> executable,
                      client->Compile(xla_computation, options));

  // Verify that the collective memory space is used.
  TF_ASSIGN_OR_RETURN(auto modules, executable->GetHloModules());
  HloInstruction* all_reduce_start =
      FindInstruction(modules[0].get(), HloOpcode::kAllReduceStart);
  EXPECT_THAT(all_reduce_start, NotNull());
  EXPECT_EQ(all_reduce_start->shape().layout().memory_space(), 1);
  EXPECT_THAT(all_reduce_start->operands(), SizeIs(1));
  const HloInstruction* input = all_reduce_start->operand(0);
  EXPECT_EQ(input->shape().layout().memory_space(), 1);

  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results,
      executable->Execute(/*argument_handles=*/{{}}, /*options=*/{}));
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].size(), 1);
  TF_ASSIGN_OR_RETURN(auto literal, results[0][0]->ToLiteralSync());
  if (node_id == 0) {
    LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 15, 11, 16}, *literal);
  } else if (node_id == 1) {
    LiteralTestUtil::ExpectR1Equal<uint32_t>({20, 25, 21, 26}, *literal);
  }
  tsl::unsetenv("XLA_FLAGS");
  return absl::OkStatus();
}

}  // namespace
}  // namespace xla::gpu

int main(int argc, char* argv[]) {
  // Save name of binary so that it may invoke itself.
  int test_id = -1;
  int node_id = -1;
  int num_nodes = -1;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("test_id", &test_id, "Which test to run."),
      tsl::Flag("node_id", &node_id, "Node ID for multiprocess tests."),
      tsl::Flag("num_nodes", &num_nodes,
                "Number of nodes for multiprocess tests."),
  };
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);
  testing::InitGoogleTest(&argc, argv);
  if (test_id >= 0) {
    absl::Status result;
    if (test_id == kTestInitialization) {
      result = xla::gpu::InitializationTestBody(node_id, num_nodes);
    } else if (test_id == kTestUserBufferWithNvshmemMalloc) {
      result =
          xla::gpu::UserBufferWithNvshmemMallocTestBody(node_id, num_nodes);
    } else {
      LOG(ERROR) << "Unknown test_id: " << test_id;
      return absl::StatusCode::kInvalidArgument;
    }
    if (!result.ok()) {
      LOG(ERROR) << result;
    }
    return result.raw_code();
  }
  return RUN_ALL_TESTS();
}
