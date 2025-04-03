/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"

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
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/service/platform_util.h"
#include "xla/status_macros.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/util/command_line_flags.h"

namespace xla {
namespace {

using ::testing::NotNull;
using ::testing::SizeIs;

HloInstruction* FindInstruction(const HloModule* module, HloOpcode opcode) {
  for (const HloComputation* computation : module->computations()) {
    if (HloInstruction* instruction =
            hlo_query::FindInstruction(computation, opcode)) {
      return instruction;
    }
  }
  return nullptr;
}

// Register a mock "mosaic_gpu" custom call op for NvshmemMemoryTest, since
// mosaic_gpu is defined in JAX and won't be available to the unit test.
static absl::Status MockMosaicGpu(ffi::AnyBuffer arg,
                                  ffi::Result<ffi::AnyBuffer> ret,
                                  absl::string_view module) {
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kMockMosaicGpu, MockMosaicGpu,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Attr<absl::string_view>("module"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "mosaic_gpu_v2",
                         PlatformUtil::CanonicalPlatformName("GPU").value(),
                         kMockMosaicGpu);

// Verify that the client can initialize NVSHMEM and that buffers used by
// mosaic_gpu_v2 custom calls are assigned to the collective memory space.
TEST(StreamExecutorGpuClientNvshmemTest, NvshmemMemoryTest) {
  tsl::setenv("XLA_FLAGS", "--xla_gpu_experimental_enable_nvshmem=true",
              /*overwrite=*/true);
  static constexpr char const* kProgram = R"(
    HloModule ffi_handler
    ENTRY main {
      param = s32[1,4]{1,0} parameter(0)
      reshape = s32[4]{0} reshape(param)
      ROOT %custom-call = s32[4] custom-call(param),
          custom_call_target="mosaic_gpu_v2",
          api_version=API_VERSION_TYPED_FFI,
          backend_config={"custom_call_backend_config": {"attributes": "{module = \"nvshmem\"}"}}
    })";
  // Nvshmem requires one gpu per process.
  GpuClientOptions client_options;
  client_options.node_id = 0;
  client_options.allowed_devices = {0};
  client_options.num_nodes = 1;
  client_options.kv_store = std::make_shared<InMemoryKeyValueStore>();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(client_options));
  xla::CompileOptions options;
  options.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_experimental_enable_nvshmem(true);

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(kProgram, {}));
  xla::XlaComputation xla_computation(hlo_module->ToProto());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtLoadedExecutable> executable,
                          client->CompileAndLoad(xla_computation, options));

  std::vector<int32_t> data{1, 2, 3, 4};
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {1, 4},
                                                    /*minor_to_major=*/{1, 0});
  shape.mutable_layout()->set_memory_space(Layout::kDefaultMemorySpace);

  PjRtDevice* const device = client->addressable_devices()[0];
  TF_EXPECT_OK(device->default_memory_space());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> input,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, *device->default_memory_space(),
          /*device_layout=*/nullptr));
  EXPECT_EQ(input->memory_space()->kind(), "device");

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<absl::string_view>> memory_kinds,
      executable->GetOutputMemoryKinds());
  EXPECT_EQ(memory_kinds.size(), 1);
  EXPECT_EQ(memory_kinds[0].size(), 1);
  EXPECT_EQ(memory_kinds[0][0], "device");

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result,
      executable->Execute({{input.get()}}, ExecuteOptions()));
  std::vector<std::unique_ptr<xla::PjRtBuffer>>& result_buffers = result[0];
  EXPECT_EQ(result_buffers[0]->memory_space()->kind(), "device");
  Shape result_shape = result_buffers[0]->on_device_shape();
  int64_t memory_space = result_shape.layout().memory_space();
  EXPECT_EQ(memory_space, 1);
  tsl::unsetenv("XLA_FLAGS");
}

TEST(StreamExecutorGpuClientNvshmemTest, UserBufferWithNvshmemMalloc) {
  const int num_nodes = 2;
  tsl::SubProcess child[num_nodes];
  for (int node_id = 0; node_id < num_nodes; ++node_id) {
    std::vector<std::string> argv;
    argv.push_back("nvhsmem_test");
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
                      client->CompileAndLoad(xla_computation, options));

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
  return absl::OkStatus();
}

}  // namespace
}  // namespace xla

int main(int argc, char* argv[]) {
  int node_id = -1;
  int num_nodes = -1;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("node_id", &node_id, "Node ID for multiprocess tests."),
      tsl::Flag("num_nodes", &num_nodes,
                "Number of nodes for multiprocess tests."),
  };
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);
  testing::InitGoogleTest(&argc, argv);
  if (node_id >= 0) {
    absl::Status result =
        xla::UserBufferWithNvshmemMallocTestBody(node_id, num_nodes);
    if (!result.ok()) {
      LOG(ERROR) << result;
    }
    return result.raw_code();
  }
  return RUN_ALL_TESTS();
}
