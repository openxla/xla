/* Copyright 2024 The OpenXLA Authors.

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

#include <memory>

#include <gtest/gtest.h>
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/debug_options_flags.h"
#include "xla/tests/test_macros.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "xla/backends/gpu/collectives/nvshmem_collectives.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

namespace xla {
namespace {

// Tests that NVSHMEM library can be loaded and initialized.

static const char* test_binary_name;

TEST(NvshmemTest, Initialization) {
  const int num_nodes = 2;
  tsl::SubProcess child[num_nodes];
  for (int node_id = 0; node_id < num_nodes; ++node_id) {
    std::vector<std::string> argv;
    argv.push_back(test_binary_name);
    argv.push_back(absl::StrFormat("--node_id=%d", node_id));
    argv.push_back(absl::StrFormat("--num_nodes=%d", num_nodes));
    child[node_id].SetProgram(test_binary_name, argv);
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

absl::Status InitializationTestBody(const int node_id, const int num_nodes) {
  std::unique_ptr<xla::DistributedRuntimeService> service;
  if (node_id == 0) {
    TF_ASSIGN_OR_RETURN(service,
                        xla::GetDistributedRuntimeService(
                            "[::]:12345", xla::CoordinationServiceImpl::Options{
                                              .num_nodes = num_nodes}));
  }

  xla::DistributedRuntimeClient::Options distributed_options;
  distributed_options.node_id = node_id;
  distributed_options.init_timeout = absl::Seconds(120);
  auto distributed_client =
      GetDistributedRuntimeClient("127.0.0.1:12345", distributed_options);
  TF_QCHECK_OK(distributed_client->Connect());
  auto kv_store =
      GetDistributedKeyValueStore(distributed_client, /*key_prefix=*/"gpu:");
  std::weak_ptr<KeyValueStoreInterface> kv_store_weak_ptr = kv_store;
  auto store_get_fn =
      [kv_store_weak_ptr](std::string_view key) -> absl::StatusOr<std::string> {
    if (std::shared_ptr<KeyValueStoreInterface> kv_store =
            kv_store_weak_ptr.lock()) {
      return kv_store->Get(key, absl::Minutes(10));
    }
    return absl::InternalError(
        "KV store is not available for nvshmem initialization");
  };
  auto store_set_fn = [kv_store_weak_ptr](
                          std::string_view key,
                          std::string_view value) -> absl::Status {
    if (std::shared_ptr<KeyValueStoreInterface> kv_store =
            kv_store_weak_ptr.lock()) {
      return kv_store->Set(key, value);
    }
    return absl::InternalError(
        "KV store is not available for nvshmem initialization");
  };
  xla::gpu::NvshmemCollectives::Default()->SetEnvInfo(
      node_id, num_nodes, 1, store_get_fn, store_set_fn);
  cudaSetDevice(node_id);
  TF_ASSIGN_OR_RETURN(void* ptr,
                      xla::gpu::NvshmemCollectives::Default()->Allocate(1024));
  TF_RET_CHECK(ptr != nullptr);
  TF_RETURN_IF_ERROR(xla::gpu::NvshmemCollectives::Default()->Deallocate(ptr));
  return absl::OkStatus();
}

}  // namespace

}  // namespace xla

int main(int argc, char* argv[]) {
  // Save name of binary so that it may invoke itself.
  xla::test_binary_name = argv[0];
  int node_id = -1;
  int num_nodes = -1;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("node_id", &node_id, "Node ID for Initialization test."),
      tsl::Flag("num_nodes", &num_nodes,
                "Number of nodes for Initialization test."),
  };
  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);
  testing::InitGoogleTest(&argc, argv);
  if (node_id >= 0) {
    return xla::InitializationTestBody(node_id, num_nodes).raw_code();
  }
  return RUN_ALL_TESTS();
}
