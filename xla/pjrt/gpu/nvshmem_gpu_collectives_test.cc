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

std::string GetDataTypeString(xla::PrimitiveType data_type) {
  switch (data_type) {
    case xla::PrimitiveType::F32:
      return "f32";
    case xla::PrimitiveType::F64:
      return "f64";
    case xla::PrimitiveType::BF16:
      return "bf16";
    case xla::PrimitiveType::F16:
      return "f16";
    case xla::PrimitiveType::U32:
      return "u32";
    case xla::PrimitiveType::U64:
      return "u64";
    case xla::PrimitiveType::S32:
      return "s32";
    case xla::PrimitiveType::S64:
      return "s64";
    default:
      throw absl::InvalidArgumentError("Invalida data type.");
  }
}
TEST(NvshmemGpuCollectivesTest, NvshmemAllReduceFloat) {
  int num_ranks = 2;

  tsl::SubProcess child[num_ranks];
  for (int rank_id = 0; rank_id < num_ranks; ++rank_id) {
    std::vector<std::string> argv;
    argv.push_back(test_binary_name);
    argv.push_back(absl::StrFormat("--rank_id=%d", rank_id));
    argv.push_back(absl::StrFormat("--num_ranks=%d", num_ranks));
    argv.push_back(
        absl::StrFormat("--input_data_type=%d", (int)xla::PrimitiveType::F32));
    child[rank_id].SetProgram(test_binary_name, argv);
    child[rank_id].SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
    child[rank_id].SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
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
}

absl::Status NvshmemCollectiveTestBody(int rank_id, int num_ranks,
                                       int input_data_type) {
  LOG(ERROR) << "NvshmemCollectiveTestBody";

  xla::PrimitiveType data_type = (xla::PrimitiveType)input_data_type;
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
  options.executable_build_options.set_run_backend_only(true);
  options.executable_build_options.set_use_spmd_partitioning(false);
  options.executable_build_options.set_num_replicas(num_ranks);
  std::string data_type_str = GetDataTypeString(data_type);
  const std::string kProgram =
      absl::StrFormat(R"(
    HloModule NvshmemAr
      apply_op {
        x = %s[] parameter(0)
        y = %s[] parameter(1)
        ROOT apply_op = %s[] add(x, y)
      }

      ENTRY test_computation {
        id = %s[] constant(10)
        start = %s[] all-reduce-start(id), to_apply=apply_op, backend_config={"collective_backend_config":{"backend":"NVSHMEM"}}
        ROOT done = %s[] all-reduce-done(start)
      })",
                      data_type_str, data_type_str, data_type_str,
                      data_type_str, data_type_str, data_type_str);

  TF_ASSIGN_OR_RETURN(auto executable,
                      CompileExecutable(kProgram, *client, options));
  TF_ASSIGN_OR_RETURN(auto result, executable->Execute({{}}, ExecuteOptions()));
  std::vector<std::unique_ptr<xla::PjRtBuffer>>& result_buffers = result[0];
  TF_ASSIGN_OR_RETURN(std::shared_ptr<xla::Literal> literal,
                      result_buffers[0]->ToLiteralSync());
  switch (data_type) {
    case xla::PrimitiveType::F32: {
      std::vector<float> ref_data{20};
      TF_RET_CHECK(literal->data<float>()[0] == ref_data[0]);
      break;
    }
    case xla::PrimitiveType::F64: {
      std::vector<double> ref_data{20};
      TF_RET_CHECK(literal->data<double>()[0] == ref_data[0]);
      break;
    }
    case xla::PrimitiveType::BF16: {
      std::vector<Eigen::bfloat16> ref_data{20};
      TF_RET_CHECK(literal->data<Eigen::bfloat16>()[0] == ref_data[0]);
      break;
    }
    case xla::PrimitiveType::F16: {
      std::vector<Eigen::half> ref_data{20};
      TF_RET_CHECK(literal->data<Eigen::half>()[0] == ref_data[0]);
      break;
    }
    case xla::PrimitiveType::U32: {
      std::vector<uint32_t> ref_data{20};
      TF_RET_CHECK(literal->data<uint32_t>()[0] == ref_data[0]);
      break;
    }
    case xla::PrimitiveType::U64: {
      std::vector<uint64_t> ref_data{20};
      TF_RET_CHECK(literal->data<uint64_t>()[0] == ref_data[0]);
      break;
    }
    case xla::PrimitiveType::S32: {
      std::vector<int32_t> ref_data{20};
      TF_RET_CHECK(literal->data<int32_t>()[0] == ref_data[0]);
      break;
    }
    case xla::PrimitiveType::S64: {
      std::vector<int64_t> ref_data{20};
      TF_RET_CHECK(literal->data<int64_t>()[0] == ref_data[0]);
      break;
    }
    default:
      throw absl::InvalidArgumentError("Invalida data type.");
  }

  return absl::OkStatus();
}
}  // namespace
}  // namespace xla

int main(int argc, char* argv[]) {
  // Save name of binary so that it may invoke itself.
  xla::test_binary_name = argv[0];
  int rank_id = -1;
  int num_ranks = -1;
  int input_data_type = (int)xla::PrimitiveType::F32;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("rank_id", &rank_id, "Rank ID for nvshmem collective test."),
      tsl::Flag("num_ranks", &num_ranks,
                "Total number of ranks for nvshmem collective test."),
      tsl::Flag("input_data_type", &input_data_type,
                "Data type to test for nvshmem collective test."),
  };
  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);
  testing::InitGoogleTest(&argc, argv);
  if (rank_id >= 0) {
    return xla::NvshmemCollectiveTestBody(rank_id, num_ranks, input_data_type)
        .raw_code();
  }
  return RUN_ALL_TESTS();
}
