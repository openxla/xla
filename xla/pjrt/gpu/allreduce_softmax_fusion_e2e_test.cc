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
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "xla/backends/gpu/collectives/nvshmem_collectives.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"

namespace xla {
namespace {

class AllReduceSoftmaxFusionE2ETest : public ::testing::Test {};

static const char* test_binary_name;

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileExecutable(
    absl::string_view program, xla::PjRtClient& client,
    xla::CompileOptions compile_options = xla::CompileOptions()) {
  TF_ASSIGN_OR_RETURN(auto hlo_module,
                      ParseAndReturnUnverifiedModule(program, {}));

  xla::XlaComputation xla_computation(hlo_module->ToProto());
  return client.CompileAndLoad(xla_computation, compile_options);
}

std::string GenerateAllReduceSoftmaxHLO(int num_ranks, int input_size = 32) {
  std::string replica_groups = "";
  for (int i = 0; i < num_ranks; ++i) {
    if (i > 0) replica_groups += ",";
    replica_groups += std::to_string(i);
  }

  return absl::StrFormat(R"(
HloModule AllReduceSoftmaxFusion, entry_computation_layout={(f32[%d]{0})->f32[%d]{0}}, replica_count=%d

region_add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

region_max {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT maximum = f32[] maximum(lhs, rhs)
}

region_sum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY main {
  input = f32[%d]{0} parameter(0)
  all_reduce = f32[%d]{0} all-reduce(input), channel_id=1, replica_groups={{%s}}, use_global_device_ids=true, to_apply=region_add, backend_config={"collective_backend_config":{"backend":"NVSHMEM"}}
  constant_neg_inf = f32[] constant(-inf)
  reduce_max = f32[] reduce(all_reduce, constant_neg_inf), dimensions={0}, to_apply=region_max
  maximum = f32[] maximum(reduce_max, constant_neg_inf)
  reshape_max = f32[1]{0} reshape(maximum)
  broadcast_max = f32[1]{0} broadcast(reshape_max), dimensions={0}
  reshape_max_scalar = f32[] reshape(broadcast_max)
  broadcast_max_vector = f32[%d]{0} broadcast(reshape_max_scalar), dimensions={}
  subtract = f32[%d]{0} subtract(all_reduce, broadcast_max_vector)
  exponential = f32[%d]{0} exponential(subtract)
  constant_zero = f32[] constant(0)
  reduce_sum = f32[] reduce(exponential, constant_zero), dimensions={0}, to_apply=region_sum
  reshape_sum = f32[1]{0} reshape(reduce_sum)
  broadcast_sum = f32[1]{0} broadcast(reshape_sum), dimensions={0}
  reshape_sum_scalar = f32[] reshape(broadcast_sum)
  broadcast_sum_vector = f32[%d]{0} broadcast(reshape_sum_scalar), dimensions={}
  ROOT divide = f32[%d]{0} divide(exponential, broadcast_sum_vector)
}
)",
                         input_size, input_size, num_ranks, input_size,
                         input_size, replica_groups, input_size, input_size,
                         input_size, input_size, input_size);
}

void RunAllReduceSoftmaxTest(bool enable_fusion, int num_ranks = 2,
                             bool use_different_input = false,
                             int input_size = 32) {
  constexpr int kMaxRanks = 8;
  EXPECT_LE(num_ranks, kMaxRanks) << "Maximum supported ranks is " << kMaxRanks;

  tsl::SubProcess child[kMaxRanks];
  for (int rank_id = 0; rank_id < num_ranks; ++rank_id) {
    std::vector<std::string> argv;
    argv.push_back(test_binary_name);
    argv.push_back(absl::StrFormat("--rank_id=%d", rank_id));
    argv.push_back(absl::StrFormat("--num_ranks=%d", num_ranks));
    argv.push_back(absl::StrFormat("--enable_fusion=%s",
                                   enable_fusion ? "true" : "false"));
    argv.push_back(absl::StrFormat("--use_different_input=%s",
                                   use_different_input ? "true" : "false"));
    argv.push_back(absl::StrFormat("--input_size=%d", input_size));

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

void RunAllReduceSoftmaxComparisonTest(int num_ranks = 2,
                                       bool use_different_input = false,
                                       int input_size = 32) {
  constexpr int kMaxRanks = 8;
  EXPECT_LE(num_ranks, kMaxRanks) << "Maximum supported ranks is " << kMaxRanks;

  // Run non-fusion test (baseline)
  tsl::SubProcess nofusion_child[kMaxRanks];
  for (int rank_id = 0; rank_id < num_ranks; ++rank_id) {
    std::vector<std::string> argv;
    argv.push_back(test_binary_name);
    argv.push_back(absl::StrFormat("--rank_id=%d", rank_id));
    argv.push_back(absl::StrFormat("--num_ranks=%d", num_ranks));
    argv.push_back("--enable_fusion=false");
    argv.push_back(absl::StrFormat("--use_different_input=%s",
                                   use_different_input ? "true" : "false"));
    argv.push_back(absl::StrFormat("--input_size=%d", input_size));
    argv.push_back("--result_output_mode=true");

    nofusion_child[rank_id].SetProgram(test_binary_name, argv);
    nofusion_child[rank_id].SetChannelAction(tsl::CHAN_STDOUT,
                                             tsl::ACTION_PIPE);
    nofusion_child[rank_id].SetChannelAction(tsl::CHAN_STDERR,
                                             tsl::ACTION_PIPE);
    ASSERT_TRUE(nofusion_child[rank_id].Start()) << "nofusion rank " << rank_id;
  }

  std::vector<std::vector<float>> nofusion_results(num_ranks);
  for (int rank_id = 0; rank_id < num_ranks; ++rank_id) {
    std::string stdout_str, stderr_str;
    int child_status =
        nofusion_child[rank_id].Communicate(nullptr, &stdout_str, &stderr_str);
    EXPECT_EQ(child_status, 0) << "nofusion rank " << rank_id << " failed";

    // Parse results
    std::vector<float> rank_results;
    std::istringstream ss(stdout_str);
    std::string line;
    while (std::getline(ss, line)) {
      if (line.find("RESULTS:") != std::string::npos) {
        size_t pos = line.find("RESULTS:") + 8;
        std::string values_str = line.substr(pos);
        std::istringstream values_stream(values_str);
        float value;
        while (values_stream >> value) {
          rank_results.push_back(value);
        }
        break;
      }
    }
    nofusion_results[rank_id] = rank_results;
  }

  // Run fusion test
  tsl::SubProcess fusion_child[kMaxRanks];
  for (int rank_id = 0; rank_id < num_ranks; ++rank_id) {
    std::vector<std::string> argv;
    argv.push_back(test_binary_name);
    argv.push_back(absl::StrFormat("--rank_id=%d", rank_id));
    argv.push_back(absl::StrFormat("--num_ranks=%d", num_ranks));
    argv.push_back("--enable_fusion=true");
    argv.push_back(absl::StrFormat("--use_different_input=%s",
                                   use_different_input ? "true" : "false"));
    argv.push_back(absl::StrFormat("--input_size=%d", input_size));
    argv.push_back("--result_output_mode=true");

    fusion_child[rank_id].SetProgram(test_binary_name, argv);
    fusion_child[rank_id].SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
    fusion_child[rank_id].SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
    ASSERT_TRUE(fusion_child[rank_id].Start()) << "fusion rank " << rank_id;
  }

  std::vector<std::vector<float>> fusion_results(num_ranks);
  for (int rank_id = 0; rank_id < num_ranks; ++rank_id) {
    std::string stdout_str, stderr_str;
    int child_status =
        fusion_child[rank_id].Communicate(nullptr, &stdout_str, &stderr_str);
    EXPECT_EQ(child_status, 0) << "fusion rank " << rank_id << " failed";

    // Parse results
    std::vector<float> rank_results;
    std::istringstream ss(stdout_str);
    std::string line;
    while (std::getline(ss, line)) {
      if (line.find("RESULTS:") != std::string::npos) {
        size_t pos = line.find("RESULTS:") + 8;
        std::string values_str = line.substr(pos);
        std::istringstream values_stream(values_str);
        float value;
        while (values_stream >> value) {
          rank_results.push_back(value);
        }
        break;
      }
    }
    fusion_results[rank_id] = rank_results;
  }

  // Show input data for debugging
  for (int rank_id = 0; rank_id < num_ranks; ++rank_id) {
    std::vector<float> input_data(input_size);
    if (use_different_input) {
      for (int i = 0; i < input_size; ++i) {
        input_data[i] = static_cast<float>(rank_id);
      }
    } else {
      for (int i = 0; i < input_size; ++i) {
        input_data[i] = 1.0f + i * 0.1f;
      }
    }

    std::cout << "Rank " << rank_id << " input: [";
    for (int i = 0; i < std::min(8, input_size); ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << input_data[i];
    }
    if (input_size > 8) {
      std::cout << ", ..., " << input_data[input_size - 1];
    }
    std::cout << "]" << std::endl;
  }

  // Compare results
  constexpr float kTolerance = 1e-5f;
  bool all_ranks_match = true;

  for (int rank_id = 0; rank_id < num_ranks; ++rank_id) {
    EXPECT_EQ(fusion_results[rank_id].size(), nofusion_results[rank_id].size())
        << "Rank " << rank_id << " result size mismatch";

    if (fusion_results[rank_id].size() != nofusion_results[rank_id].size()) {
      all_ranks_match = false;
      continue;
    }

    bool rank_match = true;
    float fusion_sum = 0.0f, nofusion_sum = 0.0f;

    for (size_t i = 0; i < fusion_results[rank_id].size(); ++i) {
      fusion_sum += fusion_results[rank_id][i];
      nofusion_sum += nofusion_results[rank_id][i];

      if (std::abs(fusion_results[rank_id][i] - nofusion_results[rank_id][i]) >=
          kTolerance) {
        rank_match = false;
      }
    }

    if (rank_match) {
      std::cout << "Rank " << rank_id << " MATCH (sum=" << nofusion_sum << ")"
                << std::endl;
    } else {
      std::cout << "Rank " << rank_id << " MISMATCH:" << std::endl;
      std::cout << "  Non-fusion sum: " << nofusion_sum << " (baseline)"
                << std::endl;
      std::cout << "  Fusion sum:     " << fusion_sum << " (under test)"
                << std::endl;

      std::cout << "  Non-fusion: [";
      for (size_t i = 0; i < nofusion_results[rank_id].size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << nofusion_results[rank_id][i];
      }
      std::cout << "]" << std::endl;

      std::cout << "  Fusion:     [";
      for (size_t i = 0; i < fusion_results[rank_id].size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << fusion_results[rank_id][i];
      }
      std::cout << "]" << std::endl;

      all_ranks_match = false;
    }
  }

  EXPECT_TRUE(all_ranks_match) << "Fusion and non-fusion results differ";
}

// Test cases
TEST(AllReduceSoftmaxFusionE2ETest, TwoGpu_SameInput_Shape32) {
  RunAllReduceSoftmaxComparisonTest(
      /*num_ranks=*/2, /*use_different_input=*/false, /*input_size=*/32);
}

TEST(AllReduceSoftmaxFusionE2ETest, TwoGpu_DiffInput_Shape32) {
  RunAllReduceSoftmaxComparisonTest(
      /*num_ranks=*/2, /*use_different_input=*/true, /*input_size=*/32);
}

TEST(AllReduceSoftmaxFusionE2ETest, TwoGpu_SameInput_Shape1024) {
  RunAllReduceSoftmaxComparisonTest(
      /*num_ranks=*/2, /*use_different_input=*/false, /*input_size=*/1024);
}

TEST(AllReduceSoftmaxFusionE2ETest, TwoGpu_DiffInput_Shape1024) {
  RunAllReduceSoftmaxComparisonTest(
      /*num_ranks=*/2, /*use_different_input=*/true, /*input_size=*/1024);
}

absl::Status AllReduceSoftmaxE2ETestBody(int rank_id, int num_ranks,
                                         bool enable_fusion,
                                         bool use_different_input = false,
                                         int input_size = 32,
                                         bool result_output_mode = false) {
  std::unique_ptr<xla::DistributedRuntimeService> service;
  if (rank_id == 0) {
    xla::CoordinationServiceImpl::Options service_options;
    service_options.num_nodes = num_ranks;
    TF_ASSIGN_OR_RETURN(service, xla::GetDistributedRuntimeService(
                                     "[::]:12345", service_options));
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

  // Initialize NVSHMEM collective system
  setenv("NVSHMEM_DEBUG", "INFO", 1);
  setenv("NVSHMEM_VERSION", "true", 1);
  setenv("NVSHMEM_INFO", "true", 1);

  auto* nvshmem_collectives = xla::gpu::NvshmemCollectives::Default();
  auto kv_store = client_options.kv_store;

  xla::gpu::GpuCollectives::Topology topology;
  topology.node_id = rank_id;
  topology.num_nodes = num_ranks;
  topology.device_count_per_process = 1;
  topology.kv_store = kv_store;
  TF_RETURN_IF_ERROR(nvshmem_collectives->InitializeTopology(topology));

  TF_ASSIGN_OR_RETURN(auto communicator,
                      nvshmem_collectives->CreateCommunicator());

  // Compile options
  xla::CompileOptions options;
  auto* debug_options =
      options.executable_build_options.mutable_debug_options();

  debug_options->set_xla_gpu_experimental_enable_nvshmem(true);
  debug_options->set_xla_gpu_use_memcpy_local_p2p(false);

  options.executable_build_options.set_run_backend_only(false);
  options.executable_build_options.set_use_spmd_partitioning(false);
  options.executable_build_options.set_num_replicas(num_ranks);

  // Fusion control
  if (!enable_fusion) {
    debug_options->add_xla_disable_hlo_passes("allreduce-softmax-fusion");
  }

  // Generate HLO
  std::string hlo_program = GenerateAllReduceSoftmaxHLO(num_ranks, input_size);

  // Compile HLO program
  TF_ASSIGN_OR_RETURN(auto executable,
                      CompileExecutable(hlo_program, *client, options));

  // Generate input
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(
      PrimitiveType::F32, {input_size}, /*minor_to_major=*/{0});
  shape.mutable_layout()->set_memory_space(Layout::kDefaultMemorySpace);

  PjRtDevice* const device = client->addressable_devices()[0];
  std::unique_ptr<PjRtBuffer> input;

  std::vector<float> input_data(input_size);
  if (use_different_input) {
    for (int i = 0; i < input_size; ++i) {
      input_data[i] = static_cast<float>(rank_id);
    }
  } else {
    for (int i = 0; i < input_size; ++i) {
      input_data[i] = 1.0f + i * 0.1f;
    }
  }

  TF_ASSIGN_OR_RETURN(
      input,
      client->BufferFromHostBuffer(
          input_data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, *device->default_memory_space(),
          /*device_layout=*/nullptr));

  TF_ASSIGN_OR_RETURN(auto result,
                      executable->Execute({{input.get()}}, ExecuteOptions()));

  std::vector<std::unique_ptr<xla::PjRtBuffer>>& result_buffers = result[0];
  TF_ASSIGN_OR_RETURN(std::shared_ptr<xla::Literal> literal,
                      result_buffers[0]->ToLiteralSync());

  auto output_data = literal->data<float>();
  int64_t output_size = literal->shape().dimensions(0);
  TF_RET_CHECK(output_size == input_size)
      << "Output size mismatch: " << output_size;

  if (result_output_mode) {
    std::cout << "RESULTS:";
    for (int64_t i = 0; i < output_size; ++i) {
      std::cout << " " << output_data[i];
    }
    std::cout << std::endl;
    return absl::OkStatus();
  }

  // Validate softmax sum
  float actual_sum = 0.0f;
  for (int64_t i = 0; i < output_size; ++i) {
    actual_sum += output_data[i];
  }
  TF_RET_CHECK(std::abs(actual_sum - 1.0f) < 1e-5f)
      << "Rank " << rank_id << " softmax sum validation failed: " << actual_sum;

  return absl::OkStatus();
}

}  // namespace
}  // namespace xla

int main(int argc, char* argv[]) {
  xla::test_binary_name = argv[0];
  int rank_id = -1;
  int num_ranks = -1;
  int input_size = 32;
  std::string enable_fusion_str = "true";
  std::string use_different_input_str = "false";
  std::string result_output_mode_str = "false";
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("rank_id", &rank_id,
                "Rank ID for AllReduce+Softmax fusion test."),
      tsl::Flag("num_ranks", &num_ranks,
                "Total number of ranks for AllReduce+Softmax fusion test."),
      tsl::Flag("enable_fusion", &enable_fusion_str,
                "Enable AllReduce+Softmax fusion (true/false)."),
      tsl::Flag("use_different_input", &use_different_input_str,
                "Use different input data per rank (true/false)."),
      tsl::Flag(
          "result_output_mode", &result_output_mode_str,
          "Output results for comparison instead of validation (true/false)."),
      tsl::Flag("input_size", &input_size,
                "Input tensor size for AllReduce+Softmax fusion test."),
  };
  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);
  testing::InitGoogleTest(&argc, argv);

  if (rank_id >= 0) {
    bool enable_fusion = (enable_fusion_str == "true");
    bool use_different_input = (use_different_input_str == "true");
    bool result_output_mode = (result_output_mode_str == "true");

    auto status = xla::AllReduceSoftmaxE2ETestBody(
        rank_id, num_ranks, enable_fusion, use_different_input, input_size,
        result_output_mode);

    return status.raw_code();
  }

  return RUN_ALL_TESTS();
}