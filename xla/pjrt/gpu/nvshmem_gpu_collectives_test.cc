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

#include <cstdint>
#include <memory>
#include <fstream>
#include <optional>
#include <string>
#include <sstream>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/debug_options_flags.h"
#include "xla/ffi/ffi.h"
#include "xla/literal_comparison.h"
#include "xla/literal_util.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/test.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/status_macros.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class NvshmemGpuCollectivesTest : public ::testing::Test {};

static const char* test_binary_name;

// static const absl::flat_hash_map< absl::string_view, xla::ResourceType >

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileExecutable(
    absl::string_view program, xla::PjRtClient& client,
    xla::CompileOptions compile_options = xla::CompileOptions()) {
  TF_ASSIGN_OR_RETURN(auto hlo_module,
                      ParseAndReturnUnverifiedModule(program, {}));

  xla::XlaComputation xla_computation(hlo_module->ToProto());
  return client.CompileAndLoad(xla_computation, compile_options);
}

void RunNvshmemTest(PrimitiveType data_type, absl::string_view test_case) {
  const int num_ranks = 2;
  tsl::SubProcess child[num_ranks];
  for (int rank_id = 0; rank_id < num_ranks; ++rank_id) {
    std::vector<std::string> argv;
    argv.push_back(test_binary_name);
    argv.push_back(absl::StrFormat("--rank_id=%d", rank_id));
    argv.push_back(absl::StrFormat("--num_ranks=%d", num_ranks));
    argv.push_back(absl::StrFormat("--input_data_type=%d", (int)data_type));
    argv.push_back(absl::StrFormat("--test_case=%s", test_case));
    argv.push_back(absl::StrFormat("--v=1"));
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

TEST(NvshmemGpuCollectivesTest, NvshmemCollectivePermuteFloat) {
  RunNvshmemTest(PrimitiveType::F32, "collective_permute");
}

// TODO(b/431602576): Re-enable this test once the bug is fixed.
TEST(NvshmemGpuCollectivesTest, DISABLED_NvshmemSendRecvFloat) {
  RunNvshmemTest(PrimitiveType::F32, "send_recv");
}

TEST(NvshmemGpuCollectivesTest, NvshmemAllReduceFloat) {
  RunNvshmemTest(PrimitiveType::F32, "all_reduce");
}

// TODO(patrios): Re-enable this test once the bug is fixed.
TEST(NvshmemGpuCollectivesTest, DISABLED_NvshmemAllReducePred) {
  RunNvshmemTest(PrimitiveType::PRED, "all_reduce");
}

TEST(NvshmemGpuCollectivesTest, NvshmemAllReduceInt8) {
  RunNvshmemTest(PrimitiveType::S8, "all_reduce");
}

TEST(NvshmemGpuCollectivesTest, NvshmemAllReduceUint8) {
  RunNvshmemTest(PrimitiveType::U8, "all_reduce");
}

absl::Status NvshmemRunAllReduce(
      PjRtClient& client, uint32_t num_ranks, 
      const CompileOptions& compile_options, PrimitiveType dtype) {
  
  const char *kProgram = R"(HloModule Test
      apply_op {
        x = $0[] parameter(0)
        y = $0[] parameter(1)
        ROOT apply_op = $0[] add(x, y)
      }
      addf {
        x = f32[] parameter(0)
        y = f32[] parameter(1)
        ROOT apply_op = f32[] add(x, y)
      }
      ENTRY all-reduce {
        num_ranks = u32[] parameter(0)
        num_ranksf = f32[]convert(num_ranks)
        id = u32[] replica-id()
        one = u32[] constant(1)
        id1 = u32[] add(id, one)
        bcast_id1 = u32$1 broadcast(id1), dimensions={}
        bast_idT = $0$1 convert(bcast_id1)
        all_reduce = $0$1 all-reduce(bast_idT), to_apply=apply_op, backend_config={"collective_backend_config":{"backend":"NVSHMEM"}}
        donef = f32$1 convert(all_reduce)
        onef = f32[] constant(1)
        p1 = f32[] add(num_ranksf, onef)
        p1mul = f32[] multiply(num_ranksf, p1)
        twof = f32[] constant(2)
        p1div2 = f32[] divide(p1mul, twof)
        truthf = f32$1 broadcast(p1div2), dimensions={}
        subf = f32$1 subtract(donef, truthf)
        subabsf = f32$1 abs(subf)
        zero = f32[] constant(0)
        ROOT final = f32[] reduce(subabsf, zero), dimensions={0,1}, to_apply=addf
      })";

  // std::ostringstream sprogram;
  // {
  // std::ifstream ifs("input.hlo");
  // if (!ifs) return absl::InternalError("Ops wrong HLO file!");
  //   sprogram << ifs.rdbuf();
  // }
  auto dtype_str = primitive_util::LowercasePrimitiveTypeName(dtype);
  auto hlo_text = absl::Substitute(kProgram, dtype_str, "[100,40]");

  TF_ASSIGN_OR_RETURN(auto executable,
                      CompileExecutable(hlo_text, client, compile_options));
  // TF_ASSIGN_OR_RETURN(auto hlo_modules, executable->GetHloModules());

  auto param = LiteralUtil::CreateFull({}, num_ranks);
  //PrimitiveTypeBitWidth
  PjRtDevice* const device = client.addressable_devices()[0];
 
    // TF_ASSIGN_OR_RETURN(auto fake_args, 
  //       xla::MakeFakeArguments(hlo_modules[0].get(), /*pseudo_random*/true,
  //       /*use_large_range*/false));
  /*  std::normal_distribution<double> generator(mean, stddev);
  return CreateLiteralWithGenerator<type, NativeT>(
      shape, [&](absl::Span<const int64_t>) {
        return static_cast<NativeT>(generator(*engine));
      });*/

  // std::minstd_rand0 engine;
  // auto func = [&](auto xtype) {
  //     auto tt = decltype(xtype);
  //     //using ElementT = primitive_util::NativeTypeOf<xtype>;
  //     return LiteralUtil::CreateRandomLiteral<tt>(
  //       shape, &engine, 0.0, /*stddev*/1.0
  //     );
  // };
  // TF_ASSIGN_OR_RETURN(auto literal, 
  //   primitive_util::ArrayTypeSwitch(func, dtype));

 TF_ASSIGN_OR_RETURN(
      auto input, client.BufferFromHostLiteral(
          param, *device->default_memory_space()));
  
  TF_ASSIGN_OR_RETURN(auto result,
                        executable->Execute({{input.get()}}, ExecuteOptions()));

  auto& result_buffers = result[0];
  TF_ASSIGN_OR_RETURN(auto output, result_buffers[0]->ToLiteralSync());
  //VLOG(0) << "Got literal output " << output->ToString();
  auto expected = LiteralUtil::CreateFull({}, 0.0f);
  return literal_comparison::Near(expected, *output,
                  ErrorSpec(1e-5, 1e-5), {}, nullptr);
}

absl::Status NvshmemRunCollectivePermute(
      PjRtClient& client, uint32_t num_ranks, 
      const CompileOptions& compile_options, PrimitiveType dtype) {

  const char *kProgram = R"(HloModule Test
addu {
  x = u32[] parameter(0)
  y = u32[] parameter(1)
  ROOT apply_op = u32[] add(x, y)
}
ENTRY Xtest {
  num_ranks = u32[] parameter(0)
  ofs = u32[] constant(33)
  id = u32[] replica-id()
  idofs = u32[] add(id, ofs)
  bcast = u32$1 broadcast(idofs), dimensions={}
  bcastT = $0$1 convert(bcast)
  coll_permute = $0$1 collective-permute(bcastT), source_target_pairs={$2}, backend_config={"collective_backend_config":{"backend":"NVSHMEM"}}
  done = u32$1 convert(coll_permute)
  one = u32[] constant(1)
  id_add = u32[] add(id, one)
  rem = u32[] remainder(id_add, num_ranks)
  add_ofs = u32[] add(rem, ofs)
  truth = u32$1 broadcast(add_ofs), dimensions={}
  sub = u32$1 subtract(done, truth)
  zero = u32[] constant(0)
  ROOT final = u32[] reduce(sub, zero), dimensions={0,1}, to_apply=addu
})";
  std::ostringstream sprogram;
  {
  std::ifstream ifs("input.hlo");
  if (!ifs) return absl::InternalError("Ops wrong HLO file!");
    sprogram << ifs.rdbuf();
  }
  std::stringstream channels;
  for (uint32_t i = 0; i < num_ranks; i++) {
    channels << '{' << (i == num_ranks-1 ? 0 : i + 1) << ',' << i << '}';
    if (i < num_ranks-1) channels << ',';
  }
  auto dtype_str = primitive_util::LowercasePrimitiveTypeName(dtype);
  auto hlo_text = absl::Substitute(kProgram, 
        dtype_str, "[10,20]", channels.str());

  TF_ASSIGN_OR_RETURN(auto executable,
                      CompileExecutable(hlo_text, client, compile_options));

  auto param = LiteralUtil::CreateFull({}, num_ranks);
  auto *device = client.addressable_devices()[0];

  TF_ASSIGN_OR_RETURN(
      auto input, client.BufferFromHostLiteral(
          param, *device->default_memory_space()));
  
  TF_ASSIGN_OR_RETURN(auto result,
                        executable->Execute({{input.get()}}, ExecuteOptions()));

  auto& result_buffers = result[0];
  TF_ASSIGN_OR_RETURN(auto output, result_buffers[0]->ToLiteralSync());

  VLOG(0) << "Got literal output " << output->ToString();
  auto expected = LiteralUtil::CreateFull({}, (uint32_t)0);
  return literal_comparison::Equal(expected, *output, nullptr);
}

absl::Status NvshmemRunSendRecv(
      PjRtClient& client, uint32_t rank_id, uint32_t num_ranks, 
      const CompileOptions& compile_options, PrimitiveType dtype) {

  const char *kProgram = R"(HloModule Test
addf {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT apply_op = f32[] add(x, y)
}
ENTRY Xtest {
  data = $0$1 iota(), iota_dimension=0
  after-all = token[] after-all()
  recv = ($0$1, f32[], token[]) recv(after-all), channel_id=0, frontend_attributes={_xla_send_recv_source_target_pairs="{{0,1}}"}, backend_config={"collective_backend_config":{"backend":"NVSHMEM"}}
  send = ($0$1, f32[], token[]) send(data, after-all), channel_id=0, control-predecessors={recv}, frontend_attributes={_xla_send_recv_source_target_pairs="{{0,1}}"}, backend_config={"collective_backend_config":{"backend":"NVSHMEM"}}
  recv-done = ($0$1, token[]) recv-done(recv), channel_id=0
  recv-data = $0$1 get-tuple-element(recv-done), index=0
  send-done = token[] send-done(send), channel_id=0, control-predecessors={recv}
  res = $0$1 copy(recv-data)
  resf = f32$1 convert(res)
  zero = f32[] constant(0)
  ROOT final = f32[] reduce(resf, zero), dimensions={0}, to_apply=addf

})";
  std::ostringstream sprogram;
  {
  std::ifstream ifs("input.hlo");
  if (!ifs) return absl::InternalError("Ops wrong HLO file!");
    sprogram << ifs.rdbuf();
  }
  int64_t N = 10000;
  auto dtype_str = primitive_util::LowercasePrimitiveTypeName(dtype);
  //auto hlo_text = sprogram.str();
  auto hlo_text = absl::Substitute(kProgram, 
        dtype_str, absl::StrFormat("[%lld]", N));
  //VLOG(0) << "input text " << hlo_text;
  
  TF_ASSIGN_OR_RETURN(auto executable,
                      CompileExecutable(hlo_text, client, compile_options));

  TF_ASSIGN_OR_RETURN(auto result,
                        executable->Execute({{ //input.get()
                        }}, ExecuteOptions()));

  auto& result_buffers = result[0];
  TF_ASSIGN_OR_RETURN(auto output, result_buffers[0]->ToLiteralSync());

  VLOG(0) << "Got literal output " << output->ToString();
  if (rank_id == 0) return absl::OkStatus(); // 0-th rank just sends

  auto expected = LiteralUtil::CreateFull({}, (float)N/2*(N-1));
  return literal_comparison::Equal(expected, *output, nullptr);
}

absl::Status NvshmemCollectiveTestBody(int rank_id, int num_ranks,
                                       int input_data_type,
                                       absl::string_view test_case) {

  xla::PrimitiveType data_type = (xla::PrimitiveType)input_data_type;
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
  // client_options.allocator_config.kind = GpuAllocatorConfig::Kind::kPlatform;
  
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
  // options.executable_build_options.set_run_backend_only(true);
  options.executable_build_options.set_use_spmd_partitioning(false);
  options.executable_build_options.set_num_replicas(num_ranks);

  if (test_case == "collective_permute") {
    return NvshmemRunCollectivePermute(*client, num_ranks, options, data_type);
  } else if (test_case == "send_recv") {
    return NvshmemRunSendRecv(*client, rank_id, num_ranks, options, data_type);
  } else if (test_case == "all_reduce") {
    VLOG(0) << "Running all reduce";
    return NvshmemRunAllReduce(*client, num_ranks, options, data_type);
  }
  return absl::InvalidArgumentError("Unknown collective type!");
}
}  // namespace
}  // namespace xla

int main(int argc, char* argv[]) {
  // Save name of binary so that it may invoke itself.
  xla::test_binary_name = argv[0];
  int rank_id = -1;
  int num_ranks = -1;
  int input_data_type = (int)xla::PrimitiveType::F32;
  std::string test_case = "all_reduce";  // Add test_case parameter
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("rank_id", &rank_id, "Rank ID for nvshmem collective test."),
      tsl::Flag("num_ranks", &num_ranks,
                "Total number of ranks for nvshmem collective test."),
      tsl::Flag("input_data_type", &input_data_type,
                "Data type to test for nvshmem collective test."),
      tsl::Flag("test_case", &test_case,
                "Test case to run (collective_permute, send_recv)."),
  };
  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);
  testing::InitGoogleTest(&argc, argv);
  if (rank_id >= 0) {
    auto s = xla::NvshmemCollectiveTestBody(rank_id, num_ranks, input_data_type,
                                          test_case);
    if (!s.ok()) {
      VLOG(0) << "Failed with " << s;
    } else {
      VLOG(0) << "Test succeeded on rank " << rank_id;
    }
    return s.raw_code();
  }
  return RUN_ALL_TESTS();
}
