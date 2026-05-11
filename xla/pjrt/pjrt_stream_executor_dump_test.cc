#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/pjrt/local_device_state.h"
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

#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/service/platform_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

TEST(PjRtStreamExecutorClientDumpTest, DumpSerializedExecutable) {
  LocalClient* local_client = xla::ClientLibrary::LocalClientOrDie();
  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          PlatformUtil::GetPlatform("Host"));
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                          platform->ExecutorForDevice(0));

  auto device_state = std::make_unique<LocalDeviceState>(
      executor, local_client, LocalDeviceState::kSynchronous, 32, false, false);
  int local_device_id = device_state->local_device_id().value();

  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  devices.emplace_back(std::make_unique<PjRtStreamExecutorDevice>(
      0, std::move(device_state), local_device_id, 0, 0, 0, "cpu"));

  std::vector<std::unique_ptr<PjRtMemorySpace>> memory_spaces;
  memory_spaces.emplace_back(std::make_unique<PjRtStreamExecutorMemorySpace>(
      0, devices.back().get(), "cpu", 0));
  devices.back()->AttachMemorySpace(memory_spaces.back().get(), true);

  auto client = std::make_unique<PjRtStreamExecutorClient>(
      "cpu", local_client, std::move(devices), 0, std::move(memory_spaces),
      nullptr, nullptr, false, nullptr);

  // Create a temp directory for dumping
  std::string dump_dir = tsl::io::JoinPath(testing::TempDir(), "xla_dump");

  CompileOptions compile_options;
  compile_options.executable_build_options.mutable_debug_options()
      ->set_xla_dump_to(dump_dir);

  XlaBuilder builder("Add");
  auto a = Parameter(&builder, 0, ShapeUtil::MakeScalarShape(F32), "a");
  auto b = Parameter(&builder, 1, ShapeUtil::MakeScalarShape(F32), "b");
  Add(a, b);
  TF_ASSERT_OK_AND_ASSIGN(auto computation, builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          client->CompileAndLoad(computation, compile_options));

  TF_ASSERT_OK_AND_ASSIGN(std::string serialized,
                          client->SerializeExecutable(*executable));

  // Now LoadSerializedExecutable
  TF_ASSERT_OK_AND_ASSIGN(
      auto loaded_exec,
      client->LoadSerializedExecutable(serialized, compile_options, {}));

  // Verify if file exists
  std::vector<std::string> files;
  TF_ASSERT_OK(tsl::Env::Default()->GetChildren(dump_dir, &files));

  bool found = false;
  for (const auto& file : files) {
    if (absl::StrContains(file, "after_optimizations")) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Dump file not found in " << dump_dir
                     << ". Files found: " << absl::StrJoin(files, ", ");
}

}  // namespace
}  // namespace xla
