/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/pjrt/pjrt_c_api_client.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "xla/hlo/builder/xla_builder.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

static void SetUpCpuPjRtApi() {
  std::string device_type = "cpu";
  auto status = ::pjrt::PjrtApi(device_type);
  if (!status.ok()) {
    // Maybe link a bazel library that will return the directory with the run-files for
    // the build ? We assume it's located in the current directory, which usually works ...
    std::string pjrt_plugin_path = "./xla/pjrt/c/pjrt_c_api_cpu_plugin.so";

    TF_ASSERT_OK_AND_ASSIGN(
      const PJRT_Api* api,
      pjrt::LoadPjrtPlugin(device_type, pjrt_plugin_path)
    );
    LOG(INFO) << "Loaded PJRT from " << pjrt_plugin_path;
  }
}

TEST(PjRtCApiClientTest, EndToEnd) {
  SetUpCpuPjRtApi();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetCApiClient("cpu"));
  LOG(INFO) << "\tplatform_name=" << client->platform_name()
    << ", platform_id=" << client->platform_id();

  // Create f(x) = x^2
  LOG(INFO) << "Create f(x) = x^2:";
  XlaBuilder builder("x*x+1");
  Shape x_shape = ShapeUtil::MakeShape(F32, {});
  auto x = Parameter(&builder, 0, x_shape, "x");
  auto f = Mul(x, x);
  auto computation = builder.Build(f).value();
  LOG(INFO) << "\tComputation built.";
  std::unique_ptr<PjRtLoadedExecutable> executable =
      client->Compile(computation, CompileOptions()).value();
  LOG(INFO) << "\tCompiled to executable.";

  // Transfer concrete x value.
  std::vector<float> data{3};
  TF_ASSERT_OK_AND_ASSIGN(
      auto x_value,
      client->BufferFromHostBuffer(
          data.data(), x_shape.element_type(), x_shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->addressable_devices()[0]));
  LOG(INFO) << "Tranferred value of x=" << data[0] << " to device.";

  // Execute function.
  ExecuteOptions execute_options;
  execute_options.non_donatable_input_indices = {0};
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results =
      executable->Execute({{x_value.get()}}, execute_options)
          .value();
  ASSERT_EQ(results[0].size(), 1);
  auto& result_buffer = results[0][0];
  // How do we get the actual result (a scalar) ? How to copy this float32 to host !?
  LOG(INFO) << "Success";
}

}  // namespace
}  // namespace xla
