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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "json/json.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend.h"  // IWYU pragma: keep - cudnn frontend headers are not hermetic
#include "third_party/cudnn_frontend/include/cudnn_frontend/backend/device_properties.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend/graph_helpers.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend_shim.h"
#include "xla/backends/gpu/target_config/cudnn_device_props.h"
#include "xla/backends/gpu/target_config/cudnn_device_props_test_util.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/test.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;

Json::Value ParseProps(
    const std::shared_ptr<cudnn_frontend::DeviceProperties>& props) {
  std::vector<uint8_t> buf;
  auto err = props->serialize(buf);
  EXPECT_FALSE(err.is_bad()) << err.get_message();
  std::string str(buf.begin(), buf.end());
  Json::Value value;
  Json::CharReaderBuilder builder;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  std::string errors;
  EXPECT_TRUE(
      reader->parse(str.data(), str.data() + str.size(), &value, &errors))
      << errors;
  return value;
}

TEST(CudnnDevicePropsTest, PreservesMemoryBusWidth) {
  if (cudnn_frontend::detail::get_backend_version() < 90800) {
    GTEST_SKIP() << "cuDNN runtime does not support DeviceProperties.";
  }
  // Query the backend's live serializer independently of the synthesized JSON.
  // Major/minor/patch alone cannot distinguish the first 9.27 builds.
  std::string name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName(name));
  if (platform->VisibleDeviceCount() == 0) {
    GTEST_SKIP() << "No visible GPU devices.";
  }
  auto live = std::make_shared<cudnn_frontend::DeviceProperties>();
  auto err = live->set_device_id(0).build();
  ASSERT_FALSE(err.is_bad()) << err.get_message();
  if (!ParseProps(live).isMember("memoryBusWidth")) {
    GTEST_SKIP() << "cuDNN runtime does not serialize memoryBusWidth.";
  }
  ASSERT_OK_AND_ASSIGN(auto proto, GetGpuTargetConfig(GpuModel::H100_SXM));
  ASSERT_OK_AND_ASSIGN(auto config, GpuTargetConfig::FromProto(proto));
  for (int64_t width : {int64_t{0}, int64_t{5120}}) {
    SCOPED_TRACE(width);
    config.device_description.set_memory_bus_width(width);
    ASSERT_OK_AND_ASSIGN(auto props,
                         BuildDeviceProperties(config.device_description));
    Json::Value json = ParseProps(props);
    ASSERT_TRUE(json.isMember("memoryBusWidth"));
    EXPECT_EQ(json["memoryBusWidth"].asInt64(), width);
  }
}

TEST(CudnnDevicePropsTest, MatchesLiveDevice) {
  ASSERT_OK_AND_ASSIGN(std::string platform_name,
                       PlatformUtil::CanonicalPlatformName("gpu"));
  std::string name = absl::AsciiStrToUpper(platform_name);
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName(name));

  bool any_compared = false;
  for (int i = 0; i < platform->VisibleDeviceCount(); ++i) {
    ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                         platform->ExecutorForDevice(i));
    const se::DeviceDescription& desc = executor->GetDeviceDescription();
    SCOPED_TRACE(absl::StrCat("device ", i, ": ", desc.name()));

    ExpectDevicePropsMatchLiveDevice(desc, i);
    if (IsSkipped()) {
      return;
    }
    any_compared = true;
  }

  if (!any_compared) {
    GTEST_SKIP() << "No visible GPU devices to compare against.";
  }
}

// Make sure the cudnn_frontend::DeviceProperties created from our checked in
// target config still matches the one from the live device. This should catch
// cases where CuDNN requires a new field to be serialized, which may not have
// been mapped in our target config.
TEST(CudnnDevicePropsTest, SpecMatchesLiveDevice) {
  ASSERT_OK_AND_ASSIGN(std::string platform_name,
                       PlatformUtil::CanonicalPlatformName("gpu"));
  std::string name = absl::AsciiStrToUpper(platform_name);
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName(name));
  ASSERT_GT(platform->VisibleDeviceCount(), 0);
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  const se::DeviceDescription& live_desc = executor->GetDeviceDescription();

  std::optional<se::GpuTargetConfigProto> spec;
  // GpuModel values are contiguous from 0, and GetGpuTargetConfig returns
  // NotFound past the last one.
  for (int i = 0;; ++i) {
    absl::StatusOr<se::GpuTargetConfigProto> proto =
        GetGpuTargetConfig(static_cast<GpuModel>(i));
    if (absl::IsNotFound(proto.status())) {
      break;
    }
    ASSERT_OK(proto);
    // Name alone doesn't distinguish MIG partitions from the full GPU.
    if (proto->gpu_device_info().name() == live_desc.name() &&
        proto->gpu_device_info().core_count() == live_desc.core_count()) {
      spec = *std::move(proto);
      break;
    }
  }
  if (!spec.has_value()) {
    GTEST_SKIP() << "No spec for " << live_desc.name();
  }
  SCOPED_TRACE(live_desc.name());

  ASSERT_OK_AND_ASSIGN(GpuTargetConfig spec_config,
                       GpuTargetConfig::FromProto(*spec));
  ExpectDevicePropsMatchLiveDevice(spec_config.device_description);
}

}  // namespace
}  // namespace xla::gpu
