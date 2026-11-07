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

#include "xla/backends/gpu/target_config/cudnn_device_props_json.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "json/json.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

TEST(CudnnDevicePropsJsonTest, PreservesOptionalMemoryBusWidth) {
  ASSERT_OK_AND_ASSIGN(auto desc,
                       stream_executor::DeviceDescription::FromProto({}));
  desc.set_cuda_compute_capability({9, 0});
  desc.set_core_count(80);
  const std::optional<int64_t> widths[] = {std::nullopt, 0, 5120};
  for (const auto width : widths) {
    SCOPED_TRACE(::testing::PrintToString(width));
    desc.set_memory_bus_width(width);
    const auto before = desc;
    const std::string json = FormatDevicePropertiesJson(desc);
    Json::Value props;
    Json::CharReaderBuilder builder;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    std::string errors;
    ASSERT_TRUE(
        reader->parse(json.data(), json.data() + json.size(), &props, &errors))
        << errors;
    EXPECT_EQ(props.isMember("memoryBusWidth"), width.has_value());
    if (width.has_value()) {
      EXPECT_EQ(props["memoryBusWidth"].asInt64(), *width);
    }

    const auto target = GetCudnnGraphTargetDevice(desc);
    ASSERT_TRUE(target.has_value());
    EXPECT_EQ(target->core_count, 80);
    EXPECT_EQ(target->serialized_device_properties, json);
    EXPECT_EQ(desc, before);
  }
}

TEST(CudnnDevicePropsJsonTest, NonCudaHasNoTargetDevice) {
  ASSERT_OK_AND_ASSIGN(auto desc,
                       stream_executor::DeviceDescription::FromProto({}));
  desc.set_rocm_compute_capability("gfx90a");
  desc.set_memory_bus_width(5120);
  EXPECT_EQ(GetCudnnGraphTargetDevice(desc), std::nullopt);
}

}  // namespace
}  // namespace xla::gpu
