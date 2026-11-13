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
#include <limits>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "json/json.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

absl::StatusOr<Json::Value> SerializedProperties(
    const stream_executor::DeviceDescription& desc,
    bool include_memory_bus_width) {
  const std::string text =
      FormatDevicePropertiesJson(desc, include_memory_bus_width);
  Json::CharReaderBuilder builder;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  Json::Value props;
  std::string errors;
  if (!reader->parse(text.data(), text.data() + text.size(), &props, &errors)) {
    return absl::InvalidArgumentError(errors);
  }
  return props;
}

TEST(CudnnDevicePropsJsonTest, RecoversPhysicalMemoryBusWidths) {
  struct Profile {
    const char* name;
    int64_t bandwidth;
    float clock_ghz;
    int width_bits;
  };
  // Values from the checked-in profiles. Float GHz storage makes the first
  // two inverted widths slightly smaller than the exact physical integer.
  for (const Profile& profile : {
           Profile{"GB200", 7928064000000, 3.996f, 7936},
           Profile{"B200", 7672320000000, 3.996f, 7680},
           Profile{"H100", 3352320000000, 2.619f, 5120},
       }) {
    SCOPED_TRACE(profile.name);
    stream_executor::DeviceDescription desc;
    desc.set_memory_bandwidth(profile.bandwidth);
    desc.set_mem_clock_ghz(profile.clock_ghz);
    ASSERT_OK_AND_ASSIGN(Json::Value props, SerializedProperties(desc, true));
    ASSERT_TRUE(props.isMember("memoryBusWidth"));
    EXPECT_EQ(props["memoryBusWidth"].asInt64(), profile.width_bits);
  }
}

TEST(CudnnDevicePropsJsonTest, InvalidBandwidthGivesZeroMemoryBusWidth) {
  for (int64_t bandwidth : {int64_t{-1}, int64_t{0}, int64_t{1},
                            std::numeric_limits<int64_t>::max()}) {
    SCOPED_TRACE(bandwidth);
    stream_executor::DeviceDescription desc;
    desc.set_memory_bandwidth(bandwidth);
    desc.set_mem_clock_ghz(1.0f);
    ASSERT_OK_AND_ASSIGN(Json::Value props, SerializedProperties(desc, true));
    ASSERT_TRUE(props.isMember("memoryBusWidth"));
    EXPECT_EQ(props["memoryBusWidth"].asInt64(), 0);
  }
}

TEST(CudnnDevicePropsJsonTest, InvalidMemoryClockGivesZeroMemoryBusWidth) {
  for (float clock : {-1.0f, 0.0f, std::numeric_limits<float>::denorm_min()}) {
    SCOPED_TRACE(clock);
    stream_executor::DeviceDescription desc;
    desc.set_memory_bandwidth(7928064000000);
    desc.set_mem_clock_ghz(clock);
    ASSERT_OK_AND_ASSIGN(Json::Value props, SerializedProperties(desc, true));
    ASSERT_TRUE(props.isMember("memoryBusWidth"));
    EXPECT_EQ(props["memoryBusWidth"].asInt64(), 0);
  }
}

TEST(CudnnDevicePropsJsonTest, MemoryBusWidthIsTheOnlyAddedKey) {
  stream_executor::DeviceDescription desc;
  desc.set_memory_bandwidth(7672320000000);
  desc.set_mem_clock_ghz(3.996f);
  ASSERT_OK_AND_ASSIGN(Json::Value without_key,
                       SerializedProperties(desc, false));
  EXPECT_FALSE(without_key.isMember("memoryBusWidth"));
  ASSERT_OK_AND_ASSIGN(Json::Value with_key, SerializedProperties(desc, true));
  with_key.removeMember("memoryBusWidth");
  EXPECT_EQ(with_key, without_key);
}

}  // namespace
}  // namespace xla::gpu
