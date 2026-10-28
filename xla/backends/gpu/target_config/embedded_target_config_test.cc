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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "tsl/platform/path.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/protobuf.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"

namespace stream_executor {
namespace {

// Older stored profiles do not supply memory bus width. Compare only that
// field's known values without weakening DeviceDescription equality globally.
DeviceDescription ForStoredSpecComparison(DeviceDescription physical,
                                         const GpuDeviceInfoProto& stored) {
  if (!stored.has_memory_bus_width()) {
    physical.set_memory_bus_width(std::nullopt);
  }
  return physical;
}

TEST(EmbeddedTargetConfigTest, LegacySpecMemoryBusWidth) {
  DeviceDescription physical;
  physical.set_memory_bus_width(7680);
  GpuDeviceInfoProto stored = physical.ToProto();
  stored.clear_memory_bus_width();
  const DeviceDescription compared = ForStoredSpecComparison(physical, stored);
  EXPECT_TRUE(tsl::protobuf::util::MessageDifferencer::Equals(compared.ToProto(),
                                                           stored));
  ASSERT_OK_AND_ASSIGN(DeviceDescription stored_description,
                       DeviceDescription::FromProto(stored));
  EXPECT_TRUE(compared.EqualsTo(stored_description));
  EXPECT_EQ(physical.memory_bus_width(), 7680);
}

TEST(EmbeddedTargetConfigTest, KnownMemoryBusWidthIsCompared) {
  DeviceDescription physical;
  physical.set_memory_bus_width(7680);
  for (int64_t width : {int64_t{0}, int64_t{5120}, int64_t{7680}}) {
    SCOPED_TRACE(width);
    GpuDeviceInfoProto stored = physical.ToProto();
    stored.set_memory_bus_width(width);
    const DeviceDescription compared = ForStoredSpecComparison(physical, stored);
    EXPECT_EQ(tsl::protobuf::util::MessageDifferencer::Equals(compared.ToProto(),
                                                           stored),
              width == 7680);
    ASSERT_OK_AND_ASSIGN(DeviceDescription stored_description,
                         DeviceDescription::FromProto(stored));
    EXPECT_EQ(compared.EqualsTo(stored_description), width == 7680);
  }
}

TEST(EmbeddedTargetConfigTest, LegacySpecRetainsOtherFields) {
  DeviceDescription physical;
  physical.set_memory_bus_width(7680);
  physical.set_core_count(148);
  GpuDeviceInfoProto stored = physical.ToProto();
  stored.clear_memory_bus_width();
  stored.set_core_count(149);
  const DeviceDescription compared = ForStoredSpecComparison(physical, stored);
  EXPECT_FALSE(tsl::protobuf::util::MessageDifferencer::Equals(compared.ToProto(),
                                                            stored));
  ASSERT_OK_AND_ASSIGN(DeviceDescription stored_description,
                       DeviceDescription::FromProto(stored));
  EXPECT_FALSE(compared.EqualsTo(stored_description));
}

TEST(EmbeddedTargetConfigTest, MissingLiveWidthDoesNotMatchKnownSpec) {
  DeviceDescription physical;
  GpuDeviceInfoProto stored = physical.ToProto();
  stored.set_memory_bus_width(0);
  const DeviceDescription compared = ForStoredSpecComparison(physical, stored);
  EXPECT_FALSE(tsl::protobuf::util::MessageDifferencer::Equals(compared.ToProto(),
                                                            stored));
  ASSERT_OK_AND_ASSIGN(DeviceDescription stored_description,
                       DeviceDescription::FromProto(stored));
  EXPECT_FALSE(compared.EqualsTo(stored_description));
}

TEST(EmbeddedTargetConfigTest, DeviceInfoMatches) {
  absl::flat_hash_map<std::string, GpuDeviceInfoProto> gpu_specs;
  for (const std::string file_name :
       {"a100_pcie_80", "a100_sxm_40", "a100_sxm_80", "a6000", "b200",
        "b200_mig", "bmg_g21", "h100_pcie", "h100_sxm", "h100_sxm_mig", "p100",
        "v100", "mi200", "rtx6000pro", "gb200", "gb300", "vr_nvl72"}) {
    GpuTargetConfigProto proto;
    std::string spec_string;
    ASSERT_OK(tsl::ReadFileToString(
        tsl::Env::Default(),
        tsl::io::JoinPath(tsl::testing::XlaSrcRoot(),
                          "backends/gpu/target_config/specs",
                          absl::StrCat(file_name, ".txtpb")),
        &spec_string));
    ASSERT_TRUE(
        tsl::protobuf::TextFormat::ParseFromString(spec_string, &proto));
    gpu_specs[proto.device_description_str()] = proto.gpu_device_info();
  }
  auto name = absl::AsciiStrToUpper(
      xla::PlatformUtil::CanonicalPlatformName("gpu").value());
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName(name));
  ASSERT_GT(platform->VisibleDeviceCount(), 0)
      << "No visible GPU devices found (cuInit may have failed).";
  bool all_skipped = true;
  for (int i = 0; i < platform->VisibleDeviceCount(); ++i) {
    ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                         platform->ExecutorForDevice(i));
    const DeviceDescription& physical_device_description =
        executor->GetDeviceDescription();

    // TODO(b/505078018): Do not skip once fixed.
    if (physical_device_description.gpu_compute_capability().IsCuda() &&
        physical_device_description.cuda_compute_capability().IsAmpere()) {
      continue;
    }

    auto it = gpu_specs.find(physical_device_description.name());
    if (it == gpu_specs.end()) {
      if (!tsl::kIsOpenSource) {
        FAIL() << "No spec file for " << physical_device_description.name();
      }
      LOG(WARNING) << "No spec file for " << physical_device_description.name();
      continue;
    }
    all_skipped = false;
    const GpuDeviceInfoProto& stored_device_info = it->second;
    const DeviceDescription compared_device_description =
        ForStoredSpecComparison(physical_device_description, stored_device_info);
    const GpuDeviceInfoProto physical_device_info =
        compared_device_description.ToProto();
    VLOG(1) << physical_device_description.name();
    VLOG(1) << physical_device_info.DebugString();
    tsl::protobuf::util::MessageDifferencer diff;
    diff.IgnoreField(
        GpuDeviceInfoProto::GetDescriptor()->FindFieldByName("pci_bus_id"));
    diff.IgnoreField(
        GpuDeviceInfoProto::GetDescriptor()->FindFieldByName("numa_node"));
    diff.IgnoreField(GpuDeviceInfoProto::GetDescriptor()->FindFieldByName(
        "kernel_mode_driver_version"));
    diff.IgnoreField(
        GpuDeviceInfoProto::GetDescriptor()->FindFieldByName("driver_version"));
    diff.IgnoreField(GpuDeviceInfoProto::GetDescriptor()->FindFieldByName(
        "runtime_version"));
    diff.IgnoreField(GpuDeviceInfoProto::GetDescriptor()->FindFieldByName(
        "compile_time_toolkit_version"));
    diff.IgnoreField(
        GpuDeviceInfoProto::GetDescriptor()->FindFieldByName("dnn_version"));
    diff.IgnoreField(
        GpuDeviceInfoProto::GetDescriptor()->FindFieldByName("cub_version"));
    diff.IgnoreField(
        DeviceInterconnectInfoProto::GetDescriptor()->FindFieldByName(
            "cluster_uuid"));
    diff.IgnoreField(
        DeviceInterconnectInfoProto::GetDescriptor()->FindFieldByName(
            "clique_id"));
    diff.IgnoreField(GpuDeviceInfoProto::GetDescriptor()->FindFieldByName(
        "device_memory_size"));
    diff.IgnoreField(
        GpuDeviceInfoProto::GetDescriptor()->FindFieldByName("model_str"));
    diff.set_message_field_comparison(
        tsl::protobuf::util::MessageDifferencer::EQUIVALENT);
    std::string result;
    diff.ReportDifferencesToString(&result);
    EXPECT_TRUE(diff.Compare(physical_device_info, stored_device_info))
        << result;
    EXPECT_NEAR(physical_device_info.device_memory_size(),
                stored_device_info.device_memory_size(), 1024 * 1024 * 1024);
    ASSERT_OK_AND_ASSIGN(DeviceDescription stored_device_description,
                         DeviceDescription::FromProto(stored_device_info));
    EXPECT_TRUE(compared_device_description.EqualsTo(
        stored_device_description,
        {DeviceDescription::CompareOptions::kIgnoreVersionNumbers,
         DeviceDescription::CompareOptions::kPortable}));
  }
  if (all_skipped) {
    GTEST_SKIP() << "No spec files for any of the available GPUs.";
  }
}

}  // namespace
}  // namespace stream_executor
