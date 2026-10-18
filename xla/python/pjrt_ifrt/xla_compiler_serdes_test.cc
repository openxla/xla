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

#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/device_test_util.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/program_serdes.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes.pb.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAreArray;
using ::testing::HasSubstr;

using XlaCompilerSerDesTestParam =
    std::tuple<SerDesVersion, test_util::DeviceTestParam>;

class XlaCompilerSerDesTest
    : public testing::TestWithParam<XlaCompilerSerDesTestParam> {
 public:
  XlaCompilerSerDesTest()
      : version_(std::get<0>(GetParam())), fixture_(std::get<1>(GetParam())) {}

  SerDesVersion version() const { return version_; }

  Client* client() { return fixture_.client(); }
  DeviceListRef GetDevices(absl::Span<const int> device_indices) {
    return fixture_.GetDevices(device_indices);
  }

 private:
  SerDesVersion version_;
  test_util::DeviceTestFixture fixture_;
};

TEST_P(XlaCompilerSerDesTest, XlaDeserializeExecutableOptionsRoundTrip) {
  xla::CompileOptions compile_options;
  compile_options.executable_build_options.set_num_replicas(2);
  DeviceListRef devices = GetDevices({0, 1});
  XlaDeserializeExecutableOptions options(compile_options, devices);

  ASSERT_OK_AND_ASSIGN(
      Serialized serialized,
      Serialize(options, std::make_unique<SerializeOptions>(version())));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XlaDeserializeExecutableOptions> deserialized,
      Deserialize<XlaDeserializeExecutableOptions>(
          serialized, std::make_unique<DeserializeProgramOptions>(client())));
  ASSERT_TRUE(deserialized->compile_options.has_value());
  EXPECT_EQ(
      deserialized->compile_options->executable_build_options.num_replicas(),
      2);
  ASSERT_TRUE(deserialized->devices.has_value());
  EXPECT_THAT((*deserialized->devices)->devices(),
              ElementsAreArray(devices->devices()));

  EXPECT_THAT(Deserialize<XlaDeserializeExecutableOptions>(serialized,
                                                           /*options=*/nullptr),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("requires Client")));

  ASSERT_OK_AND_ASSIGN(
      Serialized empty_serialized,
      Serialize(XlaDeserializeExecutableOptions(),
                std::make_unique<SerializeOptions>(version())));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XlaDeserializeExecutableOptions> empty_deserialized,
      Deserialize<XlaDeserializeExecutableOptions>(empty_serialized,
                                                   /*options=*/nullptr));
  EXPECT_FALSE(empty_deserialized->compile_options.has_value());
  EXPECT_FALSE(empty_deserialized->devices.has_value());

  XlaDeserializeExecutableOptions callback_options(
      /*compile_options=*/std::nullopt, /*devices=*/std::nullopt,
      {tsl::MakeRef<MockLoadedHostCallback>()});
  EXPECT_THAT(Serialize(callback_options,
                        std::make_unique<SerializeOptions>(version())),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     HasSubstr("loaded_host_callbacks")));
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion_NumDevices, XlaCompilerSerDesTest,
    testing::Combine(testing::ValuesIn(test_util::AllSupportedSerDesVersions()),
                     testing::Values(test_util::DeviceTestParam{
                         /*num_devices=*/2, /*num_addressable_devices=*/2})),
    [](const testing::TestParamInfo<XlaCompilerSerDesTestParam>& info) {
      return absl::StrCat("version_",
                          std::get<0>(info.param).version_number().value(),
                          "_num_devices_", std::get<1>(info.param).num_devices,
                          "_num_addressable_devices_",
                          std::get<1>(info.param).num_addressable_devices);
    });

}  // namespace
}  // namespace ifrt
}  // namespace xla
