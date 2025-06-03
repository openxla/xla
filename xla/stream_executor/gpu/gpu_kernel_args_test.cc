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

#include "xla/stream_executor/kernel.h"

#include <memory>
#include <vector>

#include "absl/strings/ascii.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/platform/test.h"

namespace stream_executor {

class GpuKernelArgsTest : public ::testing::Test {
 public:
  void SetUp() override {
    auto name = absl::AsciiStrToUpper(
        xla::PlatformUtil::CanonicalPlatformName("gpu").value());
    platform_ = PlatformManager::PlatformWithName(name).value();
  }

  Platform* platform() { return platform_; }

 private:
  Platform* platform_;
};

TEST_F(GpuKernelArgsTest, PackLargeNumberOfArguments) {
#if GOOGLE_CUDA && CUDA_VERSION >= 12010
  std::vector<DeviceMemoryBase> args(4096);
  for (int i = 0; i < 4096; ++i) {
    args[i] = DeviceMemoryBase(reinterpret_cast<void*>(0x12345678), 42);
  }
  auto result = PackKernelArgs<DeviceMemoryBase>(args, 0);
  EXPECT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result.value()->number_of_arguments(), 4096);

  args.push_back(DeviceMemoryBase(reinterpret_cast<void*>(0x12345678), 42));
  result = PackKernelArgs<DeviceMemoryBase>(args, 0);
  EXPECT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              ::testing::HasSubstr(
                  "Can't pack device memory arguments array of size 4097"));
#else
  std::vector<DeviceMemoryBase> args(1024);
  for (int i = 0; i < 1024; ++i) {
    args[i] = DeviceMemoryBase(reinterpret_cast<void*>(0x12345678), 42);
  }
  auto result = PackKernelArgs<DeviceMemoryBase>(args, 0);
  EXPECT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result.value()->number_of_arguments(), 1024);

  args.push_back(DeviceMemoryBase(reinterpret_cast<void*>(0x12345678), 42));
  result = PackKernelArgs<DeviceMemoryBase>(args, 0);
  EXPECT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              ::testing::HasSubstr(
                  "Can't pack device memory arguments array of size 1025"));
#endif
}

}  // namespace stream_executor