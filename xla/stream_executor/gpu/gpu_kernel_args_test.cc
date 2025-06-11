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
#include "xla/stream_executor/gpu/gpu_kernel.h"

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
  std::vector<int> test_limits = {4095, 1024};
  int actual_limit = 0;

  auto platform_name = platform()->Name();

  // Find the highest supported limit
  for (int limit : test_limits) {
    std::vector<DeviceMemoryBase> test_args(limit);
    for (int i = 0; i < limit; ++i) {
      test_args[i] = DeviceMemoryBase(reinterpret_cast<void*>(0x12345678), 42);
    }
    auto test_result = gpu::PackKernelArgs<DeviceMemoryBase>(test_args, 0);
    if (test_result.ok()) {
      actual_limit = limit;
      break;  // Found the highest working limit
    }
  }

  EXPECT_GT(actual_limit, 0)
      << "Platform " << platform_name
      << " should support at least some kernel arguments";

  // Test that we can pack up to the actual limit
  std::vector<DeviceMemoryBase> args(actual_limit);
  for (int i = 0; i < actual_limit; ++i) {
    args[i] = DeviceMemoryBase(reinterpret_cast<void*>(0x12345678), 42);
  }
  auto result = gpu::PackKernelArgs<DeviceMemoryBase>(args, 0);
  EXPECT_TRUE(result.ok()) << "Failed at detected limit " << actual_limit
                           << " on platform " << platform_name << ": "
                           << result.status();
  EXPECT_EQ(result.value()->number_of_arguments(), actual_limit);

  // Test that limit + 1 fails
  args.push_back(DeviceMemoryBase(reinterpret_cast<void*>(0x12345678), 42));
  result = gpu::PackKernelArgs<DeviceMemoryBase>(args, 0);
  EXPECT_FALSE(result.ok()) << "Should fail at " << (actual_limit + 1)
                            << " on platform " << platform_name;
  EXPECT_THAT(result.status().message(),
              ::testing::HasSubstr("Can't pack device memory arguments array"));
}

}  // namespace stream_executor