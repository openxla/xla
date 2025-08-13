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
#include "absl/log/log_sink.h"
#include "absl/log/scoped_mock_log.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/model/gpu_collective_performance_model.h"

namespace xla {
namespace gpu {
namespace {

using GpuPerformanceWithCollectiveModelTest = HloHardwareIndependentTestBase;

TEST_F(GpuPerformanceWithCollectiveModelTest, TestNvmlLibraryLoading) {
#if GOOGLE_CUDA
  EXPECT_TRUE(GpuPerformanceWithCollectiveModel::InitNvml());
  // After successful init, we try to use one of the
  // nvml functions to see if the result is good.
  nvmlDevice_t nvml_device;
  nvmlReturn_t get_device_result =
      xla_nvmlDeviceGetHandleByIndex(0, &nvml_device);
  EXPECT_TRUE(get_device_result == NVML_SUCCESS);

  EXPECT_TRUE(GpuPerformanceWithCollectiveModel::InitNvml());

#endif  // GOOGLE_CUDA
}

TEST_F(GpuPerformanceWithCollectiveModelTest, TestNvmlLibraryLoadingWarning) {
#if GOOGLE_CUDA && defined(PLATFORM_POSIX) && !defined(PLATFORM_GOOGLE)
  absl::ScopedMockLog mock_log(absl::MockLogDefault::kIgnoreUnexpected);

  bool warning_caught = false;

  EXPECT_CALL(mock_log,
              Log(absl::LogSeverity::kWarning, ::testing::_, ::testing::_))
      .Times(::testing::AnyNumber())
      .WillRepeatedly([&warning_caught](absl::LogSeverity severity,
                                        const std::string& filename,
                                        const std::string& message) {
        if (message.find("undefined symbol:") != std::string::npos) {
          warning_caught = true;
          EXPECT_THAT(message, ::testing::ContainsRegex("undefined symbol:"));
        }
      });

  mock_log.StartCapturingLogs();
  GpuPerformanceWithCollectiveModel::InitNvml();
  mock_log.StopCapturingLogs();

  EXPECT_TRUE(warning_caught);
#endif
}

}  // namespace
}  // namespace gpu
}  // namespace xla
