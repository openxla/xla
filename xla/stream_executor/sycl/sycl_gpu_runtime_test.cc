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
#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"

#include <gtest/gtest.h>

#include "xla/tsl/platform/status_matchers.h"

namespace stream_executor::gpu {
namespace {

TEST(SyclGpuRuntimeTest, GetDeviceCount) {
  TF_ASSERT_OK_AND_ASSIGN(int device_count, SyclDevicePool::GetDeviceCount());
  EXPECT_GT(device_count, 0);
}

TEST(SyclGpuRuntimeTest, GetDeviceOrdinal) {
  TF_ASSERT_OK_AND_ASSIGN(sycl::device sycl_device,
                          SyclDevicePool::GetDevice(kDefaultDeviceOrdinal));
  TF_ASSERT_OK_AND_ASSIGN(int device_ordinal,
                          SyclDevicePool::GetDeviceOrdinal(sycl_device));
  EXPECT_EQ(device_ordinal, kDefaultDeviceOrdinal);
}

}  // namespace
}  // namespace stream_executor::gpu
