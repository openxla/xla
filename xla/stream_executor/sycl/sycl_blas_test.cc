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
#include <vector>

#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::sycl {
namespace {

using ::testing::ElementsAre;

TEST(SyclBlasTest, DoBlasScalFloat) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithId(kSyclPlatformId));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                       executor->CreateStream());
  blas::BlasSupport* blas = executor->AsBlas();
  ASSERT_NE(blas, nullptr);

  std::vector<float> host = {1.0f, 2.0f, 3.0f, 4.0f};
  const uint64_t size_bytes = host.size() * sizeof(float);
  DeviceAddress<float> device = executor->AllocateArray<float>(host.size());
  ASSERT_THAT(stream->Memcpy(&device, host.data(), size_bytes),
              absl_testing::IsOk());

  EXPECT_TRUE(blas->DoBlasScal(stream.get(), host.size(), 0.5f, &device, 1));

  ASSERT_THAT(stream->Memcpy(host.data(), device, size_bytes),
              absl_testing::IsOk());
  ASSERT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());
  EXPECT_THAT(host, ElementsAre(0.5f, 1.0f, 1.5f, 2.0f));
  executor->Deallocate(&device);
}

}  // namespace
}  // namespace stream_executor::sycl

