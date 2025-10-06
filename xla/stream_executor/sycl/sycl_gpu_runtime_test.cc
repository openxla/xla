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

namespace stream_executor::sycl {
namespace {

TEST(SyclGpuRuntimeTest, GetDeviceCount) {
  EXPECT_THAT(SyclDevicePool::GetDeviceCount(),
              ::absl_testing::IsOkAndHolds(::testing::Gt(0)));
}

TEST(SyclGpuRuntimeTest, GetDeviceOrdinal) {
  TF_ASSERT_OK_AND_ASSIGN(::sycl::device sycl_device,
                          SyclDevicePool::GetDevice(kDefaultDeviceOrdinal));
  TF_ASSERT_OK_AND_ASSIGN(int device_ordinal,
                          SyclDevicePool::GetDeviceOrdinal(sycl_device));
  EXPECT_EQ(device_ordinal, kDefaultDeviceOrdinal);
}

TEST(SyclGpuRuntimeTest, TestStaticDeviceContext) {
  // Verify that GetDeviceContext returns the same context instance on multiple
  // calls.
  TF_ASSERT_OK_AND_ASSIGN(::sycl::context saved_sycl_context,
                          SyclDevicePool::GetDeviceContext());
  TF_ASSERT_OK_AND_ASSIGN(::sycl::context current_sycl_context,
                          SyclDevicePool::GetDeviceContext());
  EXPECT_EQ(saved_sycl_context, current_sycl_context);
}

TEST(SyclGpuRuntimeTest, TestStreamPoolCreateSynchronizeAndDestroy) {
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  ASSERT_TRUE(
      SyclStreamPool::SynchronizeStreamPool(kDefaultDeviceOrdinal).ok());

  ASSERT_TRUE(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle).ok());
  EXPECT_EQ(stream_handle, nullptr);
}

TEST(SyclGpuRuntimeTest, TestStreamPoolCreateAfterDestroy) {
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  ASSERT_TRUE(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle).ok());
  ASSERT_EQ(stream_handle, nullptr);

  // Verify that we can create a new stream after destroying the previous one.
  TF_ASSERT_OK_AND_ASSIGN(
      stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  // Clean up the stream after the test.
  ASSERT_TRUE(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle).ok());
  EXPECT_EQ(stream_handle, nullptr);
}

TEST(SyclGpuRuntimeTest, TestStreamPoolCreate_Negative) {
  constexpr int kInvalidDeviceOrdinal = -1;
  EXPECT_EQ(SyclStreamPool::GetOrCreateStream(kInvalidDeviceOrdinal,
                                              /*enable_multiple_streams=*/false)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(SyclGpuRuntimeTest, TestStreamPoolDestroy_Negative) {
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  ASSERT_TRUE(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle).ok());
  ASSERT_EQ(stream_handle, nullptr);

  // Try to destroy the stream again, which should be a no-op.
  EXPECT_EQ(SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle)
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(stream_handle, nullptr);
}

TEST(SyclGpuRuntimeTest, TestMaxStreamsPerDevice) {
  // Ensure that the maximum number of streams per device is respected.
  constexpr int kMaxStreams = 8;
  std::vector<StreamPtr> streams(kMaxStreams);
  for (int i = 0; i < kMaxStreams - 1; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(streams[i], SyclStreamPool::GetOrCreateStream(
                                            kDefaultDeviceOrdinal,
                                            /*enable_multiple_streams=*/true));
    ASSERT_NE(streams[i], nullptr);
  }

  // Attempt to create one more stream, which should fail.
  EXPECT_EQ(SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                              /*enable_multiple_streams=*/true)
                .status()
                .code(),
            absl::StatusCode::kResourceExhausted);

  // Clean up the streams created.
  for (int i = 0; i < kMaxStreams - 1; ++i) {
    ASSERT_TRUE(
        SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, streams[i]).ok());
    EXPECT_EQ(streams[i], nullptr);
  }
}

}  // namespace
}  // namespace stream_executor::sycl
