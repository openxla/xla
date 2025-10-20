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

class SyclGpuRuntimeTest : public ::testing::Test {
 public:
  std::vector<::sycl::device> sycl_devices_;

 protected:
  absl::StatusOr<void*> AllocateHostBuffer(int count) {
    TF_ASSIGN_OR_RETURN(
        void* buf, SyclMallocHost(kDefaultDeviceOrdinal, sizeof(int) * count));
    if (buf == nullptr) {
      return absl::InternalError(
          "SyclGpuRuntimeTest::AllocateHostBuffer: Failed to allocate host "
          "buffer.");
    }
    return buf;
  }

  absl::StatusOr<void*> AllocateDeviceBuffer(
      int count, int device_ordinal = kDefaultDeviceOrdinal) {
    TF_ASSIGN_OR_RETURN(void* buf,
                        SyclMallocDevice(device_ordinal, sizeof(int) * count));
    if (buf == nullptr) {
      return absl::InternalError(
          "SyclGpuRuntimeTest::AllocateDeviceBuffer: Failed to allocate "
          "device buffer.");
    }
    return buf;
  }

  void VerifyIntBuffer(void* buf, int count, int expected) {
    for (int i = 0; i < count; ++i) {
      EXPECT_EQ(static_cast<int*>(buf)[i], expected)
          << "Buffer mismatch at index " << i;
    }
  }

  absl::StatusOr<void*> AllocateAndInitHostBuffer(int count, int value) {
    TF_ASSIGN_OR_RETURN(void* buf, AllocateHostBuffer(count));
    for (int i = 0; i < count; ++i) {
      static_cast<int*>(buf)[i] = value;
    }
    return buf;
  }

  absl::StatusOr<void*> AllocateAndInitDeviceBuffer(
      int count, int value, int device_ordinal = kDefaultDeviceOrdinal) {
    TF_ASSIGN_OR_RETURN(void* buf, AllocateDeviceBuffer(count));
    TF_RETURN_IF_ERROR(
        SyclMemfillDevice(device_ordinal, buf, value, sizeof(int) * count));
    if (buf == nullptr) {
      return absl::InternalError(
          "SyclGpuRuntimeTest::AllocateAndInitDeviceBuffer: Failed to fill "
          "device buffer.");
    }
    return buf;
  }

  void FreeAndNullify(void*& ptr, int device_ordinal = kDefaultDeviceOrdinal) {
    if (ptr != nullptr) {
      EXPECT_THAT(SyclFree(device_ordinal, ptr), absl_testing::IsOk());
      EXPECT_EQ(ptr, nullptr);
    }
  }

 private:
  void SetUp() override {
    // Find the number of SYCL devices available. If there are none, skip the
    // test.
    TF_ASSERT_OK_AND_ASSIGN(int device_count, SyclDevicePool::GetDeviceCount());
    if (device_count <= 0) {
      GTEST_SKIP() << "No SYCL devices found.";
    } else {
      VLOG(2) << "Found " << device_count << " SYCL devices.";
    }

    // Initialize the device pool with available devices.
    for (int i = 0; i < device_count; ++i) {
      TF_ASSERT_OK_AND_ASSIGN(::sycl::device sycl_device,
                              SyclDevicePool::GetDevice(i));
      sycl_devices_.push_back(sycl_device);
    }
  }
};

TEST_F(SyclGpuRuntimeTest, GetDeviceCount) {
  EXPECT_THAT(SyclDevicePool::GetDeviceCount(),
              ::absl_testing::IsOkAndHolds(::testing::Gt(0)));
}

TEST_F(SyclGpuRuntimeTest, GetDeviceOrdinal) {
  TF_ASSERT_OK_AND_ASSIGN(::sycl::device sycl_device,
                          SyclDevicePool::GetDevice(kDefaultDeviceOrdinal));
  TF_ASSERT_OK_AND_ASSIGN(int device_ordinal,
                          SyclDevicePool::GetDeviceOrdinal(sycl_device));
  EXPECT_EQ(device_ordinal, kDefaultDeviceOrdinal);
}

TEST_F(SyclGpuRuntimeTest, TestStaticDeviceContext) {
  // Verify that GetDeviceContext returns the same context instance on multiple
  // calls.
  TF_ASSERT_OK_AND_ASSIGN(::sycl::context saved_sycl_context,
                          SyclDevicePool::GetDeviceContext());
  TF_ASSERT_OK_AND_ASSIGN(::sycl::context current_sycl_context,
                          SyclDevicePool::GetDeviceContext());
  EXPECT_EQ(saved_sycl_context, current_sycl_context);
}

TEST_F(SyclGpuRuntimeTest, TestDefaultStreamSynchronizeAndDestroy) {
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetDefaultStream(kDefaultDeviceOrdinal));
  ASSERT_NE(stream_handle, nullptr);

  ASSERT_TRUE(
      SyclStreamPool::SynchronizeStreamPool(kDefaultDeviceOrdinal).ok());

  ASSERT_TRUE(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle).ok());
  EXPECT_EQ(stream_handle, nullptr);
}

TEST_F(SyclGpuRuntimeTest, TestCreateStreamSynchronizeAndDestroy) {
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

TEST_F(SyclGpuRuntimeTest, TestStreamPoolCreateAfterDestroy) {
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

TEST_F(SyclGpuRuntimeTest, TestStreamPoolCreate_Negative) {
  constexpr int kInvalidDeviceOrdinal = -1;
  EXPECT_EQ(SyclStreamPool::GetOrCreateStream(kInvalidDeviceOrdinal,
                                              /*enable_multiple_streams=*/false)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(SyclGpuRuntimeTest, TestStreamPoolDestroy_Negative) {
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

TEST_F(SyclGpuRuntimeTest, TestMaxStreamsPerDevice) {
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

TEST_F(SyclGpuRuntimeTest, TestMemsetDevice) {
  constexpr int kCount = 10;
  TF_ASSERT_OK_AND_ASSIGN(
      void* src_device,
      SyclMallocDevice(kDefaultDeviceOrdinal, sizeof(char) * kCount));
  ASSERT_NE(src_device, nullptr);

  ASSERT_TRUE(SyclMemsetDevice(kDefaultDeviceOrdinal, src_device, 'A',
                               sizeof(char) * kCount)
                  .ok());

  // TODO(intel-tf): Verify the results by copying back to host once SyclMemcpy
  // is implemented.
  FreeAndNullify(src_device);
}

TEST_F(SyclGpuRuntimeTest, TestMemsetDevice_Negative) {
  constexpr int kCount = 10;
  constexpr int kInvalidDeviceOrdinal = -1;

  TF_ASSERT_OK_AND_ASSIGN(void* src_device, AllocateDeviceBuffer(kCount));
  ASSERT_NE(src_device, nullptr);

  // Attempt to memset with an invalid device ordinal.
  EXPECT_EQ(SyclMemsetDevice(kInvalidDeviceOrdinal, src_device, 'A',
                             sizeof(char) * kCount)
                .code(),
            absl::StatusCode::kInvalidArgument);

  // Attempt to memset a null pointer.
  void* null_ptr = nullptr;
  EXPECT_EQ(SyclMemsetDevice(kDefaultDeviceOrdinal, null_ptr, 'A',
                             sizeof(char) * kCount)
                .code(),
            absl::StatusCode::kInvalidArgument);

  FreeAndNullify(src_device);
}

TEST_F(SyclGpuRuntimeTest, TestMemsetDeviceAsync) {
  constexpr int kCount = 10;
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(void* device_buf, AllocateDeviceBuffer(kCount));

  ASSERT_TRUE(SyclMemsetDeviceAsync(stream_handle.get(), device_buf, 'B',
                                    sizeof(char) * kCount)
                  .ok());

  // Synchronize the stream to ensure the memset is complete before checking
  // results.
  ASSERT_TRUE(
      SyclStreamPool::SynchronizeStreamPool(kDefaultDeviceOrdinal).ok());

  // TODO(intel-tf): Verify the results by copying back to host once SyclMemcpy
  // is implemented.
  FreeAndNullify(device_buf);

  // Destroy the stream after use.
  ASSERT_TRUE(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle).ok());
  EXPECT_EQ(stream_handle, nullptr);
}

TEST_F(SyclGpuRuntimeTest, TestMemfillDeviceAsync) {
  constexpr int kCount = 10;
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(void* device_buf, AllocateDeviceBuffer(kCount));

  ASSERT_TRUE(SyclMemfillDeviceAsync(stream_handle.get(), device_buf,
                                     0xDEADC0DE, sizeof(int) * kCount)
                  .ok());

  // Synchronize the stream to ensure the fill is complete before checking
  // results.
  ASSERT_TRUE(
      SyclStreamPool::SynchronizeStreamPool(kDefaultDeviceOrdinal).ok());

  // TODO(intel-tf): Verify the results by copying back to host once SyclMemcpy
  // is implemented.
  FreeAndNullify(device_buf);

  // Destroy the stream after use.
  ASSERT_TRUE(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle).ok());
  ASSERT_EQ(stream_handle, nullptr);
}

TEST_F(SyclGpuRuntimeTest, TestMemfillDeviceAsync_Negative) {
  constexpr int kCount = 10;
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  // Attempt to fill a null pointer.
  void* null_ptr = nullptr;
  EXPECT_EQ(SyclMemfillDeviceAsync(stream_handle.get(), null_ptr, 0xFEEDEAF,
                                   sizeof(int) * kCount)
                .code(),
            absl::StatusCode::kInvalidArgument);

  // Destroy the stream after use.
  ASSERT_TRUE(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle).ok());
  EXPECT_EQ(stream_handle, nullptr);
}

TEST_F(SyclGpuRuntimeTest, TestMallocAll_Positive) {
  TF_ASSERT_OK_AND_ASSIGN(void* host_ptr, AllocateHostBuffer(/*count=*/256));
  FreeAndNullify(host_ptr);

  TF_ASSERT_OK_AND_ASSIGN(void* device_ptr,
                          AllocateDeviceBuffer(/*count=*/256));
  FreeAndNullify(device_ptr);

  TF_ASSERT_OK_AND_ASSIGN(void* shared_ptr,
                          SyclMallocShared(kDefaultDeviceOrdinal,
                                           /*byte_count=*/1024));
  EXPECT_NE(shared_ptr, nullptr);
  FreeAndNullify(shared_ptr);
}

TEST_F(SyclGpuRuntimeTest, TestMallocAll_InvalidDeviceOrdinal) {
  constexpr int kInvalidDeviceOrdinal = -1;
  EXPECT_EQ(SyclMallocHost(kInvalidDeviceOrdinal, 10).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(SyclMallocDevice(kInvalidDeviceOrdinal, 20).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(SyclMallocShared(kInvalidDeviceOrdinal, 30).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(SyclGpuRuntimeTest, TestMallocAll_ZeroAllocation) {
  constexpr size_t kByteCount = 0;
  TF_ASSERT_OK_AND_ASSIGN(void* host_ptr,
                          SyclMallocHost(kDefaultDeviceOrdinal, kByteCount));
  EXPECT_EQ(host_ptr, nullptr)
      << "Expected nullptr for zero allocation on host memory.";
  FreeAndNullify(host_ptr);

  TF_ASSERT_OK_AND_ASSIGN(void* device_ptr,
                          SyclMallocDevice(kDefaultDeviceOrdinal, kByteCount));
  EXPECT_EQ(device_ptr, nullptr)
      << "Expected nullptr for zero allocation on device memory.";
  FreeAndNullify(device_ptr);

  TF_ASSERT_OK_AND_ASSIGN(void* shared_ptr,
                          SyclMallocShared(kDefaultDeviceOrdinal, kByteCount));
  EXPECT_EQ(shared_ptr, nullptr)
      << "Expected nullptr for zero allocation on shared memory.";
  FreeAndNullify(shared_ptr);
}

TEST_F(SyclGpuRuntimeTest, TestSyclFree_Negative) {
  constexpr int kInvalidDeviceOrdinal = -1;
  void* null_ptr = nullptr;  // Null pointer should not cause issues.

  // Attempt to free with an invalid device ordinal.
  EXPECT_EQ(SyclFree(kInvalidDeviceOrdinal, null_ptr).code(),
            absl::StatusCode::kInvalidArgument);

  // Attempt to free a null pointer.
  EXPECT_EQ(SyclFree(kDefaultDeviceOrdinal, null_ptr).code(),
            absl::StatusCode::kInvalidArgument)
      << "Expected error when trying to free a null pointer.";
}

TEST_F(SyclGpuRuntimeTest, TestSyclFree_DoubleFree) {
  TF_ASSERT_OK_AND_ASSIGN(void* device_ptr, AllocateDeviceBuffer(10));
  ASSERT_TRUE(SyclFree(kDefaultDeviceOrdinal, device_ptr).ok());
  EXPECT_EQ(device_ptr, nullptr);

  // Try to free again, which should return an error.
  EXPECT_EQ(SyclFree(kDefaultDeviceOrdinal, device_ptr).code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace stream_executor::sycl
