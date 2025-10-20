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

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/backends/gpu/runtime/thunk_buffer_id.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/buffer_debug_log.h"
#include "xla/stream_executor/gpu/buffer_debug_xor_checksum_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/stream_executor/typed_kernel_factory.h"  // IWYU pragma: keep, required for KernelType::FactoryType::Create
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace se = stream_executor;

namespace stream_executor::cuda {
namespace {

using xla::gpu::BufferDebugLogEntry;
using xla::gpu::BufferDebugLogHeader;
using xla::gpu::ThunkBufferId;
using xla::gpu::ThunkId;

class ChecksumKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(platform_,
                            se::PlatformManager::PlatformWithName("CUDA"));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    TF_ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream(std::nullopt));
    allocator_ =
        std::make_unique<se::StreamExecutorMemoryAllocator>(stream_->parent());

    if (!executor_->GetDeviceDescription()
             .cuda_compute_capability()
             .IsAtLeastPascal()) {
      GTEST_SKIP()
          << "Buffer checksumming is not supported on CUDA architectures older "
             "than Pascal due to missing atomic fetch_add with system scope";
    }
  }

  template <typename T>
  absl::StatusOr<se::DeviceMemory<T>> CheckNotNull(
      se::DeviceMemory<T> device_memory, absl::string_view name) {
    if (device_memory.is_null()) {
      return absl::InternalError(
          absl::StrFormat("Device memory for %s is null", name));
    }
    return device_memory;
  }

  template <typename T>
  absl::Status AppendChecksumOnDevice(
      ThunkBufferId entry_id, const T& input,
      se::cuda::BufferDebugLog& buffer_debug_log,
      stream_executor::ThreadDim dim = stream_executor::ThreadDim(1, 1, 1)) {
    // Load kernel
    gpu::GpuKernelRegistry registry =
        gpu::GpuKernelRegistry::GetGlobalRegistry();
    TF_ASSIGN_OR_RETURN(
        auto kernel,
        registry.LoadKernel<gpu::BufferDebugXorChecksumKernel>(executor_));

    // Setup device buffers
    TF_ASSIGN_OR_RETURN(se::DeviceMemory<uint8_t> device_input,
                        CheckNotNull(executor_->AllocateArray<uint8_t>(
                                         input.size() * sizeof(input[0])),
                                     "input"));
    auto cleanup_input =
        absl::MakeCleanup([&]() { executor_->Deallocate(&device_input); });

    // Call kernel
    TF_RETURN_IF_ERROR(stream_->Memcpy(&device_input, input.data(),
                                       input.size() * sizeof(input[0])));
    TF_RETURN_IF_ERROR(kernel.Launch(dim, stream_executor::BlockDim(1, 1, 1),
                                     stream_.get(), entry_id, device_input,
                                     device_input.ElementCount(),
                                     buffer_debug_log.GetDeviceHeader(),
                                     buffer_debug_log.GetDeviceEntries()));
    TF_RETURN_IF_ERROR(stream_->BlockHostUntilDone());

    // The result gets stored in `buffer_debug_log`.
    return absl::OkStatus();
  }

  se::Platform* platform_;
  se::StreamExecutor* executor_;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<se::StreamExecutorMemoryAllocator> allocator_;
};

TEST_F(ChecksumKernelTest, ComputesCorrectChecksumForMultipleOf32Bit) {
  se::DeviceMemory<uint8_t> mem = executor_->AllocateArray<uint8_t>(1024);
  std::vector<uint8_t> input = std::vector<uint8_t>(1024, 0x55);
  // Xor with the expected checksum value.
  // Assumes the device uses little-endian byte order.
  input[1000] ^= 0x78;
  input[1001] ^= 0x56;
  input[1002] ^= 0x34;
  input[1003] ^= 0x12;
  constexpr uint32_t kExpectedChecksum = 0x12345678;

  TF_ASSERT_OK_AND_ASSIGN(
      se::cuda::BufferDebugLog device_log,
      se::cuda::BufferDebugLog::CreateOnDevice(*stream_, mem));

  TF_EXPECT_OK(AppendChecksumOnDevice(ThunkBufferId(), input, device_log));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));
  ASSERT_GE(host_log.size(), 1);
  EXPECT_EQ(host_log[0].checksum, kExpectedChecksum);
}

TEST_F(ChecksumKernelTest,
       PadsMostSignifantBitsOfIncomplete32BitInputWordWithZeros) {
  se::DeviceMemory<uint8_t> mem = executor_->AllocateArray<uint8_t>(1024);
  const std::vector<uint8_t> kInput = std::vector<uint8_t>(1023, 0x55);
  TF_ASSERT_OK_AND_ASSIGN(
      se::cuda::BufferDebugLog device_log,
      se::cuda::BufferDebugLog::CreateOnDevice(*stream_, mem));

  TF_EXPECT_OK(AppendChecksumOnDevice(ThunkBufferId(), kInput, device_log));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));
  ASSERT_GE(host_log.size(), 1);
  // Assumes the device uses little-endian byte order.
  EXPECT_EQ(host_log[0].checksum, 0x55000000);
}

TEST_F(ChecksumKernelTest, ComputesCorrectChecksumInParallel) {
  se::DeviceMemory<uint8_t> mem = executor_->AllocateArray<uint8_t>(1024);
  std::vector<uint32_t> input =
      std::vector<uint32_t>(64 * 1024 / sizeof(uint32_t), 0x55aa55aa);
  // Xor with the expected checksum value.
  input[1000] ^= 0x12345678;
  constexpr uint32_t kExpectedChecksum = 0x12345678;
  TF_ASSERT_OK_AND_ASSIGN(
      se::cuda::BufferDebugLog device_log,
      se::cuda::BufferDebugLog::CreateOnDevice(*stream_, mem));

  TF_EXPECT_OK(AppendChecksumOnDevice(ThunkBufferId(), input, device_log,
                                      se::ThreadDim(2, 4, 8)));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));
  ASSERT_GE(host_log.size(), 1);
  EXPECT_EQ(host_log[0].checksum, kExpectedChecksum);
}

TEST_F(ChecksumKernelTest, ComputesCorrectChecksumInParallelWithMaxThreads) {
  se::DeviceMemory<uint8_t> mem = executor_->AllocateArray<uint8_t>(1024);
  std::vector<uint32_t> input =
      std::vector<uint32_t>(64 * 1024 / sizeof(uint32_t), 0x55aa55aa);
  // Xor with the expected checksum value.
  input[1000] ^= 0x12345678;
  constexpr uint32_t kExpectedChecksum = 0x12345678;
  TF_ASSERT_OK_AND_ASSIGN(
      se::cuda::BufferDebugLog device_log,
      se::cuda::BufferDebugLog::CreateOnDevice(*stream_, mem));

  TF_EXPECT_OK(AppendChecksumOnDevice(ThunkBufferId(), input, device_log,
                                      se::ThreadDim(128, 4, 2)));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));
  ASSERT_GE(host_log.size(), 1);
  EXPECT_EQ(host_log[0].checksum, kExpectedChecksum);
}

TEST_F(ChecksumKernelTest, AppendsChecksumsToLog) {
  se::DeviceMemory<uint8_t> mem = executor_->AllocateArray<uint8_t>(1024);
  ThunkBufferId kId123 = ThunkBufferId::Create(ThunkId(123), 0).value();
  ThunkBufferId kId456 = ThunkBufferId::Create(ThunkId(456), 0).value();
  ThunkBufferId kId789 = ThunkBufferId::Create(ThunkId(789), 0).value();
  constexpr std::array<uint32_t, 1> kInput123 = {0x01230123};
  constexpr std::array<uint32_t, 1> kInput456 = {0x04560456};
  constexpr std::array<uint32_t, 1> kInput789 = {0x07890789};
  TF_ASSERT_OK_AND_ASSIGN(
      se::cuda::BufferDebugLog device_log,
      se::cuda::BufferDebugLog::CreateOnDevice(*stream_, mem));

  TF_EXPECT_OK(AppendChecksumOnDevice(kId123, kInput123, device_log));
  TF_EXPECT_OK(AppendChecksumOnDevice(kId456, kInput456, device_log));
  TF_EXPECT_OK(AppendChecksumOnDevice(kId789, kInput789, device_log));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));
  ASSERT_GE(host_log.size(), 3);
  EXPECT_EQ(host_log[0].entry_id, kId123);
  EXPECT_EQ(host_log[0].checksum, 0x01230123);
  EXPECT_EQ(host_log[1].entry_id, kId456);
  EXPECT_EQ(host_log[1].checksum, 0x04560456);
  EXPECT_EQ(host_log[2].entry_id, kId789);
  EXPECT_EQ(host_log[2].checksum, 0x07890789);
}

TEST_F(ChecksumKernelTest, DiscardsOverflowingChecksums) {
  se::DeviceMemory<uint8_t> mem = executor_->AllocateArray<uint8_t>(
      sizeof(BufferDebugLogHeader) + sizeof(BufferDebugLogEntry) * 2);
  ThunkBufferId kId123 = ThunkBufferId::Create(ThunkId(123), 0).value();
  ThunkBufferId kId456 = ThunkBufferId::Create(ThunkId(456), 0).value();
  ThunkBufferId kId789 = ThunkBufferId::Create(ThunkId(789), 0).value();
  constexpr std::array<uint32_t, 1> kInput123 = {0x01230123};
  constexpr std::array<uint32_t, 1> kInput456 = {0x04560456};
  constexpr std::array<uint32_t, 1> kInput789 = {0x07890789};
  TF_ASSERT_OK_AND_ASSIGN(
      se::cuda::BufferDebugLog device_log,
      se::cuda::BufferDebugLog::CreateOnDevice(*stream_, mem));

  TF_EXPECT_OK(AppendChecksumOnDevice(kId123, kInput123, device_log));
  TF_EXPECT_OK(AppendChecksumOnDevice(kId456, kInput456, device_log));
  // This entry will be discarded.
  TF_EXPECT_OK(AppendChecksumOnDevice(kId789, kInput789, device_log));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));
  ASSERT_GE(host_log.size(), 2);
  EXPECT_EQ(host_log[0].entry_id, kId123);
  EXPECT_EQ(host_log[0].checksum, 0x01230123);
  EXPECT_EQ(host_log[1].entry_id, kId456);
  EXPECT_EQ(host_log[1].checksum, 0x04560456);
}

}  // namespace
}  // namespace stream_executor::cuda
