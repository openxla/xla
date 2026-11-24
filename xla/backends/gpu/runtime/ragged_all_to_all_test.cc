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

#include "xla/backends/gpu/runtime/ragged_all_to_all.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "third_party/nccl/nccl.h"
#include "xla/backends/gpu/collectives/nccl_symmetric_memory.h"
#include "xla/backends/gpu/collectives/nccl_types.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_handle.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/gpu/ragged_all_to_all_kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::gpu {
namespace {

se::StreamExecutor* GetGpuExecutor() {
  auto* platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()).value();
  return platform->ExecutorForDevice(0).value();
}

template <typename T>
std::vector<std::vector<T>> GetExpectedOutputResults(
    absl::Span<const T> input_data, absl::Span<const int64_t> input_offsets,
    absl::Span<const int64_t> send_sizes,
    absl::Span<const int64_t> output_offsets, int64_t num_ranks,
    int64_t num_updates_per_rank, int64_t num_input_rows,
    int64_t num_row_elements) {
  std::vector<std::vector<T>> expected_output(
      num_ranks, std::vector<T>(num_input_rows * num_row_elements, 0));

  for (int64_t i = 0; i < num_ranks; ++i) {
    for (int64_t j = 0; j < num_updates_per_rank; ++j) {
      int64_t update_idx = i * num_updates_per_rank + j;
      int64_t input_offset = input_offsets[update_idx];
      int64_t send_size = send_sizes[update_idx];
      int64_t output_offset = output_offsets[update_idx];

      for (int k = 0; k < send_size * num_row_elements; ++k) {
        expected_output[i][output_offset * num_row_elements + k] =
            input_data[input_offset * num_row_elements + k];
      }
    }
  }
  return expected_output;
}

template <typename T>
stream_executor::DeviceAddressHandle CreateDeviceBuffer(
    se::StreamExecutor* executor, const std::vector<T>& data) {
  stream_executor::DeviceAddressHandle device_buffer(
      executor, executor->AllocateArray<int64_t>(data.size()));
  CHECK(!device_buffer.address().is_null());
  CHECK_OK(executor->SynchronousMemcpy(device_buffer.address_ptr(), data.data(),
                                       data.size() * sizeof(T)));
  return device_buffer;
}

template <typename T>
std::vector<std::vector<T>> CopyDeviceToHost2D(
    se::StreamExecutor* executor,
    const std::vector<stream_executor::DeviceAddressHandle>& device_buffers,
    int64_t num_elements) {
  std::vector<std::vector<T>> host_buffers;
  host_buffers.reserve(device_buffers.size());
  for (auto& device_buffer : device_buffers) {
    std::vector<T> host_buffer(num_elements);
    CHECK_OK(executor->SynchronousMemcpy(
        host_buffer.data(), device_buffer.address(), num_elements * sizeof(T)));
    host_buffers.push_back(std::move(host_buffer));
  }
  return host_buffers;
}

template <typename T>
std::vector<std::vector<T>> CopyDeviceToHost2D(
    const std::vector<std::unique_ptr<se::Stream>>& streams,
    const std::vector<std::unique_ptr<se::MemoryAllocation>>& device_buffers,
    size_t offset_bytes, int64_t num_elements) {
  const size_t copy_size = num_elements * sizeof(T);
  std::vector<std::vector<T>> host_buffers;
  host_buffers.reserve(device_buffers.size());
  CHECK_EQ(streams.size(), device_buffers.size());
  for (int i = 0; i < device_buffers.size(); ++i) {
    se::DeviceAddressBase full_buffer = device_buffers[i]->address();
    se::DeviceAddressBase sliced_address =
        full_buffer.GetByteSlice(offset_bytes, copy_size);
    std::vector<T> host_buffer(num_elements);
    CHECK_OK(streams[i]->Memcpy(host_buffer.data(), sliced_address, copy_size));
    CHECK_OK(streams[i]->BlockHostUntilDone());
    host_buffers.push_back(std::move(host_buffer));
  }
  return host_buffers;
}

absl::StatusOr<std::vector<std::unique_ptr<xla::SymmetricMemory>>>
CreateSymmetricMemory(
    std::shared_ptr<tsl::Executor> exec, const std::vector<ncclComm_t>& comms,
    const std::vector<std::unique_ptr<se::MemoryAllocation>>& buffers,
    const std::vector<se::StreamExecutor*>& executors) {
  int64_t num_devices = comms.size();
  std::vector<tsl::Future<std::unique_ptr<NcclSymmetricMemory>>>
      symmetric_memory_futures(num_devices);
  for (int i = 0; i < num_devices; ++i) {
    symmetric_memory_futures[i] = tsl::MakeFutureOn(*exec, [&, exec, i]() {
      std::shared_ptr<NcclCommState> comm_state =
          std::make_shared<NcclCommState>(comms[i]);
      return NcclSymmetricMemory::Create(comm_state, buffers[i]->address(),
                                         exec, executors[i]);
    });
  }

  std::vector<std::unique_ptr<xla::SymmetricMemory>> symmetric_memory;
  for (int i = 0; i < num_devices; ++i) {
    ABSL_ASSIGN_OR_RETURN(auto mem,
                          std::move(symmetric_memory_futures[i]).Await());
    symmetric_memory.push_back(std::move(mem));
  }
  return symmetric_memory;
}

struct KernelTestCase {
  std::string name;
  int64_t num_outputs;
  int64_t num_updates_per_output;
  int64_t num_input_rows;
  int64_t num_row_elements;
  // Updates are laid out in row-major order: all updates to the first output,
  // then all updates to the second output and so on.
  std::vector<int64_t> input_offsets;
  std::vector<int64_t> send_sizes;
  std::vector<int64_t> output_offsets;
  // Block grid the kernel is expected to be launched with:
  // (outputs, block clusters, blocks per group).
  se::BlockDim block_dims;
};

using RaggedAllToAllKernelTest = ::testing::TestWithParam<KernelTestCase>;

// Owns NCCL communicators created by `ncclCommInitAll`. Symmetric memory
// windows registered on the communicators must be destroyed first.
struct ScopedNcclComms {
  explicit ScopedNcclComms(size_t size) : comms(size, nullptr) {}
  ~ScopedNcclComms() {
    for (ncclComm_t comm : comms) {
      if (comm != nullptr) {
        ncclCommDestroy(comm);
      }
    }
  }

  ScopedNcclComms(const ScopedNcclComms&) = delete;
  ScopedNcclComms& operator=(const ScopedNcclComms&) = delete;

  std::vector<ncclComm_t> comms;
};

// The tests copy `float`, so an even number of row elements is copied as
// 8-byte vectors and an odd number as 4-byte vectors.
INSTANTIATE_TEST_SUITE_P(
    LaunchShapes, RaggedAllToAllKernelTest,
    ::testing::Values(
        // Small input: one block per update.
        KernelTestCase{"TwoUpdatesSmallInput",
                       /*num_outputs=*/2,
                       /*num_updates_per_output=*/2,
                       /*num_input_rows=*/8,
                       /*num_row_elements=*/2,
                       /*input_offsets=*/{1, 4, 0, 3},
                       /*send_sizes=*/{2, 3, 1, 2},
                       /*output_offsets=*/{0, 4, 1, 5},
                       /*block_dims=*/se::BlockDim(2, 2, 1)},
        // One update striped over nine blocks of 4-byte vectors, with a
        // zero-length send and a partial final stripe.
        KernelTestCase{"OneUpdatePartialStripe",
                       /*num_outputs=*/2,
                       /*num_updates_per_output=*/1,
                       /*num_input_rows=*/8,
                       /*num_row_elements=*/1025,
                       /*input_offsets=*/{4, 1},
                       /*send_sizes=*/{0, 5},
                       /*output_offsets=*/{2, 3},
                       /*block_dims=*/se::BlockDim(2, 1, 9)},
        // 8-byte vectors and an input large enough to hit the cap on the
        // number of blocks.
        KernelTestCase{"OneUpdateMaxBlocks",
                       /*num_outputs=*/2,
                       /*num_updates_per_output=*/1,
                       /*num_input_rows=*/8,
                       /*num_row_elements=*/16514,
                       /*input_offsets=*/{4, 1},
                       /*send_sizes=*/{0, 5},
                       /*output_offsets=*/{2, 3},
                       /*block_dims=*/se::BlockDim(2, 1, 64)},
        // The cap is not a multiple of the number of outputs.
        KernelTestCase{"OneUpdateMaxBlocksThreeOutputs",
                       /*num_outputs=*/3,
                       /*num_updates_per_output=*/1,
                       /*num_input_rows=*/8,
                       /*num_row_elements=*/16514,
                       /*input_offsets=*/{0, 2, 5},
                       /*send_sizes=*/{2, 3, 3},
                       /*output_offsets=*/{6, 0, 4},
                       /*block_dims=*/se::BlockDim(3, 1, 43)},
        // Two updates per group, with more blocks than updates in a group.
        KernelTestCase{"SixUpdatesMoreBlocksThanUpdates",
                       /*num_outputs=*/2,
                       /*num_updates_per_output=*/6,
                       /*num_input_rows=*/64,
                       /*num_row_elements=*/194,
                       /*input_offsets=*/
                       {0, 5, 10, 15, 20, 25, 7, 12, 17, 22, 27, 32},
                       /*send_sizes=*/{3, 0, 10, 7, 1, 9, 10, 2, 0, 5, 8, 4},
                       /*output_offsets=*/
                       {0, 10, 20, 30, 40, 50, 0, 10, 20, 30, 40, 50},
                       /*block_dims=*/se::BlockDim(2, 3, 3)}),
    [](const ::testing::TestParamInfo<KernelTestCase>& info) {
      return info.param.name;
    });

TEST_P(RaggedAllToAllKernelTest, BlockDim) {
  const KernelTestCase& test_case = GetParam();
  ASSERT_OK_AND_ASSIGN(
      se::BlockDim block_dims,
      GetRaggedAllToAllBlockDim(
          primitive_util::NativeToPrimitiveType<float>(), test_case.num_outputs,
          test_case.num_updates_per_output, test_case.num_input_rows,
          test_case.num_row_elements));
  EXPECT_EQ(block_dims.x, test_case.block_dims.x);
  EXPECT_EQ(block_dims.y, test_case.block_dims.y);
  EXPECT_EQ(block_dims.z, test_case.block_dims.z);
}

TEST_P(RaggedAllToAllKernelTest, KernelWithArrayOfOutputPointers) {
  using T = float;

  auto* executor = GetGpuExecutor();
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  const KernelTestCase& test_case = GetParam();
  const int64_t num_outputs = test_case.num_outputs;
  const int64_t num_update_per_output = test_case.num_updates_per_output;
  const int64_t num_input_rows = test_case.num_input_rows;
  const int64_t num_row_elements = test_case.num_row_elements;
  const int64_t n = num_input_rows * num_row_elements;

  stream_executor::DeviceAddressHandle input_buffer(
      executor, executor->AllocateArray<T>(n));

  std::vector<stream_executor::DeviceAddressHandle> output_buffers;
  for (int64_t i = 0; i < num_outputs; ++i) {
    output_buffers.emplace_back(executor, executor->AllocateArray<T>(n));
    ASSERT_TRUE(!output_buffers[i].address().is_null());
    ASSERT_OK(stream->MemZero(output_buffers[i].address_ptr(), n * sizeof(T)));
  }

  std::vector<T> input_data(n);
  absl::c_iota(input_data, 0);
  ASSERT_OK(stream->Memcpy(input_buffer.address_ptr(), input_data.data(),
                           n * sizeof(T)));

  const std::vector<int64_t>& input_offsets = test_case.input_offsets;
  const std::vector<int64_t>& send_sizes = test_case.send_sizes;
  const std::vector<int64_t>& output_offsets = test_case.output_offsets;

  stream_executor::DeviceAddressHandle input_offsets_buffer =
      CreateDeviceBuffer(executor, input_offsets);
  stream_executor::DeviceAddressHandle send_sizes_buffer =
      CreateDeviceBuffer(executor, send_sizes);
  stream_executor::DeviceAddressHandle output_offsets_buffer =
      CreateDeviceBuffer(executor, output_offsets);

  stream_executor::gpu::RaggedAllToAllOutputPtrs output_buffers_array;
  for (int64_t i = 0; i < num_outputs; ++i) {
    output_buffers_array[i] = output_buffers[i].address().opaque();
  }

  ASSERT_OK(RunRaggedAllToAllKernel(
      stream.get(), primitive_util::NativeToPrimitiveType<T>(),
      input_buffer.address(), output_buffers_array,
      input_offsets_buffer.address(), send_sizes_buffer.address(),
      output_offsets_buffer.address(), num_outputs, num_update_per_output,
      num_input_rows, num_row_elements));
  // The readback below does not go through `stream`.
  ASSERT_OK(stream->BlockHostUntilDone());

  std::vector<std::vector<T>> output_results =
      CopyDeviceToHost2D<T>(executor, output_buffers, n);

  std::vector<std::vector<T>> expected_output_results =
      GetExpectedOutputResults<T>(
          input_data, input_offsets, send_sizes, output_offsets, num_outputs,
          num_update_per_output, num_input_rows, num_row_elements);

  ASSERT_EQ(output_results.size(), expected_output_results.size());
  EXPECT_EQ(output_results, expected_output_results);
}

TEST_P(RaggedAllToAllKernelTest, KernelWithSymmetricMemory) {
  using T = float;

  const KernelTestCase& test_case = GetParam();
  const int64_t num_outputs = test_case.num_outputs;
  const int64_t num_update_per_output = test_case.num_updates_per_output;
  const int64_t num_input_rows = test_case.num_input_rows;
  const int64_t num_row_elements = test_case.num_row_elements;
  constexpr size_t output_sym_offset = 1024;  // 1KB padding for output buffers
  const int64_t n = num_input_rows * num_row_elements;

  ASSERT_OK_AND_ASSIGN(
      se::Platform * platform,
      se::PlatformManager::PlatformWithId(se::cuda::kCudaPlatformId));
  int visible_device_count =
      std::min<int64_t>(platform->VisibleDeviceCount(), num_outputs);

  if (visible_device_count < num_outputs) {
    GTEST_SKIP() << "Skipping test because there are not enough visible "
                    "devices.";
  }

  std::vector<se::StreamExecutor*> executors(visible_device_count);
  std::vector<std::unique_ptr<se::Stream>> streams(visible_device_count);

  for (int i = 0; i < visible_device_count; ++i) {
    ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                         platform->ExecutorForDevice(i));
    executors[i] = executor;

    ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
    streams[i] = std::move(stream);
  }

  if (!executors[0]
           ->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastHopper()) {
    GTEST_SKIP() << "Test requires at least Hopper architecture";
  }

  std::vector<std::unique_ptr<se::MemoryAllocator>> collective_allocators;
  for (int i = 0; i < visible_device_count; ++i) {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<se::MemoryAllocator> allocator,
        executors[i]->CreateMemoryAllocator(se::MemorySpace::kCollective));
    collective_allocators.push_back(std::move(allocator));
  }

  stream_executor::DeviceAddressHandle input_buffer(
      executors[0], executors[0]->AllocateArray<T>(n));

  std::vector<std::unique_ptr<se::MemoryAllocation>> output_buffers;
  for (int64_t i = 0; i < num_outputs; ++i) {
    size_t total_bytes = output_sym_offset + (n * sizeof(T));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::MemoryAllocation> output_buffer,
                         collective_allocators[i]->Allocate(total_bytes));
    se::DeviceAddressBase output_buffer_address = output_buffer->address();
    ASSERT_TRUE(!output_buffer_address.is_null());

    ASSERT_OK(streams[i]->MemZero(&output_buffer_address, total_bytes));

    output_buffers.push_back(std::move(output_buffer));
  }

  std::vector<T> input_data(n);
  absl::c_iota(input_data, 0);
  ASSERT_OK(streams[0]->Memcpy(input_buffer.address_ptr(), input_data.data(),
                               n * sizeof(T)));

  const std::vector<int64_t>& input_offsets = test_case.input_offsets;
  const std::vector<int64_t>& send_sizes = test_case.send_sizes;
  const std::vector<int64_t>& output_offsets = test_case.output_offsets;

  stream_executor::DeviceAddressHandle input_offsets_buffer =
      CreateDeviceBuffer(executors[0], input_offsets);
  stream_executor::DeviceAddressHandle send_sizes_buffer =
      CreateDeviceBuffer(executors[0], send_sizes);
  stream_executor::DeviceAddressHandle output_offsets_buffer =
      CreateDeviceBuffer(executors[0], output_offsets);

  // Declared before the symmetric memory, so that the windows are destroyed
  // before their communicators.
  ScopedNcclComms scoped_comms(visible_device_count);
  const std::vector<ncclComm_t>& comms = scoped_comms.comms;
  ncclResult_t result = ncclCommInitAll(
      scoped_comms.comms.data(), visible_device_count, /*devlist=*/nullptr);
  ASSERT_EQ(result, ncclSuccess);

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "nccl",
                               visible_device_count);
  auto exec =
      std::shared_ptr<tsl::Executor>(pool.AsExecutor(), [](tsl::Executor*) {});

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<xla::SymmetricMemory>>
          output_buffers_symmetric_memory,
      CreateSymmetricMemory(exec, comms, output_buffers, executors));

  ASSERT_OK_AND_ASSIGN(
      RaggedAllToAllWithSymmetricMemoryKernelType kernel,
      LoadRaggedAllToAllWithSymmetricMemoryKernel(
          executors[0], primitive_util::NativeToPrimitiveType<T>(),
          num_row_elements));
  ASSERT_OK(RunRaggedAllToAllWithSymmetricMemoryKernel(
      streams[0].get(), kernel, primitive_util::NativeToPrimitiveType<T>(),
      input_buffer.address(), output_buffers_symmetric_memory[0].get(),
      output_sym_offset, input_offsets_buffer.address(),
      send_sizes_buffer.address(), output_offsets_buffer.address(), num_outputs,
      num_update_per_output, num_input_rows, num_row_elements));
  // The kernel writes to every device from the stream of the first one, and the
  // readback of the other devices uses their own streams.
  ASSERT_OK(streams[0]->BlockHostUntilDone());

  std::vector<std::vector<T>> output_results =
      CopyDeviceToHost2D<T>(streams, output_buffers, output_sym_offset, n);

  std::vector<std::vector<T>> expected_output_results =
      GetExpectedOutputResults<T>(
          input_data, input_offsets, send_sizes, output_offsets, num_outputs,
          num_update_per_output, num_input_rows, num_row_elements);

  ASSERT_EQ(output_results.size(), expected_output_results.size());
  EXPECT_EQ(output_results, expected_output_results);
}

}  // namespace
}  // namespace xla::gpu
