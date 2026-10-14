/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/command_buffer_thunk.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tsl/profiler/lib/profiler_lock.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/memset_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/gpu/gpu_test_kernels_fatbin.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
using ::testing::Each;
using ::testing::HasSubstr;
using ::testing::Pair;

using MemoryAccess = BufferUse::MemoryAccess;
using KernelArgsPacking = se::KernelLoaderSpec::KernelArgsPacking;

namespace {

se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

struct OwningExecutableSource {
  std::string text;
  std::vector<uint8_t> binary;

  explicit operator Thunk::ExecutableSource() const { return {text, binary}; }
};

absl::StatusOr<OwningExecutableSource> ExecutableSource() {
  ABSL_ASSIGN_OR_RETURN(
      std::vector<uint8_t> fatbin,
      se::gpu::GetGpuTestKernelsFatbin(GpuExecutor()->GetPlatform()->Name()));
  return OwningExecutableSource{/*text=*/{},
                                /*binary=*/fatbin};
}

KernelArgsPacking CreateDefaultArgsPacking() {
  using Packed = absl::StatusOr<std::unique_ptr<se::KernelArgsPackedArrayBase>>;

  return [=](const se::Kernel& kernel, const se::KernelArgs& args) -> Packed {
    auto* mem_args = se::Cast<se::KernelArgsDeviceAddressArray>(&args);

    return se::PackKernelArgs(mem_args->device_addr_args(),
                              args.number_of_shared_bytes());
  };
}

// Some of the tests rely on CUDA 12.3+ features.
bool IsAtLeastCuda12300(const se::StreamExecutor* stream_executor) {
  const auto& device_description = stream_executor->GetDeviceDescription();
  const auto* cuda_cc =
      device_description.gpu_compute_capability().cuda_compute_capability();
  if (cuda_cc != nullptr) {
    // We need a recent driver to support the feature at runtime and we need a
    // recent version of the toolkit at compile time, so that we have access to
    // the driver's headers.
    if (std::min(device_description.driver_version(),
                 device_description.compile_time_toolkit_version()) >=
        stream_executor::SemanticVersion(12, 3, 0)) {
      return true;
    }
  }

  return false;
}

// Give a short alias to synchronization mode.
static constexpr auto serialize =
    CommandExecutor::SynchronizationMode::kSerialize;

class RequiresUpdateOnExecuteMemsetThunk : public Memset32BitValueThunk {
 public:
  RequiresUpdateOnExecuteMemsetThunk(Thunk::ThunkInfo thunk_info,
                                     uint32_t value,
                                     const BufferAllocation::Slice& destination,
                                     int* record_count)
      : Memset32BitValueThunk(std::move(thunk_info), value, destination),
        record_count_(record_count) {}

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override {
    ++*record_count_;
    return Memset32BitValueThunk::Record(execute_params, record_params,
                                         std::move(record_action),
                                         command_buffer);
  }

  bool requires_update_on_execute() const override { return true; }

 private:
  int* record_count_;
};

class CountingDeviceToDeviceCopyThunk : public DeviceToDeviceCopyThunk {
 public:
  CountingDeviceToDeviceCopyThunk(Thunk::ThunkInfo thunk_info,
                                  const ShapedSlice& source_buffer,
                                  const ShapedSlice& destination_buffer,
                                  int64_t mem_size, int* record_count)
      : DeviceToDeviceCopyThunk(std::move(thunk_info), source_buffer,
                                destination_buffer, mem_size),
        record_count_(record_count) {}

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override {
    ++*record_count_;
    return DeviceToDeviceCopyThunk::Record(execute_params, record_params,
                                           std::move(record_action),
                                           command_buffer);
  }

 private:
  int* record_count_;
};

// Creates execute params with a valid persistent allocation indices span
// (empty by default). Command buffer lowering is enabled only when the span
// is valid; production code always passes a valid - possibly empty - span.
Thunk::ExecuteParams CreateExecuteParams(
    const ServiceExecutableRunOptions& run_options,
    const BufferAllocations& allocations, se::Stream* stream,
    absl::Span<const BufferAllocation::Index> persistent_alloc_indices = {}) {
  return Thunk::ExecuteParams::Create(
      run_options, allocations, stream, stream, nullptr, nullptr, nullptr,
      /*additional_compute_streams=*/{}, /*execution_scoped_state=*/nullptr,
      persistent_alloc_indices);
}

}  // namespace

TEST(CommandBufferThunkTest, DeviceToDeviceCopy) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  Shape shape = ShapeUtil::MakeShape(S32, {length});
  // Prepare arguments: a=42, b=0
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> b =
      stream_executor->AllocateArray<int32_t>(length, 0);

  ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  commands.Emplace<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo(), ShapedSlice{slice_a, shape},
      ShapedSlice{slice_b, shape}, byte_length);
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo(),
                           /*devices_in_process=*/1);

  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({a, b}, 0, &allocator);

  Thunk::ExecuteParams params =
      CreateExecuteParams(run_options, allocations, stream.get());

  // Execute command buffer thunk and verify that it copied the memory.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42));

  // Try to update the command buffer with the same buffers.
  ASSERT_OK(stream->MemZero(&b, byte_length));

  // Thunk execution should automatically update underlying command buffer.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42));
}

TEST(CommandBufferThunkTest, UpdatePolicyIgnoresVaRemappedAllocations) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  Shape shape = ShapeUtil::MakeShape(S32, {length});

  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> c =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> b =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> d =
      stream_executor->AllocateArray<int32_t>(length, 0);

  ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  ASSERT_OK(stream->Memset32(&c, 7, byte_length));
  ASSERT_OK(stream->MemZero(&b, byte_length));
  ASSERT_OK(stream->MemZero(&d, byte_length));

  BufferAllocation alloc_src(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_src(&alloc_src, 0, byte_length);
  BufferAllocation::Slice slice_dst(&alloc_dst, 0, byte_length);

  CommandSequence commands;
  commands.Emplace<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo(), ShapedSlice{slice_src, shape},
      ShapedSlice{slice_dst, shape}, byte_length);
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(std::move(commands), serialize));

  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo(),
                           /*devices_in_process=*/1);

  std::vector<BufferAllocation::Index> persistent_alloc_indices = {0};

  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({a, b}, 0, &allocator);
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr,
      nullptr, /*additional_compute_streams=*/{},
      /*execution_scoped_state=*/nullptr,
      absl::MakeConstSpan(persistent_alloc_indices));

  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  std::vector<int32_t> dst(4, 0);
  ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 42));

  ASSERT_OK(stream->MemZero(&b, byte_length));
  BufferAllocations source_changed_allocations({c, b}, 0, &allocator);
  params = Thunk::ExecuteParams::Create(
      run_options, source_changed_allocations, stream.get(), stream.get(),
      nullptr, nullptr, nullptr, /*additional_compute_streams=*/{},
      /*execution_scoped_state=*/nullptr,
      absl::MakeConstSpan(persistent_alloc_indices));

  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  std::fill(dst.begin(), dst.end(), 0);
  ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 42));

  BufferAllocations dynamic_changed_allocations({c, d}, 0, &allocator);
  params = Thunk::ExecuteParams::Create(
      run_options, dynamic_changed_allocations, stream.get(), stream.get(),
      nullptr, nullptr, nullptr, /*additional_compute_streams=*/{},
      /*execution_scoped_state=*/nullptr,
      absl::MakeConstSpan(persistent_alloc_indices));

  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  std::fill(dst.begin(), dst.end(), 0);
  ASSERT_OK(stream->Memcpy(dst.data(), d, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 7));
}

TEST(CommandBufferThunkTest, AbsentPersistentAllocIndicesFallsBackToThunks) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  constexpr int64_t kLength = 4;
  constexpr int64_t kByteLength = sizeof(int32_t) * kLength;
  Shape shape = ShapeUtil::MakeShape(S32, {kLength});

  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(kLength, 0);
  se::DeviceAddress<int32_t> b =
      stream_executor->AllocateArray<int32_t>(kLength, 0);
  ASSERT_OK(stream->Memset32(&a, 42, kByteLength));
  ASSERT_OK(stream->MemZero(&b, kByteLength));

  BufferAllocation source(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation destination(/*index=*/1, kByteLength, /*color=*/0);
  BufferAllocation::Slice source_slice(&source, 0, kByteLength);
  BufferAllocation::Slice destination_slice(&destination, 0, kByteLength);

  // Command sequence that counts how many times commands are recorded into a
  // command buffer.
  int record_count = 0;
  CommandSequence commands;
  commands.Emplace<CountingDeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo(), ShapedSlice{source_slice, shape},
      ShapedSlice{destination_slice, shape}, kByteLength, &record_count);
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(std::move(commands), serialize));

  // Fallback thunk sequence that executes the same copy in op-by-op mode.
  ThunkSequence fallback_thunks;
  fallback_thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo(), ShapedSlice{source_slice, shape},
      ShapedSlice{destination_slice, shape}, kByteLength));

  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo(),
                           /*devices_in_process=*/1,
                           std::make_unique<SequentialThunk>(
                               Thunk::ThunkInfo(), std::move(fallback_thunks)));

  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({a, b}, /*device_ordinal=*/0, &allocator);

  Thunk::InitializeParams initialize_params;
  initialize_params.executor = stream_executor;
  initialize_params.buffer_allocations = &allocations;
  initialize_params.stream = stream.get();

  // With absent persistent allocation indices, initialization skips command
  // buffer creation and execution falls back to the thunk sequence.
  ASSERT_OK(thunk.Initialize(initialize_params));
  EXPECT_EQ(record_count, 0);

  Thunk::ExecuteParams absent_params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  std::vector<int32_t> result(kLength, 0);
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK(stream->MemZero(&b, kByteLength));
    ASSERT_OK(thunk.ExecuteOnStream(absent_params));
    ASSERT_OK(stream->BlockHostUntilDone());

    std::fill(result.begin(), result.end(), 0);
    ASSERT_OK(stream->Memcpy(result.data(), b, kByteLength));
    EXPECT_EQ(result, std::vector<int32_t>(kLength, 42));
    EXPECT_EQ(record_count, 0);
  }

  // Once persistent allocation indices become available, the thunk records
  // and executes the command buffer.
  std::vector<BufferAllocation::Index> persistent_alloc_indices = {0, 1};
  initialize_params.persistent_alloc_indices =
      absl::MakeConstSpan(persistent_alloc_indices);
  ASSERT_OK(thunk.Initialize(initialize_params));
  EXPECT_EQ(record_count, 1);

  Thunk::ExecuteParams present_params =
      CreateExecuteParams(run_options, allocations, stream.get(),
                          absl::MakeConstSpan(persistent_alloc_indices));

  ASSERT_OK(stream->MemZero(&b, kByteLength));
  ASSERT_OK(thunk.ExecuteOnStream(present_params));
  ASSERT_OK(stream->BlockHostUntilDone());

  std::fill(result.begin(), result.end(), 0);
  ASSERT_OK(stream->Memcpy(result.data(), b, kByteLength));
  EXPECT_EQ(result, std::vector<int32_t>(kLength, 42));
  EXPECT_EQ(record_count, 1);

  // Executing again with unchanged buffers must not trigger an update.
  ASSERT_OK(stream->MemZero(&b, kByteLength));
  ASSERT_OK(thunk.ExecuteOnStream(present_params));
  ASSERT_OK(stream->BlockHostUntilDone());

  std::fill(result.begin(), result.end(), 0);
  ASSERT_OK(stream->Memcpy(result.data(), b, kByteLength));
  EXPECT_EQ(result, std::vector<int32_t>(kLength, 42));
  EXPECT_EQ(record_count, 1);
}

TEST(CommandBufferThunkTest,
     AbsentPersistentAllocIndicesWithoutFallbackThunksIsError) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  constexpr int64_t kByteLength = sizeof(uint32_t);
  se::DeviceAddress<uint32_t> destination =
      stream_executor->AllocateArray<uint32_t>(1, 0);
  BufferAllocation allocation(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation::Slice slice(&allocation, /*offset=*/0, kByteLength);

  CommandSequence commands;
  commands.Emplace<Memset32BitValueThunk>(Thunk::ThunkInfo(), /*value=*/42,
                                          slice);
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(std::move(commands), serialize));
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo(),
                           /*devices_in_process=*/1);

  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({destination}, /*device_ordinal=*/0,
                                &allocator);
  Thunk::ExecuteParams absent_params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  EXPECT_THAT(thunk.ExecuteOnStream(absent_params),
              ::absl_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST(CommandBufferThunkTest,
     RequiresUpdateOnExecuteWithAllAllocationsPersistent) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  constexpr int64_t kByteLength = sizeof(uint32_t);
  se::DeviceAddress<uint32_t> destination =
      stream_executor->AllocateArray<uint32_t>(1, 0);
  BufferAllocation allocation(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation::Slice slice(&allocation, /*offset=*/0, kByteLength);

  int record_count = 0;
  CommandSequence commands;
  commands.Emplace<RequiresUpdateOnExecuteMemsetThunk>(
      Thunk::ThunkInfo(), /*value=*/42, slice, &record_count);
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(std::move(commands), serialize));
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo(),
                           /*devices_in_process=*/1);

  std::vector<BufferAllocation::Index> persistent_alloc_indices = {0};
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({destination}, /*device_ordinal=*/0,
                                &allocator);
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr,
      nullptr, /*additional_compute_streams=*/{},
      /*execution_scoped_state=*/nullptr,
      absl::MakeConstSpan(persistent_alloc_indices));

  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(record_count, 1);

  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(record_count, 2);
}

TEST(CommandBufferThunkTest, MemzeroThunk) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  Shape shape = ShapeUtil::MakeShape(S32, {length});

  // Prepare arguments: a=42
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  ASSERT_OK(stream->Memset32(&a, 42, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  commands.Emplace<MemzeroThunk>(Thunk::ThunkInfo(),
                                 ShapedSlice{slice_a, shape});
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo(),
                           /*devices_in_process=*/1);

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params =
      CreateExecuteParams(run_options, allocations, stream.get());

  // Execute command buffer thunk and verify that it zeroes the memory.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 0));
}

TEST(CommandBufferThunkTest, Memset32Cmd) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);

  ASSERT_OK(stream->Memset32(&a, 42, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);

  // Prepare commands sequence for constructing command buffer.
  Memset32BitValueThunk memset_thunk(Thunk::ThunkInfo(), /*value=*/84, slice_a);
  CommandSequence commands;
  commands.Append(&memset_thunk);
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo(),
                           /*devices_in_process=*/1);

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params =
      CreateExecuteParams(run_options, allocations, stream.get());

  // Execute command buffer thunk and verify that it set the memory.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 84));
}

TEST(CommandBufferThunkTest, Memset32CmdCommandBuffersDisabledDuringProfiling) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);

  ASSERT_OK(stream->Memset32(&a, 42, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);

  // Memset32BitValueThunk implements Command directly, so the same thunk is
  // used for both the fallback ThunkSequence and the CommandSequence.
  auto memset_thunk =
      std::make_unique<Memset32BitValueThunk>(Thunk::ThunkInfo(), 84, slice_a);
  Memset32BitValueThunk* memset_thunk_ptr = memset_thunk.get();
  ThunkSequence thunks;
  thunks.push_back(std::move(memset_thunk));
  auto seq_thunks =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));
  CommandSequence commands;
  commands.Append(memset_thunk_ptr);
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(std::move(commands), serialize));

  constexpr bool kProfileCommandBuffersEnabled = false;
  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo(),
                           /*devices_in_process=*/1, std::move(seq_thunks),
                           kProfileCommandBuffersEnabled);

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params =
      CreateExecuteParams(run_options, allocations, stream.get());

  ASSERT_OK_AND_ASSIGN(auto profiler_lock,
                       tsl::profiler::ProfilerLock::Acquire());
  // Execute command buffer thunk and verify that it set the memory.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 84));
}

TEST(CommandBufferThunkTest, Memset32CmdCommandBuffersEnabledDuringProfiling) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);

  ASSERT_OK(stream->Memset32(&a, 42, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);

  // Memset32BitValueThunk implements Command directly, so the same thunk is
  // used for both the fallback ThunkSequence and the CommandSequence.
  auto memset_thunk =
      std::make_unique<Memset32BitValueThunk>(Thunk::ThunkInfo(), 84, slice_a);
  Memset32BitValueThunk* memset_thunk_ptr = memset_thunk.get();
  ThunkSequence thunks;
  thunks.push_back(std::move(memset_thunk));
  auto seq_thunks =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));
  CommandSequence commands;
  commands.Append(memset_thunk_ptr);
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(std::move(commands), serialize));

  constexpr bool kProfileCommandBuffersEnabled = true;
  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo(),
                           /*devices_in_process=*/1, std::move(seq_thunks),
                           kProfileCommandBuffersEnabled);

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params =
      CreateExecuteParams(run_options, allocations, stream.get());

  ASSERT_OK_AND_ASSIGN(auto profiler_lock,
                       tsl::profiler::ProfilerLock::Acquire());

  // skip warm up iteration
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Execute command buffer thunk and verify that it set the memory.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 84));
}

TEST(CommandBufferThunkTest, Memset32CmdOnDifferentStreams) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  se::DeviceAddress<int32_t> a = stream_executor->AllocateArray<int32_t>(2, 0);
  ASSERT_OK(stream->MemZero(&a, 2 * sizeof(int32_t)));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc(/*index=*/0, a.size(), /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, 0 * sizeof(int32_t), sizeof(int32_t));
  BufferAllocation::Slice slice1(&alloc, 1 * sizeof(int32_t), sizeof(int32_t));

  // Prepare commands sequence for constructing command buffer.
  Memset32BitValueThunk memset_thunk0(Thunk::ThunkInfo(), /*value=*/12, slice0);
  Memset32BitValueThunk memset_thunk1(Thunk::ThunkInfo(), /*value=*/34, slice1);
  CommandSequence commands;
  commands.Append(&memset_thunk0);
  commands.Append(&memset_thunk1);
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo(),
                           /*devices_in_process=*/1);

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params =
      CreateExecuteParams(run_options, allocations, stream.get());

  // Execute command buffer thunk and verify that it set the memory.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(2, 0);
  ASSERT_OK(stream->Memcpy(dst.data(), a, a.size()));

  ASSERT_EQ(dst, std::vector<int32_t>({12, 34}));
}

TEST(CommandBufferThunkTest, LaunchCmd) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  Shape shape = ShapeUtil::MakeShape(S32, {length});

  // Prepare arguments: a=42, b=0
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> b =
      stream_executor->AllocateArray<int32_t>(length, 0);

  ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  std::vector<ShapedSlice> args = {
      {slice_a, shape}, {slice_a, shape}, {slice_b, shape}};  // b = a + a
  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  commands.Append(KernelThunk::MakeKernelThunk("AddI32", args, args_access,
                                               LaunchDimensions(1, 4),
                                               /*shmem_bytes=*/0,
                                               /*devices_per_host=*/1));
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo(),
                           /*devices_in_process=*/1);

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a, b}, 0, &allocator);

  Thunk::ExecuteParams params =
      CreateExecuteParams(run_options, allocations, stream.get());

  ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  ASSERT_OK(thunk.Initialize({stream_executor,
                              static_cast<Thunk::ExecutableSource>(source),
                              &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Prepare buffer allocation for updating command buffer: c=0
  se::DeviceAddress<int32_t> c =
      stream_executor->AllocateArray<int32_t>(length, 0);
  ASSERT_OK(stream->MemZero(&c, byte_length));

  // Update buffer allocation #1 to buffer `c`.
  allocations = BufferAllocations({a, c}, 0, &allocator);

  // Thunk execution should automatically update underlying command buffer.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `c` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Try to update the command buffer with the same buffers.
  ASSERT_OK(stream->MemZero(&c, byte_length));

  // Thunk execution should automatically update underlying command buffer.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `c` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));
}

TEST(CommandBufferThunkTest, CustomAddKernelLaunchCmd) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  auto packing = CreateDefaultArgsPacking();

  ASSERT_OK_AND_ASSIGN(stream_executor::KernelLoaderSpec spec,
                       stream_executor::gpu::GetAddI32TestKernelSpec(
                           stream_executor->GetPlatform()->id()));

  auto custom_kernel =
      CustomKernel("AddI32", std::move(spec), se::BlockDim(),
                   se::ThreadDim(4, 1, 1), /*shared_memory_bytes=*/0);

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  Shape shape = ShapeUtil::MakeShape(S32, {length});

  // Prepare arguments: a=42, b=0
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> b =
      stream_executor->AllocateArray<int32_t>(length, 0);

  ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  std::vector<ShapedSlice> args{
      {slice_a, shape}, {slice_a, shape}, {slice_b, shape}};  // b = a + a
  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  commands.Append(KernelThunk::MakeKernelThunk("AddI32", args, args_access,
                                               LaunchDimensions(1, 4),
                                               /*shmem_bytes=*/0,
                                               /*devices_per_host=*/1));
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo(),
                           /*devices_in_process=*/1);

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a, b}, 0, &allocator);

  Thunk::ExecuteParams params =
      CreateExecuteParams(run_options, allocations, stream.get());

  ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  ASSERT_OK(thunk.Initialize({stream_executor,
                              static_cast<Thunk::ExecutableSource>(source),
                              &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Prepare buffer allocation for updating command buffer: c=0
  se::DeviceAddress<int32_t> c =
      stream_executor->AllocateArray<int32_t>(length, 0);
  ASSERT_OK(stream->MemZero(&c, byte_length));

  // Update buffer allocation #1 to buffer `c`.
  allocations = BufferAllocations({a, c}, 0, &allocator);

  // Thunk execution should automatically update underlying command buffer.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `c` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Try to update the command buffer with the same buffers.
  ASSERT_OK(stream->MemZero(&c, byte_length));

  // Thunk execution should automatically update underlying command buffer.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `c` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));
}

TEST(CommandBufferThunkTest, GemmCmd) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  if (!IsAtLeastCuda12300(stream_executor)) {
    GTEST_SKIP() << "CUDA graph tracing is not supported";
  }

  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 4 * 3;
  int64_t out_length = sizeof(float) * 2 * 3;

  // Prepare arguments:
  // lhs = [1.0, 2.0, 3.0, 4.0
  //        5.0, 6.0, 7.0, 8.0]
  // rhs = [1.0, 1.0, 1.0
  //        1.0, 1.0, 1.0
  //        1.0, 1.0, 1.0
  //        1.0, 1.0, 1.0]
  se::DeviceAddress<float> lhs = stream_executor->AllocateArray<float>(2 * 4);
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  se::DeviceAddress<float> rhs = stream_executor->AllocateArray<float>(4 * 3);
  std::vector<float> rhs_arr(12, 1);
  ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceAddress<float> out = stream_executor->AllocateArray<float>(2 * 3);
  ASSERT_OK(stream->MemZero(&out, out_length));

  se::DeviceAddress<float> workspace =
      stream_executor->AllocateArray<float>(1024 * 1024);
  ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_lhs(/*index=*/0, lhs_length, /*color=*/0);
  BufferAllocation alloc_rhs(/*index=*/1, rhs_length, /*color=*/0);
  BufferAllocation alloc_out(/*index=*/2, out_length, /*color=*/0);
  BufferAllocation alloc_workspace(/*index=*/3, 1024 * 1024, /*color=*/0);

  BufferAllocation::Slice slice_lhs(&alloc_lhs, 0, lhs_length);
  BufferAllocation::Slice slice_rhs(&alloc_rhs, 0, rhs_length);
  BufferAllocation::Slice slice_out(&alloc_out, 0, out_length);
  BufferAllocation::Slice slice_workspace(&alloc_workspace, 0, 1024 * 1024);

  auto config = GemmConfig::For(
      ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), {}, {1},
      ShapeUtil::MakeShape(PrimitiveType::F32, {4, 3}), {}, {0},
      ShapeUtil::MakeShape(PrimitiveType::F32, {2, 3}), 1.0, 0.0, 0.0,
      PrecisionConfig::ALG_UNSET, std::nullopt,
      se::blas::kDefaultComputePrecision, false, false,
      /*scale_mode=*/se::gpu::ScaleMode::kNone,
      stream_executor->GetDeviceDescription().gpu_compute_capability());
  ASSERT_TRUE(config.ok());

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  Shape lhs_shape = ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4});
  Shape rhs_shape = ShapeUtil::MakeShape(PrimitiveType::F32, {4, 3});
  Shape output_shape = ShapeUtil::MakeShape(PrimitiveType::F32, {2, 3});
  commands.Append(std::make_unique<CublasLtMatmulThunk>(
      Thunk::ThunkInfo(), "canonical_hlo", config.value(),
      se::gpu::BlasLt::Epilogue::kDefault, 0, 0,
      ShapedSlice{slice_lhs, lhs_shape}, ShapedSlice{slice_rhs, rhs_shape},
      ShapedSlice{slice_out, output_shape},
      ShapedSlice{slice_out, output_shape}, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt));
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo(),
                           /*devices_in_process=*/1);

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({lhs, rhs, out, workspace}, 0, &allocator);

  Thunk::ExecuteParams params =
      CreateExecuteParams(run_options, allocations, stream.get());

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  ASSERT_OK(thunk.Initialize(
      {stream_executor, source, &allocations, stream.get(), stream.get()}));

  // Execute command buffer thunk and verify that it executed a GEMM.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `out` data back to host.
  std::vector<float> dst(6, 0);
  ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<float>({10, 10, 10, 26, 26, 26}));

  // Prepare buffer allocation for updating command buffer.
  se::DeviceAddress<float> updated_out =
      stream_executor->AllocateArray<float>(2 * 3);
  ASSERT_OK(stream->MemZero(&updated_out, out_length));

  // Update buffer allocation to updated `out` buffer.
  allocations =
      BufferAllocations({lhs, rhs, updated_out, workspace}, 0, &allocator);

  // Thunk execution should automatically update underlying command buffer.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `updated_out` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  ASSERT_OK(stream->Memcpy(dst.data(), updated_out, out_length));

  ASSERT_EQ(dst, std::vector<float>({10, 10, 10, 26, 26, 26}));

  // Try to update the command buffer with the same buffers.
  ASSERT_OK(stream->MemZero(&updated_out, out_length));

  // Thunk execution should automatically update underlying command buffer.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `updated_out` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  ASSERT_OK(stream->Memcpy(dst.data(), updated_out, out_length));

  ASSERT_EQ(dst, std::vector<float>({10, 10, 10, 26, 26, 26}));
}

TEST(CommandBufferThunkTest, CublasLtCmd) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  if (!IsAtLeastCuda12300(stream_executor)) {
    GTEST_SKIP() << "CUDA graph tracing is not supported";
  }

  ASSERT_OK_AND_ASSIGN(auto stream1, stream_executor->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto stream2, stream_executor->CreateStream());

  // CublasLt formula: D = alpha*(A*B) + beta*(C),

  Shape a_shape = ShapeUtil::MakeShape(F32, {2, 4});
  int64_t a_length = sizeof(float) * 2 * 4;
  Shape b_shape = ShapeUtil::MakeShape(F32, {4, 3});
  int64_t b_length = sizeof(float) * 4 * 3;
  Shape c_shape = ShapeUtil::MakeShape(F32, {2, 3});
  int64_t c_length = sizeof(float) * 2 * 3;
  Shape d_shape = ShapeUtil::MakeShape(F32, {2, 3});
  int64_t d_length = sizeof(float) * 2 * 3;

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, a_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, b_length, /*color=*/0);
  BufferAllocation alloc_c(/*index=*/2, c_length, /*color=*/0);
  BufferAllocation alloc_d(/*index=*/3, d_length, /*color=*/0);
  BufferAllocation alloc_workspace(/*index=*/4, 1024 * 1024, /*color=*/0);

  ShapedSlice slice_a{BufferAllocation::Slice{&alloc_a, 0, a_length}, a_shape};
  ShapedSlice slice_b{BufferAllocation::Slice(&alloc_b, 0, b_length), b_shape};
  ShapedSlice slice_c{BufferAllocation::Slice(&alloc_c, 0, c_length), c_shape};
  ShapedSlice slice_d{BufferAllocation::Slice(&alloc_d, 0, d_length), d_shape};
  ShapedSlice slice_workspace{
      BufferAllocation::Slice(&alloc_workspace, 0, 1024 * 1024),
      ShapeUtil::MakeShape(U8, {1024, 1024})};

  auto config = GemmConfig::For(
      /*lhs_shape*/ ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}),
      /*lhs_batch_dims*/ {}, /*lhs_contracting_dims*/ {1},
      /*rhs_shape*/ ShapeUtil::MakeShape(PrimitiveType::F32, {4, 3}),
      /*rhs_batch_dims*/ {}, /*rhs_contracting_dims*/ {0},
      /*c_shape*/ ShapeUtil::MakeShape(PrimitiveType::F32, {2, 3}),
      /*bias_shape_ptr*/ nullptr,
      /*output_shape*/ ShapeUtil::MakeShape(PrimitiveType::F32, {2, 3}),
      /*alpha_real*/ 1.0, /*alpha_imag*/ 0,
      /*beta*/ 1.0,
      /*precision_algorithm*/ PrecisionConfig::ALG_UNSET,
      /*algorithm*/ std::nullopt,
      /*compute_precision*/ se::blas::kDefaultComputePrecision,
      /*grad_x*/ false, /*grad_y*/ false,
      /*scale_mode=*/se::gpu::ScaleMode::kNone,
      stream_executor->GetDeviceDescription().gpu_compute_capability());
  ASSERT_TRUE(config.ok());

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  commands.Emplace<CublasLtMatmulThunk>(
      Thunk::ThunkInfo(), /*canonical_hlo=*/"", config.value(),
      se::gpu::BlasLt::Epilogue::kDefault, /*algorithm_idx=*/0,
      /*autotune_workspace_size=*/0, slice_a, slice_b, slice_c, slice_d,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, slice_workspace);
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo(),
                           /*devices_in_process=*/1);

  std::vector<float> a_arr_1{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> a_arr_2{2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> result_1{11, 11, 11, 27, 27, 27};
  std::vector<float> result_2{15, 15, 15, 31, 31, 31};

  auto run_cublaslt_test = [&](std::unique_ptr<se::Stream>& stream,
                               std::vector<float> a_arr,
                               std::vector<float> result) {
    se::DeviceAddress<float> a = stream_executor->AllocateArray<float>(2 * 4);
    ASSERT_OK(stream->Memcpy(&a, a_arr.data(), a_length));

    se::DeviceAddress<float> b = stream_executor->AllocateArray<float>(4 * 3);
    std::vector<float> b_arr(12, 1);
    ASSERT_OK(stream->Memcpy(&b, b_arr.data(), b_length));

    se::DeviceAddress<float> c = stream_executor->AllocateArray<float>(2 * 3);
    std::vector<float> c_arr(6, 1);
    ASSERT_OK(stream->Memcpy(&c, c_arr.data(), c_length));

    se::DeviceAddress<float> d = stream_executor->AllocateArray<float>(2 * 3);
    ASSERT_OK(stream->MemZero(&d, d_length));

    se::DeviceAddress<float> workspace =
        stream_executor->AllocateArray<float>(1024 * 1024);
    ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

    ServiceExecutableRunOptions run_options;
    stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
    BufferAllocations allocations({a, b, c, d, workspace}, 0, &allocator);

    Thunk::ExecuteParams params =
        CreateExecuteParams(run_options, allocations, stream.get());

    Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
    ASSERT_OK(thunk.Initialize(
        {stream_executor, source, &allocations, stream.get(), stream.get()}));

    // Execute command buffer thunk and verify that it executed a GEMM.
    ASSERT_OK(thunk.ExecuteOnStream(params));
    ASSERT_OK(stream->BlockHostUntilDone());

    // Copy `out` data back to host.
    std::vector<float> dst(6, 0);
    ASSERT_OK(stream->Memcpy(dst.data(), d, d_length));

    ASSERT_EQ(dst, result);

    // Prepare buffer allocation for updating command buffer.
    se::DeviceAddress<float> updated_d =
        stream_executor->AllocateArray<float>(2 * 3);
    ASSERT_OK(stream->MemZero(&updated_d, d_length));

    // Update buffer allocation to updated `d` buffer.
    allocations =
        BufferAllocations({a, b, c, updated_d, workspace}, 0, &allocator);

    // Thunk execution should automatically update underlying command
    // buffer.
    ASSERT_OK(thunk.ExecuteOnStream(params));
    ASSERT_OK(stream->BlockHostUntilDone());

    // Copy `updated_out` data back to host.
    std::fill(dst.begin(), dst.end(), 0);
    ASSERT_OK(stream->Memcpy(dst.data(), updated_d, d_length));

    ASSERT_EQ(dst, result);

    // Try to update the command buffer with the same buffers.
    ASSERT_OK(stream->MemZero(&updated_d, d_length));

    // Thunk execution should automatically update underlying command
    // buffer.
    ASSERT_OK(thunk.ExecuteOnStream(params));
    ASSERT_OK(stream->BlockHostUntilDone());

    // Copy `updated_out` data back to host.
    std::fill(dst.begin(), dst.end(), 0);
    ASSERT_OK(stream->Memcpy(dst.data(), updated_d, d_length));

    ASSERT_EQ(dst, result);
  };
  std::thread t1(run_cublaslt_test, std::ref(stream1), a_arr_1, result_1);
  std::thread t2(run_cublaslt_test, std::ref(stream2), a_arr_2, result_2);
  t1.join();
  t2.join();
}

TEST(CommandBufferThunkTest, MultipleLaunchCmd) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  Shape shape = ShapeUtil::MakeShape(S32, {length});

  // Prepare arguments: a=42, b=0
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> b =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> c =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> d =
      stream_executor->AllocateArray<int32_t>(length, 0);

  ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  ASSERT_OK(stream->MemZero(&b, byte_length));
  ASSERT_OK(stream->Memset32(&c, 21, byte_length));
  ASSERT_OK(stream->MemZero(&d, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation alloc_c(/*index=*/2, byte_length, /*color=*/0);
  BufferAllocation alloc_d(/*index=*/3, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);
  BufferAllocation::Slice slice_c(&alloc_c, 0, byte_length);
  BufferAllocation::Slice slice_d(&alloc_d, 0, byte_length);

  std::vector<ShapedSlice> args{
      {slice_a, shape}, {slice_a, shape}, {slice_b, shape}};  // b = a + a
  std::vector<ShapedSlice> args_1{
      {slice_c, shape}, {slice_c, shape}, {slice_d, shape}};  // d = c + c
  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  // Prepare commands sequence for constructing command buffer.
  CommandSequence commands;
  commands.Append(KernelThunk::MakeKernelThunk("AddI32", args, args_access,
                                               LaunchDimensions(1, 4),
                                               /*shmem_bytes=*/0,
                                               /*devices_per_host=*/1));
  commands.Append(KernelThunk::MakeKernelThunk("AddI32", args_1, args_access,
                                               LaunchDimensions(1, 4),
                                               /*shmem_bytes=*/0,
                                               /*devices_per_host=*/1));
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo(),
                           /*devices_in_process=*/1);

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a, b, c, d}, 0, &allocator);

  Thunk::ExecuteParams params =
      CreateExecuteParams(run_options, allocations, stream.get());

  ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  ASSERT_OK(thunk.Initialize({stream_executor,
                              static_cast<Thunk::ExecutableSource>(source),
                              &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Copy `d` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  ASSERT_OK(stream->Memcpy(dst.data(), d, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 21 + 21));

  BufferAllocation alloc_e(/*index=*/3, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_e(&alloc_e, 0, byte_length);

  // Prepare buffer allocation for updating command buffer: e=0
  se::DeviceAddress<int32_t> e =
      stream_executor->AllocateArray<int32_t>(length, 0);
  ASSERT_OK(stream->MemZero(&e, byte_length));

  // Update buffer allocation #1 to buffer `c`.
  allocations = BufferAllocations({a, b, c, e}, 0, &allocator);

  // Thunk execution should automatically update underlying command buffer.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Copy `e` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  ASSERT_OK(stream->Memcpy(dst.data(), e, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 21 + 21));

  // Try to update the command buffer with the same buffers.
  ASSERT_OK(stream->MemZero(&e, byte_length));

  // Thunk execution should automatically update underlying command buffer.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Copy `e` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  ASSERT_OK(stream->Memcpy(dst.data(), e, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 21 + 21));
}

TEST(CommandBufferThunkTest, ConditionalThunkCaseCommand) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  if (!IsAtLeastCuda12300(stream_executor)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  Shape shape = ShapeUtil::MakeShape(S32, {length});

  // Prepare arguments: index=0, a=42, b=0
  se::DeviceAddress<int32_t> index =
      stream_executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> b =
      stream_executor->AllocateArray<int32_t>(length, 0);

  ASSERT_OK(stream->Memset32(&index, 0, sizeof(int32_t)));
  ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_i(/*index=*/0, 1, /*color=*/0);
  BufferAllocation alloc_a(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/2, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_i(&alloc_i, 0, sizeof(int32_t));
  Shape i_shape = ShapeUtil::MakeShape(S32, {});

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  // Prepare thunk sequences for branches.
  std::vector<ThunkSequence> branch_thunks(2);

  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  {  // Case 0: b = a + a
    std::vector<ShapedSlice> args{
        {slice_a, shape}, {slice_a, shape}, {slice_b, shape}};
    branch_thunks[0].push_back(KernelThunk::MakeKernelThunk(
        "AddI32", args, args_access, LaunchDimensions(1, 4),
        /*shmem_bytes=*/0,
        /*devices_per_host=*/1));
  }

  {  // Case 1: b = b + b
    std::vector<ShapedSlice> args{
        {slice_b, shape}, {slice_b, shape}, {slice_b, shape}};
    branch_thunks[1].push_back(KernelThunk::MakeKernelThunk(
        "AddI32", args, args_access, LaunchDimensions(1, 4),
        /*shmem_bytes=*/0,
        /*devices_per_host=*/1));
  }

  // Prepare thunk sequence for command buffer conversion.
  ThunkSequence thunks = ThunkSequence::Of<ConditionalThunk>(
      Thunk::ThunkInfo(), ShapedSlice{slice_i, i_shape},
      std::move(branch_thunks), /*devices_per_host=*/1);

  ConvertToCommandsOptions options;
  options.synchronization_mode = serialize;
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       ConvertToCommands(thunks, options));

  // Construct a command buffer thunk with command sequence and fallback thunks.
  CommandBufferThunk thunk(
      std::move(executor), Thunk::ThunkInfo(), /*devices_in_process=*/1,
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks)));

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({index, a, b}, 0, &allocator);

  Thunk::ExecuteParams params =
      CreateExecuteParams(run_options, allocations, stream.get());

  ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  ASSERT_OK(thunk.Initialize({stream_executor,
                              static_cast<Thunk::ExecutableSource>(source),
                              &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Change `index` to `1` and check that it updated the `b` buffer.
  ASSERT_OK(stream->Memset32(&index, 1, sizeof(int32_t)));

  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 2 * (42 + 42)));
}

TEST(CommandBufferThunkTest, WhileThunk) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  if (!IsAtLeastCuda12300(stream_executor)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  Shape shape = ShapeUtil::MakeShape(S32, {length});

  // Prepare arguments: loop_cnt=0, num_iters=10, a=1, b=0
  se::DeviceAddress<bool> pred = stream_executor->AllocateArray<bool>(1, 0);
  se::DeviceAddress<int32_t> loop_cnt =
      stream_executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> num_iters =
      stream_executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> b =
      stream_executor->AllocateArray<int32_t>(length, 0);

  ASSERT_OK(stream->Memset32(&loop_cnt, 0, sizeof(int32_t)));
  ASSERT_OK(stream->Memset32(&num_iters, 10, sizeof(int32_t)));
  ASSERT_OK(stream->Memset32(&a, 1, byte_length));
  ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_pred(/*index=*/0, sizeof(bool), /*color=*/0);
  BufferAllocation alloc_loop_cnt(/*index=*/1, sizeof(int32_t), /*color=*/0);
  BufferAllocation alloc_num_iters(/*index=*/2, sizeof(int32_t), /*color=*/0);
  BufferAllocation alloc_a(/*index=*/3, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/4, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_pred(&alloc_pred, 0, sizeof(bool));
  BufferAllocation::Slice slice_loop_cnt(&alloc_loop_cnt, 0, sizeof(int32_t));
  BufferAllocation::Slice slice_num_iters(&alloc_num_iters, 0, sizeof(int32_t));
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  std::vector<ShapedSlice> cond_args{
      {slice_loop_cnt, shape}, {slice_pred, shape}, {slice_num_iters, shape}};
  auto cond_args_access = {MemoryAccess::kWrite, MemoryAccess::kWrite,
                           MemoryAccess::kRead};

  std::vector<ShapedSlice> body_args{
      {slice_a, shape}, {slice_b, shape}, {slice_b, shape}};  // b = a + b
  auto body_args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                           MemoryAccess::kWrite};

  // Prepare thunk sequence for loop `cond`.
  ThunkSequence cond_thunks;
  cond_thunks.push_back(KernelThunk::MakeKernelThunk(
      "IncAndCmp", cond_args, cond_args_access, LaunchDimensions(1, 1),
      /*shmem_bytes=*/0,
      /*devices_per_host=*/1));

  // Prepare thunk sequence for loop `body`.
  ThunkSequence body_thunks;
  body_thunks.push_back(KernelThunk::MakeKernelThunk(
      "AddI32", body_args, body_args_access, LaunchDimensions(1, 4),
      /*shmem_bytes=*/0,
      /*devices_per_host=*/1));

  // Prepare thunk sequence for command buffer conversion.
  ThunkSequence thunks = ThunkSequence::Of<WhileThunk>(
      Thunk::ThunkInfo(), slice_pred, std::move(cond_thunks),
      std::move(body_thunks), /*trip_count=*/std::nullopt,
      /*devices_per_host=*/1);

  ConvertToCommandsOptions options;
  options.synchronization_mode = serialize;
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       ConvertToCommands(thunks, options));

  // Construct a command buffer thunk with command sequence and fallback thunks.
  CommandBufferThunk thunk(
      std::move(executor), Thunk::ThunkInfo(), /*devices_in_process=*/1,
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks)));

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({pred, loop_cnt, num_iters, a, b}, 0,
                                &allocator);

  Thunk::ExecuteParams params =
      CreateExecuteParams(run_options, allocations, stream.get());

  ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  ASSERT_OK(thunk.Initialize({stream_executor,
                              static_cast<Thunk::ExecutableSource>(source),
                              &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value 10 times.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 10));

  // Initialize `loop_cnt` to `5` and check that we run only 5 iterations.
  ASSERT_OK(stream->Memset32(&loop_cnt, 5, sizeof(int32_t)));

  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 15));
}

TEST(CommandBufferThunkTest, ToStringPrintsNestedThunks) {
  BufferAllocation alloc_a(/*index=*/0, /*size=*/4, /*color=*/0);
  BufferAllocation::Slice slice_a(&alloc_a, /*offset=*/0, /*size=*/4);
  auto memset_thunk =
      std::make_unique<Memset32BitValueThunk>(Thunk::ThunkInfo(), 42, slice_a);
  Memset32BitValueThunk* memset_thunk_ptr = memset_thunk.get();
  CommandSequence commands;
  commands.Append(memset_thunk_ptr);
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(std::move(commands), serialize));
  ThunkSequence thunks;
  thunks.push_back(std::move(memset_thunk));
  CommandBufferThunk thunk(
      std::move(executor), Thunk::ThunkInfo(), /*devices_in_process=*/1,
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks)));
  EXPECT_THAT(thunk.ToString(/*indent=*/1),
              HasSubstr("    000: kMemset32BitValue"));
}

TEST(CommandBufferThunkTest, EvictsAndReRecordsCommandBuffer) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  constexpr int64_t kLength = 4;
  constexpr int64_t kByteLength = sizeof(int32_t) * kLength;
  Shape shape = ShapeUtil::MakeShape(S32, {kLength});

  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(kLength, 0);
  se::DeviceAddress<int32_t> b =
      stream_executor->AllocateArray<int32_t>(kLength, 0);
  ASSERT_OK(stream->Memset32(&a, 42, kByteLength));
  ASSERT_OK(stream->MemZero(&b, kByteLength));

  BufferAllocation source(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation destination(/*index=*/1, kByteLength, /*color=*/0);
  BufferAllocation::Slice source_slice(&source, 0, kByteLength);
  BufferAllocation::Slice destination_slice(&destination, 0, kByteLength);

  int record_count = 0;
  CommandSequence commands;
  commands.Emplace<CountingDeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo(), ShapedSlice{source_slice, shape},
      ShapedSlice{destination_slice, shape}, kByteLength, &record_count);
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(std::move(commands), serialize));

  CommandBufferThunk thunk1(std::move(executor), Thunk::ThunkInfo(),
                            /*devices_in_process=*/2);

  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({a, b}, /*device_ordinal=*/0, &allocator);
  std::vector<BufferAllocation::Index> persistent_alloc_indices = {0, 1};
  Thunk::ExecuteParams params =
      CreateExecuteParams(run_options, allocations, stream.get(),
                          absl::MakeConstSpan(persistent_alloc_indices));

  ASSERT_OK(thunk1.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(record_count, 1);

  // Executing again without eviction does not re-record.
  ASSERT_OK(thunk1.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(record_count, 1);

  // Constructing a new CommandBufferThunk evicts thunk1's command buffer.
  CommandBufferThunk thunk2(CommandExecutor{}, Thunk::ThunkInfo{},
                            /*devices_in_process=*/2);

  // Re-executing thunk1 recreates and re-records its command buffer.
  ASSERT_OK(stream->MemZero(&b, kByteLength));
  ASSERT_OK(thunk1.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(record_count, 2);

  std::vector<int32_t> result(kLength, 0);
  ASSERT_OK(stream->Memcpy(result.data(), b, kByteLength));
  EXPECT_EQ(result, std::vector<int32_t>(kLength, 42));
}

//===----------------------------------------------------------------------===//
// Reusing the set-up of commands in later executions
//===----------------------------------------------------------------------===//

namespace {

struct SetUpCounts {
  int prepares = 0;
  int initializes = 0;
  int records = 0;
  int executes = 0;
};

struct SetUpCountingMemsetOptions {
  Thunk::SetUpScope set_up_scope = Thunk::SetUpScope::kExecution;
  std::optional<GpuCliqueKey> clique;
  bool symmetric_memory = false;
  bool requires_warmup = false;
  bool requires_update_on_initialize = false;
};

// A memset command that requests a clique (if any) in `Prepare`, and symmetric
// memory for its destination on it if asked to. It counts how often it is
// prepared, initialized, recorded and executed.
class SetUpCountingMemsetThunk : public Memset32BitValueThunk {
 public:
  SetUpCountingMemsetThunk(uint32_t value,
                           const BufferAllocation::Slice& destination,
                           SetUpCountingMemsetOptions options,
                           SetUpCounts* counts)
      : Memset32BitValueThunk(Thunk::ThunkInfo(), value, destination),
        destination_(destination),
        options_(std::move(options)),
        counts_(counts) {}

  absl::Status Prepare(const PrepareParams& params) override {
    ++counts_->prepares;
    if (!options_.clique.has_value()) {
      return absl::OkStatus();
    }
    std::vector<GlobalDeviceId> group(options_.clique->devices().begin(),
                                      options_.clique->devices().end());
    ABSL_RETURN_IF_ERROR(params.collective_clique_requests->RequestClique(
        *options_.clique, {group}));
    if (!options_.symmetric_memory) {
      return absl::OkStatus();
    }
    return params.collective_memory_requests->RequestSymmetricAllocation(
        *options_.clique, destination_.index());
  }

  absl::Status Initialize(const InitializeParams& params) override {
    ++counts_->initializes;
    return Memset32BitValueThunk::Initialize(params);
  }

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    ++counts_->executes;
    return Memset32BitValueThunk::ExecuteOnStream(params);
  }

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override {
    ++counts_->records;
    return Memset32BitValueThunk::Record(execute_params, record_params,
                                         std::move(record_action),
                                         command_buffer);
  }

  SetUpScope set_up_scope() const override { return options_.set_up_scope; }
  bool requires_warmup() const override { return options_.requires_warmup; }
  bool requires_update_on_initialize() const override {
    return options_.requires_update_on_initialize;
  }

 private:
  BufferAllocation::Slice destination_;
  SetUpCountingMemsetOptions options_;
  SetUpCounts* counts_;
};

// A clique that thunks executed before command buffer thunks request.
const GpuCliqueKey& EarlierClique() {
  static const GpuCliqueKey* clique =
      new GpuCliqueKey({GlobalDeviceId(4), GlobalDeviceId(5)}, 2);
  return *clique;
}

// The clique and symmetric memory requests of an execution, in the order in
// which it acquires cliques and symmetric memory.
struct ExecutionRequests {
  std::vector<std::pair<size_t, GpuCliqueKey>> cliques;
  std::vector<std::tuple<size_t, GpuCliqueKey,
                         absl::btree_set<BufferAllocation::Index>>>
      symmetric_memories;
};

void ExpectSameRequests(const ExecutionRequests& a,
                        const ExecutionRequests& b) {
  EXPECT_EQ(a.cliques, b.cliques);
  EXPECT_EQ(a.symmetric_memories, b.symmetric_memories);
}

// Prepares, initializes and executes `thunks` like an executable does in one
// execution, after a thunk that requests `EarlierClique()` and symmetric
// memory on it, and returns the requests. Calls `after_prepare` after preparing
// the thunks.
absl::StatusOr<ExecutionRequests> RunExecution(
    absl::Span<CommandBufferThunk* const> thunks, se::Stream* stream,
    const BufferAllocations& allocations,
    absl::Span<const BufferAllocation::Index> persistent_alloc_indices,
    CollectiveParams* collective_params = nullptr,
    absl::FunctionRef<void()> after_prepare = [] {}) {
  Thunk::ExecutionScopedState execution_scoped_state;
  CollectiveCliqueRequests clique_requests;
  ABSL_RETURN_IF_ERROR(clique_requests.RequestClique(
      EarlierClique(), {{GlobalDeviceId(4), GlobalDeviceId(5)}}));
  CollectiveMemoryRequests memory_requests(allocations);
  ABSL_RETURN_IF_ERROR(memory_requests.RequestSymmetricAllocation(
      EarlierClique(), /*allocation=*/0));
  Thunk::PrepareParams prepare_params{collective_params,
                                      &clique_requests,
                                      &memory_requests,
                                      stream->parent(),
                                      &allocations,
                                      &execution_scoped_state};
  for (CommandBufferThunk* thunk : thunks) {
    ABSL_RETURN_IF_ERROR(thunk->Prepare(prepare_params));
  }
  after_prepare();

  Thunk::InitializeParams initialize_params;
  initialize_params.executor = stream->parent();
  initialize_params.buffer_allocations = &allocations;
  initialize_params.stream = stream;
  initialize_params.command_buffer_trace_stream = stream;
  initialize_params.collective_params = collective_params;
  initialize_params.execution_scoped_state = &execution_scoped_state;
  initialize_params.persistent_alloc_indices = persistent_alloc_indices;
  for (CommandBufferThunk* thunk : thunks) {
    ABSL_RETURN_IF_ERROR(thunk->Initialize(initialize_params));
  }

  ServiceExecutableRunOptions run_options;
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream, stream, nullptr, nullptr, nullptr,
      /*additional_compute_streams=*/{}, &execution_scoped_state,
      persistent_alloc_indices);
  for (CommandBufferThunk* thunk : thunks) {
    ABSL_RETURN_IF_ERROR(thunk->ExecuteOnStream(execute_params));
  }
  ABSL_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  ExecutionRequests requests;
  for (const CollectiveCliqueRequests::CliqueRequest& request :
       clique_requests.OrderedRequestedCliques()) {
    requests.cliques.emplace_back(request.id, request.key);
  }
  for (const CollectiveMemoryRequests::CollectiveAllocations& allocations :
       memory_requests.OrderedSymmetricAllocations()) {
    requests.symmetric_memories.emplace_back(allocations.id, allocations.clique,
                                             allocations.allocations);
  }
  return requests;
}

struct ReuseSetUpThunkOptions {
  // Gives the thunk fallback thunks that own its commands, like in production.
  bool fallback_thunks = false;
  // Makes the set-up scope of the second command `kExecution`.
  bool execution_scoped_command = false;
  bool requires_warmup = false;
  bool requires_update_on_initialize = false;
  // Index of the first of the four buffers that the thunk writes.
  int first_buffer = 0;
};

class ReuseSetUpTest : public ::testing::Test {
 protected:
  static constexpr int kNumBuffers = 8;
  static constexpr int64_t kLength = 4;
  static constexpr int64_t kByteLength = sizeof(int32_t) * kLength;
  using SetUpScope = Thunk::SetUpScope;

  void SetUp() override {
    executor_ = GpuExecutor();
    ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream());
    for (int i = 0; i < kNumBuffers; ++i) {
      buffers_.push_back(executor_->AllocateArray<int32_t>(kLength, 0));
      allocations_.emplace_back(/*index=*/i, kByteLength, /*color=*/0);
      persistent_alloc_indices_.push_back(i);
    }
  }

  void TearDown() override {
    for (se::DeviceAddress<int32_t>& buffer : buffers_) {
      executor_->Deallocate(&buffer);
    }
  }

  // Creates a thunk with four memset commands that write `i + 1` into buffer
  // `i`. Their set-up scopes are kCollectiveTopology, kCollectiveTopology,
  // kDevice and kCollectiveTopology. They request two cliques of the same size
  // and a larger one, and symmetric memory on the two cliques of the same size.
  std::unique_ptr<CommandBufferThunk> CreateThunk(
      ReuseSetUpThunkOptions options = {}) {
    std::vector<SetUpCountingMemsetOptions> commands_options = {
        {SetUpScope::kCollectiveTopology, cliques_[0],
         /*symmetric_memory=*/true},
        {options.execution_scoped_command ? SetUpScope::kExecution
                                          : SetUpScope::kCollectiveTopology,
         cliques_[1]},
        {SetUpScope::kDevice, std::nullopt},
        {SetUpScope::kCollectiveTopology, cliques_[2],
         /*symmetric_memory=*/true}};
    ThunkSequence thunks;
    for (int i = 0; i < commands_options.size(); ++i) {
      int buffer = options.first_buffer + i;
      commands_options[i].requires_warmup = options.requires_warmup;
      commands_options[i].requires_update_on_initialize =
          options.requires_update_on_initialize;
      thunks.push_back(std::make_unique<SetUpCountingMemsetThunk>(
          buffer + 1,
          BufferAllocation::Slice(&allocations_[buffer], 0, kByteLength),
          commands_options[i], &counts_[buffer]));
    }

    CommandSequence commands;
    for (std::unique_ptr<Thunk>& thunk : thunks) {
      auto* command = static_cast<SetUpCountingMemsetThunk*>(thunk.get());
      if (options.fallback_thunks) {
        commands.Append(command);
      } else {
        commands.Append(std::unique_ptr<Command>(command));
        thunk.release();
      }
    }
    CommandExecutor executor =
        CommandExecutor::Create(std::move(commands), serialize).value();
    Thunk::ThunkInfo thunk_info;
    thunk_info.thunk_id = thunk_ids_.GetNextThunkId();
    return std::make_unique<CommandBufferThunk>(
        std::move(executor), thunk_info, /*devices_in_process=*/1,
        options.fallback_thunks ? std::make_unique<SequentialThunk>(
                                      Thunk::ThunkInfo(), std::move(thunks))
                                : nullptr);
  }

  BufferAllocations Allocations(
      absl::Span<const se::DeviceAddress<int32_t>> buffers) {
    std::vector<se::DeviceAddressBase> addresses(buffers.begin(),
                                                 buffers.end());
    return BufferAllocations(addresses, /*device_ordinal=*/0, &allocator_);
  }

  // Returns the number of prepares and initializes of the commands that write
  // buffers `first_buffer` to `first_buffer + 3`.
  std::vector<std::pair<int, int>> SetUps(int first_buffer = 0) const {
    std::vector<std::pair<int, int>> set_ups;
    for (int i = first_buffer; i < first_buffer + 4; ++i) {
      set_ups.emplace_back(counts_[i].prepares, counts_[i].initializes);
    }
    return set_ups;
  }

  int Count(int SetUpCounts::* count, int first_buffer = 0) const {
    int total = 0;
    for (int i = first_buffer; i < first_buffer + 4; ++i) {
      total += counts_[i].*count;
    }
    return total;
  }

  // Expects that buffers `first_buffer` to `first_buffer + 3` of `allocations`
  // were written, and clears them.
  void ExpectWritten(const BufferAllocations& allocations,
                     int first_buffer = 0) {
    for (int i = first_buffer; i < first_buffer + 4; ++i) {
      se::DeviceAddress<int32_t> buffer(allocations.GetDeviceAddress(i));
      std::vector<int32_t> values(kLength, 0);
      ASSERT_OK(stream_->Memcpy(values.data(), buffer, kByteLength));
      ASSERT_OK(stream_->BlockHostUntilDone());
      EXPECT_EQ(values, std::vector<int32_t>(kLength, i + 1)) << "buffer " << i;
      ASSERT_OK(stream_->MemZero(&buffer, kByteLength));
    }
  }

  se::StreamExecutor* executor_;
  std::unique_ptr<se::Stream> stream_;
  stream_executor::StreamExecutorAddressAllocator allocator_{GpuExecutor()};
  std::vector<se::DeviceAddress<int32_t>> buffers_;
  std::vector<BufferAllocation> allocations_;
  std::vector<BufferAllocation::Index> persistent_alloc_indices_;
  ThunkIdGenerator thunk_ids_;
  GpuCliqueKey cliques_[3] = {
      GpuCliqueKey({GlobalDeviceId(0), GlobalDeviceId(1)}, 2),
      GpuCliqueKey({GlobalDeviceId(0), GlobalDeviceId(1), GlobalDeviceId(2),
                    GlobalDeviceId(3)},
                   4),
      GpuCliqueKey({GlobalDeviceId(2), GlobalDeviceId(3)}, 2)};
  SetUpCounts counts_[kNumBuffers];
};

TEST_F(ReuseSetUpTest, ReusesSetUpOfCommands) {
  for (bool fallback_thunks : {false, true}) {
    SCOPED_TRACE(absl::StrCat("fallback_thunks=", fallback_thunks));
    std::fill(std::begin(counts_), std::end(counts_), SetUpCounts{});
    std::unique_ptr<CommandBufferThunk> thunk =
        CreateThunk({.fallback_thunks = fallback_thunks});
    BufferAllocations allocations = Allocations(buffers_);

    // The first execution records the command buffer and the second one
    // collects the requests of commands, which later executions make instead
    // of preparing and initializing commands.
    ASSERT_OK_AND_ASSIGN(ExecutionRequests first,
                         RunExecution({thunk.get()}, stream_.get(), allocations,
                                      persistent_alloc_indices_));
    ExpectWritten(allocations);
    for (int i = 0; i < 3; ++i) {
      ASSERT_OK_AND_ASSIGN(
          ExecutionRequests later,
          RunExecution({thunk.get()}, stream_.get(), allocations,
                       persistent_alloc_indices_));
      ExpectWritten(allocations);
      ExpectSameRequests(later, first);
    }
    EXPECT_THAT(SetUps(), Each(Pair(2, 2)));
    EXPECT_EQ(Count(&SetUpCounts::records), 4);
  }
}

TEST_F(ReuseSetUpTest, SetsUpAllCommandsIfOneIsExecutionScoped) {
  std::unique_ptr<CommandBufferThunk> thunk =
      CreateThunk({.execution_scoped_command = true});
  BufferAllocations allocations = Allocations(buffers_);
  for (int i = 0; i < 4; ++i) {
    ASSERT_OK(RunExecution({thunk.get()}, stream_.get(), allocations,
                           persistent_alloc_indices_));
    ExpectWritten(allocations);
  }
  EXPECT_THAT(SetUps(), Each(Pair(4, 4)));
}

TEST_F(ReuseSetUpTest, ReusesSetUpOfEachThunkInAnExecution) {
  std::unique_ptr<CommandBufferThunk> thunk0 = CreateThunk();
  std::unique_ptr<CommandBufferThunk> thunk1 =
      CreateThunk({.first_buffer = 4});
  BufferAllocations allocations = Allocations(buffers_);
  for (int i = 0; i < 4; ++i) {
    ASSERT_OK(RunExecution({thunk0.get(), thunk1.get()}, stream_.get(),
                           allocations, persistent_alloc_indices_));
    ExpectWritten(allocations, /*first_buffer=*/0);
    ExpectWritten(allocations, /*first_buffer=*/4);
  }
  for (int first_buffer : {0, 4}) {
    EXPECT_THAT(SetUps(first_buffer), Each(Pair(2, 2)));
    EXPECT_EQ(Count(&SetUpCounts::records, first_buffer), 4);
  }
}

TEST_F(ReuseSetUpTest, ReusesSetUpOnceTheCommandBufferIsRecorded) {
  std::unique_ptr<CommandBufferThunk> thunk =
      CreateThunk({.fallback_thunks = true, .requires_warmup = true});
  BufferAllocations allocations = Allocations(buffers_);

  // The first execution warms up by executing the fallback thunks, the second
  // one records the command buffer and the third one collects the requests.
  for (int i = 0; i < 4; ++i) {
    ASSERT_OK(RunExecution({thunk.get()}, stream_.get(), allocations,
                           persistent_alloc_indices_));
    ExpectWritten(allocations);
  }
  EXPECT_THAT(SetUps(), Each(Pair(3, 3)));
  EXPECT_EQ(Count(&SetUpCounts::records), 4);
}

TEST_F(ReuseSetUpTest, UpdatesCommandBufferWithoutSettingUpCommands) {
  std::unique_ptr<CommandBufferThunk> thunk = CreateThunk();
  std::vector<BufferAllocation::Index> all_dynamic;
  BufferAllocations allocations = Allocations(buffers_);
  for (int i = 0; i < 2; ++i) {
    ASSERT_OK(
        RunExecution({thunk.get()}, stream_.get(), allocations, all_dynamic));
  }

  // Move the destination of the first memset.
  std::vector<se::DeviceAddress<int32_t>> moved_buffers = buffers_;
  moved_buffers[0] = buffers_[kNumBuffers - 1];
  BufferAllocations moved_allocations = Allocations(moved_buffers);
  ASSERT_OK(RunExecution({thunk.get()}, stream_.get(), moved_allocations,
                         all_dynamic));
  ExpectWritten(moved_allocations);
  EXPECT_THAT(SetUps(), Each(Pair(2, 2)));
  EXPECT_EQ(counts_[0].records, 2);
  EXPECT_EQ(Count(&SetUpCounts::records), 5);
}

TEST_F(ReuseSetUpTest, UpdatesCommandBufferAtInitializationWithoutSettingUp) {
  std::unique_ptr<CommandBufferThunk> thunk =
      CreateThunk({.requires_update_on_initialize = true});
  std::vector<BufferAllocation::Index> all_dynamic;
  BufferAllocations allocations = Allocations(buffers_);
  for (int i = 0; i < 4; ++i) {
    ASSERT_OK(
        RunExecution({thunk.get()}, stream_.get(), allocations, all_dynamic));
    ExpectWritten(allocations);
  }
  // Commands that require an update at initialization are recorded again in
  // every execution, also when it reuses their set-up.
  EXPECT_THAT(SetUps(), Each(Pair(2, 2)));
  EXPECT_EQ(Count(&SetUpCounts::records), 16);
}

TEST_F(ReuseSetUpTest, LaunchesReusedCommandBufferAfterEviction) {
  std::unique_ptr<CommandBufferThunk> thunk = CreateThunk();
  BufferAllocations allocations = Allocations(buffers_);
  for (int i = 0; i < 2; ++i) {
    ASSERT_OK(RunExecution({thunk.get()}, stream_.get(), allocations,
                           persistent_alloc_indices_));
  }

  // Constructing a command buffer thunk evicts the command buffers of other
  // thunks. The execution that already decided to reuse the command buffer
  // still launches it.
  ASSERT_OK(RunExecution(
      {thunk.get()}, stream_.get(), allocations, persistent_alloc_indices_,
      /*collective_params=*/nullptr, [] {
        CommandBufferThunk evicting(CommandExecutor{}, Thunk::ThunkInfo(),
                                    /*devices_in_process=*/1);
      }));
  ExpectWritten(allocations);
  EXPECT_THAT(SetUps(), Each(Pair(2, 2)));
  EXPECT_EQ(Count(&SetUpCounts::records), 4);

  // The next execution records a new command buffer.
  ASSERT_OK(RunExecution({thunk.get()}, stream_.get(), allocations,
                         persistent_alloc_indices_));
  ExpectWritten(allocations);
  EXPECT_THAT(SetUps(), Each(Pair(3, 3)));
  EXPECT_EQ(Count(&SetUpCounts::records), 8);
}

TEST_F(ReuseSetUpTest, SetsUpAllCommandsWhileProfiling) {
  std::unique_ptr<CommandBufferThunk> thunk =
      CreateThunk({.fallback_thunks = true});
  BufferAllocations allocations = Allocations(buffers_);
  for (int i = 0; i < 2; ++i) {
    ASSERT_OK(RunExecution({thunk.get()}, stream_.get(), allocations,
                           persistent_alloc_indices_));
  }

  {
    // Profiled executions execute the fallback thunks, which they set up.
    ASSERT_OK_AND_ASSIGN(auto profiler_lock,
                         tsl::profiler::ProfilerLock::Acquire());
    ASSERT_OK(RunExecution({thunk.get()}, stream_.get(), allocations,
                           persistent_alloc_indices_));
    ExpectWritten(allocations);
    EXPECT_THAT(SetUps(), Each(Pair(3, 3)));
    EXPECT_EQ(Count(&SetUpCounts::executes), 4);
  }

  ASSERT_OK(RunExecution({thunk.get()}, stream_.get(), allocations,
                         persistent_alloc_indices_));
  ExpectWritten(allocations);
  EXPECT_THAT(SetUps(), Each(Pair(3, 3)));
  EXPECT_EQ(Count(&SetUpCounts::records), 4);
}

TEST_F(ReuseSetUpTest, LaunchesReusedCommandBufferIfProfilingStartsLater) {
  std::unique_ptr<CommandBufferThunk> thunk =
      CreateThunk({.fallback_thunks = true});
  BufferAllocations allocations = Allocations(buffers_);
  for (int i = 0; i < 2; ++i) {
    ASSERT_OK(RunExecution({thunk.get()}, stream_.get(), allocations,
                           persistent_alloc_indices_));
  }

  // The fallback thunks were not set up in this execution.
  std::optional<tsl::profiler::ProfilerLock> profiler_lock;
  ASSERT_OK(RunExecution(
      {thunk.get()}, stream_.get(), allocations, persistent_alloc_indices_,
      /*collective_params=*/nullptr, [&] {
        profiler_lock.emplace(tsl::profiler::ProfilerLock::Acquire().value());
      }));
  profiler_lock.reset();
  ExpectWritten(allocations);
  EXPECT_THAT(SetUps(), Each(Pair(2, 2)));
  EXPECT_EQ(Count(&SetUpCounts::executes), 0);
}

TEST_F(ReuseSetUpTest, SetsUpCommandsAgainForAnotherTopology) {
  std::unique_ptr<CommandBufferThunk> thunk = CreateThunk();
  BufferAllocations allocations = Allocations(buffers_);

  DeviceAssignment device_assignment(/*replica_count=*/2,
                                     /*computation_count=*/1);
  device_assignment(0, 0) = 0;
  device_assignment(1, 0) = 1;
  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream_.get());
  run_options.mutable_run_options()->set_device_assignment(&device_assignment);
  ASSERT_OK_AND_ASSIGN(
      CollectiveParams collective_params,
      CollectiveParams::Create(run_options, /*async_streams=*/{},
                               LocalDeviceId(executor_->device_ordinal())));

  for (int i = 0; i < 3; ++i) {
    ASSERT_OK(RunExecution({thunk.get()}, stream_.get(), allocations,
                           persistent_alloc_indices_, &collective_params));
  }
  EXPECT_THAT(SetUps(), Each(Pair(2, 2)));

  // Changing the device assignment in place changes the topology.
  device_assignment(0, 0) = 1;
  device_assignment(1, 0) = 0;
  for (int i = 0; i < 2; ++i) {
    ASSERT_OK(RunExecution({thunk.get()}, stream_.get(), allocations,
                           persistent_alloc_indices_, &collective_params));
  }
  EXPECT_THAT(SetUps(), Each(Pair(3, 3)));
}

TEST_F(ReuseSetUpTest, SetsUpAllCommandsWithoutExecutionScopedState) {
  std::unique_ptr<CommandBufferThunk> thunk = CreateThunk();
  BufferAllocations allocations = Allocations(buffers_);
  for (int i = 0; i < 2; ++i) {
    ASSERT_OK(RunExecution({thunk.get()}, stream_.get(), allocations,
                           persistent_alloc_indices_));
  }
  for (int i = 0; i < 2; ++i) {
    CollectiveCliqueRequests clique_requests;
    CollectiveMemoryRequests memory_requests(allocations);
    Thunk::PrepareParams prepare_params{/*collective_params=*/nullptr,
                                        &clique_requests, &memory_requests,
                                        executor_, &allocations};
    ASSERT_OK(thunk->Prepare(prepare_params));
  }
  EXPECT_THAT(SetUps(), Each(Pair(4, 2)));
}

}  // namespace

}  // namespace xla::gpu
