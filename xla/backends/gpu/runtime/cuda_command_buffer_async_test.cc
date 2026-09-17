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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/memset_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/execution_graph.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_test_kernels_fatbin.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using MemoryAccess = BufferUse::MemoryAccess;

TEST(CudaCommandBufferAsyncTest, ForkJoinReplayAndUpdate) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());
  constexpr int64_t kLength = 4;
  constexpr int64_t kBytes = kLength * sizeof(int32_t);
  Shape shape = ShapeUtil::MakeShape(S32, {kLength});
  se::DeviceAddress<int32_t> a =
      stream_executor->AllocateArray<int32_t>(kLength, 0);
  se::DeviceAddress<int32_t> b =
      stream_executor->AllocateArray<int32_t>(kLength, 0);
  se::DeviceAddress<int32_t> c =
      stream_executor->AllocateArray<int32_t>(kLength, 0);
  se::DeviceAddress<int32_t> updated_c =
      stream_executor->AllocateArray<int32_t>(kLength, 0);
  BufferAllocation alloc_a(0, kBytes, 0);
  BufferAllocation alloc_b(1, kBytes, 0);
  BufferAllocation alloc_c(2, kBytes, 0);
  BufferAllocation::Slice slice_a(&alloc_a, 0, kBytes);
  BufferAllocation::Slice slice_b(&alloc_b, 0, kBytes);
  BufferAllocation::Slice slice_c(&alloc_c, 0, kBytes);

  ThunkSequence thunks;
  ThunkSequence body;
  body.Emplace<Memset32BitValueThunk>(Thunk::ThunkInfo(), 12, slice_b);
  body.Emplace<Memset32BitValueThunk>(Thunk::ThunkInfo(), 24, slice_b);
  auto start = std::make_unique<AsyncStartThunk>(
      Thunk::ThunkInfo(), ComputationStreamId(0), std::move(body));
  auto async_execution = start->async_execution();
  thunks.push_back(std::move(start));
  thunks.Emplace<Memset32BitValueThunk>(Thunk::ThunkInfo(), 42, slice_a);
  thunks.Emplace<AsyncDoneThunk>(Thunk::ThunkInfo(), async_execution);
  std::vector<ShapedSlice> args = {
      {slice_a, shape}, {slice_b, shape}, {slice_c, shape}};
  thunks.push_back(KernelThunk::MakeKernelThunk(
      "AddI32", args,
      {MemoryAccess::kRead, MemoryAccess::kRead, MemoryAccess::kWrite},
      LaunchDimensions(1, kLength), /*shmem_bytes=*/0));
  ASSERT_OK_AND_ASSIGN(
      CommandExecutor executor,
      ConvertToCommands(thunks, {CommandExecutor::SynchronizationMode::kLHS}));
  EXPECT_EQ(executor.size(), 5);
  ASSERT_FALSE(executor.execution_graph()->is_sequential());
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations({a, b, c}, 0, &allocator);
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr,
      nullptr, /*additional_compute_streams=*/{},
      /*execution_scoped_state=*/nullptr,
      /*persistent_alloc_indices=*/absl::Span<const BufferAllocation::Index>{});
  ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> fatbin,
                       se::gpu::GetGpuTestKernelsFatbin("CUDA"));
  Thunk::ExecutableSource source{/*text=*/{}, /*binary=*/fatbin};
  ASSERT_OK(
      thunk.Initialize({stream_executor, source, &allocations, stream.get()}));

  // Exercise graph creation, replay, and an update with a new output address.
  for (int i = 0; i < 3; ++i) {
    se::DeviceAddress<int32_t> output = i == 2 ? updated_c : c;
    allocations = BufferAllocations({a, b, output}, 0, &allocator);
    ASSERT_OK(stream->MemZero(&a, kBytes));
    ASSERT_OK(stream->MemZero(&b, kBytes));
    ASSERT_OK(stream->MemZero(&output, kBytes));
    ASSERT_OK(thunk.ExecuteOnStream(params));
    std::vector<int32_t> result(kLength);
    ASSERT_OK(stream->Memcpy(result.data(), output, kBytes));
    ASSERT_OK(stream->BlockHostUntilDone());
    EXPECT_EQ(result, std::vector<int32_t>(kLength, 66));
  }
  stream_executor->Deallocate(&a);
  stream_executor->Deallocate(&b);
  stream_executor->Deallocate(&c);
  stream_executor->Deallocate(&updated_c);
}

}  // namespace
}  // namespace xla::gpu
