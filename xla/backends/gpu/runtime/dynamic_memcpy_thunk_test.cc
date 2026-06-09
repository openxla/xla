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

#include "xla/backends/gpu/runtime/dynamic_memcpy_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/while_loop.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

static se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

TEST(DynamicMemcpyThunkTest, ToProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  Shape shape = ShapeUtil::MakeShape(F32, {256});

  DynamicMemcpyThunk thunk(thunk_info,
                           /*source_buffer=*/
                           {BufferAllocation::Slice(&buffer_allocations[0],
                                                    /*offset=*/0,
                                                    /*size=*/1024),
                            shape},
                           /*destination_buffer=*/
                           {BufferAllocation::Slice(&buffer_allocations[1],
                                                    /*offset=*/0,
                                                    /*size=*/1024),
                            shape},
                           /*mem_size=*/256,
                           {DynamicMemcpyThunk::Offsets{
                               /*depends_on_loop=*/true,
                               /*src_offsets=*/std::vector<int64_t>{4, 8},
                               /*dst_offsets=*/std::vector<int64_t>{16, 32},
                           }});
  ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());
  EXPECT_THAT(
      proto, EqualsProto(R"pb(
        thunk_info { profile_annotation: "profile_annotation" }
        dynamic_memcpy_thunk {
          source_buffer {
            slice { offset: 0 size: 1024 buffer_allocation_index: 0 }
            shape {
              element_type: F32
              dimensions: 256
              layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
              is_dynamic_dimension: false
            }
          }
          destination_buffer {
            slice { offset: 0 size: 1024 buffer_allocation_index: 1 }
            shape {
              element_type: F32
              dimensions: 256
              layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
              is_dynamic_dimension: false
            }
          }
          mem_size: 256
          offsets {
            depends_on_loop: true
            src_offsets: 4
            src_offsets: 8
            dst_offsets: 16
            dst_offsets: 32
          }
        }
      )pb"));
}

TEST(DynamicMemcpyThunkTest, FromProto) {
  Shape shape = ShapeUtil::MakeShape(F32, {256});
  auto dynamic_memcpy_thunk_proto = ParseTextProtoOrDie<
      DynamicMemcpyThunkProto>(
      R"pb(
        source_buffer {
          slice { offset: 0 size: 1024 buffer_allocation_index: 0 }
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
        }
        destination_buffer {
          slice { offset: 0 size: 1024 buffer_allocation_index: 1 }
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
        }
        mem_size: 256
        offsets {
          depends_on_loop: true
          src_offsets: 4
          src_offsets: 8
          dst_offsets: 16
          dst_offsets: 32
        }
      )pb");

  Thunk::ThunkInfo thunk_info{};
  thunk_info.profile_annotation = "profile_annotation";

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<DynamicMemcpyThunk> thunk,
      DynamicMemcpyThunk::FromProto(thunk_info, dynamic_memcpy_thunk_proto,
                                    buffer_allocations));

  DynamicMemcpyThunk::Offsets reference_offsets{
      /*depends_on_loop=*/true,
      /*src_offsets=*/std::vector<int64_t>{4, 8},
      /*dst_offsets=*/std::vector<int64_t>{16, 32}};
  EXPECT_EQ(thunk->offsets(), reference_offsets);
  EXPECT_EQ(thunk->source(),
            (ShapedSlice{BufferAllocation::Slice(&buffer_allocations[0],
                                                 /*offset=*/0,
                                                 /*size=*/1024),
                         shape}));
  EXPECT_EQ(thunk->destination(),
            (ShapedSlice{BufferAllocation::Slice(&buffer_allocations[1],
                                                 /*offset=*/0,
                                                 /*size=*/1024),
                         shape}));
  EXPECT_EQ(thunk->mem_size(), 256);
}

TEST(DynamicMemcpyThunkTest,
     CommandBufferUpdatesLoopDependentOffsetsInsideWhileLoopContext) {
  se::StreamExecutor* executor = GpuExecutor();
  if (executor->GetDeviceDescription().gpu_compute_capability().IsRocm()) {
    GTEST_SKIP() << "DynamicMemcpyThunk command buffer updates are not "
                    "supported on ROCm";
  }

  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  constexpr int64_t kSrcLength = 4;
  constexpr int64_t kDstLength = 1;
  constexpr int64_t kSrcBytes = sizeof(int32_t) * kSrcLength;
  constexpr int64_t kDstBytes = sizeof(int32_t) * kDstLength;

  se::DeviceAddress<int32_t> src =
      executor->AllocateArray<int32_t>(kSrcLength, 0);
  se::DeviceAddress<int32_t> dst =
      executor->AllocateArray<int32_t>(kDstLength, 0);

  std::vector<int32_t> src_data{10, 20, 30, 40};
  ASSERT_OK(stream->Memcpy(&src, src_data.data(), kSrcBytes));
  ASSERT_OK(stream->MemZero(&dst, kDstBytes));

  BufferAllocation src_alloc(/*index=*/0, kSrcBytes, /*color=*/0);
  BufferAllocation dst_alloc(/*index=*/1, kDstBytes, /*color=*/0);
  BufferAllocation::Slice src_slice(&src_alloc, 0, kSrcBytes);
  BufferAllocation::Slice dst_slice(&dst_alloc, 0, kDstBytes);
  Shape src_shape = ShapeUtil::MakeShape(S32, {kSrcLength});
  Shape dst_shape = ShapeUtil::MakeShape(S32, {kDstLength});

  DynamicMemcpyThunk dynamic_memcpy_thunk(
      Thunk::ThunkInfo(), ShapedSlice{src_slice, src_shape},
      ShapedSlice{dst_slice, dst_shape}, kDstBytes,
      DynamicMemcpyThunk::Offsets{/*depends_on_loop=*/true,
                                  /*src_offsets=*/{0, sizeof(int32_t)},
                                  /*dst_offsets=*/{0, 0}});

  CommandSequence commands;
  commands.Append(&dynamic_memcpy_thunk);
  ASSERT_OK_AND_ASSIGN(CommandExecutor command_executor,
                       CommandExecutor::Create(
                           std::move(commands),
                           CommandExecutor::SynchronizationMode::kSerialize));

  CommandBufferThunk command_buffer_thunk(
      std::move(command_executor), Thunk::ThunkInfo(),
      /*thunks=*/nullptr,
      /*enable_command_buffers_during_profiling=*/true,
      DebugOptions::ALWAYS_UPDATE);

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations({src, dst}, executor->device_ordinal(),
                                &allocator);

  Thunk::PrepareParams prepare_params;
  prepare_params.executor = executor;
  prepare_params.buffer_allocations = &allocations;
  ASSERT_OK(command_buffer_thunk.Prepare(prepare_params));

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  ASSERT_OK(command_buffer_thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(),
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr);

  ScopedWhileLoop loop("dynamic_memcpy_test", /*trip_count=*/2);
  ASSERT_OK(command_buffer_thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  std::vector<int32_t> out(kDstLength, 0);
  ASSERT_OK(stream->Memcpy(out.data(), dst, kDstBytes));
  EXPECT_EQ(out, std::vector<int32_t>({10}));

  loop.IncLoopIteration();
  ASSERT_OK(command_buffer_thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  ASSERT_OK(stream->Memcpy(out.data(), dst, kDstBytes));
  EXPECT_EQ(out, std::vector<int32_t>({20}));
}

}  // namespace
}  // namespace xla::gpu
