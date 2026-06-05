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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/while_loop.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/mock_command_buffer.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::testing::_;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

void* AddBytes(void* ptr, int64_t byte_offset) {
  return static_cast<char*>(ptr) + byte_offset;
}

Thunk::ExecuteParams MakeExecuteParams(const BufferAllocations& allocations) {
  return Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), allocations,
      /*stream=*/nullptr, /*command_buffer_trace_stream=*/nullptr,
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr);
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
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());
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

  TF_ASSERT_OK_AND_ASSIGN(
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

TEST(DynamicMemcpyThunkTest, RecordCommandBufferUsesStaticOffsets) {
  constexpr int64_t kBufferBytes = 64;
  constexpr uint64_t kCopyBytes = 8;
  constexpr int64_t kSrcOffset = 4;
  constexpr int64_t kDstOffset = 16;

  BufferAllocation src_alloc(/*index=*/0, kBufferBytes, /*color=*/0);
  BufferAllocation dst_alloc(/*index=*/1, kBufferBytes, /*color=*/0);
  Shape shape = ShapeUtil::MakeShape(U8, {kBufferBytes});

  DynamicMemcpyThunk thunk(
      Thunk::ThunkInfo(),
      ShapedSlice{BufferAllocation::Slice(&src_alloc, 0, kBufferBytes), shape},
      ShapedSlice{BufferAllocation::Slice(&dst_alloc, 0, kBufferBytes), shape},
      kCopyBytes,
      DynamicMemcpyThunk::Offsets{/*depends_on_loop=*/false,
                                  /*src_offsets=*/{kSrcOffset},
                                  /*dst_offsets=*/{kDstOffset}});
  EXPECT_FALSE(thunk.requires_update());

  std::vector<char> src_storage(kBufferBytes);
  std::vector<char> dst_storage(kBufferBytes);
  se::DeviceAddressBase src_base(src_storage.data(), kBufferBytes);
  se::DeviceAddressBase dst_base(dst_storage.data(), kBufferBytes);
  BufferAllocations allocations({src_base, dst_base}, /*device_ordinal=*/0,
                                /*memory_allocator=*/nullptr);
  Thunk::ExecuteParams execute_params = MakeExecuteParams(allocations);

  CommandStateManager state;
  Command::RecordParams record_params = {state};
  testing::NiceMock<se::MockCommandBuffer> command_buffer;
  const auto* recorded_command =
      reinterpret_cast<const se::CommandBuffer::Command*>(uintptr_t{0x1234});

  EXPECT_CALL(command_buffer, CreateMemcpyD2D(_, _, kCopyBytes, _))
      .WillOnce([&](se::DeviceAddressBase* dst,
                    const se::DeviceAddressBase& src, uint64_t size,
                    auto dependencies)
                    -> absl::StatusOr<const se::CommandBuffer::Command*> {
        EXPECT_EQ(size, kCopyBytes);
        EXPECT_TRUE(dependencies.empty());
        EXPECT_EQ(src.opaque(), AddBytes(src_base.opaque(), kSrcOffset));
        EXPECT_EQ(src.size(), kCopyBytes);
        EXPECT_EQ(dst->opaque(), AddBytes(dst_base.opaque(), kDstOffset));
        EXPECT_EQ(dst->size(), kCopyBytes);
        return recorded_command;
      });

  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* command,
                       thunk.Record(execute_params, record_params,
                                    Command::RecordCreate{/*dependencies=*/{}},
                                    &command_buffer));
  EXPECT_EQ(command, recorded_command);
}

TEST(DynamicMemcpyThunkTest, RecordCommandBufferUpdateUsesLoopOffsets) {
  constexpr int64_t kBufferBytes = 64;
  constexpr uint64_t kCopyBytes = 8;
  constexpr int64_t kSrcOffset = 12;
  constexpr int64_t kDstOffset = 24;

  BufferAllocation src_alloc(/*index=*/0, kBufferBytes, /*color=*/0);
  BufferAllocation dst_alloc(/*index=*/1, kBufferBytes, /*color=*/0);
  Shape shape = ShapeUtil::MakeShape(U8, {kBufferBytes});

  DynamicMemcpyThunk thunk(
      Thunk::ThunkInfo(),
      ShapedSlice{BufferAllocation::Slice(&src_alloc, 0, kBufferBytes), shape},
      ShapedSlice{BufferAllocation::Slice(&dst_alloc, 0, kBufferBytes), shape},
      kCopyBytes,
      DynamicMemcpyThunk::Offsets{/*depends_on_loop=*/true,
                                  /*src_offsets=*/{4, kSrcOffset},
                                  /*dst_offsets=*/{8, kDstOffset}});
  EXPECT_TRUE(thunk.requires_update());

  std::vector<char> src_storage(kBufferBytes);
  std::vector<char> dst_storage(kBufferBytes);
  se::DeviceAddressBase src_base(src_storage.data(), kBufferBytes);
  se::DeviceAddressBase dst_base(dst_storage.data(), kBufferBytes);
  BufferAllocations allocations({src_base, dst_base}, /*device_ordinal=*/0,
                                /*memory_allocator=*/nullptr);
  Thunk::ExecuteParams execute_params = MakeExecuteParams(allocations);

  CommandStateManager state;
  Command::RecordParams record_params = {state};
  testing::NiceMock<se::MockCommandBuffer> command_buffer;
  const auto* recorded_command =
      reinterpret_cast<const se::CommandBuffer::Command*>(uintptr_t{0x1234});

  ScopedWhileLoop loop("dynamic_memcpy_command_buffer",
                       /*trip_count=*/2);
  loop.IncLoopIteration();

  EXPECT_CALL(command_buffer,
              UpdateMemcpyD2D(recorded_command, _, _, kCopyBytes))
      .WillOnce([&](const se::CommandBuffer::Command* command,
                    se::DeviceAddressBase* dst,
                    const se::DeviceAddressBase& src,
                    uint64_t size) -> absl::Status {
        EXPECT_EQ(command, recorded_command);
        EXPECT_EQ(size, kCopyBytes);
        EXPECT_EQ(src.opaque(), AddBytes(src_base.opaque(), kSrcOffset));
        EXPECT_EQ(src.size(), kCopyBytes);
        EXPECT_EQ(dst->opaque(), AddBytes(dst_base.opaque(), kDstOffset));
        EXPECT_EQ(dst->size(), kCopyBytes);
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* command,
      thunk.Record(execute_params, record_params,
                   Command::RecordUpdate{recorded_command}, &command_buffer));
  EXPECT_EQ(command, recorded_command);
}

}  // namespace
}  // namespace xla::gpu
