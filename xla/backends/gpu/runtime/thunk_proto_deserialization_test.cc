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

#include "xla/backends/gpu/runtime/thunk_proto_deserialization.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/protobuf.h"

namespace xla::gpu {
namespace {
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pointer;
using ::testing::Property;
using ::testing::WhenDynamicCastTo;

TEST(ThunkProtoDeserializationTest, SequentialThunkChain) {
  constexpr ExecutionStreamId kExecutionStreamId{123};
  constexpr absl::string_view kProfileAnnotation = "profile_annotation";

  Thunk::ThunkInfo thunk_info{};
  thunk_info.execution_stream_id = kExecutionStreamId;
  thunk_info.profile_annotation = kProfileAnnotation;

  // This constructs the following thunk tree:
  // `SequentialThunk{SequentialThunk{}}`
  std::unique_ptr<Thunk> inner_thunk =
      std::make_unique<SequentialThunk>(thunk_info, ThunkSequence{});
  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(inner_thunk));
  SequentialThunk outer_thunk(thunk_info, std::move(thunk_sequence));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, outer_thunk.ToProto());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> new_thunk,
                          DeserializeThunkProto(proto, {}));

  EXPECT_THAT(new_thunk.get(),
              WhenDynamicCastTo<const SequentialThunk*>(Property(
                  &SequentialThunk::thunks,
                  ElementsAre(Pointer(WhenDynamicCastTo<const SequentialThunk*>(
                      Property(&SequentialThunk::thunks, IsEmpty())))))));
}

TEST(ThunkProtoDeserializationTest, CopyThunk) {
  ThunkProto proto;
  CHECK(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        copy_thunk {
          source_buffer { offset: 128 size: 384 buffer_allocation_index: 0 }
          destination_buffer { offset: 0 size: 256 buffer_allocation_index: 1 }
          mem_size: 256
        }
      )pb",
      &proto));

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> thunk,
                          DeserializeThunkProto(proto, buffer_allocations));

  auto* copy_thunk_ptr = dynamic_cast<CopyThunk*>(thunk.get());
  ASSERT_NE(copy_thunk_ptr, nullptr);  // Check the cast succeeded
  EXPECT_EQ(
      *copy_thunk_ptr,
      CopyThunk(thunk_info,
                BufferAllocation::Slice(&buffer_allocations[0],
                                        /*offset=*/128, /*size=*/384),
                BufferAllocation::Slice(&buffer_allocations[1], /*offset=*/0,
                                        /*size=*/256),
                /*mem_size=*/256));
}

TEST(ThunkProtoDeserializationTest, DeviceToHostCopyThunk) {
  ThunkProto proto;
  CHECK(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        device_to_host_copy_thunk {
          copy_thunk {
            source_buffer { offset: 128 size: 384 buffer_allocation_index: 0 }
            destination_buffer {
              offset: 0
              size: 256
              buffer_allocation_index: 1
            }
            mem_size: 256
          }
        }
      )pb",
      &proto));

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> thunk,
                          DeserializeThunkProto(proto, buffer_allocations));

  auto* d2h_copy_thunk_ptr = dynamic_cast<DeviceToHostCopyThunk*>(thunk.get());
  ASSERT_NE(d2h_copy_thunk_ptr, nullptr);  // Check the cast succeeded
  EXPECT_EQ(*d2h_copy_thunk_ptr,
            DeviceToHostCopyThunk(
                thunk_info,
                BufferAllocation::Slice(&buffer_allocations[0],
                                        /*offset=*/128, /*size=*/384),
                BufferAllocation::Slice(&buffer_allocations[1], /*offset=*/0,
                                        /*size=*/256),
                /*mem_size=*/256,
                /*events=*/nullptr,
                /*instr=*/nullptr));
}

TEST(ThunkProtoDeserializationTest, HostToDeviceCopyThunk) {
  ThunkProto proto;
  CHECK(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        host_to_device_copy_thunk {
          copy_thunk {
            source_buffer { offset: 128 size: 384 buffer_allocation_index: 0 }
            destination_buffer {
              offset: 0
              size: 256
              buffer_allocation_index: 1
            }
            mem_size: 256
          }
        }
      )pb",
      &proto));

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> thunk,
                          DeserializeThunkProto(proto, buffer_allocations));

  auto* h2d_copy_thunk_ptr = dynamic_cast<HostToDeviceCopyThunk*>(thunk.get());
  ASSERT_NE(h2d_copy_thunk_ptr, nullptr);  // Check the cast succeeded
  EXPECT_EQ(*h2d_copy_thunk_ptr,
            HostToDeviceCopyThunk(
                thunk_info,
                BufferAllocation::Slice(&buffer_allocations[0],
                                        /*offset=*/128, /*size=*/384),
                BufferAllocation::Slice(&buffer_allocations[1], /*offset=*/0,
                                        /*size=*/256),
                /*mem_size=*/256,
                /*events=*/nullptr,
                /*instr=*/nullptr));
}

}  // namespace
}  // namespace xla::gpu
