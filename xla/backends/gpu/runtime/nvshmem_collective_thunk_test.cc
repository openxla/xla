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

#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;

TEST(CollectiveThunkTest, NvshmemCollectiveDoneThunkProtoRoundTrip) {
  ThunkProto reference_proto =
      tsl::proto_testing::ParseTextProtoOrDie<ThunkProto>(
          R"pb(
            thunk_info {
              profile_annotation: "profile_annotation"
              execution_stream_id: 2
            }
            nvshmem_collective_done_thunk {
              thunk_kind: THUNK_KIND_NVSHMEM_ALL_REDUCE_DONE
              async_events_unique_id: 3
            }
          )pb");

  ASSERT_OK_AND_ASSIGN(
      Thunk::ThunkInfo thunk_info,
      Thunk::ThunkInfo::FromProto(reference_proto.thunk_info()));

  CollectiveThunk::AsyncEventsMap async_events_map;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<NvshmemCollectiveDoneThunk> thunk,
      NvshmemCollectiveDoneThunk::FromProto(
          thunk_info, reference_proto.nvshmem_collective_done_thunk(),
          async_events_map));
  auto event = async_events_map.find(
      AsyncEventsUniqueId{reference_proto.nvshmem_collective_done_thunk()
                              .async_events_unique_id()});
  EXPECT_NE(event, async_events_map.end());

  ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  reference_proto.mutable_nvshmem_collective_done_thunk()
      ->set_async_events_unique_id(
          absl::bit_cast<uint64_t>(event->second.get()));
  EXPECT_THAT(round_trip_proto, EqualsProto(reference_proto));
}

TEST(CollectiveThunkTest, NvshmemCollectivePermuteDoneThunkProtoRoundTrip) {
  ThunkProto reference_proto =
      tsl::proto_testing::ParseTextProtoOrDie<ThunkProto>(
          R"pb(
            thunk_info {
              profile_annotation: "profile_annotation"
              execution_stream_id: 2
            }
            nvshmem_collective_permute_done_thunk { async_events_unique_id: 3 }
          )pb");

  ASSERT_OK_AND_ASSIGN(
      Thunk::ThunkInfo thunk_info,
      Thunk::ThunkInfo::FromProto(reference_proto.thunk_info()));

  CollectiveThunk::AsyncEventsMap async_events_map;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<NvshmemCollectivePermuteDoneThunk> thunk,
      NvshmemCollectivePermuteDoneThunk::FromProto(
          thunk_info, reference_proto.nvshmem_collective_permute_done_thunk(),
          async_events_map));
  auto event = async_events_map.find(AsyncEventsUniqueId{
      reference_proto.nvshmem_collective_permute_done_thunk()
          .async_events_unique_id()});
  EXPECT_NE(event, async_events_map.end());

  ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  reference_proto.mutable_nvshmem_collective_permute_done_thunk()
      ->set_async_events_unique_id(
          absl::bit_cast<uint64_t>(event->second.get()));
  EXPECT_THAT(round_trip_proto, EqualsProto(reference_proto));
}

}  // namespace
}  // namespace xla::gpu
