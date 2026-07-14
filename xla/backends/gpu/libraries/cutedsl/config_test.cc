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

#include "xla/backends/gpu/libraries/cutedsl/config.h"

#include <cstdint>
#include <limits>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/libraries/cutedsl/config.pb.h"
#include "xla/tsl/platform/status_matchers.h"
#include "tsl/platform/protobuf.h"

namespace xla::gpu::cutedsl {

using ::xla::CollectiveOpGroupMode;
namespace wire = ::xla::gpu::cutedsl::proto;

namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

wire::CollectiveCallConfigV3 TestProto() {
  wire::CollectiveCallConfigV3 proto;
  proto.set_abi_clique_size(2);
  proto.set_group_mode(
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA);
  proto.set_communication_id(17);
  proto.add_replica_groups()->add_replica_ids(0);
  proto.mutable_replica_groups(0)->add_replica_ids(1);
  proto.add_replica_groups()->add_replica_ids(2);
  proto.mutable_replica_groups(1)->add_replica_ids(3);
  wire::PeerRegionProto* argument = proto.add_peer_regions();
  argument->set_endpoint(wire::PEER_REGION_ENDPOINT_PROTO_ARGUMENT);
  argument->set_buffer_index(2);
  argument->set_byte_offset(16);
  argument->set_byte_size(64);
  argument->set_required_alignment(16);
  argument->set_memory_kind(wire::PEER_MEMORY_KIND_PROTO_SYMMETRIC);

  wire::PeerRegionProto* result = proto.add_peer_regions();
  result->set_endpoint(wire::PEER_REGION_ENDPOINT_PROTO_RESULT);
  result->set_buffer_index(0);
  result->set_byte_offset(0);
  result->set_byte_size(128);
  result->set_required_alignment(64);
  result->set_memory_kind(wire::PEER_MEMORY_KIND_PROTO_SYMMETRIC);
  proto.set_barrier_before_launch(true);
  return proto;
}

std::string ToJson(const wire::CollectiveCallConfigV3& proto) {
  tsl::protobuf::util::JsonPrintOptions options;
  options.preserve_proto_field_names = true;
  options.always_print_enums_as_ints = true;
  std::string json;
  absl::Status status =
      tsl::protobuf::util::MessageToJsonString(proto, &json, options);
  EXPECT_TRUE(status.ok()) << status;
  return json;
}

absl::StatusOr<wire::CollectiveCallConfigV3> Parse(
    const wire::CollectiveCallConfigV3& proto) {
  return ParseAndValidateCollectiveCallConfig(ToJson(proto));
}

TEST(CollectiveConfigTest, ParsesCompleteProtoJsonConfiguration) {
  wire::CollectiveCallConfigV3 proto = TestProto();
  absl::StatusOr<wire::CollectiveCallConfigV3> parsed = Parse(proto);
  ASSERT_THAT(parsed, IsOk());

  EXPECT_EQ(parsed->group_mode(),
            CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA);
  EXPECT_EQ(parsed->abi_clique_size(), 2);
  EXPECT_EQ(parsed->communication_id(), 17);
  ASSERT_EQ(parsed->replica_groups_size(), 2);
  EXPECT_THAT(parsed->replica_groups(0).replica_ids(), ElementsAre(0, 1));
  EXPECT_THAT(parsed->replica_groups(1).replica_ids(), ElementsAre(2, 3));

  ASSERT_EQ(parsed->peer_regions_size(), 2);
  EXPECT_EQ(parsed->peer_regions(0).endpoint(),
            wire::PEER_REGION_ENDPOINT_PROTO_ARGUMENT);
  EXPECT_EQ(parsed->peer_regions(0).buffer_index(), 2);
  EXPECT_EQ(parsed->peer_regions(0).byte_offset(), 16);
  EXPECT_EQ(parsed->peer_regions(0).byte_size(), 64);
  EXPECT_EQ(parsed->peer_regions(0).required_alignment(), 16);
  EXPECT_EQ(parsed->peer_regions(0).memory_kind(),
            wire::PEER_MEMORY_KIND_PROTO_SYMMETRIC);
  EXPECT_EQ(parsed->peer_regions(1).endpoint(),
            wire::PEER_REGION_ENDPOINT_PROTO_RESULT);

  EXPECT_TRUE(parsed->barrier_before_launch());
}

TEST(CollectiveConfigTest, ParsesDkgProtoJsonEncoding) {
  constexpr absl::string_view kJson = R"json(
    {
      "abi_clique_size": "2",
      "communication_id": "17",
      "group_mode": 0,
      "peer_regions": [{
        "buffer_index": "2",
        "byte_offset": "16",
        "byte_size": "64",
        "endpoint": 0,
        "memory_kind": 0,
        "required_alignment": "16"
      }],
      "replica_groups": [{"replica_ids": ["0", "1"]}],
      "barrier_before_launch": true
    }
  )json";

  absl::StatusOr<wire::CollectiveCallConfigV3> parsed =
      ParseAndValidateCollectiveCallConfig(kJson);
  ASSERT_THAT(parsed, IsOk());
  EXPECT_EQ(parsed->abi_clique_size(), 2);
  ASSERT_EQ(parsed->peer_regions_size(), 1);
  EXPECT_EQ(parsed->peer_regions(0).buffer_index(), 2);
  EXPECT_TRUE(parsed->barrier_before_launch());
}

TEST(CollectiveConfigTest, RejectsMalformedJsonAndMissingFields) {
  EXPECT_THAT(ParseAndValidateCollectiveCallConfig("{"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Failed to parse")));

  wire::CollectiveCallConfigV3 proto = TestProto();
  proto.clear_abi_clique_size();
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("abi_clique_size")));

  proto = TestProto();
  proto.mutable_peer_regions(0)->clear_required_alignment();
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("required_alignment")));
}

TEST(CollectiveConfigTest, IgnoresUnknownJsonFields) {
  std::string json = ToJson(TestProto());
  ASSERT_EQ(json.back(), '}');
  json.pop_back();
  json.append(",\"future_field\":{\"nested\":1}}");
  EXPECT_THAT(ParseAndValidateCollectiveCallConfig(json), IsOk());
}

TEST(CollectiveConfigTest, ValidatesAbiCliqueSizeRange) {
  wire::CollectiveCallConfigV3 proto = TestProto();
  proto.set_abi_clique_size(std::numeric_limits<int32_t>::max());
  absl::StatusOr<wire::CollectiveCallConfigV3> parsed = Parse(proto);
  ASSERT_THAT(parsed, IsOk());
  EXPECT_EQ(parsed->abi_clique_size(), std::numeric_limits<int32_t>::max());

  for (int64_t invalid :
       {int64_t{0}, int64_t{-1},
        static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1}) {
    proto.set_abi_clique_size(invalid);
    EXPECT_THAT(Parse(proto),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("`abi_clique_size` must be in")));
  }
}

TEST(CollectiveConfigTest, RejectsMalformedCollectiveGroup) {
  wire::CollectiveCallConfigV3 proto = TestProto();
  proto.set_group_mode(static_cast<CollectiveOpGroupMode>(9));
  EXPECT_THAT(Parse(proto),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported collective group mode")));

  proto = TestProto();
  proto.set_communication_id(-1);
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("communication_id")));

  proto = TestProto();
  proto.mutable_replica_groups(0)->clear_replica_ids();
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("must not be empty")));

  proto = TestProto();
  proto.mutable_replica_groups(0)->mutable_replica_ids()->RemoveLast();
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("equal cardinality")));

  proto = TestProto();
  proto.mutable_replica_groups(1)->set_replica_ids(0, 1);
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("appears more than once")));

  proto = TestProto();
  proto.mutable_replica_groups(1)->set_replica_ids(1, -1);
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("must be nonnegative")));
}

TEST(CollectiveConfigTest, RejectsMalformedPeerRegions) {
  wire::CollectiveCallConfigV3 proto = TestProto();
  proto.mutable_peer_regions(0)->set_endpoint(
      static_cast<wire::PeerRegionEndpointProto>(9));
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Unsupported endpoint")));

  proto = TestProto();
  proto.mutable_peer_regions(0)->set_byte_size(0);
  EXPECT_THAT(Parse(proto),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid or overflowing byte range")));

  proto = TestProto();
  proto.mutable_peer_regions(0)->set_byte_offset(
      std::numeric_limits<int64_t>::max() - 1);
  proto.mutable_peer_regions(0)->set_byte_size(4);
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("overflowing byte range")));

  proto = TestProto();
  proto.mutable_peer_regions(0)->set_required_alignment(3);
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("positive power of two")));

  proto = TestProto();
  proto.mutable_peer_regions(0)->set_memory_kind(
      static_cast<wire::PeerMemoryKindProto>(2));
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Unsupported memory kind")));

  proto = TestProto();
  *proto.add_peer_regions() = proto.peer_regions(0);
  EXPECT_THAT(Parse(proto),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("duplicates an earlier record")));
}

TEST(CollectiveConfigTest, AcceptsMultimemRegion) {
  wire::CollectiveCallConfigV3 proto = TestProto();
  proto.mutable_peer_regions(0)->set_memory_kind(
      wire::PEER_MEMORY_KIND_PROTO_MULTIMEM);

  absl::StatusOr<wire::CollectiveCallConfigV3> parsed = Parse(proto);

  ASSERT_THAT(parsed, IsOk());
  EXPECT_EQ(parsed->peer_regions(0).memory_kind(),
            wire::PEER_MEMORY_KIND_PROTO_MULTIMEM);
}

TEST(CollectiveConfigTest, AllowsNoPeerRegionsAndDefaultsBarrierOff) {
  wire::CollectiveCallConfigV3 proto = TestProto();
  proto.clear_peer_regions();
  proto.clear_barrier_before_launch();

  absl::StatusOr<wire::CollectiveCallConfigV3> parsed = Parse(proto);
  ASSERT_THAT(parsed, IsOk());
  EXPECT_TRUE(parsed->peer_regions().empty());
  EXPECT_FALSE(parsed->barrier_before_launch());
}

}  // namespace
}  // namespace xla::gpu::cutedsl
