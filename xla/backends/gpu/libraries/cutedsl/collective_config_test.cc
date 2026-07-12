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

#include "xla/backends/gpu/libraries/cutedsl/collective_config.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"
#include "xla/backends/gpu/libraries/cutedsl/collective_config.pb.h"
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

std::string Sha256(absl::string_view bytes) {
  llvm::SHA256 hasher;
  hasher.update(llvm::StringRef(bytes.data(), bytes.size()));
  std::array<uint8_t, kModuleDigestSize> digest = hasher.final();
  return std::string(reinterpret_cast<const char*>(digest.data()),
                     digest.size());
}

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
  proto.set_module(std::string("collective\0module\xff", 18));

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

  wire::CollectiveStepProto* barrier = proto.add_steps();
  barrier->set_kind(wire::COLLECTIVE_STEP_KIND_PROTO_BARRIER);
  barrier->set_operand(0);
  wire::CollectiveStepProto* launch = proto.add_steps();
  launch->set_kind(wire::COLLECTIVE_STEP_KIND_PROTO_LAUNCH);
  launch->set_operand(2);
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

absl::StatusOr<CollectiveCallConfigV3> Parse(
    const wire::CollectiveCallConfigV3& proto) {
  return ParseCollectiveCallConfigV3(ToJson(proto));
}

TEST(CollectiveConfigTest, ParsesCompleteProtoJsonConfiguration) {
  wire::CollectiveCallConfigV3 proto = TestProto();
  absl::StatusOr<CollectiveCallConfigV3> parsed = Parse(proto);
  ASSERT_THAT(parsed, IsOk());

  EXPECT_EQ(parsed->group_mode,
            CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA);
  EXPECT_EQ(parsed->abi_clique_size, 2);
  EXPECT_EQ(parsed->communication_id, 17);
  ASSERT_EQ(parsed->replica_groups.size(), 2);
  EXPECT_THAT(parsed->replica_groups[0].replica_ids(), ElementsAre(0, 1));
  EXPECT_THAT(parsed->replica_groups[1].replica_ids(), ElementsAre(2, 3));

  EXPECT_EQ(parsed->module.bytes(), proto.module());
  EXPECT_EQ(parsed->module.sha256(), Sha256(proto.module()));

  ASSERT_EQ(parsed->peer_regions.size(), 2);
  EXPECT_EQ(parsed->peer_regions[0].endpoint, PeerRegionEndpointV3::kArgument);
  EXPECT_EQ(parsed->peer_regions[0].buffer_index, 2);
  EXPECT_EQ(parsed->peer_regions[0].byte_offset, 16);
  EXPECT_EQ(parsed->peer_regions[0].byte_size, 64);
  EXPECT_EQ(parsed->peer_regions[0].required_alignment, 16);
  EXPECT_EQ(parsed->peer_regions[1].endpoint, PeerRegionEndpointV3::kResult);

  ASSERT_EQ(parsed->steps.size(), 2);
  EXPECT_EQ(parsed->steps[0].kind, CollectiveStepKindV3::kBarrier);
  EXPECT_EQ(parsed->steps[0].operand, 0);
  EXPECT_EQ(parsed->steps[1].kind, CollectiveStepKindV3::kLaunch);
  EXPECT_EQ(parsed->steps[1].operand, 2);
}

TEST(CollectiveConfigTest, ParsesDkgProtoJsonEncoding) {
  constexpr absl::string_view kJson = R"json(
    {
      "abi_clique_size": "2",
      "communication_id": "17",
      "group_mode": 0,
      "module": "Y29sbGVjdGl2ZS1tb2R1bGU=",
      "peer_regions": [{
        "buffer_index": "2",
        "byte_offset": "16",
        "byte_size": "64",
        "endpoint": 0,
        "memory_kind": 0,
        "required_alignment": "16"
      }],
      "replica_groups": [{"replica_ids": ["0", "1"]}],
      "steps": [
        {"kind": 0, "operand": "0"},
        {"kind": 1, "operand": "2"}
      ]
    }
  )json";

  absl::StatusOr<CollectiveCallConfigV3> parsed =
      ParseCollectiveCallConfigV3(kJson);
  ASSERT_THAT(parsed, IsOk());
  EXPECT_EQ(parsed->abi_clique_size, 2);
  EXPECT_EQ(parsed->module.bytes(), "collective-module");
  ASSERT_EQ(parsed->peer_regions.size(), 1);
  EXPECT_EQ(parsed->peer_regions[0].buffer_index, 2);
  ASSERT_EQ(parsed->steps.size(), 2);
  EXPECT_EQ(parsed->steps[1].kind, CollectiveStepKindV3::kLaunch);
  EXPECT_EQ(parsed->steps[1].operand, 2);
}

TEST(CollectiveConfigTest, RejectsMalformedJsonAndMissingFields) {
  EXPECT_THAT(ParseCollectiveCallConfigV3("{"),
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

  proto = TestProto();
  proto.mutable_steps(0)->clear_operand();
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("steps[0].operand")));
}

TEST(CollectiveConfigTest, IgnoresUnknownJsonFields) {
  std::string json = ToJson(TestProto());
  ASSERT_EQ(json.back(), '}');
  json.pop_back();
  json.append(",\"future_field\":{\"nested\":1}}");
  EXPECT_THAT(ParseCollectiveCallConfigV3(json), IsOk());
}

TEST(CollectiveConfigTest, ValidatesAbiCliqueSizeRange) {
  wire::CollectiveCallConfigV3 proto = TestProto();
  proto.set_abi_clique_size(std::numeric_limits<int32_t>::max());
  absl::StatusOr<CollectiveCallConfigV3> parsed = Parse(proto);
  ASSERT_THAT(parsed, IsOk());
  EXPECT_EQ(parsed->abi_clique_size, std::numeric_limits<int32_t>::max());

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

TEST(CollectiveConfigTest, RejectsMalformedModule) {
  wire::CollectiveCallConfigV3 proto = TestProto();
  proto.clear_module();
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Missing")));

  proto = TestProto();
  proto.set_module("");
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("module` must not be empty")));
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
      static_cast<wire::PeerMemoryKindProto>(1));
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Unsupported memory kind")));

  proto = TestProto();
  *proto.add_peer_regions() = proto.peer_regions(0);
  EXPECT_THAT(Parse(proto),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("duplicates an earlier record")));
}

TEST(CollectiveConfigTest, AllowsNoPeerRegionsAndLaunchOnlySchedule) {
  wire::CollectiveCallConfigV3 proto = TestProto();
  proto.clear_peer_regions();
  proto.clear_steps();
  wire::CollectiveStepProto* launch = proto.add_steps();
  launch->set_kind(wire::COLLECTIVE_STEP_KIND_PROTO_LAUNCH);
  launch->set_operand(0);

  absl::StatusOr<CollectiveCallConfigV3> parsed = Parse(proto);
  ASSERT_THAT(parsed, IsOk());
  EXPECT_TRUE(parsed->peer_regions.empty());
  ASSERT_EQ(parsed->steps.size(), 1);
  EXPECT_EQ(parsed->steps[0].kind, CollectiveStepKindV3::kLaunch);
}

TEST(CollectiveConfigTest, RejectsMalformedSteps) {
  wire::CollectiveCallConfigV3 proto = TestProto();
  proto.clear_steps();
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("complete [kind, operand]")));

  proto = TestProto();
  proto.mutable_steps(0)->set_operand(1);
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("must have operand zero")));

  proto = TestProto();
  proto.mutable_steps(1)->set_operand(-1);
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("nonnegative function")));

  proto = TestProto();
  proto.mutable_steps()->SwapElements(0, 1);
  EXPECT_THAT(Parse(proto),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("requires all barriers before")));

  proto = TestProto();
  proto.mutable_steps(0)->set_kind(
      static_cast<wire::CollectiveStepKindProto>(9));
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Unsupported step kind")));

  proto = TestProto();
  proto.mutable_steps()->RemoveLast();
  EXPECT_THAT(Parse(proto), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("at least one launch")));
}

}  // namespace
}  // namespace xla::gpu::cutedsl
