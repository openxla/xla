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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/invoke.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla::gpu::cutedsl {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

std::string Sha256(absl::string_view bytes) {
  llvm::SHA256 hasher;
  hasher.update(llvm::StringRef(bytes.data(), bytes.size()));
  std::array<uint8_t, kCollectiveModuleDigestSizeV3> digest = hasher.final();
  return std::string(reinterpret_cast<const char*>(digest.data()),
                     digest.size());
}

struct TestAttributes {
  TestAttributes() {
    constexpr absl::string_view kModule0 = "module_zero";
    constexpr absl::string_view kModule1 = "module_one";
    module_blob = std::string(kModule0) + std::string(kModule1);
    module_offsets = {0, static_cast<int64_t>(kModule0.size()),
                      static_cast<int64_t>(module_blob.size())};
    module_keys = Sha256(kModule0) + Sha256(kModule1);
  }

  ffi::AttributesMap Build() const {
    ffi::CallFrameBuilder::AttributesBuilder attributes;
    if (schema_version_as_i32) {
      attributes.Insert("schema_version", static_cast<int32_t>(schema_version));
    } else {
      attributes.Insert("schema_version", schema_version);
    }
    attributes.Insert("group_mode", group_mode);
    attributes.Insert("communication_id", communication_id);
    attributes.Insert("replica_group_offsets", replica_group_offsets);
    attributes.Insert("replica_group_members", replica_group_members);
    attributes.Insert("module_blob", module_blob);
    attributes.Insert("module_offsets", module_offsets);
    attributes.Insert("module_keys", module_keys);
    attributes.Insert("module_index_by_rank", module_index_by_rank);
    attributes.Insert("peer_regions", peer_regions);
    if (!omit_steps) attributes.Insert("steps", steps);
    if (add_semantic_attribute) {
      attributes.Insert("buffer_role", int64_t{0});
    }
    return attributes.Build();
  }

  int64_t schema_version = kCollectiveCallSchemaVersionV3;
  int64_t group_mode =
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
  int64_t communication_id = 17;
  std::vector<int64_t> replica_group_offsets = {0, 2, 4};
  std::vector<int64_t> replica_group_members = {0, 1, 2, 3};
  std::string module_blob;
  std::vector<int64_t> module_offsets;
  std::string module_keys;
  std::vector<int64_t> module_index_by_rank = {0, 1};
  std::vector<int64_t> peer_regions = {
      static_cast<int64_t>(PeerRegionEndpointV3::kArgument),
      2,
      16,
      64,
      16,
      static_cast<int64_t>(PeerMemoryKindV3::kSymmetric),
      static_cast<int64_t>(PeerRegionEndpointV3::kResult),
      0,
      0,
      128,
      64,
      static_cast<int64_t>(PeerMemoryKindV3::kSymmetric),
  };
  std::vector<int64_t> steps = {
      static_cast<int64_t>(CollectiveStepKindV3::kBarrier),
      0,
      static_cast<int64_t>(CollectiveStepKindV3::kLaunch),
      2,
  };
  bool omit_steps = false;
  bool add_semantic_attribute = false;
  bool schema_version_as_i32 = false;
};

absl::StatusOr<CollectiveCallConfigV3> Parse(TestAttributes attributes) {
  ffi::CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attributes.Build());
  ffi::CallFrame call_frame = builder.Build();

  std::optional<CollectiveCallConfigV3> parsed;
  auto parse = [&](ffi::Dictionary dictionary) -> absl::Status {
    absl::StatusOr<CollectiveCallConfigV3> result =
        ParseCollectiveCallConfigV3(dictionary);
    if (!result.ok()) return result.status();
    parsed.emplace(std::move(*result));
    return absl::OkStatus();
  };
  auto handler = ffi::Ffi::Bind().Attrs().To(parse);
  absl::Status status = ffi::Invoke(ffi::GetXlaFfiApi(), *handler, call_frame);
  if (!status.ok()) return status;
  if (!parsed.has_value()) {
    return absl::InternalError("Test parser handler returned no configuration");
  }
  return std::move(*parsed);
}

TEST(CollectiveConfigTest, ParsesCompleteGenericConfiguration) {
  absl::StatusOr<CollectiveCallConfigV3> parsed = Parse(TestAttributes());
  ASSERT_THAT(parsed, IsOk());

  EXPECT_EQ(parsed->group_mode,
            CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA);
  EXPECT_EQ(parsed->communication_id, 17);
  ASSERT_EQ(parsed->replica_groups.size(), 2);
  EXPECT_THAT(parsed->replica_groups[0].replica_ids(), ElementsAre(0, 1));
  EXPECT_THAT(parsed->replica_groups[1].replica_ids(), ElementsAre(2, 3));

  ASSERT_EQ(parsed->modules.size(), 2);
  EXPECT_EQ(parsed->modules[0].bytes, "module_zero");
  EXPECT_EQ(parsed->modules[1].bytes, "module_one");
  EXPECT_EQ(std::string(
                reinterpret_cast<const char*>(parsed->modules[0].sha256.data()),
                parsed->modules[0].sha256.size()),
            Sha256("module_zero"));
  EXPECT_THAT(parsed->module_index_by_rank, ElementsAre(0, 1));

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

TEST(CollectiveConfigTest, RejectsWrongSchemaAndAttributeShape) {
  TestAttributes attributes;
  attributes.schema_version = 2;
  EXPECT_THAT(Parse(attributes), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("schema version 2")));

  attributes = TestAttributes();
  attributes.schema_version_as_i32 = true;
  EXPECT_THAT(Parse(attributes),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("attribute `schema_version`")));

  attributes = TestAttributes();
  attributes.omit_steps = true;
  EXPECT_THAT(Parse(attributes),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Missing CuTeDSL collective v3 attribute "
                                 "`steps`")));

  attributes = TestAttributes();
  attributes.add_semantic_attribute = true;
  EXPECT_THAT(Parse(attributes),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unknown CuTeDSL collective v3 attribute "
                                 "`buffer_role`")));
}

TEST(CollectiveConfigTest, RejectsMalformedCollectiveGroup) {
  TestAttributes attributes;
  attributes.group_mode = 9;
  EXPECT_THAT(Parse(attributes),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported collective group mode")));

  attributes = TestAttributes();
  attributes.communication_id = -1;
  EXPECT_THAT(Parse(attributes), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("communication_id")));

  attributes = TestAttributes();
  attributes.replica_group_offsets = {1, 4};
  EXPECT_THAT(Parse(attributes), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("must start with zero")));

  attributes = TestAttributes();
  attributes.replica_group_offsets = {0, 1, 4};
  EXPECT_THAT(Parse(attributes), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("equal cardinality")));

  attributes = TestAttributes();
  attributes.replica_group_members = {0, 1, 1, 3};
  EXPECT_THAT(Parse(attributes), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("appears more than once")));

  attributes = TestAttributes();
  attributes.replica_group_members = {0, 1, 2, -1};
  EXPECT_THAT(Parse(attributes), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("must be nonnegative")));
}

TEST(CollectiveConfigTest, RejectsMalformedModuleBundleOrRankMap) {
  TestAttributes attributes;
  attributes.module_offsets = {0, 0, 21};
  EXPECT_THAT(Parse(attributes),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid or empty module range")));

  attributes = TestAttributes();
  attributes.module_keys.pop_back();
  EXPECT_THAT(Parse(attributes),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("one 32-byte SHA-256 digest per module")));

  attributes = TestAttributes();
  attributes.module_keys[0] ^= 1;
  EXPECT_THAT(Parse(attributes),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not match module image 0")));

  attributes = TestAttributes();
  attributes.module_index_by_rank.clear();
  EXPECT_THAT(Parse(attributes),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("module_index_by_rank` must not be empty")));

  attributes = TestAttributes();
  attributes.module_index_by_rank = {0, 2};
  EXPECT_THAT(Parse(attributes), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("out of range")));
}

TEST(CollectiveConfigTest, DefersRankMapLengthValidationToPrepare) {
  TestAttributes attributes;
  attributes.group_mode = CollectiveOpGroupMode::
      COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION;
  attributes.module_index_by_rank = {0, 1, 0, 1};
  EXPECT_THAT(Parse(attributes), IsOk());
}

TEST(CollectiveConfigTest, RejectsMalformedPeerRegions) {
  TestAttributes attributes;
  attributes.peer_regions.pop_back();
  EXPECT_THAT(Parse(attributes), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("multiple of 6")));

  attributes = TestAttributes();
  attributes.peer_regions[0] = 9;
  EXPECT_THAT(Parse(attributes), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("Unsupported endpoint")));

  attributes = TestAttributes();
  attributes.peer_regions[3] = 0;
  EXPECT_THAT(Parse(attributes),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid or overflowing byte range")));

  attributes = TestAttributes();
  attributes.peer_regions[2] = std::numeric_limits<int64_t>::max() - 1;
  attributes.peer_regions[3] = 4;
  EXPECT_THAT(Parse(attributes), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("overflowing byte range")));

  attributes = TestAttributes();
  attributes.peer_regions[4] = 3;
  EXPECT_THAT(Parse(attributes), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("positive power of two")));

  attributes = TestAttributes();
  attributes.peer_regions[5] = 1;
  EXPECT_THAT(Parse(attributes),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported memory kind")));

  attributes = TestAttributes();
  std::vector<int64_t> duplicate(attributes.peer_regions.begin(),
                                 attributes.peer_regions.begin() + 6);
  attributes.peer_regions.insert(attributes.peer_regions.end(),
                                 duplicate.begin(), duplicate.end());
  EXPECT_THAT(Parse(attributes),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("duplicates an earlier record")));
}

TEST(CollectiveConfigTest, AllowsNoPeerRegions) {
  TestAttributes attributes;
  attributes.peer_regions.clear();
  absl::StatusOr<CollectiveCallConfigV3> parsed = Parse(attributes);
  ASSERT_THAT(parsed, IsOk());
  EXPECT_TRUE(parsed->peer_regions.empty());
}

TEST(CollectiveConfigTest, AllowsLaunchOnlySchedule) {
  TestAttributes attributes;
  attributes.steps = {
      static_cast<int64_t>(CollectiveStepKindV3::kLaunch),
      0,
  };
  absl::StatusOr<CollectiveCallConfigV3> parsed = Parse(attributes);
  ASSERT_THAT(parsed, IsOk());
  ASSERT_EQ(parsed->steps.size(), 1);
  EXPECT_EQ(parsed->steps[0].kind, CollectiveStepKindV3::kLaunch);
}

TEST(CollectiveConfigTest, RejectsMalformedStepList) {
  TestAttributes attributes;
  attributes.steps.clear();
  EXPECT_THAT(Parse(attributes),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("complete [kind, operand] records")));

  attributes = TestAttributes();
  attributes.steps = {
      static_cast<int64_t>(CollectiveStepKindV3::kBarrier),
      1,
      static_cast<int64_t>(CollectiveStepKindV3::kLaunch),
      0,
  };
  EXPECT_THAT(Parse(attributes), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("must have operand zero")));

  attributes = TestAttributes();
  attributes.steps = {
      static_cast<int64_t>(CollectiveStepKindV3::kLaunch),
      -1,
  };
  EXPECT_THAT(Parse(attributes),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("nonnegative function ordinal")));

  attributes = TestAttributes();
  attributes.steps = {
      static_cast<int64_t>(CollectiveStepKindV3::kLaunch),
      0,
      static_cast<int64_t>(CollectiveStepKindV3::kBarrier),
      0,
  };
  EXPECT_THAT(Parse(attributes),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("requires all barriers before")));

  attributes = TestAttributes();
  attributes.steps = {9, 0};
  EXPECT_THAT(Parse(attributes), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("Unsupported step kind")));

  attributes = TestAttributes();
  attributes.steps = {
      static_cast<int64_t>(CollectiveStepKindV3::kBarrier),
      0,
  };
  EXPECT_THAT(Parse(attributes), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("at least one launch")));
}

}  // namespace
}  // namespace xla::gpu::cutedsl
