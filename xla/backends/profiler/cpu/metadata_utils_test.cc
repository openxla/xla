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

#include "xla/backends/profiler/cpu/metadata_utils.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "xla/service/hlo.pb.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

TEST(MetadataUtilsTest, AddHloProtoUnderThresholdSucceeds) {
  tensorflow::profiler::XPlane raw_plane;
  tsl::profiler::XPlaneBuilder plane_builder(&raw_plane);
  MetadataXPlaneBuilder builder(&raw_plane);

  xla::HloProto hlo_proto;
  hlo_proto.mutable_hlo_module()->set_name("test_module");

  uint64_t program_id = 12345;
  builder.AddHloProto(program_id, hlo_proto);

  // Verify that the event was added correctly.
  EXPECT_EQ(raw_plane.event_metadata_size(), 1);

  const auto& event_metadata = raw_plane.event_metadata().begin()->second;
  EXPECT_EQ(event_metadata.display_name(), "test_module(12345)");

  // Verify that the HLO proto stat was recorded.
  EXPECT_GT(event_metadata.stats_size(), 0);
}

}  // namespace
}  // namespace profiler
}  // namespace xla
