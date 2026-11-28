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

#include "xla/tsl/profiler/utils/xchunk/xchunk_wire_scanner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl::profiler::xchunk {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::tensorflow::profiler::XEvent;
using ::tensorflow::profiler::XLine;
using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XSpace;
using ::testing::SizeIs;

TEST(XChunkWireScannerTest, ExtractsPlanesAndSplitsXLineEventBatches) {
  XSpace space;
  space.add_hostnames("test-host");

  XPlane* plane = space.add_planes();
  plane->set_id(7);
  plane->set_name("/host:CPU");
  (*plane->mutable_event_metadata())[1].set_id(1);
  (*plane->mutable_event_metadata())[1].set_name("ev1");
  (*plane->mutable_stat_metadata())[2].set_id(2);
  (*plane->mutable_stat_metadata())[2].set_name("st2");

  XLine* line = plane->add_lines();
  line->set_id(42);
  line->set_display_id(420);
  line->set_name("worker-thread");
  line->set_display_name("Worker Thread");
  line->set_timestamp_ns(123456);
  line->set_duration_ps(999999);
  for (int i = 0; i < 10; ++i) {
    XEvent* ev = line->add_events();
    ev->set_metadata_id(1);
    ev->set_offset_ps(i * 100);
    ev->set_duration_ps(50);
  }

  XPlane* empty_plane = space.add_planes();
  empty_plane->set_id(8);
  empty_plane->set_name("/host:metadata");

  const std::string space_wire = space.SerializeAsString();
  ASSERT_OK_AND_ASSIGN(std::vector<ScannedXPlaneWire> scanned_planes,
                       ScanXSpaceWirePlanes(space_wire, "unit-test"));
  ASSERT_THAT(scanned_planes, SizeIs(2));
  EXPECT_EQ(scanned_planes[0].name, "/host:CPU");
  ASSERT_THAT(scanned_planes[0].line_slices, SizeIs(1));
  EXPECT_EQ(scanned_planes[1].name, "/host:metadata");
  EXPECT_TRUE(scanned_planes[1].line_slices.empty());

  XPlane parsed_meta;
  ASSERT_TRUE(parsed_meta.ParseFromString(scanned_planes[0].metadata_wire));
  EXPECT_EQ(parsed_meta.id(), 7);
  EXPECT_EQ(parsed_meta.name(), "/host:CPU");
  EXPECT_EQ(parsed_meta.event_metadata_size(), 1);
  EXPECT_EQ(parsed_meta.stat_metadata_size(), 1);
  EXPECT_EQ(parsed_meta.lines_size(), 0);

  ASSERT_OK_AND_ASSIGN(
      ScannedXLineBatches line_batches,
      SplitXLineWireBatches(scanned_planes[0].line_slices[0],
                            /*first_batch_max_events=*/3,
                            /*max_events_per_batch=*/4, "unit-test"));
  // 10 events -> batches of size 3, 4, 3.
  ASSERT_THAT(line_batches.event_batch_wires, SizeIs(3));

  XLine header_only;
  ASSERT_TRUE(header_only.ParseFromString(line_batches.header_wire));
  EXPECT_EQ(header_only.id(), 42);
  EXPECT_EQ(header_only.display_id(), 420);
  EXPECT_EQ(header_only.name(), "worker-thread");
  EXPECT_EQ(header_only.display_name(), "Worker Thread");
  EXPECT_EQ(header_only.timestamp_ns(), 123456);
  EXPECT_EQ(header_only.duration_ps(), 999999);
  EXPECT_EQ(header_only.events_size(), 0);

  const std::vector<int> expected_counts = {3, 4, 3};
  int next_idx = 0;
  for (size_t b = 0; b < line_batches.event_batch_wires.size(); ++b) {
    XLine batch_line = header_only;
    ASSERT_TRUE(batch_line.MergeFromString(line_batches.event_batch_wires[b]));
    ASSERT_EQ(batch_line.events_size(), expected_counts[b]);
    for (int j = 0; j < batch_line.events_size(); ++j) {
      EXPECT_EQ(batch_line.events(j).offset_ps(), (next_idx + j) * 100);
    }
    next_idx += batch_line.events_size();
  }
}

TEST(XChunkWireScannerTest, RejectsMalformedAndTruncatedWirePayloads) {
  const std::string zero_tag("\x00", 1);
  EXPECT_THAT(ScanXSpaceWirePlanes(zero_tag),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(SplitXLineWireBatches(zero_tag, 10, 10),
              StatusIs(absl::StatusCode::kInternal));

  // Truncated length delimited XLine.events field (tag = (4 << 3) | 2 = 0x22,
  // length = 50, but only 2 bytes follow).
  constexpr absl::string_view truncated_event("\x22\x32\x01\x02", 4);
  EXPECT_THAT(SplitXLineWireBatches(truncated_event, 10, 10),
              StatusIs(absl::StatusCode::kInternal));

  // Truncated length delimited XSpace.planes field (tag = (1 << 3) | 2 = 0x0a,
  // length = 20, 1 byte follows).
  constexpr absl::string_view truncated_plane("\x0a\x14\x01", 3);
  EXPECT_THAT(ScanXSpaceWirePlanes(truncated_plane),
              StatusIs(absl::StatusCode::kInternal));

  // Empty XSpace is valid and returns zero planes.
  EXPECT_THAT(ScanXSpaceWirePlanes(""), IsOkAndHolds(SizeIs(0)));
}

}  // namespace
}  // namespace tsl::profiler::xchunk
