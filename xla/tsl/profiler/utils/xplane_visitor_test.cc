/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/tsl/profiler/utils/xplane_visitor.h"

#include <cstdint>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

TEST(XPlaneVisitorTest, GetStatTest) {
  XPlane plane;
  XPlaneBuilder xplane_builder(&plane);
  XLineBuilder xline_builder = xplane_builder.GetOrCreateLine(0);
  XEventBuilder event_builder = xline_builder.AddEvent(
      *xplane_builder.GetOrCreateEventMetadata("test_event"));

  const XStatMetadata* stat_meta =
      xplane_builder.GetOrCreateStatMetadata("test_stat");
  event_builder.AddStatValue(*stat_meta, int64_t{42});

  XPlaneVisitor xplane_visitor(&plane);
  bool event_found = false;
  xplane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
      event_found = true;
      std::optional<XStatVisitor> stat = event_visitor.GetStat(*stat_meta);
      ASSERT_TRUE(stat.has_value());
      EXPECT_EQ(stat->IntValue(), 42);

      const XStatMetadata* missing_meta =
          xplane_builder.GetOrCreateStatMetadata("missing_stat");
      std::optional<XStatVisitor> missing_stat =
          event_visitor.GetStat(*missing_meta);
      EXPECT_FALSE(missing_stat.has_value());
    });
  });
  EXPECT_TRUE(event_found);
}

TEST(XPlaneVisitorTest, GetEventOrMetadataStatTest) {
  XPlane plane;
  XPlaneBuilder xplane_builder(&plane);
  XLineBuilder xline_builder = xplane_builder.GetOrCreateLine(0);
  XEventMetadata* event_meta =
      xplane_builder.GetOrCreateEventMetadata("test_event");
  const XStatMetadata* stat_meta =
      xplane_builder.GetOrCreateStatMetadata("test_stat");
  const XStatMetadata* meta_only_stat =
      xplane_builder.GetOrCreateStatMetadata("meta_only_stat");
  const XStatMetadata* string_stat =
      xplane_builder.GetOrCreateStatMetadata("string_stat");

  XStatsBuilder<XEventMetadata> event_meta_builder(event_meta, &xplane_builder);
  event_meta_builder.AddStatValue(*stat_meta, int64_t{10});
  event_meta_builder.AddStatValue(*meta_only_stat, int64_t{30});
  event_meta_builder.AddStatValue(*string_stat, "hello");

  XEventBuilder event_builder = xline_builder.AddEvent(*event_meta);
  event_builder.AddStatValue(*stat_meta, int64_t{20});

  TypeGetter stat_type_getter =
      [](absl::string_view name) -> std::optional<int64_t> {
    if (name == "test_stat") {
      return 100;
    }
    if (name == "meta_only_stat") {
      return 200;
    }
    if (name == "string_stat") {
      return 300;
    }
    return std::nullopt;
  };
  XPlaneVisitor xplane_visitor(&plane, {}, {stat_type_getter});
  bool event_found = false;
  xplane_visitor.ForEachLine([&](const XLineVisitor& line_visitor) {
    line_visitor.ForEachEvent([&](const XEventVisitor& event_visitor) {
      event_found = true;
      std::optional<XStatVisitor> stat =
          event_visitor.GetEventOrMetadataStat(100);
      ASSERT_TRUE(stat.has_value());
      // Event stat (20) should take precedence over metadata stat (10).
      EXPECT_EQ(stat->IntValue(), 20);
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat(100, -1), 20);

      // Metadata-only stat should fall back to metadata value (30).
      std::optional<XStatVisitor> meta_stat =
          event_visitor.GetEventOrMetadataStat(200);
      ASSERT_TRUE(meta_stat.has_value());
      EXPECT_EQ(meta_stat->IntValue(), 30);
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat(200, -1), 30);

      // String stat from metadata should work with the template overload.
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat<absl::string_view>(
                    300, "default"),
                "hello");

      // Absent stat should return nullopt or default value.
      EXPECT_FALSE(event_visitor.GetEventOrMetadataStat(400).has_value());
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat(400, -1), -1);
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat<absl::string_view>(
                    400, "default"),
                "default");

      // Type mismatch should return the default value.
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat(300, -1), -1);
      EXPECT_EQ(event_visitor.GetEventOrMetadataStat<absl::string_view>(
                    100, "default"),
                "default");
    });
  });
  EXPECT_TRUE(event_found);
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
