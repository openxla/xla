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

#include "xla/tsl/profiler/utils/xchunk/xchunk.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xchunk/xchunk_visitor.h"
#include "xla/tsl/profiler/utils/xchunk/xplane_to_xchunk.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"

namespace tsl::profiler::xchunk {
namespace {

using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XSpace;
using ::testing::ElementsAre;
using ::testing::SizeIs;

constexpr int kHostId = 3;

XSpace BuildTestXSpace() {
  XSpace space;
  XPlaneBuilder plane(space.add_planes());
  plane.SetId(2);
  plane.SetName(absl::StrCat(kTpuPlanePrefix, 2));
  XStatMetadata* flow_stat =
      plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kFlow));
  XStatMetadata* group_stat =
      plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId));
  XStatMetadata* step_name_stat =
      plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kStepName));
  XStatMetadata* bytes_stat = plane.GetOrCreateStatMetadata("bytes_accessed");

  XLineBuilder ops = plane.GetOrCreateLine(/*line_id=*/10);
  ops.SetName("XLA Ops");
  ops.SetDisplayId(7);
  ops.SetTimestampNs(1000);
  XEventMetadata* fusion = plane.GetOrCreateEventMetadata("fusion.1");
  fusion->set_display_name("%fusion.1 = add(...)");
  XEventMetadata* copy = plane.GetOrCreateEventMetadata("copy.2");
  XEventMetadata* step = plane.GetOrCreateEventMetadata("step");
  // Added out of order on purpose: the producer must sort by offset.
  {
    XEventBuilder event = ops.AddEvent(*copy);
    event.SetOffsetPs(300);
    event.SetDurationPs(10);
    event.AddStatValue(*flow_stat, XFlow(42, XFlow::kFlowIn).ToStatValue());
  }
  {
    XEventBuilder event = ops.AddEvent(*fusion);
    event.SetOffsetPs(100);
    event.SetDurationPs(50);
    event.AddStatValue(*flow_stat, XFlow(42, XFlow::kFlowOut).ToStatValue());
    event.AddStatValue(*group_stat, int64_t{5});
    event.AddStatValue(*bytes_stat, int64_t{1024});
  }
  {
    XEventBuilder event = ops.AddEvent(*step);
    event.SetOffsetPs(200);
    event.SetDurationPs(20);
    event.AddStatValue(*step_name_stat, "train step 5");
  }

  XLineBuilder counters = plane.GetOrCreateLine(/*line_id=*/11);
  counters.SetName(kCounterEventsLineName);
  counters.SetTimestampNs(1000);
  XEventMetadata* memory = plane.GetOrCreateEventMetadata("memory_usage");
  XStatMetadata* value_stat = plane.GetOrCreateStatMetadata("value");
  for (int i = 0; i < 3; ++i) {
    XEventBuilder event = counters.AddEvent(*memory);
    event.SetOffsetPs(100 * i);
    event.AddStatValue(*value_stat, 1.5 * i);
  }
  // Counter event with two arguments: disqualified as a single value counter.
  {
    XEventBuilder event = counters.AddEvent(*memory);
    event.SetOffsetPs(1000);
    event.AddStatValue(*value_stat, 9.0);
    event.AddStatValue(*bytes_stat, int64_t{1});
  }

  // An empty line is not announced at all.
  plane.GetOrCreateLine(/*line_id=*/12).SetName("empty");
  return space;
}

TEST(XChunkTest, StreamsXSpaceAsChunksInOrder) {
  const XSpace space = BuildTestXSpace();
  XChunkOptions options;
  options.max_events_per_chunk = 2;
  std::vector<XChunk> chunks;
  StreamXSpaceToXChunks(space, "myhost", kHostId, options, [&](XChunk&& chunk) {
    chunks.push_back(std::move(chunk));
  });

  ASSERT_THAT(chunks, SizeIs(7));
  EXPECT_EQ(chunks[0].stream_header().hostname(), "myhost");
  EXPECT_EQ(chunks[0].stream_header().host_id(), kHostId);
  EXPECT_EQ(chunks[0].stream_header().device_family(), XChunkStreamHeader::TPU);

  const XChunkPlane& plane = chunks[1].plane();
  EXPECT_EQ(plane.plane_index(), 0);
  EXPECT_EQ(plane.plane_id(), 2);
  EXPECT_EQ(plane.process_id(),
            kHostId * kMaxDevicesPerHost + kFirstDeviceId + 2);
  EXPECT_EQ(plane.process_name(), "myhost /device:TPU:2");
  ASSERT_THAT(plane.lines(), SizeIs(2));  // the empty line is dropped
  EXPECT_EQ(plane.lines(0).line_id(), 10);
  EXPECT_EQ(plane.lines(0).display_id(), 7);
  EXPECT_EQ(plane.lines(0).name(), "XLA Ops");
  EXPECT_EQ(plane.lines(0).num_events(), 3);
  EXPECT_EQ(plane.lines(1).name(), kCounterEventsLineName);
  XChunkPlaneVisitor plane_visitor(&plane);
  EXPECT_EQ(plane_visitor.NumEventMetadata(), 4);
  EXPECT_EQ(plane_visitor.NumStatMetadata(), 0);  // include_stats = false

  // Line 10: 3 events -> 2 chunks, sorted by offset.
  const XChunkEvents& ops0 = chunks[2].events();
  EXPECT_EQ(ops0.line_id(), 10);
  EXPECT_EQ(ops0.chunk_index(), 0);
  EXPECT_FALSE(ops0.last_chunk());
  EXPECT_THAT(ops0.offset_ps_deltas(), ElementsAre(100, 100));
  EXPECT_THAT(ops0.duration_ps(), ElementsAre(50, 20));
  const XChunkEvents& ops1 = chunks[3].events();
  EXPECT_EQ(ops1.chunk_index(), 1);
  EXPECT_TRUE(ops1.last_chunk());
  EXPECT_THAT(ops1.offset_ps_deltas(), ElementsAre(300));

  std::vector<std::string> decoded;
  std::optional<XChunkLineVisitor> line = plane_visitor.GetLine(10);
  ASSERT_TRUE(line.has_value());
  for (const XChunkEvents* chunk : {&ops0, &ops1}) {
    ForEachEvent(
        plane_visitor, *line, *chunk, [&](const XChunkEventVisitor& event) {
          decoded.push_back(absl::StrFormat(
              "%s@%d+%d flow=%s group=%s", event.TraceViewerName(),
              event.TimestampPs(), event.DurationPs(),
              event.flow
                  ? absl::StrCat(event.flow->Id(), "/", event.flow->Direction())
                  : "-",
              event.group_id ? absl::StrCat(*event.group_id) : "-"));
        });
  }
  EXPECT_THAT(decoded,
              ElementsAre("%fusion.1 = add(...)@1000100+50 flow=42/2 group=5",
                          "train step 5@1000200+20 flow=- group=-",
                          "copy.2@1000300+10 flow=42/1 group=-"));

  // Counter line: 4 events in 2 chunks, 3 of them with a counter value (the
  // last one has two arguments, which disqualifies it as a counter).
  const XChunkEvents& counters0 = chunks[4].events();
  EXPECT_EQ(counters0.line_id(), 11);
  EXPECT_FALSE(counters0.last_chunk());
  EXPECT_EQ(counters0.num_events(), 2);
  EXPECT_THAT(counters0.counter_double().event_index(), ElementsAre(0, 1));
  EXPECT_THAT(counters0.counter_double().values(), ElementsAre(0.0, 1.5));
  const XChunkEvents& counters1 = chunks[5].events();
  EXPECT_TRUE(counters1.last_chunk());
  EXPECT_EQ(counters1.num_events(), 2);
  EXPECT_THAT(counters1.offset_ps_deltas(), ElementsAre(200, 800));
  EXPECT_THAT(counters1.counter_double().event_index(), ElementsAre(0));
  EXPECT_THAT(counters1.counter_double().values(), ElementsAre(3.0));

  EXPECT_EQ(chunks[6].end().num_events(), 7);
  EXPECT_EQ(chunks[6].end().num_planes(), 1);
}

TEST(XChunkTest, IncludesGenericStatColumnsWhenEnabled) {
  XSpace space = BuildTestXSpace();
  // Attach a metadata level generic stat and an event stat with an unknown
  // metadata_id to verify metadata stats are emitted and unknown stat IDs are
  // ignored.
  XPlane* raw_plane = space.mutable_planes(0);
  XPlaneBuilder builder(raw_plane);
  XStatMetadata* hlo_cat_stat = builder.GetOrCreateStatMetadata("hlo_category");
  XEventMetadata* copy_meta = builder.GetOrCreateEventMetadata("copy.2");
  tensorflow::profiler::XStat* meta_stat = copy_meta->add_stats();
  meta_stat->set_metadata_id(hlo_cat_stat->id());
  meta_stat->set_str_value("data_formatting");
  tensorflow::profiler::XStat* unknown_stat =
      raw_plane->mutable_lines(0)->mutable_events(0)->add_stats();
  unknown_stat->set_metadata_id(999999);
  unknown_stat->set_int64_value(42);

  XChunkOptions options;
  options.include_stats = true;
  std::vector<XChunk> chunks;
  StreamXSpaceToXChunks(space, "myhost", kHostId, options, [&](XChunk&& chunk) {
    chunks.push_back(std::move(chunk));
  });

  ASSERT_THAT(chunks, SizeIs(5));
  const XChunkPlane& plane = chunks[1].plane();
  XChunkPlaneVisitor plane_visitor(&plane);
  EXPECT_GT(plane_visitor.NumStatMetadata(), 0);

  const XChunkEvents& ops = chunks[2].events();
  ASSERT_EQ(ops.stats_size(), 3);
  EXPECT_EQ(plane_visitor.StatName(ops.stats(0).stat_metadata_index()),
            "bytes_accessed");
  EXPECT_THAT(ops.stats(0).event_index(), ElementsAre(0));
  EXPECT_THAT(ops.stats(0).int64_values(), ElementsAre(1024));
  EXPECT_EQ(plane_visitor.StatName(ops.stats(2).stat_metadata_index()),
            "hlo_category");
  EXPECT_THAT(ops.stats(2).event_index(), ElementsAre(2));
  EXPECT_THAT(ops.stats(2).str_values(), ElementsAre("data_formatting"));
}

TEST(XChunkTest, StreamingPlaneChunkerEmitsLineSlicesIncrementally) {
  XPlane metadata_plane;
  XPlaneBuilder meta_builder(&metadata_plane);
  meta_builder.SetId(0);
  meta_builder.SetName(kHostThreadsPlaneName);
  XEventMetadata* launch_meta =
      meta_builder.GetOrCreateEventMetadata("hipLaunchKernel");
  XStatMetadata* corr_meta = meta_builder.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kCorrelationId));

  std::vector<XChunk> chunks;
  XChunkOptions options;
  options.max_events_per_chunk = 1;
  StreamingPlaneChunker chunker(
      metadata_plane, /*plane_index=*/0, /*process_id=*/701, "myhost /host:CPU",
      kHostId, options,
      [&](XChunk&& chunk) { chunks.push_back(std::move(chunk)); });

  // Slice 1 of line 42 (2 events -> 2 chunks with max_events_per_chunk = 1,
  // is_last_slice = false):
  XPlane slice1_plane;
  XPlaneBuilder slice1_builder(&slice1_plane);
  XLineBuilder line1 = slice1_builder.GetOrCreateLine(/*line_id=*/42);
  line1.SetName("worker-0");
  line1.SetTimestampNs(5000);
  {
    XEventBuilder ev = line1.AddEvent(*launch_meta);
    ev.SetOffsetPs(100);
    ev.SetDurationPs(30);
    ev.AddStatValue(*corr_meta, uint64_t{99});
  }
  {
    XEventBuilder ev = line1.AddEvent(*launch_meta);
    ev.SetOffsetPs(200);
    ev.SetDurationPs(20);
    ev.AddStatValue(*corr_meta, uint64_t{98});
  }
  chunker.ChunkLine(
      *slice1_plane.mutable_lines(0), /*start_time_ns=*/4000,
      [&](XChunk&& chunk) { chunks.push_back(std::move(chunk)); },
      /*is_last_slice=*/false);

  // Slice 2 of line 42 (is_last_slice = true):
  XPlane slice2_plane;
  XPlaneBuilder slice2_builder(&slice2_plane);
  XLineBuilder line2 = slice2_builder.GetOrCreateLine(/*line_id=*/42);
  {
    XEventBuilder ev = line2.AddEvent(*launch_meta);
    ev.SetOffsetPs(400);
    ev.SetDurationPs(25);
    ev.AddStatValue(*corr_meta, int64_t{100});
  }
  chunker.ChunkLine(
      *slice2_plane.mutable_lines(0), /*start_time_ns=*/4000,
      [&](XChunk&& chunk) { chunks.push_back(std::move(chunk)); },
      /*is_last_slice=*/true);

  ASSERT_THAT(chunks, SizeIs(4));
  EXPECT_TRUE(chunks[0].has_plane());
  EXPECT_EQ(chunks[1].events().chunk_index(), 0);
  EXPECT_FALSE(chunks[1].events().last_chunk());
  ASSERT_TRUE(chunks[1].events().line_info.has_value());
  EXPECT_EQ(chunks[1].events().line_info->timestamp_ns(), 1000);
  EXPECT_THAT(chunks[1].events().flow().event_index(), ElementsAre(0));
  EXPECT_EQ(chunks[2].events().chunk_index(), 1);
  EXPECT_FALSE(chunks[2].events().last_chunk());
  ASSERT_TRUE(chunks[2].events().line_info.has_value());
  EXPECT_EQ(chunks[2].events().line_info->name(), "worker-0");
  EXPECT_EQ(chunks[3].events().chunk_index(), 2);
  EXPECT_TRUE(chunks[3].events().last_chunk());
  ASSERT_TRUE(chunks[3].events().line_info.has_value());
  EXPECT_THAT(chunks[3].events().flow().event_index(), ElementsAre(0));
  EXPECT_EQ(chunker.stats().num_events, 3);
}

TEST(XChunkTest, VisitorMatchesXPlaneVisitorEventByEvent) {
  XSpace space;
  XPlaneBuilder builder(space.add_planes());
  builder.SetId(0);
  builder.SetName("/device:TPU:0");
  XStatMetadata* group_stat =
      builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId));
  XLineBuilder line_builder = builder.GetOrCreateLine(/*line_id=*/1);
  line_builder.SetName("XLA Ops");
  line_builder.SetTimestampNs(2000);
  for (int i = 0; i < 5; ++i) {
    XEventMetadata* meta =
        builder.GetOrCreateEventMetadata(absl::StrCat("op_", i));
    XEventBuilder ev = line_builder.AddEvent(*meta);
    ev.SetOffsetPs(i * 1000);
    ev.SetDurationPs(500 - i * 10);
    ev.AddStatValue(*group_stat, int64_t{i + 100});
  }

  // 1. Collect events via XPlaneVisitor.
  std::vector<std::string> expected;
  XPlaneVisitor xplane_visitor = CreateTfXPlaneVisitor(&space.planes(0));
  xplane_visitor.ForEachLine([&](const XLineVisitor& xline) {
    xline.ForEachEvent([&](const XEventVisitor& xevent) {
      std::optional<XStatVisitor> group = xevent.GetStat(StatType::kGroupId);
      expected.push_back(absl::StrFormat(
          "line=%d name=%s ts=%d dur=%d group=%d", xline.Id(), xevent.Name(),
          xevent.TimestampPs(), xevent.DurationPs(),
          group.has_value() ? group->IntValue() : -1));
    });
  });

  // 2. Collect events via XChunkPlaneVisitor + ForEachEvent.
  std::vector<std::string> actual;
  std::optional<XChunkPlane> saved_plane;
  std::optional<XChunkPlaneVisitor> chunk_plane_visitor;
  XChunkOptions options;
  options.max_events_per_chunk = 2;
  StreamXSpaceToXChunks(
      space, "host0", /*host_id=*/0, options, [&](XChunk&& chunk) {
        if (chunk.has_plane()) {
          saved_plane = chunk.plane();
          chunk_plane_visitor.emplace(&*saved_plane);
        } else if (chunk.has_events()) {
          const XChunkEvents& ev_chunk = chunk.events();
          std::optional<XChunkLineVisitor> xline =
              chunk_plane_visitor->GetLine(ev_chunk.line_id());
          ForEachEvent(
              *chunk_plane_visitor, *xline, ev_chunk,
              [&](const XChunkEventVisitor& xevent) {
                actual.push_back(absl::StrFormat(
                    "line=%d name=%s ts=%d dur=%d "
                    "group=%d",
                    xline->Id(), xevent.Name(), xevent.TimestampPs(),
                    xevent.DurationPs(), xevent.group_id.value_or(-1)));
              });
        }
      });

  ASSERT_TRUE(chunk_plane_visitor.has_value());
  EXPECT_EQ(chunk_plane_visitor->EventDisplayName(999), "");
  EXPECT_EQ(actual, expected);
}

TEST(XChunkTest, CoversAsyncCountersAndProducerConsumerHostFlows) {
  XPlane plane;
  XPlaneBuilder builder(&plane);
  builder.SetId(0);
  builder.SetName(kHostThreadsPlaneName);
  auto* async_meta =
      builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kIsAsync));
  auto* prod_type_meta =
      builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProducerType));
  auto* prod_id_meta =
      builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProducerId));
  auto* cons_type_meta =
      builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kConsumerType));
  auto* cons_id_meta =
      builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kConsumerId));
  auto* u64_meta = builder.GetOrCreateStatMetadata("u64_stat");
  auto* dbl_meta = builder.GetOrCreateStatMetadata("dbl_stat");
  auto* ev_meta1 = builder.GetOrCreateEventMetadata("producer_ev");
  auto* ev_meta2 = builder.GetOrCreateEventMetadata("consumer_ev");
  auto* ev_meta3 = builder.GetOrCreateEventMetadata("two_numeric_args_ev");

  XLineBuilder line = builder.GetOrCreateLine(/*line_id=*/7);
  line.SetName(kCounterEventsLineName);
  {
    XEventBuilder ev = line.AddEvent(*ev_meta1);
    ev.SetOffsetPs(100);
    ev.SetDurationPs(10);
    ev.AddStatValue(*async_meta, int64_t{1});
    ev.AddStatValue(*prod_type_meta, uint64_t{1});
    ev.AddStatValue(*prod_id_meta, uint64_t{42});
    ev.AddStatValue(*u64_meta, uint64_t{777});
  }
  {
    XEventBuilder ev = line.AddEvent(*ev_meta2);
    ev.SetOffsetPs(200);
    ev.SetDurationPs(10);
    ev.AddStatValue(*cons_type_meta, uint64_t{1});
    ev.AddStatValue(*cons_id_meta, uint64_t{42});
    ev.AddStatValue(*dbl_meta, 3.5);
  }
  {
    XEventBuilder ev = line.AddEvent(*ev_meta3);
    ev.SetOffsetPs(300);
    ev.SetDurationPs(10);
    ev.AddStatValue(*u64_meta, uint64_t{1});
    ev.AddStatValue(*dbl_meta, 2.0);
  }

  XChunkOptions options;
  options.include_stats = true;
  std::optional<XChunkPlane> saved_plane;
  std::optional<XChunkPlaneVisitor> plane_visitor;
  std::vector<bool> async_flags;
  std::vector<std::optional<uint64_t>> u64_counters;
  std::vector<std::optional<double>> dbl_counters;
  std::vector<bool> has_flows;

  auto sink = [&](XChunk&& chunk) {
    if (chunk.has_plane()) {
      saved_plane = chunk.plane();
      plane_visitor.emplace(&*saved_plane);
    } else if (chunk.has_events()) {
      const XChunkEvents& ev_chunk = chunk.events();
      XChunkLineVisitor xline(&*ev_chunk.line_info);
      ForEachEvent(*plane_visitor, xline, ev_chunk,
                   [&](const XChunkEventVisitor& xevent) {
                     async_flags.push_back(xevent.is_async);
                     u64_counters.push_back(xevent.counter_uint64);
                     dbl_counters.push_back(xevent.counter_double);
                     has_flows.push_back(xevent.flow.has_value());
                   });
    }
  };
  StreamingPlaneChunker chunker(plane, /*plane_index=*/0, /*process_id=*/701,
                                "host0 /host:CPU", /*host_id=*/0, options,
                                sink);
  chunker.ChunkLine(plane.lines(0), /*start_time_ns=*/0, sink);

  EXPECT_THAT(async_flags, ElementsAre(true, false, false));
  EXPECT_THAT(u64_counters,
              ElementsAre(uint64_t{777}, std::nullopt, std::nullopt));
  EXPECT_THAT(dbl_counters, ElementsAre(std::nullopt, 3.5, std::nullopt));
  EXPECT_THAT(has_flows, ElementsAre(true, true, false));
}

TEST(XChunkTest, XPlaneToXChunkToXPlaneRoundTripPreservesStatsAndMetadata) {
  XPlane src_plane;
  XPlaneBuilder builder(&src_plane);
  builder.SetId(5);
  builder.SetName("/device:TPU:5");

  // Explicitly create a stat_metadata entry with ID 0 to verify ref_value = 0
  // is preserved as a kRefValue rather than falling back to kStrValue.
  XStatMetadata& ref_zero_meta = (*src_plane.mutable_stat_metadata())[0];
  ref_zero_meta.set_id(0);
  ref_zero_meta.set_name("zero_ref_target");

  auto* flow_meta =
      builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kFlow));
  auto* group_meta =
      builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId));
  auto* async_meta =
      builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kIsAsync));
  auto* step_name_meta =
      builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kStepName));
  auto* ref_stat_meta = builder.GetOrCreateStatMetadata("ref_stat");
  auto* ref_target_meta = builder.GetOrCreateStatMetadata("nonzero_ref_target");
  auto* bytes_stat_meta = builder.GetOrCreateStatMetadata("proto_bytes_stat");
  auto* str_stat_meta = builder.GetOrCreateStatMetadata("plain_str_stat");
  auto* int_stat_meta = builder.GetOrCreateStatMetadata("int_stat");
  auto* uint_stat_meta = builder.GetOrCreateStatMetadata("uint_stat");
  auto* dbl_stat_meta = builder.GetOrCreateStatMetadata("dbl_stat");

  auto* ev_meta = builder.GetOrCreateEventMetadata("op_with_all_stats");
  ev_meta->set_display_name("op_display");

  XLineBuilder line = builder.GetOrCreateLine(/*line_id=*/20);
  line.SetName("XLA Ops");
  line.SetDisplayId(20);
  line.SetTimestampNs(500);
  {
    XEventBuilder ev = line.AddEvent(*ev_meta);
    ev.SetOffsetPs(1000);
    ev.SetDurationPs(250);
    ev.AddStatValue(*flow_meta, XFlow(7, XFlow::kFlowOut).ToStatValue());
    ev.AddStatValue(*group_meta, int64_t{9});
    ev.AddStatValue(*async_meta, int64_t{1});
    ev.AddStatValue(*step_name_meta, "override_step");
    ev.AddStatValue(*ref_stat_meta, ref_zero_meta);
    ev.AddStatValue(*str_stat_meta, "hello");
    ev.AddStatValue(*int_stat_meta, int64_t{-42});
    ev.AddStatValue(*uint_stat_meta, uint64_t{4200});
    ev.AddStatValue(*dbl_stat_meta, 3.14);
    tensorflow::profiler::XStat* bytes_st =
        src_plane.mutable_lines(0)->mutable_events(0)->add_stats();
    bytes_st->set_metadata_id(bytes_stat_meta->id());
    bytes_st->set_bytes_value(std::string("\x01\x00\x02\xff", 4));
  }
  {
    XEventBuilder ev = line.AddEvent(*ev_meta);
    ev.SetOffsetPs(2000);
    ev.SetDurationPs(100);
    ev.AddStatValue(*ref_stat_meta, *ref_target_meta);
  }

  XChunkOptions options;
  options.include_stats = true;
  options.max_events_per_chunk = 1;

  XPlane dst_plane;
  // Pre populate an existing event_metadata at ID 1 to ensure incremental
  // metadata consumption does not overwrite existing metadata entries.
  XEventMetadata& pre_meta = (*dst_plane.mutable_event_metadata())[1];
  pre_meta.set_id(1);
  pre_meta.set_name("pre_existing_event");

  XChunkToXPlaneConverter converter(&dst_plane);
  ChunkXPlane(src_plane, /*plane_index=*/0, /*process_id=*/1, "proc", options,
              [&](XChunk&& chunk) { converter.Consume(std::move(chunk)); });

  // Append an incremental XChunkEvents carrying new_event_metadata_ids_vec and
  // new_stat_metadata_ids_vec.
  XChunk inc_chunk;
  XChunkEvents* inc_evs = inc_chunk.mutable_events();
  inc_evs->set_plane_index(0);
  inc_evs->set_line_id(20);
  inc_evs->set_chunk_index(2);
  inc_evs->set_last_chunk(true);
  inc_evs->set_num_events(1);
  inc_evs->new_event_metadata_ids_vec.push_back(777);
  inc_evs->new_event_names_vec.push_back("inc_op");
  inc_evs->new_event_display_names_vec.push_back("inc_disp");
  inc_evs->new_stat_metadata_ids_vec.push_back(888);
  inc_evs->new_stat_names_vec.push_back("inc_stat");
  inc_evs->add_offset_ps_deltas(3000);
  inc_evs->add_duration_ps(50);
  inc_evs->add_event_metadata_index(
      static_cast<uint32_t>(converter.current_plane().event_names_size()));
  XStatColumn* inc_scol = inc_evs->add_stats();
  inc_scol->set_stat_metadata_index(
      static_cast<uint32_t>(converter.current_plane().stat_names_size()));
  inc_scol->AddRef(0, /*ref_id=*/999, "missing_ref_target_auto_created");
  converter.Consume(std::move(inc_chunk));

  EXPECT_EQ(dst_plane.event_metadata().at(1).name(), "pre_existing_event");
  EXPECT_EQ(dst_plane.event_metadata().at(777).name(), "inc_op");
  EXPECT_EQ(dst_plane.stat_metadata().at(888).name(), "inc_stat");
  EXPECT_EQ(dst_plane.stat_metadata().at(999).name(),
            "missing_ref_target_auto_created");

  ASSERT_EQ(dst_plane.lines_size(), 1);
  const tensorflow::profiler::XLine& out_line = dst_plane.lines(0);
  ASSERT_EQ(out_line.events_size(), 3);

  // Event 0: verify ref_value == 0 and bytes_value round tripped with original
  // value_case().
  bool found_ref_zero = false;
  bool found_bytes = false;
  for (const auto& st : out_line.events(0).stats()) {
    const std::string& sname =
        dst_plane.stat_metadata().at(st.metadata_id()).name();
    if (sname == "ref_stat") {
      EXPECT_EQ(st.value_case(), tensorflow::profiler::XStat::kRefValue);
      EXPECT_EQ(st.ref_value(), 0u);
      EXPECT_EQ(dst_plane.stat_metadata().at(st.ref_value()).name(),
                "zero_ref_target");
      found_ref_zero = true;
    } else if (sname == "proto_bytes_stat") {
      EXPECT_EQ(st.value_case(), tensorflow::profiler::XStat::kBytesValue);
      EXPECT_EQ(st.bytes_value(), std::string("\x01\x00\x02\xff", 4));
      found_bytes = true;
    }
  }
  EXPECT_TRUE(found_ref_zero);
  EXPECT_TRUE(found_bytes);

  // Event 1: verify non zero ref_value round tripped.
  bool found_ref_nonzero = false;
  for (const auto& st : out_line.events(1).stats()) {
    const std::string& sname =
        dst_plane.stat_metadata().at(st.metadata_id()).name();
    if (sname == "ref_stat") {
      EXPECT_EQ(st.value_case(), tensorflow::profiler::XStat::kRefValue);
      EXPECT_EQ(dst_plane.stat_metadata().at(st.ref_value()).name(),
                "nonzero_ref_target");
      found_ref_nonzero = true;
    }
  }
  EXPECT_TRUE(found_ref_nonzero);
}

TEST(XChunkTest, SparseColumnSortsAndDeduplicatesOutOfOrderEntries) {
  SparseInt64Column col;
  col.Add(5, 50);
  col.Add(2, 20);
  col.Add(2, 25);  // in place overwrite of back()
  col.Add(5, 55);  // out of order duplicate of index 5
  col.Add(1, 10);
  EXPECT_TRUE(col.needs_sort);
  col.SortAndDeduplicate();
  EXPECT_FALSE(col.needs_sort);
  EXPECT_THAT(col.event_index(), ElementsAre(1, 2, 5));
  EXPECT_THAT(col.values(), ElementsAre(10, 25, 55));

  XStatColumn scol;
  scol.AddRef(3, /*ref_id=*/0, "ref0");
  scol.AddBytes(1, "\xaa\xbb");
  scol.AddStr(2, "plain");
  scol.AddRef(1, /*ref_id=*/7, "ref7_override");
  // Out of range uint64_t ref_id (> INT64_MAX) falls back to kStrValueTag
  // instead of wrapping to negative sentinel values (-1 / -2).
  scol.AddRef(4, /*ref_id=*/std::numeric_limits<uint64_t>::max(),
              "huge_ref_fallback");
  scol.SortAndDeduplicate();
  EXPECT_THAT(scol.event_index(), ElementsAre(1, 2, 3, 4));
  EXPECT_THAT(scol.str_values(), ElementsAre("ref7_override", "plain", "ref0",
                                             "huge_ref_fallback"));
  EXPECT_TRUE(scol.has_ref_id(0));
  EXPECT_EQ(scol.ref_id(0), 7u);
  EXPECT_FALSE(scol.has_ref_id(1));
  EXPECT_FALSE(scol.is_bytes(1));
  EXPECT_TRUE(scol.has_ref_id(2));
  EXPECT_EQ(scol.ref_id(2), 0u);
  EXPECT_FALSE(scol.has_ref_id(3));
  EXPECT_FALSE(scol.is_bytes(3));
}

TEST(XChunkTest, AppendXChunkToXPlaneClampsMismatchedAndOutOfBoundsColumns) {
  XPlane dst_plane;
  XChunk chunk;
  XChunkEvents* evs = chunk.mutable_events();
  evs->set_plane_index(0);
  evs->set_line_id(10);
  evs->set_num_events(5);  // Claims 5 events, but only 1 dense entry is added.
  evs->add_offset_ps_deltas(100);
  evs->add_duration_ps(50);
  evs->add_event_metadata_index(1);
  evs->mutable_flow()->Add(/*idx=*/99, /*val=*/123);  // Out of bounds idx
  evs->add_async_event_index(99);                     // Out of bounds idx
  XStatColumn* scol = evs->add_stats();
  scol->set_stat_metadata_index(2);
  scol->AddStr(/*idx=*/99, "oob_str");

  AppendXChunkToXPlane(chunk, /*current_plane=*/nullptr, &dst_plane);
  ASSERT_EQ(dst_plane.lines_size(), 1);
  ASSERT_EQ(dst_plane.lines(0).events_size(), 1);
  EXPECT_EQ(dst_plane.lines(0).events(0).stats_size(), 0);
}

TEST(XChunkTest, VisitorMasksPhaseBitsAndSaturatesEpochNanoToPico) {
  XChunkPlane plane;
  plane.set_plane_index(0);
  plane.AddEventMetadata(1, "split_op");
  XChunkLine* line_proto = plane.add_lines();
  line_proto->set_line_id(1);
  // Raw Unix epoch timestamp in nanoseconds (~1.79e18 ns); multiplying by
  // 1000 overflows signed int64_t unless saturated.
  line_proto->set_timestamp_ns(1790000000000000000LL);

  XChunkPlaneVisitor plane_visitor(&plane);
  XChunkLineVisitor line_visitor(line_proto);

  XChunkEvents events;
  events.set_plane_index(0);
  events.set_line_id(1);
  events.set_num_events(2);
  events.add_offset_ps_deltas(500);
  events.add_offset_ps_deltas(1000);
  events.add_duration_ps(static_cast<int64_t>(250ULL | kDurationPhaseBeginBit));
  events.add_duration_ps(static_cast<int64_t>(750ULL | kDurationPhaseEndBit));
  events.add_event_metadata_index(0);
  events.add_event_metadata_index(0);

  std::vector<int64_t> durations;
  std::vector<uint64_t> raw_durations;
  std::vector<bool> phase_begins;
  std::vector<bool> phase_ends;
  std::vector<int64_t> timestamps;
  std::vector<int64_t> end_timestamps;
  ForEachEvent(plane_visitor, line_visitor, events,
               [&](const XChunkEventVisitor& ev) {
                 durations.push_back(ev.DurationPs());
                 raw_durations.push_back(ev.RawDurationPs());
                 phase_begins.push_back(ev.IsPhaseBegin());
                 phase_ends.push_back(ev.IsPhaseEnd());
                 timestamps.push_back(ev.TimestampPs());
                 end_timestamps.push_back(ev.EndTimestampPs());
               });

  EXPECT_THAT(durations, ElementsAre(250, 750));
  EXPECT_THAT(raw_durations, ElementsAre(250ULL | kDurationPhaseBeginBit,
                                         750ULL | kDurationPhaseEndBit));
  EXPECT_THAT(phase_begins, ElementsAre(true, false));
  EXPECT_THAT(phase_ends, ElementsAre(false, true));
  EXPECT_THAT(timestamps, ElementsAre(std::numeric_limits<int64_t>::max(),
                                      std::numeric_limits<int64_t>::max()));
  EXPECT_THAT(end_timestamps, ElementsAre(std::numeric_limits<int64_t>::max(),
                                          std::numeric_limits<int64_t>::max()));

  // Also verify relative timestamps with phase bits return accurate
  // TimestampPs() and EndTimestampPs().
  XChunkPlane rel_plane = plane;
  rel_plane.lines(0).set_timestamp_ns(10);
  XChunkPlaneVisitor rel_plane_visitor(&rel_plane);
  XChunkLineVisitor rel_line_visitor(&rel_plane.lines(0));
  std::vector<int64_t> rel_ends;
  ForEachEvent(rel_plane_visitor, rel_line_visitor, events,
               [&](const XChunkEventVisitor& ev) {
                 rel_ends.push_back(ev.EndTimestampPs());
               });
  EXPECT_THAT(rel_ends, ElementsAre(10000 + 500 + 250, 10000 + 1500 + 750));

  // Verify AppendXChunkToXPlane masks phase bits on reconstructed XEvents.
  XPlane dst_plane;
  AppendXChunkToXPlane(XChunk(events), &rel_plane, &dst_plane);
  ASSERT_EQ(dst_plane.lines_size(), 1);
  ASSERT_EQ(dst_plane.lines(0).events_size(), 2);
  EXPECT_EQ(dst_plane.lines(0).events(0).duration_ps(), 250);
  EXPECT_EQ(dst_plane.lines(0).events(1).duration_ps(), 750);
}

TEST(XChunkTest, AppendXChunkToXPlaneRegistersEmptyRefValueMetadata) {
  XPlane dst_plane;
  {
    XChunkPlane plane;
    plane.set_plane_index(0);
    plane.AddEventMetadata(1, "ev");
    plane.AddStatMetadata(1, "ref_attr");
    XChunkLine* l = plane.add_lines();
    l->set_line_id(1);

    XChunkEvents evs;
    evs.set_plane_index(0);
    evs.set_line_id(1);
    evs.set_num_events(1);
    evs.add_offset_ps_deltas(100);
    evs.add_duration_ps(50);
    evs.add_event_metadata_index(0);
    XStatColumn* scol = evs.add_stats();
    scol->set_stat_metadata_index(0);
    // Reference target ID 2 with empty string value, emitted in the same chunk
    // as a flow stat that allocates a new special stat ID.
    scol->AddRef(0, /*ref_id=*/2, "");
    evs.mutable_flow()->Add(0, 42);

    XChunkToXPlaneConverter converter(&dst_plane);
    converter.Consume(XChunk(plane));
    converter.Consume(XChunk(evs));
  }

  ASSERT_TRUE(dst_plane.stat_metadata().find(2) !=
              dst_plane.stat_metadata().end());
  EXPECT_EQ(dst_plane.stat_metadata().at(2).name(), "");
  ASSERT_EQ(dst_plane.lines_size(), 1);
  ASSERT_EQ(dst_plane.lines(0).events_size(), 1);
  const auto& stats = dst_plane.lines(0).events(0).stats();
  ASSERT_EQ(stats.size(), 2);
  EXPECT_EQ(stats.Get(0).value_case(), tensorflow::profiler::XStat::kRefValue);
  EXPECT_EQ(stats.Get(0).ref_value(), 2u);
  const std::string& flow_stat_name =
      dst_plane.stat_metadata().at(stats.Get(1).metadata_id()).name();
  EXPECT_EQ(flow_stat_name, GetStatTypeStr(StatType::kFlow));
  EXPECT_NE(stats.Get(1).metadata_id(), 2);
}

TEST(XChunkTest, TraceViewerProcessIdAndStreamXSpaceHandleSubprocessAndMeta) {
  XSpace space;
  // 1. Metadata plane (should be skipped in StreamXSpaceToXChunks and not
  // collide with /device:GPU:0 in TraceViewerProcessId).
  XPlaneBuilder metadata(space.add_planes());
  metadata.SetId(0);
  metadata.SetName(kMetadataPlaneName);
  metadata.GetOrCreateLine(1).SetName("ignored_meta_line");

  // 2. Device plane with id = 0.
  XPlaneBuilder gpu0(space.add_planes());
  gpu0.SetId(0);
  gpu0.SetName(absl::StrCat(kGpuPlanePrefix, 0));
  XLineBuilder gpu_line = gpu0.GetOrCreateLine(1);
  gpu_line.SetName("Stream 1");
  gpu_line.AddEvent(*gpu0.GetOrCreateEventMetadata("kernel")).SetDurationPs(10);

  // 3. Out of order subprocess /host:CPU planes followed by main /host:CPU.
  XPlaneBuilder sub200(space.add_planes());
  sub200.SetId(2);
  sub200.SetName(absl::StrCat(kHostThreadsPlaneName, " [200]"));
  sub200.GetOrCreateLine(1)
      .AddEvent(*sub200.GetOrCreateEventMetadata("sub200"))
      .SetDurationPs(10);

  XPlaneBuilder sub100(space.add_planes());
  sub100.SetId(1);
  sub100.SetName(absl::StrCat(kHostThreadsPlaneName, " [100]"));
  sub100.GetOrCreateLine(1)
      .AddEvent(*sub100.GetOrCreateEventMetadata("sub100"))
      .SetDurationPs(10);

  XPlaneBuilder main_host(space.add_planes());
  main_host.SetId(0);
  main_host.SetName(kHostThreadsPlaneName);
  main_host.GetOrCreateLine(1)
      .AddEvent(*main_host.GetOrCreateEventMetadata("main_ev"))
      .SetDurationPs(10);

  // Direct TraceViewerProcessId checks.
  int next_custom = 0;
  int next_host_sub = 0;
  const uint32_t meta_pid =
      TraceViewerProcessId(space.planes(0), /*host_id=*/0, &next_custom);
  const uint32_t gpu0_pid =
      TraceViewerProcessId(space.planes(1), /*host_id=*/0, &next_custom);
  const uint32_t sub200_pid = TraceViewerProcessId(
      space.planes(2), /*host_id=*/0, &next_custom, &next_host_sub);
  const uint32_t sub100_pid = TraceViewerProcessId(
      space.planes(3), /*host_id=*/0, &next_custom, &next_host_sub);
  const uint32_t main_pid = TraceViewerProcessId(space.planes(4), /*host_id=*/0,
                                                 &next_custom, &next_host_sub);

  EXPECT_NE(meta_pid, gpu0_pid);
  EXPECT_EQ(gpu0_pid, kFirstDeviceId);
  EXPECT_EQ(main_pid, kHostThreadsDeviceId);
  EXPECT_EQ(sub200_pid, kHostThreadsDeviceId + 1);
  EXPECT_EQ(sub100_pid, kHostThreadsDeviceId + 2);

  // Also verify negative plane.id() on /host:metadata does not fail or collide.
  XPlane neg_meta;
  neg_meta.set_id(-1);
  neg_meta.set_name(kMetadataPlaneName);
  EXPECT_EQ(TraceViewerProcessId(neg_meta, /*host_id=*/1, &next_custom),
            kMaxDevicesPerHost);

  // StreamXSpaceToXChunks checks.
  absl::flat_hash_map<std::string, uint32_t> pid_by_plane_name;
  StreamXSpaceToXChunks(
      space, "host0", /*host_id=*/0, XChunkOptions{}, [&](XChunk&& chunk) {
        if (chunk.has_plane()) {
          pid_by_plane_name.emplace(std::string(chunk.plane().plane_name()),
                                    chunk.plane().process_id());
        }
      });

  EXPECT_FALSE(pid_by_plane_name.contains(kMetadataPlaneName));
  EXPECT_EQ(pid_by_plane_name.at("/device:GPU:0"), kFirstDeviceId);
  EXPECT_EQ(pid_by_plane_name.at("/host:CPU"), kHostThreadsDeviceId);
  // Sorted deterministically by name: [100] gets +1, [200] gets +2.
  EXPECT_EQ(pid_by_plane_name.at("/host:CPU [100]"), kHostThreadsDeviceId + 1);
  EXPECT_EQ(pid_by_plane_name.at("/host:CPU [200]"), kHostThreadsDeviceId + 2);
}

}  // namespace
}  // namespace tsl::profiler::xchunk
