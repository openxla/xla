/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/profiler/gpu/cupti_collector.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/backends/profiler/gpu/cupti_buffer_events.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

using ::tensorflow::profiler::XSpace;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(CuptiCollectorTest, TestPmSamplingDataToCounterLine) {
  PmSamples pm_samples({"metric1", "metric2"},
                       {{/*range_index=*/0,
                         /*start_timestamp_ns=*/100,
                         /*end_timestamp_ns=*/200,
                         /*metric_values=*/{1.0, 2.0}},
                        {/*range_index=*/1,
                         /*start_timestamp_ns=*/200,
                         /*end_timestamp_ns=*/300,
                         /*metric_values=*/{3.0, 4.0}}},
                       0);
  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  pm_samples.PopulateCounterLine(&plane_builder, 0);

  EXPECT_EQ(plane.lines_size(), 1);
  EXPECT_EQ(plane.lines(0).events_size(), 4);
  EXPECT_EQ(plane.event_metadata_size(), 2);
  EXPECT_EQ(plane.stat_metadata_size(), 2);
  absl::flat_hash_map<std::string, absl::flat_hash_map<uint64_t, double>>
      counter_events_values;
  for (const auto& event : plane.lines(0).events()) {
    counter_events_values[plane.event_metadata().at(event.metadata_id()).name()]
                         [event.offset_ps()] = event.stats(0).double_value();
  }
  EXPECT_THAT(counter_events_values,
              UnorderedElementsAre(
                  Pair("metric1", UnorderedElementsAre(Pair(100000, 1.0),
                                                       Pair(200000, 3.0))),
                  Pair("metric2", UnorderedElementsAre(Pair(100000, 2.0),
                                                       Pair(200000, 4.0)))));
}

TEST(CuptiCollectorTest, TestPmSamplingDefaultMetricsToCounterLineRenaming) {
  PmSamples pm_samples({"gpc__cycles_elapsed.avg.per_second", "metric2"},
                       {{/*range_index=*/0,
                         /*start_timestamp_ns=*/100,
                         /*end_timestamp_ns=*/200,
                         /*metric_values=*/{1.0, 2.0}},
                        {/*range_index=*/1,
                         /*start_timestamp_ns=*/200,
                         /*end_timestamp_ns=*/300,
                         /*metric_values=*/{3.0, 4.0}}},
                       0);
  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  pm_samples.PopulateCounterLine(&plane_builder, 0);

  EXPECT_EQ(plane.lines_size(), 1);
  EXPECT_EQ(plane.lines(0).events_size(), 4);
  EXPECT_EQ(plane.event_metadata_size(), 2);
  EXPECT_EQ(plane.stat_metadata_size(), 2);
  absl::flat_hash_map<std::string, absl::flat_hash_map<uint64_t, double>>
      counter_events_values;
  for (const auto& event : plane.lines(0).events()) {
    counter_events_values[plane.event_metadata().at(event.metadata_id()).name()]
                         [event.offset_ps()] = event.stats(0).double_value();
  }
  EXPECT_THAT(
      counter_events_values,
      UnorderedElementsAre(
          Pair("GPC Clock Frequency (Hz)",
               UnorderedElementsAre(Pair(100000, 1.0), Pair(200000, 3.0))),
          Pair("metric2",
               UnorderedElementsAre(Pair(100000, 2.0), Pair(200000, 4.0)))));
}

TEST(CuptiCollectorTest, ExportCallbackActivityAndNvtxEvents) {
  CuptiTracerCollectorOptions options;
  options.max_activity_api_events = 100;
  options.max_callback_api_events = 100;
  options.num_gpus = 1;
  std::unique_ptr<CuptiTraceCollector> collector =
      CreateCuptiCollector(options, 0, 0);

  collector->AddEvent(CuptiTracerEvent{
      /*type=*/CuptiTracerEventType::CudaGraph,
      /*source=*/CuptiTracerEventSource::Activity,
      /*name=*/"CudaGraphExec:2",
      /*annotation=*/"annotation",
      /*nvtx_range=*/"",
      /*start_time_ns=*/100,
      /*end_time_ns=*/200,
      /*device_id=*/0,
      /*correlation_id=*/8,
      /*thread_id=*/100,
      /*context_id=*/1,
      /*stream_id=*/2,
      /*graph_id=*/5,
  });

  collector->AddEvent(CuptiTracerEvent{
      /*type=*/CuptiTracerEventType::Generic,
      /*source=*/CuptiTracerEventSource::DriverCallback,
      /*name=*/"cudaGraphLaunch",
      /*annotation=*/"annotation",
      /*nvtx_range=*/"",
      /*start_time_ns=*/90,
      /*end_time_ns=*/120,
      /*device_id=*/0,
      /*correlation_id=*/8,
      /*thread_id=*/100,
      /*context_id=*/1,
      /*stream_id=*/2,
      /*graph_id=*/5,
  });

  collector->AddEvent(CuptiTracerEvent{
      /*type=*/CuptiTracerEventType::ThreadMarkerRange,
      /*source=*/CuptiTracerEventSource::Activity,
      /*name=*/"NVTX::MarkCudaGraphLaunch",
      /*annotation=*/"annotation",
      /*nvtx_range=*/"",
      /*start_time_ns=*/85,
      /*end_time_ns=*/125,
      /*device_id=*/0,
      /*correlation_id=*/0,
      /*thread_id=*/100,
      /*context_id=*/1,
      /*stream_id=*/2,
      /*graph_id=*/5,
  });

  XSpace space;
  collector->Export(&space, /*end_gpu_ns=*/210);

  // All the three planes must exist in the space:
  // Cupti-Driver-API, Cupti-NVTX, GpuDevice.
  const std::string gpu_device_plane_name = ::tsl::profiler::GpuPlaneName(0);
  const absl::flat_hash_set<absl::string_view> plane_names = {
      ::tsl::profiler::kCuptiDriverApiPlaneName,
      ::tsl::profiler::kCuptiActivityNvtxPlaneName, gpu_device_plane_name};
  int num_planes_to_check = 0;
  for (const auto& plane : space.planes()) {
    if (plane_names.contains(plane.name())) {
      ++num_planes_to_check;
    }
  }
  EXPECT_EQ(num_planes_to_check, static_cast<int>(plane_names.size()));

  // In each above plane, only one line is created, and it has one event.
  for (const auto& plane : space.planes()) {
    if (plane_names.contains(plane.name())) {
      ASSERT_EQ(plane.lines_size(), 1);
      ASSERT_EQ(plane.lines(0).events_size(), 1);
    }
  }
}

TEST(CuptiCollectorTest, ExportEnvironmentEvents) {
  CuptiTracerCollectorOptions options;
  options.max_activity_api_events = 100;
  options.max_callback_api_events = 100;
  options.num_gpus = 1;
  std::unique_ptr<CuptiTraceCollector> collector =
      CreateCuptiCollector(options, 0, 0);

  collector->AddEvent(CuptiTracerEvent{
      /*type=*/CuptiTracerEventType::Environment,
      /*source=*/CuptiTracerEventSource::Activity,
      /*name=*/"power_mw",
      /*annotation=*/"",
      /*nvtx_range=*/"",
      /*start_time_ns=*/100,
      /*end_time_ns=*/100,
      /*device_id=*/0,
      /*correlation_id=*/0,
      /*thread_id=*/0,
      /*context_id=*/0,
      /*stream_id=*/0,
      /*graph_id=*/0,
      /*scope_range_id=*/0,
      /*graph_node_id=*/0,
      /*environment_info=*/{{/*metric_value=*/1000}},
  });

  collector->AddEvent(CuptiTracerEvent{
      /*type=*/CuptiTracerEventType::Environment,
      /*source=*/CuptiTracerEventSource::Activity,
      /*name=*/"gpu_temp_c",
      /*annotation=*/"",
      /*nvtx_range=*/"",
      /*start_time_ns=*/110,
      /*end_time_ns=*/110,
      /*device_id=*/0,
      /*correlation_id=*/0,
      /*thread_id=*/0,
      /*context_id=*/0,
      /*stream_id=*/0,
      /*graph_id=*/0,
      /*scope_range_id=*/0,
      /*graph_node_id=*/0,
      /*environment_info=*/{{/*metric_value=*/60}},
  });

  XSpace space;
  collector->Export(&space, 210);

  const std::string gpu_device_plane_name = ::tsl::profiler::GpuPlaneName(0);
  int num_planes_to_check = 0;
  for (const auto& plane : space.planes()) {
    if (plane.name() != gpu_device_plane_name) {
      continue;
    }
    ++num_planes_to_check;
    EXPECT_EQ(plane.lines_size(), 1);
    EXPECT_EQ(plane.lines(0).events_size(), 2);
    absl::flat_hash_map<std::string, absl::flat_hash_map<uint64_t, double>>
        counter_events_values;
    for (const auto& event : plane.lines(0).events()) {
      counter_events_values[plane.event_metadata()
                                .at(event.metadata_id())
                                .name()][event.offset_ps()] =
          event.stats(0).double_value();
    }
    EXPECT_THAT(
        counter_events_values,
        UnorderedElementsAre(
            Pair("power_mw", UnorderedElementsAre(Pair(100000, 1000.0))),
            Pair("gpu_temp_c", UnorderedElementsAre(Pair(110000, 60.0)))));
  }
  EXPECT_EQ(num_planes_to_check, 1);
}

TEST(PmSamplesTest, PopulateCounterLineSkipsNan) {
  XSpace space;
  tsl::profiler::XPlaneBuilder plane_builder(space.add_planes());
  PmSamples pm_samples({"metric1", "metric2"},
                       {{/*range_index=*/0,
                         /*start_timestamp_ns=*/100,
                         /*end_timestamp_ns=*/200,
                         {123.0, std::numeric_limits<double>::quiet_NaN()}}},
                       /*device_id=*/0);

  uint64_t start_gpu_time_ns = 50;
  pm_samples.PopulateCounterLine(&plane_builder, start_gpu_time_ns);

  const auto& plane = space.planes(0);
  ASSERT_EQ(plane.lines_size(), 1);
  const auto& line = plane.lines(0);

  ASSERT_EQ(line.events_size(), 1);
  const auto& event = line.events(0);
  // metric2 is skipped because it's NaN.
  ASSERT_EQ(event.stats_size(), 1);
  const auto& stat = event.stats(0);
  EXPECT_EQ(plane.stat_metadata().at(stat.metadata_id()).name(), "metric1");
  EXPECT_EQ(stat.double_value(), 123.0);
}

TEST(RangeProfilingTest, PopulateRangeProfilingEventsBasic) {
  // Two metrics, one range.
  std::vector<std::string> metrics = {"sm__cycles_active.sum",
                                      "dram__bytes.sum"};
  std::vector<MetricProperties> props = {
      {/*description=*/"SM Active Cycles", /*hw_unit=*/"sm"},
      {/*description=*/"DRAM Bytes", /*hw_unit=*/"dram"},
  };
  std::vector<RangeResult> ranges = {{
      /*range_name=*/"hlo_execution",
      /*start_timestamp_ns=*/1000,
      /*end_timestamp_ns=*/2000,
      /*metric_values=*/{42.0, 100.0},
  }};
  RangeProfilerResults results(metrics, props, ranges, /*device_id=*/0);

  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  PopulateRangeProfilingEvents(&results, &plane_builder);

  // One line (one range), two events (two metrics).
  ASSERT_EQ(plane.lines_size(), 1);
  const auto& line = plane.lines(0);
  EXPECT_EQ(line.name(), "hlo_execution");
  ASSERT_EQ(line.events_size(), 2);

  // Check each event has three stats: counter_value, description, sets.
  for (const auto& event : line.events()) {
    ASSERT_EQ(event.stats_size(), 3);
  }

  // Verify event names map to raw metric names.
  absl::flat_hash_set<std::string> event_names;
  for (const auto& event : line.events()) {
    event_names.insert(
        plane.event_metadata().at(event.metadata_id()).name());
  }
  EXPECT_TRUE(event_names.contains("sm__cycles_active.sum"));
  EXPECT_TRUE(event_names.contains("dram__bytes.sum"));

  // Verify counter values and CUPTI-provided descriptions.
  using tsl::profiler::GetStatTypeStr;
  using tsl::profiler::StatType;
  for (const auto& event : line.events()) {
    std::string event_name =
        plane.event_metadata().at(event.metadata_id()).name();
    for (const auto& stat : event.stats()) {
      std::string stat_name =
          plane.stat_metadata().at(stat.metadata_id()).name();
      if (stat_name == GetStatTypeStr(StatType::kCounterValue)) {
        if (event_name == "sm__cycles_active.sum") {
          EXPECT_EQ(stat.uint64_value(), 42);
        } else {
          EXPECT_EQ(stat.uint64_value(), 100);
        }
      }
      if (stat_name ==
          GetStatTypeStr(StatType::kPerformanceCounterDescription)) {
        if (event_name == "sm__cycles_active.sum") {
          EXPECT_EQ(stat.str_value(), "SM Active Cycles");
        } else {
          EXPECT_EQ(stat.str_value(), "DRAM Bytes");
        }
      }
      if (stat_name == GetStatTypeStr(StatType::kPerformanceCounterSets)) {
        if (event_name == "sm__cycles_active.sum") {
          EXPECT_EQ(stat.str_value(), "sm");
        } else {
          EXPECT_EQ(stat.str_value(), "dram");
        }
      }
    }
  }
}

TEST(RangeProfilingTest, PopulateRangeProfilingEventsFallback) {
  // Without MetricProperties, falls back to prefix-based category and
  // display-name map.
  std::vector<std::string> metrics = {"sm__cycles_active.sum"};
  std::vector<RangeResult> ranges = {{
      /*range_name=*/"test_range",
      /*start_timestamp_ns=*/500,
      /*end_timestamp_ns=*/1500,
      /*metric_values=*/{7.0},
  }};
  // Use the constructor without properties.
  RangeProfilerResults results(metrics, ranges, /*device_id=*/0);

  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  PopulateRangeProfilingEvents(&results, &plane_builder);

  ASSERT_EQ(plane.lines_size(), 1);
  EXPECT_EQ(plane.lines(0).name(), "test_range");
  ASSERT_EQ(plane.lines(0).events_size(), 1);

  const auto& event = plane.lines(0).events(0);
  ASSERT_EQ(event.stats_size(), 3);

  // Verify fallback category (prefix "sm__" → "Streaming Multiprocessor").
  using tsl::profiler::GetStatTypeStr;
  using tsl::profiler::StatType;
  for (const auto& stat : event.stats()) {
    std::string stat_name =
        plane.stat_metadata().at(stat.metadata_id()).name();
    if (stat_name == GetStatTypeStr(StatType::kPerformanceCounterSets)) {
      EXPECT_EQ(stat.str_value(), "Streaming Multiprocessor");
    }
    if (stat_name == GetStatTypeStr(StatType::kCounterValue)) {
      EXPECT_EQ(stat.uint64_value(), 7);
    }
  }
}

TEST(RangeProfilingTest, PopulateRangeProfilingEventsMultipleRanges) {
  std::vector<std::string> metrics = {"sm__cycles_active.sum"};
  std::vector<MetricProperties> props = {
      {/*description=*/"Cycles", /*hw_unit=*/"sm"},
  };
  std::vector<RangeResult> ranges = {
      {/*range_name=*/"range_0",
       /*start_timestamp_ns=*/100,
       /*end_timestamp_ns=*/200,
       /*metric_values=*/{10.0}},
      {/*range_name=*/"range_1",
       /*start_timestamp_ns=*/300,
       /*end_timestamp_ns=*/400,
       /*metric_values=*/{20.0}},
  };
  RangeProfilerResults results(metrics, props, ranges, /*device_id=*/0);

  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  PopulateRangeProfilingEvents(&results, &plane_builder);

  // Two lines (two ranges), one event each.
  ASSERT_EQ(plane.lines_size(), 2);
  EXPECT_EQ(plane.lines(0).name(), "range_0");
  EXPECT_EQ(plane.lines(1).name(), "range_1");
  EXPECT_EQ(plane.lines(0).events_size(), 1);
  EXPECT_EQ(plane.lines(1).events_size(), 1);
}

TEST(RangeProfilingTest, PopulateRangeProfilingEventsNullResults) {
  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  PopulateRangeProfilingEvents(nullptr, &plane_builder);
  EXPECT_EQ(plane.lines_size(), 0);
}

}  // namespace
}  // namespace profiler
}  // namespace xla
