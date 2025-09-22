/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#include "tsl/platform/test.h"

#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/notification.h"
#include "google/protobuf/stubs/common.h"  // ShutdownProtobufLibrary
#include "xla/backends/profiler/gpu/rocm_collector.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"  // tsl::profiler::GpuPlaneName

namespace xla {
namespace profiler {
namespace test {

namespace tp = ::tensorflow::profiler;  // Protobufs: XSpace/XPlane
namespace tps = ::tsl::profiler;        // Schema helpers: GpuPlaneName

class RocmCollectorSanitizerTest : public ::testing::Test {
 protected:
  static void TearDownTestSuite() {
    // Helps LeakSanitizer in some environments with protobuf singletons.
    google::protobuf::ShutdownProtobufLibrary();
  }
};

// Read-only helper: find a plane by name without mutating the XSpace.
static const tp::XPlane* FindPlaneConst(const tp::XSpace& space,
                                        absl::string_view name) {
  for (const auto& p : space.planes()) {
    if (p.name() == name) return &p;
  }
  return nullptr;
}

// Safely get an event metadata name from the protobuf Map.
static std::string GetEventMetadataName(const tp::XPlane& plane, int64_t id) {
  const auto& md_map = plane.event_metadata();
  auto it = md_map.find(id);
  if (it == md_map.end()) return {};
  return it->second.name();
}

TEST_F(RocmCollectorSanitizerTest, TestAddKernelEventAndExport_TSANSafe) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 100;
  options.max_activity_api_events = 100;
  options.max_annotation_strings = 100;
  options.num_gpus = 1;

  constexpr uint64_t kStartWallTimeNs = 1000;
  constexpr uint64_t kStartGpuTimeNs = 2000;

  RocmTraceCollectorImpl collector(options, kStartWallTimeNs, kStartGpuTimeNs);

  constexpr uint32_t kCorrelationId = 42;
  constexpr uint64_t kStartTimeNs = 3000;
  constexpr uint64_t kEndTimeNs = 4000;

  // 1) API-callback side (host).
  {
    RocmTracerEvent api_event{};  // value-init all fields to zero/known state
    api_event.type = RocmTracerEventType::Kernel;
    api_event.source = RocmTracerEventSource::ApiCallback;
    api_event.domain = RocmTracerEventDomain::HIP_API;
    api_event.name = "test_rocm_kernel";  // string literal => static storage
    api_event.correlation_id = kCorrelationId;
    api_event.thread_id = 999;
    api_event.device_id = 0;  // set explicitly
    api_event.stream_id = 123;
    api_event.kernel_info = {
        /*.private_segment_size=*/32,
        /*.group_segment_size=*/1024,
        /*.block_x=*/256,
        /*.block_y=*/1,
        /*.block_z=*/1,
        /*.grid_x=*/100,
        /*.grid_y=*/1,
        /*.grid_z=*/1,
        /*.func_ptr=*/nullptr,  // avoid accidental deref/symbolization
    };
    collector.AddEvent(std::move(api_event), /*is_auxiliary=*/false);
  }

  // 2) Activity side (device).
  {
    RocmTracerEvent act_event{};
    act_event.type = RocmTracerEventType::Kernel;
    act_event.source = RocmTracerEventSource::Activity;
    act_event.domain = RocmTracerEventDomain::HIP_OPS;
    act_event.name = "test_rocm_kernel";
    act_event.correlation_id = kCorrelationId;
    act_event.start_time_ns = kStartTimeNs;
    act_event.end_time_ns = kEndTimeNs;
    act_event.device_id = 0;  // keep within [0, num_gpus)
    act_event.stream_id = 123;
    collector.AddEvent(std::move(act_event), /*is_auxiliary=*/false);
  }

  // 3) Quiesce/merge before export.
  collector.Flush();

  tp::XSpace space;
  collector.Export(&space);

  // 4) Verify without any "find-or-add" helpers.
  const std::string plane_name = tps::GpuPlaneName(/*device_ordinal=*/0);
  const tp::XPlane* gpu_plane = FindPlaneConst(space, plane_name);
  ASSERT_NE(gpu_plane, nullptr) << "GPU plane not found: " << plane_name;

  ASSERT_GT(gpu_plane->lines_size(), 0);
  const auto& line = gpu_plane->lines(0);
  ASSERT_GT(line.events_size(), 0);

  const auto& ev = line.events(0);
  // Offsets are relative to the line timestamp (start_gputime_ns).
  EXPECT_EQ(ev.offset_ps(), (kStartTimeNs - kStartGpuTimeNs) * 1000);
  EXPECT_EQ(ev.duration_ps(), (kEndTimeNs - kStartTimeNs) * 1000);

  const std::string md_name = GetEventMetadataName(*gpu_plane, ev.metadata_id());
  ASSERT_FALSE(md_name.empty());
  EXPECT_EQ(md_name, "test_rocm_kernel");
}

// Concurrency test: multiple threads add matched API+Activity pairs.
// We coordinate starts, join all threads, then Flush and Export.
TEST_F(RocmCollectorSanitizerTest, ConcurrentAddAndExport_TSANSafe) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 10'000;
  options.max_activity_api_events = 10'000;
  options.max_annotation_strings = 1'000;
  options.num_gpus = 1;

  constexpr uint64_t kStartWallTimeNs = 1'000'000;
  constexpr uint64_t kStartGpuTimeNs = 2'000'000;

  RocmTraceCollectorImpl collector(options, kStartWallTimeNs, kStartGpuTimeNs);

  constexpr int kThreads = 8;
  constexpr int kEventsPerThread = 64;

  absl::BlockingCounter ready(kThreads);
  absl::Notification go;

  auto worker = [&](int tid) {
    ready.DecrementCount();
    go.WaitForNotification();

    const uint64_t base_start = 3'000'000 + static_cast<uint64_t>(tid) * 10'000;
    const uint64_t dur_ns = 5'000;

    for (int i = 0; i < kEventsPerThread; ++i) {
      const uint32_t cid = static_cast<uint32_t>(tid * 10'000 + i);
      const uint64_t t0 = base_start + static_cast<uint64_t>(i) * 100;
      const uint64_t t1 = t0 + dur_ns;

      RocmTracerEvent api{};
      api.type = RocmTracerEventType::Kernel;
      api.source = RocmTracerEventSource::ApiCallback;
      api.domain = RocmTracerEventDomain::HIP_API;
      api.name = "kernel_concurrent";  // string literal (static)
      api.correlation_id = cid;
      api.thread_id = 1000 + tid;
      api.device_id = 0;
      api.stream_id = 1000 + tid;  // one line per worker
      collector.AddEvent(std::move(api), /*is_auxiliary=*/false);

      RocmTracerEvent act{};
      act.type = RocmTracerEventType::Kernel;
      act.source = RocmTracerEventSource::Activity;
      act.domain = RocmTracerEventDomain::HIP_OPS;
      act.name = "kernel_concurrent";
      act.correlation_id = cid;
      act.start_time_ns = t0;
      act.end_time_ns = t1;
      act.device_id = 0;
      act.stream_id = 1000 + tid;
      collector.AddEvent(std::move(act), /*is_auxiliary=*/false);
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) threads.emplace_back(worker, t);

  ready.Wait();
  go.Notify();
  for (auto& th : threads) th.join();

  collector.Flush();

  tp::XSpace space;
  collector.Export(&space);

  const std::string plane_name = tps::GpuPlaneName(/*device_ordinal=*/0);
  const tp::XPlane* gpu_plane = FindPlaneConst(space, plane_name);
  ASSERT_NE(gpu_plane, nullptr) << "GPU plane not found: " << plane_name;

  uint64_t total_events = 0;
  for (const auto& l : gpu_plane->lines()) total_events += l.events_size();
  ASSERT_EQ(total_events, static_cast<uint64_t>(kThreads * kEventsPerThread));

  // Spot check: first line has events and metadata resolves.
  if (gpu_plane->lines_size() > 0 && gpu_plane->lines(0).events_size() > 0) {
    const auto& ev = gpu_plane->lines(0).events(0);
    const std::string md_name =
        GetEventMetadataName(*gpu_plane, ev.metadata_id());
    ASSERT_FALSE(md_name.empty());
    EXPECT_EQ(md_name, "kernel_concurrent");
    EXPECT_GT(ev.duration_ps(), 0);
  }
}

}  // namespace test
}  // namespace profiler
}  // namespace xla
