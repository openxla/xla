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

#include "xla/backends/profiler/gpu/rocm_collector.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace test {

using tsl::profiler::FindOrAddMutablePlaneWithName;
using tsl::profiler::GetStatTypeStr;
using tsl::profiler::StatType;
using tsl::profiler::XSpace;

TEST(RocmCollectorTest, TestAddKernelEventAndExport) {
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

  // === 1. Add API Callback Event ===
  RocmTracerEvent api_event;
  api_event.type = RocmTracerEventType::Kernel;
  api_event.source = RocmTracerEventSource::ApiCallback;
  api_event.domain = RocmTracerEventDomain::HIP_API;
  api_event.name = "test_rocm_kernel";
  api_event.correlation_id = kCorrelationId;
  api_event.thread_id = 999;
  // KernelDetails is a union member with no default member initializer, so it
  // has to be value-initialized explicitly. It is left zeroed here on purpose:
  // ApiActivityInfoExchange copies activity->api, so the ACTIVITY side below
  // is the authoritative one -- same direction as the real tracer, which
  // builds KernelDetails in KernelEvent() on the dispatch record.
  api_event.kernel_info = KernelDetails{};

  collector.AddEvent(std::move(api_event), /*is_auxiliary=*/false);

  // === 2. Add Activity Event ===
  RocmTracerEvent activity_event;
  activity_event.type = RocmTracerEventType::Kernel;
  activity_event.source = RocmTracerEventSource::Activity;
  activity_event.domain = RocmTracerEventDomain::HIP_OPS;
  activity_event.name = "test_rocm_kernel";
  activity_event.correlation_id = kCorrelationId;
  activity_event.start_time_ns = kStartTimeNs;
  activity_event.end_time_ns = kEndTimeNs;
  activity_event.device_id = 100;
  activity_event.stream_id = 123;
  activity_event.kernel_info = KernelDetails{};
  activity_event.kernel_info.private_segment_size = 32;
  activity_event.kernel_info.group_segment_size = 1024;
  activity_event.kernel_info.workgroup_x = 256;
  activity_event.kernel_info.workgroup_y = 1;
  activity_event.kernel_info.workgroup_z = 1;
  activity_event.kernel_info.grid_x = 100;
  activity_event.kernel_info.grid_y = 1;
  activity_event.kernel_info.grid_z = 1;
  activity_event.kernel_info.arch_vgpr_count = 32;

  collector.AddEvent(std::move(activity_event), /*is_auxiliary=*/false);

  // === 3. Finalize and Export ===
  collector.Flush();

  tensorflow::profiler::XSpace space;
  collector.Export(&space);

  // === 4. Check results ===
  ASSERT_GE(space.planes_size(), 1);
  const auto* gpu_plane =
      FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu_plane, nullptr);

  ASSERT_GT(gpu_plane->lines_size(), 0);
  const auto& line = gpu_plane->lines(0);
  ASSERT_GT(line.events_size(), 0);

  const auto& event = line.events(0);
  EXPECT_EQ(event.offset_ps(), (kStartTimeNs - kStartGpuTimeNs) * 1000);
  EXPECT_EQ(event.duration_ps(), (kEndTimeNs - kStartTimeNs) * 1000);
  EXPECT_EQ(gpu_plane->event_metadata().at(event.metadata_id()).name(),
            "test_rocm_kernel");
}

// Regression test for the .front()-only iteration bug in
// ApiActivityInfoExchange. When N activity events share one
// correlation_id (the rocprofiler-sdk pattern for hipGraphLaunch-replayed
// kernels), all N must reach the exported XPlane, not just the first.
TEST(RocmCollectorTest, MultipleActivitiesPerCorrelationIdAllExported) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 100;
  options.max_activity_api_events = 100;
  options.max_annotation_strings = 100;
  options.num_gpus = 1;

  constexpr uint64_t kStartWallTimeNs = 1000;
  constexpr uint64_t kStartGpuTimeNs = 2000;
  RocmTraceCollectorImpl collector(options, kStartWallTimeNs, kStartGpuTimeNs);

  // Single correlation_id shared by all events -- mirrors a hipGraphLaunch
  // that replays a captured graph: one API call, many kernel-dispatch
  // records emitted by rocprofiler-sdk under the same correlation_id.
  constexpr uint32_t kCorrelationId = 7;
  constexpr uint32_t kDeviceId = 100;
  constexpr uint64_t kStreamId = 123;

  RocmTracerEvent api_event;
  api_event.type = RocmTracerEventType::Kernel;
  api_event.source = RocmTracerEventSource::ApiCallback;
  api_event.domain = RocmTracerEventDomain::HIP_API;
  api_event.name = "hipGraphLaunch";
  api_event.correlation_id = kCorrelationId;
  api_event.thread_id = 999;
  api_event.kernel_info = KernelDetails{};
  api_event.kernel_info.arch_vgpr_count = 32;
  collector.AddEvent(std::move(api_event), /*is_auxiliary=*/false);

  // Three GPU activity records, same correlation_id, same stream (so
  // they land on the same XLine), distinct names and timestamps.
  // Distinct register counts per dispatch: a graph replay mixes a cheap
  // elementwise kernel with an expensive MFMA one, and each must keep its own
  // KernelDetails through the api/activity merge.
  struct ActivityShape {
    const char* name;
    uint64_t start_ns;
    uint64_t end_ns;
    uint32_t arch_vgpr_count;
  };
  constexpr ActivityShape kActivities[] = {
      {"kernel_a", 3000, 3500, 8},
      {"kernel_b", 3500, 4000, 64},
      {"kernel_c", 4000, 4500, 256},
  };
  for (const auto& shape : kActivities) {
    RocmTracerEvent activity;
    activity.type = RocmTracerEventType::Kernel;
    activity.source = RocmTracerEventSource::Activity;
    activity.domain = RocmTracerEventDomain::HIP_OPS;
    activity.name = shape.name;
    activity.correlation_id = kCorrelationId;
    activity.start_time_ns = shape.start_ns;
    activity.end_time_ns = shape.end_ns;
    activity.device_id = kDeviceId;
    activity.stream_id = kStreamId;
    activity.kernel_info = KernelDetails{};
    activity.kernel_info.workgroup_x = 256;
    activity.kernel_info.workgroup_y = 1;
    activity.kernel_info.workgroup_z = 1;
    activity.kernel_info.arch_vgpr_count = shape.arch_vgpr_count;
    collector.AddEvent(std::move(activity), /*is_auxiliary=*/false);
  }

  collector.Flush();
  tensorflow::profiler::XSpace space;
  collector.Export(&space);

  const auto* gpu_plane =
      FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu_plane, nullptr);

  // Pre-fix (.front()-only) would emit just one event here. The fix
  // iterates the entire vector, so all three activity records must
  // appear on the stream line. Dense stream remapping converts the raw
  // stream_id (123) to a sequential index (0), so we look for events on
  // any device line rather than matching a specific line ID.
  int64_t details_id = -1;
  for (const auto& [id, smd] : gpu_plane->stat_metadata()) {
    if (smd.name() == GetStatTypeStr(tsl::profiler::StatType::kKernelDetails)) {
      details_id = id;
      break;
    }
  }

  size_t total_kernel_events = 0;
  absl::flat_hash_set<std::string> seen_names;
  absl::flat_hash_map<std::string, std::string> details_by_name;
  for (const auto& line : gpu_plane->lines()) {
    total_kernel_events += line.events_size();
    for (const auto& ev : line.events()) {
      const std::string& name =
          gpu_plane->event_metadata().at(ev.metadata_id()).name();
      seen_names.insert(name);
      for (const auto& stat : ev.stats()) {
        if (stat.metadata_id() != details_id) continue;
        auto it = gpu_plane->stat_metadata().find(stat.ref_value());
        if (it != gpu_plane->stat_metadata().end()) {
          details_by_name[name] = it->second.name();
        }
      }
    }
  }

  EXPECT_EQ(total_kernel_events, 3u)
      << "Expected all 3 activity records to be emitted under the same "
         "correlation_id; got "
      << total_kernel_events
      << " (this is the "
         "regression the .front()-only iteration introduced).";
  EXPECT_TRUE(seen_names.contains("kernel_a"));
  EXPECT_TRUE(seen_names.contains("kernel_b"));
  EXPECT_TRUE(seen_names.contains("kernel_c"));

  // Each dispatch must keep its OWN registers. Before the merge fix, the
  // api/activity exchange seeded the shared api_event from `.front()` and then
  // copied it back over every activity, so all three reported regs:8 -- and
  // therefore all three would report kernel_a's occupancy. No agent is
  // registered here, so regs is the non-unified max(arch, accum) == arch.
  for (const auto& shape : kActivities) {
    auto it = details_by_name.find(shape.name);
    ASSERT_NE(it, details_by_name.end())
        << shape.name << " has no kKernelDetails stat";
    EXPECT_EQ(
        it->second.rfind(absl::StrCat("regs:", shape.arch_vgpr_count, " "), 0),
        0u)
        << shape.name
        << " must report its own register count, got: " << it->second;
  }
}

// ============================================================================
// Occupancy unit tests
//
// The formula itself is pinned by a 16-row golden table in
// rocm_occupancy_test.cc, which needs no ROCm at all. What is tested here is
// the collector's half of the job: that KernelDetails reaches
// RocmDeviceOccupancyParams intact, that the agent's gfx_target_version and
// cu_count reach the model, and that the resulting XStats are the ones XProf
// expects.
// ============================================================================

// gfx942 == MI300X. Decoded as major=(v/10000)%100, minor=(v/100)%100,
// step=v%100, all decimal.
constexpr uint32_t kGfx942 = 90402;
constexpr uint32_t kCuCount = 304;  // MI300X

// Returns the first stat of `type` found on any event in `plane`, or nullptr
// if that stat was never emitted.
const tensorflow::profiler::XStat* FindEventStat(
    const tensorflow::profiler::XPlane& plane, StatType type) {
  absl::string_view key = GetStatTypeStr(type);
  int64_t stat_id = -1;
  for (const auto& [id, smd] : plane.stat_metadata()) {
    if (smd.name() == key) {
      stat_id = id;
      break;
    }
  }
  if (stat_id < 0) return nullptr;
  for (const auto& line : plane.lines()) {
    for (const auto& ev : line.events()) {
      for (const auto& stat : ev.stats()) {
        if (stat.metadata_id() == stat_id) return &stat;
      }
    }
  }
  return nullptr;
}

// Returns the kKernelDetails string of the first kernel event in `plane`, or
// the empty string if there is none. The details value is a reference into the
// plane's stat metadata, so it takes two lookups to reach.
std::string FindKernelDetails(const tensorflow::profiler::XPlane& plane) {
  const auto* stat = FindEventStat(plane, StatType::kKernelDetails);
  if (stat == nullptr) return "";
  auto it = plane.stat_metadata().find(stat->ref_value());
  if (it == plane.stat_metadata().end()) return "";
  return it->second.name();
}

// Helper: build a RocmTraceCollectorImpl and inject a paired API + Activity
// kernel event, optionally with GPU agent data for occupancy.
struct OccupancyTestFixture {
  RocmTraceCollectorImpl collector;

  static RocmTraceCollectorOptions MakeOpts() {
    RocmTraceCollectorOptions o;
    o.max_callback_api_events = 100;
    o.max_activity_api_events = 100;
    o.max_annotation_strings = 100;
    o.num_gpus = 1;
    return o;
  }

  OccupancyTestFixture()
      : collector(MakeOpts(), /*start_walltime_ns=*/1000,
                  /*start_gputime_ns=*/2000) {}

  // Injects a synthetic rocprofiler agent so the occupancy success path can be
  // exercised with no GPU and no ROCm runtime. Only the fields
  // GetDeviceCapabilities reads are set; the rest stay zeroed, and
  // mem_banks_count == 0 keeps the VRAM branch out of the way.
  void SetSyntheticAgent(uint32_t gfx_target_version, uint32_t cu_count) {
    rocprofiler_agent_v0_t agent{};
    agent.gfx_target_version = gfx_target_version;
    agent.cu_count = cu_count;
    // Matching the gfx942 row of the occupancy target table, so the
    // cross-check in GetDeviceCapabilities stays quiet.
    agent.lds_size_in_kb = 64;
    agent.max_waves_per_simd = 8;
    agent.simd_per_cu = 4;
    agent.wave_front_size = 64;
    collector.SetGpuAgents({agent});
  }

  void AddKernelPair(uint32_t arch_vgpr_count, uint32_t wg_x, uint32_t wg_y,
                     uint32_t wg_z, uint32_t smem, uint64_t start_ns,
                     uint64_t end_ns, uint32_t corr_id = 1,
                     uint32_t accum_vgpr_count = 0, uint32_t sgpr_count = 0,
                     uint32_t static_smem = 0) {
    RocmTracerEvent api;
    api.type = RocmTracerEventType::Kernel;
    api.source = RocmTracerEventSource::ApiCallback;
    api.domain = RocmTracerEventDomain::HIP_API;
    api.name = "test_kernel";
    api.correlation_id = corr_id;
    api.thread_id = 1;
    // Zeroed, exactly as the real API-callback path leaves it: the launch
    // parameters arrive on the dispatch record, not on the HIP API callback,
    // and ApiActivityInfoExchange overwrites this side with that one.
    api.kernel_info = KernelDetails{};
    collector.AddEvent(std::move(api), false);

    RocmTracerEvent act;
    act.type = RocmTracerEventType::Kernel;
    act.source = RocmTracerEventSource::Activity;
    act.domain = RocmTracerEventDomain::HIP_OPS;
    act.name = "test_kernel";
    act.correlation_id = corr_id;
    act.start_time_ns = start_ns;
    act.end_time_ns = end_ns;
    act.device_id = 0;
    act.stream_id = 1;
    // The dispatch record is where KernelDetails actually comes from: the
    // workgroup and LDS sizes straight from rocprofiler, the register counts
    // grafted on from the code-object kernel-symbol callback.
    act.kernel_info = KernelDetails{};
    act.kernel_info.arch_vgpr_count = arch_vgpr_count;
    act.kernel_info.accum_vgpr_count = accum_vgpr_count;
    act.kernel_info.sgpr_count = sgpr_count;
    act.kernel_info.workgroup_x = wg_x;
    act.kernel_info.workgroup_y = wg_y;
    act.kernel_info.workgroup_z = wg_z;
    act.kernel_info.group_segment_size = smem;
    act.kernel_info.static_group_segment_size = static_smem;
    collector.AddEvent(std::move(act), false);
  }
};

// A kernel symbol with no VGPRs at all means the code-object callback never
// ran for it (a real kernel always allocates at least one register), so there
// is nothing to model and the occupancy block must not be written.
TEST(RocmCollectorOccupancyTest, ZeroVgprCountSkipsOccupancyStats) {
  OccupancyTestFixture f;
  f.AddKernelPair(/*arch_vgpr_count=*/0, 256, 1, 1, 0, 3000, 4000);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  absl::string_view kOccKey =
      GetStatTypeStr(StatType::kTheoreticalOccupancyPct);
  for (const auto& [id, smd] : gpu->stat_metadata()) {
    EXPECT_NE(smd.name(), kOccKey)
        << "kTheoreticalOccupancyPct must not appear with no register counts";
  }
}

// Without agent data there is no gfx_target_version, and without a target the
// model has no register-file size, no LDS capacity and no granules -- every
// one of which the formula needs. Occupancy must be skipped rather than
// guessed, even though the register count here is perfectly valid.
TEST(RocmCollectorOccupancyTest, NoAgentCapabilitiesSkipsOccupancyStats) {
  OccupancyTestFixture f;
  // Valid registers, but no agent injected, so gfx_target_version_ stays 0.
  f.AddKernelPair(/*arch_vgpr_count=*/32, 256, 1, 1, 0, 3000, 4000);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  absl::string_view kOccKey =
      GetStatTypeStr(StatType::kTheoreticalOccupancyPct);
  for (const auto& [id, smd] : gpu->stat_metadata()) {
    EXPECT_NE(smd.name(), kOccKey)
        << "kTheoreticalOccupancyPct must not appear without a known target";
  }
}

// The suggested-block-size stat is gone: it was computed as
// waves_per_block * wave_front_size, which is just block_size rounded up to a
// wavefront -- an echo of the input, not a suggestion. It must not come back.
TEST(RocmCollectorOccupancyTest, SuggestedBlockSizeIsNeverEmitted) {
  OccupancyTestFixture f;
  // The agent matters here: without it occupancy is skipped entirely and the
  // assertion below would pass for the wrong reason.
  f.SetSyntheticAgent(kGfx942, kCuCount);
  f.AddKernelPair(/*arch_vgpr_count=*/32, 256, 1, 1, 0, 3000, 4000);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  ASSERT_NE(FindEventStat(*gpu, StatType::kTheoreticalOccupancyPct), nullptr)
      << "occupancy was not computed, so this test proves nothing";

  absl::string_view kSuggestedKey =
      GetStatTypeStr(StatType::kOccupancySuggestedBlockSize);
  for (const auto& [id, smd] : gpu->stat_metadata()) {
    EXPECT_NE(smd.name(), kSuggestedKey)
        << "kOccupancySuggestedBlockSize was deliberately removed";
  }
}

// The success path, end to end, with no GPU and no ROCm runtime: a synthetic
// gfx942 agent plus known register counts must land on exactly the number the
// golden table predicts. 128 arch VGPRs (no AGPRs, so the max() branch of the
// unified count applies) granulate to 128, giving 512/128 = 4 waves/SIMD out
// of 8 -> 50%.
TEST(RocmCollectorOccupancyTest, VgprLimitedKernelReportsExactOccupancy) {
  OccupancyTestFixture f;
  f.SetSyntheticAgent(kGfx942, kCuCount);
  f.AddKernelPair(/*arch_vgpr_count=*/128, 256, 1, 1, /*smem=*/0, 3000, 4000);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  const auto* occ = FindEventStat(*gpu, StatType::kTheoreticalOccupancyPct);
  ASSERT_NE(occ, nullptr) << "occupancy stat missing on the success path";
  EXPECT_DOUBLE_EQ(occ->double_value(), 50.0);

  // Whole device, not per CU: 4 workgroups/CU * 304 CUs. The pre-change code
  // reported the per-CU figure (4) under the same stat name, which XProf
  // renders as "min grid size" -- off by the CU count.
  const auto* grid = FindEventStat(*gpu, StatType::kOccupancyMinGridSize);
  ASSERT_NE(grid, nullptr);
  EXPECT_EQ(grid->int64_value(), 4 * kCuCount);
}

// The AGPR fix, end to end -- the reason this PR exists. On gfx942 the
// register file is unified, so an MFMA kernel holding 5 arch VGPRs and 128
// AGPRs really costs alignTo(5,4) + 128 = 132 VGPRs, granulated to 136: three
// waves per SIMD out of eight, 37.5%.
//
// Reading only arch_vgpr_count -- what the collector did before -- sees 5
// registers, concludes the kernel is unconstrained, and reports 100%. That is
// a 2.7x over-report on precisely the kernels a profiler user is opening the
// trace to look at.
TEST(RocmCollectorOccupancyTest, MfmaAgprsReachTheModel) {
  OccupancyTestFixture f;
  f.SetSyntheticAgent(kGfx942, kCuCount);
  f.AddKernelPair(/*arch_vgpr_count=*/5, 256, 1, 1, /*smem=*/0, 3000, 4000,
                  /*corr_id=*/1, /*accum_vgpr_count=*/128);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  const auto* occ = FindEventStat(*gpu, StatType::kTheoreticalOccupancyPct);
  ASSERT_NE(occ, nullptr);
  EXPECT_DOUBLE_EQ(occ->double_value(), 37.5)
      << "AGPRs are not reaching the occupancy model; a kernel with 128 AGPRs "
         "cannot be at full occupancy on a unified register file";

  const auto* grid = FindEventStat(*gpu, StatType::kOccupancyMinGridSize);
  ASSERT_NE(grid, nullptr);
  EXPECT_EQ(grid->int64_value(), 3 * kCuCount);
}

// LDS is the other input that only reaches the model through KernelDetails.
// 32 KiB per workgroup out of gfx942's 64 KiB per CU allows two resident
// workgroups, i.e. 512 of 2048 thread slots -> 25%, well under the 8
// workgroups the 32 VGPRs would otherwise permit.
TEST(RocmCollectorOccupancyTest, LdsLimitedKernelReportsExactOccupancy) {
  OccupancyTestFixture f;
  f.SetSyntheticAgent(kGfx942, kCuCount);
  f.AddKernelPair(/*arch_vgpr_count=*/32, 256, 1, 1, /*smem=*/32768, 3000,
                  4000);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  const auto* occ = FindEventStat(*gpu, StatType::kTheoreticalOccupancyPct);
  ASSERT_NE(occ, nullptr);
  EXPECT_DOUBLE_EQ(occ->double_value(), 25.0);

  const auto* grid = FindEventStat(*gpu, StatType::kOccupancyMinGridSize);
  ASSERT_NE(grid, nullptr);
  EXPECT_EQ(grid->int64_value(), 2 * kCuCount);
}

// kKernelDetails stat must always be present for kernel activity events,
// regardless of whether occupancy was computed.
TEST(RocmCollectorOccupancyTest, KernelDetailsAlwaysPresent) {
  OccupancyTestFixture f;
  f.AddKernelPair(/*arch_vgpr_count=*/0, 64, 1, 1, 512, 3000, 4000);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  absl::string_view kDetailsKey = GetStatTypeStr(StatType::kKernelDetails);
  bool found_details = false;
  for (const auto& [id, smd] : gpu->stat_metadata()) {
    if (smd.name() == kDetailsKey) {
      found_details = true;
      break;
    }
  }
  EXPECT_TRUE(found_details) << "kKernelDetails stat must be present for "
                                "kernel events regardless of occupancy";
}

// When the model declines, the occ_pct token must be ABSENT, not zero.
//
// This test previously asserted the opposite -- that the string contains
// "occ_pct:0" when the model is skipped -- which pinned a real defect in
// place. XProf's kernel-stats table parses this string and sets its occupancy
// column only when it finds the key (kernel_stats_utils.cc), so a literal 0
// there is read as a measured 0%, and the trace-viewer tooltip asserts a
// number nobody computed. rocm_occupancy.h is explicit: a nullopt result means
// the caller "must then emit no occupancy stats, NOT zero ones".
TEST(RocmCollectorOccupancyTest, KernelDetailsOmitsOccupancyWhenNotModelled) {
  OccupancyTestFixture f;
  // No agent -> gfx_target_version_ stays 0 -> the model is never consulted.
  f.AddKernelPair(/*arch_vgpr_count=*/0, 128, 1, 1, 0, 3000, 4000);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  const std::string details = FindKernelDetails(*gpu);
  ASSERT_FALSE(details.empty()) << "kKernelDetails stat not found";
  EXPECT_EQ(details.find("occ_pct"), std::string::npos)
      << "KernelDetails must omit the occ_pct token entirely when the model "
         "declined; got: "
      << details;
  // The rest of the string must still be there -- omission is scoped to the
  // one token, not a bail-out from emitting kernel details at all.
  EXPECT_NE(details.find("regs:"), std::string::npos) << details;
  EXPECT_NE(details.find("block:"), std::string::npos) << details;
}

// A genuine modelled result is still written, including the value 0.0. This is
// the other half of the contract: "could not model" and "modelled as zero"
// must be distinguishable in the string, not just in the numeric stat.
TEST(RocmCollectorOccupancyTest, KernelDetailsKeepsOccupancyWhenModelled) {
  OccupancyTestFixture f;
  f.SetSyntheticAgent(kGfx942, kCuCount);
  f.AddKernelPair(/*arch_vgpr_count=*/5, 256, 1, 1, /*smem=*/0, 3000, 4000,
                  /*corr_id=*/1, /*accum_vgpr_count=*/128);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  const std::string details = FindKernelDetails(*gpu);
  ASSERT_FALSE(details.empty()) << "kKernelDetails stat not found";
  EXPECT_NE(details.find("occ_pct:37.5"), std::string::npos) << details;
}

// A dispatch record with no launch geometry must reach GetOccupancy()'s
// block_size == 0 guard rather than being rounded up into a one-thread
// workgroup. Before this, std::max(workgroup_*, 1u) on all three dimensions
// produced block_size == 1, which the model happily rated at 1.5625% -- a
// confident, non-zero, unfilterable number sitting next to `block:0,0,0` in
// the very same string.
TEST(RocmCollectorOccupancyTest, ZeroWorkgroupGeometrySkipsOccupancy) {
  OccupancyTestFixture f;
  f.SetSyntheticAgent(kGfx942, kCuCount);
  // Registers present (so the outer guard passes) but every workgroup
  // dimension zero, exactly as a truncated dispatch record arrives.
  f.AddKernelPair(/*arch_vgpr_count=*/64, /*wg_x=*/0, /*wg_y=*/0, /*wg_z=*/0,
                  /*smem=*/0, 3000, 4000);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  const std::string details = FindKernelDetails(*gpu);
  ASSERT_FALSE(details.empty()) << "kKernelDetails stat not found";
  EXPECT_EQ(details.find("occ_pct"), std::string::npos)
      << "a zeroed workgroup size must not produce an occupancy figure; got: "
      << details;

  // ...and neither of the numeric occupancy stats may appear either.
  EXPECT_EQ(FindEventStat(*gpu, StatType::kTheoreticalOccupancyPct), nullptr)
      << "kTheoreticalOccupancyPct emitted for a dispatch with no geometry";
  EXPECT_EQ(FindEventStat(*gpu, StatType::kOccupancyMinGridSize), nullptr)
      << "kOccupancyMinGridSize emitted for a dispatch with no geometry";
}

// The `regs:` token must carry the same unified charge the occupancy model
// used, not arch_vgpr_count. A tooltip reading "regs:5 ... occ_pct:37.5" reads
// as a bug in the occupancy number; "regs:136 ... occ_pct:37.5" is the
// explanation for it. This is also the only place the AGPR count becomes
// visible to a user, which is the point of the whole change.
TEST(RocmCollectorOccupancyTest, KernelDetailsRegsIsTheUnifiedVgprCount) {
  OccupancyTestFixture f;
  f.SetSyntheticAgent(kGfx942, kCuCount);
  f.AddKernelPair(/*arch_vgpr_count=*/5, 256, 1, 1, /*smem=*/0, 3000, 4000,
                  /*corr_id=*/1, /*accum_vgpr_count=*/128);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  const std::string details = FindKernelDetails(*gpu);
  ASSERT_FALSE(details.empty()) << "kKernelDetails stat not found";
  // Leading token, matching the CUDA collector's string layout.
  EXPECT_EQ(details.rfind("regs:136 ", 0), 0u)
      << "expected the string to start with the unified count 136 (= "
         "alignTo(5,4) + 128); got: "
      << details;
  EXPECT_NE(details.find("occ_pct:37.5"), std::string::npos) << details;
}

// The dispatch record reports total LDS and the code-object symbol reports the
// static half; the string carries both, under CUDA's names, so XProf shows the
// same two fields for both vendors. This is the only reader of
// KernelDetails::static_group_segment_size -- if this test goes, the field
// should go with it.
TEST(RocmCollectorOccupancyTest, KernelDetailsSplitsStaticAndDynamicLds) {
  OccupancyTestFixture f;
  f.SetSyntheticAgent(kGfx942, kCuCount);
  f.AddKernelPair(/*arch_vgpr_count=*/8, 256, 1, 1, /*smem=*/16384, 3000, 4000,
                  /*corr_id=*/1, /*accum_vgpr_count=*/0, /*sgpr_count=*/0,
                  /*static_smem=*/4096);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  const std::string details = FindKernelDetails(*gpu);
  ASSERT_FALSE(details.empty()) << "kKernelDetails stat not found";
  EXPECT_NE(details.find(" static_shared:4096 "), std::string::npos) << details;
  EXPECT_NE(details.find(" dynamic_shared:12288 "), std::string::npos)
      << details;
  // The superseded total must not linger alongside the split.
  EXPECT_EQ(details.find("group_mem:"), std::string::npos) << details;
}

// When the code-object symbol lookup misses, the static figure is 0 and the
// whole allocation is reported as dynamic rather than silently vanishing.
TEST(RocmCollectorOccupancyTest, UnknownStaticLdsIsReportedAsDynamic) {
  OccupancyTestFixture f;
  f.SetSyntheticAgent(kGfx942, kCuCount);
  f.AddKernelPair(/*arch_vgpr_count=*/8, 256, 1, 1, /*smem=*/16384, 3000, 4000);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  const std::string details = FindKernelDetails(*gpu);
  ASSERT_FALSE(details.empty()) << "kKernelDetails stat not found";
  EXPECT_NE(details.find(" static_shared:0 "), std::string::npos) << details;
  EXPECT_NE(details.find(" dynamic_shared:16384 "), std::string::npos)
      << details;
}

// Without a modellable target there is no way to know how the arch and accum
// files combine, so `regs:` falls back to max() rather than guessing a sum.
// The token still has to be there -- the register count is useful even when
// the occupancy number is not available.
TEST(RocmCollectorOccupancyTest, KernelDetailsRegsPresentWithoutAgent) {
  OccupancyTestFixture f;  // no agent, so no target constants
  f.AddKernelPair(/*arch_vgpr_count=*/5, 256, 1, 1, /*smem=*/0, 3000, 4000,
                  /*corr_id=*/1, /*accum_vgpr_count=*/128);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  const std::string details = FindKernelDetails(*gpu);
  ASSERT_FALSE(details.empty());
  EXPECT_EQ(details.rfind("regs:128 ", 0), 0u) << details;
  // With no agent there is no target, so the model is never consulted and the
  // occ_pct token is omitted -- the register count still has to be there,
  // which is what this test is actually about. (It asserted "occ_pct:0" here
  // before; that was the same defect KernelDetailsOmitsOccupancyWhenNotModelled
  // covers, asserted a second time as an aside.)
  EXPECT_EQ(details.find("occ_pct"), std::string::npos) << details;
}

// The direct formula tests that used to live here -- FormulaGfx942VgprLimited,
// FormulaGfx942FullOccupancy, FormulaLdsLimited, FormulaZeroParamsReturnsEmpty
// and OccupancyParamsCacheKey -- moved to rocm_occupancy_test.cc, along with
// the model itself. They needed no GPU and no ROCm toolchain, but sat in a
// target tagged requires-gpu-amd, so they ran nowhere near often enough. The
// replacements there cover 16 golden rows instead of 3, and run in CPU CI.

}  // namespace test
}  // namespace profiler
}  // namespace xla
