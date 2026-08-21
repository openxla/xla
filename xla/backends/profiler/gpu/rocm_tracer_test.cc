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

#include "xla/backends/profiler/gpu/rocm_tracer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/rocprofiler-sdk/callback_tracing.h"
#include "rocm/include/rocprofiler-sdk/context.h"
#include "rocm/include/rocprofiler-sdk/fwd.h"
#include "rocm/include/rocprofiler-sdk/marker.h"
#include "xla/backends/profiler/gpu/rocm_collector.h"
#include "xla/backends/profiler/gpu/rocm_occupancy_test_kernel.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

using tsl::profiler::XSpace;

// Minimal mock collector implementation based on RocmTraceCollectorImpl.
class TestRocmTraceCollector : public RocmTraceCollectorImpl {
 public:
  TestRocmTraceCollector(const RocmTraceCollectorOptions& options,
                         uint64_t start_walltime_ns, uint64_t start_gputime_ns)
      : RocmTraceCollectorImpl(options, start_walltime_ns, start_gputime_ns) {}

  void Export(XSpace* space) override {
    exported_ = true;
    exported_space_ = space;
  }

  void OnEventsDropped(const std::string& reason,
                       uint64_t correlation_id) override {
    dropped_reason_ = reason;
    dropped_id_ = correlation_id;
  }

  bool exported() const { return exported_; }
  const std::string& dropped_reason() const { return dropped_reason_; }
  uint64_t dropped_id() const { return dropped_id_; }
  XSpace* exported_space() const { return exported_space_; }

 private:
  bool exported_ = false;
  std::string dropped_reason_;
  uint64_t dropped_id_ = 0;
  XSpace* exported_space_ = nullptr;
};

// Utility to create valid options for the test collector.
std::unique_ptr<TestRocmTraceCollector> CreateTestCollector() {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 2 * 1024 * 1024;
  options.max_activity_api_events = 2 * 1024 * 1024;
  options.max_annotation_strings = 1024 * 1024;
  options.num_gpus = 1;

  uint64_t walltime_ns = RocmTracer::GetTimestamp();
  uint64_t gputime_ns = RocmTracer::GetTimestamp();

  return std::make_unique<TestRocmTraceCollector>(options, walltime_ns,
                                                  gputime_ns);
}

TEST(RocmTracerTest, SingletonInstance) {
  LOG(INFO) << "Before RocmTracer::GetRocmTracerSingleton()";
  RocmTracer& tracer1 = RocmTracer::GetRocmTracerSingleton();
  RocmTracer& tracer2 = RocmTracer::GetRocmTracerSingleton();
  LOG(INFO) << "Before RocmTracer::GetRocmTracerSingleton()";
  EXPECT_EQ(&tracer1, &tracer2) << "RocmTracer must be a singleton";
}

TEST(RocmTracerTest, GpuAgentDataMatchesHipDeviceProperties) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  const auto& agents = tracer.GpuAgents();
  ASSERT_GT(agents.size(), 0u);

  hipDeviceProp_t props;
  ASSERT_EQ(hipGetDeviceProperties(&props, 0), hipSuccess);
  const auto& agent = agents[0];

  EXPECT_EQ(agent.cu_count, static_cast<uint32_t>(props.multiProcessorCount));
  // Agent clocks are in MHz, hipDeviceProp_t clocks are in KHz.
  EXPECT_EQ(static_cast<uint64_t>(agent.max_engine_clk_fcompute) * 1000,
            static_cast<uint64_t>(props.clockRate));

  auto gfx_major = (agent.gfx_target_version / 10000) % 100;
  auto gfx_minor = (agent.gfx_target_version / 100) % 100;
  EXPECT_EQ(gfx_major, static_cast<uint32_t>(props.major));
  EXPECT_EQ(gfx_minor, static_cast<uint32_t>(props.minor));

  uint64_t vram_total = 0;
  uint32_t vram_clock_mhz = 0;
  uint32_t vram_bus_width = 0;
  for (uint32_t i = 0; i < agent.mem_banks_count; ++i) {
    if (agent.mem_banks[i].heap_type == HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC ||
        agent.mem_banks[i].heap_type == HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE) {
      vram_total += agent.mem_banks[i].size_in_bytes;
      if (vram_clock_mhz == 0) {
        vram_clock_mhz = agent.mem_banks[i].mem_clk_max;
        vram_bus_width = agent.mem_banks[i].width;
      }
    }
  }
  EXPECT_EQ(vram_total, props.totalGlobalMem);
  EXPECT_EQ(static_cast<uint64_t>(vram_clock_mhz) * 1000,
            static_cast<uint64_t>(props.memoryClockRate));
  EXPECT_EQ(vram_bus_width, static_cast<uint32_t>(props.memoryBusWidth));
}

TEST(RocmTracerTest, InitialStateIsAvailable) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  EXPECT_TRUE(tracer.IsAvailable())
      << "Tracer should be available before Enable()";
}

TEST(RocmTracerTest, EnableAndDisableLifecycle) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  auto collector = CreateTestCollector();

  RocmTracerOptions tracer_options{/*max_annotation_strings=*/128};
  TF_ASSERT_OK(tracer.Enable(tracer_options, collector.get()));

  EXPECT_FALSE(tracer.IsAvailable())
      << "Tracer should not be available after Enable()";
  EXPECT_EQ(tracer.collector(), collector.get())
      << "Collector should be set after Enable()";
  ASSERT_NE(tracer.annotation_map(), nullptr)
      << "Annotation map should be initialized";

  tracer.Disable();

  EXPECT_TRUE(tracer.IsAvailable())
      << "Tracer should be available after Disable()";
}

TEST(RocmTracerTest, AnnotationMapWorks) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  AnnotationMap* map = tracer.annotation_map();
  ASSERT_NE(map, nullptr);

  uint64_t id = 42;
  std::string annotation = "matmul_fused_op";
  map->Add(id, annotation);

  absl::string_view result = map->LookUp(id);
  EXPECT_EQ(result, annotation);
}

TEST(RocmTracerTest, AnnotationMapClear) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  AnnotationMap* map = tracer.annotation_map();
  ASSERT_NE(map, nullptr);

  map->Add(100, "op_a");
  map->Add(101, "op_b");
  EXPECT_EQ(map->LookUp(100), "op_a");
  EXPECT_EQ(map->LookUp(101), "op_b");

  map->Clear();

  EXPECT_TRUE(map->LookUp(100).empty());
  EXPECT_TRUE(map->LookUp(101).empty());
}

// Simple collector that tracks received events for verification.
class EventCapturingCollector : public RocmTraceCollector {
 public:
  EventCapturingCollector() : RocmTraceCollector(MakeCollectorOptions()) {}

  void AddEvent(RocmTracerEvent&& event, bool is_auxiliary) override {
    event_count_++;
  }

  void OnEventsDropped(const std::string& reason,
                       uint64_t num_events) override {}
  void Flush() override {}
  void Export(tsl::profiler::XSpace* space) override {}

  int event_count() const { return event_count_; }

 private:
  static RocmTraceCollectorOptions MakeCollectorOptions() {
    RocmTraceCollectorOptions options;
    options.max_callback_api_events = 2 * 1024 * 1024;
    options.max_activity_api_events = 2 * 1024 * 1024;
    options.max_annotation_strings = 1024 * 1024;
    options.num_gpus = RocmTracer::GetRocmTracerSingleton().NumGpus();
    return options;
  }
  int event_count_ = 0;
};

std::unique_ptr<EventCapturingCollector> CreateEventCapturingCollector() {
  return std::make_unique<EventCapturingCollector>();
}

TEST(RocmTracerTest, CapturesHipEvents) {
#define HIP_ASSERT_OK(expr) ASSERT_EQ((expr), hipSuccess) << #expr " failed"

  int device_count = 0;
  HIP_ASSERT_OK(hipGetDeviceCount(&device_count));
  ASSERT_GT(device_count, 0) << "No HIP devices available";

  auto collector = CreateEventCapturingCollector();
  EventCapturingCollector* collector_ptr = collector.get();

  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  RocmTracerOptions tracer_options{/*max_annotation_strings=*/1024 * 1024};
  TF_ASSERT_OK(tracer.Enable(tracer_options, collector.get()));

  constexpr size_t kNumFloats = 1024;
  constexpr size_t kSize = kNumFloats * sizeof(float);
  std::vector<float> host_data(kNumFloats, 1.0f);
  void* device_data = nullptr;

  HIP_ASSERT_OK(hipMalloc(&device_data, kSize));
  HIP_ASSERT_OK(
      hipMemcpy(device_data, host_data.data(), kSize, hipMemcpyHostToDevice));
  HIP_ASSERT_OK(
      hipMemcpy(host_data.data(), device_data, kSize, hipMemcpyDeviceToHost));
  HIP_ASSERT_OK(hipDeviceSynchronize());

  tracer.Disable();
  HIP_ASSERT_OK(hipFree(device_data));

#undef HIP_ASSERT_OK

  EXPECT_GT(collector_ptr->event_count(), 0)
      << "Expected to capture at least one trace event";
}

// Regression guards: Disable() must stop the rocprofiler context it started
// in Enable(). Otherwise the buffer keeps collecting events between sessions
// and the next Enable()'s collector receives stale events.

TEST(RocmTracerTest, DisableStopsRocprofilerContext) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  ASSERT_TRUE(tracer.IsAvailable());

  auto collector = CreateTestCollector();
  RocmTracerOptions tracer_options{/*max_annotation_strings=*/128};
  TF_ASSERT_OK(tracer.Enable(tracer_options, collector.get()));

  int active = -1;
  ASSERT_EQ(rocprofiler_context_is_active(tracer.context_, &active),
            ROCPROFILER_STATUS_SUCCESS);
  EXPECT_NE(active, 0) << "Context should be active after Enable()";

  tracer.Disable();

  active = -1;
  ASSERT_EQ(rocprofiler_context_is_active(tracer.context_, &active),
            ROCPROFILER_STATUS_SUCCESS);
  EXPECT_EQ(active, 0)
      << "Disable() should call rocprofiler_stop_context(context_)";
}

TEST(RocmTracerTest, DisableIsolatesNextSession) {
  int device_count = 0;
  ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
  ASSERT_GT(device_count, 0) << "No HIP devices available";

  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  ASSERT_TRUE(tracer.IsAvailable());

  RocmTracerOptions tracer_options{/*max_annotation_strings=*/1024 * 1024};
  constexpr size_t kNumFloats = 1024;
  constexpr size_t kSize = kNumFloats * sizeof(float);
  std::vector<float> host_data(kNumFloats, 1.0f);
  void* device_data = nullptr;
  ASSERT_EQ(hipMalloc(&device_data, kSize), hipSuccess);

  // Session 1: minimal Enable -> Disable to put the rocprofiler context
  // into the post-Disable state. The 100 ms sleep before Disable lets the
  // async HIP_OPS activity record land in the buffer in time for the flush.
  auto collector1 = CreateEventCapturingCollector();
  TF_ASSERT_OK(tracer.Enable(tracer_options, collector1.get()));
  ASSERT_EQ(
      hipMemcpy(device_data, host_data.data(), kSize, hipMemcpyHostToDevice),
      hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  absl::SleepFor(absl::Milliseconds(100));
  tracer.Disable();
  ASSERT_GT(collector1->event_count(), 0)
      << "Sanity: profiler should capture events during a normal session";

  // No profiler. If Disable() stopped the context correctly, these HIP calls
  // must not be recorded into the rocprofiler-owned buffer.
  constexpr int kLeakedPairs = 50;
  for (int i = 0; i < kLeakedPairs; ++i) {
    ASSERT_EQ(
        hipMemcpy(device_data, host_data.data(), kSize, hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpy(host_data.data(), device_data, kSize, hipMemcpyDeviceToHost),
        hipSuccess);
  }
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  // Session 2: Enable -> Disable with no user HIP activity. With the fix
  // the buffer is empty here so collector2 receives zero events; with the
  // bug, leaked-window events drain into collector2.
  auto collector2 = CreateEventCapturingCollector();
  TF_ASSERT_OK(tracer.Enable(tracer_options, collector2.get()));
  tracer.Disable();

  ASSERT_EQ(hipFree(device_data), hipSuccess);

  EXPECT_EQ(collector2->event_count(), 0)
      << "Session 2 captured " << collector2->event_count()
      << " events despite no HIP activity between its Enable() and Disable();"
      << " these must have leaked from the preceding no-profiler window of "
      << kLeakedPairs << " hipMemcpy pairs";
}

// ============================================================================
// Occupancy smoke tests — require real AMD GPU hardware.
//
// These tests launch a real HIP kernel via RunOccupancyTestKernel() (defined
// in rocm_occupancy_test_kernel.hip.cc, compiled by hipcc so __global__ and
// <<<...>>> are available there), capture the profiler event through the full
// rocprofiler-sdk pipeline, and verify that the exported XSpace contains
// theoretical occupancy statistics (kTheoreticalOccupancyPct and
// kOccupancyMinGridSize).
//
// The arithmetic itself is NOT tested here -- it lives in rocm_occupancy.cc
// and is pinned by a 16-row golden table in rocm_occupancy_test.cc, which runs
// on any CPU host. What these tests cover is the plumbing that the golden
// table cannot: that the code-object callback actually fires, that the symbol's
// register counts reach KernelDetails, and that the agent's gfx_target_version
// reaches PerDeviceCollector.
// ============================================================================

TEST(RocmTracerOccupancyTest, KernelDispatchProducesOccupancyStats) {
  int device_count = 0;
  ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
  ASSERT_GT(device_count, 0) << "No HIP devices available";

  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  ASSERT_TRUE(tracer.IsAvailable());
  ASSERT_GT(tracer.GpuAgents().size(), 0u) << "No GPU agents registered";

  RocmTraceCollectorOptions col_opts;
  col_opts.max_callback_api_events = 2 * 1024 * 1024;
  col_opts.max_activity_api_events = 2 * 1024 * 1024;
  col_opts.max_annotation_strings = 1024 * 1024;
  col_opts.num_gpus = tracer.NumGpus() > 0 ? tracer.NumGpus() : 1;

  uint64_t start_wall = RocmTracer::GetTimestamp();
  uint64_t start_gpu = RocmTracer::GetTimestamp();
  auto collector =
      std::make_unique<RocmTraceCollectorImpl>(col_opts, start_wall, start_gpu);
  collector->SetGpuAgents(tracer.GpuAgents());

  RocmTracerOptions opts{/*max_annotation_strings=*/1024 * 1024};
  TF_ASSERT_OK(tracer.Enable(opts, collector.get()));

  // Launch a real kernel via the hipcc-compiled helper so the tracer captures
  // a kernel dispatch record with a valid kernel_object / func_ptr.
  const int launch_status =
      test::RunOccupancyTestKernel(/*n=*/4096, /*block_size=*/256);

  // Allow the rocprofiler-sdk buffer to flush.
  absl::SleepFor(absl::Milliseconds(200));
  // Disable BEFORE asserting. The tracer is a process-lifetime singleton
  // holding a raw pointer to `collector`; a failed ASSERT returns from the
  // test body immediately, which would destroy the collector and leave the
  // singleton dangling for the next in-flight buffer callback or the next
  // test in this binary. See DisableIsolatesNextSession.
  tracer.Disable();
  ASSERT_EQ(launch_status, 0);

  tsl::profiler::XSpace space;
  collector->Export(&space);

  // Look for the theoretical occupancy stat in any GPU plane.
  absl::string_view kOccKey = tsl::profiler::GetStatTypeStr(
      tsl::profiler::StatType::kTheoreticalOccupancyPct);
  absl::string_view kMinGridKey = tsl::profiler::GetStatTypeStr(
      tsl::profiler::StatType::kOccupancyMinGridSize);
  // Deliberately absent. The old suggested_block_size was block_size rounded
  // up to a wavefront -- an echo of the input, not a suggestion -- and was
  // deleted rather than fixed.
  absl::string_view kSugBlockKey = tsl::profiler::GetStatTypeStr(
      tsl::profiler::StatType::kOccupancySuggestedBlockSize);

  bool found_occ_pct = false;
  bool found_min_grid = false;
  bool found_sug_block = false;

  for (const auto& plane : space.planes()) {
    for (const auto& [id, smd] : plane.stat_metadata()) {
      if (smd.name() == kOccKey) found_occ_pct = true;
      if (smd.name() == kMinGridKey) found_min_grid = true;
      if (smd.name() == kSugBlockKey) found_sug_block = true;
    }
  }

  EXPECT_TRUE(found_occ_pct)
      << "XSpace must contain kTheoreticalOccupancyPct stat after a real "
         "kernel dispatch — check that the code-object callback populated "
         "arch_vgpr_count in KernelEvent() and that the agent's "
         "gfx_target_version reached PerDeviceCollector (and that this device "
         "is a target LookupTargetConstants knows: the model covers gfx9 only)";
  EXPECT_TRUE(found_min_grid)
      << "XSpace must contain kOccupancyMinGridSize stat";
  EXPECT_FALSE(found_sug_block)
      << "kOccupancySuggestedBlockSize was deliberately removed; it must not "
         "come back";

  // Verify that the kTheoreticalOccupancyPct value is in (0, 100].
  for (const auto& plane : space.planes()) {
    int64_t occ_stat_id = -1;
    for (const auto& [sid, smd] : plane.stat_metadata()) {
      if (smd.name() == kOccKey) {
        occ_stat_id = sid;
        break;
      }
    }
    if (occ_stat_id < 0) continue;

    for (const auto& line : plane.lines()) {
      for (const auto& ev : line.events()) {
        for (const auto& stat : ev.stats()) {
          if (stat.metadata_id() != occ_stat_id) continue;
          double pct = stat.double_value();
          EXPECT_GT(pct, 0.0)
              << "Theoretical occupancy must be positive for a running kernel";
          EXPECT_LE(pct, 100.0) << "Theoretical occupancy cannot exceed 100%";
        }
      }
    }
  }
}

// Verify that kKernelDetails in the XSpace contains a non-zero occ_pct
// string when occupancy is available.
TEST(RocmTracerOccupancyTest, KernelDetailsStringContainsOccupancyPct) {
  int device_count = 0;
  ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
  ASSERT_GT(device_count, 0) << "No HIP devices available";

  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  ASSERT_TRUE(tracer.IsAvailable());

  RocmTraceCollectorOptions col_opts;
  col_opts.max_callback_api_events = 2 * 1024 * 1024;
  col_opts.max_activity_api_events = 2 * 1024 * 1024;
  col_opts.max_annotation_strings = 1024 * 1024;
  col_opts.num_gpus = tracer.NumGpus() > 0 ? tracer.NumGpus() : 1;

  uint64_t start_wall = RocmTracer::GetTimestamp();
  uint64_t start_gpu = RocmTracer::GetTimestamp();
  auto collector =
      std::make_unique<RocmTraceCollectorImpl>(col_opts, start_wall, start_gpu);
  collector->SetGpuAgents(tracer.GpuAgents());

  RocmTracerOptions opts{/*max_annotation_strings=*/1024 * 1024};
  TF_ASSERT_OK(tracer.Enable(opts, collector.get()));

  const int launch_status =
      test::RunOccupancyTestKernel(/*n=*/1024, /*block_size=*/64);

  absl::SleepFor(absl::Milliseconds(200));
  // Disable before asserting -- see the note in the test above.
  tracer.Disable();
  ASSERT_EQ(launch_status, 0);

  tsl::profiler::XSpace space;
  collector->Export(&space);

  absl::string_view kDetailsKey =
      tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kKernelDetails);

  bool found_nonzero_occ = false;
  for (const auto& plane : space.planes()) {
    int64_t details_id = -1;
    for (const auto& [sid, smd] : plane.stat_metadata()) {
      if (smd.name() == kDetailsKey) {
        details_id = sid;
        break;
      }
    }
    if (details_id < 0) continue;

    for (const auto& line : plane.lines()) {
      for (const auto& ev : line.events()) {
        for (const auto& stat : ev.stats()) {
          if (stat.metadata_id() != details_id) continue;
          int64_t ref = stat.ref_value();
          auto it = plane.stat_metadata().find(ref);
          if (it == plane.stat_metadata().end()) continue;
          const std::string& details = it->second.name();
          // The details string must contain "occ_pct:" followed by a non-zero
          // value when occupancy was computed successfully.
          auto pos = details.find("occ_pct:");
          if (pos == std::string::npos) continue;
          double pct = std::stod(details.substr(pos + 8));
          if (pct <= 0.0) continue;
          found_nonzero_occ = true;
          // A non-zero occ_pct means GetOccupancy() ran, which it only does
          // when at least one register count is non-zero -- so the `regs:`
          // token on this same event must be present and non-zero. This is
          // the only place the real code-object register decode is exercised;
          // the golden table feeds the model synthetic counts.
          ASSERT_EQ(details.rfind("regs:", 0), 0u)
              << "kernel_details must lead with the regs: token: " << details;
          EXPECT_GT(std::stoul(details.substr(5)), 0u)
              << "occ_pct is non-zero, so the unified VGPR charge cannot be "
                 "zero: "
              << details;
        }
      }
    }
  }

  EXPECT_TRUE(found_nonzero_occ)
      << "kKernelDetails string must contain a non-zero occ_pct after a real "
         "kernel dispatch with occupancy support enabled";
}

}  // namespace
}  // namespace profiler
}  // namespace xla
