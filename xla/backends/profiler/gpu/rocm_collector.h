/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_COLLECTOR_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_COLLECTOR_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocprofiler-sdk/agent.h"
#include "xla/backends/profiler/gpu/rocm_occupancy.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

// `regs_per_thread` is the UNIFIED arch+accum VGPR charge from
// UnifiedVgprCount(), not kernel_info.arch_vgpr_count. Reporting the arch
// count alone next to occ_pct would show an MFMA kernel holding 5 registers
// and running at 37.5%, which reads as a bug in the occupancy number.
//
// Token names and order follow the CUDA collector's kernel-details string
// (cupti_buffer_events.h ToXStat) wherever the concept exists on both
// vendors, so one XProf tooltip parses both: `regs:`, `static_shared:`,
// `dynamic_shared:`, `grid:`, `block:`, `occ_pct:`. The single ROCm-only
// token is `private_mem:` (scratch per work-item), which has no CUPTI
// counterpart. The former `group_mem:` was the static+dynamic total; it is
// split here rather than reported alongside, because the sum carries no
// information the two parts do not and a third LDS number in the same string
// invites misreading.
//
// `occupancy_pct` is optional because GetOccupancy() is: nullopt means the
// model declined to answer (unrecognised target, missing symbol data, a
// geometry the hardware could not have made resident). In that case the
// `occ_pct:` token is omitted entirely rather than written as 0. Emitting a
// zero here would be read downstream as a measured 0% -- XProf's kernel-stats
// table parses this very string (kernel_stats_utils.cc) and sets
// occupancy_pct only when it finds the key -- and it would contradict
// rocm_occupancy.h's contract that a caller "must then emit no occupancy
// stats, NOT zero ones". A genuine modelled 0.0 is still written.
inline std::string ToXStat(const KernelDetails& kernel_info,
                           uint32_t regs_per_thread,
                           std::optional<double> occupancy_pct) {
  uint32_t grid_x = kernel_info.workgroup_x != 0
                        ? kernel_info.grid_x / kernel_info.workgroup_x
                        : 0,
           grid_y = kernel_info.workgroup_y != 0
                        ? kernel_info.grid_y / kernel_info.workgroup_y
                        : 0,
           grid_z = kernel_info.workgroup_z != 0
                        ? kernel_info.grid_z / kernel_info.workgroup_z
                        : 0;

  // The dispatch record carries the total LDS per workgroup; the code-object
  // symbol carries the static half. A zero static figure means the symbol
  // lookup missed, in which case attributing the whole total to `dynamic` is
  // the honest reading -- we know what was allocated, not how it was declared.
  const uint32_t static_shared = kernel_info.static_group_segment_size;
  const uint32_t dynamic_shared =
      kernel_info.group_segment_size > static_shared
          ? kernel_info.group_segment_size - static_shared
          : 0;

  std::string stat =
      absl::StrCat("regs:", regs_per_thread, " static_shared:", static_shared,
                   " dynamic_shared:", dynamic_shared, " grid:", grid_x, ",",
                   grid_y, ",", grid_z, " block:", kernel_info.workgroup_x, ",",
                   kernel_info.workgroup_y, ",", kernel_info.workgroup_z,
                   " private_mem:", kernel_info.private_segment_size);
  if (occupancy_pct.has_value()) {
    absl::StrAppend(&stat, " occ_pct:", *occupancy_pct);
  }
  return stat;
}

// RocmDeviceOccupancyParams, OccupancyStats, OccupancyLimiter and
// GetOccupancy() live in rocm_occupancy.h, which has no ROCm includes so that
// the model can be unit-tested on a CPU-only host.

class RocmTraceCollector {
 public:
  explicit RocmTraceCollector(const RocmTraceCollectorOptions& options)
      : options_(options) {}
  virtual ~RocmTraceCollector() {}

  // Agent data used by GetDeviceCapabilities instead of hipGetDeviceProperties
  // (which can set sticky hipGetLastError for non-visible devices on ROCm 7+).
  virtual void SetGpuAgents(std::vector<rocprofiler_agent_v0_t> /*agents*/) {}
  virtual void AddEvent(RocmTracerEvent&& event, bool is_auxiliary) = 0;
  virtual void OnEventsDropped(const std::string& reason,
                               uint64_t num_events) = 0;
  virtual void Flush() = 0;
  virtual void Export(tsl::profiler::XSpace* space) = 0;
  virtual void SetScopeRangeIdTree(ScopeRangeIdTree tree) {}

 protected:
  RocmTraceCollectorOptions options_;

 public:
  // Disable copy and move.
  RocmTraceCollector(const RocmTraceCollector&) = delete;
  RocmTraceCollector& operator=(const RocmTraceCollector&) = delete;
};

class PerDeviceCollector {
 public:
  void Export(uint64_t start_walltime_ns, uint64_t start_gputime_ns,
              uint64_t end_gputime_ns,
              tsl::profiler::XPlaneBuilder* device_plane,
              tsl::profiler::XPlaneBuilder* host_plane);

  PerDeviceCollector() = default;

  void AddEvent(RocmTracerEvent&& event);
  void GetDeviceCapabilities(const rocprofiler_agent_v0_t& agent,
                             tsl::profiler::XPlaneBuilder* device_plane);

 private:
  void CreateXEvent(const RocmTracerEvent& event,
                    tsl::profiler::XPlaneBuilder* plane, uint64_t start_gpu_ns,
                    uint64_t end_gpu_ns, tsl::profiler::XLineBuilder* line);
  void SortByStartTime();
  bool IsHostEvent(const RocmTracerEvent& event, int64_t* line_id);

 private:
  absl::Mutex events_mutex_;
  std::vector<RocmTracerEvent> events_ ABSL_GUARDED_BY(events_mutex_);
  // The fields below are written once in GetDeviceCapabilities() and read in
  // CreateXEvent(), both called sequentially from Export() after Flush().
  // No concurrent access, so no mutex guard is needed.
  //
  // A nullopt mapped value means "we tried and this launch cannot be modelled",
  // NOT "not computed yet" -- membership in the map is what says "computed".
  // Using the optional itself as the not-yet-computed sentinel would make every
  // unmodelable kernel recompute on every event, which is the cost this cache
  // exists to avoid.
  absl::flat_hash_map<RocmDeviceOccupancyParams, std::optional<OccupancyStats>>
      occupancy_cache_;
  // Kept only so the occupancy model can be keyed and the agent values
  // cross-checked against the per-target table; see GetDeviceCapabilities().
  uint32_t gfx_target_version_ = 0;
  uint32_t cu_count_ = 0;
  // Resolved once from gfx_target_version_, alongside the cross-check that
  // needs it anyway. nullopt means "target we cannot model" -- gfx10+, where
  // ToXStat falls back to a non-unified register count.
  std::optional<AmdGpuTargetConstants> target_constants_;
};  // PerDeviceCollector

class RocmTraceCollectorImpl : public RocmTraceCollector {
 public:
  RocmTraceCollectorImpl(const RocmTraceCollectorOptions& options,
                         uint64_t start_walltime_ns, uint64_t start_gputime_ns)
      : RocmTraceCollector(options),
        num_callback_events_(0),
        num_activity_events_(0),
        start_walltime_ns_(start_walltime_ns),
        start_gputime_ns_(start_gputime_ns),
        num_gpus_(options.num_gpus) {}

  void SetGpuAgents(std::vector<rocprofiler_agent_v0_t> agents) override {
    gpu_agents_ = std::move(agents);
  }

  void AddEvent(RocmTracerEvent&& event, bool is_auxiliary) override;
  void Flush() override;
  void Export(tsl::profiler::XSpace* space) override;
  void SetScopeRangeIdTree(ScopeRangeIdTree tree) override {
    scope_range_id_tree_ = std::move(tree);
  }

  void OnEventsDropped(const std::string& reason,
                       uint64_t correlation_id) override {
    VLOG(2) << "RocmTracerEvent dropped (correlation_id=" << correlation_id
            << ",) : " << reason << ".";
  }

 private:
  std::atomic<int> num_callback_events_;
  std::atomic<int> num_activity_events_;
  uint64_t start_walltime_ns_;
  uint64_t start_gputime_ns_;
  int num_gpus_;
  std::vector<rocprofiler_agent_v0_t> gpu_agents_;

  absl::Mutex event_maps_mutex_;
  absl::flat_hash_map<uint64_t, RocmTracerEvent> api_events_map_
      ABSL_GUARDED_BY(event_maps_mutex_);

  /* Some apis such as MEMSETD32 (based on an observation with ResNet50),
   trigger multiple HIP ops domain activities. We keep them in a vector and
   merge them with api activities at flush time.
 */
  absl::flat_hash_map<uint64_t, std::vector<RocmTracerEvent>>
      activity_ops_events_map_ ABSL_GUARDED_BY(event_maps_mutex_);
  // This is for the APIs that we track because we need some information from
  // them to populate the corresponding activity that we actually track.
  absl::flat_hash_map<uint64_t, RocmTracerEvent> auxiliary_api_events_map_
      ABSL_GUARDED_BY(event_maps_mutex_);

  std::vector<RocmTracerEvent> ApiActivityInfoExchange()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(event_maps_mutex_);

  void ExportScopeRangeIdTree(tsl::profiler::XSpace* space);

  absl::node_hash_map<uint32_t, PerDeviceCollector> per_device_collector_;
  ScopeRangeIdTree scope_range_id_tree_;
};  // RocmTraceCollectorImpl

std::unique_ptr<RocmTraceCollector> CreateRocmCollector(
    const RocmTraceCollectorOptions& options, uint64_t start_walltime_ns,
    uint64_t start_gputime_ns);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_COLLECTOR_H_
