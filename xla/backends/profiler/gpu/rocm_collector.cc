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

#include "xla/backends/profiler/gpu/rocm_collector.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/rocprofiler-sdk/fwd.h"
#include "rocm/include/rocprofiler-sdk/rocprofiler.h"
#include "xla/backends/profiler/gpu/rocm_occupancy.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/profiler/utils/parse_annotation.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/platform/abi.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

using tsl::Status;
using tsl::profiler::Annotation;
using tsl::profiler::FindOrAddMutablePlaneWithName;
using tsl::profiler::GetStatTypeStr;
using tsl::profiler::GpuPlaneName;
using tsl::profiler::kDeviceVendorAMD;
using tsl::profiler::ParseAnnotationStack;
using tsl::profiler::StatType;
using tsl::profiler::XEventBuilder;
using tsl::profiler::XEventMetadata;
using tsl::profiler::XLineBuilder;
using tsl::profiler::XPlaneBuilder;
using tsl::profiler::XSpace;

//==========
namespace {
// Set the all XLines of specified XPlane to starting walltime.
// Events time in both host and device planes are CUTPI timestamps.
// We set initial RocmTracer timestamp as start time for all lines to reflect
// this fact. Eventually we change line start time to corresponding
// start_walltime_ns to normalize with CPU wall time.
static void NormalizeTimeStamps(XPlaneBuilder* plane,
                                uint64_t start_walltime_ns) {
  plane->ForEachLine([&](tsl::profiler::XLineBuilder line) {
    line.SetTimestampNs(start_walltime_ns);
  });
}

std::string GetDeviceXLineName(
    int64_t stream_id, absl::flat_hash_set<RocmTracerEventType>& event_types) {
  std::string line_name = absl::StrCat("Stream #", stream_id);
  event_types.erase(RocmTracerEventType::Unsupported);
  if (event_types.empty()) {
    return line_name;
  }
  std::vector<const char*> type_names;
  for (const auto event_type : event_types) {
    type_names.emplace_back(GetRocmTracerEventTypeName(event_type));
  }
  return absl::StrCat(line_name, "(", absl::StrJoin(type_names, ","), ")");
}

void PrintRocmTracerEvent(const RocmTracerEvent& event,
                          absl::string_view message = {},
                          uint64_t start_walltime_ns = 0,
                          uint64_t start_gputime_ns = 0) {
  std::ostringstream oss;
  oss << "correlation_id=" << event.correlation_id;
  oss << ",type=" << GetRocmTracerEventTypeName(event.type);
  oss << ",source=" << GetRocmTracerEventSourceName(event.source);
  oss << ",domain=" << GetRocmTracerEventDomainName(event.domain);
  oss << ",name=" << event.name;
  oss << ",corr_id=" << event.correlation_id;
  oss << ",annotation=" << event.annotation;
  oss << ",start_time_us="
      << (start_walltime_ns + (start_gputime_ns - event.start_time_ns)) / 1000;
  oss << ",duration=" << (event.end_time_ns - event.start_time_ns) / 1000;
  oss << ",device_id=" << event.device_id;
  oss << ",thread_id=" << event.thread_id;
  oss << ",stream_id=" << event.stream_id;

  switch (event.type) {
    case RocmTracerEventType::Kernel:
      break;
    case RocmTracerEventType::MemcpyD2H:
    case RocmTracerEventType::MemcpyH2D:
    case RocmTracerEventType::MemcpyD2D:
      oss << ",num_bytes=" << event.memcpy_info.num_bytes;
      oss << ",destination=" << event.memcpy_info.destination;
      oss << ",async=" << event.memcpy_info.async;
      break;
    case RocmTracerEventType::MemoryAlloc:
      oss << ",num_bytes=" << event.memalloc_info.num_bytes;
      break;
    case RocmTracerEventType::MemcpyOther:
    case RocmTracerEventType::MemoryFree:
    case RocmTracerEventType::Memset:
    case RocmTracerEventType::Synchronization:
    case RocmTracerEventType::Generic:
      break;
    default:
      DCHECK(false);
      break;
  }
  VLOG(3) << oss.str() << ' ' << message;
}

uint64_t get_timestamp() {
  uint64_t ts;
  rocprofiler_status_t CHECKSTATUS = rocprofiler_get_timestamp(&ts);
  if (CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS) {
    const char* errstr = rocprofiler_get_status_string(CHECKSTATUS);
    LOG(ERROR) << "function rocprofiler_get_timestamp failed with error "
               << errstr;
    return 0;
  }
  return ts;
}
}  // namespace

void PerDeviceCollector::CreateXEvent(const RocmTracerEvent& event,
                                      XPlaneBuilder* plane,
                                      uint64_t start_gpu_ns,
                                      uint64_t end_gpu_ns, XLineBuilder* line) {
  if (event.start_time_ns < start_gpu_ns || event.end_time_ns > end_gpu_ns ||
      event.start_time_ns > event.end_time_ns) {
    VLOG(2) << "events have abnormal timestamps:" << event.name
            << " start time(ns): " << event.start_time_ns
            << " end time(ns): " << event.end_time_ns
            << " start gpu(ns):" << start_gpu_ns
            << " end gpu(ns):" << end_gpu_ns
            << " corr. id:" << event.correlation_id;
    return;
  }
  std::string kernel_name = tsl::port::MaybeAbiDemangle(event.name.c_str());
  if (kernel_name.empty()) {
    kernel_name = GetRocmTracerEventTypeName(event.type);
  }
  XEventMetadata* event_metadata =
      plane->GetOrCreateEventMetadata(std::move(kernel_name));
  XEventBuilder xevent = line->AddEvent(*event_metadata);
  VLOG(7) << "Adding event to line=" << line->Id();
  xevent.SetTimestampNs(event.start_time_ns);
  xevent.SetEndTimestampNs(event.end_time_ns);
  if (event.source == RocmTracerEventSource::ApiCallback) {
    xevent.AddStatValue(
        *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)),
        event.device_id);
  }
  if (event.correlation_id != RocmTracerEvent::kInvalidCorrelationId) {
    xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kCorrelationId)),
                        event.correlation_id);
  }
  if (event.scope_range_id != 0) {
    xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kScopeRangeId)),
                        event.scope_range_id);
  }
  if (!event.roctx_range.empty()) {
    xevent.AddStatValue(
        *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kNVTXRange)),
        *plane->GetOrCreateStatMetadata(event.roctx_range));
  }

  if (event.type == RocmTracerEventType::Kernel &&
      event.source == RocmTracerEventSource::Activity) {
    std::optional<double> occupancy_pct;
    if (gfx_target_version_ != 0 && (event.kernel_info.arch_vgpr_count > 0 ||
                                     event.kernel_info.accum_vgpr_count > 0)) {
      // Unused y/z dimensions arrive as 0 and mean 1, so default them -- but
      // only them. A zero x-dimension is not an unused dimension, it is a
      // dispatch record that carried no launch geometry at all (SDK skew, a
      // truncated record, the graph-replay path). Defaulting x too would turn
      // that into a fabricated one-thread workgroup, which GetOccupancy()
      // would then model at a confident 1.5625% -- non-zero, so nothing
      // downstream filters it, and flatly contradicted by the `block:0,0,0`
      // in the same event's kKernelDetails. Leaving block_size at 0 is what
      // lets the guard in GetOccupancy() decline instead.
      const uint32_t block_size =
          event.kernel_info.workgroup_x == 0
              ? 0
              : event.kernel_info.workgroup_x *
                    std::max(event.kernel_info.workgroup_y, 1u) *
                    std::max(event.kernel_info.workgroup_z, 1u);

      RocmDeviceOccupancyParams params{};
      params.arch_vgpr_count = event.kernel_info.arch_vgpr_count;
      params.accum_vgpr_count = event.kernel_info.accum_vgpr_count;
      params.sgpr_count = event.kernel_info.sgpr_count;
      params.block_size = block_size;
      // group_segment_size from the dispatch record is already the total LDS
      // per workgroup (static + runtime). Use it directly; do not add the
      // symbol's static_group_segment_size on top (that would double-count).
      params.smem_bytes = event.kernel_info.group_segment_size;
      params.gfx_target_version = gfx_target_version_;

      // Membership means computed; a nullopt mapped value means "modelled and
      // came back unmodelable". Do not use has_value() as the miss test, or
      // unmodelable launches recompute on every single event.
      auto [it, inserted] = occupancy_cache_.try_emplace(params);
      if (inserted) {
        it->second = GetOccupancy(params, cu_count_);
      }
      const std::optional<OccupancyStats>& occ = it->second;

      if (occ.has_value()) {
        occupancy_pct = occ->occupancy_pct;
        // Emitted unconditionally on a successful model, including a genuine
        // 0.0 -- CUPTI does the same, and suppressing it makes "the kernel
        // occupies nothing" indistinguishable from "we never looked". The
        // converse case, a model that declined, leaves occupancy_pct empty so
        // that neither this stat nor ToXStat's occ_pct token is written.
        xevent.AddStatValue(*plane->GetOrCreateStatMetadata(GetStatTypeStr(
                                StatType::kTheoreticalOccupancyPct)),
                            occ->occupancy_pct);
        if (occ->min_grid_size > 0) {
          xevent.AddStatValue(*plane->GetOrCreateStatMetadata(GetStatTypeStr(
                                  StatType::kOccupancyMinGridSize)),
                              static_cast<int32_t>(occ->min_grid_size));
        }
      }
    }
    // The register count that goes next to occ_pct has to be the same charge
    // the occupancy model used, or the two numbers contradict each other in
    // the tooltip. Without a known target we cannot say how the arch and
    // accum files combine, so fall back to the non-unified max() -- which is
    // also what UnifiedVgprCount would return for such a target.
    const uint32_t regs_per_thread =
        target_constants_.has_value()
            ? UnifiedVgprCount(*target_constants_,
                               event.kernel_info.arch_vgpr_count,
                               event.kernel_info.accum_vgpr_count)
            : std::max(event.kernel_info.arch_vgpr_count,
                       event.kernel_info.accum_vgpr_count);
    xevent.AddStatValue(
        *plane->GetOrCreateStatMetadata(
            GetStatTypeStr(StatType::kKernelDetails)),
        *plane->GetOrCreateStatMetadata(
            ToXStat(event.kernel_info, regs_per_thread, occupancy_pct)));
  } else if (event.type == RocmTracerEventType::MemcpyH2D ||
             event.type == RocmTracerEventType::MemcpyD2H ||
             event.type == RocmTracerEventType::MemcpyD2D ||
             event.type == RocmTracerEventType::MemcpyOther) {
    VLOG(7) << "Add Memcpy stat";
    const auto& memcpy_info = event.memcpy_info;
    std::string memcpy_details = absl::StrCat(
        // TODO(rocm-profiler): we need to discover the memory kind similar
        // to CUDA
        "kind:", "Unknown", " size:", memcpy_info.num_bytes,
        " dest:", memcpy_info.destination, " async:", memcpy_info.async);
    xevent.AddStatValue(
        *plane->GetOrCreateStatMetadata(
            GetStatTypeStr(StatType::kMemcpyDetails)),
        *plane->GetOrCreateStatMetadata(std::move(memcpy_details)));
  } else if (event.type == RocmTracerEventType::MemoryAlloc) {
    VLOG(7) << "Add MemAlloc stat";
    std::string value =
        // TODO(rocm-profiler): we need to discover the memory kind similar
        // to CUDA
        absl::StrCat("kind:", "Unknown",
                     " num_bytes:", event.memalloc_info.num_bytes);
    xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kMemallocDetails)),
                        *plane->GetOrCreateStatMetadata(std::move(value)));
  } else if (event.type == RocmTracerEventType::MemoryFree) {
    VLOG(7) << "Add MemFree stat";
    std::string value =
        // TODO(rocm-profiler): we need to discover the memory kind similar
        // to CUDA
        absl::StrCat("kind:", "Unknown",
                     " num_bytes:", event.memalloc_info.num_bytes);
    xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kMemFreeDetails)),
                        *plane->GetOrCreateStatMetadata(std::move(value)));
  } else if (event.type == RocmTracerEventType::Memset) {
    VLOG(7) << "Add Memset stat";
    auto value =
        // TODO(rocm-profiler): we need to discover the memory kind similar
        // to CUDA
        absl::StrCat("kind:", "Unknown",
                     " num_bytes:", event.memset_info.num_bytes,
                     " async:", event.memset_info.async);
    xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kMemsetDetails)),
                        *plane->GetOrCreateStatMetadata(std::move(value)));
  }
  // TODO(rocm-profiler): we need to support the following event type
  /* else if (event.type == CuptiTracerEventType::MemoryResidency) {
    VLOG(7) << "Add MemoryResidency stat";
    std::string value = absl::StrCat(
        "kind:", GetMemoryKindName(event.memory_residency_info.kind),
        " num_bytes:", event.memory_residency_info.num_bytes,
        " addr:", event.memory_residency_info.address);
    xevent.AddStatValue(*plane->GetOrCreateStatMetadata(GetStatTypeStr(
                            StatType::kMemoryResidencyDetails)),
                        *plane->GetOrCreateStatMetadata(std::move(value)));
  } */

  std::vector<Annotation> annotation_stack =
      ParseAnnotationStack(event.annotation);
  if (!annotation_stack.empty()) {
    xevent.AddStatValue(
        *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)),
        *plane->GetOrCreateStatMetadata(annotation_stack.begin()->name));
  }
  // If multiple metadata have the same key name, show the values from the
  // top of the stack (innermost annotation). Concatenate the values from
  // "hlo_op".
  absl::flat_hash_set<absl::string_view> key_set;

  for (auto annotation = annotation_stack.rbegin();
       annotation != annotation_stack.rend(); ++annotation) {
    for (const Annotation::Metadata& metadata : annotation->metadata) {
      if (key_set.insert(metadata.key).second) {
        xevent.ParseAndAddStatValue(
            *plane->GetOrCreateStatMetadata(metadata.key), metadata.value);
      }
    }
  }
}

void PerDeviceCollector::SortByStartTime() {
  absl::MutexLock lock(events_mutex_);
  std::sort(events_.begin(), events_.end(),
            [](const RocmTracerEvent& event1, const RocmTracerEvent& event2) {
              return event1.start_time_ns < event2.start_time_ns;
            });
}

bool PerDeviceCollector::IsHostEvent(const RocmTracerEvent& event,
                                     int64_t* line_id) {
  // DriverCallback(i.e. kernel launching) events are host events.
  if (event.source == RocmTracerEventSource::ApiCallback) {
    *line_id = event.thread_id;
    return true;
  } else {  // activities
    *line_id = event.stream_id;
    return false;
  }

  // TODO(rocm-profiler): do we have such a report in rocm?
  // Non-overhead activity events are device events.
  /* if (event.type != CuptiTracerEventType::Overhead) {
    *line_id = event.stream_id;
    return false;
  } */
  // Overhead events can be associated with a thread or a stream, etc.
  // If a valid thread id is specified, we consider it as a host event.
  //

  if (event.stream_id != RocmTracerEvent::kInvalidStreamId) {
    *line_id = event.stream_id;
    return false;
  } else if (event.thread_id != RocmTracerEvent::kInvalidThreadId &&
             event.thread_id != 0) {
    *line_id = event.thread_id;
    return true;
  } else {
    *line_id = tsl::profiler::kThreadIdOverhead;
    return false;
  }
}

void PerDeviceCollector::Export(uint64_t start_walltime_ns,
                                uint64_t start_gputime_ns,
                                uint64_t end_gputime_ns,
                                XPlaneBuilder* device_plane,
                                XPlaneBuilder* host_plane) {
  absl::MutexLock lock(events_mutex_);
  // Tracking event types per line.
  absl::flat_hash_map<int64_t, absl::flat_hash_set<RocmTracerEventType> >
      events_types_per_line;

  // Build dense stream remapping: raw HIP stream handles (64-bit pointer
  // values) are converted to sequential indices (0, 1, 2, ...) for clean
  // timeline lane numbering.
  absl::flat_hash_map<uint64_t, int64_t> stream_remap;
  int64_t next_stream_idx = 0;
  for (const auto& event : events_) {
    if (event.source == RocmTracerEventSource::Activity &&
        event.stream_id != RocmTracerEvent::kInvalidStreamId &&
        !stream_remap.contains(event.stream_id)) {
      stream_remap[event.stream_id] = next_stream_idx++;
    }
  }

  for (const RocmTracerEvent& event : events_) {
    int64_t line_id = RocmTracerEvent::kInvalidThreadId;
    bool is_host_event = IsHostEvent(event, &line_id);

    // Apply dense stream remapping for device events.
    if (!is_host_event) {
      auto it = stream_remap.find(static_cast<uint64_t>(line_id));
      if (it != stream_remap.end()) {
        line_id = it->second;
      }
    }

    if (line_id == RocmTracerEvent::kInvalidThreadId ||
        line_id == RocmTracerEvent::kInvalidStreamId) {
      VLOG(3) << "Ignoring event, type=" << static_cast<int>(event.type);
      continue;
    }
    auto* plane = is_host_event ? host_plane : device_plane;
    VLOG(9) << "Event"
            << " type=" << static_cast<int>(event.type)
            << " line_id=" << line_id
            << (is_host_event ? " host plane=" : " device plane=")
            << plane->Name();

    XLineBuilder line = plane->GetOrCreateLine(line_id);
    line.SetTimestampNs(start_gputime_ns);
    CreateXEvent(event, plane, start_gputime_ns, end_gputime_ns, &line);
  }

  device_plane->ForEachLine([&](XLineBuilder line) {
    line.SetName(
        GetDeviceXLineName(line.Id(), events_types_per_line[line.Id()]));
  });
  host_plane->ForEachLine([&](XLineBuilder line) {
    line.SetName(absl::StrCat("Host Threads/", line.Id()));
  });
  events_.clear();
}

void PerDeviceCollector::AddEvent(RocmTracerEvent&& event) {
  absl::MutexLock lock(events_mutex_);
  events_.emplace_back(std::move(event));
}

void PerDeviceCollector::GetDeviceCapabilities(
    const rocprofiler_agent_v0_t& agent, XPlaneBuilder* device_plane) {
  device_plane->AddStatValue(*device_plane->GetOrCreateStatMetadata(
                                 GetStatTypeStr(StatType::kDevVendor)),
                             kDeviceVendorAMD);

  // Agent clock rates are in MHz; profiler stats expect KHz.
  auto clock_rate_in_khz =
      static_cast<int64_t>(agent.max_engine_clk_fcompute) * 1000;
  if (clock_rate_in_khz) {
    device_plane->AddStatValue(
        *device_plane->GetOrCreateStatMetadata(
            GetStatTypeStr(StatType::kDevCapClockRateKHz)),
        clock_rate_in_khz);
  }

  auto core_count = agent.cu_count;
  if (core_count) {
    device_plane->AddStatValue(*device_plane->GetOrCreateStatMetadata(
                                   GetStatTypeStr(StatType::kDevCapCoreCount)),
                               core_count);
  }

  // Extract memory info from VRAM (frame buffer) banks only, skipping
  // system memory, LDS, GDS, scratch, etc.
  // TODO(ROCm): APUs share system memory and may not have frame buffer
  // banks; verify behavior on APU targets.
  if (agent.mem_banks_count > 0 && agent.mem_banks != nullptr) {
    uint64_t total_memory = 0;
    uint32_t vram_clock_mhz = 0;
    uint32_t vram_bus_width_bits = 0;
    for (uint32_t i = 0; i < agent.mem_banks_count; ++i) {
      if (agent.mem_banks[i].heap_type == HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC ||
          agent.mem_banks[i].heap_type == HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE) {
        total_memory += agent.mem_banks[i].size_in_bytes;
        if (vram_clock_mhz == 0) {
          vram_clock_mhz = agent.mem_banks[i].mem_clk_max;
          vram_bus_width_bits = agent.mem_banks[i].width;
        }
      }
    }

    if (vram_clock_mhz && vram_bus_width_bits) {
      // Agent mem_clk_max is in MHz; multiply by 10^6 to get Hz.
      // Times 2 because HBM is DDR memory; it gets two data bits per each
      // data lane.
      auto memory_bandwidth = uint64_t{2} *
                              static_cast<uint64_t>(vram_clock_mhz) * 1000 *
                              1000 * vram_bus_width_bits / 8;
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapMemoryBandwidth)),
          memory_bandwidth);
    }

    if (total_memory) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapMemorySize)),
          total_memory);
    }
  }

  // gfx_target_version encodes, all in decimal: major=((value/10000)%100),
  // minor=((value/100)%100), step=(value%100). The step renders in hex in the
  // agent *name* only, which is why gfx90a is 90010 and not 9010a.
  auto gfx_ver = agent.gfx_target_version;
  if (gfx_ver) {
    auto compute_capability_major = (gfx_ver / 10000) % 100;
    if (compute_capability_major) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapComputeCapMajor)),
          compute_capability_major);
    }
    // Emitted unguarded: 0 is a real minor version. The old `if (minor)` guard
    // meant every gfx90x part -- gfx900, gfx906, gfx90a -- published a major
    // with no minor at all. The stepping has no home in this CUDA-shaped
    // schema (it renders as a hex digit in the target name, so gfx90a is
    // major 9, minor 0, step 0xa); occupancy therefore keys off the raw
    // gfx_target_version below, never off these two stats.
    auto compute_capability_minor = (gfx_ver / 100) % 100;
    device_plane->AddStatValue(
        *device_plane->GetOrCreateStatMetadata(
            GetStatTypeStr(StatType::kDevCapComputeCapMinor)),
        compute_capability_minor);
  }

  gfx_target_version_ = agent.gfx_target_version;
  cu_count_ = agent.cu_count;

  // Cross-check the per-target table against what the runtime reports. The
  // table is the source of truth for occupancy (the agent does not expose
  // register-file size or granules at all), so a mismatch here means the table
  // needs a new entry -- e.g. a stale ROCr claiming max_waves_per_simd 10 on
  // gfx90a, or silicon disagreeing with LLVM about gfx950's LDS.
  target_constants_ = LookupTargetConstants(gfx_target_version_);
  if (target_constants_.has_value() && target_constants_->exact) {
    const AmdGpuTargetConstants& tc = *target_constants_;
    // lds_size_in_kb is the LDS capacity per CU in KiB.
    const uint32_t agent_lds_bytes = agent.lds_size_in_kb * 1024;
    if (agent_lds_bytes != 0 && agent_lds_bytes != tc.lds_per_cu) {
      LOG_FIRST_N(WARNING, 1)
          << "Occupancy target table disagrees with the runtime for " << tc.name
          << ": table lds_per_cu=" << tc.lds_per_cu << ", agent reports "
          << agent_lds_bytes << " bytes. Occupancy will use the table value.";
    }
    if (agent.max_waves_per_simd != 0 &&
        agent.max_waves_per_simd != tc.max_waves_per_simd) {
      LOG_FIRST_N(WARNING, 1)
          << "Occupancy target table disagrees with the runtime for " << tc.name
          << ": table max_waves_per_simd=" << tc.max_waves_per_simd
          << ", agent reports " << agent.max_waves_per_simd
          << ". Occupancy will use the table value.";
    }
    if (agent.wave_front_size != 0 &&
        agent.wave_front_size != tc.wave_front_size) {
      LOG_FIRST_N(WARNING, 1)
          << "Occupancy target table disagrees with the runtime for " << tc.name
          << ": table wave_front_size=" << tc.wave_front_size
          << ", agent reports " << agent.wave_front_size
          << ". Occupancy will use the table value.";
    }
  }

  // Emit the agent name (the GCN architecture string), for example "gfx942"
  if (agent.name != nullptr && agent.name[0] != '\0') {
    device_plane->AddStatValue(*device_plane->GetOrCreateStatMetadata(
                                   GetStatTypeStr(StatType::kGpuDeviceName)),
                               agent.name);
  }
}

void RocmTraceCollectorImpl::AddEvent(RocmTracerEvent&& event,
                                      bool is_auxiliary) {
  absl::MutexLock lock(event_maps_mutex_);

  if (event.source == RocmTracerEventSource::ApiCallback) {
    if (!is_auxiliary) {
      if (num_callback_events_ >= options_.max_callback_api_events) {
        LOG(WARNING)
            << "!!! Number of callback events = " << num_callback_events_
            << " is greater than/equal to the max callback api events = "
            << options_.max_callback_api_events
            << ". To collect more GPU events, please set "
               "XLA_FLAGS=--xla_gpu_rocm_max_trace_events=X ";
        return;
      }
      num_callback_events_++;
    }
    auto& map = is_auxiliary ? auxiliary_api_events_map_ : api_events_map_;
    auto [it, added] = map.emplace(event.correlation_id, std::move(event));

    if (!added) {
      OnEventsDropped("event with duplicate correlation_id was received.",
                      event.correlation_id);
      PrintRocmTracerEvent(event, ". Dropped!");
    }
  } else if (event.source == RocmTracerEventSource::Activity) {
    if (event.domain == RocmTracerEventDomain::HIP_API) {
      // we do not count HIP_OPS activities.
      if (num_activity_events_ >= options_.max_activity_api_events) {
        LOG_FIRST_N(WARNING, 1)
            << "Number of activity events (" << num_activity_events_
            << ") has reached the configured limit "
               "(xla_gpu_rocm_max_trace_events="
            << options_.max_activity_api_events
            << "). To collect more GPU events, increase "
               "XLA_FLAGS=--xla_gpu_rocm_max_trace_events=<value>.";
        return;
      }

      num_activity_events_++;
    }

    auto [it, _] = activity_ops_events_map_.emplace(
        event.correlation_id, std::vector<RocmTracerEvent>{});
    it->second.push_back(std::move(event));
  } else {
    VLOG(3) << "Dropping unknown event: " << (int)event.source
            << " domain: " << (int)event.domain;
  }
}

void RocmTraceCollectorImpl::Flush() {
  absl::MutexLock lock(event_maps_mutex_);
  auto aggregated_events = ApiActivityInfoExchange();

  VLOG(3) << "RocmTraceCollector collected " << num_callback_events_
          << " callback events, " << num_activity_events_
          << " activity events, and aggregated them into "
          << aggregated_events.size() << " events.";

  // device ids for GPUs filled in by roctracer are not zero indexed.
  // They are offset by number of CPUs on the machine
  uint32_t min_device_id = INT32_MAX;

  for (const auto& event : aggregated_events) {
    if (event.device_id < min_device_id) {
      min_device_id = event.device_id;
    }
  }

  for (auto& event : aggregated_events) {
    auto id = event.device_id - min_device_id;
    if (id < num_gpus_) {
      per_device_collector_[id].AddEvent(std::move(event));
    } else {
      PrintRocmTracerEvent(event, ". Dropped due to invalid device ID!");
    }
  }

  activity_ops_events_map_.clear();
  api_events_map_.clear();
  auxiliary_api_events_map_.clear();
}

void RocmTraceCollectorImpl::ExportScopeRangeIdTree(XSpace* space) {
  XPlaneBuilder plane(FindOrAddMutablePlaneWithName(
      space, tsl::profiler::kScopeRangeIdTreePlaneName));
  // No metadata is used for this plane, we just use the XStat to
  // transfer the map without breaking any existing proto.
  tensorflow::profiler::XStatMetadata metadata;
  for (const auto& [child_id, parent_id] : scope_range_id_tree_) {
    metadata.set_id(child_id);
    plane.AddStatValue(metadata, parent_id);
  }
}

void RocmTraceCollectorImpl::Export(XSpace* space) {
  uint64_t end_gputime_ns = get_timestamp();
  XPlaneBuilder host_plane(FindOrAddMutablePlaneWithName(
      space, tsl::profiler::kRoctracerApiPlaneName));

  VLOG(3) << "Calling RocmTraceCollectorImpl::Export num_gpus " << num_gpus_;

  for (int id = 0; id < num_gpus_; id++) {
    std::string name = GpuPlaneName(id);
    XPlaneBuilder device_plane(FindOrAddMutablePlaneWithName(space, name));
    device_plane.SetId(id);
    if (id < static_cast<int>(gpu_agents_.size())) {
      per_device_collector_[id].GetDeviceCapabilities(gpu_agents_[id],
                                                      &device_plane);
    } else {
      // Without capabilities this device has no gfx_target_version, so every
      // kernel on it silently loses its occupancy stats. Say so once rather
      // than leaving a plane that is quietly missing a column.
      LOG_FIRST_N(WARNING, 1)
          << "No rocprofiler agent for device " << id << " (only "
          << gpu_agents_.size()
          << " GPU agents were enumerated); device capabilities and "
             "theoretical occupancy will be missing for this plane.";
    }
    per_device_collector_[id].Export(start_walltime_ns_, start_gputime_ns_,
                                     end_gputime_ns, &device_plane,
                                     &host_plane);
    NormalizeTimeStamps(&device_plane, start_walltime_ns_);
  }
  NormalizeTimeStamps(&host_plane, start_walltime_ns_);
  ExportScopeRangeIdTree(space);
}

std::vector<RocmTracerEvent> RocmTraceCollectorImpl::ApiActivityInfoExchange() {
  /* Different from CUDA, roctracer activity records are not enough to fill a
    TF event. For most of the activities, we need to enable the corresponding
    API callsbacks (we call them auxiliary API callbacks) to capture the
    necessary fields from them using the correlation id. The purpose of this
    function is to let APIs and activities exchange information to reach a
    state very similar to TF CUDA and getting ready to dump the event.
  */

  std::vector<RocmTracerEvent> aggregated_events;
  size_t total_activities = 0;
  for (const auto& [_, v] : activity_ops_events_map_) {
    total_activities += v.size();
  }
  aggregated_events.reserve(api_events_map_.size() + total_activities);

  // Copy info from activity events to API callback events
  for (auto& [key, api_event] : api_events_map_) {
    auto iact = activity_ops_events_map_.find(api_event.correlation_id);

    if (iact == activity_ops_events_map_.end()) {
      PrintRocmTracerEvent(api_event, ". Dropped!");
      VLOG(1) << api_event.name << "  could not find activity counterpart!";
      continue;
    }
    const auto& item = iact->second.front();
    api_event.device_id = item.device_id;
    api_event.stream_id = item.stream_id;
    switch (api_event.type) {
      case RocmTracerEventType::Kernel:
        api_event.kernel_info = item.kernel_info;
        aggregated_events.push_back(api_event);
        break;
      case RocmTracerEventType::Memset:
      case RocmTracerEventType::MemoryAlloc:
      case RocmTracerEventType::MemoryFree:
      case RocmTracerEventType::Synchronization:
        aggregated_events.push_back(api_event);
        break;
      case RocmTracerEventType::MemcpyD2H:
      case RocmTracerEventType::MemcpyH2D:
      case RocmTracerEventType::MemcpyD2D:
      case RocmTracerEventType::MemcpyOther:
        api_event.memcpy_info = item.memcpy_info;
        aggregated_events.push_back(api_event);
        break;
      default:
        OnEventsDropped("Missing API-Activity information exchange. Dropped!",
                        api_event.correlation_id);
        PrintRocmTracerEvent(api_event, ". Dropped!");
        LOG(WARNING) << "A ROCm API event type with unimplemented activity "
                        "merge dropped! "
                        "Type="
                     << GetRocmTracerEventTypeName(api_event.type);
    }  // switch
  }    // for

  // Make sure for all activity events we have API callback events.
  //
  // `activity_iter.second` is a vector keyed by correlation_id; a single
  // hipGraphLaunch can produce many kernel-dispatch records sharing one
  // correlation_id. Iterate the whole vector; the api_event lookup is
  // invariant across it and hoisted out of the inner loop.
  for (auto& activity_iter : activity_ops_events_map_) {
    if (activity_iter.second.empty()) {
      continue;
    }
    const uint32_t corr_id = activity_iter.first;

    const RocmTracerEvent* api_event = nullptr;
    if (auto it = api_events_map_.find(corr_id); it != api_events_map_.end()) {
      api_event = &it->second;
    } else if (auto it_aux = auxiliary_api_events_map_.find(corr_id);
               it_aux != auxiliary_api_events_map_.end()) {
      api_event = &it_aux->second;
    }

    if (api_event == nullptr) {
      // Drop the entire vector together; log once per correlation_id
      // instead of per activity event (the activities all share corr_id).
      OnEventsDropped(
          "An event from activity was discarded."
          "Could not find the counterpart HIP API.",
          corr_id);
      PrintRocmTracerEvent(activity_iter.second.front(), ". Dropped!");
      continue;
    }

    for (auto& activity_event : activity_iter.second) {
      switch (activity_event.type) {
        case RocmTracerEventType::Kernel:
          // Deliberately does NOT copy kernel_info from the API event. The
          // dispatch record is the sole authoritative source: KernelEvent()
          // builds KernelDetails per dispatch (rocm_tracer.cc), while the API
          // callback path zeroes it. Copying api->activity here corrupted two
          // cases:
          //   * N dispatches under one correlation_id (a hipGraphLaunch
          //     replay) all received dispatch #1's registers and LDS, because
          //     the loop above seeds api_event from `.front()`. A 5-VGPR
          //     elementwise kernel and a 416-VGPR flash-attention kernel in
          //     one graph reported identical occupancy.
          //   * A correlation_id present only in auxiliary_api_events_map_
          //     (never visited by the loop above) wiped kernel_info to zero
          //     for every one of its dispatches, suppressing occupancy
          //     entirely.
          // The memset/memcpy cases below still need their api->activity copy;
          // only kernel_info flows the other way.
          PrintRocmTracerEvent(activity_event,
                               ". activity event from api_event.");
          aggregated_events.push_back(activity_event);
          break;

        case RocmTracerEventType::MemcpyD2H:
        case RocmTracerEventType::MemcpyH2D:
        case RocmTracerEventType::MemcpyD2D:
        case RocmTracerEventType::MemcpyOther:
          // activity_event.memcpy_info = api_event->memcpy_info;
          aggregated_events.push_back(activity_event);
          break;
        case RocmTracerEventType::Memset:
          activity_event.memset_info = api_event->memset_info;
          aggregated_events.push_back(activity_event);
          break;

        case RocmTracerEventType::MemoryAlloc:
        case RocmTracerEventType::MemoryFree:
          activity_event.device_id = api_event->device_id;
          aggregated_events.push_back(activity_event);
          break;

        case RocmTracerEventType::Synchronization:
          activity_event.device_id = api_event->device_id;
          aggregated_events.push_back(activity_event);
          break;
        default:
          OnEventsDropped("Missing API-Activity information exchange. Dropped!",
                          activity_event.correlation_id);
          PrintRocmTracerEvent(activity_event, ". Dropped!");
          LOG(WARNING) << "A ROCm activity event with unimplemented API "
                          "callback merge dropped! "
                          "Type="
                       << GetRocmTracerEventTypeName(activity_event.type);
      }  // switch
    }    // for activity_event
  }      // for activity_iter

  return aggregated_events;
}

std::unique_ptr<RocmTraceCollector> CreateRocmCollector(
    const RocmTraceCollectorOptions& options, const uint64_t start_walltime_ns,
    const uint64_t start_gputime_ns) {
  return std::make_unique<RocmTraceCollectorImpl>(options, start_walltime_ns,
                                                  start_gputime_ns);
}

}  // namespace profiler
}  // namespace xla
