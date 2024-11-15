
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

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "xla/stream_executor/rocm/roctracer_wrapper.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/abi.h"
#include "tsl/platform/env_time.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/status.h"
#include "tsl/platform/thread_annotations.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/backends/cpu/annotation_stack.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/utils/parse_annotation.h"
#include "tsl/profiler/utils/xplane_builder.h"
#include "tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/utils/xplane_utils.h"

namespace se = ::stream_executor;

namespace xla {
namespace profiler {

namespace se = ::stream_executor;
using tensorflow::ProfileOptions;
using tsl::mutex;
using tsl::mutex_lock;
// using tsl::OkStatus;
using tsl::Status;
using tsl::profiler::Annotation;
using tsl::profiler::AnnotationStack;
using tsl::profiler::FindOrAddMutablePlaneWithName;
using tsl::profiler::GetStatTypeStr;
using tsl::profiler::GpuPlaneName;
using tsl::profiler::kDeviceVendorAMD;
using tsl::profiler::kThreadIdOverhead;
using tsl::profiler::ParseAnnotationStack;
using tsl::profiler::ProfilerInterface;
// using tsl::profiler::RegisterProfilerFactory;
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

/*
std::string GetDeviceXLineName(
    int64_t stream_id, absl::flat_hash_set<RocmTracerEventType>& event_types) {
  std::string line_name = absl::StrCat("Stream #", stream_id);
  event_types.erase(RocmTracerEventType::Unsupported);
  if (event_types.empty()) return line_name;
  std::vector<const char*> type_names;
  for (const auto event_type : event_types) {
    type_names.emplace_back(GetRocmTracerEventTypeName(event_type));
  }
  return absl::StrCat(line_name, "(", absl::StrJoin(type_names, ","), ")");
}
*/
}  // namespace

static void DumpRocmTracerEvent(const RocmTracerEvent& event,
                                uint64_t start_walltime_ns,
                                uint64_t start_gputime_ns,
                                const std::string& message) {
  std::ostringstream oss;
  oss << "correlation_id=" << event.correlation_id;
  // oss << ",type=" << GetRocmTracerEventTypeName(event.type);
  // oss << ",source=" << GetRocmTracerEventSourceName(event.source);
  // oss << ",domain=" << GetRocmTracerEventDomainName(event.domain);
  oss << ",name=" << event.name;
  oss << ",annotation=" << event.annotation;
  oss << ",start_time_us="
      << (start_walltime_ns + (start_gputime_ns - event.start_time_ns)) / 1000;
  oss << ",duration=" << (event.end_time_ns - event.start_time_ns) / 1000;
  oss << ",device_id=" << event.device_id;
  oss << ",thread_id=" << event.thread_id;
  oss << ",stream_id=" << event.stream_id;

  oss << message;
  VLOG(3) << oss.str();
}

static uint64_t get_timestamp() {
    uint64_t ts;
    rocprofiler_status_t CHECKSTATUS = se::wrap::rocprofiler_get_timestamp(&ts);
    if (CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS) {
        const char* errstr = se::wrap::rocprofiler_get_status_string(CHECKSTATUS);
        LOG(ERROR) << "function rocprofiler_get_timestamp failed with error "
                   << errstr;
        return 0;
    }
    return ts;
}

struct RocmDeviceOccupancyParams {
  hipFuncAttributes attributes = {};
  int block_size = 0;
  size_t dynamic_smem_size = 0;
  void* func_ptr;

  friend bool operator==(const RocmDeviceOccupancyParams& lhs,
                         const RocmDeviceOccupancyParams& rhs) {
    return 0 == memcmp(&lhs, &rhs, sizeof(lhs));
  }

  template <typename H>
  friend H AbslHashValue(H hash_state,
                         const RocmDeviceOccupancyParams& params) {
    return H::combine(
        std::move(hash_state), params.attributes.maxThreadsPerBlock,
        params.attributes.numRegs, params.attributes.sharedSizeBytes,
        params.attributes.maxDynamicSharedSizeBytes, params.block_size,
        params.dynamic_smem_size, params.func_ptr);
  }
};

struct OccupancyStats {
  double occupancy_pct = 0.0;
  int min_grid_size = 0;
  int suggested_block_size = 0;
};

struct CorrelationInfo {
  CorrelationInfo(uint32_t t, uint32_t e) : thread_id(t), enqueue_time_ns(e) {}
  uint32_t thread_id;
  uint64_t enqueue_time_ns;
};

class PerDeviceCollector {
 private:
  OccupancyStats GetOccupancy(const RocmDeviceOccupancyParams& params) const {
    // TODO(rocm-profiler): hipOccupancyMaxActiveBlocksPerMultiprocessor only
    // return hipSuccess for HIP_API_ID_hipLaunchKernel

    OccupancyStats stats;
    int number_of_active_blocks;
    hipError_t err = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &number_of_active_blocks, params.func_ptr, params.block_size,
        params.dynamic_smem_size);

    if (err != hipError_t::hipSuccess) {
      return {};
    }

    stats.occupancy_pct = number_of_active_blocks * params.block_size * 100;
    stats.occupancy_pct /= device_properties_.maxThreadsPerMultiProcessor;

    err = hipOccupancyMaxPotentialBlockSize(
        &stats.min_grid_size, &stats.suggested_block_size,
        static_cast<const void*>(params.func_ptr), params.dynamic_smem_size, 0);

    if (err != hipError_t::hipSuccess) {
      return {};
    }

    return stats;
  }

  void CreateXEvent(const RocmTracerEvent& event, XPlaneBuilder* plane,
                    uint64_t start_gpu_ns, uint64_t end_gpu_ns,
                    XLineBuilder* line) {
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
    // std::string kernel_name = tsl::port::MaybeAbiDemangle(event.name.c_str());
    /*
    if (kernel_name.empty()) {
      kernel_name = GetRocmTracerEventTypeName(event.type);
    }
    */
    XEventMetadata* event_metadata =
        plane->GetOrCreateEventMetadata(std::move(event.name));
    XEventBuilder xevent = line->AddEvent(*event_metadata);
    // VLOG(7) << "Adding event to line=" << line->Id();
    LOG(FATAL) << "Adding event to line=" << line->Id();
    xevent.SetTimestampNs(event.start_time_ns);
    xevent.SetEndTimestampNs(event.end_time_ns);
    xevent.SetDurationNs(event.end_time_ns - event.start_time_ns);

    xevent.AddStatValue(*plane->GetOrCreateStatMetadata("DeviceId"), event.device_id);
    xevent.AddStatValue(*plane->GetOrCreateStatMetadata("CorrelationId"), event.correlation_id);

    switch (event.type) {
      case RocmTracerEventType::HIP_RUNTIME_API:
        xevent.AddStatValue(*plane->GetOrCreateStatMetadata("KernelName"), event.name);
        break;
      case RocmTracerEventType::KERNEL_DISPATCH:
        // test
        xevent.AddStatValue(*plane->GetOrCreateStatMetadata("BytesTransferred"), 1024);
                break;
      default:
        VLOG(2) << "Unhandled event type: " << static_cast<int>(event.type);
                break;
    }  
  }

  void SortByStartTime() {
    mutex_lock lock(events_mutex);
    std::sort(events.begin(), events.end(),
              [](const RocmTracerEvent& event1, const RocmTracerEvent& event2) {
                return event1.start_time_ns < event2.start_time_ns;
              });
  }

  bool IsHostEvent(const RocmTracerEvent& event, tsl::int64* line_id) {
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

 public:
  void Export(uint64_t start_walltime_ns, uint64_t start_gputime_ns,
              uint64_t end_gputime_ns, XPlaneBuilder* device_plane,
              XPlaneBuilder* host_plane) {
    int host_ev_cnt = 0, dev_ev_cnt = 0;
    mutex_lock l(events_mutex);
    // Tracking event types per line.
    absl::flat_hash_map<tsl::int64, absl::flat_hash_set<RocmTracerEventType>>
        events_types_per_line;
    for (const RocmTracerEvent& event : events) {
      int64_t line_id = RocmTracerEvent::kInvalidThreadId;
      bool is_host_event = IsHostEvent(event, &line_id);

      if (is_host_event) {
        host_ev_cnt++;
      } else {
        dev_ev_cnt++;
      }

      if (line_id == RocmTracerEvent::kInvalidThreadId ||
          line_id == RocmTracerEvent::kInvalidStreamId) {
        VLOG(3) << "Ignoring event, type=" << static_cast<int>(event.type);
        continue;
      }
      auto* plane = is_host_event ? host_plane : device_plane;
      VLOG(9) << "Event" << " type=" << static_cast<int>(event.type)
              << " line_id=" << line_id
              << (is_host_event ? " host plane=" : " device plane=")
              << plane->Name();
      XLineBuilder line = plane->GetOrCreateLine(line_id);
      line.SetTimestampNs(start_gputime_ns);
      CreateXEvent(event, plane, start_gputime_ns, end_gputime_ns, &line);
      events_types_per_line[line_id].emplace(event.type);
    }
    /*
    device_plane->ForEachLine([&](XLineBuilder line) {
      line.SetName(
          GetDeviceXLineName(line.Id(), events_types_per_line[line.Id()]));
    });
    */
    host_plane->ForEachLine([&](XLineBuilder line) {
      line.SetName(absl::StrCat("Host Threads/", line.Id()));
    });
    events.clear();
  }

  PerDeviceCollector() = default;

  void AddEvent(const RocmTracerEvent& event) {
    mutex_lock l(events_mutex);
    LOG(ERROR) << "Starting to add event";
    events.emplace_back(std::move(event));
  }

  void GetDeviceCapabilities(int32_t device_ordinal,
                             XPlaneBuilder* device_plane) {
    device_plane->AddStatValue(*device_plane->GetOrCreateStatMetadata(
                                   GetStatTypeStr(StatType::kDevVendor)),
                               kDeviceVendorAMD);

    if (hipGetDeviceProperties(&device_properties_, device_ordinal) !=
        hipSuccess)
      return;

    auto clock_rate_in_khz =
        device_properties_.clockRate;  // this is also in Khz
    if (clock_rate_in_khz) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapClockRateKHz)),
          clock_rate_in_khz);
    }

    auto core_count = device_properties_.multiProcessorCount;
    if (core_count) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapCoreCount)),
          core_count);
    }

    auto mem_clock_khz = device_properties_.memoryClockRate;
    auto mem_bus_width_bits = device_properties_.memoryBusWidth;

    if (mem_clock_khz && mem_bus_width_bits) {
      // Times 2 because HBM is DDR memory; it gets two data bits per each
      // data lane.
      auto memory_bandwidth =
          uint64_t{2} * (mem_clock_khz) * 1000 * (mem_bus_width_bits) / 8;
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapMemoryBandwidth)),
          memory_bandwidth);
    }

    size_t total_memory = device_properties_.totalGlobalMem;
    if (total_memory) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapMemorySize)),
          static_cast<uint64_t>(total_memory));
    }

    auto compute_capability_major = device_properties_.major;
    if (compute_capability_major) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapComputeCapMajor)),
          compute_capability_major);
    }
    auto compute_capability_minor = device_properties_.minor;
    if (compute_capability_minor) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapComputeCapMinor)),
          compute_capability_minor);
    }
  }

 private:
  mutex events_mutex;
  std::vector<RocmTracerEvent> events TF_GUARDED_BY(events_mutex);
  absl::flat_hash_map<uint32_t, CorrelationInfo> correlation_info_
      TF_GUARDED_BY(events_mutex);
  absl::flat_hash_map<RocmDeviceOccupancyParams, OccupancyStats>
      occupancy_cache_;
  hipDeviceProp_t device_properties_;
};

class RocmTraceCollectorImpl : public profiler::RocmTraceCollector {
 public:
  RocmTraceCollectorImpl(const RocmTraceCollectorOptions& options,
                         uint64_t start_walltime_ns, uint64_t start_gputime_ns)
      : RocmTraceCollector(options),
        start_walltime_ns_(start_walltime_ns),
        start_gputime_ns_(start_gputime_ns),
        num_gpus_(options.num_gpus) {}

  void AddEvent(RocmTracerEvent& event) override;
  void Flush() override;
  void Export(XSpace* space) override;

  /*
  void OnEventsDropped(const std::string& reason,
                       uint32_t correlation_id) override {
    LOG(INFO) << "RocmTracerEvent dropped (correlation_id=" << correlation_id
              << ",) : " << reason << ".";
  }
  */
 private:
  // std::atomic<int> num_callback_events_;
  // std::atomic<int> num_activity_events_;
  uint64_t start_walltime_ns_;
  uint64_t start_gputime_ns_;
  int num_gpus_;

  mutex event_maps_mutex_;
  std::vector<RocmTracerEvent> events_ TF_GUARDED_BY(event_maps_mutex_);
  absl::flat_hash_map<uint32_t, PerDeviceCollector> per_device_collector_;

};

void RocmTraceCollectorImpl::AddEvent(RocmTracerEvent& event) {
  LOG(ERROR) << "Starting RocmTraceCollectorImpl::AddEvent";
  mutex_lock lock(event_maps_mutex_);
  events_.push_back(std::move(event));
}

void RocmTraceCollectorImpl::Flush() {
  for (const auto& event : events_) {
    auto device_id = event.device_id;
    per_device_collector_[device_id].AddEvent(std::move(event));
  }
  events_.clear(); 
}

void RocmTraceCollectorImpl::Export(XSpace* space) {
  uint64_t end_gputime_ns = get_timestamp();
  XPlaneBuilder host_plane(FindOrAddMutablePlaneWithName(
      space, tsl::profiler::kRoctracerApiPlaneName));

  for (int device_ordinal = 0; device_ordinal < num_gpus_; ++device_ordinal) {
    std::string name = GpuPlaneName(device_ordinal);
    XPlaneBuilder device_plane(FindOrAddMutablePlaneWithName(space, name));
    device_plane.SetId(device_ordinal);
    // Calculate device capabilities before flushing, so that device
    // properties are available to the occupancy calculator in export().
    per_device_collector_[device_ordinal].GetDeviceCapabilities(device_ordinal,
                                                                &device_plane);
    per_device_collector_[device_ordinal].Export(
        start_walltime_ns_, start_gputime_ns_, end_gputime_ns, &device_plane,
        &host_plane);
    NormalizeTimeStamps(&device_plane, start_walltime_ns_);
  }
  NormalizeTimeStamps(&host_plane, start_walltime_ns_);
}

std::unique_ptr<RocmTraceCollector> CreateRocmCollector(
    const RocmTraceCollectorOptions& options, const uint64_t start_walltime_ns,
    const uint64_t start_gputime_ns) {
  return std::make_unique<RocmTraceCollectorImpl>(options, start_walltime_ns,
                                                  start_gputime_ns);
}

}  // namespace profiler
}  // namespace xla
