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

#include <cstdint>
#include <limits>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "xla/stream_executor/rocm/roctracer_wrapper.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "xla/tsl/profiler/utils/parse_annotation.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"

namespace xla {
namespace profiler {

using tsl::mutex;
using tsl::mutex_lock;
using tsl::profiler::XEvent;
using tsl::profiler::XLineBuilder;
using tsl::profiler::XPlaneBuilder;
using tsl::profiler::XSpace;

struct MemcpyDetails {
  // The amount of data copied for memcpy events.
  size_t num_bytes;
  // The destination device for peer-2-peer communication (memcpy). The source
  // device is implicit: it's the current device.
  uint32_t destination;
  // Whether or not the memcpy is asynchronous.
  bool async;
};

struct MemAllocDetails {
  // The amount of data requested for cudaMalloc events.
  uint64_t num_bytes;
};

struct MemsetDetails {
  // The number of memory elements getting set
  size_t num_bytes;
  // Whether or not the memset is asynchronous.
  bool async;
};

struct KernelDetails {
  // The number of registers used in this kernel.
  uint32_t registers_per_thread;
  // The amount of shared memory space used by a thread block.
  uint32_t static_shared_memory_usage;
  // The amount of dynamic memory space used by a thread block.
  uint32_t dynamic_shared_memory_usage;
  // X-dimension of a thread block.
  uint32_t block_x;
  // Y-dimension of a thread block.
  uint32_t block_y;
  // Z-dimension of a thread block.
  uint32_t block_z;
  // X-dimension of a grid.
  uint32_t grid_x;
  // Y-dimension of a grid.
  uint32_t grid_y;
  // Z-dimension of a grid.
  uint32_t grid_z;

  // kernel address. Used for calculating core occupancy
  void* func_ptr;
};

inline std::string ToXStat(const KernelDetails& kernel_info,
                           double occupancy_pct) {
  return absl::StrCat(
      "regs:", kernel_info.registers_per_thread,
      " static_shared:", kernel_info.static_shared_memory_usage,
      " dynamic_shared:", kernel_info.dynamic_shared_memory_usage,
      " grid:", kernel_info.grid_x, ",", kernel_info.grid_y, ",",
      kernel_info.grid_z, " block:", kernel_info.block_x, ",",
      kernel_info.block_y, ",", kernel_info.block_z,
      " occ_pct:", occupancy_pct);
}

enum class RocmTracerEventType {
  Unsupported = 0,
  Kernel,
  MemcpyH2D,
  MemcpyD2H,
  MemcpyD2D,
  MemcpyP2P,
  MemcpyOther,
  MemoryAlloc,
  MemoryFree,
  Memset,
  Synchronization,
  Generic,
};

const char* GetRocmTracerEventTypeName(const RocmTracerEventType& type);

enum class RocmTracerEventSource {
  Invalid = 0,
  ApiCallback,
  Activity,
};

const char* GetRocmTracerEventSourceName(const RocmTracerEventSource& source);

enum class RocmTracerEventDomain {
  InvalidDomain = 0,
  HIP_API,
  HIP_OPS,
};

const char* GetRocmTracerEventDomainName(const RocmTracerEventDomain& domain);
// RocmTracerSyncTypes forward declaration
enum class RocmTracerSyncTypes;

struct SynchronizationDetails {
  RocmTracerSyncTypes sync_type;
};

struct RocmTracerEvent {
  static constexpr uint32_t kInvalidDeviceId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint64_t kInvalidThreadId =
      std::numeric_limits<uint64_t>::max();
  static constexpr uint32_t kInvalidCorrelationId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint64_t kInvalidStreamId =
      std::numeric_limits<uint64_t>::max();
  RocmTracerEventType type;
  RocmTracerEventSource source = RocmTracerEventSource::Invalid;
  RocmTracerEventDomain domain;
  std::string name;
  // This points to strings in AnnotationMap, which should outlive the point
  // where serialization happens.
  absl::string_view annotation;
  absl::string_view roctx_range;
  uint64_t start_time_ns = 0;
  uint64_t end_time_ns = 0;
  uint32_t device_id = kInvalidDeviceId;
  uint32_t correlation_id = kInvalidCorrelationId;
  uint64_t thread_id = kInvalidThreadId;
  uint64_t stream_id = kInvalidStreamId;

  union {
    MemcpyDetails memcpy_info;                    // If type == Memcpy*
    MemsetDetails memset_info;                    // If type == Memset*
    MemAllocDetails memalloc_info;                // If type == MemoryAlloc
    KernelDetails kernel_info;                    // If type == Kernel
    SynchronizationDetails synchronization_info;  // If type == Synchronization
  };
};

struct RocmTraceCollectorOptions {
  // Maximum number of events to collect from callback API; if -1, no limit.
  // if 0, the callback API is enabled to build a correlation map, but no
  // events are collected.
  uint64_t max_callback_api_events;
  // Maximum number of events to collect from activity API; if -1, no limit.
  uint64_t max_activity_api_events;
  // Maximum number of annotation strings that we can accommodate.
  uint64_t max_annotation_strings;
  // Number of GPUs involved.
  uint32_t num_gpus;
};

class AnnotationMap {
 public:
  explicit AnnotationMap(uint64_t max_size) : max_size_(max_size) {}
  void Add(uint32_t correlation_id, const std::string& annotation);
  absl::string_view LookUp(uint32_t correlation_id);

 private:
  struct AnnotationMapImpl {
    // The population/consumption of annotations might happen from multiple
    // callback/activity api related threads.
    absl::Mutex mutex;
    // Annotation tends to be repetitive, use a hash_set to store the strings,
    // an use the reference to the string in the map.
    absl::node_hash_set<std::string> annotations;
    absl::flat_hash_map<uint32_t, absl::string_view> correlation_map;
  };
  const uint64_t max_size_;
  AnnotationMapImpl map_;

 public:
  // Disable copy and move.
  AnnotationMap(const AnnotationMap&) = delete;
  AnnotationMap& operator=(const AnnotationMap&) = delete;
};

// for roctracer (v1)
#if TF_ROCM_VERSION < 60300

class RocmTraceCollector {
 public:
  explicit RocmTraceCollector(const RocmTraceCollectorOptions& options)
      : options_(options), annotation_map_(options.max_annotation_strings) {}
  virtual ~RocmTraceCollector() {}

  virtual void AddEvent(RocmTracerEvent&& event, bool is_auxiliary) = 0;
  virtual void OnEventsDropped(const std::string& reason,
                               uint32_t num_events) = 0;
  virtual void Flush() = 0;
  virtual void Export(XSpace* space) = 0;

  AnnotationMap* annotation_map() { return &annotation_map_; }

 protected:
  RocmTraceCollectorOptions options_;

 private:
  AnnotationMap annotation_map_;

 public:
  // Disable copy and move.
  RocmTraceCollector(const RocmTraceCollector&) = delete;
  RocmTraceCollector& operator=(const RocmTraceCollector&) = delete;
};

#else
// for rocprofiler-sdk (v3)

class RocmTraceCollector {
 public:
  explicit RocmTraceCollector(const RocmTraceCollectorOptions& options)
      : options_(options) {}
  virtual ~RocmTraceCollector() {}

  virtual void AddEvent(RocmTracerEvent&& event, bool is_auxiliary) = 0;
  virtual void OnEventsDropped(const std::string& reason,
                               uint32_t num_events) = 0;
  virtual void Flush() = 0;
  virtual void Export(XSpace* space) = 0;

 protected:
  RocmTraceCollectorOptions options_;

 public:
  // Disable copy and move.
  RocmTraceCollector(const RocmTraceCollector&) = delete;
  RocmTraceCollector& operator=(const RocmTraceCollector&) = delete;
};
#endif

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

// FIXME: rocprofiler-sdk does not have this one yet
struct OccupancyStats {
  double occupancy_pct = 0.0;
  int min_grid_size = 0;
  int suggested_block_size = 0;
};

class PerDeviceCollector {
 public:
  void Export(uint64_t start_walltime_ns, uint64_t start_gputime_ns,
              uint64_t end_gputime_ns, XPlaneBuilder* device_plane,
              XPlaneBuilder* host_plane);

  PerDeviceCollector() = default;

  void AddEvent(RocmTracerEvent&& event);
  void GetDeviceCapabilities(int32_t device_ordinal,
                             XPlaneBuilder* device_plane);

 private:
  OccupancyStats GetOccupancy(const RocmDeviceOccupancyParams& params) const;
  void CreateXEvent(const RocmTracerEvent& event, XPlaneBuilder* plane,
                    uint64_t start_gpu_ns, uint64_t end_gpu_ns,
                    XLineBuilder* line);
  void SortByStartTime();
  bool IsHostEvent(const RocmTracerEvent& event, tsl::int64* line_id);

 private:
  mutex events_mutex_;
  std::vector<RocmTracerEvent> events_ TF_GUARDED_BY(events_mutex_);
  absl::flat_hash_map<RocmDeviceOccupancyParams, OccupancyStats>
      occupancy_cache_;
  hipDeviceProp_t device_properties_;
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

  void AddEvent(RocmTracerEvent&& event, bool is_auxiliary) override;
  void Flush() override;
  void Export(XSpace* space) override;

  void OnEventsDropped(const std::string& reason,
                       uint32_t correlation_id) override {
    LOG(INFO) << "RocmTracerEvent dropped (correlation_id=" << correlation_id
              << ",) : " << reason << ".";
  }

 private:
  std::atomic<int> num_callback_events_;
  std::atomic<int> num_activity_events_;
  uint64_t start_walltime_ns_;
  uint64_t start_gputime_ns_;
  int num_gpus_;

  mutex event_maps_mutex_;
  absl::flat_hash_map<uint32_t, RocmTracerEvent> api_events_map_
      TF_GUARDED_BY(event_maps_mutex_);

  /* Some apis such as MEMSETD32 (based on an observation with ResNet50),
   trigger multiple HIP ops domain activities. We keep them in a vector and
   merge them with api activities at flush time.
 */
  absl::flat_hash_map<uint32_t, std::vector<RocmTracerEvent>>
      activity_ops_events_map_ TF_GUARDED_BY(event_maps_mutex_);
  // This is for the APIs that we track because we need some information from
  // them to populate the corresponding activity that we actually track.
  absl::flat_hash_map<uint32_t, RocmTracerEvent> auxiliary_api_events_map_
      TF_GUARDED_BY(event_maps_mutex_);

  std::vector<RocmTracerEvent> ApiActivityInfoExchange()
      TF_EXCLUSIVE_LOCKS_REQUIRED(event_maps_mutex_);

  absl::node_hash_map<uint32_t, PerDeviceCollector> per_device_collector_;
};  // RocmTraceCollectorImpl

std::unique_ptr<RocmTraceCollector> CreateRocmCollector(
    const RocmTraceCollectorOptions& options, const uint64_t start_walltime_ns,
    const uint64_t start_gputime_ns);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_COLLECTOR_H_