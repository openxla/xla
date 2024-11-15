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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_H_

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "absl/types/optional.h"

#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/status.h"
#include "tsl/platform/types.h"

#include "xla/stream_executor/rocm/roctracer_wrapper.h"
#include "xla/backends/profiler/gpu/rocm_collector.h"

#ifdef buffered_api_tracing_client_EXPORTS
#    define CLIENT_API __attribute__((visibility("default")))
#else
#    define CLIENT_API
#endif

namespace xla {
namespace profiler {

struct RocmTracerOptions {
  std::set<uint32_t> api_tracking_set;  // actual api set we want to profile

  // map of domain --> ops for which we need to enable the API callbacks
  // If the ops vector is empty, then enable API callbacks for entire domain
  absl::flat_hash_map<rocprofiler_buffer_tracing_kind_t, std::vector<uint32_t> > api_callbacks;

  // map of domain --> ops for which we need to enable the Activity records
  // If the ops vector is empty, then enable Activity records for entire domain
  absl::flat_hash_map<rocprofiler_buffer_tracing_kind_t, std::vector<uint32_t> > activity_tracing;
};

class RocmTracer {
public:
    // Returns a pointer to singleton RocmTracer.
    static RocmTracer* GetRocmTracerSingleton();

    // Only one profile session can be live in the same time.
    bool IsAvailable() const;
    void Enable(const RocmTracerOptions& options, RocmTraceCollector* collector);

    static uint64_t GetTimestamp();
    static int NumGpus();


    void setup() CLIENT_API;
    void start() CLIENT_API;
    void stop() CLIENT_API;
    void shutdown() CLIENT_API;
    void identify(uint64_t corr_id) CLIENT_API;

private:
    // Private constructor for singleton
    RocmTracer() : is_available_(true), num_gpus_(NumGpus()) {
        LOG(INFO) << "RocmTracer initialized...";
    }

    // Private destructor
    ~RocmTracer() {
        LOG(INFO) << "RocmTracer destroyed...";
    }

    bool is_available_; // availability status
    int num_gpus_; 
    std::optional<RocmTracerOptions> options_;
    RocmTraceCollector* collector_ = nullptr;

    // Disable copy constructor and assignment operator
    RocmTracer(const RocmTracer&) = delete;
    RocmTracer& operator=(const RocmTracer&) = delete;
};  // end of RocmTracer

}  // end of namespace profiler
}  // end of namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_H_
