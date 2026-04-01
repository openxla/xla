/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_RANGE_PROFILER_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_RANGE_PROFILER_H_

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace xla {
namespace profiler {

// Range mode determines how profiling ranges are delimited.
enum class CuptiRangeMode {
  // Manual push/pop from the runner (e.g. per HLO execution).
  kUser,
  // Automatic per-kernel ranges managed by CUPTI.
  kAutoRange,
};

// Configuration for CUPTI range profiling.
// Read once during CuptiRangeProfiler creation; later changes have no effect.
struct CuptiRangeProfilerOptions {
  // Whether to enable range profiling.
  bool enable = false;
  // List of metrics to collect (e.g. "sm__cycles_elapsed.avg").
  std::vector<std::string> metrics;
  // How profiling ranges are delimited.
  CuptiRangeMode range_mode = CuptiRangeMode::kUser;
  // Default range name used when the caller does not supply one.
  std::string range_name = "hlo_execution";
  // Whether to allow metric configurations that require multiple passes.
  // Default is false: most callers (e.g. XLA workloads) cannot replay the
  // workload, so multi-pass configs are rejected at Enable() time.
  // The multi-hlo-runner sets this to true because it controls the repeat loop.
  bool allow_multipass = false;
  // What to do with decoded results after the final pass.
  // Must be safe to call from any thread.
  std::function<void(class RangeProfilerResults*)> process_results;
};

// A single profiled range with its decoded metric values.
// Counter values are aggregated across all collection passes by CUPTI.
struct RangeResult {
  std::string range_name;
  uint64_t start_timestamp_ns;
  uint64_t end_timestamp_ns;
  // Parallel to the metrics vector in CuptiRangeProfilerOptions / results.
  std::vector<double> metric_values;
};

// Container for all decoded range profiling results from one session.
class RangeProfilerResults {
 public:
  RangeProfilerResults(std::vector<std::string> metrics,
                       std::vector<RangeResult> ranges, int device_id)
      : metrics_(std::move(metrics)),
        ranges_(std::move(ranges)),
        device_id_(device_id) {}

  const std::vector<std::string>& GetMetrics() const { return metrics_; }
  const std::vector<RangeResult>& GetRanges() const { return ranges_; }
  int GetDeviceId() const { return device_id_; }

 private:
  std::vector<std::string> metrics_;
  std::vector<RangeResult> ranges_;
  int device_id_;
};

// Abstract interface for CUPTI range profiling.
//
// Range profiling collects hardware performance counters over explicitly
// delimited ranges of GPU work. Unlike PM sampling, it requires multiple
// passes of the same workload — one pass per counter group. The multi-hlo-
// runner's repeat loop maps naturally to this: each repeat is one pass.
//
// Specialize a full implementation in cupti_range_profiler_impl.h.
class CuptiRangeProfiler {
 public:
  CuptiRangeProfiler() = default;

  // Not copyable or movable.
  CuptiRangeProfiler(const CuptiRangeProfiler&) = delete;
  CuptiRangeProfiler(CuptiRangeProfiler&&) = delete;
  CuptiRangeProfiler& operator=(const CuptiRangeProfiler&) = delete;
  CuptiRangeProfiler& operator=(CuptiRangeProfiler&&) = delete;

  virtual ~CuptiRangeProfiler() = default;

  // Returns the number of passes required for the configured metrics.
  virtual int NumPasses() const = 0;

  // Per-pass lifecycle: called once per repeat of the workload.
  virtual absl::Status BeginPass() = 0;
  virtual absl::Status EndPass() = 0;

  // Range delimiters: bracket the GPU work to be measured within a pass.
  virtual absl::Status PushRange(absl::string_view name) = 0;
  virtual absl::Status PopRange() = 0;

  // Post-collection: flush counter data and decode to metric values.
  virtual absl::Status FlushAndDecode() = 0;

  // Tear down all CUPTI state.
  virtual absl::Status Deinitialize() = 0;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_RANGE_PROFILER_H_
