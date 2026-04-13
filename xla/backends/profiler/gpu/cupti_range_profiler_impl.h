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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_RANGE_PROFILER_IMPL_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_RANGE_PROFILER_IMPL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_profiler_host.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_range_profiler.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"
#include "xla/backends/profiler/gpu/cupti_range_profiler.h"

namespace xla {
namespace profiler {

// Per-device state for CUPTI range profiling using the new Range Profiling API.
// Owns the CUpti_RangeProfiler_Object* and all associated data images.
class CuptiRangeProfilerDevice {
 public:
  CuptiRangeProfilerDevice(int device_id,
                           const CuptiRangeProfilerOptions& options);
  ~CuptiRangeProfilerDevice();

  // Not copyable or movable.
  CuptiRangeProfilerDevice(const CuptiRangeProfilerDevice&) = delete;
  CuptiRangeProfilerDevice(CuptiRangeProfilerDevice&&) = delete;
  CuptiRangeProfilerDevice& operator=(const CuptiRangeProfilerDevice&) =
      delete;
  CuptiRangeProfilerDevice& operator=(CuptiRangeProfilerDevice&&) = delete;

  // Configuration: creates host object and config image.
  absl::Status CreateConfig();

  // Enable range profiling on the device context, creating the profiler object.
  absl::Status Enable();
  // Create counter data image via the range profiler object.
  absl::Status CreateCounterDataImage();
  // Set configuration (config image + counter data image) on the profiler.
  absl::Status SetConfig();

  // Per-pass lifecycle.
  absl::Status Start();
  absl::Status Stop(bool* all_passes_collected);

  // Range delimiters.
  absl::Status PushRange(absl::string_view name);
  absl::Status PopRange();

  // Post-collection: decode counter data and evaluate metrics.
  absl::Status DecodeData();
  absl::Status Decode(std::vector<RangeResult>* results);

  // Query CUPTI for per-metric properties (description, hw unit).
  // Returns one entry per enabled metric. Falls back gracefully if the
  // API is unavailable (e.g. older CUPTI).
  std::vector<MetricProperties> QueryMetricProperties();

  // Disable range profiling and destroy the profiler object.
  absl::Status Disable();

  // Query number of passes required.
  absl::Status GetNumPasses(int* num_passes);

  // Accessors.
  int device_id() const { return device_id_; }
  const std::vector<std::string>& enabled_metrics() const {
    return enabled_metrics_;
  }

 private:
  int device_id_;
  CuptiInterface* cupti_interface_;

  // Metric configuration.
  std::vector<std::string> config_metrics_;
  std::vector<std::string> enabled_metrics_;
  std::vector<const char*> c_metrics_;

  // CUPTI objects.
  std::string chip_name_;
  std::vector<uint8_t> counter_availability_image_;
  CUpti_Profiler_Host_Object* host_obj_ = nullptr;
  std::vector<uint8_t> config_image_;
  std::vector<uint8_t> counter_data_image_;

  // Range profiler object (new API). Null until Enable() is called.
  CUpti_RangeProfiler_Object* range_profiler_obj_ = nullptr;

  // Internal initialization steps.
  absl::Status GetChipName();
  absl::Status DeviceSupported();
  absl::Status CreateCounterAvailabilityImage();
  absl::Status CreateProfilerHostObj();
  absl::Status CreateConfigImage();

  // Cleanup helpers.
  void DestroyProfilerHostObj();
};

// Full implementation of CuptiRangeProfiler using the new Range Profiling API.
// Orchestrates range profiling across all devices. Unlike PM sampling, range
// profiling is synchronous: the caller drives passes via BeginPass/EndPass.
class CuptiRangeProfilerImpl : public CuptiRangeProfiler {
 public:
  static absl::StatusOr<std::unique_ptr<CuptiRangeProfilerImpl>> Create(
      int num_gpus, const CuptiRangeProfilerOptions& options);

  int NumPasses() const override { return num_passes_; }

  absl::Status BeginPass() override;
  absl::Status EndPass() override;
  absl::Status PushRange(absl::string_view name) override;
  absl::Status PopRange() override;
  absl::Status FlushAndDecode() override;
  absl::Status Deinitialize() override;

 private:
  CuptiRangeProfilerImpl() = default;

  absl::Status Initialize(int num_gpus,
                          const CuptiRangeProfilerOptions& options);

  bool initialized_ = false;
  int num_passes_ = 1;
  CuptiRangeProfilerOptions options_;
  std::vector<std::unique_ptr<CuptiRangeProfilerDevice>> devices_;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_RANGE_PROFILER_IMPL_H_
