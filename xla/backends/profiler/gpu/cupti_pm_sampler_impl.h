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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_PM_SAMPLER_IMPL_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_PM_SAMPLER_IMPL_H_

#include <thread>

#include "absl/status/status.h"
// This class requires the CUPTI PM sampling APIs to be defined and available.
// This means this cannot build with CUDA < 12.6.  Build is controled through
// bazel.
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_pmsampling.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_profiler_host.h"

#include "xla/backends/profiler/gpu/cupti_pm_sampler.h"

namespace xla {
namespace profiler {

// Information related to a decode counters pass over a single device
struct CuptiPmSamplerDecodeInfo {
  CUpti_PmSampling_DecodeStopReason decode_stop_reason =
      CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNT;
  uint8_t overflow = 0;
  size_t num_samples = 0;
  size_t num_completed = 0;
  size_t num_populated = 0;
  int device_id;
  std::vector<SamplerRange> sampler_ranges;
  std::vector<char const*> metrics;
};

// Container class for all CUPTI pm sampling infrastructure
// - Configuration
// - Enablement / disablement
// - Buffer creation
// - Worker thread creation / control
// Decoding of counter data is done in CuptiPmSamplerDecodeThread class
class CuptiPmSamplerDevice {
 public:
  // Device information
  int device_id_;

  // Creates host and sampler objects, all images
  absl::Status CreateConfig();

  // Requires config image, pm sampler object
  absl::Status SetConfig();

  // Requires pm sampler object
  absl::Status StartSampling();

  // Requires pm sampler object
  absl::Status StopSampling();

  // Requires pm sampler object, destoys pm sampler object
  absl::Status DisableSampling();

  // Collect sampling data
  // Requires pm sampler object, counter data image, fetches data from hw
  // buffer into counter data image
  absl::Status FillCounterDataImage(CuptiPmSamplerDecodeInfo& decode_info);

  // Requires counter data image
  absl::Status GetSampleCounts(CuptiPmSamplerDecodeInfo& decode_info);

  // Requires host object, pm sampler object, counter data image, metric
  // names, returns sample time, metric values
  absl::Status GetSample(SamplerRange& sample, size_t index);

  // Requires pm sampler object, counter data image, (re)initializes it
  absl::Status InitializeCounterDataImage();

  // Restores image from backup (faster than re-initializing)
  absl::Status RestoreCounterDataImage();

  // Simple warning, needed in multiple spots
  void WarnPmSamplingMetrics();

  // Constructor provides all configuration needed to set up sampling on a
  // single device
  CuptiPmSamplerDevice(int device_id, CuptiInterface* cupti_interface,
                       CuptiPmSamplerOptions* options);

  // Destructor cleans up all images and objects
  ~CuptiPmSamplerDevice();

 private:
  // Internal state
  size_t num_metrics_;
  size_t max_samples_;
  size_t hw_buf_size_;
  size_t sample_interval_ns_;
  std::vector<char const*> default_c_metrics_{
      "sm__cycles_active.sum", "sm__inst_executed_pipe_fmalite.sum",
      "pcie__read_bytes.sum", "pcie__write_bytes.sum"};
  bool warnedMetricsConfig_ = false;

  // CUPTI PM sampling objects
  // Declared roughly in order of initialization
  std::string chipName_;
  std::vector<char const*> c_metrics_;
  std::vector<uint8_t> counter_availability_image_;
  CUpti_Profiler_Host_Object* host_obj_ = nullptr;
  std::vector<uint8_t> config_image_;
  CUpti_PmSampling_Object* sampling_obj_ = nullptr;
  std::vector<uint8_t> counter_data_image_;
  std::vector<uint8_t> counter_data_image_backup_;

  // XLA interface to CUPTI
  // Needed both to call PM sampling APIs and stringify CUPTI errors
  CuptiInterface* cupti_interface_;

  // Configuration calls
  absl::Status GetChipName();
  absl::Status DeviceSupported();
  absl::Status CreateCounterAvailabilityImage();

  // Requires counter availability image
  absl::Status CreateProfilerHostObj();

  // Requires profiler host object
  absl::Status CreateConfigImage();

  // Requires config image
  size_t NumPasses();

  absl::Status InitializeProfilerAPIs();
  absl::Status CreatePmSamplerObject();

  // Requires pm sampler object
  absl::Status CreateCounterDataImage();

  // Clean up
  void DestroyCounterAvailabilityImage();
  void DestroyConfigImage();
  void DestroyCounterDataImage();
  void DestroyProfilerHostObj();
  void DestroyPmSamplerObject();
};

// Container for PM sampling decode thread
// Responsible for fetching PM sampling data from device and providing it to
// handler or other external container
class CuptiPmSamplerDecodeThread {
 public:
  CuptiPmSamplerDecodeThread(
      std::vector<std::shared_ptr<CuptiPmSamplerDevice>> devs,
      CuptiPmSamplerOptions* options);

  // Signal thread to exit; join thread
  ~CuptiPmSamplerDecodeThread() {
    nextThdState_ = kStateExiting;
    thd_->join();
  }

  // Signal and test for state transitions
  bool IsThdInitialized() { return current_thd_state_ == kStateInitialized; }
  void ThdIsInitialized() { current_thd_state_ = kStateInitialized; }

  void EnableThd() { nextThdState_ = kStateEnabled; }
  bool ShouldThdEnable() { return nextThdState_ == kStateEnabled; }
  void ThdIsEnabled() { current_thd_state_ = kStateEnabled; }
  bool IsThdEnabled() { return current_thd_state_ == kStateEnabled; }

  void DisableThd() { nextThdState_ = kStateDisabled; };
  bool ShouldThdDisable() { return nextThdState_ == kStateDisabled; }
  void ThdIsDisabled() { current_thd_state_ = kStateDisabled; }
  bool IsThdDisabled() { return current_thd_state_ == kStateDisabled; }

  void ExitThd() { nextThdState_ = kStateExiting; }
  bool ShouldThdExit() { return nextThdState_ == kStateExiting; }
  void ThdIsExiting() { current_thd_state_ = kStateExiting; }
  bool IsThdExiting() { return current_thd_state_ == kStateExiting; }

 private:
  // Spin wait sleep period, set to the min of this and all device periods
  // Space to asynchronously initialize this class and the thread it spawns
  absl::Duration decode_period_ = absl::Seconds(1);

  size_t num_metrics_;
  std::vector<char const*> c_metrics_;
  std::vector<std::string> metrics_;

  void (*process_samples)(PmSamples* samples) = nullptr;

  enum ThdState {
    // Thread is starting, not yet ready to be enabled
    kStateUninitialized,
    // Thread is ready for enablement but decoding has not yet been triggered
    kStateInitialized,
    // Thread is enabled, polling for metrics from all devices
    kStateEnabled,
    // Thread is disabled, not polling for metrics, but could be re-enabled
    kStateDisabled,
    // Thread is finishing and guaranteed to return, allowing join
    kStateExiting
  };

  // Current state of the thread (only set by worker thread)
  volatile ThdState current_thd_state_ = kStateUninitialized;

  // State thread should transition to (only set by main thread)
  volatile ThdState nextThdState_ = kStateInitialized;

  // Thread handle
  std::thread* thd_;

  // Function run by thd_
  static void ThdFunc(CuptiPmSamplerDecodeThread* control);

  // Isolate the main decode loop
  static void ThdFuncDecodeUntilDisabled(CuptiPmSamplerDecodeThread* control);

  // Devices to decode by this thread
  std::vector<std::shared_ptr<CuptiPmSamplerDevice>> devs_;
};

// Full implementation of CuptiPmSampler
class CuptiPmSamplerImpl : public CuptiPmSampler {
 public:
  // Constructor
  CuptiPmSamplerImpl() = default;

  // Not copyable or movable
  CuptiPmSamplerImpl(const CuptiPmSamplerImpl&) = delete;
  CuptiPmSamplerImpl& operator=(const CuptiPmSamplerImpl&) = delete;

  // Destructor
  ~CuptiPmSamplerImpl() override = default;

  // Initialize the PM sampler, but do not start sampling or decoding
  absl::Status Initialize(CuptiInterface* cupti_interface, size_t num_gpus,
                          CuptiPmSamplerOptions* options) override;

  // Start sampling and decoding
  absl::Status StartSampler() override;

  // Stop sampling and decoding
  absl::Status StopSampler() override;

  // Deinitialize the PM sampler
  absl::Status Deinitialize() override;

 private:
  // Interface is (at least, partially) initialized
  bool initialized_ = false;
  // Interface is (at least, partially) enabled
  bool enabled_ = false;
  // All PM sampler per-device objects
  std::vector<std::shared_ptr<CuptiPmSamplerDevice>> devices_;
  // All PM sampler per-decode-thread objects
  std::vector<std::unique_ptr<CuptiPmSamplerDecodeThread>> threads_;
  // How long to sleep before ending the decode threads
  absl::Duration decode_stop_delay_;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_PM_SAMPLER_IMPL_H_
