/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_TRACER_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_TRACER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <thread>

#include "absl/status/status.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/include/nvtx3/nvToolsExt.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"
#include "tsl/platform/types.h"

namespace xla {
namespace profiler {

// Provide safe types if CUPTI_PM_SAMPLING is not defined
// (And therefor CUPTI PM sampling headers are not included)
#if CUPTI_PM_SAMPLING == 0
typedef int CUpti_PmSampling_DecodeStopReason;
constexpr CUpti_PmSampling_DecodeStopReason
    CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNT = 0;
#endif

// Information related to a decode counters pass over a single device
struct PmSamplingDecodeInfo {
  CUpti_PmSampling_DecodeStopReason decode_stop_reason =
      CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNT;
  uint8_t overflow_ = 0;
  size_t num_samples = 0;
  size_t num_completed = 0;
  size_t num_populated = 0;
  int device_id;
  std::vector<SamplerRange> sampler_ranges;
  std::vector<char const*> metrics;
};

struct PmSamplingConfig;

struct CuptiTracerOptions {
  bool required_callback_api_events = true;
  // The callback ids that will be enabled and monitored, if empty, all
  // Callback ids to be enabled using Callback API.
  // We only care CUPTI_CB_DOMAIN_DRIVER_API domain for now. It is kind of
  // redundant to have both CUPTI_CB_DOMAIN_DRIVER_API and
  // CUPTI_CB_DOMAIN_RUNTIME_API.
  std::vector<CUpti_driver_api_trace_cbid_enum> cbids_selected;
  // Activity kinds to be collected using Activity API. If empty, the Activity
  // API is disable.
  std::vector<CUpti_ActivityKind> activities_selected;
  // Whether to call cuptiFinalize.
  bool cupti_finalize = false;
  // Whether to call cuCtxSynchronize for each device before Stop().
  bool sync_devices_before_stop = false;
  // Whether to enable NVTX tracking, we need this for TensorRT tracking.
  bool enable_nvtx_tracking = false;
  // PM sampling configuration (defaults are 2khz rate, 100ms decode)
  // Only read during creation of a PM sampling object, later changes have 
  // no effect
  struct PmSamplingConfig* pm_sampling_config;
};

// Container class for all CUPTI pm sampling infrastructure
// - Configuration
// - Enablement / disablement
// - Buffer creation
// - Worker thread creation / control
// Decoding of counter data is done in PmSamplingDecodeThread class
class PmSamplingDevice {
  // Internal state
  // Declared roughly in order of initialization
  char const* chipName_ = nullptr;
  std::vector<uint8_t> counter_availability_image_;
  CUpti_Profiler_Host_Object* host_obj_ = nullptr;
  std::vector<uint8_t> config_image_;
  CUpti_PmSampling_Object* sampling_obj_ = nullptr;
  std::vector<uint8_t> counter_data_image_;
  std::unique_ptr< const std::vector<uint8_t> > p_counter_data_image_backup_;

  // XLA interface to CUPTI
  // Needed both to call PM sampling APIs and stringify CUPTI errors
  CuptiInterface* cupti_interface_;

  // Configuration calls
  absl::Status GetChipName();
  absl::Status DeviceSupported();
  absl::Status CreateCounterAvailabilityImage();

  // Requires counter availability image
  absl::Status CreatProfilerHostObj();           

  // Requires profiler host object
  absl::Status CreateConfigImage();

  // Requires config image
  size_t NumPasses();

  absl::Status InitializeProfilerAPIs();
  absl::Status CreatePmSamplerObject();

  // Requires pm sampler object
  absl::Status CreateCounterDataImage();

  // Clean up
  absl::Status DestroyCounterAvailabilityImage();
  absl::Status DestroyProfilerHostObj();
  absl::Status DestroyConfigImage();
  absl::Status DestroyPmSamplerObject();
  absl::Status UnInitializeProfilerAPIs();
  absl::Status DestroyCounterDataImage();

  std::vector<char const*> default_metrics_{
    "sm__cycles_active.sum",
    "sm__inst_executed_pipe_fmalite.sum",
    "pcie__read_bytes.sum",
    "pcie__write_bytes.sum"
  };

  public:
  // Device information
  int device_id_;

  // PM sampling public configuration
  struct PmSamplingConfig* config_;

  // Creates host and sampler objects, all images
  absl::Status CreateConfig();

  // Requires config image, pm sampler object
  absl::Status SetConfig();

  // Requires pm sampler object
  absl::Status StartSampling();

  // Requires pm sampler object
  absl::Status StopSampling();

  // Requires pm sampler object
  absl::Status DisableSampling();

  // Collect sampling data
  // Requires pm sampler object, counter data image, fetches data from hw
  // buffer into counter data image
  absl::Status FillCounterDataImage(struct PmSamplingDecodeInfo& decodeInfo);

  // Requires counter data image
  absl::Status GetSampleCounts(struct PmSamplingDecodeInfo& decodeInfo);

  // Requires host object, pm sampler object, counter data image, metric
  // names, returns sample time, metric values
  absl::Status GetSample(SamplerRange& sample, size_t index);

  // Requires pm sampler object, counter data image, (re)initializes it
  absl::Status InitializeCounterDataImage();

  // Restores image from backup (faster than re-initializing)
  absl::Status RestoreCounterDataImage();

  // Constructor provides all configuration needed to set up sampling on a
  // single device
  PmSamplingDevice(int device_id, struct PmSamplingConfig* config);
};

// Container for PM sampling decode thread
// Responsible for fetching PM sampling data from device and providing it to
// handler or other external container
class PmSamplingDecodeThread {
  enum thd_state {
    // Thread is starting, not yet ready to be enabled
    state_uninitialized_,
    // Thread is ready for enablement but decoding has not yet been triggered
    state_initialized_,
    // Thread is enabled, polling for metrics from all devices
    state_enabled_,
    // Thread is disabled, not polling for metrics, but could be re-enabled
    state_disabled_,
    // Thread is finishing and guaranteed to return, allowing join
    state_exiting_
  };

  // Current state of the thread (only set by worker thread)
  volatile thd_state current_thd_state_ = state_uninitialized_;
    
  // State thread should transition to (only set by main thread)
  volatile thd_state nextThdState_ = state_initialized_;

  // Thread handle
  std::thread* thd_;

  // Function run by thd_
  static void ThdFunc(PmSamplingDecodeThread* control);

  // Isolate the main decode loop
  static void ThdFuncDecodeUntilDisabled(PmSamplingDecodeThread* control);

  // Devices to decode by this thread
  std::vector< std::shared_ptr<PmSamplingDevice> > devs_;

  public:
  // Spin wait sleep period, set to the min of this and all device periods
  // Space to asynchronously initialize this class and the thread it spawns
  absl::Duration decode_period_ = absl::Seconds(1);
  PmSamplingDecodeThread(std::vector< std::shared_ptr<PmSamplingDevice> >
      devs);

  // Signal thread to exit; join thread
  ~PmSamplingDecodeThread() {
    nextThdState_ = state_exiting_;
    thd_->join();
  }

  // Signal and test for state transitions
  bool IsThdInitialized() { return current_thd_state_ == state_initialized_; }
  void ThdIsInitialized() { current_thd_state_ = state_initialized_; }

  void EnableThd() { nextThdState_ = state_enabled_; }
  bool ShouldThdEnable() { return nextThdState_ == state_enabled_; }
  void ThdIsEnabled() { current_thd_state_ = state_enabled_; }
  bool IsThdEnabled() { return current_thd_state_ == state_enabled_; }

  void DisableThd() { nextThdState_ = state_disabled_; };
  bool ShouldThdDisable() { return nextThdState_ == state_disabled_; }
  void ThdIsDisabled() { current_thd_state_ = state_disabled_; }
  bool IsThdDisabled() { return current_thd_state_ == state_disabled_; }

  void ExitThd() { nextThdState_ = state_exiting_; }
  bool ShouldThdExit() { return nextThdState_ == state_exiting_; }
  void ThdIsExiting() { current_thd_state_ = state_exiting_; }
  bool IsThdExiting() { return current_thd_state_ == state_exiting_; }
};

// Should be safe on all hardware
struct PmSamplingConfig {
  // Whether to enable PM sampling
  bool enable_pm_sampling = false;
  // List of metrics to enable
  std::vector<const char*> metrics{};
  // 64MB hardware buffer (on device)
  size_t hw_buf_size = 64 * 1024 * 1024;
  // Sample interval of 500,000ns = 2khz
  size_t sample_interval = 500000;
  // Decode thread triggers every 100ms (should have 200 samples @ 2khz)
  absl::Duration decode_period = absl::Milliseconds(100);
  // Maximum samples to allocate host space for, 2.5x expected
  size_t max_samples = 500;
  // Devices per decode thread
  size_t devs_per_decode_thd = 8;
  // What to do with samples once gathered
  // Note, must be thread-safe - may be called by multiple decode threads
  // simultaneously
  void (*process_samples)(struct PmSamplingDecodeInfo* info) = nullptr;
  // All PM sampling device objects
  // Do not set manually
  std::vector< std::shared_ptr<PmSamplingDevice> > devices;
  // All PM sampling decode thread objects
  // Do not set manually
  std::vector< std::unique_ptr<PmSamplingDecodeThread> > threads;
};

class CuptiTracer;

class CuptiDriverApiHook {
 public:
  virtual ~CuptiDriverApiHook() {}

  virtual absl::Status OnDriverApiEnter(
      int device_id, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
      const CUpti_CallbackData* callback_info) = 0;
  virtual absl::Status OnDriverApiExit(
      int device_id, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
      const CUpti_CallbackData* callback_info) = 0;
  virtual absl::Status SyncAndFlush() = 0;
};

// The class use to enable cupti callback/activity API and forward the collected
// trace events to CuptiTraceCollector. There should be only one CuptiTracer
// per process.
class CuptiTracer {
 public:
  // Not copyable or movable
  CuptiTracer(const CuptiTracer&) = delete;
  CuptiTracer& operator=(const CuptiTracer&) = delete;

  // Returns a pointer to singleton CuptiTracer.
  static CuptiTracer* GetCuptiTracerSingleton();

  // Only one profile session can be live in the same time.
  bool IsAvailable() const;
  bool NeedRootAccess() const { return need_root_access_; }

  absl::Status Enable(const CuptiTracerOptions& option, CuptiTraceCollector*
      collector);
  void Disable();

  // Control threads could periodically call this function to flush the
  // collected events to the collector. Note that this function will lock the
  // per-thread data mutex and may impact the performance.
  absl::Status FlushEventsToCollector();

  // Sets the activity event buffer flush period. Set to 0 to disable the
  // periodic flush. Before using the FlushEventsToCollector() function, user
  // either need to set the activity flush period or call the
  // FlushActivityBuffers()
  absl::Status SetActivityFlushPeriod(uint32_t period_ms);

  // Force the cupti to flush activity buffers to this tracer.
  absl::Status FlushActivityBuffers();

  absl::Status HandleCallback(CUpti_CallbackDomain domain,
                              CUpti_CallbackId cbid,
                              const CUpti_CallbackData* callback_info);

  // Returns a buffer and its size for CUPTI to store activities. This buffer
  // will be reclaimed when CUPTI makes a callback to ProcessActivityBuffer.
  void RequestActivityBuffer(uint8_t** buffer, size_t* size);

  // Parses CUPTI activity events from activity buffer, and emits events for
  // CuptiTraceCollector. This function is public because called from registered
  // callback.
  absl::Status ProcessActivityBuffer(CUcontext context, uint32_t stream_id,
                                     uint8_t* buffer, size_t size);

  static uint64_t GetTimestamp();
  static int NumGpus();
  // Returns the error (if any) when using libcupti.
  static std::string ErrorIfAny();

  // Returns true if the number of annotation strings is too large. The input
  // count is the per-thread count.
  bool TooManyAnnotationStrings(size_t count) const;

  // Returns true if the total number of callback events across all threads
  // is too large.
  bool TooManyCallbackEvents() const;

  void IncCallbackEventCount() {
    num_callback_events_.fetch_add(1, std::memory_order_relaxed);
  }

  bool IsCallbackApiEventsRequired() const {
    return option_.has_value() ? option_->required_callback_api_events : false;
  }

 protected:
  // protected constructor for injecting mock cupti interface for testing.
  explicit CuptiTracer(CuptiInterface* cupti_interface);

 private:
  // Buffer size and alignment, 32K and 8 as in CUPTI samples.
  static constexpr size_t kBufferSizeInBytes = 32 * 1024;

  std::unique_ptr<CuptiActivityBufferManager> activity_buffers_;
  static_assert(std::atomic<size_t>::is_always_lock_free,
                "std::atomic<size_t> is not lock free! This may cause very bad"
                " profiling overhead in some circumstances.");
  std::atomic<size_t> cupti_dropped_activity_event_count_ = 0;
  std::atomic<size_t> num_activity_events_in_dropped_buffer_ = 0;
  std::atomic<size_t> num_activity_events_in_cached_buffer_ = 0;
  std::atomic<size_t> num_callback_events_ = 0;

  // Clear activity_buffers, reset activity event counters.
  void PrepareActivityStart();

  // Empty all per-thread callback annotations, reset callback event counter.
  void PrepareCallbackStart();

  // Gather all per-thread callback events and annotations.
  std::vector<CallbackAnnotationsAndEvents> GatherCallbackAnnotationsAndEvents(
      bool stop_recording);

  absl::Status EnableApiTracing();
  absl::Status EnablePMSampling();
  absl::Status EnableActivityTracing();
  absl::Status DisableApiTracing();
  absl::Status DisablePMSampling();
  absl::Status DisableActivityTracing();
  absl::Status Finalize();
  void ConfigureActivityUnifiedMemoryCounter(bool enable);
  absl::Status HandleNVTXCallback(CUpti_CallbackId cbid,
                                  const CUpti_CallbackData* cbdata);
  absl::Status HandleDriverApiCallback(CUpti_CallbackId cbid,
                                       const CUpti_CallbackData* cbdata);
  absl::Status HandleResourceCallback(CUpti_CallbackId cbid,
                                      const CUpti_CallbackData* cbdata);
  int num_gpus_;
  std::optional<CuptiTracerOptions> option_;
  CuptiInterface* cupti_interface_ = nullptr;
  CuptiTraceCollector* collector_ = nullptr;

  // CUPTI 10.1 and higher need root access to profile.
  bool need_root_access_ = false;

  bool api_tracing_enabled_ = false;
  bool pm_sampling_enabled_ = false;
  // Cupti handle for driver or runtime API callbacks. Cupti permits a single
  // subscriber to be active at any time and can be used to trace Cuda runtime
  // as and driver calls for all contexts and devices.
  CUpti_SubscriberHandle subscriber_;  // valid when api_tracing_enabled_.

  bool activity_tracing_enabled_ = false;

  std::unique_ptr<CuptiDriverApiHook> cupti_driver_api_hook_;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_TRACER_H_
