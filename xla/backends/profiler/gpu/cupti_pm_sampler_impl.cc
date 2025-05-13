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

#include "xla/backends/profiler/gpu/cupti_pm_sampler_impl.h"

#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_pmsampling.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_profiler_target.h"

#include <thread>

#include "absl/status/status.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace profiler {

// Full implementation of CuptiPmSampler
// This class is responsible for managing the PM sampling process, and
// requires the CUPTI PM sampling APIs to be defined and available.
// This means this cannot build with CUDA < 12.6.

// CUPTI params struct definitions are very long, macro it for convenience
// They all have a struct_size field which must be set to type_STRUCT_SIZE
// Many strucs also have a pPriv field which must be null, ie:
// CUpti_Struct_Type var = { CUpti_Struct_Type_STRUCT_SIZE, .pPriv = nullptr }
#define DEF_SIZED_PRIV_STRUCT(type, name) \
  type name = {.structSize = type##_STRUCT_SIZE, .pPriv = nullptr}

#define RETURN_IF_CUPTI_ERROR(expr)                                         \
  do {                                                                      \
    CUptiResult status = (cupti_interface_->expr);                          \
    if (ABSL_PREDICT_FALSE(status != CUPTI_SUCCESS)) {                      \
      const char* errstr = "";                                              \
      cupti_interface_->GetResultString(status, &errstr);                   \
      LOG(ERROR) << "function " << #expr << "failed with error " << errstr; \
      if (status == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES) {                  \
        return absl::PermissionDeniedError("CUPTI needs root access");      \
      } else {                                                              \
        return absl::UnknownError(absl::StrCat("CUPTI error ", errstr));    \
      }                                                                     \
    }                                                                       \
  } while (false)

#define RETURN_IF_CUDA_DRIVER_ERROR(expr)                                   \
  do {                                                                      \
    CUresult status = expr;                                                 \
    if (ABSL_PREDICT_FALSE(status != CUDA_SUCCESS)) {                       \
      const char* errstr = "";                                              \
      cuGetErrorName(status, &errstr);                                      \
      LOG(ERROR) << "function " << #expr << "failed with error " << errstr; \
      return absl::UnknownError(absl::StrCat("CUDA driver error", errstr)); \
    }                                                                       \
  } while (false)

// Constructor provides all configuration needed to set up sampling on a
// single device
CuptiPmSamplerDevice::CuptiPmSamplerDevice(int device_id,
                                           CuptiInterface* cupti_interface,
                                           CuptiPmSamplerOptions* options)
    : device_id_(device_id), cupti_interface_(GetCuptiInterface()) {
  // Provide some defaults for metrics and handler
  if (options->metrics.size() == 0) {
    c_metrics_ = default_c_metrics_;
  } else {
    c_metrics_ = options->metrics;
  }

  num_metrics_ = c_metrics_.size();
  max_samples_ = options->max_samples;
  hw_buf_size_ = options->hw_buf_size;
  sample_interval_ns_ = options->sample_interval_ns;

  if (options->process_samples == nullptr) {
    options->process_samples = [](PmSamples* info) {
      LOG(WARNING) << "(Profiling::PM Sampling) No decode handler specified, "
                   << "discarding " << info->GetSamplerRanges().size()
                   << " samples";
      return;
    };
  }
}

// Destructor cleans up all images and objects
CuptiPmSamplerDevice::~CuptiPmSamplerDevice() {
  DestroyCounterAvailabilityImage();
  DestroyConfigImage();
  DestroyCounterDataImage();
  DestroyPmSamplerObject();
  DestroyProfilerHostObj();
}

// Fetch chip name for this device
absl::Status CuptiPmSamplerDevice::GetChipName() {
  DEF_SIZED_PRIV_STRUCT(CUpti_Device_GetChipName_Params, p);
  p.deviceIndex = device_id_;
  RETURN_IF_CUPTI_ERROR(DeviceGetChipName(&p));

  chipName_ = std::string(p.pChipName);

  return absl::OkStatus();
}

// Test for device support for PM sampling
absl::Status CuptiPmSamplerDevice::DeviceSupported() {
  CUdevice cuDevice;
  RETURN_IF_CUDA_DRIVER_ERROR(cuDeviceGet(&cuDevice, device_id_));

  // CUPTI call to validate configuration
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_DeviceSupported_Params, p);
  p.cuDevice = cuDevice;
  p.api = CUPTI_PROFILER_PM_SAMPLING;
  RETURN_IF_CUPTI_ERROR(ProfilerDeviceSupported(&p));

  if (p.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED) {
    return absl::FailedPreconditionError("Device does not support pm sampling");
  }

  return absl::OkStatus();
}

// Get counter availability image size, set the image to that size,
// then initialize it
absl::Status CuptiPmSamplerDevice::CreateCounterAvailabilityImage() {
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_GetCounterAvailability_Params, p);
  p.deviceIndex = device_id_;
  RETURN_IF_CUPTI_ERROR(PmSamplingGetCounterAvailability(&p));

  counter_availability_image_.clear();
  counter_availability_image_.resize(p.counterAvailabilityImageSize);

  p.pCounterAvailabilityImage = counter_availability_image_.data();
  RETURN_IF_CUPTI_ERROR(PmSamplingGetCounterAvailability(&p));

  return absl::OkStatus();
}

// Create profiler host object
absl::Status CuptiPmSamplerDevice::CreateProfilerHostObj() {
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_Initialize_Params, p);
  p.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
  p.pChipName = chipName_.c_str();
  p.pCounterAvailabilityImage = counter_availability_image_.data();
  RETURN_IF_CUPTI_ERROR(ProfilerHostInitialize(&p));

  host_obj_ = p.pHostObject;

  return absl::OkStatus();
}

// Register metrics, resize config image, and initialize it
absl::Status CuptiPmSamplerDevice::CreateConfigImage() {
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_ConfigAddMetrics_Params, pm);
  pm.pHostObject = host_obj_;
  pm.ppMetricNames = c_metrics_.data();
  pm.numMetrics = num_metrics_;
  RETURN_IF_CUPTI_ERROR(ProfilerHostConfigAddMetrics(&pm));

  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_GetConfigImageSize_Params, ps);
  ps.pHostObject = host_obj_;
  RETURN_IF_CUPTI_ERROR(ProfilerHostGetConfigImageSize(&ps));

  config_image_.clear();
  config_image_.resize(ps.configImageSize);

  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_GetConfigImage_Params, p);
  p.pHostObject = host_obj_;
  p.pConfigImage = config_image_.data();
  p.configImageSize = config_image_.size();
  RETURN_IF_CUPTI_ERROR(ProfilerHostGetConfigImage(&p));

  return absl::OkStatus();
}

// Return number of passes
size_t CuptiPmSamplerDevice::NumPasses() {
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_GetNumOfPasses_Params, p);
  p.pConfigImage = config_image_.data();
  p.configImageSize = config_image_.size();

  if (cupti_interface_->ProfilerHostGetNumOfPasses(&p) != CUPTI_SUCCESS)
    return 0;

  return p.numOfPasses;
}

// Initialize profiler APIs - required before PM sampler specific calls.
// No visible side effects.
absl::Status CuptiPmSamplerDevice::InitializeProfilerAPIs() {
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Initialize_Params, p);
  RETURN_IF_CUPTI_ERROR(ProfilerInitialize(&p));

  return absl::OkStatus();
}

// Create pm sampling object (initializes pm sampling APIs)
absl::Status CuptiPmSamplerDevice::CreatePmSamplerObject() {
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_Enable_Params, p);
  p.deviceIndex = device_id_;
  RETURN_IF_CUPTI_ERROR(PmSamplingEnable(&p));

  sampling_obj_ = p.pPmSamplingObject;

  return absl::OkStatus();
}

// Resize and initialize counter data image
absl::Status CuptiPmSamplerDevice::CreateCounterDataImage() {
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_GetCounterDataSize_Params, p);
  p.pPmSamplingObject = sampling_obj_;
  p.numMetrics = num_metrics_;
  p.pMetricNames = c_metrics_.data();
  p.maxSamples = max_samples_;
  RETURN_IF_CUPTI_ERROR(PmSamplingGetCounterDataSize(&p));

  counter_data_image_.resize(p.counterDataSize);

  return InitializeCounterDataImage();
}

// Sets several pm sampling configuration items
absl::Status CuptiPmSamplerDevice::SetConfig() {
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_SetConfig_Params, p);
  p.pPmSamplingObject = sampling_obj_;
  p.configSize = config_image_.size();
  p.pConfig = config_image_.data();
  p.hardwareBufferSize = hw_buf_size_;
  p.samplingInterval = sample_interval_ns_;
  p.triggerMode = CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL;
  RETURN_IF_CUPTI_ERROR(PmSamplingSetConfig(&p));

  return absl::OkStatus();
}

// Start recording pm sampling data
absl::Status CuptiPmSamplerDevice::StartSampling() {
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_Start_Params, p);
  p.pPmSamplingObject = sampling_obj_;
  RETURN_IF_CUPTI_ERROR(PmSamplingStart(&p));

  return absl::OkStatus();
}

// Stop recording pm sampling data
absl::Status CuptiPmSamplerDevice::StopSampling() {
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_Stop_Params, p);
  p.pPmSamplingObject = sampling_obj_;
  RETURN_IF_CUPTI_ERROR(PmSamplingStop(&p));

  return absl::OkStatus();
}

// Disable pm sampling and destroy the pm sampling object
absl::Status CuptiPmSamplerDevice::DisableSampling() {
  // Note: currently, disabling pm sampling object finalizes all of
  // CUPTI, so do not disable here
  // TODO: Add CUPTI version test and disable once ordering is changed
  //
  // DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_Disable_Params, p);
  // p.pPmSamplingObject = sampling_obj_;
  // RETURN_IF_CUPTI_ERROR(PmSamplingDisable(&p));

  sampling_obj_ = nullptr;

  return absl::OkStatus();
}

// Fetches data from hw buffer, fills in counter data image
absl::Status CuptiPmSamplerDevice::FillCounterDataImage(
    CuptiPmSamplerDecodeInfo& decode_info) {
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_DecodeData_Params, p);
  p.pPmSamplingObject = sampling_obj_;
  p.pCounterDataImage = counter_data_image_.data();
  p.counterDataImageSize = counter_data_image_.size();
  CUptiResult ret = cupti_interface_->PmSamplingDecodeData(&p);

  // If this is CUPTI_ERROR_OUT_OF_MEMORY, hardware buffer is full
  // and session needs to be restarted
  if (ret == CUPTI_ERROR_OUT_OF_MEMORY) {
    LOG(WARNING) << "Profiling::PM Sampling - hardware buffer overflow, must "
                 << "restart session.  Decrease sample rate or increase decode "
                 << "rate to avoid this.";
  }

  if (ret != CUPTI_SUCCESS) {
    return absl::InternalError("CUPTI error during cuptiPmSamplingDecodeData");
  }

  decode_info.decode_stop_reason = p.decodeStopReason;
  decode_info.overflow = p.overflow;

  return absl::OkStatus();
}

// Gets count of samples decoded into counter data image
absl::Status CuptiPmSamplerDevice::GetSampleCounts(
    CuptiPmSamplerDecodeInfo& decode_info) {
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_GetCounterDataInfo_Params, p);
  p.pCounterDataImage = counter_data_image_.data();
  p.counterDataImageSize = counter_data_image_.size();
  RETURN_IF_CUPTI_ERROR(PmSamplingGetCounterDataInfo(&p));

  decode_info.num_samples = p.numTotalSamples;
  decode_info.num_populated = p.numPopulatedSamples;
  decode_info.num_completed = p.numCompletedSamples;

  return absl::OkStatus();
}

// Fill in a single pm sampling record
absl::Status CuptiPmSamplerDevice::GetSample(SamplerRange& sample,
                                             size_t index) {
  // First, get the start and end times
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_CounterData_GetSampleInfo_Params, ps);
  ps.pPmSamplingObject = sampling_obj_;
  ps.pCounterDataImage = counter_data_image_.data();
  ps.counterDataImageSize = counter_data_image_.size();
  ps.sampleIndex = index;
  RETURN_IF_CUPTI_ERROR(PmSamplingCounterDataGetSampleInfo(&ps));

  sample.range_index = index;
  sample.start_timestamp_ns = ps.startTimestamp;
  sample.end_timestamp_ns = ps.endTimestamp;
  sample.metric_values.resize(num_metrics_);

  // Second, get the final metric values
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_EvaluateToGpuValues_Params, p);
  p.pHostObject = host_obj_;
  p.pCounterDataImage = counter_data_image_.data();
  p.counterDataImageSize = counter_data_image_.size();
  p.ppMetricNames = c_metrics_.data();
  p.numMetrics = num_metrics_;
  p.rangeIndex = index;
  p.pMetricValues = sample.metric_values.data();
  RETURN_IF_CUPTI_ERROR(ProfilerHostEvaluateToGpuValues(&p));

  return absl::OkStatus();
}

// Initializes image, then copies this to the backup counter data image
absl::Status CuptiPmSamplerDevice::InitializeCounterDataImage() {
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_CounterDataImage_Initialize_Params, p);
  p.pPmSamplingObject = sampling_obj_;
  p.counterDataSize = counter_data_image_.size();
  p.pCounterData = counter_data_image_.data();
  RETURN_IF_CUPTI_ERROR(PmSamplingCounterDataImageInitialize(&p));

  // Stash this in a vector so it can be restored with copy semantics
  counter_data_image_backup_ = std::vector<uint8_t>(counter_data_image_);

  return absl::OkStatus();
}

// Restores image from backup (faster than re-initializing)
absl::Status CuptiPmSamplerDevice::RestoreCounterDataImage() {
  // Will use copy semantics
  counter_data_image_ = counter_data_image_backup_;

  return absl::OkStatus();
}

void CuptiPmSamplerDevice::DestroyCounterAvailabilityImage() {
  counter_availability_image_.clear();
}

// Deinitialize and destroy the profiler host object
// Must be done after decode has stopped
void CuptiPmSamplerDevice::DestroyProfilerHostObj() {
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_Deinitialize_Params, p);
  p.pHostObject = host_obj_;
  cupti_interface_->ProfilerHostDeinitialize(&p);

  host_obj_ = nullptr;
}

void CuptiPmSamplerDevice::DestroyConfigImage() { config_image_.clear(); }

// Disable sampling and destroy the pm sampling object
// Must be done after decode has stopped
void CuptiPmSamplerDevice::DestroyPmSamplerObject() {
  DisableSampling().IgnoreError();
}

void CuptiPmSamplerDevice::DestroyCounterDataImage() {
  counter_data_image_.clear();
  counter_data_image_backup_.clear();
}

absl::Status CuptiPmSamplerDevice::CreateConfig() {
  // Get chip name string
  TF_RETURN_IF_ERROR(GetChipName());

  // Test whether the hardware supports pm sampling, if not, skip device
  TF_RETURN_IF_ERROR(DeviceSupported());

  // Create counter availability image or skip device
  TF_RETURN_IF_ERROR(CreateCounterAvailabilityImage());

  // Create profiler host object or skip device
  TF_RETURN_IF_ERROR(CreateProfilerHostObj());

  // Create config image or skip device
  TF_RETURN_IF_ERROR(CreateConfigImage());

  // Test for single pass configuration or skip
  if (NumPasses() > 1) {
    return absl::InvalidArgumentError(
        "Metrics configuration requires more than one pass");
  }

  // Create PM sampler object or skip device
  TF_RETURN_IF_ERROR(CreatePmSamplerObject());

  // Create counter data image or skip device
  TF_RETURN_IF_ERROR(CreateCounterDataImage());

  return absl::OkStatus();
}

// Constructor, creates worker thread
CuptiPmSamplerDecodeThread::CuptiPmSamplerDecodeThread(
    std::vector<std::shared_ptr<CuptiPmSamplerDevice>> devs,
    CuptiPmSamplerOptions* options) {
  c_metrics_ = options->metrics;
  devs_ = devs;
  num_metrics_ = c_metrics_.size();

  for (auto metric : c_metrics_) {
    metrics_.emplace_back(metric);
  }

  decode_period_ = options->decode_period;

  process_samples = options->process_samples;

  thd_ = new std::thread(ThdFunc, this);
}

void CuptiPmSamplerDecodeThread::ThdFuncDecodeUntilDisabled(
    CuptiPmSamplerDecodeThread* control) {
  // Signal that thread is enabled for decoding
  control->ThdIsEnabled();

  // Test for exit condition on all devices
  bool all_devs_end_of_records = false;
  int extra_attempts = 0;

  // When enabled, loop over each device, decoding
  // If an error is encountered, attempt to continue with next iteration
  // instead of exiting thread
  while ((!control->ShouldThdDisable()) || (!all_devs_end_of_records) ||
         (extra_attempts < 2)) {
    VLOG(2) << "(Profiling::PM Sampling) Top of decode loop";
    // Try a few extra times to decode
    if (all_devs_end_of_records == true) extra_attempts++;

    all_devs_end_of_records = true;
    absl::Time begin = absl::Now();

    size_t decoded_samples = 0;

    // Each decode period, decode all devices assigned to it
    for (auto dev : control->devs_) {
      VLOG(2) << "(Profiling::PM Sampling)  Beginning decode for device "
              << dev->device_id_;

      absl::Time start_time = absl::Now();
      absl::Time fill_time = start_time;
      absl::Time get_count_time = start_time;
      absl::Time get_samples_time = start_time;
      absl::Time process_samples_time = start_time;
      absl::Time initialize_image_time = start_time;

      CuptiPmSamplerDecodeInfo info{.metrics = control->c_metrics_};
      if (!dev->FillCounterDataImage(info).ok()) continue;
      fill_time = absl::Now();
      if (!dev->GetSampleCounts(info).ok()) continue;
      get_count_time = absl::Now();

      // Track whether this device reached end of records
      if (info.decode_stop_reason !=
          CUPTI_PM_SAMPLING_DECODE_STOP_REASON_END_OF_RECORDS) {
        all_devs_end_of_records = false;
      } else {
        VLOG(2) << "(Profiling::PM Sampling)   End of records for device "
                << dev->device_id_;
      }

      if (info.overflow) {
        LOG(WARNING) << "(Profiling::PM Sampling) hardware buffer overflow on "
                     << "device " << dev->device_id_
                     << ", sample data has been lost";
      }

      if (info.decode_stop_reason ==
          CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNTER_DATA_FULL) {
        LOG(WARNING) << "(Profiling::PM Sampling) ran out of host buffer space "
                     << "before decoding all records from the device buffer on "
                     << "device " << dev->device_id_;
      }

      if (info.num_completed == 0) {
        VLOG(3) << "(Profiling::PM Sampling)   FillCounterDataImage took "
                << (fill_time - start_time);
        VLOG(3) << "(Profiling::PM Sampling)   GetSampleCounts took "
                << (get_count_time - fill_time);
        continue;
      }

      decoded_samples += info.num_completed;

      info.sampler_ranges.resize(info.num_completed);

      // Set each sample's info, reset samples that error
      // (should not happen)
      for (size_t i = 0; i < info.num_completed; i++) {
        if (!dev->GetSample(info.sampler_ranges[i], i).ok()) {
          LOG(WARNING) << "(Profiling::PM Sampling) Error decoding pm sample";
          info.sampler_ranges[i].range_index = 0;
          info.sampler_ranges[i].start_timestamp_ns = 0;
          info.sampler_ranges[i].end_timestamp_ns = 0;
          info.sampler_ranges[i].metric_values.clear();
        } else {
          if (VLOG_IS_ON(4)) {
            for (int j = 0; j < control->num_metrics_; j++) {
              LOG(INFO) << "            " << info.metrics[j] << "[" << i
                        << "] = " << info.sampler_ranges[i].metric_values[j];
            }
          }
        }
      }

      get_samples_time = absl::Now();

      // info now contains a list of samples and metrics,
      // hand off to process or store elsewhere
      if (control->process_samples != nullptr) {
        PmSamples samples(control->metrics_, info.sampler_ranges);
        control->process_samples(&samples);
      }

      process_samples_time = absl::Now();

      if (!dev->RestoreCounterDataImage().ok()) {
        LOG(WARNING) << "(Profiling::PM Sampling) Error resetting counter data "
                     << "image";
      }

      initialize_image_time = absl::Now();

      VLOG(3) << "(Profiling::PM Sampling)   FillCounterDataImage took "
              << (fill_time - start_time);
      VLOG(3) << "(Profiling::PM Sampling)   GetSampleCounts took "
              << (get_count_time - fill_time);
      VLOG(3) << "(Profiling::PM Sampling)   vector resize & getSample for "
              << info.num_completed << " samples took "
              << (get_samples_time - get_count_time);
      VLOG(3) << "(Profiling::PM Sampling)   external processing of samples "
              << "took " << (process_samples_time - get_samples_time);
      VLOG(3) << "(Profiling::PM Sampling)   RestoreCounterDataImage took "
              << (initialize_image_time - process_samples_time);
    }

    // Sleep until start of next period,
    // warning if decode took longer than alloted time
    absl::Time end = absl::Now();
    absl::Duration elapsed = end - begin;
    if (elapsed < control->decode_period_) {
      VLOG(2) << "(Profiling::PM Sampling)   decoded " << decoded_samples
              << ", took " << elapsed << ", sleeping for "
              << (control->decode_period_ - elapsed);
      absl::SleepFor(control->decode_period_ - elapsed);
    } else {
      VLOG(2) << "(Profiling::PM Sampling)   decoded " << decoded_samples
              << ", took " << elapsed << ", decode period is "
              << control->decode_period_;
      LOG(WARNING) << "(Profiling::PM Sampling) decode thread took longer than "
                   << "configured period to complete a single decode pass.  "
                   << "When this happens, hardware buffer may overflow and "
                   << "lose sample data.  Reduce number of devices per decode "
                   << "thread, reduce the number of metrics gathered, reduce "
                   << "the sample rate, or ensure decode threads have "
                   << "sufficient cpu resources to maintain decode faster than "
                   << "metric sampling.  Elapsed time: " << elapsed << ", "
                   << "decode period: " << control->decode_period_;
    }
  }

  VLOG(2) << "(Profiling::PM Sampling) Exited decode loop";

  // Signal thread decoding is disabled
  control->ThdIsDisabled();
}

// Control lifecycle of decode thread
void CuptiPmSamplerDecodeThread::ThdFunc(CuptiPmSamplerDecodeThread* control) {
  // Space allowed for initialization here

  control->ThdIsInitialized();

  // Wait for signal to enable decoding on devices, or exit out
  do {
    if (control->ShouldThdEnable()) {
      ThdFuncDecodeUntilDisabled(control);
    } else if (control->ShouldThdExit()) {
      break;
    }

    absl::SleepFor(control->decode_period_);
  } while (true);

  // Only reaches this point if it should exit
  control->ThdIsExiting();
}

absl::Status CuptiPmSamplerImpl::Initialize(CuptiInterface* cupti_interface,
                                            size_t num_gpus,
                                            CuptiPmSamplerOptions* options) {
  // Ensure not already initialized
  if (initialized_) return absl::AlreadyExistsError("Already initialized");

  // Wait > 1 decode periods at stop to ensure all data is flushed
  decode_stop_delay_ = options->decode_period * 1.5;

  absl::Status status;

  // PM sampling has to be enabled on individual devices
  for (int dev_idx = 0; dev_idx < num_gpus; dev_idx++) {
    // Create a new PM sampling instance for this device
    std::shared_ptr<CuptiPmSamplerDevice> dev =
        std::make_shared<CuptiPmSamplerDevice>(dev_idx, cupti_interface,
                                               options);

    // FIXME: track error codes, tear down cleanly if needed,
    // return error codes so caller can handle whether the
    // code should continue or not

    // Create all configuration needed for this device, or skip device on error
    if (status = dev->CreateConfig(); !status.ok()) break;

    // Set configuration
    if (status = dev->SetConfig(); !status.ok()) break;

    // Device is fully configured but PM sampling not yet started - push to list
    // of PM sampling devices
    devices_.push_back(std::move(dev));
  }

  // If error occurred, clean up created devices and return failure
  if (!status.ok()) {
    devices_.clear();
    return status;
  }

  // OK to have no devices that support PM sampling as long as not due to error
  if (devices_.size() < 1) return absl::OkStatus();

  // Create decode thread(s)
  for (int i = 0; i < devices_.size(); i += options->devs_per_decode_thd) {
    // Slice iterators
    auto begin = devices_.begin() + i;
    auto end = begin + options->devs_per_decode_thd;
    // Don't go past end of vector
    end = std::min(end, devices_.end());

    // Slice for this decode thread
    std::vector<std::shared_ptr<CuptiPmSamplerDevice>> slice(begin, end);

    // Create worker thread for this slice
    auto thd = std::make_unique<CuptiPmSamplerDecodeThread>(slice, options);
    threads_.push_back(std::move(thd));
  }

  // Wait for signal that all threads are ready
  for (auto& thd : threads_) {
    while (!thd->IsThdInitialized()) {
    }
  }

  initialized_ = true;

  return absl::OkStatus();
}

absl::Status CuptiPmSamplerImpl::StartSampler() {
  if (enabled_) return absl::AlreadyExistsError("Already started");

  // Start sampling on all devices
  for (auto& dev : devices_) {
    if (!dev->StartSampling().ok()) {
      LOG(WARNING) << "Profiling::PM Sampling - failed to start on device "
                   << dev->device_id_;
      // TODO: What is appropriate behavior if start thread fails?
      // Most likely should delete the sampler for this device but this would
      // need to be communicated to the decoder thread.  Should be safe to do
      // nothing, as there will just be no data to decode
    }
  }

  // Signal threads should be enabled
  for (auto& thd : threads_) {
    thd->EnableThd();
  }

  // Wait for signal that decode thread is enabled
  for (auto& thd : threads_) {
    while (!thd->IsThdEnabled()) {
    }
  }

  enabled_ = true;

  return absl::OkStatus();
}

absl::Status CuptiPmSamplerImpl::StopSampler() {
  if (!enabled_) {
    return absl::FailedPreconditionError(
        "StopSampler called before StartSampler, or failure during "
        "StartSampler");
  }

  // Stop sampling on all devices
  for (auto& dev : devices_) {
    if (!dev->StopSampling().ok()) {
      LOG(WARNING) << "Profiling::PM Sampling - failed to stop on device "
                   << dev->device_id_;
    }
  }

  // Ensure there is at least one more decode pass
  // TODO: This could be moved into the decode threads so the main thread
  // does not have to wait for the decode period
  absl::SleepFor(decode_stop_delay_);

  // Signal threads should be disabled
  for (auto& thd : threads_) {
    thd->DisableThd();
  }

  enabled_ = false;

  return absl::OkStatus();
}

absl::Status CuptiPmSamplerImpl::Deinitialize() {
  if (enabled_) {
    StopSampler().IgnoreError();
  }
  if (!initialized_) {
    return absl::FailedPreconditionError(
        "Deinitialize called before Initialize, or failure during Initialize");
  }

  // Wait for signal that decode thread is disabled
  for (const auto& thd : threads_) {
    while (!thd->IsThdDisabled()) {
    }
  }

  // Tell threads to exit
  for (auto& thd : threads_) {
    thd->ExitThd();
  }

  // Threads will soon exit, ready to join
  for (auto& thd : threads_) {
    while (!thd->IsThdExiting()) {
    }

    // Destroy decode thread (joins thread)
    thd.reset();
  }

  // Destroy all decode threads
  threads_.clear();

  // Destroy all devices
  devices_.clear();

  initialized_ = false;

  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace xla
