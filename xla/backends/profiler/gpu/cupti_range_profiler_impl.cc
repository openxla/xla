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

#include "xla/backends/profiler/gpu/cupti_range_profiler_impl.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_profiler_target.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_range_profiler.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_target.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"
#include "xla/backends/profiler/gpu/cupti_range_profiler.h"
#include "xla/backends/profiler/gpu/cupti_status.h"
#include "xla/backends/profiler/gpu/cupti_utils.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace profiler {

// CUPTI params struct definitions are very long, macro it for convenience.
// They all have a struct_size field which must be set to type_STRUCT_SIZE
// and a pPriv field which must be null.
#define DEF_SIZED_PRIV_STRUCT(type, name) \
  type name = {.structSize = type##_STRUCT_SIZE, .pPriv = nullptr}

// =============================================================================
// CuptiRangeProfilerDevice
// =============================================================================

CuptiRangeProfilerDevice::CuptiRangeProfilerDevice(
    int device_id, const CuptiRangeProfilerOptions& options)
    : device_id_(device_id),
      cupti_interface_(GetCuptiInterface()),
      config_metrics_(options.metrics),
      enabled_metrics_(options.metrics) {
  for (const auto& metric : config_metrics_) {
    c_metrics_.push_back(metric.c_str());
  }
}

CuptiRangeProfilerDevice::~CuptiRangeProfilerDevice() {
  if (range_profiler_obj_ != nullptr) {
    Disable().IgnoreError();
  }
  DestroyProfilerHostObj();
}

absl::Status CuptiRangeProfilerDevice::GetChipName() {
  DEF_SIZED_PRIV_STRUCT(CUpti_Device_GetChipName_Params, p);
  p.deviceIndex = device_id_;
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->DeviceGetChipName(&p)));
  chip_name_ = p.pChipName;
  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerDevice::DeviceSupported() {
  CUdevice cu_device;
  TF_RETURN_IF_ERROR(
      stream_executor::cuda::ToStatus(cuDeviceGet(&cu_device, device_id_)));

  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_DeviceSupported_Params, p);
  p.cuDevice = cu_device;
  p.api = CUPTI_PROFILER_RANGE_PROFILING;
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->ProfilerDeviceSupported(&p)));

  if (p.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Device ", device_id_, " does not support range profiling"));
  }
  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerDevice::CreateCounterAvailabilityImage() {
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_GetCounterAvailability_Params, p);
  p.ctx = nullptr;  // Use current context.
  TF_RETURN_IF_ERROR(
      ToStatus(cupti_interface_->ProfilerGetCounterAvailability(&p)));

  counter_availability_image_.clear();
  counter_availability_image_.resize(p.counterAvailabilityImageSize);
  p.pCounterAvailabilityImage = counter_availability_image_.data();
  TF_RETURN_IF_ERROR(
      ToStatus(cupti_interface_->ProfilerGetCounterAvailability(&p)));

  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerDevice::CreateProfilerHostObj() {
  DestroyProfilerHostObj();

  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_Initialize_Params, p);
  p.profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
  p.pChipName = chip_name_.c_str();
  p.pCounterAvailabilityImage = counter_availability_image_.data();
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->ProfilerHostInitialize(&p)));

  host_obj_ = p.pHostObject;
  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerDevice::CreateConfigImage() {
  TF_RETURN_IF_ERROR(CreateProfilerHostObj());

  // Add metrics to host object.
  LOG(INFO) << "(Profiling::Range) Adding " << c_metrics_.size()
            << " metrics to config image";
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_ConfigAddMetrics_Params, am);
  am.pHostObject = host_obj_;
  am.ppMetricNames = c_metrics_.data();
  am.numMetrics = c_metrics_.size();
  const CUptiResult status = cupti_interface_->ProfilerHostConfigAddMetrics(&am);

  if (status == CUPTI_ERROR_INVALID_PARAMETER ||
      status == CUPTI_ERROR_INVALID_METRIC_NAME) {
    // Try each metric individually to find valid ones.
    LOG(WARNING) << "(Profiling::Range) Bulk add returned error " << status
                 << ", testing " << c_metrics_.size()
                 << " metrics individually";
    std::vector<const char*> valid_metrics;
    for (const auto& metric : c_metrics_) {
      TF_RETURN_IF_ERROR(CreateProfilerHostObj());
      DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_ConfigAddMetrics_Params, test);
      test.pHostObject = host_obj_;
      const char* m = metric;
      test.ppMetricNames = &m;
      test.numMetrics = 1;
      const CUptiResult r = cupti_interface_->ProfilerHostConfigAddMetrics(&test);
      if (r == CUPTI_SUCCESS) {
        valid_metrics.push_back(metric);
      } else {
        LOG(WARNING) << "(Profiling::Range) Invalid metric: " << metric;
      }
    }
    if (valid_metrics.empty()) {
      return absl::FailedPreconditionError(
          "No valid metrics for range profiling");
    }
    LOG(INFO) << "(Profiling::Range) Pruned to " << valid_metrics.size()
              << " valid metrics (from " << c_metrics_.size() << ")";
    c_metrics_ = valid_metrics;

    // Rebuild host object with valid metrics.
    TF_RETURN_IF_ERROR(CreateProfilerHostObj());
    DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_ConfigAddMetrics_Params, retry);
    retry.pHostObject = host_obj_;
    retry.ppMetricNames = c_metrics_.data();
    retry.numMetrics = c_metrics_.size();
    TF_RETURN_IF_ERROR(
        ToStatus(cupti_interface_->ProfilerHostConfigAddMetrics(&retry)));
  } else if (status != CUPTI_SUCCESS) {
    return ToStatus(status);
  } else {
    LOG(INFO) << "(Profiling::Range) All " << c_metrics_.size()
              << " metrics accepted by CUPTI";
  }

  // Get config image size and create it.
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_GetConfigImageSize_Params, cs);
  cs.pHostObject = host_obj_;
  TF_RETURN_IF_ERROR(
      ToStatus(cupti_interface_->ProfilerHostGetConfigImageSize(&cs)));

  config_image_.clear();
  config_image_.resize(cs.configImageSize);

  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_GetConfigImage_Params, ci);
  ci.pHostObject = host_obj_;
  ci.pConfigImage = config_image_.data();
  ci.configImageSize = config_image_.size();
  TF_RETURN_IF_ERROR(
      ToStatus(cupti_interface_->ProfilerHostGetConfigImage(&ci)));

  // Update enabled_metrics_ to reflect what was actually configured.
  enabled_metrics_.clear();
  for (const auto& metric : c_metrics_) {
    enabled_metrics_.emplace_back(metric);
  }

  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerDevice::GetNumPasses(int* num_passes) {
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_GetNumOfPasses_Params, p);
  p.pConfigImage = config_image_.data();
  p.configImageSize = config_image_.size();
  TF_RETURN_IF_ERROR(
      ToStatus(cupti_interface_->ProfilerHostGetNumOfPasses(&p)));
  *num_passes = static_cast<int>(p.numOfPasses);
  LOG(INFO) << "(Profiling::Range) Device " << device_id_
            << ": CUPTI reports " << *num_passes << " pass(es) for "
            << c_metrics_.size() << " metrics (config image "
            << config_image_.size() << " bytes)";
  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerDevice::CreateConfig() {
  TF_RETURN_IF_ERROR(GetChipName());
  TF_RETURN_IF_ERROR(DeviceSupported());
  TF_RETURN_IF_ERROR(CreateCounterAvailabilityImage());
  TF_RETURN_IF_ERROR(CreateConfigImage());
  return absl::OkStatus();
}

// --- New Range Profiling API methods ---

absl::Status CuptiRangeProfilerDevice::Enable() {
  // Initialize CUPTI profiler subsystem (required before any range profiling).
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Initialize_Params, init_p);
  TF_RETURN_IF_ERROR(
      ToStatus(cupti_interface_->ProfilerInitialize(&init_p)));

  DEF_SIZED_PRIV_STRUCT(CUpti_RangeProfiler_Enable_Params, p);
  p.ctx = nullptr;  // Use current CUDA context.
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->RangeProfilerEnable(&p)));
  range_profiler_obj_ = p.pRangeProfilerObject;
  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerDevice::CreateCounterDataImage() {
  if (range_profiler_obj_ == nullptr) {
    return absl::FailedPreconditionError(
        "Range profiler not enabled; call Enable() first");
  }

  DEF_SIZED_PRIV_STRUCT(CUpti_RangeProfiler_GetCounterDataSize_Params, sz);
  sz.pRangeProfilerObject = range_profiler_obj_;
  sz.pMetricNames = c_metrics_.data();
  sz.numMetrics = c_metrics_.size();
  sz.maxNumOfRanges = 1;        // One range per pass in user mode.
  sz.maxNumRangeTreeNodes = 1;
  TF_RETURN_IF_ERROR(ToStatus(
      cupti_interface_->RangeProfilerGetCounterDataSize(&sz)));

  counter_data_image_.clear();
  counter_data_image_.resize(sz.counterDataSize, 0);

  DEF_SIZED_PRIV_STRUCT(
      CUpti_RangeProfiler_CounterDataImage_Initialize_Params, di);
  di.pRangeProfilerObject = range_profiler_obj_;
  di.pCounterData = counter_data_image_.data();
  di.counterDataSize = counter_data_image_.size();
  TF_RETURN_IF_ERROR(
      ToStatus(cupti_interface_->RangeProfilerCounterDataImageInitialize(&di)));

  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerDevice::SetConfig() {
  if (range_profiler_obj_ == nullptr) {
    return absl::FailedPreconditionError(
        "Range profiler not enabled; call Enable() first");
  }

  DEF_SIZED_PRIV_STRUCT(CUpti_RangeProfiler_SetConfig_Params, p);
  p.pRangeProfilerObject = range_profiler_obj_;
  p.pConfig = config_image_.data();
  p.configSize = config_image_.size();
  p.pCounterDataImage = counter_data_image_.data();
  p.counterDataImageSize = counter_data_image_.size();
  p.range = CUPTI_UserRange;
  p.replayMode = CUPTI_ApplicationReplay;
  p.maxRangesPerPass = 1;
  p.numNestingLevels = 1;
  p.minNestingLevel = 1;
  p.passIndex = 0;
  p.targetNestingLevel = 1;
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->RangeProfilerSetConfig(&p)));

  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerDevice::Start() {
  if (range_profiler_obj_ == nullptr) {
    return absl::FailedPreconditionError("Range profiler not enabled");
  }

  DEF_SIZED_PRIV_STRUCT(CUpti_RangeProfiler_Start_Params, p);
  p.pRangeProfilerObject = range_profiler_obj_;
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->RangeProfilerStart(&p)));
  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerDevice::Stop(bool* all_passes_collected) {
  if (range_profiler_obj_ == nullptr) {
    return absl::FailedPreconditionError("Range profiler not enabled");
  }

  DEF_SIZED_PRIV_STRUCT(CUpti_RangeProfiler_Stop_Params, p);
  p.pRangeProfilerObject = range_profiler_obj_;
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->RangeProfilerStop(&p)));
  *all_passes_collected = (p.isAllPassSubmitted != 0);
  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerDevice::PushRange(absl::string_view name) {
  if (range_profiler_obj_ == nullptr) {
    return absl::FailedPreconditionError("Range profiler not enabled");
  }

  DEF_SIZED_PRIV_STRUCT(CUpti_RangeProfiler_PushRange_Params, p);
  p.pRangeProfilerObject = range_profiler_obj_;
  std::string name_str(name);
  p.pRangeName = name_str.c_str();
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->RangeProfilerPushRange(&p)));
  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerDevice::PopRange() {
  if (range_profiler_obj_ == nullptr) {
    return absl::FailedPreconditionError("Range profiler not enabled");
  }

  DEF_SIZED_PRIV_STRUCT(CUpti_RangeProfiler_PopRange_Params, p);
  p.pRangeProfilerObject = range_profiler_obj_;
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->RangeProfilerPopRange(&p)));
  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerDevice::DecodeData() {
  if (range_profiler_obj_ == nullptr) {
    return absl::FailedPreconditionError("Range profiler not enabled");
  }

  DEF_SIZED_PRIV_STRUCT(CUpti_RangeProfiler_DecodeData_Params, p);
  p.pRangeProfilerObject = range_profiler_obj_;
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->RangeProfilerDecodeData(&p)));

  if (p.numOfRangeDropped > 0) {
    LOG(WARNING) << "(Profiling::Range) Device " << device_id_ << ": "
                 << p.numOfRangeDropped << " range(s) dropped during decode";
  }

  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerDevice::Decode(
    std::vector<RangeResult>* results) {
  // Get total number of ranges in the counter data image.
  DEF_SIZED_PRIV_STRUCT(CUpti_RangeProfiler_GetCounterDataInfo_Params, info);
  info.pCounterDataImage = counter_data_image_.data();
  info.counterDataImageSize = counter_data_image_.size();
  TF_RETURN_IF_ERROR(ToStatus(
      cupti_interface_->RangeProfilerGetCounterDataInfo(&info)));

  int num_ranges = info.numTotalRanges;
  results->resize(num_ranges);

  for (int range_idx = 0; range_idx < num_ranges; ++range_idx) {
    RangeResult& result = (*results)[range_idx];
    // Get range name.
    DEF_SIZED_PRIV_STRUCT(
        CUpti_RangeProfiler_CounterData_GetRangeInfo_Params, ri);
    ri.pCounterDataImage = counter_data_image_.data();
    ri.counterDataImageSize = counter_data_image_.size();
    ri.rangeIndex = range_idx;
    ri.rangeDelimiter = "/";
    TF_RETURN_IF_ERROR(
        ToStatus(cupti_interface_->RangeProfilerCounterDataGetRangeInfo(&ri)));
    result.range_name = ri.rangeName;

    // Evaluate metrics for this range.
    result.metric_values.resize(c_metrics_.size());
    DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_EvaluateToGpuValues_Params, ev);
    ev.pHostObject = host_obj_;
    ev.pCounterDataImage = counter_data_image_.data();
    ev.counterDataImageSize = counter_data_image_.size();
    ev.ppMetricNames = c_metrics_.data();
    ev.numMetrics = c_metrics_.size();
    ev.rangeIndex = range_idx;
    ev.pMetricValues = result.metric_values.data();
    TF_RETURN_IF_ERROR(
        ToStatus(cupti_interface_->ProfilerHostEvaluateToGpuValues(&ev)));
  }

  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerDevice::Disable() {
  if (range_profiler_obj_ == nullptr) {
    return absl::OkStatus();
  }

  DEF_SIZED_PRIV_STRUCT(CUpti_RangeProfiler_Disable_Params, p);
  p.pRangeProfilerObject = range_profiler_obj_;
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->RangeProfilerDisable(&p)));
  range_profiler_obj_ = nullptr;
  return absl::OkStatus();
}

void CuptiRangeProfilerDevice::DestroyProfilerHostObj() {
  if (host_obj_ == nullptr) return;
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_Deinitialize_Params, p);
  p.pHostObject = host_obj_;
  cupti_interface_->ProfilerHostDeinitialize(&p);
  host_obj_ = nullptr;
}

// =============================================================================
// CuptiRangeProfilerImpl
// =============================================================================

absl::Status CuptiRangeProfilerImpl::Initialize(
    int num_gpus, const CuptiRangeProfilerOptions& options) {
  if (initialized_) {
    return absl::AlreadyExistsError("Range profiler already initialized");
  }

  options_ = options;

  absl::Cleanup cleanup([this]() { devices_.clear(); });

  for (int dev_idx = 0; dev_idx < num_gpus; ++dev_idx) {
    auto dev = std::make_unique<CuptiRangeProfilerDevice>(dev_idx, options);

    // Host-side configuration (chip name, counter availability, config image).
    TF_RETURN_IF_ERROR(dev->CreateConfig());

    int passes = 0;
    TF_RETURN_IF_ERROR(dev->GetNumPasses(&passes));

    if (dev_idx == 0) {
      num_passes_ = passes;
    } else if (passes != num_passes_) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Device ", dev_idx, " requires ", passes,
          " passes but device 0 requires ", num_passes_,
          "; all devices must agree"));
    }

    // Target-side: enable range profiler, create counter data, set config.
    TF_RETURN_IF_ERROR(dev->Enable());
    TF_RETURN_IF_ERROR(dev->CreateCounterDataImage());
    TF_RETURN_IF_ERROR(dev->SetConfig());

    devices_.push_back(std::move(dev));
  }

  LOG(INFO) << "(Profiling::Range) Initialized for " << num_gpus
            << " device(s), " << num_passes_ << " pass(es) required";

  std::move(cleanup).Cancel();
  initialized_ = true;
  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerImpl::BeginPass() {
  if (!initialized_) {
    return absl::FailedPreconditionError("Range profiler not initialized");
  }
  for (auto& dev : devices_) {
    TF_RETURN_IF_ERROR(dev->Start());
  }
  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerImpl::EndPass() {
  if (!initialized_) {
    return absl::FailedPreconditionError("Range profiler not initialized");
  }
  for (auto& dev : devices_) {
    bool all_collected = false;
    TF_RETURN_IF_ERROR(dev->Stop(&all_collected));
  }
  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerImpl::PushRange(absl::string_view name) {
  for (auto& dev : devices_) {
    TF_RETURN_IF_ERROR(dev->PushRange(name));
  }
  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerImpl::PopRange() {
  for (auto& dev : devices_) {
    TF_RETURN_IF_ERROR(dev->PopRange());
  }
  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerImpl::FlushAndDecode() {
  if (!initialized_) {
    return absl::FailedPreconditionError("Range profiler not initialized");
  }

  for (auto& dev : devices_) {
    TF_RETURN_IF_ERROR(dev->DecodeData());

    std::vector<RangeResult> results;
    TF_RETURN_IF_ERROR(dev->Decode(&results));

    if (options_.process_results) {
      RangeProfilerResults profiler_results(
          dev->enabled_metrics(), std::move(results), dev->device_id());
      options_.process_results(&profiler_results);
    }
  }
  return absl::OkStatus();
}

absl::Status CuptiRangeProfilerImpl::Deinitialize() {
  if (!initialized_) {
    return absl::FailedPreconditionError("Range profiler not initialized");
  }

  for (auto& dev : devices_) {
    dev->Disable().IgnoreError();
  }

  devices_.clear();
  initialized_ = false;

  LOG(INFO) << "(Profiling::Range) Deinitialized";
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<CuptiRangeProfilerImpl>>
CuptiRangeProfilerImpl::Create(int num_gpus,
                                const CuptiRangeProfilerOptions& options) {
  std::unique_ptr<CuptiRangeProfilerImpl> profiler(
      new CuptiRangeProfilerImpl());

  if (num_gpus < 1) {
    return profiler;
  }

  TF_RETURN_IF_ERROR(profiler->Initialize(num_gpus, options));
  return profiler;
}

#undef DEF_SIZED_PRIV_STRUCT

}  // namespace profiler
}  // namespace xla
