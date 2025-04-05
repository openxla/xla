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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_WRAPPER_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_WRAPPER_H_

#include <stddef.h>
#include <stdint.h>

#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"

namespace xla {
namespace profiler {

class CuptiWrapper : public xla::profiler::CuptiInterface {
 public:
  CuptiWrapper() {}

  ~CuptiWrapper() override {}

  // CUPTI activity API
  CUptiResult ActivityDisable(CUpti_ActivityKind kind) override;

  CUptiResult ActivityEnable(CUpti_ActivityKind kind) override;

  CUptiResult ActivityFlushAll(uint32_t flag) override;

  CUptiResult ActivityGetNextRecord(uint8_t* buffer,
                                    size_t valid_buffer_size_bytes,
                                    CUpti_Activity** record) override;

  CUptiResult ActivityGetNumDroppedRecords(CUcontext context,
                                           uint32_t stream_id,
                                           size_t* dropped) override;

  CUptiResult ActivityConfigureUnifiedMemoryCounter(
      CUpti_ActivityUnifiedMemoryCounterConfig* config,
      uint32_t count) override;

  CUptiResult ActivityRegisterCallbacks(
      CUpti_BuffersCallbackRequestFunc func_buffer_requested,
      CUpti_BuffersCallbackCompleteFunc func_buffer_completed) override;

  CUptiResult ActivityUsePerThreadBuffer() override;

  CUptiResult SetActivityFlushPeriod(uint32_t period_ms) override;

  CUptiResult GetDeviceId(CUcontext context, uint32_t* deviceId) override;

  CUptiResult GetTimestamp(uint64_t* timestamp) override;

  // cuptiFinalize is only defined in CUDA8 and above.
  // To enable it in CUDA8, the environment variable CUPTI_ENABLE_FINALIZE must
  // be set to 1.
  CUptiResult Finalize() override;

  // CUPTI callback API
  CUptiResult EnableCallback(uint32_t enable, CUpti_SubscriberHandle subscriber,
                             CUpti_CallbackDomain domain,
                             CUpti_CallbackId cbid) override;

  CUptiResult EnableDomain(uint32_t enable, CUpti_SubscriberHandle subscriber,
                           CUpti_CallbackDomain domain) override;

  CUptiResult Subscribe(CUpti_SubscriberHandle* subscriber,
                        CUpti_CallbackFunc callback, void* userdata) override;

  CUptiResult Unsubscribe(CUpti_SubscriberHandle subscriber) override;

  CUptiResult GetResultString(CUptiResult result, const char** str) override;

  CUptiResult GetContextId(CUcontext context, uint32_t* context_id) override;

  CUptiResult GetStreamIdEx(CUcontext context, CUstream stream,
                            uint8_t per_thread_stream,
                            uint32_t* stream_id) override;

  CUptiResult GetGraphId(CUgraph graph, uint32_t* graph_id) override;

  CUptiResult GetGraphExecId(CUgraphExec graph_exec,
                             uint32_t* graph_id) override;

  CUptiResult SetThreadIdType(CUpti_ActivityThreadIdType type) override;

  // Profiler Host APIs
  CUptiResult ProfilerHostInitialize(
      CUpti_Profiler_Host_Initialize_Params* pParams) override;
  CUptiResult ProfilerHostDeinitialize(
      CUpti_Profiler_Host_Deinitialize_Params* pParams) override;
  CUptiResult ProfilerHostGetSupportedChips(
      CUpti_Profiler_Host_GetSupportedChips_Params* pParams) override;
  CUptiResult ProfilerHostGetBaseMetrics(
      CUpti_Profiler_Host_GetBaseMetrics_Params* pParams) override;
  CUptiResult ProfilerHostGetSubMetrics(
      CUpti_Profiler_Host_GetSubMetrics_Params* pParams) override;
  CUptiResult ProfilerHostGetMetricProperties(
      CUpti_Profiler_Host_GetMetricProperties_Params* pParams) override;
  CUptiResult ProfilerHostGetRangeName(
      CUpti_Profiler_Host_GetRangeName_Params* pParams) override;
  CUptiResult ProfilerHostEvaluateToGpuValues(
      CUpti_Profiler_Host_EvaluateToGpuValues_Params* pParams) override;
  CUptiResult ProfilerHostConfigAddMetrics(
      CUpti_Profiler_Host_ConfigAddMetrics_Params* pParams) override;
  CUptiResult ProfilerHostGetConfigImageSize(
      CUpti_Profiler_Host_GetConfigImageSize_Params* pParams) override;
  CUptiResult ProfilerHostGetConfigImage(
      CUpti_Profiler_Host_GetConfigImage_Params* pParams) override;
  CUptiResult ProfilerHostGetNumOfPasses(
      CUpti_Profiler_Host_GetNumOfPasses_Params* pParams) override;
  CUptiResult ProfilerHostGetMaxNumHardwareMetricsPerPass(
      CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params* pParams)
      override;

  // Profiler Target APIs
  CUptiResult ProfilerInitialize(
      CUpti_Profiler_Initialize_Params* pParams) override;
  CUptiResult ProfilerDeInitialize(
      CUpti_Profiler_DeInitialize_Params* pParams) override;
  CUptiResult ProfilerCounterDataImageCalculateSize(
      CUpti_Profiler_CounterDataImage_CalculateSize_Params* pParams) override;
  CUptiResult ProfilerCounterDataImageInitialize(
      CUpti_Profiler_CounterDataImage_Initialize_Params* pParams) override;
  CUptiResult ProfilerCounterDataImageCalculateScratchBufferSize(
      CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params*
          pParams) override;
  CUptiResult ProfilerCounterDataImageInitializeScratchBuffer(
      CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams)
      override;
  CUptiResult ProfilerBeginSession(
      CUpti_Profiler_BeginSession_Params* pParams) override;
  CUptiResult ProfilerEndSession(
      CUpti_Profiler_EndSession_Params* pParams) override;
  CUptiResult ProfilerSetConfig(
      CUpti_Profiler_SetConfig_Params* pParams) override;
  CUptiResult ProfilerUnsetConfig(
      CUpti_Profiler_UnsetConfig_Params* pParams) override;
  CUptiResult ProfilerBeginPass(
      CUpti_Profiler_BeginPass_Params* pParams) override;
  CUptiResult ProfilerEndPass(CUpti_Profiler_EndPass_Params* pParams) override;
  CUptiResult ProfilerEnableProfiling(
      CUpti_Profiler_EnableProfiling_Params* pParams) override;
  CUptiResult ProfilerDisableProfiling(
      CUpti_Profiler_DisableProfiling_Params* pParams) override;
  CUptiResult ProfilerIsPassCollected(
      CUpti_Profiler_IsPassCollected_Params* pParams) override;
  CUptiResult ProfilerFlushCounterData(
      CUpti_Profiler_FlushCounterData_Params* pParams) override;
  CUptiResult ProfilerPushRange(
      CUpti_Profiler_PushRange_Params* pParams) override;
  CUptiResult ProfilerPopRange(
      CUpti_Profiler_PopRange_Params* pParams) override;
  CUptiResult ProfilerGetCounterAvailability(
      CUpti_Profiler_GetCounterAvailability_Params* pParams) override;
  CUptiResult ProfilerDeviceSupported(
      CUpti_Profiler_DeviceSupported_Params* pParams) override;

  // PM sampling specific functions
  CUptiResult PmSamplingSetConfig(
      CUpti_PmSampling_SetConfig_Params* pParams) override;
  CUptiResult PmSamplingEnable(
      CUpti_PmSampling_Enable_Params* pParams) override;
  CUptiResult PmSamplingDisable(
      CUpti_PmSampling_Disable_Params* pParams) override;
  CUptiResult PmSamplingStart(CUpti_PmSampling_Start_Params* pParams) override;
  CUptiResult PmSamplingStop(CUpti_PmSampling_Stop_Params* pParams) override;
  CUptiResult PmSamplingDecodeData(
      CUpti_PmSampling_DecodeData_Params* pParams) override;
  CUptiResult PmSamplingGetCounterAvailability(
      CUpti_PmSampling_GetCounterAvailability_Params* pParams) override;
  CUptiResult PmSamplingGetCounterDataSize(
      CUpti_PmSampling_GetCounterDataSize_Params* pParams) override;
  CUptiResult PmSamplingCounterDataImageInitialize(
      CUpti_PmSampling_CounterDataImage_Initialize_Params* pParams) override;
  CUptiResult PmSamplingGetCounterDataInfo(
      CUpti_PmSampling_GetCounterDataInfo_Params* pParams) override;
  CUptiResult PmSamplingCounterDataGetSampleInfo(
      CUpti_PmSampling_CounterData_GetSampleInfo_Params* pParams) override;

  CUptiResult DeviceGetChipName(
      CUpti_Device_GetChipName_Params* pParams) override;

  void CleanUp() override {}
  bool Disabled() const override { return false; }

 private:
  CuptiWrapper(const CuptiWrapper&) = delete;
  void operator=(const CuptiWrapper&) = delete;
};

// This is an implementation of CuptiWrapper that implements all load bearing
// APIs as no-op. This is a stub that keeps XLA profiler functional, but all
// collected profiles will be empty.
class CuptiWrapperStub : public xla::profiler::CuptiInterface {
 public:
  CuptiWrapperStub() {}

  ~CuptiWrapperStub() override {}

  // CUPTI activity API
  CUptiResult ActivityDisable(CUpti_ActivityKind kind) override;

  CUptiResult ActivityEnable(CUpti_ActivityKind kind) override;

  CUptiResult ActivityFlushAll(uint32_t flag) override;

  CUptiResult ActivityGetNextRecord(uint8_t* buffer,
                                    size_t valid_buffer_size_bytes,
                                    CUpti_Activity** record) override;

  CUptiResult ActivityGetNumDroppedRecords(CUcontext context,
                                           uint32_t stream_id,
                                           size_t* dropped) override;

  CUptiResult ActivityConfigureUnifiedMemoryCounter(
      CUpti_ActivityUnifiedMemoryCounterConfig* config,
      uint32_t count) override;

  CUptiResult ActivityRegisterCallbacks(
      CUpti_BuffersCallbackRequestFunc func_buffer_requested,
      CUpti_BuffersCallbackCompleteFunc func_buffer_completed) override;

  CUptiResult ActivityUsePerThreadBuffer() override;

  CUptiResult SetActivityFlushPeriod(uint32_t period_ms) override;

  CUptiResult GetDeviceId(CUcontext context, uint32_t* deviceId) override;

  CUptiResult GetTimestamp(uint64_t* timestamp) override;

  // cuptiFinalize is only defined in CUDA8 and above.
  // To enable it in CUDA8, the environment variable CUPTI_ENABLE_FINALIZE must
  // be set to 1.
  CUptiResult Finalize() override;

  // CUPTI callback API
  CUptiResult EnableCallback(uint32_t enable, CUpti_SubscriberHandle subscriber,
                             CUpti_CallbackDomain domain,
                             CUpti_CallbackId cbid) override;

  CUptiResult EnableDomain(uint32_t enable, CUpti_SubscriberHandle subscriber,
                           CUpti_CallbackDomain domain) override;

  CUptiResult Subscribe(CUpti_SubscriberHandle* subscriber,
                        CUpti_CallbackFunc callback, void* userdata) override;

  CUptiResult Unsubscribe(CUpti_SubscriberHandle subscriber) override;

  CUptiResult GetResultString(CUptiResult result, const char** str) override;

  CUptiResult GetContextId(CUcontext context, uint32_t* context_id) override;

  CUptiResult GetStreamIdEx(CUcontext context, CUstream stream,
                            uint8_t per_thread_stream,
                            uint32_t* stream_id) override;

  CUptiResult GetGraphId(CUgraph graph, uint32_t* graph_id) override;

  CUptiResult GetGraphExecId(CUgraphExec graph_exec,
                             uint32_t* graph_id) override;

  CUptiResult SetThreadIdType(CUpti_ActivityThreadIdType type) override;

  // Profiler Host APIs
  CUptiResult ProfilerHostInitialize(
      CUpti_Profiler_Host_Initialize_Params* pParams) override;
  CUptiResult ProfilerHostDeinitialize(
      CUpti_Profiler_Host_Deinitialize_Params* pParams) override;
  CUptiResult ProfilerHostGetSupportedChips(
      CUpti_Profiler_Host_GetSupportedChips_Params* pParams) override;
  CUptiResult ProfilerHostGetBaseMetrics(
      CUpti_Profiler_Host_GetBaseMetrics_Params* pParams) override;
  CUptiResult ProfilerHostGetSubMetrics(
      CUpti_Profiler_Host_GetSubMetrics_Params* pParams) override;
  CUptiResult ProfilerHostGetMetricProperties(
      CUpti_Profiler_Host_GetMetricProperties_Params* pParams) override;
  CUptiResult ProfilerHostGetRangeName(
      CUpti_Profiler_Host_GetRangeName_Params* pParams) override;
  CUptiResult ProfilerHostEvaluateToGpuValues(
      CUpti_Profiler_Host_EvaluateToGpuValues_Params* pParams) override;
  CUptiResult ProfilerHostConfigAddMetrics(
      CUpti_Profiler_Host_ConfigAddMetrics_Params* pParams) override;
  CUptiResult ProfilerHostGetConfigImageSize(
      CUpti_Profiler_Host_GetConfigImageSize_Params* pParams) override;
  CUptiResult ProfilerHostGetConfigImage(
      CUpti_Profiler_Host_GetConfigImage_Params* pParams) override;
  CUptiResult ProfilerHostGetNumOfPasses(
      CUpti_Profiler_Host_GetNumOfPasses_Params* pParams) override;
  CUptiResult ProfilerHostGetMaxNumHardwareMetricsPerPass(
      CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params* pParams)
      override;

  // Profiler Target APIs
  CUptiResult ProfilerInitialize(
      CUpti_Profiler_Initialize_Params* pParams) override;
  CUptiResult ProfilerDeInitialize(
      CUpti_Profiler_DeInitialize_Params* pParams) override;
  CUptiResult ProfilerCounterDataImageCalculateSize(
      CUpti_Profiler_CounterDataImage_CalculateSize_Params* pParams) override;
  CUptiResult ProfilerCounterDataImageInitialize(
      CUpti_Profiler_CounterDataImage_Initialize_Params* pParams) override;
  CUptiResult ProfilerCounterDataImageCalculateScratchBufferSize(
      CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params*
          pParams) override;
  CUptiResult ProfilerCounterDataImageInitializeScratchBuffer(
      CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams)
      override;
  CUptiResult ProfilerBeginSession(
      CUpti_Profiler_BeginSession_Params* pParams) override;
  CUptiResult ProfilerEndSession(
      CUpti_Profiler_EndSession_Params* pParams) override;
  CUptiResult ProfilerSetConfig(
      CUpti_Profiler_SetConfig_Params* pParams) override;
  CUptiResult ProfilerUnsetConfig(
      CUpti_Profiler_UnsetConfig_Params* pParams) override;
  CUptiResult ProfilerBeginPass(
      CUpti_Profiler_BeginPass_Params* pParams) override;
  CUptiResult ProfilerEndPass(CUpti_Profiler_EndPass_Params* pParams) override;
  CUptiResult ProfilerEnableProfiling(
      CUpti_Profiler_EnableProfiling_Params* pParams) override;
  CUptiResult ProfilerDisableProfiling(
      CUpti_Profiler_DisableProfiling_Params* pParams) override;
  CUptiResult ProfilerIsPassCollected(
      CUpti_Profiler_IsPassCollected_Params* pParams) override;
  CUptiResult ProfilerFlushCounterData(
      CUpti_Profiler_FlushCounterData_Params* pParams) override;
  CUptiResult ProfilerPushRange(
      CUpti_Profiler_PushRange_Params* pParams) override;
  CUptiResult ProfilerPopRange(
      CUpti_Profiler_PopRange_Params* pParams) override;
  CUptiResult ProfilerGetCounterAvailability(
      CUpti_Profiler_GetCounterAvailability_Params* pParams) override;
  CUptiResult ProfilerDeviceSupported(
      CUpti_Profiler_DeviceSupported_Params* pParams) override;

  // PM sampling specific functions
  CUptiResult PmSamplingSetConfig(
      CUpti_PmSampling_SetConfig_Params* pParams) override;
  CUptiResult PmSamplingEnable(
      CUpti_PmSampling_Enable_Params* pParams) override;
  CUptiResult PmSamplingDisable(
      CUpti_PmSampling_Disable_Params* pParams) override;
  CUptiResult PmSamplingStart(CUpti_PmSampling_Start_Params* pParams) override;
  CUptiResult PmSamplingStop(CUpti_PmSampling_Stop_Params* pParams) override;
  CUptiResult PmSamplingDecodeData(
      CUpti_PmSampling_DecodeData_Params* pParams) override;
  CUptiResult PmSamplingGetCounterAvailability(
      CUpti_PmSampling_GetCounterAvailability_Params* pParams) override;
  CUptiResult PmSamplingGetCounterDataSize(
      CUpti_PmSampling_GetCounterDataSize_Params* pParams) override;
  CUptiResult PmSamplingCounterDataImageInitialize(
      CUpti_PmSampling_CounterDataImage_Initialize_Params* pParams) override;
  CUptiResult PmSamplingGetCounterDataInfo(
      CUpti_PmSampling_GetCounterDataInfo_Params* pParams) override;
  CUptiResult PmSamplingCounterDataGetSampleInfo(
      CUpti_PmSampling_CounterData_GetSampleInfo_Params* pParams) override;

  CUptiResult DeviceGetChipName(
      CUpti_Device_GetChipName_Params* pParams) override;

  void CleanUp() override {}
  bool Disabled() const override { return false; }

 private:
  CuptiWrapperStub(const CuptiWrapperStub&) = delete;
  void operator=(const CuptiWrapperStub&) = delete;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_WRAPPER_H_
