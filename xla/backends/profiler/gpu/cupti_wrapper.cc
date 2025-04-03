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

#include "xla/backends/profiler/gpu/cupti_wrapper.h"

#include "cupti_interface.h"
#include "third_party/gpus/cuda/include/cuda.h"

namespace xla {
namespace profiler {

CUptiResult CuptiWrapper::ActivityDisable(CUpti_ActivityKind kind) {
  return cuptiActivityDisable(kind);
}

CUptiResult CuptiWrapper::ActivityEnable(CUpti_ActivityKind kind) {
  return cuptiActivityEnable(kind);
}

CUptiResult CuptiWrapper::ActivityFlushAll(uint32_t flag) {
  return cuptiActivityFlushAll(flag);
}

CUptiResult CuptiWrapper::ActivityGetNextRecord(uint8_t* buffer,
                                                size_t valid_buffer_size_bytes,
                                                CUpti_Activity** record) {
  return cuptiActivityGetNextRecord(buffer, valid_buffer_size_bytes, record);
}

CUptiResult CuptiWrapper::ActivityGetNumDroppedRecords(CUcontext context,
                                                       uint32_t stream_id,
                                                       size_t* dropped) {
  return cuptiActivityGetNumDroppedRecords(context, stream_id, dropped);
}

CUptiResult CuptiWrapper::ActivityConfigureUnifiedMemoryCounter(
    CUpti_ActivityUnifiedMemoryCounterConfig* config, uint32_t count) {
  return cuptiActivityConfigureUnifiedMemoryCounter(config, count);
}

CUptiResult CuptiWrapper::ActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc func_buffer_requested,
    CUpti_BuffersCallbackCompleteFunc func_buffer_completed) {
  return cuptiActivityRegisterCallbacks(func_buffer_requested,
                                        func_buffer_completed);
}

CUptiResult CuptiWrapper::ActivityUsePerThreadBuffer() {
#if CUDA_VERSION >= 12030
  uint8_t use_per_thread_activity_buffer = 1;
  size_t value_size = sizeof(use_per_thread_activity_buffer);
  return cuptiActivitySetAttribute(
      CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER, &value_size,
      &use_per_thread_activity_buffer);
#else
  // cuptiActivitySetAttribute returns CUPTI_ERROR_INVALID_PARAMETER if invoked
  // with an invalid first parameter.
  return CUPTI_ERROR_INVALID_PARAMETER;
#endif
}

CUptiResult CuptiWrapper::SetActivityFlushPeriod(uint32_t period_ms) {
#if CUDA_VERSION >= 11010
  return cuptiActivityFlushPeriod(period_ms);
#else
  return CUPTI_ERROR_NOT_SUPPORTED;
#endif
}

CUptiResult CuptiWrapper::GetDeviceId(CUcontext context, uint32_t* deviceId) {
  return cuptiGetDeviceId(context, deviceId);
}

CUptiResult CuptiWrapper::GetTimestamp(uint64_t* timestamp) {
  return cuptiGetTimestamp(timestamp);
}

CUptiResult CuptiWrapper::Finalize() { return cuptiFinalize(); }

CUptiResult CuptiWrapper::EnableCallback(uint32_t enable,
                                         CUpti_SubscriberHandle subscriber,
                                         CUpti_CallbackDomain domain,
                                         CUpti_CallbackId cbid) {
  return cuptiEnableCallback(enable, subscriber, domain, cbid);
}

CUptiResult CuptiWrapper::EnableDomain(uint32_t enable,
                                       CUpti_SubscriberHandle subscriber,
                                       CUpti_CallbackDomain domain) {
  return cuptiEnableDomain(enable, subscriber, domain);
}

CUptiResult CuptiWrapper::Subscribe(CUpti_SubscriberHandle* subscriber,
                                    CUpti_CallbackFunc callback,
                                    void* userdata) {
  return cuptiSubscribe(subscriber, callback, userdata);
}

CUptiResult CuptiWrapper::Unsubscribe(CUpti_SubscriberHandle subscriber) {
  return cuptiUnsubscribe(subscriber);
}

CUptiResult CuptiWrapper::GetResultString(CUptiResult result,
                                          const char** str) {
  return cuptiGetResultString(result, str);
}

CUptiResult CuptiWrapper::GetContextId(CUcontext context,
                                       uint32_t* context_id) {
  return cuptiGetContextId(context, context_id);
}

CUptiResult CuptiWrapper::GetGraphId(CUgraph graph, uint32_t* graph_id) {
#if CUDA_VERSION >= 11010
  return cuptiGetGraphId(graph, graph_id);
#else
  // Do not treat it as error if the interface is not available.
  if (graph_id) *graph_id = 0;
  return CUPTI_SUCCESS;
#endif
}

CUptiResult CuptiWrapper::GetGraphExecId(CUgraphExec graph_exec,
                                         uint32_t* graph_id) {
  // TODO: (b/350105610), Using cuptiGetGraphExecId() for CUDA 12.3 and later
  return GetGraphId(reinterpret_cast<CUgraph>(graph_exec), graph_id);
}

CUptiResult CuptiWrapper::SetThreadIdType(CUpti_ActivityThreadIdType type) {
  return cuptiSetThreadIdType(type);
}

CUptiResult CuptiWrapper::GetStreamIdEx(CUcontext context, CUstream stream,
                                        uint8_t per_thread_stream,
                                        uint32_t* stream_id) {
  return cuptiGetStreamIdEx(context, stream, per_thread_stream, stream_id);
}

// Profiler Host APIs
// If CUPTI_PM_SAMPLING is not set, no definition for these following CUPTI
// functions will exist, however the references to them will be eliminated by
// the compiler.
CUptiResult CuptiWrapper::ProfilerHostInitialize(
    CUpti_Profiler_Host_Initialize_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerHostInitialize(pParams);
}

CUptiResult CuptiWrapper::ProfilerHostDeinitialize(
    CUpti_Profiler_Host_Deinitialize_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerHostDeinitialize(pParams);
}

CUptiResult CuptiWrapper::ProfilerHostGetSupportedChips(
    CUpti_Profiler_Host_GetSupportedChips_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerHostGetSupportedChips(pParams);
}

CUptiResult CuptiWrapper::ProfilerHostGetBaseMetrics(
    CUpti_Profiler_Host_GetBaseMetrics_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerHostGetBaseMetrics(pParams);
}

CUptiResult CuptiWrapper::ProfilerHostGetSubMetrics(
    CUpti_Profiler_Host_GetSubMetrics_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerHostGetSubMetrics(pParams);
}

CUptiResult CuptiWrapper::ProfilerHostGetMetricProperties(
    CUpti_Profiler_Host_GetMetricProperties_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerHostGetMetricProperties(pParams);
}

CUptiResult CuptiWrapper::ProfilerHostGetRangeName(
    CUpti_Profiler_Host_GetRangeName_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerHostGetRangeName(pParams);
}

CUptiResult CuptiWrapper::ProfilerHostEvaluateToGpuValues(
    CUpti_Profiler_Host_EvaluateToGpuValues_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerHostEvaluateToGpuValues(pParams);
}

CUptiResult CuptiWrapper::ProfilerHostConfigAddMetrics(
    CUpti_Profiler_Host_ConfigAddMetrics_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerHostConfigAddMetrics(pParams);
}

CUptiResult CuptiWrapper::ProfilerHostGetConfigImageSize(
    CUpti_Profiler_Host_GetConfigImageSize_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerHostGetConfigImageSize(pParams);
}

CUptiResult CuptiWrapper::ProfilerHostGetConfigImage(
    CUpti_Profiler_Host_GetConfigImage_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerHostGetConfigImage(pParams);
}

CUptiResult CuptiWrapper::ProfilerHostGetNumOfPasses(
    CUpti_Profiler_Host_GetNumOfPasses_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerHostGetNumOfPasses(pParams);
}

CUptiResult CuptiWrapper::ProfilerHostGetMaxNumHardwareMetricsPerPass(
    CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerHostGetMaxNumHardwareMetricsPerPass(pParams);
}

// Profiler Target APIs
CUptiResult CuptiWrapper::ProfilerInitialize(
    CUpti_Profiler_Initialize_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerInitialize(pParams);
}

CUptiResult CuptiWrapper::ProfilerDeInitialize(
    CUpti_Profiler_DeInitialize_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerDeInitialize(pParams);
}

CUptiResult CuptiWrapper::ProfilerCounterDataImageCalculateSize(
    CUpti_Profiler_CounterDataImage_CalculateSize_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerCounterDataImageCalculateSize(pParams);
}

CUptiResult CuptiWrapper::ProfilerCounterDataImageInitialize(
    CUpti_Profiler_CounterDataImage_Initialize_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerCounterDataImageInitialize(pParams);
}

CUptiResult CuptiWrapper::ProfilerCounterDataImageCalculateScratchBufferSize(
    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params*
        pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerCounterDataImageCalculateScratchBufferSize(pParams);
}

CUptiResult CuptiWrapper::ProfilerCounterDataImageInitializeScratchBuffer(
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerCounterDataImageInitializeScratchBuffer(pParams);
}

CUptiResult CuptiWrapper::ProfilerBeginSession(
    CUpti_Profiler_BeginSession_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerBeginSession(pParams);
}

CUptiResult CuptiWrapper::ProfilerEndSession(
    CUpti_Profiler_EndSession_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerEndSession(pParams);
}

CUptiResult CuptiWrapper::ProfilerSetConfig(
    CUpti_Profiler_SetConfig_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerSetConfig(pParams);
}

CUptiResult CuptiWrapper::ProfilerUnsetConfig(
    CUpti_Profiler_UnsetConfig_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerUnsetConfig(pParams);
}

CUptiResult CuptiWrapper::ProfilerBeginPass(
    CUpti_Profiler_BeginPass_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerBeginPass(pParams);
}

CUptiResult CuptiWrapper::ProfilerEndPass(
    CUpti_Profiler_EndPass_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerEndPass(pParams);
}

CUptiResult CuptiWrapper::ProfilerEnableProfiling(
    CUpti_Profiler_EnableProfiling_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerEnableProfiling(pParams);
}

CUptiResult CuptiWrapper::ProfilerDisableProfiling(
    CUpti_Profiler_DisableProfiling_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerDisableProfiling(pParams);
}

CUptiResult CuptiWrapper::ProfilerIsPassCollected(
    CUpti_Profiler_IsPassCollected_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerIsPassCollected(pParams);
}

CUptiResult CuptiWrapper::ProfilerFlushCounterData(
    CUpti_Profiler_FlushCounterData_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerFlushCounterData(pParams);
}

CUptiResult CuptiWrapper::ProfilerPushRange(
    CUpti_Profiler_PushRange_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerPushRange(pParams);
}

CUptiResult CuptiWrapper::ProfilerPopRange(
    CUpti_Profiler_PopRange_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerPopRange(pParams);
}

CUptiResult CuptiWrapper::ProfilerGetCounterAvailability(
    CUpti_Profiler_GetCounterAvailability_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerGetCounterAvailability(pParams);
}

CUptiResult CuptiWrapper::ProfilerDeviceSupported(
    CUpti_Profiler_DeviceSupported_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiProfilerDeviceSupported(pParams);
}

CUptiResult CuptiWrapper::PmSamplingSetConfig(
    CUpti_PmSampling_SetConfig_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiPmSamplingSetConfig(pParams);
}

CUptiResult CuptiWrapper::PmSamplingEnable(
    CUpti_PmSampling_Enable_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiPmSamplingEnable(pParams);
}

CUptiResult CuptiWrapper::PmSamplingDisable(
    CUpti_PmSampling_Disable_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiPmSamplingDisable(pParams);
}

CUptiResult CuptiWrapper::PmSamplingStart(
    CUpti_PmSampling_Start_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiPmSamplingStart(pParams);
}

CUptiResult CuptiWrapper::PmSamplingStop(
    CUpti_PmSampling_Stop_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiPmSamplingStop(pParams);
}

CUptiResult CuptiWrapper::PmSamplingDecodeData(
    CUpti_PmSampling_DecodeData_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiPmSamplingDecodeData(pParams);
}

CUptiResult CuptiWrapper::PmSamplingGetCounterAvailability(
    CUpti_PmSampling_GetCounterAvailability_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiPmSamplingGetCounterAvailability(pParams);
}

CUptiResult CuptiWrapper::PmSamplingGetCounterDataSize(
    CUpti_PmSampling_GetCounterDataSize_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiPmSamplingGetCounterDataSize(pParams);
}

CUptiResult CuptiWrapper::PmSamplingCounterDataImageInitialize(
    CUpti_PmSampling_CounterDataImage_Initialize_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiPmSamplingCounterDataImageInitialize(pParams);
}

CUptiResult CuptiWrapper::PmSamplingGetCounterDataInfo(
    CUpti_PmSampling_GetCounterDataInfo_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiPmSamplingGetCounterDataInfo(pParams);
}

CUptiResult CuptiWrapper::PmSamplingCounterDataGetSampleInfo(
    CUpti_PmSampling_CounterData_GetSampleInfo_Params* pParams) {
  if (! CUPTI_PM_SAMPLING) return CUPTI_ERROR_NOT_SUPPORTED;
  return cuptiPmSamplingCounterDataGetSampleInfo(pParams);
}

CUptiResult CuptiWrapper::DeviceGetChipName(
    CUpti_Device_GetChipName_Params* pParams) {
  return cuptiDeviceGetChipName(pParams);
}

}  // namespace profiler
}  // namespace xla
