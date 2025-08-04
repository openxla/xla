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

// This translation unit is **self‑contained**: it provides minimal stub
// implementations for the rocprofiler callbacks that XLA needs to register
// (toolInit / toolFinialize / code_object_callback).  They do nothing except
// keep the compiler and linker happy.  Once real logging is implemented, you
// can replace the stubs with the actual logic.

#include "xla/backends/profiler/gpu/rocm_tracer.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <time.h>
#include <unistd.h>
#include <chrono>
#include <unistd.h>  // For standard sysconf

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "rocm/rocm_config.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/mem.h"

// for roctracer (v1)
#if TF_ROCM_VERSION < 60300
namespace xla {
namespace profiler {

namespace se = ::stream_executor;
using tsl::mutex;
using tsl::mutex_lock;
using tsl::profiler::AnnotationStack;
constexpr uint32_t RocmTracerEvent::kInvalidDeviceId;

#define RETURN_IF_ROCTRACER_ERROR(expr)                                     \
  do {                                                                      \
    roctracer_status_t status = expr;                                       \
    if (status != ROCTRACER_STATUS_SUCCESS) {                               \
      const char* errstr = se::wrap::roctracer_error_string();              \
      LOG(ERROR) << "function " << #expr << "failed with error " << errstr; \
      return tsl::errors::Internal(                                         \
          absl::StrCat("roctracer call error", errstr));                    \
    }                                                                       \
  } while (false)

namespace {

// GetCachedTID() caches the thread ID in thread-local storage (which is a
// userspace construct) to avoid unnecessary system calls. Without this caching,
// it can take roughly 98ns, while it takes roughly 1ns with this caching.
int64_t GetCachedTID() {
  static thread_local int64_t current_thread_id =
      tsl::Env::Default()->GetCurrentThreadId();
  return current_thread_id;
}

const char* GetActivityDomainName(uint32_t domain) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_API:
      return "HSA API";
    case ACTIVITY_DOMAIN_HSA_OPS:
      return "HSA OPS";
    case ACTIVITY_DOMAIN_HIP_OPS:
      return "HIP OPS/HCC/VDI";
    case ACTIVITY_DOMAIN_HIP_API:
      return "HIP API";
    case ACTIVITY_DOMAIN_KFD_API:
      return "KFD API";
    case ACTIVITY_DOMAIN_EXT_API:
      return "EXT API";
    case ACTIVITY_DOMAIN_ROCTX:
      return "ROCTX";
    case ACTIVITY_DOMAIN_HSA_EVT:
      return "HSA envents";
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

std::string GetActivityDomainOpName(uint32_t domain, uint32_t op) {
  std::ostringstream oss;
  oss << GetActivityDomainName(domain) << " - ";
  switch (domain) {
    case ACTIVITY_DOMAIN_HIP_API:
      oss << hip_api_name(op);
      break;
    default:
      oss << op;
      break;
  }
  return oss.str();
}

const char* GetActivityPhaseName(uint32_t phase) {
  switch (phase) {
    case ACTIVITY_API_PHASE_ENTER:
      return "ENTER";
    case ACTIVITY_API_PHASE_EXIT:
      return "EXIT";
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

inline void DumpApiCallbackData(uint32_t domain, uint32_t cbid,
                                const void* cbdata) {
  std::ostringstream oss;
  oss << "API callback for " << GetActivityDomainName(domain);
  if (domain == ACTIVITY_DOMAIN_HIP_API) {
    const hip_api_data_t* data =
        reinterpret_cast<const hip_api_data_t*>(cbdata);
    oss << " - " << hip_api_name(cbid);
    oss << ", correlation_id=" << data->correlation_id;
    oss << ", phase=" << GetActivityPhaseName(data->phase);
    switch (cbid) {
      case HIP_API_ID_hipModuleLaunchKernel:
      case HIP_API_ID_hipExtModuleLaunchKernel:
      case HIP_API_ID_hipHccModuleLaunchKernel:
      case HIP_API_ID_hipLaunchKernel:
      case HIP_API_ID_hipExtLaunchKernel:
        break;
      case HIP_API_ID_hipMemcpyDtoH:
        oss << ", sizeBytes=" << data->args.hipMemcpyDtoH.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyDtoHAsync:
        oss << ", sizeBytes=" << data->args.hipMemcpyDtoHAsync.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyHtoD:
        oss << ", sizeBytes=" << data->args.hipMemcpyHtoD.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyHtoDAsync:
        oss << ", sizeBytes=" << data->args.hipMemcpyHtoDAsync.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyDtoD:
        oss << ", sizeBytes=" << data->args.hipMemcpyDtoD.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyDtoDAsync:
        oss << ", sizeBytes=" << data->args.hipMemcpyDtoDAsync.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyAsync:
        oss << ", sizeBytes=" << data->args.hipMemcpyAsync.sizeBytes;
        break;
      case HIP_API_ID_hipMemsetD32:
        oss << ", value=" << data->args.hipMemsetD32.value;
        oss << ", count=" << data->args.hipMemsetD32.count;
        break;
      case HIP_API_ID_hipMemsetD32Async:
        oss << ", value=" << data->args.hipMemsetD32Async.value;
        oss << ", count=" << data->args.hipMemsetD32Async.count;
        break;
      case HIP_API_ID_hipMemsetD8:
        oss << ", value=" << data->args.hipMemsetD8.value;
        oss << ", count=" << data->args.hipMemsetD8.count;
        break;
      case HIP_API_ID_hipMemsetD8Async:
        oss << ", value=" << data->args.hipMemsetD8Async.value;
        oss << ", count=" << data->args.hipMemsetD8Async.count;
        break;
      case HIP_API_ID_hipMalloc:
        oss << ", size=" << data->args.hipMalloc.size;
        break;
      case HIP_API_ID_hipFree:
        oss << ", ptr=" << data->args.hipFree.ptr;
        break;
      case HIP_API_ID_hipStreamSynchronize:
        break;
      case HIP_API_ID_hipStreamWaitEvent:  // ignore all aux HIP API Events
      case HIP_API_ID_hipHostFree:
      case HIP_API_ID_hipHostMalloc:
      case HIP_API_ID_hipSetDevice:
        break;
      default:
        VLOG(3) << "Warning: HIP API is not handled: HIP_API_ID_"
                << hip_api_name(cbid);
        break;
    }
  } else {
    oss << ": " << cbid;
  }
  VLOG(3) << oss.str();
}

void DumpActivityRecord(const roctracer_record_t* record,
                        std::string extra_info) {
  std::ostringstream oss;
  oss << "Activity callback for " << GetActivityDomainName(record->domain);
  oss << ", op name= "
      << se::wrap::roctracer_op_string(record->domain, record->op,
                                       record->kind);
  oss << ", correlation_id=" << record->correlation_id;
  oss << ", begin_ns=" << record->begin_ns;
  oss << ", end_ns=" << record->end_ns;
  oss << ", duration=" << record->end_ns - record->begin_ns;
  oss << ", device_id=" << record->device_id;
  oss << ", queue_id=" << record->queue_id;
  oss << ", process_id=" << record->process_id;
  oss << ", thread_id=" << record->thread_id;
  oss << ", external_id=" << record->external_id;
  oss << ", bytes=" << record->bytes;
  oss << ", domain=" << record->domain;
  oss << ", op=" << record->op;
  oss << ", kind=" << record->kind;
  oss << ", extra_info=" << extra_info;
  VLOG(3) << oss.str();
}

}  // namespace

const char* GetRocmTracerEventTypeName(const RocmTracerEventType& type) {
  switch (type) {
    case RocmTracerEventType::Kernel:
      return "Kernel";
    case RocmTracerEventType::MemcpyH2D:
      return "MemcpyH2D";
    case RocmTracerEventType::MemcpyD2H:
      return "MemcpyD2H";
    case RocmTracerEventType::MemcpyD2D:
      return "MemcpyD2D";
    case RocmTracerEventType::MemcpyP2P:
      return "MemcpyP2P";
    case RocmTracerEventType::MemcpyOther:
      return "MemcpyOther";
    case RocmTracerEventType::MemoryAlloc:
      return "MemoryAlloc";
    case RocmTracerEventType::MemoryFree:
      return "MemoryFree";
    case RocmTracerEventType::Memset:
      return "Memset";
    case RocmTracerEventType::Synchronization:
      return "Synchronization";
    case RocmTracerEventType::Generic:
      return "Generic";
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

const char* GetRocmTracerEventSourceName(const RocmTracerEventSource& source) {
  switch (source) {
    case RocmTracerEventSource::ApiCallback:
      return "ApiCallback";
      break;
    case RocmTracerEventSource::Activity:
      return "Activity";
      break;
    case RocmTracerEventSource::Invalid:
      return "Invalid";
      break;
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

// FIXME(rocm-profiler): These domain names are not consistent with the
// GetActivityDomainName function
const char* GetRocmTracerEventDomainName(const RocmTracerEventDomain& domain) {
  switch (domain) {
    case RocmTracerEventDomain::HIP_API:
      return "HIP_API";
      break;
    case RocmTracerEventDomain::HIP_OPS:
      return "HIP_OPS";
      break;
    default:
      VLOG(3) << "RocmTracerEventDomain::InvalidDomain";
      DCHECK(false);
      return "";
  }
  return "";
}

absl::Status RocmApiCallbackImpl::operator()(uint32_t domain, uint32_t cbid,
                                             const void* cbdata) {
  /* Some APIs such as hipMalloc, implicitly work on th devices set by the
    user using APIs such as hipSetDevice. API callbacks and activity records
    for functions like hipMalloc does not return the device id (CUDA does). To
    solve this we need to track the APIs that select the device (such as
    hipSetDevice) for each thread.
    */

  thread_local uint32_t default_device = hipGetStreamDeviceId(nullptr);

  // DumpApiCallbackData(domain, cbid, cbdata);

  if (domain != ACTIVITY_DOMAIN_HIP_API) return tsl::OkStatus();

  const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(cbdata);

  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    if (options_.api_tracking_set.find(cbid) !=
        options_.api_tracking_set.end()) {
      mutex_lock lock(api_call_start_mutex_);
      api_call_start_time_.emplace(data->correlation_id,
                                   RocmTracer::GetTimestamp());
    }

    if (cbid == HIP_API_ID_hipSetDevice) {
      default_device = hipGetStreamDeviceId(nullptr);
    }
  } else if (data->phase == ACTIVITY_API_PHASE_EXIT) {
    uint64_t enter_time = 0, exit_time = 0;

    if (options_.api_tracking_set.find(cbid) !=
        options_.api_tracking_set.end()) {
      mutex_lock lock(api_call_start_mutex_);
      if (api_call_start_time_.find(data->correlation_id) !=
          api_call_start_time_.end()) {
        enter_time = api_call_start_time_.at(data->correlation_id);
        api_call_start_time_.erase(data->correlation_id);
      } else {
        LOG(WARNING) << "An API exit callback received without API enter "
                        "with same correlation id. Event droped!";
        return tsl::OkStatus();  // This API does not belong to us.
      }
      exit_time = RocmTracer::GetTimestamp();
    }
    // Set up the map from correlation id to annotation string.
    const std::string& annotation = AnnotationStack::Get();
    if (!annotation.empty()) {
      collector_->annotation_map()->Add(data->correlation_id, annotation);
    }

    if (options_.api_tracking_set.find(cbid) ==
        options_.api_tracking_set.end()) {
      VLOG(3) << "API callback is from the auxilarity list. Corr. id="
              << data->correlation_id;
    }
    DumpApiCallbackData(domain, cbid, cbdata);

    switch (cbid) {
      // star in comments means it does not exist in the driver wrapper
      case HIP_API_ID_hipModuleLaunchKernel:
      case HIP_API_ID_hipExtModuleLaunchKernel:  // *
      case HIP_API_ID_hipHccModuleLaunchKernel:  // *
      case HIP_API_ID_hipLaunchKernel:           // *
      case HIP_API_ID_hipExtLaunchKernel:

        this->AddKernelEventUponApiExit(cbid, data, enter_time, exit_time);

        // Add the correlation_ids for these events to the pending set
        // so that we can explicitly wait for their corresponding
        // HIP runtime activity records, before exporting the trace data
        tracer_->AddToPendingActivityRecords(data->correlation_id);
        break;
      case HIP_API_ID_hipMemcpy:
      case HIP_API_ID_hipMemcpyDtoH:
      case HIP_API_ID_hipMemcpyDtoHAsync:
      case HIP_API_ID_hipMemcpyHtoD:
      case HIP_API_ID_hipMemcpyHtoDAsync:
      case HIP_API_ID_hipMemcpyDtoD:
      case HIP_API_ID_hipMemcpyDtoDAsync:
      case HIP_API_ID_hipMemcpyAsync:
        this->AddNormalMemcpyEventUponApiExit(cbid, data, enter_time,
                                              exit_time);
        tracer_->AddToPendingActivityRecords(data->correlation_id);
        break;
      case HIP_API_ID_hipMemset:
      case HIP_API_ID_hipMemsetAsync:
      case HIP_API_ID_hipMemsetD32:
      case HIP_API_ID_hipMemsetD32Async:
      case HIP_API_ID_hipMemsetD16:
      case HIP_API_ID_hipMemsetD16Async:
      case HIP_API_ID_hipMemsetD8:
      case HIP_API_ID_hipMemsetD8Async:
        this->AddMemsetEventUponApiExit(cbid, data, enter_time, exit_time);
        break;
      case HIP_API_ID_hipMalloc:
      case HIP_API_ID_hipMallocPitch:
      case HIP_API_ID_hipHostMalloc:
      case HIP_API_ID_hipFree:
      case HIP_API_ID_hipHostFree:
        this->AddMallocFreeEventUponApiExit(cbid, data, default_device,
                                            enter_time, exit_time);
        break;
      case HIP_API_ID_hipStreamSynchronize:
      case HIP_API_ID_hipStreamWaitEvent:
        // case HIP_API_ID_hipEventSynchronize:
        this->AddSynchronizeEventUponApiExit(cbid, data, enter_time, exit_time);
        break;
      case HIP_API_ID_hipSetDevice:
        // we track this ID only to find the device ID
        //  for the current thread.
        break;
      default:
        //
        LOG(WARNING) << "API call "
                     << se::wrap::roctracer_op_string(ACTIVITY_DOMAIN_HIP_API,
                                                      cbid, 0)
                     << ", corr. id=" << data->correlation_id
                     << " dropped. No capturing function was found!";
        // AddGenericEventUponApiExit(cbid, data);
        break;
    }
  }
  return tsl::OkStatus();
}

void RocmApiCallbackImpl::AddKernelEventUponApiExit(uint32_t cbid,
                                                    const hip_api_data_t* data,
                                                    const uint64_t enter_time,
                                                    const uint64_t exit_time) {
  /*
  extra fields:
    kernel_info, domain

  missing fields:
    context_id
  */
  RocmTracerEvent event;

  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = RocmTracerEventType::Kernel;
  event.source = RocmTracerEventSource::ApiCallback;
  event.thread_id = GetCachedTID();
  event.correlation_id = data->correlation_id;
  event.start_time_ns = enter_time;
  event.end_time_ns = exit_time;

  switch (cbid) {
    case HIP_API_ID_hipModuleLaunchKernel: {
      const hipFunction_t kernelFunc = data->args.hipModuleLaunchKernel.f;
      if (kernelFunc != nullptr) event.name = hipKernelNameRef(kernelFunc);

      event.kernel_info.dynamic_shared_memory_usage =
          data->args.hipModuleLaunchKernel.sharedMemBytes;
      event.kernel_info.block_x = data->args.hipModuleLaunchKernel.blockDimX;
      event.kernel_info.block_y = data->args.hipModuleLaunchKernel.blockDimY;
      event.kernel_info.block_z = data->args.hipModuleLaunchKernel.blockDimZ;
      event.kernel_info.grid_x = data->args.hipModuleLaunchKernel.gridDimX;
      event.kernel_info.grid_y = data->args.hipModuleLaunchKernel.gridDimY;
      event.kernel_info.grid_z = data->args.hipModuleLaunchKernel.gridDimZ;
      event.kernel_info.func_ptr = kernelFunc;
      const hipStream_t& stream = data->args.hipModuleLaunchKernel.stream;
      // TODO(rocm-profiler): wrap this API if possible.
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
    case HIP_API_ID_hipExtModuleLaunchKernel: {
      const hipFunction_t kernelFunc = data->args.hipExtModuleLaunchKernel.f;
      if (kernelFunc != nullptr) event.name = hipKernelNameRef(kernelFunc);

      event.kernel_info.dynamic_shared_memory_usage =
          data->args.hipExtModuleLaunchKernel.sharedMemBytes;
      unsigned int blockDimX =
          data->args.hipExtModuleLaunchKernel.localWorkSizeX;
      unsigned int blockDimY =
          data->args.hipExtModuleLaunchKernel.localWorkSizeY;
      unsigned int blockDimZ =
          data->args.hipExtModuleLaunchKernel.localWorkSizeZ;

      event.kernel_info.block_x = blockDimX;
      event.kernel_info.block_y = blockDimY;
      event.kernel_info.block_z = blockDimZ;
      event.kernel_info.grid_x =
          data->args.hipExtModuleLaunchKernel.globalWorkSizeX / blockDimX;
      event.kernel_info.grid_y =
          data->args.hipExtModuleLaunchKernel.globalWorkSizeY / blockDimY;
      event.kernel_info.grid_z =
          data->args.hipExtModuleLaunchKernel.globalWorkSizeZ / blockDimZ;
      event.kernel_info.func_ptr = kernelFunc;
      const hipStream_t& stream = data->args.hipExtModuleLaunchKernel.hStream;
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
    case HIP_API_ID_hipHccModuleLaunchKernel: {
      const hipFunction_t kernelFunc = data->args.hipHccModuleLaunchKernel.f;
      if (kernelFunc != nullptr) event.name = hipKernelNameRef(kernelFunc);

      event.kernel_info.dynamic_shared_memory_usage =
          data->args.hipHccModuleLaunchKernel.sharedMemBytes;
      event.kernel_info.block_x = data->args.hipHccModuleLaunchKernel.blockDimX;
      event.kernel_info.block_y = data->args.hipHccModuleLaunchKernel.blockDimY;
      event.kernel_info.block_z = data->args.hipHccModuleLaunchKernel.blockDimZ;
      event.kernel_info.grid_x =
          data->args.hipHccModuleLaunchKernel.globalWorkSizeX /
          event.kernel_info.block_x;
      event.kernel_info.grid_y =
          data->args.hipHccModuleLaunchKernel.globalWorkSizeY /
          event.kernel_info.block_y;
      event.kernel_info.grid_z =
          data->args.hipHccModuleLaunchKernel.globalWorkSizeZ /
          event.kernel_info.block_z;
      event.kernel_info.func_ptr = kernelFunc;
      const hipStream_t& stream = data->args.hipHccModuleLaunchKernel.hStream;
      event.device_id = hipGetStreamDeviceId(stream);
      event.kernel_info.dynamic_shared_memory_usage =
          data->args.hipHccModuleLaunchKernel.sharedMemBytes;
    } break;
    case HIP_API_ID_hipLaunchKernel: {
      const void* func_addr = data->args.hipLaunchKernel.function_address;
      hipStream_t stream = data->args.hipLaunchKernel.stream;
      if (func_addr != nullptr)
        event.name = hipKernelNameRefByPtr(func_addr, stream);

      event.kernel_info.dynamic_shared_memory_usage =
          data->args.hipLaunchKernel.sharedMemBytes;
      event.kernel_info.block_x = data->args.hipLaunchKernel.dimBlocks.x;
      event.kernel_info.block_y = data->args.hipLaunchKernel.dimBlocks.y;
      event.kernel_info.block_z = data->args.hipLaunchKernel.dimBlocks.z;
      event.kernel_info.grid_x = data->args.hipLaunchKernel.numBlocks.x;
      event.kernel_info.grid_y = data->args.hipLaunchKernel.numBlocks.y;
      event.kernel_info.grid_z = data->args.hipLaunchKernel.numBlocks.z;
      event.kernel_info.func_ptr = (void*)func_addr;
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
    case HIP_API_ID_hipExtLaunchKernel: {
      const void* func_addr = data->args.hipExtLaunchKernel.function_address;
      hipStream_t stream = data->args.hipExtLaunchKernel.stream;
      if (func_addr != nullptr)
        event.name = hipKernelNameRefByPtr(func_addr, stream);

      event.kernel_info.dynamic_shared_memory_usage =
          data->args.hipExtLaunchKernel.sharedMemBytes;
      event.kernel_info.block_x = data->args.hipExtLaunchKernel.dimBlocks.x;
      event.kernel_info.block_y = data->args.hipExtLaunchKernel.dimBlocks.y;
      event.kernel_info.block_z = data->args.hipExtLaunchKernel.dimBlocks.z;
      event.kernel_info.grid_x = data->args.hipExtLaunchKernel.numBlocks.x;
      event.kernel_info.grid_y = data->args.hipExtLaunchKernel.numBlocks.y;
      event.kernel_info.grid_z = data->args.hipExtLaunchKernel.numBlocks.z;
      event.kernel_info.func_ptr = const_cast<void*>(func_addr);
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
  }
  bool is_auxiliary =
      options_.api_tracking_set.find(cbid) == options_.api_tracking_set.end();
  collector_->AddEvent(std::move(event), is_auxiliary);
}

void RocmApiCallbackImpl::AddNormalMemcpyEventUponApiExit(
    uint32_t cbid, const hip_api_data_t* data, uint64_t enter_time,
    uint64_t exit_time) {
  /*
    missing:
      device_id(partially, have only for async), context_id,
    memcpy_info.kind(CUPTI puts CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN),
      memcpy_info.destination(partially, only for async)( CUPTI puts device_id),

    extra:
      domain, name,
  */
  // for CUDA, it does NOT capture stream id for these types

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.name = se::wrap::roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid, 0);
  event.source = RocmTracerEventSource::ApiCallback;
  event.thread_id = GetCachedTID();
  event.correlation_id = data->correlation_id;
  event.start_time_ns = enter_time;
  event.end_time_ns = exit_time;

  /* The general hipMemcpy or hipMemcpyAsync can support any kind of memory
  copy operation, such as H2D, D2D, P2P, and D2H. Here we use MemcpyOther for
  all api calls with HipMemcpy(+Async) to carry-on this generality.
  We also assume that if we want to copy data BETWEEN devices, we do not use
  hipMemcpy(+Async) or hipMemcpyDtoD(+Async) as we explicitly always set the
  destenation as the source device id). Ultimately, to figure out the actual
  device we can use hipPointerGetAttributes but we do not do that now .In the
  other words, we assume we use hipMemcpyPeer to achieve the copy between
  devices.
  */

  switch (cbid) {
    case HIP_API_ID_hipMemcpyDtoH: {
      event.type = RocmTracerEventType::MemcpyD2H;
      event.memcpy_info.num_bytes = data->args.hipMemcpyDtoH.sizeBytes;
      event.memcpy_info.async = false;
    } break;
    case HIP_API_ID_hipMemcpyDtoHAsync: {
      event.type = RocmTracerEventType::MemcpyD2H;
      const hipStream_t& stream = data->args.hipMemcpyDtoHAsync.stream;
      event.device_id = hipGetStreamDeviceId(stream);
      event.memcpy_info.num_bytes = data->args.hipMemcpyDtoHAsync.sizeBytes;
      event.memcpy_info.async = true;
      event.memcpy_info.destination = event.device_id;
    } break;
    case HIP_API_ID_hipMemcpyHtoD: {
      event.type = RocmTracerEventType::MemcpyH2D;
      event.memcpy_info.num_bytes = data->args.hipMemcpyHtoD.sizeBytes;
      event.memcpy_info.async = false;
      // we set the destenattion device id for it using the device id we get
      // from activities when they exchange information before flushing
    } break;
    case HIP_API_ID_hipMemcpyHtoDAsync: {
      event.type = RocmTracerEventType::MemcpyH2D;
      const hipStream_t& stream = data->args.hipMemcpyHtoDAsync.stream;
      event.device_id = hipGetStreamDeviceId(stream);
      event.memcpy_info.num_bytes = data->args.hipMemcpyHtoDAsync.sizeBytes;
      event.memcpy_info.async = true;
      event.memcpy_info.destination = event.device_id;
    } break;
    case HIP_API_ID_hipMemcpyDtoD: {
      event.type = RocmTracerEventType::MemcpyD2D;
      event.memcpy_info.num_bytes = data->args.hipMemcpyDtoD.sizeBytes;
      event.memcpy_info.async = false;
    } break;
    case HIP_API_ID_hipMemcpyDtoDAsync: {
      event.type = RocmTracerEventType::MemcpyD2D;
      const hipStream_t& stream = data->args.hipMemcpyDtoDAsync.stream;
      event.device_id = hipGetStreamDeviceId(stream);
      event.memcpy_info.num_bytes = data->args.hipMemcpyDtoDAsync.sizeBytes;
      event.memcpy_info.async = true;
      event.memcpy_info.destination = event.device_id;
    } break;
    case HIP_API_ID_hipMemcpy: {
      event.type = RocmTracerEventType::MemcpyOther;
      event.memcpy_info.num_bytes = data->args.hipMemcpy.sizeBytes;
      event.memcpy_info.async = false;
    } break;
    case HIP_API_ID_hipMemcpyAsync: {
      event.type = RocmTracerEventType::MemcpyOther;
      const hipStream_t& stream = data->args.hipMemcpyAsync.stream;
      event.device_id = hipGetStreamDeviceId(stream);
      event.memcpy_info.num_bytes = data->args.hipMemcpyAsync.sizeBytes;
      event.memcpy_info.async = true;
      event.memcpy_info.destination = event.device_id;
    } break;
    default:
      LOG(WARNING) << "Unsupported Memcpy API for profiling observed for cbid="
                   << cbid << ". Event dropped!";
      return;
      break;
  }

  bool is_auxiliary =
      options_.api_tracking_set.find(cbid) == options_.api_tracking_set.end();
  collector_->AddEvent(std::move(event), is_auxiliary);
}
void RocmApiCallbackImpl::AddMemcpyPeerEventUponApiExit(
    uint32_t cbid, const hip_api_data_t* data, uint64_t enter_time,
    uint64_t exit_time) {
  /*
    missing: context_id, memcpy_info.kind

    extra: domain, name,
  */

  RocmTracerEvent event;
  event.type = RocmTracerEventType::MemcpyP2P;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.name = se::wrap::roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid, 0);
  event.source = RocmTracerEventSource::ApiCallback;
  event.thread_id = GetCachedTID();
  event.correlation_id = data->correlation_id;
  event.start_time_ns = enter_time;
  event.end_time_ns = exit_time;

  switch (cbid) {
    case HIP_API_ID_hipMemcpyPeer:
      event.device_id = data->args.hipMemcpyPeer.srcDeviceId;
      event.memcpy_info.destination = data->args.hipMemcpyPeer.dstDeviceId;
      event.memcpy_info.num_bytes = data->args.hipMemcpyPeer.sizeBytes;
      event.memcpy_info.async = false;
      break;
    case HIP_API_ID_hipMemcpyPeerAsync:
      event.device_id = data->args.hipMemcpyPeerAsync.srcDevice;
      event.memcpy_info.destination = data->args.hipMemcpyPeerAsync.dstDeviceId;
      event.memcpy_info.num_bytes = data->args.hipMemcpyPeerAsync.sizeBytes;
      event.memcpy_info.async = true;
      break;
    default:
      LOG(WARNING)
          << "Unsupported MemcpyPeer API for profiling observed for cbid="
          << cbid << ". Event dropped!";
      return;
      break;
  }

  bool is_auxiliary =
      options_.api_tracking_set.find(cbid) == options_.api_tracking_set.end();
  collector_->AddEvent(std::move(event), is_auxiliary);
}
void RocmApiCallbackImpl::AddMemsetEventUponApiExit(uint32_t cbid,
                                                    const hip_api_data_t* data,
                                                    uint64_t enter_time,
                                                    uint64_t exit_time) {
  /*
    misses:
      device_id(only avail. for async), context_id

    extras:
      domain, name
  */

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.name = se::wrap::roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid, 0);
  event.source = RocmTracerEventSource::ApiCallback;
  event.thread_id = GetCachedTID();
  event.correlation_id = data->correlation_id;
  event.start_time_ns = enter_time;
  event.end_time_ns = exit_time;

  switch (cbid) {
    case HIP_API_ID_hipMemsetD8:
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_bytes = data->args.hipMemsetD8.count;
      event.memset_info.async = false;
      break;
    case HIP_API_ID_hipMemsetD8Async: {
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_bytes = data->args.hipMemsetD8Async.count;
      event.memset_info.async = true;
      const hipStream_t& stream = data->args.hipMemsetD8Async.stream;
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
    case HIP_API_ID_hipMemsetD16:
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_bytes = 2 * data->args.hipMemsetD16.count;
      event.memset_info.async = false;
      break;
    case HIP_API_ID_hipMemsetD16Async: {
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_bytes = 2 * data->args.hipMemsetD16Async.count;
      event.memset_info.async = true;
      const hipStream_t& stream = data->args.hipMemsetD16Async.stream;
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
    case HIP_API_ID_hipMemsetD32:
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_bytes = 4 * data->args.hipMemsetD32.count;
      event.memset_info.async = false;
      break;
    case HIP_API_ID_hipMemsetD32Async: {
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_bytes = 4 * data->args.hipMemsetD32Async.count;
      event.memset_info.async = true;
      const hipStream_t& stream = data->args.hipMemsetD32Async.stream;
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
    case HIP_API_ID_hipMemset:
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_bytes = data->args.hipMemset.sizeBytes;
      event.memset_info.async = false;
      break;
    case HIP_API_ID_hipMemsetAsync: {
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_bytes = data->args.hipMemsetAsync.sizeBytes;
      event.memset_info.async = true;
      const hipStream_t& stream = data->args.hipMemsetAsync.stream;
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
    default:
      LOG(WARNING) << "Unsupported Memset API for profiling observed for cbid="
                   << cbid << ". Event dropped!";
      return;
      break;
  }

  bool is_auxiliary =
      options_.api_tracking_set.find(cbid) == options_.api_tracking_set.end();
  collector_->AddEvent(std::move(event), is_auxiliary);
}

void RocmApiCallbackImpl::AddMallocFreeEventUponApiExit(
    uint32_t cbid, const hip_api_data_t* data, uint32_t device_id,
    uint64_t enter_time, uint64_t exit_time) {
  /*
    misses: context_id

    extras: domain
  */

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = (cbid == HIP_API_ID_hipFree || cbid == HIP_API_ID_hipHostFree)
                   ? RocmTracerEventType::MemoryFree
                   : RocmTracerEventType::MemoryAlloc;
  event.source = RocmTracerEventSource::ApiCallback;
  event.name = se::wrap::roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid, 0);
  event.device_id = device_id;
  event.thread_id = GetCachedTID();
  // We do not set stream_id (probably to zero as Malloc etc. commands seems
  // to run on  default stream). Later we use the unassigned stream_id as a
  // feature to assign events to host or device.
  event.correlation_id = data->correlation_id;
  event.start_time_ns = enter_time;
  event.end_time_ns = exit_time;

  switch (cbid) {
    case HIP_API_ID_hipMalloc:
      event.memalloc_info.num_bytes = data->args.hipMalloc.size;
      break;
    case HIP_API_ID_hipMallocPitch:
      event.memalloc_info.num_bytes = data->args.hipMallocPitch.pitch__val *
                                      data->args.hipMallocPitch.height;
      break;
    case HIP_API_ID_hipHostMalloc:
      event.memalloc_info.num_bytes = data->args.hipHostMalloc.size;
      break;
    case HIP_API_ID_hipFree:
    case HIP_API_ID_hipHostFree:
      event.memalloc_info.num_bytes = 0;
      break;
    default:
      LOG(WARNING)
          << "Unsupported Malloc/Free API for profiling observed for cbid="
          << cbid << ". Event dropped!";
      return;
      break;
  }

  bool is_auxiliary =
      options_.api_tracking_set.find(cbid) == options_.api_tracking_set.end();
  collector_->AddEvent(std::move(event), is_auxiliary);
}

void RocmApiCallbackImpl::AddSynchronizeEventUponApiExit(
    uint32_t cbid, const hip_api_data_t* data, uint64_t enter_time,
    uint64_t exit_time) {
  // TODO(rocm-profiler): neither CUDA and nor we capture annotaint for this
  // event
  /*
    misses: context_id

    extras: domain,
  */

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = RocmTracerEventType::Synchronization;
  event.source = RocmTracerEventSource::ApiCallback;
  event.name = se::wrap::roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid, 0);
  event.thread_id = GetCachedTID();
  event.correlation_id = data->correlation_id;
  event.start_time_ns = enter_time;
  event.end_time_ns = exit_time;

  switch (cbid) {
    case HIP_API_ID_hipStreamSynchronize: {
      event.synchronization_info.sync_type =
          RocmTracerSyncTypes::StreamSynchronize;
      const hipStream_t& stream = data->args.hipStreamSynchronize.stream;
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
    case HIP_API_ID_hipStreamWaitEvent: {
      event.synchronization_info.sync_type = RocmTracerSyncTypes::StreamWait;
      const hipStream_t& stream = data->args.hipStreamWaitEvent.stream;
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
    default:
      LOG(WARNING)
          << "Unsupported Synchronization API for profiling observed for cbid="
          << cbid << ". Event dropped!";
      return;
      break;
  }
  bool is_auxiliary =
      options_.api_tracking_set.find(cbid) == options_.api_tracking_set.end();
  collector_->AddEvent(std::move(event), is_auxiliary);
}

absl::Status RocmActivityCallbackImpl::operator()(const char* begin,
                                                  const char* end) {
  // we do not dump activities in this set in logger

  static std::set<activity_op_t> dump_excluded_activities = {
      HIP_API_ID_hipGetDevice,
      HIP_API_ID_hipSetDevice,
      HIP_API_ID___hipPushCallConfiguration,
      HIP_API_ID___hipPopCallConfiguration,
      HIP_API_ID_hipEventQuery,
      HIP_API_ID_hipCtxSetCurrent,
      HIP_API_ID_hipEventRecord,
      HIP_API_ID_hipEventQuery,
      HIP_API_ID_hipGetDeviceProperties,
      HIP_API_ID_hipPeekAtLastError,
      HIP_API_ID_hipModuleGetFunction,
      HIP_API_ID_hipEventCreateWithFlags};

  const roctracer_record_t* record =
      reinterpret_cast<const roctracer_record_t*>(begin);
  const roctracer_record_t* end_record =
      reinterpret_cast<const roctracer_record_t*>(end);

  while (record < end_record) {
    // DumpActivityRecord(record);

    switch (record->domain) {
      // HIP API activities.
      case ACTIVITY_DOMAIN_HIP_API:
        switch (record->op) {
          case HIP_API_ID_hipModuleLaunchKernel:
          case HIP_API_ID_hipExtModuleLaunchKernel:
          case HIP_API_ID_hipHccModuleLaunchKernel:
          case HIP_API_ID_hipLaunchKernel:
          case HIP_API_ID_hipExtLaunchKernel:
            DumpActivityRecord(record, std::to_string(__LINE__));
            AddHipKernelActivityEvent(record);
            break;
          case HIP_API_ID_hipMemcpyDtoH:
          case HIP_API_ID_hipMemcpyHtoD:
          case HIP_API_ID_hipMemcpyDtoD:
          case HIP_API_ID_hipMemcpyDtoHAsync:
          case HIP_API_ID_hipMemcpyHtoDAsync:
          case HIP_API_ID_hipMemcpyDtoDAsync:
          case HIP_API_ID_hipMemcpyAsync:
          case HIP_API_ID_hipMemcpy:
            DumpActivityRecord(record, std::to_string(__LINE__));
            AddNormalHipMemcpyActivityEvent(record);
            break;
          case HIP_API_ID_hipMemset:
          case HIP_API_ID_hipMemsetAsync:
          case HIP_API_ID_hipMemsetD32:
          case HIP_API_ID_hipMemsetD32Async:
          case HIP_API_ID_hipMemsetD16:
          case HIP_API_ID_hipMemsetD16Async:
          case HIP_API_ID_hipMemsetD8:
          case HIP_API_ID_hipMemsetD8Async:
            DumpActivityRecord(record, std::to_string(__LINE__));
            AddHipMemsetActivityEvent(record);
            break;

          case HIP_API_ID_hipMalloc:
          case HIP_API_ID_hipMallocPitch:
          case HIP_API_ID_hipHostMalloc:
          case HIP_API_ID_hipFree:
          case HIP_API_ID_hipHostFree:
            DumpActivityRecord(record, std::to_string(__LINE__));
            AddHipMallocActivityEvent(record);
            break;
          case HIP_API_ID_hipStreamSynchronize:
          case HIP_API_ID_hipStreamWaitEvent:
            // case HIP_API_ID_hipStreamWaitEvent:
            DumpActivityRecord(record, std::to_string(__LINE__));
            AddHipStreamSynchronizeActivityEvent(record);
            break;

          default:
            if (dump_excluded_activities.find(record->op) ==
                dump_excluded_activities.end()) {
              std::string drop_message(
                  "\nNot in the API tracked activities. Dropped!");
              DumpActivityRecord(record, drop_message);
            }
            break;
        }  // switch (record->op).
        break;

      // HCC ops activities.
      case ACTIVITY_DOMAIN_HIP_OPS:

        switch (record->op) {
          case HIP_OP_ID_DISPATCH:
            DumpActivityRecord(record, std::to_string(__LINE__));
            AddHccKernelActivityEvent(record);
            tracer_->RemoveFromPendingActivityRecords(record->correlation_id);
            break;
          case HIP_OP_ID_COPY:
            switch (record->kind) {
              // TODO(rocm-profiler): use enum instead.
              case 4595:   /*CopyDeviceToHost*/
              case 4596:   /*CopyDeviceToDevice*/
              case 4597: { /*CopyHostToDevice*/
                /*MEMCPY*/
                // roctracer returns CopyHostToDevice for hipMemcpyDtoD API
                //  Please look at the issue #53 in roctracer GitHub repo.
                DumpActivityRecord(record, "");
                AddNormalHipOpsMemcpyActivityEvent(record);
                tracer_->RemoveFromPendingActivityRecords(
                    record->correlation_id);
              } break;
              case 4615: /*FillBuffer*/
                /*MEMSET*/
                DumpActivityRecord(record, "");
                AddHipOpsMemsetActivityEvent(record);
                break;
              case 4606: /*MARKER*/
                // making the log shorter.
                // markers are with 0ns duration.
                break;
              default:
                std::string drop_message(
                    "\nNot in the HIP-OPS-COPY tracked activities. Dropeed!");
                DumpActivityRecord(record, drop_message);
                break;
            }  // switch (record->kind)
            break;
          default:
            std::string drop_message(
                "\nNot in the HIP-OPS tracked activities. Dropped!");
            DumpActivityRecord(record, drop_message);
            break;
        }  // switch (record->op).
        break;
      default:
        std::string drop_message(
            "\nNot in the tracked domain activities. Dropped!");
        DumpActivityRecord(record, drop_message);
        break;
    }

    RETURN_IF_ROCTRACER_ERROR(static_cast<roctracer_status_t>(
#if TF_ROCM_VERSION >= 50300
        se::wrap::roctracer_next_record(record, &record)
#else
        roctracer_next_record(record, &record)
#endif
            ));
  }

  return tsl::OkStatus();
}

void RocmActivityCallbackImpl::AddHipKernelActivityEvent(
    const roctracer_record_t* record) {
  /*
  missing:
   name, device_id(got from hcc), context_id, stream_id(got from hcc),
 nvtx_range, kernel_info

  extra:
   domain
 activity record contains process/thread ID
 */
  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = RocmTracerEventType::Kernel;
  event.source = RocmTracerEventSource::Activity;
  // event.name =  /* we use the API name instead*/
  //    se::wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  // TODO(rocm-profiler): CUDA uses device id and correlation ID for finding
  // annotations.
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);

  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;

  collector_->AddEvent(std::move(event), false);
}

void RocmActivityCallbackImpl::AddNormalHipMemcpyActivityEvent(
    const roctracer_record_t* record) {
  /*
  ---------------NormalMemcpy-------------------
    misses:context_id, memcpy_info.kind, memcpy_info.srckind,
  memcpy_info.dstkind, memcpy_info.num_bytes, memcpy_info.destenation,
  device_id, stream_id,

    extras: domain
  ---------------PeerMemcpy---------------------
    misses: device_id, context_id, stream_id, memcpy_info.kind,
      memcpy_info.num_bytes, memcpy_info.destination,
    extras:
      domain,
  */

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.source = RocmTracerEventSource::Activity;
  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);
  // TODO(roc-profiler): record->bytes is not a valid value
  // event.memcpy_info.num_bytes = record->bytes;
  event.name =
      se::wrap::roctracer_op_string(record->domain, record->op, record->kind);
  switch (record->op) {
    case HIP_API_ID_hipMemcpyDtoH:
    case HIP_API_ID_hipMemcpyDtoHAsync:
      event.type = RocmTracerEventType::MemcpyD2H;
      event.memcpy_info.async =
          (record->op == HIP_API_ID_hipMemcpyDtoHAsync) ? true : false;
      break;
    case HIP_API_ID_hipMemcpyHtoD:
    case HIP_API_ID_hipMemcpyHtoDAsync:
      event.type = RocmTracerEventType::MemcpyH2D;
      event.memcpy_info.async =
          (record->op == HIP_API_ID_hipMemcpyHtoDAsync) ? true : false;
      break;
    case HIP_API_ID_hipMemcpyDtoD:
    case HIP_API_ID_hipMemcpyDtoDAsync:
      event.type = RocmTracerEventType::MemcpyD2D;
      event.memcpy_info.async =
          (record->op == HIP_API_ID_hipMemcpyDtoDAsync) ? true : false;
      break;
    case HIP_API_ID_hipMemcpy:
    case HIP_API_ID_hipMemcpyAsync:
      event.type = RocmTracerEventType::MemcpyOther;
      event.memcpy_info.async =
          (record->op == HIP_API_ID_hipMemcpyAsync) ? true : false;
      break;
    case HIP_API_ID_hipMemcpyPeer:
    case HIP_API_ID_hipMemcpyPeerAsync:
      event.type = RocmTracerEventType::MemcpyP2P;
      event.memcpy_info.async =
          (record->op == HIP_API_ID_hipMemcpyPeerAsync) ? true : false;
      break;
    default:
      LOG(WARNING) << "Unsupported Memcpy/MemcpyPeer activity for profiling "
                      "observed for cbid="
                   << record->op << ". Event dropped!";
      return;
      break;
  }

  collector_->AddEvent(std::move(event), false);
}

void RocmActivityCallbackImpl::AddHipMemsetActivityEvent(
    const roctracer_record_t* record) {
  /*
    misses:
      device_id, context_id, stram_id, memset_info.num_bytes
      memset_info.kind

    extras:
      domain, annotation
  */

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.source = RocmTracerEventSource::Activity;
  event.name =
      se::wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);

  event.type = RocmTracerEventType::Memset;

  switch (record->op) {
    case HIP_API_ID_hipMemset:
      event.memset_info.async = false;
      break;
    case HIP_API_ID_hipMemsetAsync:
      event.memset_info.async = true;
      break;
    case HIP_API_ID_hipMemsetD8:
      event.memset_info.async = false;
      break;
    case HIP_API_ID_hipMemsetD8Async:
      event.memset_info.async = true;
      break;
    case HIP_API_ID_hipMemsetD16:
      event.memset_info.async = false;
      break;
    case HIP_API_ID_hipMemsetD16Async:
      event.memset_info.async = true;
      break;
    case HIP_API_ID_hipMemsetD32:
      event.memset_info.async = false;
      break;
    case HIP_API_ID_hipMemsetD32Async:
      event.memset_info.async = true;
      break;
  }

  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;

  collector_->AddEvent(std::move(event), false);
}

void RocmActivityCallbackImpl::AddHipMallocActivityEvent(
    const roctracer_record_t* record) {
  /*
    misses: device_id, context_id, memory_residency_info (num_byts, kind,
    address)

    extras:
      annotation, domain,
  */

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = RocmTracerEventType::MemoryAlloc;
  event.source = RocmTracerEventSource::Activity;
  event.name =
      se::wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);
  // similar to CUDA we set this to the default stream
  event.stream_id = 0;
  event.start_time_ns = record->begin_ns;
  // making sure it does not have 0ns duration. Otherwise, it may not show up in
  // the trace view
  event.end_time_ns = std::max(record->end_ns, record->begin_ns + 1);

  collector_->AddEvent(std::move(event), false);
}

void RocmActivityCallbackImpl::AddHipStreamSynchronizeActivityEvent(
    const roctracer_record_t* record) {
  /*
  misses: context_id, device_id (cuda also does not provide but we can get from
  API-CB)

  extras: domain, synchronization_info.sync_type, annotation
  */

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = RocmTracerEventType::Synchronization;
  event.source = RocmTracerEventSource::Activity;
  event.name =
      se::wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);
  event.start_time_ns = record->begin_ns;

  // making sure it does not have 0ns duration. Otherwise, it may not show up in
  // the trace view
  event.end_time_ns = std::max(record->end_ns, record->begin_ns + 1);

  switch (record->op) {
    case HIP_API_ID_hipStreamSynchronize:
      event.synchronization_info.sync_type =
          RocmTracerSyncTypes::StreamSynchronize;
      break;
    case HIP_API_ID_hipStreamWaitEvent:
      event.synchronization_info.sync_type = RocmTracerSyncTypes::StreamWait;
      break;
    default:
      event.synchronization_info.sync_type = RocmTracerSyncTypes::InvalidSync;
      break;
  }
  collector_->AddEvent(std::move(event), false);
}

// TODO(rocm-profiler): rename this function. this is HIP-OP
void RocmActivityCallbackImpl::AddHccKernelActivityEvent(
    const roctracer_record_t* record) {
  /*
   missing:
     name, context_id, nvtx_range, kernel_info

   extra:
     domain (thread id from the HIP activity)

   activity record contains device/stream ID
 */
  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_OPS;
  event.type = RocmTracerEventType::Kernel;
  event.source = RocmTracerEventSource::Activity;
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);
  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;
  event.device_id = record->device_id;
  event.stream_id = record->queue_id;

  collector_->AddEvent(std::move(event), false);
}

void RocmActivityCallbackImpl::AddNormalHipOpsMemcpyActivityEvent(
    const roctracer_record_t* record) {
  /*
    misses:
      type, name(the name set here is not clear enough but we keep it for
    debug), context_id, memcpy_info.kind, memcpy_info.num_bytes,
    memcpy_info.async, memcpy_info.src_mem_kind, memcpy_info.dst_mem_kind

    extras:
      domain,

  */

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_OPS;
  event.source = RocmTracerEventSource::Activity;
  event.name =  // name is stored for debug
      se::wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);

  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;
  event.device_id = record->device_id;
  event.memcpy_info.destination = event.device_id;
  event.stream_id = record->queue_id;

  // we set the type as MemcpyOther as HIP-OPS activity record does not carry
  // this information
  event.type = RocmTracerEventType::MemcpyOther;

  collector_->AddEvent(std::move(event), false);
}

void RocmActivityCallbackImpl::AddHipOpsMemsetActivityEvent(
    const roctracer_record_t* record) {
  /*
    misses:
      name (name recorder here is not clear enough for Memset. We only capture
    it for debug), context_id, memset_info.kind, memset_info.num_bytes,
    memset_info.async

    extras:
      dommain, annotation,

  */

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_OPS;
  event.source = RocmTracerEventSource::Activity;
  event.name =  // name is stored for debug
      se::wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);

  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;
  event.device_id = record->device_id;
  event.stream_id = record->queue_id;

  event.type = RocmTracerEventType::Memset;

  collector_->AddEvent(std::move(event), false);
}

/* static */ RocmTracer* RocmTracer::GetRocmTracerSingleton() {
  static auto* singleton = new RocmTracer();
  return singleton;
}

// FIXME(rocm-profiler): we should also check if we have AMD GPUs
bool RocmTracer::IsAvailable() const {
  return !activity_tracing_enabled_ && !api_tracing_enabled_;  // &&NumGpus()
}

int RocmTracer::NumGpus() {
  static int num_gpus = []() -> int {
    if (hipInit(0) != hipSuccess) {
      return 0;
    }
    int gpu_count;
    if (hipGetDeviceCount(&gpu_count) != hipSuccess) {
      return 0;
    }
    LOG(INFO) << "Profiler found " << gpu_count << " GPUs";
    return gpu_count;
  }();
  return num_gpus;
}

void RocmTracer::Enable(const RocmTracerOptions& options,
                        RocmTraceCollector* collector) {
  options_ = options;
  collector_ = collector;
  api_cb_impl_ = new RocmApiCallbackImpl(options, this, collector);
  activity_cb_impl_ = new RocmActivityCallbackImpl(options, this, collector);

  // From ROCm 3.5 onwards, the following call is required.
  // don't quite know what it does (no documentation!), only that without it
  // the call to enable api/activity tracing will run into a segfault
  se::wrap::roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, nullptr);

  EnableApiTracing().IgnoreError();
  EnableActivityTracing().IgnoreError();
  LOG(INFO) << "GpuTracer started";
}

void RocmTracer::Disable() {
  // TODO(rocm-profiler): TF has a SyncAndFlush() function
  // to be called before disabling. It makes sure all the contexts
  // has finished all the tasks before shutting down the profiler
  DisableApiTracing().IgnoreError();
  DisableActivityTracing().IgnoreError();
  delete api_cb_impl_;
  delete activity_cb_impl_;
  collector_->Flush();
  collector_ = nullptr;
  options_.reset();
  LOG(INFO) << "GpuTracer stopped";
}

void ApiCallback(uint32_t domain, uint32_t cbid, const void* cbdata,
                 void* user_data) {
  RocmTracer* tracer = reinterpret_cast<RocmTracer*>(user_data);
  tracer->ApiCallbackHandler(domain, cbid, cbdata).IgnoreError();
}

absl::Status RocmTracer::ApiCallbackHandler(uint32_t domain, uint32_t cbid,
                                            const void* cbdata) {
  if (api_tracing_enabled_)
    TF_RETURN_IF_ERROR((*api_cb_impl_)(domain, cbid, cbdata));
  return tsl::OkStatus();
}

absl::Status RocmTracer::EnableApiTracing() {
  if (api_tracing_enabled_) return tsl::OkStatus();
  api_tracing_enabled_ = true;

  for (auto& iter : options_->api_callbacks) {
    activity_domain_t domain = iter.first;
    std::vector<uint32_t>& ops = iter.second;
    if (ops.size() == 0) {
      VLOG(3) << "Enabling API tracing for domain "
              << GetActivityDomainName(domain);
      RETURN_IF_ROCTRACER_ERROR(se::wrap::roctracer_enable_domain_callback(
          domain, ApiCallback, this));
    } else {
      VLOG(3) << "Enabling API tracing for " << ops.size() << " ops in domain "
              << GetActivityDomainName(domain);
      for (auto& op : ops) {
        VLOG(3) << "Enabling API tracing for "
                << GetActivityDomainOpName(domain, op);
        RETURN_IF_ROCTRACER_ERROR(se::wrap::roctracer_enable_op_callback(
            domain, op, ApiCallback, this));
      }
    }
  }
  return tsl::OkStatus();
}

absl::Status RocmTracer::DisableApiTracing() {
  if (!api_tracing_enabled_) return tsl::OkStatus();
  api_tracing_enabled_ = false;

  for (auto& iter : options_->api_callbacks) {
    activity_domain_t domain = iter.first;
    std::vector<uint32_t>& ops = iter.second;
    if (ops.size() == 0) {
      VLOG(3) << "Disabling API tracing for domain "
              << GetActivityDomainName(domain);
      RETURN_IF_ROCTRACER_ERROR(
          se::wrap::roctracer_disable_domain_callback(domain));
    } else {
      VLOG(3) << "Disabling API tracing for " << ops.size() << " ops in domain "
              << GetActivityDomainName(domain);
      for (auto& op : ops) {
        VLOG(3) << "Disabling API tracing for "
                << GetActivityDomainOpName(domain, op);
        RETURN_IF_ROCTRACER_ERROR(
            se::wrap::roctracer_disable_op_callback(domain, op));
      }
    }
  }
  return tsl::OkStatus();
}

void ActivityCallback(const char* begin, const char* end, void* user_data) {
  RocmTracer* tracer = reinterpret_cast<RocmTracer*>(user_data);
  tracer->ActivityCallbackHandler(begin, end).IgnoreError();
}

absl::Status RocmTracer::ActivityCallbackHandler(const char* begin,
                                                 const char* end) {
  if (activity_tracing_enabled_) {
    TF_RETURN_IF_ERROR((*activity_cb_impl_)(begin, end));
  } else {
    LOG(WARNING) << "ActivityCallbackHandler called when "
                    "activity_tracing_enabled_ is false";

    VLOG(3) << "Dropped Activity Records Start";
    const roctracer_record_t* record =
        reinterpret_cast<const roctracer_record_t*>(begin);
    const roctracer_record_t* end_record =
        reinterpret_cast<const roctracer_record_t*>(end);
    while (record < end_record) {
      DumpActivityRecord(record,
                         "activity_tracing_enabled_ is false. Dropped!");
#if TF_ROCM_VERSION >= 50300
      RETURN_IF_ROCTRACER_ERROR(static_cast<roctracer_status_t>(
          se::wrap::roctracer_next_record(record, &record)));
#else
      RETURN_IF_ROCTRACER_ERROR(static_cast<roctracer_status_t>(
          roctracer_next_record(record, &record)));
#endif
    }
    VLOG(3) << "Dropped Activity Records End";
  }
  return tsl::OkStatus();
}

absl::Status RocmTracer::EnableActivityTracing() {
  if (activity_tracing_enabled_) return tsl::OkStatus();
  activity_tracing_enabled_ = true;

  if (!options_->activity_tracing.empty()) {
    // Create the memory pool to store activity records in
    if (se::wrap::roctracer_default_pool_expl(nullptr) == NULL) {
      roctracer_properties_t properties{};
      properties.buffer_size = 0x1000;
      properties.buffer_callback_fun = ActivityCallback;
      properties.buffer_callback_arg = this;
      VLOG(3) << "Creating roctracer activity buffer: buff-size="
              << properties.buffer_size;
      RETURN_IF_ROCTRACER_ERROR(
          se::wrap::roctracer_open_pool_expl(&properties, nullptr));
    }
  }

  for (auto& iter : options_->activity_tracing) {
    activity_domain_t domain = iter.first;
    std::vector<uint32_t>& ops = iter.second;
    if (ops.size() == 0) {
      VLOG(3) << "Enabling Activity tracing for domain "
              << GetActivityDomainName(domain);
      RETURN_IF_ROCTRACER_ERROR(
          se::wrap::roctracer_enable_domain_activity_expl(domain, nullptr));
    } else {
      VLOG(3) << "Enabling Activity tracing for " << ops.size()
              << " ops in domain " << GetActivityDomainName(domain);
      for (auto& op : ops) {
        VLOG(3) << "Enabling Activity tracing for "
                << GetActivityDomainOpName(domain, op);
        // roctracer library has not exported "roctracer_enable_op_activity"
        RETURN_IF_ROCTRACER_ERROR(
            se::wrap::roctracer_enable_op_activity_expl(domain, op, nullptr));
      }
    }
  }

  return tsl::OkStatus();
}

absl::Status RocmTracer::DisableActivityTracing() {
  if (!activity_tracing_enabled_) return tsl::OkStatus();

  for (auto& iter : options_->activity_tracing) {
    activity_domain_t domain = iter.first;
    std::vector<uint32_t>& ops = iter.second;
    if (ops.size() == 0) {
      VLOG(3) << "Disabling Activity tracing for domain "
              << GetActivityDomainName(domain);
      RETURN_IF_ROCTRACER_ERROR(
          se::wrap::roctracer_disable_domain_activity(domain));
    } else {
      VLOG(3) << "Disabling Activity tracing for " << ops.size()
              << " ops in domain " << GetActivityDomainName(domain);
      for (auto& op : ops) {
        VLOG(3) << "Disabling Activity tracing for "
                << GetActivityDomainOpName(domain, op);
        RETURN_IF_ROCTRACER_ERROR(
            se::wrap::roctracer_disable_op_activity(domain, op));
      }
    }
  }

  // TODO(rocm-profiler): this stopping mechanism needs improvement.
  // Flush the activity buffer BEFORE setting the activity_tracing_enable_
  // flag to FALSE. This is because the activity record callback routine is
  // gated by the same flag
  VLOG(3) << "Flushing roctracer activity buffer";
  RETURN_IF_ROCTRACER_ERROR(se::wrap::roctracer_flush_activity_expl(nullptr));
  // roctracer_flush_buf();

  // Explicitly wait for (almost) all pending activity records
  // The choice of all of the following is based what seemed to work
  // best when enabling tracing on a large testcase (BERT)
  // * 100 ms as the initial sleep duration AND
  // * 1 as the initial threshold value
  // * 6 as the maximum number of iterations
  int duration_ms = 100;
  size_t threshold = 1;
  for (int i = 0; i < 6; i++, duration_ms *= 2, threshold *= 2) {
    if (GetPendingActivityRecordsCount() < threshold) break;
    VLOG(3) << "Wait for pending activity records :" << " Pending count = "
            << GetPendingActivityRecordsCount()
            << ", Threshold = " << threshold;
    VLOG(3) << "Wait for pending activity records : sleep for " << duration_ms
            << " ms";
    tsl::profiler::SleepForMillis(duration_ms);
  }
  ClearPendingActivityRecordsCount();

  activity_tracing_enabled_ = false;

  return tsl::OkStatus();
}

/*static*/ uint64_t RocmTracer::GetTimestamp() {
  uint64_t ts;
  if (se::wrap::roctracer_get_timestamp(&ts) != ROCTRACER_STATUS_SUCCESS) {
    const char* errstr = se::wrap::roctracer_error_string();
    LOG(ERROR) << "function roctracer_get_timestamp failed with error "
               << errstr;
    // Return 0 on error.
    return 0;
  }
  return ts;
}

}  // namespace profiler
}  // namespace xla

#else
// for rocprofiler-sdk
namespace xla {
namespace profiler {

using tsl::profiler::AnnotationStack;

// represents the maximum number of chars
static constexpr int kMaxSymbolSize = 1024;
// represents an invalid or uninitialized device ID used in RocmTracer events.
constexpr uint32_t RocmTracerEvent::kInvalidDeviceId;

std::string demangle(const char* name) {
#ifndef _MSC_VER
  if (!name) {
    return "";
  }

  if (strlen(name) > kMaxSymbolSize) {
    return name;
  }

  int status;
  size_t len = 0;
  char* demangled = abi::__cxa_demangle(name, nullptr, &len, &status);
  if (status != 0) {
    return name;
  }
  std::string res(demangled);
  // The returned buffer must be freed!
  free(demangled);
  return res;
#else
  // TODO: demangling on Windows
  if (!name) {
    return "";
  } else {
    return name;
  }
#endif
}

std::string demangle(const std::string& name) { return demangle(name.c_str()); }

inline auto GetCallbackTracingNames() {
  return rocprofiler::sdk::get_callback_tracing_names();
}

std::vector<rocprofiler_agent_v0_t> GetGpuDeviceAgents();

//-----------------------------------------------------------------------------
const char* GetRocmTracerEventSourceName(const RocmTracerEventSource& source) {
  switch (source) {
    case RocmTracerEventSource::ApiCallback:
      return "ApiCallback";
      break;
    case RocmTracerEventSource::Activity:
      return "Activity";
      break;
    case RocmTracerEventSource::Invalid:
      return "Invalid";
      break;
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

// FIXME(rocm-profiler): These domain names are not consistent with the
// GetActivityDomainName function
const char* GetRocmTracerEventDomainName(const RocmTracerEventDomain& domain) {
  switch (domain) {
    case RocmTracerEventDomain::HIP_API:
      return "HIP_API";
      break;
    case RocmTracerEventDomain::HIP_OPS:
      return "HIP_OPS";
      break;
    default:
      LOG(WARNING) << "RocmTracerEventDomain::InvalidDomain";
      DCHECK(false);
      return "";
  }
  return "";
}

const char* GetRocmTracerEventTypeName(const RocmTracerEventType& type) {
#define OO(x)                  \
  case RocmTracerEventType::x: \
    return #x;
  switch (type) {
    OO(Kernel)
    OO(MemcpyH2D)
    OO(MemcpyD2H)
    OO(MemcpyD2D)
    OO(MemcpyOther)
    OO(MemoryAlloc)
    OO(MemoryFree)
    OO(Memset)
    OO(Synchronization)
    OO(Generic)
    default:;
  }
#undef OO
  DCHECK(false);
  return "";
}

//-----------------------------------------------------------------------------
// copy api calls
bool isCopyApi(uint32_t id) {
  switch (id) {
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2D:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DFromArray:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DFromArrayAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DToArray:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DToArrayAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy3D:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy3DAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAtoH:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoD:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoDAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoH:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoHAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromArray:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbol:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbolAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoA:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoD:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoDAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyParam2D:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyParam2DAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeer:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeerAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToArray:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbol:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbolAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyWithStream:
      return true;
      break;
    default:;
  }
  return false;
}

// ----------------------------------------------------------------------------
// Stub implementations for RocmTracer static functions expected by
// rocprofiler-sdk.
// ----------------------------------------------------------------------------
RocmTracer& RocmTracer::i() {
  static RocmTracer obj;
  return obj;
}

bool RocmTracer::IsAvailable() const {
  return !activity_tracing_enabled_ && !api_tracing_enabled_;  // &&NumGpus()
}

/*static*/ uint64_t RocmTracer::GetTimestamp() {
  uint64_t ts;
  if (rocprofiler_get_timestamp(&ts) != ROCPROFILER_STATUS_SUCCESS) {
    LOG(ERROR) << "function rocprofiler_get_timestamp failed with error ";
    return 0;
  }
  return ts;
}

void RocmTracer::Enable(const RocmTracerOptions& options,
                        RocmTraceCollector* collector) {
  absl::MutexLock lock(&collector_mutex_);
  if (collector_ != nullptr) {
    LOG(WARNING) << "ROCM tracer is already running!";
    return;
  }
  options_ = options;
  collector_ = collector;
  AnnotationMap(options_->max_annotation_strings);
  api_tracing_enabled_ = true;
  activity_tracing_enabled_ = true;
  rocprofiler_start_context(context_);
  LOG(INFO) << "GpuTracer started with number of GPUs" << NumGpus();
}

void RocmTracer::HipApiEvent(const rocprofiler_record_header_t* hdr,
                             RocmTracerEvent* traced_event) {
  const auto& rec =
      *static_cast<const rocprofiler_buffer_tracing_hip_api_record_t*>(
          hdr->payload);

  traced_event->type = RocmTracerEventType::Kernel;
  traced_event->source = RocmTracerEventSource::ApiCallback;
  traced_event->domain = RocmTracerEventDomain::HIP_API;
  traced_event->name = "??";
  traced_event->start_time_ns = rec.start_timestamp;
  traced_event->end_time_ns = rec.end_timestamp;
  traced_event->device_id = RocmTracerEvent::kInvalidDeviceId;
  traced_event->correlation_id = rec.correlation_id.internal;
  traced_event->annotation =
      annotation_map()->LookUp(traced_event->correlation_id);
  traced_event->thread_id = rec.thread_id;
  traced_event->stream_id = RocmTracerEvent::kInvalidStreamId;
  traced_event->kernel_info = KernelDetails{};

  absl::MutexLock lock(&kernel_lock_);
  if (static_cast<size_t>(rec.kind) < name_info_.size()) {
    auto& vec = name_info_[rec.kind];
    traced_event->name = vec[rec.operation];
  }
  if (isCopyApi(rec.operation)) {
    // actually one needs to set the real type
    traced_event->type = RocmTracerEventType::MemcpyOther;
  }
}

void RocmTracer::MemcpyEvent(const rocprofiler_record_header_t* hdr,
                             RocmTracerEvent* traced_event) {
  const auto& rec =
      *static_cast<const rocprofiler_buffer_tracing_memory_copy_record_t*>(
          hdr->payload);

#define OO(src, target)                               \
  case ROCPROFILER_MEMORY_COPY_##src:                 \
    traced_event->type = RocmTracerEventType::target; \
    traced_event->name = #target;                     \
    break;

  switch (rec.operation) {
    OO(NONE, MemcpyOther)
    OO(HOST_TO_HOST, MemcpyOther)
    OO(HOST_TO_DEVICE, MemcpyH2D)
    OO(DEVICE_TO_HOST, MemcpyD2H)
    OO(DEVICE_TO_DEVICE, MemcpyD2D)
    default:
      LOG(WARNING) << "Unexpected memcopy operation " << rec.operation;
      traced_event->type = RocmTracerEventType::MemcpyOther;
  }
#undef OO
  const auto &src_gpu = agents_[static_cast<uint32_t>(rec.src_agent_id.handle)],
             &dst_gpu = agents_[static_cast<uint32_t>(rec.dst_agent_id.handle)];

  // Assign device_id based on copy direction
  if (traced_event->type == RocmTracerEventType::MemcpyH2D &&
      dst_gpu.type == ROCPROFILER_AGENT_TYPE_GPU) {
    traced_event->device_id = dst_gpu.id.handle;  // Destination is GPU
  } else if (traced_event->type == RocmTracerEventType::MemcpyD2H &&
             src_gpu.type == ROCPROFILER_AGENT_TYPE_GPU) {
    traced_event->device_id = src_gpu.id.handle;  // Source is GPU
  } else if (traced_event->type == RocmTracerEventType::MemcpyD2D) {
    // Prefer destination GPU for D2D
    traced_event->device_id = dst_gpu.id.handle;
  } else {
    // Fallback for MemcpyOther or HOST_TO_HOST
    if (dst_gpu.type == ROCPROFILER_AGENT_TYPE_GPU) {
      traced_event->device_id = dst_gpu.id.handle;
    } else if (src_gpu.type == ROCPROFILER_AGENT_TYPE_GPU) {
      traced_event->device_id = src_gpu.id.handle;
    } else {
      LOG(WARNING) << "No GPU ID available for memory copy operation: "
                   << traced_event->name << ", src_agent_type=" << src_gpu.type
                   << ", dst_agent_type=" << dst_gpu.type;
      traced_event->device_id = 0;  // Invalid ID or default
    }
  }

  traced_event->source = RocmTracerEventSource::Activity;
  traced_event->domain = RocmTracerEventDomain::HIP_OPS;
  traced_event->start_time_ns = rec.start_timestamp;
  traced_event->end_time_ns = rec.end_timestamp;
  traced_event->correlation_id = rec.correlation_id.internal;
  traced_event->annotation =
      annotation_map()->LookUp(traced_event->correlation_id);
  traced_event->thread_id = rec.thread_id;
  // we do not know valid stream ID for memcpy
  // rec.stream_id.handle;
  traced_event->stream_id = RocmTracerEvent::kInvalidStreamId;
  traced_event->memcpy_info = MemcpyDetails{
      .num_bytes = rec.bytes,
      .destination = static_cast<uint32_t>(dst_gpu.id.handle),
      .async = false,
  };

  LOG(INFO) << "copy bytes: " << traced_event->memcpy_info.num_bytes
            << " stream: " << traced_event->stream_id << " src_id "
            << traced_event->device_id << " dst_id "
            << traced_event->memcpy_info.destination;
}

void RocmTracer::KernelEvent(const rocprofiler_record_header_t* hdr,
                             RocmTracerEvent* traced_event) {
  const auto& rec =
      *static_cast<const rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(
          hdr->payload);

  const auto& kinfo = rec.dispatch_info;
  traced_event->type = RocmTracerEventType::Kernel;
  traced_event->source = RocmTracerEventSource::Activity;
  traced_event->domain = RocmTracerEventDomain::HIP_OPS;
  traced_event->name = "??";
  traced_event->start_time_ns = rec.start_timestamp;
  traced_event->end_time_ns = rec.end_timestamp;
  traced_event->device_id = agents_[kinfo.agent_id.handle].id.handle;
  traced_event->correlation_id = rec.correlation_id.internal;
  traced_event->annotation =
      annotation_map()->LookUp(traced_event->correlation_id);
  traced_event->thread_id = rec.thread_id;
  traced_event->stream_id = kinfo.queue_id.handle;
  traced_event->kernel_info = KernelDetails{
      .registers_per_thread = 0,
      .static_shared_memory_usage = 0,
      .dynamic_shared_memory_usage = 0,
      .block_x = kinfo.workgroup_size.x,
      .block_y = kinfo.workgroup_size.y,
      .block_z = kinfo.workgroup_size.z,
      .grid_x = kinfo.grid_size.x,
      .grid_y = kinfo.grid_size.y,
      .grid_z = kinfo.grid_size.z,
      .func_ptr = nullptr,
  };

  auto it = kernel_info_.find(kinfo.kernel_id);
  if (it != kernel_info_.end()) traced_event->name = it->second.name;
}

void RocmTracer::TracingCallback(rocprofiler_context_id_t context,
                                 rocprofiler_buffer_id_t buffer_id,
                                 rocprofiler_record_header_t** headers,
                                 size_t num_headers, uint64_t drop_count) {
  if (collector() == nullptr) return;
  if (num_headers == 0) return;
  assert(drop_count == 0 && "drop count should be zero for lossless policy");

  if (headers == nullptr) {
    LOG(ERROR)
        << "rocprofiler invoked a buffer callback with a null pointer to the "
           "array of headers. this should never happen";
    return;
  }

  for (size_t i = 0; i < num_headers; i++) {
    RocmTracerEvent event;
    auto header = headers[i];

    if (header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) continue;

    switch (header->kind) {
      case ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API:
        HipApiEvent(header, &event);
        break;

      case ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH:
        KernelEvent(header, &event);
        break;

      case ROCPROFILER_BUFFER_TRACING_MEMORY_COPY:
        MemcpyEvent(header, &event);
        break;

      default:
        continue;
    }  // switch

    absl::MutexLock lock(&collector_mutex_);
    if (collector()) {
      collector()->AddEvent(std::move(event), false);
    }
  }  // for
}

void RocmTracer::CodeObjectCallback(
    rocprofiler_callback_tracing_record_t record, void* callback_data) {
  if (record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
      record.operation == ROCPROFILER_CODE_OBJECT_LOAD) {
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD) {
      // mainly for debugging
      LOG(WARNING)
          << "Callback phase unload without registering kernel names ...";
    }
  } else if (record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
             record.operation ==
                 ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) {
    auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD) {
      absl::MutexLock lock(&kernel_lock_);
      kernel_info_.emplace(
          data->kernel_id,
          ProfilerKernelInfo{demangle(data->kernel_name), *data});
    } else if (record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD) {
      // FIXME: clear these?  At minimum need kernel names at shutdown, async
      // completion We don't erase it just in case a buffer callback still needs
      // this kernel_info_.erase(data->kernel_id);
    }
  }
}

static void code_object_callback(rocprofiler_callback_tracing_record_t record,
                                 rocprofiler_user_data_t* user_data,
                                 void* callback_data) {
  RocmTracer::i().CodeObjectCallback(record, callback_data);
}

static void tool_tracing_callback(rocprofiler_context_id_t context,
                                  rocprofiler_buffer_id_t buffer_id,
                                  rocprofiler_record_header_t** headers,
                                  size_t num_headers, void* user_data,
                                  uint64_t drop_count) {
  RocmTracer::i().TracingCallback(context, buffer_id, headers, num_headers,
                                  drop_count);
}

int RocmTracer::toolInit(rocprofiler_client_finalize_t fini_func,
                         void* tool_data) {
  // Gather API names
  name_info_ = GetCallbackTracingNames();

  // Gather agent info
  num_gpus_ = 0;
  for (const auto& agent : GetGpuDeviceAgents()) {
    LOG(INFO) << "agent id = " << agent.id.handle
              << ", dev = " << agent.device_id
              << ", name = " << (agent.name ? agent.name : "null");
    agents_[agent.id.handle] = agent;
    if (agent.type == ROCPROFILER_AGENT_TYPE_GPU) {
      num_gpus_++;
    }
  }

  // Utility context to gather code‑object info
  rocprofiler_create_context(&utility_context_);

  // buffered tracing
  auto code_object_ops = std::vector<rocprofiler_tracing_operation_t>{
      ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};

  rocprofiler_configure_callback_tracing_service(
      utility_context_, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
      code_object_ops.data(), code_object_ops.size(), code_object_callback,
      nullptr);

  rocprofiler_start_context(utility_context_);
  LOG(INFO) << "rocprofiler start utilityContext";

  // a multiple of the page size, and the gap allows the buffer to absorb bursts
  // of GPU events
  constexpr auto buffer_size_bytes = 20 * 4096;
  constexpr auto buffer_watermark_bytes = 2 * 4096;

  // Utility context to gather code‑object info
  rocprofiler_create_context(&context_);

  rocprofiler_create_buffer(context_, buffer_size_bytes, buffer_watermark_bytes,
                            ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                            tool_tracing_callback, tool_data, &buffer_);

  rocprofiler_configure_buffer_tracing_service(
      context_, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, nullptr, 0,
      buffer_);

  rocprofiler_configure_buffer_tracing_service(
      context_, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0,
      buffer_);

  rocprofiler_configure_buffer_tracing_service(
      context_, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, nullptr, 0, buffer_);

  {
    // for annotations
    const rocprofiler_tracing_operation_t* hip_ops = nullptr;
    size_t hip_ops_count = 0;

    rocprofiler_configure_callback_tracing_service(
        context_, ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API, hip_ops,
        hip_ops_count,
        [](rocprofiler_callback_tracing_record_t record,
           rocprofiler_user_data_t*, void*) {
          if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
            const std::string& annotation =
                tsl::profiler::AnnotationStack::Get();
            if (!annotation.empty()) {
              RocmTracer::i().annotation_map()->Add(
                  record.correlation_id.internal, annotation);
            }
          }
        },
        nullptr);
  }

  auto client_thread = rocprofiler_callback_thread_t{};
  rocprofiler_create_callback_thread(&client_thread);
  rocprofiler_assign_callback_thread(buffer_, client_thread);

  int isValid = 0;
  rocprofiler_context_is_valid(context_, &isValid);
  if (isValid == 0) {
    context_.handle = 0;  // Leak on failure.
    return -1;
  }

  return 0;
}

void RocmTracer::toolFinalize(void* tool_data) {
  auto& obj = RocmTracer::i();
  LOG(INFO) << "Calling toolFinalize!";
  rocprofiler_stop_context(obj.utility_context_);
  obj.utility_context_.handle = 0;
  rocprofiler_stop_context(obj.context_);
  // flush buffer here or in disable?
  obj.context_.handle = 0;
}

void RocmTracer::Disable() {
  absl::MutexLock lock(&collector_mutex_);
  collector_->Flush();
  collector_ = nullptr;
  api_tracing_enabled_ = false;
  activity_tracing_enabled_ = false;
  LOG(INFO) << "GpuTracer stopped";
}

// ----------------------------------------------------------------------------
// Helper that returns all device agents (GPU + CPU for now).
// ----------------------------------------------------------------------------
std::vector<rocprofiler_agent_v0_t> GetGpuDeviceAgents() {
  std::vector<rocprofiler_agent_v0_t> agents;

  rocprofiler_query_available_agents_cb_t iterate_cb =
      [](rocprofiler_agent_version_t agents_ver, const void** agents_arr,
         size_t num_agents, void* udata) {
        if (agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0) {
          LOG(ERROR) << "unexpected rocprofiler agent version: " << agents_ver;
          return ROCPROFILER_STATUS_ERROR;
        }
        auto* agents_vec =
            static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
        for (size_t i = 0; i < num_agents; ++i) {
          const auto* agent =
              static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
          agents_vec->push_back(*agent);
        }
        return ROCPROFILER_STATUS_SUCCESS;
      };

  rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                     iterate_cb, sizeof(rocprofiler_agent_t),
                                     static_cast<void*>(&agents));
  return agents;
}

static int toolInitStatic(rocprofiler_client_finalize_t finalize_func,
                          void* tool_data) {
  return RocmTracer::i().toolInit(finalize_func, tool_data);
}

// ----------------------------------------------------------------------------
// C‑linkage entry‑point expected by rocprofiler-sdk.
// ----------------------------------------------------------------------------
extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(
    uint32_t version, const char* runtime_version, uint32_t priority,
    rocprofiler_client_id_t* id) {
  auto& obj = RocmTracer::i();  // Ensure constructed, critical for tracing.

  id->name = "XLA-with-rocprofiler-sdk";
  obj.client_id_ = id;

  LOG(INFO) << "Configure rocprofiler-sdk...";

  const uint32_t major = version / 10000;
  const uint32_t minor = (version % 10000) / 100;
  const uint32_t patch = version % 100;

  std::stringstream info;
  info << id->name << " Configure XLA with rocprofv3... (priority=" << priority
       << ") is using rocprofiler-sdk v" << major << '.' << minor << '.'
       << patch << " (" << runtime_version << ')';
  LOG(INFO) << info.str();

  static rocprofiler_tool_configure_result_t cfg{
      sizeof(rocprofiler_tool_configure_result_t), &toolInitStatic,
      &RocmTracer::toolFinalize, nullptr};

  return &cfg;
}

}  // namespace profiler
}  // namespace xla

void __attribute__((constructor)) init_rocm_lib() {
  rocprofiler_force_configure(xla::profiler::rocprofiler_configure);
}

#endif  //