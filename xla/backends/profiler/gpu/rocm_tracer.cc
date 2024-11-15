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

#include "xla/backends/profiler/gpu/common/call_stack.hpp"
#include "xla/backends/profiler/gpu/common/defines.hpp"
#include "xla/backends/profiler/gpu/common/filesystem.hpp"
#include "xla/backends/profiler/gpu/common/name_info.hpp"

#include "xla/backends/profiler/gpu/rocm_tracer.h"
#include "xla/backends/profiler/gpu/rocm_collector.h"
#include "xla/stream_executor/rocm/roctracer_wrapper.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "rocm/rocm_config.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/mem.h"
#include "tsl/profiler/backends/cpu/annotation_stack.h"
#include "tsl/profiler/utils/time_utils.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(
  uint32_t version, const char* runtime_version, uint32_t priority,
  rocprofiler_client_id_t* id
);

template <typename Tp = std::string_view>
using buffer_name_info_t = rocprofiler::sdk::utility::name_info<rocprofiler_buffer_tracing_kind_t, Tp>;

namespace se = ::stream_executor;

namespace xla {
namespace profiler {
namespace {
using xla::common::buffer_name_info;
using xla::common::call_stack_t;
using xla::common::source_location;

using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;

rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;
rocprofiler_context_id_t      client_ctx       = {};
rocprofiler_buffer_id_t       client_buffer    = {};
buffer_name_info              client_name_info = {};
kernel_symbol_map_t           client_kernels   = {};

void
print_call_stack(const call_stack_t& _call_stack)
{
    LOG(ERROR) << "print out call stack...";
    common::print_call_stack("api_buffered_trace.log", _call_stack);
    LOG(ERROR) << "complete print out call stack...";
}

void
tool_code_object_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t*              user_data,
                          void*                                 callback_data)
{
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            // flush the buffer to ensure that any lookups for the client kernel names for the code
            // object are completed
            auto flush_status = se::wrap::rocprofiler_flush_buffer(client_buffer);
            if(flush_status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
                ROCPROFILER_CALL(flush_status, "buffer flush");
        }
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
            record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            client_kernels.emplace(data->kernel_id, *data);
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            client_kernels.erase(data->kernel_id);
        }
    }

    (void) user_data;
    (void) callback_data;
}

template <typename Tp>
inline buffer_name_info_t<Tp>
rocm_get_buffer_tracing_names()
{
    auto cb_name_info = buffer_name_info_t<Tp>{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb = [](rocprofiler_buffer_tracing_kind_t kindv,
                                               rocprofiler_tracing_operation_t   operation,
                                               void*                             data_v) {
        auto* name_info_v = static_cast<buffer_name_info_t<Tp>*>(data_v);

        const char* name = nullptr;
        auto        status =
            se::wrap::rocprofiler_query_buffer_tracing_kind_operation_name(kindv, operation, &name, nullptr);
        if(status == rocprofiler::sdk::success_v && name) name_info_v->emplace(kindv, operation, name);
        return 0;
    };

    //
    //  callback for each buffer kind (i.e. domain)
    //
    static auto tracing_kind_cb = [](rocprofiler_buffer_tracing_kind_t kind, void* data) {
        //  store the buffer kind name
        auto*       name_info_v = static_cast<buffer_name_info_t<Tp>*>(data);
        const char* name        = nullptr;
        auto        status      = se::wrap::rocprofiler_query_buffer_tracing_kind_name(kind, &name, nullptr);
        if(status == rocprofiler::sdk::success_v && name) name_info_v->emplace(kind, name);

        se::wrap::rocprofiler_iterate_buffer_tracing_kind_operations(kind, tracing_kind_operation_cb, data);
        return 0;
    };

    se::wrap::rocprofiler_iterate_buffer_tracing_kinds(tracing_kind_cb, &cb_name_info);

    return cb_name_info;
}

void
tool_tracing_callback(rocprofiler_context_id_t      context,
                      rocprofiler_buffer_id_t       buffer_id,
                      rocprofiler_record_header_t** headers,
                      size_t                        num_headers,
                      void*                         user_data,
                      uint64_t                      drop_count)
{
    assert(user_data != nullptr);
    assert(drop_count == 0 && "drop count should be zero for lossless policy");

   /*
    if(num_headers == 0)
        throw std::runtime_error{
            "rocprofiler invoked a buffer callback with no headers. this should never happen"};
    else if(headers == nullptr)
        throw std::runtime_error{"rocprofiler invoked a buffer callback with a null pointer to the "
                                 "array of headers. this should never happen"};
    */

    auto* collector_impl = static_cast<RocmTraceCollector*>(user_data);
    LOG(ERROR) << "Number of gpus = " << &collector_impl;


    LOG(INFO) << "Number of heads = " << num_headers;
    LOG(INFO) << "Tracing category = " << ROCPROFILER_BUFFER_CATEGORY_TRACING;
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        RocmTracerEvent event;

        auto kind_name = std::string{};
        LOG(INFO) << "head category = " << header->category;
        LOG(INFO) << "head kind = " << header->kind;

        if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING)
        {
            const char* _name = nullptr;
            auto        _kind = static_cast<rocprofiler_buffer_tracing_kind_t>(header->kind);
            ROCPROFILER_CALL(se::wrap::rocprofiler_query_buffer_tracing_kind_name(_kind, &_name, nullptr),
                             "query buffer tracing kind name");
            if(_name)
            {
                static size_t len = 15;

                kind_name = std::string{_name};
                len       = std::max(len, kind_name.length());
                kind_name.resize(len, ' ');
                kind_name += " :: ";
                LOG(INFO) << "kind name = " << kind_name;
            }
        }

        if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(header->payload);

            event.type = RocmTracerEventType::HIP_RUNTIME_API;
            event.start_time_ns = record->start_timestamp;
            event.end_time_ns = record->end_timestamp;
            // event.device_id = record->dispatch_info.agent_id.handle;;
            // event.stream_id = record->stream_id;
            event.correlation_id = record->correlation_id.internal;
            event.name = client_name_info[record->kind][record->operation];

            auto info = std::stringstream{};
            info << "tid=" << record->thread_id << ", context=" << context.handle
                 << ", buffer_id=" << buffer_id.handle
                 << ", cid=" << record->correlation_id.internal
                 << ", extern_cid=" << record->correlation_id.external.value
                 << ", kind=" << record->kind << ", operation=" << record->operation
                 << ", start=" << record->start_timestamp << ", stop=" << record->end_timestamp
                 << ", name=" << client_name_info[record->kind][record->operation];

            if(record->start_timestamp > record->end_timestamp)
            {
                auto msg = std::stringstream{};
                msg << "hip api: start > end (" << record->start_timestamp << " > "
                    << record->end_timestamp
                    << "). diff = " << (record->start_timestamp - record->end_timestamp);
                std::cerr << "threw an exception " << msg.str() << "\n" << std::flush;
                // throw std::runtime_error{msg.str()};
            }

            static_cast<call_stack_t*>(user_data)->emplace_back(
                source_location{__FUNCTION__, __FILE__, __LINE__, kind_name + info.str()});
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);
            
            event.type = RocmTracerEventType::KERNEL_DISPATCH;
            event.start_time_ns = record->start_timestamp;
            event.end_time_ns = record->end_timestamp;
            event.device_id = record->dispatch_info.agent_id.handle;
            event.stream_id = record->dispatch_info.queue_id.handle;
            event.correlation_id = record->correlation_id.internal;
            event.name = client_kernels.at(record->dispatch_info.kernel_id).kernel_name;

            auto info = std::stringstream{};

            info << "tid=" << record->thread_id << ", context=" << context.handle
                 << ", buffer_id=" << buffer_id.handle
                 << ", cid=" << record->correlation_id.internal
                 << ", extern_cid=" << record->correlation_id.external.value
                 << ", kind=" << record->kind << ", operation=" << record->operation
                 << ", agent_id=" << record->dispatch_info.agent_id.handle
                 << ", queue_id=" << record->dispatch_info.queue_id.handle
                 << ", kernel_id=" << record->dispatch_info.kernel_id
                 << ", kernel=" << client_kernels.at(record->dispatch_info.kernel_id).kernel_name
                 << ", kernel group segement size " << client_kernels.at(record->dispatch_info.kernel_id).group_segment_size
                 << ", kernel private segement size " << client_kernels.at(record->dispatch_info.kernel_id).private_segment_size
                 << ", kernel scalar general purpose register count " << client_kernels.at(record->dispatch_info.kernel_id).sgpr_count
                 << ", kernel arch_vpgr_count " << client_kernels.at(record->dispatch_info.kernel_id).arch_vgpr_count
                 << ", keernel accum_vpgr_count " << client_kernels.at(record->dispatch_info.kernel_id).accum_vgpr_count
                 << ", start=" << record->start_timestamp << ", stop=" << record->end_timestamp
                 << ", private_segment_size=" << record->dispatch_info.private_segment_size
                 << ", group_segment_size=" << record->dispatch_info.group_segment_size
                 << ", workgroup_size=(" << record->dispatch_info.workgroup_size.x << ","
                 << record->dispatch_info.workgroup_size.y << ","
                 << record->dispatch_info.workgroup_size.z << "), grid_size=("
                 << record->dispatch_info.grid_size.x << "," << record->dispatch_info.grid_size.y
                 << "," << record->dispatch_info.grid_size.z << ")";

            if(record->start_timestamp > record->end_timestamp)
                printf("kernel dispatch: start > end");
                // throw std::runtime_error("kernel dispatch: start > end");

            static_cast<call_stack_t*>(user_data)->emplace_back(
                source_location{__FUNCTION__, __FILE__, __LINE__, kind_name + info.str()});
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(header->payload);

            auto info = std::stringstream{};

            info << "tid=" << record->thread_id << ", context=" << context.handle
                 << ", buffer_id=" << buffer_id.handle
                 << ", cid=" << record->correlation_id.internal
                 << ", extern_cid=" << record->correlation_id.external.value
                 << ", kind=" << record->kind << ", operation=" << record->operation
                 << ", src_agent_id=" << record->src_agent_id.handle
                 << ", dst_agent_id=" << record->dst_agent_id.handle
                 << ", direction=" << record->operation << ", start=" << record->start_timestamp
                 << ", stop=" << record->end_timestamp
                 << ", name=" << client_name_info.at(record->kind, record->operation);

            if(record->start_timestamp > record->end_timestamp)
                printf("memory copy: start > end \n");
                // throw std::runtime_error("memory copy: start > end");

            static_cast<call_stack_t*>(user_data)->emplace_back(
                source_location{__FUNCTION__, __FILE__, __LINE__, kind_name + info.str()});
        }
        else
        {
            auto _msg = std::stringstream{};
            _msg << "unexpected rocprofiler_record_header_t category + kind: (" << header->category
                 << " + " << header->kind << ")";
            std::cout << _msg.str() << std::endl;
            // throw std::runtime_error{_msg.str()};
        }
        // Pass the created event to RocmTraceCollectorImpl
        LOG(ERROR) << "Prepare transferring event to RocmTraceCollectorImpl";
        collector_impl->AddEvent(event);
    }   
}

int tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    assert(tool_data != nullptr);

    auto* call_stack_v = static_cast<call_stack_t*>(tool_data);

    call_stack_v->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    client_name_info = rocm_get_buffer_tracing_names<std::string_view>();
    // client_name_info = get_default_buffer_tracing_names();

    for(const auto& itr : client_name_info)
    {
        auto name_idx = std::stringstream{};
        name_idx << " [" << std::setw(3) << itr.value << "]";
        call_stack_v->emplace_back(
            source_location{"rocprofiler_buffer_tracing_kind_names          " + name_idx.str(),
                            __FILE__,
                            __LINE__,
                            std::string{itr.name}});

        for(auto [didx, ditr] : itr.items())
        {
            auto operation_idx = std::stringstream{};
            operation_idx << " [" << std::setw(3) << didx << "]";
            call_stack_v->emplace_back(source_location{
                "rocprofiler_buffer_tracing_kind_operation_names" + operation_idx.str(),
                __FILE__,
                __LINE__,
                std::string{"- "} + std::string{*ditr}});
        }
    }

    client_fini_func = fini_func;

    ROCPROFILER_CALL(se::wrap::rocprofiler_create_context(&client_ctx), "context creation");

    auto code_object_ops = std::vector<rocprofiler_tracing_operation_t>{
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};

    ROCPROFILER_CALL(
        se::wrap::rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       code_object_ops.data(),
                                                       code_object_ops.size(),
                                                       tool_code_object_callback,
                                                       nullptr),
        "code object tracing service configure");

    constexpr auto buffer_size_bytes      = 4096;
    constexpr auto buffer_watermark_bytes = buffer_size_bytes - (buffer_size_bytes / 8);

    ROCPROFILER_CALL(se::wrap::rocprofiler_create_buffer(client_ctx,
                                               buffer_size_bytes,
                                               buffer_watermark_bytes,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_callback,
                                               tool_data,
                                               &client_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(
        se::wrap::rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, nullptr, 0, client_buffer),
        "buffer tracing service configure");
    
    ROCPROFILER_CALL(
        se::wrap::rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, client_buffer),
        "buffer tracing service for kernel dispatch configure");

    ROCPROFILER_CALL(
        se::wrap::rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, nullptr, 0, client_buffer),
        "buffer tracing service for memory copy configure");

    int valid_ctx = 0;
    ROCPROFILER_CALL(se::wrap::rocprofiler_context_is_valid(client_ctx, &valid_ctx),
                     "context validity check");
    if(valid_ctx == 0)
    {
        // notify rocprofiler that initialization failed
        // and all the contexts, buffers, etc. created
        // should be ignored
        return -1;
    }

    ROCPROFILER_CALL(se::wrap::rocprofiler_start_context(client_ctx), "rocprofiler context start");

    // no errors
    return 0;
}

void tool_fini(void* tool_data){
    assert(tool_data != nullptr);

    auto* _call_stack = static_cast<call_stack_t*>(tool_data);
    _call_stack->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    print_call_stack(*_call_stack);

    delete _call_stack;
}
}  // end of namespace

int RocmTracer::NumGpus() {
    static int num_gpus = []() -> int {
        if (hipInit(0) != hipSuccess) {
            return 0;
        }
        int gpu_count;
        if (hipGetDeviceCount(&gpu_count) != hipSuccess) {
            return 0;
        }
        LOG(ERROR) << "Profiler found " << gpu_count << " GPUs.";
        return gpu_count;
    }();
    return num_gpus;
}

void RocmTracer::Enable(const RocmTracerOptions& options, RocmTraceCollector* collector) {
    options_ = options;
    collector_ = collector;
    LOG(ERROR) << "GpuTracer started";
}

/*static*/ uint64_t RocmTracer::GetTimestamp() {
    uint64_t ts;
    rocprofiler_status_t CHECKSTATUS = se::wrap::rocprofiler_get_timestamp(&ts);
    if (CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS) {
        const char* errstr = se::wrap::rocprofiler_get_status_string(CHECKSTATUS);
        LOG(ERROR) << "function rocprofiler_get_timestamp failed with error "
                   << errstr;
        return 0;
    }
    return ts;
}

void RocmTracer::setup(){
    if(int status = 0;
       se::wrap::rocprofiler_is_initialized(&status) == ROCPROFILER_STATUS_SUCCESS && status == 0){
        ROCPROFILER_CALL(se::wrap::rocprofiler_force_configure(&rocprofiler_configure),
                         "force configuration");
    }
}

void RocmTracer::shutdown(){
    if(client_id){
        ROCPROFILER_CALL(se::wrap::rocprofiler_flush_buffer(client_buffer), "buffer flush");
        client_fini_func(*client_id);
    }
}

void RocmTracer::start(){
    ROCPROFILER_CALL(se::wrap::rocprofiler_start_context(client_ctx), "context start");
}

void RocmTracer::identify(uint64_t val){
    auto _tid = rocprofiler_thread_id_t{};
    se::wrap::rocprofiler_get_thread_id(&_tid);
    rocprofiler_user_data_t user_data = {};
    user_data.value                   = val;
    se::wrap::rocprofiler_push_external_correlation_id(client_ctx, _tid, user_data);
}

void RocmTracer::stop(){
    ROCPROFILER_CALL(se::wrap::rocprofiler_stop_context(client_ctx), "context stop");
}

/* static */ RocmTracer* RocmTracer::GetRocmTracerSingleton() {
  static RocmTracer singleton;  // Changed to a direct static instance
  return &singleton;
}

bool RocmTracer::IsAvailable() const {
  return is_available_;
}

}  // namespace profiler
}  // namespace xla

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // set the client name
    id->name = "XLA-with-rocprofiler-sdk";

    // store client info
    xla::profiler::client_id = id;
    LOG(ERROR) << "Configure rocprofiler-sdk...\n";

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << id->name << "Configure XLA with rocprofv3... (priority=" << priority << ") is using rocprofiler-sdk v" << major << "."
         << minor << "." << patch << " (" << runtime_version << ")";

    std::clog << info.str() << std::endl;

    auto* client_tool_data = new std::vector<>{};

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &xla::profiler::tool_init,
                                            &xla::profiler::tool_fini,
                                            static_cast<void*>(client_tool_data)};

    // return pointer to configure data
    return &cfg;
}


