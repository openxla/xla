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
/*
using tsl::mutex;
using tsl::mutex_lock;
using tsl::profiler::AnnotationStack;
*/
    
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
    common::print_call_stack("api_buffered_trace.log", _call_stack);
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
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        auto kind_name = std::string{};
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
            }
        }

        if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
           (header->kind == ROCPROFILER_BUFFER_TRACING_HSA_CORE_API ||
            header->kind == ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API ||
            header->kind == ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API ||
            header->kind == ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API))
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_hsa_api_record_t*>(header->payload);
            auto info = std::stringstream{};
            info << "tid=" << record->thread_id << ", context=" << context.handle
                 << ", buffer_id=" << buffer_id.handle
                 << ", cid=" << record->correlation_id.internal
                 << ", extern_cid=" << record->correlation_id.external.value
                 << ", kind=" << record->kind << ", operation=" << record->operation
                 << ", start=" << record->start_timestamp << ", stop=" << record->end_timestamp
                 << ", name=" << client_name_info.at(record->kind, record->operation);

            if(record->start_timestamp > record->end_timestamp)
            {
                auto msg = std::stringstream{};
                msg << "hsa api: start > end (" << record->start_timestamp << " > "
                    << record->end_timestamp
                    << "). diff = " << (record->start_timestamp - record->end_timestamp);
                std::cerr << "threw an exception " << msg.str() << "\n" << std::flush;
                // throw std::runtime_error{msg.str()};
            }

            static_cast<call_stack_t*>(user_data)->emplace_back(
                source_location{__FUNCTION__, __FILE__, __LINE__, kind_name + info.str()});
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(header->payload);
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
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_page_migration_record_t*>(header->payload);

            auto info = std::stringstream{};

            info << "kind=" << record->kind << ", operation=" << record->operation
                 << ", pid=" << record->pid << ", start=" << record->start_timestamp
                 << ", stop=" << record->end_timestamp
                 << ", name=" << client_name_info.at(record->kind, record->operation);

            switch(record->operation)
            {
                case ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE:
                {
                    info << ", page_fault=(" << record->page_fault.read_fault << ", "
                         << record->page_fault.migrated << ", " << record->page_fault.node_id
                         << ", " << std::hex << "0x" << record->page_fault.address << ")";
                    break;
                }
                case ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT:
                {
                    info << ", page_migrate=(" << std::hex << "0x"
                         << record->page_migrate.start_addr << ", 0x"
                         << record->page_migrate.end_addr << ", " << std::dec
                         << record->page_migrate.from_node << ", " << record->page_migrate.to_node
                         << ", " << record->page_migrate.prefetch_node << ", "
                         << record->page_migrate.preferred_node << ", "
                         << record->page_migrate.trigger << ")";
                    break;
                }
                case ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND:
                {
                    info << ", queue_suspend=(" << record->queue_suspend.rescheduled << ", "
                         << record->queue_suspend.node_id << ", " << record->queue_suspend.trigger
                         << ")";
                    break;
                }
                case ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU:
                {
                    info << ", unmap_from_gpu=(" << record->unmap_from_gpu.node_id << std::hex
                         << ", 0x" << record->unmap_from_gpu.start_addr << ", 0x"
                         << record->unmap_from_gpu.end_addr << ", " << std::dec
                         << record->unmap_from_gpu.trigger << ")";
                    break;
                }
                case ROCPROFILER_PAGE_MIGRATION_NONE:
                case ROCPROFILER_PAGE_MIGRATION_LAST:
                {
                    // throw std::runtime_error{"unexpected page migration value"};
                    printf("page migration: start > end\n");
                    break;
                }
            }

            if(record->start_timestamp > record->end_timestamp)
                printf("page migration: start > end\n");
                // throw std::runtime_error("page migration: start > end");

            static_cast<call_stack_t*>(user_data)->emplace_back(
                source_location{__FUNCTION__, __FILE__, __LINE__, kind_name + info.str()});
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_scratch_memory_record_t*>(header->payload);

            auto info = std::stringstream{};

            auto _elapsed =
                std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
                    std::chrono::nanoseconds{record->end_timestamp - record->start_timestamp})
                    .count();

            info << "tid=" << record->thread_id << ", context=" << context.handle
                 << ", buffer_id=" << buffer_id.handle
                 << ", cid=" << record->correlation_id.internal
                 << ", extern_cid=" << record->correlation_id.external.value
                 << ", kind=" << record->kind << ", operation=" << record->operation
                 << ", agent_id=" << record->agent_id.handle
                 << ", queue_id=" << record->queue_id.handle << ", thread_id=" << record->thread_id
                 << ", elapsed=" << std::setprecision(3) << std::fixed << _elapsed
                 << " usec, flags=" << record->flags
                 << ", name=" << client_name_info.at(record->kind, record->operation);

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
    }
}

void thread_precreate(rocprofiler_runtime_library_t lib, void* tool_data)
{
    static_cast<call_stack_t*>(tool_data)->emplace_back(
        source_location{__FUNCTION__,
                        __FILE__,
                        __LINE__,
                        std::string{"internal thread about to be created by rocprofiler (lib="} +
                            std::to_string(lib) + ")"});
}

void thread_postcreate(rocprofiler_runtime_library_t lib, void* tool_data)
{
    static_cast<call_stack_t*>(tool_data)->emplace_back(
        source_location{__FUNCTION__,
                        __FILE__,
                        __LINE__,
                        std::string{"internal thread was created by rocprofiler (lib="} +
                            std::to_string(lib) + ")"});
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
                            "test.."});
                            // std::string{itr.name}});

        for(auto [didx, ditr] : itr.items())
        {
            auto operation_idx = std::stringstream{};
            operation_idx << " [" << std::setw(3) << didx << "]";
            call_stack_v->emplace_back(source_location{
                "rocprofiler_buffer_tracing_kind_operation_names" + operation_idx.str(),
                __FILE__,
                __LINE__,
                "test..."});
                // std::string{"- "} + std::string{*ditr}});
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

    /*
    ROCPROFILER_CALL(se::wrap::rocprofiler_create_buffer(client_ctx,
                                               buffer_size_bytes,
                                               buffer_watermark_bytes,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_callback,
                                               tool_data,
                                               &client_buffer),
                     "buffer creation");
    */
    for(auto itr :
        {ROCPROFILER_BUFFER_TRACING_HSA_CORE_API, ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API})
    {
        ROCPROFILER_CALL(se::wrap::rocprofiler_configure_buffer_tracing_service(
                             client_ctx, itr, nullptr, 0, client_buffer),
                         "buffer tracing service configure");
    }

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

    // May have incompatible kernel so only emit a warning here
    ROCPROFILER_WARN(se::wrap::rocprofiler_configure_buffer_tracing_service(
        client_ctx, ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION, nullptr, 0, client_buffer));

    ROCPROFILER_CALL(
        se::wrap::rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY, nullptr, 0, client_buffer),
        "buffer tracing service for page migration configure");

    auto client_thread = rocprofiler_callback_thread_t{};
    ROCPROFILER_CALL(se::wrap::rocprofiler_create_callback_thread(&client_thread),
                     "creating callback thread");

    ROCPROFILER_CALL(se::wrap::rocprofiler_assign_callback_thread(client_buffer, client_thread),
                     "assignment of thread for buffer");

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

void tool_fini(void* tool_data)
{
    assert(tool_data != nullptr);

    auto* _call_stack = static_cast<call_stack_t*>(tool_data);
    _call_stack->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    print_call_stack(*_call_stack);

    delete _call_stack;
}
}  // namespace

void
setup()
{
    if(int status = 0;
       se::wrap::rocprofiler_is_initialized(&status) == ROCPROFILER_STATUS_SUCCESS && status == 0)
    {
        ROCPROFILER_CALL(se::wrap::rocprofiler_force_configure(&rocprofiler_configure),
                         "force configuration");
    }
}

void
shutdown()
{
    if(client_id)
    {
        ROCPROFILER_CALL(se::wrap::rocprofiler_flush_buffer(client_buffer), "buffer flush");
        client_fini_func(*client_id);
    }
}

void
start()
{
    ROCPROFILER_CALL(se::wrap::rocprofiler_start_context(client_ctx), "context start");
}

void
identify(uint64_t val)
{
    auto _tid = rocprofiler_thread_id_t{};
    se::wrap::rocprofiler_get_thread_id(&_tid);
    rocprofiler_user_data_t user_data = {};
    user_data.value                   = val;
    se::wrap::rocprofiler_push_external_correlation_id(client_ctx, _tid, user_data);
}

void
stop()
{
    ROCPROFILER_CALL(se::wrap::rocprofiler_stop_context(client_ctx), "context stop");
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
    id->name = "XLA-with-rocprofv3";

    // store client info
    xla::profiler::client_id = id;
    std::cout << "Configure rocprofv3...\n" <<std::flush;

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << id->name << "Configure XLA with rocprofv3... (priority=" << priority << ") is using rocprofiler-sdk v" << major << "."
         << minor << "." << patch << " (" << runtime_version << ")";

    std::clog << info.str() << std::endl;

    auto* client_tool_data = new std::vector<xla::profiler::source_location>{};

    client_tool_data->emplace_back(
        xla::common::source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});

    ROCPROFILER_CALL(se::wrap::rocprofiler_at_internal_thread_create(
                         xla::profiler::thread_precreate,
                         xla::profiler::thread_postcreate,
                         ROCPROFILER_LIBRARY | ROCPROFILER_HSA_LIBRARY | ROCPROFILER_HIP_LIBRARY |
                             ROCPROFILER_MARKER_LIBRARY,
                         static_cast<void*>(client_tool_data)),
                     "registration for thread creation notifications");

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &xla::profiler::tool_init,
                                            &xla::profiler::tool_fini,
                                            static_cast<void*>(client_tool_data)};

    // return pointer to configure data
    return &cfg;
}


