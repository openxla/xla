/* Copyright 2017 The OpenXLA Authors.

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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/map_util.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/scoped_module_handle.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/vmm_device_address_allocator.h"
#include "xla/util.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::gpu {

absl::StatusOr<const GpuExecutable::BufferAllocToDeviceMemoryMap*>
GpuExecutable::ResolveConstantGlobals(se::Stream* stream) {
  se::StreamExecutor* executor = stream->parent();

  absl::MutexLock lock(module_handle_mutex_);
  auto it = module_globals_.find(executor);
  if (it != module_globals_.end()) {
    return it->second.get();
  }

  se::MultiModuleLoaderSpec module_spec;
  if (!binary().empty()) {
    module_spec.AddCudaCubinInMemory(binary());
  }
  module_spec.AddCudaPtxInMemory(text().c_str());

  auto globals = std::make_unique<BufferAllocToDeviceMemoryMap>();
  se::ModuleHandle module_handle;
  // The CUDA driver isn't able to load a PTX and a binary which are both empty.
  // It's okay if we skip loading in this case; if the module isn't loaded, all
  // symbol lookups will fail, just as they should for an empty module.
  if (!(executor->GetPlatform()->id() == se::cuda::kCudaPlatformId &&
        binary().empty() && text().empty())) {
    ASSIGN_OR_RETURN(module_handle, executor->LoadModule(module_spec));
  }

  // A flag signalling if constant initialization submitted memcpy operations
  // to the `stream`.
  int submitted_mem_copies = 0;

  for (const ConstantInfo& info : constants_) {
    absl::StatusOr<se::DeviceAddressBase> global_status;
    if (static_cast<bool>(module_handle)) {
      global_status = executor->GetSymbol(info.symbol_name, module_handle);
    }

    se::DeviceAddressBase global;

    CHECK(static_cast<bool>(module_handle) && global_status.ok());
    // The constant was defined in the PTX and has been allocated by the CUDA
    // driver.
    global = *global_status;
    XLA_VLOG_DEVICE(3, executor->device_ordinal()) << absl::StreamFormat(
        "Resolved global %s to %p", info.symbol_name, global.opaque());

    if (!info.content.span().empty()) {
      // This means the constant did not have an initializer in the PTX and
      // therefore must be initialized by XLA here.
      RETURN_IF_ERROR(stream->Memcpy(&global, info.content.span().data(),
                                     info.content.span().size()));
      submitted_mem_copies = true;
    }

    if (info.allocation_index != -1) {
      InsertOrDie(globals.get(), info.allocation_index, global);
    }
  }

  // Wait for the completion of all host->device transfers, to guarantee that
  // destructor will not race with any operations in flight (deallocate
  // xla::Literal owned by the HLO module).
  if (submitted_mem_copies) {
    CHECK_OK(stream->BlockHostUntilDone());
  }

  module_handles_.emplace(executor,
                          se::ScopedModuleHandle(executor, module_handle));
  return module_globals_.emplace(executor, std::move(globals))
      .first->second.get();
}

absl::StatusOr<se::DeviceAddressBase> GpuExecutable::BufferForAllocation(
    ParameterBufferResolver get_parameter_buffer,
    const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
    const BufferAllocation& allocation,
    se::DeviceAddressAllocator* const memory_allocator, int device_ordinal,
    int64_t arg_idx,
    const absl::flat_hash_map<LogicalBuffer::Color, int64_t>&
        allocate_granularity) {
  if (allocation.is_thread_local()) {
    return se::DeviceAddressBase{};
  }
  if (allocation.is_entry_computation_parameter()) {
    ASSIGN_OR_RETURN(ParameterBuffer registered_buffer,
                     get_parameter_buffer(allocation));
    if (registered_buffer.buffer.is_null() &&
        registered_buffer.buffer.size() > 0) {
      return FailedPrecondition(
          "Cannot run XLA computation because pointer to (sub-)buffer at "
          "index %s of parameter %d was null.  All pointers to "
          "(sub-)buffers must not be null, unless the (sub-)buffer has "
          "zero elements.",
          allocation.param_shape_index().ToString(),
          registered_buffer.parameter_number);
    }
    return registered_buffer.buffer;
  }
  if (allocation.is_constant()) {
    auto it = globals->find(arg_idx);
    if (it == globals->end()) {
      return se::DeviceAddressBase();
    }
    return it->second;
  }

  // Allocate each allocation that might escape, or is the temp buffer.
  CHECK(allocation.maybe_live_out() || allocation.IsPreallocatedTempBuffer());
  int64_t buffer_size = allocation.size();
  se::DeviceAddressBase buffer_address;
  if (buffer_size > 0) {
    // Maybe round up buffer allocation size to the requested granularity.
    if (auto it = allocate_granularity.find(allocation.color());
        it != allocate_granularity.end()) {
      buffer_size = RoundUpTo(buffer_size, it->second);
    }
    ASSIGN_OR_RETURN(
        se::ScopedDeviceAddress<uint8_t> buffer,
        memory_allocator->Allocate(device_ordinal, buffer_size,
                                   /*retry_on_failure=*/true,
                                   /*memory_space=*/allocation.color()));
    buffer_address = buffer.Release();
  }
  return buffer_address;
}

absl::Status CheckAlignment(const BufferAllocation& allocation,
                            se::DeviceAddressBase buffer, int arg_idx) {
  const int64_t expected_alignment = [&] {
    if (allocation.is_entry_computation_parameter()) {
      return kEntryParameterAlignBytes;
    }
    if (allocation.is_constant()) {
      return kConstantBufferAlignBytes;
    }
    return kXlaAllocatedBufferAlignBytes;
  }();
  if (!buffer.is_null() &&
      reinterpret_cast<uintptr_t>(buffer.opaque()) % expected_alignment != 0) {
    return Internal(
        "Address of buffer %d must be a multiple of %x, but "
        "was %p",
        arg_idx, expected_alignment, buffer.opaque());
  }
  return absl::OkStatus();
}

// Resolve GpuCollectives instance that we should use for the run.
// TODO(ezhulenev): We have almost identical method in `collective_params.cc`,
// this one has to be removed.
static GpuCollectives* ResolveGpuCollectives(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options) {
  auto* gpu_options = run_options->run_options().gpu_executable_run_options();
  if (gpu_options && gpu_options->collectives()) {
    return gpu_options->collectives();
  }

  absl::string_view platform_name =
      run_options->run_options().stream()->parent()->GetPlatform()->Name();

  // If debug options specify a collectives implementation by name, look it up
  // in the registry. Otherwise, use the default (highest-priority) one.
  if (debug_options &&
      !debug_options->xla_gpu_collectives_implementation().empty()) {
    absl::StatusOr<Collectives*> collectives = CollectivesRegistry::Get(
        platform_name, debug_options->xla_gpu_collectives_implementation());
    CHECK_OK(collectives)  // Crash OK
        << "Failed to get GPU collectives implementation: "
        << debug_options->xla_gpu_collectives_implementation();
    return absl::down_cast<GpuCollectives*>(*collectives);
  }

  return GpuCollectives::Default(platform_name);
}

absl::StatusOr<BufferAllocations> GpuExecutable::GenerateBufferAllocations(
    const ServiceExecutableRunOptions* run_options,
    ParameterBufferResolver get_parameter_buffer,
    const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
    se::DeviceAddressAllocator* const memory_allocator, int device_ordinal) {
  tsl::profiler::TraceMe hlo_module_activity(
      [&] { return std::string("Build buffer allocations"); },
      tsl::profiler::TraceMeLevel::kInfo);

  const DebugOptions* debug_options =
      has_module() ? &module_config().debug_options() : nullptr;

  absl::flat_hash_map<LogicalBuffer::Color, int64_t> allocate_granularity;
  if (auto* collectives = ResolveGpuCollectives(run_options, debug_options)) {
    // BFC allocator ignores memory alignment and always allocates 256 byte
    // aligned buffers, however for collective memory underlying libraries
    // require larger alignment. We conservatively round up all allocation
    // sizes to the alignment requirement. Proper fix must be done in BFC
    // allocator and all the other allocator adaptors that we have in XLA, but
    // this is left as an exercise for curious reader. The raw memory allocator
    // that backs the BFC allocator uses correct granularity and alignment.
    static constexpr int64_t kCollectiveMemoryColor = 1;
    allocate_granularity[kCollectiveMemoryColor] =
        collectives->SymmetricMemoryAlignment();
  }

  // Tag allocations made in this invocation as multi-device for VMM reuse.
  se::DeviceAddressVmmAllocator::DeviceAssignmentScope
      vmm_device_assignment_scope(
          run_options->run_options().device_assignment());

  absl::Span<const BufferAllocation* const> allocations = GetAllocations();
  const int64_t num_buffers = allocations.size();
  std::vector<se::DeviceAddressBase> buffers;
  buffers.reserve(num_buffers);
  for (int64_t i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = *allocations[i];
    ASSIGN_OR_RETURN(
        buffers.emplace_back(),
        BufferForAllocation(get_parameter_buffer, globals, allocation,
                            memory_allocator, device_ordinal, i,
                            allocate_granularity));
    RETURN_IF_ERROR(CheckAlignment(allocation, buffers.back(), i));
  }
  return {{buffers, device_ordinal, memory_allocator}};
}

absl::StatusOr<se::DeviceAddressBase>
GpuExecutable::AllocateCopyProtectedOutputBuffer(
    const ServiceExecutableRunOptions* run_options,
    BufferAllocations& buffer_allocations, const ShapeIndex& index,
    const BufferAllocation& allocation, int device_ordinal,
    se::DeviceAddressAllocator* const memory_allocator) {
  // The caller guards this against aliasing pass-through params, as we do not
  // need to write into the output buffer in that case.
  XLA_VLOG_DEVICE(3, device_ordinal)
      << "Using copy-protection: aliasing is specified, but the "
         "buffer is not donated; allocating a fresh buffer";
  int64_t allocation_size =
      ShapeUtil::ByteSizeOf(ShapeUtil::GetSubshape(result_shape(), index));
  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> allocated_buffer =
      memory_allocator->Allocate(device_ordinal, allocation_size,
                                 /*retry_on_failure=*/true,
                                 /*memory_space=*/allocation.color());
  if (!allocated_buffer.ok()) {
    return VerboseAllocationError(allocated_buffer.status());
  }
  se::DeviceAddressBase result_buffer = allocated_buffer->Release();
  se::DeviceAddressBase& aliased_buffer =
      buffer_allocations.GetMutableDeviceAddress(allocation.index());
  CHECK_EQ(aliased_buffer.size(), result_buffer.size());
  RETURN_IF_ERROR(run_options->stream()->MemcpyD2D(
      &result_buffer, aliased_buffer, aliased_buffer.size()));
  aliased_buffer = result_buffer;
  return result_buffer;
}

absl::Status GpuExecutable::VerboseAllocationError(absl::Status s) {
  return ResourceExhausted(
      "%s\n%s\n", s.message(),
      buffer_assignment_->ToVerboseString(alias_info_.get(),
                                          debug_buffer_assignment_show_max_));
}

}  // namespace xla::gpu
