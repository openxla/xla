/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_executable_buffer_allocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_address_vmm_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {
namespace {

// Number of profiling executions before the SKIP_PROFILED transition.
constexpr int64_t kProfileStepsLimit = 3;

uint64_t RoundUpToGranularity(uint64_t size, uint64_t granularity) {
  if (granularity == 0) {
    return size;
  }
  return ((size + granularity - 1) / granularity) * granularity;
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

struct CollectedAllocationIndices {
  GpuExecutableBufferAllocator::AllocationIndexSet constant;
  GpuExecutableBufferAllocator::AllocationIndexSet persistent;
  GpuExecutableBufferAllocator::AllocationIndexSet va_remapped;
  GpuExecutableBufferAllocator::AllocationIndexSet profile_candidates;
};

CollectedAllocationIndices CollectAllocationIndices(
    absl::Span<const BufferAllocation* const> allocations,
    const ThunkExecutor* thunk_executor,
    DebugOptions::CommandBufferUpdateMode update_mode) {
  CollectedAllocationIndices indices;
  if (thunk_executor == nullptr) {
    return indices;
  }

  CHECK_OK(thunk_executor->thunks().WalkNested(
      [&](const Thunk* thunk) -> absl::Status {
        auto* command_buffer_thunk =
            dynamic_cast<const CommandBufferThunk*>(thunk);
        if (command_buffer_thunk == nullptr) {
          return absl::OkStatus();
        }
        for (BufferAllocation::Index index :
             command_buffer_thunk->allocs_indices()) {
          if (index < 0 || static_cast<size_t>(index) >= allocations.size()) {
            continue;
          }
          const BufferAllocation& allocation = *allocations[index];
          if (allocation.size() == 0) {
            continue;
          }
          if (allocation.is_constant()) {
            indices.constant.insert(index);
          } else if (update_mode == DebugOptions::SKIP_TEMP &&
                     allocation.IsPreallocatedTempBuffer()) {
            indices.va_remapped.insert(index);
          } else if (update_mode == DebugOptions::SKIP_PROFILED &&
                     !allocation.is_thread_local()) {
            indices.profile_candidates.insert(index);
          }
        }
        return absl::OkStatus();
      }));

  indices.persistent = indices.constant;
  indices.persistent.insert(indices.va_remapped.begin(),
                            indices.va_remapped.end());
  return indices;
}

}  // namespace

absl::StatusOr<uint64_t>
GpuExecutableBufferAllocator::Remapping::GetReservationOffset(
    BufferAllocation::Index idx) const {
  auto it = allocation_to_reservation_offset.find(idx);
  if (it == allocation_to_reservation_offset.end()) {
    return Internal("No VA reservation offset for allocation %d", idx);
  }
  return it->second;
}

absl::StatusOr<uint64_t>
GpuExecutableBufferAllocator::Remapping::GetMappingSize(
    BufferAllocation::Index idx) const {
  auto it = allocation_to_mapping_size.find(idx);
  if (it == allocation_to_mapping_size.end()) {
    return Internal("No VA mapping size for allocation %d", idx);
  }
  return it->second;
}

GpuExecutableBufferAllocator::ExecutionScope::ExecutionScope(
    GpuExecutableBufferAllocator* owner, Remapping* remapping,
    se::DeviceAddressVmmAllocator* vmm_allocator,
    std::unique_ptr<absl::MutexLock> remap_lock)
    : owner_(owner),
      remapping_(remapping),
      vmm_allocator_(vmm_allocator),
      remap_lock_(std::move(remap_lock)) {}

GpuExecutableBufferAllocator::ExecutionScope::~ExecutionScope() {
  if (step_aliases_ != nullptr && !step_aliases_->aliases.empty()) {
    // Normally ReleaseStepAliases runs inside ExecuteWithBufferAllocations;
    // reaching this point means buffer allocation generation or output
    // handling failed after some aliases were installed.
    absl::Status status = ReleaseStepAliases(/*allocs=*/nullptr);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to release command buffer VA remapping aliases "
                    "for module "
                 << owner_->module_name_ << ": " << status;
    }
  }
}

bool GpuExecutableBufferAllocator::ExecutionScope::remap_active() const {
  if (remapping_ == nullptr) {
    return false;
  }
  if (owner_->update_mode_ == DebugOptions::SKIP_PROFILED) {
    return remapping_->phase == Remapping::ProfilePhase::kActive;
  }
  return true;
}

const GpuExecutableBufferAllocator::AllocationIndexSet&
GpuExecutableBufferAllocator::ExecutionScope::active_remap_set() const {
  if (owner_->update_mode_ == DebugOptions::SKIP_PROFILED) {
    return remapping_->profiled_va_remapped_alloc_indices;
  }
  return owner_->va_remapped_alloc_indices_;
}

absl::Status GpuExecutableBufferAllocator::ExecutionScope::PrepareReservation(
    const ServiceExecutableRunOptions* run_options, int device_ordinal,
    const absl::flat_hash_map<LogicalBuffer::Color, int64_t>&
        allocate_granularity) {
  if (!remap_active()) {
    return absl::OkStatus();
  }

  uint64_t granularity =
      vmm_allocator_->GetAllocationGranularity(run_options->stream()->parent());
  if (remapping_->va_reservation != nullptr &&
      remapping_->granularity != granularity) {
    return Internal(
        "Command buffer VA remapping granularity changed for module %s: "
        "previous=%u current=%u",
        owner_->module_name_, remapping_->granularity, granularity);
  }
  if (remapping_->va_reservation != nullptr) {
    return absl::OkStatus();
  }

  // First execution on this executor creates the persistent reservation. Later
  // executions reuse the same reservation and deterministic layout.
  remapping_->granularity = granularity;
  remapping_->total_size = 0;
  remapping_->allocation_to_reservation_offset.clear();
  remapping_->allocation_to_mapping_size.clear();
  for (BufferAllocation::Index idx : active_remap_set()) {
    const BufferAllocation& allocation = *owner_->allocations_[idx];
    uint64_t buffer_size = allocation.size();
    if (auto it = allocate_granularity.find(allocation.color());
        it != allocate_granularity.end()) {
      buffer_size =
          RoundUpToGranularity(buffer_size, static_cast<uint64_t>(it->second));
    }
    uint64_t mapping_size =
        RoundUpToGranularity(buffer_size, remapping_->granularity);
    remapping_->allocation_to_reservation_offset[idx] = remapping_->total_size;
    remapping_->allocation_to_mapping_size[idx] = mapping_size;
    remapping_->total_size = remapping_->total_size + mapping_size;
  }
  ASSIGN_OR_RETURN(
      remapping_->va_reservation,
      vmm_allocator_->CreateReservation(run_options->stream()->parent(),
                                        remapping_->total_size));
  XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
      "VA remapping: reserved range for module %s VA=%p total_size=%u "
      "granularity=%u",
      owner_->module_name_, remapping_->va_reservation->address().opaque(),
      remapping_->total_size, remapping_->granularity);
  return absl::OkStatus();
}

bool GpuExecutableBufferAllocator::ExecutionScope::ShouldRemapAllocation(
    BufferAllocation::Index index) const {
  return remap_active() && active_remap_set().contains(index);
}

void GpuExecutableBufferAllocator::ExecutionScope::ObserveAllocationAddresses(
    const BufferAllocations& allocs) {
  for (BufferAllocation::Index idx : owner_->profile_candidate_alloc_indices_) {
    se::DeviceAddressBase addr = allocs.GetDeviceAddress(idx);
    if (addr.is_null()) {
      remapping_->unstable_alloc_indices.insert(idx);
      continue;
    }
    auto [it, inserted] =
        remapping_->last_observed_address.try_emplace(idx, addr);
    if (!inserted && !it->second.IsSameAs(addr)) {
      remapping_->unstable_alloc_indices.insert(idx);
      it->second = addr;
    }
  }
}

se::DeviceAddressBase
GpuExecutableBufferAllocator::ExecutionScope::ReservationSlice(
    uint64_t offset, uint64_t size) const {
  return se::DeviceAddressBase(
      static_cast<char*>(remapping_->va_reservation->address().opaque()) +
          offset,
      size);
}

void GpuExecutableBufferAllocator::ExecutionScope::RecordStepAlias(
    int device_ordinal, BufferAllocation::Index index,
    uint64_t reservation_offset, uint64_t mapping_size,
    se::DeviceAddressBase external_address) {
  if (step_aliases_ == nullptr) {
    step_aliases_ = std::make_unique<StepAliases>();
    step_aliases_->device_ordinal = device_ordinal;
  }
  step_aliases_->aliases.push_back(
      {index, reservation_offset, mapping_size, external_address});
  step_aliases_->external_address_by_index[index] = external_address;
}

absl::Status GpuExecutableBufferAllocator::ExecutionScope::ReleaseStepAliases(
    BufferAllocations* allocs) {
  if (step_aliases_ == nullptr) {
    return absl::OkStatus();
  }
  absl::Status status;
  for (const StepAlias& alias : step_aliases_->aliases) {
    if (allocs != nullptr) {
      allocs->GetMutableDeviceAddress(alias.index) = alias.external_address;
    }
    absl::Status unmap_status = vmm_allocator_->UnMap(
        step_aliases_->device_ordinal, remapping_->va_reservation.get(),
        alias.reservation_offset, alias.mapping_size);
    if (!unmap_status.ok() && status.ok()) {
      status = unmap_status;
    }
  }
  step_aliases_->aliases.clear();
  return status;
}

se::DeviceAddressBase
GpuExecutableBufferAllocator::ExecutionScope::ResolveOutputBuffer(
    BufferAllocation::Index index, se::DeviceAddressBase current) const {
  if (step_aliases_ == nullptr) {
    return current;
  }
  auto it = step_aliases_->external_address_by_index.find(index);
  if (it == step_aliases_->external_address_by_index.end()) {
    return current;
  }
  return it->second;
}

absl::StatusOr<se::ScopedDeviceAddress<uint8_t>>
GpuExecutableBufferAllocator::ExecutionScope::AllocateBuffer(
    int device_ordinal, const BufferAllocation& allocation,
    int64_t buffer_size) {
  ASSIGN_OR_RETURN(uint64_t va_offset,
                   remapping_->GetReservationOffset(allocation.index()));
  uint64_t mapping_size = RoundUpToGranularity(
      static_cast<uint64_t>(buffer_size), remapping_->granularity);
  return vmm_allocator_->Allocate(
      device_ordinal, mapping_size, /*retry_on_failure=*/true,
      /*memory_space=*/allocation.color(), remapping_->va_reservation.get(),
      va_offset, mapping_size, /*return_reservation_address=*/true);
}

absl::StatusOr<se::DeviceAddressBase>
GpuExecutableBufferAllocator::ExecutionScope::MapParameterBuffer(
    int device_ordinal, const BufferAllocation& allocation,
    se::DeviceAddressBase buffer) {
  if (buffer.is_null()) {
    return Internal(
        "Command buffer VA remapping selected parameter allocation %d, but "
        "the parameter buffer is null for this execution",
        allocation.index());
  }
  ASSIGN_OR_RETURN(uint64_t va_offset,
                   remapping_->GetReservationOffset(allocation.index()));
  ASSIGN_OR_RETURN(uint64_t mapping_size,
                   remapping_->GetMappingSize(allocation.index()));
  // Map() reactivates a matching stale mapping from the previous execution,
  // so a parameter that keeps its address across executions performs no VMM
  // driver calls here.
  RETURN_IF_ERROR(vmm_allocator_->Map(device_ordinal, buffer,
                                      remapping_->va_reservation.get(),
                                      va_offset, mapping_size));
  RecordStepAlias(device_ordinal, allocation.index(), va_offset, mapping_size,
                  buffer);
  return ReservationSlice(va_offset, mapping_size);
}

absl::StatusOr<se::DeviceAddressBase>
GpuExecutableBufferAllocator::ExecutionScope::AllocateEscapingBuffer(
    int device_ordinal, const BufferAllocation& allocation) {
  ASSIGN_OR_RETURN(uint64_t va_offset,
                   remapping_->GetReservationOffset(allocation.index()));
  ASSIGN_OR_RETURN(uint64_t mapping_size,
                   remapping_->GetMappingSize(allocation.index()));
  // The returned allocator-owned address escapes to the caller, which may
  // hold it beyond this execution and must be able to Deallocate() it. The
  // same physical allocation is also aliased at the reservation slice, which
  // is the stable address recorded in command buffers.
  ASSIGN_OR_RETURN(
      se::ScopedDeviceAddress<uint8_t> allocated,
      vmm_allocator_->Allocate(
          device_ordinal, mapping_size, /*retry_on_failure=*/true,
          /*memory_space=*/allocation.color(), remapping_->va_reservation.get(),
          va_offset, mapping_size, /*return_reservation_address=*/false));
  se::DeviceAddressBase allocator_address = allocated.Release();
  RecordStepAlias(device_ordinal, allocation.index(), va_offset, mapping_size,
                  allocator_address);
  XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
      "VA remapping: escaping allocation %d external=%p reservation=%p",
      allocation.index(), allocator_address.opaque(),
      ReservationSlice(va_offset, mapping_size).opaque());
  return ReservationSlice(va_offset, mapping_size);
}

absl::StatusOr<se::DeviceAddressBase>
GpuExecutableBufferAllocator::ExecutionScope::BufferForAllocation(
    ParameterBufferResolver get_parameter_buffer,
    const BufferAllocToDeviceMemoryMap* globals,
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
        registered_buffer.buffer.size() > 0 &&
        !registered_buffer.allow_null_buffer) {
      return FailedPrecondition(
          "Cannot run XLA computation because pointer to (sub-)buffer at "
          "index %s of parameter %d was null.  All pointers to "
          "(sub-)buffers must not be null, unless the (sub-)buffer has "
          "zero elements.",
          allocation.param_shape_index().ToString(),
          registered_buffer.parameter_number);
    }
    if (ShouldRemapAllocation(allocation.index())) {
      return MapParameterBuffer(device_ordinal, allocation,
                                registered_buffer.buffer);
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
  int64_t buffer_size = allocation.size();
  se::DeviceAddressBase buffer_address;
  if (buffer_size > 0) {
    // Maybe round up buffer allocation size to the requested granularity.
    if (auto it = allocate_granularity.find(allocation.color());
        it != allocate_granularity.end()) {
      buffer_size = RoundUpTo(buffer_size, it->second);
    }
    absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> buffer;
    if (ShouldRemapAllocation(allocation.index())) {
      if (allocation.maybe_live_out()) {
        return AllocateEscapingBuffer(device_ordinal, allocation);
      }
      buffer = AllocateBuffer(device_ordinal, allocation, buffer_size);
    } else {
      buffer = memory_allocator->Allocate(device_ordinal, buffer_size,
                                          /*retry_on_failure=*/true,
                                          /*memory_space=*/allocation.color());
    }
    ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> scoped_buffer,
                     std::move(buffer));
    buffer_address = scoped_buffer.Release();
  }
  return buffer_address;
}

absl::StatusOr<BufferAllocations>
GpuExecutableBufferAllocator::ExecutionScope::GenerateBufferAllocations(
    const ServiceExecutableRunOptions* run_options,
    ParameterBufferResolver get_parameter_buffer,
    const BufferAllocToDeviceMemoryMap* globals,
    se::DeviceAddressAllocator* const memory_allocator, int device_ordinal) {
  tsl::profiler::TraceMe hlo_module_activity(
      [&] { return std::string("Build buffer allocations"); },
      tsl::profiler::TraceMeLevel::kInfo);

  absl::flat_hash_map<LogicalBuffer::Color, int64_t> allocate_granularity;
  if (run_options && run_options->stream()) {
    absl::StatusOr<uint64_t> collective_memory_granularity =
        run_options->stream()->parent()->GetCollectiveMemoryGranularity();
    if (collective_memory_granularity.ok()) {
      // BFC allocator ignores memory alignment and always allocates 256 byte
      // aligned buffers, however for collective memory underlying libraries
      // require larger alignment. We conservatively round up all allocation
      // sizes to the alignment requirement. Proper fix must be done in BFC
      // allocator and all the other allocator adaptors that we have in XLA.
      static constexpr int64_t kCollectiveMemoryColor = 1;
      allocate_granularity[kCollectiveMemoryColor] =
          *collective_memory_granularity;
    }
  }

  // Tag allocations made in this invocation as multi-device for VMM reuse.
  se::DeviceAddressVmmAllocator::DeviceAssignmentScope
      vmm_device_assignment_scope(
          run_options->run_options().device_assignment());

  const int64_t num_buffers = owner_->allocations_.size();
  RETURN_IF_ERROR(
      PrepareReservation(run_options, device_ordinal, allocate_granularity));

  std::vector<se::DeviceAddressBase> buffers;
  buffers.reserve(num_buffers);
  for (int64_t i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = *owner_->allocations_[i];
    ASSIGN_OR_RETURN(
        buffers.emplace_back(),
        BufferForAllocation(get_parameter_buffer, globals, allocation,
                            memory_allocator, device_ordinal, i,
                            allocate_granularity));
    RETURN_IF_ERROR(CheckAlignment(allocation, buffers.back(), i));
  }
  return BufferAllocations(buffers, device_ordinal, memory_allocator);
}

absl::StatusOr<se::DeviceAddressBase>
GpuExecutableBufferAllocator::ExecutionScope::AllocateCopyProtectedOutputBuffer(
    const ServiceExecutableRunOptions* run_options,
    BufferAllocations& buffer_allocations, const ShapeIndex& index,
    const BufferAllocation& allocation, int device_ordinal,
    se::DeviceAddressAllocator* const memory_allocator,
    absl::FunctionRef<absl::Status(absl::Status)> allocation_error) {
  // The caller guards this against aliasing pass-through params, as we do not
  // need to write into the output buffer in that case.
  XLA_VLOG_DEVICE(3, device_ordinal)
      << "Using copy-protection: aliasing is specified, but the "
         "buffer is not donated; allocating a fresh buffer";
  if (remapping_ != nullptr &&
      owner_->update_mode_ == DebugOptions::SKIP_PROFILED) {
    if (remapping_->phase == Remapping::ProfilePhase::kProfiling) {
      // Copy-protection redirects this allocation to a fresh address each
      // execution, so it can never be VA-remapped.
      remapping_->unstable_alloc_indices.insert(allocation.index());
    } else if (remapping_->phase == Remapping::ProfilePhase::kActive &&
               remapping_->profiled_va_remapped_alloc_indices.contains(
                   allocation.index())) {
      // Redirecting a persistent-declared allocation would leave command
      // buffers reading the stale reservation address; failing loudly beats
      // silent corruption. This means the buffer donation behavior changed
      // after the profiling executions, which SKIP_PROFILED assumes does not
      // happen.
      return Internal(
          "Copy-protection triggered for VA-remapped allocation %d of module "
          "%s: buffer donation behavior changed after the SKIP_PROFILED "
          "profiling executions",
          allocation.index(), owner_->module_name_);
    }
  }
  int64_t allocation_size = ShapeUtil::ByteSizeOf(
      ShapeUtil::GetSubshape(owner_->result_shape_, index));
  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> allocated_buffer =
      memory_allocator->Allocate(device_ordinal, allocation_size,
                                 /*retry_on_failure=*/true,
                                 /*memory_space=*/allocation.color());
  if (!allocated_buffer.ok()) {
    return allocation_error(allocated_buffer.status());
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

absl::Status
GpuExecutableBufferAllocator::ExecutionScope::ExecuteWithBufferAllocations(
    BufferAllocations& owning_buffer_allocations, int device_ordinal,
    absl::FunctionRef<
        absl::Status(const BufferAllocations&,
                     std::optional<absl::Span<const BufferAllocation::Index>>
                         persistent_alloc_indices)>
        execute) {
  if (remapping_ == nullptr) {
    return execute(owning_buffer_allocations,
                   absl::MakeConstSpan(owner_->constant_alloc_indices_));
  }

  if (owner_->update_mode_ == DebugOptions::SKIP_PROFILED) {
    if (remapping_->phase == Remapping::ProfilePhase::kProfiling) {
      // Profiling executions must pass std::nullopt: the persistent
      // allocation indices may transition from absent to present only once,
      // and the profiled set is not known yet.
      ObserveAllocationAddresses(owning_buffer_allocations);
      ++remapping_->profiled_steps;
      XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
          "VA remapping: module %s profiling execution %d of %d",
          owner_->module_name_, remapping_->profiled_steps, kProfileStepsLimit);
      return execute(owning_buffer_allocations, std::nullopt);
    }

    DCHECK(remapping_->phase == Remapping::ProfilePhase::kActive);
    XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
        "VA remapping: module %s executing with %d profiled command buffer "
        "allocation(s)",
        owner_->module_name_,
        remapping_->profiled_va_remapped_alloc_indices.size());
    absl::Status execute_status = execute(
        owning_buffer_allocations,
        absl::MakeConstSpan(remapping_->profiled_persistent_alloc_indices));
    // Release the per-execution aliases even when execution failed, and
    // rewrite remapped entries to their external addresses so TearDown and
    // result handling never see reservation addresses.
    absl::Status release_status =
        ReleaseStepAliases(&owning_buffer_allocations);
    RETURN_IF_ERROR(execute_status);
    return release_status;
  }

  XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
      "VA remapping: module %s executing with %d command buffer "
      "allocation(s)",
      owner_->module_name_, owner_->va_remapped_alloc_indices_.size());
  return execute(owning_buffer_allocations,
                 absl::MakeConstSpan(owner_->persistent_alloc_indices_));
}

GpuExecutableBufferAllocator::GpuExecutableBufferAllocator(
    absl::string_view module_name,
    absl::Span<const BufferAllocation* const> allocations,
    const Shape& result_shape, const DebugOptions* debug_options,
    ThunkExecutor* thunk_executor)
    : module_name_(module_name),
      allocations_(allocations.begin(), allocations.end()),
      result_shape_(result_shape),
      debug_options_(debug_options) {
  update_mode_ = debug_options_ != nullptr
                     ? debug_options_->xla_gpu_command_buffer_update_mode()
                     : DebugOptions::ALWAYS_UPDATE;
  CHECK(update_mode_ == DebugOptions::ALWAYS_UPDATE ||
        update_mode_ == DebugOptions::SKIP_TEMP ||
        update_mode_ == DebugOptions::SKIP_PROFILED)
      << "Unsupported command buffer update mode: " << update_mode_;

  CollectedAllocationIndices indices =
      CollectAllocationIndices(allocations_, thunk_executor, update_mode_);
  constant_alloc_indices_.assign(indices.constant.begin(),
                                 indices.constant.end());
  persistent_alloc_indices_.assign(indices.persistent.begin(),
                                   indices.persistent.end());
  va_remapped_alloc_indices_ = std::move(indices.va_remapped);
  profile_candidate_alloc_indices_ = std::move(indices.profile_candidates);

  VLOG(3) << "Command buffer allocation policy: collected "
          << persistent_alloc_indices_.size()
          << " persistent allocation indices, "
          << va_remapped_alloc_indices_.size()
          << " VA-remapped allocation indices, and "
          << profile_candidate_alloc_indices_.size()
          << " profile candidate allocation indices for module "
          << module_name_;
}

void GpuExecutableBufferAllocator::TransitionProfiledRemapping(
    Remapping* remapping, se::DeviceAddressVmmAllocator* vmm_allocator,
    int device_ordinal) {
  AllocationIndexSet selected;
  // Parameter buffers are owned by the caller and can only be remapped by
  // aliasing them with Map(), which requires the exact address to be an
  // active allocator address of `vmm_allocator` and allows at most one alias
  // per allocator address. Count observed parameter addresses to drop
  // parameters that share a buffer.
  absl::flat_hash_map<const void*, int64_t> parameter_address_count;

  for (BufferAllocation::Index idx : profile_candidate_alloc_indices_) {
    if (remapping->unstable_alloc_indices.contains(idx)) {
      continue;
    }
    auto it = remapping->last_observed_address.find(idx);
    if (it == remapping->last_observed_address.end() || it->second.is_null()) {
      continue;
    }
    const BufferAllocation& allocation = *allocations_[idx];
    if (allocation.is_entry_computation_parameter()) {
      se::MemoryAllocation* raw =
          vmm_allocator->GetRawAllocation(device_ordinal, it->second);
      if (raw == nullptr) {
        XLA_VLOG_DEVICE(2, device_ordinal) << absl::StreamFormat(
            "VA remapping: module %s parameter allocation %d at %p is not an "
            "exact VMM allocator address; not remapping it",
            module_name_, idx, it->second.opaque());
        continue;
      }
      ++parameter_address_count[it->second.opaque()];
    }
    selected.insert(idx);
  }

  // Drop parameters whose observed address backs more than one selected
  // parameter allocation: Map() supports only one reservation alias per
  // allocator address.
  for (auto it = selected.begin(); it != selected.end();) {
    const BufferAllocation& allocation = *allocations_[*it];
    bool duplicate_parameter_address = false;
    if (allocation.is_entry_computation_parameter()) {
      const se::DeviceAddressBase& observed =
          remapping->last_observed_address.at(*it);
      duplicate_parameter_address =
          parameter_address_count.at(observed.opaque()) > 1;
    }
    if (duplicate_parameter_address) {
      XLA_VLOG_DEVICE(2, device_ordinal) << absl::StreamFormat(
          "VA remapping: module %s parameter allocation %d shares its buffer "
          "with another parameter; not remapping it",
          module_name_, *it);
      it = selected.erase(it);
    } else {
      ++it;
    }
  }

  if (selected.empty()) {
    remapping->phase = Remapping::ProfilePhase::kDisabled;
    XLA_VLOG_DEVICE(1, device_ordinal) << absl::StreamFormat(
        "VA remapping: module %s profile found no stable command buffer "
        "allocations; passing only constants as persistent",
        module_name_);
    return;
  }

  remapping->profiled_va_remapped_alloc_indices = std::move(selected);
  AllocationIndexSet persistent(constant_alloc_indices_.begin(),
                                constant_alloc_indices_.end());
  persistent.insert(remapping->profiled_va_remapped_alloc_indices.begin(),
                    remapping->profiled_va_remapped_alloc_indices.end());
  remapping->profiled_persistent_alloc_indices.assign(persistent.begin(),
                                                      persistent.end());
  remapping->phase = Remapping::ProfilePhase::kActive;
  XLA_VLOG_DEVICE(1, device_ordinal) << absl::StreamFormat(
      "VA remapping: module %s profile selected %d of %d command buffer "
      "allocation(s) for remapping after %d profiling execution(s)",
      module_name_, remapping->profiled_va_remapped_alloc_indices.size(),
      profile_candidate_alloc_indices_.size(), remapping->profiled_steps);
}

GpuExecutableBufferAllocator::~GpuExecutableBufferAllocator() {
  absl::MutexLock lock(remappings_mutex_);
  for (auto& [executor, remapping] : remappings_) {
    absl::MutexLock remap_lock(remapping.mutex);
    if (remapping.vmm_allocator == nullptr) {
      continue;
    }
    absl::Status status = remapping.vmm_allocator->SynchronizePendingOperations(
        executor->device_ordinal());
    if (!status.ok()) {
      LOG(ERROR) << "Failed to synchronize command buffer VA remapping "
                    "deferred operations for module "
                 << module_name_ << ": " << status;
    }
  }
}

absl::StatusOr<GpuExecutableBufferAllocator::ExecutionScope>
GpuExecutableBufferAllocator::CreateExecutionScope(
    const ServiceExecutableRunOptions* run_options,
    se::DeviceAddressAllocator* memory_allocator, int device_ordinal) {
  auto scope_without_remapping = [&] {
    return ExecutionScope(this, nullptr, nullptr, nullptr);
  };

  const bool profiled_mode = update_mode_ == DebugOptions::SKIP_PROFILED;
  if (profiled_mode ? profile_candidate_alloc_indices_.empty()
                    : va_remapped_alloc_indices_.empty()) {
    return scope_without_remapping();
  }

  auto* vmm_allocator =
      dynamic_cast<se::DeviceAddressVmmAllocator*>(memory_allocator);
  if (vmm_allocator == nullptr) {
    return scope_without_remapping();
  }

  ASSIGN_OR_RETURN(se::Stream * allocator_stream,
                   vmm_allocator->GetStream(device_ordinal));
  if (allocator_stream != run_options->stream()) {
    return Internal(
        "Command buffer VA remapping requires the VMM allocator stream "
        "and execution stream to match");
  }

  Remapping* remapping = nullptr;
  se::StreamExecutor* executor = run_options->stream()->parent();
  {
    absl::MutexLock lock(remappings_mutex_);
    // This is the lifetime remapping object for this executable/executor. It
    // owns the VA reservation reused by later ExecuteAsyncOnStream calls.
    remapping = &remappings_[executor];
  }

  auto remap_lock = std::make_unique<absl::MutexLock>(&remapping->mutex);
  if (remapping->vmm_allocator != nullptr &&
      remapping->vmm_allocator != vmm_allocator) {
    return Internal(
        "Command buffer VA remapping for module %s changed VMM allocator for "
        "executor %p",
        module_name_, executor);
  }
  remapping->vmm_allocator = vmm_allocator;

  if (profiled_mode) {
    // The lock is held through the whole execution in the profiling phase as
    // well, so concurrent executions cannot interleave profile observations
    // or race the phase transition.
    if (remapping->phase == Remapping::ProfilePhase::kInactive) {
      remapping->phase = Remapping::ProfilePhase::kProfiling;
    }
    if (remapping->phase == Remapping::ProfilePhase::kProfiling &&
        remapping->profiled_steps >= kProfileStepsLimit) {
      TransitionProfiledRemapping(remapping, vmm_allocator, device_ordinal);
    }
    if (remapping->phase == Remapping::ProfilePhase::kDisabled) {
      // Nothing to remap for this executor; behave like a scope without
      // remapping, which passes only constants as persistent.
      return scope_without_remapping();
    }
  }

  // Deferred deallocations and unmaps from the previous execution are left
  // pending on purpose: Allocate() and Map() reactivate a compatible stale
  // mapping at the same reservation VA, which keeps the same physical
  // allocation across executions and avoids waiting for the previous execution
  // to retire. Incompatible stale mappings are completed lazily and per-record
  // by the fresh-mapping paths.
  return ExecutionScope(this, remapping, vmm_allocator, std::move(remap_lock));
}

}  // namespace gpu
}  // namespace xla
