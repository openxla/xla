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
#include "absl/cleanup/cleanup.h"
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
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {
namespace {

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
  GpuExecutableBufferAllocator::AllocationIndexSet profiled_candidate;
  GpuExecutableBufferAllocator::AllocationIndexSet profiled_temp;
};

CollectedAllocationIndices CollectAllocationIndices(
    absl::Span<const BufferAllocation* const> allocations,
    const ThunkExecutor* thunk_executor, bool persist_temp_allocations,
    const absl::flat_hash_set<BufferAllocation::Index>&
        returned_output_alloc_indices) {
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
            continue;
          }
          if (allocation.is_entry_computation_parameter() ||
              returned_output_alloc_indices.contains(index)) {
            indices.profiled_candidate.insert(index);
          } else if (allocation.IsPreallocatedTempBuffer()) {
            indices.profiled_candidate.insert(index);
            indices.profiled_temp.insert(index);
            if (persist_temp_allocations) {
              indices.va_remapped.insert(index);
            }
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

GpuExecutableBufferAllocator::ExecutionScope::ExecutionScope(
    GpuExecutableBufferAllocator* owner, Remapping* remapping,
    se::DeviceAddressVmmAllocator* vmm_allocator,
    std::unique_ptr<absl::MutexLock> remap_lock, bool va_remap_enabled,
    bool profile_disabled_fallback)
    : owner_(owner),
      remapping_(remapping),
      vmm_allocator_(vmm_allocator),
      remap_lock_(std::move(remap_lock)),
      va_remap_enabled_(va_remap_enabled),
      profile_disabled_fallback_(profile_disabled_fallback) {}

GpuExecutableBufferAllocator::ExecutionScope::ExecutionScope(
    ExecutionScope&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)),
      remapping_(std::exchange(other.remapping_, nullptr)),
      vmm_allocator_(std::exchange(other.vmm_allocator_, nullptr)),
      remap_lock_(std::move(other.remap_lock_)),
      va_remap_enabled_(std::exchange(other.va_remap_enabled_, false)),
      profile_disabled_fallback_(
          std::exchange(other.profile_disabled_fallback_, false)),
      profile_observation_pending_(
          std::exchange(other.profile_observation_pending_, false)),
      device_ordinal_(std::exchange(other.device_ordinal_, -1)),
      copy_protected_alloc_indices_(
          std::move(other.copy_protected_alloc_indices_)),
      execution_aliases_(std::move(other.execution_aliases_)),
      reservation_aliases_(std::move(other.reservation_aliases_)),
      cleanup_allocator_(std::exchange(other.cleanup_allocator_, nullptr)),
      cleanup_owning_buffers_(std::move(other.cleanup_owning_buffers_)) {}

GpuExecutableBufferAllocator::ExecutionScope::~ExecutionScope() {
  absl::Status status = UnmapAliases();
  if (!status.ok()) {
    LOG(ERROR) << "Failed to release command buffer VA aliases for module "
               << (owner_ == nullptr ? "<unknown>" : owner_->module_name_)
               << ": " << status;
  }
  if (cleanup_allocator_ != nullptr) {
    for (auto it = cleanup_owning_buffers_.rbegin();
         it != cleanup_owning_buffers_.rend(); ++it) {
      absl::Status deallocate_status =
          cleanup_allocator_->Deallocate(device_ordinal_, *it);
      if (!deallocate_status.ok()) {
        LOG(ERROR) << "Failed to release an owning buffer while destroying "
                      "an execution scope for module "
                   << (owner_ == nullptr ? "<unknown>" : owner_->module_name_)
                   << ": " << deallocate_status;
      }
    }
  }
}

absl::Status GpuExecutableBufferAllocator::ExecutionScope::PrepareReservation(
    const ServiceExecutableRunOptions* run_options, int device_ordinal,
    const absl::flat_hash_map<LogicalBuffer::Color, int64_t>&
        allocate_granularity) {
  if (!va_remap_enabled()) {
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
  for (BufferAllocation::Index idx : remapping_->reservation_alloc_indices) {
    const BufferAllocation& allocation = *owner_->allocations_[idx];
    uint64_t buffer_size = allocation.size();
    if (auto it = allocate_granularity.find(allocation.color());
        it != allocate_granularity.end()) {
      buffer_size =
          RoundUpToGranularity(buffer_size, static_cast<uint64_t>(it->second));
    }
    remapping_->allocation_to_reservation_offset[idx] = remapping_->total_size;
    uint64_t mapping_size =
        RoundUpToGranularity(buffer_size, remapping_->granularity);
    remapping_->allocation_to_mapping_size[idx] = mapping_size;
    remapping_->total_size += mapping_size;
  }
  if (remapping_->total_size == 0) {
    return absl::OkStatus();
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
  if (!va_remap_enabled()) {
    return false;
  }
  if (owner_->update_mode_ == DebugOptions::SKIP_PROFILED &&
      remapping_->profile_phase == Remapping::ProfilePhase::kActive) {
    return remapping_->va_remapped_alloc_indices.contains(index);
  }
  return remapping_->reservation_alloc_indices.contains(index);
}

absl::StatusOr<se::ScopedDeviceAddress<uint8_t>>
GpuExecutableBufferAllocator::ExecutionScope::AllocateBuffer(
    int device_ordinal, const BufferAllocation& allocation, int64_t buffer_size,
    bool return_reservation_address) {
  ASSIGN_OR_RETURN(uint64_t va_offset,
                   remapping_->GetReservationOffset(allocation.index()));
  auto mapping_size_it =
      remapping_->allocation_to_mapping_size.find(allocation.index());
  if (mapping_size_it == remapping_->allocation_to_mapping_size.end()) {
    return Internal("No VA reservation size for allocation %d",
                    allocation.index());
  }
  uint64_t mapping_size = mapping_size_it->second;
  if (static_cast<uint64_t>(buffer_size) > mapping_size) {
    return Internal(
        "VA reservation for allocation %d is too small: reserved=%u, "
        "requested=%u",
        allocation.index(), mapping_size, buffer_size);
  }
  return vmm_allocator_->Allocate(
      device_ordinal, mapping_size, /*retry_on_failure=*/true,
      /*memory_space=*/allocation.color(), remapping_->va_reservation.get(),
      va_offset, mapping_size, return_reservation_address);
}

absl::StatusOr<se::DeviceAddressBase>
GpuExecutableBufferAllocator::ExecutionScope::ReservationAddress(
    BufferAllocation::Index index) const {
  if (remapping_->va_reservation == nullptr) {
    return Internal("No VA reservation for allocation %d", index);
  }
  ASSIGN_OR_RETURN(uint64_t offset, remapping_->GetReservationOffset(index));
  auto size_it = remapping_->allocation_to_mapping_size.find(index);
  if (size_it == remapping_->allocation_to_mapping_size.end()) {
    return Internal("No VA reservation size for allocation %d", index);
  }
  return remapping_->va_reservation->address().GetByteSlice(offset,
                                                            size_it->second);
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
    const bool should_remap = ShouldRemapAllocation(allocation.index());
    const bool create_output_alias =
        should_remap &&
        owner_->returned_output_alloc_indices_.contains(allocation.index());
    std::optional<se::DeviceAddressBase> reservation_address;
    uint64_t reservation_offset = 0;
    if (create_output_alias) {
      ASSIGN_OR_RETURN(reservation_address,
                       ReservationAddress(allocation.index()));
      ASSIGN_OR_RETURN(reservation_offset,
                       remapping_->GetReservationOffset(allocation.index()));
    }
    if (should_remap) {
      bool return_reservation_address = !create_output_alias;
      buffer = AllocateBuffer(device_ordinal, allocation, buffer_size,
                              return_reservation_address);
    } else {
      buffer = memory_allocator->Allocate(device_ordinal, buffer_size,
                                          /*retry_on_failure=*/true,
                                          /*memory_space=*/allocation.color());
    }
    ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> scoped_buffer,
                     std::move(buffer));
    buffer_address = scoped_buffer.Release();
    if (create_output_alias) {
      execution_aliases_[allocation.index()] = *reservation_address;
      reservation_aliases_.push_back(
          ReservationAlias{reservation_offset, reservation_address->size()});
    }
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
  device_ordinal_ = device_ordinal;

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
  std::vector<se::DeviceAddressBase> owning_buffers;
  owning_buffers.reserve(num_buffers);
  absl::Cleanup cleanup = [&] {
    absl::Status unmap_status = UnmapAliases();
    if (!unmap_status.ok()) {
      LOG(ERROR) << "Failed to release VA aliases after buffer allocation "
                    "failure for module "
                 << owner_->module_name_ << ": " << unmap_status;
    }
    for (auto it = owning_buffers.rbegin(); it != owning_buffers.rend(); ++it) {
      absl::Status deallocate_status =
          memory_allocator->Deallocate(device_ordinal, *it);
      if (!deallocate_status.ok()) {
        LOG(ERROR) << "Failed to release buffer after allocation failure for "
                      "module "
                   << owner_->module_name_ << ": " << deallocate_status;
        cleanup_allocator_ = memory_allocator;
        cleanup_owning_buffers_.push_back(*it);
      }
    }
  };
  for (int64_t i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = *owner_->allocations_[i];
    ASSIGN_OR_RETURN(
        buffers.emplace_back(),
        BufferForAllocation(get_parameter_buffer, globals, allocation,
                            memory_allocator, device_ordinal, i,
                            allocate_granularity));
    if (!allocation.is_thread_local() &&
        !allocation.is_entry_computation_parameter() &&
        !allocation.is_constant() && !buffers.back().is_null()) {
      owning_buffers.push_back(buffers.back());
    }
    RETURN_IF_ERROR(CheckAlignment(allocation, buffers.back(), i));
  }
  std::move(cleanup).Cancel();
  return BufferAllocations(buffers, device_ordinal, memory_allocator);
}

absl::StatusOr<se::DeviceAddressBase>
GpuExecutableBufferAllocator::ExecutionScope::AllocateCopyProtectedOutputBuffer(
    const ServiceExecutableRunOptions* run_options,
    BufferAllocations& buffer_allocations, const ShapeIndex& index,
    const BufferAllocation& allocation, int device_ordinal,
    se::DeviceAddressAllocator* const memory_allocator,
    absl::FunctionRef<absl::Status(absl::Status)> allocation_error) {
  if (owner_->update_mode_ == DebugOptions::SKIP_PROFILED &&
      remapping_ != nullptr &&
      remapping_->profile_phase == Remapping::ProfilePhase::kActive &&
      remapping_->va_remapped_alloc_indices.contains(allocation.index())) {
    return FailedPrecondition(
        "Command buffer profiled allocation %d became copy-protected after "
        "the persistent allocation policy was frozen for module %s",
        allocation.index(), owner_->module_name_);
  }
  if (owner_->update_mode_ == DebugOptions::SKIP_PROFILED) {
    copy_protected_alloc_indices_.insert(allocation.index());
  }

  // The caller guards this against aliasing pass-through params, as we do not
  // need to write into the output buffer in that case.
  XLA_VLOG_DEVICE(3, device_ordinal)
      << "Using copy-protection: aliasing is specified, but the "
         "buffer is not donated; allocating a fresh buffer";
  int64_t allocation_size = ShapeUtil::ByteSizeOf(
      ShapeUtil::GetSubshape(owner_->result_shape_, index));
  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> allocated_buffer =
      memory_allocator->Allocate(device_ordinal, allocation_size,
                                 /*retry_on_failure=*/true,
                                 /*memory_space=*/allocation.color());
  if (!allocated_buffer.ok()) {
    return allocation_error(allocated_buffer.status());
  }
  se::DeviceAddressBase result_buffer = allocated_buffer->cref();
  se::DeviceAddressBase& aliased_buffer =
      buffer_allocations.GetMutableDeviceAddress(allocation.index());
  CHECK_EQ(aliased_buffer.size(), result_buffer.size());
  RETURN_IF_ERROR(run_options->stream()->MemcpyD2D(
      &result_buffer, aliased_buffer, aliased_buffer.size()));
  result_buffer = allocated_buffer->Release();
  aliased_buffer = result_buffer;
  return result_buffer;
}

absl::Status GpuExecutableBufferAllocator::ExecutionScope::UnmapAliases() {
  if (reservation_aliases_.empty()) {
    return absl::OkStatus();
  }
  if (vmm_allocator_ == nullptr || remapping_ == nullptr ||
      remapping_->va_reservation == nullptr || device_ordinal_ < 0) {
    return Internal(
        "Cannot release command buffer VA aliases without an "
        "active VMM execution scope");
  }

  absl::Status first_error = absl::OkStatus();
  std::vector<ReservationAlias> aliases_still_active;
  aliases_still_active.reserve(reservation_aliases_.size());
  for (const ReservationAlias& alias : reservation_aliases_) {
    absl::Status status =
        vmm_allocator_->UnMap(device_ordinal_, remapping_->va_reservation.get(),
                              alias.offset, alias.size);
    if (!status.ok()) {
      if (first_error.ok()) {
        first_error = status;
      }
      aliases_still_active.push_back(alias);
    }
  }
  reservation_aliases_ = std::move(aliases_still_active);
  return first_error;
}

absl::Status
GpuExecutableBufferAllocator::ExecutionScope::ReleaseReservationAliases() {
  return UnmapAliases();
}

absl::Status
GpuExecutableBufferAllocator::ExecutionScope::CommitProfileObservation(
    const BufferAllocations& owning_buffer_allocations) {
  if (remapping_ == nullptr) {
    return Internal("Missing profiled command buffer state for module %s",
                    owner_->module_name_);
  }
  if (owning_buffer_allocations.size() != owner_->allocations_.size()) {
    return Internal(
        "Profiled command buffer allocation count mismatch for module %s: "
        "expected=%u actual=%u",
        owner_->module_name_, owner_->allocations_.size(),
        owning_buffer_allocations.size());
  }

  switch (remapping_->profile_phase) {
    case Remapping::ProfilePhase::kObserveFirst:
      remapping_->first_observed_addresses.assign(
          owning_buffer_allocations.buffers().begin(),
          owning_buffer_allocations.buffers().end());
      remapping_->copy_protected_alloc_indices.insert(
          copy_protected_alloc_indices_.begin(),
          copy_protected_alloc_indices_.end());
      remapping_->profile_phase = Remapping::ProfilePhase::kObserveSecond;
      XLA_VLOG_DEVICE(3, owning_buffer_allocations.device_ordinal())
          << "Command buffer allocation profiling completed first "
             "observation for module "
          << owner_->module_name_;
      return absl::OkStatus();

    case Remapping::ProfilePhase::kObserveSecond:
      remapping_->copy_protected_alloc_indices.insert(
          copy_protected_alloc_indices_.begin(),
          copy_protected_alloc_indices_.end());
      remapping_->reservation_alloc_indices.clear();
      for (BufferAllocation::Index index :
           owner_->profiled_candidate_alloc_indices_) {
        if (remapping_->copy_protected_alloc_indices.contains(index)) {
          continue;
        }
        if (owner_->profiled_temp_alloc_indices_.contains(index)) {
          remapping_->reservation_alloc_indices.insert(index);
          continue;
        }
        const se::DeviceAddressBase& first =
            remapping_->first_observed_addresses[index];
        se::DeviceAddressBase current =
            owning_buffer_allocations.GetDeviceAddress(index);
        if (!first.is_null() && first.IsSameAs(current)) {
          remapping_->reservation_alloc_indices.insert(index);
        }
      }
      remapping_->first_observed_addresses.clear();
      remapping_->profile_phase = Remapping::ProfilePhase::kActivating;
      XLA_VLOG_DEVICE(3, owning_buffer_allocations.device_ordinal())
          << absl::StreamFormat(
                 "Command buffer allocation profiling selected %d "
                 "allocation(s) for activation in module %s",
                 remapping_->reservation_alloc_indices.size(),
                 owner_->module_name_);
      return absl::OkStatus();

    case Remapping::ProfilePhase::kActivating:
    case Remapping::ProfilePhase::kActive:
    case Remapping::ProfilePhase::kDisabled:
      return FailedPrecondition(
          "Command buffer allocation profiling is not observing for module "
          "%s",
          owner_->module_name_);
  }
}

absl::StatusOr<BufferAllocations>
GpuExecutableBufferAllocator::ExecutionScope::BuildExecutionBufferAllocations(
    const BufferAllocations& owning_buffer_allocations, int device_ordinal) {
  CHECK(owner_->update_mode_ == DebugOptions::SKIP_PROFILED);
  CHECK(remapping_ != nullptr);
  const bool activating =
      remapping_->profile_phase == Remapping::ProfilePhase::kActivating;
  const bool active =
      remapping_->profile_phase == Remapping::ProfilePhase::kActive;
  if (!activating && !active) {
    return FailedPrecondition(
        "Command buffer allocation remapping is not active for module %s",
        owner_->module_name_);
  }

  std::vector<se::DeviceAddressBase> execution_buffers(
      owning_buffer_allocations.buffers().begin(),
      owning_buffer_allocations.buffers().end());
  AllocationIndexSet selected_alloc_indices;
  const AllocationIndexSet& candidates =
      active ? remapping_->va_remapped_alloc_indices
             : remapping_->reservation_alloc_indices;

  struct MappedSource {
    BufferAllocation::Index canonical_index;
    se::DeviceAddressBase source;
    se::DeviceAddressBase reservation_alias;
  };
  std::vector<MappedSource> mapped_sources;
  if (activating) {
    remapping_->parameter_alias_canonical_indices.clear();
  }

  for (BufferAllocation::Index index : candidates) {
    const BufferAllocation& allocation = *owner_->allocations_[index];
    if (copy_protected_alloc_indices_.contains(index)) {
      if (active) {
        return FailedPrecondition(
            "Command buffer profiled allocation %d became copy-protected "
            "after the persistent allocation policy was frozen for module %s",
            index, owner_->module_name_);
      }
      continue;
    }

    if (allocation.is_entry_computation_parameter()) {
      se::DeviceAddressBase source =
          owning_buffer_allocations.GetDeviceAddress(index);
      auto matching_source = mapped_sources.end();

      if (active) {
        auto canonical_it =
            remapping_->parameter_alias_canonical_indices.find(index);
        if (canonical_it ==
            remapping_->parameter_alias_canonical_indices.end()) {
          return Internal(
              "Profiled parameter allocation %d has no frozen alias group "
              "for module %s",
              index, owner_->module_name_);
        }
        BufferAllocation::Index canonical_index = canonical_it->second;
        for (auto it = mapped_sources.begin(); it != mapped_sources.end();
             ++it) {
          if (it->canonical_index == canonical_index) {
            matching_source = it;
            break;
          }
        }
        if (canonical_index != index) {
          if (matching_source == mapped_sources.end() ||
              !matching_source->source.IsSameAs(source)) {
            return FailedPrecondition(
                "Aliasing of profiled parameter allocation %d changed after "
                "the persistent allocation policy was frozen for module %s",
                index, owner_->module_name_);
          }
          execution_buffers[index] = matching_source->reservation_alias;
          selected_alloc_indices.insert(index);
          continue;
        }
        for (auto it = mapped_sources.begin(); it != mapped_sources.end();
             ++it) {
          if (it->source.IsSameAs(source)) {
            return FailedPrecondition(
                "Aliasing of profiled parameter allocation %d changed after "
                "the persistent allocation policy was frozen for module %s",
                index, owner_->module_name_);
          }
        }
      } else {
        for (auto it = mapped_sources.begin(); it != mapped_sources.end();
             ++it) {
          if (it->source.IsSameAs(source)) {
            matching_source = it;
            break;
          }
        }
      }

      if (matching_source != mapped_sources.end()) {
        execution_buffers[index] = matching_source->reservation_alias;
        selected_alloc_indices.insert(index);
        remapping_->parameter_alias_canonical_indices[index] =
            matching_source->canonical_index;
        continue;
      }

      auto mapping_size_it = remapping_->allocation_to_mapping_size.find(index);
      if (mapping_size_it == remapping_->allocation_to_mapping_size.end()) {
        return Internal("No VA reservation size for allocation %d", index);
      }
      uint64_t mapping_size = mapping_size_it->second;
      if (activating) {
        ASSIGN_OR_RETURN(bool can_map,
                         vmm_allocator_->CanMapAsNewReservationAlias(
                             device_ordinal, source, mapping_size));
        if (!can_map) {
          XLA_VLOG_DEVICE(3, device_ordinal)
              << "Command buffer allocation profiling excluded unsupported "
                 "parameter allocation "
              << index << " in module " << owner_->module_name_;
          continue;
        }
      }

      ASSIGN_OR_RETURN(uint64_t offset,
                       remapping_->GetReservationOffset(index));
      ASSIGN_OR_RETURN(se::DeviceAddressBase reservation_alias,
                       ReservationAddress(index));
      RETURN_IF_ERROR(vmm_allocator_->Map(device_ordinal, source,
                                          remapping_->va_reservation.get(),
                                          offset, mapping_size));
      reservation_aliases_.push_back(ReservationAlias{offset, mapping_size});
      execution_buffers[index] = reservation_alias;
      mapped_sources.push_back(MappedSource{index, source, reservation_alias});
      selected_alloc_indices.insert(index);
      if (activating) {
        remapping_->parameter_alias_canonical_indices[index] = index;
      }
      continue;
    }

    if (owner_->returned_output_alloc_indices_.contains(index)) {
      auto alias = execution_aliases_.find(index);
      if (alias == execution_aliases_.end()) {
        return Internal(
            "VA-remapped output allocation %d has no reservation alias for "
            "module %s",
            index, owner_->module_name_);
      }
      execution_buffers[index] = alias->second;
      selected_alloc_indices.insert(index);
      continue;
    }

    if (allocation.IsPreallocatedTempBuffer()) {
      ASSIGN_OR_RETURN(se::DeviceAddressBase expected,
                       ReservationAddress(index));
      if (!execution_buffers[index].IsSameAs(expected)) {
        return Internal(
            "VA-remapped temp allocation %d has unexpected address for "
            "module %s: expected=%p actual=%p",
            index, owner_->module_name_, expected.opaque(),
            execution_buffers[index].opaque());
      }
      selected_alloc_indices.insert(index);
      continue;
    }

    return Internal("Unsupported profiled allocation %d for module %s", index,
                    owner_->module_name_);
  }

  if (activating) {
    remapping_->copy_protected_alloc_indices.insert(
        copy_protected_alloc_indices_.begin(),
        copy_protected_alloc_indices_.end());
    remapping_->va_remapped_alloc_indices = selected_alloc_indices;
    AllocationIndexSet persistent(owner_->constant_alloc_indices_.begin(),
                                  owner_->constant_alloc_indices_.end());
    persistent.insert(selected_alloc_indices.begin(),
                      selected_alloc_indices.end());
    remapping_->persistent_alloc_indices.assign(persistent.begin(),
                                                persistent.end());
    remapping_->profile_phase = Remapping::ProfilePhase::kActive;
    XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
        "Command buffer allocation profiling activated %d persistent "
        "allocation(s), including %d VA-remapped allocation(s), for module %s",
        remapping_->persistent_alloc_indices.size(),
        selected_alloc_indices.size(), owner_->module_name_);
  } else if (selected_alloc_indices != remapping_->va_remapped_alloc_indices) {
    return Internal(
        "Profiled command buffer allocation set changed after activation for "
        "module %s",
        owner_->module_name_);
  }

  return BufferAllocations(execution_buffers, device_ordinal,
                           owning_buffer_allocations.memory_allocator());
}

absl::Status
GpuExecutableBufferAllocator::ExecutionScope::ExecuteWithBufferAllocations(
    const BufferAllocations& owning_buffer_allocations, int device_ordinal,
    absl::FunctionRef<
        absl::Status(const BufferAllocations&,
                     std::optional<absl::Span<const BufferAllocation::Index>>
                         persistent_alloc_indices)>
        execute) {
  device_ordinal_ = device_ordinal;
  if (owner_->update_mode_ != DebugOptions::SKIP_PROFILED) {
    if (!va_remap_enabled()) {
      return execute(owning_buffer_allocations,
                     absl::MakeConstSpan(owner_->constant_alloc_indices_));
    }
    XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
        "VA remapping: module %s executing with %d command buffer "
        "allocation(s)",
        owner_->module_name_, owner_->va_remapped_alloc_indices_.size());
    return execute(owning_buffer_allocations,
                   absl::MakeConstSpan(owner_->persistent_alloc_indices_));
  }

  if (profile_disabled_fallback_ || remapping_ == nullptr ||
      remapping_->profile_phase == Remapping::ProfilePhase::kDisabled) {
    return execute(owning_buffer_allocations,
                   absl::MakeConstSpan(owner_->constant_alloc_indices_));
  }

  if (remapping_->profile_phase == Remapping::ProfilePhase::kObserveFirst ||
      remapping_->profile_phase == Remapping::ProfilePhase::kObserveSecond) {
    absl::Status execute_status =
        execute(owning_buffer_allocations, std::nullopt);
    if (execute_status.ok()) {
      profile_observation_pending_ = true;
    }
    return execute_status;
  }

  absl::StatusOr<BufferAllocations> execution_buffer_allocations =
      BuildExecutionBufferAllocations(owning_buffer_allocations,
                                      device_ordinal);
  if (!execution_buffer_allocations.ok()) {
    absl::Status unmap_status = UnmapAliases();
    if (!unmap_status.ok()) {
      LOG(ERROR) << "Failed to release command buffer VA aliases after "
                    "activation failure for module "
                 << owner_->module_name_ << ": " << unmap_status;
    }
    return execution_buffer_allocations.status();
  }

  absl::Status execute_status =
      execute(*execution_buffer_allocations,
              absl::MakeConstSpan(remapping_->persistent_alloc_indices));
  absl::Status unmap_status = UnmapAliases();
  if (!execute_status.ok()) {
    if (!unmap_status.ok()) {
      LOG(ERROR) << "Failed to release command buffer VA aliases after "
                    "execution failure for module "
                 << owner_->module_name_ << ": " << unmap_status;
    }
    return execute_status;
  }
  return unmap_status;
}

absl::Status
GpuExecutableBufferAllocator::ExecutionScope::CommitSuccessfulExecution(
    const BufferAllocations& owning_buffer_allocations) {
  if (!profile_observation_pending_) {
    return absl::OkStatus();
  }
  RETURN_IF_ERROR(CommitProfileObservation(owning_buffer_allocations));
  profile_observation_pending_ = false;
  return absl::OkStatus();
}

GpuExecutableBufferAllocator::GpuExecutableBufferAllocator(
    absl::string_view module_name,
    absl::Span<const BufferAllocation* const> allocations,
    const Shape& result_shape, const DebugOptions* debug_options,
    ThunkExecutor* thunk_executor,
    absl::Span<const BufferAllocation::Index> returned_output_alloc_indices)
    : module_name_(module_name),
      allocations_(allocations.begin(), allocations.end()),
      result_shape_(result_shape),
      debug_options_(debug_options),
      returned_output_alloc_indices_(returned_output_alloc_indices.begin(),
                                     returned_output_alloc_indices.end()) {
  update_mode_ = debug_options_ != nullptr
                     ? debug_options_->xla_gpu_command_buffer_update_mode()
                     : DebugOptions::ALWAYS_UPDATE;
  CHECK(update_mode_ == DebugOptions::ALWAYS_UPDATE ||
        update_mode_ == DebugOptions::SKIP_TEMP ||
        update_mode_ == DebugOptions::SKIP_PROFILED)
      << "Unsupported command buffer update mode: " << update_mode_;

  CollectedAllocationIndices indices = CollectAllocationIndices(
      allocations_, thunk_executor, update_mode_ == DebugOptions::SKIP_TEMP,
      returned_output_alloc_indices_);
  constant_alloc_indices_.assign(indices.constant.begin(),
                                 indices.constant.end());
  persistent_alloc_indices_.assign(indices.persistent.begin(),
                                   indices.persistent.end());
  va_remapped_alloc_indices_ = std::move(indices.va_remapped);
  profiled_candidate_alloc_indices_ = std::move(indices.profiled_candidate);
  profiled_temp_alloc_indices_ = std::move(indices.profiled_temp);

  VLOG(3) << "Command buffer allocation policy: collected "
          << persistent_alloc_indices_.size()
          << " persistent allocation indices and "
          << va_remapped_alloc_indices_.size()
          << " statically VA-remapped allocation indices and "
          << profiled_candidate_alloc_indices_.size()
          << " profiled candidate allocation indices for module "
          << module_name_;
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
  auto scope_without_remapping = [&](bool profile_disabled_fallback = false) {
    return ExecutionScope(this, nullptr, nullptr, nullptr,
                          /*va_remap_enabled=*/false,
                          profile_disabled_fallback);
  };

  if (update_mode_ == DebugOptions::ALWAYS_UPDATE ||
      (update_mode_ == DebugOptions::SKIP_TEMP &&
       va_remapped_alloc_indices_.empty())) {
    return scope_without_remapping();
  }

  auto* vmm_allocator =
      dynamic_cast<se::DeviceAddressVmmAllocator*>(memory_allocator);
  if (vmm_allocator == nullptr) {
    if (update_mode_ != DebugOptions::SKIP_PROFILED) {
      return scope_without_remapping();
    }

    se::Stream* stream =
        run_options == nullptr ? nullptr : run_options->stream();
    if (stream == nullptr) {
      LOG(WARNING) << "Disabling profiled command buffer VA remapping for "
                      "module "
                   << module_name_ << " because no VMM allocator is available";
      return scope_without_remapping(/*profile_disabled_fallback=*/true);
    }

    Remapping* remapping = nullptr;
    se::StreamExecutor* executor = stream->parent();
    {
      absl::MutexLock lock(remappings_mutex_);
      remapping = &remappings_[executor];
    }
    auto remap_lock = std::make_unique<absl::MutexLock>(&remapping->mutex);
    if (remapping->vmm_allocator != nullptr ||
        remapping->profile_phase != Remapping::ProfilePhase::kObserveFirst) {
      if (remapping->profile_phase == Remapping::ProfilePhase::kDisabled) {
        return ExecutionScope(this, remapping, nullptr, std::move(remap_lock),
                              /*va_remap_enabled=*/false);
      }
      return Internal(
          "Profiled command buffer allocator for module %s changed from VMM "
          "to a non-VMM allocator",
          module_name_);
    }
    remapping->profile_phase = Remapping::ProfilePhase::kDisabled;
    LOG(WARNING) << "Disabling profiled command buffer VA remapping for module "
                 << module_name_ << " because no VMM allocator is available";
    return ExecutionScope(this, remapping, nullptr, std::move(remap_lock),
                          /*va_remap_enabled=*/false);
  }

  if (run_options == nullptr || run_options->stream() == nullptr) {
    return InvalidArgument(
        "Command buffer VA remapping requires an execution stream");
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
  if (update_mode_ == DebugOptions::SKIP_PROFILED &&
      remapping->profile_phase == Remapping::ProfilePhase::kDisabled) {
    return ExecutionScope(this, remapping, nullptr, std::move(remap_lock),
                          /*va_remap_enabled=*/false);
  }
  if (remapping->vmm_allocator != nullptr &&
      remapping->vmm_allocator != vmm_allocator) {
    return Internal(
        "Command buffer VA remapping for module %s changed VMM allocator for "
        "executor %p",
        module_name_, executor);
  }
  remapping->vmm_allocator = vmm_allocator;

  if (update_mode_ == DebugOptions::SKIP_TEMP) {
    if (remapping->reservation_alloc_indices.empty()) {
      remapping->reservation_alloc_indices = va_remapped_alloc_indices_;
      remapping->va_remapped_alloc_indices = va_remapped_alloc_indices_;
      remapping->persistent_alloc_indices = persistent_alloc_indices_;
    } else if (remapping->reservation_alloc_indices !=
               va_remapped_alloc_indices_) {
      return Internal(
          "Command buffer VA remapping allocation set changed for module %s",
          module_name_);
    }
    return ExecutionScope(this, remapping, vmm_allocator, std::move(remap_lock),
                          /*va_remap_enabled=*/true);
  }

  bool va_remap_enabled = false;
  if (remapping->profile_phase == Remapping::ProfilePhase::kActivating) {
    va_remap_enabled = !remapping->reservation_alloc_indices.empty();
  } else if (remapping->profile_phase == Remapping::ProfilePhase::kActive) {
    va_remap_enabled = !remapping->va_remapped_alloc_indices.empty();
  }

  // Deferred deallocations and unmaps from the previous execution are left
  // pending on purpose: Allocate() and Map() reactivate compatible stale
  // mappings at the same reservation VA, avoiding a stream wait and fresh VMM
  // calls when physical allocations are reused on the next execution.
  return ExecutionScope(this, remapping, vmm_allocator, std::move(remap_lock),
                        va_remap_enabled);
}

}  // namespace gpu
}  // namespace xla
