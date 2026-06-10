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

#include "xla/service/gpu/command_buffer_va_remapping.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/vmm_device_address_allocator.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

uint64_t RoundUpToGranularity(uint64_t size, uint64_t granularity) {
  if (granularity == 0) {
    return size;
  }
  return ((size + granularity - 1) / granularity) * granularity;
}

}  // namespace

absl::StatusOr<std::unique_ptr<CommandBufferVaRemapping>>
CommandBufferVaRemapping::Create(
    DebugOptions::CommandBufferUpdateMode update_mode,
    ThunkExecutor* thunk_executor,
    absl::Span<const BufferAllocation* const> allocations,
    absl::string_view module_name) {
  absl::btree_set<BufferAllocation::Index> update_allocation_indices;
  absl::btree_set<BufferAllocation::Index> allocation_indices;

  if (thunk_executor != nullptr &&
      update_mode == DebugOptions::TEMP_NEVER_UPDATE) {
    RETURN_IF_ERROR(thunk_executor->thunks().WalkNested(
        [&](const Thunk* thunk) -> absl::Status {
          auto* command_buffer_thunk =
              dynamic_cast<const CommandBufferThunk*>(thunk);
          if (command_buffer_thunk == nullptr) {
            return absl::OkStatus();
          }
          return command_buffer_thunk->WalkCommands(
              [&](const Command* command) -> absl::Status {
                for (const BufferUse& use : command->buffer_uses()) {
                  BufferAllocation::Index index = use.slice().index();
                  bool is_preallocated_temp_buffer = false;
                  if (index >= 0 &&
                      static_cast<size_t>(index) < allocations.size()) {
                    const BufferAllocation& allocation = *allocations[index];
                    if (allocation.is_constant() || allocation.size() == 0) {
                      continue;
                    }
                    is_preallocated_temp_buffer =
                        allocation.IsPreallocatedTempBuffer();
                  }
                  update_allocation_indices.insert(index);
                  if (is_preallocated_temp_buffer) {
                    allocation_indices.insert(index);
                  }
                }
                return absl::OkStatus();
              });
        }));
    VLOG(3) << "VA remapping: collected " << allocation_indices.size()
            << " VA-remap allocation indexes and "
            << update_allocation_indices.size()
            << " update allocation indexes for module " << module_name;
  }

  return std::unique_ptr<CommandBufferVaRemapping>(new CommandBufferVaRemapping(
      update_mode, std::string(module_name),
      std::move(update_allocation_indices), std::move(allocation_indices)));
}

CommandBufferVaRemapping::CommandBufferVaRemapping(
    DebugOptions::CommandBufferUpdateMode update_mode, std::string module_name,
    absl::btree_set<BufferAllocation::Index> update_allocation_indices,
    absl::btree_set<BufferAllocation::Index> allocation_indices)
    : update_mode_(update_mode),
      module_name_(std::move(module_name)),
      update_allocation_indices_(std::move(update_allocation_indices)),
      allocation_indices_(std::move(allocation_indices)) {}

CommandBufferVaRemapping::~CommandBufferVaRemapping() {
  absl::MutexLock lock(va_remaps_mutex_);
  for (auto& [executor, va_remap] : va_remaps_) {
    absl::MutexLock remap_lock(va_remap.mutex);
    if (va_remap.vmm_allocator == nullptr) {
      continue;
    }
    absl::Status status = va_remap.vmm_allocator->SynchronizePendingOperations(
        executor->device_ordinal());
    if (!status.ok()) {
      LOG(ERROR) << "Failed to synchronize command buffer VA remapping "
                    "deferred operations for module "
                 << module_name_ << ": " << status;
    }
  }
}

absl::StatusOr<uint64_t>
CommandBufferVaRemapping::VaRemapping::GetReservationOffset(
    BufferAllocation::Index idx) const {
  auto it = allocation_to_reservation_offset.find(idx);
  if (it == allocation_to_reservation_offset.end()) {
    return Internal("No VA reservation offset for allocation %d", idx);
  }
  return it->second;
}

absl::StatusOr<CommandBufferVaRemapping::MemoryReservationAlias>
CommandBufferVaRemapping::ScopedExecution::GetReservationAlias(
    BufferAllocation::Index idx) const {
  auto it = allocation_to_reservation_aliases_.find(idx);
  if (it == allocation_to_reservation_aliases_.end()) {
    return Internal("No VA reservation alias for allocation %d", idx);
  }
  return it->second;
}

absl::StatusOr<std::unique_ptr<CommandBufferVaRemapping::ScopedExecution>>
CommandBufferVaRemapping::BeginExecution(
    const ServiceExecutableRunOptions* run_options,
    se::DeviceAddressAllocator* const memory_allocator, int device_ordinal) {
  if (update_mode_ == DebugOptions::ALWAYS_UPDATE) {
    return nullptr;
  }
  if (update_mode_ != DebugOptions::TEMP_NEVER_UPDATE) {
    return Internal("Unsupported command buffer update mode: %d", update_mode_);
  }
  if (allocation_indices_.empty()) {
    return nullptr;
  }

  auto* vmm_allocator =
      dynamic_cast<se::DeviceAddressVmmAllocator*>(memory_allocator);
  if (vmm_allocator == nullptr) {
    return nullptr;
  }

  ASSIGN_OR_RETURN(se::Stream * allocator_stream,
                   vmm_allocator->GetStream(device_ordinal));
  if (allocator_stream != run_options->stream()) {
    return Internal(
        "Command buffer VA remapping requires the VMM allocator stream "
        "and execution stream to match");
  }

  VaRemapping* va_remap = nullptr;
  se::StreamExecutor* executor = run_options->stream()->parent();
  {
    absl::MutexLock lock(va_remaps_mutex_);
    // This is the lifetime remapping object for this executable/executor. It
    // owns the VA reservation reused by later ExecuteAsyncOnStream calls.
    va_remap = &va_remaps_[executor];
  }

  auto va_remap_lock = std::make_unique<absl::MutexLock>(&va_remap->mutex);
  if (va_remap->vmm_allocator != nullptr &&
      va_remap->vmm_allocator != vmm_allocator) {
    return Internal(
        "Command buffer VA remapping for module %s changed VMM allocator for "
        "executor %p",
        module_name_, executor);
  }
  va_remap->vmm_allocator = vmm_allocator;
  return std::make_unique<ScopedExecution>(*va_remap, *vmm_allocator,
                                           std::move(va_remap_lock));
}

absl::Status CommandBufferVaRemapping::PrepareReservation(
    const ServiceExecutableRunOptions* run_options, int device_ordinal,
    absl::Span<const BufferAllocation* const> allocations,
    const absl::flat_hash_map<LogicalBuffer::Color, int64_t>&
        allocate_granularity,
    ScopedExecution* execution) {
  if (execution == nullptr) {
    return absl::OkStatus();
  }

  VaRemapping& va_remap = execution->remapping_;
  uint64_t granularity = execution->vmm_allocator_.GetAllocationGranularity(
      run_options->stream()->parent());
  if (va_remap.va_reservation != nullptr &&
      va_remap.granularity != granularity) {
    return Internal(
        "Command buffer VA remapping granularity changed for module %s: "
        "previous=%u current=%u",
        module_name_, va_remap.granularity, granularity);
  }
  if (va_remap.va_reservation != nullptr) {
    return absl::OkStatus();
  }

  // First execution on this executor creates the persistent reservation. Later
  // executions reuse the same reservation and deterministic layout.
  va_remap.granularity = granularity;
  va_remap.total_size = 0;
  va_remap.allocation_to_reservation_offset.clear();
  for (BufferAllocation::Index idx : allocation_indices_) {
    const BufferAllocation& allocation = *allocations[idx];
    uint64_t buffer_size = allocation.size();
    if (auto it = allocate_granularity.find(allocation.color());
        it != allocate_granularity.end()) {
      buffer_size =
          RoundUpToGranularity(buffer_size, static_cast<uint64_t>(it->second));
    }
    va_remap.allocation_to_reservation_offset[idx] = va_remap.total_size;
    va_remap.total_size =
        va_remap.total_size +
        RoundUpToGranularity(buffer_size, va_remap.granularity);
  }
  ASSIGN_OR_RETURN(va_remap.va_reservation,
                   execution->vmm_allocator_.CreateReservation(
                       run_options->stream()->parent(), va_remap.total_size));
  XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
      "VA remapping: reserved range for module %s VA=%p total_size=%u "
      "granularity=%u",
      module_name_, va_remap.va_reservation->address().opaque(),
      va_remap.total_size, va_remap.granularity);
  return absl::OkStatus();
}

absl::StatusOr<se::ScopedDeviceAddress<uint8_t>>
CommandBufferVaRemapping::Allocate(int device_ordinal,
                                   const BufferAllocation& allocation,
                                   int64_t buffer_size,
                                   bool return_reservation_address,
                                   ScopedExecution& execution) {
  VaRemapping& va_remap = execution.remapping_;
  ASSIGN_OR_RETURN(uint64_t va_offset,
                   va_remap.GetReservationOffset(allocation.index()));
  uint64_t mapping_size = RoundUpToGranularity(
      static_cast<uint64_t>(buffer_size), va_remap.granularity);
  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> buffer =
      execution.vmm_allocator_.Allocate(
          device_ordinal, mapping_size, /*retry_on_failure=*/true,
          /*memory_space=*/allocation.color(), va_remap.va_reservation.get(),
          va_offset, mapping_size, return_reservation_address);
  if (buffer.ok() && !return_reservation_address) {
    se::DeviceAddressBase reservation_address =
        va_remap.va_reservation->address().GetByteSlice(va_offset,
                                                        mapping_size);
    execution.allocation_to_reservation_aliases_[allocation.index()] =
        MemoryReservationAlias{va_offset, mapping_size, reservation_address};
  }
  return buffer;
}

bool CommandBufferVaRemapping::ShouldRemapAllocation(
    BufferAllocation::Index index, const ScopedExecution* execution) const {
  if (execution == nullptr || !allocation_indices_.contains(index)) {
    return false;
  }
  const VaRemapping& va_remap = execution->remapping_;
  if (!va_remap.update_policy_ready) {
    return true;
  }
  return va_remap.policy_va_remapped_index_set.contains(index);
}

absl::Status CommandBufferVaRemapping::UpdateAllocationPolicy(
    ScopedExecution& execution) {
  VaRemapping& va_remap = execution.remapping_;
  if (update_mode_ == DebugOptions::ALWAYS_UPDATE ||
      va_remap.update_policy_ready) {
    return absl::OkStatus();
  }

  if (update_mode_ != DebugOptions::TEMP_NEVER_UPDATE) {
    return Internal("Unsupported command buffer update mode: %d", update_mode_);
  }

  va_remap.policy_va_remapped_indices.assign(allocation_indices_.begin(),
                                             allocation_indices_.end());
  va_remap.policy_va_remapped_index_set = allocation_indices_;
  va_remap.policy_dynamic_alloc_indices.clear();
  absl::c_set_difference(
      update_allocation_indices_, allocation_indices_,
      std::back_inserter(va_remap.policy_dynamic_alloc_indices));
  va_remap.update_policy_ready = true;
  return absl::OkStatus();
}

Thunk::CommandBufferUpdateInfo
CommandBufferVaRemapping::GetCommandBufferUpdateInfo(
    const ScopedExecution& execution) const {
  const VaRemapping& va_remap = execution.remapping_;
  return Thunk::CommandBufferUpdateInfo{
      va_remap.update_policy_ready,
      absl::MakeConstSpan(va_remap.policy_va_remapped_indices),
      absl::MakeConstSpan(va_remap.policy_dynamic_alloc_indices)};
}

absl::StatusOr<BufferAllocations>
CommandBufferVaRemapping::BuildBufferAllocations(
    const BufferAllocations& owning_buffer_allocations, int device_ordinal,
    absl::Span<const BufferAllocation* const> allocations,
    ScopedExecution& execution) {
  VaRemapping& va_remap = execution.remapping_;
  std::vector<se::DeviceAddressBase> execution_buffers;
  execution_buffers.reserve(owning_buffer_allocations.size());

  struct SourceMapping {
    se::DeviceAddressBase source_address;
    MemoryReservationAlias alias;
  };
  std::vector<SourceMapping> source_mappings;

  for (BufferAllocation::Index i = 0; i < owning_buffer_allocations.size();
       ++i) {
    se::DeviceAddressBase owning_address =
        owning_buffer_allocations.GetDeviceAddress(i);
    if (!ShouldRemapAllocation(i, &execution)) {
      execution_buffers.push_back(owning_address);
      continue;
    }
    if (owning_address.is_null()) {
      return Internal("Command buffer allocation %d has null address", i);
    }

    ASSIGN_OR_RETURN(uint64_t va_offset, va_remap.GetReservationOffset(i));

    if (execution.allocation_to_reservation_aliases_.contains(i)) {
      ASSIGN_OR_RETURN(MemoryReservationAlias alias,
                       execution.GetReservationAlias(i));
      execution_buffers.push_back(alias.reservation_address);
      execution.aliases_to_unmap_.push_back(alias);
      continue;
    }

    const BufferAllocation& allocation = *allocations[i];
    if (!allocation.is_entry_computation_parameter()) {
      se::DeviceAddressBase expected_reservation_address =
          va_remap.va_reservation->address().GetByteSlice(
              va_offset, owning_address.size());
      if (!owning_address.IsSameAs(expected_reservation_address)) {
        return Internal(
            "Command buffer allocation %d expected reservation-backed "
            "allocator address %p but got %p",
            i, expected_reservation_address.opaque(), owning_address.opaque());
      }
      execution_buffers.push_back(owning_address);
      continue;
    }

    bool reused_source_mapping = false;
    for (const SourceMapping& source_mapping : source_mappings) {
      if (source_mapping.source_address.IsSameAs(owning_address)) {
        execution_buffers.push_back(source_mapping.alias.reservation_address);
        reused_source_mapping = true;
        break;
      }
    }
    if (reused_source_mapping) {
      continue;
    }

    uint64_t mapping_size =
        RoundUpToGranularity(owning_address.size(), va_remap.granularity);
    MemoryReservationAlias alias{
        va_offset, mapping_size,
        va_remap.va_reservation->address().GetByteSlice(va_offset,
                                                        mapping_size)};
    RETURN_IF_ERROR(execution.vmm_allocator_.Map(
        device_ordinal, owning_address, va_remap.va_reservation.get(),
        alias.reservation_offset, alias.size));
    XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
        "VA remapping: Mapped allocation %d for module %s from %p to %p "
        "size=%u",
        i, module_name_, owning_address.opaque(),
        alias.reservation_address.opaque(), alias.size);
    execution.aliases_to_unmap_.push_back(alias);
    source_mappings.push_back(SourceMapping{owning_address, alias});
    execution_buffers.push_back(alias.reservation_address);
  }

  return BufferAllocations(execution_buffers,
                           owning_buffer_allocations.device_ordinal(),
                           owning_buffer_allocations.memory_allocator());
}

absl::Status CommandBufferVaRemapping::UnmapAliases(
    int device_ordinal, ScopedExecution& execution) {
  VaRemapping& va_remap = execution.remapping_;
  absl::Status status;
  absl::flat_hash_set<void*> unmapped_aliases;
  auto unmap_alias = [&](const MemoryReservationAlias& alias) {
    if (alias.reservation_address.is_null()) {
      return;
    }
    if (!unmapped_aliases.insert(alias.reservation_address.opaque()).second) {
      return;
    }
    absl::Status unmap_status = execution.vmm_allocator_.UnMap(
        device_ordinal, va_remap.va_reservation.get(), alias.reservation_offset,
        alias.size);
    if (!unmap_status.ok() && status.ok()) {
      status = unmap_status;
    }
  };

  for (const MemoryReservationAlias& alias : execution.aliases_to_unmap_) {
    unmap_alias(alias);
  }
  for (const auto& [_, alias] : execution.allocation_to_reservation_aliases_) {
    unmap_alias(alias);
  }
  return status;
}

}  // namespace xla::gpu
