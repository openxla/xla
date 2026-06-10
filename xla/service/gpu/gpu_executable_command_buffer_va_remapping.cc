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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable.h"
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
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla::gpu {

using ::tsl::profiler::ScopedAnnotation;

void GpuExecutable::CollectCommandBufferAllocationIndexes() {
  // Populate command_buffer_allocation_indexes_ with buffer indices accessed by
  // command buffer thunks. Skip constant and zero-size allocations since they
  // don't need VA remapping (constants are allocated as global values with
  // fixed addresses; zero-size allocations have nothing to map).
  //
  // The set of collected indices depends on xla_gpu_command_buffer_update_mode:
  //   ALWAYS_UPDATE - collect nothing (VA remapping disabled)
  //   NEVER_UPDATE - collect all allocations from all command buffer
  //     commands
  //   CAPTURE_CMD_NEVER_UPDATE - collect only allocations from traced
  //     commands, including collective commands recorded by CollectiveThunk
  if (thunk_executor_) {
    DebugOptions::CommandBufferUpdateMode update_mode =
        has_module() ? module_config()
                           .debug_options()
                           .xla_gpu_command_buffer_update_mode()
                     : DebugOptions::ALWAYS_UPDATE;

    if (update_mode == DebugOptions::NEVER_UPDATE ||
        update_mode == DebugOptions::CAPTURE_CMD_NEVER_UPDATE) {
      CHECK_OK(thunk_executor_->thunks().WalkNested(
          [&](const Thunk* t) -> absl::Status {
            auto* cbt = dynamic_cast<const CommandBufferThunk*>(t);
            if (cbt == nullptr) return absl::OkStatus();
            return cbt->WalkCommands([&](const Command* cmd) -> absl::Status {
              if (update_mode == DebugOptions::CAPTURE_CMD_NEVER_UPDATE &&
                  !cmd->IsTracedCommand()) {
                return absl::OkStatus();
              }
              for (const BufferUse& use : cmd->buffer_uses()) {
                BufferAllocation::Index index = use.slice().index();
                if (index >= 0 && index < allocation_ptrs_.size()) {
                  const BufferAllocation* alloc = allocation_ptrs_[index];
                  if (alloc->is_constant() || alloc->size() == 0) continue;
                }
                command_buffer_allocation_indexes_.insert(index);
              }
              return absl::OkStatus();
            });
          }));
      VLOG(3) << "VA remapping: collected "
              << command_buffer_allocation_indexes_.size()
              << " allocation indexes for module " << module_name_;
    }
    // update_mode == ALWAYS_UPDATE: collect nothing.
  }
}

// VA remapping execution flow for 2 consecutive calls on the same executor:
//
// clang-format off
// NOLINTBEGIN
//                   +---------------------+---------------------++---------------------+---------------------+
// GPU               |  VA1 Execute        |  VA2 Execute        ||  VA1 Execute        |  VA2 Execute        |
//                   +---------------------+---------------------++---------------------+---------------------+
//         +---------++---------+           +---------++---------+ +---------++---------++---------+           +---------+
// CPU     | VA1 Map || VA2 Map |           |VA1 UnMap|| VA1 Map | |VA2 UnMap|| VA2 Map ||VA1 UnMap|           |VA2 UnMap|
//         +---------++---------+           +---------++---------+ +---------++---------++---------+           +---------+
// NOLINTEND
// clang-format on
absl::Status GpuExecutable::ExecuteThunksWithVaRemapping(
    const BufferAllocations& buffer_allocations,
    const ServiceExecutableRunOptions* run_options,
    se::StreamExecutor* executor, int64_t unique_id,
    Thunk::ExecutableSource executable_source, bool block_host_until_done,
    bool collective_use_minimal_resource) {
  // Get or create VaRanges for this executor and VA range index. We hold
  // va_ranges_mutex_ briefly just to access/create the VaRanges entry.
  // The VA range index allows multiplexing: with kNumVaReservationSets=2
  // reservations, the CPU can remap one range while the GPU executes the other.
  int command_buffer_va_range_idx =
      run_options->run_options().command_buffer_va_range_idx();
  VaRanges* va_ranges = nullptr;
  {
    absl::MutexLock lock(va_ranges_mutex_);
    auto va_ranges_key = std::make_pair(executor, command_buffer_va_range_idx);
    va_ranges = &module_va_ranges_[va_ranges_key];
  }

  XLA_VLOG_DEVICE(3, executor->device_ordinal())
      << "VA remapping: module " << module_name_
      << " va_range_idx=" << command_buffer_va_range_idx
      << " num_allocations=" << command_buffer_allocation_indexes_.size();

  // Get the DeviceAddressVmmAllocator to look up physical allocations.
  // vmm_allocator is guaranteed non-null here because
  // use_command_buffer_va_remapping already checked for it.
  se::DeviceAddressVmmAllocator* vmm_allocator =
      dynamic_cast<se::DeviceAddressVmmAllocator*>(run_options->allocator());
  if (vmm_allocator == nullptr) {
    return Internal("DeviceAddressVmmAllocator cast failed unexpectedly");
  }

  uint64_t granularity = vmm_allocator->GetAllocationGranularity(executor);
  auto round_up_to_granularity = [granularity](uint64_t size) -> uint64_t {
    if (granularity == 0) {
      return size;
    }
    return ((size + granularity - 1) / granularity) * granularity;
  };

  // Acquire per-executor mutex to protect VA range operations.
  // This ensures only one thread uses the VA ranges at a time for this
  // executor.
  absl::MutexLock va_lock(va_ranges->mutex);

  // Initialize VA ranges if this is first use (va_reservation is null).
  if (va_ranges->va_reservation == nullptr) {
    ScopedAnnotation annotation_va_reserve([&] {
      return absl::StrFormat("command_buffer_va_range_reserve:#module=%s#",
                             module_name_);
    });

    // Calculate total size for all command buffer allocations, rounding each
    // allocation up to the allocation granularity.
    uint64_t total_va_size = 0;
    for (BufferAllocation::Index i : command_buffer_allocation_indexes_) {
      const uint64_t size = buffer_allocations.GetDeviceAddress(i).size();
      total_va_size += round_up_to_granularity(size);
    }

    // Reserve a single large VA range for all command buffer allocations.
    ASSIGN_OR_RETURN(va_ranges->va_reservation,
                     vmm_allocator->CreateReservation(executor, total_va_size));
    ASSIGN_OR_RETURN(va_ranges->unmap_event, executor->CreateEvent());

    XLA_VLOG_DEVICE(3, executor->device_ordinal()) << absl::StreamFormat(
        "VA remapping: Reserved single VA range for module %s "
        "VA: %p total_size: %d granularity: %d",
        module_name_, va_ranges->va_reservation->address().opaque(),
        total_va_size, granularity);
  } else {
    ScopedAnnotation annotation_va_unmap([&] {
      return absl::StrFormat("command_buffer_va_range_unmap:#module=%s#",
                             module_name_);
    });

    // VA range is already initialized; wait for the unmap event to be marked
    // and then do the VA unmapping.
    RETURN_IF_ERROR(va_ranges->unmap_event->Synchronize());

    // Unmap physical addresses from the single reserved VA range.
    // Clearing ScopedMappings calls UnMap via their destructors.
    va_ranges->scoped_mapping.reset();
  }

  // Build a map from allocation index to its offset within va_reservation.
  // Iterate through command_buffer_allocation_indexes_ in order (btree_set
  // provides deterministic iteration order) and accumulate offsets.
  absl::flat_hash_map<BufferAllocation::Index, uint64_t> allocation_va_offsets;
  uint64_t current_offset = 0;
  for (BufferAllocation::Index idx : command_buffer_allocation_indexes_) {
    const uint64_t size = buffer_allocations.GetDeviceAddress(idx).size();
    allocation_va_offsets[idx] = current_offset;
    current_offset += round_up_to_granularity(size);
  }

  if (!allocation_va_offsets.empty() && va_ranges->va_reservation == nullptr) {
    return Internal("Reserved VA address range is null");
  }

  // Map physical memory to reserved VA addresses.
  std::vector<se::DeviceAddressBase> mapped_buffers;
  mapped_buffers.reserve(buffer_allocations.size());

  {
    ScopedAnnotation annotation_va_remap([&] {
      return absl::StrFormat("command_buffer_va_range_remap:#module=%s#",
                             module_name_);
    });

    // Collect mapping descriptors for the batch MapTo call. Descriptors are
    // accumulated in reservation_offset order (guaranteed because
    // allocation_va_offsets was built from a sorted btree_set and the loop
    // below iterates allocation indices in ascending order).
    std::vector<se::MemoryReservation::MappingDescriptor> mapping_descriptors;

    const BufferAllocation::Index num_allocations =
        static_cast<BufferAllocation::Index>(buffer_allocations.size());
    for (BufferAllocation::Index i = 0; i < num_allocations; ++i) {
      se::DeviceAddressBase original_buffer =
          buffer_allocations.GetDeviceAddress(i);

      // Only do VA mapping for allocations accessed by CommandBufferThunk.
      auto offset_it = allocation_va_offsets.find(i);
      if (offset_it == allocation_va_offsets.end()) {
        // Not a command buffer allocation (or zero-size), use the original
        // buffer.
        mapped_buffers.push_back(original_buffer);
        continue;
      }

      // This allocation is used by command buffer - validate it's not null.
      if (original_buffer.is_null()) {
        return Internal("Command buffer allocation %d has null address", i);
      }

      // Get the physical memory allocation from the VMM allocator.
      se::MemoryAllocation* raw_alloc = vmm_allocator->GetRawAllocation(
          executor->device_ordinal(), original_buffer);
      if (raw_alloc == nullptr) {
        return Internal(
            "No raw allocation found for command buffer allocation %d", i);
      }
      const uint64_t mapping_size = raw_alloc->address().size();

      // Calculate the sub-range VA address for this allocation.
      uint64_t va_offset = offset_it->second;
      void* sub_range_ptr = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(
              va_ranges->va_reservation->address().opaque()) +
          va_offset);
      se::DeviceAddressBase sub_range_va(sub_range_ptr, original_buffer.size());

      XLA_VLOG_DEVICE(3, executor->device_ordinal()) << absl::StreamFormat(
          "Mapping allocation %d physical: %p -> VA: %p "
          "(offset: %d) size: %d",
          i, original_buffer.opaque(), sub_range_va.opaque(), va_offset,
          original_buffer.size());

      mapping_descriptors.push_back(
          {va_offset, /*allocation_offset=*/0, mapping_size, raw_alloc});

      // Use VA address for execution.
      mapped_buffers.push_back(
          se::DeviceAddressBase(sub_range_va.opaque(), original_buffer.size()));
    }

    // Batch-map all command buffer allocations into the reserved VA range in
    // a single call. This maps the contiguous range formed by the descriptors
    // and enables device access before returning.
    if (!mapping_descriptors.empty()) {
      ASSIGN_OR_RETURN(se::MemoryReservation::ScopedMapping scoped_mapping,
                       va_ranges->va_reservation->MapTo(
                           absl::MakeSpan(mapping_descriptors)));
      va_ranges->scoped_mapping = std::move(scoped_mapping);
    }
  }

  if (VLOG_IS_ON(3)) {
    void* va_base = (va_ranges->va_reservation != nullptr)
                        ? va_ranges->va_reservation->address().opaque()
                        : nullptr;
    XLA_VLOG_DEVICE(3, executor->device_ordinal()) << absl::StreamFormat(
        "VA remapping: Mapped %d allocations to single VA range at %p",
        allocation_va_offsets.size(), va_base);
    for (const auto& [alloc_idx, va_offset] : allocation_va_offsets) {
      se::DeviceAddressBase physical_addr =
          buffer_allocations.GetDeviceAddress(alloc_idx);
      void* va_ptr = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(va_base) + va_offset);
      XLA_VLOG_DEVICE(3, executor->device_ordinal()) << absl::StreamFormat(
          "  allocation[%d] physical: %p -> VA: %p (offset: %d) size: %d",
          alloc_idx, physical_addr.opaque(), va_ptr, va_offset,
          physical_addr.size());
    }
  }

  BufferAllocations remapped_buffer_allocations(
      mapped_buffers, buffer_allocations.device_ordinal(),
      buffer_allocations.memory_allocator());

  // Execute thunks with remapped addresses.
  RETURN_IF_ERROR(ExecuteThunksImpl(
      has_module() ? &module_config().debug_options() : nullptr, module_name_,
      unique_id, *thunk_executor_, executable_source, run_options,
      remapped_buffer_allocations, block_host_until_done,
      num_additional_streams_, collective_memory_cache_,
      collective_use_minimal_resource));

  // Record event so VA range can be reclaimed after GPU finishes.
  RETURN_IF_ERROR(
      run_options->stream()->RecordEvent(va_ranges->unmap_event.get()));

  return absl::OkStatus();
}

}  // namespace xla::gpu
