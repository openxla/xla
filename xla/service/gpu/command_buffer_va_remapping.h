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

#ifndef XLA_SERVICE_GPU_COMMAND_BUFFER_VA_REMAPPING_H_
#define XLA_SERVICE_GPU_COMMAND_BUFFER_VA_REMAPPING_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
class DeviceAddressVmmAllocator;
}  // namespace stream_executor

namespace xla::gpu {

// Manages stable virtual-address mappings for command-buffer allocations.
//
// Command buffers can avoid updates when selected allocations are mapped at
// deterministic virtual addresses across executions. This class owns the
// per-executable allocation policy and the per-executor VMM reservations used
// to materialize that policy at execution time.
class CommandBufferVaRemapping {
 public:
  class ScopedExecution;

  static absl::StatusOr<std::unique_ptr<CommandBufferVaRemapping>> Create(
      DebugOptions::CommandBufferUpdateMode update_mode,
      ThunkExecutor* thunk_executor,
      absl::Span<const BufferAllocation* const> allocations,
      absl::string_view module_name);

  ~CommandBufferVaRemapping();

  CommandBufferVaRemapping(const CommandBufferVaRemapping&) = delete;
  CommandBufferVaRemapping& operator=(const CommandBufferVaRemapping&) = delete;

  const absl::btree_set<BufferAllocation::Index>& allocation_indices() const {
    return allocation_indices_;
  }

  absl::StatusOr<std::unique_ptr<ScopedExecution>> BeginExecution(
      const ServiceExecutableRunOptions* run_options,
      se::DeviceAddressAllocator* memory_allocator, int device_ordinal);

  absl::Status PrepareReservation(
      const ServiceExecutableRunOptions* run_options, int device_ordinal,
      absl::Span<const BufferAllocation* const> allocations,
      const absl::flat_hash_map<LogicalBuffer::Color, int64_t>&
          allocate_granularity,
      ScopedExecution* execution);

  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, const BufferAllocation& allocation,
      int64_t buffer_size, bool return_reservation_address,
      ScopedExecution& execution);

  bool ShouldRemapAllocation(BufferAllocation::Index index,
                             const ScopedExecution* execution) const;

  absl::Status UpdateAllocationPolicy(ScopedExecution& execution);

  Thunk::CommandBufferUpdateInfo GetCommandBufferUpdateInfo(
      const ScopedExecution& execution) const;

  absl::StatusOr<BufferAllocations> BuildBufferAllocations(
      const BufferAllocations& owning_buffer_allocations, int device_ordinal,
      absl::Span<const BufferAllocation* const> allocations,
      ScopedExecution& execution);

  absl::Status UnmapAliases(int device_ordinal, ScopedExecution& execution);

 private:
  struct MemoryReservationAlias {
    uint64_t reservation_offset = 0;
    uint64_t size = 0;
    se::DeviceAddressBase reservation_address;
  };

  struct VaRemapping {
    absl::Mutex mutex;

    uint64_t granularity = 0;
    uint64_t total_size = 0;
    absl::flat_hash_map<BufferAllocation::Index, uint64_t>
        allocation_to_reservation_offset;

    std::unique_ptr<se::MemoryReservation> va_reservation;
    se::DeviceAddressVmmAllocator* vmm_allocator = nullptr;

    bool update_policy_ready = false;
    std::vector<BufferAllocation::Index> policy_va_remapped_indices;
    std::vector<BufferAllocation::Index> policy_dynamic_alloc_indices;
    absl::btree_set<BufferAllocation::Index> policy_va_remapped_index_set;

    absl::StatusOr<uint64_t> GetReservationOffset(
        BufferAllocation::Index idx) const;
  };

 public:
  // Per-execution state for command-buffer VA remapping. Access is serialized
  // by VaRemapping::mutex, which is held for the full execution.
  class ScopedExecution {
   public:
    ScopedExecution(VaRemapping& remapping,
                    se::DeviceAddressVmmAllocator& vmm_allocator,
                    std::unique_ptr<absl::MutexLock> lock)
        : remapping_(remapping),
          vmm_allocator_(vmm_allocator),
          lock_(std::move(lock)) {}

   private:
    friend class CommandBufferVaRemapping;

    absl::StatusOr<MemoryReservationAlias> GetReservationAlias(
        BufferAllocation::Index idx) const;

    VaRemapping& remapping_;
    se::DeviceAddressVmmAllocator& vmm_allocator_;
    std::unique_ptr<absl::MutexLock> lock_;
    absl::flat_hash_map<BufferAllocation::Index, MemoryReservationAlias>
        allocation_to_reservation_aliases_;
    std::vector<MemoryReservationAlias> aliases_to_unmap_;
  };

 private:
  CommandBufferVaRemapping(
      DebugOptions::CommandBufferUpdateMode update_mode,
      std::string module_name,
      absl::btree_set<BufferAllocation::Index> update_allocation_indices,
      absl::btree_set<BufferAllocation::Index> allocation_indices);

  DebugOptions::CommandBufferUpdateMode update_mode_;
  std::string module_name_;

  // Buffer allocation indices accessed by command buffer thunks. Using
  // btree_set for deterministic iteration order.
  absl::btree_set<BufferAllocation::Index> update_allocation_indices_;

  // Buffer allocation indices that can be VA-remapped for command buffer
  // execution. This is a subset of update_allocation_indices_.
  absl::btree_set<BufferAllocation::Index> allocation_indices_;

  // Persistent command-buffer VA remapping state, keyed by executor. A single
  // executable can run on multiple executors, but each executor reuses its own
  // VA reservation across executions.
  absl::Mutex va_remaps_mutex_;
  absl::node_hash_map<stream_executor::StreamExecutor*, VaRemapping> va_remaps_
      ABSL_GUARDED_BY(va_remaps_mutex_);
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_COMMAND_BUFFER_VA_REMAPPING_H_
