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

#ifndef XLA_SERVICE_GPU_COMMAND_BUFFER_VA_REMAPPER_H_
#define XLA_SERVICE_GPU_COMMAND_BUFFER_VA_REMAPPER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// Owns the command-buffer virtual-address remapping state for one
// GpuExecutable. The remapper keeps stable virtual address reservations across
// executions and maps the current physical buffers into those reservations
// before delegating to thunk execution.
class CommandBufferVaRemapper {
 public:
  using ExecuteFn = absl::FunctionRef<absl::Status(const BufferAllocations&)>;

  void CaptureAllocations(ThunkExecutor& thunk_executor,
                          absl::Span<const BufferAllocation* const> allocations,
                          DebugOptions::CommandBufferUpdateMode update_mode,
                          absl::string_view module_name);

  bool ShouldUse(DebugOptions::CommandBufferUpdateMode update_mode,
                 se::DeviceAddressAllocator* memory_allocator) const;

  size_t num_allocations() const { return allocation_indexes_.size(); }

  absl::Status Execute(const BufferAllocations& buffer_allocations,
                       const ServiceExecutableRunOptions* run_options,
                       se::StreamExecutor* executor,
                       absl::string_view module_name, ExecuteFn execute);

 private:
  // State for VA remapping of command buffer allocations on a single executor.
  struct VaRanges {
    // Mutex to protect VA range operations (map/execute/unmap) for this
    // executor. This ensures only one thread can use the VA ranges at a time.
    absl::Mutex mutex;

    // Single large virtual address reservation covering all command buffer
    // allocations. nullptr until first use.
    std::unique_ptr<se::MemoryReservation> va_reservation;

    // Event used to synchronize VA range reuse. When the device has completed
    // the task that uses the VA range, it marks the event, letting the host
    // know the VA range can be remapped to other physical addresses.
    std::unique_ptr<se::Event> unmap_event;

    // RAII wrapper that keeps the VA->physical mapping active.
    // Reset (auto-unmapping) before each re-use of the VA range.
    std::optional<se::MemoryReservation::ScopedMapping> scoped_mapping;
  };

  // Buffer allocation indices accessed by command buffer thunks. Using
  // btree_set for deterministic iteration order.
  absl::btree_set<BufferAllocation::Index> allocation_indexes_;

  // Separate mutex for VA ranges to avoid contention with unrelated executable
  // state during VA remapping operations which may involve GPU synchronization.
  absl::Mutex va_ranges_mutex_;
  absl::node_hash_map<std::pair<se::StreamExecutor*, int>, VaRanges>
      module_va_ranges_ ABSL_GUARDED_BY(va_ranges_mutex_);
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_COMMAND_BUFFER_VA_REMAPPER_H_
