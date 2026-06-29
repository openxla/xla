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

#ifndef XLA_SERVICE_GPU_GPU_EXECUTABLE_BUFFER_ALLOCATOR_H_
#define XLA_SERVICE_GPU_GPU_EXECUTABLE_BUFFER_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// Internal, pure (uncached) helpers for the ROCm VMM command-buffer
// optimizations, exposed for unit testing. Production code calls the cached
// wrappers defined in the .cc; these take the raw env value explicitly so the
// decision logic can be tested without process-global state.
namespace vmm_internal {

// Whether per-slot skip-remap is enabled for `platform_name`, given the raw
// XLA_VMM_SKIP_REMAP value (`env_value == nullptr` means unset). ROCm-only:
// default ON for ROCm, always OFF elsewhere.
bool ParseVmmRemapSkipEnabled(absl::string_view platform_name,
                              const char* env_value);

// Copy-into-shadow byte threshold for `platform_name`, given the raw
// XLA_VMM_TMP_COPY_THRESHOLD value (`env_value == nullptr` means unset).
// ROCm-only; returns 0 (disabled) elsewhere and for empty/non-numeric values.
uint64_t ParseVmmCopyThresholdBytes(absl::string_view platform_name,
                                    const char* env_value);

}  // namespace vmm_internal

class ThunkExecutor;

// Owns executable-scoped buffer allocation state for one GpuExecutable.
class GpuExecutableBufferAllocator {
 private:
  struct VaRanges;

 public:
  struct ParameterBuffer {
    se::DeviceAddressBase buffer;
    int64_t parameter_number = 0;
    bool allow_null_buffer = false;
  };

  // Resolves the device address backing an entry-computation-parameter
  // allocation. Returning `allow_null_buffer` is used for skipped tuple
  // index-table allocations.
  using ParameterBufferResolver =
      absl::FunctionRef<absl::StatusOr<ParameterBuffer>(
          const BufferAllocation& allocation)>;

  using BufferAllocToDeviceMemoryMap =
      absl::flat_hash_map<BufferAllocation::Index, se::DeviceAddressBase>;

  using AllocationIndexSet = absl::btree_set<BufferAllocation::Index>;

  // Execution-scoped buffer allocation state. Command-buffer VA remapping is
  // inactive when `command_buffer_active()` is false.
  class ExecutionScope {
   public:
    ExecutionScope(const ExecutionScope&) = delete;
    ExecutionScope& operator=(const ExecutionScope&) = delete;
    ExecutionScope(ExecutionScope&&) = default;
    ExecutionScope& operator=(ExecutionScope&&) = default;

    bool command_buffer_active() const { return va_ranges_ != nullptr; }

    // Builds the BufferAllocations for an execution. Entry-computation
    // parameter buffers are obtained from `get_parameter_buffer`; all other
    // allocations are resolved internally, including collective-memory
    // granularity rounding and alignment checking.
    absl::StatusOr<BufferAllocations> GenerateBufferAllocations(
        const ServiceExecutableRunOptions* run_options,
        ParameterBufferResolver get_parameter_buffer,
        const BufferAllocToDeviceMemoryMap* globals,
        se::DeviceAddressAllocator* memory_allocator, int device_ordinal);

    // Copy-protection for an aliased output that was not donated at runtime:
    // allocates a fresh result buffer for the output at `index`, copies the
    // contents of the aliased buffer (allocation `allocation`) into it, and
    // redirects the aliased entry in `buffer_allocations` to the fresh buffer.
    // Returns the newly allocated result buffer.
    absl::StatusOr<se::DeviceAddressBase> AllocateCopyProtectedOutputBuffer(
        const ServiceExecutableRunOptions* run_options,
        BufferAllocations& buffer_allocations, const ShapeIndex& index,
        const BufferAllocation& allocation, int device_ordinal,
        se::DeviceAddressAllocator* memory_allocator,
        absl::FunctionRef<absl::Status(absl::Status)> allocation_error);

    absl::Status ExecuteWithBufferAllocations(
        const BufferAllocations& owning_buffer_allocations, int device_ordinal,
        absl::FunctionRef<absl::Status(const BufferAllocations&)> execute);

   private:
    friend class GpuExecutableBufferAllocator;

    explicit ExecutionScope(GpuExecutableBufferAllocator* owner);
    ExecutionScope(GpuExecutableBufferAllocator* owner, VaRanges* va_ranges,
                   const ServiceExecutableRunOptions* run_options);

    absl::StatusOr<se::DeviceAddressBase> BufferForAllocation(
        ParameterBufferResolver get_parameter_buffer,
        const BufferAllocToDeviceMemoryMap* globals,
        const BufferAllocation& allocation,
        se::DeviceAddressAllocator* memory_allocator, int device_ordinal,
        int64_t arg_idx,
        const absl::flat_hash_map<LogicalBuffer::Color, int64_t>&
            allocate_granularity);
    absl::Status ExecuteWithVaRemapping(
        const BufferAllocations& owning_buffer_allocations, int device_ordinal,
        absl::FunctionRef<absl::Status(const BufferAllocations&)> execute);

    GpuExecutableBufferAllocator* owner_ = nullptr;
    VaRanges* va_ranges_ = nullptr;
    const ServiceExecutableRunOptions* run_options_ = nullptr;
  };

  static absl::StatusOr<AllocationIndexSet>
  CollectCommandBufferAllocationIndexes(
      ThunkExecutor* thunk_executor,
      absl::Span<const BufferAllocation* const> allocations,
      DebugOptions::CommandBufferUpdateMode update_mode);

  GpuExecutableBufferAllocator(
      absl::string_view module_name,
      absl::Span<const BufferAllocation* const> allocations,
      const Shape& result_shape, const DebugOptions* debug_options,
      DebugOptions::CommandBufferUpdateMode update_mode,
      AllocationIndexSet allocation_indexes);

  size_t command_buffer_allocation_count() const {
    return command_buffer_allocation_indexes_.size();
  }

  absl::StatusOr<ExecutionScope> CreateExecutionScope(
      const ServiceExecutableRunOptions* run_options,
      se::DeviceAddressAllocator* memory_allocator, int device_ordinal);

 private:
  // State for VA remapping of command buffer allocations on a single executor.
  struct VaRanges {
    // Mutex to protect VA range operations (map/execute/unmap) for this
    // executor. This ensures only one thread can use the VA ranges at a time.
    absl::Mutex mutex;

    // Single large virtual address reservation covering all command buffer
    // allocations. nullptr until first use.
    std::unique_ptr<se::MemoryReservation> va_reservation ABSL_GUARDED_BY(mutex);

    // Event used to synchronize VA range reuse. When the device has completed
    // the task that uses the VA range, it marks the event, letting the host
    // know the VA range can be remapped to other physical addresses.
    std::unique_ptr<se::Event> unmap_event ABSL_GUARDED_BY(mutex);

    // RAII wrapper that keeps the VA->physical mapping active.
    // Reset (auto-unmapping) before each re-use of the VA range.
    std::optional<se::MemoryReservation::ScopedMapping> scoped_mapping
        ABSL_GUARDED_BY(mutex);

    // ROCm skip-remap booking. Source (BFC) device address mapped into each
    // command-buffer slot during the previous step, in ascending
    // reservation-offset order (same order the mapping descriptors are built).
    // A slot whose source address is unchanged this step keeps its existing
    // mapping instead of being unmapped+remapped, and when no slot changes the
    // whole unmap/map/SetAccess (and the unmap-event sync) is skipped. Empty
    // until the first mapping is established. ROCm-only; see
    // VmmRemapSkipEnabled().
    std::vector<const void*> last_mapped_src_addrs ABSL_GUARDED_BY(mutex);

    // Debug-only companion to last_mapped_src_addrs: the per-slot reservation
    // offsets from the previous step, used to DCHECK that the slot->allocation
    // mapping is stable across steps (the per-slot address comparison is only
    // valid if each slot maps the same allocation). Populated only in non-NDEBUG
    // builds; declared unconditionally to keep the struct layout identical
    // across translation units. ROCm-only.
    std::vector<uint64_t> last_mapped_offsets ABSL_GUARDED_BY(mutex);

    // ROCm copy-into-shadow (env XLA_VMM_TMP_COPY_THRESHOLD): small
    // command-buffer slices (tiny scale/scalar/metric buffers that churn their
    // address every step) are kept OUT of the VA reservation and instead given
    // a stable shadow device buffer here -- allocated once, fixed address baked
    // into the command buffer -- and refreshed with a stream-ordered D2D copy
    // each step (real->shadow before execute, shadow->real after) rather than
    // an expensive hipMemUnmap/Map/SetAccess. Keyed by allocation index; kept
    // for the run's lifetime. See VmmCopyThresholdBytes().
    absl::flat_hash_map<BufferAllocation::Index, se::DeviceAddressBase>
        small_shadow ABSL_GUARDED_BY(mutex);

    // Executor that owns the `small_shadow` device buffers. Set when this
    // VaRanges is created; used by the destructor to free the shadow buffers
    // (DeviceAddressBase is a non-owning pointer+size, so they must be freed
    // explicitly). nullptr means no shadow buffers were ever allocated.
    se::StreamExecutor* executor = nullptr;

    VaRanges() = default;
    // Runs at executor/allocator teardown when no other thread can touch this
    // VaRanges, so it accesses the mutex-guarded members without locking
    // (ABSL_NO_THREAD_SAFETY_ANALYSIS).
    ~VaRanges() ABSL_NO_THREAD_SAFETY_ANALYSIS {
      if (executor != nullptr) {
        // The last step's command buffer may still be reading/writing these
        // shadow buffers on the device (unmap_event is recorded after
        // execute()); freeing memory still in use by a kernel is UB on ROCm, so
        // wait for that GPU work before deallocating.
        if (unmap_event != nullptr) {
          unmap_event->Synchronize().IgnoreError();
        }
        for (auto& [index, shadow] : small_shadow) {
          executor->Deallocate(&shadow);
        }
      }
    }

    // Non-movable/non-copyable: stored in a node_hash_map (stable addresses,
    // never relocated) and holds a mutex + RAII mapping.
    VaRanges(const VaRanges&) = delete;
    VaRanges& operator=(const VaRanges&) = delete;
  };

  std::string module_name_;
  std::vector<const BufferAllocation*> allocations_;
  Shape result_shape_;
  const DebugOptions* debug_options_ = nullptr;
  DebugOptions::CommandBufferUpdateMode update_mode_;
  AllocationIndexSet command_buffer_allocation_indexes_;

  // Separate mutex for VA ranges to avoid contention with executable module
  // handle state during VA remapping operations, which may synchronize with GPU
  // work.
  absl::Mutex va_ranges_mutex_;
  absl::node_hash_map<se::StreamExecutor*, VaRanges> module_va_ranges_
      ABSL_GUARDED_BY(va_ranges_mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_BUFFER_ALLOCATOR_H_
