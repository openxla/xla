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

#include "xla/stream_executor/stream_executor_vmm_allocator.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_memory_reservation.h"
#include "xla/stream_executor/cuda/cuda_raw_memory_allocation.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/numbers.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace stream_executor {

static absl::Status DeviceNotFoundError(int device_ordinal) {
  return absl::NotFoundError(
      absl::StrFormat("No device with ordinal %d registered in "
                      "DeviceAddressVmmAllocator",
                      device_ordinal));
}

// Returns the completed timeline value from pinned host memory using an
// acquire load, so all GPU writes prior to this value are visible.
// Uses __atomic_load_n rather than std::atomic<> because the pointer is
// volatile (GPU-written pinned memory) and reinterpret_cast to
// std::atomic<uint64_t>* would discard the volatile qualifier.
static uint64_t LoadTimeline(const volatile uint64_t* pinned_timeline) {
  return __atomic_load_n(pinned_timeline, __ATOMIC_ACQUIRE);
}

DeviceAddressVmmAllocator::DeviceAddressVmmAllocator(const Platform* platform)
    : DeviceAddressAllocator(platform) {}

absl::StatusOr<std::unique_ptr<DeviceAddressVmmAllocator>>
DeviceAddressVmmAllocator::Create(const Platform* platform,
                                  absl::Span<const DeviceConfig> devices) {
  absl::flat_hash_set<int> seen_ordinals;
  for (const DeviceConfig& cfg : devices) {
    DCHECK_NE(cfg.executor, nullptr);
    DCHECK_NE(cfg.stream, nullptr);
    int ordinal = cfg.executor->device_ordinal();
    DCHECK(seen_ordinals.insert(ordinal).second)
        << "Duplicate device ordinal: " << ordinal;
  }

  auto allocator = absl::WrapUnique(new DeviceAddressVmmAllocator(platform));

  for (const DeviceConfig& cfg : devices) {
    int ordinal = cfg.executor->device_ordinal();

    // Verify that the device supports 64-bit stream memory operations
    // (cuStreamWriteValue64), which requires compute capability >= 7.0.
    CUdevice cu_device;
    TF_RETURN_IF_ERROR(
        cuda::ToStatus(cuDeviceGet(&cu_device, ordinal), "cuDeviceGet"));
    int supported = 0;
    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cuDeviceGetAttribute(&supported,
                             CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,
                             cu_device),
        "cuDeviceGetAttribute"));
    if (!supported) {
      return absl::UnimplementedError(absl::StrFormat(
          "Device %d does not support 64-bit stream memory operations "
          "(cuStreamWriteValue64 requires compute capability >= 7.0). "
          "Query CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS returned "
          "false.",
          ordinal));
    }

    // Allocate one uint64_t of pinned host memory as the per-device timeline
    // counter, then obtain the device-side pointer used by
    // cuStreamWriteValue64. CU_MEMHOSTALLOC_PORTABLE makes it accessible from
    // all CUDA contexts (important for multi-device scenarios).
    void* host_ptr = nullptr;
    CUdeviceptr dev_ptr = 0;
    {
      std::unique_ptr<ActivateContext> activation = cfg.executor->Activate();
      TF_RETURN_IF_ERROR(cuda::ToStatus(
          cuMemHostAlloc(&host_ptr, sizeof(uint64_t), CU_MEMHOSTALLOC_PORTABLE),
          "cuMemHostAlloc for timeline counter"));
      *static_cast<volatile uint64_t*>(host_ptr) = 0;
      if (auto status =
              cuda::ToStatus(cuMemHostGetDevicePointer(&dev_ptr, host_ptr,
                                                       /*flags=*/0),
                             "cuMemHostGetDevicePointer");
          !status.ok()) {
        cuMemFreeHost(host_ptr);
        return status;
      }
    }

    CUmemAllocationProp alloc_props = {};
    alloc_props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    alloc_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    alloc_props.location.id = cu_device;
    alloc_props.requestedHandleTypes =
        static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_NONE);
    size_t granularity = 0;
    if (auto s = cuda::ToStatus(
            cuMemGetAllocationGranularity(&granularity, &alloc_props,
                                          CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
            "cuMemGetAllocationGranularity");
        !s.ok()) {
      LOG(ERROR) << "Failed to get allocation granularity for device "
                 << ordinal << ": " << s;
    }

    auto state = std::make_unique<PerDeviceState>();
    state->executor = cfg.executor;
    state->stream = cfg.stream;
    state->pa_budget = cfg.pa_budget;
    state->allocation_granularity = static_cast<uint64_t>(granularity);
    state->pinned_timeline = static_cast<volatile uint64_t*>(host_ptr);
    state->timeline_dev_ptr = static_cast<uint64_t>(dev_ptr);

    VLOG(3) << "DeviceAddressVmmAllocator: registering device " << ordinal
            << " with pa_budget " << cfg.pa_budget;
    allocator->per_device_.emplace(ordinal, std::move(state));
  }

  return allocator;
}

absl::StatusOr<std::unique_ptr<DeviceAddressVmmAllocator>>
DeviceAddressVmmAllocator::Create(StreamExecutor* executor, Stream* stream,
                                  uint64_t pa_budget) {
  return Create(executor->GetPlatform(),
                {{DeviceConfig{executor, stream, pa_budget}}});
}

// ABSL_NO_THREAD_SAFETY_ANALYSIS because clang's thread-safety analysis cannot
// reason through the spin-wait path where we read pending_deallocations without
// the lock. In the destructor no concurrent callers exist.
DeviceAddressVmmAllocator::~DeviceAddressVmmAllocator()
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  for (auto& [ordinal, state] : per_device_) {
    // Spin-wait for any pending GPU work to complete before freeing physical
    // memory. In the destructor there are no concurrent callers, so reading
    // pending_deallocations without the lock is safe.
    if (state->pinned_timeline != nullptr &&
        !state->pending_deallocations.empty()) {
      uint64_t last_seqno = state->pending_deallocations.back().seqno;
      while (LoadTimeline(state->pinned_timeline) < last_seqno) {
        absl::SleepFor(absl::Microseconds(50));
      }
    }

    {
      absl::MutexLock lock(&state->mu);
      for (auto& pending : state->pending_deallocations) {
        DoDeallocate(*state, pending.mem);
      }
      state->pending_deallocations.clear();
    }

    // Free the pinned timeline allocation. cuMemFreeHost is safe to call
    // without context activation for CU_MEMHOSTALLOC_PORTABLE memory.
    if (state->pinned_timeline != nullptr) {
      auto status = cuda::ToStatus(
          cuMemFreeHost(const_cast<uint64_t*>(state->pinned_timeline)),
          "cuMemFreeHost for timeline counter");
      if (!status.ok()) {
        LOG(WARNING) << "Failed to free pinned timeline memory for device "
                     << ordinal << ": " << status;
      }
    }
  }
}

DeviceAddressVmmAllocator::PerDeviceState*
DeviceAddressVmmAllocator::GetPerDeviceState(int device_ordinal) const {
  auto it = per_device_.find(device_ordinal);
  if (it == per_device_.end()) {
    return nullptr;
  }
  return it->second.get();
}

void DeviceAddressVmmAllocator::ProcessCompletedPendingDeallocations(
    PerDeviceState& state) {
  // Single atomic read covers all entries whose seqno is <= completed.
  uint64_t completed = LoadTimeline(state.pinned_timeline);
  while (!state.pending_deallocations.empty()) {
    if (state.pending_deallocations.front().seqno > completed) {
      break;
    }
    DoDeallocate(state, state.pending_deallocations.front().mem);
    state.pending_deallocations.pop_front();
  }
}

// ABSL_NO_THREAD_SAFETY_ANALYSIS because clang's thread-safety analysis cannot
// reason through manual Unlock()/Lock() pairs. The declaration in the header
// retains no ABSL_EXCLUSIVE_LOCKS_REQUIRED annotation to reflect that this
// method manages locking internally.
void DeviceAddressVmmAllocator::WaitPendingDeallocationsToComplete(
    PerDeviceState& state, uint64_t size) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (state.pending_deallocations.empty()) {
    return;
  }

  uint64_t accumulated_size = 0;
  size_t count_to_wait = 0;
  uint64_t rounded_size = RoundUpToGranularity(state, size);
  uint64_t target_seqno = 0;

  // Target 1.1x the requested size to provide some headroom.
  uint64_t target_size = rounded_size + rounded_size / 10;

  for (const auto& pending : state.pending_deallocations) {
    accumulated_size += RoundUpToGranularity(state, pending.mem.size());
    target_seqno = pending.seqno;
    ++count_to_wait;
    if (accumulated_size >= target_size) {
      break;
    }
  }

  // Move selected entries out of the deque while holding the lock, so no
  // other thread can observe or free them.
  std::vector<PendingDeallocation> selected;
  selected.reserve(count_to_wait);
  for (size_t i = 0; i < count_to_wait; ++i) {
    selected.push_back(std::move(state.pending_deallocations.front()));
    state.pending_deallocations.pop_front();
  }

  // Release the lock before spin-waiting to avoid stalling other threads for
  // potentially milliseconds while the GPU drains its work queue.
  state.mu.Unlock();
  {
    tsl::profiler::ScopedAnnotation annotation([&] {
      return absl::StrFormat(
          "WaitPendingDeallocations:#count=%zu,target_size=%s#", count_to_wait,
          tsl::strings::HumanReadableNumBytes(target_size));
    });

    // Poll until the GPU writes a timeline value >= target_seqno.
    // Since timeline values are written in stream order, this guarantees all
    // earlier pending deallocations have also completed.
    // Sleep 10us per iteration to release the CPU core while waiting rather
    // than hot-spinning.
    while (LoadTimeline(state.pinned_timeline) < target_seqno) {
      absl::SleepFor(absl::Microseconds(50));
    }
  }
  // Reacquire the lock before modifying the maps.
  state.mu.Lock();

  for (auto& item : selected) {
    DoDeallocate(state, item.mem);
  }
}

void DeviceAddressVmmAllocator::DoDeallocate(PerDeviceState& state,
                                             DeviceAddressBase mem) {
  VLOG(3) << absl::StreamFormat(
      "Actually freeing virtual address %p (size=%uB) on device ordinal %d",
      mem.opaque(), mem.size(), state.executor->device_ordinal());

  // Erase the ScopedMapping first: its destructor unmaps the physical memory
  // from the virtual address range.
  state.scoped_mappings.erase(mem.opaque());
  // Erase the reservation next: its destructor frees the virtual address range.
  state.reservations.erase(mem.opaque());
  // Erase the raw allocation last: its destructor releases the physical memory.
  state.raw_allocations.erase(mem.opaque());

  uint64_t rounded_size = RoundUpToGranularity(state, mem.size());
  DCHECK_GE(state.pa_allocated, rounded_size);
  state.pa_allocated -= rounded_size;
}

absl::StatusOr<DeviceAddressBase> DeviceAddressVmmAllocator::AllocateWithBudget(
    PerDeviceState& state, uint64_t size) {
  uint64_t rounded_size = RoundUpToGranularity(state, size);
  if (state.pa_allocated + rounded_size > state.pa_budget) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Not enough PA budget for allocation: pa_allocated=%uB, "
        "rounded_size=%uB, pa_budget=%uB",
        state.pa_allocated, rounded_size, state.pa_budget));
  }

  // Create physical memory allocation (cuMemCreate).
  TF_ASSIGN_OR_RETURN(auto raw_alloc, gpu::CudaRawMemoryAllocation::Create(
                                          state.executor, size));
  const uint64_t padded_size = raw_alloc->address().size();

  // Reserve virtual address range (cuMemAddressReserve).
  TF_ASSIGN_OR_RETURN(auto reservation,
                      gpu::CudaMemoryReservation::Create(state.executor, size));

  // Map physical memory into the virtual address range and enable access.
  TF_ASSIGN_OR_RETURN(
      auto scoped_mapping,
      reservation->MapTo(/*reservation_offset=*/0, /*allocation_offset=*/0,
                         padded_size, *raw_alloc));

  void* va_ptr = reservation->address().opaque();

  // Store tracking entries. Destruction order matters: scoped_mappings must
  // be erased before reservations, which must be erased before raw_allocations.
  state.raw_allocations.emplace(va_ptr, std::move(raw_alloc));
  state.reservations.emplace(va_ptr, std::move(reservation));
  state.scoped_mappings.emplace(va_ptr, std::move(scoped_mapping));

  state.pa_allocated += rounded_size;
  // Return the original requested size, not the padded size.
  return DeviceAddressBase(va_ptr, size);
}

// Allocation flow with retry:
//
// Allocate(device_ordinal, size)
//           │
//           ▼
// ┌─────────────────────────────────┐
// │ Reuse pending deallocation      │──found──► return
// │ with matching size?             │
// └─────────────────────────────────┘
//           │ not found
//           ▼
// ┌─────────────────────────────────┐
// │ Allocate new physical +         │──OK──► return
// │ virtual memory                  │
// └─────────────────────────────────┘
//           │ failed
//           ▼
// ┌─────────────────────────────────┐
// │ Free any GPU-completed          │
// │ pending deallocations           │
// │ (non-blocking)                  │
// └─────────────────────────────────┘
//           │
//           ▼
// ┌─────────────────────────────────┐
// │ Allocate new physical +         │──OK──► return
// │ virtual memory                  │
// └─────────────────────────────────┘
//           │ failed
//           ▼
// ┌─────────────────────────────────┐
// │ Block until GPU frees           │
// │ enough pending memory           │
// └─────────────────────────────────┘
//           │
//           ▼
// ┌─────────────────────────────────┐
// │ Allocate new physical +         │──OK──► return
// │ virtual memory                  │
// └─────────────────────────────────┘
//           │ failed
//           ▼
//    ResourceExhaustedError
absl::StatusOr<ScopedDeviceAddress<uint8_t>>
DeviceAddressVmmAllocator::Allocate(int device_ordinal, uint64_t size,
                                    bool /*retry_on_failure*/,
                                    int64_t /*memory_space*/) {
  if (size == 0) {
    return ScopedDeviceAddress<uint8_t>(DeviceAddressBase(), device_ordinal,
                                        this);
  }

  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }

  absl::MutexLock lock(&state->mu);

  // Try to reuse a completed pending deallocation with matching size.
  std::optional<DeviceAddressBase> reused =
      TryReusePendingDeallocation(*state, size);
  if (reused.has_value()) {
    return ScopedDeviceAddress<uint8_t>(*reused, device_ordinal, this);
  }

  absl::StatusOr<DeviceAddressBase> result = AllocateWithBudget(*state, size);

  // If allocation failed (e.g., out of memory), try processing pending
  // deallocations to free memory, then retry.
  if (!result.ok()) {
    ProcessCompletedPendingDeallocations(*state);
    result = AllocateWithBudget(*state, size);
  }

  if (!result.ok()) {
    WaitPendingDeallocationsToComplete(*state, size);
    result = AllocateWithBudget(*state, size);
  }

  if (!result.ok()) {
    return result.status();
  }

  VLOG(3) << absl::StreamFormat(
      "Allocated virtual address %s (%uB) on device ordinal %d: %p",
      tsl::strings::HumanReadableNumBytes(size), size, device_ordinal,
      result->opaque());

  return ScopedDeviceAddress<uint8_t>(*result, device_ordinal, this);
}

absl::Status DeviceAddressVmmAllocator::Deallocate(int device_ordinal,
                                                   DeviceAddressBase mem) {
  if (mem.is_null()) {
    return absl::OkStatus();
  }

  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }

  absl::MutexLock lock(&state->mu);

  VLOG(3) << absl::StreamFormat(
      "Queueing deferred deallocation for virtual address %p (size=%uB) "
      "on device ordinal %d",
      mem.opaque(), mem.size(), device_ordinal);

  // Get the underlying CUDA stream handle.
  CUstream cu_stream =
      static_cast<CUstream>(state->stream->platform_specific_handle().stream);

  // Assign the next sequence number and enqueue a GPU write to the pinned
  // timeline when the stream reaches this point. The CPU polls the timeline
  // value to know when it is safe to free the memory.
  uint64_t seqno = state->next_seqno++;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuStreamWriteValue64(cu_stream,
                           static_cast<CUdeviceptr>(state->timeline_dev_ptr),
                           seqno, /*flags=*/0),
      "cuStreamWriteValue64"));

  state->pending_deallocations.push_back({mem, seqno});

  return absl::OkStatus();
}

absl::StatusOr<Stream*> DeviceAddressVmmAllocator::GetStream(
    int device_ordinal) {
  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }
  return state->stream;
}

absl::StatusOr<StreamExecutor*> DeviceAddressVmmAllocator::GetStreamExecutor(
    int device_ordinal) const {
  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }
  return state->executor;
}

MemoryAllocation* DeviceAddressVmmAllocator::GetRawAllocation(
    int device_ordinal, DeviceAddressBase addr) const {
  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return nullptr;
  }
  absl::MutexLock lock(&state->mu);
  auto it = state->raw_allocations.find(addr.opaque());
  if (it == state->raw_allocations.end()) {
    return nullptr;
  }
  return it->second.get();
}

MemoryReservation* DeviceAddressVmmAllocator::GetReservation(
    int device_ordinal, DeviceAddressBase addr) const {
  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return nullptr;
  }
  absl::MutexLock lock(&state->mu);
  auto it = state->reservations.find(addr.opaque());
  if (it == state->reservations.end()) {
    return nullptr;
  }
  return it->second.get();
}

std::optional<DeviceAddressBase>
DeviceAddressVmmAllocator::TryReusePendingDeallocation(PerDeviceState& state,
                                                       uint64_t size) {
  uint64_t rounded_size = RoundUpToGranularity(state, size);
  for (auto it = state.pending_deallocations.begin();
       it != state.pending_deallocations.end(); ++it) {
    if (RoundUpToGranularity(state, it->mem.size()) != rounded_size) {
      continue;
    }

    DeviceAddressBase reused_mem(it->mem.opaque(), size);
    VLOG(3) << absl::StreamFormat(
        "Reusing pending deallocation: address=%p original_size=%uB "
        "new_size=%uB rounded_size=%uB device=%d",
        reused_mem.opaque(), it->mem.size(), size, rounded_size,
        state.executor->device_ordinal());
    state.pending_deallocations.erase(it);

    return reused_mem;
  }

  return std::nullopt;
}

uint64_t DeviceAddressVmmAllocator::RoundUpToGranularity(
    const PerDeviceState& state, uint64_t size) const {
  if (state.allocation_granularity == 0) {
    return size;
  }
  return ((size + state.allocation_granularity - 1) /
          state.allocation_granularity) *
         state.allocation_granularity;
}

}  // namespace stream_executor
