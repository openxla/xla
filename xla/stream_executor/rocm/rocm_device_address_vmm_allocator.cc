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

#include "xla/stream_executor/rocm/rocm_device_address_vmm_allocator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_memory_reservation.h"
#include "xla/stream_executor/rocm/rocm_raw_memory_allocation.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/vmm_device_address_allocator.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {

RocmDeviceAddressVmmAllocator::RocmDeviceAddressVmmAllocator(
    const Platform* platform)
    : DeviceAddressVmmAllocator(platform) {}

absl::StatusOr<std::unique_ptr<RocmDeviceAddressVmmAllocator>>
RocmDeviceAddressVmmAllocator::Create(const Platform* platform,
                                      absl::Span<const DeviceConfig> devices) {
  auto allocator =
      absl::WrapUnique(new RocmDeviceAddressVmmAllocator(platform));
  TF_RETURN_IF_ERROR(PopulateDevices(allocator.get(), devices));
  return allocator;
}

absl::StatusOr<std::unique_ptr<RocmDeviceAddressVmmAllocator>>
RocmDeviceAddressVmmAllocator::Create(StreamExecutor* executor, Stream* stream,
                                      uint64_t pa_budget) {
  return Create(executor->GetPlatform(),
                {{DeviceConfig{executor, stream, pa_budget}}});
}

absl::StatusOr<std::unique_ptr<RocmDeviceAddressVmmAllocator>>
RocmDeviceAddressVmmAllocator::Create(
    const Platform* platform, double memory_fraction,
    std::optional<int64_t> gpu_system_memory_size,
    absl::Span<const std::pair<StreamExecutor*, Stream*>> devices) {
  LOG(INFO) << "Using VMM (Virtual Memory Management) allocator for ROCm.";
  std::vector<DeviceConfig> device_configs;
  device_configs.reserve(devices.size());
  for (const auto& [executor, stream] : devices) {
    int64_t free_memory;
    int64_t total_memory;
    if (!executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
      return absl::UnavailableError(
          absl::StrFormat("Failed to query available memory from device %i",
                          executor->device_ordinal()));
    }
    uint64_t pa_budget = total_memory * memory_fraction;
    if (gpu_system_memory_size.has_value()) {
      pa_budget = gpu_system_memory_size.value();
    }
    LOG(INFO) << "VMM allocator pa_budget for device "
              << executor->device_ordinal() << ": " << pa_budget << " bytes.";
    device_configs.push_back({executor, stream, pa_budget});
  }
  return Create(platform, device_configs);
}

absl::Status RocmDeviceAddressVmmAllocator::InitializeDeviceState(
    PerDeviceState& state) {
  int ordinal = state.executor->device_ordinal();

  // Allocate signal memory for the per-device timeline counter.
  // hipStreamWriteValue64 on AMD hardware requires the target pointer to be
  // allocated with hipMallocSignalMemory.
  void* signal_ptr = nullptr;
  {
    std::unique_ptr<ActivateContext> activation = state.executor->Activate();
    TF_RETURN_IF_ERROR(ToStatus(
        wrap::hipExtMallocWithFlags(&signal_ptr, sizeof(uint64_t),
                                    hipMallocSignalMemory),
        "hipExtMallocWithFlags(hipMallocSignalMemory) for timeline counter"));
    *static_cast<volatile uint64_t*>(signal_ptr) = 0;
  }

  // Set timeline fields and destroy_fn immediately so that any early return
  // below still cleans up the signal memory via ~DeviceAddressVmmAllocator.
  state.pinned_timeline = static_cast<volatile uint64_t*>(signal_ptr);
  state.timeline_dev_ptr = reinterpret_cast<uint64_t>(signal_ptr);
  state.destroy_fn = [signal_ptr, ordinal]() {
    auto status =
        ToStatus(wrap::hipFree(signal_ptr), "hipFree for signal memory");
    if (!status.ok()) {
      LOG(WARNING) << "Failed to free signal memory for device " << ordinal
                   << ": " << status;
    }
  };

  // Query allocation granularity for this device.
  hipDevice_t hip_device;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipDeviceGet(&hip_device, ordinal), "hipDeviceGet"));

  hipMemAllocationProp alloc_props = {};
  alloc_props.type = hipMemAllocationTypePinned;
  alloc_props.location.type = hipMemLocationTypeDevice;
  alloc_props.location.id = hip_device;
  alloc_props.requestedHandleTypes = hipMemHandleTypeNone;
  size_t granularity = 0;
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipMemGetAllocationGranularity(
          &granularity, &alloc_props,
          hipMemAllocationGranularityRecommended),
      "hipMemGetAllocationGranularity"));

  state.allocation_granularity = static_cast<uint64_t>(granularity);

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>>
RocmDeviceAddressVmmAllocator::CreateAllocation(StreamExecutor* executor,
                                                uint64_t size) {
  return RocmRawMemoryAllocation::Create(executor, size);
}

absl::StatusOr<std::unique_ptr<MemoryReservation>>
RocmDeviceAddressVmmAllocator::CreateReservation(StreamExecutor* executor,
                                                 uint64_t size) {
  return RocmMemoryReservation::Create(executor, size);
}

absl::Status RocmDeviceAddressVmmAllocator::EnqueueDeferredDeallocation(
    PerDeviceState& state, uint64_t seqno) {
  hipStream_t hip_stream =
      static_cast<hipStream_t>(state.stream->platform_specific_handle().stream);
  return ToStatus(
      wrap::hipStreamWriteValue64(
          hip_stream, reinterpret_cast<void*>(state.timeline_dev_ptr), seqno,
          0),
      "hipStreamWriteValue64");
}

}  // namespace stream_executor::gpu
