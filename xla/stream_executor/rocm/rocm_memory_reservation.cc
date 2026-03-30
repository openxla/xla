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

#include "xla/stream_executor/rocm/rocm_memory_reservation.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_raw_memory_allocation.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {

absl::StatusOr<std::unique_ptr<RocmMemoryReservation>>
RocmMemoryReservation::Create(StreamExecutor* executor, uint64_t size) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  hipDevice_t device;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipDeviceGet(&device, executor->device_ordinal())));

  hipMemAllocationProp props = {};
  props.type = hipMemAllocationTypePinned;
  props.location.type = hipMemLocationTypeDevice;
  props.location.id = device;
  props.requestedHandleTypes = hipMemHandleTypeNone;

  size_t granularity = 0;
  TF_RETURN_IF_ERROR(ToStatus(wrap::hipMemGetAllocationGranularity(
      &granularity, &props, hipMemAllocationGranularityRecommended)));

  uint64_t padded_size = xla::RoundUpTo<uint64_t>(size, granularity);

  void* ptr = nullptr;
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipMemAddressReserve(&ptr, padded_size, granularity, nullptr,
                                    0ULL)));

  return std::unique_ptr<RocmMemoryReservation>(
      new RocmMemoryReservation(executor, static_cast<char*>(ptr), padded_size));
}

RocmMemoryReservation::RocmMemoryReservation(StreamExecutor* executor,
                                             char* ptr, uint64_t size)
    : executor_(executor), ptr_(ptr), size_(size) {}

DeviceAddressBase RocmMemoryReservation::address() const {
  return DeviceAddressBase(ptr_, size_);
}

absl::Status RocmMemoryReservation::Map(size_t reservation_offset,
                                        size_t allocation_offset, size_t size,
                                        MemoryAllocation& allocation) {
  auto* rocm_alloc = dynamic_cast<RocmRawMemoryAllocation*>(&allocation);
  if (rocm_alloc == nullptr) {
    return absl::InvalidArgumentError(
        "RocmMemoryReservation::Map requires a RocmRawMemoryAllocation");
  }
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  return ToStatus(wrap::hipMemMap(ptr_ + reservation_offset, size,
                                  allocation_offset, rocm_alloc->GetHandle(),
                                  0ULL));
}

absl::Status RocmMemoryReservation::SetAccess(uint64_t reservation_offset,
                                              size_t size) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  hipMemAccessDesc desc = {};
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = static_cast<int>(executor_->device_ordinal());
  desc.flags = hipMemAccessFlagsProtReadWrite;
  return ToStatus(
      wrap::hipMemSetAccess(ptr_ + reservation_offset, size, &desc, 1));
}

absl::Status RocmMemoryReservation::UnMap(size_t offset, size_t size) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  return ToStatus(wrap::hipMemUnmap(ptr_ + offset, size));
}

RocmMemoryReservation::~RocmMemoryReservation() {
  if (ptr_ == nullptr) {
    return;
  }
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  auto unmap_status =
      ToStatus(wrap::hipMemUnmap(ptr_, size_), "Error unmapping ROCm memory");
  if (!unmap_status.ok()) {
    LOG(ERROR) << unmap_status.message();
  }
  auto free_status = ToStatus(wrap::hipMemAddressFree(ptr_, size_),
                              "Error freeing ROCm address range");
  if (!free_status.ok()) {
    LOG(ERROR) << free_status.message();
  }
}

}  // namespace stream_executor::gpu
