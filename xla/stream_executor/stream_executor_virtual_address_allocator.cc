/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/stream_executor_virtual_address_allocator.h"

#include <cstdint>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/numbers.h"

namespace stream_executor {

DeviceVirtualAddressAllocator::DeviceVirtualAddressAllocator(
    StreamExecutor* executor)
    : DeviceAddressAllocator(executor->GetPlatform()) {
  stream_executors_ = {executor};
}

DeviceVirtualAddressAllocator::DeviceVirtualAddressAllocator(
    const Platform* platform,
    absl::Span<StreamExecutor* const> stream_executors)
    : DeviceAddressAllocator(platform),
      stream_executors_(stream_executors.begin(), stream_executors.end()) {}

absl::StatusOr<ScopedDeviceAddress<uint8_t>>
DeviceVirtualAddressAllocator::Allocate(int device_ordinal, uint64_t size,
                                        bool retry_on_failure,
                                        int64_t memory_space) {
  // Handle zero-size allocation.
  if (size == 0) {
    return ScopedDeviceAddress<uint8_t>(DeviceAddressBase(), device_ordinal,
                                        this);
  }

  TF_ASSIGN_OR_RETURN(StreamExecutor * executor,
                      GetStreamExecutor(device_ordinal));

  // Use VmmAllocateMemory which handles the full flow: reserve virtual address,
  // allocate physical memory, map, and set access permissions.
  TF_ASSIGN_OR_RETURN(DeviceAddressBase result,
                      executor->VmmAllocateMemory(size));

  VLOG(3) << absl::StreamFormat(
      "Allocated virtual address %s (%uB) on device ordinal %d: %p",
      tsl::strings::HumanReadableNumBytes(size), size, device_ordinal,
      result.opaque());

  return ScopedDeviceAddress<uint8_t>(result, device_ordinal, this);
}

absl::Status DeviceVirtualAddressAllocator::Deallocate(int device_ordinal,
                                                       DeviceAddressBase mem) {
  if (mem.is_null()) {
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(StreamExecutor * executor,
                      GetStreamExecutor(device_ordinal));

  VLOG(3) << absl::StreamFormat(
      "Freeing virtual address %p (size=%uB) on device ordinal %d",
      mem.opaque(), mem.size(), device_ordinal);

  // StreamExecutor::Deallocate handles VMM memory deallocation internally
  // (unmap, release physical memory, free virtual address).
  executor->Deallocate(&mem);

  return absl::OkStatus();
}

absl::StatusOr<StreamExecutor*>
DeviceVirtualAddressAllocator::GetStreamExecutor(int device_ordinal) const {
  if (device_ordinal < 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "device ordinal value (%d) must be non-negative", device_ordinal));
  }
  for (StreamExecutor* se : stream_executors_) {
    if (se->device_ordinal() == device_ordinal) {
      return se;
    }
  }
  return absl::NotFoundError(
      absl::StrFormat("Device %s:%d present but not supported",
                      platform()->Name(), device_ordinal));
}

bool DeviceVirtualAddressAllocator::AllowsAsynchronousDeallocation() const {
  return false;
}

absl::StatusOr<Stream*> DeviceVirtualAddressAllocator::GetStream(
    int device_ordinal) {
  CHECK(!AllowsAsynchronousDeallocation())
      << "The logic below only works for synchronous allocators";
  TF_ASSIGN_OR_RETURN(StreamExecutor * executor,
                      GetStreamExecutor(device_ordinal));
  absl::MutexLock lock(&mutex_);
  if (!streams_.count(device_ordinal)) {
    TF_ASSIGN_OR_RETURN(auto stream, executor->CreateStream());
    auto stream_ptr = stream.get();
    stream_ptr->SetName("DeviceVirtualAddressAllocator");
    streams_.emplace(device_ordinal, std::move(stream));
    return stream_ptr;
  }
  return streams_.at(device_ordinal).get();
}

}  // namespace stream_executor
