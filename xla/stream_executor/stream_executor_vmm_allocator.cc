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

#include "xla/stream_executor/stream_executor_vmm_allocator.h"

#include <cstdint>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/numbers.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace stream_executor {

DeviceAddressVmmAllocator::DeviceAddressVmmAllocator(StreamExecutor* executor,
                                                     Stream* stream,
                                                     uint64_t pa_budget)
    : DeviceAddressAllocator(executor->GetPlatform()),
      allocation_granularity_(executor->GetAllocationGranularity()) {
  stream_executors_ = {executor};
  streams_[executor->device_ordinal()] = stream;
  allocator_pa_budget_[executor->device_ordinal()] = pa_budget;
  VLOG(3) << "DeviceAddressVmmAllocator created for device "
          << executor->device_ordinal() << " with pa_budget " << pa_budget;
}

DeviceAddressVmmAllocator::DeviceAddressVmmAllocator(
    const Platform* platform, absl::Span<const DeviceInfo> devices)
    : DeviceAddressAllocator(platform) {
  CHECK_EQ(devices.size(), 1)
      << "DeviceAddressVmmAllocator only supports single-device "
         "instantiation. Got "
      << devices.size() << " devices.";
  stream_executors_.reserve(devices.size());
  for (const auto& device : devices) {
    stream_executors_.push_back(device.executor);
    streams_[device.executor->device_ordinal()] = device.stream;
    allocator_pa_budget_[device.executor->device_ordinal()] = device.pa_budget;
    // Initialize allocation granularity from the first device.
    if (allocation_granularity_ == 0) {
      allocation_granularity_ = device.executor->GetAllocationGranularity();
    }
    VLOG(3) << "DeviceAddressVmmAllocator created for device "
            << device.executor->device_ordinal() << " with pa_budget "
            << device.pa_budget;
  }
}

DeviceAddressVmmAllocator::~DeviceAddressVmmAllocator() {
  // Process all remaining pending deallocations synchronously.
  for (auto& pending : pending_deallocations_) {
    // Wait for the event to complete.
    if (pending.event) {
      auto status = pending.event->Synchronize();
      if (!status.ok()) {
        LOG(WARNING) << "Failed to synchronize event during cleanup: "
                     << status;
      }
    }
    DoDeallocate(pending.device_ordinal, pending.mem);
  }
  pending_deallocations_.clear();
}

void DeviceAddressVmmAllocator::ProcessCompletedPendingDeallocations() {
  // Process pending deallocations whose events have completed.
  while (!pending_deallocations_.empty()) {
    auto& front = pending_deallocations_.front();

    // Check if the event has completed.
    if (front.event) {
      Event::Status status = front.event->PollForStatus();
      if (status == Event::Status::kPending) {
        // Not ready yet, stop processing.
        break;
      }
      if (status == Event::Status::kError) {
        LOG(WARNING) << "Event error while processing pending deallocation";
      }
    }

    // Event completed (or no event), perform the actual deallocation.
    DoDeallocate(front.device_ordinal, front.mem);

    pending_deallocations_.pop_front();
  }
}

void DeviceAddressVmmAllocator::WaitPendingDeallocationsToComplete(
    uint64_t size) {
  if (pending_deallocations_.empty()) {
    return;
  }

  // Determine how many pending deallocations from the front of the queue
  // we need to wait for to free enough memory for the requested size.
  // Use rounded sizes since physical memory is allocated at granularity.
  uint64_t accumulated_size = 0;
  size_t count_to_wait = 0;
  uint64_t rounded_size = RoundUpToGranularity(size);

  // Target 1.1x the requested size to provide some headroom.
  uint64_t target_size = rounded_size + rounded_size / 10;

  for (const auto& pending : pending_deallocations_) {
    accumulated_size += RoundUpToGranularity(pending.mem.size());
    ++count_to_wait;
    if (accumulated_size >= target_size) {
      break;
    }
  }

  if (count_to_wait == 0) {
    return;
  }

  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat(
        "WaitPendingDeallocations:#count=%zu,target_size=%s#", count_to_wait,
        tsl::strings::HumanReadableNumBytes(target_size));
  });

  // Wait only for the last event from the selected deallocations. Since events
  // are recorded in order on the stream, waiting for the last event guarantees
  // all previous events have also completed.
  auto& last_selected = pending_deallocations_[count_to_wait - 1];

  if (last_selected.event) {
    auto status = last_selected.event->Synchronize();
    if (!status.ok()) {
      LOG(WARNING) << "Failed to synchronize event during "
                      "WaitPendingDeallocationsToComplete: "
                   << status;
    }
  }

  // Now process and deallocate all selected pending deallocations.
  for (size_t i = 0; i < count_to_wait && !pending_deallocations_.empty();
       ++i) {
    auto& front = pending_deallocations_.front();
    DoDeallocate(front.device_ordinal, front.mem);
    pending_deallocations_.pop_front();
  }
}

void DeviceAddressVmmAllocator::DoDeallocate(int device_ordinal,
                                             DeviceAddressBase mem) {
  auto executor_or = GetStreamExecutor(device_ordinal);
  if (!executor_or.ok()) {
    LOG(WARNING) << "Failed to get executor for deallocation: "
                 << executor_or.status();
    return;
  }
  StreamExecutor* executor = executor_or.value();

  VLOG(3) << absl::StreamFormat(
      "Actually freeing virtual address %p (size=%uB) on device ordinal %d",
      mem.opaque(), mem.size(), device_ordinal);

  executor->Deallocate(&mem);
  allocator_info_[device_ordinal].pa_allocated -=
      RoundUpToGranularity(mem.size());
}

absl::StatusOr<DeviceAddressBase> DeviceAddressVmmAllocator::AllocateWithBudget(
    int device_ordinal, StreamExecutor* executor, uint64_t size) {
  if (size == 0) {
    return DeviceAddressBase(nullptr, 0);
  }
  uint64_t rounded_size = RoundUpToGranularity(size);
  uint64_t pa_allocated = allocator_info_[device_ordinal].pa_allocated;
  if (pa_allocated + rounded_size > allocator_pa_budget_[device_ordinal]) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Not enough PA budget for allocation: pa_allocated=%uB, "
        "rounded_size=%uB, pa_budget=%uB",
        pa_allocated, rounded_size, allocator_pa_budget_[device_ordinal]));
  }

  allocator_info_[device_ordinal].pa_allocated += rounded_size;
  return executor->VmmAllocateMemory(size);
}

absl::StatusOr<ScopedDeviceAddress<uint8_t>>
DeviceAddressVmmAllocator::Allocate(int device_ordinal, uint64_t size,
                                    bool retry_on_failure,
                                    int64_t memory_space) {
  // Handle zero-size allocation.
  if (size == 0) {
    return ScopedDeviceAddress<uint8_t>(DeviceAddressBase(), device_ordinal,
                                        this);
  }

  TF_ASSIGN_OR_RETURN(StreamExecutor * executor,
                      GetStreamExecutor(device_ordinal));

  // Calculate rounded size for reuse matching.
  uint64_t rounded_size = RoundUpToGranularity(size);

  // Try to reuse a completed pending deallocation with matching size.
  std::optional<DeviceAddressBase> reused =
      TryReusePendingDeallocation(device_ordinal, size, rounded_size);
  if (reused.has_value()) {
    return ScopedDeviceAddress<uint8_t>(*reused, device_ordinal, this);
  }

  // Use AllocateWithBudget which handles the full flow: reserve virtual
  // address, allocate physical memory, map, and set access permissions.
  absl::StatusOr<DeviceAddressBase> result =
      AllocateWithBudget(device_ordinal, executor, size);

  // If allocation failed (e.g., out of memory), try processing pending
  // deallocations to free memory, then retry.
  if (!result.ok()) {
    ProcessCompletedPendingDeallocations();
    // Retry allocation after freeing memory.
    result = AllocateWithBudget(device_ordinal, executor, size);
  }

  if (!result.ok()) {
    WaitPendingDeallocationsToComplete(size);
    // Retry allocation after freeing memory.
    result = AllocateWithBudget(device_ordinal, executor, size);
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

  TF_ASSIGN_OR_RETURN(StreamExecutor * executor,
                      GetStreamExecutor(device_ordinal));

  VLOG(3) << absl::StreamFormat(
      "Queueing deferred deallocation for virtual address %p (size=%uB) "
      "on device ordinal %d",
      mem.opaque(), mem.size(), device_ordinal);

  // Create an event and record it on the stream.
  TF_ASSIGN_OR_RETURN(auto event, executor->CreateEvent());

  // Get or create the stream for this device.
  TF_ASSIGN_OR_RETURN(Stream * stream, GetStream(device_ordinal));

  // Record the event on the stream - deallocation will happen after all
  // currently enqueued work completes.
  TF_RETURN_IF_ERROR(stream->RecordEvent(event.get()));

  // Add this deallocation to the pending queue. Completed deallocations will
  // be processed (freed or reused) during the next Allocate() call.
  pending_deallocations_.push_back(
      PendingDeallocation{device_ordinal, mem, std::move(event)});

  return absl::OkStatus();
}

absl::StatusOr<StreamExecutor*> DeviceAddressVmmAllocator::GetStreamExecutor(
    int device_ordinal) const {
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

absl::StatusOr<Stream*> DeviceAddressVmmAllocator::GetStream(
    int device_ordinal) {
  auto it = streams_.find(device_ordinal);
  if (it == streams_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("No stream registered for device ordinal %d. "
                        "DeviceAddressVmmAllocator requires streams to be "
                        "provided at construction time.",
                        device_ordinal));
  }
  return it->second;
}

uint64_t DeviceAddressVmmAllocator::RoundUpToGranularity(uint64_t size) const {
  if (allocation_granularity_ == 0) {
    return size;
  }
  return ((size + allocation_granularity_ - 1) / allocation_granularity_) *
         allocation_granularity_;
}

std::optional<DeviceAddressBase>
DeviceAddressVmmAllocator::TryReusePendingDeallocation(int device_ordinal,
                                                       uint64_t size,
                                                       uint64_t rounded_size) {
  // Search for a pending deallocation with matching rounded size on the same
  // device whose event has already completed.
  for (auto it = pending_deallocations_.begin();
       it != pending_deallocations_.end(); ++it) {
    if (it->device_ordinal != device_ordinal) {
      continue;
    }

    // Check if the rounded size matches.
    uint64_t pending_rounded_size = RoundUpToGranularity(it->mem.size());
    if (pending_rounded_size != rounded_size) {
      continue;
    }

    // We don't need to check if the event has completed and ordering is
    // guaranteed by the stream.
    uint64_t original_size = it->mem.size();
    // Create the reused address with the new requested size but same underlying
    // memory (which has the rounded size).
    DeviceAddressBase reused_mem(it->mem.opaque(), size, it->mem.raw_handle());
    pending_deallocations_.erase(it);

    VLOG(3) << absl::StreamFormat(
        "Reusing pending deallocation: address=%p original_size=%uB "
        "new_size=%uB rounded_size=%uB device=%d",
        reused_mem.opaque(), original_size, size, rounded_size, device_ordinal);

    return reused_mem;
  }

  return std::nullopt;
}

}  // namespace stream_executor
