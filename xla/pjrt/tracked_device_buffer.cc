/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/pjrt/tracked_device_buffer.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/event_pool.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/se_raw_buffer.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/event.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"

namespace xla {

ShapedBuffer RawSEDeviceMemory::AsShapedBuffer(
    PjRtDevice* device, const Shape& on_device_shape) const {
  ShapedBuffer shaped_buffer(on_device_shape, device->local_device_id().value(),
                             device->local_hardware_id().value());
  ShapeTree<se::DeviceMemoryBase>::iterator iterator =
      shaped_buffer.buffers().begin();
  CHECK(iterator != shaped_buffer.buffers().end());
  iterator->second = mem();
  ++iterator;
  CHECK(iterator == shaped_buffer.buffers().end());
  return shaped_buffer;
}

class AllocatedRawSEDeviceMemory : public RawSEDeviceMemory {
 public:
  AllocatedRawSEDeviceMemory(se::DeviceMemoryBase value, int device_ordinal,
                             se::DeviceMemoryAllocator* allocator)
      : RawSEDeviceMemory(value),
        allocator_(allocator),
        device_ordinal_(device_ordinal) {}

  ~AllocatedRawSEDeviceMemory() override {
    if (allocator_) {
      absl::Status status = allocator_->Deallocate(device_ordinal_, mem());
      if (!status.ok()) {
        LOG(ERROR) << "Buffer deallocation failed: " << status;
      }
    }
  }

  void UnsafeReleaseMemory() override { allocator_ = nullptr; }

 private:
  se::DeviceMemoryAllocator* allocator_;
  int device_ordinal_;
};

tsl::RCReference<RawSEDeviceMemory> RawSEDeviceMemory::Create(
    se::DeviceMemoryBase value, PjRtLocalDeviceId device_id,
    se::DeviceMemoryAllocator* allocator) {
  return tsl::MakeRef<AllocatedRawSEDeviceMemory>(value, device_id.value(),
                                                  allocator);
}

class ForeignRawSEDeviceMemory : public RawSEDeviceMemory {
 public:
  ForeignRawSEDeviceMemory(se::DeviceMemoryBase value,
                           absl::AnyInvocable<void() &&> on_delete_callback)
      : RawSEDeviceMemory(value),
        on_delete_callback_(std::move(on_delete_callback)) {}

  ~ForeignRawSEDeviceMemory() override { std::move(on_delete_callback_)(); }

  void UnsafeReleaseMemory() override {
    LOG(FATAL) << "ForeignRawSEDeviceMemory cannot be donated.";
  }

 private:
  absl::AnyInvocable<void() &&> on_delete_callback_;
};

tsl::RCReference<RawSEDeviceMemory> RawSEDeviceMemory::CreateForeign(
    se::DeviceMemoryBase value,
    absl::AnyInvocable<void() &&> on_delete_callback) {
  return tsl::MakeRef<ForeignRawSEDeviceMemory>(value,
                                                std::move(on_delete_callback));
}

ShapedBuffer TrackedDeviceBuffer::AsShapedBuffer(
    const Shape& on_device_shape) const {
  ShapedBuffer shaped_buffer(on_device_shape,
                             device_->local_device_id().value(),
                             device_->local_hardware_id().value());
  ShapeTree<se::DeviceMemoryBase>::iterator iterator =
      shaped_buffer.buffers().begin();
  if (device_memory_) {
    CHECK(iterator != shaped_buffer.buffers().end());
    iterator->second = device_memory_->mem();
    ++iterator;
  }
  CHECK(iterator == shaped_buffer.buffers().end());
  return shaped_buffer;
}

TrackedDeviceBuffer::TrackedDeviceBuffer(
    PjRtDevice* device, tsl::RCReference<RawSEDeviceMemory> device_memory,
    absl::Span<const BufferSequencingEventRef> definition_events)
    : device_(device),
      device_memory_(std::move(device_memory)),
      definition_events_(std::make_move_iterator(definition_events.begin()),
                         std::make_move_iterator(definition_events.end())),
      in_use_(true) {}

TrackedDeviceBuffer::~TrackedDeviceBuffer() = default;

void TrackedDeviceBuffer::ReleaseDeviceMemory() {
  device_memory_ = tsl::RCReference<RawSEDeviceMemory>();
}

void TrackedDeviceBuffer::ConfirmDonation() {
  // As a sanity check ensure no more usage events can be added to the buffer.
  LockUseAndTransferUsageEvents();
  // Release the memory so that no new usage is possible.
  ReleaseDeviceMemory();
}

void TrackedDeviceBuffer::AddUsageEvent(BufferSequencingEventRef event,
                                        bool reference_held) {
  CHECK(in_use_);

  // If the event is 0, it means that the event is not recorded yet and the task
  // related to this event is deferred, so just add it.
  if (!event->IsDefined()) {
    usage_events_.push_back({event, reference_held});
    return;
  }
  auto* usage_stream = event->definition_stream();

  for (auto& existing : usage_events_) {
    // If the existing event is 0, it means that the event is not recorded yet
    // and the task related to this event is deferred, so don't replace it.
    if (!existing.event->IsDefined()) continue;
    if (existing.event->definition_stream() == usage_stream) {
      if (*existing.event < *event) {
        existing.event = event;
        existing.reference_held = reference_held;
      }
      return;
    }
  }
  usage_events_.push_back({event, reference_held});
}

TrackedDeviceBuffer::StreamAndEventContainer
TrackedDeviceBuffer::LockUseAndTransferUsageEvents() {
  CHECK(in_use_);
  in_use_ = false;
  return std::move(usage_events_);
}

std::vector<tsl::RCReference<tsl::AsyncValue>>
TrackedDeviceBuffer::GetAsyncValueDefinitionEvents() {
  std::vector<tsl::RCReference<tsl::AsyncValue>> avs;
  avs.reserve(definition_events_.size());
  for (const auto& ev : definition_events_) {
    avs.push_back(ev.CopyRCRef());
  }
  return avs;
}

tsl::RCReference<CommonPjRtRawBuffer> TrackedDeviceBuffer::GetRawBuffer(
    PjRtMemorySpace* memory_space) {
  return tsl::MakeRef<PjRtStreamExecutorRawBuffer>(
      tensorflow::down_cast<PjRtStreamExecutorClient*>(memory_space->client()),
      memory_space,
      tensorflow::down_cast<PjRtStreamExecutorDevice*>(
          memory_space->devices()[0])
          ->local_device_state(),
      device_memory_);
}

void TrackedDeviceBuffer::AddUsageEvent(
    tsl::RCReference<PjRtDeviceEvent> event) {
  if (event) {
    AddUsageEvent(
        tensorflow::down_cast<PjRtStreamExecutorDeviceEvent*>(event.get())
            ->event(),
        true);
  }
}

void GetDeviceBufferEvents(
    const TrackedDeviceBuffer& buffer, bool get_usage_events,
    absl::flat_hash_set<BufferSequencingEvent*>* events) {
  if (get_usage_events) {
    for (const auto& e : buffer.usage_events()) {
      events->insert(&*e.event);
    }
  } else {
    for (const auto& e : buffer.definition_events()) {
      events->insert(&*e);
    }
  }
}

void WaitForBufferDefinitionEventsOnStream(
    absl::Span<const BufferSequencingEventRef> definition_events,
    se::Stream* stream) {
  if (definition_events.size() <= 1) {
    for (const auto& event : definition_events) {
      event->WaitForEventOnStream(stream);
    }
  } else {
    absl::flat_hash_set<BufferSequencingEvent*> events;
    for (const auto& event : definition_events) {
      if (events.emplace(&*event).second) {
        event->WaitForEventOnStream(stream);
      }
    }
  }
}

}  // namespace xla
