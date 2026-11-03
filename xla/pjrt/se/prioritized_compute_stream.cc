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

#include "xla/pjrt/se/prioritized_compute_stream.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/pjrt/se/buffer_sequencing_event.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace {

// Innermost ScopedDeferredDeallocationCollector alive on this thread, if any
// (see ScopedDeferredDeallocationCollector::Current()).
ABSL_CONST_INIT thread_local ScopedDeferredDeallocationCollector*
    current_deferred_deallocation_collector = nullptr;

// Thin adapter installed as the execution allocator of a prioritized compute
// stream: allocations go straight to the device allocator, deallocations go
// through PrioritizedComputeStream::ReleaseAfterPendingWork().
class CompletionOrderedAllocator : public se::DeviceAddressAllocator {
 public:
  CompletionOrderedAllocator(se::DeviceAddressAllocator* base,
                             PrioritizedComputeStream* compute_stream)
      : se::DeviceAddressAllocator(base->platform()),
        base_(base),
        compute_stream_(compute_stream) {}

  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override {
    ABSL_ASSIGN_OR_RETURN(
        se::ScopedDeviceAddress<uint8_t> address,
        base_->Allocate(device_ordinal, size, retry_on_failure, memory_space));
    // Re-own the address so that any release goes through this allocator.
    return se::ScopedDeviceAddress<uint8_t>(address.Release(), device_ordinal,
                                            this);
  }

  absl::Status Deallocate(int device_ordinal,
                          se::DeviceAddressBase mem) override {
    if (!mem.is_null()) {
      compute_stream_->ReleaseAfterPendingWork(device_ordinal, mem);
    }
    return absl::OkStatus();
  }

  bool AllowsAsynchronousDeallocation() const override {
    return base_->AllowsAsynchronousDeallocation();
  }

  absl::StatusOr<se::Stream*> GetStream(int device_ordinal) override {
    return compute_stream_->stream();
  }

  bool ClearAllocatorStats(int device_ordinal) override {
    return base_->ClearAllocatorStats(device_ordinal);
  }

 private:
  se::DeviceAddressAllocator* const base_;
  PrioritizedComputeStream* const compute_stream_;
};

}  // namespace

std::unique_ptr<PrioritizedComputeStream> PrioritizedComputeStream::WrapDefault(
    int priority, se::Stream* default_stream) {
  return absl::WrapUnique(new PrioritizedComputeStream(
      priority, default_stream, /*is_default=*/true, /*owned_stream=*/nullptr));
}

std::unique_ptr<PrioritizedComputeStream> PrioritizedComputeStream::CreateOwned(
    int priority, std::unique_ptr<se::Stream> stream) {
  se::Stream* raw_stream = stream.get();
  return absl::WrapUnique(new PrioritizedComputeStream(
      priority, raw_stream, /*is_default=*/false, std::move(stream)));
}

PrioritizedComputeStream::PrioritizedComputeStream(
    int priority, se::Stream* stream, bool is_default,
    std::unique_ptr<se::Stream> owned_stream)
    : priority_(priority),
      stream_(stream),
      is_default_(is_default),
      owned_stream_(std::move(owned_stream)) {}

PrioritizedComputeStream::~PrioritizedComputeStream() = default;

void PrioritizedComputeStream::ResetOwnedStream() {
  if (owned_stream_ != nullptr) {
    stream_ = nullptr;
    owned_stream_.reset();
  }
}

void PrioritizedComputeStream::InitializeAllocator(
    se::DeviceAddressAllocator* base_allocator) {
  base_allocator_ = base_allocator;
  allocator_ =
      std::make_unique<CompletionOrderedAllocator>(base_allocator, this);
}

void PrioritizedComputeStream::ReleaseAfterPendingWork(
    int device_ordinal, se::DeviceAddressBase memory) {
  CHECK(base_allocator_ != nullptr);
  // Inside an execution's enqueue on this thread: let the caller tie the
  // release to that execution's definition event.
  if (ScopedDeferredDeallocationCollector* collector =
          ScopedDeferredDeallocationCollector::Current();
      collector != nullptr && collector->stream() == this) {
    collector->deallocations_.push_back({device_ordinal, memory});
    return;
  }
  BufferSequencingEventRef last_event = this->last_event();
  if (!last_event || last_event.IsAvailable()) {
    absl::Status status = base_allocator_->Deallocate(device_ordinal, memory);
    if (!status.ok()) {
      LOG(ERROR) << "Deallocation failed: " << status;
    }
    return;
  }
  last_event.AndThen([base = base_allocator_, device_ordinal, memory]() {
    absl::Status status = base->Deallocate(device_ordinal, memory);
    if (!status.ok()) {
      LOG(ERROR) << "Deferred deallocation failed: " << status;
    }
  });
}

BufferSequencingEventRef PrioritizedComputeStream::last_event() {
  absl::MutexLock lock(mu_);
  return last_event_;
}

void PrioritizedComputeStream::set_last_event(BufferSequencingEventRef event) {
  absl::MutexLock lock(mu_);
  last_event_ = std::move(event);
}

void PrioritizedComputeStream::PerformDeferredDeallocations(
    absl::Span<const DeferredDeallocation> deallocations) {
  CHECK(base_allocator_ != nullptr);
  for (const DeferredDeallocation& deallocation : deallocations) {
    absl::Status status = base_allocator_->Deallocate(
        deallocation.device_ordinal, deallocation.memory);
    if (!status.ok()) {
      LOG(ERROR) << "Deferred deallocation failed: " << status;
    }
  }
}

ScopedDeferredDeallocationCollector*
ScopedDeferredDeallocationCollector::Current() {
  return current_deferred_deallocation_collector;
}

ScopedDeferredDeallocationCollector::ScopedDeferredDeallocationCollector(
    PrioritizedComputeStream* stream)
    : stream_(stream), previous_(current_deferred_deallocation_collector) {
  current_deferred_deallocation_collector = this;
}

ScopedDeferredDeallocationCollector::~ScopedDeferredDeallocationCollector() {
  CHECK_EQ(current_deferred_deallocation_collector, this)
      << "ScopedDeferredDeallocationCollector destroyed out of order";
  current_deferred_deallocation_collector = previous_;
  if (deallocations_.empty()) {
    return;
  }
  // Not taken by the caller: defer to the stream's most recent execution.
  BufferSequencingEventRef last_event = stream_->last_event();
  std::vector<PrioritizedComputeStream::DeferredDeallocation> deallocations =
      std::move(deallocations_);
  if (!last_event || last_event.IsAvailable()) {
    stream_->PerformDeferredDeallocations(deallocations);
    return;
  }
  last_event.AndThen(
      [stream = stream_, deallocations = std::move(deallocations)]() {
        stream->PerformDeferredDeallocations(deallocations);
      });
}

std::vector<PrioritizedComputeStream::DeferredDeallocation>
ScopedDeferredDeallocationCollector::Take() {
  return std::move(deallocations_);
}

}  // namespace xla
