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

#ifndef XLA_PJRT_SE_PRIORITIZED_COMPUTE_STREAM_H_
#define XLA_PJRT_SE_PRIORITIZED_COMPUTE_STREAM_H_

#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/pjrt/se/buffer_sequencing_event.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"

namespace xla {

// Name of the ExecuteOptions::custom_options that carries the requested GPU
// execution priority.
inline constexpr absl::string_view kExecutionPriorityCustomOption =
    "execution_priority";

// A compute stream created at a given priority, together with the bookkeeping
// needed to run executions on it concurrently with the device's other compute
// streams. Every device has at least its default compute stream; additional
// prioritized compute streams exist only when
// `LocalDeviceState::StreamOptions::additional_compute_stream_priorities` is
// set. Instances are owned by the `LocalDeviceState` and live as long as it
// does.
class PrioritizedComputeStream {
 public:
  // Wraps the device's default compute stream without taking ownership of it.
  static std::unique_ptr<PrioritizedComputeStream> WrapDefault(
      int priority, se::Stream* default_stream);

  // Wraps an additional compute stream and takes ownership of it.
  static std::unique_ptr<PrioritizedComputeStream> CreateOwned(
      int priority, std::unique_ptr<se::Stream> stream);

  ~PrioritizedComputeStream();
  PrioritizedComputeStream(const PrioritizedComputeStream&) = delete;
  PrioritizedComputeStream& operator=(const PrioritizedComputeStream&) = delete;

  // Priority the stream was created with, in the platform's convention.
  int priority() const { return priority_; }

  // The compute stream itself.
  se::Stream* stream() const { return stream_; }

  // True for the device's default compute stream
  // (`LocalDeviceState::compute_stream()`).
  bool is_default() const { return is_default_; }

  // Destroys the owned stream, if any, while keeping the rest of this object
  // (such as `allocator()`) alive for pending deferred-deallocation callbacks.
  // No-op for the default compute stream.
  void ResetOwnedStream();

  // Installs `allocator()` on top of `base_allocator`, which must outlive this
  // object.
  void InitializeAllocator(se::DeviceAddressAllocator* base_allocator);

  // Allocator that executions on this stream must allocate through, or nullptr
  // when the device has no additional prioritized compute streams. It forwards
  // allocations to the device allocator and routes deallocations through
  // `ReleaseAfterPendingWork()`, so that memory used by work on one compute
  // stream can never be recycled by work on another compute stream that
  // overtakes it.
  se::DeviceAddressAllocator* allocator() const { return allocator_.get(); }

  // Returns `memory` to the device allocator once the work enqueued on this
  // stream so far has completed on the device. While an execution is being
  // enqueued on the current thread (see ScopedDeferredDeallocationCollector),
  // the release is collected instead so that the caller can tie it to that
  // execution's definition event.
  void ReleaseAfterPendingWork(int device_ordinal,
                               se::DeviceAddressBase memory);

  // Definition event of the most recent execution enqueued on this stream, or
  // a null reference if none has been recorded yet.
  BufferSequencingEventRef last_event();
  void set_last_event(BufferSequencingEventRef event);

  // A deallocation deferred by `ReleaseAfterPendingWork()`.
  struct DeferredDeallocation {
    int device_ordinal;
    se::DeviceAddressBase memory;
  };

  // Performs deferred deallocations against the device allocator.
  void PerformDeferredDeallocations(
      absl::Span<const DeferredDeallocation> deallocations);

 private:
  PrioritizedComputeStream(int priority, se::Stream* stream, bool is_default,
                           std::unique_ptr<se::Stream> owned_stream);

  const int priority_;
  se::Stream* stream_;
  const bool is_default_;
  // Owns `stream_` for streams other than the default compute stream, which is
  // owned by the `LocalDeviceState`.
  std::unique_ptr<se::Stream> owned_stream_;
  // The device allocator that `allocator_` wraps; set by InitializeAllocator().
  se::DeviceAddressAllocator* base_allocator_ = nullptr;
  std::unique_ptr<se::DeviceAddressAllocator> allocator_;
  absl::Mutex mu_;
  BufferSequencingEventRef last_event_ ABSL_GUARDED_BY(mu_);
};

// While alive on the current thread, releases requested through
// `stream->ReleaseAfterPendingWork()` (and therefore through
// `stream->allocator()`) are collected instead of performed. The collected
// deallocations are returned by `Take()`; the caller performs them through
// `PrioritizedComputeStream::PerformDeferredDeallocations` once the execution
// that released the memory has completed on the device. Deallocations that
// were not taken are deferred to the stream's last event when the collector is
// destroyed.
class ScopedDeferredDeallocationCollector {
 public:
  explicit ScopedDeferredDeallocationCollector(
      PrioritizedComputeStream* stream);
  ~ScopedDeferredDeallocationCollector();
  ScopedDeferredDeallocationCollector(
      const ScopedDeferredDeallocationCollector&) = delete;
  ScopedDeferredDeallocationCollector& operator=(
      const ScopedDeferredDeallocationCollector&) = delete;

  // The innermost collector alive on the calling thread, or nullptr. Collectors
  // are tracked per thread because an enqueue is a property of the thread
  // performing it: several threads may enqueue on the same stream at once, and
  // the frees issued by each enqueue must be tied to that enqueue's execution.
  static ScopedDeferredDeallocationCollector* Current();

  // The stream whose releases this collector captures.
  PrioritizedComputeStream* stream() const { return stream_; }

  std::vector<PrioritizedComputeStream::DeferredDeallocation> Take();

 private:
  friend class PrioritizedComputeStream;

  PrioritizedComputeStream* const stream_;
  std::vector<PrioritizedComputeStream::DeferredDeallocation> deallocations_;
  ScopedDeferredDeallocationCollector* const previous_;
};

}  // namespace xla

#endif  // XLA_PJRT_SE_PRIORITIZED_COMPUTE_STREAM_H_
