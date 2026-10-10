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

#ifndef XLA_BACKENDS_GPU_RUNTIME_LAUNCH_ORDERING_H_
#define XLA_BACKENDS_GPU_RUNTIME_LAUNCH_ORDERING_H_

#include <cstddef>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/event_pool.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

// The GPU dispatches kernels in the order they become ready, not in the order
// the host issued them, so a kernel issued later can take the SMs first and
// make an earlier one wait.
//
// A thunk whose kernel records an event once all its blocks are dispatched is a
// producer. Every later thunk is a consumer: before it launches, the executor
// waits on that event (LaunchOrdering::WaitForProducers), so its kernels reach
// the SMs after the producer's kernel, while that kernel still runs. A slot is
// the index of a producer's event. The map is built once per executable. It
// gives each producer its slot and each thunk the slot of the last producer
// ahead of it, for example in
//
//   fusion.0, all-reduce.1, fusion.2, all-gather.3, fusion.4
//
// all-reduce.1 is slot 0, all-gather.3 is slot 1. fusion.2 and all-gather.3
// wait on slot 0, fusion.4 waits on slot 1.
class LaunchDependencyMap {
 public:
  using SlotId = size_t;

  explicit LaunchDependencyMap(
      absl::Span<const Thunk* const> execution_order = {}) {
    std::optional<SlotId> pending;
    for (const Thunk* thunk : execution_order) {
      if (pending) last_producer_[thunk] = *pending;
      if (thunk->RecordsLaunchCompletion()) {
        pending = num_slots_++;
        recorded_slot_[thunk] = *pending;
      }
    }
  }

  // Slot `thunk` records into, if it is a producer.
  std::optional<SlotId> RecordSlot(const Thunk* thunk) const {
    auto it = recorded_slot_.find(thunk);
    return it == recorded_slot_.end() ? std::nullopt
                                      : std::make_optional(it->second);
  }

  // Slot of the last producer ahead of `thunk`, if any.
  std::optional<SlotId> LastProducerSlot(const Thunk* thunk) const {
    auto it = last_producer_.find(thunk);
    return it == last_producer_.end() ? std::nullopt
                                      : std::make_optional(it->second);
  }

  size_t num_slots() const { return num_slots_; }

 private:
  absl::flat_hash_map<const Thunk*, SlotId> recorded_slot_;
  absl::flat_hash_map<const Thunk*, SlotId> last_producer_;
  size_t num_slots_ = 0;
};

// The events of one execution. A producer binds its slot when it takes its
// event. A slot its producer never binds is skipped.
struct LaunchOrdering {
  const LaunchDependencyMap& map;
  absl::Span<const EventPool::Event> events;

  // Makes `stream` wait on every bound producer ahead of `thunk`.
  absl::Status WaitForProducers(const Thunk* thunk,
                                stream_executor::Stream* stream) const {
    std::optional<LaunchDependencyMap::SlotId> last =
        map.LastProducerSlot(thunk);
    if (!last.has_value()) return absl::OkStatus();
    for (size_t& cursor = waited[stream]; cursor <= *last; ++cursor) {
      if (!event_bound[cursor]) continue;
      ABSL_RETURN_IF_ERROR(stream->WaitFor(events[cursor]->get()));
    }
    return absl::OkStatus();
  }

  // Binds `slot`, and rewinds streams that already waited past it.
  void ClaimSlot(LaunchDependencyMap::SlotId slot) const {
    event_bound[slot] = true;
    for (auto& [stream, cursor] : waited) {
      if (cursor > slot) cursor = slot;
    }
  }

  // Per stream: slots below this value have already been waited on.
  mutable absl::flat_hash_map<stream_executor::Stream*, size_t> waited;

  // Slots whose producer took its event. Nothing waits on the rest.
  mutable std::vector<bool> event_bound = std::vector<bool>(map.num_slots());
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_LAUNCH_ORDERING_H_
