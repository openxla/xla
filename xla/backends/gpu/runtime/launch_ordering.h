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
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/event_pool.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"

namespace xla::gpu {

// The GPU dispatches kernels in the order they become ready, not in the order
// the host issued them, so a kernel issued later can take the SMs first and
// make an earlier one wait.
//
// A kernel can record an event once all its blocks are dispatched. The thunk
// that records it is a producer, and every later thunk is a consumer whose
// stream waits on the events recorded before it, so its kernels start after
// the kernel of the producer is on the SMs, while that kernel still runs. A
// wait needs only what happened before it, so one LaunchOrdering per execution
// logs the launches as they happen. In
//
//   fusion.0, all-reduce.1, fusion.2, all-gather.3, fusion.4
//
// fusion.2 and all-gather.3 wait on all-reduce.1, fusion.4 on all-gather.3. A
// stream waits on each event once, and not on events recorded on itself.
class LaunchOrdering {
 public:
  explicit LaunchOrdering(LaunchEventPool* pool) : pool_(pool) {}

  // Returns the event the producer records on the kernel it launches on stream.
  absl::StatusOr<stream_executor::Event*> TakeLaunchEvent(
      stream_executor::Stream* stream, absl::string_view producer) {
    ABSL_ASSIGN_OR_RETURN(LaunchEventPool::Event event,
                          pool_->GetOrCreateEvent());
    stream_executor::Event* raw = event->get();
    XLA_VLOG_DEVICE(3, stream->parent()->device_ordinal())
        << "LaunchOrdering: " << producer << " records launch event "
        << launches_.size() << " on stream " << stream;
    launches_.push_back({stream, std::move(event)});
    return raw;
  }

  // Waits on launches other streams recorded since the stream last waited here.
  absl::StatusOr<int> WaitForProducers(stream_executor::Stream* stream,
                                       absl::string_view consumer) {
    int waits = 0;
    for (size_t& cursor = waited_[stream]; cursor < launches_.size();
         ++cursor) {
      if (launches_[cursor].stream == stream) continue;
      ABSL_RETURN_IF_ERROR(stream->WaitFor(launches_[cursor].event->get()));
      ++waits;
    }
    if (waits > 0) {
      XLA_VLOG_DEVICE(3, stream->parent()->device_ordinal())
          << "LaunchOrdering: " << consumer << " waits on " << waits
          << " launch event(s) on stream " << stream;
    }
    return waits;
  }

 private:
  struct Launch {
    stream_executor::Stream* stream;
    LaunchEventPool::Event event;
  };

  LaunchEventPool* pool_;
  std::vector<Launch> launches_;
  absl::flat_hash_map<stream_executor::Stream*, size_t> waited_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_LAUNCH_ORDERING_H_
