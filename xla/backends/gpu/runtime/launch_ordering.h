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

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/event_pool.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

// Launch-ordering events, kept apart from every other user of EventPool.
class LaunchEventPool : public EventPool {
 public:
  using EventPool::EventPool;
};

// Kernels on one stream run in order. Across streams the GPU dispatches
// kernels as they become ready, not in the order the host issued them, so a
// compute kernel issued after a collective can take the SMs first and make the
// collective wait.
//
// A launch-completion event fires once all blocks of a kernel are dispatched,
// unlike a recorded event, which fires once the kernel completes. The thunk
// whose kernel records one is a producer, and every later thunk is a consumer
// whose stream waits on the events recorded before it, so its kernels start
// after the kernel of the producer is on the SMs, while that kernel still
// runs. Producers are NCCL, ragged all-to-all and XLA collective kernels.
// Compute kernels record no event and keep stream order among themselves. A
// wait needs only the latest launch of each other stream, so one
// LaunchOrdering per execution keeps one launch per stream. In
//
//   fusion.0, all-reduce.1, fusion.2, all-gather.3, fusion.4
//
// fusion.2 and all-gather.3 wait on all-reduce.1, fusion.4 on all-gather.3. A
// stream waits on each event once, and not on events recorded on itself.
class LaunchOrdering {
 public:
  explicit LaunchOrdering(LaunchEventPool* pool) : pool_(pool) {}

  // Returns the event the producer must record on its launch on stream.
  absl::StatusOr<stream_executor::Event*> TakeLaunchEvent(
      stream_executor::Stream* stream, absl::string_view producer);

  // Waits on launches other streams recorded since the stream last waited here.
  absl::Status WaitForLaunchEvents(stream_executor::Stream* stream,
                                   absl::string_view consumer);

 private:
  struct Launch {
    size_t number;  // counts the launches of the execution from one
    LaunchEventPool::Event event;
  };

  LaunchEventPool* pool_;
  size_t launch_count_ = 0;
  absl::flat_hash_map<stream_executor::Stream*, Launch> latest_;
  absl::flat_hash_map<stream_executor::Stream*, size_t> waited_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_LAUNCH_ORDERING_H_
