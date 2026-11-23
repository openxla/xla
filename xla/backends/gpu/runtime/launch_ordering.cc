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

#include "xla/backends/gpu/runtime/launch_ordering.h"

#include <cstddef>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/event_pool.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::gpu {

absl::StatusOr<stream_executor::Event*> LaunchOrdering::TakeLaunchEvent(
    stream_executor::Stream* stream, absl::string_view producer) {
  ABSL_ASSIGN_OR_RETURN(LaunchEventPool::Event event,
                        pool_->GetOrCreateEvent());
  stream_executor::Event* raw = event->get();
  XLA_VLOG_DEVICE(3, stream->parent()->device_ordinal())
      << producer << " records launch event " << launch_count_ << " on "
      << stream;
  latest_[stream] = {++launch_count_, std::move(event)};
  return raw;
}

absl::Status LaunchOrdering::WaitForLaunchEvents(
    stream_executor::Stream* stream, absl::string_view consumer) {
  int waits = 0;
  size_t& waited = waited_[stream];
  for (auto& [producer, launch] : latest_) {
    if (producer == stream || launch.number <= waited) {
      continue;
    }
    ABSL_RETURN_IF_ERROR(stream->WaitFor(launch.event->get()));
    ++waits;
  }
  waited = launch_count_;
  if (waits > 0) {
    XLA_VLOG_DEVICE(3, stream->parent()->device_ordinal())
        << consumer << " waits on " << waits << " launch events on " << stream;
    tsl::profiler::TraceMe::InstantActivity([&] {
      return absl::StrFormat("LaunchOrdering: %s waits on %d launch(es)",
                             consumer, waits);
    });
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu
