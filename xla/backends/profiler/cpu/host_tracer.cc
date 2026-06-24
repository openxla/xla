/* Copyright 2018 The OpenXLA Authors.

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
#include "xla/backends/profiler/cpu/host_tracer.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/profiler/backends/cpu/host_tracer_utils.h"
#include "xla/tsl/profiler/backends/cpu/threadpool_listener.h"
#include "xla/tsl/profiler/backends/cpu/traceme_recorder.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/lib/profiler_collection.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

#define TRACEME_FILTER_DEFAULT_MASK std::numeric_limits<uint64_t>::max()

// Controls TraceMeRecorder and converts TraceMeRecorder::Events into XEvents.
//
// Thread-safety: This class is go/thread-compatible.
class HostTracer : public tsl::profiler::ProfilerInterface {
 public:
  explicit HostTracer(int host_trace_level);
  explicit HostTracer(int host_trace_level, uint64_t filter_mask,
                      bool enable_source_location = true);
  ~HostTracer() override;

  absl::Status Start() override;  // TENSORFLOW_STATUS_OK

  absl::Status Stop() override;  // TENSORFLOW_STATUS_OK

  absl::Status CollectData(  // TENSORFLOW_STATUS_OK
      tensorflow::profiler::XSpace* space) override;

  absl::StatusOr<tsl::profiler::ProfilerInterface::ConsumeResult> Consume()
      override;

  absl::Status Serialize(std::any data,
                         tensorflow::profiler::XSpace* space) override;

 private:
  // Level of host tracing.
  const int host_trace_level_;

  // Filter mask for host tracing.
  const uint64_t filter_mask_;

  // Whether to enable source location in TraceMe encode.
  const bool enable_source_location_;

  // True if currently recording.
  bool recording_ = false;

  // Timestamp at the start of tracing.
  uint64_t start_timestamp_ns_ = 0;

  // Container of all traced events.
  tsl::profiler::TraceMeRecorder::Events events_;
};

HostTracer::HostTracer(int host_trace_level)
    : host_trace_level_(host_trace_level),
      filter_mask_(TRACEME_FILTER_DEFAULT_MASK),
      enable_source_location_(true) {}

HostTracer::HostTracer(int host_trace_level, uint64_t filter_mask,
                       bool enable_source_location)
    : host_trace_level_(host_trace_level),
      filter_mask_(filter_mask),
      enable_source_location_(enable_source_location) {}

HostTracer::~HostTracer() { Stop().IgnoreError(); }  // NOLINT

absl::Status HostTracer::Start() {  // TENSORFLOW_STATUS_OK
  LOG(INFO) << "HostTracer::Start called";
  if (recording_) {
    return absl::InternalError("TraceMeRecorder already started");
  }

  // All TraceMe captured should have a timestamp greater or equal to
  // start_timestamp_ns_ to prevent timestamp underflow in XPlane.
  // Therefore this have to be done before TraceMeRecorder::Start.
  start_timestamp_ns_ = tsl::profiler::GetCurrentTimeNanos();
  recording_ = tsl::profiler::TraceMeRecorder::Start(
      host_trace_level_, filter_mask_, enable_source_location_);
  if (!recording_) {
    return absl::InternalError("Failed to start TraceMeRecorder");
  }
  return absl::OkStatus();
}

absl::Status HostTracer::Stop() {  // TENSORFLOW_STATUS_OK
  if (!recording_) {
    return absl::InternalError("TraceMeRecorder not started");
  }
  events_ = tsl::profiler::TraceMeRecorder::Stop();
  recording_ = false;
  return absl::OkStatus();
}

absl::Status HostTracer::CollectData(  // TENSORFLOW_STATUS_OK
    tensorflow::profiler::XSpace* space) {
  VLOG(2) << "Collecting data to XSpace from HostTracer.";
  if (recording_) {
    return absl::InternalError("TraceMeRecorder not stopped");
  }
  if (events_.empty()) {
    return absl::OkStatus();
  }
  tensorflow::profiler::XPlane* plane =
      tsl::profiler::FindOrAddMutablePlaneWithName(
          space, tsl::profiler::kHostThreadsPlaneName);
  ConvertCompleteEventsToXPlane(start_timestamp_ns_, std::exchange(events_, {}),
                                plane);
  return absl::OkStatus();
}

absl::StatusOr<tsl::profiler::ProfilerInterface::ConsumeResult>
HostTracer::Consume() {
  LOG(INFO) << "HostTracer::Consume called, recording=" << recording_;
  tsl::profiler::TraceMeRecorder::Events events;
  if (recording_) {
    events = tsl::profiler::TraceMeRecorder::Flush();
  } else {
    events = std::exchange(events_, {});
  }
  size_t total_events = 0;
  for (const auto& thread : events) {
    total_events += thread.events.size();
  }
  LOG(INFO) << "HostTracer::Consume: flushed " << events.size()
            << " threads, total events=" << total_events;

  size_t estimated_size = 0;
  for (const auto& thread : events) {
    estimated_size += sizeof(thread.thread);
    estimated_size += thread.thread.name.capacity();
    estimated_size +=
        thread.events.size() * sizeof(tsl::profiler::TraceMeRecorder::Event);
    for (const auto& event : thread.events) {
      estimated_size += event.name.capacity();
    }
  }

  tsl::profiler::ProfilerInterface::ConsumeResult result;
  result.data = std::any(std::move(events));
  result.estimated_size_bytes = estimated_size;
  return result;
}

absl::Status HostTracer::Serialize(std::any data,
                                   tensorflow::profiler::XSpace* space) {
  LOG(INFO) << "HostTracer::Serialize called";
  if (space == nullptr) {
    return absl::InvalidArgumentError("XSpace pointer cannot be null.");
  }
  auto* events = std::any_cast<tsl::profiler::TraceMeRecorder::Events>(&data);
  if (events == nullptr) {
    return absl::InvalidArgumentError("Invalid data type passed to Serialize.");
  }
  if (events->empty()) {
    LOG(INFO) << "HostTracer::Serialize: events is empty, doing nothing";
    return absl::OkStatus();
  }
  size_t total_events = 0;
  for (const auto& thread : *events) {
    total_events += thread.events.size();
  }
  LOG(INFO) << "HostTracer::Serialize: serializing " << events->size()
            << " threads, total events=" << total_events;
  tensorflow::profiler::XPlane* plane =
      tsl::profiler::FindOrAddMutablePlaneWithName(
          space, tsl::profiler::kHostThreadsPlaneName);
  ConvertCompleteEventsToXPlane(start_timestamp_ns_, std::move(*events), plane);
  events->clear();
  return absl::OkStatus();
}

}  // namespace

std::unique_ptr<tsl::profiler::ProfilerInterface> CreateHostTracer(
    const HostTracerOptions& options) {
  if (options.trace_level == 0) return nullptr;
  std::vector<std::unique_ptr<tsl::profiler::ProfilerInterface>> profilers;
  profilers.push_back(
      std::make_unique<HostTracer>(options.trace_level, options.filter_mask,
                                   options.enable_source_location));
  profilers.push_back(
      std::make_unique<tsl::profiler::ThreadpoolProfilerInterface>());
  return std::make_unique<tsl::profiler::ProfilerCollection>(
      std::move(profilers));
}

}  // namespace profiler
}  // namespace xla
