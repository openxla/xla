/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/backends/profiler/gpu/rocm_collector.h"
#include "xla/backends/profiler/gpu/rocm_tracer.h"
#include "xla/backends/profiler/gpu/rocm_tracer_options_utils.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/debug_options_flags.h"
#include "xla/tsl/platform/env_time.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

using tensorflow::ProfileOptions;
using tsl::profiler::AnnotationStack;
using tsl::profiler::ProfilerInterface;
using tsl::profiler::XSpace;

namespace {

// Used unless ProfileOptions.advanced_configuration overrides them.
constexpr uint64_t kDefaultMaxAnnotationStrings = 4 * 1024 * 1024;
constexpr uint64_t kDefaultMaxTraceEvents = 4 * 1024 * 1024;

// The callback and activity event caps used unless a gpu_max_* key sets them.
// The deprecated --xla_gpu_rocm_max_trace_events replaces the default, because
// remote capture forwards only a fixed set of keys, so its callers cannot set
// the gpu_max_* keys. Remove the flag once remote capture forwards them.
uint64_t DefaultMaxTraceEvents() {
  int64_t flag = GetDebugOptionsFromFlags().xla_gpu_rocm_max_trace_events();
  if (flag <= 0) return kDefaultMaxTraceEvents;
  return static_cast<uint64_t>(std::min(flag, kMaxRocmTraceEvents));
}

}  // namespace

// GpuTracer for ROCm GPU.
class GpuTracer : public profiler::ProfilerInterface {
 public:
  GpuTracer(RocmTracer* rocm_tracer, const ProfileOptions& profile_options)
      : profile_options_(profile_options), rocm_tracer_(rocm_tracer) {
    LOG(INFO) << "GpuTracer created.";
  }
  ~GpuTracer() override {}

  // GpuTracer interface:
  absl::Status Start() override;
  absl::Status Stop() override;
  absl::Status CollectData(XSpace* space) override;

 private:
  absl::Status DoStart();
  absl::Status DoStop();

  enum State {
    kNotStarted,
    kStartedOk,
    kStartedError,
    kStoppedOk,
    kStoppedError
  };
  State profiling_state_ = State::kNotStarted;

  const ProfileOptions profile_options_;
  RocmTracerOptionDiagnostics option_diagnostics_;
  RocmTracer* rocm_tracer_;
  std::unique_ptr<RocmTraceCollector> rocm_trace_collector_;
};

absl::Status GpuTracer::DoStart() {
  RocmTracerOptions tracer_options;
  tracer_options.max_annotation_strings = kDefaultMaxAnnotationStrings;
  RocmTraceCollectorOptions trace_collector_options;
  const uint64_t default_max_events = DefaultMaxTraceEvents();
  trace_collector_options.max_callback_api_events = default_max_events;
  trace_collector_options.max_activity_api_events = default_max_events;
  trace_collector_options.num_gpus = rocm_tracer_->NumGpus();
  option_diagnostics_ = UpdateRocmTracerOptionsFromProfilerOptions(
      profile_options_, trace_collector_options.num_gpus, tracer_options,
      trace_collector_options);

  // Failing Start() drops the whole GPU trace, so a bad key only does that
  // when the caller asks for it.
  if (!option_diagnostics_.errors.empty() &&
      profile_options_.raise_error_on_start_failure()) {
    // The errors are in the returned status, and the caller logs it.
    for (const std::string& warning : option_diagnostics_.warnings) {
      LOG(WARNING) << warning;
    }
    return absl::InvalidArgumentError(
        absl::StrJoin(option_diagnostics_.errors, " "));
  }

  AnnotationStack::Enable(true);
  uint64_t start_gputime_ns = RocmTracer::GetTimestamp();
  uint64_t start_walltime_ns = tsl::EnvTime::NowNanos();

  rocm_trace_collector_ = CreateRocmCollector(
      trace_collector_options, start_walltime_ns, start_gputime_ns);
  rocm_trace_collector_->SetGpuAgents(rocm_tracer_->GpuAgents());

  absl::Status status =
      rocm_tracer_->Enable(tracer_options, rocm_trace_collector_.get());
  if (!status.ok()) {
    AnnotationStack::Enable(false);
    // CollectData() is not called after a failed Start().
    AppendOptionDiagnostics(option_diagnostics_, /*space=*/nullptr);
    return status;
  }
  return absl::OkStatus();
}

absl::Status GpuTracer::Start() {
  absl::Status status = DoStart();
  if (status.ok()) {
    profiling_state_ = State::kStartedOk;
    return absl::OkStatus();
  } else {
    profiling_state_ = State::kStartedError;
    return status;
  }
}

absl::Status GpuTracer::DoStop() {
  rocm_tracer_->Disable();
  AnnotationStack::Enable(false);
  return absl::OkStatus();
}

absl::Status GpuTracer::Stop() {
  if (profiling_state_ == State::kStartedOk) {
    absl::Status status = DoStop();
    profiling_state_ = status.ok() ? State::kStoppedOk : State::kStoppedError;
  }
  return absl::OkStatus();
}

absl::Status GpuTracer::CollectData(XSpace* space) {
  switch (profiling_state_) {
    case State::kNotStarted:
      VLOG(3) << "No trace data collected, session wasn't started";
      return absl::OkStatus();
    case State::kStartedOk:
      return absl::FailedPreconditionError(
          "Cannot collect trace before stopping");
    case State::kStartedError:
      LOG(ERROR) << "Cannot collect, roctracer failed to start";
      return absl::OkStatus();
    case State::kStoppedError:
      VLOG(3) << "No trace data collected";
      return absl::OkStatus();
    case State::kStoppedOk: {
      AppendOptionDiagnostics(option_diagnostics_, space);
      if (rocm_trace_collector_) {
        rocm_trace_collector_->SetScopeRangeIdTree(
            rocm_tracer_->annotation_map()->TakeScopeRangeIdTree());
        rocm_trace_collector_->Export(space);
      }
      return absl::OkStatus();
    }
  }
  return absl::InternalError(
      absl::StrCat("Invalid profiling state: ", profiling_state_));
}

// Not in anonymous namespace for testing purposes.
std::unique_ptr<profiler::ProfilerInterface> CreateGpuTracer(
    const ProfileOptions& options) {
  if (options.device_type() != ProfileOptions::GPU &&
      options.device_type() != ProfileOptions::UNSPECIFIED)
    return nullptr;
  auto& rocm_tracer = profiler::RocmTracer::GetRocmTracerSingleton();
  if (!rocm_tracer.IsAvailable()) return nullptr;
  return std::make_unique<profiler::GpuTracer>(&rocm_tracer, options);
}

auto register_rocm_gpu_tracer_factory = [] {
  RegisterProfilerFactory(&CreateGpuTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace xla
