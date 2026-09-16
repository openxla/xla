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

  // Seeds both option structs from the hardcoded defaults and the
  // --xla_gpu_rocm_max_trace_events flag, then applies
  // ProfileOptions.advanced_configuration on top. Records any complaint in
  // option_diagnostics_ instead of failing.
  void BuildOptions(uint32_t num_gpus, RocmTracerOptions& tracer_options,
                    RocmTraceCollectorOptions& collector_options);

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
  // Whether option_diagnostics_ has already been written into an XSpace.
  bool diagnostics_delivered_ = false;
  RocmTracer* rocm_tracer_;
  std::unique_ptr<RocmTraceCollector> rocm_trace_collector_;
};

void GpuTracer::BuildOptions(uint32_t num_gpus,
                             RocmTracerOptions& tracer_options,
                             RocmTraceCollectorOptions& collector_options) {
  // Rebuilt from scratch on every start, so that a second Start() on the same
  // tracer does not report the first session's complaints a second time.
  option_diagnostics_ = RocmTracerOptionDiagnostics{};
  diagnostics_delivered_ = false;

  // Layer 1: hardcoded defaults, unchanged from what this file used before
  // advanced_configuration existed.
  collector_options.num_gpus = num_gpus;
  // The AnnotationMap size is deliberately NOT seeded from the flag below.
  // GetRocmTracerOptions() hardcoded 4M here and the flag never reached it, so
  // routing the flag to it now would silently shrink the annotation cache for
  // everyone who already lowers --xla_gpu_rocm_max_trace_events to bound trace
  // memory -- and a full AnnotationMap costs op names on every later kernel,
  // with no diagnostic. Only gpu_max_annotation_strings overrides this.
  tracer_options.max_annotation_strings = kDefaultMaxAnnotationStrings;

  // Layer 2: the legacy flag. Retained as a fallback for users who already
  // depend on it; each advanced_configuration key overrides its own field.
  const auto& dbg = xla::GetDebugOptionsFromFlags();
  int64_t max_events = dbg.xla_gpu_rocm_max_trace_events();
  VLOG(2) << "max number of events to be trace from flag = " << max_events;
  if (max_events <= 0) {
    max_events = 4 * 1024 * 1024;
  }
  if (max_events > 1'000'000'000LL) {
    max_events = 1'000'000'000LL;
  }
  VLOG(3) << "maximum number of events to be traced = " << max_events;

  collector_options.max_callback_api_events = max_events;
  collector_options.max_activity_api_events = max_events;
  collector_options.max_annotation_strings = max_events;

  // Layer 3: ProfileOptions.advanced_configuration.
  if (profile_options_.version() == 0 &&
      !profile_options_.advanced_configuration().empty()) {
    // Unreachable in practice: ProfilerSession::GetOptions drops the map
    // before we ever see it when version() is zero. Logged in case a caller
    // reaches the tracer factory directly with a hand-built proto.
    VLOG(1) << "ProfileOptions.version() is 0; advanced_configuration is "
               "normally stripped by ProfilerSession for such options.";
  }
  UpdateRocmTracerOptionsFromProfilerOptions(
      profile_options_, tracer_options, collector_options, option_diagnostics_);

  // Post-parse fixup for num_gpus, mirroring device_tracer_cuda.cc. That file
  // spells the first test `<= 0`, which is tautological on an unsigned field
  // and warns under -Wtautological-unsigned-zero-compare; `== 0` is what it
  // means and what it compiles to. This is a reset to all, not a clamp: an
  // out-of-range ask profiles every GPU, not one.
  if (collector_options.num_gpus == 0 ||
      collector_options.num_gpus > num_gpus) {
    if (collector_options.num_gpus != 0) {
      LOG(WARNING) << "The provided number of GPUs ("
                   << collector_options.num_gpus
                   << ") is invalid. Profiling will be done on all available "
                      "GPUs ("
                   << num_gpus << ").";
      option_diagnostics_.warnings.push_back(absl::StrCat(
          "advanced_configuration key 'gpu_num_chips_to_profile_per_task'=",
          collector_options.num_gpus, " is out of range; profiling all ",
          num_gpus, " GPUs."));
    }
    collector_options.num_gpus = num_gpus;
  }
}

absl::Status GpuTracer::DoStart() {
  AnnotationStack::Enable(true);
  uint64_t start_gputime_ns = RocmTracer::GetTimestamp();
  uint64_t start_walltime_ns = tsl::EnvTime::NowNanos();

  RocmTracerOptions tracer_options;
  RocmTraceCollectorOptions trace_collector_options;
  BuildOptions(rocm_tracer_->NumGpus(), tracer_options,
               trace_collector_options);

  // Deliberately different from the CUPTI path. A bad key does not abort the
  // session, because on the default JAX path that costs the user the entire
  // GPU plane with nothing in the trace to explain it. The diagnostics are
  // carried to CollectData instead. Callers that explicitly asked for start
  // failures to be fatal still get that.
  if (!option_diagnostics_.errors.empty() &&
      profile_options_.raise_error_on_start_failure()) {
    AnnotationStack::Enable(false);
    // Only the errors can travel in the Status, and CollectData will never be
    // reached on this path, so the warnings would otherwise be lost outright.
    // Log them here rather than discard them.
    for (const std::string& warning : option_diagnostics_.warnings) {
      LOG(WARNING) << warning;
    }
    return absl::InvalidArgumentError(
        absl::StrJoin(option_diagnostics_.errors, " "));
  }

  rocm_trace_collector_ = CreateRocmCollector(
      trace_collector_options, start_walltime_ns, start_gputime_ns);
  rocm_trace_collector_->SetGpuAgents(rocm_tracer_->GpuAgents());

  absl::Status status =
      rocm_tracer_->Enable(tracer_options, rocm_trace_collector_.get());
  if (!status.ok()) {
    AnnotationStack::Enable(false);
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
  // Delivered before the state switch, so the stopped-with-errors states are
  // covered as well as the happy path -- CUPTI's add_errors/add_warnings calls
  // sit under kStoppedOk alone.
  //
  // Be precise about why that matters, because the obvious stronger claim is
  // false: ProfilerController::CollectData only forwards to us when the status
  // from Start() was ok (profiler_controller.cc), so in the kStartedError case
  // this function is never called at all and the branch below is reachable
  // only by a direct ProfilerInterface user such as device_tracer_rocm_test.
  // What actually protects the user is upstream of here -- a bad key does not
  // fail Start() on this backend, so the session reaches CollectData normally
  // and the diagnostics land. The one path that still loses them is a caller
  // that sets raise_error_on_start_failure: its errors surface in the returned
  // Status, but the warnings do not survive.
  //
  // kStartedOk is not terminal: the caller is collecting before Stop(), gets an
  // error below, and will call again once stopped. And CollectData may legally
  // be called more than once from a terminal state. Latch on delivery so that
  // neither case puts the same message into the XSpace twice; BuildOptions
  // clears the latch along with the diagnostics on the next Start().
  if (profiling_state_ != State::kStartedOk && !diagnostics_delivered_) {
    diagnostics_delivered_ = true;
    AppendOptionDiagnostics(option_diagnostics_, space);
  }
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
