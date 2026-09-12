/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_OPTIONS_UTILS_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_OPTIONS_UTILS_H_

#include <string>
#include <vector>

#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

// Diagnostics produced while interpreting
// ProfileOptions.advanced_configuration. Recorded when the tracer starts and
// delivered when it collects, so that a mistyped or unsupported key reaches
// the XSpace the user opens rather than only the server's stderr.
struct RocmTracerOptionDiagnostics {
  // Keys that were not recognised, or whose value had the wrong oneof type.
  std::vector<std::string> errors;
  // Keys that are recognised but have no effect on the ROCm backend, plus
  // caveats attached to keys that do.
  std::vector<std::string> warnings;

  bool empty() const { return errors.empty() && warnings.empty(); }
};

// Applies the gpu_* entries of profile_options.advanced_configuration() to the
// two ROCm option structs. Both structs must already hold the caller's
// defaults; only keys that are actually present are touched.
//
// This function never fails. Unlike the CUPTI equivalent
// (UpdateCuptiTracerOptionsFromProfilerOptions), an unrecognised key neither
// aborts parsing nor prevents the tracer from starting: it is recorded in
// diagnostics.errors, and the remaining keys are still applied.
//
// This function never fails and has no opinion about what a non-empty error
// list means; that is the caller's decision. A caller that wants CUPTI's
// hard-failure behaviour can inspect diagnostics.errors and honour
// ProfileOptions.raise_error_on_start_failure itself. `diagnostics` is
// appended to, never cleared, so a caller reusing one across sessions must
// reset it.
//
// The reason for the difference is that on the default JAX path a parse
// failure is not visible to the user at all: it sets kStartedError, and that
// branch of CollectData logs and returns OkStatus without ever reaching
// XSpace.errors, so one typo silently costs the whole GPU plane. Reporting
// into the XSpace and still producing a trace strictly dominates that.
void UpdateRocmTracerOptionsFromProfilerOptions(
    const tensorflow::ProfileOptions& profile_options,
    RocmTracerOptions& tracer_options,
    RocmTraceCollectorOptions& collector_options,
    RocmTracerOptionDiagnostics& diagnostics);

// Appends diagnostics to space->errors() and space->warnings(), and mirrors
// them to LOG. Safe to call with empty diagnostics or a null space.
void AppendOptionDiagnostics(const RocmTracerOptionDiagnostics& diagnostics,
                             tensorflow::profiler::XSpace* space);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_OPTIONS_UTILS_H_
