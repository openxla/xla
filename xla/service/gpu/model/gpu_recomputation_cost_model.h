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

#ifndef XLA_SERVICE_GPU_MODEL_GPU_RECOMPUTATION_COST_MODEL_H_
#define XLA_SERVICE_GPU_MODEL_GPU_RECOMPUTATION_COST_MODEL_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/decision.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Advisory estimates for a dependency-preserving region, NOT for merging its
// kernels. Unknown instruction profiles, occupancy and inter-kernel cache state
// make these uncalibrated; they must never independently authorize a rewrite.
struct RecomputationEstimate {
  double duration_ns = 0;
  double compute_ns = 0;
  double read_ns = 0;
  double write_ns = 0;
  int64_t bytes_read = 0;
  int64_t bytes_written = 0;
  int64_t kernels = 0;
};
absl::StatusOr<RecomputationEstimate> EstimateRecomputationRegion(
    HloModule& module, const se::DeviceDescription& device);

// One ABBA block: A=materialize, B=recompute. Opposite orders mitigate linear
// drift. All four samples are warm, whole-region GPU execution times.
struct RecomputationSample {
  double before_first_ns;
  double after_first_ns;
  double after_second_ns;
  double before_second_ns;
};
struct RecomputationMeasurement {
  double before_ns = 0;
  double after_ns = 0;
  double gain_ns = 0;
  double uncertainty_ns = 0;
  double required_gain_ns = 0;
  Decision decision = Decision::Forbid("not measured");
};

// Require >=25 ABBA blocks, a gain exceeding both 1% and 1us,
// a three-standard-error margin, and agreement between the two orders. This is
// a conservative tuning policy, not a calibrated statistical guarantee or an
// end-to-end latency estimate. Noise, invalid samples and small gains reject.
RecomputationMeasurement EvaluateRecomputationMeasurements(
    absl::Span<const RecomputationSample> samples);

}  // namespace xla::gpu
#endif  // XLA_SERVICE_GPU_MODEL_GPU_RECOMPUTATION_COST_MODEL_H_
