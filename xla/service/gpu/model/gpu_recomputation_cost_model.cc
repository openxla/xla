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

#include "xla/service/gpu/model/gpu_recomputation_cost_model.h"

#include <algorithm>
#include <cmath>

#include "absl/status/status_macros.h"
#include "absl/time/time.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/combined_gpu_performance_model.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

absl::StatusOr<RecomputationEstimate> EstimateRecomputationRegion(
    HloModule& module, const se::DeviceDescription& device) {
  GpuHloCostAnalysis analysis({.shape_size =
                                   [](const Shape& shape) {
                                     return ShapeUtil::ByteSizeOf(
                                         shape, /*pointer_size=*/8);
                                   },
                               .count_multiple_input_accesses = true},
                              device);
  ABSL_RETURN_IF_ERROR(module.entry_computation()->Accept(&analysis));
  // Fresh caches: before/after graphs must never share instruction estimates.
  mlir::MLIRContext context;
  context.disableMultithreading();
  RegisterSymbolicExprStorage(&context);
  HloFusionAnalysisCache fusion_cache(device);
  const auto& debug = module.config().debug_options();
  CombinedGpuPerformanceModel model(
      device, fusion_cache, context,
      [](const Shape& shape) { return ShapeUtil::ByteSizeOf(shape, 8); },
      debug.xla_gpu_experimental_enable_tiling_propagation(),
      debug.xla_gpu_experimental_enable_same_shape_multi_output_fusion());
  RecomputationEstimate total;
  for (HloInstruction* instruction :
       module.entry_computation()->instructions()) {
    if (instruction->opcode() != HloOpcode::kFusion) continue;
    ABSL_ASSIGN_OR_RETURN(auto estimate, model.EstimateRunTimeForInstruction(
                                             instruction, &analysis));
    total.duration_ns += absl::ToDoubleNanoseconds(
        estimate.exec_time + GpuPerformanceModelBase::kKernelLaunchOverhead);
    total.compute_ns += absl::ToDoubleNanoseconds(estimate.compute_time);
    total.read_ns += absl::ToDoubleNanoseconds(estimate.read_time);
    total.write_ns += absl::ToDoubleNanoseconds(estimate.write_time);
    total.bytes_read += estimate.bytes_read;
    total.bytes_written += estimate.bytes_written;
    ++total.kernels;
  }
  return total;
}

RecomputationMeasurement EvaluateRecomputationMeasurements(
    absl::Span<const RecomputationSample> samples) {
  RecomputationMeasurement result;
  if (samples.size() < 25) {
    result.decision = Decision::Forbid("fewer than 25 ABBA blocks");
    return result;
  }
  double first_gain = 0;
  double second_gain = 0;
  double squared_deviations = 0;
  int64_t count = 0;
  for (const auto& sample : samples) {
    for (double ns : {sample.before_first_ns, sample.after_first_ns,
                      sample.after_second_ns, sample.before_second_ns}) {
      if (!std::isfinite(ns) || ns <= 0) {
        result.decision = Decision::Forbid("invalid profiling duration");
        return result;
      }
    }
    const double before =
        (sample.before_first_ns + sample.before_second_ns) / 2;
    const double after = (sample.after_first_ns + sample.after_second_ns) / 2;
    const double gain = before - after;
    ++count;
    result.before_ns += (before - result.before_ns) / count;
    result.after_ns += (after - result.after_ns) / count;
    const double delta = gain - result.gain_ns;
    result.gain_ns += delta / count;
    squared_deviations += delta * (gain - result.gain_ns);
    first_gain +=
        (sample.before_first_ns - sample.after_first_ns - first_gain) / count;
    second_gain +=
        (sample.before_second_ns - sample.after_second_ns - second_gain) /
        count;
  }
  result.uncertainty_ns =
      3 * std::sqrt(std::max(0.0, squared_deviations) / (count * (count - 1)));
  result.required_gain_ns = std::max(1000.0, 0.01 * result.before_ns);
  if (std::min(first_gain, second_gain) <= result.required_gain_ns ||
      result.gain_ns - result.uncertainty_ns <= result.required_gain_ns) {
    result.decision =
        Decision::Forbid("gain does not exceed noise/order/policy margin");
  } else {
    result.decision = Decision::Allow();
  }
  return result;
}
}  // namespace xla::gpu
