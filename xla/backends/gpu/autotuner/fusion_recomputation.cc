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

#include "xla/backends/gpu/autotuner/fusion_recomputation.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "xla/backends/gpu/autotuner/gpu_profiler.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/decision.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/model/gpu_recomputation_cost_model.h"
#include "xla/service/gpu_topology.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

absl::StatusOr<Decision> Evaluate(const FusionRecomputeVariants& variants,
                                  Compiler* compiler,
                                  const Compiler::GpuTargetConfig& target,
                                  se::StreamExecutor* executor,
                                  se::DeviceAddressAllocator* allocator) {
  if (executor == nullptr)
    return Decision::Forbid("requires a profiling device");
  auto compile = [&](const HloModule& source) {
    auto module = source.Clone();
    auto& debug = module->mutable_config().mutable_debug_options();
    GpuCodegenBackend::AdjustDebugOptionsForAutotuning(debug);
    debug.set_xla_gpu_experimental_fusion_recomputation(0);
    Compiler::CompileOptions options;
    options.device_allocator = allocator;
    options.gpu_topology = GetSingleDeviceGpuTopology("", target);
    // These modules already have the final fusion boundaries. Do NOT run the
    // fusion pipeline again or measure a fused-vs-unfused surrogate.
    return compiler->RunBackend(std::move(module), executor, options);
  };
  ABSL_ASSIGN_OR_RETURN(auto before, compile(*variants.before));
  ABSL_ASSIGN_OR_RETURN(auto after, compile(*variants.after));
  auto temp_bytes = [](Executable* executable) -> absl::StatusOr<int64_t> {
    const auto* gpu = dynamic_cast<GpuExecutable*>(executable);
    if (!gpu) return absl::InternalError("expected a GPU executable");
    int64_t bytes = 0;
    for (const BufferAllocation& allocation : gpu->allocations()) {
      if (allocation.IsPreallocatedTempBuffer()) bytes += allocation.size();
    }
    return bytes;
  };
  ABSL_ASSIGN_OR_RETURN(int64_t before_temp, temp_bytes(before.get()));
  ABSL_ASSIGN_OR_RETURN(int64_t after_temp, temp_bytes(after.get()));
  if (after_temp > before_temp) {
    return Decision::Forbid("compiled region temporary allocation increases");
  }
  auto profiler = GpuProfiler::Create(
      executor, {/*redzone_padding_bytes=*/256, /*should_init_buffers=*/true},
      allocator);
  if (!profiler) return absl::InternalError("could not create GPU profiler");
  ABSL_ASSIGN_OR_RETURN(auto inputs,
                        profiler->CreateInputBuffers(before.get()));
  // Verify all externally observable region outputs, not just the gradient.
  {
    ABSL_ASSIGN_OR_RETURN(auto reference,
                          profiler->Profile(before.get(), *inputs));
    ABSL_ASSIGN_OR_RETURN(auto output, profiler->Profile(after.get(), *inputs));
    if (!reference.output_buffer || !output.output_buffer) {
      return absl::InternalError("missing profiling output");
    }
    ABSL_RETURN_IF_ERROR(profiler->CheckOutputBuffer(
        *output.output_buffer, *reference.output_buffer, /*rtol=*/1e-5));
    ABSL_RETURN_IF_ERROR(profiler->CheckInputBuffers(*inputs));
  }
  std::vector<RecomputationSample> samples;
  std::string samples_csv =
      "block,before_first_ns,after_first_ns,after_second_ns,before_second_ns\n";
  for (int block = 0; block < 25; ++block) {
    std::array<double, 4> ns;
    for (int i = 0; i < 4; ++i) {
      ABSL_ASSIGN_OR_RETURN(
          auto profile,
          profiler->Profile((i == 0 || i == 3) ? before.get() : after.get(),
                            *inputs));
      ns[i] = absl::ToDoubleNanoseconds(profile.duration);
    }
    samples.push_back({ns[0], ns[1], ns[2], ns[3]});
    absl::StrAppendFormat(&samples_csv, "%d,%.0f,%.0f,%.0f,%.0f\n", block,
                          ns[0], ns[1], ns[2], ns[3]);
  }
  ABSL_RETURN_IF_ERROR(profiler->CheckInputBuffers(*inputs));
  const auto measurement = EvaluateRecomputationMeasurements(samples);
  std::string report;
  absl::StrAppendFormat(
      &report,
      "eliminated_bytes=%d\nmeasured_before_ns=%.0f\nmeasured_after_ns=%.0f\n"
      "gain_ns=%.0f\nuncertainty_ns=%.0f\nrequired_gain_ns=%.0f\naccepted=%d\n",
      variants.eliminated_bytes, measurement.before_ns, measurement.after_ns,
      measurement.gain_ns, measurement.uncertainty_ns,
      measurement.required_gain_ns, measurement.decision.IsAllowed());
  absl::StrAppendFormat(
      &report, "compiled_before_temp_bytes=%d\ncompiled_after_temp_bytes=%d\n",
      before_temp, after_temp);
  // The analytical model is diagnostic until its errors have been calibrated
  // on held-out kernels/devices. It includes dtype/op profiles, coalescing,
  // indexing multiplicity and compute-memory overlap from the existing model;
  // timings above additionally observe codegen, registers, spills and caches.
  for (auto [label, module] : {std::pair{"before", variants.before.get()},
                               std::pair{"after", variants.after.get()}}) {
    auto estimate =
        EstimateRecomputationRegion(*module, target.device_description);
    if (estimate.ok()) {
      absl::StrAppendFormat(
          &report,
          "uncalibrated_%s: total_ns=%.0f compute_ns=%.0f read_ns=%.0f "
          "write_ns=%.0f bytes_read=%d bytes_written=%d kernels=%d\n",
          label, estimate->duration_ns, estimate->compute_ns, estimate->read_ns,
          estimate->write_ns, estimate->bytes_read, estimate->bytes_written,
          estimate->kernels);
    } else {
      absl::StrAppend(&report, "uncalibrated_", label, ": ",
                      estimate.status().ToString(), "\n");
    }
  }
  absl::StrAppend(&report, "device=", target.device_description.ToString(),
                  "\n");
  VLOG(1) << "Fusion recomputation: " << report;
  DumpToFileInDir(*variants.before, "fusion-recomputation", "csv", samples_csv);
  DumpToFileInDir(*variants.before, "fusion-recomputation", "txt", report);
  DumpToFileInDir(*variants.before, "fusion-recomputation", "before.hlo",
                  variants.before->ToString());
  DumpToFileInDir(*variants.before, "fusion-recomputation", "after.hlo",
                  variants.after->ToString());
  if (before->has_module() && after->has_module()) {
    DumpToFileInDir(*variants.before, "fusion-recomputation",
                    "compiled-before.hlo", before->module().ToString());
    DumpToFileInDir(*variants.before, "fusion-recomputation",
                    "compiled-after.hlo", after->module().ToString());
  }
  return measurement.decision;
}
}  // namespace

RecomputeFusionSideOutputs::Evaluator MakeFusionRecomputationEvaluator(
    Compiler* compiler, const Compiler::GpuTargetConfig& target,
    se::StreamExecutor* executor, se::DeviceAddressAllocator* allocator) {
  // Canonical graphs include constants, layouts and backend configs. The cache
  // lifetime fixes compiler version, device, flags and profiling policy. Unlike
  // a process-global cache it cannot reuse a result on a different GPU/context.
  auto cache = std::make_shared<absl::flat_hash_map<std::string, Decision>>();
  return [compiler, &target, executor, allocator,
          cache](const FusionRecomputeVariants& v) -> absl::StatusOr<Decision> {
    auto print = HloPrintOptions::Canonical()
                     .set_print_backend_config(true)
                     .set_print_large_constants(true);
    const std::string key = absl::StrCat(
        v.before->config().debug_options().SerializeAsString(), "\nBEFORE\n",
        v.before->ToString(print), "\nAFTER\n", v.after->ToString(print));
    auto found = cache->find(key);
    if (found != cache->end()) return found->second;
    ABSL_ASSIGN_OR_RETURN(auto decision,
                          Evaluate(v, compiler, target, executor, allocator));
    cache->emplace(key, decision);
    return decision;
  };
}
}  // namespace xla::gpu
