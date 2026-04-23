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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_GPU_FUSION_COST_MODEL_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_GPU_FUSION_COST_MODEL_H_

#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/fusion_cost_model.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

class GpuFusionCostModel : public FusionCostModel {
 public:
  GpuFusionCostModel(const se::DeviceDescription& device_info,
                     GpuHloCostAnalysis::Options cost_analysis_options,
                     mlir::MLIRContext* mlir_context);

  absl::StatusOr<RunTimes> EstimateRunTimes(
      const HloInstruction* producer,
      absl::Span<const HloInstruction* const> consumers) override {
    return EstimateRunTimes(producer, consumers,
                            /*use_multi_output_fusion=*/false);
  }

  absl::StatusOr<RunTimes> EstimateRunTimes(
      const HloInstruction* producer,
      absl::Span<const HloInstruction* const> consumers,
      bool use_multi_output_fusion);

  bool WouldExplodeIrSize(const HloInstruction* producer,
                          const HloInstruction* consumer) override;
  void OnInstructionFused(HloInstruction* producer, HloInstruction* consumer,
                          HloInstruction* fusion) override;
  void Invalidate(const HloInstruction* instruction) override;
  void ClearCaches() override;

  absl::Status Revisit(const HloInstruction* instruction) override;

  const se::DeviceDescription& device_info() const { return device_info_; }
  HloFusionAnalysisCache& fusion_analysis_cache() {
    return *fusion_analysis_cache_;
  }
  GpuHloCostAnalysis& cost_analysis() { return *cost_analysis_; }
  GpuPerformanceModelWithIndexingAnalysis* gpu_indexing_performance_model()
      const {
    return gpu_indexing_performance_model_.get();
  }
  FusionInfoCache& fusion_info_cache() { return fusion_info_cache_; }

  std::optional<BlockLevelParameters> GetTritonBlockLevelParameters(
      const HloInstruction& producer, const HloInstruction& consumer,
      bool use_multi_output_fusion) const;

  absl::Status Prepare(const HloComputation* computation) override;
  absl::Mutex& mutex() const { return caches_mutex_; }

  void UpdatePerformanceModelCache(const HloInstruction* instruction);
  absl::Duration EstimateRunTimeForInstruction(
      const HloInstruction* instruction);

  absl::StatusOr<RunTimes> EstimateTritonRunTimes(
      const HloInstruction* producer, const HloInstruction* consumer,
      bool use_multi_output_fusion);

 private:
  void UpdatePerformanceModelCacheLocked(const HloInstruction* instruction)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(caches_mutex_);
  se::DeviceDescription device_info_;
  GpuHloCostAnalysis::Options cost_analysis_options_;
  std::unique_ptr<GpuHloCostAnalysis> cost_analysis_;
  HloCostAnalysis::ShapeSizeFunction shape_size_;
  mutable absl::Mutex triton_params_cache_mutex_;
  absl::flat_hash_map<
      std::tuple<const HloInstruction*, const HloInstruction*, bool>,
      BlockLevelParameters>
      triton_params_cache_ ABSL_GUARDED_BY(triton_params_cache_mutex_);
  mlir::MLIRContext* mlir_context_;
  FusionInfoCache fusion_info_cache_;

  std::unique_ptr<GpuPerformanceModelCache> gpu_performance_model_cache_;
  std::unique_ptr<HloFusionAnalysisCache> fusion_analysis_cache_;
  std::unique_ptr<GpuPerformanceModel> gpu_performance_model_;
  std::unique_ptr<GpuPerformanceModelWithIndexingAnalysis>
      gpu_indexing_performance_model_;

  absl::flat_hash_map<
      std::tuple<const HloInstruction*, const HloInstruction*, bool>, RunTimes>
      triton_runtimes_cache_ ABSL_GUARDED_BY(caches_mutex_);
  mutable absl::Mutex caches_mutex_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_GPU_FUSION_COST_MODEL_H_
