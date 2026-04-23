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

#include "xla/backends/gpu/transforms/gpu_fusion_cost_model.h"

#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/transforms/priority_fusion.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/fusion_cost_model.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

GpuFusionCostModel::GpuFusionCostModel(
    const se::DeviceDescription& device_info,
    GpuHloCostAnalysis::Options cost_analysis_options,
    mlir::MLIRContext* mlir_context)
    : device_info_(device_info),
      cost_analysis_options_(cost_analysis_options),
      cost_analysis_(std::make_unique<GpuHloCostAnalysis>(cost_analysis_options,
                                                          device_info)),
      shape_size_(cost_analysis_options.shape_size),
      mlir_context_(mlir_context),
      fusion_info_cache_(device_info),
      gpu_performance_model_cache_(
          std::make_unique<GpuPerformanceModelCache>()),
      fusion_analysis_cache_(
          std::make_unique<HloFusionAnalysisCache>(device_info)),
      gpu_performance_model_(std::make_unique<GpuPerformanceModel>(
          device_info, *fusion_analysis_cache_, *gpu_performance_model_cache_,
          mlir_context)),
      gpu_indexing_performance_model_(
          std::make_unique<GpuPerformanceModelWithIndexingAnalysis>(
              &device_info, fusion_analysis_cache_.get(), shape_size_,
              mlir_context)) {}

void GpuFusionCostModel::UpdatePerformanceModelCache(
    const HloInstruction* instruction) {
  absl::MutexLock lock(&caches_mutex_);
  UpdatePerformanceModelCacheLocked(instruction);
}

absl::Duration GpuFusionCostModel::EstimateRunTimeForInstruction(
    const HloInstruction* instruction) {
  absl::MutexLock lock(&caches_mutex_);
  UpdatePerformanceModelCacheLocked(instruction);
  auto runtime_data = gpu_performance_model_cache_->Get(*instruction);
  return runtime_data ? runtime_data->exec_time : absl::ZeroDuration();
}

void GpuFusionCostModel::UpdatePerformanceModelCacheLocked(
    const HloInstruction* instruction) {
  if (gpu_performance_model_cache_->Get(*instruction)) {
    return;
  }

  // Handle non-fusible instructions by setting their runtime to zero.
  // This ensures that they are present in the cache for GpuPerformanceModel.
  if (instruction->opcode() == HloOpcode::kParameter ||
      instruction->shape().IsToken() ||
      (instruction->shape().IsTuple() &&
       instruction->opcode() != HloOpcode::kFusion) ||
      instruction->opcode() == HloOpcode::kGetTupleElement ||
      instruction->opcode() == HloOpcode::kConstant ||
      IsFusibleBitcast(*instruction)) {
    gpu_performance_model_cache_->Set(*instruction,
                                      EstimateRunTimeData::Zero());
    return;
  }

  if (IsGenericTritonFusion(*instruction)) {
    auto runtime_data_or =
        gpu_indexing_performance_model_->EstimateRunTimeForTriton(instruction);
    if (runtime_data_or.ok()) {
      gpu_performance_model_cache_->Set(*instruction, *runtime_data_or);
      return;
    }
  }
  auto runtime_data = gpu_performance_model_->EstimateRunTimeForInstruction(
      instruction, cost_analysis_.get());
  gpu_performance_model_cache_->Set(*instruction, runtime_data);
}

absl::Status GpuFusionCostModel::Prepare(const HloComputation* computation) {
  gpu_performance_model_cache_ = std::make_unique<GpuPerformanceModelCache>();
  fusion_analysis_cache_ =
      std::make_unique<HloFusionAnalysisCache>(device_info_);
  gpu_performance_model_ = std::make_unique<GpuPerformanceModel>(
      device_info_, *fusion_analysis_cache_, *gpu_performance_model_cache_,
      mlir_context_);
  gpu_indexing_performance_model_ =
      std::make_unique<GpuPerformanceModelWithIndexingAnalysis>(
          &device_info_, fusion_analysis_cache_.get(), shape_size_,
          mlir_context_);

  cost_analysis_ = std::make_unique<GpuHloCostAnalysis>(cost_analysis_options_,
                                                        device_info_);
  auto status = computation->Accept(cost_analysis_.get());
  if (!status.ok()) {
    return status;
  }

  for (auto* instruction : computation->instructions()) {
    if (instruction->opcode() == HloOpcode::kConstant) {
      continue;
    }

    UpdatePerformanceModelCache(instruction);
  }

  return absl::OkStatus();
}

absl::Status GpuFusionCostModel::Revisit(const HloInstruction* instruction) {
  auto status = cost_analysis_->RevisitInstruction(instruction);
  if (!status.ok()) {
    return status;
  }

  UpdatePerformanceModelCache(instruction);

  return absl::OkStatus();
}

absl::StatusOr<FusionCostModel::RunTimes> GpuFusionCostModel::EstimateRunTimes(
    const HloInstruction* producer,
    absl::Span<const HloInstruction* const> consumers,
    bool use_multi_output_fusion) {
  absl::MutexLock lock(&caches_mutex_);
  UpdatePerformanceModelCacheLocked(producer);
  for (auto consumer : consumers) {
    UpdatePerformanceModelCacheLocked(consumer);
  }

  if (consumers.size() == 1 && !use_multi_output_fusion &&
      gpu_indexing_performance_model_) {
    const HloInstruction* consumer = consumers[0];
    auto it = triton_runtimes_cache_.find(
        {producer, consumer, use_multi_output_fusion});
    if (it != triton_runtimes_cache_.end()) {
      return it->second;
    }
  }

  auto res = gpu_performance_model_->EstimateRunTimes(
      producer, cost_analysis_.get(), consumers);
  return FusionCostModel::RunTimes{res.time_unfused, res.time_fused};
}

void GpuFusionCostModel::Invalidate(const HloInstruction* instruction) {
  absl::MutexLock lock(&caches_mutex_);
  (void)cost_analysis_->RemoveInstruction(instruction);
  gpu_performance_model_cache_->Invalidate(*instruction);

  // NOLINTNEXTLINE(custom-deterministic-iteration-order)
  for (auto it = triton_runtimes_cache_.begin();
       it != triton_runtimes_cache_.end();) {
    if (std::get<0>(it->first) == instruction ||
        std::get<1>(it->first) == instruction) {
      triton_runtimes_cache_.erase(it++);
    } else {
      ++it;
    }
  }
  fusion_info_cache_.Invalidate(instruction);
  if (instruction->parent() != nullptr) {
    fusion_analysis_cache_->Invalidate(*instruction);
  }
}

void GpuFusionCostModel::ClearCaches() {
  absl::MutexLock lock(&caches_mutex_);
  fusion_analysis_cache_->Clear();
  gpu_performance_model_cache_->Clear();
  triton_runtimes_cache_.clear();
}

bool GpuFusionCostModel::WouldExplodeIrSize(const HloInstruction* producer,
                                            const HloInstruction* consumer) {
  return cost_analysis_->ProducerConsumerMergedTooLarge(*producer, *consumer);
}

std::optional<BlockLevelParameters>
GpuFusionCostModel::GetTritonBlockLevelParameters(
    const HloInstruction& producer, const HloInstruction& consumer,
    bool use_multi_output_fusion) const {
  absl::MutexLock lock(triton_params_cache_mutex_);
  auto it = triton_params_cache_.find(
      {&producer, &consumer, use_multi_output_fusion});
  if (it != triton_params_cache_.end()) {
    return it->second;
  }
  return std::nullopt;
}

void GpuFusionCostModel::OnInstructionFused(HloInstruction* producer,
                                            HloInstruction* consumer,
                                            HloInstruction* fusion) {
  absl::MutexLock lock(&caches_mutex_);
  (void)cost_analysis_->RemoveInstruction(producer);
  (void)cost_analysis_->RemoveInstruction(consumer);
  (void)fusion->Accept(cost_analysis_.get());
}

absl::StatusOr<FusionCostModel::RunTimes>
GpuFusionCostModel::EstimateTritonRunTimes(const HloInstruction* producer,
                                           const HloInstruction* consumer,
                                           bool use_multi_output_fusion) {
  absl::MutexLock lock(&caches_mutex_);
  if (!gpu_indexing_performance_model_) {
    return absl::InternalError(
        "GPU indexing performance model is not available.");
  }

  auto fusion = HloFusionAdaptor::ForProducerConsumer(producer, consumer,
                                                      use_multi_output_fusion);
  absl::StatusOr<TiledRunTimeDataOrError> result_or_status =
      gpu_indexing_performance_model_->TryFindBestTilingForFusion(*fusion);

  if (!result_or_status.ok()) {
    return result_or_status.status();
  }

  if (const auto* fusion_decision =
          std::get_if<FusionDecision>(&*result_or_status)) {
    return absl::InternalError(
        absl::StrCat("Triton tiling failed: ", fusion_decision->Explain()));
  }

  const auto& tiled_data = std::get<gpu::TiledRunTimeData>(*result_or_status);

  auto p_data = gpu_performance_model_cache_->Get(*producer);
  auto producer_runtime = p_data ? p_data->exec_time : absl::ZeroDuration();
  auto c_data = gpu_performance_model_cache_->Get(*consumer);
  auto consumer_runtime = c_data ? c_data->exec_time : absl::ZeroDuration();

  RunTimes run_times = {producer_runtime + consumer_runtime +
                            2 * GpuPerformanceModelBase::kKernelLaunchOverhead,
                        tiled_data.runtime_data.exec_time};

  triton_runtimes_cache_[{producer, consumer, use_multi_output_fusion}] =
      run_times;
  gpu_performance_model_cache_->Set(*producer, *consumer, run_times.fused);

  {
    absl::MutexLock triton_lock(&triton_params_cache_mutex_);
    triton_params_cache_[{producer, consumer, use_multi_output_fusion}] =
        tiled_data.block_level_parameters;
  }

  return run_times;
}

}  // namespace gpu
}  // namespace xla
