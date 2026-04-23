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

#include "xla/backends/gpu/transforms/gpu_priority_fusion.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/backends/gpu/transforms/gpu_fusion_cost_model.h"
#include "xla/backends/gpu/transforms/priority_fusion.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_dfs_reachability.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace gpu {

GpuPriorityFusion::GpuPriorityFusion(
    tsl::thread::ThreadPool* thread_pool, const se::DeviceDescription& device,
    const AliasInfo* alias_info,
    GpuHloCostAnalysis::Options cost_analysis_options,
    mlir::MLIRContext* mlir_context)
    : PriorityFusion(
          thread_pool, alias_info,
          std::make_unique<GpuFusionCostModel>(
              device, std::move(cost_analysis_options), mlir_context)),
      device_info_(device),
      fusion_info_cache_(device) {
  gpu_cost_model_ = static_cast<GpuFusionCostModel*>(cost_model());
}

FusionDecision GpuPriorityFusion::IsTritonSupported(
    const HloInstruction& instruction) {
  if (IsGenericTritonFusion(instruction)) {
    return FusionDecision::Allow();
  }

  if (!instruction.shape().IsArray()) {
    return FusionDecision::Forbid("not an array");
  }

  if (instruction.opcode() != HloOpcode::kFusion) {
    return IsTritonSupportedInstruction(instruction,
                                        device_info_.gpu_compute_capability());
  }

  for (const HloInstruction* instr :
       instruction.fused_instructions_computation()->instructions()) {
    if (auto codegen_decision = IsTritonSupportedInstruction(
            *instr, device_info_.gpu_compute_capability());
        !codegen_decision) {
      return codegen_decision;
    }
  }

  return FusionDecision::Allow();
}

FusionDecision GpuPriorityFusion::CanFuseTriton(const HloInstruction* producer,
                                                const HloInstruction* consumer,
                                                bool use_multi_output_fusion) {
  return [&]() -> FusionDecision {
    if (!IsFusible(*producer)) {
      return FusionDecision::Forbid("the producer is not fusible");
    }

    if (!IsFusible(*consumer)) {
      return FusionDecision::Forbid("the consumer is not fusible");
    }

    if (producer->GetModule() == nullptr) {
      return FusionDecision::Forbid("producer has no module");
    }

    bool triton_heroless_fusion_enabled =
        producer->GetModule()
            ->config()
            .debug_options()
            .xla_gpu_experimental_enable_triton_heroless_priority_fusion();
    if (!(IsGenericTritonFusion(*producer) ||
          IsGenericTritonFusion(*consumer) || triton_heroless_fusion_enabled)) {
      return FusionDecision::Forbid("triton heroless fusion is not enabled");
    }

    if (auto fusion_decision = IsTritonSupported(*producer); !fusion_decision) {
      return fusion_decision;
    }

    if (auto fusion_decision = IsTritonSupported(*consumer); !fusion_decision) {
      return fusion_decision;
    }

    if (auto fits_budget =
            FusionFitsInParameterLimit(*consumer, *producer,
                                       /*is_consumer_producer_fusion=*/true);
        !fits_budget) {
      return fits_budget;
    }

    auto* gpu_cost_model = static_cast<GpuFusionCostModel*>(cost_model());
    if (gpu_cost_model) {
      absl::StatusOr<FusionCostModel::RunTimes> result_or_status =
          gpu_cost_model->EstimateTritonRunTimes(producer, consumer,
                                                 use_multi_output_fusion);
      if (!result_or_status.ok()) {
        return FusionDecision::Forbid(
            absl::StrCat("EstimateTritonRunTimes returned status: ",
                         result_or_status.status().message()));
      }
    }

    return FusionDecision::Allow();
  }();
}

FusionDecision GpuPriorityFusion::BackendCanFuse(HloInstruction* producer,
                                                 HloInstruction* consumer) {
  bool use_multi_output_fusion =
      (consumer->opcode() == HloOpcode::kFusion &&
       consumer->fusion_kind() == HloInstruction::FusionKind::kInput);
  FusionDecision can_fuse_triton =
      CanFuseTriton(producer, consumer, use_multi_output_fusion);
  if (IsGenericTritonFusion(*producer) || IsGenericTritonFusion(*consumer)) {
    return can_fuse_triton;
  }
  if (!producer->shape().IsArray() || !consumer->shape().IsArray()) {
    return FusionDecision::Forbid("not an array");
  }
  if (can_fuse_triton) {
    return can_fuse_triton;
  }

  if (IsFusibleBitcast(*consumer)) {
    if (consumer->opcode() == HloOpcode::kBitcast &&
        producer->opcode() == HloOpcode::kBroadcast &&
        absl::c_all_of(consumer->users(), [](const HloInstruction* user) {
          return user->opcode() == HloOpcode::kTranspose;
        })) {
      return FusionDecision::Allow();
    }
    return FusionDecision::Forbid(
        "not fusing into a single bitcast as consumer");
  }

  // Scatter is special as it has no elemental version but is still input
  // fusible. Block attempts to create scatter fusions we can't codegen.
  if (auto can_fuse = CanEmitInputFusedScatter(*producer, *consumer);
      !can_fuse) {
    return can_fuse;
  }

  // Avoid fusing reduce into reduce. Our cost model doesn't currently
  // understand this case due to a lack of tiling analysis.
  auto contains_significant_reduce = [&](const HloInstruction* instr) {
    auto fusion = HloFusionAdaptor::ForInstruction(instr);
    bool result = HloAnyOf(*fusion, [](auto node) {
      if (node.opcode() != HloOpcode::kReduce || !node.shape().IsArray()) {
        return false;
      }

      int64_t reduction_size =
          ShapeUtil::ElementsIn(node.instruction().operand(0)->shape()) /
          ShapeUtil::ElementsIn(node.shape());

      // Small reductions are emitted using the elemental emitter anyway.
      return reduction_size >= 16;
    });
    return result;
  };
  if (contains_significant_reduce(producer) &&
      contains_significant_reduce(consumer)) {
    return FusionDecision::Forbid(
        "both the producer and the consumer contain a reduce");
  }

  if (producer->opcode() == HloOpcode::kTranspose &&
      contains_significant_reduce(consumer)) {
    return FusionDecision::Forbid(
        "Do not fuse transpose into a significant reduction");
  }

  // Avoid doing fusions into the output of an "input" fusion when it would
  // switch it to the loop emitter. This often occurs during epilog fusion for
  // reductions, which suffer from limited emitter support.
  const auto& producer_analysis =
      gpu_cost_model_->fusion_analysis_cache().Get(*producer);

  if (producer_analysis.emitter_fusion_kind() ==
      HloFusionAnalysis::EmitterFusionKind::kReduction) {
    const auto& analysis_fused =
        gpu_cost_model_->fusion_analysis_cache().Get(*producer, *consumer);

    if (analysis_fused.emitter_fusion_kind() ==
        HloFusionAnalysis::EmitterFusionKind::kLoop) {
      return FusionDecision::Forbid(
          "fusion into output of a reduce fusion would create a loop fusion");
    }
  }

  // Avoid cases where we'd create a fusion that hit limitations in ptxas.
  // Would be nice to model this with cost instead.
  if (auto fits_budget =
          FusionFitsInBudget(*consumer, *producer, device_info_,
                             /*is_consumer_producer_fusion=*/true,
                             &gpu_cost_model_->fusion_info_cache());
      !fits_budget) {
    return fits_budget;
  }

  // Also check that our emitter can handle the fusion node. We currently can
  // have exponential time/memory requirements for emitting certain fusion
  // kernels, in which case we don't want to fuse.
  if (gpu_cost_model_->cost_analysis().ProducerConsumerMergedTooLarge(
          *producer, *consumer)) {
    return FusionDecision::Forbid(
        "the fusion would result in an overly large code duplication");
  }

  return InstructionFusion::ShouldFuseInPlaceOp(producer, consumer, alias_info_,
                                                std::nullopt);
}

bool GpuPriorityFusion::IsFusible(const HloInstruction& instr) {
  bool fusible = false;
  if (instr.opcode() == HloOpcode::kFusion) {
    fusible = IsGenericTritonFusion(instr) ||
              instr.fusion_kind() != HloInstruction::FusionKind::kCustom;
  } else {
    fusible = PriorityFusion::IsFusible(instr);
  }

  return fusible;
}

HloInstruction::FusionKind GpuPriorityFusion::ChooseKind(
    const HloInstruction* producer, const HloInstruction* consumer,
    bool use_multi_output_fusion) {
  absl::MutexLock lock(&gpu_cost_model_->mutex());

  const auto& analysis =
      gpu_cost_model_->fusion_analysis_cache().Get(*producer, *consumer);

  switch (analysis.emitter_fusion_kind()) {
    case HloFusionAnalysis::EmitterFusionKind::kDynamicMemcpy:
    case HloFusionAnalysis::EmitterFusionKind::kLoop:
      return HloInstruction::FusionKind::kLoop;
    case HloFusionAnalysis::EmitterFusionKind::kTriton:
    case HloFusionAnalysis::EmitterFusionKind::kCustomFusion:
    case HloFusionAnalysis::EmitterFusionKind::kCuDnn:
    case HloFusionAnalysis::EmitterFusionKind::kSort:
      return HloInstruction::FusionKind::kCustom;
    case HloFusionAnalysis::EmitterFusionKind::kConcatenate:
    case HloFusionAnalysis::EmitterFusionKind::kReduction:
    case HloFusionAnalysis::EmitterFusionKind::kTranspose:
    case HloFusionAnalysis::EmitterFusionKind::kScatter:
      return HloInstruction::FusionKind::kInput;
  }
  return HloInstruction::FusionKind::kLoop;
}

HloInstruction* GpuPriorityFusion::Fuse(HloInstruction* producer,
                                        HloInstruction* consumer,
                                        bool use_multi_output_fusion) {
  // PriorityFusion::Fuse will call ChooseKind, which in turn calls IsTriton.
  // We rely on the caching within CanFuseTriton for efficiency.
  bool is_triton = false;
  std::optional<BlockLevelParameters> triton_params;
  is_triton = IsGenericTritonFusion(*producer) ||
              IsGenericTritonFusion(*consumer) ||
              CanFuseTriton(producer, consumer, use_multi_output_fusion);

  if (is_triton && gpu_cost_model_) {
    absl::MutexLock lock(&gpu_cost_model_->mutex());
    triton_params = gpu_cost_model_->GetTritonBlockLevelParameters(
        *producer, *consumer, use_multi_output_fusion);
  }

  HloInstruction* fusion =
      PriorityFusion::Fuse(producer, consumer, use_multi_output_fusion);

  if (is_triton) {
    absl::MutexLock lock(&gpu_cost_model_->mutex());
    fusion->set_fusion_kind(HloInstruction::FusionKind::kCustom);
    GpuBackendConfig gpu_backend_config;
    gpu_backend_config.mutable_fusion_backend_config()->set_kind(
        kTritonFusionKind);
    if (triton_params) {
      *gpu_backend_config.mutable_fusion_backend_config()
           ->mutable_block_level_fusion_config() =
          triton_params->ToBlockLevelFusionConfig();
    }
    CHECK_OK(fusion->set_backend_config(gpu_backend_config));
  }
  return fusion;
}
std::optional<HloInstruction*>
GpuPriorityFusion::GetPreferredMultiOutputConsumer(
    const HloInstruction* producer, HloDfsReachability* reachability) {
  bool triton_multi_output_fusion_enabled = false;
  if (producer->GetModule() != nullptr) {
    triton_multi_output_fusion_enabled =
        producer->GetModule()
            ->config()
            .debug_options()
            .xla_gpu_unsupported_enable_triton_multi_output_fusion();
  }

  if (!triton_multi_output_fusion_enabled) {
    return std::nullopt;
  }

  if (producer == producer->parent()->root_instruction()) {
    return std::nullopt;
  }

  std::vector<HloInstruction*> possible_consumers;
  for (const auto& user : producer->users()) {
    if (IsFusibleBitcast(*user)) {
      continue;
    }
    if (CanFuseTriton(producer, user, /*use_multi_output_fusion=*/true) &&
        !OperandReachableFromProducer(producer, user, reachability)) {
      possible_consumers.push_back(user);
    }
  }
  if (possible_consumers.size() == 1) {
    return possible_consumers[0];
  }

  return std::nullopt;
}

void GpuPriorityFusion::PopulateFusionProcessDump(
    FusionProcessDumpProto* dump) {
  *dump->mutable_gpu_device_info() = device_info_.ToProto();
}

std::vector<HloComputation*> GpuPriorityFusion::GetFusibleComputations(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return xla::gpu::GetFusibleComputations(*module, execution_threads);
}

}  // namespace gpu
}  // namespace xla
