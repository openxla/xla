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

#include "xla/backends/gpu/transforms/dynamic_slice_copy_fusion_analysis.h"

#include <memory>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

bool HasDynamicSliceConfig(const HloInstruction* instr) {
  auto config = instr->backend_config<GpuBackendConfig>();
  return config.ok() && config->has_dynamic_slice_config();
}

bool IsSlicingInstruction(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kSlice ||
         instr->opcode() == HloOpcode::kDynamicSlice ||
         instr->opcode() == HloOpcode::kDynamicUpdateSlice;
}

bool IsSlicingInstructionCompatible(const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kSlice) {
    return IsContiguousSlice(*instr) &&
           ShapeUtil::ByteStrides(instr->operand(0)->shape()).has_value();
  }

  return HasDynamicSliceConfig(instr);
}

bool AllSlicingInstructionsCompatible(const HloComputation* computation) {
  for (const HloInstruction* instr : computation->instructions()) {
    if (IsSlicingInstruction(instr) && !IsSlicingInstructionCompatible(instr)) {
      return false;
    }
  }
  return true;
}

bool IsBitcastOrReshape(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kBitcast ||
         instr->opcode() == HloOpcode::kReshape;
}

const HloInstruction* WalkThroughBitcastsAndReshapes(
    const HloInstruction* instr) {
  while (IsBitcastOrReshape(instr)) {
    instr = instr->operand(0);
  }
  return instr;
}

std::optional<DynamicSliceCopyFusionAnalysis> FindMemcpyFusionCandidate(
    const HloComputation* computation) {
  const HloInstruction* root = computation->root_instruction();
  const HloInstruction* ds_or_dus = WalkThroughBitcastsAndReshapes(root);

  if (!HasDynamicSliceConfig(ds_or_dus)) {
    return std::nullopt;
  }

  if (ds_or_dus->opcode() == HloOpcode::kDynamicSlice) {
    return DynamicSliceCopyFusionAnalysis{/*slicing=*/ds_or_dus,
                                          /*copy_operand=*/root};
  }

  if (ds_or_dus->opcode() == HloOpcode::kDynamicUpdateSlice) {
    return DynamicSliceCopyFusionAnalysis{
        /*slicing=*/ds_or_dus,
        /*copy_operand=*/ds_or_dus->operand(1)};
  }

  return std::nullopt;
}

bool CanLowerAsDynamicSliceCopyFusion(
    const DynamicSliceCopyFusionAnalysis& analysis) {
  auto resolve_copy_hero_parameters = [](const HloInstruction* operand) {
    std::unique_ptr<HloInstruction> copy =
        HloInstruction::CreateUnary(operand->shape(), HloOpcode::kCopy,
                                    const_cast<HloInstruction*>(operand));
    return DynamicSliceFusion::ResolveParameters(copy.get());
  };

  auto parameters = resolve_copy_hero_parameters(analysis.copy_operand);
  if (!parameters.ok()) {
    return false;
  }

  for (const DynamicSliceFusion::Parameter& parameter : *parameters) {
    if (parameter.slice_config.has_value()) {
      continue;
    }

    // Without DynamicSliceConfig, DynamicSliceFusion will pass the original
    // parameter buffer base address to the embedded copy thunk. This is only
    // correct for unsliced pass-through operands.
    if (ShapeUtil::ByteSizeOf(parameter.slice_shape) !=
        ShapeUtil::ByteSizeOf(parameter.parameter_shape)) {
      return false;
    }
  }

  if (analysis.slicing->opcode() == HloOpcode::kDynamicSlice) {
    return true;
  }

  return DynamicSliceFusion::ResolveResults(analysis.copy_operand).ok();
}

std::optional<DynamicSliceCopyFusionAnalysis> AnalyzeExistingCopyHeroFusion(
    const HloFusionInstruction& fusion) {
  std::optional<std::string> custom_name = GetCustomFusionConfigName(&fusion);
  if (!custom_name.has_value() ||
      *custom_name != kDynamicSliceFusionConfigName) {
    return std::nullopt;
  }

  const HloInstruction* hero =
      DynamicSliceFusion::FindHero(fusion.fused_instructions_computation());
  if (hero == nullptr || hero->opcode() != HloOpcode::kCopy) {
    return std::nullopt;
  }

  return DynamicSliceCopyFusionAnalysis{/*slicing=*/nullptr,
                                        /*copy_operand=*/hero->operand(0),
                                        /*existing_copy_hero=*/hero};
}

}  // namespace

absl::StatusOr<std::optional<DynamicSliceCopyFusionAnalysis>>
AnalyzeDynamicSliceCopyFusion(const HloFusionInstruction& fusion) {
  if (auto analysis = AnalyzeExistingCopyHeroFusion(fusion)) {
    return analysis;
  }

  if (fusion.fusion_kind() == HloInstruction::FusionKind::kCustom) {
    return std::nullopt;
  }

  std::optional<DynamicSliceCopyFusionAnalysis> analysis =
      FindMemcpyFusionCandidate(fusion.fused_instructions_computation());
  if (!analysis.has_value() ||
      !AllSlicingInstructionsCompatible(
          fusion.fused_instructions_computation()) ||
      !CanLowerAsDynamicSliceCopyFusion(*analysis)) {
    return std::nullopt;
  }

  return analysis;
}

bool IsDynamicSliceCopyFusion(const HloInstruction* instr) {
  auto* fusion = DynCast<HloFusionInstruction>(instr);
  if (fusion == nullptr) {
    return false;
  }

  auto analysis = AnalyzeDynamicSliceCopyFusion(*fusion);
  return analysis.ok() && analysis->has_value();
}

}  // namespace xla::gpu
