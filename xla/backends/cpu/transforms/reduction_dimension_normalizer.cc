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

#include "xla/backends/cpu/transforms/reduction_dimension_normalizer.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_rewrite_utils.h"
#include "xla/service/decision.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

Decision CanNormalizeReduction(const HloInstruction* hlo) {
  if ((hlo->opcode() != HloOpcode::kReduce &&
       hlo->opcode() != HloOpcode::kReduceWindow) ||
      !hlo->shape().IsArray() || hlo->shape().is_dynamic() ||
      hlo->HasControlDependencies()) {
    return Decision::Forbid("Expected a static, single-output reduction.");
  }
  const Shape& input = hlo->operand(0)->shape();
  if (input.is_dynamic() || !ShapeUtil::HasDegenerateDimensions(input)) {
    return Decision::Forbid("No static degenerate input dimensions.");
  }
  if (hlo->opcode() == HloOpcode::kReduceWindow) {
    for (int64_t dim = 0; dim < input.dimensions().size(); ++dim) {
      if (input.dimensions(dim) == 1 && !window_util::IsTrivialWindowDimension(
                                            hlo->window().dimensions(dim))) {
        return Decision::Forbid("Degenerate input has a nontrivial window.");
      }
    }
  }
  return Decision::Allow();
}

HloInstruction* ReshapeTo(const Shape& shape, HloInstruction* operand) {
  // Reuse the normalized value behind the shape-restoring reshape inserted
  // for an earlier producer. This keeps the entire chain at the smaller rank,
  // while other users can continue to observe the original shape.
  if (operand->opcode() == HloOpcode::kReshape &&
      !operand->HasControlDependencies() &&
      ShapeUtil::Equal(shape, operand->operand(0)->shape())) {
    return operand->mutable_operand(0);
  }
  if (ShapeUtil::Equal(shape, operand->shape())) {
    return operand;
  }
  return operand->parent()->AddInstruction(
      HloInstruction::CreateReshape(shape, operand));
}

}  // namespace

absl::StatusOr<bool> ReductionDimensionNormalizer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    absl::flat_hash_set<HloInstruction*> candidates;
    std::vector<HloInstruction*> worklist;
    for (HloInstruction* hlo : computation->instructions()) {
      if ((hlo->opcode() == HloOpcode::kReduce ||
           hlo->opcode() == HloOpcode::kReduceWindow) &&
          CanNormalizeReduction(hlo)) {
        candidates.insert(hlo);
        worklist.push_back(hlo->mutable_operand(0));
      }
    }
    while (!worklist.empty()) {
      HloInstruction* hlo = worklist.back();
      worklist.pop_back();
      if (!hlo->shape().IsArray() || hlo->shape().is_dynamic() ||
          hlo->HasControlDependencies() ||
          !ShapeUtil::HasDegenerateDimensions(hlo->shape()) ||
          !IsTrivialElementwise(*hlo) ||
          !absl::c_all_of(hlo->operands(),
                          [&](HloInstruction* operand) {
                            return operand->shape().is_static() &&
                                   ShapeUtil::SameDimensions(hlo->shape(),
                                                             operand->shape());
                          }) ||
          !candidates.insert(hlo).second) {
        continue;
      }
      for (HloInstruction* operand : hlo->operands()) {
        worklist.push_back(operand);
      }
    }

    for (HloInstruction* hlo : computation->MakeInstructionPostOrder()) {
      if (!candidates.contains(hlo)) {
        continue;
      }
      Shape result_shape = ShapeUtil::DropDegenerateDimensions(hlo->shape());
      std::unique_ptr<HloInstruction> replacement;
      if (hlo->opcode() == HloOpcode::kReduce ||
          hlo->opcode() == HloOpcode::kReduceWindow) {
        const Shape& input_shape = hlo->operand(0)->shape();
        std::vector<int64_t> removed_dims;
        std::vector<int64_t> reduction_dims;
        Window window;
        for (int64_t dim = 0; dim < input_shape.dimensions().size(); ++dim) {
          if (input_shape.dimensions(dim) == 1) {
            removed_dims.push_back(dim);
          } else if (hlo->opcode() == HloOpcode::kReduceWindow) {
            *window.add_dimensions() = hlo->window().dimensions(dim);
          } else if (absl::c_linear_search(hlo->dimensions(), dim)) {
            reduction_dims.push_back(dim -
                                     static_cast<int64_t>(removed_dims.size()));
          }
        }
        HloInstruction* input =
            ReshapeTo(ShapeUtil::DropDegenerateDimensions(input_shape),
                      hlo->mutable_operand(0));
        if (hlo->opcode() == HloOpcode::kReduce) {
          replacement = HloInstruction::CreateReduce(
              result_shape, input, hlo->mutable_operand(1), reduction_dims,
              hlo->to_apply());
        } else {
          result_shape =
              ShapeUtil::DeleteDimensions(removed_dims, hlo->shape());
          replacement = HloInstruction::CreateReduceWindow(
              result_shape, input, hlo->mutable_operand(1), window,
              hlo->to_apply());
        }
      } else {
        std::vector<HloInstruction*> operands;
        for (HloInstruction* operand : hlo->operands()) {
          operands.push_back(ReshapeTo(
              ShapeUtil::DropDegenerateDimensions(operand->shape()), operand));
        }
        replacement = hlo->CloneWithNewOperands(result_shape, operands);
      }
      hlo->SetupDerivedInstruction(replacement.get());
      replacement->clear_sharding();
      if (!ShapeUtil::Equal(result_shape, hlo->shape())) {
        HloInstruction* normalized =
            computation->AddInstruction(std::move(replacement));
        replacement = HloInstruction::CreateReshape(hlo->shape(), normalized);
      }
      ABSL_RETURN_IF_ERROR(
          computation->ReplaceWithNewInstruction(hlo, std::move(replacement)));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla::cpu
