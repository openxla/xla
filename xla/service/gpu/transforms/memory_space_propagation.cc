/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/memory_space_propagation.h"

#include <optional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

bool Propagate(ShapeIndexView index, HloInstruction* instruction,
               const Shape& src_shape, const HloDataflowAnalysis* analysis) {
  bool modified = false;

  const HloValue& value =
      analysis->GetValueDefinedAt(instruction, ShapeIndex(index));
  auto memory_space_to_propagate = src_shape.layout().memory_space();
  for (const HloPosition& position : value.positions()) {
    auto* instruction = position.instruction;
    Shape* shape = ShapeUtil::GetMutableSubshape(instruction->mutable_shape(),
                                                 position.index);
    if (!shape->has_layout() ||
        shape->layout().memory_space() == memory_space_to_propagate) {
      continue;
    }
    shape->mutable_layout()->set_memory_space(memory_space_to_propagate);
    modified = true;
  }

  // We don't propagate from S(x) operand to S(x) output because it's
  // pure memory transfer without any computation in S(x). Computations
  // including transposes should be defined explicitly in JAX.
  return modified;
}

}  // namespace

absl::StatusOr<bool> MemorySpacePropagation::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool modified = false;
  TF_ASSIGN_OR_RETURN(auto dataflow_analysis,
                      HloDataflowAnalysis::Run(*module, /*ssa_form=*/false,
                                               /*bitcast_defines_value=*/true));

  absl::flat_hash_set<HloInstruction*> visited_instructions;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (visited_instructions.contains(instruction)) {
        continue;
      }
      visited_instructions.insert(instruction);
      ShapeUtil::ForEachLeafShape(
          instruction->shape(),
          [&, analysis = dataflow_analysis.get()](const Shape& sub_shape,
                                                  const ShapeIndex& index) {
            if (sub_shape.has_layout() &&
                sub_shape.layout().memory_space() !=
                    xla::Layout::kDefaultMemorySpace &&
                analysis->ValueIsDefinedAt(instruction, index)) {
              modified |= Propagate(index, instruction, sub_shape, analysis);
            }
          });
    }
  }
  return modified;
}
}  // namespace xla::gpu
