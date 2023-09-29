/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/dynamic_slice_to_slice.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

namespace xla {
StatusOr<bool> DynamicSliceToSlice::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(3,
                 "DynamicSliceToSlice::Run(), before:\n" + module->ToString());
  bool changed = false;
  std::vector<HloInstruction*> instructions;
  for (auto* comp : module->MakeComputationPostOrder()) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(instructions),
                    HloPredicateIsOp<HloOpcode::kDynamicSlice>);
  }

  for (HloInstruction* instruction : instructions) {
    absl::Span<HloInstruction* const> start_indices =
        absl::Span<HloInstruction* const>(instruction->operands()).subspan(1);

    // Vectors that store the arguments to the slice operation. Note that the
    // type of the start and limit values are always int64_t regardless of the
    // dynamic index argument types.
    std::vector<int64_t> start_indices_const;
    std::vector<int64_t> limit_indices;
    start_indices_const.reserve(start_indices.size());
    int64_t i = 0;
    bool is_static = true;
    for (HloInstruction* const start_index : start_indices) {
      if (!start_index->IsConstant()) {
        is_static = false;
        break;
      }
      std::optional<int64_t> idx = Cast<HloConstantInstruction>(start_index)
                                       ->literal()
                                       .GetFirstInteger();

      CHECK(idx.has_value());

      int64_t clamped_start =
          std::clamp(idx.value(), (int64_t)0,
                     instruction->operand(0)->shape().dimensions(i) -
                         instruction->slice_sizes(i));
      start_indices_const.push_back(clamped_start);
      limit_indices.push_back(clamped_start + instruction->slice_sizes(i++));
    }
    if (is_static) {
      // Replace the instruction with the static version
      HloInstruction* slice =
          instruction->AddInstruction(HloInstruction::CreateSlice(
              instruction->shape(), instruction->mutable_operand(0),
              start_indices_const, limit_indices, {}));
      XLA_VLOG_LINES(3, "Replacing " + instruction->ToString() + " with " +
                            slice->ToString());
      TF_RETURN_IF_ERROR(
          instruction->parent()->ReplaceInstruction(instruction, slice));

      changed |= true;
    }
  }

  XLA_VLOG_LINES(3,
                 "DynamicSliceToSlice::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
