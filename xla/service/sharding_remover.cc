/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/sharding_remover.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "tsl/platform/errors.h"

namespace xla {

// Remove Sharding custom-call instruction by assigning its users to
// to its operand.
StatusOr<bool> ShardingRemover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  CHECK(module->config().num_partitions() == 1)
      << "Number of partitions must be 1";
  bool changed = false;

  const absl::flat_hash_set<absl::string_view> to_remove_sharding_ops = {
      "Sharding", "SPMDShardToFullShape", "SPMDFullToShardShape"};

  for (HloComputation* computation : module->computations(execution_threads)) {
    auto instructions = computation->MakeInstructionPostOrder();
    std::reverse(instructions.begin(), instructions.end());
    for (HloInstruction* instruction : instructions) {
      if (instruction->opcode() != HloOpcode::kCustomCall) {
        continue;
      }

      if (!to_remove_sharding_ops.contains(instruction->custom_call_target())) {
        continue;
      }
      CHECK(instruction->operand_count() == 1)
          << "Sharding instruction must have exactly one operand";
      if (instruction->custom_call_target() != "Sharding" &&
          instruction->has_sharding() && !instruction->sharding().IsManual()) {
        const int64_t num_tiles = instruction->sharding().TotalNumTiles();
        CHECK(num_tiles == 1) << absl::StrFormat(
            "Sharding instruction: %s must have exactly one tile, but it has "
            "%d instead.",
            instruction->ToString(), num_tiles);
      }
      const HloInstruction* operand = instruction->operand(0);
      if (operand->has_sharding() && !operand->sharding().IsManual()) {
        const int64_t num_tiles = operand->sharding().TotalNumTiles();
        CHECK(num_tiles == 1) << absl::StrFormat(
            "Operand: %s of sharding instruction: %s must have exactly one "
            "tile, but it has %d instead.",
            instruction->ToString(), operand->ToString(), num_tiles);
      }
      TF_RETURN_IF_ERROR(
          instruction->ReplaceAllUsesWith(instruction->mutable_operand(0)));
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla
