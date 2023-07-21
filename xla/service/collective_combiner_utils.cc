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

#include "xla/service/collective_combiner_utils.h"

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

void MaybeUpdateSchedulePostCombining(
    HloComputation* computation, HloInstruction* combined,
    absl::Span<HloInstruction* const> to_combine,
    const absl::flat_hash_map<HloInstruction*, HloInstruction*>& replacements) {
  HloModule* module = computation->parent();
  if (!module->has_schedule()) return;

  const auto& sequence = module->schedule().sequence(computation);
  std::vector<HloInstruction*> new_sequence;
  new_sequence.reserve(sequence.size());

  for (HloInstruction* instr : sequence.instructions()) {
    auto it = replacements.find(instr);
    if (it == replacements.end()) {
      new_sequence.push_back(instr);
      continue;
    }
    if (instr != to_combine[to_combine.size() - 1]) continue;
    new_sequence.push_back(combined);
    for (HloInstruction* c : to_combine) {
      new_sequence.push_back(replacements.at(c));
    }
  }
  module->schedule().set_sequence(computation, new_sequence);
}

}  // namespace xla
