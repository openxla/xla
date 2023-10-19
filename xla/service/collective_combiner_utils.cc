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

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/statusor.h"

namespace xla {

namespace internal {
StatusOr<int64_t> SizeFromArrayShapedInstruction(
    const HloInstruction* instruction) {
  TF_RET_CHECK(instruction->shape().IsArray());
  return ShapeUtil::ByteSizeOf(instruction->shape());
}
}  // namespace internal

// If the module has been scheduled, updates its schedule to reflect the
// transformation described by the `replacements` map (i.e. keys are replaced by
// values) and the introduction of the combining instruction (`combined`).
//
// Correctness requires no dependencies in the schedule between the instructions
// to be combined; we thus schedule the combiner (and subsequent replacements,
// i.e. get-tuple-element's) where the first element of `to_combine` is.
static void MaybeUpdateSchedulePostCombining(
    HloComputation* computation, HloInstruction* combined,
    HloInstruction* combined_end, absl::Span<HloInstruction* const> to_combine,
    absl::Span<HloInstruction* const> to_combine_ends,
    const absl::flat_hash_map<HloInstruction*, HloInstruction*>& replacements) {
  HloModule* module = computation->parent();
  if (!module->has_schedule()) return;

  const auto& sequence = module->schedule().sequence(computation);
  std::vector<HloInstruction*> new_sequence;
  new_sequence.reserve(sequence.size());

  for (HloInstruction* instr : sequence.instructions()) {
    // 'combined' or 'combined_end' might have been added by module->Update(),
    // e.g. after adding a new async op.
    // Skip them here to insert them at the right place below.
    if (instr == combined || instr == combined_end) {
      continue;
    }
    auto it = replacements.find(instr);
    if (it == replacements.end()) {
      new_sequence.push_back(instr);
      continue;
    }
    if (instr == to_combine.front()) {
      new_sequence.push_back(combined);
    }
    if (it->second == nullptr) {
      // Removed instruction.
      continue;
    }
    if (instr == to_combine_ends.back()) {
      // Last "end" instruction; schedule the combined end and the tuples.
      new_sequence.push_back(combined_end);
      for (HloInstruction* c : to_combine_ends) {
        new_sequence.push_back(replacements.at(c));
      }
    }
  }
  module->schedule().set_sequence(computation, new_sequence);
}

Status CombineCollectives(HloComputation* computation, HloInstruction* combined,
                          HloInstruction* combined_end,
                          absl::Span<HloInstruction* const> to_combine,
                          absl::Span<HloInstruction* const> to_combine_ends,
                          bool is_async) {
  absl::flat_hash_map<HloInstruction*, HloInstruction*> replacements;

  if (is_async) {
    auto maybe_disconnect_called_computation =
        [](HloInstruction* op) -> Status {
      if (!op->IsAsynchronous()) {
        return OkStatus();
      }
      TF_RET_CHECK(op->called_computations().size() == 1);
      HloComputation* called = op->called_computations().front();
      called->RemoveAsyncInstruction(op);
      return OkStatus();
    };

    // Replace the Start/Done ops with the combined Start/Done ops.
    for (int64_t i = 0; i < to_combine.size(); ++i) {
      HloInstruction* start = to_combine[i];
      HloInstruction* done = to_combine_ends[i];

      // Copy metadata from the first iteration so that start/done
      // have matching suffixes (e.g. start.X, done.X).
      if (i == 0) {
        combined->set_metadata(start->metadata());
        combined_end->set_metadata(done->metadata());
      }

      // Start op.
      replacements[start] = nullptr;
      TF_RETURN_IF_ERROR(combined->CopyAllControlDepsFrom(start));
      TF_RETURN_IF_ERROR(start->DropAllControlDeps());
      TF_RETURN_IF_ERROR(start->ReplaceAllUsesWithDifferentShape(combined));
      TF_RETURN_IF_ERROR(maybe_disconnect_called_computation(start));
      TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(
          start, /*cleanup=*/std::nullopt,
          /*ignore_control_dependencies=*/true));

      // Done op.
      const Shape& shape = done->shape();
      TF_RET_CHECK(shape.IsArray());
      HloInstruction* tuple = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(shape, combined_end, i));
      replacements[to_combine_ends[i]] = tuple;
      TF_RETURN_IF_ERROR(maybe_disconnect_called_computation(done));
      TF_ASSIGN_OR_RETURN(bool changed, computation->ReplaceInstruction(
                                            done, tuple,
                                            /*preserve_sharding=*/false,
                                            /*relay_control_dependency=*/true));
      DCHECK(changed);
    }
  } else {
    // Replace all the smaller collectives with elements of the tuple output
    // of the single bigger collective.
    for (int64_t i = 0; i < to_combine.size(); ++i) {
      auto replace_with = HloInstruction::CreateGetTupleElement(
          to_combine[i]->shape(), combined, i);
      TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
          to_combine[i], std::move(replace_with)));
    }
  }

  MaybeUpdateSchedulePostCombining(computation, combined, combined_end,
                                   to_combine, to_combine_ends, replacements);

  return OkStatus();
}

}  // namespace xla
