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

#include "xla/hlo/ir/hlo_original_value_util.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/utils/pointer_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

void CopyOriginalValue(const HloInstruction* src_instruction,
                       HloInstruction* dest_instruction, bool clone,
                       bool issue_warning) {
  if (!src_instruction || !dest_instruction ||
      !ShapeUtil::Compatible(src_instruction->shape(),
                             dest_instruction->shape())) {
    if (issue_warning) {
      LOG(WARNING)
          << "Expect the new instruction to have the same shape with the old "
             "instruction when moving over original_value";
    }
    return;
  }

  std::shared_ptr<OriginalValue> original_value =
      src_instruction->original_value();
  if (!original_value) {
    return;
  }

  if (!clone || original_value->is_synthetic_call()) {
    dest_instruction->set_original_value(original_value);
    return;
  }

  // Deep clone the tree.
  auto cloned_tree = std::make_shared<OriginalValue>(
      original_value->tree(), original_value->call_hierarchy());
  dest_instruction->set_original_value(cloned_tree);
}

void AppendEmptyElementsToOriginalValue(HloInstruction* instruction) {
  std::shared_ptr<OriginalValue> old_original_value =
      instruction->original_value();
  const Shape& shape = instruction->shape();
  if (old_original_value == nullptr ||
      old_original_value->is_synthetic_call() ||
      !old_original_value->IsTuple() || !shape.IsTuple()) {
    return;
  }
  const int64_t num_old_elements = old_original_value->tree().num_children();
  if (num_old_elements >= shape.tuple_shapes().size()) {
    return;
  }
  auto new_original_value = std::make_shared<OriginalValue>(
      shape, old_original_value->call_hierarchy());
  for (int64_t i = 0; i < num_old_elements; ++i) {
    if (!new_original_value->mutable_tree()
             ->CopyCompatibleSubtreeFrom(old_original_value->tree(), {i}, {i})
             .ok()) {
      // The existing elements don't match the leading elements of the shape,
      // so the shape was not widened by appending elements.
      return;
    }
  }
  instruction->set_original_value(std::move(new_original_value));
}

void DeduplicateOriginalValues(HloModule* module) {
  absl::flat_hash_set<std::shared_ptr<OriginalValue>,
                      PointeeHash<OriginalValue>, PointeeEqual<OriginalValue>>
      unique_original_values;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (std::shared_ptr<OriginalValue> original_value =
              instruction->original_value()) {
        auto p = unique_original_values.insert(original_value);
        if (!p.second) {
          // Reassign the pointer with the existing identical object and release
          // the duplicate.
          instruction->set_original_value(*p.first);
        }
      }
    }
  }
}
}  // namespace xla
