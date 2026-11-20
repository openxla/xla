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

#include "xla/hlo/transforms/simplifiers/hlo_rematerialization_data_structures.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"

namespace xla {

void HloRematInstructionList::RegisterItem(HloRematItem* item) {
  const HloInstruction* inst = item->instruction;
  const int32_t id = inst->local_id();
  if (ABSL_PREDICT_TRUE(id >= 0)) {
    const size_t idx = static_cast<size_t>(id);
    if (idx >= items_by_local_id_.size()) {
      items_by_local_id_.resize(
          std::max(idx + 1, items_by_local_id_.size() * 2), nullptr);
    }
    CHECK_EQ(items_by_local_id_[idx], nullptr)
        << "inserting inst twice " << inst->name();
    items_by_local_id_[idx] = item;
  } else {
    CHECK(fallback_item_map_.emplace(inst, item).second)
        << "inserting inst twice " << inst->name();
  }
}

HloRematItem* HloRematInstructionList::CreateItem(HloInstruction* inst) {
  HloRematItem& item = items_.emplace_back();
  item.instruction = inst;
  item.has_control_successors = !inst->control_successors().empty();
  RegisterItem(&item);
  return &item;
}

absl::Status HloRematInstructionList::UpdateFromSequence(
    const HloInstructionSequence& order, bool preserve_denylist) {
  absl::flat_hash_set<const HloInstruction*> denylist;
  if (preserve_denylist) {
    for (const HloRematItem& existing : items_) {
      if (existing.denylisted) {
        denylist.insert(existing.instruction);
      }
    }
  }
  items_.clear();
  items_by_local_id_.clear();
  fallback_item_map_.clear();
  first_ = nullptr;
  last_skip_node_ = nullptr;
  first_skip_node_ = nullptr;

  int32_t max_local_id = -1;
  for (HloInstruction* inst : order.instructions()) {
    max_local_id = std::max(max_local_id, inst->local_id());
  }
  if (max_local_id >= 0) {
    items_by_local_id_.assign(static_cast<size_t>(max_local_id) + 1, nullptr);
  }

  int64_t position = 0;
  HloRematItem* last = nullptr;
  for (HloInstruction* inst : order.instructions()) {
    HloRematItem& item = items_.emplace_back();
    item.instruction = inst;
    if (preserve_denylist && !denylist.empty() && denylist.contains(inst)) {
      item.denylisted = true;
    }
    item.is_dead = inst->IsDead();
    item.has_control_successors = !inst->control_successors().empty();
    item.next = nullptr;
    item.prev = last;
    if (last == nullptr) {
      first_ = &item;
    } else {
      last->next = &item;
    }
    last = &item;

    // Initially position numbers are uniquely assigned in order. Later as
    // instructions are added with InsertBefore* methods, some instructions
    // may have duplicate position numbers, but the values will be guaranteed
    // to be monotonically increasing through the list, and so is still useful
    // for quickly(-ish) determining the order of arbitrary instructions in
    // the list.
    item.position = position++;
    RegisterItem(&item);
  }
  return absl::OkStatus();
}

void HloRematInstructionList::InsertBeforeInstructions(
    HloRematItem* to_insert,
    absl::Span<HloRematItem* const> before_instructions) {
  VLOG(3) << "InsertBeforeInstructions: " << to_insert->instruction->name()
          << " before {"
          << absl::StrJoin(before_instructions, ", ",
                           [](std::string* out, HloRematItem* item) {
                             absl::StrAppend(out, item->instruction->name());
                           })
          << "}";

  // Find the minimal position number of any instruction in
  // 'before_instructions'.
  CHECK(!before_instructions.empty());
  HloRematItem* min_position_item = nullptr;
  for (HloRematItem* item : before_instructions) {
    if (min_position_item == nullptr ||
        item->position < min_position_item->position) {
      min_position_item = item;
    }
  }

  // Because more than one instruction in 'before_instructions' may have a
  // position number of 'min_position_number', find the first such instruction
  // with position number 'min_position_number'.

  // First find first instruction with the min position.
  CHECK(min_position_item != nullptr);
  while (min_position_item->prev != nullptr &&
         min_position_item->position == min_position_item->prev->position) {
    min_position_item = min_position_item->prev;
  }

  // Now scan forwards until we find one of the before_instructions.
  while (!absl::c_linear_search(before_instructions, min_position_item)) {
    min_position_item = min_position_item->next;
  }
  return InsertBefore(to_insert, min_position_item);
}

void HloRematInstructionList::PromoteNodesToSkip(
    absl::FunctionRef<bool(HloRematItem*)> should_promote) {
  int64_t count = 0;
  for (auto* item = first(); item != nullptr; item = next(item)) {
    if (!item->denylisted && !item->is_dead && !item->pinned_by_overwrite &&
        item->rematerializable != 0 && should_promote(item)) {
      count += 1;
      if (first_skip_node_ == nullptr) {
        first_skip_node_ = item;
      }
      item->is_skip_node = true;
      item->prev_skip_node = last_skip_node_;
      if (last_skip_node_ != nullptr) {
        last_skip_node_->next_skip_node = item;
      }
      last_skip_node_ = item;
    }
  }
  VLOG(1) << " Rematerialization has " << count << " items in express lane";
}

void HloRematInstructionList::RemoveFromSkipList(HloRematItem* item) {
  if (!item->is_skip_node) {
    return;
  }
  if (item->prev_skip_node != nullptr) {
    item->prev_skip_node->next_skip_node = item->next_skip_node;
  } else {
    first_skip_node_ = item->next_skip_node;
  }
  if (item->next_skip_node != nullptr) {
    item->next_skip_node->prev_skip_node = item->prev_skip_node;
  } else {
    last_skip_node_ = item->prev_skip_node;
  }
  item->prev_skip_node = nullptr;
  item->next_skip_node = nullptr;
  item->is_skip_node = false;
}

void HloRematInstructionList::MarkDead(HloRematItem* item) {
  item->is_dead = true;
  RemoveFromSkipList(item);
  fallback_item_map_[item->instruction] = item;
}

void HloRematInstructionList::InsertAfterInstructions(
    HloRematItem* to_insert,
    absl::Span<HloRematItem* const> after_instructions) {
  VLOG(3) << "InsertAfterInstructions: " << to_insert->instruction->name()
          << " after {"
          << absl::StrJoin(after_instructions, ", ",
                           [](std::string* out, HloRematItem* item) {
                             absl::StrAppend(out, item->instruction->name());
                           })
          << "}";

  // Find the max position number of any instruction in
  // 'after_instructions'.
  CHECK(!after_instructions.empty());
  HloRematItem* max_position_item = nullptr;
  for (HloRematItem* item : after_instructions) {
    if (max_position_item == nullptr ||
        item->position > max_position_item->position) {
      max_position_item = item;
    }
  }
  // No rematerializable instruction should be inserted at the end of the
  // computation.
  CHECK(max_position_item != nullptr);
  CHECK(max_position_item->next != nullptr);
  InsertBeforeInstructions(to_insert, {max_position_item->next});
}

void HloRematInstructionList::InsertBefore(HloRematItem* item,
                                           HloRematItem* before) {
  VLOG(3) << "InsertBefore: " << item->instruction->name() << " before "
          << before->instruction->name();
  if (!item->denylisted && !item->is_dead && !item->pinned_by_overwrite &&
      item->rematerializable != 0) {
    item->is_skip_node = true;
    // Find the next express node starting from 'before'. Set up the node's
    // express pointers.
    HloRematItem* cursor = before;
    while (cursor != nullptr && !cursor->is_skip_node) {
      cursor = cursor->next;
    }
    CHECK(cursor == nullptr || cursor->is_skip_node);
    if (cursor == nullptr) {
      item->prev_skip_node = last_skip_node_;
      if (last_skip_node_ != nullptr) {
        last_skip_node_->next_skip_node = item;
      }
      item->next_skip_node = nullptr;
      last_skip_node_ = item;
    } else {
      CHECK(cursor->is_skip_node);
      item->prev_skip_node = cursor->prev_skip_node;
      if (item->prev_skip_node != nullptr) {
        item->prev_skip_node->next_skip_node = item;
      }
      item->next_skip_node = cursor;
      cursor->prev_skip_node = item;
    }
    if (first_skip_node_ == cursor) {
      first_skip_node_ = item;
    }
  }
  // Insert new item into linked list.
  item->prev = before->prev;
  item->next = before;
  before->prev = item;
  if (item->prev != nullptr) {
    item->prev->next = item;
  } else {
    first_ = item;
  }

  // Assign the same position number to the newly added instruction as
  // 'before'. This guarantees monotonicity of the position numbers, but not
  // uniqueness.
  item->position = before->position;
}
}  // namespace xla
