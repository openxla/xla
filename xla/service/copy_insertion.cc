/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/copy_insertion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/frontend_attributes.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_operand_index.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/ir/ptrvec.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/service/call_graph.h"
#include "xla/service/compile_time_cap.h"
#include "xla/service/copy_removal.h"
#include "xla/service/dump.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"

namespace xla {
namespace {

bool IsReadonlyEntryParameterValue(const HloValue& value) {
  const HloComputation* computation = value.defining_instruction()->parent();
  return value.defining_instruction()->opcode() == HloOpcode::kParameter &&
         computation == computation->parent()->entry_computation() &&
         !computation->parent()->input_output_alias_config().ParameterHasAlias(
             value.defining_instruction()->parameter_number(), value.index());
}

bool IsConstantValue(const HloValue& value) {
  return value.defining_instruction()->opcode() == HloOpcode::kConstant;
}

bool ValueIsReadOnly(const HloValue& value) {
  return IsConstantValue(value) || IsReadonlyEntryParameterValue(value);
}

// Data structure describing the action which should be taken on parts of a
// computation buffers, with respect to the adding of special case copies.
struct SpecialCaseCopyPolicy {
  // Insert a copy if the same buffer is found at multiple indices within the
  // output tuple.
  bool copy_root_replicated_buffers = false;
  // If true, insert a copy if a buffer coming from a constant or a parameter
  // is found within the output tuple.
  bool copy_parameters_and_constants = false;
};

// Returns true if any conditional instruction calling `node` defines a new Phi
// value at `index`. If the conditional simply passes through reaching values at
// `index` without merging, no replicated root copies are needed.
bool ShouldCopyConditionalRootAt(const CallGraphNode& node,
                                 const HloAliasAnalysis& alias_analysis,
                                 const ShapeIndex& index) {
  const HloDataflowAnalysis& dataflow = alias_analysis.dataflow_analysis();
  for (const CallSite& caller_callsite : node.caller_callsites()) {
    if (caller_callsite.instruction()->opcode() == HloOpcode::kConditional) {
      if (!ShapeUtil::IndexIsValid(caller_callsite.instruction()->shape(),
                                   index)) {
        return true;  // Conservatively copy if index is out of bounds.
      }
      const HloValueSet& value_set =
          dataflow.GetValueSet(caller_callsite.instruction(), index);
      if (absl::c_any_of(value_set.values(), [](const HloValue* value) {
            return value->is_phi();
          })) {
        return true;
      }
    }
  }
  return false;
}

SpecialCaseCopyPolicy GetSpecialCaseCopyPolicy(const CallGraphNode& node,
                                               HloModule* module,
                                               HloComputation* computation) {
  SpecialCaseCopyPolicy policy;
  if (computation == module->entry_computation()) {
    policy.copy_parameters_and_constants = true;
    policy.copy_root_replicated_buffers = true;
  }
  return policy;
}

bool ShouldCopyRootValue(const HloValue& value,
                         const SpecialCaseCopyPolicy& policy) {
  if (policy.copy_parameters_and_constants) {
    return ValueIsReadOnly(value);
  }
  return false;
}

bool IsWhileLoopCopyDisabled(const HloInstruction* instruction,
                             const CallGraph& call_graph) {
  if (instruction == nullptr || instruction->parent() == nullptr) {
    return false;
  }
  absl::flat_hash_set<const HloComputation*> visited;
  std::vector<const HloComputation*> worklist = {instruction->parent()};
  while (!worklist.empty()) {
    const HloComputation* curr = worklist.back();
    worklist.pop_back();
    if (!visited.insert(curr).second) {
      continue;
    }
    const CallGraphNode& node = call_graph.GetNode(curr);
    for (const CallSite& callsite : node.caller_callsites()) {
      const HloInstruction* caller = callsite.instruction();
      if (caller != nullptr && caller->opcode() == HloOpcode::kWhile &&
          frontend::HasDisableWhileLoopCopiesAttr(caller)) {
        return true;
      }
      if (caller != nullptr && caller->parent() != nullptr) {
        worklist.push_back(caller->parent());
      }
    }
  }
  return false;
}

using CopyListTree = ShapeTree<absl::InlinedVector<HloInstruction*, 1>>;

CopyInsertion::PassthroughAttachment ClassifyValidatedPassthrough(
    const HloInstruction* inst,
    const CopyInsertion::PassthroughClassifier& classify_passthrough) {
  CHECK_NE(inst, nullptr);
  if (classify_passthrough == nullptr) {
    return CopyInsertion::PassthroughAttachment::kNone;
  }
  CopyInsertion::PassthroughAttachment attachment = classify_passthrough(inst);
  if (attachment != CopyInsertion::PassthroughAttachment::kNone) {
    CHECK_EQ(inst->operand_count(), 1) << inst->ToString();
    CHECK(ShapeUtil::Compatible(inst->shape(), inst->operand(0)->shape()))
        << inst->ToString();
  }
  return attachment;
}

// Clones `marker` onto `new_operand` with `new_shape` in `comp` while
// preserving any control predecessors and control successors attached to
// `marker`.
//
// REQUIRES: `marker`, `new_operand`, and `comp` are not null, and
// `ShapeUtil::Compatible(new_shape, marker->shape())`.
HloInstruction* ClonePassthroughPreservingControlDeps(
    const HloInstruction* marker, HloInstruction* new_operand,
    const Shape& new_shape, HloComputation* comp) {
  HloInstruction* clone = comp->AddInstruction(
      marker->CloneWithNewOperands(new_shape, {new_operand}));
  for (HloInstruction* pred : marker->control_predecessors()) {
    CHECK_OK(pred->AddControlDependencyTo(clone));
  }
  for (HloInstruction* succ : marker->control_successors()) {
    CHECK_OK(clone->AddControlDependencyTo(succ));
  }
  return clone;
}

// Copies an array leaf `leaf` to `output_shape`, placing the new kCopy after
// any kProducer passthroughs and before any kConsumer passthroughs (except when
// `is_entry_root_copy` is true, where the entry root kCopy itself is the final
// consumer).
//
// REQUIRES: `leaf` and `comp` are not null, `leaf->shape().IsArray()`, and
// `ShapeUtil::Compatible(output_shape, leaf->shape())`.
absl::StatusOr<HloInstruction*> CopyArrayPreservingPassthroughs(
    HloInstruction* leaf, const Shape& output_shape,
    const HloInstruction* target_user, bool is_entry_root_copy,
    bool allow_in_place_rewire, HloComputation* comp,
    HloInstruction** recorded_copy,
    const CopyInsertion::PassthroughClassifier& classify_passthrough) {
  std::vector<HloInstruction*> wrappers;
  HloInstruction* curr = leaf;
  while (ClassifyValidatedPassthrough(curr, classify_passthrough) !=
         CopyInsertion::PassthroughAttachment::kNone) {
    wrappers.push_back(curr);
    curr = curr->mutable_operand(0);
  }
  HloInstruction* base = curr;

  if (wrappers.empty() || is_entry_root_copy) {
    HloInstruction* copy = comp->AddInstruction(
        HloInstruction::CreateUnary(output_shape, HloOpcode::kCopy, leaf));
    if (recorded_copy != nullptr) {
      *recorded_copy = copy;
    }
    return copy;
  }

  std::vector<HloInstruction*> producer_wrappers;
  std::vector<HloInstruction*> consumer_wrappers;
  bool seen_consumer = false;
  bool ordered_prod_then_cons = true;
  for (auto it = wrappers.rbegin(); it != wrappers.rend(); ++it) {
    HloInstruction* w = *it;
    if (ClassifyValidatedPassthrough(w, classify_passthrough) ==
        CopyInsertion::PassthroughAttachment::kProducer) {
      if (seen_consumer) {
        ordered_prod_then_cons = false;
      }
      producer_wrappers.push_back(w);
    } else {
      seen_consumer = true;
      consumer_wrappers.push_back(w);
    }
  }

  if (consumer_wrappers.empty()) {
    HloInstruction* copy = comp->AddInstruction(
        HloInstruction::CreateUnary(output_shape, HloOpcode::kCopy, leaf));
    if (recorded_copy != nullptr) {
      *recorded_copy = copy;
    }
    return copy;
  }

  bool can_rewire_in_place = allow_in_place_rewire && ordered_prod_then_cons &&
                             ShapeUtil::Equal(output_shape, leaf->shape());
  if (can_rewire_in_place) {
    can_rewire_in_place = target_user == nullptr
                              ? leaf->user_count() == 0
                              : (leaf->user_count() == 1 && !leaf->IsRoot() &&
                                 target_user->OperandIndices(leaf).size() == 1);
    for (size_t i = 0; i + 1 < consumer_wrappers.size() && can_rewire_in_place;
         ++i) {
      can_rewire_in_place &= consumer_wrappers[i]->user_count() == 1 &&
                             !consumer_wrappers[i]->IsRoot();
    }
  }
  if (can_rewire_in_place) {
    HloInstruction* prod_tail =
        producer_wrappers.empty() ? base : producer_wrappers.back();
    HloInstruction* copy = comp->AddInstruction(HloInstruction::CreateUnary(
        prod_tail->shape(), HloOpcode::kCopy, prod_tail));
    if (recorded_copy != nullptr) {
      *recorded_copy = copy;
    }
    ABSL_RETURN_IF_ERROR(
        prod_tail->ReplaceUseWith(consumer_wrappers.front(), 0, copy));
    return leaf;
  }

  HloInstruction* prod_tail = base;
  for (HloInstruction* w : producer_wrappers) {
    prod_tail = (w->operand(0) == prod_tail)
                    ? w
                    : ClonePassthroughPreservingControlDeps(w, prod_tail,
                                                            w->shape(), comp);
  }
  HloInstruction* copy = comp->AddInstruction(
      HloInstruction::CreateUnary(output_shape, HloOpcode::kCopy, prod_tail));
  if (recorded_copy != nullptr) {
    *recorded_copy = copy;
  }
  HloInstruction* cons_tail = copy;
  for (HloInstruction* w : consumer_wrappers) {
    cons_tail =
        ClonePassthroughPreservingControlDeps(w, cons_tail, output_shape, comp);
  }
  return cons_tail;
}

// Deep copies `instruction` for `target_user` (or for a computation root when
// `target_user` is null), placing each copied array leaf between any kProducer
// and kConsumer passthrough wrappers so both stay attached to their targets.
//
// REQUIRES: `instruction` is not null and belongs to a computation, and any
// non null `indices_to_copy` or `copies_added` is shape compatible with
// `instruction->shape()`.
absl::StatusOr<HloInstruction*> DeepCopyConsumerOperand(
    HloInstruction* instruction, const HloInstruction* target_user,
    const ShapeTree<bool>* indices_to_copy,
    ShapeTree<HloInstruction*>* copies_added,
    const CopyInsertion::PassthroughClassifier& classify_passthrough) {
  TF_RET_CHECK(instruction != nullptr);
  HloComputation* computation = instruction->parent();
  TF_RET_CHECK(computation != nullptr);
  if (indices_to_copy != nullptr) {
    TF_RET_CHECK(
        ShapeUtil::Compatible(instruction->shape(), indices_to_copy->shape()));
  }
  if (copies_added != nullptr) {
    TF_RET_CHECK(
        ShapeUtil::Compatible(instruction->shape(), copies_added->shape()));
  }
  if (classify_passthrough == nullptr) {
    return computation->DeepCopyInstruction(instruction, indices_to_copy,
                                            copies_added);
  }

  const bool is_entry_root_copy =
      target_user == nullptr && computation->IsEntryComputation();

  if (instruction->shape().IsArray()) {
    if (indices_to_copy != nullptr && !indices_to_copy->element({})) {
      return instruction;
    }
    HloInstruction* recorded = nullptr;
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * result,
        CopyArrayPreservingPassthroughs(
            instruction, instruction->shape(), target_user, is_entry_root_copy,
            /*allow_in_place_rewire=*/true, computation, &recorded,
            classify_passthrough));
    if (copies_added != nullptr) {
      *copies_added->mutable_element({}) = recorded;
    }
    return result;
  }

  if (!instruction->shape().IsTuple()) {
    return computation->DeepCopyInstruction(instruction, indices_to_copy,
                                            copies_added);
  }

  // If `instruction` itself is wrapped in tuple shaped kConsumer passthroughs
  // (such as tuple V0 coloring markers), unwrap them, deep copy the underlying
  // tuple, and reattach or clone the tuple kConsumer chain on top.
  if (!is_entry_root_copy &&
      ClassifyValidatedPassthrough(instruction, classify_passthrough) ==
          CopyInsertion::PassthroughAttachment::kConsumer) {
    std::vector<HloInstruction*> tuple_chain;
    HloInstruction* curr = instruction;
    while (ClassifyValidatedPassthrough(curr, classify_passthrough) ==
           CopyInsertion::PassthroughAttachment::kConsumer) {
      tuple_chain.push_back(curr);
      curr = curr->mutable_operand(0);
    }
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * copied_base,
        DeepCopyConsumerOperand(curr, /*target_user=*/nullptr, indices_to_copy,
                                copies_added, classify_passthrough));
    if (copied_base == curr) {
      return instruction;
    }
    bool can_rewire_in_place =
        ShapeUtil::Equal(copied_base->shape(), curr->shape()) &&
        (target_user == nullptr
             ? instruction->user_count() == 0
             : (instruction->user_count() == 1 && !instruction->IsRoot() &&
                target_user->OperandIndices(instruction).size() == 1));
    for (size_t i = 1; i < tuple_chain.size() && can_rewire_in_place; ++i) {
      can_rewire_in_place &=
          tuple_chain[i]->user_count() == 1 && !tuple_chain[i]->IsRoot();
    }
    if (can_rewire_in_place) {
      ABSL_RETURN_IF_ERROR(
          curr->ReplaceUseWith(tuple_chain.back(), 0, copied_base));
      return instruction;
    }
    HloInstruction* rebuilt = copied_base;
    for (auto it = tuple_chain.rbegin(); it != tuple_chain.rend(); ++it) {
      rebuilt = ClonePassthroughPreservingControlDeps(
          *it, rebuilt, (*it)->shape(), computation);
    }
    return rebuilt;
  }

  // Tuple path: walk downward from `(instruction, leaf_index)` across nested
  // kTuple and passthrough instructions so leaves that actually carry
  // passthrough wrappers keep kProducer and kConsumer wrappers on the right
  // side of the new kCopy, while leaves without passthroughs copy `leaf`
  // directly (matching HloComputation::DeepCopyInstruction and preserving any
  // explicit tuple subshape layout or tiling on `leaf->shape()`).
  return computation->DeepCopyInstructionWithCustomCopier(
      instruction,
      [&](HloInstruction* leaf, const ShapeIndex& leaf_index,
          HloComputation* comp) -> HloInstruction* {
        if (indices_to_copy != nullptr &&
            !indices_to_copy->element(leaf_index)) {
          return leaf;
        }
        if (!leaf->shape().IsArray()) {
          return leaf;
        }
        HloInstruction* underlying = instruction;
        bool resolved_to_leaf = true;
        bool crossed_passthrough = false;
        for (int64_t idx : leaf_index) {
          while (
              ClassifyValidatedPassthrough(underlying, classify_passthrough) !=
              CopyInsertion::PassthroughAttachment::kNone) {
            crossed_passthrough = true;
            underlying = underlying->mutable_operand(0);
          }
          if (underlying->opcode() == HloOpcode::kTuple) {
            underlying = underlying->mutable_operand(idx);
          } else {
            resolved_to_leaf = false;
            break;
          }
        }
        const bool has_leaf_passthrough =
            resolved_to_leaf &&
            (crossed_passthrough ||
             ClassifyValidatedPassthrough(underlying, classify_passthrough) !=
                 CopyInsertion::PassthroughAttachment::kNone);
        if (!has_leaf_passthrough) {
          HloInstruction* copy =
              comp->AddInstruction(HloInstruction::CreateUnary(
                  leaf->shape(), HloOpcode::kCopy, leaf));
          if (copies_added != nullptr) {
            *copies_added->mutable_element(leaf_index) = copy;
          }
          return copy;
        }
        HloInstruction* recorded = nullptr;
        absl::StatusOr<HloInstruction*> copied =
            CopyArrayPreservingPassthroughs(
                underlying, leaf->shape(), /*target_user=*/nullptr,
                is_entry_root_copy, /*allow_in_place_rewire=*/false, comp,
                &recorded, classify_passthrough);
        CHECK_OK(copied.status());
        if (copies_added != nullptr) {
          *copies_added->mutable_element(leaf_index) = recorded;
        }
        return *copied;
      });
}

// Returns true if every non passthrough user of the underlying producer of
// `operand` carries the disjoint read/write regions attribute, memoized by
// unwrapped producer in `disjoint_base_cache`.
//
// REQUIRES: `operand` and `disjoint_base_cache` are not null, and the caller
// has already verified `HasDisjointReadWriteRegionsAttr` on the candidate op.
bool AllNonPassthroughUsersHaveDisjointRegions(
    const HloInstruction* operand,
    const CopyInsertion::PassthroughClassifier& classify_passthrough,
    absl::flat_hash_map<const HloInstruction*, bool>* disjoint_base_cache) {
  CHECK_NE(operand, nullptr);
  CHECK_NE(disjoint_base_cache, nullptr);
  const HloInstruction* base = operand;
  while (ClassifyValidatedPassthrough(base, classify_passthrough) !=
         CopyInsertion::PassthroughAttachment::kNone) {
    base = base->operand(0);
  }
  if (auto it = disjoint_base_cache->find(base);
      it != disjoint_base_cache->end()) {
    return it->second;
  }
  absl::flat_hash_set<const HloInstruction*> visited = {base};
  std::vector<const HloInstruction*> worklist = {base};
  bool all_disjoint = true;
  while (!worklist.empty()) {
    const HloInstruction* curr = worklist.back();
    worklist.pop_back();
    for (const HloInstruction* user : curr->users()) {
      if (ClassifyValidatedPassthrough(user, classify_passthrough) !=
          CopyInsertion::PassthroughAttachment::kNone) {
        if (visited.insert(user).second) {
          worklist.push_back(user);
        }
        continue;
      }
      if (!frontend::HasDisjointReadWriteRegionsAttr(user)) {
        all_disjoint = false;
        worklist.clear();
        break;
      }
    }
  }
  disjoint_base_cache->emplace(base, all_disjoint);
  return all_disjoint;
}

// Deep copies `instruction` at `indices_to_copy` and rewires its downstream
// uses (and computation root when `replace_root` is true and `instruction` or
// an attached kProducer passthrough is the root), placing copies after any
// attached kProducer passthroughs without introducing marker to marker edges.
//
// REQUIRES: `instruction` is not null and belongs to a computation,
// `indices_to_copy` is shape compatible with `instruction->shape()`, and any
// non null `copies_added` is shape compatible with `instruction->shape()`.
absl::Status InsertProducerCopyImpl(
    HloInstruction* instruction, const ShapeTree<bool>& indices_to_copy,
    std::optional<absl::Span<HloInstruction* const>> users_override,
    bool replace_root,
    const absl::flat_hash_set<const HloInstruction*>* skip_users,
    CopyListTree* copies_added,
    const CopyInsertion::PassthroughClassifier& classify_passthrough) {
  TF_RET_CHECK(instruction != nullptr);
  HloComputation* computation = instruction->parent();
  TF_RET_CHECK(computation != nullptr);
  TF_RET_CHECK(
      ShapeUtil::Compatible(instruction->shape(), indices_to_copy.shape()));
  if (copies_added != nullptr) {
    TF_RET_CHECK(
        ShapeUtil::Compatible(instruction->shape(), copies_added->shape()));
  }

  auto should_skip_user = [&](const HloInstruction* u) {
    return skip_users != nullptr && skip_users->contains(u);
  };

  std::vector<HloInstruction*> initial_users;
  if (users_override.has_value()) {
    for (HloInstruction* u : *users_override) {
      if (!should_skip_user(u)) {
        initial_users.push_back(u);
      }
    }
  } else {
    for (HloInstruction* u : instruction->users()) {
      if (!should_skip_user(u)) {
        initial_users.push_back(u);
      }
    }
  }

  auto record_single_tree = [&](const ShapeTree<HloInstruction*>& single_tree) {
    if (copies_added == nullptr) {
      return;
    }
    single_tree.ForEachElement(
        [&](const ShapeIndex& idx, HloInstruction* copy) {
          if (copy != nullptr) {
            copies_added->mutable_element(idx)->push_back(copy);
          }
        });
  };

  if (classify_passthrough == nullptr) {
    const bool update_root = replace_root && instruction->IsRoot();
    ShapeTree<HloInstruction*> single_tree(instruction->shape(), nullptr);
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * deep_copy,
        computation->DeepCopyInstruction(
            instruction, &indices_to_copy,
            copies_added != nullptr ? &single_tree : nullptr));
    record_single_tree(single_tree);
    for (HloInstruction* user : initial_users) {
      ABSL_RETURN_IF_ERROR(
          instruction->ReplaceUseWithDifferentShape(user, deep_copy));
    }
    if (update_root) {
      computation->set_root_instruction(deep_copy);
    }
    return absl::OkStatus();
  }

  // Array producer path: keep every attached kProducer passthrough on its
  // existing operand (preventing cycles) and place a kCopy immediately after
  // each owner (`instruction` or attached kProducer) that has downstream uses
  // or is the computation root. Also separate mixed `kConsumer -> kProducer`
  // chains so `kProducer` stays before the kCopy and `kConsumer` stays after.
  if (instruction->shape().IsArray() && indices_to_copy.element({})) {
    std::vector<HloInstruction*> owners = {instruction};
    absl::flat_hash_set<HloInstruction*> visited_owners = {instruction};
    absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
        owner_uses;

    auto process_user_of_owner = [&](HloInstruction* owner,
                                     HloInstruction* user) -> absl::Status {
      if (should_skip_user(user)) {
        return absl::OkStatus();
      }
      auto att = ClassifyValidatedPassthrough(user, classify_passthrough);
      if (att == CopyInsertion::PassthroughAttachment::kProducer) {
        if (visited_owners.insert(user).second) {
          owners.push_back(user);
        }
        return absl::OkStatus();
      }
      if (att == CopyInsertion::PassthroughAttachment::kConsumer) {
        // Check if any kProducer passthrough sits downstream of `user`; if so,
        // rebuild each such path as `owner -> m_out_clone -> copy ->
        // m_in_clone` via DeepCopyConsumerOperand on the outer passthrough.
        std::vector<HloInstruction*> stack = {user};
        absl::flat_hash_set<HloInstruction*> seen = {user};
        bool has_downstream_producer_marker = false;
        while (!stack.empty()) {
          HloInstruction* curr = stack.back();
          stack.pop_back();
          for (HloInstruction* sub : curr->users()) {
            auto sub_att =
                ClassifyValidatedPassthrough(sub, classify_passthrough);
            if (sub_att == CopyInsertion::PassthroughAttachment::kProducer) {
              has_downstream_producer_marker = true;
              break;
            }
            if (sub_att == CopyInsertion::PassthroughAttachment::kConsumer &&
                seen.insert(sub).second) {
              stack.push_back(sub);
            }
          }
        }
        if (has_downstream_producer_marker) {
          // Walk to the maximal passthrough tails reachable from `user` and
          // rebuild each tail for its non passthrough consumers.
          std::vector<HloInstruction*> tails = {user};
          absl::flat_hash_set<HloInstruction*> visited_tails = {user};
          for (size_t ti = 0; ti < tails.size(); ++ti) {
            HloInstruction* t = tails[ti];
            std::vector<HloInstruction*> non_pass_users;
            for (HloInstruction* tu : t->users()) {
              if (should_skip_user(tu)) {
                continue;
              }
              if (ClassifyValidatedPassthrough(tu, classify_passthrough) !=
                  CopyInsertion::PassthroughAttachment::kNone) {
                if (visited_tails.insert(tu).second) {
                  tails.push_back(tu);
                }
              } else {
                non_pass_users.push_back(tu);
              }
            }
            const bool tail_is_root = replace_root && t->IsRoot();
            if (!non_pass_users.empty() || tail_is_root) {
              HloInstruction* recorded = nullptr;
              ABSL_ASSIGN_OR_RETURN(
                  HloInstruction * rebuilt,
                  CopyArrayPreservingPassthroughs(
                      t, t->shape(), /*target_user=*/nullptr,
                      /*is_entry_root_copy=*/tail_is_root &&
                          non_pass_users.empty() &&
                          computation->IsEntryComputation(),
                      /*allow_in_place_rewire=*/false, computation, &recorded,
                      classify_passthrough));
              if (copies_added != nullptr && recorded != nullptr) {
                copies_added->mutable_element({})->push_back(recorded);
              }
              for (HloInstruction* nu : non_pass_users) {
                ABSL_RETURN_IF_ERROR(
                    t->ReplaceUseWithDifferentShape(nu, rebuilt));
              }
              if (tail_is_root) {
                computation->set_root_instruction(rebuilt);
              }
            }
          }
          return absl::OkStatus();
        }
      }
      owner_uses[owner].push_back(user);
      return absl::OkStatus();
    };

    for (HloInstruction* u : initial_users) {
      ABSL_RETURN_IF_ERROR(process_user_of_owner(instruction, u));
    }
    for (size_t idx = 1; idx < owners.size(); ++idx) {
      HloInstruction* marker = owners[idx];
      std::vector<HloInstruction*> m_users = marker->users();
      for (HloInstruction* u : m_users) {
        ABSL_RETURN_IF_ERROR(process_user_of_owner(marker, u));
      }
    }

    // When `instruction` has both attached kProducer markers and direct users,
    // route any direct user that cannot reach `primary_marker` through
    // `primary_marker`'s copy so a single kCopy serves both without introducing
    // cycles.
    std::vector<HloInstruction*> shared_direct_users;
    if (owners.size() > 1) {
      auto direct_it = owner_uses.find(instruction);
      if (direct_it != owner_uses.end() && !direct_it->second.empty()) {
        HloInstruction* primary_marker = owners.back();
        std::unique_ptr<HloReachabilityMap> reachability =
            HloReachabilityMap::Build(computation);
        std::vector<HloInstruction*> remaining_direct_users;
        for (HloInstruction* u : direct_it->second) {
          if (!reachability->IsReachable(u, primary_marker)) {
            shared_direct_users.push_back(u);
          } else {
            remaining_direct_users.push_back(u);
          }
        }
        direct_it->second = std::move(remaining_direct_users);
      }
    }

    bool recorded_any_copy = false;
    for (HloInstruction* owner : owners) {
      const auto it = owner_uses.find(owner);
      const bool has_uses = it != owner_uses.end() && !it->second.empty();
      const bool is_primary = owners.size() > 1 && owner == owners.back() &&
                              !shared_direct_users.empty();
      const bool owner_is_root = replace_root && owner->IsRoot();
      if (!has_uses && !is_primary && !owner_is_root) {
        continue;
      }
      HloInstruction* copy = computation->AddInstruction(
          HloInstruction::CreateUnary(owner->shape(), HloOpcode::kCopy, owner));
      if (copies_added != nullptr) {
        copies_added->mutable_element({})->push_back(copy);
      }
      recorded_any_copy = true;
      if (has_uses) {
        for (HloInstruction* user : it->second) {
          ABSL_RETURN_IF_ERROR(owner->ReplaceUseWithDifferentShape(user, copy));
        }
      }
      if (is_primary) {
        for (HloInstruction* user : shared_direct_users) {
          ABSL_RETURN_IF_ERROR(
              instruction->ReplaceUseWithDifferentShape(user, copy));
        }
      }
      if (owner_is_root) {
        computation->set_root_instruction(copy);
      }
    }
    if (!recorded_any_copy && copies_added != nullptr) {
      HloInstruction* copy =
          computation->AddInstruction(HloInstruction::CreateUnary(
              instruction->shape(), HloOpcode::kCopy, instruction));
      copies_added->mutable_element({})->push_back(copy);
    }
    return absl::OkStatus();
  }

  // Tuple producer path: preserve both tuple shaped kProducer passthroughs on
  // `instruction` and array kProducer passthroughs attached to GTEs of
  // `instruction`.
  if (instruction->shape().IsTuple()) {
    std::vector<HloInstruction*> tuple_producer_markers;
    std::vector<HloInstruction*> gtes_with_producer_markers;
    std::vector<HloInstruction*> direct_users;
    absl::flat_hash_set<HloInstruction*> visited_tuple_markers;

    for (HloInstruction* u : initial_users) {
      if (ClassifyValidatedPassthrough(u, classify_passthrough) ==
          CopyInsertion::PassthroughAttachment::kProducer) {
        if (visited_tuple_markers.insert(u).second) {
          tuple_producer_markers.push_back(u);
        }
        continue;
      }
      if (u->opcode() == HloOpcode::kGetTupleElement && u->shape().IsArray() &&
          indices_to_copy.element({u->tuple_index()}) &&
          absl::c_any_of(u->users(), [&](const HloInstruction* gu) {
            return ClassifyValidatedPassthrough(gu, classify_passthrough) ==
                   CopyInsertion::PassthroughAttachment::kProducer;
          })) {
        gtes_with_producer_markers.push_back(u);
        continue;
      }
      direct_users.push_back(u);
    }

    if (!tuple_producer_markers.empty() ||
        !gtes_with_producer_markers.empty()) {
      for (size_t i = 0; i < tuple_producer_markers.size(); ++i) {
        HloInstruction* marker = tuple_producer_markers[i];
        std::vector<HloInstruction*> m_direct_users;
        for (HloInstruction* mu : marker->users()) {
          if (should_skip_user(mu)) {
            continue;
          }
          if (ClassifyValidatedPassthrough(mu, classify_passthrough) ==
              CopyInsertion::PassthroughAttachment::kProducer) {
            if (visited_tuple_markers.insert(mu).second) {
              tuple_producer_markers.push_back(mu);
            }
          } else {
            m_direct_users.push_back(mu);
          }
        }
        const bool marker_is_root = replace_root && marker->IsRoot();
        if (!m_direct_users.empty() || marker_is_root) {
          ShapeTree<HloInstruction*> single_tree(marker->shape(), nullptr);
          ABSL_ASSIGN_OR_RETURN(
              HloInstruction * marker_copy,
              computation->DeepCopyInstruction(
                  marker, &indices_to_copy,
                  copies_added != nullptr ? &single_tree : nullptr));
          record_single_tree(single_tree);
          for (HloInstruction* mu : m_direct_users) {
            ABSL_RETURN_IF_ERROR(
                marker->ReplaceUseWithDifferentShape(mu, marker_copy));
          }
          if (marker_is_root) {
            computation->set_root_instruction(marker_copy);
          }
        }
      }

      for (HloInstruction* gte : gtes_with_producer_markers) {
        ShapeTree<bool> leaf_indices(gte->shape(), true);
        CopyListTree leaf_copies(gte->shape());
        ABSL_RETURN_IF_ERROR(InsertProducerCopyImpl(
            gte, leaf_indices, /*users_override=*/std::nullopt, replace_root,
            skip_users, copies_added != nullptr ? &leaf_copies : nullptr,
            classify_passthrough));
        if (copies_added != nullptr) {
          auto& dst = *copies_added->mutable_element({gte->tuple_index()});
          const auto& src = leaf_copies.element({});
          dst.insert(dst.end(), src.begin(), src.end());
        }
      }

      const bool update_root = replace_root && instruction->IsRoot();
      if (!direct_users.empty() || update_root ||
          (copies_added != nullptr && tuple_producer_markers.empty())) {
        ShapeTree<HloInstruction*> single_tree(instruction->shape(), nullptr);
        ABSL_ASSIGN_OR_RETURN(
            HloInstruction * deep_copy,
            computation->DeepCopyInstruction(
                instruction, &indices_to_copy,
                copies_added != nullptr ? &single_tree : nullptr));
        record_single_tree(single_tree);
        for (HloInstruction* u : direct_users) {
          ABSL_RETURN_IF_ERROR(
              instruction->ReplaceUseWithDifferentShape(u, deep_copy));
        }
        if (update_root) {
          computation->set_root_instruction(deep_copy);
        }
      }
      return absl::OkStatus();
    }
  }

  const bool update_root = replace_root && instruction->IsRoot();
  ShapeTree<HloInstruction*> single_tree(instruction->shape(), nullptr);
  ABSL_ASSIGN_OR_RETURN(HloInstruction * deep_copy,
                        computation->DeepCopyInstruction(
                            instruction, &indices_to_copy,
                            copies_added != nullptr ? &single_tree : nullptr));
  record_single_tree(single_tree);
  for (HloInstruction* user : initial_users) {
    ABSL_RETURN_IF_ERROR(
        instruction->ReplaceUseWithDifferentShape(user, deep_copy));
  }
  if (update_root) {
    computation->set_root_instruction(deep_copy);
  }
  return absl::OkStatus();
}

// Adds control dependencies from every kCopy in `from_copy_tree` at each leaf
// index to the matching kCopy in `to_copy_tree`.
//
// REQUIRES: `ShapeUtil::Compatible(from_copy_tree.shape(),
// to_copy_tree.shape())`, and for each index `from_copy_tree.element(index)` is
// non empty iff `to_copy_tree.element(index) != nullptr`.
absl::Status AddControlEdgesBetweenCopyTrees(
    const CopyListTree& from_copy_tree,
    const ShapeTree<HloInstruction*>& to_copy_tree) {
  TF_RET_CHECK(
      ShapeUtil::Compatible(from_copy_tree.shape(), to_copy_tree.shape()));
  for (const auto& pair : from_copy_tree) {
    const ShapeIndex& index = pair.first;
    const auto& from_copies = pair.second;
    HloInstruction* to_copy = to_copy_tree.element(index);
    if (from_copies.empty()) {
      TF_RET_CHECK(to_copy == nullptr);
      continue;
    }
    TF_RET_CHECK(to_copy != nullptr);
    for (HloInstruction* from_copy : from_copies) {
      TF_RET_CHECK(from_copy != nullptr);
      ABSL_RETURN_IF_ERROR(from_copy->AddControlDependencyTo(to_copy));
    }
  }
  return absl::OkStatus();
}

// Returns whether the instruction is an asynchronous operation that produces
// implicit non-copyable values. The whole result of the instruction is
// considered as non-copyable for the purpose of copy insertion.
//
// TODO(bixia): May consider migrating these ops to use explicit buffer types.
bool IsImplicitNonCopyable(const HloInstruction* instruction) {
  // Currently, the verifier only allows the pipelining of Send/Recv. As such,
  // here we only handle the ops allowed by
  // HloDataflowAnalysis::IsAsynchronousOperationStart that pass through its
  // operand for now. For the ops that don't pass through its operand, we need
  // to add a copy of its operand for the straight line case in order to allow
  // all ops in HloDataflowAnalysis::IsAsynchronousOperationStart.
  //
  // TODO(bixia): Handle ops that may modify its operand and add kCopyStart
  // here.
  HloOpcode opcode = instruction->opcode();
  return opcode == HloOpcode::kSend || opcode == HloOpcode::kRecv ||
         opcode == HloOpcode::kCopyStart;
}

// Returns whether the instruction produces explicit buffer values or uses
// explicit buffer values.
bool IsExplicitNonCopyable(const HloInstruction* instruction) {
  auto has_buffer = [](const Shape& shape) {
    bool has_buffer = false;
    ShapeUtil::ForEachSubshape(
        shape, [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.IsBuffer()) {
            has_buffer = true;
          }
        });
    return has_buffer;
  };
  if (has_buffer(instruction->shape())) {
    return true;
  }
  for (auto* operand : instruction->operands()) {
    if (has_buffer(operand->shape())) {
      return true;
    }
  }
  return false;
}

// Returns true if the instruction produces non-copyable results.
bool IsNonCopyable(const HloInstruction* instruction) {
  return IsImplicitNonCopyable(instruction) ||
         IsExplicitNonCopyable(instruction);
}

// Returns true if the value at the given index in the while init is a
// non-copyable value produced by and implicit or explicit non-copyable
// instruction.
bool IsNonCopyableInWhileInit(const HloInstruction* while_init,
                              const ShapeIndex& index) {
  // Handle explicit non-copyable values, that is, explicit buffers.
  bool non_copyable = false;
  ShapeUtil::ForEachSubshape(
      ShapeUtil::GetSubshape(while_init->shape(), index),
      [&](const Shape& subshape, const ShapeIndex& /*index*/) {
        if (subshape.IsBuffer()) {
          non_copyable = true;
        }
      });
  if (non_copyable) {
    return true;
  }

  if (index.empty()) {
    return false;
  }
  int64_t i = index.front();
  return i < while_init->operand_count() &&
         IsImplicitNonCopyable(while_init->operand(i));
}

// Compute the indices of the loop state which need copies in order to avoid
// live range interference. Generally, an element in the loop state does not
// need to be copied if the element is passed through transparently through the
// body.
//
// Returns whether any indices need to be copied.
bool IndicesToCopyForWhile(const HloDataflowAnalysis& dataflow,
                           const HloInstruction* xla_while,
                           ShapeTree<bool>* indices_to_copy) {
  DCHECK(ShapeUtil::Compatible(indices_to_copy->shape(), xla_while->shape()));

  bool any_copies = false;
  const HloInstruction* init = xla_while->operand(0);
  for (auto& pair : *indices_to_copy) {
    const ShapeIndex& index = pair.first;
    bool& should_copy = pair.second;
    if (IsNonCopyableInWhileInit(init, index)) {
      // Do not copy non-copyable values, instead, we will add copies for
      // transitioning into and out of non-copyable values.
      should_copy = false;
      continue;
    }
    if (dataflow.GetValueSet(init, index).values().size() > 1 ||
        dataflow.GetValueSet(xla_while, index).values().size() > 1) {
      // If there is any ambiguity, then loop state must be copied.
      should_copy = true;
    } else {
      // If the output of the while instruction is not the same as the init
      // value of the while, then this element is not passed through the body
      // transparently and must be copied.
      should_copy = dataflow.GetUniqueValueAt(xla_while, index) !=
                    dataflow.GetUniqueValueAt(init, index);
    }
    any_copies |= should_copy;
  }
  return any_copies;
}

// Compute the indices of the conditional outputs which need copies. Umambiguous
// buffers(buffer with only one value) don't need copies.
bool IndicesToCopyForConditional(const HloDataflowAnalysis& dataflow,
                                 const HloInstruction* xla_conditional,
                                 ShapeTree<bool>* indices_to_copy) {
  DCHECK(ShapeUtil::Compatible(indices_to_copy->shape(),
                               xla_conditional->shape()));

  bool any_copies = false;
  for (auto& pair : *indices_to_copy) {
    const ShapeIndex& index = pair.first;
    bool& should_copy = pair.second;

    CHECK_EQ(dataflow.GetValueSet(xla_conditional, index).values().size(), 1);

    auto value = dataflow.GetValueSet(xla_conditional, index).values()[0];
    // The conditional must be copied if the value is a phi.
    should_copy =
        value->is_phi() && value->defining_instruction() == xla_conditional;
    any_copies |= should_copy;
  }
  return any_copies;
}

// Add kCopy instructions around the given kWhile instruction to eliminate any
// possible live range interference of HLO values assuming a dependency-based
// ordering. Copies are added conservatively. There  likely are copies which are
// not strictly necessary, but they are removed later in the pass via
// RemoveUnnecessaryCopies.
//
// Elements (each ShapeIndex) in the loop state are considered independently.  A
// copy is added to each element of the loop state which is modified in the
// while body. For each such element, a total of three kCopy instructions are
// added at following locations:
//
//   (1) The init value is copied before the kWhile instruction. Before:
//
//           (Init)
//             |
//           kWhile
//             |
//            ...
//
//       After:
//
//           (Init)
//             |
//           kCopy
//             |
//           kWhile
//             |
//            ...
//
//       This copy is necessary in case the init value is simultaneously live
//       with the kWhile.
//
//   (2) Copies are added to the parameter and root of the while body
//       computation. Before:
//
//           kParameter
//               |
//              ...
//               |
//           (body root)
//
//       After:
//
//           kParameter
//               |
//             kCopy ----------+
//               |             |
//              ...           ctrl
//               |            edge
//           (body root)       |
//               |             |
//             kCopy <---------+
//
//       The root kCopy becomes the new root of the computation. Both copies are
//       necessary to any potential interference between the parameter value and
//       the root value. The control edge prevents potential interference
//       between the copies themselves.
//
// If the loop state is a tuple then the above kCopy instructions are a deep
// copy constructed of kCopy, kGetTupleElement, and kTuple instruction as
// constructed by HloInstruction::DeepCopyInstruction.
bool IsAsyncOperationStateDefinition(const HloInstruction* instruction) {
  return HloDataflowAnalysis::IsAsynchronousOperationStart(
             instruction->opcode()) ||
         instruction->opcode() == HloOpcode::kAsyncUpdate;
}

// Returns true if `index` of `while_instr` carries an async pipelined state
// across loop iterations when `xla_disable_while_loop_copies` is set.
//
// REQUIRES: `while_instr` is not null.
bool IsAsyncPipelinedWhileTupleIndex(
    const HloInstruction* while_instr, const ShapeIndex& index,
    const CopyInsertion::PassthroughClassifier& classify_passthrough) {
  CHECK_NE(while_instr, nullptr);
  if (!frontend::HasDisableWhileLoopCopiesAttr(while_instr) || index.empty()) {
    return false;
  }
  auto is_passthrough = [&](const HloInstruction* inst) {
    return ClassifyValidatedPassthrough(inst, classify_passthrough) !=
           CopyInsertion::PassthroughAttachment::kNone;
  };
  auto unwrap_wrappers = [&](const HloInstruction* inst) {
    while (inst->opcode() == HloOpcode::kBitcast || is_passthrough(inst)) {
      inst = inst->operand(0);
    }
    return inst;
  };
  // Walk through passthrough wrappers and nested GTEs from a loop state GTE to
  // check whether it feeds an async completion or update op.
  auto has_async_done_or_update_user = [&](const HloInstruction* gte) {
    absl::flat_hash_set<const HloInstruction*> visited = {gte};
    std::vector<const HloInstruction*> worklist = {gte};
    while (!worklist.empty()) {
      const HloInstruction* curr = worklist.back();
      worklist.pop_back();
      for (const HloInstruction* u : curr->users()) {
        if (is_passthrough(u) || u->opcode() == HloOpcode::kGetTupleElement) {
          if (visited.insert(u).second) {
            worklist.push_back(u);
          }
          continue;
        }
        if (HloDataflowAnalysis::IsAsynchronousOperationDone(u->opcode()) ||
            u->opcode() == HloOpcode::kAsyncUpdate) {
          return true;
        }
      }
    }
    return false;
  };
  int64_t tuple_idx = index[0];

  // 1. Check while init operand
  const HloInstruction* while_init = while_instr->operand(0);
  if (while_init->opcode() == HloOpcode::kTuple &&
      tuple_idx < while_init->operand_count()) {
    const HloInstruction* init_elem =
        unwrap_wrappers(while_init->operand(tuple_idx));
    if (IsAsyncOperationStateDefinition(init_elem)) {
      return true;
    }
  }

  // 2. Check while body
  HloComputation* while_body = while_instr->while_body();
  if (while_body != nullptr) {
    // Check root instruction of while body
    const HloInstruction* root = while_body->root_instruction();
    if (root != nullptr && root->opcode() == HloOpcode::kTuple &&
        tuple_idx < root->operand_count()) {
      const HloInstruction* root_elem =
          unwrap_wrappers(root->operand(tuple_idx));
      if (IsAsyncOperationStateDefinition(root_elem)) {
        return true;
      }
    }

    // Check while body parameter users
    if (while_body->num_parameters() > 0) {
      const HloInstruction* while_param = while_body->parameter_instruction(0);
      for (const HloInstruction* user : while_param->users()) {
        if (user->opcode() == HloOpcode::kGetTupleElement &&
            user->tuple_index() == tuple_idx &&
            has_async_done_or_update_user(user)) {
          return true;
        }
      }
    }
  }

  // 3. Check while instruction users
  for (const HloInstruction* user : while_instr->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement &&
        user->tuple_index() == tuple_idx &&
        has_async_done_or_update_user(user)) {
      return true;
    }
  }

  return false;
}

bool IsDisjointInPlaceWhileTupleIndex(const HloInstruction* while_instr,
                                      const ShapeIndex& index) {
  if (!frontend::HasDisableWhileLoopCopiesAttr(while_instr) || index.empty()) {
    return false;
  }
  int64_t tuple_idx = index[0];
  const HloComputation* while_body = while_instr->while_body();
  if (while_body == nullptr || while_body->num_parameters() == 0) {
    return false;
  }
  const HloInstruction* while_param = while_body->parameter_instruction(0);

  const HloInstruction* param_gte = nullptr;
  for (const HloInstruction* user : while_param->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement &&
        user->tuple_index() == tuple_idx) {
      param_gte = user;
      break;
    }
  }
  if (param_gte == nullptr) {
    return false;
  }

  auto is_transparent_wrapper = [](const HloInstruction* instr) {
    return instr->opcode() == HloOpcode::kBitcast ||
           instr->opcode() == HloOpcode::kBitcastConvert ||
           instr->opcode() == HloOpcode::kReshape ||
           instr->opcode() == HloOpcode::kCopy ||
           (instr->opcode() == HloOpcode::kCustomCall &&
            instr->operand_count() == 1);
  };

  // Ensure param_gte has no competing readers inside the loop body unless the
  // loop is explicitly annotated as disjoint.
  const HloInstruction* single_reader = param_gte;
  while (single_reader->user_count() == 1 &&
         is_transparent_wrapper(single_reader->users()[0])) {
    single_reader = single_reader->users()[0];
  }
  if (single_reader->user_count() != 1 &&
      !frontend::HasDisjointReadWriteRegionsAttr(while_instr)) {
    return false;
  }

  const HloInstruction* root = while_body->root_instruction();
  if (root == nullptr || root->opcode() != HloOpcode::kTuple ||
      tuple_idx >= root->operand_count()) {
    return false;
  }
  const HloInstruction* root_elem = root->operand(tuple_idx);
  while (is_transparent_wrapper(root_elem)) {
    root_elem = root_elem->operand(0);
  }

  const HloInstruction* curr = root_elem;
  while (curr != nullptr && curr != single_reader && curr != param_gte) {
    const HloInstruction* next = nullptr;
    if (curr->opcode() == HloOpcode::kDynamicUpdateSlice) {
      if (!frontend::HasDisjointReadWriteRegionsAttr(curr) &&
          !frontend::HasDisjointReadWriteRegionsAttr(while_instr)) {
        return false;
      }
      next = curr->operand(0);
    } else if (curr->IsAsynchronous() &&
               curr->async_wrapped_opcode() == HloOpcode::kDynamicUpdateSlice) {
      const HloInstruction* async_start = nullptr;
      if (curr->opcode() == HloOpcode::kAsyncDone) {
        async_start = curr->operand(0);
      } else if (curr->opcode() == HloOpcode::kAsyncStart) {
        async_start = curr;
      }
      if (async_start == nullptr || async_start->operand_count() == 0) {
        return false;
      }
      if (!frontend::HasDisjointReadWriteRegionsAttr(curr) &&
          !frontend::HasDisjointReadWriteRegionsAttr(async_start) &&
          !frontend::HasDisjointReadWriteRegionsAttr(while_instr)) {
        return false;
      }
      next = async_start->operand(0);
    } else if (is_transparent_wrapper(curr)) {
      next = curr->operand(0);
    } else {
      return false;
    }
    while (next != nullptr && is_transparent_wrapper(next)) {
      next = next->operand(0);
    }
    curr = next;
  }
  return curr == single_reader || curr == param_gte;
}

// Only such a while makes AddCopiesForWhile read HloBuffers.
// LINT.IfChange(disabled_copies_prescan)
bool HasWhileWithDisabledCopies(const HloModule& module) {
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kWhile &&
          frontend::HasDisableWhileLoopCopiesAttr(instruction)) {
        return true;
      }
    }
  }
  return false;
}
// LINT.ThenChange(:disabled_copies_buffer_reads)

// `alias_analysis` is only consulted for a while annotated with
// xla_disable_while_loop_copies and may be null otherwise.
absl::Status AddCopiesForWhile(
    const HloDataflowAnalysis& dataflow, const HloAliasAnalysis* alias_analysis,
    HloInstruction* xla_while,
    const CopyInsertion::PassthroughClassifier& classify_passthrough) {
  VLOG(2) << "Adding copies for kWhile instruction " << xla_while->name();
  TF_RET_CHECK(xla_while->opcode() == HloOpcode::kWhile);

  ShapeTree<bool> indices_to_copy(xla_while->shape());
  IndicesToCopyForWhile(dataflow, xla_while, &indices_to_copy);

  // LINT.IfChange(disabled_copies_buffer_reads)
  if (frontend::HasDisableWhileLoopCopiesAttr(xla_while)) {
    VLOG(2) << "While loop copy insertion partially disabled via frontend "
               "attribute for "
            << xla_while->name();
    TF_RET_CHECK(alias_analysis != nullptr);
    // LINT.ThenChange(:disabled_copies_prescan)
    // Clear copy requirement for scalars, async pipelined, and disjoint
    // in-place tuple indices.
    for (auto& pair : indices_to_copy) {
      const Shape& subshape =
          ShapeUtil::GetSubshape(xla_while->shape(), pair.first);
      if (ShapeUtil::IsScalar(subshape) ||
          IsAsyncPipelinedWhileTupleIndex(xla_while, pair.first,
                                          classify_passthrough) ||
          IsDisjointInPlaceWhileTupleIndex(xla_while, pair.first)) {
        pair.second = false;
      }
    }
    // Deep copy any duplicate buffers in while_init outside the loop.
    HloInstruction* while_init = xla_while->mutable_operand(0);
    if (while_init->shape().IsTuple()) {
      ShapeTree<bool> init_indices_to_copy(while_init->shape(), false);
      absl::flat_hash_set<const HloBuffer*> seen_buffers;
      bool need_copy = false;
      ShapeUtil::ForEachSubshape(
          while_init->shape(),
          [&](const Shape& subshape, const ShapeIndex& index) {
            if (!subshape.IsArray() || index.empty() ||
                IsAsyncPipelinedWhileTupleIndex(xla_while, {index[0]},
                                                classify_passthrough)) {
              return;
            }
            std::vector<const HloBuffer*> buffers =
                alias_analysis->ComputeBuffersAt(while_init, index);
            for (const HloBuffer* buffer : buffers) {
              if (!seen_buffers.insert(buffer).second) {
                *init_indices_to_copy.mutable_element(index) = true;
                need_copy = true;
                break;
              }
            }
          });
      if (need_copy) {
        ABSL_ASSIGN_OR_RETURN(HloInstruction * while_init_copy,
                              DeepCopyConsumerOperand(while_init, xla_while,
                                                      &init_indices_to_copy,
                                                      /*copies_added=*/nullptr,
                                                      classify_passthrough));
        if (while_init_copy != while_init) {
          ABSL_RETURN_IF_ERROR(
              while_init->ReplaceUseWith(xla_while, while_init_copy));
        }
      }
    }
  }

  bool any_copies = false;
  for (const auto& pair : indices_to_copy) {
    if (pair.second) {
      any_copies = true;
      break;
    }
  }
  if (!any_copies) {
    VLOG(2) << "No copies necessary for kWhile instruction "
            << xla_while->name();
    return absl::OkStatus();
  }

  VLOG(2) << "Adding copies for " << xla_while->name() << " at indices:";
  for (auto& pair : indices_to_copy) {
    if (pair.second) {
      VLOG(2) << "  " << pair.first;
    }
  }

  // Deep copy init, placing leaf copies before any kConsumer passthroughs.
  HloInstruction* while_init = xla_while->mutable_operand(0);
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * while_init_copy,
      DeepCopyConsumerOperand(while_init, xla_while, &indices_to_copy,
                              /*copies_added=*/nullptr, classify_passthrough));
  if (while_init_copy != while_init) {
    ABSL_RETURN_IF_ERROR(
        while_init->ReplaceUseWith(xla_while, while_init_copy));
  }

  // Deep copy the parameter and the root. Extend a control edge from the copy
  // of the parameter value to the corresponding copy value of the root.
  HloComputation* body = xla_while->while_body();
  HloInstruction* param = body->parameter_instruction(0);
  HloInstruction* root = body->root_instruction();

  // If param is the root then all indices should have been passed through the
  // while body and we should have returned early above.
  TF_RET_CHECK(param != root);

  // Copy users before making a deep copy of the parameter as the deep copy
  // will create new users of the parameter (eg, the GTE instructions of the
  // deep copy).
  std::vector<HloInstruction*> param_users = param->users();
  CopyListTree from_copy_tree(param->shape());
  ABSL_RETURN_IF_ERROR(InsertProducerCopyImpl(
      param, indices_to_copy, param_users, /*replace_root=*/false,
      /*skip_users=*/nullptr, &from_copy_tree, classify_passthrough));

  root = body->root_instruction();
  ShapeTree<HloInstruction*> to_copy_tree(root->shape(),
                                          /*init_value=*/nullptr);
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * root_copy,
      DeepCopyConsumerOperand(root, /*target_user=*/nullptr, &indices_to_copy,
                              &to_copy_tree, classify_passthrough));
  ABSL_RETURN_IF_ERROR(
      AddControlEdgesBetweenCopyTrees(from_copy_tree, to_copy_tree));
  body->set_root_instruction(root_copy);
  return absl::OkStatus();
}

// Add copies for the operands of in-place operations. RemoveUnnecessaryCopies
// will remove the unnecessary copies.
absl::Status AddCopiesForInPlaceOperation(
    HloInstruction* in_place_op, int64_t operand_number,
    const CopyInsertion::PassthroughClassifier& classify_passthrough) {
  VLOG(2) << "Adding copies for in-place operation " << in_place_op->name();
  HloInstruction* operand = in_place_op->mutable_operand(operand_number);
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * deep_copy,
      DeepCopyConsumerOperand(operand, in_place_op, /*indices_to_copy=*/nullptr,
                              /*copies_added=*/nullptr, classify_passthrough));
  if (deep_copy != operand) {
    ABSL_RETURN_IF_ERROR(
        operand->ReplaceUseWith(in_place_op, operand_number, deep_copy));
  }
  return absl::OkStatus();
}

// Orders the other readers of the operand of `copy` before `copy`, the copy
// added for an operand of `in_place_op`, if they are already predecessors of
// `in_place_op`. Such readers must run before `in_place_op` anyway, so the new
// control dependencies do not constrain the schedule beyond what `in_place_op`
// already requires, but they let RemoveUnnecessaryCopies prove that `copy` is
// redundant. Without them, chains of "read, then update in place" steps on the
// same buffer (e.g. gather + scatter updates of an embedding table across
// unrolled training steps) keep every copy, since eliding each copy requires
// the reader of the next step to be ordered before the next copy.
//
// Readers that are copies are skipped: removing either of two copies of the
// same operand feeding in-place ops would make both ops write the same buffer.
// `reachability` may be stale; it is only used to find paths that already
// exist, and instructions missing from it are skipped.
absl::Status OrderPredecessorReadersBeforeCopy(
    HloInstruction* copy, const HloInstruction* in_place_op,
    const HloReachabilityMap& reachability) {
  for (HloInstruction* reader : copy->operand(0)->users()) {
    // Asynchronous ops keep reading their operands until the matching done,
    // so ordering the start before `copy` would not order the read.
    if (reader != copy && reader != in_place_op &&
        reader->opcode() != HloOpcode::kCopy &&
        !HloDataflowAnalysis::IsAsynchronousOperationStart(reader->opcode()) &&
        reachability.IsPresent(reader) &&
        reachability.IsReachable(reader, in_place_op)) {
      VLOG(3) << "Ordering " << reader->name() << " before " << copy->name();
      ABSL_RETURN_IF_ERROR(reader->AddControlDependencyTo(copy));
    }
  }
  return absl::OkStatus();
}

// Conservatively adds copies before root instruction of entry computation and
// each aliased parameter to resolve interference of aliased input and output
// buffer. We later rely on RemoveUnnecessaryCopies to drop the unnecessary
// ones.
absl::Status AddCopiesForAliasedInputOutputs(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    const CopyInsertion::PassthroughClassifier& classify_passthrough) {
  HloComputation* entry = module->entry_computation();
  if (!HloInstruction::IsThreadIncluded(entry->execution_thread(),
                                        execution_threads)) {
    return absl::OkStatus();
  }
  HloInstruction* root = entry->root_instruction();

  ShapeTree<bool> output_indices_to_copy(root->shape());
  std::vector<std::optional<CopyListTree>> copied_parameters(
      entry->num_parameters());
  bool has_alias = false;
  for (auto* param : entry->parameter_instructions()) {
    bool param_has_alias = false;
    ShapeTree<bool> param_indices_to_copy(param->shape());

    module->input_output_alias_config().ForEachAlias(
        [&](const ShapeIndex& output_index,
            const HloInputOutputAliasConfig::Alias& alias) {
          if (alias.parameter_number == param->parameter_number()) {
            param_has_alias = true;
            *(param_indices_to_copy.mutable_element(alias.parameter_index)) =
                true;
            *(output_indices_to_copy.mutable_element(output_index)) = true;
          }
        });

    if (!param_has_alias) {
      continue;
    }

    TF_RET_CHECK(param->parameter_number() < entry->num_parameters());
    TF_RET_CHECK(!copied_parameters[param->parameter_number()]);

    has_alias = true;
    // Store a snapshot of users before copying, as copying introduces new users
    // of the instruction.
    std::vector<HloInstruction*> users = param->users();
    CopyListTree param_copy_tree(param->shape());
    ABSL_RETURN_IF_ERROR(InsertProducerCopyImpl(
        param, param_indices_to_copy, users, /*replace_root=*/true,
        /*skip_users=*/nullptr, &param_copy_tree, classify_passthrough));
    root = entry->root_instruction();

    copied_parameters[param->parameter_number()] = std::move(param_copy_tree);
  }

  if (!has_alias) {
    return absl::OkStatus();
  }

  // Add copies before root instruction.
  ShapeTree<HloInstruction*> output_copy_tree(root->shape(),
                                              /*init_value=*/nullptr);

  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * root_copied,
      DeepCopyConsumerOperand(root, /*target_user=*/nullptr,
                              &output_indices_to_copy, &output_copy_tree,
                              classify_passthrough));

  // Add control dependencies between the input/output copies.
  ABSL_RETURN_IF_ERROR(
      module->input_output_alias_config().ForEachAliasWithStatus(
          [&](const ShapeIndex& output_index,
              const HloInputOutputAliasConfig::Alias& alias) -> absl::Status {
            if (!copied_parameters[alias.parameter_number]) {
              return absl::OkStatus();
            }
            const auto& from_copies =
                copied_parameters[alias.parameter_number]->element(
                    alias.parameter_index);
            HloInstruction* to = output_copy_tree.element(output_index);

            TF_RET_CHECK(!from_copies.empty());
            TF_RET_CHECK(to != nullptr);
            for (HloInstruction* from : from_copies) {
              TF_RET_CHECK(from != nullptr);
              ABSL_RETURN_IF_ERROR(from->AddControlDependencyTo(to));
            }
            return absl::OkStatus();
          }));

  entry->set_root_instruction(root_copied);

  return absl::OkStatus();
}

// Removes any control dependencies to or from the given instruction.
absl::Status StripControlDependenciesFrom(HloInstruction* instruction) {
  while (!instruction->control_successors().empty()) {
    ABSL_RETURN_IF_ERROR(instruction->RemoveControlDependencyTo(
        instruction->control_successors().front()));
  }

  while (!instruction->control_predecessors().empty()) {
    ABSL_RETURN_IF_ERROR(
        instruction->control_predecessors().front()->RemoveControlDependencyTo(
            instruction));
  }

  return absl::OkStatus();
}

}  // namespace

// We add copies for all phi indices of the true and false computation
// roots, in order to resolve interference. We later rely on
// RemoveUnnecessaryCopies to drop the unnecessary ones.
absl::Status CopyInsertion::AddCopiesForConditional(
    const HloDataflowAnalysis& dataflow, HloInstruction* conditional) {
  VLOG(2) << "Adding copies for kConditional instruction "
          << conditional->name();
  ShapeTree<bool> indices_to_copy(conditional->shape());
  TF_RET_CHECK(conditional->opcode() == HloOpcode::kConditional);
  if (!IndicesToCopyForConditional(dataflow, conditional, &indices_to_copy)) {
    VLOG(2) << "No copies necessary for kConditional instruction "
            << conditional->name();
    return absl::OkStatus();
  }

  for (HloComputation* computation : conditional->branch_computations()) {
    HloInstruction* root = computation->root_instruction();
    std::vector<HloInstruction*> users = root->users();
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * deep_copy,
        DeepCopyConsumerOperand(root, /*target_user=*/nullptr, &indices_to_copy,
                                /*copies_added=*/nullptr));
    if (deep_copy != root) {
      for (HloInstruction* user : users) {
        ABSL_RETURN_IF_ERROR(root->ReplaceUseWith(user, deep_copy));
      }
      computation->set_root_instruction(deep_copy);
    }
  }
  return absl::OkStatus();
}

// Returns the output index corresponding to the non-copyable value for in the
// given operand index through output_operand_aliasing or std::nullopt if there
// isn't such an output index.
std::optional<int64_t> FindAliasingOutputIndexForInput(const HloInstruction* op,
                                                       int64_t operand_index) {
  for (const auto& [output_index, input] : op->output_operand_aliasing()) {
    if (input.first == operand_index) {
      if (output_index.empty()) {
        // Output is not a tuple.
        return 0;
      }
      // The verifier guarantees no nested non-copyable values.
      CHECK_EQ(output_index.size(), 1);
      return output_index.front();
    }
  }

  return std::nullopt;
}

// Returns the input index corresponding to the non-copyable value for in the
// given output index through output_operand_aliasing or std::nullopt if there
// isn't such an input index.
std::optional<int64_t> FindAliasingInputIndexForOutput(const HloInstruction* op,
                                                       int64_t output_index) {
  for (const auto& [output, input] : op->output_operand_aliasing()) {
    // The verifier guarantees no nested non-copyable values.
    CHECK(input.second.empty());
    if ((output.empty() && output_index == 0) ||
        (output.size() == 1 && output.front() == output_index)) {
      return input.first;
    }
  }

  return std::nullopt;
}

// Given an `op` with _xla_non_copyable_attribute, returns true if it is the
// start of the chain of explicit non-copyable values containing its operand
// in `operand_index`.
bool IsChainStart(const HloInstruction* op, int64_t operand_index) {
  if (op->IsCustomCall(kPinCustomCallTarget) ||
      op->IsCustomCall(kCreateBufferCustomCallTarget)) {
    return true;
  }

  // The producer of the operand.
  const HloInstruction* producer = op->operand(operand_index);
  int64_t output_index = 0;
  const HloInstruction* gte = nullptr;

  // There can be at most one GetTupleElement in the chain as we assume no
  // buffers nested in operands.
  if (producer->opcode() == HloOpcode::kGetTupleElement) {
    gte = producer;
    output_index = producer->tuple_index();
    producer = producer->operand(0);
  }

  // If the producer for the operand doesn't produces the non-copyable value in
  // question, `op` is the start of the chain.
  if (!IsExplicitNonCopyable(producer)) {
    return true;
  }

  if (producer->opcode() != HloOpcode::kCustomCall) {
    // This is an instruction, such as tuple or while-loop, that pass through
    // the non-copyable values.
    return false;
  }

  if (!FindAliasingInputIndexForOutput(producer, output_index).has_value()) {
    // The operand in question doesn't connect to a non-copyable value in its
    // producer, `op` is the start of the chain.
    return true;
  }

  if ((!gte && producer->user_count() == 1) ||
      (gte && gte->user_count() == 1)) {
    // Producer is in the same chain of `op` and is before `op`.
    return false;
  }

  return true;
}

// Finds the unique user for the non-copyable value in the given output index.
// Returns the unique user and the operand index in the user or std::nullopt if
// there isn't such a unique user.
std::optional<std::pair<HloInstruction*, int64_t>> FindUniqueUser(
    HloInstruction* op, int64_t output_index) {
  if (!op->shape().IsTuple()) {
    CHECK_EQ(output_index, 0) << " expect output_index to be 0 for non-tuple ";
    if (op->user_count() != 1) {
      return std::nullopt;
    }
    HloInstruction* unique_user = op->users().front();
    std::vector<int64_t> operand_indices = unique_user->operand_indices(op);
    if (operand_indices.size() != 1) {
      // The user uses op multiple times. We don't consider this as a unique
      // user for the purpose of finding non-copyable chain.
      return std::nullopt;
    }
    return std::make_pair(unique_user, operand_indices.front());
  }

  // The non-copyable op produces a tuple result. Look for a unique
  // GetTupleElement user for the output_index.
  HloInstruction* unique_gte = nullptr;
  for (HloInstruction* user : op->users()) {
    CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
    if (user->tuple_index() != output_index) {
      continue;
    }
    if (unique_gte) {
      // There are multiple GetTupleElement for the same output_index.
      return std::nullopt;
    }
    unique_gte = user;
  }

  if (!unique_gte || unique_gte->user_count() != 1) {
    return std::nullopt;
  }

  // Return the unique user of the GetTupleElement.
  return FindUniqueUser(unique_gte, 0);
}

// Walks down a chain of explicit non-copyable ops inside a while-body until we
// reach the end. Returns the last op in the chain and the corresponding output
// index in the last op. This routine is used to help recognizing a rotated
// non-copyable chain inside a while-body.
std::pair<HloInstruction*, int64_t> WalkDownNonCopyableChain(
    HloInstruction* op, int64_t output_index) {
  HloComputation* parent = op->parent();
  VLOG(2) << "WalkDownNonCopyableChain start: " << op->ToString()
          << " output_index " << output_index;
  std::optional<std::pair<HloInstruction*, int64_t>> unique_user_and_operand =
      FindUniqueUser(op, output_index);

  while (unique_user_and_operand.has_value()) {
    HloInstruction* unique_user = unique_user_and_operand.value().first;
    int64_t unique_user_operand_index = unique_user_and_operand.value().second;
    if (!IsExplicitNonCopyable(unique_user)) {
      break;
    }
    if (unique_user->opcode() == HloOpcode::kTuple &&
        parent->root_instruction() == unique_user) {
      // We reach the while-root, which is the end of the chain.
      return std::make_pair(unique_user, unique_user_operand_index);
    }
    std::optional<int64_t> unique_user_output_index =
        FindAliasingOutputIndexForInput(unique_user, unique_user_operand_index);
    if (!unique_user_output_index.has_value()) {
      // We reach the end of the chain, as unique_user is not in the same chain.
      break;
    }
    output_index = *unique_user_output_index;
    op = unique_user;
    unique_user_and_operand = FindUniqueUser(op, output_index);
  }

  VLOG(2) << "WalkDownNonCopyableChain end: " << op->ToString()
          << " output_index " << output_index;
  return std::make_pair(op, output_index);
}

// Walks up a chain of explicit non-copyable ops inside a while-body until we
// reach the start. Returns the first op in the chain and the corresponding
// output index in the first op. This routine is used to help recognizing a
// rotated non-copyable chain inside a while-body.
std::pair<HloInstruction*, int64_t> WalkUpNonCopyableChain(
    HloInstruction* op, int64_t operand_index) {
  VLOG(2) << "WalkUpNonCopyableChain start: " << op->ToString()
          << " operand_index " << operand_index;

  while (HloInstruction* producer = op->mutable_operand(operand_index)) {
    int64_t output_index = 0;
    if (producer->opcode() == HloOpcode::kGetTupleElement) {
      output_index = producer->tuple_index();
      producer = producer->mutable_operand(0);
    }
    if (!IsExplicitNonCopyable(producer)) {
      break;
    }
    std::optional<int64_t> producer_operand_index =
        FindAliasingInputIndexForOutput(producer, output_index);
    if (!producer_operand_index.has_value()) {
      break;
    }

    std::optional<std::pair<HloInstruction*, int64_t>> unique_user_and_operand =
        FindUniqueUser(producer, output_index);
    if (!unique_user_and_operand.has_value()) {
      break;
    }
    CHECK_EQ(unique_user_and_operand.value().first, op);
    op = producer;
    operand_index = *producer_operand_index;
  }

  VLOG(2) << "WalkUpNonCopyableChain end:" << op->ToString()
          << " operand_index " << operand_index;
  return std::make_pair(op, operand_index);
}

// The information of a rotated explicit non-copyable chain.
//
// Below is an example of a rotated chain, where the first part includes b1->b2
// and the second part includes b3.
//
// while_body {
//   param = (b(f32[16]), f32[16]) parameter(0)
//   b0 = b(f32[16]) get-tuple-element(param), index=0
//   v0 = f32[16] get-tuple-element(param), index=1
//   b1 = b(f32[16]) custom-call(v0,  b0), custom_call_target="up0"
//      output_to_operand_aliasing={{}:(1, {})}
//   b2 = f32[16] custom-call(b1), custom_call_target="unpin"
//   v1 = f32[16]add(v0, b2)
//   b3 = b(f32[16]) custom-call(v1), custom_call_target="pin"
//   ROOT tuple = (b(f32[16]), f32[16]) tuple(b3, v1)
// }
// We record the information for the end of the first part and the start of the
// second part of the chain. In this example, the recorded information is:
//     first_part_end = b2, output_index = 0
//     second_part_start = b1, operand_index = 0
struct RotatedChainInfo {
  HloInstruction* first_part_end;
  int64_t output_index;  // output index in the first part end op.
  HloInstruction* second_part_start;
};

// Given the start of an explicit non-copyable chain, tries to find the rotated
// chain, using the op as the first part start.
std::optional<RotatedChainInfo> FindRotatedChainInfo(
    const HloComputation* while_body, HloInstruction* second_part_start) {
  VLOG(2) << "FindRotatedChainInfo start: " << second_part_start->ToString();

  HloInstruction* second_part_end;
  int64_t second_part_end_output_index;
  std::tie(second_part_end, second_part_end_output_index) =
      WalkDownNonCopyableChain(second_part_start, /*output_index=*/0);

  if (second_part_end != while_body->root_instruction()) {
    return std::nullopt;
  }

  HloInstruction* first_part_start = nullptr;
  for (HloInstruction* user : while_body->parameter_instruction(0)->users()) {
    if (Match(user, match::GetTupleElement(match::Parameter(0),
                                           second_part_end_output_index))) {
      first_part_start = user;
      break;
    }
  }
  if (first_part_start == nullptr) {
    return std::nullopt;
  }
  VLOG(2) << "FindRotatedChainInfo first_part_start: "
          << first_part_start->ToString();

  HloInstruction* first_part_end;
  int64_t first_part_end_output_index;
  std::tie(first_part_end, first_part_end_output_index) =
      WalkDownNonCopyableChain(first_part_start, /*output_index=*/0);

  VLOG(2) << "FindRotatedChainInfo: " << first_part_end->ToString()
          << " output_index " << first_part_end_output_index
          << second_part_start->ToString();
  return RotatedChainInfo{first_part_end, first_part_end_output_index,
                          second_part_start};
}

// Adds copies for a rotated explicit non-copyable chain to transition in and
// out of non-copyable values.
absl::Status AddCopiesForNonCopyableTransitionsRotatedCase(
    const RotatedChainInfo& chain) {
  HloInstruction* start_op = chain.second_part_start;
  HloInstruction* end_op = chain.first_part_end;
  HloComputation* while_body = start_op->parent();
  // Handle aliasing input for the op, where we transition from copyable to
  // non-copyable.
  if (!chain.second_part_start->operands().empty()) {
    HloInstruction* operand = start_op->mutable_operand(0);
    HloInstruction* copied_operand =
        while_body->AddInstruction(HloInstruction::CreateUnary(
            operand->shape(), HloOpcode::kCopy, operand));
    VLOG(2) << "Transition from copyable to non-copyable:  copy "
            << operand->ToString() << " for " << start_op->ToString()
            << " output_index ";
    ABSL_RETURN_IF_ERROR(operand->ReplaceUseWith(start_op, copied_operand));
    ABSL_RETURN_IF_ERROR(end_op->AddControlDependencyTo(copied_operand));
  }

  // Add a control dependency from the rotated end_op of the chain to the
  // start_op of the chain guarantee disjoint live times of the buffer.
  ABSL_RETURN_IF_ERROR(end_op->AddControlDependencyTo(start_op));

  // Insert copies for the result produced by the end_op of the chain where we
  // transition from non-copyable to copyable.

  PtrVec<HloInstruction*> users = end_op->users();
  if (users.empty()) {
    return absl::OkStatus();
  }

  if (!end_op->shape().IsTuple()) {
    HloInstruction* copy = while_body->AddInstruction(
        HloInstruction::CreateUnary(end_op->shape(), HloOpcode::kCopy, end_op));
    ABSL_RETURN_IF_ERROR(copy->AddControlDependencyTo(start_op));
    return end_op->ReplaceAllUsesWith(copy);
  }

  for (auto user : users) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return FailedPrecondition("Expect user to be a GetTupleElement: %s",
                                user->ToString());
    }
    if (user->tuple_index() == chain.output_index) {
      VLOG(2) << "Transition from non-copyable to copyable: copy "
              << user->ToString() << " for all users";
      HloInstruction* copy = while_body->AddInstruction(
          HloInstruction::CreateUnary(user->shape(), HloOpcode::kCopy, user));
      ABSL_RETURN_IF_ERROR(copy->AddControlDependencyTo(start_op));
      ABSL_RETURN_IF_ERROR(user->ReplaceAllUsesWith(copy));
    }
  }

  return absl::OkStatus();
}

// The output with index `output_index` of `chain_start` and the input with
// index `operand_index` of `start_op` are explicit non-copyable values. If
// `chain_start` is the beginning of a non-copyable chain, we add copies to
// transition in and out of the chain.
absl::Status CopyInsertion::AddCopiesForExplicitNonCopyableTransitions(
    HloInstruction* chain_start) {
  VLOG(2) << "AddCopiesForExplicitNonCopyableTransitions: "
          << chain_start->ToString();
  HloComputation* parent = chain_start->parent();
  if (!parent->caller_instructions(HloOpcode::kWhile).empty()) {
    std::optional<RotatedChainInfo> rotated_chain_info =
        FindRotatedChainInfo(parent, chain_start);
    if (rotated_chain_info.has_value()) {
      return AddCopiesForNonCopyableTransitionsRotatedCase(
          rotated_chain_info.value());
    }
  }

  // For non-rotated cases, we only need to add a copy for the operand going
  // into chain_start, to transition from copyable to non-copyable.
  if (!chain_start->operands().empty()) {
    HloInstruction* operand = chain_start->mutable_operand(0);
    VLOG(2) << "Transition from copyable to non-copyable:  copy "
            << operand->ToString() << " for " << chain_start->ToString();
    HloInstruction* copied_operand =
        parent->AddInstruction(HloInstruction::CreateUnary(
            operand->shape(), HloOpcode::kCopy, operand));
    ABSL_RETURN_IF_ERROR(operand->ReplaceUseWith(chain_start, copied_operand));
  }

  return absl::OkStatus();
}

// If `chain_start` is the head of a chain of non-copyable ops inside a while
// loop, and part of the chain is rotated to the next iteration, returns the
// chain end in the rotated part. Otherwise, returns nullptr.
HloInstruction* FindEndOpForRotatedNonCopyableChain(
    const HloComputation* while_body, const HloInstruction* chain_start) {
  // Non-copyable op must have a single user.
  if (chain_start->user_count() != 1) {
    return nullptr;
  }
  HloInstruction* unique_user = chain_start->users().front();
  if (unique_user->opcode() != HloOpcode::kTuple || !unique_user->IsRoot()) {
    return nullptr;
  }
  int64_t index = unique_user->operand_index(chain_start);
  for (const HloInstruction* it :
       while_body->parameter_instruction(0)->users()) {
    const auto* gte = DynCast<HloGetTupleElementInstruction>(it);
    if (gte->tuple_index() == index) {
      CHECK_EQ(gte->user_count(), 1)
          << "non-copyable value in next loop iteration must "
             "be consumed by unique instruction.";
      HloInstruction* next_unique_user = gte->users().front();
      if (HloDataflowAnalysis::IsAsynchronousOperationDone(
              next_unique_user->opcode())) {
        return next_unique_user;
      }
      // Explicit non-copyable chain ends with a custom-call.
      if (chain_start->opcode() == HloOpcode::kCustomCall &&
          next_unique_user->opcode() == HloOpcode::kCustomCall) {
        return next_unique_user;
      }
    }
  }
  return nullptr;
}

// Adds copies for non-copyable transitioning between copyable and non-copyable
// for a chain start with `chain_start` and part of the chain is rotated to the
// next iteration that ends with `chain_end`.
absl::Status AddCopiesForNonCopyableTransitionsRotatedCase(
    HloInstruction* chain_start, HloInstruction* chain_end) {
  HloComputation* while_body = chain_start->parent();
  bool disable_loop_copies = false;
  for (const HloInstruction* caller :
       while_body->caller_instructions(HloOpcode::kWhile)) {
    if (frontend::HasDisableWhileLoopCopiesAttr(caller)) {
      disable_loop_copies = true;
      break;
    }
  }
  // Handle aliasing input for the op, where we transition from copyable to
  // non-copyable.
  if (!chain_start->operands().empty() && !disable_loop_copies) {
    // A chain_start may have multiple operands, but we assume only the first
    // operand is a buffer aliasing with the output, which is true currently.
    HloInstruction* operand = chain_start->mutable_operand(0);
    HloInstruction* copied_operand =
        while_body->AddInstruction(HloInstruction::CreateUnary(
            operand->shape(), HloOpcode::kCopy, operand));
    ABSL_RETURN_IF_ERROR(operand->ReplaceUseWith(chain_start, copied_operand));
    ABSL_RETURN_IF_ERROR(chain_end->AddControlDependencyTo(copied_operand));
  }

  // The chain_end is rotated and semantically paired with the chain_start of
  // the previous iteration. We add a control dependency from the chain_end to
  // the chain_start to in the same lexical iteration guarantee disjoint live
  // times of the buffers involved.
  ABSL_RETURN_IF_ERROR(chain_end->AddControlDependencyTo(chain_start));

  // If chain_end has users, insert copies for the result produced by the
  // chain_end with aliasing input and output buffers, where we transition from
  // non-copyable to copyable.
  PtrVec<HloInstruction*> users = chain_end->users();
  if (users.empty()) {
    return absl::OkStatus();
  }
  ShapeTree<HloInstruction*> copies_added(chain_end->shape());
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * copy,
      while_body->DeepCopyInstruction(chain_end, /*indices_to_copy=*/nullptr,
                                      &copies_added));
  for (auto [shape_index, instr] : copies_added) {
    if (instr != nullptr) {
      ABSL_RETURN_IF_ERROR(instr->AddControlDependencyTo(chain_start));
    }
  }
  for (HloInstruction* it : users) {
    ABSL_RETURN_IF_ERROR(chain_end->ReplaceUseWith(it, copy));
  }
  return absl::OkStatus();
}

// Adds the needed copies for transitioning into and out of non-copyable values,
// to prevent overlapping live times of buffers. This is needed when the unique
// user of the non-copyable op is rotated (also called pipelined) in a
// while-loop. In particular, if a non-copyable op has an input aliasing with
// its output, such as async Send, we make a copy of its input to transition
// from copyable to non-copyable. If a non-copyable op's unique user produces an
// output aliasing with its input, such as async Recv, we make a copy of the
// output produced by the unique user, to transition out of non-copyable to
// copyable. We also add control-flow edges between the copies and the
// non-copyable op to guarantee disjoint live times of the buffers invovled.
//
// Using async Send and Recv as examples, here is the transformation:
//
// Before:
//
//      kParameter               kParameter
//          |                        |
//      kSendDone                kRecvDone (end of a non-copyable chain)
//                                   |
//         ...                    consumer
//
//       producer                   ...
//          |
//        kSend                    kRecv   (start of a non-copyable op)
//          |                        |
//     (body root)              (body root)
//
//
// After:
//
//      kParameter                kParameter
//          |                         |
//      kSendDone ----+           kRecvDone
//                    |               |
//                   ctrl           kCopy ----+
//       producer    edge             |       |
//          |         |            consumer  ctrl
//        kCopy <-----+                      edge
//          |                                 |
//        kSend                     kRecv <---+
//          |                         |
//     (body root)               (body root)
//
absl::Status CopyInsertion::AddCopiesForNonCopyableTransitions(
    HloInstruction* chain_start) {
  if (!IsImplicitNonCopyable(chain_start)) {
    if (chain_start->IsCustomCall(kPinCustomCallTarget) ||
        chain_start->IsCustomCall(kCreateBufferCustomCallTarget)) {
      ABSL_RETURN_IF_ERROR(
          AddCopiesForExplicitNonCopyableTransitions(chain_start));
    }
    return absl::OkStatus();
  }

  // Handle implicitly chained non-copyable ops, that is async op chains.
  if (chain_start->users().empty()) {
    return absl::OkStatus();
  }

  // Currently non-copyable ops can have at most one user.
  if (chain_start->users().size() != 1) {
    return absl::InvalidArgumentError(
        "Non-copyable op must have a single user.");
  }

  HloInstruction* unique_user = chain_start->users().front();
  // If start feeds directly into done, the live time is contained and we don't
  // need to add any copies.
  if (HloDataflowAnalysis::IsAsynchronousOperationDone(unique_user->opcode())) {
    return absl::OkStatus();
  }

  HloComputation* parent = chain_start->parent();
  // If a start op with an operand is fed into a pipelined while-loop, we
  // need to make a copy of the operand and use the copy in the start op.
  if (chain_start->operand_count() > 0 &&
      unique_user->opcode() == HloOpcode::kTuple &&
      unique_user->users().size() == 1 &&
      unique_user->users().front()->opcode() == HloOpcode::kWhile) {
    if (frontend::HasDisableWhileLoopCopiesAttr(unique_user->users().front())) {
      return absl::OkStatus();
    }
    HloInstruction* operand = chain_start->mutable_operand(0);
    HloInstruction* copied_operand =
        parent->AddInstruction(HloInstruction::CreateUnary(
            operand->shape(), HloOpcode::kCopy, operand));
    ABSL_RETURN_IF_ERROR(operand->ReplaceUseWith(chain_start, copied_operand));
    return absl::OkStatus();
  }

  // For other cases where a non-copyable chain is outside of the while loop,
  // live times are disjoint. No copies are needed.
  if (parent->caller_instructions(HloOpcode::kWhile).empty()) {
    return absl::OkStatus();
  }

  // For async start ops, the end of the chain is the async done op.
  HloInstruction* chain_end =
      FindEndOpForRotatedNonCopyableChain(parent, chain_start);
  if (chain_end) {
    return AddCopiesForNonCopyableTransitionsRotatedCase(chain_start,
                                                         chain_end);
  }
  return absl::OkStatus();
}

// Add kCopy instructions to the given module to guarantee there is no
// live-range interference. Generally interference can only occur around kWhile
// instructions which have update-in-place semantics.
absl::Status CopyInsertion::AddCopiesToResolveInterference(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Only the xla_disable_while_loop_copies branch of AddCopiesForWhile reads
  // HloBuffers; everything else reads the dataflow analysis.
  // HloAliasAnalysis::Run calls HloDataflowAnalysis::Run(*module,
  // /*ssa_form=*/true, /*bitcast_defines_value=*/false) and passes alias_info_
  // only to CreateBuffers, so the same two booleans below yield the identical
  // dataflow analysis. The analysis is built before the walk on purpose: copies
  // inserted by earlier iterations would change the buffers a later build
  // would see.
  std::unique_ptr<HloAliasAnalysis> alias_analysis;
  std::unique_ptr<HloDataflowAnalysis> owned_dataflow;
  const HloDataflowAnalysis* dataflow = nullptr;
  if (HasWhileWithDisabledCopies(*module)) {
    ABSL_ASSIGN_OR_RETURN(alias_analysis, RunAliasAnalysis(module));
    dataflow = &alias_analysis->dataflow_analysis();
  } else {
    ABSL_ASSIGN_OR_RETURN(
        owned_dataflow,
        HloDataflowAnalysis::Run(
            *module, /*ssa_form=*/true, /*bitcast_defines_value=*/false,
            /*execution_threads=*/{}, /*propagate_through_calls=*/true,
            /*precompute_uses=*/std::nullopt,
            /*propagate_through_control_flow=*/true,
            MakePassthroughPredicate()));
    dataflow = owned_dataflow.get();
  }
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (computation->IsAsyncComputation()) {
      continue;
    }
    absl::flat_hash_map<const HloInstruction*, bool> disjoint_base_cache;
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        ABSL_RETURN_IF_ERROR(AddCopiesForWhile(*dataflow, alias_analysis.get(),
                                               instruction,
                                               classify_passthrough_));
      } else if (instruction->opcode() == HloOpcode::kConditional) {
        ABSL_RETURN_IF_ERROR(AddCopiesForConditional(*dataflow, instruction));
      } else if (IsNonCopyable(instruction)) {
        // We currently assume that we don't have a custom-call with
        // output-to-operand aliases for both buffers and non-buffers.
        ABSL_RETURN_IF_ERROR(AddCopiesForNonCopyableTransitions(instruction));
      } else {
        // When an operand is a tuple, we avoid copying the operand multiple
        // times by recording and checking the operand number of operands that
        // have been copied.
        absl::flat_hash_set<int64_t> copied_operands;
        HloOpcode op_code = instruction->opcode();
        int64_t nr_previous_bound_operands = 0;
        const HloInstruction* instr_to_get_aliases = instruction;
        if (op_code == HloOpcode::kAsyncStart ||
            op_code == HloOpcode::kAsyncUpdate) {
          // No operands bound, so we don't need to insert copies for them.
          if ((op_code == HloOpcode::kAsyncStart &&
               instruction->operand_count() == 0) ||
              (op_code == HloOpcode::kAsyncUpdate &&
               instruction->operand_count() == 1)) {
            continue;
          }

          if (op_code == HloOpcode::kAsyncStart &&
              instruction->output_operand_aliasing().empty()) {
            instr_to_get_aliases = instruction->async_wrapped_instruction();
          }

          if (op_code == HloOpcode::kAsyncUpdate) {
            CHECK_GE(instruction->operand_count(), 2);
            nr_previous_bound_operands =
                xla::hlo_instruction_utils::async::GetAsyncBoundOperands(
                    Cast<HloAsyncInstruction>(instruction->operand(0)))
                    .size();
          }
        }

        for (const auto& operand_and_output_index :
             alias_info_->GetInPlaceInputOutputPairs(instr_to_get_aliases)) {
          const HloOperandIndex& operand_index = operand_and_output_index.first;
          int64_t logical_operand_number = operand_index.operand_number;
          int64_t operand_index_in_this_intr = logical_operand_number;

          if (op_code == HloOpcode::kAsyncUpdate ||
              op_code == HloOpcode::kAsyncStart) {
            int64_t async_base_operand_idx =
                (op_code == HloOpcode::kAsyncUpdate) ? 1 : 0;
            int64_t nr_operands_bound_in_this_intr =
                instruction->operand_count() - async_base_operand_idx;

            if (logical_operand_number >= nr_previous_bound_operands &&
                logical_operand_number < nr_previous_bound_operands +
                                             nr_operands_bound_in_this_intr) {
              operand_index_in_this_intr =
                  async_base_operand_idx +
                  (logical_operand_number - nr_previous_bound_operands);
            } else {
              continue;
            }
          }

          if (copied_operands.contains(operand_index_in_this_intr)) {
            continue;
          }
          if ((op_code == HloOpcode::kAsyncUpdate ||
               op_code == HloOpcode::kAsyncDone) &&
              operand_index_in_this_intr == 0) {
            continue;
          }

          // Skip copies for aliasing input/output pairs when the instruction or
          // its enclosing while loop is annotated with
          // xla_disjoint_read_write_regions, as fusion may wrap companion
          // dynamic-slice readers into fusions without preserving the
          // attribute on every user.
          bool has_disjoint_loop = false;
          if (instruction->parent() != nullptr) {
            for (const HloInstruction* caller :
                 instruction->parent()->caller_instructions(
                     HloOpcode::kWhile)) {
              if (frontend::HasDisjointReadWriteRegionsAttr(caller)) {
                has_disjoint_loop = true;
                break;
              }
            }
          }
          if (has_disjoint_loop ||
              (frontend::HasDisjointReadWriteRegionsAttr(instruction) &&
               AllNonPassthroughUsersHaveDisjointRegions(
                   instruction->operand(operand_index_in_this_intr),
                   classify_passthrough_, &disjoint_base_cache))) {
            continue;
          }
          if ((instruction->opcode() == HloOpcode::kAsyncDone ||
               instruction->opcode() == HloOpcode::kAsyncUpdate) &&
              operand_index.operand_number == 0) {
            continue;
          }
          copied_operands.insert(operand_index_in_this_intr);
          ABSL_RETURN_IF_ERROR(AddCopiesForInPlaceOperation(
              instruction, operand_index_in_this_intr, classify_passthrough_));
        }
      }
    }
  }

  ABSL_RETURN_IF_ERROR(AddCopiesForAliasedInputOutputs(
      module, execution_threads, classify_passthrough_));
  return absl::OkStatus();
}

CopyInsertion::PassthroughAttachment CopyInsertion::ClassifyPassthrough(
    const HloInstruction* instruction) const {
  return ClassifyValidatedPassthrough(instruction, classify_passthrough_);
}

HloDataflowAnalysis::PassthroughPredicate
CopyInsertion::MakePassthroughPredicate() const {
  if (classify_passthrough_ == nullptr) {
    return nullptr;
  }
  return [classifier = classify_passthrough_](const HloInstruction* inst) {
    return ClassifyValidatedPassthrough(inst, classifier) !=
           PassthroughAttachment::kNone;
  };
}

absl::StatusOr<std::unique_ptr<HloAliasAnalysis>>
CopyInsertion::RunAliasAnalysis(const HloModule* module) const {
  TF_RET_CHECK(module != nullptr);
  return HloAliasAnalysis::Run(module, alias_info_, MakePassthroughPredicate());
}

absl::Status CopyInsertion::InsertRootCopyUnwrappingBitcast(
    HloComputation* computation, const ShapeTree<bool>& indices_to_copy,
    std::vector<HloInstruction*>* dead_bitcasts) const {
  TF_RET_CHECK(computation != nullptr);
  TF_RET_CHECK(dead_bitcasts != nullptr);
  HloInstruction* root = computation->root_instruction();
  TF_RET_CHECK(root != nullptr);
  TF_RET_CHECK(ShapeUtil::Compatible(root->shape(), indices_to_copy.shape()));

  HloInstruction* target = root;
  if (classify_passthrough_ != nullptr && indices_to_copy.element({}) &&
      root->opcode() == HloOpcode::kBitcast && root->user_count() == 0 &&
      ShapeUtil::Equal(root->shape(), root->operand(0)->shape()) &&
      IsPassthrough(root->operand(0))) {
    target = root->mutable_operand(0);
  }

  ABSL_ASSIGN_OR_RETURN(HloInstruction * copied,
                        DeepCopyConsumerOperand(target, /*target_user=*/nullptr,
                                                &indices_to_copy));
  if (target != root) {
    std::vector<HloInstruction*> preds(root->control_predecessors().begin(),
                                       root->control_predecessors().end());
    std::vector<HloInstruction*> succs(root->control_successors().begin(),
                                       root->control_successors().end());
    ABSL_RETURN_IF_ERROR(StripControlDependenciesFrom(root));
    for (HloInstruction* pred : preds) {
      ABSL_RETURN_IF_ERROR(pred->AddControlDependencyTo(copied));
    }
    for (HloInstruction* succ : succs) {
      ABSL_RETURN_IF_ERROR(copied->AddControlDependencyTo(succ));
    }
  }
  computation->set_root_instruction(copied);
  if (target != root && root->user_count() == 0) {
    dead_bitcasts->push_back(root);
  }
  return absl::OkStatus();
}

absl::Status CopyInsertion::RemoveDeadRootBitcasts(
    HloModule* module, absl::Span<HloInstruction* const> dead_bitcasts) const {
  TF_RET_CHECK(module != nullptr);
  for (HloInstruction* bitcast : dead_bitcasts) {
    if (bitcast->parent() == nullptr || bitcast->user_count() != 0) {
      continue;
    }
    HloComputation* computation = bitcast->parent();
    if (module->has_schedule() &&
        module->schedule().is_computation_scheduled(computation)) {
      module->schedule()
          .GetOrCreateSequence(computation)
          .remove_instruction(bitcast);
    }
    ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(bitcast));
  }
  return absl::OkStatus();
}

absl::StatusOr<HloInstruction*> CopyInsertion::DeepCopyConsumerOperand(
    HloInstruction* instruction, const HloInstruction* target_user,
    const ShapeTree<bool>* indices_to_copy,
    ShapeTree<HloInstruction*>* copies_added) const {
  return ::xla::DeepCopyConsumerOperand(instruction, target_user,
                                        indices_to_copy, copies_added,
                                        classify_passthrough_);
}

absl::Status CopyInsertion::InsertProducerCopy(
    HloInstruction* instruction, const ShapeTree<bool>& indices_to_copy,
    std::optional<absl::Span<HloInstruction* const>> users_override,
    bool replace_root,
    const absl::flat_hash_set<const HloInstruction*>* skip_users) const {
  return InsertProducerCopyImpl(
      instruction, indices_to_copy, users_override, replace_root, skip_users,
      /*copies_added=*/nullptr, classify_passthrough_);
}

absl::Status CopyInsertion::WrapRootPassthroughsInBitcast(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) const {
  TF_RET_CHECK(module != nullptr);
  if (classify_passthrough_ == nullptr) {
    return absl::OkStatus();
  }
  for (HloComputation* computation : module->computations(execution_threads)) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    HloInstruction* root = computation->root_instruction();
    if (root == nullptr || !IsPassthrough(root)) {
      continue;
    }
    std::vector<HloInstruction*> added_instructions;
    HloInstruction* new_root = nullptr;
    if (root->shape().IsArray()) {
      new_root = computation->AddInstruction(
          HloInstruction::CreateBitcast(root->shape(), root));
      added_instructions.push_back(new_root);
    } else if (root->shape().IsTuple()) {
      const size_t before_count = computation->instruction_count();
      std::vector<HloInstruction*> existing(computation->instructions().begin(),
                                            computation->instructions().end());
      absl::flat_hash_set<HloInstruction*> existing_set(existing.begin(),
                                                        existing.end());
      ABSL_ASSIGN_OR_RETURN(
          new_root,
          computation->DeepCopyInstructionWithCustomCopier(
              root,
              [](HloInstruction* leaf, const ShapeIndex& /*leaf_index*/,
                 HloComputation* comp) -> HloInstruction* {
                if (!leaf->shape().IsArray()) {
                  return leaf;
                }
                return comp->AddInstruction(
                    HloInstruction::CreateBitcast(leaf->shape(), leaf));
              }));
      if (computation->instruction_count() > before_count) {
        for (HloInstruction* inst : computation->instructions()) {
          if (!existing_set.contains(inst)) {
            added_instructions.push_back(inst);
          }
        }
      }
    }
    if (new_root == nullptr || new_root == root) {
      continue;
    }
    computation->set_root_instruction(new_root);
    if (module->has_schedule() &&
        module->schedule().is_computation_scheduled(computation)) {
      HloInstructionSequence& seq =
          module->schedule().GetOrCreateSequence(computation);
      auto it = absl::c_find(seq.instructions(), root);
      TF_RET_CHECK(it != seq.instructions().end());
      int64_t insert_idx = (it - seq.instructions().begin()) + 1;
      for (HloInstruction* added : added_instructions) {
        if (insert_idx < seq.size()) {
          seq.insert_instruction(added, insert_idx);
        } else {
          seq.push_back(added);
        }
        ++insert_idx;
      }
    }
  }
  return absl::OkStatus();
}

absl::Status CopyInsertion::AddSpecialCaseCopies(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    CustomBufferAnalysisFn custom_buffer_analysis) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  return AddSpecialCaseCopies(*call_graph, execution_threads, module,
                              custom_buffer_analysis);
}

absl::Status CopyInsertion::AddSpecialCaseCopies(
    const CallGraph& call_graph,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    HloModule* module, CustomBufferAnalysisFn custom_buffer_analysis) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                        RunAliasAnalysis(module));
  ABSL_ASSIGN_OR_RETURN(
      SpecialCaseCopies special_case_copies,
      ComputeSpecialCaseCopies(call_graph, execution_threads, module,
                               *alias_analysis, custom_buffer_analysis));
  ABSL_RETURN_IF_ERROR(InsertSpecialCaseCopies(special_case_copies, module));
  ABSL_RETURN_IF_ERROR(
      WrapRootPassthroughsInBitcast(module, execution_threads));
  if (module->has_schedule() && !special_case_copies.empty()) {
    ABSL_RETURN_IF_ERROR(module->schedule().Update());
  }
  return absl::OkStatus();
}

absl::StatusOr<CopyInsertion::SpecialCaseCopies>
CopyInsertion::ComputeSpecialCaseCopies(
    const CallGraph& call_graph,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    HloModule* module, const HloAliasAnalysis& alias_analysis,
    CustomBufferAnalysisFn custom_buffer_analysis) {
  SpecialCaseCopies result;
  HloInstructionMap<ShapeTree<bool>>& instructions_to_copy =
      result.instructions_to_copy;
  absl::flat_hash_map<std::pair<HloInstruction*, int64_t>, size_t>
      operand_copy_index;
  auto add_operand_index_to_copy = [&](HloInstruction* user,
                                       int64_t operand_number,
                                       const ShapeIndex& index) {
    if (IsWhileLoopCopyDisabled(user, call_graph) ||
        user->parent()->IsAsyncComputation()) {
      return;
    }
    const Shape& op_shape = user->operand(operand_number)->shape();
    if (ShapeUtil::GetSubshape(op_shape, index).IsBuffer()) {
      return;
    }
    auto [it, inserted] = operand_copy_index.try_emplace(
        std::make_pair(user, operand_number), result.operand_copies.size());
    if (inserted) {
      result.operand_copies.push_back(OperandCopy{
          .user = user,
          .operand_number = operand_number,
          .indices_to_copy = ShapeTree<bool>(op_shape, /*init_value=*/false),
      });
    }
    *result.operand_copies[it->second].indices_to_copy.mutable_element(index) =
        true;
  };
  auto unwrap_passthrough = [this](const HloInstruction* inst) {
    while (IsPassthrough(inst)) {
      inst = inst->operand(0);
    }
    return inst;
  };
  auto add_index_to_copy = [&instructions_to_copy, &call_graph](
                               HloInstruction* instruction,
                               const ShapeIndex& index) {
    if (IsWhileLoopCopyDisabled(instruction, call_graph)) {
      return;
    }
    // Buffers are non-copyable and needed copies are added to transition
    // in and out non-copyable values.
    if (ShapeUtil::GetSubshape(instruction->shape(), index).IsBuffer()) {
      return;
    }
    // Copies for async computations are determined by the async start/done
    // instructions.
    if (instruction->parent()->IsAsyncComputation()) {
      return;
    }
    VLOG(2) << "Adding index to copy: " << instruction->ToString() << "@"
            << index.ToString();
    auto it = instructions_to_copy.find(instruction);
    if (it == instructions_to_copy.end()) {
      auto it_added = instructions_to_copy.emplace(
          std::piecewise_construct, std::forward_as_tuple(instruction),
          std::forward_as_tuple(instruction->shape(), /*init_value=*/false));
      it = it_added.first;
    }
    *it->second.mutable_element(index) = true;
  };

  // Iterate through values of all constants and entry parameters. These values
  // are special because they are held in read-only buffers. If any of these
  // values share a buffer with other values (for example, the init value of a
  // while is a constant) then copy the value at its definition and replace all
  // its uses with the copy.
  // Also, locate all input-output aliasing violations for operations that
  // cannot be done in place. Such aliasing can be created when some copies are
  // removed too aggressively by CopyRemoval.
  for (const HloValue* value : alias_analysis.dataflow_analysis().values()) {
    const HloBuffer& buffer = alias_analysis.GetBufferContainingValue(*value);
    if (buffer.values().size() > 1 && ValueIsReadOnly(*value)) {
      VLOG(2) << "Value " << value->ToShortString()
              << " is read only, but its buffer contains more than one value. "
                 "Copying.";
      add_index_to_copy(value->defining_instruction(), value->defining_index());
    }

    for (const HloValue* value2 : buffer.values()) {
      // Find HloValues that share a position and use, which would cause the use
      // and operand to share buffers. Check if this is allowed and insert a
      // copy if it isn't.
      if (value2 == value) {
        continue;
      }
      HloPosition position = value2->defining_position();
      for (const HloUse& use : value->GetUses()) {
        // We already handle the copy of pin custom-call operands and shouldn't
        // add another copy here.
        if (!use.instruction->IsCustomCall(kPinCustomCallTarget) &&
            use.instruction == position.instruction) {
          VLOG(3) << "Same instruction: " << position.instruction->ToString();
          if ((use.instruction->opcode() == HloOpcode::kAsyncUpdate ||
               use.instruction->opcode() == HloOpcode::kAsyncDone) &&
              use.operand_number == 0) {
            continue;
          }
          if (!alias_analysis.dataflow_analysis().CanShareOperandBufferWithUser(
                  /*operand=*/use.instruction->mutable_operand(
                      use.operand_number),
                  /*operand_index=*/use.operand_index,
                  /*user=*/position.instruction,
                  /*user_index=*/position.index, alias_info_)) {
            VLOG(2) << "Adding back copy: "
                    << use.instruction->operand(use.operand_number)->ToString()
                    << "@" << use.operand_index.ToString()
                    << " instr: " << position.instruction->ToString() << "@"
                    << position.index;
            HloInstruction* operand =
                use.instruction->mutable_operand(use.operand_number);
            const HloInstruction* unwrapped = unwrap_passthrough(operand);
            bool shares_base_with_other_operand = false;
            if (classify_passthrough_ != nullptr) {
              for (int64_t op_idx = 0;
                   op_idx < use.instruction->operand_count(); ++op_idx) {
                if (op_idx != use.operand_number &&
                    unwrap_passthrough(use.instruction->operand(op_idx)) ==
                        unwrapped) {
                  shares_base_with_other_operand = true;
                  break;
                }
              }
            }
            if (shares_base_with_other_operand) {
              add_operand_index_to_copy(use.instruction, use.operand_number,
                                        use.operand_index);
            } else {
              add_index_to_copy(operand, use.operand_index);
            }
          }
        }
      }
    }
  }

  if (custom_buffer_analysis) {
    VLOG(2) << "Running custom buffer analysis";
    custom_buffer_analysis(module, alias_analysis, add_index_to_copy);
  }

  // Identify copies which must be added at root instructions
  for (HloComputation* computation : module->computations(execution_threads)) {
    const CallGraphNode& node = call_graph.GetNode(computation);
    if (node.context() == CallContext::kEmbedded) {
      continue;
    }
    TF_RET_CHECK(node.context() == CallContext::kControlFlow);
    SpecialCaseCopyPolicy policy =
        GetSpecialCaseCopyPolicy(node, module, computation);
    HloInstruction* root = computation->root_instruction();

    // Mark nondistinct/ambiguous indices.
    absl::flat_hash_map<const HloBuffer*, ShapeIndex> seen;
    ShapeUtil::ForEachSubshape(
        root->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          bool copy_replicated =
              policy.copy_root_replicated_buffers ||
              ShouldCopyConditionalRootAt(node, alias_analysis, index);
          std::vector<const HloBuffer*> buffers_at_index =
              alias_analysis.ComputeBuffersAt(root, index);
          bool buffer_seen_before = false;
          for (const HloBuffer* buffer : buffers_at_index) {
            buffer_seen_before |= !seen.emplace(buffer, index).second;
          }

          if (buffer_seen_before && copy_replicated &&
              computation == module->entry_computation() &&
              module->input_output_alias_config().OutputHasAlias(index) &&
              buffers_at_index.size() == 1) {
            std::optional<HloInputOutputAliasConfig::Alias> alias =
                module->input_output_alias_config().GetAliasedParameter(index);
            CHECK(alias) << "Alias does not exist";
            const ShapeIndex& other_index = seen[buffers_at_index[0]];
            VLOG(2) << "Output indices " << index.ToString() << " and "
                    << other_index.ToString() << " are both aliased to "
                    << alias->parameter_number << " copying " << other_index;
            add_index_to_copy(root, other_index);
            return;
          }

          if (buffers_at_index.size() > 1 ||
              (buffer_seen_before && copy_replicated)) {
            VLOG(2) << "Index " << index << " of computation "
                    << computation->name() << " (" << root->name()
                    << ") has ambiguous or non-distinct buffer. Copying.";
            add_index_to_copy(root, index);
          }
        });

    for (const auto& pair :
         alias_analysis.dataflow_analysis().GetInstructionValueSet(root)) {
      const ShapeIndex& index = pair.first;
      const HloValueSet& value_set = pair.second;
      for (const HloValue* value : value_set.values()) {
        if (ShouldCopyRootValue(*value, policy)) {
          VLOG(2) << "Root of (" << root->name() << ") of computation("
                  << computation->name()
                  << ") has constant or parameter value at index " << index
                  << ". Copying.";
          add_index_to_copy(root, index);
        }
      }
    }
  }
  return result;
}

absl::Status CopyInsertion::InsertSpecialCaseCopies(
    const SpecialCaseCopies& special_case_copies, HloModule* module) const {
  TF_RET_CHECK(module != nullptr);

  // 1. Copy individual conflicting operand edges (such as multi operand uses of
  // the same underlying producer across passthrough wrappers).
  for (const OperandCopy& op_copy : special_case_copies.operand_copies) {
    HloInstruction* operand =
        op_copy.user->mutable_operand(op_copy.operand_number);
    ABSL_ASSIGN_OR_RETURN(HloInstruction * deep_copy,
                          DeepCopyConsumerOperand(operand, op_copy.user,
                                                  &op_copy.indices_to_copy));
    if (deep_copy != operand) {
      ABSL_RETURN_IF_ERROR(operand->ReplaceUseWith(
          op_copy.user, op_copy.operand_number, deep_copy));
    }
  }

  // 2. Process pure root entries before producer entries so any temporary
  // identity root bitcast over a passthrough is unwrapped cleanly and never
  // double copied when its underlying producer is also in the map.
  std::vector<HloInstruction*> dead_bitcasts;
  absl::flat_hash_set<const HloInstruction*> handled_in_pass_1;
  absl::flat_hash_set<const HloComputation*> computations_with_new_root_copy;
  absl::flat_hash_set<const HloInstruction*> inserted_root_copies;
  for (const auto& [instruction, indices_to_copy] :
       special_case_copies.instructions_to_copy) {
    HloComputation* computation = instruction->parent();
    if (instruction == computation->root_instruction() &&
        instruction->user_count() == 0) {
      ABSL_RETURN_IF_ERROR(InsertRootCopyUnwrappingBitcast(
          computation, indices_to_copy, &dead_bitcasts));
      handled_in_pass_1.insert(instruction);
      computations_with_new_root_copy.insert(computation);
      HloInstruction* new_root = computation->root_instruction();
      inserted_root_copies.insert(new_root);
      const HloInstruction* curr = new_root;
      while (IsPassthrough(curr)) {
        curr = curr->operand(0);
      }
      if (curr->opcode() == HloOpcode::kCopy) {
        inserted_root_copies.insert(curr);
      }
    }
  }

  // 3. Process remaining consumer passthrough and producer entries, skipping
  // any root entries and root copies already handled in step 2.
  for (const auto& [instruction, indices_to_copy] :
       special_case_copies.instructions_to_copy) {
    if (handled_in_pass_1.contains(instruction)) {
      continue;
    }
    HloComputation* computation = instruction->parent();
    if (computations_with_new_root_copy.contains(computation) &&
        instruction->user_count() == 1 &&
        inserted_root_copies.contains(instruction->users()[0])) {
      continue;
    }

    if (ClassifyPassthrough(instruction) == PassthroughAttachment::kConsumer) {
      std::vector<HloInstruction*> users;
      for (HloInstruction* u : instruction->users()) {
        if (!inserted_root_copies.contains(u)) {
          users.push_back(u);
        }
      }
      const bool is_root =
          !computations_with_new_root_copy.contains(computation) &&
          instruction->IsRoot();
      if (users.empty() && !is_root) {
        continue;
      }
      ABSL_ASSIGN_OR_RETURN(
          HloInstruction * rebuilt,
          DeepCopyConsumerOperand(
              instruction, users.size() == 1 && !is_root ? users[0] : nullptr,
              &indices_to_copy));
      if (rebuilt != instruction) {
        for (HloInstruction* user : users) {
          ABSL_RETURN_IF_ERROR(instruction->ReplaceUseWith(user, rebuilt));
        }
        if (is_root) {
          computation->set_root_instruction(rebuilt);
        }
      }
      continue;
    }

    ABSL_RETURN_IF_ERROR(InsertProducerCopy(
        instruction, indices_to_copy, /*users_override=*/std::nullopt,
        /*replace_root=*/!computations_with_new_root_copy.contains(computation),
        &inserted_root_copies));
  }

  return RemoveDeadRootBitcasts(module, dead_bitcasts);
}

static int64_t GetNumExistingCopies(
    const HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  int64_t num_existing_copies = 0;
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCopy) {
        ++num_existing_copies;
      }
    }
  }
  return num_existing_copies;
}

absl::Status CopyInsertion::RemoveUnnecessaryCopies(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    bool insert_post_scheduling_control_dependencies) {
  XLA_VLOG_LINES(
      4, module->ToString(HloPrintOptions().set_syntax_sugar_async_ops(false)));

  // Use SequentialHloOrdering if the module has a schedule. The schedule can
  // provide more information on the ordering, allowing for detecting more
  // redundant copies.
  std::unique_ptr<HloOrdering> ordering;
  if (module->has_schedule()) {
    ordering = std::make_unique<SequentialHloOrdering>(module->schedule());
  } else {
    ordering = std::make_unique<DependencyHloOrdering>(module);
  }

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                        RunAliasAnalysis(module));
  CopyRemover copy_remover(*module, *alias_analysis, alias_info_,
                           ordering.get(), execution_threads, view_color_);
  if (VLOG_IS_ON(3)) {
    LOG(INFO) << "Removing unnecessary copies in " << module->name();
    LOG(INFO) << "Buffer values, in dependency order: ";
    for (const HloBuffer& buffer : alias_analysis->buffers()) {
      LOG(INFO) << "    HloBuffer " << buffer.id();
    }
  }

  int64_t num_existing_copies = GetNumExistingCopies(module, execution_threads);
  bool changed = true;
  int64_t num_iterations = -1;
  VLOG(6) << "Copy removal analyzing module (" << module->name()
          << ") with instruction count = " << module->instruction_count();
  BoundNonLinearCompilerAnalysis allowance(module, name(), 10);
  while (changed) {
    CHECK_LE(++num_iterations, num_existing_copies);
    changed = false;
    VLOG(2) << "RemoveUnnecessaryCopies running fixpoint iteration "
            << num_iterations << " of copy elision";
    for (HloComputation* computation :
         module->computations(execution_threads)) {
      VLOG(2) << "computation:" << computation->name();
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->opcode() != HloOpcode::kCopy) {
          continue;
        }
        // The region_analysis_cost_now is always set to
        // use_region_based_live_range_analysis_ if it is < 0, in which case the
        // analysis is always performed.
        int64_t region_analysis_cost_now =
            (use_region_based_live_range_analysis_ == 0)
                ? 0
                : std::min(allowance.analysis_allowance(),
                           use_region_based_live_range_analysis_);
        if (copy_remover.TryElideCopy(
                instruction, &region_analysis_cost_now,
                insert_post_scheduling_control_dependencies,
                should_skip_removal_)) {
          changed = true;
          ABSL_RETURN_IF_ERROR(StripControlDependenciesFrom(instruction));
          ABSL_RETURN_IF_ERROR(
              instruction->ReplaceAllUsesWith(instruction->mutable_operand(0)));
          VLOG(3) << "Copy removed successfully: " << instruction->ToString();
          XLA_VLOG_LINES(
              6, absl::StrCat("   Resulting Module: ", module->ToString()));
        }
        if (allowance.ContinueAnalysis() && region_analysis_cost_now > 0) {
          VLOG(6) << "Copy Insertion analyzing module cost: "
                  << region_analysis_cost_now;
          VLOG(6) << "instruction:" << instruction->ToString();
          allowance.DeductCost(region_analysis_cost_now);
          VLOG(6) << "allowance:" << allowance.analysis_allowance();
        }
      }
    }
  }
  // Wrap any root passthrough instructions in bitcasts after copy elision
  // settles so HloValue::IsRootOf remains valid throughout the fixpoint loop.
  return WrapRootPassthroughsInBitcast(module, execution_threads);
}

absl::Status CopyInsertion::OrderPredecessorReadersBeforeInPlaceCopies(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (computation->IsAsyncComputation()) {
      continue;
    }
    // Built on first use.
    std::unique_ptr<HloReachabilityMap> reachability;
    for (HloInstruction* instruction : computation->instructions()) {
      for (const auto& [operand_index, output_index] :
           alias_info_->GetInPlaceInputOutputPairs(instruction)) {
        if (!operand_index.operand_index.empty()) {
          continue;
        }
        HloInstruction* copy =
            instruction->mutable_operand(operand_index.operand_number);
        if (copy->opcode() != HloOpcode::kCopy ||
            copy->operand(0)->user_count() < 2) {
          continue;
        }
        if (reachability == nullptr) {
          reachability = HloReachabilityMap::Build(computation);
        }
        ABSL_RETURN_IF_ERROR(OrderPredecessorReadersBeforeCopy(
            copy, instruction, *reachability));
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> CopyInsertion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Copy insertion is performed in three steps:
  //
  // (1) Add copies conservatively to guarantee that there is no live-range
  //     interference. This is done simplistically and usually results in more
  //     copies than is strictly necessary.
  //
  // (2) Using a more fine-grained analysis, remove as many copies that were
  //     added in (1) as possible while ensuring no live-range interference.
  //
  // (3) Add copies to resolve issues not related to live range interference
  //     such as parameters and constants live out of the entry computation.
  //
  // We add copies then remove them (step (1) then (2)) rather than simply
  // adding only the copies that are necessary because, in general, it is
  // difficult to figure out the minimal set of copies to add once there is
  // interference. On the other hand, it is easy to determine if removing a copy
  // will introduce interference.
  //
  // The final copy insertion in (3) is done separately to simplify the
  // implementation of copy removal in (2) which is the most complicated part of
  // the pass. As is, copy removal only has to reason about live range
  // interference. If all copies were added in step (1) then copy removal would
  // also have to reason about things like constants and parameters live out of
  // the computation.
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  if (!call_graph->IsFlattened()) {
    return FailedPrecondition(
        "Call graph must be flattened before copy insertion.");
  }

  int64_t num_copies_before = GetNumExistingCopies(module, execution_threads);

  ABSL_RETURN_IF_ERROR(
      AddCopiesToResolveInterference(module, execution_threads));

  // Simplify the tuple structures introduced by the deep copies. This should be
  // done before removing copies (RemoveUnnecessaryCopies) because tuple
  // simplification changes dependencies in the graph which changes live range
  // interference in the graph. Also run DCE to remove the dead Tuple/GTE
  // instructions introduced by tuple simplification.
  TupleSimplifier tuple_simplifier;
  HloDCE dce;
  ABSL_RETURN_IF_ERROR(
      tuple_simplifier.Run(module, execution_threads).status());
  ABSL_RETURN_IF_ERROR(dce.Run(module, execution_threads).status());
  // Done after tuple simplification so that the orderings only rely on paths
  // that copy removal also sees.
  if (order_predecessor_readers_before_in_place_copies_) {
    ABSL_RETURN_IF_ERROR(
        OrderPredecessorReadersBeforeInPlaceCopies(module, execution_threads));
  }
  DumpHloModuleDuringPassIfEnabled(
      name(), "after adding copies to resolve interference", *module);

  ABSL_RETURN_IF_ERROR(RemoveUnnecessaryCopies(module, execution_threads));
  DumpHloModuleDuringPassIfEnabled(name(), "after removing unnecessary copies",
                                   *module);
  ABSL_RETURN_IF_ERROR(
      AddSpecialCaseCopies(*call_graph, execution_threads, module,
                           /*custom_buffer_analysis=*/nullptr));
  DumpHloModuleDuringPassIfEnabled(name(), "after adding special-case copies",
                                   *module);

  ABSL_RETURN_IF_ERROR(
      tuple_simplifier.Run(module, execution_threads).status());
  ABSL_RETURN_IF_ERROR(dce.Run(module, execution_threads).status());
  ABSL_RETURN_IF_ERROR(
      WrapRootPassthroughsInBitcast(module, execution_threads));
  if (module->has_schedule()) {
    ABSL_RETURN_IF_ERROR(module->schedule().Update());
  }

  VLOG(1) << "Num copies before copy-insertion: " << num_copies_before;
  VLOG(1) << "Num copies after copy-insertion: "
          << GetNumExistingCopies(module, execution_threads);

  return true;
}
}  // namespace xla
