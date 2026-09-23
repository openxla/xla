/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/copy_fusion.h"

#include <cstdint>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/call_graph.h"
#include "xla/service/decision.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {
namespace {

// Returns the instructions that need the buffer of `value`, reached through
// the given `users` of `value`. Bitcasts, tuples and get-tuple-elements only
// forward the buffer, so the search continues through them; it follows a tuple
// only into the get-tuple-elements that read the element holding the value. A
// root tuple needs all of its buffers together and is itself a consumer.
absl::flat_hash_set<const HloInstruction*> FindMaterializingConsumers(
    const HloInstruction* value, absl::Span<HloInstruction* const> users) {
  // An instruction that forwards the value, and where in its shape it lives.
  using ForwardedValue = std::pair<const HloInstruction*, ShapeIndex>;
  std::vector<ForwardedValue> worklist;
  absl::flat_hash_set<ForwardedValue> visited;
  absl::flat_hash_set<const HloInstruction*> consumers;
  auto visit_users = [&](const HloInstruction* instruction,
                         const ShapeIndex& index,
                         absl::Span<HloInstruction* const> users) {
    for (const HloInstruction* user : users) {
      switch (user->opcode()) {
        case HloOpcode::kBitcast:
          worklist.emplace_back(user, index);
          break;
        case HloOpcode::kTuple:
          for (int64_t i = 0; i < user->operand_count(); ++i) {
            if (user->operand(i) == instruction) {
              ShapeIndex nested_index = index;
              nested_index.push_front(i);
              worklist.emplace_back(user, std::move(nested_index));
            }
          }
          break;
        case HloOpcode::kGetTupleElement:
          if (!index.empty() && user->tuple_index() == index.front()) {
            worklist.emplace_back(user,
                                  ShapeIndex(ShapeIndexView(index).subspan(1)));
          }
          break;
        default:
          consumers.insert(user);
      }
    }
  };
  visit_users(value, ShapeIndex(), users);
  while (!worklist.empty()) {
    ForwardedValue forwarded = std::move(worklist.back());
    worklist.pop_back();
    if (!visited.insert(forwarded).second) {
      continue;
    }
    const HloInstruction* instruction = forwarded.first;
    if (instruction->IsRoot()) {
      consumers.insert(instruction);
      continue;
    }
    visit_users(instruction, forwarded.second, instruction->users());
  }
  return consumers;
}

// Whether `fusion` is a large fill with a scalar literal that can be
// materialized again anywhere without changing the program.
Decision IsRematerializableConstantFill(const HloInstruction& fusion) {
  // Keep small fills together to avoid additional kernel launches.
  constexpr int64_t kMinRematerializedFillBytes = 1024 * 1024;
  if (fusion.fusion_kind() != HloInstruction::FusionKind::kLoop ||
      fusion.operand_count() != 0 || !fusion.shape().IsArray() ||
      fusion.HasControlDependencies() || fusion.has_sharding()) {
    return Decision::Forbid("Not an independent loop fill");
  }
  const HloComputation* computation = fusion.fused_instructions_computation();
  const HloInstruction* root = computation->root_instruction();
  if (computation->instruction_count() != 2 ||
      root->opcode() != HloOpcode::kBroadcast ||
      root->operand(0)->opcode() != HloOpcode::kConstant ||
      !ShapeUtil::IsScalar(root->operand(0)->shape()) ||
      root->HasControlDependencies() ||
      root->operand(0)->HasControlDependencies()) {
    return Decision::Forbid("Not a broadcast of a scalar literal");
  }
  if (ShapeUtil::ByteSizeOfElements(fusion.shape()) <
      kMinRematerializedFillBytes) {
    return Decision::Forbid("Fill is too small to split");
  }
  return Decision::Allow();
}

// Whether `copy` can be replaced by an independent instance of `fill`.
bool IsRematerializableCopy(const HloInstruction& fill,
                            const HloInstruction& copy) {
  return copy.operand(0) == &fill && !copy.has_sharding();
}

// Replaces copies of the constant `fill` with independent fills when their
// consumers are disjoint from the consumers that keep the original fill
// alive: its `other_users`, including copies this pass could not fuse, and the
// copies that stay fused with it. Splitting the initialization of buffers that
// a consumer needs together would add launches without separating lifetimes.
// This only screens obvious overlaps; the scheduler still decides whether the
// independent fills reduce peak memory.
//
// Rematerialized copies are removed from `copies`. The remaining copies are
// left to the multi-output rewrite. Returns whether the computation changed.
absl::StatusOr<bool> RematerializeIndependentCopies(
    HloComputation* computation, HloInstruction* fill,
    std::vector<HloInstruction*>& copies,
    std::vector<HloInstruction*>& other_users) {
  bool changed = false;
  // When nothing else uses the fill, one copy is redundant: the fill can take
  // its place, and that copy's consumers then keep the original fill alive.
  if (other_users.empty()) {
    auto redundant =
        absl::c_find_if(copies, [fill](const HloInstruction* copy) {
          return IsRematerializableCopy(*fill, *copy);
        });
    if (redundant == copies.end()) {
      return false;
    }
    HloInstruction* copy = *redundant;
    other_users.assign(copy->users().begin(), copy->users().end());
    ABSL_RETURN_IF_ERROR(
        computation
            ->ReplaceInstruction(copy, fill, /*preserve_sharding=*/false,
                                 /*relay_control_dependency=*/false,
                                 /*remove_unused_operands=*/false)
            .status());
    copies.erase(redundant);
    changed = true;
  }

  absl::flat_hash_set<const HloInstruction*> retaining_consumers =
      FindMaterializingConsumers(fill, other_users);
  for (const HloInstruction* copy : copies) {
    if (!IsRematerializableCopy(*fill, *copy)) {
      retaining_consumers.merge(
          FindMaterializingConsumers(copy, copy->users()));
    }
  }

  std::vector<HloInstruction*> remaining_copies;
  for (HloInstruction* copy : copies) {
    if (!IsRematerializableCopy(*fill, *copy)) {
      remaining_copies.push_back(copy);
      continue;
    }
    const bool shares_consumer =
        absl::c_any_of(FindMaterializingConsumers(copy, copy->users()),
                       [&retaining_consumers](const HloInstruction* consumer) {
                         return retaining_consumers.contains(consumer);
                       });
    if (shares_consumer) {
      VLOG(4) << "Not rematerializing " << copy->name()
              << ": it shares a consumer with the original fill";
      remaining_copies.push_back(copy);
      continue;
    }
    ABSL_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
        copy, fill->Clone("rematerialized"),
        /*preserve_sharding=*/false, /*relay_control_dependency=*/false,
        /*remove_unused_operands=*/false));
    changed = true;
  }
  copies = std::move(remaining_copies);
  return changed;
}

}  // namespace

bool OnlyElementwiseOpsReachableFromParams(HloComputation* fused_computation) {
  std::queue<const HloInstruction*> q;
  absl::flat_hash_set<const HloInstruction*> visited;
  for (auto param : fused_computation->parameter_instructions()) {
    q.push(param);
    visited.insert(param);
  }
  while (!q.empty()) {
    const HloInstruction* hlo = q.front();
    q.pop();
    for (auto user : hlo->users()) {
      if ((!user->IsElementwiseOnOperand(user->operand_index(hlo)) ||
           HloPredicateIsOp<HloOpcode::kCopy>(user)) &&
          HloPredicateIsNotOp<HloOpcode::kBitcast, HloOpcode::kTuple>(user)) {
        return false;
      }
      if (visited.insert(user).second) {
        q.push(user);
      }
    }
  }
  return true;
}

absl::StatusOr<bool> CopyFusion::DoCopyFusion(
    HloComputation* computation, std::unique_ptr<CallGraph> call_graph) {
  bool changed = false;
  std::vector<HloInstruction*> defs_before_uses =
      computation->MakeInstructionPostOrder();

  for (HloInstruction* hlo : defs_before_uses) {
    if (HloPredicateIsNotOp<HloOpcode::kFusion>(hlo) || hlo->IsCustomFusion()) {
      continue;
    }
    std::vector<HloInstruction*> copies;
    std::vector<HloInstruction*> other_users;
    HloComputation* fused_computation = hlo->fused_instructions_computation();
    if (!OnlyElementwiseOpsReachableFromParams(fused_computation)) {
      continue;
    }
    HloInstruction* root = fused_computation->root_instruction();
    if (IsReductionFromOrToContiguousDimensions(*root, device_description_) ||
        HloPredicateIsOp<HloOpcode::kScatter>(root) ||
        (hlo->IsMultiOutputFusion() &&
         absl::c_all_of(root->operands(),
                        HloPredicateIsOp<HloOpcode::kSlice>))) {
      continue;
    }
    for (auto user : hlo->users()) {
      HloInstruction* copy_user = user;
      // Skip get-tuple-element ops.
      if (HloPredicateIsOp<HloOpcode::kGetTupleElement>(copy_user) &&
          copy_user->user_count() == 1) {
        if (IsReductionFromOrToContiguousDimensions(
                *(root->operand(copy_user->tuple_index())),
                device_description_)) {
          other_users.push_back(user);
          continue;
        }
        copy_user = copy_user->users()[0];
      }
      // Skip bitcast ops.
      if (HloPredicateIsOp<HloOpcode::kBitcast>(copy_user) &&
          copy_user->user_count() == 1) {
        copy_user = copy_user->users()[0];
      }
      if (HloPredicateIsOp<HloOpcode::kCopy>(copy_user) &&
          copy_user->shape() == copy_user->operand(0)->shape() &&
          !copy_user->shape().IsTuple() &&
          !copy_user->HasControlDependencies() &&
          FusionFitsInBudget(*hlo, *copy_user, device_description_)) {
        copies.push_back(copy_user);
      } else {
        other_users.push_back(user);
      }
    }
    if (copies.empty()) {
      continue;
    }

    // Sharing a constant fill between an early copy and a later in-place user
    // keeps the original fill live until that later user. Materialize such
    // copies independently so the scheduler can delay the original fill.
    // Unlike the multi-output rewrite below, this preserves independent
    // lifetimes while still producing the separate writable values required
    // by copy insertion.
    if (const Decision fill = IsRematerializableConstantFill(*hlo);
        fill.IsAllowed()) {
      ABSL_ASSIGN_OR_RETURN(const bool rematerialized,
                            RematerializeIndependentCopies(
                                computation, hlo, copies, other_users));
      changed |= rematerialized;
      if (copies.empty()) {
        continue;
      }
    } else {
      VLOG(4) << "Not rematerializing copies of " << hlo->name() << ": "
              << fill.Explain();
    }

    auto fusion_adaptor = HloFusionAdaptor::ForComputation(fused_computation);
    auto dynamic_update_slices =
        GetOutputDefiningDynamicUpdateSlices(fusion_adaptor->GetRoots());
    // Skip dynamic update slice fusions which might be emitted in-place.
    if (!dynamic_update_slices.empty() &&
        (HloPredicateIsNotOp<HloOpcode::kTuple>(root) ||
         dynamic_update_slices.size() == root->shape().tuple_shapes().size())) {
      continue;
    }

    int64_t num_outputs =
        hlo->IsMultiOutputFusion() ? root->operand_count() : int64_t{1};
    int64_t total_outputs = num_outputs + copies.size();

    if (total_outputs > MaxOperandsAndOutputsPerFusion()) {
      VLOG(1) << "Skipping fusion as it would exceed "
                 "MaxOperandsAndOutputsPerFusion(): "
              << total_outputs << " > " << MaxOperandsAndOutputsPerFusion();
      continue;
    }

    changed = true;

    HloInstruction::InstructionVector tuple_elements;
    tuple_elements.reserve(copies.size() + num_outputs);
    if (hlo->IsMultiOutputFusion()) {
      for (HloInstruction* operand : root->operands()) {
        tuple_elements.push_back(operand);
      }
    } else {
      tuple_elements.push_back(root);
    }

    for (auto copy : copies) {
      HloInstruction* user = copy;
      std::vector<HloInstruction*> operand_chain;
      operand_chain.push_back(user);
      while (user->operand(0) != hlo) {
        user = user->mutable_operand(0);
        operand_chain.push_back(user);
      }
      HloInstruction* clone_operand = root;
      if (hlo->IsMultiOutputFusion()) {
        clone_operand = root->mutable_operand(user->tuple_index());
        CHECK_EQ(operand_chain.back()->opcode(), HloOpcode::kGetTupleElement);
        operand_chain.pop_back();
      }
      for (int64_t i = operand_chain.size() - 1; i >= 0; --i) {
        HloInstruction* user = operand_chain[i];
        clone_operand = fused_computation->AddInstruction(
            user->CloneWithNewOperands(user->shape(), {clone_operand}));
      }
      tuple_elements.push_back(clone_operand);
    }

    HloInstruction* new_root = fused_computation->AddInstruction(
        HloInstruction::CreateTuple(tuple_elements));
    fused_computation->set_root_instruction(new_root,
                                            /*accept_different_shape=*/true);
    // Creates a new original value for the fusion instruction and the new root
    // of the fused computation.
    if (hlo->original_value() != nullptr) {
      std::shared_ptr<xla::OriginalValue> new_original_value =
          xla::OriginalValue::CreateFromInstruction(new_root);
      new_root->set_original_value(new_original_value);
      hlo->set_original_value(new_original_value);
    }
    *hlo->mutable_shape() = new_root->shape();
    for (HloInstruction* caller :
         call_graph->GetComputationCallers(fused_computation)) {
      if (caller->opcode() == HloOpcode::kFusion) {
        if (caller->has_sharding()) {
          caller->clear_sharding();
        }
      }
    }

    if (HloPredicateIsOp<HloOpcode::kTuple>(root)) {
      ABSL_RETURN_IF_ERROR(fused_computation->RemoveInstruction(root));
    } else {
      auto get_tuple_element_root = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(hlo, 0));
      ABSL_RETURN_IF_ERROR(hlo->ReplaceAllUsesWithDifferentShape(
          other_users, get_tuple_element_root));
    }
    for (int64_t i = 0; i < copies.size(); ++i) {
      auto get_tuple_element = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(hlo, num_outputs + i));
      ABSL_RETURN_IF_ERROR(
          computation->ReplaceInstruction(copies[i], get_tuple_element));
    }
  }
  return changed;
}

absl::StatusOr<bool> CopyFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Only for the entry computation we can be sure that the copies do not share
  // a buffer with a parameter of the fusion that it will be fused with. For
  // example while loop computations have tuple parameters that need to share
  // the buffers with the output tuples, and copies inserted by the
  // CopyInsertion pass will share a buffer with the tuple output (and thus
  // with the tuple input as well).
  return DoCopyFusion(module->entry_computation(), CallGraph::Build(module));
}

}  // namespace gpu
}  // namespace xla
