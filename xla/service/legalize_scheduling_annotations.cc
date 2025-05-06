/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/legalize_scheduling_annotations.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/ptrvec.h"
#include "xla/service/scheduling_annotations_util.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

// Given a group of annotated instructions (sources), find all reachable
// instructions from them in the same computation.
absl::flat_hash_set<HloInstruction*> PropagateAnnotationFromSources(
    const std::vector<HloInstruction*>& sources,
    const HloComputation* computation) {
  absl::flat_hash_set<HloInstruction*> to_annotate;
  auto reachability = HloReachabilityMap::Build(computation);
  // worklist contains instructions that can reach any source instruction.
  std::queue<HloInstruction*> work_queue;
  absl::flat_hash_set<HloInstruction*> visited;
  absl::flat_hash_set<HloInstruction*> sources_set(sources.begin(),
                                                   sources.end());
  for (HloInstruction* instr : sources) {
    for (HloInstruction* another_instr : sources) {
      if (instr == another_instr) {
        continue;
      }
      if (reachability->IsReachable(instr, another_instr)) {
        work_queue.push(instr);
        visited.insert(instr);
        break;
      }
    }
  }

  while (!work_queue.empty()) {
    auto* instr = work_queue.front();
    work_queue.pop();
    if (!sources_set.contains(instr)) {
      to_annotate.insert(instr);
    }
    for (const PtrVec<HloInstruction*>& users :
         {instr->users(), instr->control_successors()}) {
      for (HloInstruction* user : users) {
        if (visited.contains(user)) {
          continue;
        }
        // Add user to work queue if it reaches any source instruction.
        for (HloInstruction* source : sources) {
          if (reachability->IsReachable(user, source)) {
            work_queue.push(user);
            visited.insert(user);
            break;
          }
        }
      }
    }
  }
  return to_annotate;
}

// Attach the annotation ID to the given instructions. Returns error if any of
// the instructions already has an annotation.
absl::Status AttachAnnotation(
    Annotation annotation,
    const absl::flat_hash_set<HloInstruction*>& instructions) {
  for (HloInstruction* instr : instructions) {
    TF_ASSIGN_OR_RETURN(std::optional<Annotation> instr_annotation,
                        GetSchedulingAnnotation(instr));
    if (instr_annotation) {
      return absl::InternalError("Trying to propagate scheduling annotation " +
                                 annotation.ToString() + " to " +
                                 std::string(instr->name()) +
                                 " but it has an existing annotation: " +
                                 instr_annotation->ToString());
    }
    LOG(INFO) << "Propagating annotation " << annotation.ToString() << " to "
              << instr->name();
    TF_RETURN_IF_ERROR(SetSchedulingAnnotation(instr, annotation));
  }
  return absl::OkStatus();
}

bool IsSupportedAsyncOp(HloInstruction* instr) {
  return HloPredicateIsOp<
      HloOpcode::kAllGatherDone, HloOpcode::kAllGatherStart,
      HloOpcode::kAllReduceDone, HloOpcode::kAllReduceStart,
      HloOpcode::kCollectivePermuteDone, HloOpcode::kCollectivePermuteStart,
      HloOpcode::kAsyncDone, HloOpcode::kAsyncStart, HloOpcode::kSendDone,
      HloOpcode::kSend, HloOpcode::kRecvDone, HloOpcode::kRecv>(instr);
}

absl::Status CheckStartDoneAnnotationConsistency(
    const absl::flat_hash_map<
        Annotation,
        absl::flat_hash_map<HloComputation*, std::vector<HloInstruction*>>>&
        annotation_to_instruction,
    const absl::flat_hash_map<HloInstruction*, Annotation>&
        instruction_to_annotation) {
  for (const auto& [annotation, comp_inst_vector] : annotation_to_instruction) {
    for (const auto& [comp, annotated_instructions] : comp_inst_vector) {
      for (HloInstruction* instr : annotated_instructions) {
        CHECK(instruction_to_annotation.contains(instr));
        CHECK(instruction_to_annotation.at(instr) == annotation);
        if (HloPredicateIsOp<
                HloOpcode::kAllGatherDone, HloOpcode::kAllReduceDone,
                HloOpcode::kCollectivePermuteDone, HloOpcode::kAsyncDone>(
                instr) &&
            (!instruction_to_annotation.contains(instr->operand(0)) ||
             instruction_to_annotation.at(instr->mutable_operand(0)) !=
                 annotation)) {
          return absl::InternalError(absl::StrCat(
              "Done instruction's operand is not annotated with the same id: ",
              instr->operand(0)->name(),
              ", annotation: ", annotation.ToString()));
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status CheckGapBetweenAnnotatedInstructions(
    const absl::flat_hash_map<
        Annotation,
        absl::flat_hash_map<HloComputation*, std::vector<HloInstruction*>>>&
        annotation_to_instruction,
    const absl::flat_hash_map<HloInstruction*, Annotation>&
        instruction_to_annotation) {
  absl::flat_hash_map<HloInstruction*, HloInstruction*> parent;
  for (const auto& [annotation, comp_inst_vector] : annotation_to_instruction) {
    for (const auto& [comp, annotated_instructions] : comp_inst_vector) {
      // First find the frontier nodes that are not annotated with id but use an
      // annotated instruction with id.
      std::vector<HloInstruction*> stack;
      absl::flat_hash_set<HloInstruction*> visited;
      for (HloInstruction* instr : annotated_instructions) {
        CHECK(instruction_to_annotation.contains(instr));
        CHECK(instruction_to_annotation.at(instr) == annotation);
        for (const PtrVec<HloInstruction*>& users :
             {instr->users(), instr->control_successors()}) {
          for (HloInstruction* user : users) {
            if (!visited.contains(user) &&
                (!instruction_to_annotation.contains(user) ||
                 instruction_to_annotation.at(user) != annotation)) {
              stack.push_back(user);
              parent[user] = instr;
              visited.insert(user);
              VLOG(2) << "Annotation : " << annotation.ToString()
                      << ", frontier using a root: " << user->name();
            }
          }
        }
      }
      VLOG(2) << "Annotation : " << annotation.ToString() << ", frontier has "
              << stack.size() << " instructions";
      // Traverse the HLO graph starting from the frontier instructions and move
      // to the users. If there are gaps in the annotation, the traversal will
      // hit an instruction that is annotated with the same id.
      while (!stack.empty()) {
        HloInstruction* instr = stack.back();
        stack.pop_back();
        for (const PtrVec<HloInstruction*>& users :
             {instr->users(), instr->control_successors()}) {
          for (HloInstruction* user : users) {
            if (instruction_to_annotation.contains(user) &&
                instruction_to_annotation.at(user) == annotation) {
              LOG(INFO) << "PATH: " << user->name();
              HloInstruction* current = instr;
              LOG(INFO) << "PATH: " << current->name();
              while (parent.contains(current)) {
                current = parent[current];
                LOG(INFO) << "PATH: " << current->name();
              }
              return absl::UnimplementedError(absl::StrCat(
                  "Support for annotation groups with gaps doesn't "
                  "exist yet, annotation: ",
                  annotation.ToString(), ", instr: ", user->name(),
                  " has the same annotation in its operand tree but "
                  "has gaps on the way from that operand to itself."));
            }
            if (visited.contains(user)) {
              continue;
            }
            stack.push_back(user);
            parent[user] = instr;
            visited.insert(user);
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> HaulAnnotationToFusionInstruction(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    const absl::flat_hash_map<
        Annotation,
        absl::flat_hash_map<HloComputation*, std::vector<HloInstruction*>>>&
        annotation_to_instruction,
    const absl::flat_hash_map<HloInstruction*, Annotation>&
        instruction_to_annotation,
    std::function<bool(HloInstruction*)> keep_sync_annotation) {
  bool changed = false;
  for (HloComputation* computation : module->computations(execution_threads)) {
    if (!computation->IsFusionComputation() ||
        !keep_sync_annotation(computation->FusionInstruction()) ||
        instruction_to_annotation.contains(computation->FusionInstruction())) {
      continue;
    }
    changed = true;
    std::optional<Annotation> seen_annotation;
    for (HloInstruction* instr : computation->instructions()) {
      TF_ASSIGN_OR_RETURN(std::optional<Annotation> annotation,
                          GetSchedulingAnnotation(instr));
      if (!annotation) {
        continue;
      }
      if (!seen_annotation) {
        seen_annotation = annotation;
        continue;
      }
      if (seen_annotation != annotation) {
        return absl::InternalError(absl::StrCat(
            "Found a fusion with multiple annotations in the fused "
            "computation. fusion: ",
            computation->FusionInstruction()->name(), ", annotations: ",
            seen_annotation->ToString(), " and ", annotation->ToString()));
      }
    }
    // No fused instructions are annotated, nothing to do.
    if (!seen_annotation) {
      continue;
    }
    TF_RETURN_IF_ERROR(SetSchedulingAnnotation(computation->FusionInstruction(),
                                               seen_annotation->ToString()));
  }
  return changed;
}

absl::StatusOr<bool> RemoveLoopIterationAnnotation(HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instr : computation->instructions()) {
      TF_ASSIGN_OR_RETURN(bool removed,
                          RemoveSchedulingAnnotationIterationId(instr));
      changed |= removed;
    }
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> LegalizeSchedulingAnnotations::PropagateAnnotations(
    const HloComputation* computation,
    const absl::btree_map<Annotation, std::vector<HloInstruction*>>&
        annotation_to_instruction) {
  bool changed = false;
  for (auto& [annotation, sources] : annotation_to_instruction) {
    absl::flat_hash_set<HloInstruction*> to_annotate =
        PropagateAnnotationFromSources(sources, computation);
    changed |= (!to_annotate.empty());
    auto status = AttachAnnotation(annotation, to_annotate);
    if (!status.ok()) {
      return status;
    }
  }
  return changed;
}

bool LegalizeSchedulingAnnotations::KeepSchedulingAnnotation(
    HloInstruction* instr) {
  const auto& attrs = instr->frontend_attributes().map();
  if (attrs.contains(kXlaSchedulingGroupIdAttr) &&
      attrs.at(kXlaSchedulingGroupIdAttr) == kXlaNoOpSchedulingGroup) {
    return false;
  }

  return IsSupportedAsyncOp(instr) || config_.keep_sync_annotation(instr);
}

absl::StatusOr<bool> LegalizeSchedulingAnnotations::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  // Remove loop iteration annotation if requested.
  if (config_.remove_loop_iteration_annotation_only) {
    TF_ASSIGN_OR_RETURN(bool removed, RemoveLoopIterationAnnotation(module));
    changed |= removed;
    return changed;
  }

  absl::flat_hash_map<HloInstruction*, Annotation> instruction_to_annotation;
  absl::flat_hash_map<
      Annotation,
      absl::flat_hash_map<HloComputation*, std::vector<HloInstruction*>>>
      annotation_to_instruction;
  // Filter the annotated ops (using config) to keep the annotations only in the
  // desired sync ops. Annotations in all async ops are kept.
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instr : computation->instructions()) {
      if (HasSchedulingAnnotation(instr) && !KeepSchedulingAnnotation(instr)) {
        changed |= RemoveSchedulingAnnotation(instr);
      }
    }
  }

  // Find the annotated instructions and save relevant information.
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : computation->instructions()) {
      TF_ASSIGN_OR_RETURN(std::optional<Annotation> annotation,
                          GetSchedulingAnnotation(instr));
      if (!annotation) {
        continue;
      }
      instruction_to_annotation[instr] = *annotation;
      annotation_to_instruction[*annotation][computation].push_back(instr);
    }
  }

  // Move the annotation from inside fusion computation to the caller
  // instruction if the caller doesn't have an annotation. Return an error if
  // there are some fused instructions with different annotations.
  TF_ASSIGN_OR_RETURN(
      bool haul_annotation_to_top_level,
      HaulAnnotationToFusionInstruction(
          module, execution_threads, annotation_to_instruction,
          instruction_to_annotation, config_.keep_sync_annotation));
  changed |= haul_annotation_to_top_level;

  if (annotation_to_instruction.empty()) {
    return changed;
  }

  if (config_.check_start_done_annotation_consistency) {
    auto status = CheckStartDoneAnnotationConsistency(
        annotation_to_instruction, instruction_to_annotation);
    if (!status.ok()) {
      return status;
    }
  }

  // Either propagate the annotation to fill the gaps between instructions with
  // the same annotation ID or check (and return error) if there are gaps.
  if (config_.propagate_annotation) {
    // Propagate the annotation to fill the gaps between instructions with the
    // same annotation ID.
    for (HloComputation* computation :
         module->MakeNonfusionComputations(execution_threads)) {
      absl::btree_map<Annotation, std::vector<HloInstruction*>>
          per_computation_annotation_to_instruction;
      for (const auto& [annotation, comp_inst_vector] :
           annotation_to_instruction) {
        if (comp_inst_vector.contains(computation)) {
          per_computation_annotation_to_instruction[annotation] =
              comp_inst_vector.at(computation);
        }
      }
      if (per_computation_annotation_to_instruction.empty()) {
        continue;
      }
      auto result = PropagateAnnotations(
          computation, per_computation_annotation_to_instruction);
      if (!result.ok()) {
        return result.status();
      }
      changed |= result.value();
    }
  } else {
    auto result = CheckGapBetweenAnnotatedInstructions(
        annotation_to_instruction, instruction_to_annotation);
    if (!result.ok()) {
      return result;
    }
  }

  return changed;
}
}  // namespace xla
