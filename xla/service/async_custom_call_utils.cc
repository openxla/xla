/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/async_custom_call_utils.h"

#include <memory>
#include <optional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/service/call_inliner.h"

namespace xla {

std::optional<std::string> MatchingStartTarget(absl::string_view done_target) {
  absl::string_view base = done_target;
  if (!absl::ConsumeSuffix(&base, "-done")) {
    return std::nullopt;
  }
  return absl::StrCat(base, "-start");
}

absl::flat_hash_set<const HloInstruction*> FindCallsAlongAsyncPaths(
    const HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    absl::FunctionRef<bool(const HloInstruction*)> is_done) {
  absl::flat_hash_set<const HloInstruction*> calls_to_inline;
  CallInliner default_inliner;
  std::unique_ptr<CallGraph> call_graph;
  auto can_traverse_call = [&](const HloInstruction* call) {
    if (call->opcode() != HloOpcode::kCall ||
        call->called_computations().size() != 1) {
      return false;
    }
    if (!HloInstruction::IsThreadIncluded(call->parent()->execution_thread(),
                                          execution_threads) ||
        !HloInstruction::IsThreadIncluded(call->to_apply()->execution_thread(),
                                          execution_threads)) {
      return false;
    }
    if (call_graph == nullptr) {
      call_graph = CallGraph::Build(module, execution_threads);
    }
    return default_inliner.ShouldInline(*call_graph,
                                        const_cast<HloInstruction*>(call));
  };

  hlo_instruction_utils::async::AsyncTraceOptions options;
  options.traverse_while = false;
  options.can_traverse_call = can_traverse_call;
  options.matched_calls = &calls_to_inline;

  for (const HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (const HloInstruction* instr :
         computation->MakeInstructionPostOrder()) {
      if (!is_done(instr) || instr->operand_count() == 0) {
        continue;
      }
      std::optional<std::string> expected_start =
          MatchingStartTarget(instr->custom_call_target());
      if (!expected_start.has_value()) {
        continue;
      }
      hlo_instruction_utils::async::TraceAsyncDataflow(
          instr->operand(0),
          [&](const HloInstruction* candidate) {
            return candidate->IsCustomCall(*expected_start);
          },
          /*backward_steps=*/nullptr, options);
    }
  }
  return calls_to_inline;
}

absl::StatusOr<bool> InlineCallsAlongAsyncPaths(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    absl::FunctionRef<bool(const HloInstruction*)> is_done) {
  absl::flat_hash_set<const HloInstruction*> calls_to_inline =
      FindCallsAlongAsyncPaths(module, execution_threads, is_done);
  if (calls_to_inline.empty()) {
    return false;
  }
  CallInliner inliner(
      /*single_call_site=*/false,
      /*composites_to_preserve=*/{},
      /*override_policy=*/
      [&calls_to_inline](const CallGraph&, const HloInstruction* instruction) {
        return calls_to_inline.contains(instruction)
                   ? CallInliner::InlineOverridePolicy::kAllowInline
                   : CallInliner::InlineOverridePolicy::kProhibitInline;
      });
  return inliner.Run(module, execution_threads);
}

absl::Status FinishRewrite(
    HloComputation* computation, HloInstruction* start_call,
    HloInstruction* done_call, HloInstruction* async_start,
    HloInstruction* async_done,
    absl::Span<const hlo_instruction_utils::async::AsyncTraceStep> forward_path,
    HloInstruction* final_result, bool propagate_annotations) {
  ABSL_ASSIGN_OR_RETURN(HloInstruction * done_operand,
                        hlo_instruction_utils::async::PropagateDataflow(
                            forward_path, async_start));
  if (done_operand != async_done->operand(0)) {
    ABSL_RETURN_IF_ERROR(
        async_done->ReplaceOperandWithDifferentShape(0, done_operand));
  }

  if (propagate_annotations) {
    async_start->set_metadata(start_call->metadata());
    async_done->set_metadata(done_call->metadata());
    if (final_result != async_done) {
      final_result->set_metadata(done_call->metadata());
    }
    async_start->add_frontend_attributes(start_call->frontend_attributes());
    async_done->add_frontend_attributes(done_call->frontend_attributes());
  }

  for (HloInstruction* pred : start_call->control_predecessors()) {
    ABSL_RETURN_IF_ERROR(pred->AddControlDependencyTo(async_start));
  }
  for (HloInstruction* succ : start_call->control_successors()) {
    ABSL_RETURN_IF_ERROR(async_start->AddControlDependencyTo(succ));
  }
  for (HloInstruction* pred : done_call->control_predecessors()) {
    ABSL_RETURN_IF_ERROR(pred->AddControlDependencyTo(async_done));
  }
  for (HloInstruction* succ : done_call->control_successors()) {
    ABSL_RETURN_IF_ERROR(final_result->AddControlDependencyTo(succ));
  }

  ABSL_RETURN_IF_ERROR(done_call->ReplaceAllUsesWith(final_result));
  ABSL_RETURN_IF_ERROR(done_call->DropAllControlDeps());
  ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(done_call));

  if (start_call->user_count() == 0 && !start_call->IsRoot()) {
    ABSL_RETURN_IF_ERROR(start_call->DropAllControlDeps());
    ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(start_call));
  }
  return absl::OkStatus();
}

}  // namespace xla
