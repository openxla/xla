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

#include "xla/service/async_computation_utils.h"

#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/side_effect_util.h"

namespace xla {
namespace {

bool IsAsyncBarrierComputation(const HloComputation* computation) {
  return absl::c_all_of(
      computation->instructions(),
      [root = computation->root_instruction()](const HloInstruction* instr) {
        return root == instr || instr->opcode() == HloOpcode::kParameter ||
               instr->opcode() == HloOpcode::kGetTupleElement;
      });
}

// Given an computation that needs to run async but contains more instructions
// than just parameters and a single root, builds a new computation that
// contains only parameters and a single root which can be run async. The root
// of the new computation just calls into the original input computation.
std::unique_ptr<HloComputation> BuildAsyncWrapperComputation(
    HloComputation* computation) {
  HloComputation::Builder builder(
      absl::StrCat(computation->name(), ".async_wrapper"));

  // Add parameters to the wrapper.
  std::vector<HloInstruction*> parameters;
  for (const HloInstruction* parameter :
       computation->parameter_instructions()) {
    HloInstruction* wrapper_parameter = builder.AddInstruction(
        HloInstruction::CreateParameter(parameter->parameter_number(),
                                        parameter->shape(), parameter->name()));
    wrapper_parameter->set_metadata(parameter->metadata());
    parameters.push_back(wrapper_parameter);
  }

  // Add the root call instruction.
  HloInstruction* original_root = computation->root_instruction();
  HloInstruction* call = builder.AddInstruction(HloInstruction::CreateCall(
      original_root->shape(), parameters, computation));
  call->set_metadata(original_root->metadata());

  // Build and return the wrapper computation.
  std::unique_ptr<HloComputation> wrapper = builder.Build(call);
  wrapper->SetExecutionThread(computation->execution_thread());
  return wrapper;
}

bool CallRequiresSingleInstruction(const HloInstruction* caller,
                                   const HloComputation* computation) {
  if (!caller->IsAsynchronous() ||
      computation->execution_thread() != HloInstruction::kMainExecutionThread ||
      IsAsyncBarrierComputation(computation)) {
    return false;
  }

  // The call is treated as a collective group if it is an async computation
  // whose async-start had the collective group annotation, or if it itself has
  // a collective group annotation.
  HloInstruction* async_chain_start = caller->async_chain_start();
  bool call_is_collective_group =
      (async_chain_start != nullptr &&
       async_chain_start->get_frontend_attribute(kCollectiveGroupMarkerAttr)
           .has_value()) ||
      caller->get_frontend_attribute(kCollectiveGroupMarkerAttr).has_value();

  return !call_is_collective_group;
}

}  // namespace

bool AsyncComputationRequiresSingleInstruction(
    const HloComputation& computation) {
  return absl::c_any_of(computation.caller_instructions(),
                        [&computation](const HloInstruction* caller) {
                          return CallRequiresSingleInstruction(caller,
                                                               &computation);
                        });
}

absl::StatusOr<bool> LegalizeAsyncComputations(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    // Skip computations that don't need to be single-instruction or are already
    // single-instruction.
    if (!AsyncComputationRequiresSingleInstruction(*computation) ||
        computation->CanExpandIntoSingleInstruction()) {
      continue;
    }

    // Build and add a new, single-instruction computation that wraps the
    // current computation.
    HloComputation* wrapper = module->AddEmbeddedComputation(
        BuildAsyncWrapperComputation(computation));

    // Modify async callers without collective group annotations to call into
    // the wrapper.
    const auto callers = computation->caller_instructions();
    for (HloInstruction* caller : callers) {
      if (CallRequiresSingleInstruction(caller, computation)) {
        caller->ReplaceCalledComputations([&](HloComputation* callee) {
          return callee == computation ? wrapper : callee;
        });
      }
    }

    changed = true;
  }
  return changed;
}

}  // namespace xla
