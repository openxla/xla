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

#include "xla/hlo/transforms/simplifiers/alias_copy_avoidance.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/ptrvec.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

// Whether an instruction of `readers` depends on `writer` through data or
// control edges, so that it runs after `writer` in every schedule.
bool AnyDependsOn(const HloInstruction* writer,
                  const absl::flat_hash_set<const HloInstruction*>& readers) {
  absl::flat_hash_set<const HloInstruction*> visited = {writer};
  std::vector<const HloInstruction*> worklist = {writer};
  while (!worklist.empty()) {
    const HloInstruction* instruction = worklist.back();
    worklist.pop_back();
    for (const PtrVec<HloInstruction*>* successors :
         {&instruction->users(), &instruction->control_successors()}) {
      for (const HloInstruction* successor : *successors) {
        if (readers.contains(successor)) {
          return true;
        }
        if (visited.insert(successor).second) {
          worklist.push_back(successor);
        }
      }
    }
  }
  return false;
}

// Whether the alias of the parameter value `input` with the output value
// `output` cannot be kept without a copy of the parameter.
bool AliasNeedsCopy(const HloValue& input, const HloValue& output) {
  const HloInstruction* writer = output.defining_instruction();
  if (writer->opcode() == HloOpcode::kParameter ||
      writer->opcode() == HloOpcode::kConstant || input.live_out_of_module()) {
    return false;
  }
  // A reader inside a called computation is reached through its caller, which
  // uses the value too.
  absl::flat_hash_set<const HloInstruction*> readers;
  for (const HloUse& use : input.GetUses()) {
    readers.insert(use.instruction);
  }
  return AnyDependsOn(writer, readers);
}

}  // namespace

absl::StatusOr<bool> AliasCopyAvoidance::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HloComputation* entry = module->entry_computation();
  if (!HloInstruction::IsThreadIncluded(entry->execution_thread(),
                                        execution_threads)) {
    return false;
  }
  HloInputOutputAliasConfig& aliases = module->input_output_alias_config();
  HloInputOutputAliasConfig kept(aliases.shape());
  // The values copy insertion sees. Built for the first optional alias.
  std::unique_ptr<HloDataflowAnalysis> dataflow;
  // Parameters may be the elements of one wide tuple.
  auto is_entry_parameter = [entry](const HloValue& value) {
    return value.defining_instruction()->opcode() == HloOpcode::kParameter &&
           value.defining_instruction()->parent() == entry;
  };
  bool changed = false;
  ABSL_RETURN_IF_ERROR(aliases.ForEachAliasWithStatus(
      [&](const ShapeIndex& output_index,
          const HloInputOutputAliasConfig::Alias& alias) -> absl::Status {
        if (!alias.must_alias()) {
          if (dataflow == nullptr) {
            ABSL_ASSIGN_OR_RETURN(dataflow,
                                  HloDataflowAnalysis::Run(
                                      *module, /*ssa_form=*/true,
                                      /*bitcast_defines_value=*/false,
                                      /*execution_threads=*/{},
                                      /*propagate_through_calls=*/true,
                                      /*precompute_uses=*/is_entry_parameter));
          }
          const HloValue& input = dataflow->GetValueDefinedAt(
              entry->parameter_instruction(alias.parameter_number),
              alias.parameter_index);
          // An output of a flattened call graph holds one value.
          if (absl::c_all_of(
                  dataflow->GetValueSet(entry->root_instruction(), output_index)
                      .values(),
                  [&](const HloValue* output) {
                    return AliasNeedsCopy(input, *output);
                  })) {
            changed = true;
            return absl::OkStatus();
          }
        }
        return kept.SetUpAlias(output_index, alias.parameter_number,
                               alias.parameter_index, alias.kind);
      }));
  if (changed) {
    aliases = std::move(kept);
  }
  return changed;
}

}  // namespace xla
