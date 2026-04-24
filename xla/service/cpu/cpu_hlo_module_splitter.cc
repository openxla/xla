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

#include "xla/service/cpu/cpu_hlo_module_splitter.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {

namespace {

bool FrontendAttributesAllowInlining(const HloInstruction* instruction) {
  auto it = instruction->frontend_attributes().map().find("inlineable");
  if (it != instruction->frontend_attributes().map().end()) {
    return it->second == "true";
  }
  return true;
}

absl::StatusOr<HloInstruction*> CreateBoundaryCopy(HloComputation* comp,
                                                   HloInstruction* inst) {
  if (inst->shape().IsToken()) {
    return inst;
  }
  if (!inst->shape().IsTuple()) {
    return comp->AddInstruction(
        HloInstruction::CreateUnary(inst->shape(), HloOpcode::kCopy, inst));
  }

  ShapeTree<bool> indices_to_copy(inst->shape(), true);
  ShapeUtil::ForEachSubshape(
      inst->shape(), [&](const Shape& s, const ShapeIndex& index) {
        if (s.IsToken()) {
          *indices_to_copy.mutable_element(index) = false;
        }
      });
  return comp->DeepCopyInstruction(inst, &indices_to_copy);
}

}  // namespace

absl::StatusOr<bool> CpuHloModuleSplitter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  submodules_.clear();
  bool changed = false;
  absl::flat_hash_map<HloComputation*, HloModule*> extracted_modules;
  absl::flat_hash_set<std::string> used_names;

  // We use post-order to process callees before callers.
  std::vector<HloComputation*> computations =
      module->MakeComputationPostOrder(execution_threads);

  for (HloComputation* comp : computations) {
    std::vector<HloInstruction*> instructions =
        comp->MakeInstructionPostOrder();
    for (HloInstruction* inst : instructions) {
      if (inst->opcode() == HloOpcode::kCall &&
          !FrontendAttributesAllowInlining(inst)) {
        HloComputation* callee = inst->to_apply();

        if (extracted_modules.find(callee) == extracted_modules.end()) {
          std::string base_name(callee->name());
          std::string name = base_name;
          int suffix = 0;
          while (used_names.contains(name)) {
            name = absl::StrCat(base_name, ".", ++suffix);
          }
          used_names.insert(name);

          auto sub_module = std::make_unique<HloModule>(name, module->config());
          HloCloneContext context(sub_module.get());
          HloComputation* cloned_callee =
              sub_module->DeepCloneComputation(callee, &context);
          sub_module->ReplaceEntryComputation(cloned_callee);
          extracted_modules[callee] = sub_module.get();
          submodules_.push_back(std::move(sub_module));
        }

        // Replace call with custom-call.
        std::vector<HloInstruction*> operands;
        for (HloInstruction* operand : inst->operands()) {
          TF_ASSIGN_OR_RETURN(HloInstruction * copy,
                              CreateBoundaryCopy(comp, operand));
          operands.push_back(copy);
        }

        auto* custom_call = Cast<HloCustomCallInstruction>(
            comp->AddInstruction(HloInstruction::CreateCustomCall(
                inst->shape(), operands, "__xla_cpu_multi_module_call",
                /*opaque=*/extracted_modules[callee]->name(),
                CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED)));

        if (callee->HasSideEffect()) {
          custom_call->set_custom_call_has_side_effect(true);
        }

        TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(custom_call));
        TF_RETURN_IF_ERROR(comp->RemoveInstruction(inst));
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace xla::cpu
