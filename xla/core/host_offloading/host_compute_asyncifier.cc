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

#include "xla/core/host_offloading/host_compute_asyncifier.h"

#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/casts.h"
#include "xla/core/host_offloading/hlo_host_device_type_call_wrapper.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/layout.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

// Removes tiles and memory spaces from all instructions inside `computation`.
void RemoveTilesAndMemorySpaces(HloComputation* computation) {
  for (HloInstruction* instruction : computation->instructions()) {
    VLOG(1) << absl::StreamFormat(
        "Removing tiles and memory spaces from \"%s\".",
        instruction->ToString());
    ShapeUtil::ForEachMutableSubshape(
        instruction->mutable_shape(),
        [](Shape* subshape, const ShapeIndex& subshape_index) {
          if (!subshape->has_layout()) {
            return;
          }
          Layout* layout = subshape->mutable_layout();
          layout->clear_tiles();
          layout->set_memory_space(Layout::kDefaultMemorySpace);
        });
  }
}

// Clones `computation` into `module`, preserving its instruction schedule if
// scheduled.
HloComputation* CloneComputation(HloModule* module,
                                 HloComputation* computation) {
  if (!module->has_schedule() ||
      !module->schedule().is_computation_scheduled(computation)) {
    return module->AddEmbeddedComputation(
        computation->Clone(/*suffix=*/"host"));
  }
  auto [clone, clone_sequence] =
      computation->CloneWithSchedule(/*suffix=*/"host");
  HloComputation* clone_ptr = module->AddEmbeddedComputation(std::move(clone));
  module->schedule().set_sequence(clone_ptr, clone_sequence);
  return clone_ptr;
}

// Specializes computations in `execution_threads` that are reachable via both
// pure non-host call paths and paths containing at least one host device-type
// call, ensuring each computation has at most two versions:
// - Non-host version: all call paths from the entry computation consist solely
//   of non-host device-type calls.
// - Host version: every call path from the entry computation contains at least
//   one host device-type call.
void SpecializeComputationsForHost(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    absl::FunctionRef<bool(HloInstruction*)> is_host_call) {
  if (!module->has_entry_computation()) {
    return;
  }

  absl::flat_hash_set<const HloComputation*> non_host_comps;
  absl::flat_hash_set<const HloComputation*> host_comps;
  non_host_comps.insert(module->entry_computation());

  std::vector<HloComputation*> post_order =
      module->MakeComputationPostOrder(execution_threads);
  for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
    HloComputation* computation = *it;
    if (computation == module->entry_computation()) {
      continue;
    }

    absl::InlinedVector<HloInstruction*, 1> callers =
        computation->caller_instructions();
    bool is_non_host_comp = false;
    bool is_host_comp = false;
    for (HloInstruction* caller : callers) {
      const HloComputation* caller_comp = caller->parent();
      const bool host_call = is_host_call(caller);
      if (non_host_comps.contains(caller_comp) && !host_call) {
        is_non_host_comp = true;
      }
      if (host_comps.contains(caller_comp) || host_call) {
        is_host_comp = true;
      }
    }

    if (is_non_host_comp && is_host_comp) {
      HloComputation* host_clone = CloneComputation(module, computation);
      non_host_comps.insert(computation);
      host_comps.insert(host_clone);

      // Sort callers by unique_id for deterministic topological sort edge
      // insertion.
      absl::c_sort(callers,
                   [](const HloInstruction* a, const HloInstruction* b) {
                     return a->unique_id() < b->unique_id();
                   });
      for (HloInstruction* caller : callers) {
        const HloComputation* caller_comp = caller->parent();
        if (host_comps.contains(caller_comp) || is_host_call(caller)) {
          caller->ReplaceCalledComputations([&](HloComputation* callee) {
            return callee == computation ? host_clone : callee;
          });
        }
      }
    } else if (is_non_host_comp) {
      non_host_comps.insert(computation);
    } else if (is_host_comp) {
      host_comps.insert(computation);
    }
  }
}

}  // namespace

absl::StatusOr<bool> HostComputeAsyncifier::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto is_host_call = [&](HloInstruction* instr) {
    return instr->opcode() == HloOpcode::kCall &&
           backend_config_device_type_is_host_fn_(instr);
  };

  SpecializeComputationsForHost(module, execution_threads, is_host_call);

  bool modified = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* call : computation->instructions()) {
      if (!is_host_call(call)) {
        continue;
      }
      VLOG(1) << "Host Call: " << call->name();

      HloComputation* parent_computation = call->parent();
      HloComputation* host_computation = call->called_computations().front();
      HloCallInstruction* call_instr =
          absl::down_cast<HloCallInstruction*>(call);
      CHECK_NE(call_instr, nullptr);

      ABSL_ASSIGN_OR_RETURN(
          HloCallInstruction * call_instr_no_tuple_operands,
          HloHostDeviceTypeCallWrapper::RemoveTupleParameters(call_instr));
      ABSL_RETURN_IF_ERROR(TupleSimplifier().Run(module).status());
      ABSL_ASSIGN_OR_RETURN(
          HloInstruction * call_instr_no_constants,
          HloHostDeviceTypeCallWrapper::MaterializeConstantsOnHostComputation(
              call_instr_no_tuple_operands));

      VLOG(1) << "Call instruction without constants: "
              << call_instr_no_constants->name();

      TF_RET_CHECK(call_instr_no_constants->operands().size() ==
                   host_computation->num_parameters())
          << "Expected the number of operands to match the number of "
             "parameters "
             "of the host called computation.";
      TF_RET_CHECK(
          ShapeUtil::Equal(host_computation->root_instruction()->shape(),
                           call_instr_no_constants->shape()))
          << "Shape mismatch between the host computation and the "
             "corresponding "
             "host call.";

      ABSL_ASSIGN_OR_RETURN(
          HloInstruction * async_done,
          parent_computation->CreateAsyncInstructions(
              call_instr_no_constants, {ShapeUtil::MakeScalarShape(U32)},
              HloInstruction::kHostThread,
              /*replace=*/true, /*override_names=*/true));
      if (call_instr_no_constants->has_frontend_attributes()) {
        HloInstruction* async_start = async_done->async_chain_start();
        async_start->set_frontend_attributes(
            call_instr_no_constants->frontend_attributes());
        async_done->set_frontend_attributes(
            call_instr_no_constants->frontend_attributes());
      }
      VLOG(1) << "Turning " << call_instr_no_constants->name()
              << " into an async instruction " << async_done->name();

      VLOG(1) << "Replacing" << call_instr_no_constants->name() << " with "
              << async_done->name();
      ABSL_RETURN_IF_ERROR(
          call_instr_no_constants->ReplaceAllUsesWith(async_done));

      RemoveTilesAndMemorySpaces(host_computation);

      modified = true;
    }
  }

  if (modified) {
    ABSL_RETURN_IF_ERROR(HloDCE().Run(module).status());

    if (module->has_schedule()) {
      ABSL_RETURN_IF_ERROR(module->schedule().Update());
    }
  }
  return modified;
}

}  // namespace xla
