/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/backends/gpu/hlo/transforms/reduce_scatter_creator.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_opt_utils.h"
#include "xla/service/decision.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace gpu {
namespace {

// Decides whether `instruction` is a broadcast of a scalar that can be
// re-created with a different shape. A broadcast with control dependencies
// cannot, as the new broadcast would not be ordered by them.
Decision CanRecreateScalarBroadcast(const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kBroadcast ||
      !ShapeUtil::IsScalar(instruction->operand(0)->shape())) {
    return Decision::Forbid("not a broadcast of a scalar");
  }
  if (instruction->HasControlDependencies()) {
    return Decision::Forbid("the scalar broadcast has control dependencies");
  }
  return Decision::Allow();
}

// Decides whether a dynamic-slice of `user` can be replaced by `user` applied
// to a dynamic-slice of `operand`. This is the case if `user` only applies a
// per-element function to `operand`: it is either a copy, or an elementwise
// operation whose remaining operands are all broadcasts of scalars.
Decision CanSliceCommuteWithUser(const HloInstruction* user,
                                 const HloInstruction* operand) {
  if (user->HasSideEffect()) {
    return Decision::Forbid("the user has side effects");
  }
  if (user->HasControlDependencies()) {
    return Decision::Forbid("the user has control dependencies");
  }
  if (!user->shape().IsArray() || user->shape().is_dynamic()) {
    return Decision::Forbid("the user does not have a static array shape");
  }
  if (!ShapeUtil::SameDimensions(user->shape(), operand->shape())) {
    return Decision::Forbid("the user changes the dimensions");
  }
  if (user->opcode() == HloOpcode::kCopy) {
    return Decision::Allow();
  }
  if (!user->IsElementwise()) {
    return Decision::Forbid("the user is neither a copy nor elementwise");
  }
  if (user->OperandIndices(operand).size() != 1) {
    return Decision::Forbid("the user consumes the operand more than once");
  }
  if (!absl::c_all_of(user->operands(), [&](const HloInstruction* other) {
        return other == operand ||
               CanRecreateScalarBroadcast(other).IsAllowed();
      })) {
    return Decision::Forbid(
        "another operand of the user is not a scalar broadcast without "
        "control dependencies");
  }
  return Decision::Allow();
}

// Rewrites
//   dynamic-slice(f_n(...f_1(all-reduce)...), offsets)
// into
//   f_n(...f_1(dynamic-slice(all-reduce, offsets))...)
// where every f_i is a copy or an elementwise operation whose other operands
// are scalar broadcasts, and every instruction of the chain has a single user.
// The SPMD partitioner emits such chains, for example when a gradient is scaled
// before it is sliced to the shard of the current partition. Moving the
// dynamic-slice next to the all-reduce exposes the reduce-scatter pattern, and
// the elementwise operations then run on the slice only.
absl::StatusOr<bool> SinkDynamicSliceThroughElementwise(
    HloAllReduceInstruction* ar) {
  if (!ar->shape().IsArray() || ar->shape().is_dynamic()) {
    return false;
  }
  std::vector<HloInstruction*> chain;
  HloInstruction* current = ar;
  HloInstruction* ds = nullptr;
  while (current->user_count() == 1) {
    HloInstruction* user = *current->users().begin();
    if (user->opcode() == HloOpcode::kDynamicSlice) {
      if (user->operand(0) == current &&
          user->OperandIndices(current).size() == 1 &&
          !user->HasControlDependencies() && !user->shape().is_dynamic()) {
        ds = user;
      }
      break;
    }
    Decision can_commute = CanSliceCommuteWithUser(user, current);
    if (can_commute.IsForbidden()) {
      VLOG(2) << "Cannot move a dynamic-slice above " << user->name() << ": "
              << can_commute.Explain();
      break;
    }
    chain.push_back(user);
    current = user;
  }
  if (ds == nullptr || chain.empty()) {
    return false;
  }

  HloComputation* computation = ar->parent();
  // The new shapes get the default layout. This pass runs before layout
  // assignment, where every shape still has the default layout.
  auto sliced_shape = [&](const HloInstruction* instruction) {
    return ShapeUtil::MakeShape(instruction->shape().element_type(),
                                ds->shape().dimensions());
  };
  std::vector<HloInstruction*> ds_operands(ds->operands().begin(),
                                           ds->operands().end());
  ds_operands[0] = ar;
  HloInstruction* result = computation->AddInstruction(
      ds->CloneWithNewOperands(sliced_shape(ar), ds_operands));
  HloInstruction* previous = ar;
  for (HloInstruction* instruction : chain) {
    std::vector<HloInstruction*> operands;
    operands.reserve(instruction->operand_count());
    for (HloInstruction* operand : instruction->operands()) {
      if (operand == previous) {
        operands.push_back(result);
      } else {
        // Clone the scalar broadcast to keep its metadata.
        operands.push_back(
            computation->AddInstruction(operand->CloneWithNewOperands(
                sliced_shape(operand), {operand->mutable_operand(0)})));
      }
    }
    result = computation->AddInstruction(
        instruction->CloneWithNewOperands(sliced_shape(instruction), operands));
    previous = instruction;
  }
  ABSL_RETURN_IF_ERROR(ds->ReplaceAllUsesWith(result));
  ABSL_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(ds));
  return true;
}

}  // namespace

absl::StatusOr<bool> ReduceScatterCreator::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const HloModuleConfig& config = module->config();
  int64_t next_channel_id = hlo_query::NextChannelId(*module);

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (HloPredicateIsNotOp<HloOpcode::kAllReduce>(instruction)) {
        continue;
      }
      auto* ar = Cast<HloAllReduceInstruction>(instruction);
      // The dynamic-slice is moved next to the all-reduce before matching, and
      // deliberately stays there even if no reduce-scatter is created below
      // (for example because the slice offset is not a per-device shard
      // offset). The move is exact and never increases the amount of work: the
      // elementwise operations run on the slice instead of the full all-reduce
      // result.
      ABSL_ASSIGN_OR_RETURN(bool sunk, SinkDynamicSliceThroughElementwise(ar));
      changed |= sunk;
      auto ar_spec = MatchReduceScatter(ar, config.num_partitions(),
                                        config.replica_count(),
                                        /*allow_multiple_split_dims=*/false,
                                        /*allow_intervening_reshape=*/true);
      if (!ar_spec) {
        VLOG(2) << "Cannot match reduce-scatter " << ar->ToString();
        continue;
      }

      HloInstruction* ds = ar_spec->dynamic_slice;

      // Convert to all-reduce scatter. The output shape of the all-reduce
      // scatter will the same as the input shape, except the split dim size is
      // that of the result of the dynamic slice.
      const int64_t split_dim = ar_spec->split_dim;
      Shape scatter_shape = ar->shape();
      const int64_t split_dim_size = scatter_shape.dimensions(split_dim);
      HloInstruction* rs_input = ar->mutable_operand(0);
      const int64_t scatter_dim_size = split_dim_size / ar_spec->group_size;
      TF_RET_CHECK(scatter_dim_size * ar_spec->group_size <= split_dim_size);
      if (split_dim_size % ar_spec->group_size != 0) {
        // The dynamic-slice does not evenly split the scatter dim. In that
        // case, create a reduce-scatter with the relevant slice of the
        // all-reduce input.
        scatter_shape.set_dimensions(split_dim,
                                     scatter_dim_size * ar_spec->group_size);
        rs_input = computation->AddInstruction(HloInstruction::CreateSlice(
            scatter_shape, rs_input,
            std::vector<int64_t>(scatter_shape.dimensions().size(), 0),
            scatter_shape.dimensions(),
            std::vector<int64_t>(scatter_shape.dimensions().size(), 1)));
      }
      scatter_shape.set_dimensions(split_dim, scatter_dim_size);

      std::optional<int64_t> channel_id;
      if (ar->channel_id()) {
        // We cannot reuse the channel_id on all-reduce for reduce-scatter.
        channel_id = next_channel_id++;
      }

      HloInstruction* ars =
          computation->AddInstruction(HloInstruction::CreateReduceScatter(
              scatter_shape, {rs_input}, ar->to_apply(), ar->device_list(),
              ar->constrain_layout(), channel_id, ar->use_global_device_ids(),
              ar_spec->split_dim));
      ar->SetupDerivedInstruction(ars);

      // SetupDerivedInstruction only copies backend configuration for matching
      // opcodes. At this point in the GPU pipeline, all-reduce and
      // reduce-scatter use compatible CollectiveBackendConfig fields, so
      // explicitly preserve the source collective configuration.
      ars->CopyBackendConfigFrom(ar);

      // If there was an intervening reshape, reshape the non-split dimensions
      // to match that existing reshape. Basically we can just reshape the ars
      // result to the dynamic slice shape.
      HloInstruction* result = ars;
      HloInstruction* reshape = nullptr;
      if (ds->operand(0) != ar) {
        reshape = ds->mutable_operand(0);
        result = computation->AddInstruction(
            HloInstruction::CreateReshape(ds->shape(), result));
      }

      // Note that RemoveInstructionAndUnusedOperands may not always remove the
      // all-reduce operand of the dynamic-slice, so remove all the dead
      // instructions manually.
      ABSL_RETURN_IF_ERROR(ds->ReplaceAllUsesWith(result));
      ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(ds));
      if (reshape) {
        ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(reshape));
      }
      ABSL_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(ar));
      changed = true;
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
