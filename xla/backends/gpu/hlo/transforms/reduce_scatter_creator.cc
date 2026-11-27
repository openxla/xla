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
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_opt_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace gpu {
namespace {

// Re-creates `chain`, the copies and elementwise operations between `ar` and
// the dynamic-slice that was matched as a reduce-scatter, on the result of
// `reduce_scatter`. Slicing commutes with these operations, so applying them
// to the shard produced by the reduce-scatter gives the same result as slicing
// their output. Returns the last re-created instruction.
HloInstruction* RecreateElementwiseOnReduceScatter(
    absl::Span<HloInstruction* const> chain, const HloInstruction* ar,
    HloInstruction* reduce_scatter, int64_t split_dim) {
  HloComputation* computation = reduce_scatter->parent();
  // The shape of `instruction` with the split dimension shrunk to the size of
  // the shard. The instructions in the chain and their scalar broadcasts have
  // the dimensions of `ar`, so only the split dimension changes. Copying the
  // shape keeps its layout, such as the element size of sub-byte types.
  auto scattered_shape = [&](const HloInstruction* instruction) {
    Shape shape = instruction->shape();
    shape.set_dimensions(split_dim,
                         reduce_scatter->shape().dimensions(split_dim));
    return shape;
  };
  HloInstruction* result = reduce_scatter;
  const HloInstruction* previous = ar;
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
                scattered_shape(operand), {operand->mutable_operand(0)})));
      }
    }
    result = computation->AddInstruction(instruction->CloneWithNewOperands(
        scattered_shape(instruction), operands));
    previous = instruction;
  }
  return result;
}

// Removes `chain`, which is dead after its dynamic-slice user was removed,
// together with the scalar broadcasts that are no longer used.
absl::Status RemoveElementwiseChain(absl::Span<HloInstruction* const> chain) {
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    HloInstruction* instruction = *it;
    HloComputation* computation = instruction->parent();
    std::vector<HloInstruction*> broadcasts;
    for (HloInstruction* operand : instruction->operands()) {
      if (operand->opcode() == HloOpcode::kBroadcast &&
          !absl::c_linear_search(broadcasts, operand)) {
        broadcasts.push_back(operand);
      }
    }
    ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
    for (HloInstruction* broadcast : broadcasts) {
      if (broadcast->IsDead()) {
        ABSL_RETURN_IF_ERROR(
            computation->RemoveInstructionAndUnusedOperands(broadcast));
      }
    }
  }
  return absl::OkStatus();
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
      auto ar_spec = MatchReduceScatter(
          ar, config.num_partitions(), config.replica_count(),
          /*allow_multiple_split_dims=*/false,
          /*allow_intervening_reshape=*/true, /*min_rank=*/1,
          HloPredicateIsOp<HloOpcode::kPartitionId>,
          HloPredicateIsOp<HloOpcode::kReplicaId>,
          /*allow_intervening_bitcast=*/false,
          /*allow_intervening_elementwise=*/true);
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

      const std::vector<HloInstruction*>& elementwise =
          ar_spec->intervening_elementwise;
      HloInstruction* result = ars;
      HloInstruction* reshape = nullptr;
      if (!elementwise.empty()) {
        // If there were intervening copies or elementwise operations, re-create
        // them on the result of the reduce-scatter.
        result =
            RecreateElementwiseOnReduceScatter(elementwise, ar, ars, split_dim);
      } else if (ds->operand(0) != ar) {
        // If there was an intervening reshape, reshape the non-split dimensions
        // to match that existing reshape. Basically we can just reshape the ars
        // result to the dynamic slice shape.
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
      ABSL_RETURN_IF_ERROR(RemoveElementwiseChain(elementwise));
      ABSL_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(ar));
      changed = true;
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
