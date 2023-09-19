/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/all_gather_combiner.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/collective_combiner_utils.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace {

// Determine the preferred all-gather dimension.
// In case we want to combine all-gathers with different dimensions, prefer the
// symbolic representation for the case in which the major-most dimension of the
// layout is gathered.
int64_t PreferredAllGatherDim(const HloAllGatherInstruction* const all_gather,
                              bool combine_major_most_layout_dim) {
  if (!combine_major_most_layout_dim) {
    return all_gather->all_gather_dimension();
  }
  if (all_gather->all_gather_dimension() ==
      HloAllGatherInstruction::kMajorMostLayoutDimension) {
    return HloAllGatherInstruction::kMajorMostLayoutDimension;
  }

  // If all operands gather the major-most dimension of the layout, use a
  // symbolic dimension representation.
  bool is_all_operands_major_most_layout_dim = absl::c_all_of(
      all_gather->operands(), [&](const HloInstruction* operand) {
        return operand->shape().has_layout() &&
               LayoutUtil::Major(operand->shape().layout(), 0) ==
                   all_gather->all_gather_dimension();
      });
  return is_all_operands_major_most_layout_dim
             ? HloAllGatherInstruction::kMajorMostLayoutDimension
             : all_gather->all_gather_dimension();
}

// Combines the elements of to_combine into a single AllGather op. All entries
// in to_combine must be AllGather ops with exactly one operand and the same
// preferred all_gather_dimension.
Status CombineAllGathers(absl::Span<HloInstruction* const> to_combine,
                         bool combine_major_most_layout_dim) {
  if (to_combine.size() < 2) {
    return OkStatus();
  }
  VLOG(1) << "Combined " << to_combine.size() << " AllGather ops";

  HloComputation& computation = *to_combine.back()->parent();
  int64_t all_gather_dimension =
      PreferredAllGatherDim(Cast<HloAllGatherInstruction>(to_combine.front()),
                            combine_major_most_layout_dim);

  // Create a single bigger AllGather of the operands of the smaller AllGather.
  std::vector<HloInstruction*> operands;
  std::vector<const Shape*> output_shapes;
  VLOG(1) << "Combining set";
  for (HloInstruction* hlo : to_combine) {
    VLOG(1) << "Set element: " << hlo->ToString();
    TF_RET_CHECK(hlo->opcode() == HloOpcode::kAllGather);
    TF_RET_CHECK(hlo->operands().size() == 1);
    TF_RET_CHECK(PreferredAllGatherDim(Cast<HloAllGatherInstruction>(hlo),
                                       combine_major_most_layout_dim) ==
                 all_gather_dimension);

    TF_RET_CHECK(hlo->shape().IsArray());
    for (HloInstruction* operand : hlo->operands()) {
      operands.push_back(operand);
      output_shapes.push_back(&hlo->shape());
    }
  }

  HloInstruction* combined;
  // AllGather ops with more than one operand produce a tuple.
  TF_RET_CHECK(operands.size() >= 2);
  combined = computation.AddInstruction(HloInstruction::CreateAllGather(
      ShapeUtil::MakeTupleShapeWithPtrs(output_shapes), operands,
      all_gather_dimension, to_combine.front()->replica_groups(),
      /*constrain_layout=*/false, to_combine.front()->channel_id(),
      Cast<HloAllGatherInstruction>(to_combine.front())
          ->use_global_device_ids()));

  // We have to propagate the sharding manually because Domain instructions are
  // not guaranteed to preserve it for side effecting instructions.
  combined->set_sharding(
      hlo_sharding_util::CreateTupleSharding(combined->shape(), to_combine));
  VLOG(1) << "Replacing with : " << combined->ToString();

  // Replace all the smaller AllGathers with elements of the tuple output
  // of the single bigger AllGather.
  for (int64_t i = 0; i < to_combine.size(); ++i) {
    auto replace_with = HloInstruction::CreateGetTupleElement(
        to_combine[i]->shape(), combined, i);
    TF_RETURN_IF_ERROR(computation.ReplaceWithNewInstruction(
        to_combine[i], std::move(replace_with)));
  }
  return OkStatus();
}

// The group key encapsulates all of the properties which must match for it to
// be possible to combine the instructions.
using GroupKey =
    std::tuple<int64_t, int64_t, bool, bool, std::vector<std::vector<int64_t>>>;

// Returns a key that will be equal for instructions that might be combined, or
// different if not.
std::optional<GroupKey> CombineKey(const HloInstruction* instruction,
                                   const HloDomainMap& domain_map,
                                   bool combine_major_most_layout_dim) {
  if (instruction->opcode() != HloOpcode::kAllGather) {
    return std::nullopt;
  }

  const auto* ag = Cast<HloAllGatherInstruction>(instruction);

  std::vector<std::vector<int64_t>> replica_groups;
  replica_groups.reserve(ag->replica_groups().size());
  for (const ReplicaGroup& replica_group : ag->replica_groups()) {
    replica_groups.push_back(
        std::vector<int64_t>(replica_group.replica_ids().begin(),
                             replica_group.replica_ids().end()));
  }

  return GroupKey{PreferredAllGatherDim(ag, combine_major_most_layout_dim),
                  domain_map.GetDomainMetadataId(ag),
                  ag->channel_id().has_value(), ag->use_global_device_ids(),
                  replica_groups};
}

}  // namespace

AllGatherCombiner::AllGatherCombiner(int64_t combine_threshold_in_bytes,
                                     int64_t combine_threshold_count,
                                     bool combine_major_most_layout_dim)
    : combine_threshold_in_bytes_(combine_threshold_in_bytes),
      combine_threshold_count_(combine_threshold_count),
      combine_major_most_layout_dim_(combine_major_most_layout_dim) {}

StatusOr<bool> AllGatherCombiner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Running AllGatherCombiner with threshold of "
          << combine_threshold_in_bytes_ << " bytes";

  if (combine_threshold_in_bytes_ <= 0 || combine_threshold_count_ <= 0) {
    VLOG(1) << "Skip AllGatherCombiner because the threshold is zero";
    return false;
  }

  if (hlo_query::ContainsLayoutConstrainedCollective(*module,
                                                     HloOpcode::kAllGather)) {
    VLOG(1) << "Skip AllGatherCombiner because the module contains "
               "all-gather with constrained layouts";
    return false;
  }

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(auto domain_map, HloDomainMap::Create(computation, ""));

    auto key_fn = [&](const HloInstruction* instruction) {
      return CombineKey(instruction, *domain_map,
                        combine_major_most_layout_dim_);
    };
    auto combine_fn =
        [&](absl::Span<HloInstruction* const> to_combine) -> Status {
      return CombineAllGathers(to_combine, combine_major_most_layout_dim_);
    };

    TF_ASSIGN_OR_RETURN(
        bool computation_changed,
        CombineInstructionsByKey<GroupKey>(computation, key_fn, combine_fn,
                                           combine_threshold_in_bytes_,
                                           combine_threshold_count_));
    changed |= computation_changed;
  }

  return changed;
}

}  // namespace xla
