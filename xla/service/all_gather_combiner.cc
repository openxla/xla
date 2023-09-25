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

#include <algorithm>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/service/collective_combiner_utils.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace {

// Combines the elements of to_combine into a single AllGather op. All entries
// in to_combine must be AllGather ops with exactly one operand and the same
// all_gather_dimension.
Status CombineAllGathers(absl::Span<HloInstruction* const> to_combine,
                         absl::Span<HloInstruction* const> to_combine_ends) {
  if (to_combine.size() < 2) {
    return OkStatus();
  }
  VLOG(1) << "Combined " << to_combine.size() << " AllGather ops";

  bool is_async = !to_combine_ends.empty();
  HloComputation& computation = *to_combine.back()->parent();
  int64_t all_gather_dimension =
      Cast<HloAllGatherInstruction>(to_combine.front())->all_gather_dimension();

  // Create a single bigger AllGather of the operands of the smaller AllGather.
  std::vector<HloInstruction*> operands;
  std::vector<const Shape*> output_shapes;
  VLOG(1) << "Combining set";
  for (HloInstruction* hlo : to_combine) {
    VLOG(1) << "Set element: " << hlo->ToString();
    TF_RET_CHECK(hlo->opcode() == (is_async ? HloOpcode::kAllGatherStart
                                            : HloOpcode::kAllGather));
    TF_RET_CHECK(hlo->operands().size() == 1);
    TF_RET_CHECK(Cast<HloAllGatherInstruction>(hlo)->all_gather_dimension() ==
                 all_gather_dimension);
    const Shape* shape;
    if (is_async) {
      TF_RET_CHECK(hlo->shape().IsTuple());
      TF_RET_CHECK(hlo->shape().tuple_shapes_size() == 2);
      shape = &hlo->shape().tuple_shapes(1);
    } else {
      shape = &hlo->shape();
    }
    TF_RET_CHECK(shape->IsArray());
    for (HloInstruction* operand : hlo->operands()) {
      operands.push_back(operand);
      output_shapes.push_back(shape);
    }
  }

  HloInstruction* combined;
  // AllGather ops with more than one operand produce a tuple.
  TF_RET_CHECK(operands.size() >= 2);
  auto create = [&](auto& f) {
    Shape shape;
    if (is_async) {
      std::vector<const Shape*> operand_shapes;
      for (HloInstruction* operand : operands) {
        operand_shapes.push_back(&operand->shape());
      }
      shape = ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeTupleShapeWithPtrs(operand_shapes),
           ShapeUtil::MakeTupleShapeWithPtrs(output_shapes)});
    } else {
      shape = ShapeUtil::MakeTupleShapeWithPtrs(output_shapes);
    }
    return computation.AddInstruction(
        f(shape, operands, all_gather_dimension,
          to_combine.front()->replica_groups(),
          /*constrain_layout=*/false, to_combine.front()->channel_id(),
          Cast<HloAllGatherInstruction>(to_combine.front())
              ->use_global_device_ids()));
  };
  combined = is_async ? create(HloInstruction::CreateAllGatherStart)
                      : create(HloInstruction::CreateAllGather);

  // We have to propagate the sharding manually because Domain instructions are
  // not guaranteed to preserve it for side effecting instructions.
  combined->set_sharding(
      hlo_sharding_util::CreateTupleSharding(combined->shape(), to_combine));
  VLOG(1) << "Replacing with : " << combined->ToString();

  HloInstruction* combined_end =
      is_async ? computation.AddInstruction(HloInstruction::CreateUnary(
                     combined->shape().tuple_shapes(1),
                     HloOpcode::kAllGatherDone, combined))
               : nullptr;

  return CombineCollectives(&computation, combined, combined_end, to_combine,
                            to_combine_ends, is_async);
}

// The group key encapsulates all of the properties which must match for it to
// be possible to combine the instructions.
using GroupKey =
    std::tuple<int64_t, int64_t, bool, bool, std::vector<std::vector<int64_t>>>;

// Returns a key that will be equal for instructions that might be combined, or
// different if not.
std::optional<GroupKey> CombineKey(const HloInstruction* instruction,
                                   const HloDomainMap& domain_map,
                                   bool is_async) {
  HloOpcode opcode =
      is_async ? HloOpcode::kAllGatherStart : HloOpcode::kAllGather;

  if (instruction->opcode() != opcode) {
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

  return GroupKey{ag->all_gather_dimension(),
                  domain_map.GetDomainMetadataId(ag),
                  ag->channel_id().has_value(), ag->use_global_device_ids(),
                  replica_groups};
}

}  // namespace

AllGatherCombiner::AllGatherCombiner(int64_t combine_threshold_in_bytes,
                                     int64_t combine_threshold_count,
                                     bool is_async,
                                     std::string_view async_strategy)
    : combine_threshold_in_bytes_(combine_threshold_in_bytes),
      combine_threshold_count_(combine_threshold_count),
      is_async_(is_async),
      async_strategy_(async_strategy == "near" ? kNear : kTrivial) {}

StatusOr<bool> AllGatherCombiner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Running " << name() << " with threshold of "
          << combine_threshold_in_bytes_ << " bytes";

  if (combine_threshold_in_bytes_ <= 0 || combine_threshold_count_ <= 0) {
    VLOG(1) << "Skip " << name() << " because the threshold is zero";
    return false;
  }

  if (hlo_query::ContainsLayoutConstrainedCollective(*module,
                                                     HloOpcode::kAllGather)) {
    VLOG(1) << "Skip " << name()
            << " because the module contains "
               "all-gather with constrained layouts";
    return false;
  }

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    HloModule* module = computation->parent();
    if (is_async_) {
      TF_RET_CHECK(module->has_schedule());
    }
    TF_ASSIGN_OR_RETURN(auto domain_map, HloDomainMap::Create(computation, ""));

    auto key_fn = [&domain_map, this](const HloInstruction* instruction) {
      return CombineKey(instruction, *domain_map, is_async_);
    };

    auto size_fn =
        [this](const HloInstruction* instruction) -> StatusOr<int64_t> {
      if (!is_async_) {
        return internal::SizeFromArrayShapedInstruction(instruction);
      }
      TF_RET_CHECK(instruction->opcode() == HloOpcode::kAllGatherStart);
      // AllGatherStart has a tuple shape: (input_shape, output_shape). We are
      // only interested in the output shape.
      TF_RET_CHECK(instruction->shape().IsTuple());
      TF_RET_CHECK(instruction->shape().tuple_shapes_size() == 2);
      const Shape& output_shape = instruction->shape().tuple_shapes(1);
      TF_RET_CHECK(output_shape.IsArray());
      return ShapeUtil::ByteSizeOf(output_shape);
    };

    TF_ASSIGN_OR_RETURN(
        bool computation_changed,
        CombineInstructionsByKey<GroupKey>(
            computation, key_fn, &CombineAllGathers,
            combine_threshold_in_bytes_, combine_threshold_count_, is_async_,
            async_strategy_, size_fn));
    changed |= computation_changed;
  }

  return changed;
}

}  // namespace xla
