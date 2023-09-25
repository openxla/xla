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

#include "xla/service/all_reduce_combiner.h"

#include <algorithm>
#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <string>
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
#include "xla/service/all_reduce_key.h"
#include "xla/service/collective_combiner_utils.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

// Combines the elements of to_combine into a single AllReduce op. All
// entries in to_combine must be AllReduce ops with exactly one operand
// and the same reduction operation.
Status CombineAllReduces(absl::Span<HloInstruction* const> to_combine,
                         absl::Span<HloInstruction* const> to_combine_ends) {
  if (to_combine.size() < 2) {
    return OkStatus();
  }
  VLOG(1) << "Combined " << to_combine.size() << " CRS ops";

  bool is_async = !to_combine_ends.empty();
  HloComputation& computation = *to_combine.back()->parent();
  HloComputation* reduction = to_combine[0]->to_apply();
  const HloOpcode type = reduction->root_instruction()->opcode();

  // Create a single bigger AllReduce of the operands of the smaller
  // AllReduces.
  std::vector<HloInstruction*> operands;
  std::vector<const Shape*> operand_shapes;
  VLOG(1) << "Combining set";
  for (HloInstruction* hlo : to_combine) {
    VLOG(1) << "Set element: " << hlo->ToString();
    TF_RET_CHECK(hlo->opcode() == (is_async ? HloOpcode::kAllReduceStart
                                            : HloOpcode::kAllReduce));
    TF_RET_CHECK(hlo->operands().size() == 1);
    TF_RET_CHECK(hlo->to_apply() == reduction ||
                 (hlo->to_apply()->instruction_count() == 3 &&
                  hlo->to_apply()->num_parameters() == 2 &&
                  hlo->to_apply()->root_instruction()->opcode() == type));
    TF_RET_CHECK(hlo->shape().IsArray());
    for (HloInstruction* operand : hlo->operands()) {
      operands.push_back(operand);
      operand_shapes.push_back(&operand->shape());
    }
  }

  HloInstruction* combined;
  // AllReduce ops with more than one operand produce a tuple.
  TF_RET_CHECK(operands.size() >= 2);
  auto create = [&](auto& f) {
    return computation.AddInstruction(
        f(ShapeUtil::MakeTupleShapeWithPtrs(operand_shapes), operands,
          reduction, to_combine.front()->replica_groups(),
          /*constrain_layout=*/false, to_combine.front()->channel_id(),
          Cast<HloAllReduceInstruction>(to_combine.front())
              ->use_global_device_ids()));
  };
  combined = is_async ? create(HloInstruction::CreateAllReduceStart)
                      : create(HloInstruction::CreateAllReduce);
  // We have to propagate the sharding manually because Domain instructions are
  // not guaranteed to preserve it for side effecting instructions.
  combined->set_sharding(
      hlo_sharding_util::CreateTupleSharding(combined->shape(), to_combine));
  VLOG(1) << "Replacing with : " << combined->ToString();

  HloInstruction* combined_end =
      is_async ? computation.AddInstruction(HloInstruction::CreateUnary(
                     combined->shape(), HloOpcode::kAllReduceDone, combined))
               : nullptr;

  return CombineCollectives(&computation, combined, combined_end, to_combine,
                            to_combine_ends, is_async);
}
}  // namespace

AllReduceCombiner::AllReduceCombiner(int64_t combine_threshold_in_bytes,
                                     int64_t combine_threshold_count,
                                     bool is_async,
                                     std::string_view async_strategy)
    : combine_threshold_in_bytes_(combine_threshold_in_bytes),
      combine_threshold_count_(combine_threshold_count),
      is_async_(is_async),
      async_strategy_(async_strategy == "near" ? kNear : kTrivial) {}

StatusOr<bool> AllReduceCombiner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Running " << name() << " with threshold of "
          << combine_threshold_in_bytes_ << " bytes";

  if (combine_threshold_in_bytes_ <= 0 || combine_threshold_count_ <= 0) {
    VLOG(1) << "Skip " << name() << " because the threshold is zero";
    return false;
  }

  if (hlo_query::ContainsLayoutConstrainedAllReduce(*module)) {
    VLOG(1) << "Skip " << name()
            << " because the module contains all-reduce "
               "with constrained layouts";
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

    auto key_fn =
        [&domain_map, this](
            const HloInstruction* instruction) -> std::optional<AllReduceKey> {
      HloOpcode opcode =
          is_async_ ? HloOpcode::kAllReduceStart : HloOpcode::kAllReduce;
      if (instruction->opcode() != opcode) {
        return std::nullopt;
      }
      return GetAllReduceKey(instruction, domain_map.get());
    };

    TF_ASSIGN_OR_RETURN(
        bool computation_changed,
        CombineInstructionsByKey<AllReduceKey>(
            computation, key_fn, &CombineAllReduces,
            combine_threshold_in_bytes_, combine_threshold_count_, is_async_,
            async_strategy_));
    changed |= computation_changed;
  }

  return changed;
}

}  // namespace xla
