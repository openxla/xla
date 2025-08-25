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

#include "xla/service/gpu/transforms/l2_prefetch_scheduler.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape_util.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/transforms/command_buffer_scheduling.h"
#include "xla/service/gpu/kernels/prefetch_kernel_common.h"

namespace xla::gpu {

namespace {

bool NoNeedToPrefetchFor(const HloInstruction& hlo) {
  return HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kGetTupleElement,
                          HloOpcode::kTuple>(&hlo);
}

HloInstruction* SkipBitcasts(HloInstruction* hlo) {
  if (HloPredicateIsOp<HloOpcode::kBitcast>(hlo)) {
    return SkipBitcasts(hlo->mutable_operand(0));
  }
  return hlo;
}

HloInstruction* SkipBitcastsAndGTEs(HloInstruction* hlo) {
  if (HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kGetTupleElement>(hlo)) {
    return SkipBitcastsAndGTEs(hlo->mutable_operand(0));
  }
  return hlo;
}

constexpr absl::string_view kPrefetchOpportunity = "l2_prefetch_opportunity";
constexpr absl::string_view kNumBlocks = "prefetch_num_blocks";

}  // namespace

absl::StatusOr<bool> L2PrefetchScheduler::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  HloSchedule& schedule = module->schedule();

  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    // Create async pairs for annotated custom calls.
    for (HloInstruction* hlo : computation->instructions()) {
      if (HloPredicateIsOp<HloOpcode::kCustomCall>(hlo) &&
          hlo->get_frontend_attribute(kPrefetchOpportunity).value_or("") !=
              "") {
        TF_ASSIGN_OR_RETURN(
            hlo, computation->CreateAsyncInstructions(hlo, {}, "async"));
        changed = true;
      }
    }

    if (changed) {
      TF_RETURN_IF_ERROR(schedule.Update(execution_threads));
    }

    // Move parameters and constants to the front to simplify the analysis.
    TF_ASSIGN_OR_RETURN(
        bool moved, CommandBufferScheduling::MoveParametersAndConstantsToFront(
                        computation));
    changed |= moved;

    // Same for GTEs.
    TF_ASSIGN_OR_RETURN(moved, MoveGTEsRightAfterTupleDefinition(computation));
    changed |= moved;

    HloInstructionSequence& sequence =
        schedule.GetOrCreateSequence(computation);

    for (HloInstruction* done : computation->instructions()) {
      const auto& prefetch_size_attr =
          done->get_frontend_attribute(kPrefetchOpportunity);
      if (HloPredicateIsOp<HloOpcode::kAllReduceDone, HloOpcode::kAllGatherDone,
                           HloOpcode::kAsyncDone>(done)) {
        // Skip instructions without annotation or with empty one.
        if (prefetch_size_attr.value_or("") == "") {
          continue;
        }
      } else {
        continue;
      }

      int64_t max_prefetch_size = default_max_prefetch_size_;
      if (prefetch_size_attr.has_value()) {
        if (!absl::SimpleAtoi(prefetch_size_attr.value(), &max_prefetch_size)) {
          continue;
        }
      }

      int num_blocks;
      if (!absl::SimpleAtoi(
              done->get_frontend_attribute(kNumBlocks).value_or(""),
              &num_blocks)) {
        num_blocks = kPrefetchDefaultBlocks;
      }
      if (num_blocks <= 0) {
        continue;
      }

      std::vector<HloInstruction*> to_prefetch;
      // Prefetched HLO value -> its users. Used to add control dependencies.
      absl::flat_hash_map<HloInstruction*, absl::flat_hash_set<HloInstruction*>>
          targets;

      // Traverse the schedule to find prefetchable instructions.

      int64_t seen_bytes = 0;
      int64_t prefetch_operands_total_bytes = 0;

      auto stop_schedule_traversal = [&]() {
        if (to_prefetch.size() >= kMaxNumPrefetchBuffers) {
          return true;
        }
        if (seen_bytes >= max_cache_traffic_) {
          return true;
        }
        if (prefetch_operands_total_bytes >= max_prefetch_size) {
          return true;
        }
        return false;
      };

      bool past_done = false;
      bool past_start = false;
      absl::flat_hash_set<const HloInstruction*> not_to_prefetch;
      HloInstruction* start = done->mutable_operand(0);
      // Operands of the prefetch host will get into L2 because host reads them.
      for (HloInstruction* operand : start->operands()) {
        not_to_prefetch.insert(SkipBitcasts(operand));
      }
      not_to_prefetch.insert(done);
      HloInstruction* last_non_noop_before_start = nullptr;

      for (auto it = sequence.instructions().begin();
           it != sequence.instructions().end(); ++it) {
        if (*it == start) {
          past_start = true;
        }
        if (!past_start && HloPredicateIsOp<HloOpcode::kFusion>(*it)) {
          last_non_noop_before_start = *it;
        }
        if (stop_schedule_traversal()) {
          break;
        }
        HloInstruction* successor = *it;
        if (!past_done) {
          if (successor == done) {
            past_done = true;
            if (*(it - 1) != start) {
              // Skip operations already overlapped with anything else.
              break;
            }
          }
          continue;
        }
        if (NoNeedToPrefetchFor(*successor)) {
          continue;
        }
        // Successor outputs will get into L2 as successors execute.
        not_to_prefetch.insert(successor);
        for (HloInstruction* candidate : successor->operands()) {
          if (targets.contains(candidate)) {
            targets[candidate].insert(successor);
            continue;
          }
          if (not_to_prefetch.contains(SkipBitcastsAndGTEs(candidate))) {
            continue;
          }
          if (!candidate->shape().IsArray()) {
            continue;
          }
          candidate = SkipBitcasts(candidate);
          seen_bytes += ShapeUtil::ByteSizeOfElements(candidate->shape());
          to_prefetch.push_back(candidate);
          targets[candidate].insert(successor);
          prefetch_operands_total_bytes +=
              ShapeUtil::ByteSizeOfElements(candidate->shape());
          if (stop_schedule_traversal()) {
            break;
          }
        }

        seen_bytes += ShapeUtil::ByteSizeOf(
            successor->shape(), HloCostAnalysis::kDefaultPointerSize);
      }

      if (to_prefetch.empty()) {
        continue;
      }

      HloComputation::Builder builder(kL2Prefetch);
      std::vector<HloInstruction*> fusion_parameters;
      fusion_parameters.reserve(to_prefetch.size());
      std::vector<HloInstruction*> custom_call_inputs;
      custom_call_inputs.reserve(to_prefetch.size());
      int prefetch_total_bytes = 0;
      for (const HloInstruction* input : to_prefetch) {
        fusion_parameters.push_back(
            builder.AddInstruction(HloInstruction::CreateParameter(
                fusion_parameters.size(), input->shape(), "p")));
        if (ShapeUtil::ByteSizeOfElements(input->shape()) >
            max_prefetch_size - prefetch_total_bytes) {
          // Slice an input to fit into the prefetch size limit.
          HloInstruction* bitcast =
              builder.AddInstruction(HloInstruction::CreateBitcast(
                  ShapeUtil::MakeShape(input->shape().element_type(),
                                       {ShapeUtil::ElementsIn(input->shape())},
                                       {false}),
                  fusion_parameters.back()));
          const int slice_element_count =
              (max_prefetch_size - prefetch_total_bytes) /
              ShapeUtil::ByteSizeOfPrimitiveType(
                  bitcast->shape().element_type());
          custom_call_inputs.push_back(
              builder.AddInstruction(HloInstruction::CreateSlice(
                  ShapeUtil::MakeShape(bitcast->shape().element_type(),
                                       {slice_element_count}, {false}),
                  bitcast, {0}, {slice_element_count}, {1})));
        } else {
          custom_call_inputs.push_back(fusion_parameters.back());
        }
        prefetch_total_bytes +=
            ShapeUtil::ByteSizeOfElements(fusion_parameters.back()->shape());
      }
      HloInstruction* custom_call =
          builder.AddInstruction(HloInstruction::CreateCustomCall(
              ShapeUtil::MakeNil(), custom_call_inputs, kL2Prefetch));
      Cast<HloCustomCallInstruction>(custom_call)
          ->set_custom_call_has_side_effect(true);
      custom_call->set_frontend_attribute(kNumBlocks,
                                          std::to_string(num_blocks));
      HloComputation* prefetch_computation =
          module->AddEmbeddedComputation(builder.Build());
      HloInstruction* fusion =
          computation->AddInstruction(HloInstruction::CreateFusion(
              prefetch_computation->root_instruction()->shape(),
              HloInstruction::FusionKind::kCustom, to_prefetch,
              prefetch_computation));
      // Control dependencies constrain prefetch to start with the host op
      // and end before any of its targets start.
      for (HloInstruction* operand : start->mutable_operands()) {
        TF_RETURN_IF_ERROR(operand->AddControlDependencyTo(fusion));
      }
      if (last_non_noop_before_start) {
        TF_RETURN_IF_ERROR(
            last_non_noop_before_start->AddControlDependencyTo(fusion));
      }
      for (HloInstruction* operand : to_prefetch) {
        for (HloInstruction* target : targets[operand]) {
          TF_RETURN_IF_ERROR(fusion->AddControlDependencyTo(target));
        }
      }
      TF_ASSIGN_OR_RETURN(auto gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
      gpu_config.mutable_fusion_backend_config()->set_kind(
          std::string(kL2Prefetch));
      TF_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

      const std::vector<int64_t>& ids = sequence.ids();
      const int64_t done_index =
          std::find(ids.begin(), ids.end(), done->unique_id()) - ids.begin();
      sequence.insert_instruction(fusion, done_index);

      changed = true;
    }
  }

  if (changed) {
    TF_RETURN_IF_ERROR(schedule.Update(execution_threads));
  }

  return changed;
}

}  // namespace xla::gpu
