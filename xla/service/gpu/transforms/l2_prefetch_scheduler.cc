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

HloInstruction* SkipBitcasts(HloInstruction& hlo) {
  if (HloPredicateIsOp<HloOpcode::kBitcast>(&hlo)) {
    return SkipBitcasts(*hlo.mutable_operand(0));
  }
  return &hlo;
}

HloInstruction* SkipBitcastsAndGTEs(HloInstruction& hlo) {
  if (HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kGetTupleElement>(
          &hlo)) {
    return SkipBitcastsAndGTEs(*hlo.mutable_operand(0));
  }
  return &hlo;
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

    TF_ASSIGN_OR_RETURN(
        bool moved, CommandBufferScheduling::MoveParametersAndConstantsToFront(
                        computation));
    changed |= moved;

    TF_ASSIGN_OR_RETURN(moved, MoveGTEsRightAfterTupleDefinition(computation));
    changed |= moved;

    HloInstructionSequence& sequence =
        schedule.GetOrCreateSequence(computation);

    for (HloInstruction* done : computation->instructions()) {
      int64_t max_prefetch_size = default_max_prefetch_size_;
      const auto& prefetch_size_attr =
          done->get_frontend_attribute(kPrefetchOpportunity);
      // TODO: RS?
      if (HloPredicateIsOp<HloOpcode::kAllReduceDone, HloOpcode::kAllGatherDone,
                           HloOpcode::kAsyncDone>(done)) {
        if (prefetch_size_attr.value_or("") == "") {
          continue;
        }
      } else {
        continue;
      }

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
      absl::flat_hash_map<HloInstruction*, absl::flat_hash_set<HloInstruction*>>
          targets;

      int64_t seen_bytes = 0;
      int64_t prefetch_size = 0;

      auto stop = [&]() {
        if (to_prefetch.size() >= kMaxNumPrefetchBuffers) {
          return true;
        }
        if (seen_bytes >= max_cache_traffic_) {
          return true;
        }
        if (prefetch_size >= max_prefetch_size) {
          return true;
        }
        return false;
      };

      bool past_done = false;
      bool past_start = false;
      absl::flat_hash_set<const HloInstruction*> to_ignore;
      HloInstruction* start = done->mutable_operand(0);
      for (HloInstruction* operand : start->operands()) {
        to_ignore.insert(SkipBitcasts(*operand));
      }
      to_ignore.insert(done);
      HloInstruction* last_fusion_before_start = nullptr;
      for (auto it = sequence.instructions().begin();
           it != sequence.instructions().end(); ++it) {
        if (*it == start) {
          past_start = true;
        }
        if (!past_start && HloPredicateIsOp<HloOpcode::kFusion>(*it)) {
          last_fusion_before_start = *it;
        }
        if (stop()) {
          break;
        }
        HloInstruction* successor = *it;
        if (!past_done) {
          if (successor == done) {
            past_done = true;
            if (*(it - 1) != start) {
              // skip already overlapped ops
              break;
            }
          }
          continue;
        }
        if (NoNeedToPrefetchFor(*successor)) {
          continue;
        }
        to_ignore.insert(successor);
        for (HloInstruction* operand : successor->operands()) {
          if (targets.contains(operand)) {
            targets[operand].insert(successor);
            continue;
          }
          if (to_ignore.contains(SkipBitcastsAndGTEs(*operand))) {
            continue;
          }
          if (!LayoutUtil::IsDenseArray(operand->shape())) {
            continue;
          }
          operand = SkipBitcasts(*operand);
          // TODO: first collect all meaningful candidates, then prioritize
          seen_bytes += ShapeUtil::ByteSizeOfElements(operand->shape());
          to_prefetch.push_back(operand);
          targets[operand].insert(successor);
          prefetch_size += ShapeUtil::ByteSizeOfElements(operand->shape());
          if (stop()) {
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
      std::vector<HloInstruction*> params;
      params.reserve(to_prefetch.size());
      std::vector<HloInstruction*> cc_inputs;
      cc_inputs.reserve(to_prefetch.size());
      prefetch_size = 0;
      for (const HloInstruction* input : to_prefetch) {
        params.push_back(builder.AddInstruction(HloInstruction::CreateParameter(
            params.size(), input->shape(), "p")));
        if (ShapeUtil::ByteSizeOfElements(input->shape()) >
            max_prefetch_size - prefetch_size) {
          HloInstruction* bitcast =
              builder.AddInstruction(HloInstruction::CreateBitcast(
                  ShapeUtil::MakeShape(input->shape().element_type(),
                                       {ShapeUtil::ElementsIn(input->shape())},
                                       {false}),
                  params.back()));
          const int slice_element_count = (max_prefetch_size - prefetch_size) /
                                          ShapeUtil::ByteSizeOfPrimitiveType(
                                              bitcast->shape().element_type());
          // TODO: it's probably better to prefetch some uniform subset rather
          // than the beginning
          cc_inputs.push_back(
              builder.AddInstruction(HloInstruction::CreateSlice(
                  ShapeUtil::MakeShape(bitcast->shape().element_type(),
                                       {slice_element_count}, {false}),
                  bitcast, {0}, {slice_element_count}, {1})));
        } else {
          cc_inputs.push_back(params.back());
        }
        prefetch_size += ShapeUtil::ByteSizeOfElements(params.back()->shape());
      }
      HloInstruction* custom_call =
          builder.AddInstruction(HloInstruction::CreateCustomCall(
              ShapeUtil::MakeNil(), cc_inputs, kL2Prefetch));
      Cast<HloCustomCallInstruction>(custom_call)
          ->set_custom_call_has_side_effect(true);
      custom_call->set_frontend_attribute(kNumBlocks,
                                          std::to_string(num_blocks));
      builder.AddInstruction(HloInstruction::CreateTuple(params));
      HloComputation* prefetch_computation =
          module->AddEmbeddedComputation(builder.Build());
      HloInstruction* fusion =
          computation->AddInstruction(HloInstruction::CreateFusion(
              prefetch_computation->root_instruction()->shape(),
              HloInstruction::FusionKind::kCustom, to_prefetch,
              prefetch_computation));
      // FIXME
      TF_RETURN_IF_ERROR(
          start->mutable_operand(0)->AddControlDependencyTo(fusion));
      if (last_fusion_before_start) {
        TF_RETURN_IF_ERROR(
            last_fusion_before_start->AddControlDependencyTo(fusion));
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
