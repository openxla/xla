/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/priority_fusion.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_dfs_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/transforms/fusion_cost_model.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/map_util.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/fusion_process_dump.pb.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class PriorityFusionQueue {
  using Priority = absl::Duration;

 public:
  PriorityFusionQueue(PriorityFusion* parent, HloComputation* computation,
                      FusionCostModel* cost_model, const AliasInfo* alias_info,
                      tsl::thread::ThreadPool* thread_pool,
                      FusionProcessDumpProto* fusion_process_dump)
      : cost_model_(cost_model),
        computation_(computation),
        thread_pool_(thread_pool),
        fusion_process_dump_(fusion_process_dump),
        alias_info_(alias_info),
        parent_(parent),
        reachability_(HloDfsReachability::Build(computation)) {
    dump_fusion_visualization_ = computation->parent()
                                     ->config()
                                     .debug_options()
                                     .xla_dump_fusion_visualization();

    (void)parent_->cost_model()->Prepare(computation);
    std::vector<HloInstruction*> instructions;
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (HloPredicateIsOp<HloOpcode::kParameter>(instruction) ||
          instruction->user_count() == 0 || !parent_->IsFusible(*instruction) ||
          HloPredicateIsOp<HloOpcode::kTuple, HloOpcode::kGetTupleElement>(
              instruction)) {
        continue;
      }
      instructions.push_back(instruction);
    }
    ComputeAndSetPriorities(instructions);
  }

  void ComputeAndSetPriorities(
      const std::vector<HloInstruction*>& instructions) {
    std::vector<Priority> priorities = ComputePriorities(instructions);

    for (auto [instruction, priority] : llvm::zip(instructions, priorities)) {
      auto key = std::make_pair(priority, instruction->unique_id());

      auto reverse_it = reverse_map_.find(instruction);
      if (reverse_it != reverse_map_.end()) {
        const PriorityQueue::iterator& queue_it = reverse_it->second;
        if (key == queue_it->first) {
          continue;
        }
        producer_priority_queue_.erase(queue_it);
        reverse_map_.erase(reverse_it);
      }

      if (priority < absl::ZeroDuration()) {
        continue;
      }

      auto emplace_result = producer_priority_queue_.emplace(key, instruction);
      reverse_map_.emplace(instruction, emplace_result.first);
    }
  }

  std::vector<Priority> ComputePriorities(
      const std::vector<HloInstruction*>& instructions) {
    auto schedule_or_run = [this](std::function<void()> fn) {
      if (thread_pool_) {
        thread_pool_->Schedule(std::move(fn));
      } else {
        fn();
      }
    };
    absl::BlockingCounter counter(instructions.size());
    std::vector<Priority> priorities(instructions.size());

    for (size_t i = 0; i < instructions.size(); ++i) {
      schedule_or_run([&, i] {
        priorities[i] = CalculateProducerPriority(instructions[i]);
        counter.DecrementCount();
      });
    }
    counter.Wait();
    return priorities;
  }

  bool DequeueNextProducer() {
    current_producer_ = nullptr;
    current_consumers_.clear();

    while (!producer_priority_queue_.empty() && current_consumers_.empty()) {
      auto next_it = std::prev(producer_priority_queue_.end());

      current_producer_ = next_it->second;
      producer_priority_queue_.erase(next_it);
      reverse_map_.erase(current_producer_);

      current_consumers_ = current_producer_->users();
      auto preferred_consumer = GetPreferredConsumer(current_producer_);
      if (preferred_consumer) {
        current_consumers_ = {*preferred_consumer};
      }

      if (IsFusibleBitcast(*current_producer_)) {
        // We don't check if bitcasts can be fused with all consumers, so we
        // have to do it here.
        llvm::erase_if(current_consumers_, [&](HloInstruction* consumer) {
          return !CanFuseCached(current_producer_, consumer);
        });
      }
    }

    return !current_consumers_.empty();
  }

  std::optional<HloInstruction*> GetPreferredConsumer(
      HloInstruction* producer) {
    absl::MutexLock lock(preferred_consumer_mutex_);
    auto it = preferred_consumer_.find(producer);
    if (it == preferred_consumer_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  absl::Status UpdatePriorities() {
    std::vector<HloInstruction*> instructions(to_update_priority_.begin(),
                                              to_update_priority_.end());
    absl::c_sort(instructions,
                 [](const HloInstruction* a, const HloInstruction* b) {
                   return a->unique_id() < b->unique_id();
                 });

    for (auto instruction : instructions) {
      TF_RETURN_IF_ERROR(cost_model_->Revisit(instruction));
    }

    ComputeAndSetPriorities(instructions);

    to_update_priority_.clear();
    operands_to_new_consumers_.clear();
    operands_to_removed_consumers_runtimes_.clear();
    return absl::OkStatus();
  }

  void PreFusion(HloInstruction* producer, HloInstruction* consumer) {
    if (cost_model_) {
      cost_model_->PreInstructionFused(producer, consumer);
    }
    if (dump_fusion_visualization_) {
      RegisterFusionState(
          *computation_,
          absl::StrCat("About to fuse |", producer->name(), "| into |",
                       consumer->name(), "| inside PriorityFusion"),
          *consumer, producer);
    }
  }

  void InvalidateCaches(HloInstruction* instruction) {
    {
      absl::MutexLock lock(&can_fuse_cache_mutex_);
      can_fuse_cache_.erase(instruction);
      for (const HloInstruction* operand : instruction->operands()) {
        auto it = can_fuse_cache_.find(operand);
        if (it != can_fuse_cache_.end()) {
          it->second.erase(instruction);
        }
      }
    }
    cost_model_->Invalidate(instruction);
  }

  absl::Status ComputeRuntimesOfRemovedConsumers(
      HloInstruction* producer, const std::vector<HloInstruction*>& consumers) {
    absl::flat_hash_set<HloInstruction*> all_operands;
    for (auto* op : producer->operands()) {
      all_operands.insert(op);
    }
    for (auto* consumer : consumers) {
      for (auto* op : consumer->operands()) {
        all_operands.insert(op);
      }
    }

    std::vector<HloInstruction*> sorted_operands(all_operands.begin(),
                                                 all_operands.end());
    llvm::sort(sorted_operands,
               [](const HloInstruction* a, const HloInstruction* b) {
                 return a->unique_id() < b->unique_id();
               });

    for (auto* operand : sorted_operands) {
      if (!reverse_map_.contains(operand)) {
        continue;
      }
      std::vector<HloInstruction*> removed_consumers;
      for (auto* consumer : consumers) {
        if (absl::c_linear_search(consumer->operands(), operand)) {
          if (!IsFusibleBitcast(*consumer)) {
            removed_consumers.push_back(consumer);
          }
        }
      }
      if (absl::c_linear_search(producer->operands(), operand)) {
        if (!IsFusibleBitcast(*producer)) {
          removed_consumers.push_back(producer);
        }
      }
      TF_ASSIGN_OR_RETURN(
          FusionCostModel::RunTimes runtimes,
          cost_model_->EstimateRunTimes(operand, removed_consumers));
      operands_to_removed_consumers_runtimes_.emplace(operand, runtimes);
    }
    return absl::OkStatus();
  }

  void OnFusingInstruction(HloInstruction* fusion,
                           HloInstruction* original_producer,
                           HloInstruction* original_consumer,
                           absl::string_view original_producer_name,
                           absl::string_view original_consumer_name,
                           int64_t original_consumer_operand_index,
                           int64_t original_consumer_unique_id) {
    bool creates_multi_output_fusion = false;
    {
      absl::MutexLock lock(preferred_consumer_mutex_);
      creates_multi_output_fusion =
          preferred_consumer_.contains(original_producer);
    }

    if (fusion_process_dump_) {
      auto* fusion_step =
          fusion_process_dump_->add_fusion_steps()->mutable_fusion();
      fusion_step->set_fusion_name(fusion->name());
      fusion_step->set_producer_name(original_producer_name);
      fusion_step->set_consumer_name(original_consumer_name);
    }

    if (dump_fusion_visualization_) {
      VLOG(2) << "Fusing " << original_producer_name << " into "
              << original_consumer_name
              << " (kind: " << xla::ToString(fusion->fusion_kind()) << ")";

      RegisterFusionState(
          *computation_,
          absl::StrCat("Fused |", original_producer_name, "| into |",
                       fusion->name(), "| inside PriorityFusion"),
          *fusion);
    }

    if (fusion == original_consumer) {
      absl::MutexLock lock(preferred_consumer_mutex_);
      preferred_consumer_.erase(original_consumer);
    } else {
      reachability_->OnInstructionReplaced(/*previous=*/original_consumer,
                                           /*now=*/fusion);
      RemoveInstruction(original_consumer);
    }
    if (creates_multi_output_fusion) {
      reachability_ = HloDfsReachability::Build(computation_);
    }

    for (HloInstruction* operand : fusion->operands()) {
      if (operand == original_producer ||
          HloPredicateIsOp<HloOpcode::kConstant, HloOpcode::kGetTupleElement>(
              operand)) {
        continue;
      }
      if (!parent_->IsFusible(*operand)) {
        continue;
      }

      to_update_priority_.insert(operand);
      operands_to_new_consumers_[operand].push_back(fusion);

      absl::MutexLock lock(preferred_consumer_mutex_);
      auto it = preferred_consumer_.find(operand);
      if (it != preferred_consumer_.end() && it->second == original_consumer) {
        preferred_consumer_.erase(it);
      }
    }
    to_update_priority_.insert(fusion);
  }

  void RemoveInstruction(HloInstruction* instruction) {
    to_update_priority_.erase(instruction);

    auto reverse_it = reverse_map_.find(instruction);
    if (reverse_it == reverse_map_.end()) {
      return;
    }
    producer_priority_queue_.erase(reverse_it->second);
    reverse_map_.erase(reverse_it);
    absl::MutexLock lock(preferred_consumer_mutex_);
    preferred_consumer_.erase(instruction);
  }

  HloInstruction* current_producer() { return current_producer_; }

  const std::vector<HloInstruction*>& current_consumers() {
    return current_consumers_;
  }

 private:
  absl::Duration CalculateProducerPriority(HloInstruction* producer) {
    {
      absl::MutexLock lock(&preferred_consumer_mutex_);
      preferred_consumer_.erase(producer);
    }
    // Bitcasts should always be fused first, since they are no-ops.
    if (IsFusibleBitcast(*producer)) {
      return absl::InfiniteDuration();
    }
    // We always fuse constants, but the cost model doesn't handle them very
    // well: fusing constants changes costs significantly. Also, there's no
    // point recomputing priorities. Therefore, we fuse all of them at the end.
    if (producer->opcode() == HloOpcode::kConstant) {
      return -absl::InfiniteDuration();
    }

    bool has_non_bitcast_user = false;
    FusionDecision failed_decision = FusionDecision::Allow();
    for (auto* consumer : producer->users()) {
      if (IsFusibleBitcast(*consumer)) {
        continue;
      }
      has_non_bitcast_user = true;
      if (auto decision = CanFuseCached(producer, consumer); !decision) {
        failed_decision = decision;
        break;
      }
    }

    if (!failed_decision || !has_non_bitcast_user) {
      if (auto preferred_consumer = parent_->GetPreferredMultiOutputConsumer(
              producer, reachability_.get())) {
        std::vector<HloInstruction*> consumers = {*preferred_consumer};
        auto run_times_or = cost_model_->EstimateRunTimes(producer, consumers);
        if (!run_times_or.ok()) {
          VLOG(2) << "Cost model failed for producer " << producer->name()
                  << ": " << run_times_or.status().message();
          return -absl::InfiniteDuration();
        }
        FusionCostModel::RunTimes run_times = *run_times_or;
        {
          absl::MutexLock lock(&preferred_consumer_mutex_);
          preferred_consumer_[producer] = *preferred_consumer;
        }
        return run_times.unfused - run_times.fused;
      }

      if (fusion_process_dump_) {
        absl::MutexLock lock(&fusion_process_dump_mutex_);
        auto* step = fusion_process_dump_->add_fusion_steps()
                         ->mutable_producer_ineligible();
        step->set_producer_name(producer->name());
        step->set_reason(failed_decision.Explain());
      }
      return -absl::InfiniteDuration();
    }

    auto removed_consumers_runtime_it =
        operands_to_removed_consumers_runtimes_.find(producer);
    bool is_incremental_update =
        reverse_map_.contains(producer) &&
        removed_consumers_runtime_it !=
            operands_to_removed_consumers_runtimes_.end();

    absl::Duration current_priority = absl::ZeroDuration();
    if (is_incremental_update) {
      current_priority = reverse_map_.at(producer)->first.first;
    }

    std::vector<HloInstruction*> consumers;
    if (is_incremental_update) {
      auto it = operands_to_new_consumers_.find(producer);
      if (it != operands_to_new_consumers_.end()) {
        consumers = it->second;
      }
    } else {
      for (auto* user : producer->users()) {
        if (!IsFusibleBitcast(*user)) {
          consumers.push_back(user);
        }
      }
    }

    auto rt_or = cost_model_->EstimateRunTimes(producer, consumers);
    if (!rt_or.ok()) {
      VLOG(2) << "Cost model failed for producer " << producer->name() << ": "
              << rt_or.status().message();
      return -absl::InfiniteDuration();
    }
    FusionCostModel::RunTimes rt = *rt_or;

    if (is_incremental_update) {
      const FusionCostModel::RunTimes& removed_rt =
          removed_consumers_runtime_it->second;
      rt.unfused -= removed_rt.unfused;
      rt.fused -= removed_rt.fused;
    }

    absl::Duration priority = current_priority + rt.unfused - rt.fused;

    if (fusion_process_dump_) {
      absl::MutexLock lock(&fusion_process_dump_mutex_);
      auto* step =
          fusion_process_dump_->add_fusion_steps()->mutable_update_priority();
      step->set_producer_name(producer->name());
      for (auto* consumer : consumers) {
        step->add_consumer_names(std::string(consumer->name()));
      }
      step->set_us_fused(absl::ToDoubleMicroseconds(rt.fused));
      step->set_us_unfused(absl::ToDoubleMicroseconds(rt.unfused));
    }

    return priority;
  }

  FusionDecision CanFuse(HloInstruction* producer, HloInstruction* consumer) {
    if (producer == producer->parent()->root_instruction()) {
      return FusionDecision::Forbid(
          "not fusing into the output of the root instruction");
    }
    if (!parent_->IsFusible(*producer)) {
      return FusionDecision::Forbid("the producer is not fusible");
    }
    if (!parent_->IsFusible(*consumer)) {
      return FusionDecision::Forbid("the consumer is not fusible");
    }
    if (auto backend = parent_->BackendCanFuse(producer, consumer); !backend) {
      return backend;
    }
    if (cost_model_->WouldExplodeIrSize(producer, consumer)) {
      return FusionDecision::Forbid(
          "the fusion would result in an overly large code duplication");
    }
    return InstructionFusion::ShouldFuseInPlaceOp(producer, consumer,
                                                  alias_info_, std::nullopt);
  }

  FusionDecision CanFuseCached(HloInstruction* producer,
                               HloInstruction* consumer) {
    {
      absl::MutexLock lock(&can_fuse_cache_mutex_);
      auto& producer_cache = can_fuse_cache_[producer];

      auto it = producer_cache.find(consumer);
      if (it != producer_cache.end()) {
        return it->second;
      }
    }
    auto fusion_decision = CanFuse(producer, consumer);

    {
      absl::MutexLock lock(&can_fuse_cache_mutex_);
      can_fuse_cache_[producer].insert_or_assign(consumer, fusion_decision);
    }

    return fusion_decision;
  }

  bool OperandReachableFromProducer(const HloInstruction* producer,
                                    const HloInstruction* consumer) {
    return gpu::OperandReachableFromProducer(producer, consumer,
                                             reachability_.get());
  }

  FusionCostModel* cost_model_;  // not owned
  HloComputation* computation_;
  tsl::thread::ThreadPool* thread_pool_;
  FusionProcessDumpProto* fusion_process_dump_;
  const AliasInfo* alias_info_;

  using PriorityQueue = std::map<std::pair<Priority, int64_t>, HloInstruction*>;
  PriorityQueue producer_priority_queue_;

  absl::flat_hash_map<HloInstruction*, PriorityQueue::iterator> reverse_map_;

  absl::flat_hash_map<HloInstruction*, HloInstruction*> preferred_consumer_
      ABSL_GUARDED_BY(preferred_consumer_mutex_);
  absl::Mutex preferred_consumer_mutex_;

  HloInstruction* current_producer_;
  std::vector<HloInstruction*> current_consumers_;

  PriorityFusion* parent_;
  absl::flat_hash_set<HloInstruction*> to_update_priority_;
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
      operands_to_new_consumers_;
  absl::flat_hash_map<HloInstruction*, FusionCostModel::RunTimes>
      operands_to_removed_consumers_runtimes_;

  absl::Mutex fusion_process_dump_mutex_;

  absl::flat_hash_map<
      const HloInstruction*,
      absl::flat_hash_map<const HloInstruction*, FusionDecision>>
      can_fuse_cache_ ABSL_GUARDED_BY(can_fuse_cache_mutex_);
  absl::Mutex can_fuse_cache_mutex_;

  std::unique_ptr<HloDfsReachability> reachability_;

  bool dump_fusion_visualization_;
};

bool OperandReachableFromProducer(const HloInstruction* producer,
                                  const HloInstruction* consumer,
                                  HloDfsReachability* reachability) {
  for (const auto* consumer_operand : consumer->operands()) {
    CHECK(reachability->IsPresent(consumer_operand) &&
          reachability->IsPresent(producer))
        << "Reachability map is incomplete. This should never "
           "happen.";
    if (producer != consumer_operand &&
        reachability->IsReachable(producer, consumer_operand)) {
      return true;
    }
  }
  return false;
}

bool IsSmallConstant(const HloInstruction* instr) {
  return HloPredicateIsOp<HloOpcode::kConstant>(instr) &&
         instr->shape().IsArray() && ShapeUtil::ElementsIn(instr->shape()) <= 1;
}

PriorityFusion::PriorityFusion(tsl::thread::ThreadPool* thread_pool,
                               const AliasInfo* alias_info,
                               std::unique_ptr<FusionCostModel> cost_model)
    : alias_info_(alias_info),
      thread_pool_(thread_pool),
      cost_model_(std::move(cost_model)) {}

bool PriorityFusion::ConsumeFuel(HloInstruction* producer,
                                 HloInstruction* consumer) {
  return xla::ConsumeFuel(name(), [&] {
    return absl::StrFormat("Not fusing producer %s with consumer %s",
                           producer->name(), consumer->name());
  });
}

// Bitcasts are fusible if they don't change the bit width.
bool IsFusibleBitcast(const HloInstruction& instr) {
  return instr.opcode() == HloOpcode::kBitcast &&
         hlo_instruction_utils::KeepsBitwidth(instr);
}

// TODO(b/326639148): This logic is quite GPU-centric. For a truly shared base
// class, this list might eventually need to move to the cost model or a
// separate 'fusion policy' interface.
bool PriorityFusion::IsFusible(const HloInstruction& instr) {
  if (instr.IsCustomFusion()) {
    return false;
  }

  if (instr.IsElementwise()) {
    return true;
  }

  if (IsFusibleBitcast(instr)) {
    return true;
  }

  switch (instr.opcode()) {
    case HloOpcode::kFusion:
    case HloOpcode::kCopy:
    case HloOpcode::kIota:
    case HloOpcode::kConstant:
    case HloOpcode::kReduce:
    case HloOpcode::kBroadcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGather:
    case HloOpcode::kPad:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kScatter:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
      return true;
    default:
      return false;
  }
}

FusionDecision PriorityFusion::CanFuseConstant(HloInstruction* constant,
                                               HloInstruction* user) {
  if (auto fusion_decision = BackendCanFuse(constant, user); !fusion_decision) {
    return fusion_decision;
  }

  if (!IsFusible(*user)) {
    return FusionDecision::Forbid("User is not fusible");
  }

  return FusionDecision::Allow();
}

absl::StatusOr<bool> PriorityFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool dump_enabled =
      DumpingEnabledForHloPass(name(), module->config().debug_options());
  if (dump_enabled) {
    fusion_process_dump_ = std::make_unique<FusionProcessDumpProto>();
    PopulateFusionProcessDump(fusion_process_dump_.get());
  }

  auto fusible_computations = GetFusibleComputations(module, execution_threads);

  if (dump_enabled) {
    fusion_process_dump_->set_hlo_module_before_fusion(
        module->ToString(HloPrintOptions::ShortParsable()));
  }

  bool changed = false;
  for (auto* computation : fusible_computations) {
    CHECK(!computation->IsFusionComputation());

    TF_RETURN_IF_ERROR(cost_model_->Prepare(computation));

    PriorityFusionQueue fusion_queue(this, computation, cost_model_.get(),
                                     alias_info_, thread_pool_,
                                     fusion_process_dump_.get());

    while (fusion_queue.DequeueNextProducer()) {
      auto producer = fusion_queue.current_producer();

      auto preferred_consumer = fusion_queue.GetPreferredConsumer(producer);
      std::vector<HloInstruction*> consumers = fusion_queue.current_consumers();
      bool use_multi_output_fusion = preferred_consumer.has_value();

      absl::flat_hash_set<int64_t> pre_fusion_consumer_ids;

      TF_RETURN_IF_ERROR(
          fusion_queue.ComputeRuntimesOfRemovedConsumers(producer, consumers));

      for (auto* consumer : consumers) {
        if (IsFusibleBitcast(*consumer)) {
          continue;
        }
        if (!ConsumeFuel(producer, consumer)) {
          continue;
        }

        VLOG(5) << "next: " << consumer->name() << "(" << consumer << ") + "
                << producer->name() << "(" << producer << ")";

        int64_t consumer_operand_index = consumer->operand_index(producer);

        fusion_queue.PreFusion(producer, consumer);
        int64_t consumer_pre_fusion_id = consumer->unique_id();

        absl::string_view producer_name = producer->name();
        absl::string_view consumer_name = consumer->name();

        fusion_queue.InvalidateCaches(producer);
        for (auto* c : consumers) {
          fusion_queue.InvalidateCaches(c);
        }

        auto fusion_instruction =
            Fuse(producer, consumer, use_multi_output_fusion);

        fusion_queue.OnFusingInstruction(
            fusion_instruction, producer, consumer, producer_name,
            consumer_name, consumer_operand_index, consumer_pre_fusion_id);

        changed = true;
      }

      if (use_multi_output_fusion || producer->user_count() == 0) {
        fusion_queue.RemoveInstruction(producer);

        if (!use_multi_output_fusion && producer->parent() != nullptr) {
          producer->DetachFromOperandsAndUsers();
          TF_RETURN_IF_ERROR(computation->RemoveInstruction(producer));
        }
      }

      TF_RETURN_IF_ERROR(fusion_queue.UpdatePriorities());
    }

    std::vector<HloInstruction*> constants;
    for (auto* instruction : computation->instructions()) {
      if (IsSmallConstant(instruction)) {
        constants.push_back(instruction);
      }
    }

    for (auto* constant : constants) {
      auto users = constant->users();
      for (auto* user : users) {
        if (CanFuseConstant(constant, user)) {
          Fuse(constant, user, /*use_multi_output_fusion=*/false);
          changed = true;
        }
      }
    }
  }

  cost_model_->ClearCaches();

  if (dump_enabled) {
    DumpPerModuleProtobufToFile(*module, *fusion_process_dump_,
                                module->config().debug_options(),
                                "priority_fusion_dump");
  }

  return changed;
}

HloInstruction* PriorityFusion::Fuse(HloInstruction* producer,
                                     HloInstruction* consumer,
                                     bool use_multi_output_fusion) {
  VLOG(2) << "Fusing " << producer->ToString() << " into "
          << consumer->ToString();

  HloComputation* computation = consumer->parent();
  auto kind = ChooseKind(producer, consumer, use_multi_output_fusion);
  HloInstruction* fusion_instruction = consumer;

  if (HloPredicateIsNotOp<HloOpcode::kFusion>(fusion_instruction)) {
    fusion_instruction = computation->AddInstruction(
        HloInstruction::CreateFusion(consumer->shape(), kind, consumer));
    CHECK_OK(computation->ReplaceInstruction(consumer, fusion_instruction));
  } else if (kind != fusion_instruction->fusion_kind()) {
    fusion_instruction->set_fusion_kind(kind);
  }

  fusion_instruction->set_called_computations_execution_thread(
      computation->execution_thread());

  if (HloPredicateIsOp<HloOpcode::kFusion>(producer)) {
    if (use_multi_output_fusion) {
      fusion_instruction->MergeFusionInstructionIntoMultiOutput(producer);
    } else {
      fusion_instruction->MergeFusionInstruction(producer);
    }
  } else {
    if (use_multi_output_fusion) {
      fusion_instruction->FuseInstructionIntoMultiOutput(producer);
      CHECK_EQ(0, producer->user_count());
      CHECK_OK(producer->parent()->RemoveInstruction(producer));
    } else {
      fusion_instruction->FuseInstruction(producer);
    }
  }

  if (fusion_instruction != consumer) {
    VLOG(2) << "       created new fusion: " << fusion_instruction->ToString();
  }

  cost_model_->OnInstructionFused(producer, consumer, fusion_instruction);

  return fusion_instruction;
}

}  // namespace gpu
}  // namespace xla
