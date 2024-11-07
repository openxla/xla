/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/analysis/hlo_replication_analysis.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/map_util.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Determines whether an HLO instruction is replicated at index based on current
// knowledge in hlo_replication. When cross_partition_spmd is true, the
// instruction must be replicated across all partitions on each replica.
// Similarly, when cross_partition_spmd is false, the instruction must be
// replicated across all replicas on ecah partition.
HloReplicationAnalysis::HloReplication
HloReplicationAnalysis::DetermineHloInstructionIsReplicated(
    const HloInstruction* hlo, const ShapeIndex& index,
    bool cross_partition_spmd,
    const absl::flat_hash_map<const HloInstruction*, ShapeTree<HloReplication>>&
        hlo_replication,
    bool support_partial_replication) {
  struct IDs {
    int64_t pid;
    int64_t rid;
  };
  const auto merge_operand_replication = [&hlo_replication](
                                             const HloInstruction* inst) {
    HloReplication replication = HloReplication::ReplicatedOnAllDevices();
    for (auto operand : inst->operands()) {
      auto operand_it = hlo_replication.find(operand);
      if (operand_it == hlo_replication.end()) {
        replication = replication.Merge(HloReplication::UniqueOnAllDevices());
      } else {
        replication = replication.Merge(operand_it->second.element({}));
      }
    }
    return replication;
  };

  if (hlo->opcode() == HloOpcode::kAllReduce ||
      hlo->opcode() == HloOpcode::kAllGather) {
    // All-reduce/all-gather returns same values across partitions/replicas as
    // long as its operands are replicated.
    HloReplication replication = merge_operand_replication(hlo);
    if (replication.IsReplicatedOnAllDevices()) {
      return replication;
    }
    if (!hlo->channel_id().has_value()) {
      // This is cross-replica-only.
      if (cross_partition_spmd) {
        return replication;
      }
      if (hlo->replica_groups().empty() || hlo->replica_groups().size() == 1) {
        return HloReplication::ReplicatedOnAllDevices();
      }
      if (support_partial_replication) {
        std::vector<std::vector<int64_t>> device_sets;
        for (const ReplicaGroup& replica_group : hlo->replica_groups()) {
          std::vector<int64_t> device_set;
          for (auto id : replica_group.replica_ids()) {
            device_set.push_back(id);
          }
          device_sets.push_back(device_set);
        }
        return HloReplication::PartiallyReplicated(device_sets);
      } else {
        return HloReplication::UniqueOnAllDevices();
      }
    } else {
      bool global_id;
      if (hlo->opcode() == HloOpcode::kAllReduce) {
        global_id = Cast<HloAllReduceInstruction>(hlo)->use_global_device_ids();
      } else {
        global_id = Cast<HloAllGatherInstruction>(hlo)->use_global_device_ids();
      }
      if (global_id) {
        const int64_t num_partitions =
            hlo->GetModule()->config().num_partitions();
        const int64_t replica_count =
            hlo->GetModule()->config().replica_count();
        // Holds the partition IDs of all replica groups in which a given
        // replica participates.
        std::vector<std::vector<absl::flat_hash_set<int64_t>>>
            partitions_for_replica(replica_count);
        // Holds the replica IDs of all replica groups in which a given
        // partition participates.
        std::vector<std::vector<absl::flat_hash_set<int64_t>>>
            replicas_for_partition(num_partitions);
        std::vector<std::vector<int64_t>> device_sets;
        for (const auto& group : hlo->replica_groups()) {
          std::vector<int64_t> device_set;
          std::vector<IDs> pids_and_rids;
          // Convert the global IDs in the replica group to partition and
          // replica IDs.
          for (int64_t id : group.replica_ids()) {
            int64_t rid = id / num_partitions;
            int64_t pid = id % num_partitions;
            pids_and_rids.emplace_back(IDs{pid, rid});
            if (support_partial_replication) {
              device_set.push_back(cross_partition_spmd ? pid : rid);
            }
          }
          if (support_partial_replication) {
            device_sets.push_back(std::move(device_set));
          }
          if (cross_partition_spmd) {
            // Collect the partition IDs from all devices in the current replica
            // group.
            for (int64_t rid = 0; rid < replica_count; ++rid) {
              absl::flat_hash_set<int64_t> partitions;
              for (IDs pid_and_rid : pids_and_rids) {
                if (pid_and_rid.rid == rid) {
                  partitions.insert(pid_and_rid.pid);
                }
              }
              if (!partitions.empty()) {
                partitions_for_replica[rid].push_back(std::move(partitions));
              }
            }
          } else {
            // Collect the replica IDs from all devices in the current replica
            // group.
            for (int64_t pid = 0; pid < num_partitions; ++pid) {
              absl::flat_hash_set<int64_t> replicas;
              for (IDs pid_and_rid : pids_and_rids) {
                if (pid_and_rid.pid == pid) {
                  replicas.insert(pid_and_rid.rid);
                }
              }
              if (!replicas.empty()) {
                replicas_for_partition[pid].push_back(std::move(replicas));
              }
            }
          }
        }
        bool fully_replicated_across_partitions = true;
        bool fully_replicated_across_replicas = true;
        if (cross_partition_spmd) {
          // In the fully replicated case, there is one set of partition IDs on
          // each replica. Since the flattened ID replica groups must contain
          // every device, the size of the set is the number of partitions.
          fully_replicated_across_partitions &=
              partitions_for_replica[0].size() == 1 &&
              partitions_for_replica[0][0].size() == num_partitions;
          for (auto partitions_for_current_replica : partitions_for_replica) {
            fully_replicated_across_partitions &=
                partitions_for_current_replica == partitions_for_replica[0];
          }
        } else {
          // In the fully replicated case, there is one set of replica IDs on
          // each partition. Since the flattened ID replica groups must contain
          // every device, the size of the set is the number of replicas.
          fully_replicated_across_replicas &=
              replicas_for_partition[0].size() == 1 &&
              replicas_for_partition[0][0].size() == replica_count;
          for (auto replicas_for_current_partition : replicas_for_partition) {
            fully_replicated_across_replicas &=
                replicas_for_current_partition == replicas_for_partition[0];
          }
        }
        if ((cross_partition_spmd && fully_replicated_across_partitions) ||
            (!cross_partition_spmd && fully_replicated_across_replicas)) {
          return HloReplication::ReplicatedOnAllDevices();
        } else if (support_partial_replication) {
          return HloReplication::PartiallyReplicated(device_sets);
        } else {
          return HloReplication::UniqueOnAllDevices();
        }
      }
      if (cross_partition_spmd) {
        return HloReplication::ReplicatedOnAllDevices();
      }
      if (hlo->replica_groups().empty() || hlo->replica_groups().size() == 1) {
        return HloReplication::ReplicatedOnAllDevices();
      } else {
        return HloReplication::UniqueOnAllDevices();
      }
    }
  }
  if (hlo->HasSideEffectNoRecurse()) {
    return HloReplication::UniqueOnAllDevices();
  }
  if (hlo->opcode() == HloOpcode::kReplicaId) {
    // ReplicaId returns the same value for all partitions in each replica.
    return cross_partition_spmd ? HloReplication::ReplicatedOnAllDevices()
                                : HloReplication::UniqueOnAllDevices();
  }
  if (hlo->opcode() == HloOpcode::kPartitionId) {
    // PartitionId returns the same value for all replicas in each partition.
    return cross_partition_spmd ? HloReplication::UniqueOnAllDevices()
                                : HloReplication::ReplicatedOnAllDevices();
  }
  auto it = hlo_replication.find(hlo);
  if (hlo->opcode() == HloOpcode::kParameter) {
    // Parameters should have been processed.
    CHECK(it != hlo_replication.end());
    return it->second.element(index);
  }
  if (it != hlo_replication.end() &&
      it->second.element(index).IsUniqueOnAllDevices()) {
    // The HLO is already marked as non-replicated.
    return it->second.element(index);
  }

  if (hlo->opcode() == HloOpcode::kConstant) {
    return HloReplication::ReplicatedOnAllDevices();
  }

  if (hlo->opcode() == HloOpcode::kCustomCall &&
      (hlo->custom_call_target() == "X64SplitLow" ||
       hlo->custom_call_target() == "X64SplitHigh" ||
       hlo->custom_call_target() == "X64Combine")) {
    return merge_operand_replication(hlo);
  }

  // Pattern-match and process cases where the HLO is partially replicated.
  if (support_partial_replication) {
    // Below is a very specific pattern to match the SPMD pipeline case.
    if (hlo->opcode() == HloOpcode::kDynamicSlice) {
      const HloInstruction* ds_buffer = hlo->operand(0);
      if (hlo->dynamic_slice_sizes().size() == 1 &&
          hlo->dynamic_slice_sizes()[0] == 1 &&
          ds_buffer->opcode() == HloOpcode::kConstant &&
          ds_buffer->shape().rank() == 1 &&
          ds_buffer->shape().element_type() == PrimitiveType::S32 &&
          ((cross_partition_spmd &&
            hlo->operand(1)->opcode() == HloOpcode::kPartitionId) ||
           (!cross_partition_spmd &&
            hlo->operand(1)->opcode() == HloOpcode::kReplicaId))) {
        const HloModule* hlo_module = hlo->GetModule();
        int64_t num_devices = cross_partition_spmd
                                  ? hlo_module->config().num_partitions()
                                  : hlo_module->config().replica_count();
        absl::flat_hash_map<int64_t, std::vector<int64_t>> value_to_device_set;
        for (int64_t device_id = 0; device_id < num_devices; ++device_id) {
          std::optional<int64_t> value =
              ds_buffer->literal().GetIntegralAsS64({device_id});
          value_to_device_set[*value].push_back(device_id);
        }
        std::vector<std::vector<int64_t>> device_sets;
        for (const auto& value_and_device_set : value_to_device_set) {
          device_sets.push_back(value_and_device_set.second);
        }
        return HloReplication::PartiallyReplicated(device_sets);
      }
    }
  }

  if (hlo->IsElementwise() ||                             //
      hlo->opcode() == HloOpcode::kConcatenate ||         //
      hlo->opcode() == HloOpcode::kConvolution ||         //
      hlo->opcode() == HloOpcode::kDot ||                 //
      hlo->opcode() == HloOpcode::kReduce ||              //
      hlo->opcode() == HloOpcode::kBroadcast ||           //
      hlo->opcode() == HloOpcode::kTranspose ||           //
      hlo->opcode() == HloOpcode::kReshape ||             //
      hlo->opcode() == HloOpcode::kBitcast ||             //
      hlo->opcode() == HloOpcode::kReverse ||             //
      hlo->opcode() == HloOpcode::kGather ||              //
      hlo->opcode() == HloOpcode::kScatter ||             //
      hlo->opcode() == HloOpcode::kIota ||                //
      hlo->opcode() == HloOpcode::kPad ||                 //
      hlo->opcode() == HloOpcode::kSlice ||               //
      hlo->opcode() == HloOpcode::kDynamicSlice ||        //
      hlo->opcode() == HloOpcode::kDynamicUpdateSlice ||  //
      hlo->opcode() == HloOpcode::kReduceWindow ||        //
      hlo->opcode() == HloOpcode::kCopy) {
    return merge_operand_replication(hlo);
  }
  return HloReplication::UniqueOnAllDevices();
}

bool HloReplicationAnalysis::ComputeHloReplicationOnComputation(
    const HloComputation* computation, bool mark_everything_not_replicated) {
  bool changed = false;
  for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
    // Assigns the shape tree to dest if dest doesn't have one yet, or combines
    // it with the existing one by and'ing them. Returns if anything is updated.
    auto assign_or_combine_shapetree =
        [&](ShapeTree<HloReplication>&& to_combine,
            const HloInstruction* dest) {
          auto it = hlo_replication_.find(dest);
          if (it == hlo_replication_.end()) {
            hlo_replication_[dest] = std::move(to_combine);
            return true;
          }
          bool updated = false;
          it->second.ForEachMutableElement(
              [&](const ShapeIndex& index, HloReplication* element) {
                HloReplication new_replication =
                    element->Merge(to_combine.element(index));
                if (!element->Equal(new_replication)) {
                  *element = std::move(new_replication);
                  updated = true;
                }
              });
          return updated;
        };
    // Assigns or combines source's shape tree to dest. Returns if anything is
    // updated.
    auto propagate_shapetree = [&](const HloInstruction* source,
                                   const HloInstruction* dest) {
      auto source_it = hlo_replication_.find(source);
      if (source_it == hlo_replication_.end()) {
        return false;
      }
      return assign_or_combine_shapetree(
          ShapeTree<HloReplication>(source_it->second), dest);
    };
    // For the opcodes below that we do special handling, we don't need to
    // explicitly check mark_everything_not_replicated because if it is set, the
    // operands should already be marked as not replicated.
    if (inst->opcode() == HloOpcode::kWhile) {
      // Since while body's input and output alias each other, we need to run it
      // multiple times until a fixed point is reached.
      while (true) {
        // First, propagate the input's and body root's shape trees to the
        // parameters of the body and condition.
        bool updated = propagate_shapetree(
            inst->operand(0),
            inst->while_condition()->parameter_instruction(0));
        updated |= propagate_shapetree(
            inst->while_body()->root_instruction(),
            inst->while_condition()->parameter_instruction(0));
        updated |= propagate_shapetree(
            inst->operand(0), inst->while_body()->parameter_instruction(0));
        updated |=
            propagate_shapetree(inst->while_body()->root_instruction(),
                                inst->while_body()->parameter_instruction(0));
        // Compute the condition.
        updated |= ComputeHloReplicationOnComputation(
            inst->while_condition(), mark_everything_not_replicated);
        // Compute the body. If the condition is not replicated, the while body
        // should be different across replicas.
        if (!ContainsKey(loops_known_with_same_iterations_, inst) &&
            !hlo_replication_[inst->while_condition()->root_instruction()]
                 .element({})
                 .IsReplicatedOnAllDevices()) {
          updated |= ComputeHloReplicationOnComputation(
              inst->while_body(), /*mark_everything_not_replicated=*/true);
        } else {
          updated |= ComputeHloReplicationOnComputation(
              inst->while_body(), mark_everything_not_replicated);
        }
        if (!updated) {
          break;
        }
        changed = true;
      }
      // Propagate the input's and body root's shape trees to the while HLO.
      changed |= propagate_shapetree(inst->operand(0), inst);
      changed |=
          propagate_shapetree(inst->while_body()->root_instruction(), inst);
    } else if (inst->opcode() == HloOpcode::kCall ||
               inst->opcode() == HloOpcode::kFusion) {
      auto called = inst->called_computations().front();
      for (int64_t i = 0; i < inst->operand_count(); ++i) {
        changed |= propagate_shapetree(inst->operand(i),
                                       called->parameter_instruction(i));
      }
      changed |= ComputeHloReplicationOnComputation(
          called, mark_everything_not_replicated);
      changed |= propagate_shapetree(called->root_instruction(), inst);
    } else if (inst->opcode() == HloOpcode::kConditional) {
      // Propagate inputs' shape trees to the called computations' parameters.
      for (int64_t i = 0; i < inst->called_computations().size(); ++i) {
        changed |= propagate_shapetree(
            inst->operand(i + 1),
            inst->called_computations()[i]->parameter_instruction(0));
      }
      // If the condition is not replicated, the conditional result should be
      // different across replicas.
      if (!hlo_replication_[inst->operand(0)]
               .element({})
               .IsReplicatedOnAllDevices()) {
        for (auto called : inst->called_computations()) {
          changed |= ComputeHloReplicationOnComputation(
              called,
              /*mark_everything_not_replicated=*/true);
        }
        changed |= assign_or_combine_shapetree(
            ShapeTree<HloReplication>(inst->shape(),
                                      HloReplication::UniqueOnAllDevices()),
            inst);
      } else {
        for (auto called : inst->called_computations()) {
          changed |= ComputeHloReplicationOnComputation(
              called, mark_everything_not_replicated);
          changed |= propagate_shapetree(called->root_instruction(), inst);
        }
      }
    } else if (inst->opcode() == HloOpcode::kTuple) {
      ShapeTree<HloReplication> shape_tree(
          inst->shape(), HloReplication::ReplicatedOnAllDevices());
      for (int64_t i = 0; i < inst->operand_count(); ++i) {
        shape_tree.CopySubtreeFrom(hlo_replication_[inst->operand(i)], {}, {i});
      }
      changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
    } else if (inst->opcode() == HloOpcode::kOptimizationBarrier) {
      ShapeTree<HloReplication> shape_tree = hlo_replication_[inst->operand(0)];
      changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
    } else if (inst->opcode() == HloOpcode::kGetTupleElement) {
      ShapeTree<HloReplication> shape_tree(
          inst->shape(), HloReplication::ReplicatedOnAllDevices());
      shape_tree.CopySubtreeFrom(hlo_replication_[inst->operand(0)],
                                 {inst->tuple_index()}, {});
      changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
    } else if (inst->opcode() == HloOpcode::kInfeed && cross_partition_spmd_) {
      ShapeTree<HloReplication> shape_tree(
          inst->shape(), HloReplication::UniqueOnAllDevices());
      if (inst->has_sharding()) {
        auto sharding = inst->sharding().GetAsShapeTree(inst->shape());
        shape_tree.ForEachMutableElement(
            [&sharding](const ShapeIndex& index, HloReplication* data) {
              *data = sharding.element(index).IsReplicated()
                          ? HloReplication::ReplicatedOnAllDevices()
                          : HloReplication::UniqueOnAllDevices();
            });
      }
      changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
    } else {
      if (mark_everything_not_replicated) {
        changed |= assign_or_combine_shapetree(
            ShapeTree<HloReplication>(inst->shape(),
                                      HloReplication::UniqueOnAllDevices()),
            inst);
      } else {
        ShapeTree<HloReplication> shape_tree(
            inst->shape(), HloReplication::ReplicatedOnAllDevices());
        ShapeUtil::ForEachSubshape(
            inst->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
              *shape_tree.mutable_element(index) =
                  DetermineHloInstructionIsReplicated(
                      inst, index, cross_partition_spmd_, hlo_replication_,
                      support_partial_replication_);
            });
        changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
      }
    }
  }
  return changed;
}

absl::Status HloReplicationAnalysis::ComputeHloReplication() {
  // Add entry parameters to the above sets according to user annotation.
  // Replicated modules read from `parameter_replicated_at_leaf_buffers` whereas
  // SPMD partitioned modules read from HloSharding attributes.
  auto entry = module_->entry_computation();
  for (int i = 0; i < entry->num_parameters(); ++i) {
    auto param = entry->parameter_instruction(i);
    ShapeTree<HloReplication> shape_tree(param->shape(),
                                         HloReplication::UniqueOnAllDevices());
    const auto& replication = param->parameter_replicated_at_leaf_buffers();
    int leaf_index = 0;
    absl::Status status = ShapeUtil::ForEachSubshapeWithStatus(
        param->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (!ShapeUtil::IsLeafIndex(param->shape(), index)) {
            return absl::OkStatus();
          }
          if (cross_partition_spmd_ && param->has_sharding()) {
            // In cross-partition spmd mode, set parameter replication status
            // based on the parameter's sharding.
            TF_ASSIGN_OR_RETURN(auto sharding_tree,
                                param->sharding().AsShapeTree(param->shape()));
            *shape_tree.mutable_element(index) =
                sharding_tree.element(index).IsReplicated()
                    ? HloReplication::ReplicatedOnAllDevices()
                    : HloReplication::UniqueOnAllDevices();
          }
          if (replication) {
            // If parameter replication status has been set explicitly, use that
            // instead.
            if (!cross_partition_spmd_ && (*replication)[leaf_index]) {
              // Setting parameter replication status for replicas in
              // non cross-partition spmd mode.
              *shape_tree.mutable_element(index) =
                  HloReplication::ReplicatedOnAllDevices();
            }
            if (cross_partition_spmd_ && !(*replication)[leaf_index]) {
              // Setting paramemter replication status for partitions in
              // cross-partition spmd mode.
              *shape_tree.mutable_element(index) =
                  HloReplication::UniqueOnAllDevices();
            }
            ++leaf_index;
          }
          return absl::OkStatus();
        });
    TF_RETURN_IF_ERROR(status);
    hlo_replication_[param] = std::move(shape_tree);
  }
  ComputeHloReplicationOnComputation(entry,
                                     /*mark_everything_not_replicated=*/false);
  return absl::OkStatus();
}

bool HloReplicationAnalysis::HloInstructionIsReplicatedAt(
    const HloInstruction* inst, const ShapeIndex& index) const {
  auto it = hlo_replication_.find(inst);
  if (it == hlo_replication_.end()) {
    return false;
  }
  return it->second.element(index).IsReplicatedOnAllDevices();
}

bool HloReplicationAnalysis::HloInstructionIsReplicatedAt(
    const HloInstruction* inst, const ShapeIndex& index,
    absl::Span<const ReplicaGroup> replica_groups) const {
  auto it = hlo_replication_.find(inst);
  if (it == hlo_replication_.end()) {
    return false;
  }
  VLOG(5) << "HloInstructionIsReplicatedAt is called on " << inst->name()
          << ", index: " << index.ToString()
          << ", replication: " << it->second.element(index).ToString();
  if (replica_groups.empty()) {
    return it->second.element(index).IsReplicatedOnAllDevices();
  }
  if (it->second.element(index).IsReplicatedOnAllDevices()) {
    return true;
  }
  if (it->second.element(index).IsUniqueOnAllDevices()) {
    return false;
  }
  for (const ReplicaGroup& replica_group : replica_groups) {
    if (!it->second.element(index).IsReplicatedWithinSubgroup(
            replica_group.replica_ids())) {
      return false;
    }
  }
  return true;
}

/* static */ absl::StatusOr<std::unique_ptr<HloReplicationAnalysis>>
HloReplicationAnalysis::Run(const HloModule* module,
                            bool cross_partition_spmd) {
  const absl::flat_hash_set<const HloInstruction*> empty;
  return Run(module, cross_partition_spmd, &empty);
}

/* static */ absl::StatusOr<std::unique_ptr<HloReplicationAnalysis>>
HloReplicationAnalysis::Run(const HloModule* module, bool cross_partition_spmd,
                            const absl::flat_hash_set<const HloInstruction*>*
                                loops_known_with_same_iterations) {
  auto analysis = absl::WrapUnique(new HloReplicationAnalysis(
      module, cross_partition_spmd, loops_known_with_same_iterations,
      /*support_partial_replication=*/false));
  TF_RETURN_IF_ERROR(analysis->ComputeHloReplication());
  return analysis;
}

/* static */ absl::StatusOr<std::unique_ptr<HloReplicationAnalysis>>
HloReplicationAnalysis::RunWithPartialReplication(const HloModule* module,
                                                  bool cross_partition_spmd) {
  const absl::flat_hash_set<const HloInstruction*> empty;
  auto analysis = absl::WrapUnique(
      new HloReplicationAnalysis(module, cross_partition_spmd, &empty,
                                 /*support_partial_replication=*/true));
  TF_RETURN_IF_ERROR(analysis->ComputeHloReplication());
  return analysis;
}

HloReplicationAnalysis::HloReplication::HloReplication()
    : state_(State::kReplicatedOnAllDevices) {}

HloReplicationAnalysis::HloReplication::HloReplication(
    HloReplicationAnalysis::HloReplication::State state,
    absl::Span<const std::vector<int64_t>> device_sets)
    : state_(state), device_sets_(device_sets.begin(), device_sets.end()) {
  CHECK(state == State::kPartiallyReplicated || device_sets_.empty());
}

HloReplicationAnalysis::HloReplication
HloReplicationAnalysis::HloReplication::ReplicatedOnAllDevices() {
  return HloReplication(State::kReplicatedOnAllDevices, {});
}

HloReplicationAnalysis::HloReplication
HloReplicationAnalysis::HloReplication::UniqueOnAllDevices() {
  return HloReplication(State::kUniqueOnAllDevices, {});
}

HloReplicationAnalysis::HloReplication
HloReplicationAnalysis::HloReplication::PartiallyReplicated(
    absl::Span<const std::vector<int64_t>> device_sets) {
  return HloReplication(State::kPartiallyReplicated, device_sets);
}

HloReplicationAnalysis::HloReplication
HloReplicationAnalysis::HloReplication::Merge(
    const HloReplication& other) const {
  switch (state_) {
    case State::kReplicatedOnAllDevices:
      return other;
    case State::kUniqueOnAllDevices:
      return *this;
    case State::kPartiallyReplicated: {
      switch (other.state_) {
        case State::kReplicatedOnAllDevices:
          return *this;
        case State::kUniqueOnAllDevices:
          return other;
        case State::kPartiallyReplicated: {
          std::vector<int64_t> merged_device_set;
          std::vector<std::vector<int64_t>> merged_device_sets;
          int other_set_idx = 0;
          int other_sets_idx = 0;
          for (std::vector<int64_t> device_set : device_sets_) {
            for (int64_t device_id : device_set) {
              if (other_set_idx == other.device_sets_[other_sets_idx].size()) {
                ++other_sets_idx;
                other_set_idx = 0;
                if (!merged_device_set.empty()) {
                  merged_device_sets.push_back(std::move(merged_device_set));
                  merged_device_set.clear();
                }
              }
              CHECK(other_sets_idx < other.device_sets_.size());
              if (other.device_sets_[other_sets_idx][other_set_idx] !=
                  device_id) {
                return UniqueOnAllDevices();
              }
              merged_device_set.push_back(device_id);
              ++other_set_idx;
            }
            if (!merged_device_set.empty()) {
              merged_device_sets.push_back(std::move(merged_device_set));
              merged_device_set.clear();
            }
          }
          CHECK(other_sets_idx >= other.device_sets_.size() - 1 &&
                other_set_idx >= other.device_sets_.back().size() - 1);

          return PartiallyReplicated(merged_device_sets);
        }
      }
    }
  }
}

bool HloReplicationAnalysis::HloReplication::Equal(
    const HloReplication& other) const {
  if (state_ != other.state_) {
    return false;
  }
  return absl::c_equal(device_sets_, other.device_sets_);
}

bool HloReplicationAnalysis::HloReplication::IsReplicatedOnAllDevices() const {
  return state_ == State::kReplicatedOnAllDevices;
}

bool HloReplicationAnalysis::HloReplication::IsUniqueOnAllDevices() const {
  return state_ == State::kUniqueOnAllDevices;
}

bool HloReplicationAnalysis::HloReplication::IsReplicatedWithinSubgroup(
    absl::Span<const int64_t> device_ids) const {
  // All device sets must contain all or none of the elements of device_ids.
  for (std::vector<int64_t> device_set : device_sets_) {
    int match_count = 0;
    for (int64_t device_id : device_ids) {
      match_count += std::find(device_set.begin(), device_set.end(),
                               device_id) != device_set.end();
    }
    if (match_count && match_count != device_ids.size()) {
      return false;
    }
  }

  return true;
}

std::string HloReplicationAnalysis::HloReplication::ToString() const {
  switch (state_) {
    case State::kReplicatedOnAllDevices:
      return "ReplicatedOnAllDevices";
    case State::kUniqueOnAllDevices:
      return "UniqueOnAllDevices";
    case State::kPartiallyReplicated:
      std::ostringstream oss;
      oss << "PartiallyReplicated{";
      for (int k = 0; k < device_sets_.size(); ++k) {
        oss << "{";
        oss << absl::StrJoin(device_sets_[k], ",");
        oss << (k == device_sets_.size() - 1 ? "}" : "},");
      }
      oss << "}";
      return oss.str();
  }
}

}  // namespace xla
