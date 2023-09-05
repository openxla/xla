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
#include "xla/service/gpu/loop_double_buffer_transformer.h"

#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/numeric/bits.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/service/while_loop_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

struct LoopInputConsumerInfo {
  int64_t operand_index;
  int64_t tuple_index;
};

HloInstruction* SetChannelIdForNewCollective(HloInstruction* new_instr,
                                             HloModule* module) {
  std::unordered_map<int64_t, int64_t> old_to_new_channel_id_map;
  std::unordered_map<int64_t, HloComputation*> channel_id_comp_map;
  if (HloAsyncInstruction::ClassOf(new_instr) &&
      hlo_query::IsCollectiveCommunicationOp(
          DynCast<HloAsyncInstruction>(new_instr)
              ->async_wrapped_instruction()
              ->opcode())) {
    HloInstruction* wrapped_instr =
        DynCast<HloAsyncInstruction>(new_instr)->async_wrapped_instruction();
    int64_t new_channel_id;
    int64_t old_channel_id = *wrapped_instr->channel_id();
    if (old_to_new_channel_id_map.find(old_channel_id) ==
        old_to_new_channel_id_map.end()) {
      new_channel_id = hlo_query::NextChannelId(*module);
      VLOG(2) << "Generated new channel id " << new_channel_id;
      old_to_new_channel_id_map[old_channel_id] = new_channel_id;
    } else {
      VLOG(2) << "Found existing channel id for old channel id "
              << old_channel_id;
      new_channel_id = old_to_new_channel_id_map[old_channel_id];
    }
    VLOG(2) << "Setting channel id to " << new_channel_id;

    wrapped_instr->set_channel_id(new_channel_id);
    if (channel_id_comp_map.find(new_channel_id) == channel_id_comp_map.end()) {
      channel_id_comp_map[new_channel_id] = new_instr->called_computations()[0];
    } else {
      channel_id_comp_map[new_channel_id]->AddAsyncInstruction(*new_instr);
    }
  } else if (hlo_query::IsCollectiveCommunicationOp(new_instr->opcode()) ||
             hlo_query::IsAsyncCollectiveStartOp(new_instr->opcode())) {
    new_instr->set_channel_id(hlo_query::NextChannelId(*module));
  }
  return new_instr;
}
StatusOr<bool> LoopDoubleBufferTransformer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  std::vector<HloInstruction*> while_instrs;
  for (auto* comp : module->computations(execution_threads)) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(while_instrs),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }
  VLOG(2) << "Processing " << while_instrs.size() << " while loops.";

  for (HloInstruction* while_instr : while_instrs) {
    TF_ASSIGN_OR_RETURN(auto config,
                        while_instr->backend_config<WhileLoopBackendConfig>());
    if (!config.has_known_trip_count()) {
      VLOG(2) << while_instr->ToString()
              << " doesn't have exact trip count, skipping double buffering "
                 "for now";
      changed |= changed;
      continue;
    }
    int64_t exact_trip_count = config.known_trip_count().n();
    VLOG(2) << "Processing while loop " << while_instr->ToString()
            << " with trip count: " << exact_trip_count;

    HloComputation* while_body = while_instr->while_body();

    CHECK(while_body->root_instruction()->opcode() == HloOpcode::kTuple);
    VLOG(2) << "Processing root " << while_body->root_instruction()->ToString();
    int64_t loop_output_count = while_body->root_instruction()->operand_count();

    std::vector<HloInstruction*> old_loop_roots(loop_output_count);
    std::vector<HloInstruction*> new_loop_roots(loop_output_count);
    // This encapulates a mapping from instruction to LoopInputConsumerInfo.
    std::unordered_map<HloInstruction*, std::vector<LoopInputConsumerInfo>>
        loop_input_consumers;
    for (int64_t i = 0; i < loop_output_count; i++) {
      old_loop_roots[i] = while_body->root_instruction()->mutable_operand(i);
    }
    HloInstruction* input_parameter = while_body->parameter_instructions()[0];
    VLOG(2) << "Processing input parameter " << input_parameter->ToString();

    // This captures all functional nodes that consume the input tuple along
    // with what operand index the GTE is feeding into and corresponding tuple
    // index.
    for (auto user : input_parameter->users()) {
      CHECK(user->opcode() == HloOpcode::kGetTupleElement);
      for (auto consumer : user->users()) {
        // If input is directly returned, we don't need to bookkeep it.
        if (consumer != while_body->root_instruction()) {
          LoopInputConsumerInfo input_consumer_info;
          input_consumer_info.operand_index = consumer->operand_index(user);
          input_consumer_info.tuple_index = user->tuple_index();
          loop_input_consumers[const_cast<HloInstruction*>(consumer)].push_back(
              input_consumer_info);
        }
      }
    }
    std::unordered_map<HloInstruction*, HloInstruction*> old_to_new_map;
    std::vector<HloInstruction*> all_old_instructions;
    std::vector<HloInstruction*> all_new_instructions;
    std::unordered_set<HloInstruction*> skip_control_dep_injection;

    absl::c_copy(while_body->instructions(),
                 std::back_inserter(all_old_instructions));

    if (exact_trip_count % 2) {
      VLOG(2) << "Found loops with odd trip count, 1 iteration will be pealed "
                 "outside of the main body.";
      HloInstruction* input_tuple = while_instr->mutable_operand(0);
      CHECK(input_tuple->opcode() == HloOpcode::kTuple);
      HloComputation* parent_comp = while_instr->parent();
      std::vector<HloInstruction*> loop_inputs(loop_output_count);
      for (int64_t i = 0; i < loop_output_count; i++) {
        loop_inputs[i] = input_tuple->mutable_operand(i);
      }
      for (auto old_instr : all_old_instructions) {
        // This is to track mappings of old->new channel id for async
        // collecitves wrapped in the form of HloAsyncInstruction, the start and
        // done need to have the same unique channel id.
        if (old_instr == input_parameter ||
            old_instr == while_body->root_instruction() ||
            old_instr->IsUserOf(input_parameter)) {
          continue;
        }
        VLOG(2) << "Peeling instruction " << old_instr->ToString();
        HloInstruction* new_instr = parent_comp->AddInstruction(
            old_instr->Clone(/*suffix=*/"peeled_double_buffer"));
        new_instr = SetChannelIdForNewCollective(new_instr, module);
        old_to_new_map[old_instr] = new_instr;
        all_new_instructions.push_back(new_instr);
        VLOG(2) << "Added instruction " << new_instr->ToString()
                << " to parent computation.";
      }
      // Handle existing control dependencies.
      for (auto old_instr : all_old_instructions) {
        if (old_to_new_map.find(old_instr) != old_to_new_map.end()) {
          HloInstruction* new_instr = old_to_new_map[old_instr];
          VLOG(2) << "Processing control predecessors for peeled instruction "
                  << new_instr->ToString();
          if (old_instr->control_predecessors().size() > 0) {
            std::vector<HloInstruction*> new_control_pred;
            for (auto pred : old_instr->control_predecessors()) {
              CHECK(old_to_new_map.find(pred) != old_to_new_map.end());
              new_control_pred.push_back(old_to_new_map[pred]);
            }

            TF_RETURN_IF_ERROR(new_instr->DropAllControlDeps());
            for (auto new_pred : new_control_pred) {
              TF_RETURN_IF_ERROR(new_pred->AddControlDependencyTo(new_instr));
              VLOG(2) << "Adding " << new_pred->ToString()
                      << " to control dependency of peeled instruction: "
                      << new_instr->ToString();
            }
          }
        }
      }
      // Replace all operands with newly cloned instructions.
      for (auto instr : all_new_instructions) {
        for (int64_t op_index = 0; op_index < instr->operand_count();
             op_index++) {
          VLOG(2) << "Processing operand " << op_index << " of "
                  << instr->ToString();
          HloInstruction* operand = instr->mutable_operand(op_index);
          if (operand == input_parameter ||
              operand->IsUserOf(input_parameter)) {
            VLOG(2) << "Operand " << op_index << " is an input parameter.";
            continue;
          }
          VLOG(2) << "Changing operand " << op_index << " of instruction "
                  << instr->ToString() << " from " << operand->ToString()
                  << " to " << old_to_new_map[operand]->ToString();
          TF_RETURN_IF_ERROR(
              instr->ReplaceOperandWith(op_index, old_to_new_map[operand]));
        }
      }

      // Connect the peeled instructions with previous while loop input
      // producers.
      for (auto& instr_index_pair : loop_input_consumers) {
        VLOG(2) << "Processing peeled instruction: "
                << instr_index_pair.first->ToString();
        HloInstruction* instr = old_to_new_map[instr_index_pair.first];
        // Get the operand index of its consumer
        for (auto consumer_info : instr_index_pair.second) {
          int64_t op_index = consumer_info.operand_index;
          HloInstruction* new_operand = loop_inputs[consumer_info.tuple_index];
          VLOG(2) << "Changing operand " << op_index
                  << " of peeled instruction " << instr->ToString() << " from "
                  << instr->operand(op_index)->ToString() << " to "
                  << new_operand->ToString();
          if (instr->operand(op_index) != new_operand) {
            TF_RETURN_IF_ERROR(
                instr->ReplaceOperandWith(op_index, new_operand));
          }
        }
      }
      // Make a new tuple as the new root
      for (int64_t i = 0; i < loop_output_count; i++) {
        if (old_to_new_map.find(old_loop_roots[i]) != old_to_new_map.end()) {
          new_loop_roots[i] = old_to_new_map[old_loop_roots[i]];
        } else {
          new_loop_roots[i] = loop_inputs[i];
        }
      }
      TF_RETURN_IF_ERROR(parent_comp->ReplaceWithNewInstruction(
          input_tuple, HloInstruction::CreateTuple(new_loop_roots)));
      old_to_new_map.clear();
      all_new_instructions.clear();
      exact_trip_count -= 1;
    }

    for (auto old_instr : all_old_instructions) {
      // This is to track mappings of old->new channel id for async collecitves
      // wrapped in the form of HloAsyncInstruction, the start and done need to
      // have the same unique channel id.
      if (old_instr == input_parameter ||
          old_instr == while_body->root_instruction() ||
          old_instr->IsUserOf(input_parameter)) {
        continue;
      }
      VLOG(2) << "Cloning instruction " << old_instr->ToString();
      HloInstruction* new_instr = while_body->AddInstruction(
          old_instr->Clone(/*suffix=*/"double_buffer_clone"));
      // If an elementwise instruction with constant operand is present, we
      // won't inject control dependency at the end to allow more constant
      // folding opportunities.
      if (old_instr->IsElementwiseBinary() && old_instr->HasConstantOperand()) {
        skip_control_dep_injection.insert(old_instr);
      }
      new_instr = SetChannelIdForNewCollective(new_instr, module);
      old_to_new_map[old_instr] = new_instr;
      all_new_instructions.push_back(new_instr);
      VLOG(2) << "Added instruction " << new_instr->ToString();
    }
    // Handle existing control dependencies.
    for (auto old_instr : all_old_instructions) {
      if (old_to_new_map.find(old_instr) != old_to_new_map.end()) {
        HloInstruction* new_instr = old_to_new_map[old_instr];
        VLOG(2) << "Processing control predecessors for "
                << new_instr->ToString();
        if (old_instr->control_predecessors().size() > 0) {
          std::vector<HloInstruction*> new_control_pred;
          for (auto pred : old_instr->control_predecessors()) {
            CHECK(old_to_new_map.find(pred) != old_to_new_map.end());
            new_control_pred.push_back(old_to_new_map[pred]);
          }

          TF_RETURN_IF_ERROR(new_instr->DropAllControlDeps());
          for (auto new_pred : new_control_pred) {
            TF_RETURN_IF_ERROR(new_pred->AddControlDependencyTo(new_instr));
            VLOG(2) << "Adding " << new_pred->ToString()
                    << " to control dependency of " << new_instr->ToString();
          }
        }
      }
    }
    for (auto instr : all_new_instructions) {
      for (int64_t op_index = 0; op_index < instr->operand_count();
           op_index++) {
        VLOG(2) << "Processing operand " << op_index << " of "
                << instr->ToString();
        HloInstruction* operand = instr->mutable_operand(op_index);
        if (operand == input_parameter || operand->IsUserOf(input_parameter)) {
          VLOG(2) << "Operand " << op_index << " is an input parameter.";
          continue;
        }
        VLOG(2) << "Changing operand " << op_index << " of instruction "
                << instr->ToString() << " from " << operand->ToString()
                << " to " << old_to_new_map[operand]->ToString();
        TF_RETURN_IF_ERROR(
            instr->ReplaceOperandWith(op_index, old_to_new_map[operand]));
      }
    }

    // Connect the boundary, meaning adding previous roots as operands of
    // instructions in the next iteration
    for (auto& instr_index_pair : loop_input_consumers) {
      VLOG(2) << "Processing boundary instruction: "
              << instr_index_pair.first->ToString();
      HloInstruction* instr = old_to_new_map[instr_index_pair.first];
      // Get the operand index of its consumer
      for (auto consumer_info : instr_index_pair.second) {
        int64_t op_index = consumer_info.operand_index;
        HloInstruction* new_operand = old_loop_roots[consumer_info.tuple_index];
        VLOG(2) << "Changing boundary operand " << op_index
                << " of instruction " << instr->ToString() << " from "
                << instr->operand(op_index)->ToString() << " to "
                << new_operand->ToString();
        if (instr->operand(op_index) != new_operand) {
          TF_RETURN_IF_ERROR(instr->ReplaceOperandWith(op_index, new_operand));
        }
      }
      if (skip_control_dep_injection.find(instr_index_pair.first) ==
              skip_control_dep_injection.end() &&
          !IsCollective(instr)) {
        for (auto& old_root : old_loop_roots) {
          TF_RETURN_IF_ERROR(old_root->AddControlDependencyTo(instr));
        }
      }
    }

    // Make a new tuple as the new root
    for (int64_t i = 0; i < loop_output_count; i++) {
      if (old_to_new_map.find(old_loop_roots[i]) != old_to_new_map.end()) {
        new_loop_roots[i] = old_to_new_map[old_loop_roots[i]];
      } else {
        new_loop_roots[i] = old_loop_roots[i];
      }
    }
    TF_RETURN_IF_ERROR(while_body->ReplaceWithNewInstruction(
        while_body->root_instruction(),
        HloInstruction::CreateTuple(new_loop_roots)));

    WhileLoopBackendConfig new_config;
    new_config.mutable_known_trip_count()->set_n((exact_trip_count / 2));
    TF_RETURN_IF_ERROR(while_instr->set_backend_config(new_config));
    changed = true;
  }

  VLOG(2) << "LoopDoubleBufferTransformer output: " << module->ToString();

  return changed;
}

}  // end namespace gpu
}  // end namespace xla
