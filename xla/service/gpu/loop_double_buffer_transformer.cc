/* Copyright 2023 The OpenXLA Authors.

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

#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/flatten_call_graph.h"
#include "xla/status.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

void SetChannelIdForNewCollective(HloInstruction* new_instr,
                                  const HloModule* module) {
  // This is to track mappings of old->new channel id for async collectives
  // wrapped in the form of HloAsyncInstruction, the start and done need to
  // have the same unique channel id.
  absl::flat_hash_map<int64_t, int64_t> old_to_new_channel_id_map;
  absl::flat_hash_map<int64_t, HloComputation*> channel_id_comp_map;
  if (new_instr->IsAsynchronous() && hlo_query::IsCollectiveCommunicationOp(
                                         new_instr->async_wrapped_opcode())) {
    HloInstruction* wrapped_instr =
        DynCast<HloAsyncInstruction>(new_instr)->async_wrapped_instruction();
    int64_t old_channel_id = *wrapped_instr->channel_id();
    int64_t new_channel_id = old_to_new_channel_id_map[old_channel_id];
    if (old_to_new_channel_id_map.find(old_channel_id) ==
        old_to_new_channel_id_map.end()) {
      new_channel_id = hlo_query::NextChannelId(*module);
      VLOG(2) << "Generated new channel id " << new_channel_id;
      old_to_new_channel_id_map[old_channel_id] = new_channel_id;
    }

    VLOG(2) << "Setting channel id to " << new_channel_id;

    wrapped_instr->set_channel_id(new_channel_id);
    if (channel_id_comp_map.find(new_channel_id) == channel_id_comp_map.end()) {
      channel_id_comp_map[new_channel_id] =
          new_instr->async_wrapped_computation();
    } else {
      channel_id_comp_map[new_channel_id]->AddAsyncStart(new_instr);
    }
  } else if (hlo_query::IsCollectiveCommunicationOp(new_instr->opcode()) ||
             hlo_query::IsAsyncCollectiveStartOp(new_instr)) {
    new_instr->set_channel_id(hlo_query::NextChannelId(*module));
  }
}

absl::StatusOr<std::vector<std::pair<int64_t, int64_t>>> ParseVectorOfPairs(
    std::basic_string<char>& str) {
  TF_ASSIGN_OR_RETURN(std::vector<ReplicaGroup> replica_groups,
                      ParseReplicaGroupsOnly(str));
  std::vector<std::pair<int64_t, int64_t>> res;
  res.reserve(replica_groups.size());
  for (const ReplicaGroup& replica_group : replica_groups) {
    TF_RET_CHECK(replica_group.replica_ids_size() == 2);
    int64_t a = replica_group.replica_ids(0);
    int64_t b = replica_group.replica_ids(1);
    res.emplace_back(a, b);
  }
  return res;
}

absl::Status SetSendRecvValidationForPeeledInstr(HloInstruction* new_instr,
                                                 HloInstruction* old_instr) {
  auto opcode = new_instr->opcode();
  assert(opcode == old_instr->opcode() &&
         "cloned instruction and original instruction have different opcodes");
  if (!HloPredicateIsOp<HloOpcode::kCollectivePermute,
                        HloOpcode::kCollectivePermuteStart, HloOpcode::kSend,
                        HloOpcode::kRecv>(old_instr)) {
    return absl::OkStatus();
  }

  google::protobuf::Map<std::string, std::string> attribute_map =
      new_instr->frontend_attributes().map();
  if (!attribute_map.contains(kSendRecvValidationAttr)) {
    return absl::OkStatus();
  }

  VLOG(3) << "Original send-recv iterations: "
          << attribute_map.at(kSendRecvValidationAttr);

  TF_ASSIGN_OR_RETURN(
      auto send_recv_validation_attr,
      ParseVectorOfPairs(attribute_map.at(kSendRecvValidationAttr)));

  uint64_t n_pairs = send_recv_validation_attr.size();
  if (n_pairs == 0) {
    return absl::OkStatus();
  }
  std::vector<std::pair<int64_t, int64_t>> send_recv_validation_attr_updated(
      n_pairs, {0, 0});
  bool is_forward_cycle = (send_recv_validation_attr[0].first == 0);
  if (is_forward_cycle)
    send_recv_validation_attr_updated[0] = {0, 1};
  else
    send_recv_validation_attr_updated[send_recv_validation_attr.size() - 1] = {
        0, 1};

  FrontendAttributes attributes;
  // Copy the attributes from old_instr attributes
  for (auto& [key, val] : old_instr->frontend_attributes().map()) {
    if (key == kSendRecvValidationAttr) continue;
    (*attributes.mutable_map())[key] = val;
  }
  std::string send_recv_validation_attr_str =
      "{" +
      absl::StrJoin(send_recv_validation_attr_updated, ",",
                    absl::PairFormatter(
                        [](std::string* out, int64_t value) {
                          absl::StrAppend(out, "{", value);
                        },
                        ",",
                        [](std::string* out, int64_t value) {
                          absl::StrAppend(out, value, "}");
                        })) +
      "}";
  VLOG(3) << "New send-recv iterations for peeled instruction: "
          << send_recv_validation_attr_str;
  (*attributes.mutable_map())[kSendRecvValidationAttr] =
      send_recv_validation_attr_str;
  new_instr->set_frontend_attributes(attributes);
  return absl::OkStatus();
}

absl::Status SetSendRecvValidation(HloInstruction* new_instr,
                                   HloInstruction* old_instr, bool is_peeled) {
  auto opcode = new_instr->opcode();
  assert(opcode == old_instr->opcode() &&
         "cloned instruction and original instruction have different opcodes");
  if (!HloPredicateIsOp<HloOpcode::kCollectivePermute,
                        HloOpcode::kCollectivePermuteStart, HloOpcode::kSend,
                        HloOpcode::kRecv>(old_instr)) {
    return absl::OkStatus();
  }
  google::protobuf::Map<std::string, std::string> attribute_map =
      new_instr->frontend_attributes().map();
  if (!attribute_map.contains(kSendRecvValidationAttr)) {
    return absl::OkStatus();
  }
  VLOG(3) << "Original send-recv iterations: "
          << attribute_map.at(kSendRecvValidationAttr);

  TF_ASSIGN_OR_RETURN(
      auto send_recv_validation_attr,
      ParseVectorOfPairs(attribute_map.at(kSendRecvValidationAttr)));

  if (send_recv_validation_attr.size() == 0) {
    return absl::OkStatus();
  }

  std::vector<std::pair<int64_t, int64_t>> send_recv_iterations_old_instr,
      send_recv_iterations_new_instr;
  send_recv_iterations_old_instr.reserve(send_recv_validation_attr.size());
  send_recv_iterations_new_instr.reserve(send_recv_validation_attr.size());
  for (std::pair<int64_t, int64_t> pair : send_recv_validation_attr) {
    int64_t a = pair.first;
    int64_t b = pair.second;
    if (is_peeled) {
      send_recv_iterations_old_instr.emplace_back(std::floor(a / 2.0),
                                                  std::floor(b / 2.0));
      send_recv_iterations_new_instr.emplace_back(
          std::max(0.0, std::floor((a - 1) / 2.0)), std::floor((b - 1) / 2.0));
    } else {
      send_recv_iterations_old_instr.emplace_back(std::ceil(a / 2.0),
                                                  std::ceil(b / 2.0));
      send_recv_iterations_new_instr.emplace_back(std::floor(a / 2.0),
                                                  std::floor(b / 2.0));
    }
  }

  std::string iteration_instances_old_instr =
      "{" +
      absl::StrJoin(send_recv_iterations_old_instr, ",",
                    absl::PairFormatter(
                        [](std::string* out, int64_t value) {
                          absl::StrAppend(out, "{", value);
                        },
                        ",",
                        [](std::string* out, int64_t value) {
                          absl::StrAppend(out, value, "}");
                        })) +
      "}";

  std::string iteration_instances_new_instr =
      "{" +
      absl::StrJoin(send_recv_iterations_new_instr, ",",
                    absl::PairFormatter(
                        [](std::string* out, int64_t value) {
                          absl::StrAppend(out, "{", value);
                        },
                        ",",
                        [](std::string* out, int64_t value) {
                          absl::StrAppend(out, value, "}");
                        })) +
      "}";

  VLOG(3) << "New send-recv iterations for original instruction: "
          << iteration_instances_old_instr;
  VLOG(3) << "New send-recv iterations for new instruction: "
          << iteration_instances_new_instr;

  FrontendAttributes attributes;
  // Copy the attributes from old_instr attributes
  for (auto& [key, val] : old_instr->frontend_attributes().map()) {
    if (key == kSendRecvValidationAttr) continue;
    (*attributes.mutable_map())[key] = val;
  }
  (*attributes.mutable_map())[kSendRecvValidationAttr] =
      iteration_instances_old_instr;
  old_instr->set_frontend_attributes(attributes);

  // Do the same for new instr
  attributes.clear_map();
  for (auto& [key, val] : new_instr->frontend_attributes().map()) {
    if (key == kSendRecvValidationAttr) continue;
    (*attributes.mutable_map())[key] = val;
  }
  (*attributes.mutable_map())[kSendRecvValidationAttr] =
      iteration_instances_new_instr;
  new_instr->set_frontend_attributes(attributes);
  return absl::OkStatus();
}

absl::Status PeelInstructionsForOddTripCount(HloModule* module,
                                             HloInstruction* while_instr) {
  std::string suffix = "peeled_double_buffer";
  absl::flat_hash_map<HloInstruction*, HloInstruction*> old_to_new_map;
  HloComputation* while_body = while_instr->while_body();
  HloInstruction* input_parameter = while_body->parameter_instruction(0);
  HloInstruction* input_tuple = while_instr->mutable_operand(0);

  auto old_loop_roots = while_body->root_instruction()->mutable_operands();
  HloComputation* parent_comp = while_instr->parent();
  old_to_new_map[input_parameter] = input_tuple;

  for (HloInstruction* old_instr : while_body->MakeInstructionPostOrder()) {
    if (old_to_new_map.find(old_instr) != old_to_new_map.end()) {
      continue;
    }
    VLOG(2) << "Peeling instruction " << old_instr->ToString();
    std::vector<HloInstruction*> new_operands(old_instr->operand_count());
    for (int64_t i = 0; i < old_instr->operand_count(); i++) {
      new_operands[i] = old_to_new_map[old_instr->mutable_operand(i)];
    }
    HloInstruction* new_instr =
        parent_comp->AddInstruction(old_instr->CloneWithNewOperands(
            old_instr->shape(), new_operands, suffix));

    SetChannelIdForNewCollective(new_instr, module);
    TF_CHECK_OK(SetSendRecvValidationForPeeledInstr(new_instr, old_instr));
    old_to_new_map[old_instr] = new_instr;
    VLOG(2) << "Added instruction " << new_instr->ToString()
            << " to parent computation.";
  }

  std::vector<HloInstruction*> new_roots;
  for (HloInstruction* instr : old_loop_roots) {
    new_roots.push_back(old_to_new_map[instr]);
  }
  TF_RETURN_IF_ERROR(while_instr->ReplaceOperandWith(
      0, old_to_new_map[while_body->root_instruction()]));
  VLOG(2) << "Replaced with new input tuple "
          << while_instr->operand(0)->ToString();

  // Handle existing control dependencies.
  for (HloInstruction* old_instr : while_body->MakeInstructionPostOrder()) {
    if (old_to_new_map.find(old_instr) != old_to_new_map.end()) {
      HloInstruction* new_instr = old_to_new_map[old_instr];
      VLOG(2) << "Processing control predecessors for peeled instruction "
              << new_instr->ToString();
      std::vector<HloInstruction*> new_control_pred(
          old_instr->control_predecessors().size());
      for (HloInstruction* pred : old_instr->control_predecessors()) {
        new_control_pred.push_back(old_to_new_map[pred]);
      }

      TF_RETURN_IF_ERROR(new_instr->DropAllControlDeps());
      for (HloInstruction* new_pred : new_control_pred) {
        TF_RETURN_IF_ERROR(new_pred->AddControlDependencyTo(new_instr));
        VLOG(2) << "Adding " << new_pred->ToString()
                << " to control dependency of peeled instruction: "
                << new_instr->ToString();
      }
    }
  }
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<bool> LoopDoubleBufferTransformer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  std::vector<HloInstruction*> while_instrs;
  for (auto comp : module->MakeNonfusionComputations()) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(while_instrs),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }
  VLOG(2) << "Processing " << while_instrs.size() << " while loops.";

  for (HloInstruction* while_instr : while_instrs) {
    TF_ASSIGN_OR_RETURN(WhileLoopBackendConfig config,
                        while_instr->backend_config<WhileLoopBackendConfig>());
    if (!config.has_known_trip_count()) {
      VLOG(2) << while_instr->ToString()
              << " doesn't have exact trip count, skipping double buffering "
                 "for now";
      continue;
    }
    int64_t exact_trip_count = config.known_trip_count().n();
    VLOG(2) << "Processing while loop " << while_instr->ToString()
            << " with trip count: " << exact_trip_count;

    HloComputation* while_body = while_instr->while_body();

    VLOG(2) << "Processing root " << while_body->root_instruction()->ToString();

    auto old_loop_roots = while_body->root_instruction()->mutable_operands();
    HloInstruction* input_parameter = while_body->parameter_instruction(0);
    VLOG(2) << "Processing input parameter " << input_parameter->ToString();
    absl::flat_hash_map<HloInstruction*, HloInstruction*> old_to_new_map;
    absl::flat_hash_set<HloInstruction*> skip_control_dep_injection;
    bool is_peeled = exact_trip_count % 2;
    if (is_peeled) {
      VLOG(2) << "Found loops with odd trip count, 1 iteration will be peeled "
                 "outside of the main body.";
      TF_RETURN_IF_ERROR(PeelInstructionsForOddTripCount(module, while_instr));
      exact_trip_count -= 1;
    }
    std::string suffix = "double_buffer_clone";
    old_to_new_map[input_parameter] = while_body->root_instruction();
    for (HloInstruction* old_instr : while_body->MakeInstructionPostOrder()) {
      if (old_to_new_map.find(old_instr) != old_to_new_map.end()) {
        continue;
      }
      VLOG(2) << "Cloning instruction " << old_instr->ToString();
      std::vector<HloInstruction*> new_operands;
      for (HloInstruction* old_operand : old_instr->mutable_operands()) {
        new_operands.push_back(old_to_new_map[old_operand]);
      }
      HloInstruction* new_instr =
          while_body->AddInstruction(old_instr->CloneWithNewOperands(
              old_instr->shape(), new_operands, suffix));

      // If an elementwise instruction with constant operand is present, we
      // won't inject control dependency at the end to allow more constant
      // folding opportunities.
      if (old_instr->IsElementwiseBinary() && old_instr->HasConstantOperand()) {
        skip_control_dep_injection.insert(old_instr);
      }
      SetChannelIdForNewCollective(new_instr, module);
      TF_CHECK_OK(SetSendRecvValidation(new_instr, old_instr, is_peeled));
      old_to_new_map[old_instr] = new_instr;
      VLOG(2) << "Added instruction " << new_instr->ToString();
    }

    while_body->set_root_instruction(
        old_to_new_map[while_body->root_instruction()]);
    VLOG(2) << "Replaced with new root "
            << while_body->root_instruction()->ToString();

    // Handle existing control dependencies.
    for (HloInstruction* old_instr : while_body->MakeInstructionPostOrder()) {
      if (old_to_new_map.find(old_instr) != old_to_new_map.end()) {
        HloInstruction* new_instr = old_to_new_map[old_instr];
        VLOG(2) << "Processing control predecessors for "
                << new_instr->ToString();
        std::vector<HloInstruction*> new_control_pred(
            old_instr->control_predecessors().size());
        for (HloInstruction* pred : old_instr->control_predecessors()) {
          new_control_pred.push_back(old_to_new_map[pred]);
        }

        TF_RETURN_IF_ERROR(new_instr->DropAllControlDeps());
        for (HloInstruction* new_pred : new_control_pred) {
          TF_RETURN_IF_ERROR(new_pred->AddControlDependencyTo(new_instr));
          VLOG(2) << "Adding " << new_pred->ToString()
                  << " to control dependency of " << new_instr->ToString();
        }
      }
    }
    for (HloInstruction* input_consumer : input_parameter->users()) {
      for (HloInstruction* old_input : input_consumer->users()) {
        if (old_to_new_map.find(old_input) != old_to_new_map.end()) {
          HloInstruction* new_input = old_to_new_map[old_input];
          if (skip_control_dep_injection.find(old_input) ==
                  skip_control_dep_injection.end() &&
              !IsCollective(old_input)) {
            for (HloInstruction* old_root : old_loop_roots) {
              TF_RETURN_IF_ERROR(old_root->AddControlDependencyTo(new_input));
            }
          }
        }
      }
    }
    WhileLoopBackendConfig new_config;
    new_config.mutable_known_trip_count()->set_n((exact_trip_count / 2));
    TF_RETURN_IF_ERROR(while_instr->set_backend_config(new_config));
    changed = true;
  }

  VLOG(2) << "LoopDoubleBufferTransformer output: " << module->ToString();

  // Run necessary cleanup to ensure LoopDoubleBufferTransformer behaves
  // correctly.
  if (changed) {
    // The call graph will not be flat if one of the loops that was unrolled
    // contains any kind of call to another computation---since the call will
    // be duplicated, thereby adding a second callsite for that computation.
    TF_RETURN_IF_ERROR(
        FlattenCallGraph().Run(module, execution_threads).status());
  }

  return changed;
}

}  // end namespace gpu
}  // end namespace xla
