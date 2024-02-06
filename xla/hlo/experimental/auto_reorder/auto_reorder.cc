#include "xla/hlo/experimental/auto_reorder/auto_reorder.h"
namespace xla {
constexpr int64_t kPointerSize = 8;
// get shape byte size, f32 have 4 bytes;
int64_t ShapeSize(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, kPointerSize);
}

tsl::Status AutoReorderPass::RebuildHloOrdering(
    HloSchedule& module_schedule, HloComputation* entry_computation) {
  bool is_debug = false;
  // module_schedule.remove_computation(entry_computation);
  // module_schedule.GetOrCreateSequence(entry_computation);
  auto status = module_schedule.UpdateComputationSchedule(entry_computation);

  if (!status.ok()) {
    return status;
  } else {
  }
  status = module_schedule.Update({});
  if (!status.ok()) {
    VLOG(2) << "Update error:" << status.message() << std::endl;
    return status;
  }
  // SequentialHloOrdering seq_ordering(module_schedule);
  // auto seqs = seq_ordering.SequentialOrder(*entry_computation);
  // module_schedule.set_sequence(entry_computation, *seqs);

  auto new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  for (auto i = 0; i < new_instruction_sequence.size(); i++) {
    auto inst = new_instruction_sequence.at(i);
  }
  status = module_schedule.Verify();
  if (!status.ok()) {
    return status;
  }
  return OkStatus();
}
tsl::StatusOr<std::vector<HloInstruction*>>
AutoReorderPass::ScheduleComputation(HloComputation* computation) {
  int64_t current_pos = 0;
  auto post_order_instructions = computation->MakeInstructionPostOrder();
  HloScheduleGraph schedule_graph(&post_order_instructions,
                                  /*alias_analysis=*/nullptr,
                                  latency_estimator_.get(),
                                  async_tracker_.get());
  async_tracker_->PostProcessScheduleGraph(&schedule_graph,
                                           latency_estimator_.get());
  // we don't need InitializeGraphAnalysis for init node status;
  
    
  auto solver_ = absl::make_unique<LinearProgramScheduler<
      LPContainer<const HloInstruction*>, const HloInstruction*>
    >();

  // scan instructions, get every instruction cost and deps
  for (HloInstruction* instr : post_order_instructions) {
    const HloGraphNode& instr_node = schedule_graph.GetNode(instr);
    VLOG(2) << instr->ToShortString() << "flops cost :" << instr_node.GetCost();
    auto addEdge = [&](const xla::HloInstruction* from_inst, 
                      LPContainer<const HloInstruction*>* dst_node,
                       NodeType edge_type) {
      auto operand_lp_node = solver_->FindInstructionLPNode(from_inst);
      if (!operand_lp_node.ok()) {
        VLOG(2) << "operand_lp_node not found:" << from_inst->ToShortString();
        return false;
      }
      auto operand_node = schedule_graph.GetNode(from_inst);
      CostType edge_cost =
          latency_estimator_->GetLatencyBetween(operand_node, instr_node);
      VLOG(2) << from_inst->ToShortString() + " should execute before " +
                     instr->ToShortString();
      dst_node->AddDep(operand_lp_node.value(), edge_cost);
      return true;
    };

    CostType cost = std::ceil(instr_node.GetCost());
    // there are 2 type now: 1. compute 2. communication
    if (async_tracker_->IsSupportedAsyncStart(*instr) ||
        async_tracker_->IsSupportedAsyncDone(*instr)) {
      // communication
      // GetCost return float, floor to int
      auto current_inst_lp_node = solver_->FindLPNodeOrCreate(
          instr, cost, NodeType::kCommunication);
      // add current node as constraint

      if (async_tracker_->IsSupportedAsyncDone(*instr)) {
        // create a edge, which is communication
        auto operand_inst = instr->operand(0);
        auto is_success = addEdge(operand_inst, current_inst_lp_node,
                                  NodeType::kCommunication);
        TF_RET_CHECK(is_success)
            << "operand_lp_node not found:" << operand_inst->ToShortString();
      } else {
        // add it's operands to his deps
        for (auto i = 0; i < instr->operand_count(); i++) {
          auto operand_inst = instr->operand(i);
          auto is_success = addEdge(operand_inst, current_inst_lp_node,
                                    NodeType::kCompute);
          TF_RET_CHECK(is_success)
              << "operand_lp_node not found:" << operand_inst->ToShortString();
        }
      }

      TF_CHECK_OK(solver_->AddConstraint(current_inst_lp_node));

    } else {  // compute
      auto current_inst_lp_node =
          solver_->FindLPNodeOrCreate(instr, cost, NodeType::kCompute);
      // when adding edge node, current node have no add to Constraint?
      for (auto i = 0; i < instr->operand_count(); i++) {
        auto operand_inst = instr->operand(i);
        auto is_success = addEdge(operand_inst, current_inst_lp_node,
                                  NodeType::kCompute);
        TF_RET_CHECK(is_success)
            << "operand_lp_node not found:" << operand_inst->ToShortString();
      }
      TF_CHECK_OK(solver_->AddConstraint(current_inst_lp_node));
    }
  }
  auto status = solver_->Solve();
  if(reorder::solve_debug){
    solver_->RenderGantt(absl::StrCat("gantt_", computation->name()));
    solver_->RenderGraphviz(absl::StrCat("gantt_", computation->name()));
  }
  
  if (status.ok()) {
    // return instruction order by solver
    std::vector<HloInstruction*> new_schedule;
    auto sorted_nodes = solver_->GetSortedNodes();
    for (auto node : sorted_nodes) {
      new_schedule.push_back(
          const_cast<xla::HloInstruction*>(node->GetValue()));
    }
    return new_schedule;
  }
  TF_RET_CHECK(status.ok()) << "Solver error:" << status.message();
  return status;
}
tsl::Status AutoReorderPass::MoveInstruction(HloComputation* src_computation,
                                             absl::string_view src_name,
                                             HloComputation* dst_computation) {
  bool is_debug = true;

  // Move instruction from src_computation to dst_computation.
  auto src_instruction = src_computation->GetInstructionWithName(src_name);
  // step 1: found src_instruction input args and output args
  std::vector<HloInstruction*>
      src_inputs;  // instruction which outputs is needed by src_instruction
  std::vector<HloInstruction*>
      src_outputs;  // instruction which input is src_instruction's output
  for (auto i = 0; i < src_instruction->operand_count(); i++) {
    auto src_input = src_instruction->mutable_operand(i);
    src_inputs.push_back(src_input);
  }
  std::vector<xla::HloInstruction*> user_insts = src_instruction->users();
  for (auto i = 0; i < src_instruction->user_count(); i++) {
    src_outputs.push_back(user_insts.at(i));
  }
  // step 2: create Send Instruction for input args, create Recv Instruction for
  // output args
  int64_t channel_id = 0;
  std::vector<HloInstruction*> dst_inputs;
  std::vector<HloInstruction*> send_params;
  dst_inputs.reserve(src_inputs.size());
  send_params.reserve(src_inputs.size());
  for (size_t i = 0; i < src_inputs.size(); i++) {
    channel_id++;
    auto src_input = src_inputs.at(i);
    auto src_input_shape = src_input->shape();
    // src_instruction
    auto token = src_computation->AddInstruction(HloInstruction::CreateToken());

    auto send_inst = src_computation->AddInstruction(HloInstruction::CreateSend(
        src_input, token, channel_id, false /*is_host_transfer*/));
    auto send_done = src_computation->AddInstruction(
        HloInstruction::CreateSendDone(send_inst));
    token = dst_computation->AddInstruction(HloInstruction::CreateToken());
    auto recv_inst = dst_computation->AddInstruction(
        HloInstruction::CreateRecv(src_input_shape, token, channel_id,
                                   false /*is_host_transfer*/),
        "dst_recv" + std::to_string(i));
    auto recv_done = dst_computation->AddInstruction(
        HloInstruction::CreateRecvDone(recv_inst));
    HloInstruction* recv_parameter = dst_computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(recv_done, 0));

    dst_inputs.push_back(recv_parameter);
  }
  channel_id++;
  // step3: clone same instruction to dst_computation
  auto dst_inst =
      dst_computation->AddInstruction(src_instruction->CloneWithNewOperands(
          src_instruction->shape(), dst_inputs));

  // step4 :create Send Instruction from dst_compuation, create Recv Instruction
  // in src_computation
  auto token = dst_computation->AddInstruction(HloInstruction::CreateToken());

  auto ret_send_inst =
      dst_computation->AddInstruction(HloInstruction::CreateSend(
          dst_inst, token, channel_id, false /*is_host_transfer*/));
  auto send_done = dst_computation->AddInstruction(
      HloInstruction::CreateSendDone(ret_send_inst));

  // create recv in src_computation, create token node,so recv_inst will be
  // executed by scheduler
  token = src_computation->AddInstruction(HloInstruction::CreateToken());

  auto recv_inst = src_computation->AddInstruction(
      HloInstruction::CreateRecv(dst_inst->shape(), token, channel_id,
                                 false /*is_host_transfer*/),
      "src_recv_ret");
  auto recv_done = src_computation->AddInstruction(
      HloInstruction::CreateRecvDone(recv_inst));
  HloInstruction* recv_parameter = src_computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(recv_done, 0));

  // step5: replace instruction which use src_instruction's output with Recv
  // Instruction
  for (size_t i = 0; i < src_outputs.size(); i++) {
    /* code */
    auto src_output = src_outputs.at(i);
    // add dependency
    auto status = src_instruction->ReplaceUseWith(src_output, recv_parameter);
    if (!status.ok()) {
      VLOG(2) << "ReplaceUseWith error:" << status.message() << std::endl;
    }
    absl::flat_hash_map<int, HloInstruction*> new_instruction_uses;
    int operand_num = 0;
    for (const HloInstruction* operand : src_output->operands()) {
      if (operand->unique_id() == src_instruction->unique_id()) {
        new_instruction_uses[operand_num] = recv_parameter;
      }
      operand_num++;
    }
    for (auto it = new_instruction_uses.begin();
         it != new_instruction_uses.end(); ++it) {
      status = src_output->ReplaceOperandWith(it->first, it->second);
      if (!status.ok()) {
        VLOG(2) << "ReplaceOperandWith error:" << status.message() << std::endl;
      }
    }
  }
  // step6: remove src_instruction
  src_instruction->DetachFromOperandsAndUsers();
  auto status = src_computation->RemoveInstruction(src_instruction);
  if (!status.ok()) {
    VLOG(2) << "RemoveInstruction error:" << status.message() << std::endl;
    return status;
  } else {
    VLOG(3) << "RemoveInstruction success"
            << src_computation->instruction_count() << std::endl;
    return OkStatus();
  }
}
StatusOr<bool> AutoReorderPass::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // about reorder: be careful about RNG, such as dropout, random_shuffle,
  // random_uniform;
  // HloCostAnalysis, get instruction cost
  HloComputation* entry_computation = module->entry_computation();

  // Currently we expect that a schedule that minimizes memory pressure is
  // provided as a base. It's not necessary for the algorithm itself but it
  // allows us to not having to think for now about memory pressure.
  std::vector<HloComputation*> computations_to_schedule;
  computations_to_schedule.reserve(module->computation_count());
  // Collect which computations have latency hiding opportunities.
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (auto* instr : computation->instructions()) {
      if (async_tracker_->IsSupportedAsyncStart(*instr) ||
          async_tracker_->IsSupportedAsyncDone(*instr)) {
        computations_to_schedule.push_back(computation);
        break;
      }
    }
  }
  if (computations_to_schedule.empty()) {
    return false;
  }

  absl::flat_hash_map<HloComputation*, std::vector<HloInstruction*>>
      saved_schedules;
  // TF_RETURN_IF_ERROR(scheduler_core_->InitializeScheduler(module)); //TODO:
  // we don't limit memory usage
  for (HloComputation* computation : computations_to_schedule) {
    TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> new_schedule,
                        ScheduleComputation(computation));
    VLOG(2) << "new_schedule length:" << new_schedule.size()
            << " computation instruction_count:"
            << computation->instruction_count();
    saved_schedules[computation] = std::move(new_schedule);
  }

  // TODO: now memory is not in constraction
  // LOG(INFO) << "AutoReorderPass current memory usage: "
  //           << scheduler_core_->GetMemoryPeak() << " bytes.";
  for (HloComputation* computation : computations_to_schedule) {
    // VLOG(1) << "Statistics before scheduling:";
    VLOG(4) << "sequences length:" << module->schedule().sequences().size()
            << std::endl;
    module->schedule().set_sequence(
        computation, absl::MakeConstSpan(saved_schedules[computation]));
    VLOG(1) << "Statistics after scheduling:";
    // LogScheduleStatistics(computation);
  }
  return true;

}  // AutoReorderPass::Run
CanonicalAsyncOp GpuGetCanonicalAsyncOp(const HloInstruction& hlo) {
  switch (hlo.opcode()) {
    case HloOpcode::kSend:
      return {HloOpcode::kAsyncStart, HloOpcode::kSend};
    case HloOpcode::kSendDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kSend};
    case HloOpcode::kRecv:
      return {HloOpcode::kAsyncStart, HloOpcode::kRecv};
    case HloOpcode::kRecvDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kRecv};
    default:
      return DefaultGetCanonicalAsyncOp(hlo);
  }
}

}  // namespace xla
