// TODO: license

#include "xla/service/experimental/complete_strategy_graph.h"


namespace xla {

namespace {

void CompleteInstructionStrategiesGraph(std::unordered_map<HloInstruction*,
    std::shared_ptr<InstructionStrategies>>& map) {

  std::vector<std::shared_ptr<InstructionStrategies>> strats;

  // connect edges for all instruction nodes
  for (auto& [instr, instr_strats] : map) {
    // add all operand strategies to the instr_strats
    strats.clear();
    for (HloInstruction* operand : instr->operands()) {
      strats.push_back(map.at(operand));
    }
    instr_strats->set_operand_strats(strats);

    // add all user strategies to instr_strats
    strats.clear();
    for (HloInstruction* operand : instr->users()) {
      strats.push_back(map.at(operand));
    }
    instr_strats->set_user_strats(strats);
  }
}

// Output shardings of sharding strategies in first argument will be
// have their resharding costs evaluated with the operand sharding strategies
// of the second argument from the appropriate index
std::shared_ptr<ReshardingCostMatrix> ConstructReshardingFromStrategies(
    std::shared_ptr<InstructionStrategies> instr_strats,
    std::shared_ptr<InstructionStrategies> user_instr_strats) {
  // extract instructions from their strategies objects
  const HloInstruction* instr = instr_strats->orig_instr();
  const HloInstruction* user_instr = user_instr_strats->orig_instr();

  // get the shape of the data that is resharded between these two operations
  const Shape& shape = instr->shape();

  // build vector of the output shardings from the instruction sharding strats
  std::vector<std::shared_ptr<HloSharding>> instr_shardings;
  for (ShardingStrategy& strat : instr_strats->sharding_strats()) {
    instr_shardings.push_back(strat.result_sharding());
  } 

  // build vector of operand shardings from the user instruction sharding strats
  // determine what the operand index of the first instruction is in user
  int op_idx = user_instr->operand_index(instr);
  std::vector<std::shared_ptr<HloSharding>> user_instr_shardings;
  for (ShardingStrategy& strat : user_instr_strats->sharding_strats()) {
    user_instr_shardings.push_back(strat.GetOpSharding(op_idx));
  }

  return std::make_shared<ReshardingCostMatrix>(
    shape, 
    instr_shardings, 
    user_instr_shardings);
}

// Construct reshardings between all instr and user of instr pairs
// and store them in their appropriate strategies
void EstimateReshardingCosts(std::unordered_map<HloInstruction*, 
    std::shared_ptr<InstructionStrategies>>& map) {
  
  // iterate through (instr, user of instr) pairs and create
  // resharding matrices from them
  for (auto& [instr, instr_strats] : map) {
    std::vector<std::shared_ptr<ReshardingCostMatrix>> resharding_matrices;
    for (auto& user_instr_strats : instr_strats->user_strats()) {
      resharding_matrices.push_back(ConstructReshardingFromStrategies(
        instr_strats, user_instr_strats
      ));
    }
    instr_strats->set_resharding_matrices(resharding_matrices);
  }
  
  return;
}

} // namespace

void CompleteStrategyGraph(std::unordered_map<HloInstruction*,
    std::shared_ptr<InstructionStrategies>>& map) {
  CompleteInstructionStrategiesGraph(map);
  EstimateReshardingCosts(map);
  return;
}

} // xla