// TODO: license

#include "xla/service/experimental/resharding_cost_matrix.h"

#include "xla/service/experimental/resharding_cost_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

ReshardingCostMatrix::ReshardingCostMatrix(const Shape& shape, 
    std::vector<std::shared_ptr<HloSharding>>& strats1, 
    std::vector<std::shared_ptr<HloSharding>>& strats2) :
      num_rows_(strats1.size()), 
      num_cols_(strats2.size()),
      costs_(num_rows_, std::vector<uint64_t>(num_cols_, 0)) {

  ReshardingCostEvaluator evaluator;
  assert(num_rows_ == strats1.size() && num_cols_ == strats2.size());

  // iterate through each pair and fill the costs_ matrix
  for (int r = 0; r < num_rows_; r++) {
    for (int c = 0; c < num_cols_; c++) {
      costs_[r][c] = evaluator.Evaluate(
        shape, 
        *strats1[r].get(), 
        *strats2[c].get());
    }
  }

  return;

}

uint64_t ReshardingCostMatrix::CostAt(int r, int c) {
  assert(0 <= r && r < num_rows());
  assert(0 <= c && c < num_cols());

  return costs_[r][c];
}

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

} // xla