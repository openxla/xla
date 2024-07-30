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

std::string ReshardingCostMatrix::ToString() {

  std::string s = "";
  for (int r = 0; r < num_rows(); r++) {
    for (int c = 0; c < num_cols(); c++) {
      s += std::to_string(costs_[r][c]) + " ";
    }
    s += "\n";
  }

  return s;
}

} // xla