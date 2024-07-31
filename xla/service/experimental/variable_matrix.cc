// TODO: license

#include "xla/service/experimental/variable_matrix.h"

#include "xla/service/experimental/fix_log.h"

namespace xla {

VariableMatrix::VariableMatrix(std::shared_ptr<MPSolver> solver, 
    int num_rows, int num_cols, bool integer, int lb, int ub) : 
      num_rows_(num_rows),
      num_cols_(num_cols),
      solver_(solver),
      matrix_(num_rows, std::vector<MPVariable*>(num_cols))  {

  assert(num_rows > 0 && num_cols > 0);

  // fill the matrix with MPVariables
  for (int r = 0; r < num_rows_; r++) {
    for (int c = 0; c < num_cols_; c++) {
      matrix_[r][c] = solver_->MakeVar(lb, ub, integer, "");
    }
  }

  return;
}

LinearExpr VariableMatrix::SumRow(int r) {
  assert(0 <= r && r < num_rows_);
  assert(size() > 0); // create constraint only with non-empty matrix

  LinearExpr row_sum;
  for (int c = 0; c < num_cols_; c++) {
    row_sum += matrix_[r][c];
  }

  return row_sum;
}

LinearExpr VariableMatrix::SumCol(int c) {
  assert(0 <= c && c < num_cols_);
  assert(size() > 0); // create constraint only with non-empty matrix

  LinearExpr col_sum;
  for (int r = 0; r < num_rows_; r++) {
    col_sum += matrix_[r][c];
  }

  return col_sum;
}

LinearExpr VariableMatrix::Sum() {
  assert(size() > 0); // create constraint only with non-empty matrix

  LinearExpr sum;
  for (int r = 0; r < num_rows_; r++) {
    for (int c = 0; c < num_cols_; c++) {
      sum += matrix_[r][c];
    }
  }

  return sum;
}

void VariableMatrix::SetCoefficient(int r, int c, uint64_t coeff) {
  assert(size() > 0); // create constraint only with non-empty matrix
  assert(0 <= r && r < num_rows_);
  assert(0 <= c && c < num_cols_);

  solver_->MutableObjective()->SetCoefficient(matrix_[r][c], coeff);
  return;
}

void VariableMatrix::SetCoefficients(
    std::shared_ptr<ReshardingCostMatrix> cost_matrix) {
  assert(size() > 0); // create constraint only with non-empty matrix
  assert(cost_matrix->num_rows() == num_rows_);
  assert(cost_matrix->num_cols() == num_cols_);

  for (int r = 0; r < num_rows_; r++) {
    for (int c = 0; c < num_cols_; c++) {
      SetCoefficient(r, c, cost_matrix->CostAt(r, c));
    }
  }

  return;
}

std::string VariableMatrix::ToString(std::string delimiter) {
  std::string s = "";
  s += "[" + std::to_string(num_rows_) + "," + std::to_string(num_cols_) + "]" + "\n";

  for (int r = 0; r < num_rows_; r++) {
    for (int c = 0; c < num_cols_; c++) {
      s += std::to_string((int)(matrix_[r][c]->solution_value()));
      if (c != num_cols_ - 1) {
        s += delimiter;
      }
    }
    s += "\n";
  }

  return s;
}

} // xla