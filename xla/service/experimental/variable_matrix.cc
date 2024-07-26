// TODO: license

#include "xla/service/experimental/variable_matrix.h"

#include "xla/service/experimental/fix_log.h"

namespace xla {

VariableMatrix::VariableMatrix(MPSolver* solver, int num_rows, int num_cols,
    bool integer, int lb, int ub) : 
      num_rows_(num_rows),
      num_cols_(num_cols),
      solver_(solver),
      matrix_(num_rows, std::vector<MPVariable*>(num_cols))  {

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

  LinearExpr row_sum;
  for (int c = 0; c < num_cols_; c++) {
    row_sum += matrix_[r][c];
  }

  return row_sum;
}

LinearExpr VariableMatrix::SumCol(int c) {
  assert(0 <= c && c < num_cols_);

  LinearExpr col_sum;
  for (int r = 0; r < num_rows_; r++) {
    col_sum += matrix_[r][c];
  }

  return col_sum;
}

LinearExpr VariableMatrix::Sum() {
  LinearExpr sum;
  for (int r = 0; r < num_rows_; r++) {
    for (int c = 0; c < num_cols_; c++) {
      sum += matrix_[r][c];
    }
  }

  return sum;
}

void VariableMatrix::SetCoefficient(int r, int c, uint64_t coeff) {
  solver_->MutableObjective()->SetCoefficient(matrix_[r][c], coeff);
  return;
}

} // xla