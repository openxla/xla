// TODO: license

#include "xla/service/experimental/variable_matrix.h"

#include "xla/service/experimental/fix_log.h"

namespace xla {

VariableMatrix::VariableMatrix(MPSolver* solver, int num_rows, int num_cols,
    int lb, int ub) : 
      num_rows_(num_rows),
      num_cols_(num_cols),
      solver_(solver),
      matrix_(num_rows, std::vector<MPVariable*>(num_cols))  {

  // fill the matrix with MPVariables
  for (int r = 0; r < num_rows_; r++) {
    for (int c = 0; c < num_cols_; c++) {
      matrix_[r][c] = solver_->MakeIntVar(lb, ub, "");
    }
  }

  return;
}

} // xla