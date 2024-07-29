// TODO: license

#ifndef XLA_SERVICE_EXPERIMENTAL_VARIABLE_MATRIX_H_
#define XLA_SERVICE_EXPERIMENTAL_VARIABLE_MATRIX_H_

#include "xla/service/experimental/resharding_cost_matrix.h"

#include "ortools/linear_solver/linear_solver.h"

#include "stdint.h"

using ::operations_research::MPSolver;
using ::operations_research::MPObjective;
using ::operations_research::MPVariable;
using ::operations_research::LinearExpr;

namespace xla {

// This class represents a matrix of MPVariables and provides methods 
// to take specific sums and setting coefficients of various variables
// Note: indexes into array use 0-indexing
class VariableMatrix {
public:
  // defines num_rows * num_cols MPVariables with lower and upper bounds
  // of lb and ub for the provided solver
  // note: num_rows and num_cols must be > 0
  VariableMatrix(std::shared_ptr<MPSolver> solver, int num_rows, int num_cols, 
    bool integer, int lb, int ub);

  // returns a linear expression of the sum of the r'th row 
  // no equality constraint
  LinearExpr SumRow(int r);

  // returns a linear expression of the sum of the c'th column
  // no equality constraint
  LinearExpr SumCol(int c);

  // returns a linear expression of the sum of all elements in the array
  // no equality constraint
  LinearExpr Sum();

  // sets the coefficient of the variable in position [r, c] of the matrix
  // for the solver objective
  void SetCoefficient(int r, int c, uint64_t coeff);

  // sets the coefficients of the variables from a ReshardingCostMatrix
  // ReshardingCostMatrix must have same dimensions as this VariableMatrix
  void SetCoefficients(std::shared_ptr<ReshardingCostMatrix> cost_matrix);

private:
  // number of rows in matrix
  int num_rows_;

  // number of columns in matrix
  int num_cols_;

  // solver for which to create variables for
  std::shared_ptr<MPSolver> solver_;

  // matrix of MPVariables that the class represents
  // TODO: optimize using a single vector
  std::vector<std::vector<MPVariable*>> matrix_; 

};

} // xla

#endif // XLA_SERVICE_EXPERIMENTAL_VARIABLE_MATRIX_H_
