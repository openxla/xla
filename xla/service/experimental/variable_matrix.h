// TODO: license

#ifndef XLA_SERVICE_EXPERIMENTAL_VARIABLE_MATRIX_H_
#define XLA_SERVICE_EXPERIMENTAL_VARIABLE_MATRIX_H_

#include "ortools/linear_solver/linear_solver.h"

#include "stdint.h"

using ::operations_research::MPSolver;
using ::operations_research::MPObjective;
using ::operations_research::MPVariable;
using ::operations_research::LinearExpr;

namespace xla {

// This class represents a matrix of MPVariables and provides methods for
// taking specific sums and setting coefficients of various variables

class VariableMatrix {
public:
  // defines num_rows * num_cols MPVariables with lower and upper bounds
  // of lb and ub for the provided solver
  VariableMatrix(MPSolver* solver, int num_rows, int num_cols, int lb, int ub);

  // returns a linear expression of the sum of the i'th row 
  // no equality constraint
  LinearExpr SumRow(int i) const;

  // returns a linear expression of the sum of the i'th column
  // no equality constraint
  LinearExpr SumCol(int i) const;

  // returns a linear expression of the sum of all elements in the array
  // no equality constraint
  LinearExpr Sum(int i) const;

  // sets the coefficient of the variable in position [i, j] of the matrix
  // for the solver objective
  void SetCoefficient(int i, int j, uint64_t c);

private:
  // number of rows in matrix
  int num_rows_;

  // number of columns in matrix
  int num_cols_;

  // solver for which to create variables for
  MPSolver* solver_;

  // matrix of MPVariables that the class represents
  std::vector<std::vector<MPVariable*>> matrix_; 

};

} // xla

#endif // XLA_SERVICE_EXPERIMENTAL_VARIABLE_MATRIX_H_
