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
  LinearExpr SumRow(int i);

  // returns a linear expression of the sum of the i'th column
  // no equality constraint
  LinearExpr SumCol(int i);

  // returns a linear expression of the sum of all elements in the array
  // no equality constraint
  LinearExpr Sum(int i);

  // sets the coefficient of the variable in position [i, j] of the matrix
  // for the solver objective
  void SetCoefficient(int i, int j, uint64_t c);

};

} // xla

#endif // XLA_SERVICE_EXPERIMENTAL_VARIABLE_MATRIX_H_
