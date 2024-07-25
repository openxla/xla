// TODO: license

#ifndef XLA_SERVICE_EXPERIMENTAL_SOLVER_BUILDER_H_
#define XLA_SERVICE_EXPERIMENTAL_SOLVER_BUILDER_H_

#include "xla/service/experimental/instruction_strategies.h"

namespace xla {

class SolverBuilder {
public:

  // setup variables within the solver
  virtual void CreateVars(std::shared_ptr<InstructionStrategies> strats) = 0;

  // setup variable constraints
  virtual void AddConstraints(std::shared_ptr<InstructionStrategies> strats) = 0;

  // setup the objective
  virtual void AddInObjective(std::shared_ptr<InstructionStrategies> strats) = 0;

  // return the solver built by the solver builder
  virtual bool Solve() = 0;

  // get the index of the sharding strategy after solving
  virtual int GetStratIdx(std::shared_ptr<InstructionStrategies> strats) = 0;
};

} // xla

#endif // XLA_SERVICE_EXPERIMENTAL_SOLVER_BUILDER_H_