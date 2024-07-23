// TODO: license

#include "xla/service/experimental/simple_solver_builder.h"

namespace xla {

SimpleSolverBuilder::SimpleSolverBuilder() {
  return;
}

// setup variables within the solver
void CreateVars(std::shared_ptr<InstructionStrategies> strats) {
  return;
}

// setup variable constraints
void AddConstraints(std::shared_ptr<InstructionStrategies> strats) {
  return;
}

// setup the objective
void AddInObjective(std::shared_ptr<InstructionStrategies> strats) {
  return;
}

// return the solver built by the solver builder
bool Solve() {
  return false;
}


int GetStratIdx(InstructionStrategies& strats) {
  return 0;
}

} // xla