// TODO: license

#include "xla/service/experimental/simple_solver_builder.h"

#include "tsl/platform/logging.h"
#include "tsl/platform/errors.h"

namespace xla {

SimpleSolverBuilder::SimpleSolverBuilder() :
    solver_(MPSolver::CreateSolver("SCIP")) {
  return;
}

// setup variables within the solver
void SimpleSolverBuilder::CreateVars(std::shared_ptr<InstructionStrategies> strats) {

  // ignore if instruction strategies already inside
  if (var_map_.count(strats) > 0) {
    return;
  }

  // create a variable for each strategy
  solver_->MakeBoolVarArray(
    strats->sharding_strats().size(), 
    "", 
    &var_map_[strats]
  );

  VLOG(5) << "\tLength of variable map: " << var_map_[strats].size();

  return;
}

// setup variable constraints
void SimpleSolverBuilder::AddConstraints(std::shared_ptr<InstructionStrategies> strats) {
  return;
}

// setup the objective
void SimpleSolverBuilder::AddInObjective(std::shared_ptr<InstructionStrategies> strats) {
  return;
}

// return the solver built by the solver builder
bool SimpleSolverBuilder::Solve() {
  return false;
}


int SimpleSolverBuilder::GetStratIdx(std::shared_ptr<InstructionStrategies> strats) {
  return 0;
}

} // xla