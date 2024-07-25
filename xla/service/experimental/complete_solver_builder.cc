// TODO: license

#include "xla/service/experimental/complete_solver_builder.h"

#include "tsl/platform/logging.h"
#include "tsl/platform/errors.h"

#include "xla/service/experimental/fix_log.h"

using ::operations_research::LinearExpr;

namespace xla {


CompleteSolverBuilder::CompleteSolverBuilder() :
    solver_(MPSolver::CreateSolver("SCIP")),
    objective_(solver_->MutableObjective()) {

  objective_->SetMinimization();
  return;
}

// setup variables within the solver
void CompleteSolverBuilder::CreateVars(std::shared_ptr<InstructionStrategies> strats) {

  // ignore if instruction strategies already inside and incorporated
  if (var_map_.count(strats) > 0 && var_map_[strats].size() > 0) {
    return;
  }

  // ignore instruction that doesn't have any sharding strategies
  int num_strats = strats->sharding_strats().size();
  if (num_strats == 0) {
    return;
  }

  // TODO: implement

  return;
}

// setup variable constraints
void CompleteSolverBuilder::AddConstraints(std::shared_ptr<InstructionStrategies> strats) {

  // ignore if no sharding strategies for instruction 
  if (strats->sharding_strats().size() == 0) {
    return;
  }

  // TODO: implement

  return;
}

// setup the objective
void CompleteSolverBuilder::AddInObjective(std::shared_ptr<InstructionStrategies> strats) {

  // ignore if no sharding strategy for instruction
  if (strats->sharding_strats().size() == 0) {
    return;
  }

  // TODO: implement

  return;
}

// call the solver and return whether found an optimal result
bool CompleteSolverBuilder::Solve() {
  // attempt to solve
  const MPSolver::ResultStatus result_status = solver_->Solve();
  if (result_status != MPSolver::OPTIMAL) {
    return false;
  }

  return true;
}


int CompleteSolverBuilder::GetStratIdx(std::shared_ptr<InstructionStrategies> strats) {

  // ignore if no sharding strategies for instruction
  if (strats->sharding_strats().size() == 0) {
    return 0;
  }

  // TODO: implement

  // should not have reached this point with a valid solution
  assert(0);

  return -1;
}

} // xla