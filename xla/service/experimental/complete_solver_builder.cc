// TODO: license

#include "xla/service/experimental/complete_solver_builder.h"

#include "tsl/platform/logging.h"
#include "tsl/platform/errors.h"

#include "xla/service/experimental/fix_log.h"

namespace xla {

CompleteSolverBuilder::CompleteSolverBuilder() :
    solver_(MPSolver::CreateSolver("SCIP")),
    objective_(solver_->MutableObjective()) {

  objective_->SetMinimization();
  return;
}

// setup variables within the solver
void CompleteSolverBuilder::CreateVars(std::shared_ptr<InstructionStrategies> strats) {

  if (strats->sharding_strats().size() == 0) {
    return;
  }

  // create variables representing which sharding strategy is chosen
  solver_->MakeBoolVarArray(
    strats->sharding_strats().size(),
    "",
    &var_map_[strats].comp_vars
  );

  // for each user, create a matrix representing the resharding choices
  int num_rows = strats->num_sharding_strats();
  int num_cols;
  for (auto& user_strats : strats->user_strats()) {
    var_map_[strats].resharding_vars.push_back(
      std::make_shared<VariableMatrix>(
        solver_,
        num_rows, num_cols,
        true, 0, 1 /* binary variable specification */
      )
    );
  }
  

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