
#include "xla/service/experimental/sharding_strategy_solver.h"

#include "xla/service/experimental/simple_solver_builder.h"

// #include "absl/log/log.h"

#include "tsl/platform/logging.h"
#include "tsl/platform/errors.h"

#include "xla/service/experimental/fix_log.h"

namespace xla {

// incorporate the object that defines solver setup
ShardingStrategySolver::ShardingStrategySolver() {
  return;
}

// Sets the shardings of the HloInstructions based on the best sharding strategy
// selected from the solver
bool ShardingStrategySolver::Solve(std::unordered_map<HloInstruction*, 
    std::shared_ptr<InstructionStrategies>> strat_map) {

  VLOG(5) << "Initializing Solver";

  // initialize a builder
  SimpleSolverBuilder builder;

  // create variables, construct their constraints, and add to the objective
  for (auto& [instr, strats] : strat_map) {
    builder.CreateVars(strats);
  }

  for (auto& [instr, strats] : strat_map) {
    builder.AddConstraints(strats);
  }

  for (auto& [instr, strats] : strat_map) {
    builder.AddInObjective(strats);
  }

  // solve the problem
  if (!builder.Solve()) {
    return false;
  }

  // success, determine which sharding to load into the instruction
  for (auto& [instr, strats] : strat_map) {
    strats->set_chosen_strat(builder.GetStratIdx(strats));
  }

  return true;
}

} // xla