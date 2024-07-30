
#include "xla/service/experimental/sharding_strategy_selector.h"

#include "xla/service/experimental/complete_solver_builder.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/errors.h"
#include "xla/service/experimental/fix_log.h"

namespace xla {

// Sets the shardings of the HloInstructions based on the best sharding strategy
// selected from the solver
bool ShardingStrategySelector::Select(std::unordered_map<HloInstruction*, 
    std::shared_ptr<InstructionStrategies>> strat_map) {

  // initialize a builder
  CompleteSolverBuilder builder;

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
    if (strats->num_sharding_strats() > 0) {
      strats->set_chosen_strat(builder.GetStratIdx(strats));
    }
  }

  return true;
}

} // xla