// TODO: license?

#ifndef XLA_SERVICE_EXPERIMENTAL_SHARDING_STRATEGY_SOLVER_H_
#define XLA_SERVICE_EXPERIMENTAL_SHARDING_STRATEGY_SOLVER_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/experimental/instruction_strategies.h"
#include "ortools/linear_solver/linear_solver.h"

#include <unordered_map>

using ::operations_research::MPSolver;
using ::operations_research::MPObjective;

namespace xla {

class ShardingStrategySolver {
public:
  // TODO: pass in an object that specifies how to setup the code
  ShardingStrategySolver();

  bool Solve(std::unordered_map<HloInstruction*, 
    std::shared_ptr<InstructionStrategies>> strat_map);

};

} // xla


#endif // XLA_SERVICE_EXPERIMENTAL_SHARDING_STRATEGY_SOLVER_H_