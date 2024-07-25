// TODO: license

#ifndef XLA_SERVICE_EXPERIMENTAL_COMPLETE_SOLVER_BUILDER_H_
#define XLA_SERVICE_EXPERIMENTAL_COMPLETE_SOLVER_BUILDER_H_

#include "xla/service/experimental/solver_builder.h"
#include "ortools/linear_solver/linear_solver.h"

using ::operations_research::MPSolver;
using ::operations_research::MPObjective;
using ::operations_research::MPVariable;

namespace xla {

// This solver builder will ignore the resharding costs between 
// instructions and will only perform the naive optimization of choosing a 
// sharding strategy based off of their costs
class CompleteSolverBuilder : SolverBuilder {
public:
  CompleteSolverBuilder();

  // setup variables within the solver
  void CreateVars(std::shared_ptr<InstructionStrategies> strats) override final;

  // setup variable constraints
  void AddConstraints(std::shared_ptr<InstructionStrategies> strats) override final;

  // setup the objective
  void AddInObjective(std::shared_ptr<InstructionStrategies> strats) override final;

  // return the solver built by the solver builder
  bool Solve() override final;

  // get the index of the sharding strategy after solving
  int GetStratIdx(std::shared_ptr<InstructionStrategies> strats) override final;

private:
  // solver that will be built
  std::unique_ptr<MPSolver> solver_;

  // objective to optimization problem
  MPObjective* const objective_;

  // map to hold the solver variables associated with an instruction
  std::unordered_map<std::shared_ptr<InstructionStrategies>, std::vector<MPVariable*>> var_map_;
};

} // xla

#endif // XLA_SERVICE_EXPERIMENTAL_SIMPLE_SOLVER_BUILDER_H_