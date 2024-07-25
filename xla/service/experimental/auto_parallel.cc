// TODO: add copyright 

#include "xla/service/experimental/auto_parallel.h"

#include "xla/service/experimental/instruction_strategies.h"
#include "xla/service/experimental/sharding_strategy.h"
#include "xla/service/experimental/sharding_strategy_solver.h"
#include "xla/service/experimental/resharding_cost_evaluator.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/errors.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"

#include <stdint.h>

#include "xla/service/experimental/debug.h"


namespace xla {

namespace {

  /*********************************************************/
  /* Resharding Cost Estimation                            */
  /*********************************************************/

  void CompleteInstructionStrategiesGraph(std::unordered_map<HloInstruction*,
      std::shared_ptr<InstructionStrategies>>& map) {

    std::vector<std::shared_ptr<InstructionStrategies>> strats;

    // connect edges for all instruction nodes
    for (auto& [instr, instr_strats] : map) {
      // add all operand strategies to the instr_strats
      strats.clear();
      for (HloInstruction* operand : instr->operands()) {
        strats.push_back(map[operand]);
      }
      instr_strats->set_operand_strats(strats);

      // add all user strategies to instr_strats
      strats.clear();
      for (HloInstruction* operand : instr->users()) {
        strats.push_back(map[operand]);
      }
      instr_strats->set_user_strats(strats);
    }
  }

  void EstimateReshardingCosts(std::unordered_map<HloInstruction*, 
      std::shared_ptr<InstructionStrategies>>& map) {
    
    // for each instruction, for each user of it, for each sharding strategy
    // recalculate resharding costs

    ReshardingCostEvaluator evaluator;

    for (auto& [instr, instr_strats] : map) {
      const Shape& shape = instr->shape();
      
      // don't loop if no users
      if (instr->user_count() == 0) {
        continue;
      }

      // deteremine costs for each sharding strategy 
      for (ShardingStrategy& strat: instr_strats->sharding_strats()) {
        // TODO: can cache this iteration to reduce time
        std::shared_ptr<HloSharding> in_sharding = strat.result_sharding();

        for (HloInstruction* user : instr->users()) {
          int op_idx = user->operand_index(instr);
          std::vector<uint64_t> resharding_costs;
          for (ShardingStrategy& out_strat : map[user]->sharding_strats()) {
            uint64_t cost = evaluator.Evaluate(
              shape, *in_sharding.get(), *out_strat.GetOpSharding(op_idx).get()
            );
            resharding_costs.push_back(cost);
          }
          strat.AddUserReshardingCosts(resharding_costs);
        }
      }

    }
    
    return;
  }

  /*********************************************************/
  /* Additional Helper Functions                           */
  /*********************************************************/

  // TODO: determine if this is a valid approach to determine 
  // if the module is the main module for the computations
  bool ShardableModule(HloModule* module) {

    std::string name = module->name();
    return name.find("convert_element_type") == std::string::npos &&
      name.find("broadcast_in_dim") == std::string::npos &&
      name.find("_multi_slice") == std::string::npos;
  }

}   // namespace


  /*********************************************************/
  /* AutoParallelizer Pass Implementation                  */
  /*********************************************************/

  // overriden functions from class
  // modifies the sharding specification of the instructions in the module
  // TODO: need to ignore the default modules that are not the core computation
  //  e.g. jit_convert_element_type, jit_broadcast_in_dim, jit__multi_slice
  // otherwise, just too much time spent on these things
  absl::StatusOr<bool> AutoParallelizer::Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) {

    VLOG(5) << "Starting AutoParallelizer Run";

    if (!ShardableModule(module)) {
      VLOG(5) << LOG_HEADER(0) << "module = " 
        << module->name() << " not shardable";
      return false;
    }

    // create a clone of the module, then run off of that 
    VLOG(5) << LOG_HEADER(0) << "module: " << module->name();

    std::unordered_map<HloInstruction*, 
      std::shared_ptr<InstructionStrategies>> info_map;

    // TODO: shouldn't I only be doing this for the main computation?
    // construct relevant sharding information
    for (HloComputation* computation : module->computations()) {
      for (HloInstruction* instr : computation->instructions()) {
        assert(info_map.count(instr) == 0);
        info_map[instr] = std::make_shared<InstructionStrategies>(instr);
      }
    }

    CompleteInstructionStrategiesGraph(info_map);
    EstimateReshardingCosts(info_map);

    // TODO: refactor to ShardingStrategySelector
    ShardingStrategySolver solver;
    bool successful = solver.Solve(info_map);
    VLOG(5) << successful;

    VLOG(5) << "Done AutoParallelizer Run";
    
    return true;
  }

}   // namespace xla