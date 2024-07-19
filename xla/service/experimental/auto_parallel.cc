// TODO: add copyright 

#include "xla/service/experimental/auto_parallel.h"

#include "xla/service/experimental/instruction_strategies.h"
#include "xla/service/experimental/sharding_strategy.h"
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

namespace xla {

namespace {

  /*********************************************************/
  /* Debugging                                             */
  /*********************************************************/
  // TODO: move into separate file

  std::string LOG_HEADER(int x, const char c[]="AutoParallel: ") {
    return ((x == 0) ? (c) : ((LOG_HEADER(x - 1, c)) + "\t"));
  }

  void PrintProtoList(std::function<int()> length_fn, std::function<int64_t(int)> getter, int depth=3, std::string list_name="array") {
    int n = length_fn();

    std::string s = "";

    for (int i = 0; i < n; i++) {
      s += std::to_string(getter(i)) + " ";
    }

    VLOG(5) << LOG_HEADER(depth, "InstInfo: ") << list_name << s;

    return; 
  }

  void PrintShardingInfo(OpSharding sharding, int depth=3) {

    VLOG(5) << LOG_HEADER(depth + 1, "SharInfo: ") << "sharding: ";

    std::function<int64_t(int)> getter = [&sharding](int index) {
      return (sharding.*static_cast<int64_t (OpSharding::*)(int) const>(&OpSharding::tile_assignment_dimensions))(index);
    };

    PrintProtoList(
      std::bind(&OpSharding::tile_assignment_dimensions_size, &sharding),
      getter,
      depth + 1, "tile_assignment_dimensions:"
    );

    getter = [&sharding](int index) {
      return (sharding.*static_cast<int64_t (OpSharding::*)(int) const>(&OpSharding::tile_assignment_devices))(index);
    };

    PrintProtoList(
      std::bind(&OpSharding::tile_assignment_devices_size, &sharding),
      getter,
      depth + 1, "tile_assignment_devices:"
    );

    getter = [&sharding](int index) {
      return (sharding.*static_cast<int64_t (OpSharding::*)(int) const>(&OpSharding::iota_reshape_dims))(index);
    };

    PrintProtoList(
      std::bind(&OpSharding::iota_reshape_dims_size, &sharding),
      getter,
      depth + 1, "iota_reshape_dims:"
    );

    getter = [&sharding](int index) {
      return (sharding.*static_cast<int32_t (OpSharding::*)(int) const>(&OpSharding::iota_transpose_perm))(index);
    };

    PrintProtoList(
      std::bind(&OpSharding::iota_transpose_perm_size, &sharding),
      getter,
      depth + 1, "iota_transpose_perm:"
    );
  }

  void PrintInstructionInfo(HloInstruction* instruction, int depth=3) {

    int64_t num_operands = instruction->operand_count();
    VLOG(5) << LOG_HEADER(depth, "InstInfo: ") << "Name: " << instruction->name() << " " << instruction;
    VLOG(5) << LOG_HEADER(depth + 1, "InstInfo: ") << "num operands: " << num_operands;
    VLOG(5) << LOG_HEADER(depth + 1, "InstInfo: ") << "sharded: " << instruction->has_sharding();
    VLOG(5) << LOG_HEADER(depth + 1, "InstInfo: ") << "shape: " << instruction->shape().ToString();

    if (instruction->has_sharding()) {
      // convert to Proto and print out proto elements
      PrintShardingInfo(instruction->sharding_ptr()->ToProto(), depth + 1);
    }

    HloInstruction::InstructionVector operands = instruction->operands();
    for (int i = 0; i < num_operands; i++) {
      VLOG(5) << LOG_HEADER(depth + 1, "InstInfo: ") << "op " << i << ": " << operands[i]->name() << " " << operands[i]->shape().ToString() << " " << operands[i];
    }

    return;
  }

  void PrintComputationInfo(HloComputation* computation, int depth=3) {
    VLOG(5) << LOG_HEADER(depth, "CompInfo: ") << "Name: " << computation->name() << " " << computation;
    VLOG(5) << LOG_HEADER(depth, "CompInfo: ") << "Instruction Count: " << computation->instruction_count();

    for (HloInstruction* instr : computation->instructions()) {
      PrintInstructionInfo(instr, depth + 1);
    }
  }

  void PrintModuleInfo(HloModule* module, int depth=1) {

    VLOG(5) << LOG_HEADER(depth, "ModuInfo: ") << "Name: " << module->name() << " " << module;
    VLOG(5) << LOG_HEADER(depth, "ModuInfo: ") << "Computation count: " << module->computation_count();

    for (HloComputation* computation : module->computations()) {
      PrintComputationInfo(computation, depth + 1);
    }

  }

  /*********************************************************/
  /* Resharding Cost Estimation                            */
  /*********************************************************/

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

      // add all user strategies to current instructions strategies
      std::vector<std::shared_ptr<InstructionStrategies>> all_user_strats;
      for (HloInstruction* user : instr->users()) {
        all_user_strats.push_back(map[user]);
      }
      instr_strats->set_user_strats(all_user_strats);

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

    if (!ShardableModule(module)) {
      VLOG(5) << LOG_HEADER(0) << "module = " 
        << module->name() << " not shardable";
      return false;
    }

    VLOG(5) << "Testing AutoParallelizer Run";

    // create a clone of the module, then run off of that 
    std::unique_ptr<HloModule> module_clone = module->Clone();
    VLOG(5) << LOG_HEADER(0) << "module: " << module_clone->name();

    std::unordered_map<HloInstruction*, 
      std::shared_ptr<InstructionStrategies>> info_map;

    // TODO: shouldn't I only be doing this for the main computation?
    // construct relevant sharding information
    for (HloComputation* computation : module_clone->computations()) {
      for (HloInstruction* instr : computation->instructions()) {
        assert(info_map.count(instr) == 0);
        info_map[instr] = std::make_shared<InstructionStrategies>(instr);
      }
    }

    VLOG(5) << "Starting to evaluate the resharding costs";
    EstimateReshardingCosts(info_map);

    VLOG(5) << "Number of instructions: " << info_map.size();

    VLOG(5) << "Done Testing AutoParallelizer Run";
    
    return true;
  }

}   // namespace xla