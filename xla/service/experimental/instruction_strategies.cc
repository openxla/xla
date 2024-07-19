// TODO: license

#include "xla/service/experimental/instruction_strategies.h"

#include "xla/service/experimental/instruction_to_module.h"
#include "xla/service/experimental/sharding_enumeration.h"
#include "xla/service/experimental/sharding_strategy_evaluator.h"

namespace xla {

/*********************************************************/
/* InstructionStrategies Class                         */
/*********************************************************/

InstructionStrategies::InstructionStrategies(HloInstruction* orig_instr) 
  : orig_instr_(orig_instr),
    sharding_strats_(EnumerateShardingStrategies(orig_instr)) {

  // create a single instruction module which will then be used for evaluating
  // all of the sharding strats
  std::unique_ptr<HloModule> single_instr_module = 
    CreateModuleFromInstruction(orig_instr);

  // estimate costs of each sharding strategy
  for (int i = 0; i < sharding_strats_.size(); i++) {
    EvaluateShardingStrat(single_instr_module.get(), &sharding_strats_[i]);
  }

  return;
}

void InstructionStrategies::set_chosen_strat(int idx) {
  assert(0 <= idx && idx < sharding_strats_.size());
  sharding_strats_[idx].ApplyToInstruction(orig_instr_);
  return;
}

} // xla