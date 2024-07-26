// TODO: license

#ifndef XLA_SERVICE_EXPERIMENTAL_INSTRUCTION_STRATEGIES_H_
#define XLA_SERVICE_EXPERIMENTAL_INSTRUCTION_STRATEGIES_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/experimental/sharding_strategy.h"

#include <vector>

namespace xla {

class InstructionStrategies {
public:
  InstructionStrategies(HloInstruction* orig_instr);
  ~InstructionStrategies() = default;
  InstructionStrategies(const InstructionStrategies& info) = default;

  // accessors
  std::vector<ShardingStrategy>& sharding_strats() { 
    return sharding_strats_;
  };

  int num_sharding_strats() {
    return sharding_strats_.size();
  }

  void set_operand_strats(
      std::vector<std::shared_ptr<InstructionStrategies>>& operand_strats) {
    operand_strats_ = operand_strats;
  }

  void set_user_strats(
      std::vector<std::shared_ptr<InstructionStrategies>>& user_strats) {
    user_strats_ = user_strats;
  }

  // takes the index of sharding_strats_ and sets the sharding
  // of the instruction
  void set_chosen_strat(int idx);

private:

  // Points to the original instruction that will have its
  // sharding strategies enumerated. Eventually, this instruction
  // will be modified with a sharding strategy provided by the solvers
  HloInstruction* orig_instr_;

  // Pointers to strategies of operands of this instruction
  std::vector<std::shared_ptr<InstructionStrategies>> operand_strats_;

  // Pointers to strategies of users of this instruction
  std::vector<std::shared_ptr<InstructionStrategies>> user_strats_;

  // vector of sharding strategies for the given instruction
  std::vector<ShardingStrategy> sharding_strats_;

};

} // xla

#endif // XLA_SERVICE_EXPERIMENTAL_INSTRUCTION_STRATEGIES_H_