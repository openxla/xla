// TODO: license?

#ifndef XLA_SERVICE_EXPERIMENTAL_SHARDING_STRATEGY_SELECTOR_H_
#define XLA_SERVICE_EXPERIMENTAL_SHARDING_STRATEGY_SELECTOR_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/experimental/instruction_strategies.h"

#include <unordered_map>

namespace xla {

class ShardingStrategySelector {
public:
  // TODO: pass in an object that specifies how to setup the code
  ShardingStrategySelector() = default;

  bool Select(std::unordered_map<HloInstruction*, 
    std::shared_ptr<InstructionStrategies>> strat_map);

};

} // xla


#endif // XLA_SERVICE_EXPERIMENTAL_SHARDING_STRATEGY_SELECTOR_H_