// TODO: license

#ifndef XLA_SERVICE_EXPERIMENTAL_COMPLETE_STRATEGY_GRAPH_H_
#define XLA_SERVICE_EXPERIMENTAL_COMPLETE_STRATEGY_GRAPH_H_

#include "xla/service/experimental/instruction_strategies.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// connects various instruction strategies by their users and operands
// and computes the resharding costs between two instructions
// all of this information is stored in the various InstructionStrategies
void CompleteStrategyGraph(std::unordered_map<HloInstruction*,
  std::shared_ptr<InstructionStrategies>>& map);

} // xla

#endif // XLA_SERVICE_EXPERIMENTAL_COMPLETE_STRATEGY_GRAPH_H_