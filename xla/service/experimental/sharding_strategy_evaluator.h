// TODO: license

#ifndef XLA_SERVICE_EXPERIMENTAL_SHARDING_STRATEGY_EVALUATOR_H_
#define XLA_SERVICE_EXPERIMENTAL_SHARDING_STRATEGY_EVALUATOR_H_

#include "xla/service/experimental/sharding_strategy.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// This function will evaluate the sharding strategy on the 
// single-instruction module by applying the input shardings from the strat
// onto the operands of the module's root instruction, running GSPMD,
// and evaluating the communication costs of the resulting module
// The strat parameter will be updated with this cost and the resulting
// output sharding
void EvaluateShardingStrat(const HloModule* module, ShardingStrategy* strat);

}

#endif // XLA_SERVICE_EXPERIMENTAL_SHARDING_STRATEGY_EVALUATOR_H_