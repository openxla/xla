// TODO: add license

#ifndef XLA_SERVICE_EXPERIMENTAL_RESHARDING_COST_EVALUATOR_H_
#define XLA_SERVICE_EXPERIMENTAL_RESHARDING_COST_EVALUATOR_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"

#include <stdint.h>

namespace xla {

class ReshardingCostEvaluator{
public:
  ReshardingCostEvaluator() = default;
  ~ReshardingCostEvaluator() = default;

  // This function returns a heuristic cost of converting a shape array
  // from a specific sharding to another sharding
  // NOTE: shape must be an array type
  uint64_t Evaluate(const Shape& shape, const HloSharding& from_sharding,
    const HloSharding& to_sharding);

};

} // xla


#endif // XLA_SERVICE_EXPERIMENTAL_RESHARDING_COST_EVALUATOR_H_