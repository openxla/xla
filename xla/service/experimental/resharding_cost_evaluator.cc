// TODO: add license

#include "xla/service/experimental/resharding_cost_evaluator.h"

#include "xla/service/experimental/shape_utils.h"

namespace xla {

uint64_t ReshardingCostEvaluator::Evaluate(const Shape& shape, 
    const HloSharding& from_sharding, const HloSharding& to_sharding) {
  
  // shape must be an array
  assert(shape.IsArray());

  // if resharding necessary, return size of shape as heuristic 
  return from_sharding == to_sharding ? 0 : NumBytesFromShape(shape);
}

}