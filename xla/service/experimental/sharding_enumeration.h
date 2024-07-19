// TODO: license

#ifndef XLA_SERVICE_EXPERIMENTAL_SHARDING_ENUMERATION_H_
#define XLA_SERVICE_EXPERIMENTAL_SHARDING_ENUMERATION_H_

#include "xla/service/experimental/sharding_strategy.h"
#include "xla/hlo/ir/hlo_instruction.h"

#include <vector>

namespace xla {

std::vector<ShardingStrategy> EnumerateShardingStrategies(
  HloInstruction* instruction);

}

#endif // XLA_SERVICE_EXPERIMENTAL_SHARDING_ENUMERATION_H_


