// TODO: license

#include "xla/service/experimental/sharding_strategy.h"

namespace xla {

void ShardingStrategy::AddOpSharding(HloSharding sharding) {
  operand_shardings_.push_back(std::make_shared<HloSharding>(sharding));
}

void ShardingStrategy::set_result_sharding(HloSharding result_sharding) {
  result_sharding_ = std::make_shared<HloSharding>(result_sharding);
}

void ShardingStrategy::AddUserReshardingCosts(std::vector<uint64_t> costs) {
  resharding_costs_.push_back(std::move(costs));
}

} // xla