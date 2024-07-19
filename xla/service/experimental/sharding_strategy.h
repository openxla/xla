// TODO: license
#ifndef XLA_SERVICE_EXPERIMENTAL_SHARDING_STRATEGY_H_
#define XLA_SERVICE_EXPERIMENTAL_SHARDING_STRATEGY_H_

#include "xla/hlo/ir/hlo_sharding.h"

namespace xla {

class ShardingStrategy {
public:
  ShardingStrategy() = default;
  ~ShardingStrategy() = default;
  ShardingStrategy(const ShardingStrategy& s) = default;
  ShardingStrategy(ShardingStrategy&& s) = default;

  // cost getters and setters
  uint64_t cost() const { return cost_; }
  void set_cost(uint64_t cost) { cost_ = cost; }

  // modifying the operand_shardings
  // TODO: accept a shared pointer
  void AddOpSharding(HloSharding sharding);
  std::shared_ptr<HloSharding> GetOpSharding(int op_idx) {
    return operand_shardings_[op_idx];
  };
  int64_t NumOpShardings() { return operand_shardings_.size(); }

  // modifying resulting sharding
  // TODO: accept a shared pointer
  void set_result_sharding(HloSharding result_sharding);
  std::shared_ptr<HloSharding> result_sharding() { return result_sharding_; };

  void AddUserReshardingCosts(std::vector<uint64_t> resharding_costs);

private:
  // TODO: make these shared_ptr<const HloSharding>
  // The sharding of each operand of an instruction. Using shared_ptr
  // as noted by HloInstruction due to large size for many element tuples
  // This vector will be filled by enumerating incomplete sharding strategies
  std::vector<std::shared_ptr<HloSharding>> operand_shardings_;

  // Cost of this specific instruction sharding. This will be assigned
  // after evaluating the cost of the complete HloModule after performing
  // sharding propagation through SPMD.
  uint64_t cost_;

  // TODO: make these shared_ptr<const HloSharding>
  // Sharding of result of computing instruction. This will be completed
  // by GSPMD when given the input shardings and determining the output
  // shardings.
  std::shared_ptr<HloSharding> result_sharding_;

  // For each user of this instruction, have a resharding cost for
  // each of the user's potential sharding strategies
  // Cost vectors are added in user order
  std::vector<std::vector<uint64_t>> resharding_costs_;

};

} // xla

#endif // XLA_SERVICE_EXPERIMENTAL_SHARDING_STRATEGY_H_