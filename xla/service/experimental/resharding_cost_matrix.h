// TODO: license

#ifndef XLA_SERVICE_EXPERIMENTAL_RESHARDING_COST_MATRIX_H_
#define XLA_SERVICE_EXPERIMENTAL_RESHARDING_COST_MATRIX_H_

#include "xla/shape.h"
#include "xla/hlo/ir/hlo_sharding.h"

#include <vector>
#include <stdint.h>

namespace xla {

class ReshardingCostMatrix {
public:

  // construct matrix between two InstructionStrategies objects
  // resulting matrix will have shape |s1| x |s2| where |s| is the number
  // of sharding strategies in s
  ReshardingCostMatrix(const Shape& shape, 
    std::vector<std::shared_ptr<HloSharding>>& strats1, 
    std::vector<std::shared_ptr<HloSharding>>& strats2);

  // returns the number of elements in the matrix (num_rows * num_cols)
  int size() const { return num_rows_ * num_cols_; };

  // returns the number of rows in the matrix (|s1|)
  int num_rows() const { return num_rows_; }

  // returns the number of cols in the matrix (|s2|)
  int num_cols() const { return num_cols_; }

  // returns the resharding cost between the r'th sharding strategy of s1
  // and the c'th sharding strategy of s2
  uint64_t CostAt(int r, int c);

private:

  int num_rows_;
  int num_cols_;

  // matrix for storing resharding cost information
  // TODO: optimize using a single vector
  std::vector<std::vector<uint64_t>> costs_;

};

} // xla

#endif // XLA_SERVICE_EXPERIMENTAL_RESHARDING_COST_MATRIX_H_