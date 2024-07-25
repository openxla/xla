// TODO: license

#include "xla/service/experimental/sharding_enumeration.h"

#include "xla/service/experimental/device_mesh.h"
#include "xla/hlo/ir/hlo_sharding.h"

namespace xla {

// enumerate sharding from the number of dimensions in the data
// TODO: could be cached
// Constructs a vector of rank * (rank + 1) shardings
std::vector<HloSharding> EnumerateShardingsFromRank(int rank) {

  // two device dimensions currently (assume 4 (nodes) x 8 (gpus per node))
  std::vector<HloSharding> shardings;

  int mesh_x_dim_size = DeviceMesh::XDimSize();
  int mesh_y_dim_size = DeviceMesh::YDimSize();

  // note: this code is only acceptable for a 2D mesh grid,
  // would require more complicated solution for higher-dimensional grids
  for (int x_idx = 0; x_idx < rank; x_idx++) {
    for (int y_idx = 0; y_idx < rank; y_idx++) {
      // TODO: have a simple boolean for whether we would like to shard
      // both mesh grid dimensions on the same data dimension

      // construct tile_assignment_dims
      std::vector<int64_t> tile_assignment_dims(rank, 1);
      tile_assignment_dims[x_idx] *= mesh_x_dim_size;
      tile_assignment_dims[y_idx] *= mesh_y_dim_size;

      // NOTE: intentionally may add two shardings if x_idx == y_idx
      // (i.e. when sharding a single data dimension on all devices)
      // because ordering of machines may influence resulting communication
      // costs and overall problem. Adding both shardings to be complete
      // construct the iota_reshape_dims and iota_tranpose_perm
      if (x_idx <= y_idx) {
        shardings.push_back(HloSharding::IotaTile(
          tile_assignment_dims,
          { mesh_x_dim_size * mesh_y_dim_size },
          { 0 }
        ));
      }
      if (y_idx <= x_idx) {
        shardings.push_back(HloSharding::IotaTile(
          tile_assignment_dims,
          { mesh_x_dim_size, mesh_y_dim_size },
          { 1, 0 }
        ));
      }
    }
  }

  return shardings;
}

// assuming a 2D mesh grid, enumerates all choice 2 shardings of data
// TODO: determine if tuples of data will need to be considered for sharding
std::vector<HloSharding> EnumerateGeneralOpSharding(HloInstruction* operand, 
    HloInstruction* instruction) {
  
  // operand requires sharding
  assert(operand->has_sharding());

  // only sharding array types of data, otherwise no sharding options
  const Shape op_shape = operand->shape();
  if (!op_shape.IsArray()) {
    return {};
  }

  return EnumerateShardingsFromRank(op_shape.rank());
}

// TODO: figure out a better way to deal with tuples for data
std::vector<HloSharding> EnumerateTupleOpSharding(HloInstruction* operand,
    HloInstruction* instruction) {
  return {};
}

// Enumerates the shardings of a single operand instruction
// depending on the user instruction of the operand and whether it is sharded.
// This is a general function for iterating through shardings of a single
// TODO: should give integer argument here and in EnumerateGeneralOpSharding
std::vector<HloSharding> EnumerateOpSharding(
    HloInstruction* operand, HloInstruction* instruction) {
  
  // if sharding already exists for the instruction, only have that sharding
  if (operand->has_sharding()) {
    return { operand->sharding() };
  }

  // otherwise, perform sharding based on type of instruction
  // we are sharding operations for (may want to case on Dot product)
  switch (instruction->opcode()) {
  case HloOpcode::kTuple:
    return EnumerateTupleOpSharding(operand, instruction);
  default:
    return EnumerateGeneralOpSharding(operand, instruction);
  }

}

// Combine shardings for each operator to form sharding strategies
std::vector<ShardingStrategy> CombineShardingVectors(
    std::vector<std::vector<HloSharding>> sharding_vecs) {
  int num_vecs = sharding_vecs.size();

  if (num_vecs == 0) {
    return {};
  } else if (num_vecs == 1) {
    // only one operator, map each sharding to a separate ShardingStrategy
    std::vector<ShardingStrategy> strats;
    for (HloSharding sharding : sharding_vecs[0]) {
      ShardingStrategy strat;
      strat.AddOpSharding(sharding);
      strats.push_back(strat);
    }
    return strats;
  }

  // otherwise recurse
  std::vector<HloSharding> shardings = sharding_vecs[num_vecs - 1];
  std::vector<ShardingStrategy> sub_strats = CombineShardingVectors(
    std::vector<std::vector<HloSharding>>(sharding_vecs.begin(), 
      sharding_vecs.end() - 1)
  );

  std::vector<ShardingStrategy> strats;
  for (HloSharding sharding : shardings) {
    for (ShardingStrategy strat : sub_strats) {
      // copy the existing sub_strat and add the new sharding
      strat.AddOpSharding(sharding);
      strats.push_back(strat);
    }
  }
  
  return strats;
}

// Enumerates all possible sharding strategies on the inputs of the current
// instruction
// TODO: need to make instruction sharding use shared pointers
// going to be many identical copies of the same sharding in memory
// for larger problems
std::vector<ShardingStrategy> EnumerateShardingStrategies(
    HloInstruction* instruction) {

  // enumerate through the shardings for each operator of the instruction
  std::vector<std::vector<HloSharding>> all_op_shardings;

  // TODO: pass index of operand to distinguish from other operands
  // if necessary
  HloInstruction::InstructionVector operands = instruction->operands();
  for (HloInstruction* op : operands) {
    all_op_shardings.push_back(EnumerateOpSharding(op, instruction));
  }

  return CombineShardingVectors(all_op_shardings);
} 

}