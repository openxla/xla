// TODO: license

#ifndef XLA_SERVICE_EXPERIMENTAL_SHARDING_ENUMERATION_H_
#define XLA_SERVICE_EXPERIMENTAL_SHARDING_ENUMERATION_H_

#include "xla/service/experimental/sharding_strategy.h"
#include "xla/hlo/ir/hlo_instruction.h"

#include <vector>

// assuming two dimensional mesh heirarchy of nodes and GPUs within nodes
#define NUM_MESH_DIM 2  /* number of dimensions in the mesh grid */
#define MESH_X_DIM 2 /* number of nodes */
#define MESH_Y_DIM 4 /* number of gpus per node */
#define DEVICE_COUNT (MESH_X_DIM * MESH_Y_DIM) /* total number of devices */

namespace xla {

std::vector<ShardingStrategy> EnumerateShardingStrategies(
  HloInstruction* instruction);

}

#endif // XLA_SERVICE_EXPERIMENTAL_SHARDING_ENUMERATION_H_


