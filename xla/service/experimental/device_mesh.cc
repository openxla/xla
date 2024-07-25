// TODO: license

#include "xla/service/experimental/device_mesh.h"

// assuming two dimensional mesh heirarchy of nodes and GPUs within nodes
#define NUM_MESH_DIM 2  /* number of dimensions in the mesh grid */
#define MESH_X_DIM 2 /* number of nodes */
#define MESH_Y_DIM 4 /* number of gpus per node */
#define DEVICE_COUNT (MESH_X_DIM * MESH_Y_DIM) /* total number of devices */

namespace xla {

int DeviceMesh::NumDim() { return NUM_MESH_DIM; }
int DeviceMesh::XDimSize() { return MESH_X_DIM; }
int DeviceMesh::YDimSize() { return MESH_Y_DIM; }
int DeviceMesh::DeviceCount() { return MESH_X_DIM * MESH_Y_DIM; }

} // xla