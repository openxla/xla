// TODO: license

#ifndef XLA_SERVICE_EXPERIMENTAL_MESH_H_
#define XLA_SERVICE_EXPERIMENTAL_MESH_H_

#include <stdint.h>

namespace xla {

// NOTE: class for now in case Mesh initialized from other variables in future
class Mesh {
public:
  static int NumDim();
  static int XDimSize();
  static int YDimSize();
  static int DeviceCount();
};

} // xla

#endif // XLA_SERVICE_EXPERIMENTAL_MESH_H_