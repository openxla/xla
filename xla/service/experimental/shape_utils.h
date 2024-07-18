// TODO: add license

#ifndef XLA_SERVICE_EXPERIMENTAL_SHAPE_UTILS_H_
#define XLA_SERVICE_EXPERIMENTAL_SHAPE_UTILS_H_

#include "xla/shape.h"

#include <stdint.h>

namespace xla {

// Returns the number of bytes allocated by a specific type
// TODO: need to determine what the correct value is for types smaller than
// a byte
uint64_t NumBytesFromType(const PrimitiveType type);

// This function returns the number of elements in a shape
// Shape must be an array shape
uint64_t NumElementsFromShape(const Shape& shape);

// This function returns the number of bytes that would be allocated in an
// object of type shape. Shpae must be either an array or tuple. If tuple,
// then result is the sum of shapes of the tuple's leaves which must be arrays
uint64_t NumBytesFromShape(const Shape& shape);

} // xla


#endif // XLA_SERVICE_EXPERIMENTAL_SHAPE_UTILS_H_