// TODO: add license

#include "xla/service/experimental/shape_utils.h"

namespace xla {

// This function returns the number of bytes of a primitive type
uint64_t NumBytesFromType(const PrimitiveType type) {
  switch (type) {
  case PrimitiveType::PRED:
  case PrimitiveType::S2:
  case PrimitiveType::S4:
  case PrimitiveType::S8:
  case PrimitiveType::U2:
  case PrimitiveType::U4:
  case PrimitiveType::U8:
  case PrimitiveType::F8E5M2:
  case PrimitiveType::F8E4M3FN:
  case PrimitiveType::F8E4M3B11FNUZ:
  case PrimitiveType::F8E5M2FNUZ:
  case PrimitiveType::F8E4M3FNUZ:
    return 1;
  case PrimitiveType::S16:
  case PrimitiveType::U16:
  case PrimitiveType::F16:
    return 2;
  case PrimitiveType::S32:
  case PrimitiveType::U32:
  case PrimitiveType::F32:
    return 4;
  case PrimitiveType::S64:
  case PrimitiveType::U64:
  case PrimitiveType::F64:
  case PrimitiveType::C64:
    return 8;
  case PrimitiveType::C128:
    return 16;
  default:
    // TODO: determine appropriate way to support the other value types
    assert(0);
    return 0;
  }
}

// This function returns the number of elements in a shape
uint64_t NumElementsFromShape(const Shape& shape) {
  assert(shape.IsArray());

  uint64_t num_elems = 1;
  int num_dims = shape.dimensions_size();

  for (int i = 0; i < num_dims; i++) {
    num_elems *= shape.dimensions(i);
  }

  return num_elems;
}

// This function returns the number of bytes taken by a shape
uint64_t NumBytesFromShape(const Shape& shape) {
  assert(!shape.IsToken() && !shape.IsOpaque());

  // base-case in a tuple-tree is an array
  if (shape.IsArray()) {
    return NumElementsFromShape(shape) * NumBytesFromType(shape.element_type());
  }

  assert(shape.IsTuple());
  int num_tuple_shapes = shape.tuple_shapes_size();
  uint64_t total_bytes = 0;
  for (int i = 0; i < num_tuple_shapes; i++) {
    total_bytes += NumBytesFromShape(shape.tuple_shapes(i));
  }

  return total_bytes;
}

}