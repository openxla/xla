/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/tpu/x64_literal_linearizer.h"

#include <type_traits>
#include <utility>

#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"
#include "third_party/tensorflow/core/profiler/lib/traceme.h"

namespace xla::tpu {

namespace {

template <typename T>
struct FailHelper : std::false_type {};

template <typename X64Type, typename X32Type>
void X64ToTupleByType(const LiteralSlice& literal, Literal* tuple) {
  auto data = literal.data<X64Type>();
  auto* first = tuple->data<X32Type>({0}).data();
  auto* second = tuple->data<X32Type>({1}).data();
  for (const X64Type& v : data) {
    if constexpr (std::is_same_v<X64Type, int64_t> ||
                  std::is_same_v<X64Type, uint64_t>) {
      static_assert(std::is_same_v<X32Type, uint32_t>);
      *first = (v & 0xffffffff);
      *second = (v >> 32);
    } else if constexpr (std::is_same_v<X64Type, complex64>) {
      static_assert(std::is_same_v<X32Type, float>);
      *first = v.real();
      *second = v.imag();
    } else if constexpr (std::is_same_v<X64Type, double>) {
      static_assert(std::is_same_v<X32Type, float>);
      std::tie(*first, *second) = SplitF64ToF32(v);
    } else {
      static_assert(FailHelper<X64Type>{});
    }
    ++first;
    ++second;
  }
}

/* static */ Shape X64ComponentShape(const Shape& shape) {
  if (!ShapeUtil::ElementHasBitWidth(shape, 64)) {
    return shape;
  }
  if (ShapeUtil::ElementIsComplex(shape)) {
    return ShapeUtil::ComplexComponentShape(shape);
  }
  if (ShapeUtil::ElementIsFloating(shape)) {
    return ShapeUtil::ChangeElementType(shape, F32);
  }
  return ShapeUtil::ChangeElementType(shape, U32);
}

}  // namespace

/* static */ StatusOr<Literal> X64LiteralLinearizer::X64ToTuple(
    const LiteralSlice& literal) {
  tensorflow::profiler::TraceMe trace_me(
      [&] {
        return absl::StrCat("X64ToTuple ",
                            ShapeUtil::HumanStringWithLayout(literal.shape()));
      },
      /*level=*/2);
  TF_RET_CHECK(ShapeUtil::ElementHasBitWidth(literal.shape(), 64));
  Shape component_shape = X64ComponentShape(literal.shape());
  Literal tuple(
      ShapeUtil::MakeTupleShapeWithPtrs({&component_shape, &component_shape}));

  switch (literal.shape().element_type()) {
    case C64:
      X64ToTupleByType<complex64, float>(literal, &tuple);
      break;
    case S64:
      X64ToTupleByType<int64_t, uint32_t>(literal, &tuple);
      break;
    case U64:
      X64ToTupleByType<uint64_t, uint32_t>(literal, &tuple);
      break;
    case F64:
      X64ToTupleByType<double, float>(literal, &tuple);
      break;
    default:
      return Unimplemented("Unsupported X64 type: %s",
                           PrimitiveType_Name(literal.shape().element_type()));
  }
  return std::move(tuple);
}

/* static */ Status X64LiteralLinearizer::X64FromTuple(
    const LiteralSlice& low, const LiteralSlice& high,
    MutableBorrowingLiteral literal) {
  tensorflow::profiler::TraceMe trace_me(
      [&] {
        return absl::StrCat("X64FromTuple ",
                            ShapeUtil::HumanStringWithLayout(literal.shape()));
      },
      /*level=*/2);
  TF_RET_CHECK(ShapeUtil::ElementHasBitWidth(literal.shape(), 64));
  Shape component_shape = X64ComponentShape(literal.shape());
  TF_RET_CHECK(ShapeUtil::Compatible(low.shape(), component_shape));
  TF_RET_CHECK(ShapeUtil::Compatible(high.shape(), component_shape));
  switch (literal.shape().element_type()) {
    case C64: {
      LiteralSlice real = low;
      LiteralSlice imag = high;
      TF_RETURN_IF_ERROR(
          literal.Populate<complex64>([&](absl::Span<const int64_t> indices) {
            return complex64(real.Get<float>(indices),
                             imag.Get<float>(indices));
          }));
      break;
    }
    case S64:
      TF_RETURN_IF_ERROR(
          literal.Populate<int64_t>([&](absl::Span<const int64_t> indices) {
            return static_cast<int64_t>(
                low.Get<uint32_t>(indices) +
                (static_cast<uint64_t>(high.Get<uint32_t>(indices)) << 32));
          }));
      break;
    case U64:
      TF_RETURN_IF_ERROR(
          literal.Populate<uint64_t>([&](absl::Span<const int64_t> indices) {
            return low.Get<uint32_t>(indices) +
                   (static_cast<uint64_t>(high.Get<uint32_t>(indices)) << 32);
          }));
      break;
    case F64:
      TF_RETURN_IF_ERROR(
          literal.Populate<double>([&](absl::Span<const int64_t> indices) {
            return low.Get<float>(indices) +
                   static_cast<double>(high.Get<float>(indices));
          }));
      break;
    default:
      return Unimplemented("Unsupported complex type: %s",
                           PrimitiveType_Name(literal.shape().element_type()));
  }
  return OkStatus();
}

}  // namespace xla::tpu
