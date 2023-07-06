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

#ifndef XLA_SERVICE_TPU_X64_LITERAL_LINEARIZER_H_
#define XLA_SERVICE_TPU_X64_LITERAL_LINEARIZER_H_

#include "xla/literal.h"
#include "xla/statusor.h"

namespace xla::tpu {

class X64LiteralLinearizer {
 public:
  // Converts a x64 literal to a Tuple of two component literals.
  static StatusOr<Literal> X64ToTuple(const LiteralSlice& literal);

  // Converts two literals to a X64 literal, where the low and high 32-bit
  // components are pulled from respective input literals.
  static Status X64FromTuple(const LiteralSlice& low, const LiteralSlice& high,
                             MutableBorrowingLiteral literal);
};

}  // namespace xla::tpu

#endif  //  XLA_SERVICE_TPU_X64_LITERAL_LINEARIZER_H_
