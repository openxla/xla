/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tools/mlir_interpreter/interpreter_value.h"

#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/ADT/ArrayRef.h"
#include "xla/tools/mlir_interpreter/tensor_or_memref.h"

namespace mlir {
namespace interpreter {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(InterpreterValueTest, FillUnitTensor) {
  auto t = TensorOrMemref<int64_t>::Empty({});
  t.buffer->storage[0] = 42;
  InterpreterValue v{t};
  v.Fill([](llvm::ArrayRef<int64_t>) { return InterpreterValue{int64_t{43}}; });
  ASSERT_EQ(t.buffer->storage[0], 43);
}

TEST(InterpreterValueTest, Fill1DTensor) {
  auto t = TensorOrMemref<int64_t>::Empty({3});
  InterpreterValue v{t};
  v.Fill([](llvm::ArrayRef<int64_t> indices) {
    return InterpreterValue{indices[0]};
  });
  ASSERT_EQ(t.buffer->storage[0], 0);
  ASSERT_EQ(t.buffer->storage[1], 1);
  ASSERT_EQ(t.buffer->storage[2], 2);
}

TEST(InterpreterValueTest, FillZeroSizedTensor) {
  auto t = TensorOrMemref<int64_t>::Empty({0, 1});
  InterpreterValue v{t};
  v.Fill([](llvm::ArrayRef<int64_t> indices) {
    LOG(FATAL) << "Callback should not be called";
    return InterpreterValue{indices[0]};
  });
}

TEST(InterpreterValueTest, TypedAlike) {
  InterpreterValue v{TensorOrMemref<int32_t>::Empty({})};
  auto typed_alike = v.TypedAlike({1, 2, 3});
  ASSERT_TRUE(
      std::holds_alternative<TensorOrMemref<int32_t>>(typed_alike.storage));
  ASSERT_THAT(typed_alike.view().sizes, ElementsAre(1, 2, 3));
}

TEST(InterpreterValueTest, AsUnitTensor) {
  InterpreterValue v{42};
  InterpreterValue wrapped = v.AsUnitTensor();
  ASSERT_THAT(wrapped.view().sizes, IsEmpty());
  ASSERT_EQ(
      std::get<TensorOrMemref<int32_t>>(wrapped.storage).buffer->storage[0],
      42);
}

TEST(InterpreterValueTest, IsTensor) {
  ASSERT_FALSE(InterpreterValue{42}.IsTensor());
  ASSERT_TRUE(InterpreterValue{TensorOrMemref<int32_t>::Empty({})}.IsTensor());
}

TEST(InterpreterValueTest, AsInt) {
  ASSERT_EQ(InterpreterValue{int64_t{42}}.AsInt(), 42);
  ASSERT_EQ(InterpreterValue{int32_t{42}}.AsInt(), 42);
  ASSERT_EQ(InterpreterValue{int16_t{42}}.AsInt(), 42);
}

TEST(InterpreterValueTest, CloneTensor) {
  auto tensor = TensorOrMemref<int64_t>::Empty({3});
  tensor.buffer->storage = {1, 2, 3};

  InterpreterValue wrapped{tensor};
  auto clone = wrapped.Clone();
  tensor.buffer->storage = {4, 5, 6};

  auto& cloned_tensor = std::get<TensorOrMemref<int64_t>>(clone.storage);
  ASSERT_THAT(cloned_tensor.buffer->storage, ElementsAre(1, 2, 3));
}

TEST(InterpreterValueTest, CloneScalar) {
  InterpreterValue value{42};
  auto clone = value.Clone();
  ASSERT_THAT(std::get<int32_t>(clone.storage), 42);
}

TEST(InterpreterValueTest, ToString) {
  InterpreterValue value{TensorOrMemref<int64_t>::Empty({3})};
  ASSERT_EQ(value.ToString(), "TensorOrMemref<3xi64>: [0, 0, 0]");
}

TEST(InterpreterValueTest, ToString2d) {
  InterpreterValue value{TensorOrMemref<int64_t>::Empty({3, 2})};
  ASSERT_EQ(value.ToString(),
            "TensorOrMemref<3x2xi64>: [[0, 0], [0, 0], [0, 0]]");
}

TEST(InterpreterValueTest, ToString0d) {
  InterpreterValue value{TensorOrMemref<int64_t>::Empty({})};
  ASSERT_EQ(value.ToString(), "TensorOrMemref<i64>: 0");
}

TEST(InterpreterValueTest, ToStringComplex) {
  InterpreterValue value{std::complex<float>{}};
  ASSERT_EQ(value.ToString(), "complex<f32>: 0.000000e+00+0.000000e+00i");
}

}  // namespace
}  // namespace interpreter
}  // namespace mlir
