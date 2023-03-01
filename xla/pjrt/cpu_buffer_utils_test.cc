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

#include "xla/pjrt/cpu_buffer_utils.h"

#include <cstring>
#include <memory>

#include <gtest/gtest.h>
#include "xla/literal_util.h"
#include "xla/runtime/cpu_event.h"
#include "xla/tests/literal_test_util.h"
#include "tsl/platform/statusor.h"
#include "tfrt/concurrency/async_value_ref.h"  // from @tf_runtime

namespace xla {
namespace {

TEST(CpuBufferUtilsTest, LeafBuffer) {
  auto literal = LiteralUtil::CreateR2<uint32_t>({{1, 2}, {3, 4}});
  const Shape& shape = literal.shape();
  ASSERT_FALSE(shape.IsTuple());

  size_t byte_size = ShapeUtil::ByteSizeOf(shape);
  TF_ASSERT_OK_AND_ASSIGN(auto buffer,
                          MaybeOwningCpuMemory::AllocateShared(byte_size));
  auto definition_event = tsl::MakeAvailableAsyncValueRef<runtime::CpuEvent>();
  std::memcpy(buffer->data(), literal.untyped_data(), buffer->size());
  TrackedTfrtCpuDeviceBuffer tracked_buffer(/*is_tuple=*/false, {buffer},
                                            definition_event,
                                            /*on_delete_callback_=*/nullptr);

  auto expected_literal = std::make_unique<Literal>(shape);
  CopyCpuBufferToLiteral(shape, &tracked_buffer, expected_literal.get());
  EXPECT_TRUE(LiteralTestUtil::Equal(literal, *expected_literal));
}

TEST(CpuBufferUtilsTest, TupledBuffer) {
  auto scalar = LiteralUtil::CreateR0<float>(1.0);
  auto matrix = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto literal = LiteralUtil::MakeTuple({&scalar, &matrix});
  const Shape& shape = literal.shape();
  ASSERT_TRUE(shape.IsTuple());

  size_t byte_size_0 = ShapeUtil::ByteSizeOf(shape.tuple_shapes(0));
  size_t byte_size_1 = ShapeUtil::ByteSizeOf(shape.tuple_shapes(1));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_0,
                          MaybeOwningCpuMemory::AllocateShared(byte_size_0));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_1,
                          MaybeOwningCpuMemory::AllocateShared(byte_size_1));
  auto definition_event = tsl::MakeAvailableAsyncValueRef<runtime::CpuEvent>();
  TrackedTfrtCpuDeviceBuffer tracked_buffer(
      /*is_tuple=*/true, {buffer_0, buffer_1}, definition_event,
      /*on_delete_callback_=*/nullptr);

  CopyLiteralSliceToLeafCpuBuffer(0, &tracked_buffer,
                                  LiteralSlice(literal, {0}));
  CopyLiteralSliceToLeafCpuBuffer(1, &tracked_buffer,
                                  LiteralSlice(literal, {1}));
  auto expected_literal = std::make_unique<Literal>(shape);
  CopyCpuBufferToLiteral(shape, &tracked_buffer, expected_literal.get());
  EXPECT_TRUE(LiteralTestUtil::Equal(literal, *expected_literal));
}

}  // namespace
}  // namespace xla
