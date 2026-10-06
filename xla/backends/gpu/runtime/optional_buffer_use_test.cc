/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/optional_buffer_use.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>

#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

class OptionalBufferUseTest : public ::testing::Test {
 protected:
  BufferAllocation allocation_{/*index=*/0, /*size=*/1024, /*color=*/0};
  ShapedSlice shaped_slice_{
      BufferAllocation::Slice(&allocation_, /*offset=*/16, /*size=*/64),
      ShapeUtil::MakeShape(F32, {16})};
};

TEST_F(OptionalBufferUseTest, ShapedSliceNulloptIsSkipped) {
  Thunk::BufferUses uses;
  AppendOptionalBufferUse(uses, &BufferUse::Read, std::optional<ShapedSlice>());
  EXPECT_THAT(uses, IsEmpty());
}

TEST_F(OptionalBufferUseTest, ShapedSliceValueIsAppendedWithAccess) {
  Thunk::BufferUses uses;
  AppendOptionalBufferUse(uses, &BufferUse::Read,
                          std::optional<ShapedSlice>(shaped_slice_));
  AppendOptionalBufferUse(uses, &BufferUse::Write,
                          std::optional<ShapedSlice>(shaped_slice_));
  AppendOptionalBufferUse(uses, &BufferUse::Scratch,
                          std::optional<const ShapedSlice>(shaped_slice_));

  const BufferAllocation::Slice& slice = shaped_slice_.slice;
  const Shape& shape = shaped_slice_.shape;
  EXPECT_THAT(uses, ElementsAre(BufferUse::Read(slice, shape),
                                BufferUse::Write(slice, shape),
                                BufferUse::Scratch(slice, shape)));
  EXPECT_EQ(uses[0].shape(), shape);
}

TEST_F(OptionalBufferUseTest, SliceAndShapeNulloptIsSkipped) {
  Thunk::BufferUses uses;
  AppendOptionalBufferUse(uses, &BufferUse::Write, std::nullopt, std::nullopt);
  EXPECT_THAT(uses, IsEmpty());
}

TEST_F(OptionalBufferUseTest, SliceAndShapeValueIsAppended) {
  Thunk::BufferUses uses;
  AppendOptionalBufferUse(uses, &BufferUse::Write, shaped_slice_.slice,
                          shaped_slice_.shape);
  EXPECT_THAT(uses, ElementsAre(BufferUse::Write(shaped_slice_.slice,
                                                 shaped_slice_.shape)));
  EXPECT_EQ(uses[0].shape(), shaped_slice_.shape);
}

TEST_F(OptionalBufferUseTest, SliceWithoutShapeDies) {
  Thunk::BufferUses uses;
  EXPECT_DEATH(AppendOptionalBufferUse(uses, &BufferUse::Read,
                                       shaped_slice_.slice, std::nullopt),
               "Missing shape");
}

}  // namespace
}  // namespace xla::gpu
