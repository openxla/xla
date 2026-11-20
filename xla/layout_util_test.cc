/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/layout_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/layout.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::ContainsRegex;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

class LayoutUtilTest : public ::testing::Test {
 protected:
  Shape MakeShapeWithLayout(PrimitiveType element_type,
                            absl::Span<const int64_t> dimensions,
                            absl::Span<const int64_t> minor_to_major,
                            absl::Span<const Tile> tiles = {}) {
    Shape shape = ShapeUtil::MakeShape(element_type, dimensions);
    *shape.mutable_layout() = LayoutUtil::MakeLayout(minor_to_major, tiles);
    return shape;
  }
};

TEST_F(LayoutUtilTest, TupleLayoutComparison) {
  Shape shape =
      ShapeUtil::MakeTupleShape({MakeShapeWithLayout(F32, {2, 3}, {0, 1})});
  Shape other_shape =
      ShapeUtil::MakeTupleShape({MakeShapeWithLayout(F32, {2, 2}, {0, 1})});

  Shape tuple0 = ShapeUtil::MakeTupleShape({});
  Shape tuple1 = ShapeUtil::MakeTupleShape({shape});
  Shape tuple2 = ShapeUtil::MakeTupleShape({shape, shape});

  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(tuple0, tuple0));
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(tuple0, tuple1));
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(tuple0, tuple2));
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(tuple1, tuple0));
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(tuple2, tuple0));

  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(tuple1, tuple1));
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(tuple1, tuple2));
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(tuple2, tuple1));

  Shape other_tuple2 = ShapeUtil::MakeTupleShape({shape, other_shape});
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(tuple2, tuple2));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(tuple2, other_tuple2));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(other_tuple2, tuple2));
}

TEST_F(LayoutUtilTest, CopyLayoutDenseArray) {
  Shape src = MakeShapeWithLayout(F32, {2, 3}, {0, 1});
  Shape dst = MakeShapeWithLayout(F32, {2, 3}, {1, 0});

  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));

  // Should work if destination has no layout.
  dst.clear_layout();
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));

  // If source is cleared, then destination should be cleared.
  src.clear_layout();
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_TRUE(dst.has_layout());
  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_FALSE(dst.has_layout());
}

TEST_F(LayoutUtilTest, CopyLayoutTuple) {
  Shape src = ShapeUtil::MakeTupleShape(
      {MakeShapeWithLayout(F32, {2, 3}, {0, 1}),
       MakeShapeWithLayout(F32, {42, 123}, {1, 0}),
       ShapeUtil::MakeTupleShape(
           {MakeShapeWithLayout(F32, {}, {}),
            MakeShapeWithLayout(F32, {1, 2, 3}, {0, 2, 1})})});
  Shape dst = ShapeUtil::MakeTupleShape(
      {MakeShapeWithLayout(F32, {2, 3}, {1, 0}),
       MakeShapeWithLayout(F32, {42, 123}, {1, 0}),
       ShapeUtil::MakeTupleShape(
           {MakeShapeWithLayout(F32, {}, {}),
            MakeShapeWithLayout(F32, {1, 2, 3}, {1, 2, 0})})});

  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
}

TEST_F(LayoutUtilTest, CopyLayoutNotCompatibleSameRank) {
  Shape src = MakeShapeWithLayout(F32, {123, 42, 7}, {2, 0, 1});
  Shape dst = MakeShapeWithLayout(F32, {2, 3, 5}, {1, 0});
  ASSERT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
}

TEST_F(LayoutUtilTest, CopyLayoutNotCompatibleDifferentRank) {
  Shape src = MakeShapeWithLayout(F32, {123, 42, 7}, {2, 0, 1});
  Shape dst = MakeShapeWithLayout(F32, {2, 3}, {1, 0});
  auto status = LayoutUtil::CopyLayoutBetweenShapes(src, &dst);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(), ContainsRegex("cannot copy layout from shape"));
}

TEST_F(LayoutUtilTest, CopyLayoutNotCompatibleTuple) {
  Shape src =
      ShapeUtil::MakeTupleShape({MakeShapeWithLayout(F32, {2, 3}, {0, 1}),
                                 MakeShapeWithLayout(F32, {42, 123}, {1, 0}),
                                 ShapeUtil::MakeTupleShape({MakeShapeWithLayout(
                                     F32, {1, 2, 3}, {0, 2, 1})})});
  Shape dst = ShapeUtil::MakeTupleShape(
      {MakeShapeWithLayout(F32, {2, 3}, {1, 0}),
       MakeShapeWithLayout(F32, {42, 123}, {1, 0}),
       ShapeUtil::MakeTupleShape(
           {MakeShapeWithLayout(F32, {}, {}),
            MakeShapeWithLayout(F32, {1, 2, 3}, {1, 2, 0})})});

  auto status = LayoutUtil::CopyLayoutBetweenShapes(src, &dst);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(), ContainsRegex("cannot copy layout from shape"));
}

TEST_F(LayoutUtilTest, CopyLayoutBogusLayout) {
  Shape src = ShapeUtil::MakeShape(F32, {2, 3});
  Shape dst = ShapeUtil::MakeShape(F32, {2, 3});
  // Set layout to invalid value.
  *src.mutable_layout() = LayoutUtil::MakeLayout({1, 2, 3, 4});

  auto status = LayoutUtil::CopyLayoutBetweenShapes(src, &dst);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              ContainsRegex("layout minor_to_major field contains .* "
                            "elements, but shape has"));
}

TEST_F(LayoutUtilTest, CopyTokenLayout) {
  Shape src = ShapeUtil::MakeTokenShape();
  Shape dst = ShapeUtil::MakeTokenShape();

  // Layouts are trivially the same for token types and copying layouts should
  // be a nop.
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
}

TEST_F(LayoutUtilTest, CopyOpaqueLayout) {
  Shape src = ShapeUtil::MakeOpaqueShape();
  Shape dst = ShapeUtil::MakeOpaqueShape();

  // Layouts are trivially the same for opaque types and copying layouts should
  // be a nop.
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
}

TEST_F(LayoutUtilTest, CopyTupleLayoutWithTokenAndOpaque) {
  Shape src = ShapeUtil::MakeTupleShape(
      {MakeShapeWithLayout(F32, {2, 3}, {0, 1}),
       MakeShapeWithLayout(F32, {42, 123}, {1, 0}), ShapeUtil::MakeTokenShape(),
       ShapeUtil::MakeTupleShape(
           {ShapeUtil::MakeOpaqueShape(), MakeShapeWithLayout(F32, {}, {}),
            MakeShapeWithLayout(F32, {1, 2, 3}, {0, 2, 1})})});
  Shape dst = ShapeUtil::MakeTupleShape(
      {MakeShapeWithLayout(F32, {2, 3}, {1, 0}),
       MakeShapeWithLayout(F32, {42, 123}, {1, 0}), ShapeUtil::MakeTokenShape(),
       ShapeUtil::MakeTupleShape(
           {ShapeUtil::MakeOpaqueShape(), MakeShapeWithLayout(F32, {}, {}),
            MakeShapeWithLayout(F32, {1, 2, 3}, {1, 2, 0})})});

  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
}

TEST_F(LayoutUtilTest, ClearLayoutTuple) {
  Shape shape = ShapeUtil::MakeTupleShape(
      {MakeShapeWithLayout(F32, {2, 3}, {1, 0}),
       MakeShapeWithLayout(F32, {42, 123}, {1, 0}),
       ShapeUtil::MakeTupleShape(
           {MakeShapeWithLayout(F32, {}, {}),
            MakeShapeWithLayout(F32, {1, 2, 3}, {1, 2, 0})})});
  EXPECT_TRUE(LayoutUtil::HasLayout(shape));
  EXPECT_TRUE(shape.tuple_shapes(0).has_layout());
  EXPECT_TRUE(shape.tuple_shapes(2).tuple_shapes(1).has_layout());

  LayoutUtil::ClearLayout(&shape);

  EXPECT_FALSE(LayoutUtil::HasLayout(shape));
  EXPECT_FALSE(shape.tuple_shapes(0).has_layout());
  EXPECT_FALSE(shape.tuple_shapes(2).tuple_shapes(1).has_layout());
}

TEST_F(LayoutUtilTest, ClearLayoutOpaqueAndToken) {
  // Opaque and token types trivially have layouts.
  for (Shape shape :
       {ShapeUtil::MakeOpaqueShape(), ShapeUtil::MakeTokenShape()}) {
    EXPECT_TRUE(LayoutUtil::HasLayout(shape));
    LayoutUtil::ClearLayout(&shape);
    EXPECT_TRUE(LayoutUtil::HasLayout(shape));
  }
}

TEST_F(LayoutUtilTest, SetToDefaultLayoutTuple) {
  Shape shape = ShapeUtil::MakeTupleShape(
      {MakeShapeWithLayout(F32, {2, 3, 4}, {1, 0, 2}),
       MakeShapeWithLayout(F32, {42, 123, 7}, {1, 2, 0}),
       ShapeUtil::MakeTupleShape(
           {MakeShapeWithLayout(F32, {}, {}),
            MakeShapeWithLayout(F32, {1, 2, 3, 4}, {3, 1, 2, 0})})});
  EXPECT_FALSE(LayoutUtil::Equal(shape.tuple_shapes(0).layout(),
                                 shape.tuple_shapes(1).layout()));
  LayoutUtil::SetToDefaultLayout(&shape);
  EXPECT_TRUE(LayoutUtil::Equal(shape.tuple_shapes(0).layout(),
                                shape.tuple_shapes(1).layout()));
  EXPECT_TRUE(LayoutUtil::Equal(
      LayoutUtil::GetDefaultLayoutForShape(shape.tuple_shapes(0)),
      shape.tuple_shapes(1).layout()));
}

TEST_F(LayoutUtilTest, DefaultLayoutGettersMajorToMinor) {
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({1, 0}),
                                LayoutUtil::GetDefaultLayoutForR2()));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({2, 1, 0}),
                                LayoutUtil::GetDefaultLayoutForR3()));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({3, 2, 1, 0}),
                                LayoutUtil::GetDefaultLayoutForR4()));
  EXPECT_TRUE(
      LayoutUtil::Equal(LayoutUtil::MakeLayout({4, 3, 2, 1, 0}),
                        LayoutUtil::GetDefaultLayoutForShape(
                            ShapeUtil::MakeShape(F32, {10, 20, 30, 15, 25}))));
}

TEST_F(LayoutUtilTest, MakeDescending) {
  EXPECT_TRUE(
      LayoutUtil::Equal(LayoutUtil::MakeDescendingLayout(8),
                        LayoutUtil::MakeLayout({7, 6, 5, 4, 3, 2, 1, 0})));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeDescendingLayout(5),
                                LayoutUtil::MakeLayout({4, 3, 2, 1, 0})));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeDescendingLayout(1),
                                LayoutUtil::MakeLayout({0})));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeDescendingLayout(0),
                                LayoutUtil::MakeLayout({})));
}

TEST_F(LayoutUtilTest, MakeAscending) {
  EXPECT_TRUE(
      LayoutUtil::Equal(LayoutUtil::MakeAscendingLayout(8),
                        LayoutUtil::MakeLayout({0, 1, 2, 3, 4, 5, 6, 7})));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeAscendingLayout(5),
                                LayoutUtil::MakeLayout({0, 1, 2, 3, 4})));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeAscendingLayout(1),
                                LayoutUtil::MakeLayout({0})));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeAscendingLayout(0),
                                LayoutUtil::MakeLayout({})));
}

TEST_F(LayoutUtilTest, HumanStringWithTiling) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3, 4}, {0, 1, 2});
  Tile* tile;

  // No tiling.
  EXPECT_EQ(ShapeUtil::HumanStringWithLayout(shape), "f32[2,3,4]{0,1,2}");

  // 2D tile.
  tile = shape.mutable_layout()->add_tiles();
  tile->add_dimensions(512);
  tile->add_dimensions(1024);
  EXPECT_EQ(ShapeUtil::HumanStringWithLayout(shape),
            "f32[2,3,4]{0,1,2:T(512,1024)}");

  // 1D tile.
  shape.mutable_layout()->clear_tiles();
  tile = shape.mutable_layout()->add_tiles();
  tile->add_dimensions(512);
  EXPECT_EQ(ShapeUtil::HumanStringWithLayout(shape),
            "f32[2,3,4]{0,1,2:T(512)}");

  // 2 tiles.
  shape = ShapeUtil::MakeShapeWithDenseLayout(BF16, {2, 3, 4}, {1, 2, 0});
  tile = shape.mutable_layout()->add_tiles();
  tile->add_dimensions(16);
  tile->add_dimensions(256);
  tile = shape.mutable_layout()->add_tiles();
  tile->add_dimensions(2);
  tile->add_dimensions(1);
  EXPECT_EQ(ShapeUtil::HumanStringWithLayout(shape),
            "bf16[2,3,4]{1,2,0:T(16,256)(2,1)}");

  // PRED with element size of 8 bits.
  shape = ShapeUtil::MakeShapeWithDenseLayout(PRED, {8, 8, 8}, {0, 2, 1});
  tile = shape.mutable_layout()->add_tiles();
  tile->add_dimensions(8);
  tile->add_dimensions(128);
  EXPECT_EQ(ShapeUtil::HumanStringWithLayout(shape),
            "pred[8,8,8]{0,2,1:T(8,128)}");

  // PRED with element size of 32 bits.
  shape.mutable_layout()->clear_tiles();
  tile = shape.mutable_layout()->add_tiles();
  tile->add_dimensions(8);
  tile->add_dimensions(128);
  shape.mutable_layout()->set_element_size_in_bits(32);
  EXPECT_EQ(ShapeUtil::HumanStringWithLayout(shape),
            "pred[8,8,8]{0,2,1:T(8,128)E(32)}");

  // No tile. PRED with element size of 32 bits.
  shape.mutable_layout()->clear_tiles();
  shape.mutable_layout()->set_element_size_in_bits(32);
  EXPECT_EQ(ShapeUtil::HumanStringWithLayout(shape),
            "pred[8,8,8]{0,2,1:E(32)}");

  // Tile with negative dimension size for combining dimensions.
  shape = ShapeUtil::MakeShapeWithDenseLayout(BF16, {2, 3, 1004}, {2, 1, 0});
  tile = shape.mutable_layout()->add_tiles();
  tile->add_dimensions(2);
  tile->add_dimensions(Tile::kCombineDimension);
  tile->add_dimensions(128);
  EXPECT_EQ(ShapeUtil::HumanStringWithLayout(shape),
            "bf16[2,3,1004]{2,1,0:T(2,*,128)}");

  // Tile with two negative dimensions.
  shape =
      ShapeUtil::MakeShapeWithDenseLayout(BF16, {8, 2, 3, 1004}, {3, 2, 1, 0});
  tile = shape.mutable_layout()->add_tiles();
  tile->add_dimensions(2);
  tile->add_dimensions(Tile::kCombineDimension);
  tile->add_dimensions(Tile::kCombineDimension);
  tile->add_dimensions(128);
  EXPECT_EQ(ShapeUtil::HumanStringWithLayout(shape),
            "bf16[8,2,3,1004]{3,2,1,0:T(2,*,*,128)}");
}

TEST_F(LayoutUtilTest, LinearIndexWithMatchDimension) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {16, 256}, {1, 0});
  Tile* tile = shape.mutable_layout()->add_tiles();
  tile->add_dimensions(Tile::kMatchDimension);
  tile->add_dimensions(128);

  // Tile (d, 128) on [16, 256] means each tile has size (16, 128).
  // (0, 0) -> 0
  EXPECT_EQ(LayoutUtil::LinearIndex(shape, {0, 0}), 0);
  EXPECT_EQ(LayoutUtil::LinearIndexForNestedTiling(shape, {0, 0}), 0);
  // (1, 0) -> row 1 of tile 0 -> 128
  EXPECT_EQ(LayoutUtil::LinearIndex(shape, {1, 0}), 128);
  EXPECT_EQ(LayoutUtil::LinearIndexForNestedTiling(shape, {1, 0}), 128);
  // (0, 128) -> row 0 of tile 1 -> 16 * 128 = 2048
  EXPECT_EQ(LayoutUtil::LinearIndex(shape, {0, 128}), 2048);
  EXPECT_EQ(LayoutUtil::LinearIndexForNestedTiling(shape, {0, 128}), 2048);
}

TEST_F(LayoutUtilTest, ValidateLayout_ValidArrayLayout) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3}, {0, 1});
  auto status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/false);
  EXPECT_TRUE(status.ok());
  status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/true);
  EXPECT_TRUE(status.ok());
}

TEST_F(LayoutUtilTest, ValidateLayout_InvalidArrayLayout) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  *shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1, 2});
  auto status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/false);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("layout minor_to_major field "
                        "contains 3 elements, but shape has 2 dimensions"));
  status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/true);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("layout minor_to_major field "
                        "contains 3 elements, but shape has 2 dimensions"));
}

TEST_F(LayoutUtilTest, ValidateLayout_MissingArrayLayout) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  LayoutUtil::ClearLayout(&shape);
  auto status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/false);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("shape f32[2,3] does not have a layout"));
  status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/true);
  EXPECT_TRUE(status.ok());
}

TEST_F(LayoutUtilTest, ValidateLayout_TupleSubshapesWithMissingLayouts) {
  Shape sub_1_1_1 = ShapeUtil::MakeShape(F32, {1, 2});
  Shape sub_1_1 = ShapeUtil::MakeTupleShape({sub_1_1_1});
  Shape sub_1_2 = ShapeUtil::MakeShape(F32, {1, 2});
  LayoutUtil::ClearLayout(&sub_1_2);
  Shape sub_1 = ShapeUtil::MakeTupleShape({sub_1_1, sub_1_2});
  Shape sub_2_1 = ShapeUtil::MakeShape(F32, {9});
  LayoutUtil::ClearLayout(&sub_2_1);
  Shape sub_2 = ShapeUtil::MakeTupleShape({sub_2_1});
  Shape shape = ShapeUtil::MakeTupleShape({sub_1, sub_2});

  auto status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/false);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("shape f32[1,2] does not have a layout"));
  status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/true);
  EXPECT_TRUE(status.ok());

  // Add invalid layout on one of sub-shapes.
  *shape.mutable_tuple_shapes(1)->mutable_tuple_shapes(0)->mutable_layout() =
      LayoutUtil::MakeLayout({0, 2, 3});

  status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/true);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("layout minor_to_major field "
                        "contains 3 elements, but shape has 1 dimensions"));
}

TEST_F(LayoutUtilTest, ValidateLayout_TileValidation) {
  // Valid oversized tile: Shape rank 1, but tile has 2 dimensions.
  // This is allowed because the shape is padded to 2D during tiling.
  {
    Shape shape = ShapeUtil::MakeShape(F32, {8});
    Layout layout;
    layout.add_minor_to_major(0);
    *layout.add_tiles() = Tile({2, 2});
    *shape.mutable_layout() = layout;
    auto status = LayoutUtil::ValidateLayoutInShape(
        shape, /*allow_missing_layouts=*/false);
    EXPECT_TRUE(status.ok());

    // Test round-trip indexing with padding
    EXPECT_EQ(LayoutUtil::LinearIndexForNestedTiling(shape, {5}), 9);
    EXPECT_THAT(LayoutUtil::DelinearizeIndexForNestedTiling(shape, 9),
                ElementsAre(5));
  }

  // Valid nested oversized tile: Shape rank 2, but nested tiles have total rank
  // exceeding the shape rank at the step.
  // Initially rank 2.
  // Tile 1 (rank 2) -> no padding, rank becomes 2 + 2 = 4.
  // Tile 2 (rank 5) -> pads rank from 4 to 5, new rank becomes 5 + 5 = 10.
  // This is allowed because the shape is padded to rank 5 at the second step.
  {
    Shape shape = ShapeUtil::MakeShape(F32, {8, 8});
    Layout layout;
    layout.add_minor_to_major(1);
    layout.add_minor_to_major(0);
    *layout.add_tiles() = Tile({2, 2});
    *layout.add_tiles() = Tile({2, 2, 2, 2, 2});
    *shape.mutable_layout() = layout;
    auto status = LayoutUtil::ValidateLayoutInShape(
        shape, /*allow_missing_layouts=*/false);
    EXPECT_TRUE(status.ok());
  }

  // Invalid: Tile with dimension <= 0 (excluding kCombineDimension)
  {
    Shape shape = ShapeUtil::MakeShape(F32, {8, 8});
    Layout layout;
    layout.add_minor_to_major(1);
    layout.add_minor_to_major(0);
    *layout.add_tiles() = Tile({2, -2});
    *shape.mutable_layout() = layout;
    auto status = LayoutUtil::ValidateLayoutInShape(
        shape, /*allow_missing_layouts=*/false);
    EXPECT_FALSE(status.ok());
    EXPECT_THAT(status.message(), HasSubstr("layout has invalid tiles"));
  }

  // Invalid: Tile with dimension == 0
  {
    Shape shape = ShapeUtil::MakeShape(F32, {8, 8});
    Layout layout;
    layout.add_minor_to_major(1);
    layout.add_minor_to_major(0);
    *layout.add_tiles() = Tile({2, 0});
    *shape.mutable_layout() = layout;
    auto status = LayoutUtil::ValidateLayoutInShape(
        shape, /*allow_missing_layouts=*/false);
    EXPECT_FALSE(status.ok());
    EXPECT_THAT(status.message(), HasSubstr("layout has invalid tiles"));
  }

  // Valid scalar tiling (allowed for backward compatibility / TPU vector
  // padding)
  {
    Shape shape = ShapeUtil::MakeShape(U32, {});
    Layout layout;
    *layout.add_tiles() = Tile({128});
    *shape.mutable_layout() = layout;
    auto status = LayoutUtil::ValidateLayoutInShape(
        shape, /*allow_missing_layouts=*/false);
    EXPECT_TRUE(status.ok());
  }

  // Valid 1D shape with 2D tile (allowed due to padding logic)
  {
    Shape shape = ShapeUtil::MakeShape(F32, {3});
    Layout layout;
    layout.add_minor_to_major(0);
    *layout.add_tiles() = Tile({8, 128});
    *shape.mutable_layout() = layout;
    auto status = LayoutUtil::ValidateLayoutInShape(
        shape, /*allow_missing_layouts=*/false);
    EXPECT_TRUE(status.ok());

    // Test round-trip indexing
    EXPECT_EQ(LayoutUtil::LinearIndexForNestedTiling(shape, {2}), 2);
    EXPECT_THAT(LayoutUtil::DelinearizeIndexForNestedTiling(shape, 2),
                ElementsAre(2));
  }
}

TEST_F(LayoutUtilTest, MoveDimToMajor) {
  const Layout layout = LayoutUtil::MakeLayout({2, 1, 0});
  Layout new_layout = LayoutUtil::MoveDimToMajor(layout, 0);
  EXPECT_EQ(new_layout, layout);

  new_layout = LayoutUtil::MoveDimToMajor(layout, 1);
  EXPECT_EQ(new_layout, LayoutUtil::MakeLayout({2, 0, 1}));
}

TEST_F(LayoutUtilTest, MoveDimToMinor) {
  const Layout layout = LayoutUtil::MakeLayout({2, 0, 3, 1});
  EXPECT_EQ(LayoutUtil::MoveDimToMinor(layout, 2), layout);
  EXPECT_EQ(LayoutUtil::MoveDimToMinor(layout, 3),
            LayoutUtil::MakeLayout({3, 2, 0, 1}));
}

TEST_F(LayoutUtilTest, StridesIsMajorToMinor) {
  std::vector<int64_t> byte_strides = {3960, 440, 44, 4};
  EXPECT_TRUE(LayoutUtil::ByteStridesIsMajorToMinor(
      byte_strides, {8, 9, 10, 11}, PrimitiveType::F32));
}

TEST_F(LayoutUtilTest, StridesNotMajorToMinorInnerMostStrideIncorrect) {
  std::vector<int64_t> byte_strides = {1880, 220, 22, 2};
  EXPECT_FALSE(LayoutUtil::ByteStridesIsMajorToMinor(
      byte_strides, {8, 9, 10, 11}, PrimitiveType::F32));
}

TEST_F(LayoutUtilTest, StridesNotMajorToMinor) {
  std::vector<int64_t> byte_strides = {1880, 440, 44, 4};
  EXPECT_FALSE(LayoutUtil::ByteStridesIsMajorToMinor(
      byte_strides, {8, 9, 10, 11}, PrimitiveType::F32));
}

TEST_F(LayoutUtilTest, HasCustomElementSizeInBits) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 2});
  EXPECT_FALSE(LayoutUtil::HasCustomElementSizeInBits(shape));

  shape = ShapeUtil::MakeShape(F32, {1, 2});
  shape.mutable_layout()->set_element_size_in_bits(0);
  EXPECT_FALSE(LayoutUtil::HasCustomElementSizeInBits(shape));

  shape = ShapeUtil::MakeShape(F32, {1, 2});
  shape.mutable_layout()->set_element_size_in_bits(32);
  EXPECT_TRUE(LayoutUtil::HasCustomElementSizeInBits(shape));

  shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {1, 2}),
                                  ShapeUtil::MakeShape(F32, {1, 2})}),
       ShapeUtil::MakeShape(F32, {1, 2})});
  EXPECT_FALSE(LayoutUtil::HasCustomElementSizeInBits(shape));

  shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {1, 2}),
                                  ShapeUtil::MakeShape(F32, {1, 2})}),
       ShapeUtil::MakeShape(F32, {1, 2})});
  ShapeUtil::GetMutableSubshape(&shape, {0, 1})
      ->mutable_layout()
      ->set_element_size_in_bits(32);
  EXPECT_TRUE(LayoutUtil::HasCustomElementSizeInBits(shape));
}

TEST_F(LayoutUtilTest, MaxSplitSize) {
  Shape shape = ShapeUtil::MakeShape(F32, {150, 200, 100});
  *shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1, 2})
                                .add_split_configs(SplitConfig(0, {30}))
                                .add_split_configs(SplitConfig(1, {40, 130}));

  EXPECT_EQ(LayoutUtil::MaxSplitSize(shape, 0), 150);
  EXPECT_EQ(LayoutUtil::MaxSplitSize(shape, 1), 90);
  EXPECT_EQ(LayoutUtil::MaxSplitSize(shape, 2), 70);
}

TEST_F(LayoutUtilTest, MaxElementsInPerSplit) {
  Shape shape = ShapeUtil::MakeShape(F32, {150, 200, 100});
  *shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1, 2});
  EXPECT_EQ(LayoutUtil::MaxElementsInPerSplit(shape), 150 * 200 * 100);

  *shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1, 2})
                                .add_split_configs(SplitConfig(0, {30}))
                                .add_split_configs(SplitConfig(1, {40, 130}));
  EXPECT_EQ(LayoutUtil::MaxElementsInPerSplit(shape), 150 * 90 * 70);
}

TEST_F(LayoutUtilTest, LinearIndexForNestedTilingRoundTrip1D) {
  // bf16[2048]{0:T(1024)(128)(2,1)}
  Shape shape_1D =
      MakeShapeWithLayout(BF16,
                          /*dimensions=*/{2048}, /*minor_to_major=*/{0},
                          /*tiles=*/{Tile({1024}), Tile({128}), Tile({2, 1})});
  EXPECT_EQ(LayoutUtil::LinearIndexForNestedTiling(shape_1D, {0}), 0);

  EXPECT_EQ(LayoutUtil::LinearIndexForNestedTiling(shape_1D, {128}), 1);
  EXPECT_THAT(LayoutUtil::DelinearizeIndexForNestedTiling(shape_1D, 1),
              ElementsAre(128));

  EXPECT_EQ(LayoutUtil::LinearIndexForNestedTiling(shape_1D, {1}), 2);
  EXPECT_THAT(LayoutUtil::DelinearizeIndexForNestedTiling(shape_1D, 2),
              ElementsAre(1));
}

TEST_F(LayoutUtilTest, LinearIndexForNestedTilingRoundTrip2D) {
  // bf16[2,8,128]{2,1,0:T(8,128)(2,1)}
  Shape shape_2D = MakeShapeWithLayout(
      BF16, /*dimensions=*/{2, 8, 128}, /*minor_to_major=*/{2, 1, 0},
      /*tiles=*/{Tile({8, 128}), Tile({2, 1})});

  EXPECT_EQ(LayoutUtil::LinearIndexForNestedTiling(shape_2D, {0, 0, 0}), 0);
  EXPECT_THAT(LayoutUtil::DelinearizeIndexForNestedTiling(shape_2D, 0),
              ElementsAre(0, 0, 0));

  EXPECT_EQ(LayoutUtil::LinearIndexForNestedTiling(shape_2D, {0, 2, 0}), 256);
  EXPECT_THAT(LayoutUtil::DelinearizeIndexForNestedTiling(shape_2D, 256),
              ElementsAre(0, 2, 0));
}

TEST_F(LayoutUtilTest, LinearIndexWithAndWithoutTiles) {
  // | 0 1 | 2 3 |
  // | 4 5 | 6 7 |
  // |-----|-----|
  // | 8 9 |10 11|
  // |12 13|14 15|
  // |-----|-----|
  // |16 17|18 19|
  // |20 21|22 23|
  // |-----|-----|
  // |24 25|26 27|
  // |28 29|30 31|
  //
  // Memory layout (untiled):
  // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 ...
  constexpr int linear_index_of_element_13_untiled = 13;
  // Memory layout (tiled):
  // 0 1 4 5 2 3 6 7 8 9 12 13 ...
  constexpr int linear_index_of_element_13_tiled = 11;
  Shape shape_2D =
      MakeShapeWithLayout(F32, /*dimensions=*/{8, 4}, /*minor_to_major=*/{1, 0},
                          /*tiles=*/{Tile({2, 2})});
  EXPECT_EQ(LayoutUtil::LinearIndexForNestedTiling(shape_2D, {3, 1}),
            linear_index_of_element_13_tiled);
  shape_2D.mutable_layout()->clear_tiles();
  EXPECT_EQ(LayoutUtil::LinearIndexForNestedTiling(shape_2D, {3, 1}),
            linear_index_of_element_13_untiled);
}

TEST_F(LayoutUtilTest, LinearIndexForNestedTiling_OversizedTilesPadding) {
  // Rank 0 (scalar) tiled with rank 1 tile: u32[]{:T(128)}
  {
    Shape shape = ShapeUtil::MakeShape(U32, {});
    Layout layout;
    *layout.add_tiles() = Tile({128});
    *shape.mutable_layout() = layout;

    EXPECT_EQ(LayoutUtil::LinearIndexForNestedTiling(shape, {}), 0);
    EXPECT_THAT(LayoutUtil::DelinearizeIndexForNestedTiling(shape, 0),
                ElementsAre());
  }

  // Rank 1 shape with rank 2 tile: f32[3]{:T(8, 128)}
  // Padded to [1, 3]. Tile T(8, 128) splits it:
  // - Outer: [Ceil(1, 8), Ceil(3, 128)] = [1, 1]
  // - Inner: [8, 128]
  // Tiled space size = 1 * 1 * 8 * 128 = 1024.
  // For logical index {2}, padded to {0, 2}.
  // Outer indices: {0 / 8, 2 / 128} = {0, 0}.
  // Inner indices: {0 % 8, 2 % 128} = {0, 2}.
  // Linear index: 0 * (1 * 8 * 128) + 0 * (8 * 128) + 0 * 128 + 2 = 2.
  {
    Shape shape = ShapeUtil::MakeShape(F32, {3});
    Layout layout;
    layout.add_minor_to_major(0);
    *layout.add_tiles() = Tile({8, 128});
    *shape.mutable_layout() = layout;

    EXPECT_EQ(LayoutUtil::LinearIndexForNestedTiling(shape, {2}), 2);
    EXPECT_THAT(LayoutUtil::DelinearizeIndexForNestedTiling(shape, 2),
                ElementsAre(2));
  }

  // Rank 1 shape with rank 2 tile: f32[8]{:T(2, 2)}
  // Padded to [1, 8]. Tile T(2, 2) splits it:
  // - Outer: [Ceil(1, 2), Ceil(8, 2)] = [1, 4]
  // - Inner: [2, 2]
  // Tiled space size = 1 * 4 * 2 * 2 = 16.
  // For logical index {5}, padded to {0, 5}.
  // Outer indices: {0 / 2, 5 / 2} = {0, 2}.
  // Inner indices: {0 % 2, 5 % 2} = {0, 1}.
  // Linear index in 16-element space:
  // 0 * (4 * 2 * 2) + 2 * (2 * 2) + 0 * 2 + 1 = 8 + 1 = 9.
  {
    Shape shape = ShapeUtil::MakeShape(F32, {8});
    Layout layout;
    layout.add_minor_to_major(0);
    *layout.add_tiles() = Tile({2, 2});
    *shape.mutable_layout() = layout;

    EXPECT_EQ(LayoutUtil::LinearIndexForNestedTiling(shape, {5}), 9);
    EXPECT_THAT(LayoutUtil::DelinearizeIndexForNestedTiling(shape, 9),
                ElementsAre(5));
  }
}

struct IsUntiledLayoutTestCase {
  std::vector<int64_t> shape;
  std::vector<Tile> tiles;
  bool expected_result;
};

using IsUntiledLayoutTest = ::testing::TestWithParam<IsUntiledLayoutTestCase>;

TEST_P(IsUntiledLayoutTest, IsUntiledLayout) {
  IsUntiledLayoutTestCase params = GetParam();
  EXPECT_EQ(LayoutUtil::IsUntiledLayout(params.tiles, params.shape),
            params.expected_result);
}

INSTANTIATE_TEST_SUITE_P(IsUntiledLayoutTests, IsUntiledLayoutTest,
                         ::testing::ValuesIn<IsUntiledLayoutTestCase>(
                             {{{24, 128}, {Tile({8, 128})}, true},
                              {{4, 256}, {Tile({1, 128})}, true},
                              {{2, 3, 4}, {Tile({8, 128})}, false}}));

TEST_F(LayoutUtilTest, ValidateBitPlanesFp6AndS64Layouts) {
  using ::absl_testing::IsOk;
  using ::absl_testing::StatusIs;

  const std::vector<BitPlane> fp6_2d_planes = {
      BitPlane(0, 4, {Tile({8, 1})}),
      BitPlane(4, 2, {Tile({16, 1})}),
  };
  const std::vector<BitPlane> fp6_1d_planes = {
      BitPlane(0, 4, {Tile({128}), Tile({8, 1})}),
      BitPlane(4, 2, {Tile({128}), Tile({16, 1})}),
  };

  // Valid FP6 2D compact, multi-strip, match-dim, planar, and 1D layouts.
  EXPECT_THAT(ShapeUtil::MakeValidatedShapeWithDenseLayout(
                  F6E3M2FN, {16, 128}, {1, 0}, {Tile({16, 128})}, 1, 0, 0, {},
                  fp6_2d_planes),
              IsOk());
  EXPECT_THAT(
      ShapeUtil::MakeValidatedShapeWithDenseLayout(
          F6E2M3FN, {256, 128}, {1, 0}, {Tile({256, 128}), Tile({16, 128})}, 1,
          0, 0, {}, fp6_2d_planes),
      IsOk());
  EXPECT_THAT(ShapeUtil::MakeValidatedShapeWithDenseLayout(
                  F6E3M2FN, {20, 128}, {1, 0},
                  {Tile({Tile::kMatchDimension, 128}), Tile({16, 128})}, 1, 0,
                  0, {}, fp6_2d_planes),
              IsOk());
  EXPECT_THAT(ShapeUtil::MakeValidatedShapeWithDenseLayout(
                  F6E3M2FN, {256, 128}, {1, 0}, {Tile({256, 128})}, 1, 0, 0, {},
                  fp6_2d_planes),
              IsOk());
  EXPECT_THAT(
      ShapeUtil::MakeValidatedShapeWithDenseLayout(
          F6E3M2FN, {130}, {0}, {Tile({2048})}, 1, 0, 0, {}, fp6_1d_planes),
      IsOk());
  EXPECT_THAT(ShapeUtil::MakeValidatedShapeWithDenseLayout(
                  F6E3M2FN, {4096}, {0}, {Tile({32768}), Tile({2048})}, 1, 0, 0,
                  {}, fp6_1d_planes),
              IsOk());
  // MSB-first offsets and padded odd/sub-byte widths (U4 in 4b+2b).
  EXPECT_THAT(
      ShapeUtil::MakeValidatedShapeWithDenseLayout(
          F6E3M2FN, {16, 128}, {1, 0}, {Tile({16, 128})}, 1, 0, 0, {},
          {BitPlane(4, 2, {Tile({16, 1})}), BitPlane(0, 4, {Tile({8, 1})})}),
      IsOk());
  EXPECT_THAT(
      ShapeUtil::MakeValidatedShapeWithDenseLayout(
          U4, {16, 128}, {1, 0}, {Tile({16, 128})}, 1, 0, 0, {}, fp6_2d_planes),
      IsOk());

  // Valid S64/U64 whole-tensor split (empty outer tiles) and per-tile splits.
  const std::vector<BitPlane> s64_2d_planes = {
      BitPlane(0, 32, {Tile({8, 128})}),
      BitPlane(32, 32, {Tile({8, 128})}),
  };
  EXPECT_THAT(ShapeUtil::MakeValidatedShapeWithDenseLayout(
                  S64, {10, 130}, {1, 0}, {}, 1, 0, 0, {}, s64_2d_planes),
              IsOk());
  EXPECT_THAT(ShapeUtil::MakeValidatedShapeWithDenseLayout(
                  U64, {10, 130}, {1, 0}, {}, 1, 0, 0, {},
                  {BitPlane(0, 32, {Tile({Tile::kMatchDimension, 128})}),
                   BitPlane(32, 32, {Tile({Tile::kMatchDimension, 128})})}),
              IsOk());
  EXPECT_THAT(
      ShapeUtil::MakeValidatedShapeWithDenseLayout(
          S64, {100}, {0}, {}, 1, 0, 0, {},
          {BitPlane(0, 32, {Tile({1024})}), BitPlane(32, 32, {Tile({1024})})}),
      IsOk());
  EXPECT_THAT(
      ShapeUtil::MakeValidatedShapeWithDenseLayout(
          S64, {}, {}, {}, 1, 0, /*memory_space=*/1, {},
          {BitPlane(0, 32, {Tile({128})}), BitPlane(32, 32, {Tile({128})})}),
      IsOk());
  EXPECT_THAT(ShapeUtil::MakeValidatedShapeWithDenseLayout(
                  C64, {10, 130}, {1, 0}, {}, 1, 0, 0, {}, s64_2d_planes),
              IsOk());
  EXPECT_THAT(ShapeUtil::MakeValidatedShapeWithDenseLayout(
                  F6E3M2FN, {2, 8, 128}, {2, 1, 0},
                  {Tile({Tile::kCombineDimension, 16, 128})}, 1, 0, 0, {},
                  fp6_2d_planes),
              IsOk());
  EXPECT_THAT(
      ShapeUtil::MakeValidatedShapeWithDenseLayout(
          S64, {2, 5, 130}, {2, 1, 0}, {}, 1, 0, 0, {},
          {BitPlane(0, 32, {Tile({Tile::kCombineDimension, 8, 128})}),
           BitPlane(32, 32, {Tile({Tile::kCombineDimension, 8, 128})})}),
      IsOk());
  EXPECT_THAT(
      ShapeUtil::MakeValidatedShapeWithDenseLayout(
          S64, {100}, {0}, {},
          /*tail_padding_alignment_in_elements=*/128, 0, 0, {},
          {BitPlane(0, 32, {Tile({1})}), BitPlane(32, 32, {Tile({1})})}),
      IsOk());
  EXPECT_THAT(ShapeUtil::MakeValidatedShapeWithDenseLayout(
                  S64, {16, 128}, {1, 0}, {Tile({16, 128}), Tile({1, 128})}, 1,
                  0, 0, {}, {BitPlane(0, 32), BitPlane(32, 32)}),
              IsOk());

  // 0D scalar in S(1) must still validate bit_planes, and non-empty bit_planes
  // on a 2D shape in S(1) skips the AUTO early return so minor_to_major size
  // validation still rejects an empty minor_to_major.
  EXPECT_THAT(ShapeUtil::MakeValidatedShapeWithDenseLayout(
                  S64, {}, {}, {}, 1, 0, /*memory_space=*/1, {},
                  {BitPlane(0, 64, {Tile({128})})}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("at least 2 planes")));
  Shape auto_with_bp = ShapeUtil::MakeShape(F6E3M2FN, {16, 128});
  auto_with_bp.mutable_layout()->clear_minor_to_major();
  auto_with_bp.mutable_layout()->set_memory_space(1);
  auto_with_bp.mutable_layout()->add_bit_planes(BitPlane(0, 4, {Tile({8, 1})}));
  auto_with_bp.mutable_layout()->add_bit_planes(
      BitPlane(4, 2, {Tile({16, 1})}));
  EXPECT_THAT(
      LayoutUtil::ValidateLayoutInShape(auto_with_bp),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("layout minor_to_major field contains 0 elements")));

  // Invalid cases: E(n) + BP, single plane, overlap, gap below/above elem_bits,
  // incomplete coverage, non-dividing outer subtile, 'd' in tiles.back(),
  // non-dividing plane tile, '*'/'d' in plane.tiles with outer tiles, and
  // non-strip-aligned L(n).
  EXPECT_THAT(
      ShapeUtil::MakeValidatedShapeWithDenseLayout(
          F6E3M2FN, {16, 128}, {1, 0}, {Tile({16, 128})}, 1,
          /*element_size_in_bits=*/6, 0, {}, fp6_2d_planes),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "cannot specify both element_size_in_bits (6) and bit_planes")));
  EXPECT_THAT(ShapeUtil::MakeValidatedShapeWithDenseLayout(
                  U4, {16, 128}, {1, 0}, {Tile({16, 128})}, 1, 0, 0, {},
                  {BitPlane(0, 4, {Tile({8, 1})})}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("at least 2 planes")));
  EXPECT_THAT(
      ShapeUtil::MakeValidatedShapeWithDenseLayout(
          F6E3M2FN, {16, 128}, {1, 0}, {Tile({16, 128})}, 1, 0, 0, {},
          {BitPlane(0, 4, {Tile({8, 1})}), BitPlane(3, 3, {Tile({16, 1})})}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("overlapping bit ranges [0, 4) and [3, 6)")));
  EXPECT_THAT(
      ShapeUtil::MakeValidatedShapeWithDenseLayout(
          F6E3M2FN, {16, 128}, {1, 0}, {Tile({16, 128})}, 1, 0, 0, {},
          {BitPlane(0, 2, {Tile({16, 1})}), BitPlane(4, 2, {Tile({16, 1})})}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("contiguous from bit 0 (gap between [0, 2) and "
                         "[4, 6))")));
  EXPECT_THAT(
      ShapeUtil::MakeValidatedShapeWithDenseLayout(
          F6E3M2FN, {16, 128}, {1, 0}, {Tile({16, 128})}, 1, 0, 0, {},
          {BitPlane(0, 6, {Tile({16, 1})}), BitPlane(100, 2, {Tile({16, 1})})}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("contiguous from bit 0 (gap between [0, 6) and "
                         "[100, 102))")));
  EXPECT_THAT(
      ShapeUtil::MakeValidatedShapeWithDenseLayout(
          F6E3M2FN, {16, 128}, {1, 0}, {Tile({16, 128})}, 1, 0, 0, {},
          {BitPlane(0, 2, {Tile({16, 1})}), BitPlane(2, 2, {Tile({16, 1})})}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("must cover all 6 bits of element type "
                         "(covered up to 4)")));
  EXPECT_THAT(
      ShapeUtil::MakeValidatedShapeWithDenseLayout(
          F6E3M2FN, {256, 128}, {1, 0}, {Tile({20, 128}), Tile({16, 128})}, 1,
          0, 0, {}, fp6_2d_planes),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("subtile dimension 16 does not evenly divide "
                         "parent dimension 20")));
  EXPECT_THAT(
      ShapeUtil::MakeValidatedShapeWithDenseLayout(
          F6E3M2FN, {20, 128}, {1, 0}, {Tile({Tile::kMatchDimension, 128})}, 1,
          0, 0, {}, fp6_2d_planes),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("kMatchDimension ('d') is not allowed in the "
                         "innermost outer tile")));
  EXPECT_THAT(ShapeUtil::MakeValidatedShapeWithDenseLayout(
                  F6E3M2FN, {8, 128}, {1, 0}, {Tile({8, 128})}, 1, 0, 0, {},
                  fp6_2d_planes),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("subtile dimension 16 does not evenly divide "
                                 "parent dimension 8")));
  EXPECT_THAT(ShapeUtil::MakeValidatedShapeWithDenseLayout(
                  F6E3M2FN, {16, 128}, {1, 0}, {Tile({16, 128})}, 1, 0, 0, {},
                  {BitPlane(0, 4, {Tile({Tile::kMatchDimension, 1})}),
                   BitPlane(4, 2, {Tile({16, 1})})}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be strictly positive (no '*' or 'd')")));
  EXPECT_THAT(
      ShapeUtil::MakeValidatedShapeWithDenseLayout(
          F6E3M2FN, {16, 128}, {1, 0}, {Tile({16, 128})},
          /*tail_padding_alignment_in_elements=*/100, 0, 0, {}, fp6_2d_planes),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("tail_padding_alignment_in_elements (100) "
                         "must be a multiple of the innermost tile "
                         "element count (2048)")));
}

TEST_F(LayoutUtilTest, ClearTilesAndBitOffsetForBitPlane) {
  Shape fp6_1d = ShapeUtil::MakeShapeWithDenseLayout(
      F6E3M2FN, {130}, {0}, {Tile({2048})}, 1, 0, 0, {},
      {BitPlane(0, 4, {Tile({128}), Tile({8, 1})}),
       BitPlane(4, 2, {Tile({128}), Tile({16, 1})})});
  // Index 129 wraps to (row=1, lane=1) -> plane 1 bit 8192 + (1*16 + 1)*2 =
  // 8226 (byte 1028), while index 127 sits at (row=0, lane=127) -> plane 1 bit
  // 8192 + (127*16 + 0)*2 = 12256 (byte 1532).
  EXPECT_EQ(LayoutUtil::BitOffsetForBitPlane(fp6_1d, {129}, 1), 8226);
  EXPECT_EQ(LayoutUtil::BitOffsetForBitPlane(fp6_1d, {127}, 1), 12256);

  Shape s64_2d = ShapeUtil::MakeShapeWithDenseLayout(
      S64, {10, 130}, {1, 0}, {}, 1, 0, 0, {},
      {BitPlane(0, 32, {Tile({8, 128})}), BitPlane(32, 32, {Tile({8, 128})})});
  // Plane 0 pads [10, 130] to [16, 256] = 4096 elements = 16384 bytes.
  EXPECT_EQ(LayoutUtil::BitOffsetForBitPlane(s64_2d, {0, 0}, 0), 0);
  EXPECT_EQ(LayoutUtil::BitOffsetForBitPlane(s64_2d, {0, 0}, 1), 16384 * 8);

  LayoutUtil::ClearTiles(&fp6_1d);
  EXPECT_TRUE(fp6_1d.layout().tiles().empty());
  EXPECT_TRUE(fp6_1d.layout().bit_planes().empty());
}

TEST_F(LayoutUtilTest, PhysicalDimensionsAndResolveTileStep) {
  using absl_testing::IsOk;
  using absl_testing::StatusIs;

  Shape split_shape = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {10, 20}, {0, 1}, {}, /*tail_padding_alignment_in_elements=*/1,
      /*element_size_in_bits=*/0, /*memory_space=*/0, {SplitConfig(0, {6})});
  // minor_to_major={0, 1} puts logical dim 1 (major, split at 6 -> max 14)
  // before logical dim 0 (minor, size 10).
  EXPECT_THAT(LayoutUtil::PhysicalDimensions(split_shape), ElementsAre(14, 10));

  Shape unbounded_shape =
      ShapeUtil::MakeShape(F32, {Shape::kUnboundedSize, 128});
  *unbounded_shape.mutable_layout() = LayoutUtil::MakeLayout({1, 0});
  EXPECT_THAT(LayoutUtil::PhysicalDimensions(unbounded_shape),
              ElementsAre(Shape::kUnboundedSize, 128));

  // Pad 1D shape [20] to 2D and resolve kMatchDimension against (16, 128).
  const std::vector<int64_t> sub_tile_dims = {16, 128};
  absl::StatusOr<LayoutUtil::ResolvedTileStep> step =
      LayoutUtil::ResolveTileStep({20}, Tile({Tile::kMatchDimension, 128}),
                                  sub_tile_dims);
  ASSERT_THAT(step, IsOk());
  EXPECT_EQ(step->leading_pad_dims, 1);
  EXPECT_EQ(step->suffix_start, 0);
  EXPECT_THAT(step->padded_suffix, ElementsAre(1, 20));
  EXPECT_THAT(step->effective_tile_dims, ElementsAre(16, 128));
  EXPECT_THAT(step->next_shape, ElementsAre(1, 1, 16, 128));

  // Combine ('*') + zero-sized dimension handling.
  absl::StatusOr<LayoutUtil::ResolvedTileStep> zero_step =
      LayoutUtil::ResolveTileStep(
          {2, 0, 128},
          Tile({Tile::kCombineDimension, Tile::kMatchDimension, 128}),
          sub_tile_dims);
  ASSERT_THAT(zero_step, IsOk());
  EXPECT_THAT(zero_step->effective_tile_dims, ElementsAre(16, 128));
  EXPECT_THAT(zero_step->next_shape, ElementsAre(0, 1, 16, 128));

  // Unbounded dimension combining, kMatchDimension, and next_shape propagation.
  absl::StatusOr<LayoutUtil::ResolvedTileStep> unbounded_step =
      LayoutUtil::ResolveTileStep(
          {2, Shape::kUnboundedSize, 128},
          Tile({Tile::kCombineDimension, Tile::kMatchDimension, 128}),
          sub_tile_dims);
  ASSERT_THAT(unbounded_step, IsOk());
  EXPECT_THAT(unbounded_step->effective_tile_dims, ElementsAre(16, 128));
  EXPECT_THAT(unbounded_step->next_shape,
              ElementsAre(Shape::kUnboundedSize, 1, 16, 128));

  // Error branches: empty tile, trailing '*', non-positive dim, and overflow.
  using ::testing::HasSubstr;
  EXPECT_THAT(LayoutUtil::ResolveTileStep({16, 128}, Tile(), {}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("layout has empty tile dimensions")));
  EXPECT_THAT(LayoutUtil::ResolveTileStep(
                  {16, 128}, Tile({16, Tile::kCombineDimension}), {}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("tile cannot end with kCombineDimension")));
  EXPECT_THAT(LayoutUtil::ResolveTileStep({16, 128}, Tile({0, 128}), {}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be positive")));
  EXPECT_THAT(
      LayoutUtil::ResolveTileStep({std::numeric_limits<int64_t>::max(), 2},
                                  Tile({Tile::kCombineDimension, 16}), {}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("combined tile dimension overflows")));
  EXPECT_THAT(LayoutUtil::ResolveTileStep(
                  {std::numeric_limits<int64_t>::max(), 128},
                  Tile({Tile::kMatchDimension, 128}), sub_tile_dims),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("kMatchDimension tile size overflows")));
}

TEST_F(LayoutUtilTest, ValidateBitPlanesRejectsOverflow) {
  using absl_testing::StatusIs;
  using ::testing::HasSubstr;

  // Combining SplitConfig with bit_planes is rejected.
  Shape split_shape = ShapeUtil::MakeShape(S64, {20, 128});
  Layout split_bp_layout(
      {1, 0}, /*tiles=*/{}, PRIMITIVE_TYPE_INVALID, PRIMITIVE_TYPE_INVALID,
      /*tail_padding_alignment_in_elements=*/1, /*element_size_in_bits=*/0,
      /*memory_space=*/0, {SplitConfig(0, {6})}, /*physical_shape=*/nullptr,
      /*dynamic_shape_metadata_prefix_bytes=*/0,
      {BitPlane(0, 32, {Tile({8, 128})}), BitPlane(32, 32, {Tile({8, 128})})});
  EXPECT_THAT(
      LayoutUtil::ValidateLayoutForShape(split_bp_layout, split_shape),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("cannot specify both split_configs and bit_planes")));

  // Total bit_width across disjoint planes overflows int64_t.
  Shape scalar_shape = ShapeUtil::MakeShape(F6E3M2FN, {16, 128});
  Layout overflow_width_layout(
      {1, 0}, {Tile({16, 128})}, PRIMITIVE_TYPE_INVALID, PRIMITIVE_TYPE_INVALID,
      /*tail_padding_alignment_in_elements=*/1, /*element_size_in_bits=*/0,
      /*memory_space=*/0, /*split_configs=*/{}, /*physical_shape=*/nullptr,
      /*dynamic_shape_metadata_prefix_bytes=*/0,
      {BitPlane(0, 6, {Tile({8, 1})}),
       BitPlane(6, std::numeric_limits<int64_t>::max() - 6, {Tile({16, 1})})});
  EXPECT_THAT(
      LayoutUtil::ValidateLayoutForShape(overflow_width_layout, scalar_shape),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("overflows")));

  // Static tensor with outer tiles whose total bit count overflows int64_t.
  Shape large_shape = ShapeUtil::MakeShape(F6E3M2FN, {1LL << 31, 1LL << 30});
  Layout outer_tiled_bp(
      {1, 0}, {Tile({16, 128})}, PRIMITIVE_TYPE_INVALID, PRIMITIVE_TYPE_INVALID,
      /*tail_padding_alignment_in_elements=*/1, /*element_size_in_bits=*/0,
      /*memory_space=*/0, /*split_configs=*/{}, /*physical_shape=*/nullptr,
      /*dynamic_shape_metadata_prefix_bytes=*/0,
      {BitPlane(0, 4, {Tile({8, 1})}), BitPlane(4, 2, {Tile({16, 1})})});
  EXPECT_THAT(LayoutUtil::ValidateLayoutForShape(outer_tiled_bp, large_shape),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("bit count overflows")));

  // Static tensor with empty outer tiles whose per-plane bit sum overflows.
  Shape large_s64 = ShapeUtil::MakeShape(S64, {1LL << 29, 1LL << 29});
  Layout whole_tensor_bp(
      {1, 0}, {}, PRIMITIVE_TYPE_INVALID, PRIMITIVE_TYPE_INVALID,
      /*tail_padding_alignment_in_elements=*/1, /*element_size_in_bits=*/0,
      /*memory_space=*/0, /*split_configs=*/{}, /*physical_shape=*/nullptr,
      /*dynamic_shape_metadata_prefix_bytes=*/0,
      {BitPlane(0, 32, {Tile({8, 128})}), BitPlane(32, 32, {Tile({8, 128})})});
  EXPECT_THAT(LayoutUtil::ValidateLayoutForShape(whole_tensor_bp, large_s64),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("bit count overflows")));
}

void BM_MakeDescendingLayout(::testing::benchmark::State& state) {
  const int64_t num_dims = state.range(0);
  for (auto s : state) {
    tsl::testing::DoNotOptimize(LayoutUtil::MakeDescendingLayout(num_dims));
  }
}

// NOLINTNEXTLINE(misc-include-cleaner)
BENCHMARK(BM_MakeDescendingLayout)->Arg(4)->Arg(8);

void BM_MakeAscendingLayout(::testing::benchmark::State& state) {
  const int64_t num_dims = state.range(0);
  for (auto s : state) {
    tsl::testing::DoNotOptimize(LayoutUtil::MakeAscendingLayout(num_dims));
  }
}

// NOLINTNEXTLINE(misc-include-cleaner)
BENCHMARK(BM_MakeAscendingLayout)->Arg(4)->Arg(8);

}  // namespace
}  // namespace xla
