/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/layout.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

#include "absl/hash/hash_testing.h"
#include "xla/hlo/testlib/test.h"
#include "xla/layout_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

TEST(Layout, ToStringForEmpty) { EXPECT_EQ(Layout().ToString(), "{}"); }

TEST(Layout, ToStringForMinorToMajorOnly) {
  EXPECT_EQ(Layout({1, 2, 0}).ToString(), "{1,2,0}");
}

TEST(Layout, ToStringForTiles) {
  EXPECT_EQ(Layout({3, 2, 1, 0}, {Tile({42, 123}), Tile({4, 5})}).ToString(),
            "{3,2,1,0:T(42,123)(4,5)}");
}

TEST(Layout, ToStringForTileWithCombinedDimensions) {
  EXPECT_EQ(Layout({3, 2, 1, 0}, {Tile({Tile::kCombineDimension,
                                        Tile::kCombineDimension, 42, 123})})
                .ToString(),
            "{3,2,1,0:T(*,*,42,123)}");
}

TEST(Layout, ToStringForTileWithMatchedDimensions) {
  EXPECT_EQ(
      Layout({3, 2, 1, 0}, {Tile({Tile::kMatchDimension, 128})}).ToString(),
      "{3,2,1,0:T(d,128)}");
}

TEST(Layout, ToStringForTailPaddingAlignment) {
  EXPECT_EQ(Layout({3, 2, 1, 0})
                .set_tail_padding_alignment_in_elements(100)
                .ToString(),
            "{3,2,1,0:L(100)}");
}

TEST(Layout, ToStringForIndexPrimitiveType) {
  EXPECT_EQ(Layout({3, 2, 1, 0})
                .set_index_primitive_type(PrimitiveType::U32)
                .ToString(),
            "{3,2,1,0:#(u32)}");
}

TEST(Layout, ToStringForPointerPrimitiveType) {
  EXPECT_EQ(Layout({3, 2, 1, 0})
                .set_pointer_primitive_type(PrimitiveType::U16)
                .ToString(),
            "{3,2,1,0:*(u16)}");
}

TEST(Layout, ToStringForElementSize) {
  EXPECT_EQ(Layout({3, 2, 1, 0}).set_element_size_in_bits(42).ToString(),
            "{3,2,1,0:E(42)}");
}

TEST(Layout, ToStringForMemorySpace) {
  EXPECT_EQ(Layout({3, 2, 1, 0}).set_memory_space(3).ToString(),
            "{3,2,1,0:S(3)}");
}

TEST(Layout, ToStringForSplitConfigs) {
  EXPECT_EQ(Layout({0, 1})
                .add_split_configs(SplitConfig(0, {3}))
                .add_split_configs(SplitConfig(1, {0, 4}))
                .ToString(),
            "{0,1:SC(0:3)(1:0,4)}");
}

TEST(Layout, ToStringForPhysicalShape) {
  Layout layout({0, 1});
  *layout.mutable_physical_shape() = ShapeUtil::MakeShape(S32, {10, 20});
  EXPECT_EQ(layout.ToString(), "{0,1:P(s32[10,20]{1,0})}");
}

TEST(Layout, ToStringForDynamicShapeMetadataPrefixBytes) {
  EXPECT_EQ(
      Layout({0, 1}).set_dynamic_shape_metadata_prefix_bytes(123).ToString(),
      "{0,1:M(123)}");
}

TEST(Layout, ToStringForMutipleProperties) {
  EXPECT_EQ(Layout({3, 2, 1, 0}, {Tile({42, 123}), Tile({4, 5})})
                .set_tail_padding_alignment_in_elements(100)
                .set_element_size_in_bits(42)
                .ToString(),
            "{3,2,1,0:T(42,123)(4,5)L(100)E(42)}");
}

TEST(Layout, StreamOut) {
  {
    std::ostringstream oss;
    oss << Tile({7, 8});
    EXPECT_EQ(oss.str(), "(7,8)");
  }

  {
    std::ostringstream oss;
    oss << Layout({0, 1, 2});
    EXPECT_EQ(oss.str(), "{0,1,2}");
  }
}

TEST(Layout, Equality) {
  EXPECT_EQ(Layout(), Layout());
  const std::vector<int64_t> empty_dims;
  EXPECT_EQ(Layout(empty_dims), Layout(empty_dims));
  EXPECT_EQ(Layout(), Layout(empty_dims));
  EXPECT_EQ(Layout({0, 1, 2, 3}), Layout({0, 1, 2, 3}));
  EXPECT_NE(Layout({0, 1, 2, 3}), Layout({0, 1, 2}));
  EXPECT_EQ(Layout({0, 1, 2}, {Tile({42, 44})}),
            Layout({0, 1, 2}, {Tile({42, 44})}));
  EXPECT_NE(Layout({0, 1, 2}, {Tile({42, 44})}),
            Layout({0, 1, 2}, {Tile({42, 45})}));
  EXPECT_NE(Layout({0, 1, 2}, {Tile({42, 44})}), Layout({0, 1, 2, 3}));
  EXPECT_EQ(Layout({0, 1, 2}).set_element_size_in_bits(33),
            Layout({0, 1, 2}).set_element_size_in_bits(33));
  EXPECT_NE(Layout({0, 1, 2}).set_element_size_in_bits(33),
            Layout({0, 1, 2}).set_element_size_in_bits(7));
  EXPECT_EQ(Layout({0, 1, 2}).set_memory_space(3),
            Layout({0, 1, 2}).set_memory_space(3));
  EXPECT_NE(Layout({0, 1, 2}).set_memory_space(1),
            Layout({0, 1, 2}).set_memory_space(3));
  EXPECT_FALSE(
      Layout::Equal()(Layout({0, 1, 2}, {Tile({42, 44})}), Layout({0, 1, 2})));
  EXPECT_EQ(Layout({0, 1, 2}).add_split_configs(SplitConfig(0, {2})),
            Layout({0, 1, 2}).add_split_configs(SplitConfig(0, {2})));
  EXPECT_NE(Layout({0, 1, 2}).add_split_configs(SplitConfig(0, {2})),
            Layout({0, 1, 2}).add_split_configs(SplitConfig(0, {3})));
  EXPECT_TRUE(Layout::Equal().IgnoreTiles()(Layout({0, 1, 2}, {Tile({42, 44})}),
                                            Layout({0, 1, 2})));
  EXPECT_FALSE(Layout::Equal()(Layout({0, 1, 2}, {}, PRIMITIVE_TYPE_INVALID,
                                      PRIMITIVE_TYPE_INVALID, 1, 32),
                               Layout({0, 1, 2}, {}, PRIMITIVE_TYPE_INVALID,
                                      PRIMITIVE_TYPE_INVALID, 1, 1)));
  EXPECT_TRUE(Layout::Equal().IgnoreElementSize()(
      Layout({0, 1, 2}).set_element_size_in_bits(32),
      Layout({0, 1, 2}).set_element_size_in_bits(1)));
  EXPECT_TRUE(Layout::Equal().IgnoreMemorySpace()(
      Layout({0, 1, 2}).set_memory_space(1),
      Layout({0, 1, 2}).set_memory_space(3)));
  EXPECT_TRUE(Layout::Equal().IgnoreSplitConfigs()(
      Layout({0, 1, 2}).add_split_configs(SplitConfig(0, {2})),
      Layout({0, 1, 2}).add_split_configs(SplitConfig(0, {3}))));
  EXPECT_FALSE(
      Layout::Equal()(Layout({0, 1, 2}).add_split_configs(SplitConfig(0, {2})),
                      Layout({0, 1, 2})));
  EXPECT_TRUE(Layout::Equal().MinorToMajorOnly()(
      Layout({0, 1, 2}).add_split_configs(SplitConfig(0, {2})),
      Layout({0, 1, 2})));
}

TEST(Layout, LayoutToFromProto) {
  // Round-trips a Layout through proto de/serialization.
  auto expect_unchanged = [](const Layout& layout) {
    const auto layout_proto = layout.ToProto();
    const auto from_proto_result = Layout::FromProto(layout_proto);
    TF_ASSERT_OK(from_proto_result);
    EXPECT_EQ(layout, from_proto_result.value());
  };

  expect_unchanged(Layout());
  expect_unchanged(Layout({1, 3, 2, 0}));
  expect_unchanged(Layout({0, 1}).set_element_size_in_bits(42));
  expect_unchanged(Layout({3, 2, 1, 0}, {Tile({42, 123}), Tile({4, 5})}));
  expect_unchanged(Layout({1, 0}, {}));
  expect_unchanged(Layout(
      {1, 0}, {}, PRIMITIVE_TYPE_INVALID, PRIMITIVE_TYPE_INVALID, 1, 0, 0, {},
      std::make_unique<Shape>(ShapeUtil::MakeShape(S32, {10, 10}))));
  expect_unchanged(Layout({0, 1}, {Tile({123})})
                       .add_split_configs(SplitConfig(0, {3}))
                       .add_split_configs(SplitConfig(1, {0, 4})));
}

TEST(Layout, LayoutToFromProtoInplace) {
  // Round-trips a Layout through proto de/serialization. Start from a
  // non-empty proto to verify that `ToProto` clears the given proto first.
  auto expect_unchanged = [](const Layout& layout) {
    LayoutProto layout_proto;
    layout_proto.add_tiles();
    layout.ToProto(layout_proto);
    const auto from_proto_result = Layout::FromProto(layout_proto);
    TF_ASSERT_OK(from_proto_result);
    EXPECT_EQ(layout, *from_proto_result) << layout_proto.DebugString();
  };

  expect_unchanged(Layout());
  expect_unchanged(Layout({1, 3, 2, 0}));
  expect_unchanged(Layout({0, 1}).set_element_size_in_bits(42));
  expect_unchanged(Layout({3, 2, 1, 0}, {Tile({42, 123}), Tile({4, 5})}));
  expect_unchanged(Layout({1, 0}, {}));
  expect_unchanged(Layout(
      {1, 0}, {}, PRIMITIVE_TYPE_INVALID, PRIMITIVE_TYPE_INVALID, 1, 0, 0, {},
      std::make_unique<Shape>(ShapeUtil::MakeShape(S32, {10, 10}))));
  expect_unchanged(Layout({0, 1}, {Tile({123})})
                       .add_split_configs(SplitConfig(0, {3}))
                       .add_split_configs(SplitConfig(1, {0, 4})));
}

TEST(Layout, DeleteDimensionWorksForDeletingLastDimFromDenseLayout) {
  Layout layout({0, 1});
  ASSERT_EQ(layout.minor_to_major().size(), 2);

  layout.DeleteDimension(1);
  EXPECT_THAT(layout.minor_to_major(), ElementsAre(0));
}

TEST(Layout, DeleteDimensionWorksForDeletingNonLastDimFromDenseLayout) {
  Layout layout({1, 0});
  ASSERT_EQ(layout.minor_to_major().size(), 2);

  layout.DeleteDimension(0);
  EXPECT_THAT(layout.minor_to_major(), ElementsAre(0));
}

TEST(Layout, DeleteDimensionUpdatesSplitConfigs) {
  Layout layout({2, 0, 1});
  layout.add_split_configs(SplitConfig(0, {10}));
  layout.add_split_configs(SplitConfig(1, {20}));
  layout.add_split_configs(SplitConfig(2, {30}));

  layout.DeleteDimension(0);

  EXPECT_THAT(layout.minor_to_major(), ElementsAre(1, 0));

  ASSERT_EQ(layout.split_configs().size(), 2);

  EXPECT_EQ(layout.split_configs(0).dimension(), 0);
  EXPECT_THAT(layout.split_configs(0).split_indices(), ElementsAre(10));

  EXPECT_EQ(layout.split_configs(1).dimension(), 1);
  EXPECT_THAT(layout.split_configs(1).split_indices(), ElementsAre(30));
}

TEST(Layout, BitPlanesZeroInlineSizeOverhead) {
  struct BaselineLayoutSize {
    PrimitiveType index_primitive_type : 8;
    PrimitiveType pointer_primitive_type : 8;
    int8_t memory_space;
    int64_t element_size_in_bits;
    DimensionVector minor_to_major;
    TileVector tiles;
    std::vector<SplitConfig> split_configs;
    int64_t tail_padding_alignment_in_elements;
    std::unique_ptr<Shape> physical_shape;
    int64_t dynamic_shape_metadata_prefix_bytes;
  };
  static_assert(sizeof(Layout) == sizeof(BaselineLayoutSize),
                "ColdData pointer must add zero bytes to sizeof(Layout).");
}

TEST(Layout, BitPlanesToStringEqualityHashAndProtoRoundTrip) {
  Layout fp6_2d({1, 0}, {Tile({256, 128}), Tile({16, 128})});
  fp6_2d.add_bit_planes(BitPlane(0, 4, {Tile({8, 1})}))
      .add_bit_planes(BitPlane(4, 2, {Tile({16, 1})}))
      .set_memory_space(1);
  EXPECT_EQ(fp6_2d.ToString(),
            "{1,0:T(256,128)(16,128)BP(4:(8,1))(2:(16,1))S(1)}");

  Layout msb_first({1, 0}, {Tile({16, 128})});
  msb_first.add_bit_planes(BitPlane(4, 2, {Tile({16, 1})}))
      .add_bit_planes(BitPlane(0, 4, {Tile({8, 1})}));
  EXPECT_EQ(msb_first.ToString(), "{1,0:T(16,128)BP(2@4:(16,1))(4@0:(8,1))}");

  Layout s64_0d;
  s64_0d.add_bit_planes(BitPlane(0, 32, {Tile({128})}))
      .add_bit_planes(BitPlane(32, 32, {Tile({128})}));
  EXPECT_EQ(s64_0d.ToString(), "{:BP(32:(128))(32:(128))}");

  Layout s64_linear({0});
  s64_linear.set_tail_padding_alignment_in_elements(128)
      .add_bit_planes(BitPlane(0, 32, {Tile({1})}))
      .add_bit_planes(BitPlane(32, 32, {Tile({1})}));
  EXPECT_EQ(s64_linear.ToString(), "{0:L(128)BP(32:(1))(32:(1))}");

  Layout untiled_bp({1, 0}, {Tile({16, 128})});
  untiled_bp.add_bit_planes(BitPlane(0, 32)).add_bit_planes(BitPlane(32, 32));
  EXPECT_EQ(untiled_bp.ToString(), "{1,0:T(16,128)BP(32)(32)}");

  EXPECT_NE(fp6_2d, msb_first);
  EXPECT_TRUE(Layout::Equal().IgnoreTiles().IgnoreMemorySpace()(
      fp6_2d, Layout({1, 0})));
  EXPECT_TRUE(Layout::Equal().IgnoreBitPlanes()(
      untiled_bp, Layout({1, 0}, {Tile({16, 128})})));
  EXPECT_TRUE(Layout::Equal().MinorToMajorOnly()(fp6_2d, Layout({1, 0})));
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      Layout(),
      Layout({1, 0}),
      fp6_2d,
      msb_first,
      s64_0d,
      s64_linear,
      untiled_bp,
  }));

  for (const Layout& l : {fp6_2d, msb_first, s64_0d, s64_linear, untiled_bp}) {
    LayoutProto proto;
    proto.add_bit_planes()->set_bit_width(99);
    l.ToProto(proto);
    ASSERT_OK_AND_ASSIGN(Layout roundtrip, Layout::FromProto(proto));
    EXPECT_EQ(roundtrip, l);
  }

  // Unvalidated INT64_MAX bit_width formats without signed overflow.
  Layout huge_width({0});
  huge_width.add_bit_planes(BitPlane(0, std::numeric_limits<int64_t>::max()))
      .add_bit_planes(BitPlane(0, 1));
  EXPECT_EQ(huge_width.ToString(), "{0:BP(9223372036854775807@0)(1@0)}");
}

TEST(Layout, DeleteDimensionHandlesBitPlanes) {
  Layout batch_fp6({2, 1, 0}, {Tile({256, 128}), Tile({16, 128})});
  batch_fp6.add_bit_planes(BitPlane(0, 4, {Tile({8, 1})}))
      .add_bit_planes(BitPlane(4, 2, {Tile({16, 1})}));
  // Deleting untiled major dim 0 keeps tiles and bit_planes intact.
  batch_fp6.DeleteDimension(0);
  EXPECT_EQ(batch_fp6.ToString(),
            "{1,0:T(256,128)(16,128)BP(4:(8,1))(2:(16,1))}");
  // Deleting tiled minor dim 1 clears tiles and bit_planes.
  batch_fp6.DeleteDimension(1);
  EXPECT_EQ(batch_fp6.ToString(), "{0}");

  // Deleting a major dimension consumed by a higher-rank subtile step clears
  // tiles and bit_planes rather than leaving an invalid subtile rank.
  Layout higher_rank_subtile({1, 0}, {Tile({2048}), Tile({2, 2, 1024})});
  higher_rank_subtile.add_bit_planes(BitPlane(0, 32))
      .add_bit_planes(BitPlane(32, 32));
  higher_rank_subtile.DeleteDimension(0);
  EXPECT_EQ(higher_rank_subtile.ToString(), "{0}");

  Layout whole_tensor_higher_rank_subtile({1, 0});
  whole_tensor_higher_rank_subtile
      .add_bit_planes(BitPlane(0, 32, {Tile({2048}), Tile({2, 2, 1024})}))
      .add_bit_planes(BitPlane(32, 32, {Tile({2048}), Tile({2, 2, 1024})}));
  whole_tensor_higher_rank_subtile.DeleteDimension(0);
  EXPECT_EQ(whole_tensor_higher_rank_subtile.ToString(), "{0}");
}

}  // namespace
}  // namespace xla
