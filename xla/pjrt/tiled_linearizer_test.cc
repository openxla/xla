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

#include "xla/pjrt/tiled_linearizer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/transpose.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;

constexpr int64_t kTcLaneCount = 128;
constexpr size_t kParallelChunkTargetBytes = 2 << 20;

struct PlanExecutionStats {
  int plan_calls = 0;
  int max_plan_parallelism = 0;
  size_t max_output_bytes = 0;
};

std::vector<uint8_t> MakeDeterministicHostBytes(size_t count,
                                                PrimitiveType type) {
  const uint8_t mask = (type == PRED) ? uint8_t{0x01} : uint8_t{0xff};
  std::vector<uint8_t> bytes(count);
  uint64_t state = 0x9e3779b97f4a7c15ULL;
  for (size_t i = 0; i < count; ++i) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    bytes[i] =
        static_cast<uint8_t>((state * 0x2545f4914f6cdd1dULL) >> 56) & mask;
  }
  return bytes;
}

Layout MakeTiledLayout(absl::Span<const int64_t> minor_to_major,
                       absl::Span<const Tile> tiles, int bits_per_elem = 0,
                       int64_t prefix_bytes = 0) {
  Layout layout = LayoutUtil::MakeLayout(minor_to_major, tiles);
  layout.set_element_size_in_bits(bits_per_elem);
  layout.set_dynamic_shape_metadata_prefix_bytes(prefix_bytes);
  return layout;
}

TiledLinearizer::ParallelismOptions MakeThreadPoolOptions(
    tsl::thread::ThreadPool* pool, int max_parallelism,
    PlanExecutionStats* stats = nullptr) {
  TiledLinearizer::ParallelismOptions opts;
  opts.max_parallelism = max_parallelism;
  if (max_parallelism > 1 && pool != nullptr) {
    opts.run_plan = [pool, stats](const TransposePlan& plan, const void* a,
                                  void* b) {
      if (stats != nullptr) {
        ++stats->plan_calls;
        stats->max_plan_parallelism =
            std::max(stats->max_plan_parallelism, plan.Parallelism());
        stats->max_output_bytes =
            std::max<size_t>(stats->max_output_bytes,
                             static_cast<size_t>(plan.OutputNumElems()) *
                                 plan.ElemSizeInBytes());
      }
      plan.Execute(a, b, [pool](std::function<void()> fn) {
        pool->Schedule(std::move(fn));
      });
    };
  }
  return opts;
}

size_t ComputeTiledElements(const Shape& shape, const Layout& layout) {
  if (shape.dimensions().empty()) {
    return 1;
  }
  if (shape.dimensions().size() == 1) {
    return shape.dimensions(0);
  }
  size_t elems = 1;
  const auto m2m = layout.minor_to_major();
  const auto tile0 = layout.tiles(0).dimensions();
  const size_t ndim = shape.dimensions().size();
  const size_t untiled = ndim - tile0.size();
  for (size_t d = 0; d < ndim; ++d) {
    const int64_t tile = d < untiled ? 1 : tile0[d - untiled];
    elems *= RoundUpTo<int64_t>(shape.dimensions(m2m[ndim - 1 - d]), tile);
  }
  return elems;
}

size_t ComputePackedBytes(const Shape& shape, const Layout& layout,
                          size_t extra_sublanes = 0) {
  const int bits =
      layout.element_size_in_bits() != 0
          ? layout.element_size_in_bits()
          : ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type()) * CHAR_BIT;
  const size_t sublane_elems = static_cast<size_t>(kTcLaneCount * (32 / bits));
  const size_t padded_elems =
      RoundUpTo<size_t>(ComputeTiledElements(shape, layout), sublane_elems) +
      extra_sublanes * sublane_elems;
  return (padded_elems * static_cast<size_t>(bits)) / CHAR_BIT;
}

// Reference oracle using a single 1-byte TransposePlan + scalar bit shifts.
absl::StatusOr<std::vector<uint8_t>> ReferenceLinearizeSubbyte(
    size_t dest_size_bytes, const void* data, const Shape& host_shape,
    absl::Span<const int64_t> byte_strides, const Layout& resolved_layout,
    bool set_padding_to_nan, absl::Span<const uint32_t> dynamic_sizes) {
  const int bits = resolved_layout.element_size_in_bits();
  const int packing_factor = 32 / bits;
  const size_t sublane_bytes = kTcLaneCount * packing_factor;
  const uint8_t pad_byte = set_padding_to_nan ? 0xff : 0x00;
  const size_t prefix_bytes =
      dynamic_sizes.empty()
          ? 0
          : resolved_layout.dynamic_shape_metadata_prefix_bytes();
  const size_t unpacked_bytes =
      (dest_size_bytes - prefix_bytes) * (CHAR_BIT / bits);

  std::vector<uint8_t> out(dest_size_bytes, pad_byte);
  for (size_t i = 0; i < dynamic_sizes.size(); ++i) {
    for (size_t b = 0; b < sizeof(uint32_t); ++b) {
      out[i * sizeof(uint32_t) + b] =
          static_cast<uint8_t>((dynamic_sizes[i] >> (b * CHAR_BIT)) & 0xff);
    }
  }

  absl::InlinedVector<int64_t, 4> dims(host_shape.dimensions().begin(),
                                       host_shape.dimensions().end());
  absl::InlinedVector<int64_t, 4> strides(byte_strides.begin(),
                                          byte_strides.end());
  absl::InlinedVector<int64_t, 4> perm;
  if (dims.empty()) {
    dims = {1};
    strides = {1};
    perm = {0};
  } else {
    perm.assign(resolved_layout.minor_to_major().rbegin(),
                resolved_layout.minor_to_major().rend());
  }
  absl::InlinedVector<int64_t, 2> tiling;
  if (dims.size() > 1 && !resolved_layout.tiles().empty()) {
    const auto t0 = resolved_layout.tiles(0).dimensions();
    tiling.assign(t0.begin(), t0.end());
  }

  std::vector<uint8_t> unpacked(unpacked_bytes, pad_byte);
  if (ComputeTiledElements(host_shape, resolved_layout) > 0) {
    TransposePlan::Options opts;
    opts.elem_size_in_bytes = 1;
    opts.dims = dims;
    opts.permutation = perm;
    opts.input_striding = TransposePlan::Striding{strides};
    opts.output_tiling = TransposePlan::Tiling{tiling};
    opts.tile_padding = pad_byte;
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<TransposePlan> plan,
                          TransposePlan::Create(opts));
    plan->Execute(data, unpacked.data());
  }

  const uint8_t mask = static_cast<uint8_t>((1u << bits) - 1u);
  for (size_t s = 0; s < unpacked_bytes / sublane_bytes; ++s) {
    const absl::Span<const uint8_t> sublane =
        absl::MakeConstSpan(unpacked).subspan(s * sublane_bytes, sublane_bytes);
    for (int64_t lane = 0; lane < kTcLaneCount; ++lane) {
      uint32_t word = 0;
      for (int p = 0; p < packing_factor; ++p) {
        word |= static_cast<uint32_t>(sublane[p * kTcLaneCount + lane] & mask)
                << (p * bits);
      }
      const size_t byte_off =
          prefix_bytes + (s * kTcLaneCount + lane) * sizeof(uint32_t);
      for (size_t b = 0; b < sizeof(uint32_t); ++b) {
        out[byte_off + b] =
            static_cast<uint8_t>((word >> (b * CHAR_BIT)) & 0xff);
      }
    }
  }
  return out;
}

absl::Span<uint8_t> MutableLiteralBytes(MutableLiteralBase* literal) {
  return absl::MakeSpan(static_cast<uint8_t*>(literal->untyped_data()),
                        literal->size_bytes());
}

absl::Status VerifyLinearizeAndRoundTrip(
    const Shape& host_shape, const Layout& device_layout,
    bool set_padding_to_nan, size_t extra_sublanes = 0,
    absl::Span<const uint32_t> dynamic_sizes = {},
    const Layout* oracle_override = nullptr) {
  const Layout& oracle_layout =
      oracle_override != nullptr ? *oracle_override : device_layout;
  const int64_t num_elems = ShapeUtil::ElementsIn(host_shape);
  Literal src(host_shape);
  if (num_elems > 0) {
    const std::vector<uint8_t> raw =
        MakeDeterministicHostBytes(num_elems, host_shape.element_type());
    absl::c_copy(raw, MutableLiteralBytes(&src).begin());
  }
  absl::InlinedVector<int64_t, 4> strides(host_shape.dimensions().size());
  if (!strides.empty()) {
    ABSL_RETURN_IF_ERROR(
        ShapeUtil::UnpackedByteStrides(host_shape, absl::MakeSpan(strides)));
  }

  const size_t prefix_bytes =
      dynamic_sizes.empty()
          ? 0
          : device_layout.dynamic_shape_metadata_prefix_bytes();
  const size_t payload_bytes =
      ComputePackedBytes(host_shape, oracle_layout, extra_sublanes);
  const size_t total_bytes = prefix_bytes + payload_bytes;
  ABSL_ASSIGN_OR_RETURN(
      std::vector<uint8_t> expected,
      ReferenceLinearizeSubbyte(total_bytes, src.untyped_data(), host_shape,
                                strides, oracle_layout, set_padding_to_nan,
                                dynamic_sizes));

  std::vector<uint8_t> actual_serial(total_bytes, 0xa5);
  ABSL_RETURN_IF_ERROR(TiledLinearizer::LinearizeUsingTransposeInto(
      absl::MakeSpan(actual_serial), src.untyped_data(), host_shape, strides,
      device_layout, set_padding_to_nan, dynamic_sizes,
      TiledLinearizer::ParallelismOptions{}));
  TF_RET_CHECK(actual_serial == expected);

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "linearizer_test", 8);
  PlanExecutionStats h2d_stats;
  std::vector<uint8_t> actual_parallel(total_bytes, 0x5a);
  ABSL_RETURN_IF_ERROR(TiledLinearizer::LinearizeUsingTransposeInto(
      absl::MakeSpan(actual_parallel), src.untyped_data(), host_shape, strides,
      device_layout, set_padding_to_nan, dynamic_sizes,
      MakeThreadPoolOptions(&pool, 8, &h2d_stats)));
  TF_RET_CHECK(actual_parallel == expected);

  const size_t sublane_bytes =
      kTcLaneCount * (32 / oracle_layout.element_size_in_bits());
  if (ComputeTiledElements(host_shape, oracle_layout) >
      kParallelChunkTargetBytes + sublane_bytes) {
    TF_RET_CHECK(h2d_stats.plan_calls >= 2);
    TF_RET_CHECK(h2d_stats.max_output_bytes > 0);
    TF_RET_CHECK(h2d_stats.max_output_bytes <= kParallelChunkTargetBytes);
  }
  if (payload_bytes > 0) {
    Literal dst(host_shape);
    ABSL_RETURN_IF_ERROR(TiledLinearizer::Delinearize(
        host_shape, absl::MakeConstSpan(actual_parallel).subspan(prefix_bytes),
        device_layout, &dst, MakeThreadPoolOptions(&pool, 8)));
    TF_RET_CHECK(dst == src);
  }
  return absl::OkStatus();
}

TEST(TiledLinearizerTest, MultiMibS4PermutationsPartialTilesAndSplits) {
  for (int64_t major_tile : {8, 64}) {
    const Shape aligned = ShapeUtil::MakeShapeWithDescendingLayout(
        S4, {6, 4096 / major_tile, major_tile, 128});
    for (const auto& m2m : std::vector<std::vector<int64_t>>{
             {3, 2, 1, 0}, {1, 3, 0, 2}, {2, 0, 3, 1}}) {
      SCOPED_TRACE(
          absl::StrCat("tile=", major_tile, " m2m=", absl::StrJoin(m2m, ",")));
      EXPECT_OK(VerifyLinearizeAndRoundTrip(
          aligned,
          MakeTiledLayout(m2m, {Tile({major_tile, 128}), Tile({8, 1})}, 4),
          /*set_padding_to_nan=*/false));
    }
  }
  const Shape partial =
      ShapeUtil::MakeShapeWithDescendingLayout(S4, {3, 68, 67, 131});
  for (const auto& m2m :
       std::vector<std::vector<int64_t>>{{3, 2, 0, 1}, {2, 3, 0, 1}}) {
    for (bool nan : {false, true}) {
      SCOPED_TRACE(absl::StrCat("m2m=", absl::StrJoin(m2m, ","), " nan=", nan));
      EXPECT_OK(VerifyLinearizeAndRoundTrip(
          partial, MakeTiledLayout(m2m, {Tile({8, 128}), Tile({8, 1})}, 4), nan,
          /*extra_sublanes=*/3));
    }
  }

  // Inner tile-grid splits, 1D layout, sublane-step amplification + carry, and
  // `Tile::kMatchDimension` intra-tile splitting (including pure-padding tail).
  EXPECT_OK(VerifyLinearizeAndRoundTrip(
      ShapeUtil::MakeShapeWithDescendingLayout(S4, {131, 300003}),
      MakeTiledLayout({1, 0}, {Tile({8, 128}), Tile({8, 1})}, 4), true));
  EXPECT_OK(VerifyLinearizeAndRoundTrip(
      ShapeUtil::MakeShapeWithDescendingLayout(S4, {3500111}),
      MakeTiledLayout({0}, {Tile({1024}), Tile({128}), Tile({8, 1})}, 4), false,
      /*extra_sublanes=*/2));
  EXPECT_OK(VerifyLinearizeAndRoundTrip(
      ShapeUtil::MakeShapeWithDescendingLayout(S4, {10, 5001, 2, 128}),
      MakeTiledLayout({3, 2, 1, 0}, {Tile({2, 128})}, 4), true));
  EXPECT_OK(VerifyLinearizeAndRoundTrip(
      ShapeUtil::MakeShapeWithDescendingLayout(S4, {5, 8401, 2, 128}),
      MakeTiledLayout({3, 2, 1, 0}, {Tile({2, 128})}, 4), true));

  const Layout resolved_match =
      MakeTiledLayout({1, 0}, {Tile({20008, 128}), Tile({8, 1})}, 4);
  EXPECT_OK(VerifyLinearizeAndRoundTrip(
      ShapeUtil::MakeShapeWithDescendingLayout(S4, {20003, 256}),
      MakeTiledLayout({1, 0},
                      {Tile({Tile::kMatchDimension, 128}), Tile({8, 1})}, 4),
      true, /*extra_sublanes=*/0, /*dynamic_sizes=*/{}, &resolved_match));
  EXPECT_OK(VerifyLinearizeAndRoundTrip(
      ShapeUtil::MakeShapeWithDescendingLayout(S4, {16000, 256}),
      resolved_match, true));
}

TEST(TiledLinearizerTest, SubbyteTypesEdgeCasesThresholdAndValidation) {
  EXPECT_OK(VerifyLinearizeAndRoundTrip(
      ShapeUtil::MakeShapeWithDescendingLayout(U4, {25, 997, 130}),
      MakeTiledLayout({2, 0, 1}, {Tile({8, 128}), Tile({8, 1})}, 4), false));
  for (PrimitiveType type : {S2, U2}) {
    SCOPED_TRACE(PrimitiveType_Name(type));
    EXPECT_OK(VerifyLinearizeAndRoundTrip(
        ShapeUtil::MakeShapeWithDescendingLayout(type, {17, 1500, 131}),
        MakeTiledLayout({2, 1, 0}, {Tile({16, 128}), Tile({16, 1})}, 2), true));
  }
  EXPECT_OK(VerifyLinearizeAndRoundTrip(
      ShapeUtil::MakeShapeWithDescendingLayout(PRED, {65, 257, 259}),
      MakeTiledLayout({1, 2, 0}, {Tile({256, 128}), Tile({32, 1})}, 1), false));
  EXPECT_OK(VerifyLinearizeAndRoundTrip(
      ShapeUtil::MakeShape(S4, {}),
      MakeTiledLayout({}, {Tile({128}), Tile({8, 1})}, 4), false));
  EXPECT_OK(VerifyLinearizeAndRoundTrip(
      ShapeUtil::MakeShapeWithDescendingLayout(S4, {4, 0, 128}),
      MakeTiledLayout({2, 1, 0}, {Tile({8, 128}), Tile({8, 1})}, 4), true, 2));
  const std::vector<uint32_t> dyn_sizes = {5, 1020, 128};
  EXPECT_OK(VerifyLinearizeAndRoundTrip(
      ShapeUtil::MakeShapeWithDescendingLayout(S4, {5, 4200, 128}),
      MakeTiledLayout({2, 1, 0}, {Tile({8, 128}), Tile({8, 1})}, 4, 1024),
      false, /*extra_sublanes=*/1, dyn_sizes));

  // Negative + padded host byte strides and 2 MiB parallel H2D threshold.
  const Shape chunked_shape =
      ShapeUtil::MakeShapeWithDescendingLayout(S4, {2050, 8, 128});
  const Layout tc_layout =
      MakeTiledLayout({2, 1, 0}, {Tile({8, 128}), Tile({8, 1})}, 4);
  const int64_t row_pitch = 8 * 160;
  std::vector<uint8_t> backing =
      MakeDeterministicHostBytes(2050 * row_pitch, S4);
  const uint8_t* neg_origin = &backing[2049 * row_pitch];
  const std::vector<int64_t> neg_strides = {-row_pitch, 160, 1};
  const size_t packed_bytes = ComputePackedBytes(chunked_shape, tc_layout);
  ASSERT_OK_AND_ASSIGN(
      std::vector<uint8_t> expected_strided,
      ReferenceLinearizeSubbyte(packed_bytes, neg_origin, chunked_shape,
                                neg_strides, tc_layout, false, {}));
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "threshold_pool", 8);
  std::vector<uint8_t> actual_strided(packed_bytes, 0);
  ASSERT_OK(TiledLinearizer::LinearizeUsingTransposeInto(
      absl::MakeSpan(actual_strided), neg_origin, chunked_shape, neg_strides,
      tc_layout, false, {}, MakeThreadPoolOptions(&pool, 4)));
  EXPECT_EQ(actual_strided, expected_strided);

  auto check_h2d_threshold = [&](const Shape& shape, int calls, int parallelism,
                                 size_t max_out) {
    std::vector<uint8_t> src(ShapeUtil::ElementsIn(shape), 0x37);
    absl::InlinedVector<int64_t, 4> strides(3);
    ASSERT_OK(ShapeUtil::UnpackedByteStrides(shape, absl::MakeSpan(strides)));
    std::vector<uint8_t> dst(ComputePackedBytes(shape, tc_layout), 0);
    PlanExecutionStats stats;
    ASSERT_OK(TiledLinearizer::LinearizeUsingTransposeInto(
        absl::MakeSpan(dst), src.data(), shape, strides, tc_layout, false, {},
        MakeThreadPoolOptions(&pool, 8, &stats)));
    EXPECT_EQ(stats.plan_calls, calls);
    EXPECT_EQ(stats.max_plan_parallelism, parallelism);
    EXPECT_EQ(stats.max_output_bytes, max_out);
  };
  check_h2d_threshold(
      ShapeUtil::MakeShapeWithDescendingLayout(S4, {3, 4096, 128}), 1, 2,
      3 * 4096 * 128);
  check_h2d_threshold(
      ShapeUtil::MakeShapeWithDescendingLayout(S4, {4, 4096, 128}), 1, 2,
      4 * 4096 * 128);
  check_h2d_threshold(chunked_shape, 2, 8, kParallelChunkTargetBytes);

  const Shape small = ShapeUtil::MakeShapeWithDescendingLayout(S4, {3, 2, 128});
  std::vector<uint8_t> small_src(3 * 2 * 128, 0);
  absl::InlinedVector<int64_t, 4> small_strides(3);
  ASSERT_OK(
      ShapeUtil::UnpackedByteStrides(small, absl::MakeSpan(small_strides)));
  std::vector<uint8_t> aligned_dst(ComputePackedBytes(small, tc_layout));
  TiledLinearizer::ParallelismOptions bad_opts;
  bad_opts.max_parallelism = 0;
  EXPECT_THAT(TiledLinearizer::LinearizeUsingTransposeInto(
                  absl::MakeSpan(aligned_dst), small_src.data(), small,
                  small_strides, tc_layout, false, {}, bad_opts),
              StatusIs(absl::StatusCode::kInvalidArgument));
  std::vector<uint8_t> misaligned_dst((3 * 2 * 128 * 4) / CHAR_BIT, 0);
  EXPECT_THAT(
      TiledLinearizer::LinearizeUsingTransposeInto(
          absl::MakeSpan(misaligned_dst), small_src.data(), small,
          small_strides, MakeTiledLayout({2, 1, 0}, {Tile({2, 128})}, 4), false,
          {}, TiledLinearizer::ParallelismOptions{}),
      StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace xla
