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

#include "xla/pjrt/common_pjrt_client.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <memory>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "benchmark/benchmark.h"  // IWYU pragma: keep
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/pjrt/tiled_linearizer.h"
#include "xla/pjrt/transpose.h"
#include "xla/primitive_util.h"
#include "xla/runtime/device_id.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

TEST(TransferIncarnationsTest, EmptySnapshotReturnsNullopt) {
  ASSERT_OK_AND_ASSIGN(auto transfer_incarnations,
                       TransferIncarnations(/*src_process_index=*/1,
                                            /*dst_process_index=*/2, {}));
  EXPECT_FALSE(transfer_incarnations.has_value());
}

TEST(TransferIncarnationsTest, SnapshotKeepsSourceThenDestination) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(2), IncarnationId(20)},
      {TaskId(1), IncarnationId(10)},
  };

  ASSERT_OK_AND_ASSIGN(
      auto forward,
      TransferIncarnations(/*src_process_index=*/2, /*dst_process_index=*/1,
                           incarnations));
  ASSERT_TRUE(forward.has_value());
  EXPECT_THAT(*forward, ElementsAre(IncarnationId(20), IncarnationId(10)));

  ASSERT_OK_AND_ASSIGN(
      auto swapped,
      TransferIncarnations(/*src_process_index=*/1, /*dst_process_index=*/2,
                           incarnations));
  ASSERT_TRUE(swapped.has_value());
  EXPECT_THAT(*swapped, ElementsAre(IncarnationId(10), IncarnationId(20)));
}

TEST(TransferIncarnationsTest, SameTaskKeepsBothIncarnations) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(3), IncarnationId(7)},
  };

  ASSERT_OK_AND_ASSIGN(
      auto transfer_incarnations,
      TransferIncarnations(/*src_process_index=*/3, /*dst_process_index=*/3,
                           incarnations));
  ASSERT_TRUE(transfer_incarnations.has_value());
  EXPECT_THAT(*transfer_incarnations,
              ElementsAre(IncarnationId(7), IncarnationId(7)));
}

TEST(TransferIncarnationsTest, MissingTaskFails) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(1), IncarnationId(10)},
  };

  EXPECT_THAT(TransferIncarnations(/*src_process_index=*/1,
                                   /*dst_process_index=*/2, incarnations),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

template <typename T>
Literal MakePatternLiteral(const Shape& shape, absl::Span<const T> palette) {
  Literal literal(shape);
  absl::Span<T> data = literal.data<T>();
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = palette[i % palette.size()];
  }
  return literal;
}

std::vector<uint8_t> PackLiteralBytes(const Literal& literal) {
  const PrimitiveType element_type = literal.shape().element_type();
  const int bit_width = primitive_util::BitWidth(element_type);
  const size_t num_elements =
      static_cast<size_t>(ShapeUtil::ElementsIn(literal.shape()));
  const size_t packed_bytes = CeilOfRatio<size_t>(num_elements, 8 / bit_width);
  std::vector<uint8_t> packed(packed_bytes, 0);
  absl::Span<const char> src(static_cast<const char*>(literal.untyped_data()),
                             num_elements);
  absl::Span<char> dst(reinterpret_cast<char*>(packed.data()), packed.size());
  primitive_util::PackIntN(element_type, src, dst);
  return packed;
}

TEST(CommonPjRtClientDelinearizeTest, SubBytePackedMatchingMinorToMajor) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(CpuClientOptions{}));
  auto* common_client = absl::down_cast<CommonPjRtClient*>(client.get());
  ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memory_space,
      client->addressable_devices()[0]->default_memory_space());

  const std::vector<s4> s4_palette = {
      s4(-8), s4(-7), s4(-6), s4(-5), s4(-4), s4(-3), s4(-2), s4(-1),
      s4(0),  s4(1),  s4(2),  s4(3),  s4(4),  s4(5),  s4(6),  s4(7)};
  const std::vector<u4> u4_palette = {
      u4(0), u4(1), u4(2),  u4(3),  u4(4),  u4(5),  u4(6),  u4(7),
      u4(8), u4(9), u4(10), u4(11), u4(12), u4(13), u4(14), u4(15)};
  const std::vector<std::vector<int64_t>> dims_cases = {
      {}, {0}, {1}, {3, 5}, {4, 4}};

  for (const std::vector<int64_t>& dims : dims_cases) {
    for (int64_t mem_space :
         {Layout::kDefaultMemorySpace, Layout::kHostMemorySpace}) {
      Shape s4_host_shape = ShapeUtil::MakeShape(S4, dims);
      Literal expected_s4 = MakePatternLiteral<s4>(s4_host_shape, s4_palette);
      std::vector<uint8_t> packed_s4 = PackLiteralBytes(expected_s4);
      Shape s4_device_shape = s4_host_shape;
      s4_device_shape.mutable_layout()->set_element_size_in_bits(4);
      s4_device_shape.mutable_layout()->set_memory_space(mem_space);
      Literal actual_s4(s4_host_shape);
      EXPECT_THAT(common_client->Delinearize(packed_s4, s4_device_shape,
                                             &actual_s4, memory_space),
                  IsOk());
      EXPECT_EQ(actual_s4, expected_s4);

      if (ShapeUtil::ElementsIn(s4_host_shape) % 2 == 1) {
        // Non-zero high-nibble padding in the final packed byte must be masked.
        packed_s4.back() |= 0xF0;
        Literal actual_padded_s4(s4_host_shape);
        EXPECT_THAT(common_client->Delinearize(packed_s4, s4_device_shape,
                                               &actual_padded_s4, memory_space),
                    IsOk());
        EXPECT_EQ(actual_padded_s4, expected_s4);
      }

      Shape u4_host_shape = ShapeUtil::MakeShape(U4, dims);
      Literal expected_u4 = MakePatternLiteral<u4>(u4_host_shape, u4_palette);
      std::vector<uint8_t> packed_u4 = PackLiteralBytes(expected_u4);
      Shape u4_device_shape = u4_host_shape;
      u4_device_shape.mutable_layout()->set_element_size_in_bits(4);
      u4_device_shape.mutable_layout()->set_memory_space(mem_space);
      Literal actual_u4(u4_host_shape);
      EXPECT_THAT(common_client->Delinearize(packed_u4, u4_device_shape,
                                             &actual_u4, memory_space),
                  IsOk());
      EXPECT_EQ(actual_u4, expected_u4);
    }
  }

  const std::vector<s2> s2_palette = {s2(-2), s2(-1), s2(0), s2(1)};
  Shape s2_host_shape = ShapeUtil::MakeShape(S2, {3, 3});
  Literal expected_s2 = MakePatternLiteral<s2>(s2_host_shape, s2_palette);
  std::vector<uint8_t> packed_s2 = PackLiteralBytes(expected_s2);
  Shape s2_device_shape = s2_host_shape;
  s2_device_shape.mutable_layout()->set_element_size_in_bits(2);
  Literal actual_s2(s2_host_shape);
  EXPECT_THAT(common_client->Delinearize(packed_s2, s2_device_shape, &actual_s2,
                                         memory_space),
              IsOk());
  EXPECT_EQ(actual_s2, expected_s2);

  const std::vector<u2> u2_palette = {u2(0), u2(1), u2(2), u2(3)};
  Shape u2_host_shape = ShapeUtil::MakeShape(U2, {3, 3});
  Literal expected_u2 = MakePatternLiteral<u2>(u2_host_shape, u2_palette);
  std::vector<uint8_t> packed_u2 = PackLiteralBytes(expected_u2);
  Shape u2_device_shape = u2_host_shape;
  u2_device_shape.mutable_layout()->set_element_size_in_bits(2);
  Literal actual_u2(u2_host_shape);
  EXPECT_THAT(common_client->Delinearize(packed_u2, u2_device_shape, &actual_u2,
                                         memory_space),
              IsOk());
  EXPECT_EQ(actual_u2, expected_u2);
}

TEST(CommonPjRtClientDelinearizeTest,
     SubBytePackedNonDefaultMatchingMinorToMajor) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(CpuClientOptions{}));
  auto* common_client = absl::down_cast<CommonPjRtClient*>(client.get());
  ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memory_space,
      client->addressable_devices()[0]->default_memory_space());

  Shape col_major_shape = ShapeUtil::MakeShapeWithDenseLayout(
      S4, {3, 5}, /*minor_to_major=*/{0, 1});
  const std::vector<s4> palette = {s4(-8), s4(-5), s4(-1), s4(0),
                                   s4(2),  s4(4),  s4(7)};
  Literal expected = MakePatternLiteral<s4>(col_major_shape, palette);
  std::vector<uint8_t> packed = PackLiteralBytes(expected);

  Shape device_shape = col_major_shape;
  device_shape.mutable_layout()->set_element_size_in_bits(4);
  device_shape.mutable_layout()->set_memory_space(Layout::kHostMemorySpace);

  Literal actual(col_major_shape);
  EXPECT_THAT(
      common_client->Delinearize(packed, device_shape, &actual, memory_space),
      IsOk());
  EXPECT_EQ(actual, expected);
}

TEST(CommonPjRtClientDelinearizeTest, SubBytePackedTransposedMinorToMajor) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(CpuClientOptions{}));
  auto* common_client = absl::down_cast<CommonPjRtClient*>(client.get());
  ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memory_space,
      client->addressable_devices()[0]->default_memory_space());

  Literal row_major = LiteralUtil::CreateR2<s4>(
      {{s4(-8), s4(-3), s4(1)}, {s4(2), s4(5), s4(7)}});
  std::vector<uint8_t> packed = PackLiteralBytes(row_major);

  Shape device_shape = ShapeUtil::MakeShapeWithDenseLayout(
      S4, {2, 3}, /*minor_to_major=*/{1, 0});
  device_shape.mutable_layout()->set_element_size_in_bits(4);

  Literal actual(ShapeUtil::MakeShapeWithDenseLayout(
      S4, {2, 3}, /*minor_to_major=*/{0, 1}));
  EXPECT_THAT(
      common_client->Delinearize(packed, device_shape, &actual, memory_space),
      IsOk());
  // In {0, 1} layout, dimension 0 is minor: (0,0), (1,0), (0,1), (1,1), ...
  EXPECT_THAT(actual.data<s4>(),
              ElementsAre(s4(-8), s4(2), s4(-3), s4(5), s4(1), s4(7)));
}

TEST(CommonPjRtClientDelinearizeTest, HostMemorySpaceMatchingMinorToMajor) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(CpuClientOptions{}));
  auto* common_client = absl::down_cast<CommonPjRtClient*>(client.get());
  ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memory_space,
      client->addressable_devices()[0]->default_memory_space());

  Literal expected =
      LiteralUtil::CreateR2<float>({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
  Shape device_shape = expected.shape();
  device_shape.mutable_layout()->set_memory_space(Layout::kHostMemorySpace);

  absl::Span<const uint8_t> src_bytes(
      static_cast<const uint8_t*>(expected.untyped_data()),
      expected.size_bytes());
  Literal actual(expected.shape());
  EXPECT_THAT(common_client->Delinearize(src_bytes, device_shape, &actual,
                                         memory_space),
              IsOk());
  EXPECT_EQ(actual, expected);

  Shape empty_shape = ShapeUtil::MakeShape(F32, {0});
  Shape empty_device_shape = empty_shape;
  empty_device_shape.mutable_layout()->set_memory_space(
      Layout::kHostMemorySpace);
  Literal actual_empty(empty_shape);
  EXPECT_THAT(common_client->Delinearize(absl::Span<const uint8_t>(),
                                         empty_device_shape, &actual_empty,
                                         memory_space),
              IsOk());
  EXPECT_EQ(actual_empty, Literal(empty_shape));
}

TEST(CommonPjRtClientDelinearizeTest,
     SubByteHostMemoryForDeviceManagerRoundTrip) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(CpuClientOptions{}));
  PjRtHostMemoryForDeviceManager* manager =
      client->GetPjRtHostMemoryForDeviceManager();
  ASSERT_NE(manager, nullptr);

  const std::vector<s4> palette = {s4(-8), s4(-5), s4(-1), s4(0),
                                   s4(2),  s4(4),  s4(7)};
  Shape host_shape = ShapeUtil::MakeShape(S4, {3, 5});
  Shape device_shape = host_shape;
  device_shape.mutable_layout()->set_element_size_in_bits(4);
  Literal expected = MakePatternLiteral<s4>(host_shape, palette);

  ASSERT_OK_AND_ASSIGN(
      PjRtChunk device_chunk,
      manager->ToDeviceLayout(expected.untyped_data(), expected.size_bytes(),
                              host_shape, device_shape));
  Literal actual(host_shape);
  EXPECT_THAT(manager->ToHostLayout(device_chunk.data(), device_chunk.size(),
                                    device_shape, actual.untyped_data(),
                                    actual.size_bytes(), host_shape),
              IsOk());
  EXPECT_EQ(actual, expected);
}

TEST(CommonPjRtClientDelinearizeTest,
     SubByteBufferFromHostLiteralToLiteralRoundTrip) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(CpuClientOptions{}));
  const std::vector<s4> palette = {s4(-8), s4(-5), s4(-1), s4(0),
                                   s4(2),  s4(4),  s4(7)};
  Shape host_shape = ShapeUtil::MakeShape(S4, {3, 5});
  Literal expected = MakePatternLiteral<s4>(host_shape, palette);

  for (PjRtMemorySpace* memory_space : client->memory_spaces()) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtBuffer> buffer,
                         client->BufferFromHostLiteral(expected, memory_space));
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> actual,
                         buffer->ToLiteral().Await());
    EXPECT_EQ(*actual, expected);

    Layout col_major_device_layout = LayoutUtil::MakeLayout({0, 1});
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtBuffer> col_major_buffer,
                         client->BufferFromHostLiteral(
                             expected, memory_space, &col_major_device_layout));
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> col_major_to_default,
                         col_major_buffer->ToLiteral().Await());
    EXPECT_EQ(*col_major_to_default, expected);

    Literal col_major_literal(ShapeUtil::MakeShapeWithDenseLayout(
        S4, {3, 5}, /*minor_to_major=*/{0, 1}));
    ASSERT_OK(col_major_buffer->ToLiteral(&col_major_literal).Await());
    EXPECT_EQ(col_major_literal, expected.Relayout(col_major_device_layout));
  }
}

// Returns a literal of `shape` holding small integer values (no NaNs), so that
// literals that round-trip through linearization compare equal.
Literal MakeLinearizerTestLiteral(const Shape& shape) {
  Literal literal(shape);
  primitive_util::ArrayTypeSwitch(
      [&](auto ptype) {
        using NativeT = primitive_util::NativeTypeOf<ptype>;
        absl::Span<NativeT> data = literal.data<NativeT>();
        for (size_t i = 0; i < data.size(); ++i) {
          const int value = static_cast<int>(i % 7);
          if constexpr (ptype == PRED) {
            data[i] = value % 2 == 1;
          } else if constexpr (primitive_util::IsComplexType(ptype)) {
            data[i] = NativeT(value + 1, value - 3);
          } else if constexpr (primitive_util::IsUnsignedIntegralType(ptype)) {
            data[i] = static_cast<NativeT>(value);
          } else {
            data[i] = static_cast<NativeT>(value - 3);
          }
        }
      },
      shape.element_type());
  return literal;
}

// Returns the byte strides of a dense major-to-minor array of `shape`.
std::vector<int64_t> MajorToMinorByteStrides(const Shape& shape) {
  std::vector<int64_t> strides(shape.dimensions().size());
  int64_t stride = primitive_util::ByteWidth(shape.element_type());
  for (int64_t i = static_cast<int64_t>(strides.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape.dimensions(i);
  }
  return strides;
}

// Linearizes `literal` into a buffer of `num_bytes` bytes that initially holds
// copies of `initial_word`, and returns the resulting bytes.
absl::StatusOr<std::vector<uint8_t>> LinearizeIntoInitializedBuffer(
    const Literal& literal, const Layout& device_layout, size_t num_bytes,
    bool set_padding_to_nan,
    const TiledLinearizer::ParallelismOptions& parallelism_options,
    uint32_t initial_word) {
  std::vector<uint32_t> buffer(CeilOfRatio(num_bytes, sizeof(uint32_t)),
                               initial_word);
  absl::Span<uint8_t> dest(reinterpret_cast<uint8_t*>(buffer.data()),
                           num_bytes);
  ABSL_RETURN_IF_ERROR(TiledLinearizer::LinearizeUsingTransposeInto(
      dest, literal.untyped_data(), literal.shape(),
      MajorToMinorByteStrides(literal.shape()), device_layout,
      set_padding_to_nan, /*dynamic_sizes=*/{}, parallelism_options));
  return std::vector<uint8_t>(dest.begin(), dest.end());
}

// Returns the offset of the first byte at which `a` and `b` differ, or
// `std::nullopt` if they are identical.
std::optional<size_t> FirstDifference(absl::Span<const uint8_t> a,
                                      absl::Span<const uint8_t> b) {
  if (a.size() != b.size()) {
    return std::min(a.size(), b.size());
  }
  const auto mismatch = absl::c_mismatch(a, b);
  if (mismatch.first == a.end()) {
    return std::nullopt;
  }
  return static_cast<size_t>(mismatch.first - a.begin());
}

struct LinearizeIntoCase {
  PrimitiveType type;
  std::vector<int64_t> dims;
  std::vector<Tile> tiles;
  std::vector<int64_t> minor_to_major = {};  // Empty means major-to-minor.
  int64_t element_size_in_bits = 0;
};

// `LinearizeUsingTransposeInto` fills only the padding that `TransposePlan`
// does not write. Checks that together they write every byte of `dest`, by
// linearizing into buffers that initially hold the padding pattern (which
// reproduces filling all of `dest` with padding first) or one of two poison
// patterns: a byte that is left unwritten keeps its initial value and makes the
// results differ. (Sub-tile packed sub-byte types are transposed into an
// internal scratch buffer, which MSAN checks instead.)
TEST(TiledLinearizerTest, LinearizeUsingTransposeIntoWritesEveryByte) {
  constexpr uint32_t kNanPadWord = 0x7fc07fc0;
  const Tile k8x128({8, 128});
  const std::vector<LinearizeIntoCase> cases = {
      // Rank 0 and 1: the elements are contiguous, only the tail is padding.
      {F32, {}, {Tile({128})}},
      {F32, {7}, {Tile({128})}},
      {BF16, {1025}, {Tile({1024})}},
      {S8, {130}, {Tile({1024})}},
      // 1D tiling: each row ends with a partial tile.
      {F32, {5, 130}, {Tile({128})}},
      {F32, {3, 2, 127}, {Tile({128})}},
      {BF16, {3, 300}, {Tile({256})}},
      // 2D tiling, with and without a partial tile-row (dim0 % t0 != 0) and a
      // partial tile-column (dim1 % t1 != 0).
      {F32, {8, 128}, {k8x128}},
      {F32, {19, 260}, {k8x128}},
      {F32, {16, 260}, {k8x128}},
      {F32, {19, 256}, {k8x128}},
      {F32, {2, 19, 260}, {k8x128}},
      {S32, {2, 3, 9, 129}, {k8x128}},
      // 2D tiling without any complete tile, or without elements.
      {F32, {3, 5}, {k8x128}},
      {F32, {3, 300}, {k8x128}},
      {F32, {20, 100}, {k8x128}},
      {F32, {0, 7}, {k8x128}},
      // Transposing layouts and other tile shapes.
      {U32, {300, 260}, {k8x128}, {0, 1}},
      {F32, {3, 260, 140}, {k8x128}, {1, 2, 0}},
      {F32, {19, 260}, {Tile({2, 128})}},
      {F32, {5, 260}, {Tile({4, 128})}},
      {F32, {6, 260}, {Tile({Tile::kMatchDimension, 128})}},
      // Sub-tile packing, repacked in place after the transpose.
      {BF16, {19, 260}, {k8x128, Tile({2, 1})}},
      {F16, {2, 33, 130}, {Tile({16, 128}), Tile({2, 1})}},
      {S8, {40, 260}, {Tile({32, 128}), Tile({4, 1})}},
      {PRED, {40, 129}, {Tile({32, 128}), Tile({4, 1})}},
      // 8- and 16-byte elements, linearized as 4-byte components.
      {S64, {}, {Tile({256})}},
      {U64, {1000}, {Tile({1024})}},
      {S64, {11, 130}, {k8x128}},
      {F64, {11, 130}, {k8x128}},
      {C64, {11, 130}, {k8x128}},
      {C128, {11, 130}, {k8x128}},
      // Sub-byte elements: sub-tile packed ones go through a scratch buffer,
      // the others are packed by `TransposePlan`.
      {S4, {19, 260}, {k8x128, Tile({8, 1})}, {}, 4},
      {U4, {2, 9, 129}, {k8x128, Tile({8, 1})}, {}, 4},
      {S4, {7}, {Tile({1024})}, {}, 4},
      {U4, {5, 130}, {Tile({128})}, {}, 4},
      // Large enough for `TransposePlan` to split the work into chunks.
      {F32, {2, 1030, 260}, {k8x128}},
  };
  TiledLinearizer::ParallelismOptions serial;
  TiledLinearizer::ParallelismOptions parallel;
  parallel.max_parallelism = 4;
  int max_chunks = 1;
  parallel.run_plan = [&max_chunks](const TransposePlan& plan, const void* a,
                                    void* b) {
    max_chunks = std::max(max_chunks, plan.Parallelism());
    plan.Execute(a, b);
  };
  for (const LinearizeIntoCase& c : cases) {
    const Shape host_shape = ShapeUtil::MakeShape(c.type, c.dims);
    const Layout device_layout = LayoutUtil::MakeLayout(
        c.minor_to_major.empty() ? host_shape.layout().minor_to_major()
                                 : absl::MakeConstSpan(c.minor_to_major),
        c.tiles, /*tail_padding_alignment_in_elements=*/1,
        /*index_primitive_type=*/PRIMITIVE_TYPE_INVALID,
        /*pointer_primitive_type=*/PRIMITIVE_TYPE_INVALID,
        c.element_size_in_bits);
    Shape device_shape = host_shape;
    *device_shape.mutable_layout() = device_layout;
    const size_t array_bytes =
        RoundUpTo<size_t>(ShapeUtil::ArraySize(device_shape), sizeof(uint32_t));
    const Literal literal = MakeLinearizerTestLiteral(host_shape);
    for (size_t tail_bytes : {0, 2048}) {
      for (bool set_padding_to_nan : {false, true}) {
        for (const TiledLinearizer::ParallelismOptions* parallelism :
             {&serial, &parallel}) {
          SCOPED_TRACE(::testing::Message()
                       << ShapeUtil::HumanStringWithLayout(device_shape)
                       << " tail_bytes=" << tail_bytes
                       << " set_padding_to_nan=" << set_padding_to_nan
                       << " max_parallelism=" << parallelism->max_parallelism);
          const size_t num_bytes = array_bytes + tail_bytes;
          ASSERT_OK_AND_ASSIGN(
              std::vector<uint8_t> expected,
              LinearizeIntoInitializedBuffer(
                  literal, device_layout, num_bytes, set_padding_to_nan,
                  *parallelism, set_padding_to_nan ? kNanPadWord : 0));
          for (uint32_t poison : {0xa5a5a5a5u, 0x5a5a5a5au}) {
            ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> actual,
                                 LinearizeIntoInitializedBuffer(
                                     literal, device_layout, num_bytes,
                                     set_padding_to_nan, *parallelism, poison));
            const std::optional<size_t> diff =
                FirstDifference(actual, expected);
            EXPECT_FALSE(diff.has_value())
                << "byte " << diff.value_or(0) << " of " << num_bytes
                << " was not written (initial word 0x" << std::hex << poison
                << ")";
          }
          if (tail_bytes == 0) {
            Literal round_trip(host_shape);
            EXPECT_THAT(
                TiledLinearizer::Delinearize(
                    host_shape, expected, device_layout, &round_trip, serial),
                IsOk());
            EXPECT_EQ(round_trip, literal);
          }
        }
      }
    }
  }
  // Some case must have been split into several chunks.
  EXPECT_GT(max_chunks, 1);
}

// NOLINTBEGIN(misc-include-cleaner): benchmark/benchmark.h is the public
// umbrella header across internal and OSS Google Benchmark versions.
void BM_DelinearizeInt4(benchmark::State& state) {
  const int64_t num_elements = state.range(0);
  absl::StatusOr<std::unique_ptr<PjRtClient>> client =
      GetXlaPjrtCpuClient(CpuClientOptions{});
  CHECK_OK(client);
  auto* common_client = absl::down_cast<CommonPjRtClient*>(client->get());
  absl::StatusOr<PjRtMemorySpace*> memory_space =
      (*client)->addressable_devices()[0]->default_memory_space();
  CHECK_OK(memory_space);

  Shape host_shape = ShapeUtil::MakeShape(S4, {num_elements});
  Shape device_shape = host_shape;
  device_shape.mutable_layout()->set_element_size_in_bits(4);

  const size_t packed_bytes =
      static_cast<size_t>(CeilOfRatio<int64_t>(num_elements, 2));
  std::vector<uint8_t> packed(packed_bytes, 0x78);
  Literal literal(host_shape);

  for (auto _ : state) {
    CHECK_OK(common_client->Delinearize(packed, device_shape, &literal,
                                        *memory_space));
    benchmark::DoNotOptimize(literal.untyped_data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * num_elements);
}

BENCHMARK(BM_DelinearizeInt4)->Arg(4096)->Arg(1 << 20)->Arg(64 << 20);
// NOLINTEND(misc-include-cleaner)

}  // namespace
}  // namespace xla
