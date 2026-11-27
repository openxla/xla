/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/copy_thunk.h"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

TEST(CopyThunkTest, CopyEmptyShape) {
  auto src = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto dst = LiteralUtil::CreateR2<float>({{0.0, 0.0}, {0.0, 0.0}});

  BufferAllocations allocations = CreateBufferAllocations(src, dst);
  auto [src_alloc, dst_alloc] = CreateBufferAllocation(src, dst);

  BufferAllocation::Slice src_slice =
      CreateBufferAllocationSlice(src_alloc, 0, 0);
  BufferAllocation::Slice dst_slice =
      CreateBufferAllocationSlice(src_alloc, 0, 0);

  Shape shape = ShapeUtil::MakeShape(F32, {0, 2});

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      CopyThunk::Create({"copy"}, src_slice, shape, dst_slice, shape));

  Thunk::ExecuteParams params = {nullptr, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());
}

TEST(CopyThunkTest, CopySameShape) {
  auto src = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto dst = LiteralUtil::CreateR2<float>({{0.0, 0.0}, {0.0, 0.0}});

  BufferAllocations allocations = CreateBufferAllocations(src, dst);

  auto [src_alloc, dst_alloc] = CreateBufferAllocation(src, dst);
  auto [src_slice, dst_slice] =
      CreateBufferAllocationSlice(src_alloc, dst_alloc);

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CopyThunk::Create({"copy"}, src_slice, src.shape(), dst_slice,
                                    dst.shape()));

  Thunk::ExecuteParams params = {nullptr, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_EQ(src, dst);
}

TEST(CopyThunkTest, CopyTransposed) {
  auto src = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto dst = LiteralUtil::CreateR2<float>({{0.0, 0.0}, {0.0, 0.0}});

  BufferAllocations allocations = CreateBufferAllocations(src, dst);

  auto [src_alloc, dst_alloc] = CreateBufferAllocation(src, dst);
  auto [src_slice, dst_slice] =
      CreateBufferAllocationSlice(src_alloc, dst_alloc);

  Shape transposed_shape = src.shape();
  *transposed_shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CopyThunk::Create({"copy"}, src_slice, transposed_shape,
                                    dst_slice, dst.shape()));

  Thunk::ExecuteParams params = {nullptr, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_EQ(dst, LiteralUtil::CreateR2<float>({{1.0, 3.0}, {2.0, 4.0}}));
}

TEST(CopyThunkTest, CopyTransposedEmptyShape) {
  auto src = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto dst = LiteralUtil::CreateR2<float>({{0.0, 0.0}, {0.0, 0.0}});

  BufferAllocations allocations = CreateBufferAllocations(src, dst);
  auto [src_alloc, dst_alloc] = CreateBufferAllocation(src, dst);

  BufferAllocation::Slice src_slice =
      CreateBufferAllocationSlice(src_alloc, 0, 0);
  BufferAllocation::Slice dst_slice =
      CreateBufferAllocationSlice(src_alloc, 0, 0);

  Shape shape = ShapeUtil::MakeShape(F32, {0, 2});

  Shape transposed_shape = shape;
  *transposed_shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CopyThunk::Create({"copy"}, src_slice, transposed_shape,
                                    dst_slice, shape));

  Thunk::ExecuteParams params = {nullptr, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());
}

TEST(CopyThunkTest, RejectsPackedLayoutChangeInvalidShapesAndUndersizedSlices) {
  BufferAllocation alloc(0, 128, 0);
  BufferAllocation::Slice s8(&alloc, 0, 8);
  BufferAllocation::Slice s4(&alloc, 16, 4);
  BufferAllocation::Slice s64(&alloc, 0, 64);

  // Unpacked s4[8]{0} (8 B) -> packed s4[8]{0:E(4)} (4 B) must fail cleanly
  // rather than overflowing the 4-byte destination buffer; equal packed layout
  // s4[8]{0:E(4)} -> s4[8]{0:E(4)} remains supported.
  Shape s4_unpacked = ShapeUtil::MakeShape(S4, {8});
  Shape s4_packed = s4_unpacked;
  s4_packed.mutable_layout()->set_element_size_in_bits(4);
  EXPECT_FALSE(
      CopyThunk::Create({"copy"}, s8, s4_unpacked, s4, s4_packed).ok());
  EXPECT_FALSE(
      CopyThunk::Create({"copy"}, s4, s4_packed, s8, s4_unpacked).ok());

  std::vector<uint8_t> packed_src = {0x12, 0x34, 0x56, 0x78};
  std::vector<uint8_t> packed_dst(4, 0);
  BufferAllocations packed_allocs(
      BufferAllocations::Buffers{se::DeviceAddressBase(packed_src.data(), 4),
                                 se::DeviceAddressBase(packed_dst.data(), 4)});
  BufferAllocation pa0(0, 4, 0), pa1(1, 4, 0);
  TF_ASSERT_OK_AND_ASSIGN(
      auto packed_thunk,
      CopyThunk::Create({"copy"}, BufferAllocation::Slice(&pa0, 0, 4),
                        s4_packed, BufferAllocation::Slice(&pa1, 0, 4),
                        s4_packed));
  Thunk::ExecuteParams packed_params = {nullptr, &packed_allocs};
  auto packed_ev = packed_thunk->Execute(packed_params);
  tsl::BlockUntilReady(packed_ev);
  ASSERT_FALSE(packed_ev.IsError());
  EXPECT_EQ(packed_dst, packed_src);

  // Tiled layout change, non-array shape, missing layout, and undersized slice.
  Shape f32_4x4 = ShapeUtil::MakeShape(F32, {4, 4});
  Shape f32_tiled = f32_4x4;
  f32_tiled.mutable_layout()->add_tiles()->add_dimensions(2);
  f32_tiled.mutable_layout()->mutable_tiles(0)->add_dimensions(2);
  EXPECT_FALSE(CopyThunk::Create({"copy"}, s64, f32_4x4, s64, f32_tiled).ok());

  Shape tup = ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {2})});
  EXPECT_FALSE(CopyThunk::Create({"copy"}, s8, tup, s8, tup).ok());

  Shape no_layout = ShapeUtil::MakeShape(F32, {2});
  LayoutUtil::ClearLayout(&no_layout);
  EXPECT_FALSE(CopyThunk::Create({"copy"}, s8, no_layout, s8, no_layout).ok());

  EXPECT_FALSE(CopyThunk::Create({"copy"}, s4, ShapeUtil::MakeShape(F32, {2}),
                                 s8, ShapeUtil::MakeShape(F32, {2}))
                   .ok());
}

TEST(CopyThunkTest, HandlesOverlappingParallelCopyAndOverlappingTranspose) {
  // 2 MiB + 4 KiB overlapping forward copy on an 8-thread intra-op pool
  // (previously raced across adjacent 1 MiB blocks and corrupted output).
  constexpr size_t kCopyElems = 512 * 1024;  // 2 MiB (2 parallel blocks)
  constexpr size_t kShiftElems = 128;
  std::vector<uint32_t> buf(kCopyElems + kShiftElems);
  std::iota(buf.begin(), buf.end(), uint32_t{1});

  BufferAllocations allocs(BufferAllocations::Buffers{
      se::DeviceAddressBase(buf.data(), buf.size() * sizeof(uint32_t))});
  BufferAllocation alloc(0, buf.size() * sizeof(uint32_t), 0);
  const int64_t copy_bytes = kCopyElems * sizeof(uint32_t);
  BufferAllocation::Slice src(&alloc, 0, copy_bytes);
  BufferAllocation::Slice dst(&alloc, kShiftElems * sizeof(uint32_t),
                              copy_bytes);
  Shape shape = ShapeUtil::MakeShape(U32, {static_cast<int64_t>(kCopyElems)});

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "copy_test", 8);
  Eigen::ThreadPoolDevice device(pool.AsEigenThreadPool(), pool.NumThreads());
  Thunk::ExecuteParams params = {nullptr, &allocs, nullptr, &device};

  // Self-copy is a safe no-op.
  TF_ASSERT_OK_AND_ASSIGN(auto self_thunk,
                          CopyThunk::Create({"self"}, src, shape, src, shape));
  auto self_ev = self_thunk->Execute(params);
  tsl::BlockUntilReady(self_ev);
  ASSERT_FALSE(self_ev.IsError());

  // Forward-overlapping multi-block copy preserves all elements race-free.
  TF_ASSERT_OK_AND_ASSIGN(auto ov_thunk,
                          CopyThunk::Create({"ov"}, src, shape, dst, shape));
  auto ov_ev = ov_thunk->Execute(params);
  tsl::BlockUntilReady(ov_ev);
  ASSERT_FALSE(ov_ev.IsError());
  for (size_t i = 0; i < kCopyElems; ++i) {
    ASSERT_EQ(buf[kShiftElems + i], static_cast<uint32_t>(i + 1)) << "i=" << i;
  }

  // In-place transpose stages through scratch instead of corrupting elements.
  std::vector<float> mat = {1, 2, 3,
                            4, 5, 6};  // 2x3 col-major: [[1,3,5],[2,4,6]]
  BufferAllocations mat_allocs(BufferAllocations::Buffers{
      se::DeviceAddressBase(mat.data(), mat.size() * sizeof(float))});
  BufferAllocation mat_alloc(0, mat.size() * sizeof(float), 0);
  BufferAllocation::Slice mat_slice(&mat_alloc, 0, mat.size() * sizeof(float));
  Shape col_major = ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3}, {0, 1});
  Shape row_major = ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3}, {1, 0});
  TF_ASSERT_OK_AND_ASSIGN(
      auto tr_thunk,
      CopyThunk::Create({"tr"}, mat_slice, col_major, mat_slice, row_major));
  Thunk::ExecuteParams mat_params = {nullptr, &mat_allocs};
  auto tr_ev = tr_thunk->Execute(mat_params);
  tsl::BlockUntilReady(tr_ev);
  ASSERT_FALSE(tr_ev.IsError());
  EXPECT_EQ(mat, (std::vector<float>{1, 3, 5, 2, 4, 6}));
}

}  // namespace
}  // namespace xla::cpu
