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

#include "xla/backends/cpu/runtime/topk_thunk.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "absl/base/casts.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/topk_lib.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

using ::testing::ElementsAre;

TEST(TopKThunkTest, RejectsInvalidDimensionsOverflowAndUndersizedSlices) {
  BufferAllocation alloc(/*index=*/0, /*size=*/64, /*color=*/0);
  BufferAllocation::Slice s16(&alloc, 0, 16);
  BufferAllocation::Slice s8(&alloc, 16, 8);
  BufferAllocation::Slice s4(&alloc, 24, 4);
  BufferAllocation::Slice i8(&alloc, 32, 8);

  // k > input_size (previously caused heap-buffer-overflow).
  EXPECT_FALSE(TopKThunk::Create({"topk"}, s4, s8, i8, /*batch_size=*/1,
                                 /*input_size=*/1, /*k=*/2)
                   .ok());
  // Negative dimensions.
  EXPECT_FALSE(TopKThunk::Create({"topk"}, s16, s8, i8, -1, 2, 1).ok());
  EXPECT_FALSE(TopKThunk::Create({"topk"}, s16, s8, i8, 1, -2, 1).ok());
  EXPECT_FALSE(TopKThunk::Create({"topk"}, s16, s8, i8, 1, 2, -1).ok());
  // input_size exceeding int32 index range and int64 byte size overflow.
  const int64_t too_wide =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  BufferAllocation huge(/*index=*/1,
                        /*size=*/std::numeric_limits<int64_t>::max(),
                        /*color=*/0);
  BufferAllocation::Slice shuge(&huge, 0, std::numeric_limits<int64_t>::max());
  EXPECT_FALSE(TopKThunk::Create({"topk"}, shuge, s4, s4, 1, too_wide, 1).ok());
  EXPECT_FALSE(TopKThunk::Create({"topk"}, shuge, shuge, shuge,
                                 int64_t{1} << 33, int64_t{1} << 30, 1)
                   .ok());
  // Undersized input, output, and index slices.
  EXPECT_FALSE(TopKThunk::Create({"topk"}, s4, s8, i8, 1, 4, 2).ok());
  EXPECT_FALSE(TopKThunk::Create({"topk"}, s16, s4, i8, 1, 4, 2).ok());
  EXPECT_FALSE(TopKThunk::Create({"topk"}, s16, s8, s4, 1, 4, 2).ok());
}

TEST(TopKThunkTest, ZeroBatchAndZeroKAreNoOps) {
  std::vector<float> buf(4, 7.0f);
  std::vector<int32_t> idx(4, 9);
  BufferAllocations allocations(
      BufferAllocations::Buffers{se::DeviceAddressBase(buf.data(), 16),
                                 se::DeviceAddressBase(idx.data(), 16)});
  BufferAllocation a0(0, 16, 0);
  BufferAllocation a1(1, 16, 0);
  BufferAllocation::Slice empty0(&a0, 0, 0);
  BufferAllocation::Slice empty1(&a1, 0, 0);
  BufferAllocation::Slice vals(&a0, 0, 16);

  Thunk::ExecuteParams params = {nullptr, &allocations};
  TF_ASSERT_OK_AND_ASSIGN(
      auto zero_batch,
      TopKThunk::Create({"topk"}, empty0, empty0, empty1, 0, 4, 2));
  auto ev0 = zero_batch->Execute(params);
  tsl::BlockUntilReady(ev0);
  EXPECT_FALSE(ev0.IsError());

  TF_ASSERT_OK_AND_ASSIGN(
      auto zero_k, TopKThunk::Create({"topk"}, vals, empty0, empty1, 2, 2, 0));
  auto ev1 = zero_k->Execute(params);
  tsl::BlockUntilReady(ev1);
  EXPECT_FALSE(ev1.IsError());

  TF_ASSERT_OK_AND_ASSIGN(
      auto zero_all,
      TopKThunk::Create({"topk"}, empty0, empty0, empty1, 2, 0, 0));
  auto ev2 = zero_all->Execute(params);
  tsl::BlockUntilReady(ev2);
  EXPECT_FALSE(ev2.IsError());
}

TEST(TopKThunkTest, HandlesOverlappingInputAndRejectsOverlappingOutputs) {
  // Batch 0: [1, 5, 3] -> top-2 [5, 3] idx [1, 2]
  // Batch 1: [9, 2, 8] -> top-2 [9, 8] idx [0, 2]
  // When out_values aliases values + 1 (forward overlap), writing batch 0's
  // output [5, 3] used to clobber values[2] and batch 1's values[3] before
  // batch 1 ran.
  std::vector<float> shared = {0.0f, 1.0f, 5.0f, 3.0f, 9.0f, 2.0f, 8.0f};
  std::vector<int32_t> indices(4, -1);
  BufferAllocations allocations(BufferAllocations::Buffers{
      se::DeviceAddressBase(shared.data(), shared.size() * sizeof(float)),
      se::DeviceAddressBase(indices.data(), indices.size() * sizeof(int32_t))});
  BufferAllocation a0(0, shared.size() * sizeof(float), 0);
  BufferAllocation a1(1, indices.size() * sizeof(int32_t), 0);
  BufferAllocation::Slice val_slice(&a0, sizeof(float), 6 * sizeof(float));
  BufferAllocation::Slice out_slice(&a0, 2 * sizeof(float), 4 * sizeof(float));
  BufferAllocation::Slice idx_slice(&a1, 0, 4 * sizeof(int32_t));

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      TopKThunk::Create({"topk"}, val_slice, out_slice, idx_slice,
                        /*batch_size=*/2, /*input_size=*/3, /*k=*/2));
  Thunk::ExecuteParams params = {nullptr, &allocations};
  auto ev = thunk->Execute(params);
  tsl::BlockUntilReady(ev);
  ASSERT_FALSE(ev.IsError());

  EXPECT_THAT(std::vector<float>(shared.begin() + 2, shared.begin() + 6),
              ElementsAre(5.0f, 3.0f, 9.0f, 8.0f));
  EXPECT_THAT(indices, ElementsAre(1, 2, 0, 2));

  // Direct internal::TopK also snapshots when out_indices overlaps values.
  alignas(float) uint8_t raw[7 * sizeof(float)];
  const float init_vals[6] = {1.0f, 5.0f, 3.0f, 9.0f, 2.0f, 8.0f};
  std::memcpy(raw, init_vals, sizeof(init_vals));
  float direct_out[4] = {};
  internal::TopK<float>(2, 3, 2, reinterpret_cast<const float*>(raw),
                        direct_out,
                        reinterpret_cast<int32_t*>(raw + sizeof(float)));
  EXPECT_THAT(direct_out, ElementsAre(5.0f, 3.0f, 9.0f, 8.0f));
  int32_t direct_idx[4] = {};
  std::memcpy(direct_idx, raw + sizeof(float), sizeof(direct_idx));
  EXPECT_THAT(direct_idx, ElementsAre(1, 2, 0, 2));

  // Overlapping output and indices slices must be rejected.
  TF_ASSERT_OK_AND_ASSIGN(
      auto bad,
      TopKThunk::Create({"topk"}, val_slice, out_slice, out_slice, 2, 3, 2));
  auto bad_ev = bad->Execute(params);
  tsl::BlockUntilReady(bad_ev);
  EXPECT_TRUE(bad_ev.IsError());
}

TEST(TopKThunkTest, PreservesNanPayloadBits) {
  const uint32_t pos_nan_bits = 0x7fc12345u;
  std::vector<float> vals = {1.0f, absl::bit_cast<float>(pos_nan_bits), -0.0f,
                             +0.0f};
  std::vector<float> out(3, 0.0f);
  std::vector<int32_t> idx(3, -1);
  internal::TopK<float>(1, 4, 3, vals.data(), out.data(), idx.data());
  EXPECT_EQ(absl::bit_cast<uint32_t>(out[0]), pos_nan_bits);
  EXPECT_EQ(out[1], 1.0f);
  EXPECT_EQ(absl::bit_cast<uint32_t>(out[2]), 0x00000000u);
  EXPECT_THAT(idx, ElementsAre(1, 0, 3));
}

}  // namespace
}  // namespace xla::cpu
