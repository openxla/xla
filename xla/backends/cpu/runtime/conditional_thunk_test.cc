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

#include "xla/backends/cpu/runtime/conditional_thunk.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

TEST(ConditionalThunkTest, BufferUses) {
  BufferAllocation alloc(0, 1024, 0);
  Shape branch_index_slice_shape = ShapeUtil::MakeShape(S32, {1});
  BufferAllocation::Slice branch_index_slice(&alloc, 0, sizeof(int32_t));
  Shape read_slice_shape = ShapeUtil::MakeShape(F32, {4});
  BufferAllocation::Slice read_slice(&alloc, 10, 12);

  std::vector<ThunkSequence> branch_sequences(1);
  branch_sequences[0].push_back(std::make_unique<BufferUseThunk>(
      BufferUse::Read(read_slice, read_slice_shape)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, ConditionalThunk::Create({"conditional"}, branch_index_slice,
                                           std::move(branch_sequences)));

  EXPECT_EQ(thunk->buffer_uses().size(), 2);
  EXPECT_EQ(thunk->buffer_uses()[0],
            BufferUse::Read(branch_index_slice, branch_index_slice_shape));
  EXPECT_EQ(thunk->buffer_uses()[1],
            BufferUse::Read(read_slice, read_slice_shape));
}

TEST(ConditionalThunkTest, ResourceUses) {
  BufferAllocation alloc(0, 1024, 0);
  BufferAllocation::Slice branch_index_slice(&alloc, 0, sizeof(int32_t));

  auto token = Resource::Create(Resource::kToken);

  std::vector<ThunkSequence> branch_sequences(1);
  branch_sequences[0].push_back(
      std::make_unique<ResourceUseThunk>(ResourceUse::Read(token)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, ConditionalThunk::Create({"conditional"}, branch_index_slice,
                                           std::move(branch_sequences)));

  EXPECT_EQ(thunk->resource_uses().size(), 1);
  EXPECT_EQ(thunk->resource_uses()[0], ResourceUse::Read(token));
}

class RecordBranchThunk final : public Thunk {
 public:
  RecordBranchThunk(int id, int* out)
      : Thunk(Kind::kKernel, {"rec"}), id_(id), out_(out) {}
  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams&) final {
    *out_ = id_;
    return OkExecuteEvent();
  }
  BufferUses buffer_uses() const final { return {}; }

 private:
  int id_;
  int* out_;
};

TEST(ConditionalThunkTest,
     ValidatesArityAndHandlesNonCanonicalPredAndMisalignedS32) {
  BufferAllocation alloc(0, 16, 0);
  BufferAllocation::Slice pred_slice(&alloc, 0, sizeof(uint8_t));
  BufferAllocation::Slice misaligned_s32_slice(&alloc, 1, sizeof(int32_t));

  // Reject empty branch list and PRED with != 2 branches.
  EXPECT_FALSE(ConditionalThunk::Create({"c"}, pred_slice, {}).ok());
  EXPECT_FALSE(ConditionalThunk::Create({"c"}, misaligned_s32_slice, {}).ok());
  EXPECT_FALSE(
      ConditionalThunk::Create({"c"}, pred_slice, std::vector<ThunkSequence>(1))
          .ok());
  EXPECT_FALSE(
      ConditionalThunk::Create({"c"}, pred_slice, std::vector<ThunkSequence>(3))
          .ok());

  // Non-canonical non-zero PRED bytes (0x02, 0x80, 0xFF) execute branch 0
  // without invalid-bool UB; 0x00 executes branch 1.
  alignas(8) uint8_t raw[16] = {};
  BufferAllocations allocations(
      BufferAllocations::Buffers{se::DeviceAddressBase(raw, sizeof(raw))});
  Thunk::ExecuteParams params = {nullptr, &allocations};

  int taken = -1;
  std::vector<ThunkSequence> pred_branches(2);
  pred_branches[0].push_back(std::make_unique<RecordBranchThunk>(0, &taken));
  pred_branches[1].push_back(std::make_unique<RecordBranchThunk>(1, &taken));
  TF_ASSERT_OK_AND_ASSIGN(auto pred_thunk,
                          ConditionalThunk::Create({"pred_c"}, pred_slice,
                                                   std::move(pred_branches)));
  for (uint8_t b :
       {uint8_t{0x01}, uint8_t{0x02}, uint8_t{0x80}, uint8_t{0xFF}}) {
    raw[0] = b;
    taken = -1;
    auto ev = pred_thunk->Execute(params);
    tsl::BlockUntilReady(ev);
    ASSERT_FALSE(ev.IsError());
    EXPECT_EQ(taken, 0) << "byte=" << static_cast<int>(b);
  }
  raw[0] = 0x00;
  taken = -1;
  auto false_ev = pred_thunk->Execute(params);
  tsl::BlockUntilReady(false_ev);
  ASSERT_FALSE(false_ev.IsError());
  EXPECT_EQ(taken, 1);

  // Misaligned S32 index loads cleanly and clamps out-of-range values to last.
  std::vector<ThunkSequence> s32_branches(3);
  for (int b = 0; b < 3; ++b) {
    s32_branches[b].push_back(std::make_unique<RecordBranchThunk>(b, &taken));
  }
  TF_ASSERT_OK_AND_ASSIGN(
      auto s32_thunk, ConditionalThunk::Create({"s32_c"}, misaligned_s32_slice,
                                               std::move(s32_branches)));
  for (auto [idx, expected] :
       std::vector<std::pair<int32_t, int>>{{-5, 2}, {0, 0}, {1, 1}, {99, 2}}) {
    std::memcpy(raw + 1, &idx, sizeof(idx));
    taken = -1;
    auto ev = s32_thunk->Execute(params);
    tsl::BlockUntilReady(ev);
    ASSERT_FALSE(ev.IsError());
    EXPECT_EQ(taken, expected);
  }
}

}  // namespace
}  // namespace xla::cpu
