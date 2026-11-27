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

#include "xla/backends/cpu/runtime/infeed_thunk.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "tsl/platform/statusor.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/xfeed_manager.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

TEST(InfeedThunkTest, BufferAndResourceUses) {
  BufferAllocation alloc(0, 1024, 0);
  BufferAllocation::Slice infeed_slice(&alloc, 10, 40);
  Shape infeed_shape = ShapeUtil::MakeShape(F32, {10});

  InfeedThunk::InfeedBuffer infeed_buffer{infeed_slice, infeed_shape};

  auto consume_token = Resource::Create(Resource::kToken);
  auto produce_token = Resource::Create(Resource::kToken);

  TF_ASSERT_OK_AND_ASSIGN(auto thunk,
                          InfeedThunk::Create({"infeed"}, {infeed_buffer},
                                              {consume_token, produce_token}));

  EXPECT_EQ(thunk->buffer_uses().size(), 1);
  EXPECT_EQ(thunk->buffer_uses()[0],
            BufferUse::Write(infeed_slice, infeed_shape));

  EXPECT_EQ(thunk->resource_uses().size(), 2);
  EXPECT_EQ(thunk->resource_uses()[0], ResourceUse::Read(consume_token));
  EXPECT_EQ(thunk->resource_uses()[1], ResourceUse::Write(produce_token));
}

class SelfDeletingXfeedBuffer final : public XfeedBuffer {
 public:
  SelfDeletingXfeedBuffer(int32_t len, void* ptr, bool* done_ok,
                          bool* done_called)
      : len_(len), ptr_(ptr), done_ok_(done_ok), done_called_(done_called) {}
  int32_t length() override { return len_; }
  void* data() override { return ptr_; }
  void Done(absl::StatusOr<Shape> shape) override {
    *done_called_ = true;
    *done_ok_ = shape.ok();
    delete this;
  }

 private:
  int32_t len_;
  void* ptr_;
  bool* done_ok_;
  bool* done_called_;
};

TEST(InfeedThunkTest, ReleasesQueueOnMismatchNullDataAndZeroLength) {
  std::vector<int32_t> dst(2, 0);
  BufferAllocations allocations(
      BufferAllocations::Buffers{se::DeviceAddressBase(dst.data(), 8)});
  BufferAllocation alloc(0, 8, 0);
  BufferAllocation::Slice slice(&alloc, 0, 8);
  Shape shape = ShapeUtil::MakeShape(S32, {2});
  auto t0 = Resource::Create(Resource::kToken);
  auto t1 = Resource::Create(Resource::kToken);
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, InfeedThunk::Create({"infeed"}, {{slice, shape}}, {t0, t1}));

  XfeedManager xfeed;
  Thunk::ExecuteParams params = {nullptr, &allocations, &xfeed};

  // Size mismatch releases the buffer (including self-deleting Done()) with an
  // error and leaves the queue usable for the next execution.
  int32_t short_src = 7;
  bool bad_ok = true, bad_done = false;
  xfeed.infeed()->EnqueueBuffersAtomically(
      {new SelfDeletingXfeedBuffer(4, &short_src, &bad_ok, &bad_done)});
  auto bad_ev = thunk->Execute(params);
  tsl::BlockUntilReady(bad_ev);
  EXPECT_TRUE(bad_ev.IsError());
  EXPECT_TRUE(bad_done);
  EXPECT_FALSE(bad_ok);

  // Null data on non-empty buffer returns error and releases queue.
  bool null_ok = true, null_done = false;
  xfeed.infeed()->EnqueueBuffersAtomically(
      {new SelfDeletingXfeedBuffer(8, nullptr, &null_ok, &null_done)});
  auto null_ev = thunk->Execute(params);
  tsl::BlockUntilReady(null_ev);
  EXPECT_TRUE(null_ev.IsError());
  EXPECT_TRUE(null_done);
  EXPECT_FALSE(null_ok);

  // Subsequent valid infeed succeeds on the same XfeedManager.
  int32_t good_src[2] = {11, 22};
  bool good_ok = false, good_done = false;
  xfeed.infeed()->EnqueueBuffersAtomically(
      {new SelfDeletingXfeedBuffer(8, good_src, &good_ok, &good_done)});
  auto good_ev = thunk->Execute(params);
  tsl::BlockUntilReady(good_ev);
  ASSERT_FALSE(good_ev.IsError());
  EXPECT_TRUE(good_done);
  EXPECT_TRUE(good_ok);
  EXPECT_EQ(dst[0], 11);
  EXPECT_EQ(dst[1], 22);

  // Zero-byte infeed with nullptr data succeeds without calling memcpy on null.
  BufferAllocation::Slice empty_slice(&alloc, 0, 0);
  Shape empty_shape = ShapeUtil::MakeShape(S32, {0});
  TF_ASSERT_OK_AND_ASSIGN(
      auto empty_thunk,
      InfeedThunk::Create({"infeed0"}, {{empty_slice, empty_shape}}, {t0, t1}));
  bool zero_ok = false, zero_done = false;
  xfeed.infeed()->EnqueueBuffersAtomically(
      {new SelfDeletingXfeedBuffer(0, nullptr, &zero_ok, &zero_done)});
  auto zero_ev = empty_thunk->Execute(params);
  tsl::BlockUntilReady(zero_ev);
  EXPECT_FALSE(zero_ev.IsError());
  EXPECT_TRUE(zero_done);
  EXPECT_TRUE(zero_ok);
}

}  // namespace
}  // namespace xla::cpu
