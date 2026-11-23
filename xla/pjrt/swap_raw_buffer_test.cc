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

#include "xla/pjrt/swap_raw_buffer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tsl/platform/mem.h"
#include "xla/future.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/runtime/device_id.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/framework/bfc_allocator.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::NotNull;

constexpr size_t kSwapChunkSizeBytes = 8 * 1024 * 1024;

auto ChunkWithSize(size_t size) {
  return AllOf(Field("data", &SwapChunkStorage::Chunk::data, NotNull()),
               Field("size", &SwapChunkStorage::Chunk::size, size));
}

class AlignedCpuSubAllocator : public tsl::SubAllocator {
 public:
  AlignedCpuSubAllocator()
      : tsl::SubAllocator(/*alloc_visitors=*/{}, /*free_visitors=*/{}) {}

  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override {
    *bytes_received = num_bytes;
    return tsl::port::AlignedMalloc(num_bytes,
                                    static_cast<std::align_val_t>(alignment));
  }

  void Free(void* ptr, size_t /*num_bytes*/) override {
    tsl::port::AlignedFree(ptr);
  }

  bool SupportsCoalescing() const override { return false; }
};

struct BoundedHostAllocator {
  // Non-owning pointer into `host_allocator` for inspecting BFC stats.
  tsl::BFCAllocator* bfc;
  // Real `BasicHostMemoryAllocator` owning `bfc`.
  std::unique_ptr<BasicHostMemoryAllocator> host_allocator;
};

absl::StatusOr<BoundedHostAllocator> MakeBoundedHostAllocator(
    size_t total_memory_bytes) {
  auto sub_allocator = std::make_unique<AlignedCpuSubAllocator>();
  tsl::BFCAllocator::Options opts = {
      .allow_growth = false,
      .allow_retry_on_failure = false,
      .garbage_collection = false,
      .fragmentation_fraction = 0,
      .enable_spatial_partitioning = false,
  };
  auto bfc = std::make_unique<tsl::BFCAllocator>(
      std::move(sub_allocator), total_memory_bytes, "swap_test_bfc", opts);
  tsl::BFCAllocator* bfc_ptr = bfc.get();
  return BoundedHostAllocator{
      .bfc = bfc_ptr,
      .host_allocator =
          std::make_unique<BasicHostMemoryAllocator>(std::move(bfc)),
  };
}

TEST(SwapRawBufferTest, SwapDeviceEventPromiseForwardsSwapEvents) {
  auto [ready_promise, ready_event] = CreateSwapLinkedEventPromise();
  EXPECT_FALSE(ready_event.async_value()->IsAvailable());
  ready_promise.SetReady();
  EXPECT_TRUE(ready_event.async_value()->IsConcrete());
  EXPECT_TRUE(static_cast<bool>(ready_event.down_cast<SwapEvent>()));

  auto [err_promise, err_event] = CreateSwapLinkedEventPromise();
  err_promise.SetError(absl::InternalError("swap promise error"));
  ASSERT_TRUE(err_event.async_value()->IsError());
  EXPECT_THAT(err_event.async_value()->GetError(),
              StatusIs(absl::StatusCode::kInternal));

  auto [ok_future_promise, ok_future] = MakePromise();
  PjRtDeviceEventRef forwarded_src = ToSwapEvent(std::move(ok_future));
  auto [fwd_promise, fwd_event] = CreateSwapLinkedEventPromise();
  fwd_promise.Set(std::move(forwarded_src));
  EXPECT_FALSE(fwd_event.async_value()->IsAvailable());
  ok_future_promise.Set();
  tsl::BlockUntilReady(fwd_event.async_value());
  EXPECT_TRUE(fwd_event.async_value()->IsConcrete());

  auto [fail_future_promise, fail_future] = MakePromise();
  PjRtDeviceEventRef failed_event = ToSwapEvent(std::move(fail_future));
  fail_future_promise.Set(absl::UnavailableError("future failed"));
  tsl::BlockUntilReady(failed_event.async_value());
  ASSERT_TRUE(failed_event.async_value()->IsError());
  EXPECT_THAT(failed_event.async_value()->GetError(),
              StatusIs(absl::StatusCode::kUnavailable));
}

TEST(SwapRawBufferTest, RejectsInvalidAllocationInputs) {
  EXPECT_THAT(SwapChunkStorage::Allocate(/*total_bytes=*/1024,
                                         /*allocator=*/nullptr, /*numa_node=*/0,
                                         LocalDeviceId(0)),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      SwapRawBuffer::Allocate(/*memory_space=*/nullptr,
                              /*on_device_bytes_count=*/1024,
                              /*allocator=*/nullptr, /*numa_node=*/0,
                              LocalDeviceId(0), /*async_work_runner=*/nullptr),
      StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(SwapRawBufferTest, ChunkStorageAllocatesFixed8MiBChunksAndExactRemainder) {
  ASSERT_OK_AND_ASSIGN(BoundedHostAllocator alloc,
                       MakeBoundedHostAllocator(64 * 1024 * 1024));

  ASSERT_OK_AND_ASSIGN(
      SwapChunkStorage empty,
      SwapChunkStorage::Allocate(/*total_bytes=*/0, alloc.host_allocator.get(),
                                 /*numa_node=*/0, LocalDeviceId(0)));
  EXPECT_EQ(empty.total_bytes(), 0);
  EXPECT_THAT(empty.chunks(), IsEmpty());
  EXPECT_EQ(alloc.bfc->GetStats()->bytes_in_use, 0);

  const size_t exact_multiple = 2 * kSwapChunkSizeBytes;
  {
    ASSERT_OK_AND_ASSIGN(
        SwapChunkStorage exact,
        SwapChunkStorage::Allocate(exact_multiple, alloc.host_allocator.get(),
                                   /*numa_node=*/0, LocalDeviceId(0)));
    EXPECT_EQ(exact.total_bytes(), exact_multiple);
    EXPECT_THAT(exact.chunks(),
                ElementsAre(ChunkWithSize(kSwapChunkSizeBytes),
                            ChunkWithSize(kSwapChunkSizeBytes)));
    EXPECT_GT(alloc.bfc->GetStats()->bytes_in_use, 0);
  }
  EXPECT_EQ(alloc.bfc->GetStats()->bytes_in_use, 0);

  const size_t remainder_bytes = 12345;
  const size_t non_multiple = 2 * kSwapChunkSizeBytes + remainder_bytes;
  {
    ASSERT_OK_AND_ASSIGN(
        SwapChunkStorage chunked,
        SwapChunkStorage::Allocate(non_multiple, alloc.host_allocator.get(),
                                   /*numa_node=*/0, LocalDeviceId(0)));
    EXPECT_EQ(chunked.total_bytes(), non_multiple);
    EXPECT_THAT(chunked.chunks(),
                ElementsAre(ChunkWithSize(kSwapChunkSizeBytes),
                            ChunkWithSize(kSwapChunkSizeBytes),
                            ChunkWithSize(remainder_bytes)));
  }
  EXPECT_EQ(alloc.bfc->GetStats()->bytes_in_use, 0);
}

TEST(SwapRawBufferTest,
     ChunkStorageOomReleasesEarlierChunksAndReturnsResourceExhausted) {
  ASSERT_OK_AND_ASSIGN(
      BoundedHostAllocator alloc,
      MakeBoundedHostAllocator(2 * kSwapChunkSizeBytes + 2048));
  EXPECT_THAT(SwapChunkStorage::Allocate(2 * kSwapChunkSizeBytes + 4096,
                                         alloc.host_allocator.get(),
                                         /*numa_node=*/0, LocalDeviceId(0)),
              StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_EQ(alloc.bfc->GetStats()->bytes_in_use, 0);
}

TEST(SwapRawBufferTest, CopyRawAndSliceSpanMultipleChunks) {
  ASSERT_OK_AND_ASSIGN(BoundedHostAllocator alloc,
                       MakeBoundedHostAllocator(64 * 1024 * 1024));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> pjrt_client,
                       GetXlaPjrtCpuClient(CpuClientOptions{}));
  auto* common_client = absl::down_cast<CommonPjRtClient*>(pjrt_client.get());
  SwapMemorySpace swap_space(/*id=*/100,
                             common_client->addressable_devices()[0]);

  const size_t total_bytes = 2 * kSwapChunkSizeBytes + 777;
  ASSERT_OK_AND_ASSIGN(
      tsl::RCReference<SwapRawBuffer> buffer,
      SwapRawBuffer::Allocate(swap_space.ToCApiPtr(), total_bytes,
                              alloc.host_allocator.get(), /*numa_node=*/0,
                              LocalDeviceId(0),
                              common_client->async_work_runner()));
  EXPECT_EQ(buffer->GetHostPointer(), nullptr);
  EXPECT_EQ(buffer->OpaqueDeviceMemoryDataPointer(), nullptr);
  EXPECT_THAT(
      buffer->storage().chunks(),
      ElementsAre(ChunkWithSize(kSwapChunkSizeBytes),
                  ChunkWithSize(kSwapChunkSizeBytes), ChunkWithSize(777)));

  std::vector<uint8_t> expected(total_bytes);
  std::iota(expected.begin(), expected.end(), uint8_t{0});
  ASSERT_OK_AND_ASSIGN(
      PjRtDeviceEventRef h2d_event,
      buffer->CopyRawHostToDeviceAndReturnEvent(
          expected.data(), /*offset=*/0, total_bytes, /*dependencies=*/{}));
  tsl::BlockUntilReady(h2d_event.async_value());
  EXPECT_FALSE(h2d_event.async_value()->IsError());

  EXPECT_THAT(buffer->Slice(-1, 16),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(buffer->Slice(0, total_bytes + 1),
              StatusIs(absl::StatusCode::kInvalidArgument));

  const int64_t slice_offset = kSwapChunkSizeBytes - 64;
  const int64_t slice_size = 256;
  ASSERT_OK_AND_ASSIGN(PjRtRawBufferRef sliced,
                       buffer->Slice(slice_offset, slice_size));
  std::vector<uint8_t> patch(slice_size, 211);
  ASSERT_OK_AND_ASSIGN(PjRtDeviceEventRef patch_event,
                       sliced->CopyRawHostToDeviceAndReturnEvent(
                           patch.data(), /*offset=*/0, slice_size));
  tsl::BlockUntilReady(patch_event.async_value());
  absl::c_copy(patch, expected.begin() + slice_offset);

  std::vector<uint8_t> actual(total_bytes, 0);
  ASSERT_OK_AND_ASSIGN(
      PjRtDeviceEventRef d2h_event,
      buffer->CopyRawDeviceToHostAndReturnEvent(
          actual.data(), /*offset=*/0, total_bytes, /*dependencies=*/{}));
  tsl::BlockUntilReady(d2h_event.async_value());
  EXPECT_EQ(actual, expected);
}

TEST(SwapRawBufferTest, ScheduleCopyToCopiesBetweenSwapBuffers) {
  ASSERT_OK_AND_ASSIGN(BoundedHostAllocator alloc,
                       MakeBoundedHostAllocator(64 * 1024 * 1024));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> pjrt_client,
                       GetXlaPjrtCpuClient(CpuClientOptions{}));
  auto* common_client = absl::down_cast<CommonPjRtClient*>(pjrt_client.get());
  SwapMemorySpace swap_space(/*id=*/100,
                             common_client->addressable_devices()[0]);

  const size_t total_bytes = kSwapChunkSizeBytes + 1024;
  ASSERT_OK_AND_ASSIGN(
      tsl::RCReference<SwapRawBuffer> src,
      SwapRawBuffer::Allocate(swap_space.ToCApiPtr(), total_bytes,
                              alloc.host_allocator.get(), /*numa_node=*/0,
                              LocalDeviceId(0),
                              common_client->async_work_runner()));
  ASSERT_OK_AND_ASSIGN(
      tsl::RCReference<SwapRawBuffer> dst,
      SwapRawBuffer::Allocate(swap_space.ToCApiPtr(), total_bytes,
                              alloc.host_allocator.get(), /*numa_node=*/0,
                              LocalDeviceId(0),
                              common_client->async_work_runner()));

  std::vector<uint8_t> expected(total_bytes);
  std::iota(expected.begin(), expected.end(), uint8_t{3});
  ASSERT_OK_AND_ASSIGN(
      PjRtDeviceEventRef h2d_event,
      src->CopyRawHostToDeviceAndReturnEvent(expected.data(), /*offset=*/0,
                                             total_bytes, /*dependencies=*/{}));

  ASSERT_OK_AND_ASSIGN(auto copy_def, common_client->CreateLinkedEventPromise(
                                          &swap_space, "def"));
  ASSERT_OK_AND_ASSIGN(auto copy_use, common_client->CreateLinkedEventPromise(
                                          &swap_space, "use"));
  src->ScheduleCopyTo({std::move(h2d_event)}, dst, std::move(copy_def.first),
                      std::move(copy_use.first), /*allocation_event=*/{});
  tsl::BlockUntilReady(copy_def.second.async_value());
  tsl::BlockUntilReady(copy_use.second.async_value());
  EXPECT_FALSE(copy_def.second.async_value()->IsError());

  std::vector<uint8_t> actual(total_bytes, 0);
  ASSERT_OK_AND_ASSIGN(
      PjRtDeviceEventRef d2h_event,
      dst->CopyRawDeviceToHostAndReturnEvent(actual.data(), /*offset=*/0,
                                             total_bytes, /*dependencies=*/{}));
  tsl::BlockUntilReady(d2h_event.async_value());
  EXPECT_EQ(actual, expected);
}

TEST(SwapRawBufferTest, ScheduleCopyFromCopiesBetweenSwapBuffers) {
  ASSERT_OK_AND_ASSIGN(BoundedHostAllocator alloc,
                       MakeBoundedHostAllocator(16 * 1024 * 1024));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> pjrt_client,
                       GetXlaPjrtCpuClient(CpuClientOptions{}));
  auto* common_client = absl::down_cast<CommonPjRtClient*>(pjrt_client.get());
  SwapMemorySpace swap_space(/*id=*/100,
                             common_client->addressable_devices()[0]);

  const size_t total_bytes = 4096;
  ASSERT_OK_AND_ASSIGN(
      tsl::RCReference<SwapRawBuffer> src,
      SwapRawBuffer::Allocate(swap_space.ToCApiPtr(), total_bytes,
                              alloc.host_allocator.get(), /*numa_node=*/0,
                              LocalDeviceId(0),
                              common_client->async_work_runner()));
  ASSERT_OK_AND_ASSIGN(
      tsl::RCReference<SwapRawBuffer> dst,
      SwapRawBuffer::Allocate(swap_space.ToCApiPtr(), total_bytes,
                              alloc.host_allocator.get(), /*numa_node=*/0,
                              LocalDeviceId(0),
                              common_client->async_work_runner()));

  std::vector<uint8_t> expected(total_bytes, 57);
  ASSERT_OK_AND_ASSIGN(
      PjRtDeviceEventRef h2d_event,
      src->CopyRawHostToDeviceAndReturnEvent(expected.data(), /*offset=*/0,
                                             total_bytes, /*dependencies=*/{}));

  using EventPromisePair =
      std::pair<PjRtDeviceEventPromiseRef, PjRtDeviceEventRef>;
  ASSERT_OK_AND_ASSIGN(
      EventPromisePair copy_def,
      common_client->CreateLinkedEventPromise(&swap_space, "def"));
  ASSERT_OK_AND_ASSIGN(
      EventPromisePair copy_use,
      common_client->CreateLinkedEventPromise(&swap_space, "use"));
  dst->ScheduleCopyFrom({std::move(h2d_event)}, src, std::move(copy_def.first),
                        std::move(copy_use.first), /*allocation_event=*/{});
  tsl::BlockUntilReady(copy_def.second.async_value());
  tsl::BlockUntilReady(copy_use.second.async_value());
  EXPECT_FALSE(copy_def.second.async_value()->IsError());

  std::vector<uint8_t> actual(total_bytes, 0);
  ASSERT_OK_AND_ASSIGN(
      PjRtDeviceEventRef d2h_event,
      dst->CopyRawDeviceToHostAndReturnEvent(actual.data(), /*offset=*/0,
                                             total_bytes, /*dependencies=*/{}));
  tsl::BlockUntilReady(d2h_event.async_value());
  EXPECT_EQ(actual, expected);
}

TEST(SwapRawBufferTest, ScheduleCopyRejectsSizeMismatch) {
  ASSERT_OK_AND_ASSIGN(BoundedHostAllocator alloc,
                       MakeBoundedHostAllocator(16 * 1024 * 1024));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> pjrt_client,
                       GetXlaPjrtCpuClient(CpuClientOptions{}));
  auto* common_client = absl::down_cast<CommonPjRtClient*>(pjrt_client.get());
  SwapMemorySpace swap_space(/*id=*/100,
                             common_client->addressable_devices()[0]);

  ASSERT_OK_AND_ASSIGN(
      tsl::RCReference<SwapRawBuffer> small_buf,
      SwapRawBuffer::Allocate(
          swap_space.ToCApiPtr(), /*on_device_bytes_count=*/64,
          alloc.host_allocator.get(), /*numa_node=*/0, LocalDeviceId(0),
          common_client->async_work_runner()));
  ASSERT_OK_AND_ASSIGN(
      tsl::RCReference<SwapRawBuffer> large_buf,
      SwapRawBuffer::Allocate(swap_space.ToCApiPtr(),
                              /*on_device_bytes_count=*/128,
                              alloc.host_allocator.get(), /*numa_node=*/0,
                              LocalDeviceId(0),
                              common_client->async_work_runner()));

  auto [to_def_promise, to_def_event] = CreateSwapLinkedEventPromise();
  auto [to_use_promise, to_use_event] = CreateSwapLinkedEventPromise();
  absl::Status to_alloc_status = absl::OkStatus();
  small_buf->ScheduleCopyTo(
      /*transfer_dependency_events=*/{}, large_buf, std::move(to_def_promise),
      std::move(to_use_promise),
      [&](absl::Status status) { to_alloc_status = std::move(status); });
  tsl::BlockUntilReady(to_def_event.async_value());
  tsl::BlockUntilReady(to_use_event.async_value());
  EXPECT_THAT(to_alloc_status, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(to_def_event.async_value()->GetError(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  auto [from_def_promise, from_def_event] = CreateSwapLinkedEventPromise();
  auto [from_use_promise, from_use_event] = CreateSwapLinkedEventPromise();
  absl::Status from_alloc_status = absl::OkStatus();
  small_buf->ScheduleCopyFrom(
      /*transfer_dependency_events=*/{}, large_buf, std::move(from_def_promise),
      std::move(from_use_promise),
      [&](absl::Status status) { from_alloc_status = std::move(status); });
  tsl::BlockUntilReady(from_def_event.async_value());
  tsl::BlockUntilReady(from_use_event.async_value());
  EXPECT_THAT(from_alloc_status, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(from_def_event.async_value()->GetError(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SwapRawBufferTest, ScheduleCopyFromCopiesIntoSwapAndPropagatesError) {
  ASSERT_OK_AND_ASSIGN(BoundedHostAllocator alloc,
                       MakeBoundedHostAllocator(16 * 1024 * 1024));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> pjrt_client,
                       GetXlaPjrtCpuClient(CpuClientOptions{}));
  auto* common_client = absl::down_cast<CommonPjRtClient*>(pjrt_client.get());
  PjRtDevice* device = common_client->addressable_devices()[0];
  ASSERT_OK_AND_ASSIGN(PjRtMemorySpace * host_space,
                       device->default_memory_space());
  SwapMemorySpace swap_space(/*id=*/100, device);

  const size_t total_bytes = 4096;
  ASSERT_OK_AND_ASSIGN(PjRtRawBufferRef src,
                       common_client->AllocateRawBuffer(host_space, total_bytes,
                                                        /*retry_on_oom=*/false,
                                                        /*allocate_after=*/{}));
  std::vector<uint8_t> expected(total_bytes, 91);
  ASSERT_OK_AND_ASSIGN(
      PjRtDeviceEventRef h2d_event,
      src->CopyRawHostToDeviceAndReturnEvent(expected.data(), /*offset=*/0,
                                             total_bytes, /*dependencies=*/{}));

  ASSERT_OK_AND_ASSIGN(
      tsl::RCReference<SwapRawBuffer> dst,
      SwapRawBuffer::Allocate(swap_space.ToCApiPtr(), total_bytes,
                              alloc.host_allocator.get(), /*numa_node=*/0,
                              LocalDeviceId(0),
                              common_client->async_work_runner()));
  ASSERT_OK_AND_ASSIGN(auto copy_def, common_client->CreateLinkedEventPromise(
                                          &swap_space, "def"));
  ASSERT_OK_AND_ASSIGN(auto copy_use, common_client->CreateLinkedEventPromise(
                                          host_space, "use"));
  src->ScheduleCopyTo({std::move(h2d_event)}, dst, std::move(copy_def.first),
                      std::move(copy_use.first), /*allocation_event=*/{});
  tsl::BlockUntilReady(copy_def.second.async_value());
  tsl::BlockUntilReady(copy_use.second.async_value());
  EXPECT_FALSE(copy_def.second.async_value()->IsError());

  std::vector<uint8_t> actual(total_bytes, 0);
  ASSERT_OK_AND_ASSIGN(
      PjRtDeviceEventRef d2h_event,
      dst->CopyRawDeviceToHostAndReturnEvent(actual.data(), /*offset=*/0,
                                             total_bytes, /*dependencies=*/{}));
  tsl::BlockUntilReady(d2h_event.async_value());
  EXPECT_EQ(actual, expected);

  ASSERT_OK_AND_ASSIGN(
      auto err_dep, common_client->CreateLinkedEventPromise(host_space, "err"));
  ASSERT_OK_AND_ASSIGN(auto err_def, common_client->CreateLinkedEventPromise(
                                         &swap_space, "err_def"));
  ASSERT_OK_AND_ASSIGN(auto err_use, common_client->CreateLinkedEventPromise(
                                         host_space, "err_use"));
  err_dep.first.SetError(absl::AbortedError("source failed"));
  src->ScheduleCopyTo({std::move(err_dep.second)}, dst,
                      std::move(err_def.first), std::move(err_use.first),
                      /*allocation_event=*/{});
  tsl::BlockUntilReady(err_def.second.async_value());
  tsl::BlockUntilReady(err_use.second.async_value());
  EXPECT_THAT(err_def.second.async_value()->GetError(),
              StatusIs(absl::StatusCode::kAborted));
}

}  // namespace
}  // namespace xla
