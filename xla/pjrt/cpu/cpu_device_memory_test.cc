/* Copyright 2022, 2026 The OpenXLA Authors.

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

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/cpu/abstract_cpu_buffer.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/pjrt/cpu/raw_buffer.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::tsl::MakeConstructedAsyncValueRef;
using ::tsl::thread::ThreadPool;

TEST(CpuDeviceMemoryTest, Basic) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetPjRtCpuClient(CpuClientOptions()));
  PjRtMemorySpace* memory_space = client->memory_spaces()[0];
  std::string expected = "tracked_cpu_device_buffer_test";
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer, CpuRawBuffer::Allocate(memory_space, expected.size()));

  auto definition_event = MakeConstructedAsyncValueRef<CpuEvent>();

  ThreadPool thread_pool(tsl::Env::Default(), "tracked_buffer_test",
                         /*num_threads=*/4);

  thread_pool.Schedule([&]() {
    std::memcpy(buffer->buffer()->untyped_data(), expected.data(),
                expected.size());
    definition_event.SetStateConcrete();
  });

  BlockUntilReady(definition_event);

  auto result = buffer->buffer();
  ASSERT_TRUE(result.IsAvailable());
  EXPECT_EQ(std::string(static_cast<const char*>(result->untyped_data()),
                        result->size_bytes()),
            expected);
}

TEST(CpuDeviceMemoryTest, BasicError) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetPjRtCpuClient(CpuClientOptions()));
  PjRtMemorySpace* memory_space = client->memory_spaces()[0];
  TF_ASSERT_OK_AND_ASSIGN(auto buffer,
                          CpuRawBuffer::Allocate(memory_space, 64));

  auto definition_event = MakeConstructedAsyncValueRef<CpuEvent>();

  ThreadPool thread_pool(tsl::Env::Default(), "tracked_buffer_test",
                         /*num_threads=*/4);

  thread_pool.Schedule([&]() {
    definition_event.SetError(
        Internal("tracked_cpu_device_buffer_test error."));
  });

  absl::InlinedVector<PjRtDeviceEventRef, 2> definition_events;
  definition_events.push_back(PjRtDeviceEventRef(definition_event));

  BlockUntilReady(definition_event);

  ASSERT_TRUE(definition_event.IsError());
  EXPECT_EQ(definition_event.GetError().message(),
            "tracked_cpu_device_buffer_test error.");
}

TEST(CpuDeviceMemoryTest, DelayedAllocation) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetPjRtCpuClient(CpuClientOptions()));
  PjRtMemorySpace* memory_space = client->memory_spaces()[0];
  std::string expected = "tracked_cpu_device_buffer_test";

  auto buffer = CpuDeviceMemory::CreateDelayedMemory();
  auto malloc_event = MakeConstructedAsyncValueRef<CpuEvent>();
  malloc_event.AndThen([buffer, buffer_size = expected.size()]() mutable {
    CHECK_OK(CpuDeviceMemory::AllocateInto(buffer_size, buffer.AsPtr()));
  });

  auto definition_event = MakeConstructedAsyncValueRef<CpuEvent>();
  absl::InlinedVector<PjRtDeviceEventRef, 2> definition_events;
  definition_events.push_back(PjRtDeviceEventRef(definition_event));

  auto raw_buffer = tsl::MakeRef<CpuRawBuffer>(memory_space, buffer,
                                               /*buffer_size=*/expected.size(),
                                               /*is_mutable=*/true);
  auto result = raw_buffer->buffer();

  ASSERT_FALSE(result.IsAvailable());
  ASSERT_EQ(raw_buffer->GetOnDeviceSizeInBytes(), expected.size());

  ThreadPool thread_pool(tsl::Env::Default(), "tracked_buffer_test",
                         /*num_threads=*/4);

  thread_pool.Schedule([&]() {
    malloc_event.SetStateConcrete();
    std::memcpy(buffer->untyped_data(), expected.data(), expected.size());
    definition_event.SetStateConcrete();
  });

  BlockUntilReady(definition_event);

  EXPECT_EQ(std::string(static_cast<const char*>(result->untyped_data()),
                        result->size_bytes()),
            expected);
}

TEST(CpuDeviceMemoryTest, PackOrCopyOobWrite) {
  Shape shape_large = ShapeUtil::MakeShape(S4, {16});
  std::vector<s4> data(16, s4(1));
  Literal literal = LiteralUtil::CreateR1<s4>(data);

  std::vector<char> dst(4, 0);

  // Asserts that the mismatched target size triggers the CHECK abort securely.
  EXPECT_DEATH(PackOrCopy(S4, literal, dst.data(), 4),
               "Mismatched packed target size in PackOrCopy");
}

TEST(CpuDeviceMemoryTest, CacheIdentityTracksWritesAndSlices) {
  ASSERT_OK_AND_ASSIGN(auto memory, CpuDeviceMemory::Allocate(64));
  // Writes before the first identity do not prevent subsequent caching.
  memory->InvalidateCacheIdentity();
  auto identity = memory->GetCacheIdentity();
  ASSERT_NE(identity, nullptr);
  EXPECT_EQ(identity, memory->GetCacheIdentity());
  auto slice = CpuDeviceMemory::CreateSlicedMemory(memory, 16, 32);
  EXPECT_EQ(identity, slice->GetCacheIdentity());
  slice->InvalidateCacheIdentity();
  auto updated = memory->GetCacheIdentity();
  ASSERT_NE(updated, nullptr);
  EXPECT_NE(identity, updated);
  EXPECT_EQ(updated, slice->GetCacheIdentity());
  slice->DisableCacheIdentity();
  EXPECT_EQ(memory->GetCacheIdentity(), nullptr);
  memory->InvalidateCacheIdentity();
  EXPECT_EQ(slice->GetCacheIdentity(), nullptr);
}

TEST(CpuDeviceMemoryTest, DeferredWriteInvalidatesAfterItsDependencies) {
  ASSERT_OK_AND_ASSIGN(auto client, GetPjRtCpuClient(CpuClientOptions()));
  ASSERT_OK_AND_ASSIGN(auto raw,
                       CpuRawBuffer::Allocate(client->memory_spaces()[0], 64));
  alignas(64) char bytes[64];
  std::memset(bytes, 1, sizeof(bytes));
  auto identity = raw->buffer()->GetCacheIdentity();
  PjRtRawBufferInterface* interface = raw.get();
  EXPECT_EQ(interface->GetHostPointerForInternalUse(
                PjRtRawBufferInterface::HostAccess::kRead),
            raw->buffer()->untyped_data());
  EXPECT_EQ(identity, raw->buffer()->GetCacheIdentity());
  std::memset(interface->GetHostPointerForInternalUse(
                  PjRtRawBufferInterface::HostAccess::kWrite),
              2, sizeof(bytes));
  EXPECT_NE(identity, raw->buffer()->GetCacheIdentity());
  identity = raw->buffer()->GetCacheIdentity();
  ASSERT_NE(identity, nullptr);
  auto pending = tsl::MakeConstructedAsyncValueRef<CpuEvent>();
  ASSERT_OK_AND_ASSIGN(auto written,
                       raw->CopyRawHostToDeviceAndReturnEvent(
                           bytes, 0, 64, {PjRtDeviceEventRef(pending)}));
  auto completion = written.down_cast<CpuEvent>();
  EXPECT_FALSE(completion.IsAvailable());
  EXPECT_EQ(identity, raw->buffer()->GetCacheIdentity());
  pending.SetStateConcrete();
  tsl::BlockUntilReady(completion);
  ASSERT_FALSE(completion.IsError());
  EXPECT_EQ(std::memcmp(raw->buffer()->untyped_data(), bytes, sizeof(bytes)),
            0);
  EXPECT_NE(identity, raw->buffer()->GetCacheIdentity());
  EXPECT_NE(raw->buffer()->GetCacheIdentity(), nullptr);
}

}  // namespace
}  // namespace xla
