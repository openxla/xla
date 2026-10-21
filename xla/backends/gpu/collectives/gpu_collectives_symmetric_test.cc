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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "xla/backends/gpu/collectives/collectives_test_util.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/future.h"
#include "xla/runtime/device_id.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::gpu {
namespace {

static constexpr GlobalDeviceId kD0(0);
static constexpr GlobalDeviceId kD1(1);
static constexpr GlobalDeviceId kD2(2);
static constexpr GlobalDeviceId kD3(3);

TEST(GpuCollectivesSymmetricTest, PutAndWaitSignal) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       PlatformUtil::GetPlatform("gpu"));

  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, 2));
  if (!executors[0]->CanEnablePeerAccessTo(executors[1])) {
    GTEST_SKIP() << "Test requires peer access between devices";
  }

  // TODO(rocm): API available in RCCL 2.30, should be enabled in the future
  auto cc = executors[0]->GetDeviceDescription().gpu_compute_capability();
  if (!cc.IsCuda()) {
    GTEST_SKIP() << "Test requires CUDA";
  }
  if (!cc.cuda_compute_capability()->IsAtLeastHopper()) {
    GTEST_SKIP() << "Test requires at least Hopper architecture";
  }

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executors, {kD0, kD1}));

  ASSERT_OK_AND_ASSIGN(auto allocators, CreateMemoryAllocators(executors));

  constexpr size_t kNumFloats = 4;
  constexpr size_t kNumBytes = kNumFloats * sizeof(float);

  ASSERT_OK_AND_ASSIGN(auto send_allocs, Allocate(allocators, kNumBytes));
  ASSERT_OK_AND_ASSIGN(auto recv_allocs, Allocate(allocators, kNumBytes));

  ASSERT_OK_AND_ASSIGN(auto stream0, executors[0]->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto stream1, executors[1]->CreateStream());

  float h_send0[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float h_send1[] = {5.0f, 6.0f, 7.0f, 8.0f};

  se::DeviceAddressBase send0_addr = send_allocs[0]->address();
  se::DeviceAddressBase send1_addr = send_allocs[1]->address();
  se::DeviceAddressBase recv0_addr = recv_allocs[0]->address();
  se::DeviceAddressBase recv1_addr = recv_allocs[1]->address();

  ASSERT_OK(stream0->Memcpy(&send0_addr, h_send0, kNumBytes));
  ASSERT_OK(stream1->Memcpy(&send1_addr, h_send1, kNumBytes));
  ASSERT_OK(stream0->MemZero(&recv0_addr, kNumBytes));
  ASSERT_OK(stream1->MemZero(&recv1_addr, kNumBytes));
  ASSERT_OK(stream0->BlockHostUntilDone());
  ASSERT_OK(stream1->BlockHostUntilDone());

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "collectives", 2);
  tsl::Executor& exec = *pool.AsExecutor();

  auto fsymm_send = CreateSymmetricMemory(exec, comms, send_allocs);
  ASSERT_OK_AND_ASSIGN(auto symm_send,
                       AwaitSymmetricMemory(std::move(fsymm_send)));

  auto fsymm_recv = CreateSymmetricMemory(exec, comms, recv_allocs);
  ASSERT_OK_AND_ASSIGN(auto symm_recv,
                       AwaitSymmetricMemory(std::move(fsymm_recv)));

  GpuSignalDesc signal_desc(0, 0);

  auto f0 = MakeFutureOn<void>(exec, [&]() -> absl::Status {
    GpuCollectives::Executor gpu_exec(stream0.get());
    ABSL_RETURN_IF_ERROR(comms[0]
                             ->Put(send0_addr, symm_recv[0].get(), 0, kNumBytes,
                                   RankId(1), gpu_exec)
                             .Await());
    return comms[0]->WaitSignal(RankId(1), 1, signal_desc, gpu_exec).Await();
  });

  auto f1 = MakeFutureOn<void>(exec, [&]() -> absl::Status {
    GpuCollectives::Executor gpu_exec(stream1.get());
    ABSL_RETURN_IF_ERROR(comms[1]
                             ->Put(send1_addr, symm_recv[1].get(), 0, kNumBytes,
                                   RankId(0), gpu_exec)
                             .Await());
    return comms[1]->WaitSignal(RankId(0), 1, signal_desc, gpu_exec).Await();
  });

  ASSERT_OK(f0.Await());
  ASSERT_OK(f1.Await());

  ASSERT_OK(stream0->BlockHostUntilDone());
  ASSERT_OK(stream1->BlockHostUntilDone());

  float h_recv0[kNumFloats];
  float h_recv1[kNumFloats];
  ASSERT_OK(stream0->Memcpy(h_recv0, recv0_addr, kNumBytes));
  ASSERT_OK(stream1->Memcpy(h_recv1, recv1_addr, kNumBytes));
  ASSERT_OK(stream0->BlockHostUntilDone());
  ASSERT_OK(stream1->BlockHostUntilDone());

  EXPECT_THAT(h_recv0, testing::ElementsAre(5.0f, 6.0f, 7.0f, 8.0f));
  EXPECT_THAT(h_recv1, testing::ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
}

TEST(GpuCollectivesSymmetricTest, PutAndWaitSignals) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       PlatformUtil::GetPlatform("gpu"));

  if (platform->VisibleDeviceCount() < 4) {
    GTEST_SKIP() << "Test requires at least 4 GPUs";
  }

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, 4));
  for (se::StreamExecutor* executor : executors) {
    auto cc = executor->GetDeviceDescription().gpu_compute_capability();
    if (!cc.IsCuda()) {
      GTEST_SKIP() << "Test requires CUDA";
    }
    if (!cc.cuda_compute_capability()->IsAtLeastHopper()) {
      GTEST_SKIP() << "Test requires at least Hopper architecture";
    }
  }
  for (size_t peer = 1; peer < executors.size(); ++peer) {
    if (!executors[peer]->CanEnablePeerAccessTo(executors[0])) {
      GTEST_SKIP() << "Test requires peer access to device 0";
    }
  }

  ASSERT_OK_AND_ASSIGN(auto comms,
                       CreateCommunicators(executors, {kD0, kD1, kD2, kD3}));
  ASSERT_OK_AND_ASSIGN(auto allocators, CreateMemoryAllocators(executors));

  constexpr size_t kChunkFloats = 4;
  constexpr size_t kChunkBytes = kChunkFloats * sizeof(float);
  constexpr size_t kRecvFloats = 3 * kChunkFloats;
  constexpr size_t kRecvBytes = kRecvFloats * sizeof(float);

  ASSERT_OK_AND_ASSIGN(auto send_allocs, Allocate(allocators, kChunkBytes));
  ASSERT_OK_AND_ASSIGN(auto recv_allocs, Allocate(allocators, kRecvBytes));

  ASSERT_OK_AND_ASSIGN(auto stream0, executors[0]->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto stream1, executors[1]->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto stream2, executors[2]->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto stream3, executors[3]->CreateStream());

  float h_send1[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float h_send2[] = {5.0f, 6.0f, 7.0f, 8.0f};
  float h_send3[] = {9.0f, 10.0f, 11.0f, 12.0f};

  se::DeviceAddressBase send1_addr = send_allocs[1]->address();
  se::DeviceAddressBase send2_addr = send_allocs[2]->address();
  se::DeviceAddressBase send3_addr = send_allocs[3]->address();
  se::DeviceAddressBase recv0_addr = recv_allocs[0]->address();

  ASSERT_OK(stream1->Memcpy(&send1_addr, h_send1, kChunkBytes));
  ASSERT_OK(stream2->Memcpy(&send2_addr, h_send2, kChunkBytes));
  ASSERT_OK(stream3->Memcpy(&send3_addr, h_send3, kChunkBytes));
  std::array<se::Stream*, 4> streams = {stream0.get(), stream1.get(),
                                        stream2.get(), stream3.get()};
  for (size_t i = 0; i < executors.size(); ++i) {
    se::DeviceAddressBase recv_addr = recv_allocs[i]->address();
    ASSERT_OK(streams[i]->MemZero(&recv_addr, kRecvBytes));
    ASSERT_OK(streams[i]->BlockHostUntilDone());
  }

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "collectives", 4);
  tsl::Executor& exec = *pool.AsExecutor();

  auto fsymm_send = CreateSymmetricMemory(exec, comms, send_allocs);
  ASSERT_OK_AND_ASSIGN(auto symm_send,
                       AwaitSymmetricMemory(std::move(fsymm_send)));

  auto fsymm_recv = CreateSymmetricMemory(exec, comms, recv_allocs);
  ASSERT_OK_AND_ASSIGN(auto symm_recv,
                       AwaitSymmetricMemory(std::move(fsymm_recv)));

  GpuSignalDesc signal_desc(0, 0);

  auto f0 = MakeFutureOn<void>(exec, [&]() -> absl::Status {
    GpuCollectives::Executor gpu_exec(stream0.get());
    std::array<Communicator::PeerWaitDesc, 3> peer_wait_descs = {
        Communicator::PeerWaitDesc{RankId(1), 1, signal_desc},
        Communicator::PeerWaitDesc{RankId(2), 1, signal_desc},
        Communicator::PeerWaitDesc{RankId(3), 1, signal_desc}};
    return comms[0]->WaitSignals(peer_wait_descs, gpu_exec).Await();
  });

  auto f1 = MakeFutureOn<void>(exec, [&]() -> absl::Status {
    GpuCollectives::Executor gpu_exec(stream1.get());
    return comms[1]
        ->Put(send1_addr, symm_recv[1].get(), /*offset=*/0, kChunkBytes,
              RankId(0), gpu_exec)
        .Await();
  });

  auto f2 = MakeFutureOn<void>(exec, [&]() -> absl::Status {
    GpuCollectives::Executor gpu_exec(stream2.get());
    return comms[2]
        ->Put(send2_addr, symm_recv[2].get(), /*offset=*/kChunkBytes,
              kChunkBytes, RankId(0), gpu_exec)
        .Await();
  });

  auto f3 = MakeFutureOn<void>(exec, [&]() -> absl::Status {
    GpuCollectives::Executor gpu_exec(stream3.get());
    return comms[3]
        ->Put(send3_addr, symm_recv[3].get(), /*offset=*/2 * kChunkBytes,
              kChunkBytes, RankId(0), gpu_exec)
        .Await();
  });

  ASSERT_OK(f0.Await());
  ASSERT_OK(f1.Await());
  ASSERT_OK(f2.Await());
  ASSERT_OK(f3.Await());

  ASSERT_OK(stream0->BlockHostUntilDone());

  float h_recv0[kRecvFloats];
  ASSERT_OK(stream0->Memcpy(h_recv0, recv0_addr, kRecvBytes));
  ASSERT_OK(stream0->BlockHostUntilDone());

  EXPECT_THAT(h_recv0,
              testing::ElementsAre(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                                   8.0f, 9.0f, 10.0f, 11.0f, 12.0f));

  ASSERT_OK(stream1->BlockHostUntilDone());
  ASSERT_OK(stream2->BlockHostUntilDone());
  ASSERT_OK(stream3->BlockHostUntilDone());
}

}  // namespace
}  // namespace xla::gpu
