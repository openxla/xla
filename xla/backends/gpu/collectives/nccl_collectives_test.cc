/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/collectives/nccl_collectives.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/future.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::gpu {
namespace {

TEST(NcclCollectives, CreateSymmetricMemory) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor0,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor1,
                       platform->ExecutorForDevice(1));

  GpuCollectives::Device device0(executor0);
  GpuCollectives::Device device1(executor1);

  NcclCollectives collectives;

  ASSERT_OK_AND_ASSIGN(CliqueId clique_id, collectives.CreateUniqueCliqueId());
  CliqueIds clique_ids(clique_id);

  GpuCliqueKey clique_key({GlobalDeviceId(0), GlobalDeviceId(1)},
                          /*num_local_participants=*/2);

  Collectives::DeviceRank rank0(&device0, RankId(0));
  Collectives::DeviceRank rank1(&device1, RankId(1));

  ASSERT_OK_AND_ASSIGN(auto comms, collectives.CreateCommunicators(
                                       clique_key, clique_ids, {rank0, rank1},
                                       GpuCollectives::Config{}));
  ASSERT_EQ(comms.size(), 2);

  GpuCommunicator* comm0 = dynamic_cast<GpuCommunicator*>(comms[0].get());
  GpuCommunicator* comm1 = dynamic_cast<GpuCommunicator*>(comms[1].get());

  EXPECT_TRUE(comm0->platform_host_comm().handle);
  EXPECT_TRUE(comm1->platform_host_comm().handle);

  // Create memory allocators that allocate physical memory in the collective
  // memory space, which makes them compatible with symmetric memory
  // requirements.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<se::MemoryAllocator> allocator0,
      executor0->CreateMemoryAllocator(se::MemorySpace::kCollective));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<se::MemoryAllocator> allocator1,
      executor1->CreateMemoryAllocator(se::MemorySpace::kCollective));

  // Allocate device memory on each participating rank.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::MemoryAllocation> alloc0,
                       allocator0->Allocate(1024));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::MemoryAllocation> alloc1,
                       allocator1->Allocate(1024));

  // Because symmetric memory creating is a collective operation, we must call
  // it from a thead pool to avoid deadlocks.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "nccl", 2);
  tsl::Executor& exec = *pool.AsExecutor();

  // Register allocated buffers as symmetric memory.
  auto fsymm0 = Future<std::unique_ptr<SymmetricMemory>>::MakeOn(
      exec, [&] { return comm0->CreateSymmetricMemory(alloc0->address()); });
  auto fsymm1 = Future<std::unique_ptr<SymmetricMemory>>::MakeOn(
      exec, [&] { return comm1->CreateSymmetricMemory(alloc1->address()); });

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<SymmetricMemory> symm0,
                       std::move(fsymm0).Await());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<SymmetricMemory> symm1,
                       std::move(fsymm1).Await());
}

}  // namespace
}  // namespace xla::gpu