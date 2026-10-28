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

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk_multigpu_test_utils.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"

namespace xla::gpu {
namespace {

constexpr int kNumDevices = 2;

TEST(CollectiveMemoryMultiGpuTest, ArenaWindowRegistrationAndSubSliceReuse) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  GpuExecutableRunOptions::DeviceIdMap id_map;
  for (int i = 0; i < kNumDevices; ++i) {
    id_map[LocalDeviceId(i)] = GlobalDeviceId(i);
  }
  GpuCliqueKey clique_key({GlobalDeviceId(0), GlobalDeviceId(1)}, kNumDevices);
  std::vector<std::vector<GlobalDeviceId>> device_groups = {
      {GlobalDeviceId(0), GlobalDeviceId(1)}};

  ASSERT_OK(RunOnDevices(
      kNumDevices, "arena_window_reuse", [&](int d) -> absl::Status {
        se::StreamExecutor* executor = GetGpuExecutor(d);
        ABSL_ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
                              executor->CreateStream());
        se::StreamExecutorAddressAllocator allocator(executor);

        constexpr int64_t kSmallSize = 16 * 1024;
        constexpr int64_t kLargeSize = 64 * 1024;
        se::DeviceAddressBase base_slab = executor->Allocate(
            kLargeSize, static_cast<int64_t>(se::MemorySpace::kCollective));
        if (base_slab.is_null()) {
          return absl::InternalError("Failed to allocate collective slab");
        }

        auto run_steps = [&]() -> absl::Status {
          ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase expected_arena,
                                executor->GetMemoryRange(base_slab));
          EXPECT_EQ(expected_arena.opaque(), base_slab.opaque());
          EXPECT_GE(expected_arena.size(), kLargeSize);

          se::DeviceAddressBase small_slice =
              base_slab.GetByteSlice(0, kSmallSize);
          se::DeviceAddressBase large_slice =
              base_slab.GetByteSlice(0, kLargeSize);
          se::DeviceAddressBase mid_slice =
              base_slab.GetByteSlice(kSmallSize, kSmallSize);

          GpuExecutableRunOptions gpu_options;
          gpu_options.set_gpu_global_device_ids(id_map);

          ServiceExecutableRunOptions run_options;
          run_options.mutable_run_options()->set_stream(stream.get());
          run_options.mutable_run_options()->set_device_assignment(
              &device_assignment);
          run_options.mutable_run_options()->set_gpu_executable_run_options(
              &gpu_options);
          run_options.mutable_run_options()->set_local_device_count(
              kNumDevices);

          ABSL_ASSIGN_OR_RETURN(
              CollectiveParams collective_params,
              CollectiveParams::Create(run_options, /*async_streams=*/{},
                                       LocalDeviceId(d)));

          CollectiveCliqueRequests clique_requests;
          ABSL_RETURN_IF_ERROR(
              clique_requests.RequestClique(clique_key, device_groups));
          ABSL_ASSIGN_OR_RETURN(
              CollectiveCliques cliques,
              AcquireCollectiveCliques(collective_params, clique_requests));

          // Step 1: Requesting a 16 KiB sub-slice registers the entire
          // underlying physical slab on the communicator.
          std::vector<se::DeviceAddressBase> bufs_1 = {small_slice};
          BufferAllocations allocs_1(bufs_1, d, &allocator);
          CollectiveMemoryRequests reqs_1(allocs_1);
          ABSL_RETURN_IF_ERROR(
              reqs_1.RequestSymmetricAllocation(clique_key, 0));
          ABSL_ASSIGN_OR_RETURN(
              CollectiveMemory mem_1,
              AcquireCollectiveMemory(collective_params, cliques, reqs_1));
          auto [sym_1, off_1] =
              mem_1.FindSymmetricMemory(clique_key, BufferAllocation::Index(0));
          if (sym_1 == nullptr) {
            return absl::InternalError("sym_1 is null");
          }
          EXPECT_EQ(sym_1->addr().opaque(), expected_arena.opaque());
          EXPECT_EQ(sym_1->addr().size(), expected_arena.size());
          EXPECT_EQ(off_1, 0);

          // Step 2: Requesting a 64 KiB slice on the same physical slab hits
          // the communicator cache and reuses sym_1 without re-registering.
          std::vector<se::DeviceAddressBase> bufs_2 = {large_slice};
          BufferAllocations allocs_2(bufs_2, d, &allocator);
          CollectiveMemoryRequests reqs_2(allocs_2);
          ABSL_RETURN_IF_ERROR(
              reqs_2.RequestSymmetricAllocation(clique_key, 0));
          ABSL_ASSIGN_OR_RETURN(
              CollectiveMemory mem_2,
              AcquireCollectiveMemory(collective_params, cliques, reqs_2));
          auto [sym_2, off_2] =
              mem_2.FindSymmetricMemory(clique_key, BufferAllocation::Index(0));
          if (sym_2 == nullptr) {
            return absl::InternalError("sym_2 is null");
          }
          EXPECT_EQ(sym_2, sym_1);
          EXPECT_EQ(sym_2->addr().size(), expected_arena.size());
          EXPECT_EQ(off_2, 0);

          // Step 3: Requesting a 16 KiB sub-slice at offset 16 KiB hits the
          // communicator cache and returns offset 16 KiB relative to sym_1.
          std::vector<se::DeviceAddressBase> bufs_3 = {mid_slice};
          BufferAllocations allocs_3(bufs_3, d, &allocator);
          CollectiveMemoryRequests reqs_3(allocs_3);
          ABSL_RETURN_IF_ERROR(
              reqs_3.RequestSymmetricAllocation(clique_key, 0));
          ABSL_ASSIGN_OR_RETURN(
              CollectiveMemory mem_3,
              AcquireCollectiveMemory(collective_params, cliques, reqs_3));
          auto [sym_3, off_3] =
              mem_3.FindSymmetricMemory(clique_key, BufferAllocation::Index(0));
          if (sym_3 == nullptr) {
            return absl::InternalError("sym_3 is null");
          }
          EXPECT_EQ(sym_3, sym_1);
          EXPECT_EQ(sym_3->addr().size(), expected_arena.size());
          EXPECT_EQ(off_3, kSmallSize);
          return absl::OkStatus();
        };

        absl::Status status = run_steps();
        executor->Deallocate(&base_slab);
        return status;
      }));
}

}  // namespace
}  // namespace xla::gpu
