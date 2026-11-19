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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_cache.h"
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
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"

namespace xla::gpu {
namespace {

constexpr int kNumDevices = 2;

// Common setup for all tests: skips unless there are `kNumDevices` CUDA GPUs
// of Hopper or newer with peer access between them, and builds the device
// assignment and the local-to-global device id map.
class CollectiveMemoryMultiGpuTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!HasEnoughGpus(kNumDevices)) {
      GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
    }
    se::StreamExecutor* executor0 = GetGpuExecutor(0);
    auto cc = executor0->GetDeviceDescription().gpu_compute_capability();
    if (!cc.IsCuda() || !cc.cuda_compute_capability()->IsAtLeastHopper()) {
      GTEST_SKIP() << "Test requires CUDA Hopper or newer";
    }
    if (!executor0->CanEnablePeerAccessTo(GetGpuExecutor(1))) {
      GTEST_SKIP() << "Test requires peer access between devices";
    }
    device_assignment_.emplace(MakeDeviceAssignment(kNumDevices));
    for (int i = 0; i < kNumDevices; ++i) {
      id_map_[LocalDeviceId(i)] = GlobalDeviceId(i);
    }
  }

  std::optional<DeviceAssignment> device_assignment_;
  GpuExecutableRunOptions::DeviceIdMap id_map_;
};

TEST_F(CollectiveMemoryMultiGpuTest, ArenaWindowRegistrationAndSubSliceReuse) {
  GpuCliqueKey clique_key({GlobalDeviceId(0), GlobalDeviceId(1)}, kNumDevices);
  std::vector<std::vector<GlobalDeviceId>> device_groups = {
      {GlobalDeviceId(0), GlobalDeviceId(1)}};

  ASSERT_OK(RunOnDevices(
      kNumDevices, "arena_window_reuse", [&](int d) -> absl::Status {
        SCOPED_TRACE(absl::StrCat("device ", d));
        se::StreamExecutor* executor = GetGpuExecutor(d);
        ABSL_ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
                              executor->CreateStream());
        se::StreamExecutorAddressAllocator allocator(executor);

        constexpr int64_t kSmallSize = 16 * 1024;
        constexpr int64_t kLargeSize = 64 * 1024;
        se::DeviceAddressBase base_arena = executor->Allocate(
            kLargeSize, static_cast<int64_t>(se::MemorySpace::kCollective));
        if (base_arena.is_null()) {
          return absl::InternalError("Failed to allocate collective arena");
        }

        absl::Cleanup release_arena = [&] {
          ReleaseSymmetricMemory(LocalDeviceId(d), base_arena);
          executor->Deallocate(&base_arena);
        };

        ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase expected_arena,
                              executor->GetMemoryRange(base_arena));
        EXPECT_EQ(expected_arena.opaque(), base_arena.opaque());
        EXPECT_GE(expected_arena.size(), kLargeSize);

        GpuExecutableRunOptions gpu_options;
        gpu_options.set_gpu_global_device_ids(id_map_);

        ServiceExecutableRunOptions run_options;
        run_options.mutable_run_options()->set_stream(stream.get());
        run_options.mutable_run_options()->set_device_assignment(
            &*device_assignment_);
        run_options.mutable_run_options()->set_gpu_executable_run_options(
            &gpu_options);
        run_options.mutable_run_options()->set_local_device_count(kNumDevices);

        ABSL_ASSIGN_OR_RETURN(
            CollectiveParams collective_params,
            CollectiveParams::Create(run_options,
                                     /*async_streams=*/{}, LocalDeviceId(d)));
        collective_params.cache_nccl_window_in_communicator = true;
        // Unused when windows are cached on the communicator.
        CollectiveMemoryCache memory_cache(kNumDevices);

        CollectiveCliqueRequests clique_requests;
        ABSL_RETURN_IF_ERROR(
            clique_requests.RequestClique(clique_key, device_groups));
        ABSL_ASSIGN_OR_RETURN(
            CollectiveCliques cliques,
            AcquireCollectiveCliques(collective_params, clique_requests));

        // Acquires symmetric memory for a single buffer and returns its window
        // and the buffer offset inside the window. The window stays alive
        // because the communicator caches it.
        auto acquire = [&](se::DeviceAddressBase buffer)
            -> absl::StatusOr<std::pair<SymmetricMemory*, size_t>> {
          std::vector<se::DeviceAddressBase> buffers = {buffer};
          BufferAllocations allocations(buffers, d, &allocator);
          CollectiveMemoryRequests requests(allocations);
          ABSL_RETURN_IF_ERROR(
              requests.RequestSymmetricAllocation(clique_key, 0));
          ABSL_ASSIGN_OR_RETURN(
              CollectiveMemory memory,
              AcquireCollectiveMemory(collective_params, cliques, requests,
                                      memory_cache));
          auto sym = memory.FindSymmetricMemory(clique_key,
                                                BufferAllocation::Index(0));
          if (sym.first == nullptr) {
            return absl::InternalError("Symmetric memory not found");
          }
          return sym;
        };

        // A 16 KiB sub-slice registers the entire underlying physical arena on
        // the communicator.
        ABSL_ASSIGN_OR_RETURN((auto [arena_window, small_offset]),
                              acquire(base_arena.GetByteSlice(0, kSmallSize)));
        EXPECT_EQ(arena_window->addr().opaque(), expected_arena.opaque());
        EXPECT_EQ(arena_window->addr().size(), expected_arena.size());
        EXPECT_EQ(small_offset, 0);

        // A 64 KiB slice of the same arena reuses the cached window.
        ABSL_ASSIGN_OR_RETURN((auto [large_window, large_offset]),
                              acquire(base_arena.GetByteSlice(0, kLargeSize)));
        EXPECT_EQ(large_window, arena_window);
        EXPECT_EQ(large_offset, 0);

        // A 16 KiB sub-slice at offset 16 KiB reuses the cached window at
        // offset 16 KiB.
        ABSL_ASSIGN_OR_RETURN(
            (auto [mid_window, mid_offset]),
            acquire(base_arena.GetByteSlice(kSmallSize, kSmallSize)));
        EXPECT_EQ(mid_window, arena_window);
        EXPECT_EQ(mid_offset, kSmallSize);

        // A null buffer gets no window.
        std::vector<se::DeviceAddressBase> null_buffers = {
            se::DeviceAddressBase()};
        BufferAllocations null_allocations(null_buffers, d, &allocator);
        CollectiveMemoryRequests null_requests(null_allocations);
        ABSL_RETURN_IF_ERROR(
            null_requests.RequestSymmetricAllocation(clique_key, 0));
        ABSL_ASSIGN_OR_RETURN(
            CollectiveMemory null_memory,
            AcquireCollectiveMemory(collective_params, cliques, null_requests,
                                    memory_cache));
        EXPECT_EQ(
            null_memory
                .FindSymmetricMemory(clique_key, BufferAllocation::Index(0))
                .first,
            nullptr);
        return absl::OkStatus();
      }));
}

}  // namespace
}  // namespace xla::gpu
