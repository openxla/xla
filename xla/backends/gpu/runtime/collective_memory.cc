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

#include "xla/backends/gpu/runtime/collective_memory.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"
#include "xla/util.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::gpu {

CollectiveMemory::CollectiveMemory(
    const BufferAllocations& buffers,
    absl::flat_hash_map<Key, std::unique_ptr<SymmetricMemory>> sym_memories)
    : buffers_(buffers), sym_memories_(std::move(sym_memories)) {}

std::pair<SymmetricMemory*, size_t> CollectiveMemory::FindSymmetricMemory(
    const GpuCliqueKey& clique, BufferAllocation::Index allocation) const {
  auto it = sym_memories_.find(std::make_pair(clique, allocation));
  if (it == sym_memories_.end()) {
    return std::make_pair(nullptr, 0);
  }
  return std::make_pair(it->second.get(), 0);
}

std::pair<SymmetricMemory*, size_t> CollectiveMemory::FindSymmetricMemory(
    const GpuCliqueKey& clique, se::DeviceAddressBase addr) const {
  auto allocation = buffers_.FindAllocationIndex(addr);
  if (!allocation.has_value()) {
    return std::make_pair(nullptr, 0);
  }

  // Find offset from the base allocation.
  se::DeviceAddressBase base = buffers_.GetDeviceAddress(*allocation);
  size_t offset = tsl::safe_reinterpret_cast<uintptr_t>(addr.opaque()) -
                  tsl::safe_reinterpret_cast<uintptr_t>(base.opaque());

  auto [sym, sym_offset] = FindSymmetricMemory(clique, *allocation);
  return std::make_pair(sym, sym_offset + offset);
}

absl::StatusOr<CollectiveMemory> AcquireCollectiveMemory(
    const CollectiveParams& params, const CollectiveCliques& cliques,
    const CollectiveMemoryRequests& requests) {
  // We rely on determenistic order of memory requests, to guarantee that all
  // ranks create symmetric memory in identical order, otherwise we can get
  // a deadlock.
  std::vector<CollectiveMemoryRequests::SymmetricAllocations> allocs =
      requests.OrderedSymmetricAllocations();
  if (allocs.empty()) {
    return CollectiveMemory(requests.buffers(), /*sym_memories=*/{});
  }

  VLOG(2) << absl::StreamFormat(
      "[%d] Acquire %d collective memories for global device id %v; run_id=%v",
      params.executor->device_ordinal(), allocs.size(), params.global_device_id,
      params.run_id);
  absl::Time start = absl::Now();

  for (size_t i = 0; i < allocs.size(); ++i) {
    const CollectiveMemoryRequests::SymmetricAllocations& r = allocs[i];
    VLOG(2) << absl::StreamFormat(
        "[%d]    symmetric memory #%d (global device %v): id=%d; clique=%v; "
        "allocations=[%s]",
        params.executor->device_ordinal(), i, params.global_device_id, r.id,
        r.key, absl::StrJoin(r.allocations, ", "));
  }

  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode("AcquireCollectiveMemory",
                                        {{"num_memories", allocs.size()}});
  });

  absl::flat_hash_map<CollectiveMemory::Key, std::unique_ptr<SymmetricMemory>>
      sym_memories;

  for (const CollectiveMemoryRequests::SymmetricAllocations& r : allocs) {
    std::optional<RankId> rank = r.key.rank(params.global_device_id);

    if (!rank.has_value()) {
      return Internal("Can't find global device id %v in clique key %s",
                      params.global_device_id, r.key.ToString());
    }

    // TODO(ezhulenev): All of the buffer allocations that we make symmetric
    // are created from the same underlying memory allocator. We can
    // significantly improve performance with a few tricks:
    //
    // 1. Coalesce adjacent allocations and create one large symmetric region.
    // 2. Create one big symmetric region from [start, end] addresses, we might
    //    have unused gaps in the middle, but it doesn't matter, we will ignore
    //    them.
    // 3. Cache symmetric memories in a process-level cache.
    //
    // Currently it's very simple proof of concept.

    ASSIGN_OR_RETURN(GpuCommunicator * comm, cliques.GetComm(r.key, *rank));
    for (BufferAllocation::Index i : r.allocations) {
      ASSIGN_OR_RETURN(
          std::unique_ptr<SymmetricMemory> symm,
          comm->CreateSymmetricMemory(requests.buffers().GetDeviceAddress(i)));
      sym_memories[std::make_pair(r.key, i)] = std::move(symm);
    }
  }

  VLOG(2) << absl::StreamFormat(
      "[%d] Acquired %d collective memories in %s for global device id %v; "
      "run_id=%v",
      params.executor->device_ordinal(), allocs.size(),
      absl::FormatDuration(absl::Now() - start), params.global_device_id,
      params.run_id);

  return CollectiveMemory(requests.buffers(), std::move(sym_memories));
}

}  // namespace xla::gpu
