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

#ifndef XLA_PJRT_GPU_GPU_DMA_BUF_CACHE_H_
#define XLA_PJRT_GPU_GPU_DMA_BUF_CACHE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/dma_buf.h"
#include "xla/tsl/framework/allocator.h"

namespace xla {

// Caches exported Linux dma-buf file descriptors for GPU suballocator chunks
// and notifies registered watchers when dma-buf regions are created or closed.
//
// Chunks are recorded via `alloc_visitor()` and invalidated via
// `free_visitor()`. When `GetOrExport()` is called for a buffer contained in a
// tracked chunk, the entire chunk is lazily exported once and shared across all
// buffers in that chunk. Buffers outside any single tracked chunk fall back to
// an uncached export owned by the returned `ExportedDmaBuf`. Instances must be
// owned by `std::shared_ptr` before calling `alloc_visitor()`,
// `free_visitor()`, `GetOrExport()`, or `WatchRegions()`.
class GpuDmaBufCache : public std::enable_shared_from_this<GpuDmaBufCache> {
 public:
  struct ExportedDmaBuf {
    // Shared ownership of the underlying dma-buf handle.
    std::shared_ptr<stream_executor::gpu::DmaBuf> dma_buf;

    // Byte offset of the requested buffer from the start of `dma_buf`.
    uint64_t offset = 0;
  };

  using ExportFn =
      absl::FunctionRef<absl::StatusOr<stream_executor::gpu::DmaBuf>(
          const stream_executor::DeviceAddressBase&)>;

  GpuDmaBufCache() = default;
  GpuDmaBufCache(const GpuDmaBufCache&) = delete;
  GpuDmaBufCache& operator=(const GpuDmaBufCache&) = delete;

  // Returns suballocator visitors that track chunk lifetimes.
  tsl::SubAllocator::Visitor alloc_visitor();
  tsl::SubAllocator::Visitor free_visitor();

  // Returns a cached dma-buf covering `buffer_range` (if inside a tracked
  // chunk), or falls back to an uncached export via `export_fn`.
  absl::StatusOr<ExportedDmaBuf> GetOrExport(
      int32_t device_ordinal,
      const stream_executor::DeviceAddressBase& buffer_range,
      ExportFn export_fn);

  // Subscribes to dma-buf regions for `device_ordinal`. Replays `on_register`
  // synchronously for all currently live regions on `device_ordinal` and
  // notifies for any new regions created on that device while the returned
  // watcher is alive.
  absl::StatusOr<std::unique_ptr<PjRtExternalDmaRegionWatcher>> WatchRegions(
      int32_t device_ordinal, PjRtExternalDmaRegionCallbacks callbacks);

 private:
  struct ChunkState {
    explicit ChunkState(stream_executor::DeviceAddressBase range)
        : range(range) {}

    // Full suballocator chunk address range.
    const stream_executor::DeviceAddressBase range;

    // Serializes lazy export of this chunk across concurrent `GetOrExport`
    // callers. Never acquired by `RecordFree`.
    absl::Mutex export_mu;

    // Lazily exported dma-buf for `range`, or nullptr if not yet exported.
    // Protected by `GpuDmaBufCache::mu_`.
    std::shared_ptr<stream_executor::gpu::DmaBuf> dma_buf;

    // True once `RecordFree` has removed this chunk from `chunks_`.
    // Protected by `GpuDmaBufCache::mu_`.
    bool freed = false;
  };

  struct LiveRegion {
    // Device ordinal owning the exported region.
    int32_t device_ordinal = 0;

    // Exported region metadata.
    PjRtExternalDmaRegion region;

    // Weak reference to the live dma-buf so `live_regions_` does not extend
    // its lifetime.
    std::weak_ptr<stream_executor::gpu::DmaBuf> dma_buf;
  };

  struct WatcherState {
    WatcherState(int32_t device_ordinal,
                 PjRtExternalDmaRegionCallbacks callbacks)
        : device_ordinal(device_ordinal), callbacks(std::move(callbacks)) {}

    // Device ordinal this watcher subscribes to.
    const int32_t device_ordinal;

    // User-supplied register/unregister callbacks.
    PjRtExternalDmaRegionCallbacks callbacks;

    // Serializes all callback invocations for this watcher and prevents
    // `~PjRtExternalDmaRegionWatcher()` from returning while a callback is in
    // flight.
    absl::Mutex cb_mu;

    // Live region IDs currently registered with this watcher.
    absl::btree_map<uint64_t, PjRtExternalDmaRegion> registered_regions
        ABSL_GUARDED_BY(cb_mu);

    // True once the watcher has been unsubscribed.
    bool unsubscribed ABSL_GUARDED_BY(cb_mu) = false;
  };

  void RecordAlloc(void* ptr, int32_t device_ordinal, size_t bytes);
  void RecordFree(void* ptr, int32_t device_ordinal, size_t bytes);

  // Returns the chunk's dma-buf (exporting lazily if needed), or nullptr if the
  // chunk was freed concurrently.
  absl::StatusOr<std::shared_ptr<stream_executor::gpu::DmaBuf>>
  GetOrExportChunk(int32_t device_ordinal, ChunkState& chunk,
                   ExportFn export_fn);

  std::shared_ptr<stream_executor::gpu::DmaBuf> RegisterNewDmaBuf(
      int32_t device_ordinal, stream_executor::gpu::DmaBuf raw_dma_buf);
  void UnregisterDmaBuf(int32_t device_ordinal, uint64_t region_id,
                        const PjRtExternalDmaRegion& region);
  void UnwatchRegions(std::shared_ptr<WatcherState> watcher);

  // Protects `chunks_`, `next_region_id_`, `live_regions_`, and `watchers_`.
  // Never held across user callbacks or driver export syscalls.
  absl::Mutex mu_;

  // Tracked suballocator chunks keyed by device ordinal.
  absl::flat_hash_map<int32_t, std::vector<std::shared_ptr<ChunkState>>> chunks_
      ABSL_GUARDED_BY(mu_);

  // Monotonically increasing live region ID.
  uint64_t next_region_id_ ABSL_GUARDED_BY(mu_) = 0;

  // Currently live dma-buf regions keyed by region ID.
  absl::btree_map<uint64_t, LiveRegion> live_regions_ ABSL_GUARDED_BY(mu_);

  // Active region watchers.
  std::vector<std::shared_ptr<WatcherState>> watchers_ ABSL_GUARDED_BY(mu_);
};

}  // namespace xla

#endif  // XLA_PJRT_GPU_GPU_DMA_BUF_CACHE_H_
