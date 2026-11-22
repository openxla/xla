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

#include "xla/pjrt/gpu/gpu_dma_buf_cache.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/dma_buf.h"
#include "xla/tsl/framework/allocator.h"

namespace xla {

namespace {

bool RangeContains(const stream_executor::DeviceAddressBase& outer,
                   const stream_executor::DeviceAddressBase& inner) {
  if (inner.opaque() == nullptr || inner.size() == 0) {
    return false;
  }
  const uintptr_t outer_start = reinterpret_cast<uintptr_t>(outer.opaque());
  const uintptr_t inner_start = reinterpret_cast<uintptr_t>(inner.opaque());
  if (inner_start < outer_start) {
    return false;
  }
  const uint64_t rel_offset = static_cast<uint64_t>(inner_start - outer_start);
  return rel_offset <= outer.size() &&
         inner.size() <= outer.size() - rel_offset;
}

GpuDmaBufCache::ExportedDmaBuf MakeChunkExport(
    const stream_executor::DeviceAddressBase& chunk_range,
    uintptr_t buffer_start,
    std::shared_ptr<stream_executor::gpu::DmaBuf> dma_buf) {
  const uintptr_t chunk_start =
      reinterpret_cast<uintptr_t>(chunk_range.opaque());
  const uint64_t offset =
      static_cast<uint64_t>(buffer_start - chunk_start) + dma_buf->offset();
  return GpuDmaBufCache::ExportedDmaBuf{
      /*dma_buf=*/std::move(dma_buf),
      /*offset=*/offset,
  };
}

}  // namespace

tsl::SubAllocator::Visitor GpuDmaBufCache::alloc_visitor() {
  return [self = shared_from_this()](void* ptr, int32_t device_ordinal,
                                     size_t bytes) {
    self->RecordAlloc(ptr, device_ordinal, bytes);
  };
}

tsl::SubAllocator::Visitor GpuDmaBufCache::free_visitor() {
  return [self = shared_from_this()](void* ptr, int32_t device_ordinal,
                                     size_t bytes) {
    self->RecordFree(ptr, device_ordinal, bytes);
  };
}

void GpuDmaBufCache::RecordAlloc(void* ptr, int32_t device_ordinal,
                                 size_t bytes) {
  if (ptr == nullptr || bytes == 0) {
    return;
  }
  absl::MutexLock lock(mu_);
  chunks_[device_ordinal].push_back(std::make_shared<ChunkState>(
      stream_executor::DeviceAddressBase(ptr, bytes)));
}

void GpuDmaBufCache::RecordFree(void* ptr, int32_t device_ordinal,
                                size_t /*bytes*/) {
  if (ptr == nullptr) {
    return;
  }
  // Collect removed dma_bufs and drop them after releasing `mu_` so watcher
  // `on_unregister` callbacks never run under `mu_`.
  std::vector<std::shared_ptr<stream_executor::gpu::DmaBuf>> to_release;
  {
    absl::MutexLock lock(mu_);
    auto it = chunks_.find(device_ordinal);
    if (it == chunks_.end()) {
      return;
    }
    std::vector<std::shared_ptr<ChunkState>>& device_chunks = it->second;
    device_chunks.erase(
        std::remove_if(device_chunks.begin(), device_chunks.end(),
                       [&](std::shared_ptr<ChunkState>& chunk) {
                         if (chunk->range.opaque() == ptr) {
                           chunk->freed = true;
                           if (chunk->dma_buf != nullptr) {
                             to_release.push_back(std::move(chunk->dma_buf));
                           }
                           return true;
                         }
                         return false;
                       }),
        device_chunks.end());
  }
}

std::shared_ptr<stream_executor::gpu::DmaBuf> GpuDmaBufCache::RegisterNewDmaBuf(
    int32_t device_ordinal, stream_executor::gpu::DmaBuf raw_dma_buf) {
  const PjRtExternalDmaRegion region = {
      /*fd=*/raw_dma_buf.fd(),
      /*size=*/raw_dma_buf.size(),
  };

  uint64_t region_id = 0;
  std::shared_ptr<stream_executor::gpu::DmaBuf> dma_buf;
  std::vector<std::shared_ptr<WatcherState>> watchers;
  {
    absl::MutexLock lock(mu_);
    region_id = ++next_region_id_;
    auto* raw_ptr = new stream_executor::gpu::DmaBuf(std::move(raw_dma_buf));
    std::weak_ptr<GpuDmaBufCache> weak_self = weak_from_this();
    dma_buf = std::shared_ptr<stream_executor::gpu::DmaBuf>(
        raw_ptr, [weak_self = std::move(weak_self), device_ordinal, region_id,
                  region](stream_executor::gpu::DmaBuf* ptr) {
          if (std::shared_ptr<GpuDmaBufCache> self = weak_self.lock()) {
            self->UnregisterDmaBuf(device_ordinal, region_id, region);
          }
          delete ptr;
        });
    live_regions_.emplace(region_id, LiveRegion{
                                         /*device_ordinal=*/device_ordinal,
                                         /*region=*/region,
                                         /*dma_buf=*/dma_buf,
                                     });
    watchers = watchers_;
  }

  for (const std::shared_ptr<WatcherState>& watcher : watchers) {
    if (watcher->device_ordinal != device_ordinal) {
      continue;
    }
    absl::MutexLock cb_lock(watcher->cb_mu);
    if (!watcher->unsubscribed &&
        watcher->registered_regions.try_emplace(region_id, region).second) {
      watcher->callbacks.on_register(region);
    }
  }
  return dma_buf;
}

void GpuDmaBufCache::UnregisterDmaBuf(int32_t device_ordinal,
                                      uint64_t region_id,
                                      const PjRtExternalDmaRegion& region) {
  std::vector<std::shared_ptr<WatcherState>> watchers;
  {
    absl::MutexLock lock(mu_);
    live_regions_.erase(region_id);
    watchers = watchers_;
  }
  for (const std::shared_ptr<WatcherState>& watcher : watchers) {
    if (watcher->device_ordinal != device_ordinal) {
      continue;
    }
    absl::MutexLock cb_lock(watcher->cb_mu);
    if (watcher->registered_regions.erase(region_id) > 0 &&
        watcher->callbacks.on_unregister != nullptr) {
      watcher->callbacks.on_unregister(region);
    }
  }
}

void GpuDmaBufCache::UnwatchRegions(std::shared_ptr<WatcherState> watcher) {
  {
    // Unregister while `watcher` is still in `watchers_` so any concurrent
    // `UnregisterDmaBuf` blocks on `cb_mu` and keeps every fd open until
    // `on_unregister` returns.
    absl::MutexLock cb_lock(watcher->cb_mu);
    watcher->unsubscribed = true;
    if (watcher->callbacks.on_unregister != nullptr) {
      for (const auto& [id, region] : watcher->registered_regions) {
        watcher->callbacks.on_unregister(region);
      }
    }
    watcher->registered_regions.clear();
  }
  absl::MutexLock lock(mu_);
  watchers_.erase(std::remove(watchers_.begin(), watchers_.end(), watcher),
                  watchers_.end());
}

absl::StatusOr<std::shared_ptr<stream_executor::gpu::DmaBuf>>
GpuDmaBufCache::GetOrExportChunk(int32_t device_ordinal, ChunkState& chunk,
                                 ExportFn export_fn) {
  std::shared_ptr<stream_executor::gpu::DmaBuf> discarded;
  {
    absl::MutexLock export_lock(chunk.export_mu);
    {
      absl::MutexLock lock(mu_);
      if (chunk.dma_buf != nullptr) {
        return chunk.dma_buf;
      }
      if (chunk.freed) {
        return nullptr;
      }
    }
    ABSL_ASSIGN_OR_RETURN(stream_executor::gpu::DmaBuf exported_chunk,
                          export_fn(chunk.range));
    std::shared_ptr<stream_executor::gpu::DmaBuf> dma_buf =
        RegisterNewDmaBuf(device_ordinal, std::move(exported_chunk));
    {
      absl::MutexLock lock(mu_);
      if (!chunk.freed) {
        chunk.dma_buf = dma_buf;
        return dma_buf;
      }
    }
    // The chunk was freed while `export_fn` was in flight; drop the chunk-wide
    // export after releasing `export_mu`.
    discarded = std::move(dma_buf);
  }
  discarded.reset();
  return nullptr;
}

absl::StatusOr<GpuDmaBufCache::ExportedDmaBuf> GpuDmaBufCache::GetOrExport(
    int32_t device_ordinal,
    const stream_executor::DeviceAddressBase& buffer_range,
    ExportFn export_fn) {
  const uintptr_t buffer_start =
      reinterpret_cast<uintptr_t>(buffer_range.opaque());

  std::shared_ptr<ChunkState> chunk;
  {
    absl::MutexLock lock(mu_);
    auto it = chunks_.find(device_ordinal);
    if (it != chunks_.end()) {
      for (const std::shared_ptr<ChunkState>& candidate : it->second) {
        if (RangeContains(candidate->range, buffer_range)) {
          if (candidate->dma_buf != nullptr) {
            return MakeChunkExport(candidate->range, buffer_start,
                                   candidate->dma_buf);
          }
          chunk = candidate;
          break;
        }
      }
    }
  }

  if (chunk != nullptr) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<stream_executor::gpu::DmaBuf> dma_buf,
                          GetOrExportChunk(device_ordinal, *chunk, export_fn));
    if (dma_buf != nullptr) {
      return MakeChunkExport(chunk->range, buffer_start, std::move(dma_buf));
    }
  }

  // Fallback for untracked / multi-chunk / non-BFC / concurrently-freed ranges.
  ABSL_ASSIGN_OR_RETURN(stream_executor::gpu::DmaBuf exported,
                        export_fn(buffer_range));
  std::shared_ptr<stream_executor::gpu::DmaBuf> dma_buf =
      RegisterNewDmaBuf(device_ordinal, std::move(exported));
  const uint64_t offset = dma_buf->offset();
  return ExportedDmaBuf{
      /*dma_buf=*/std::move(dma_buf),
      /*offset=*/offset,
  };
}

absl::StatusOr<std::unique_ptr<PjRtExternalDmaRegionWatcher>>
GpuDmaBufCache::WatchRegions(int32_t device_ordinal,
                             PjRtExternalDmaRegionCallbacks callbacks) {
  if (!callbacks.on_register) {
    return absl::InvalidArgumentError(
        "WatchRegions requires a non-null on_register callback.");
  }

  auto watcher =
      std::make_shared<WatcherState>(device_ordinal, std::move(callbacks));
  std::vector<std::shared_ptr<stream_executor::gpu::DmaBuf>> pinned_dma_bufs;
  std::vector<std::pair<uint64_t, PjRtExternalDmaRegion>> snapshot;
  {
    absl::MutexLock lock(mu_);
    watchers_.push_back(watcher);
    for (const auto& [id, live] : live_regions_) {
      if (live.device_ordinal != device_ordinal) {
        continue;
      }
      if (std::shared_ptr<stream_executor::gpu::DmaBuf> pinned =
              live.dma_buf.lock()) {
        pinned_dma_bufs.push_back(std::move(pinned));
        snapshot.emplace_back(id, live.region);
      }
    }
  }

  {
    absl::MutexLock cb_lock(watcher->cb_mu);
    for (const auto& [id, region] : snapshot) {
      if (watcher->registered_regions.try_emplace(id, region).second) {
        watcher->callbacks.on_register(region);
      }
    }
  }
  return std::make_unique<PjRtExternalDmaRegionWatcher>(
      [self = shared_from_this(), watcher = std::move(watcher)]() mutable {
        self->UnwatchRegions(std::move(watcher));
      });
}

}  // namespace xla
