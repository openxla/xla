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

#include "xla/service/cpu/onednn_weight_cache.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>

#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/mem.h"

namespace xla::cpu {
namespace {

constexpr char kCapacityEnvVar[] = "XLA_ONEDNN_WEIGHT_CACHE_CAPACITY_BYTES";
constexpr size_t kPackedWeightAlignment = 64;

}  // namespace

OneDnnPackedWeights::~OneDnnPackedWeights() {
  tsl::port::AlignedFree(data_);
  live_bytes_->fetch_sub(size_bytes_, std::memory_order_relaxed);
}

OneDnnWeightCache::LookupResult OneDnnWeightCache::LookupOrCreate(
    OneDnnWeightCacheKey key) {
  absl::MutexLock lock(mu_);
  // Keep the owner alive while accessing its tied allocation.
  auto source = key.source_identity.lock();
  if (!source) {
    return {LookupKind::kBypass, nullptr};
  }
  auto entry = entries_.find(key);
  if (entry != entries_.end()) {
    if (!entry->second->ready) {
      return {LookupKind::kBypass, nullptr};
    }
    auto weights = entry->second->weights.Lock();
    lru_.splice(lru_.begin(), lru_, entry->second);
    ++stats_.hits;
    VLOG(1) << "oneDNN weight cache hit: source=" << key.source_data
            << " packed_bytes=" << weights->size_bytes_;
    return {LookupKind::kHit, std::move(weights)};
  }

  const size_t packed_bytes = key.packed_desc.get_size();
  if (packed_bytes == 0 || packed_bytes > capacity_bytes_) {
    return {LookupKind::kBypass, nullptr};
  }

  // Removed entries can still have execution-held payload; charge live bytes.
  PruneExpiredLocked();
  while (packed_bytes >
             capacity_bytes_ - live_bytes_->load(std::memory_order_relaxed) &&
         TryEvictOneLocked()) {
  }
  if (packed_bytes >
      capacity_bytes_ - live_bytes_->load(std::memory_order_relaxed)) {
    return {LookupKind::kBypass, nullptr};
  }

  void* packed_data = tsl::port::AlignedMalloc(
      packed_bytes, static_cast<std::align_val_t>(kPackedWeightAlignment));
  if (packed_data == nullptr) {
    LOG_FIRST_N(WARNING, 1) << "oneDNN weight cache could not allocate "
                            << packed_bytes << " bytes; using unpacked weights";
    return {LookupKind::kBypass, nullptr};
  }

  live_bytes_->fetch_add(packed_bytes, std::memory_order_relaxed);
  auto tied = source->Tie(std::unique_ptr<OneDnnPackedWeights>(
      new OneDnnPackedWeights(key, packed_data, packed_bytes, live_bytes_)));
  auto weights = tied.Lock();
  lru_.push_front({std::move(key), std::move(tied)});
  entries_.emplace(lru_.front().key, lru_.begin());
  VLOG(1) << "oneDNN weight cache miss: source=" << weights->key_.source_data
          << " packed_bytes=" << packed_bytes;
  return {LookupKind::kMiss, std::move(weights)};
}

void OneDnnWeightCache::Complete(
    const std::shared_ptr<OneDnnPackedWeights>& weights, bool success) {
  absl::MutexLock lock(mu_);
  auto entry = entries_.find(weights->key_);
  if (entry == entries_.end() || entry->second->ready) {
    LOG(ERROR) << "oneDNN weight cache completion has no reservation";
    return;
  }

  if (!success || entry->second->weights.Expired()) {
    EraseLocked(entry->second);
    return;
  }

  entry->second->ready = true;
  ++stats_.insertions;
  VLOG(1) << "oneDNN weight cache insert: source=" << weights->key_.source_data
          << " packed_bytes=" << weights->size_bytes_;
}

OneDnnWeightCache::Stats OneDnnWeightCache::stats() const {
  absl::MutexLock lock(mu_);
  return {stats_.hits, stats_.insertions,
          live_bytes_->load(std::memory_order_relaxed)};
}

bool OneDnnWeightCache::TryEvictOneLocked() {
  for (auto it = lru_.end(); it != lru_.begin();) {
    --it;
    if (!it->ready) {
      continue;
    }
    auto weights = it->weights.Lock();
    // The source identity and this local reference are the only expected
    // owners. Active executions prevent eviction until they release theirs.
    if (weights && weights.use_count() != 2) {
      continue;
    }
    EraseLocked(it);
    return true;
  }
  return false;
}

void OneDnnWeightCache::EraseLocked(LruList::iterator it) {
  entries_.erase(it->key);
  lru_.erase(it);
}

void OneDnnWeightCache::PruneExpiredLocked() {
  for (auto it = lru_.begin(); it != lru_.end();) {
    auto current = it++;
    if (current->ready && current->weights.Expired()) {
      EraseLocked(current);
    }
  }
}

OneDnnWeightCache& GlobalOneDnnWeightCache() {
  static OneDnnWeightCache* cache = [] {
    int64_t capacity = 0;
    absl::Status status =
        tsl::ReadInt64FromEnvVar(kCapacityEnvVar, 0, &capacity);
    if (!status.ok()) {
      LOG(WARNING) << "Ignoring invalid " << kCapacityEnvVar << ": " << status;
      capacity = 0;
    } else if (capacity < 0) {
      LOG(WARNING) << "Ignoring negative " << kCapacityEnvVar << ": "
                   << capacity;
      capacity = 0;
    }
    if (capacity > 0) {
      LOG(INFO) << "oneDNN weight cache enabled with capacity " << capacity
                << " bytes";
    }
    return new OneDnnWeightCache(static_cast<size_t>(capacity));
  }();
  return *cache;
}

}  // namespace xla::cpu
