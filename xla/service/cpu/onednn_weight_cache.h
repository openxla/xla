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

#ifndef XLA_SERVICE_CPU_ONEDNN_WEIGHT_CACHE_H_
#define XLA_SERVICE_CPU_ONEDNN_WEIGHT_CACHE_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/synchronization/mutex.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "xla/tsl/util/tied_ref.h"

namespace xla::cpu {

// Identifies a packed representation without retaining the source allocation.
struct OneDnnWeightCacheKey {
  // Distinguishes slices that share the same content generation.
  const void* source_data = nullptr;
  dnnl::memory::desc source_desc;
  dnnl::memory::desc packed_desc;
  // Ownership identifies a content generation, not equality of matrix values.
  std::weak_ptr<tsl::TiedAny> source_identity;

  bool operator==(const OneDnnWeightCacheKey& other) const {
    return source_data == other.source_data &&
           source_desc == other.source_desc &&
           packed_desc == other.packed_desc &&
           !source_identity.owner_before(other.source_identity) &&
           !other.source_identity.owner_before(source_identity);
  }

  template <typename H>
  friend H AbslHashValue(H h, const OneDnnWeightCacheKey& key) {
    // oneDNN exposes semantic equality for memory descriptors but no matching
    // hash. Hash the source address and let equality resolve collisions
    // between layouts or generations of the same allocation.
    return H::combine(std::move(h), key.source_data);
  }
};

class OneDnnPackedWeights {
 public:
  ~OneDnnPackedWeights();
  OneDnnPackedWeights(const OneDnnPackedWeights&) = delete;
  OneDnnPackedWeights& operator=(const OneDnnPackedWeights&) = delete;

  void* data() const { return data_; }

 private:
  friend class OneDnnWeightCache;
  OneDnnPackedWeights(OneDnnWeightCacheKey key, void* data, size_t size_bytes,
                      std::shared_ptr<std::atomic<size_t>> live_bytes)
      : key_(std::move(key)),
        data_(data),
        size_bytes_(size_bytes),
        live_bytes_(std::move(live_bytes)) {}

  OneDnnWeightCacheKey key_;
  void* data_;
  size_t size_bytes_;
  std::shared_ptr<std::atomic<size_t>> live_bytes_;
};

// Byte-bounded LRU. Packed weights are tied to the source content identity;
// invalidation or source release frees them after active executions finish.
// Concurrent misses for one key bypass until Complete() publishes its entry.
// A source identity belongs to one cache; its Tie/Lock/release operations are
// serialized by that cache's mutex. The runtime uses the process-global cache.
class OneDnnWeightCache {
 public:
  enum class LookupKind { kHit, kMiss, kBypass };

  struct LookupResult {
    LookupKind kind;
    std::shared_ptr<OneDnnPackedWeights> weights;
  };

  struct Stats {
    uint64_t hits = 0;
    uint64_t insertions = 0;
    // Packed payload, including pending and execution-held allocations;
    // excludes source storage and cache metadata.
    size_t live_bytes = 0;
  };

  explicit OneDnnWeightCache(size_t capacity_bytes)
      : capacity_bytes_(capacity_bytes) {}

  size_t capacity_bytes() const { return capacity_bytes_; }

  LookupResult LookupOrCreate(OneDnnWeightCacheKey key);

  // Publishes a successfully reordered entry or releases its reservation.
  // Must be called exactly once for every kMiss result.
  void Complete(const std::shared_ptr<OneDnnPackedWeights>& weights,
                bool success);

  Stats stats() const;

 private:
  struct Entry {
    OneDnnWeightCacheKey key;
    tsl::TiedRef<OneDnnPackedWeights> weights;
    bool ready = false;
  };
  using LruList = std::list<Entry>;

  bool TryEvictOneLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void EraseLocked(LruList::iterator it) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void PruneExpiredLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const size_t capacity_bytes_;
  // Executions can retain packed storage after the cache itself is destroyed.
  const std::shared_ptr<std::atomic<size_t>> live_bytes_ =
      std::make_shared<std::atomic<size_t>>(0);
  mutable absl::Mutex mu_;
  LruList lru_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<OneDnnWeightCacheKey, LruList::iterator> entries_
      ABSL_GUARDED_BY(mu_);
  Stats stats_ ABSL_GUARDED_BY(mu_);
};

// Returns a process-global cache whose capacity is read once from
// XLA_ONEDNN_WEIGHT_CACHE_CAPACITY_BYTES. The default capacity is zero.
OneDnnWeightCache& GlobalOneDnnWeightCache();

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_ONEDNN_WEIGHT_CACHE_H_
