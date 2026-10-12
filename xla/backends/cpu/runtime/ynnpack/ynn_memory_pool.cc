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

#include "xla/backends/cpu/runtime/ynnpack/ynn_memory_pool.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <thread>  // NOLINT
#include <vector>

#include "absl/synchronization/mutex.h"
#include "slinky/base/util.h"
#include "slinky/runtime/memory_pool.h"

namespace xla::cpu {

YnnMemoryPool::~YnnMemoryPool() {
  absl::MutexLock lock(mu_);
  for (auto& [thread, pool] : pools_) {
    for (auto b = pool->evict_any(); b.ptr; b = pool->evict_any()) {
      slinky::deallocate_bytes(b.ptr);
    }
  }
}

slinky::memory_pool& YnnMemoryPool::ThreadPool() {
  std::unique_ptr<slinky::memory_pool>& pool =
      pools_[std::this_thread::get_id()];
  if (pool == nullptr) {
    pool = std::make_unique<slinky::memory_pool>();
  }
  return *pool;
}

void* YnnMemoryPool::Allocate(void* context, size_t size, size_t alignment) {
  YnnMemoryPool* self = static_cast<YnnMemoryPool*>(context);
  std::vector<void*> stale;
  {
    absl::MutexLock lock(self->mu_);
    slinky::memory_pool& pool = self->ThreadPool();
    slinky::memory_pool::block block = pool.allocate(size, alignment);
    if (block.ptr) {
      self->sizes_[block.ptr] = block.size;
      return block.ptr;
    }
    for (auto b = pool.evict_stale(); b.ptr; b = pool.evict_stale()) {
      stale.push_back(b.ptr);
    }
  }
  for (void* ptr : stale) {
    slinky::deallocate_bytes(ptr);
  }
  void* ptr = slinky::allocate_bytes(
      size, std::max(alignment, alignof(std::max_align_t)));
  if (ptr) {
    absl::MutexLock lock(self->mu_);
    self->sizes_[ptr] = size;
  }
  return ptr;
}

void YnnMemoryPool::Free(void* context, void* ptr, size_t size) {
  YnnMemoryPool* self = static_cast<YnnMemoryPool*>(context);
  absl::MutexLock lock(self->mu_);
  auto it = self->sizes_.find(ptr);
  self->ThreadPool().free(ptr, it->second);
  self->sizes_.erase(it);
}

size_t YnnMemoryPool::retained_bytes() const {
  absl::MutexLock lock(mu_);
  size_t bytes = 0;
  for (const auto& [thread, pool] : pools_) {
    bytes += pool->retained_size();
  }
  return bytes;
}

}  // namespace xla::cpu
