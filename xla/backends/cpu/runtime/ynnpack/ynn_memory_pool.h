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

#ifndef XLA_BACKENDS_CPU_RUNTIME_YNNPACK_YNN_MEMORY_POOL_H_
#define XLA_BACKENDS_CPU_RUNTIME_YNNPACK_YNN_MEMORY_POOL_H_

#include <cstddef>
#include <memory>
#include <thread>  // NOLINT

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "slinky/runtime/memory_pool.h"

namespace xla::cpu {

// Allocates the internal buffers of the YNNPACK runtimes of one execution.
//
// A YNNPACK runtime frees its internal buffers at the end of every invocation,
// so a fusion invoked in a loop would fault the pages of its buffers in on
// every iteration. This keeps freed buffers in `slinky::memory_pool`s (the pool
// YNNPACK uses within one invocation) for the rest of the execution, and frees
// them all when destroyed at the end of the execution. Nothing is kept between
// executions.
//
// Each thread has its own pool, which keeps the buffers it freed, so a buffer
// is reused by the thread whose cache holds it. A pool reuses a freed buffer
// for a request of up to twice its size, and releases buffers that went unused
// through two requests it could not serve; buffers of a thread that stops
// allocating stay until the end of the execution. Thread-safe.
class YnnMemoryPool {
 public:
  YnnMemoryPool() = default;
  ~YnnMemoryPool();

  YnnMemoryPool(const YnnMemoryPool&) = delete;
  YnnMemoryPool& operator=(const YnnMemoryPool&) = delete;

  // Matches `ynn_allocate_fn` and `ynn_free_fn`, with `context` a pool.
  static void* Allocate(void* context, size_t size, size_t alignment);
  static void Free(void* context, void* ptr, size_t size);

  size_t retained_bytes() const;

 private:
  // The pool of the calling thread.
  slinky::memory_pool& ThreadPool() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::thread::id, std::unique_ptr<slinky::memory_pool>>
      pools_ ABSL_GUARDED_BY(mu_);
  // Sizes of the buffers in use, which can exceed the requested size of a
  // reused buffer.
  absl::flat_hash_map<void*, size_t> sizes_ ABSL_GUARDED_BY(mu_);
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_YNNPACK_YNN_MEMORY_POOL_H_
