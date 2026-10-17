/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/platform/threadpool.h"

#include <algorithm>
#include <atomic>
#include <cfenv>  // NOLINT
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/base/optimization.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/context.h"
#include "tsl/platform/denormal.h"
#include "tsl/platform/numa.h"
#include "tsl/platform/setround.h"
#include "tsl/platform/tracing.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/threadpool_interface.h"

#ifdef DNNL_AARCH64_USE_ACL
#include "tsl/platform/cpu_info.h"
#endif  // DNNL_AARCH64_USE_ACL

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

#ifdef TENSORFLOW_THREADSCALING_EXPERIMENTAL
ABSL_FLAG(float, tensorflow_num_threads_scale_factor, 1.0,
          "Allows to scale all Tensorflow ThreadPools. Total number of threads "
          "in a given ThreadPool equals to num_threads * "
          "tensorflow_num_threads_scale_factor. Default scale factor of 1 is a "
          "no-op.");
#endif  // TENSORFLOW_THREADSCALING_EXPERIMENTAL

namespace tsl::thread {

struct EigenEnvironment {
  using EnvThread = Thread;

  struct TaskImpl {
    std::function<void()> fn;
    Context context;
    uint64_t trace_id;
  };

  struct Task {
    Task() = default;

    Task(std::function<void()> fn, Context context, uint64_t trace_id)
        : f(TaskImpl{std::move(fn), std::move(context), trace_id}) {}

    Task(Task&&) = default;
    Task& operator=(Task&&) = default;

    std::optional<TaskImpl> f;
  };

  Env* const env;
  const ThreadOptions thread_options;
  const std::string name;

  EigenEnvironment(Env* env, const ThreadOptions& thread_options,
                   std::string name)
      : env(env), thread_options(thread_options), name(std::move(name)) {}

  EnvThread* CreateThread(std::function<void()> f) {
    return env->StartThread(thread_options, name, [this, f = std::move(f)]() {
      // Set the processor flag to flush denormals to zero.
      port::ScopedFlushDenormal flush;
      // Set the processor rounding mode to ROUND TO NEAREST.
      tsl::port::ScopedSetRound round(FE_TONEAREST);
      if (thread_options.numa_node != port::kNUMANoAffinity) {
        port::NUMASetThreadNodeAffinity(thread_options.numa_node);
      }
      f();
    });
  }

  Task CreateTask(std::function<void()> f) {
    uint64_t id = 0;
    if (ABSL_PREDICT_FALSE(tracing::EventCollector::IsEnabled())) {
      id = tracing::GetUniqueArg();
      tracing::RecordEvent(tracing::EventCategory::kScheduleClosure, id);
    }
    return Task(std::move(f), Context(ContextKind::kThread), id);
  }

  void ExecuteTask(const Task& t) {
    WithContext wc(t.f->context);
    tracing::ScopedRegion region(tracing::EventCategory::kRunClosure,
                                 t.f->trace_id);
    t.f->fn();
  }
};

ThreadPool::ThreadPool(Env* env, const std::string& name, int num_threads)
    : ThreadPool(env, ThreadOptions(), name, num_threads, true, nullptr) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const std::string& name, int num_threads)
    : ThreadPool(env, thread_options, name, num_threads, true, nullptr) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const std::string& name, int num_threads,
                       bool low_latency_hint, Eigen::Allocator* allocator)
    : executor_(this) {
  CHECK_GE(num_threads, 1);

#ifdef DNNL_AARCH64_USE_ACL
  // To avoid cost of swapping in and out threads from running processes
  // we do not use all available cores to parallelise TF operations.
  if (num_threads == tsl::port::NumTotalCPUs() && num_threads >= 16) {
    num_threads = num_threads - 1;
  }
#endif  // DNNL_AARCH64_USE_ACL

#ifdef TENSORFLOW_THREADSCALING_EXPERIMENTAL
  CHECK_GT(absl::GetFlag(FLAGS_tensorflow_num_threads_scale_factor), 0);
  num_threads *= absl::GetFlag(FLAGS_tensorflow_num_threads_scale_factor);
  if (num_threads < 1) num_threads = 1;
#endif  // TENSORFLOW_THREADSCALING_EXPERIMENTAL

  eigen_threadpool_ =
      std::make_unique<Eigen::ThreadPoolTempl<EigenEnvironment>>(
          num_threads, low_latency_hint,
          EigenEnvironment(env, thread_options, "tf_" + name));
  underlying_threadpool_ = eigen_threadpool_.get();
  threadpool_device_ = std::make_unique<Eigen::ThreadPoolDevice>(
      underlying_threadpool_, num_threads, allocator);
}

ThreadPool::ThreadPool(thread::ThreadPoolInterface* user_threadpool)
    : executor_(this) {
  underlying_threadpool_ = user_threadpool;
  threadpool_device_ = std::make_unique<Eigen::ThreadPoolDevice>(
      underlying_threadpool_, underlying_threadpool_->NumThreads(), nullptr);
}

ThreadPool::~ThreadPool() {}

void ThreadPool::Schedule(std::function<void()> fn) {
  CHECK(fn != nullptr);
  underlying_threadpool_->Schedule(std::move(fn));
}

int ThreadPool::NumShardsUsedByFixedBlockSizeScheduling(
    const int64_t total, const int64_t block_size) {
  if (block_size <= 0 || total <= 1 || total <= block_size ||
      NumThreads() == 1) {
    return 1;
  }
  return (total + block_size - 1) / block_size;
}

int ThreadPool::NumShardsUsedByTransformRangeConcurrently(
    const int64_t block_size, const int64_t total) {
  return NumShardsUsedByFixedBlockSizeScheduling(total, block_size);
}

void ThreadPool::ParallelFor(int64_t total,
                             const SchedulingParams& scheduling_params,
                             const std::function<void(int64_t, int64_t)>& fn) {
  switch (scheduling_params.strategy()) {
    case SchedulingStrategy::kAdaptive: {
      if (scheduling_params.cost_per_unit().has_value()) {
        ParallelFor(total, *scheduling_params.cost_per_unit(), fn);
      }
      break;
    }
    case SchedulingStrategy::kFixedBlockSize: {
      if (scheduling_params.block_size().has_value()) {
        ParallelForFixedBlockSizeScheduling(
            total, *scheduling_params.block_size(), fn);
      }
      break;
    }
  }
}

void ThreadPool::TransformRangeConcurrently(
    const int64_t block_size, const int64_t total,
    const std::function<void(int64_t, int64_t)>& fn) {
  ParallelFor(total,
              SchedulingParams(SchedulingStrategy::kFixedBlockSize,
                               /*cost_per_unit=*/std::nullopt, block_size),
              fn);
}

namespace {

// Shared state for `ParallelForFixedBlockSizeScheduling`.
//
// Blocks are claimed dynamically via an atomic index, by both the calling
// thread and any helper tasks scheduled onto the pool. This lets the caller
// actively work-steal blocks instead of passively blocking, which guarantees
// forward progress even if no pool thread is ever available to run a helper
// (e.g. a nested call made from inside a pool thread of a saturated or tiny
// pool).
//
// The state is reference counted because helper tasks may begin running (and
// discover that no blocks remain) after the caller has already returned.
class FixedBlockSizeState {
 public:
  FixedBlockSizeState(std::function<void(int64_t, int64_t)> fn, int64_t total,
                      int64_t block_size, int num_blocks)
      : fn_(std::move(fn)),
        total_(total),
        block_size_(block_size),
        num_blocks_(num_blocks),
        blocks_remaining_(num_blocks) {}

  // Claims and runs blocks until there are none left to claim.
  void RunBlocks() {
    while (true) {
      const int64_t block = next_block_.fetch_add(1, std::memory_order_relaxed);
      if (block >= num_blocks_) {
        return;
      }
      const int64_t first = block * block_size_;
      const int64_t last = std::min(first + block_size_, total_);
      fn_(first, last);
      blocks_remaining_.DecrementCount();
    }
  }

  // Returns true if every block has been claimed (but not necessarily
  // finished).
  bool AllBlocksClaimed() const {
    return next_block_.load(std::memory_order_relaxed) >= num_blocks_;
  }

  // Waits until every block has finished running.
  void WaitUntilAllBlocksFinished() { blocks_remaining_.Wait(); }

 private:
  const std::function<void(int64_t, int64_t)> fn_;
  const int64_t total_;
  const int64_t block_size_;
  const int64_t num_blocks_;

  // Index of the next block to be claimed.
  std::atomic<int64_t> next_block_{0};

  // Number of blocks that have not yet finished running.
  BlockingCounter blocks_remaining_;
};

}  // namespace

// This functionality is similar to parallelFor, except that reasoning about
// the number of shards used is significantly easier.
void ThreadPool::ParallelForFixedBlockSizeScheduling(
    const int64_t total, const int64_t block_size,
    const std::function<void(int64_t, int64_t)>& fn) {
  const int num_shards_used =
      NumShardsUsedByFixedBlockSizeScheduling(total, block_size);
  if (num_shards_used == 1) {
    fn(0, total);
    return;
  }

  auto state = std::make_shared<FixedBlockSizeState>(fn, total, block_size,
                                                     num_shards_used);

  // The caller always participates, so schedule enough helpers such that at
  // most `NumThreads()` threads execute blocks concurrently (matching the
  // previous behavior).
  const int num_helpers = std::min(num_shards_used, NumThreads()) - 1;
  for (int i = 0; i < num_helpers; ++i) {
    // Stop scheduling once already started helpers have claimed everything;
    // additional helpers would just be no-ops.
    if (state->AllBlocksClaimed()) {
      break;
    }
    Schedule([state]() { state->RunBlocks(); });
  }

  // Instead of blocking, the caller works through any blocks that have not
  // yet been claimed by a helper. If the pool is too small (or fully
  // occupied, e.g. by callers blocked in this function) the caller ends up
  // running every block itself rather than deadlocking.
  state->RunBlocks();

  // All blocks have been claimed; wait for blocks that are still running on
  // helper threads. These are actively executing, so this always terminates.
  state->WaitUntilAllBlocksFinished();
}

void ThreadPool::ParallelFor(int64_t total, int64_t cost_per_unit,
                             const std::function<void(int64_t, int64_t)>& fn) {
  CHECK_GE(total, 0);
  CHECK_EQ(total, (int64_t)(Eigen::Index)total);
  threadpool_device_->parallelFor(
      total, Eigen::TensorOpCost(0, 0, cost_per_unit),
      [&fn](Eigen::Index first, Eigen::Index last) { fn(first, last); });
}

void ThreadPool::ParallelForWithWorkerId(
    int64_t total, const SchedulingParams& scheduling_params,
    const std::function<void(int64_t, int64_t, int)>& fn) {
  ParallelFor(total, scheduling_params,
              [this, &fn](int64_t start, int64_t limit) {
                // We may use the current thread to do some work synchronously.
                // When calling CurrentThreadId() from outside of the thread
                // pool, we get -1, so we can shift every id up by 1.
                int id = CurrentThreadId() + 1;
                fn(start, limit, id);
              });
}

int ThreadPool::NumThreads() const {
  return underlying_threadpool_->NumThreads();
}

int ThreadPool::CurrentThreadId() const {
  return underlying_threadpool_->CurrentThreadId();
}

void ThreadPool::ScheduleWithHint(std::function<void()> fn, int start,
                                  int limit) {
  underlying_threadpool_->ScheduleWithHint(std::move(fn), start, limit);
}

Eigen::ThreadPoolInterface* ThreadPool::AsEigenThreadPool() const {
  DCHECK(underlying_threadpool_ != nullptr);
  return underlying_threadpool_;
}

tsl::Executor* absl_nonnull ThreadPool::AsExecutor() { return &executor_; }

ThreadPool::ThreadPoolExecutor::ThreadPoolExecutor(ThreadPool* thread_pool)
    : thread_pool_(thread_pool) {}

void ThreadPool::ThreadPoolExecutor::Execute(Task task) {
  auto* task_ptr = new Task(std::move(task));
  thread_pool_->Schedule([task_ptr] {
    std::move((*task_ptr))();
    delete task_ptr;
  });
}

}  // namespace tsl::thread
