/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_ONEDNN_ONEDNN_THREADPOOL_H_
#define XLA_BACKENDS_CPU_RUNTIME_ONEDNN_ONEDNN_THREADPOOL_H_

#include <cstddef>
#include <cstdint>
#include <functional>

#include "dnnl_threadpool.hpp"
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
#include "xla/backends/cpu/runtime/parallel_loop_runner.h"
#include "xla/backends/cpu/runtime/work_queue.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

class OneDnnThreadPool final
    : public dnnl::threadpool_interop::threadpool_iface {
 public:
  explicit OneDnnThreadPool(Eigen::ThreadPoolInterface* thread_pool)
      : thread_pool_(thread_pool) {}
  explicit OneDnnThreadPool(ParallelLoopRunner* runner) : runner_(runner) {
    use_runner_ = true;
    dnnl_threadpool_interop_set_max_concurrency(runner_->num_threads());
  }

  int get_num_threads() const final {
    if (use_runner_) {
      return runner_->num_threads();
    }
    return thread_pool_->NumThreads();
  }

  bool get_in_parallel() const final {
    if (use_runner_) {
      // TODO(intel-tf): this is a temporary fix without which
      // oneDNN runs single-threaded.
      return false;
    }
    return thread_pool_->CurrentThreadId() >= 0;
  }

  uint64_t get_flags() const final {
    if (use_runner_) {
      // oneDNN is asynchronous when using oneDNN custom call
      // XLA implementation with thunk runtime.
      return ASYNCHRONOUS;
    }
    return 0;
  }

#ifdef ENABLE_ONEDNN_ASYNC
  // The wait() method only exists with oneDNN's experimental support
  // for asynchronous execution determined by the ENABLE_ONEDNN_ASYNC.
  void wait() override {
    if (use_runner_) {
      // While performing asynchronous execution, wait() method is
      // needed to notify the user that the output is ready.
      // oneDNN will not call wait() inside the library to avoid deadlock.
      tsl::BlockUntilReady(runner_->done_event());
    }
  }
#endif  // ENABLE_ONEDNN_ASYNC

  void parallel_for(int n, const std::function<void(int, int)>& fn) final {
    if (use_runner_) {
      // oneDNN ThreadPool instance is only created with ParallelLoopRunner
      // when using the asynchronous execution of oneDNN custom calls.
      // Therefore, we don't block here.
      runner_->Parallelize(
          ParallelLoopRunner::RangeDim{static_cast<size_t>(n)},
          [fn, n](ParallelLoopRunner::RangeIndex i) { fn(i.offset, n); });
      return;
    }

    // It is perfectly safe to block here as Worker implements work stealing
    // that guarantees forward progress and deadlock freedom, even if we are
    // running in the same thread pool as the Eigen thread_pool.
    tsl::BlockUntilReady(Worker::Parallelize(thread_pool_,
                                             thread_pool_->NumThreads(), n,
                                             [fn, n](size_t i) { fn(i, n); }));
  }

  const void set_thread_pool(Eigen::ThreadPoolInterface* thread_pool) {
    thread_pool_ = thread_pool;
  }

 private:
  Eigen::ThreadPoolInterface* thread_pool_;
  ParallelLoopRunner* runner_;
  bool use_runner_ =
      false;  // indicates if we are using ParallelLoopRunner
              // which is used to return a future from the
              // oneDNN custom call's FFI handler for asynchronous execution.
              // TODO(intel-tf): Remove this flag when
              // oneDNN supports asynchronous execution by default.
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_ONEDNN_ONEDNN_THREADPOOL_H_
