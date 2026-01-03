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

#ifndef XLA_PJRT_GPU_TFRT_STREAM_ACCESSOR_GUARD_H_
#define XLA_PJRT_GPU_TFRT_STREAM_ACCESSOR_GUARD_H_

#include "absl/log/check.h"

namespace xla {

// RAII handle that guards against `se::Stream` and `se::StreamExecutor` access
// on the current thread.
//
// When this handle is alive on a thread, calling a stream accessor on the same
// thread will cause the process to crash. This is used in tests to validate
// that TFRT GPU does not call CUDA inline, which is important because of the
// following reasons:
//
// 1. CUDA uses synchronization primitives that are not compatible with
//    cooperatively scheduled threads. For example, Google's fiber scheduler
//    will not be aware of blocking code inside CUDA, which results in
//    suboptimal scheduling decisions.
//
// 2. TFRT GPU's goal is to perform GPU operations asynchronously. If CUDA API
//    is being called inline, this is likely an indication that the work is not
//    being performed asynchronously by accident.
//
// Guarding against stream and stream executor accessors is sufficient because
// these are the only ways for TFRT GPU to invoke CUDA APIs.
class TfrtGpuStreamAccessorGuard {
 public:
  TfrtGpuStreamAccessorGuard() { ++depth_; }
  ~TfrtGpuStreamAccessorGuard() { --depth_; }

  TfrtGpuStreamAccessorGuard(const TfrtGpuStreamAccessorGuard&) = delete;
  TfrtGpuStreamAccessorGuard& operator=(const TfrtGpuStreamAccessorGuard&) =
      delete;

  static bool IsAllowed() { return depth_ == 0; }

  static void AssertAllowed() {
    CHECK(IsAllowed())
        << "TFRT GPU requires stream accessor to be never called inline on the "
           "PjRt API caller's thread; call CUDA APIs on the `AsyncWorkRunner` "
           "owned by `TfrtGpuClient` instead";
  }

 private:
  static thread_local int depth_;
};

}  // namespace xla

#endif  // XLA_PJRT_GPU_TFRT_STREAM_ACCESSOR_GUARD_H_
