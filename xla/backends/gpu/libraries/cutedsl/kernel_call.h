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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_KERNEL_CALL_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_KERNEL_CALL_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/libraries/cutedsl/ffi_abi.h"
#include "xla/backends/gpu/libraries/cutedsl/module.h"
#include "xla/ffi/ffi.h"

namespace xla::gpu::cutedsl {

// A loaded function whose module is retained for the function's lifetime.
class LoadedKernel {
 public:
  LoadedKernel(const LoadedKernel&) = default;
  LoadedKernel& operator=(const LoadedKernel&) = default;
  LoadedKernel(LoadedKernel&&) = default;
  LoadedKernel& operator=(LoadedKernel&&) = default;

  absl::Status Run(void** arguments, size_t argument_count) const;

 private:
  friend class KernelCallState;

  LoadedKernel(std::shared_ptr<LoadedModule> module,
               LoadedModule::FunctionHandle function)
      : module_(std::move(module)), function_(function) {}

  std::shared_ptr<LoadedModule> module_;
  LoadedModule::FunctionHandle function_;
};

// Owns the module image and lazily prepares its generated kernel. FFI instance
// state is shared across executions, so all mutable access is synchronized.
class KernelCallState {
 public:
  explicit KernelCallState(ModuleImage image) : image_(std::move(image)) {}

  absl::StatusOr<LoadedKernel> Prepare();
  absl::StatusOr<LoadedKernel> prepared_kernel() const;

 private:
  ModuleImage image_;
  ModuleLoader loader_;
  mutable absl::Mutex mutex_;
  std::optional<LoadedKernel> kernel_ ABSL_GUARDED_BY(mutex_);
};

// Owns the buffer descriptors for one generated kernel invocation.
class KernelArguments {
 public:
  static absl::StatusOr<KernelArguments> Create(ffi::RemainingArgs arguments,
                                                ffi::RemainingRets results);

  KernelArguments(const KernelArguments&) = delete;
  KernelArguments& operator=(const KernelArguments&) = delete;
  KernelArguments(KernelArguments&&) = default;
  KernelArguments& operator=(KernelArguments&&) = default;

  absl::Status Run(const LoadedKernel& kernel, void* stream);
  absl::Status Run(const LoadedKernel& kernel, void* stream,
                   CollectiveContextAbi& collective_context);

 private:
  static constexpr size_t kInlineBufferCount = 8;

  KernelArguments() = default;

  absl::InlinedVector<CuteXlaFfiBuffer, kInlineBufferCount> buffers_;
};

}  // namespace xla::gpu::cutedsl

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_KERNEL_CALL_H_
