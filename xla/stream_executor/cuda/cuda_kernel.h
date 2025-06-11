/* Copyright 2019 The OpenXLA Authors.

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

// The CUDA implementation of the StreamExecutor functionality.
// CUDA inclusions are ideally confined to this implementation file.
//
// The notions from the StreamExecutor basically correspond to the CUDA streams
// programming model provided by the libcuda.so driver APIs, so we don't have
// to do much more than wrap the calls to the libraries appropriately.
#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_KERNEL_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_KERNEL_H_

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/gpu/gpu_kernel.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/logging.h"

namespace stream_executor::gpu {

template <typename ArgType>
inline absl::StatusOr<std::unique_ptr<KernelArgsPackedArrayBase>>
PackKernelArgs(absl::Span<const ArgType> args, uint32_t shared_mem_bytes) {
#if CUDA_VERSION >= 12010
  static constexpr int kKernelArgsLimit = 4095;
#else
  static constexpr int kKernelArgsLimit = 1024;
#endif
  if (args.size() > kKernelArgsLimit)
    return absl::InvalidArgumentError(absl::StrCat(
        "Can't pack device memory arguments array of size ", args.size(),
        " which is larger than the maximum supported size of ",
        kKernelArgsLimit));

  // Specialize kernel arguments array for small sizes to allocate a smaller
  // chunk of memory and hopefully hit a small allocations cache.
  if (args.size() <= 4) {
    return internal::PackKernelArgs<4>(args, shared_mem_bytes);
  } else if (args.size() <= 8) {
    return internal::PackKernelArgs<8>(args, shared_mem_bytes);
  } else if (args.size() <= 16) {
    return internal::PackKernelArgs<16>(args, shared_mem_bytes);
  } else if (args.size() <= 32) {
    return internal::PackKernelArgs<32>(args, shared_mem_bytes);
  } else if (args.size() <= 64) {
    return internal::PackKernelArgs<64>(args, shared_mem_bytes);
  } else if (args.size() <= 256) {
    return internal::PackKernelArgs<256>(args, shared_mem_bytes);
  } else if (args.size() <= 512) {
    return internal::PackKernelArgs<512>(args, shared_mem_bytes);
  }

  return internal::PackKernelArgs<kKernelArgsLimit>(args, shared_mem_bytes);
}

template <typename ArgType>
inline absl::StatusOr<std::unique_ptr<KernelArgsPackedArrayBase>>
PackKernelArgs(absl::Span<const ArgType> args, const KernelMetadata &metadata) {
  uint32_t shared_mem_bytes = metadata.shared_memory_bytes().value_or(0);
  return PackKernelArgs(args, shared_mem_bytes);
}

class CudaKernel : public Kernel {
 public:
  explicit CudaKernel(StreamExecutor *executor) : executor_(executor) {}

  // Note that the function is unloaded when the module is unloaded, and the
  // module that the function is contained in is owned by the StreamExecutor.
  ~CudaKernel() override { executor_->UnloadKernel(this); }

  // As arity cannot be reflected upon using the CUDA API, the arity is
  // explicitly set during the StreamExecutor::GetKernel initialization process.
  void set_arity(unsigned arity) { arity_ = arity; }
  unsigned Arity() const override { return arity_; }

  absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(
      ThreadDim threads, size_t dynamic_shared_memory_bytes) const override;

  // Simple accessor methods.
  CUfunction gpu_function() const { return gpu_function_; }
  void set_gpu_function(CUfunction gpu_function) {
    gpu_function_ = gpu_function;
  }

  // Collects metadata for the specified kernel.
  absl::StatusOr<KernelMetadata> GetKernelMetadata();

 private:
  absl::Status Launch(const ThreadDim &thread_dims, const BlockDim &block_dims,
                      const std::optional<ClusterDim> &cluster_dims,
                      Stream *stream, const KernelArgs &args) override;

  StreamExecutor *executor_ = nullptr;

  CUfunction gpu_function_ = nullptr;  // wrapped CUDA kernel handle
  unsigned arity_ = 0;  // number of formal parameters the kernel takes
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_KERNEL_H_
