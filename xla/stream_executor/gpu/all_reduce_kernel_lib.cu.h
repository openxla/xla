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
#ifndef XLA_STREAM_EXECUTOR_GPU_ALL_REDUCE_KERNEL_LIB_CU_H_
#define XLA_STREAM_EXECUTOR_GPU_ALL_REDUCE_KERNEL_LIB_CU_H_

#include <array>
#include <cassert>
#include <cstdint>

#include "third_party/gpus/cuda/include/cuda/atomic"
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"

namespace stream_executor::gpu {

constexpr int64_t kNumElementsPerThread = 4;

template <typename T>
union Vec;

template <>
union alignas(16) Vec<float> {
  using PackedType = int4;

  float data[4];
  PackedType packed;
};

template <>
union alignas(8) Vec<__nv_bfloat16> {
  using PackedType = int2;

  __nv_bfloat16 data[4];
  PackedType packed;
};

template <typename T>
__device__ __forceinline__ Vec<T> VecLoad(T* addr) {
  Vec<T> vec;
  vec.packed = *(reinterpret_cast<typename Vec<T>::PackedType*>(addr));
  return vec;
}

template <typename T>
__device__ __forceinline__ void VecStore(T* addr, const Vec<T>& vec) {
  *(reinterpret_cast<typename Vec<T>::PackedType*>(addr)) = vec.packed;
}

template <typename T>
__device__ __forceinline__ void VecAdd(Vec<T>& res, const Vec<T>& vec) {
  res.data[0] += vec.data[0];
  res.data[1] += vec.data[1];
  res.data[2] += vec.data[2];
  res.data[3] += vec.data[3];
}

__device__ __forceinline__ bool CompareExchange(uint32_t* addr,
                                                uint32_t compare,
                                                uint32_t val) {
#if __CUDA_ARCH__ >= 600
  ::cuda::atomic_ref<uint32_t, ::cuda::thread_scope_system> ref(*addr);
  return ref.compare_exchange_strong(compare, val,
                                     ::cuda::memory_order_acq_rel);
#else
  assert(false);
  return true;
#endif
}

__device__ __forceinline__ void PutSignalFlag(uint32_t* addr) {
  while (!CompareExchange(addr, 0, 1)) {
  }
}

__device__ __forceinline__ void WaitSignalFlag(uint32_t* addr) {
  while (!CompareExchange(addr, 1, 0)) {
  }
}

__device__ __forceinline__ void SyncRemoteBlocks(
    std::array<uint32_t* __restrict__, kMaxNumAllReduceInputPtrs>
        signal_pad_ptrs,
    int64_t rank, int64_t num_ranks) {
  if (threadIdx.x < num_ranks) {
    auto target_rank = threadIdx.x;
    PutSignalFlag(signal_pad_ptrs[target_rank] + blockIdx.x * num_ranks + rank);
    WaitSignalFlag(signal_pad_ptrs[rank] + blockIdx.x * num_ranks +
                   target_rank);
  }
}

template <typename T>
__global__ void AllReduceKernelImpl(
    std::array<T* __restrict__, kMaxNumAllReduceInputPtrs> remote_input_ptrs,
    T* __restrict__ local_input_ptr, T* __restrict__ output_ptr, int64_t rank,
    int64_t num_ranks, int64_t num_elements,
    std::array<uint32_t* __restrict__, kMaxNumAllReduceInputPtrs>
        signal_flags_ptrs) {
  int64_t offset =
      kNumElementsPerThread * (blockIdx.x * blockDim.x + threadIdx.x);
  int64_t stride = kNumElementsPerThread * blockDim.x * gridDim.x;

  // Copy data from local input buffer to remote input buffer.
  for (int i = offset; i < num_elements; i += stride) {
    VecStore(remote_input_ptrs[rank] + i, VecLoad(local_input_ptr + i));
  }

  SyncRemoteBlocks(signal_flags_ptrs, rank, num_ranks);
  __syncthreads();

  for (int i = offset; i < num_elements; i += stride) {
    Vec<T> sum = VecLoad(remote_input_ptrs[0] + i);

    // Since `remote_input_ptrs` are provided in rank order, we get stable
    // reduction results on all devices.
#pragma unroll
    for (int j = 1; j < kMaxNumAllReduceInputPtrs; ++j) {
      if (j < num_ranks) {
        VecAdd(sum, VecLoad(remote_input_ptrs[j] + i));
      }
    }

    VecStore(output_ptr + i, sum);
  }

  __syncthreads();
  SyncRemoteBlocks(signal_flags_ptrs, rank, num_ranks);
}

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_ALL_REDUCE_KERNEL_LIB_CU_H_
