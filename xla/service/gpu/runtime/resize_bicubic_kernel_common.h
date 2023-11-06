/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_RESIZE_BICUBIC_KERNEL_COMMON_H_
#define XLA_SERVICE_GPU_RUNTIME_RESIZE_BICUBIC_KERNEL_COMMON_H_

// Contains shared declarations between resize_bicubic_kernel.cc and resize_bicubic_kernel.cu.cc
// but avoids including ABSL, etc. which some CUDA compilers cannot
// handle.

namespace xla::gpu {

constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK = 1024;
constexpr uint32_t CUDA_THREADS_PER_BLOCK_FALLBACK = 256;

template <typename scalar_t>
void* GetResizeBicubicKernel();

template <typename scalar_t>
void* GetResizeBicubicGradKernel();

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_RESIZE_BICUBIC_KERNEL_COMMON_H_
