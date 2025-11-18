/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_KERNELS_CK_GEMM_ADAPTOR_CU_H_
#define XLA_SERVICE_GPU_KERNELS_CK_GEMM_ADAPTOR_CU_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "ck_tile/host/kernel_launch.hpp"

#include "xla/service/gpu/kernels/ck_gemm.h"

namespace xla::gpu::kernel::gemm_universal {

// Template library implementing an adaptor from a ck_tile kernel to StreamExecutor primitives

template <typename Tag>
Dim3 Adaptor<Tag>::ThreadDim() const {
  auto block_shape = Traits<Tag>::Kernel::BlockSize();
  return Dim3{block_shape.x, block_shape.y, block_shape.z};
}

template <typename Tag>
Dim3 Adaptor<Tag>::BlockDim(int32_t m, int32_t n, int32_t k) const {
  auto grid = Traits<Tag>::Kernel::GridSize(m, n, /*k_batch=*/1);
  return Dim3{grid.x, grid.y, grid.z};
}

template <typename Tag>
void Adaptor<Tag>::Initialize(void *params, const Arguments &args,
                              int32_t device_sms) const {
  // Convert ck operation arguments to a device kernel parameters.
  new (params) typename Traits<Tag>::Arguments {
    args.a_ptr,
    args.b_ptr,
    args.c_ptr,
    args.M,
    args.N,
    args.K,
    args.stride_A,
    args.stride_B,
    args.stride_C,
    args.k_batch
  };
}

template <typename Tag>
void *DeviceKernel<Tag>::symbol() const {
  using Kernel = typename Traits<Tag>::Kernel;
  using Arguments = typename Traits<Tag>::Arguments;
  return reinterpret_cast<void *>(ck_tile::kentry<CK_TILE_MAX_THREAD_PER_BLOCK, CK_TILE_MIN_BLOCK_PER_CU, Kernel, Arguments>);
};

//===----------------------------------------------------------------------===//
// ck_tile kernel traits helper
//===----------------------------------------------------------------------===//

#define XLA_GPU_DEFINE_CK_GEMM_TRAITS(TAG, KERNEL) \
  template <>                                              \
  struct Traits<TAG> {                                     \
    using Kernel = KERNEL;                             \
    using Arguments = typename Kernel::GemmKernelArgs;       \
  }

}  // namespace xla::gpu::kernel::gemm_universal

#endif  // XLA_SERVICE_GPU_KERNELS_CK_GEMM_ADAPTOR_CU_H_
