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

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "xla/service/gpu/kernels/ck_gemm_adaptor.cu.h"
#include "xla/service/gpu/kernels/ck_gemm_kernel_helper.cu.h"

namespace xla::gpu::kernel::gemm_universal {

using CkGemmKernel = CkGemmKernelHelper<
  256, 256, 64, // M_Tile, N_Tile, K_Tile
  2,   2,   1,  // M_Warp, N_Warp, K_Warp
  32,  32,  16,  // M_Warp_Tile, N_Warp_Tile, K_Warp_Tile, 
  ck_tile::bf16_t, ck_tile::bf16_t, // ADataType, BDataType
  float, ck_tile::bf16_t, // AccDataType, CDataType
  ck_tile::tensor_layout::gemm::RowMajor,  // ALayout
  ck_tile::tensor_layout::gemm::RowMajor,  // BLayout
  ck_tile::tensor_layout::gemm::RowMajor>; // Chayout

XLA_GPU_DEFINE_CK_GEMM_TRAITS(BF16xBF16ToBF16, CkGemmKernel::GemmKernel);

template class Adaptor<BF16xBF16ToBF16>;
template class DeviceKernel<BF16xBF16ToBF16>;

}  // namespace xla::gpu::kernel::gemm_universal
