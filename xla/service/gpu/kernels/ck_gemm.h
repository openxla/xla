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

#ifndef XLA_SERVICE_GPU_KERNELS_CK_GEMM_H_
#define XLA_SERVICE_GPU_KERNELS_CK_GEMM_H_

#include <cstdint>
#include <optional>

namespace xla::gpu::kernel::gemm_universal {

// Use tag-based template specializations to avoid including
// ck_tile headers into regular libraries, and specialize templates in separate
// ROCM build targets that have no dependencies on other parts of XLA
struct F16xF16ToF16 {};
struct BF16xBF16ToBF16 {};

// Matches GemmKernelArguments in ck_tile
struct Arguments {
    const void* a_ptr;
    const void* b_ptr;
    void* c_ptr;
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t stride_A;
    int32_t stride_B;
    int32_t stride_C;
    int32_t k_batch;
};

struct ArgsIndices {
  int64_t lhs;
  int64_t rhs;
  int64_t out;
};

//===----------------------------------------------------------------------===//
// ck_tile Host Side Adaptor
//===----------------------------------------------------------------------===//

template <typename Tag>
struct Traits;

struct Dim3 {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};

// Type-erased adaptor that has all details required for launching
// a ck_tile kernel on a device. 
template <typename Tag>
class Adaptor {
 public:
  Dim3 BlockDim(int32_t m, int32_t n, int32_t k) const;
  Dim3 ThreadDim() const;
  void Initialize(void* params, const Arguments& args, int32_t device_sms) const;
};

//===----------------------------------------------------------------------===//
// ck_tile Device Side Adaptor
//===----------------------------------------------------------------------===//

template <typename Tag>
class DeviceKernel {
 public:
  void* symbol() const;
};

}  // namespace xla::gpu::kernel::gemm_universal

#endif  // XLA_SERVICE_GPU_KERNELS_CK_GEMM_H_
