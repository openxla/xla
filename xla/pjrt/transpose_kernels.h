/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_PJRT_TRANSPOSE_KERNELS_H_
#define XLA_PJRT_TRANSPOSE_KERNELS_H_

#include <cstdint>

namespace xla {

// Maximum number of elements along each dimension of a macroblock. When the
// inner microkernel block size `bs` is at least `kMaxOuterBlockElems`, the
// outer block dimensions (`outer_bs_a` and `outer_bs_b`) are always 1.
inline constexpr int kMaxOuterBlockElems = 16;

// Maximum input byte stride (`lda`) for which the 128-bit square microkernel
// is preferred over the 256-bit rectangular microkernel when `lda >= ldb`.
inline constexpr int kMaxSquare128StrideBytes = 64;

template <int packed>
void TransposeMacroKernelDispatchPacked(const char* a, int64_t lda,
                                        int outer_bs_a, char* b, int64_t ldb,
                                        int outer_bs_b);

template <typename T, int bs>
void TransposeMacroKernelDispatch(const char* a, int64_t lda, int outer_bs_a,
                                  char* b, int64_t ldb, int outer_bs_b) {
  constexpr int packed = sizeof(T) | (bs << 16);
  TransposeMacroKernelDispatchPacked<packed>(a, lda, outer_bs_a, b, ldb,
                                             outer_bs_b);
}

template <typename T, int bs>
void TransposeMicroKernelDispatch(const char* a, int64_t lda, char* b,
                                  int64_t ldb) {
  TransposeMacroKernelDispatch<T, bs>(a, lda, /*outer_bs_a=*/1, b, ldb,
                                      /*outer_bs_b=*/1);
}

}  // namespace xla

#endif  // XLA_PJRT_TRANSPOSE_KERNELS_H_
