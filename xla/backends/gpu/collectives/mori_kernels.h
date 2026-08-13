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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_MORI_KERNELS_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_MORI_KERNELS_H_

#include <cstddef>
#include <cstdint>

// The CollectivesFacade owns the per-device staging + Run* entry points. Host
// includers (mori_communicator.cc) see decl-only Run* templates; the device TU
// (mori_kernels.cu.cc, compiled as HIP) pulls in the full device path and emits
// the explicit float instantiations that resolve the host's references.
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

// By default the wiring builds against an inert stub facade (mori_stub.h) so it
// compiles/links without the @roc_mori library. Define XLA_GPU_USE_REAL_MORI to
// pull in the real device facade instead (and restore the @roc_mori deps in the
// mori_kernels BUILD target).
#if defined(XLA_GPU_USE_REAL_MORI)
#include "mori/collective/collectives_facade.hpp"
#else
#include "xla/backends/gpu/collectives/mori_stub.h"
#endif

// Shared (PrimitiveType -> C++ storage type) and (ReductionKind -> functor)
// tables used by both the host communicator (dispatch) and the device TU
// (explicit instantiations), so the two agree on every symbol.

// X(PT, CT): PT = xla::PrimitiveType enumerator, CT = C++ storage type.
#define MORI_FOR_EACH_DTYPE(X) \
  X(F16, __half)               \
  X(BF16, hip_bfloat16)        \
  X(S8, int8_t)                \
  X(U8, uint8_t)               \
  X(S32, int32_t)              \
  X(U32, uint32_t)             \
  X(S64, int64_t)              \
  X(U64, uint64_t)             \
  X(F32, float)                \
  X(F64, double)

// X(PT, CT, RK, OP): RK = xla::ReductionKind enumerator, OP = functor template.
#define MORI_FOR_EACH_OP(X, PT, CT) \
  X(PT, CT, SUM, ::SumOp)           \
  X(PT, CT, MIN, ::MinOp)           \
  X(PT, CT, MAX, ::MaxOp)           \
  X(PT, CT, PRODUCT, ::ProdOp)

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_MORI_KERNELS_H_
