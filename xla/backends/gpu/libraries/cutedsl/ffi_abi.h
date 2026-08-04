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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_FFI_ABI_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_FFI_ABI_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace xla::gpu::cutedsl {

// A POD descriptor matching cutlass.jax.types.JaxArray. Generated CuTeDSL
// wrappers receive a pointer to one descriptor for each XLA buffer.
struct CuteXlaFfiBuffer {
  void* buffer;
  const int64_t* shape;
};

static_assert(std::is_standard_layout_v<CuteXlaFfiBuffer>);
static_assert(std::is_trivially_copyable_v<CuteXlaFfiBuffer>);
static_assert(alignof(CuteXlaFfiBuffer) == alignof(void*));
static_assert(sizeof(CuteXlaFfiBuffer) == 2 * sizeof(void*));
static_assert(offsetof(CuteXlaFfiBuffer, buffer) == 0);
static_assert(offsetof(CuteXlaFfiBuffer, shape) == sizeof(void*));

// A host-only POD decoded by cutlass.jax.collective's outer JIT wrapper.
struct alignas(8) CollectiveContextAbi {
  // Device array containing numeric peer or multimem addresses in
  // region-major order. Multimem rows repeat the rank-local alias.
  const uint64_t* peer_addresses;
  int32_t rank;
  int32_t clique_size;
};

static_assert(std::is_standard_layout_v<CollectiveContextAbi>);
static_assert(std::is_trivially_copyable_v<CollectiveContextAbi>);
static_assert(alignof(CollectiveContextAbi) == 8);
static_assert(sizeof(CollectiveContextAbi) == 16);
static_assert(offsetof(CollectiveContextAbi, peer_addresses) == 0);
static_assert(offsetof(CollectiveContextAbi, rank) == 8);
static_assert(offsetof(CollectiveContextAbi, clique_size) == 12);

}  // namespace xla::gpu::cutedsl

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_FFI_ABI_H_
