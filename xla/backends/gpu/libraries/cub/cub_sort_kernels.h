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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_KERNELS_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_KERNELS_H_

#include <optional>

#include "absl/status/statusor.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_elf_utils.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Embedded KernelLoaderSpecs and optional static histogram attributes required
// to emit a multi-tile 1D Onesweep radix sort.
struct CubSortOnesweepKernelSpecs {
  stream_executor::KernelLoaderSpec histogram;
  // Statically extracted from the matched CUBIN ELF when present; nullopt when
  // falling back to PTX JIT (callers should use a conservative occupancy
  // fallback).
  std::optional<stream_executor::cuda::CudaKernelFuncAttributes>
      histogram_attrs;
  stream_executor::KernelLoaderSpec exclusive_sum;
  stream_executor::KernelLoaderSpec onesweep;
};

// Returns the embedded KernelLoaderSpec for the unified SingleTile radix sort
// kernel (handling both unbatched and batched sorts whose segment size fits
// within `CubRadixSortPolicySpec::single_tile.tile_items()`).
//
// Prefers the single-architecture CUBIN ELF matching `compute_capability`.
// When no matching CUBIN is embedded (for example, PTX-only builds), falls back
// to the fatbinary only if the highest JIT-eligible PTX entry maps to the same
// canonical policy tier as `CanonicalCubSortPolicySmVersion(cc.major)`.
//
// Kernel ABI:
// - Keys-only (arity 3): `(const Key* __restrict__ d_keys_in,
//                          Key* __restrict__ d_keys_out,
//                          int32_t segment_size)`
// - Pairs (arity 5):     `(const Key* __restrict__ d_keys_in,
//                          Key* __restrict__ d_keys_out,
//                          const Value* __restrict__ d_values_in,
//                          Value* __restrict__ d_values_out,
//                          int32_t segment_size)`
// Launch geometry: `BlockDim(num_segments)`,
// `ThreadDim(policy.single_tile.block_threads)`, with
// `0 <= segment_size <= policy.single_tile.tile_items()`. Input and output
// buffers must be non-aliasing (`__restrict__`).
absl::StatusOr<stream_executor::KernelLoaderSpec>
GetCubSortSingleTileKernelSpec(
    PrimitiveType key_type, std::optional<PrimitiveType> value_type,
    bool descending,
    const stream_executor::CudaComputeCapability& compute_capability);

// Returns the single-architecture CUBIN KernelLoaderSpecs (plus statically
// parsed histogram function attributes for occupancy calculation) for a
// multi-tile unbatched Onesweep radix sort.
absl::StatusOr<CubSortOnesweepKernelSpecs> GetCubSortOnesweepKernelSpecs(
    PrimitiveType key_type, std::optional<PrimitiveType> value_type,
    bool descending,
    const stream_executor::CudaComputeCapability& compute_capability);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_KERNELS_H_
