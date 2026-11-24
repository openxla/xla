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

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wpass-failed"
#endif

#include <cstdint>

#include "cub/block/block_scan.cuh"
#include "xla/backends/gpu/libraries/cub/cub_sort_kernels_impl.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_policy.h"

namespace xla::gpu::internal {
namespace {

// Out-of-place exclusive prefix sum over the 256 histogram bins of each
// Onesweep pass (one 256-thread block per pass), plus zero-initialization of
// the per-(portion, pass) dynamic tile counters (strided by 256 bytes so each
// Onesweep launch owns a disjoint slice). Out-of-place I/O keeps
// CustomKernelThunk::buffer_uses() strictly pure-Read / pure-Write and avoids
// separate per-pass MemZero launches for `d_ctrs`.
__launch_bounds__(kCubSortExclusiveSumBlockThreads) __global__
    void CubRadixSortExclusiveSumKernel(
        const CubGlobalOffset* __restrict__ d_bins_in,
        CubGlobalOffset* __restrict__ d_bins_out,
        int32_t* __restrict__ d_ctrs_out, int32_t num_portions) {
  static_assert(kCubSortOnesweepRadixDigits ==
                kCubSortExclusiveSumBlockThreads);
  using BlockScan =
      cub::BlockScan<CubGlobalOffset, kCubSortExclusiveSumBlockThreads>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  const int bin_offset =
      static_cast<int>(blockIdx.x) * kCubSortOnesweepRadixDigits +
      static_cast<int>(threadIdx.x);
  CubGlobalOffset bin = d_bins_in[bin_offset];
  BlockScan(temp_storage).ExclusiveSum(bin, bin);
  d_bins_out[bin_offset] = bin;

  constexpr int64_t kCtrStrideInt32 =
      kCubSortScratchAlignmentBytes / static_cast<int64_t>(sizeof(int32_t));
  for (int64_t p = static_cast<int64_t>(threadIdx.x);
       p < static_cast<int64_t>(num_portions);
       p += kCubSortExclusiveSumBlockThreads) {
    const int64_t ctr_slot =
        p * static_cast<int64_t>(gridDim.x) + static_cast<int64_t>(blockIdx.x);
    d_ctrs_out[ctr_slot * kCtrStrideInt32] = 0;
  }
}

}  // namespace

KernelHostRef GetCubSortExclusiveSumRef() {
  return {reinterpret_cast<const void*>(&CubRadixSortExclusiveSumKernel),
          /*arity=*/4};
}

}  // namespace xla::gpu::internal
