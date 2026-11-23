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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_POLICY_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_POLICY_H_

#include <cstdint>

namespace xla::gpu {

// Global 64-bit offset type used by CUB's Onesweep histogram, exclusive-sum,
// and scatter kernels. `unsigned long long` is required because CUDA's
// `atomicAdd` has an `unsigned long long*` overload but no `unsigned long*`
// (`uint64_t*`) overload on Linux LP64.
using CubGlobalOffset =
    unsigned long long;  // NOLINT(runtime/int) NOLINT(google-runtime-int)
static_assert(sizeof(CubGlobalOffset) == 8);

// Maximum static shared memory per thread block assumed by CUB's
// RegBoundScaling (48 KiB).
inline constexpr int kCubSortMaxStaticSmemBytes = 48 * 1024;

// Onesweep always uses 8-bit radix digits (256 bins per pass).
inline constexpr int kCubSortOnesweepRadixBits = 8;
inline constexpr int kCubSortOnesweepRadixDigits = 1
                                                   << kCubSortOnesweepRadixBits;

// Exclusive sum over the 256 bins of each pass runs one 256-thread block per
// pass.
inline constexpr int kCubSortExclusiveSumBlockThreads = 256;

// Alignment in bytes for independent sub-allocations inside the CUB sort
// scratch buffer.
inline constexpr int64_t kCubSortScratchAlignmentBytes = 256;

// Upper bound on items per Onesweep portion before flooring to a multiple of
// the Onesweep tile size (matches CUB's DispatchRadixSort::InvokeOnesweep).
inline constexpr int32_t kCubSortMaxPortionLimit = (1 << 28) - 1;

// Thread-block geometry and radix bit width for a single radix sort kernel.
struct CubRadixSortBlockSpec {
  int block_threads;
  int items_per_thread;
  int radix_bits;

  constexpr int tile_items() const { return block_threads * items_per_thread; }
};

// Complete pinned tuning policy for all XLA-owned CUDA CUB radix sort kernels
// at a canonical SM tier. Both the host-side scratch/launch planner and the
// device-side XlaChainedPolicy construct their parameters from this struct so
// host and device stay strictly synchronized across CCCL versions.
//
// Note: Unlike CUB's default `policy_hub` (which disables Onesweep on SM50 and
// for 1B/2B keys on SM60..SM80), XLA intentionally uses Onesweep for all
// unbatched multi-tile sorts across SM50..SM100 to keep a single uniform
// unbatched multi-pass codepath within the 48 KiB static shared-memory budget.
struct CubRadixSortPolicySpec {
  int sm_version;

  // Scaled block geometries used by host planning and verified via
  // static_assert against CUB's policy structs on the device side.
  CubRadixSortBlockSpec single_tile;
  CubRadixSortBlockSpec histogram;
  CubRadixSortBlockSpec exclusive_sum;
  CubRadixSortBlockSpec onesweep;
  CubRadixSortBlockSpec segmented;
  CubRadixSortBlockSpec alt_segmented;

  // Nominal 4-byte parameters passed into CUB's AgentRadixSort*Policy
  // templates on the device side (which apply RegBoundScaling internally).
  int single_tile_nominal_threads;
  int single_tile_nominal_items;
  int histogram_nominal_4b_num_parts;
  int onesweep_nominal_threads;
  int onesweep_nominal_items;
  int onesweep_rank_num_parts;
  bool onesweep_use_raking_memoize;
  int segmented_nominal_threads;
  int segmented_nominal_items;
  int alt_segmented_nominal_threads;
  int alt_segmented_nominal_items;
  bool segmented_use_warp_transpose;
};

namespace detail {

constexpr int CeilDivInt(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

struct ScaledBlockDims {
  int block_threads;
  int items_per_thread;
};

// Mirrors cub::detail::RegBoundScaling<Nominal4ByteBlockThreads,
// Nominal4ByteItemsPerThread, DominantT>.
constexpr ScaledBlockDims RegBoundScaling(int nominal_4b_block_threads,
                                          int nominal_4b_items_per_thread,
                                          int dominant_byte_size) {
  int denom = dominant_byte_size > 4 ? dominant_byte_size : 4;
  int scaled_items = (nominal_4b_items_per_thread * 4) / denom;
  int items_per_thread = scaled_items > 1 ? scaled_items : 1;
  int smem_max_threads = CeilDivInt(kCubSortMaxStaticSmemBytes /
                                        (dominant_byte_size * items_per_thread),
                                    32) *
                         32;
  int block_threads = nominal_4b_block_threads < smem_max_threads
                          ? nominal_4b_block_threads
                          : smem_max_threads;
  return {block_threads, items_per_thread};
}

// Returns Hopper's (sm_90) pre-scaling nominal `(block_threads,
// items_per_thread)` Onesweep launch geometry for small (1- or 2-byte) keys
// with 8-byte global offsets, keyed by `(key_byte_size, value_byte_size)`
// (`value_byte_size == 0` for keys-only). The caller subsequently passes these
// nominal 4-byte dimensions through `RegBoundScaling` to compute the final
// block geometry. Mirrors `cub::detail::radix::sm90_small_key_tuning`.
constexpr ScaledBlockDims GetSm90SmallKeyNominalOnesweep(int key_byte_size,
                                                         int value_byte_size) {
  if (key_byte_size == 1) {
    switch (value_byte_size) {
      case 0:
        return {512, 19};
      case 1:
        return {448, 16};
      case 2:
        return {512, 14};
      case 4:
        return {512, 14};
      case 8:
        return {384, 18};
      case 16:
        return {512, 22};
      default:
        return {384, 23};
    }
  }
  if (key_byte_size == 2) {
    switch (value_byte_size) {
      case 0:
        return {512, 19};
      case 1:
        return {384, 16};
      case 2:
        return {448, 16};
      case 4:
        return {512, 12};
      case 8:
        return {512, 23};
      case 16:
        return {576, 22};
      default:
        return {384, 23};
    }
  }
  return {384, 23};
}

// Returns Blackwell's (sm_100) pre-scaling nominal `(block_threads,
// items_per_thread)` Onesweep launch geometry for small (1- or 2-byte) keys
// with 8-byte global offsets, keyed by `(key_byte_size, value_byte_size)`
// (`value_byte_size == 0` for keys-only). The caller subsequently passes these
// nominal 4-byte dimensions through `RegBoundScaling` to compute the final
// block geometry. Mirrors `cub::detail::radix::sm100_small_key_tuning`.
constexpr ScaledBlockDims GetSm100SmallKeyNominalOnesweep(int key_byte_size,
                                                          int value_byte_size) {
  if (key_byte_size == 1) {
    switch (value_byte_size) {
      case 0:
        return {512, 19};
      case 1:
        return {384, 20};
      case 2:
        return {256, 22};
      case 4:
        return {384, 15};
      case 8:
        return {384, 18};
      case 16:
        return {512, 22};
      default:
        return {384, 23};
    }
  }
  if (key_byte_size == 2) {
    switch (value_byte_size) {
      case 0:
        return {384, 20};
      case 1:
        return {384, 15};
      case 2:
        return {576, 15};
      case 4:
        return {512, 18};
      case 8:
        return {512, 23};
      case 16:
        return {576, 22};
      default:
        return {384, 23};
    }
  }
  return {384, 23};
}

}  // namespace detail

// Maps a CUDA compute capability major version to one of the pinned major-SM
// policy tiers {500, 600, 700, 800, 900, 1000}.
//
// Tiers branch strictly on the major version so that when FindCubinForArch
// selects a lower-minor CUBIN of the same major architecture (for example,
// sm_100 for sm_103 or sm_80 for sm_89), the CUBIN's active device policy is
// guaranteed to match the host planner's policy.
constexpr int CanonicalCubSortPolicySmVersion(int cc_major) {
  if (cc_major >= 10) {
    return 1000;
  }
  if (cc_major >= 9) {
    return 900;
  }
  if (cc_major >= 8) {
    return 800;
  }
  if (cc_major >= 7) {
    return 700;
  }
  if (cc_major >= 6) {
    return 600;
  }
  return 500;
}

// Returns the pinned tuning policy for the given canonical SM version,
// `key_byte_size` (1, 2, 4, or 8), `value_byte_size` (0 for keys-only, or
// 2, 4, 8 for key-value pairs), and whether the key type is IEEE-754 float32.
constexpr CubRadixSortPolicySpec GetCubRadixSortPolicySpec(
    int sm_version, int key_byte_size, int value_byte_size,
    bool is_float32_key) {
  const int dominant_byte_size =
      value_byte_size > key_byte_size ? value_byte_size : key_byte_size;
  const bool keys_only = value_byte_size == 0;

  const int single_tile_radix_bits = key_byte_size > 1 ? 6 : 5;
  const int segmented_radix_bits = key_byte_size > 1 ? 6 : 5;
  const int alt_segmented_radix_bits = segmented_radix_bits - 1;

  const int single_tile_nom_threads = 256;
  const int single_tile_nom_items = 19;
  const detail::ScaledBlockDims single_tile_dims = detail::RegBoundScaling(
      single_tile_nom_threads, single_tile_nom_items, dominant_byte_size);

  const int segmented_nom_threads = 192;
  const int segmented_nom_items = sm_version < 600 ? 31 : 39;
  const detail::ScaledBlockDims segmented_dims = detail::RegBoundScaling(
      segmented_nom_threads, segmented_nom_items, dominant_byte_size);

  const int alt_segmented_nom_threads = sm_version < 600 ? 256 : 384;
  const int alt_segmented_nom_items = 11;
  const detail::ScaledBlockDims alt_segmented_dims = detail::RegBoundScaling(
      alt_segmented_nom_threads, alt_segmented_nom_items, dominant_byte_size);

  const int histo_threads = sm_version >= 800 ? 128 : 256;
  const int histo_items = sm_version >= 800 ? 16 : 8;
  const int histo_nom_4b_parts =
      (sm_version >= 600 && sm_version < 800) ? 8 : 1;

  int onesweep_nom_threads = 256;
  int onesweep_nom_items = 21;
  int onesweep_rank_parts = 1;
  bool onesweep_raking_memoize = false;

  if (sm_version >= 900) {
    onesweep_rank_parts = 1;
    onesweep_raking_memoize = true;
    if (key_byte_size < 4) {
      detail::ScaledBlockDims small =
          sm_version >= 1000 ? detail::GetSm100SmallKeyNominalOnesweep(
                                   key_byte_size, value_byte_size)
                             : detail::GetSm90SmallKeyNominalOnesweep(
                                   key_byte_size, value_byte_size);
      onesweep_nom_threads = small.block_threads;
      onesweep_nom_items = small.items_per_thread;
    } else if (key_byte_size == 4) {
      onesweep_nom_threads = 384;
      if (keys_only) {
        onesweep_nom_items = 19 - (is_float32_key ? 1 : 0);
      } else {
        onesweep_nom_items = value_byte_size < 8 ? 17 : 29;
      }
    } else {
      onesweep_nom_threads = 384;
      onesweep_nom_items = value_byte_size < 8 ? 30 : 24;
    }
  } else if (sm_version >= 800) {
    onesweep_nom_threads = 384;
    onesweep_nom_items = (key_byte_size == 4 && !keys_only) ? 17 : 21;
    onesweep_rank_parts = 1;
    onesweep_raking_memoize = true;
  } else if (sm_version >= 700) {
    onesweep_nom_threads = 256;
    onesweep_nom_items = (key_byte_size == 4 && value_byte_size == 4) ? 46 : 23;
    onesweep_rank_parts = 4;
    onesweep_raking_memoize = false;
  } else if (sm_version >= 600) {
    onesweep_nom_threads = 256;
    onesweep_nom_items = 29;
    onesweep_rank_parts = 2;
    onesweep_raking_memoize = false;
  }

  const detail::ScaledBlockDims onesweep_dims = detail::RegBoundScaling(
      onesweep_nom_threads, onesweep_nom_items, dominant_byte_size);

  return CubRadixSortPolicySpec{
      /*sm_version=*/sm_version,
      /*single_tile=*/
      {single_tile_dims.block_threads, single_tile_dims.items_per_thread,
       single_tile_radix_bits},
      /*histogram=*/{histo_threads, histo_items, kCubSortOnesweepRadixBits},
      /*exclusive_sum=*/
      {kCubSortExclusiveSumBlockThreads, 1, kCubSortOnesweepRadixBits},
      /*onesweep=*/
      {onesweep_dims.block_threads, onesweep_dims.items_per_thread,
       kCubSortOnesweepRadixBits},
      /*segmented=*/
      {segmented_dims.block_threads, segmented_dims.items_per_thread,
       segmented_radix_bits},
      /*alt_segmented=*/
      {alt_segmented_dims.block_threads, alt_segmented_dims.items_per_thread,
       alt_segmented_radix_bits},
      /*single_tile_nominal_threads=*/single_tile_nom_threads,
      /*single_tile_nominal_items=*/single_tile_nom_items,
      /*histogram_nominal_4b_num_parts=*/histo_nom_4b_parts,
      /*onesweep_nominal_threads=*/onesweep_nom_threads,
      /*onesweep_nominal_items=*/onesweep_nom_items,
      /*onesweep_rank_num_parts=*/onesweep_rank_parts,
      /*onesweep_use_raking_memoize=*/onesweep_raking_memoize,
      /*segmented_nominal_threads=*/segmented_nom_threads,
      /*segmented_nominal_items=*/segmented_nom_items,
      /*alt_segmented_nominal_threads=*/alt_segmented_nom_threads,
      /*alt_segmented_nominal_items=*/alt_segmented_nom_items,
      /*segmented_use_warp_transpose=*/sm_version < 600,
  };
}

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_POLICY_H_
