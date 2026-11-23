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

#include "xla/backends/gpu/libraries/cub/cub_sort_kernels_impl.h"

#include <cstdint>
#include <cstring>
#include <type_traits>

// Clang can't always unroll all CUB loops; silence the warning to avoid build
// breaks with -Werror (matches stream_executor/cuda/cub_sort_kernel_cuda_impl).
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wpass-failed"
#endif

#include "cub/agent/agent_radix_sort_downsweep.cuh"
#include "cub/agent/agent_radix_sort_histogram.cuh"
#include "cub/agent/agent_radix_sort_onesweep.cuh"
#include "cub/agent/agent_radix_sort_upsweep.cuh"
#include "cub/block/block_load.cuh"
#include "cub/block/block_radix_sort.cuh"
#include "cub/block/block_scan.cuh"
#include "cub/util_device.cuh"
#include "cub/util_ptx.cuh"
#include "cub/util_type.cuh"
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_policy.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::internal {
namespace {

// Device-side CUB policy constructed directly from GetCubRadixSortPolicySpec so
// that host planning and compiled device CUBINs stay strictly locked across
// CCCL versions.
template <int kSmVersion, typename KeyT, typename ValueT>
struct XlaCubRadixSortPolicy {
  static constexpr bool kKeysOnly = std::is_same_v<ValueT, cub::NullType>;
  static constexpr int kValueByteSize =
      kKeysOnly ? 0 : static_cast<int>(sizeof(ValueT));
  using DominantType =
      std::conditional_t<(sizeof(ValueT) > sizeof(KeyT)), ValueT, KeyT>;

  static constexpr CubRadixSortPolicySpec kSpec =
      GetCubRadixSortPolicySpec(kSmVersion, static_cast<int>(sizeof(KeyT)),
                                kValueByteSize, std::is_same_v<KeyT, float>);

  using SingleTilePolicy = cub::AgentRadixSortDownsweepPolicy<
      kSpec.single_tile_nominal_threads, kSpec.single_tile_nominal_items,
      DominantType, cub::BLOCK_LOAD_DIRECT, cub::LOAD_LDG,
      cub::RADIX_RANK_MEMOIZE, cub::BLOCK_SCAN_WARP_SCANS,
      kSpec.single_tile.radix_bits>;

  using HistogramPolicy = cub::AgentRadixSortHistogramPolicy<
      kSpec.histogram.block_threads, kSpec.histogram.items_per_thread,
      kSpec.histogram_nominal_4b_num_parts, KeyT, kSpec.histogram.radix_bits>;

  using OnesweepPolicy = cub::AgentRadixSortOnesweepPolicy<
      kSpec.onesweep_nominal_threads, kSpec.onesweep_nominal_items,
      DominantType, kSpec.onesweep_rank_num_parts,
      cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      kSpec.onesweep_use_raking_memoize ? cub::BLOCK_SCAN_RAKING_MEMOIZE
                                        : cub::BLOCK_SCAN_WARP_SCANS,
      cub::RADIX_SORT_STORE_DIRECT, kSpec.onesweep.radix_bits>;

  using SegmentedPolicy = cub::AgentRadixSortDownsweepPolicy<
      kSpec.segmented_nominal_threads, kSpec.segmented_nominal_items,
      DominantType,
      kSpec.segmented_use_warp_transpose ? cub::BLOCK_LOAD_WARP_TRANSPOSE
                                         : cub::BLOCK_LOAD_TRANSPOSE,
      cub::LOAD_DEFAULT, cub::RADIX_RANK_MEMOIZE, cub::BLOCK_SCAN_WARP_SCANS,
      kSpec.segmented.radix_bits>;

  using AltSegmentedPolicy = cub::AgentRadixSortDownsweepPolicy<
      kSpec.alt_segmented_nominal_threads, kSpec.alt_segmented_nominal_items,
      DominantType,
      kSpec.segmented_use_warp_transpose ? cub::BLOCK_LOAD_WARP_TRANSPOSE
                                         : cub::BLOCK_LOAD_TRANSPOSE,
      cub::LOAD_DEFAULT, cub::RADIX_RANK_MEMOIZE, cub::BLOCK_SCAN_WARP_SCANS,
      kSpec.alt_segmented.radix_bits>;

  static_assert(SingleTilePolicy::BLOCK_THREADS ==
                kSpec.single_tile.block_threads);
  static_assert(SingleTilePolicy::ITEMS_PER_THREAD ==
                kSpec.single_tile.items_per_thread);
  static_assert(SingleTilePolicy::RADIX_BITS == kSpec.single_tile.radix_bits);

  static_assert(HistogramPolicy::BLOCK_THREADS ==
                kSpec.histogram.block_threads);
  static_assert(HistogramPolicy::ITEMS_PER_THREAD ==
                kSpec.histogram.items_per_thread);
  static_assert(HistogramPolicy::RADIX_BITS == kSpec.histogram.radix_bits);
  // Pair shards reuse the keys-only shard's histogram kernel; lock the
  // value-independence invariant at compile time.
  static constexpr CubRadixSortPolicySpec kKeysOnlySpec =
      GetCubRadixSortPolicySpec(kSmVersion, static_cast<int>(sizeof(KeyT)),
                                /*value_byte_size=*/0,
                                std::is_same_v<KeyT, float>);
  static_assert(kSpec.histogram.block_threads ==
                kKeysOnlySpec.histogram.block_threads);
  static_assert(kSpec.histogram.items_per_thread ==
                kKeysOnlySpec.histogram.items_per_thread);
  static_assert(kSpec.histogram.radix_bits ==
                kKeysOnlySpec.histogram.radix_bits);
  static_assert(kSpec.histogram_nominal_4b_num_parts ==
                kKeysOnlySpec.histogram_nominal_4b_num_parts);

  static_assert(OnesweepPolicy::BLOCK_THREADS == kSpec.onesweep.block_threads);
  static_assert(OnesweepPolicy::ITEMS_PER_THREAD ==
                kSpec.onesweep.items_per_thread);
  static_assert(OnesweepPolicy::RADIX_BITS == kSpec.onesweep.radix_bits);

  static_assert(SegmentedPolicy::BLOCK_THREADS ==
                kSpec.segmented.block_threads);
  static_assert(SegmentedPolicy::ITEMS_PER_THREAD ==
                kSpec.segmented.items_per_thread);
  static_assert(SegmentedPolicy::RADIX_BITS == kSpec.segmented.radix_bits);

  static_assert(AltSegmentedPolicy::BLOCK_THREADS ==
                kSpec.alt_segmented.block_threads);
  static_assert(AltSegmentedPolicy::ITEMS_PER_THREAD ==
                kSpec.alt_segmented.items_per_thread);
  static_assert(AltSegmentedPolicy::RADIX_BITS ==
                kSpec.alt_segmented.radix_bits);
};

// Major-SM policy chain (500..1000) so minor-SM CUBIN fallbacks (e.g., sm_100
// on sm_103) always match CanonicalCubSortPolicySmVersion(cc.major).
template <typename KeyT, typename ValueT>
struct XlaChainedPolicy {
  struct Policy500 : cub::ChainedPolicy<500, Policy500, Policy500>,
                     XlaCubRadixSortPolicy<500, KeyT, ValueT> {};
  struct Policy600 : cub::ChainedPolicy<600, Policy600, Policy500>,
                     XlaCubRadixSortPolicy<600, KeyT, ValueT> {};
  struct Policy700 : cub::ChainedPolicy<700, Policy700, Policy600>,
                     XlaCubRadixSortPolicy<700, KeyT, ValueT> {};
  struct Policy800 : cub::ChainedPolicy<800, Policy800, Policy700>,
                     XlaCubRadixSortPolicy<800, KeyT, ValueT> {};
  struct Policy900 : cub::ChainedPolicy<900, Policy900, Policy800>,
                     XlaCubRadixSortPolicy<900, KeyT, ValueT> {};
  struct Policy1000 : cub::ChainedPolicy<1000, Policy1000, Policy900>,
                      XlaCubRadixSortPolicy<1000, KeyT, ValueT> {};

  using MaxPolicy = Policy1000;
};

template <bool kDescending, typename KeyT>
__device__ __forceinline__ KeyT MakeDefaultOobKey() {
  using Traits = cub::Traits<KeyT>;
  typename Traits::UnsignedBits default_key_bits =
      kDescending ? Traits::LOWEST_KEY : Traits::MAX_KEY;
  KeyT default_key;
  std::memcpy(&default_key, &default_key_bits, sizeof(KeyT));
  return default_key;
}

//===----------------------------------------------------------------------===//
// 1. Unified SingleTile kernels (unbatched and batched segments <= tile_items)
//===----------------------------------------------------------------------===//

template <typename ChainedPolicyT, bool kDescending, typename KeyT>
__launch_bounds__(ChainedPolicyT::ActivePolicy::SingleTilePolicy::BLOCK_THREADS)
    __global__
    void CubRadixSortSingleTileKeysKernel(const KeyT* __restrict__ d_keys_in,
                                          KeyT* __restrict__ d_keys_out,
                                          int32_t segment_size) {
  using Policy = typename ChainedPolicyT::ActivePolicy::SingleTilePolicy;
  constexpr int kBlockThreads = Policy::BLOCK_THREADS;
  constexpr int kItemsPerThread = Policy::ITEMS_PER_THREAD;

  using BlockRadixSort =
      cub::BlockRadixSort<KeyT, kBlockThreads, kItemsPerThread, cub::NullType,
                          Policy::RADIX_BITS,
                          (Policy::RANK_ALGORITHM == cub::RADIX_RANK_MEMOIZE),
                          Policy::SCAN_ALGORITHM>;
  using BlockLoadKeys = cub::BlockLoad<KeyT, kBlockThreads, kItemsPerThread,
                                       Policy::LOAD_ALGORITHM>;

  __shared__ union {
    typename BlockRadixSort::TempStorage sort;
    typename BlockLoadKeys::TempStorage load_keys;
  } temp_storage;

  const int64_t base_offset =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(segment_size);
  const KeyT* seg_keys_in = d_keys_in + base_offset;
  KeyT* seg_keys_out = d_keys_out + base_offset;

  KeyT keys[kItemsPerThread];
  BlockLoadKeys(temp_storage.load_keys)
      .Load(seg_keys_in, keys, segment_size,
            MakeDefaultOobKey<kDescending, KeyT>());
  __syncthreads();

  if constexpr (kDescending) {
    BlockRadixSort(temp_storage.sort)
        .SortDescendingBlockedToStriped(keys, /*begin_bit=*/0,
                                        /*end_bit=*/sizeof(KeyT) * 8);
  } else {
    BlockRadixSort(temp_storage.sort)
        .SortBlockedToStriped(keys, /*begin_bit=*/0,
                              /*end_bit=*/sizeof(KeyT) * 8);
  }

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int item_offset =
        item * kBlockThreads + static_cast<int>(threadIdx.x);
    if (item_offset < segment_size) {
      seg_keys_out[item_offset] = keys[item];
    }
  }
}

template <typename ChainedPolicyT, bool kDescending, typename KeyT,
          typename ValT>
__launch_bounds__(ChainedPolicyT::ActivePolicy::SingleTilePolicy::BLOCK_THREADS)
    __global__
    void CubRadixSortSingleTilePairsKernel(const KeyT* __restrict__ d_keys_in,
                                           KeyT* __restrict__ d_keys_out,
                                           const ValT* __restrict__ d_values_in,
                                           ValT* __restrict__ d_values_out,
                                           int32_t segment_size) {
  using Policy = typename ChainedPolicyT::ActivePolicy::SingleTilePolicy;
  constexpr int kBlockThreads = Policy::BLOCK_THREADS;
  constexpr int kItemsPerThread = Policy::ITEMS_PER_THREAD;

  using BlockRadixSort =
      cub::BlockRadixSort<KeyT, kBlockThreads, kItemsPerThread, ValT,
                          Policy::RADIX_BITS,
                          (Policy::RANK_ALGORITHM == cub::RADIX_RANK_MEMOIZE),
                          Policy::SCAN_ALGORITHM>;
  using BlockLoadKeys = cub::BlockLoad<KeyT, kBlockThreads, kItemsPerThread,
                                       Policy::LOAD_ALGORITHM>;
  using BlockLoadValues = cub::BlockLoad<ValT, kBlockThreads, kItemsPerThread,
                                         Policy::LOAD_ALGORITHM>;

  __shared__ union {
    typename BlockRadixSort::TempStorage sort;
    typename BlockLoadKeys::TempStorage load_keys;
    typename BlockLoadValues::TempStorage load_values;
  } temp_storage;

  const int64_t base_offset =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(segment_size);
  const KeyT* seg_keys_in = d_keys_in + base_offset;
  KeyT* seg_keys_out = d_keys_out + base_offset;
  const ValT* seg_values_in = d_values_in + base_offset;
  ValT* seg_values_out = d_values_out + base_offset;

  KeyT keys[kItemsPerThread];
  ValT values[kItemsPerThread];

  BlockLoadKeys(temp_storage.load_keys)
      .Load(seg_keys_in, keys, segment_size,
            MakeDefaultOobKey<kDescending, KeyT>());
  __syncthreads();

  // Mirror CUB's register-pressure workaround: moving segment_size through shfl
  // prevents ptxas from holding prior guarded-load predicates across loads.
  static_assert(kBlockThreads % 32 == 0);
  const int32_t valid_items =
      cub::ShuffleIndex<32>(segment_size, 0, 0xffffffff);
  BlockLoadValues(temp_storage.load_values)
      .Load(seg_values_in, values, valid_items);
  __syncthreads();

  if constexpr (kDescending) {
    BlockRadixSort(temp_storage.sort)
        .SortDescendingBlockedToStriped(keys, values, /*begin_bit=*/0,
                                        /*end_bit=*/sizeof(KeyT) * 8);
  } else {
    BlockRadixSort(temp_storage.sort)
        .SortBlockedToStriped(keys, values, /*begin_bit=*/0,
                              /*end_bit=*/sizeof(KeyT) * 8);
  }

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int item_offset =
        item * kBlockThreads + static_cast<int>(threadIdx.x);
    if (item_offset < valid_items) {
      seg_keys_out[item_offset] = keys[item];
      seg_values_out[item_offset] = values[item];
    }
  }
}

//===----------------------------------------------------------------------===//
// 2. Onesweep Histogram (keys-only shards, reused by pair shards) & Scatter
//===----------------------------------------------------------------------===//

template <typename ChainedPolicyT, bool kDescending, typename KeyT>
__launch_bounds__(ChainedPolicyT::ActivePolicy::HistogramPolicy::BLOCK_THREADS)
    __global__
    void CubRadixSortHistogramKernel(CubGlobalOffset* __restrict__ d_bins_out,
                                     const KeyT* __restrict__ d_keys_in,
                                     CubGlobalOffset num_items,
                                     int32_t begin_bit, int32_t end_bit) {
  using HistogramPolicy =
      typename ChainedPolicyT::ActivePolicy::HistogramPolicy;
  using Agent = cub::detail::radix_sort::AgentRadixSortHistogram<
      HistogramPolicy, kDescending, KeyT, CubGlobalOffset>;
  __shared__ typename Agent::TempStorage temp_storage;
  Agent agent(temp_storage, d_bins_out, d_keys_in, num_items, begin_bit,
              end_bit);
  agent.Process();
}

template <typename ChainedPolicyT, bool kDescending, typename KeyT>
__launch_bounds__(ChainedPolicyT::ActivePolicy::OnesweepPolicy::BLOCK_THREADS)
    __global__ void CubRadixSortOnesweepKeysKernel(
        int32_t* __restrict__ d_lookback, int32_t* __restrict__ d_ctrs,
        CubGlobalOffset* __restrict__ d_bins_out,
        const CubGlobalOffset* __restrict__ d_bins_in,
        KeyT* __restrict__ d_keys_out, const KeyT* __restrict__ d_keys_in,
        int64_t portion_item_offset, int32_t portion_num_items,
        int32_t current_bit, int32_t num_bits) {
  using OnesweepPolicy = typename ChainedPolicyT::ActivePolicy::OnesweepPolicy;
  using Agent = cub::detail::radix_sort::AgentRadixSortOnesweep<
      OnesweepPolicy, kDescending, KeyT, cub::NullType, CubGlobalOffset,
      int32_t>;
  __shared__ typename Agent::TempStorage temp_storage;
  Agent agent(temp_storage, d_lookback, d_ctrs, d_bins_out, d_bins_in,
              d_keys_out, d_keys_in + portion_item_offset,
              /*d_values_out=*/nullptr, /*d_values_in=*/nullptr,
              portion_num_items, current_bit, num_bits);
  agent.Process();
}

template <typename ChainedPolicyT, bool kDescending, typename KeyT,
          typename ValT>
__launch_bounds__(ChainedPolicyT::ActivePolicy::OnesweepPolicy::BLOCK_THREADS)
    __global__ void CubRadixSortOnesweepPairsKernel(
        int32_t* __restrict__ d_lookback, int32_t* __restrict__ d_ctrs,
        CubGlobalOffset* __restrict__ d_bins_out,
        const CubGlobalOffset* __restrict__ d_bins_in,
        KeyT* __restrict__ d_keys_out, const KeyT* __restrict__ d_keys_in,
        ValT* __restrict__ d_values_out, const ValT* __restrict__ d_values_in,
        int64_t portion_item_offset, int32_t portion_num_items,
        int32_t current_bit, int32_t num_bits) {
  using OnesweepPolicy = typename ChainedPolicyT::ActivePolicy::OnesweepPolicy;
  using Agent = cub::detail::radix_sort::AgentRadixSortOnesweep<
      OnesweepPolicy, kDescending, KeyT, ValT, CubGlobalOffset, int32_t>;
  __shared__ typename Agent::TempStorage temp_storage;
  Agent agent(temp_storage, d_lookback, d_ctrs, d_bins_out, d_bins_in,
              d_keys_out, d_keys_in + portion_item_offset, d_values_out,
              d_values_in + portion_item_offset, portion_num_items, current_bit,
              num_bits);
  agent.Process();
}

//===----------------------------------------------------------------------===//
// 3. Uniform-segment multi-pass batched radix sort kernels (no H2D offsets)
//===----------------------------------------------------------------------===//

template <typename SegmentedPolicy, bool kDescending, typename KeyT,
          typename ValueT>
__device__ __forceinline__ void RunUniformSegmentedPass(
    const KeyT* __restrict__ d_keys_in, KeyT* __restrict__ d_keys_out,
    const ValueT* __restrict__ d_values_in, ValueT* __restrict__ d_values_out,
    int32_t segment_size, int32_t current_bit, int32_t pass_bits) {
  constexpr int kBlockThreads = SegmentedPolicy::BLOCK_THREADS;
  constexpr int kRadixBits = SegmentedPolicy::RADIX_BITS;
  constexpr int kRadixDigits = 1 << kRadixBits;
  static_assert(kBlockThreads >= kRadixDigits);

  using BlockUpsweep =
      cub::detail::radix_sort::AgentRadixSortUpsweep<SegmentedPolicy, KeyT,
                                                     int32_t>;
  using DigitScan = cub::BlockScan<int32_t, kBlockThreads>;
  using BlockDownsweep = cub::detail::radix_sort::AgentRadixSortDownsweep<
      SegmentedPolicy, kDescending, KeyT, ValueT, int32_t>;
  constexpr int kBinsTrackedPerThread = BlockDownsweep::BINS_TRACKED_PER_THREAD;

  __shared__ union {
    typename BlockUpsweep::TempStorage upsweep;
    typename BlockDownsweep::TempStorage downsweep;
    struct {
      volatile int32_t reverse_counts_in[kRadixDigits];
      volatile int32_t reverse_counts_out[kRadixDigits];
      typename DigitScan::TempStorage scan;
    };
  } temp_storage;

  if (segment_size <= 0) {
    return;
  }

  // Recompute base_offset before upsweep and again before downsweep (matching
  // CUB's DeviceSegmentedRadixSortKernel) so segment pointers are not kept live
  // in registers across upsweep + BlockScan, reducing register pressure in
  // AltSegmented.
  int32_t bin_count[kBinsTrackedPerThread] = {};
  {
    const int64_t upsweep_offset =
        static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(segment_size);
    BlockUpsweep upsweep(temp_storage.upsweep, d_keys_in + upsweep_offset,
                         current_bit, pass_bits);
    upsweep.ProcessRegion(int32_t{0}, segment_size);
    __syncthreads();

    upsweep.ExtractCounts(bin_count);
    __syncthreads();
  }

  if constexpr (kDescending) {
#pragma unroll
    for (int track = 0; track < kBinsTrackedPerThread; ++track) {
      const int bin_idx =
          static_cast<int>(threadIdx.x) * kBinsTrackedPerThread + track;
      if ((kBlockThreads == kRadixDigits) || (bin_idx < kRadixDigits)) {
        temp_storage.reverse_counts_in[bin_idx] = bin_count[track];
      }
    }
    __syncthreads();
#pragma unroll
    for (int track = 0; track < kBinsTrackedPerThread; ++track) {
      const int bin_idx =
          static_cast<int>(threadIdx.x) * kBinsTrackedPerThread + track;
      if ((kBlockThreads == kRadixDigits) || (bin_idx < kRadixDigits)) {
        bin_count[track] =
            temp_storage.reverse_counts_in[kRadixDigits - bin_idx - 1];
      }
    }
  }

  int32_t bin_offset[kBinsTrackedPerThread];
  DigitScan(temp_storage.scan).ExclusiveSum(bin_count, bin_offset);

  if constexpr (kDescending) {
#pragma unroll
    for (int track = 0; track < kBinsTrackedPerThread; ++track) {
      const int bin_idx =
          static_cast<int>(threadIdx.x) * kBinsTrackedPerThread + track;
      if ((kBlockThreads == kRadixDigits) || (bin_idx < kRadixDigits)) {
        temp_storage.reverse_counts_out[bin_idx] = bin_offset[track];
      }
    }
    __syncthreads();
#pragma unroll
    for (int track = 0; track < kBinsTrackedPerThread; ++track) {
      const int bin_idx =
          static_cast<int>(threadIdx.x) * kBinsTrackedPerThread + track;
      if ((kBlockThreads == kRadixDigits) || (bin_idx < kRadixDigits)) {
        bin_offset[track] =
            temp_storage.reverse_counts_out[kRadixDigits - bin_idx - 1];
      }
    }
  }

  __syncthreads();

  const int64_t downsweep_offset =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(segment_size);
  const ValueT* seg_values_in = std::is_same_v<ValueT, cub::NullType>
                                    ? nullptr
                                    : d_values_in + downsweep_offset;
  ValueT* seg_values_out = std::is_same_v<ValueT, cub::NullType>
                               ? nullptr
                               : d_values_out + downsweep_offset;
  BlockDownsweep downsweep(temp_storage.downsweep, bin_offset, segment_size,
                           d_keys_in + downsweep_offset,
                           d_keys_out + downsweep_offset, seg_values_in,
                           seg_values_out, current_bit, pass_bits);
  downsweep.ProcessRegion(int32_t{0}, segment_size);
}

template <typename ChainedPolicyT, bool kAltDigitBits, bool kDescending,
          typename KeyT>
__launch_bounds__(
    kAltDigitBits
        ? ChainedPolicyT::ActivePolicy::AltSegmentedPolicy::BLOCK_THREADS
        : ChainedPolicyT::ActivePolicy::SegmentedPolicy::BLOCK_THREADS)
    __global__ void CubUniformSegmentedRadixSortKeysKernel(
        const KeyT* __restrict__ d_keys_in, KeyT* __restrict__ d_keys_out,
        int32_t segment_size, int32_t current_bit, int32_t pass_bits) {
  using SegmentedPolicy = std::conditional_t<
      kAltDigitBits, typename ChainedPolicyT::ActivePolicy::AltSegmentedPolicy,
      typename ChainedPolicyT::ActivePolicy::SegmentedPolicy>;
  RunUniformSegmentedPass<SegmentedPolicy, kDescending, KeyT, cub::NullType>(
      d_keys_in, d_keys_out, /*d_values_in=*/nullptr, /*d_values_out=*/nullptr,
      segment_size, current_bit, pass_bits);
}

template <typename ChainedPolicyT, bool kAltDigitBits, bool kDescending,
          typename KeyT, typename ValT>
__launch_bounds__(
    kAltDigitBits
        ? ChainedPolicyT::ActivePolicy::AltSegmentedPolicy::BLOCK_THREADS
        : ChainedPolicyT::ActivePolicy::SegmentedPolicy::BLOCK_THREADS)
    __global__ void CubUniformSegmentedRadixSortPairsKernel(
        const KeyT* __restrict__ d_keys_in, KeyT* __restrict__ d_keys_out,
        const ValT* __restrict__ d_values_in, ValT* __restrict__ d_values_out,
        int32_t segment_size, int32_t current_bit, int32_t pass_bits) {
  using SegmentedPolicy = std::conditional_t<
      kAltDigitBits, typename ChainedPolicyT::ActivePolicy::AltSegmentedPolicy,
      typename ChainedPolicyT::ActivePolicy::SegmentedPolicy>;
  RunUniformSegmentedPass<SegmentedPolicy, kDescending, KeyT, ValT>(
      d_keys_in, d_keys_out, d_values_in, d_values_out, segment_size,
      current_bit, pass_bits);
}

template <typename ReturnT, typename... Args>
KernelHostRef MakeKernelHostRef(ReturnT (*fn)(Args...)) {
  return {reinterpret_cast<const void*>(fn), sizeof...(Args)};
}

template <typename KeyT>
CubSortTypeKernelRefs GetKeysRefsImpl(bool descending) {
  using Policy = typename XlaChainedPolicy<KeyT, cub::NullType>::MaxPolicy;
  if (descending) {
    return {
        /*single_tile=*/MakeKernelHostRef(
            &CubRadixSortSingleTileKeysKernel<Policy, true, KeyT>),
        /*onesweep=*/
        MakeKernelHostRef(&CubRadixSortOnesweepKeysKernel<Policy, true, KeyT>),
        /*segmented_primary=*/
        MakeKernelHostRef(
            &CubUniformSegmentedRadixSortKeysKernel<Policy, false, true, KeyT>),
        /*segmented_alt=*/
        MakeKernelHostRef(
            &CubUniformSegmentedRadixSortKeysKernel<Policy, true, true, KeyT>),
    };
  }
  return {
      /*single_tile=*/MakeKernelHostRef(
          &CubRadixSortSingleTileKeysKernel<Policy, false, KeyT>),
      /*onesweep=*/
      MakeKernelHostRef(&CubRadixSortOnesweepKeysKernel<Policy, false, KeyT>),
      /*segmented_primary=*/
      MakeKernelHostRef(
          &CubUniformSegmentedRadixSortKeysKernel<Policy, false, false, KeyT>),
      /*segmented_alt=*/
      MakeKernelHostRef(
          &CubUniformSegmentedRadixSortKeysKernel<Policy, true, false, KeyT>),
  };
}

template <typename KeyT>
KernelHostRef GetHistogramRefImpl(bool descending) {
  using Policy = typename XlaChainedPolicy<KeyT, cub::NullType>::MaxPolicy;
  if (descending) {
    return MakeKernelHostRef(&CubRadixSortHistogramKernel<Policy, true, KeyT>);
  }
  return MakeKernelHostRef(&CubRadixSortHistogramKernel<Policy, false, KeyT>);
}

template <typename KeyT, typename ValT>
CubSortTypeKernelRefs GetPairsRefsImpl(bool descending) {
  using Policy = typename XlaChainedPolicy<KeyT, ValT>::MaxPolicy;
  if (descending) {
    return {
        /*single_tile=*/MakeKernelHostRef(
            &CubRadixSortSingleTilePairsKernel<Policy, true, KeyT, ValT>),
        /*onesweep=*/
        MakeKernelHostRef(
            &CubRadixSortOnesweepPairsKernel<Policy, true, KeyT, ValT>),
        /*segmented_primary=*/
        MakeKernelHostRef(
            &CubUniformSegmentedRadixSortPairsKernel<Policy, false, true, KeyT,
                                                     ValT>),
        /*segmented_alt=*/
        MakeKernelHostRef(
            &CubUniformSegmentedRadixSortPairsKernel<Policy, true, true, KeyT,
                                                     ValT>),
    };
  }
  return {
      /*single_tile=*/MakeKernelHostRef(
          &CubRadixSortSingleTilePairsKernel<Policy, false, KeyT, ValT>),
      /*onesweep=*/
      MakeKernelHostRef(
          &CubRadixSortOnesweepPairsKernel<Policy, false, KeyT, ValT>),
      /*segmented_primary=*/
      MakeKernelHostRef(
          &CubUniformSegmentedRadixSortPairsKernel<Policy, false, false, KeyT,
                                                   ValT>),
      /*segmented_alt=*/
      MakeKernelHostRef(
          &CubUniformSegmentedRadixSortPairsKernel<Policy, true, false, KeyT,
                                                   ValT>),
  };
}

template <PrimitiveType kKeyType>
struct CudaKeyType;
template <>
struct CudaKeyType<BF16> {
  using type = __nv_bfloat16;
};
template <>
struct CudaKeyType<F16> {
  using type = __half;
};
template <>
struct CudaKeyType<F32> {
  using type = float;
};
template <>
struct CudaKeyType<F64> {
  using type = double;
};
template <>
struct CudaKeyType<S8> {
  using type = int8_t;
};
template <>
struct CudaKeyType<S16> {
  using type = int16_t;
};
template <>
struct CudaKeyType<S32> {
  using type = int32_t;
};
template <>
struct CudaKeyType<S64> {
  using type = int64_t;
};
template <>
struct CudaKeyType<U8> {
  using type = uint8_t;
};
template <>
struct CudaKeyType<U16> {
  using type = uint16_t;
};
template <>
struct CudaKeyType<U32> {
  using type = uint32_t;
};
template <>
struct CudaKeyType<U64> {
  using type = uint64_t;
};

template <int kBitWidth>
struct CudaValueType;
template <>
struct CudaValueType<16> {
  using type = uint16_t;
};
template <>
struct CudaValueType<32> {
  using type = uint32_t;
};
template <>
struct CudaValueType<64> {
  using type = uint64_t;
};

}  // namespace

template <PrimitiveType kKeyType>
CubSortTypeKernelRefs GetCubSortKeysRefs(bool descending) {
  return GetKeysRefsImpl<typename CudaKeyType<kKeyType>::type>(descending);
}

template <PrimitiveType kKeyType>
KernelHostRef GetCubSortHistogramRef(bool descending) {
  return GetHistogramRefImpl<typename CudaKeyType<kKeyType>::type>(descending);
}

template <PrimitiveType kKeyType, int kValueBitWidth>
CubSortTypeKernelRefs GetCubSortPairsRefs(bool descending) {
  return GetPairsRefsImpl<typename CudaKeyType<kKeyType>::type,
                          typename CudaValueType<kValueBitWidth>::type>(
      descending);
}

// Floating point keys.
#ifdef CUB_TYPE_BF16
template CubSortTypeKernelRefs GetCubSortKeysRefs<BF16>(bool);
template KernelHostRef GetCubSortHistogramRef<BF16>(bool);
#endif
#ifdef CUB_TYPE_F16
template CubSortTypeKernelRefs GetCubSortKeysRefs<F16>(bool);
template KernelHostRef GetCubSortHistogramRef<F16>(bool);
#endif
#ifdef CUB_TYPE_F32
template CubSortTypeKernelRefs GetCubSortKeysRefs<F32>(bool);
template KernelHostRef GetCubSortHistogramRef<F32>(bool);
#endif
#ifdef CUB_TYPE_F64
template CubSortTypeKernelRefs GetCubSortKeysRefs<F64>(bool);
template KernelHostRef GetCubSortHistogramRef<F64>(bool);
#endif

// Signed integer keys.
#ifdef CUB_TYPE_S8
template CubSortTypeKernelRefs GetCubSortKeysRefs<S8>(bool);
template KernelHostRef GetCubSortHistogramRef<S8>(bool);
#endif
#ifdef CUB_TYPE_S16
template CubSortTypeKernelRefs GetCubSortKeysRefs<S16>(bool);
template KernelHostRef GetCubSortHistogramRef<S16>(bool);
#endif
#ifdef CUB_TYPE_S32
template CubSortTypeKernelRefs GetCubSortKeysRefs<S32>(bool);
template KernelHostRef GetCubSortHistogramRef<S32>(bool);
#endif
#ifdef CUB_TYPE_S64
template CubSortTypeKernelRefs GetCubSortKeysRefs<S64>(bool);
template KernelHostRef GetCubSortHistogramRef<S64>(bool);
#endif

// Unsigned integer keys.
#ifdef CUB_TYPE_U8
template CubSortTypeKernelRefs GetCubSortKeysRefs<U8>(bool);
template KernelHostRef GetCubSortHistogramRef<U8>(bool);
#endif
#ifdef CUB_TYPE_U16
template CubSortTypeKernelRefs GetCubSortKeysRefs<U16>(bool);
template KernelHostRef GetCubSortHistogramRef<U16>(bool);
#endif
#ifdef CUB_TYPE_U32
template CubSortTypeKernelRefs GetCubSortKeysRefs<U32>(bool);
template KernelHostRef GetCubSortHistogramRef<U32>(bool);
#endif
#ifdef CUB_TYPE_U64
template CubSortTypeKernelRefs GetCubSortKeysRefs<U64>(bool);
template KernelHostRef GetCubSortHistogramRef<U64>(bool);
#endif

// Pairs with 8-bit key.
#ifdef CUB_TYPE_U8_B16
template CubSortTypeKernelRefs GetCubSortPairsRefs<U8, 16>(bool);
#endif
#ifdef CUB_TYPE_U8_B32
template CubSortTypeKernelRefs GetCubSortPairsRefs<U8, 32>(bool);
#endif
#ifdef CUB_TYPE_U8_B64
template CubSortTypeKernelRefs GetCubSortPairsRefs<U8, 64>(bool);
#endif

// Pairs with 16-bit key.
#ifdef CUB_TYPE_U16_B16
template CubSortTypeKernelRefs GetCubSortPairsRefs<U16, 16>(bool);
#endif
#ifdef CUB_TYPE_U16_B32
template CubSortTypeKernelRefs GetCubSortPairsRefs<U16, 32>(bool);
#endif
#ifdef CUB_TYPE_U16_B64
template CubSortTypeKernelRefs GetCubSortPairsRefs<U16, 64>(bool);
#endif

// Pairs with signed 32-bit key.
#ifdef CUB_TYPE_S32_B16
template CubSortTypeKernelRefs GetCubSortPairsRefs<S32, 16>(bool);
#endif
#ifdef CUB_TYPE_S32_B32
template CubSortTypeKernelRefs GetCubSortPairsRefs<S32, 32>(bool);
#endif
#ifdef CUB_TYPE_S32_B64
template CubSortTypeKernelRefs GetCubSortPairsRefs<S32, 64>(bool);
#endif

// Pairs with unsigned 32-bit key.
#ifdef CUB_TYPE_U32_B16
template CubSortTypeKernelRefs GetCubSortPairsRefs<U32, 16>(bool);
#endif
#ifdef CUB_TYPE_U32_B32
template CubSortTypeKernelRefs GetCubSortPairsRefs<U32, 32>(bool);
#endif
#ifdef CUB_TYPE_U32_B64
template CubSortTypeKernelRefs GetCubSortPairsRefs<U32, 64>(bool);
#endif

// Pairs with 64-bit key.
#ifdef CUB_TYPE_U64_B16
template CubSortTypeKernelRefs GetCubSortPairsRefs<U64, 16>(bool);
#endif
#ifdef CUB_TYPE_U64_B32
template CubSortTypeKernelRefs GetCubSortPairsRefs<U64, 32>(bool);
#endif
#ifdef CUB_TYPE_U64_B64
template CubSortTypeKernelRefs GetCubSortPairsRefs<U64, 64>(bool);
#endif

// Pairs with f32 key.
#ifdef CUB_TYPE_F32_B16
template CubSortTypeKernelRefs GetCubSortPairsRefs<F32, 16>(bool);
#endif
#ifdef CUB_TYPE_F32_B32
template CubSortTypeKernelRefs GetCubSortPairsRefs<F32, 32>(bool);
#endif
#ifdef CUB_TYPE_F32_B64
template CubSortTypeKernelRefs GetCubSortPairsRefs<F32, 64>(bool);
#endif

}  // namespace xla::gpu::internal
