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
#include "cub/block/block_load.cuh"
#include "cub/block/block_radix_sort.cuh"
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
template <int kSmVersion, typename Key, typename Value>
struct XlaCubRadixSortPolicy {
  static constexpr bool kKeysOnly = std::is_same_v<Value, cub::NullType>;
  static constexpr int kValueByteSize =
      kKeysOnly ? 0 : static_cast<int>(sizeof(Value));
  using DominantType =
      std::conditional_t<(sizeof(Value) > sizeof(Key)), Value, Key>;

  static constexpr CubRadixSortPolicySpec kSpec =
      GetCubRadixSortPolicySpec(kSmVersion, static_cast<int>(sizeof(Key)),
                                kValueByteSize, std::is_same_v<Key, float>);

  using SingleTilePolicy = cub::AgentRadixSortDownsweepPolicy<
      kSpec.single_tile.nominal_threads, kSpec.single_tile.nominal_items,
      DominantType, cub::BLOCK_LOAD_DIRECT, cub::LOAD_LDG,
      cub::RADIX_RANK_MEMOIZE, cub::BLOCK_SCAN_WARP_SCANS,
      kSpec.single_tile.radix_bits>;

  using HistogramPolicy = cub::AgentRadixSortHistogramPolicy<
      kSpec.histogram.block_threads, kSpec.histogram.items_per_thread,
      kSpec.histogram_nominal_4b_num_parts, Key, kSpec.histogram.radix_bits>;

  using OnesweepPolicy = cub::AgentRadixSortOnesweepPolicy<
      kSpec.onesweep.nominal_threads, kSpec.onesweep.nominal_items,
      DominantType, kSpec.onesweep_rank_num_parts,
      cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      kSpec.onesweep_use_raking_memoize ? cub::BLOCK_SCAN_RAKING_MEMOIZE
                                        : cub::BLOCK_SCAN_WARP_SCANS,
      cub::RADIX_SORT_STORE_DIRECT, kSpec.onesweep.radix_bits>;

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
      GetCubRadixSortPolicySpec(kSmVersion, static_cast<int>(sizeof(Key)),
                                /*value_byte_size=*/0,
                                std::is_same_v<Key, float>);
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
};

// Major-SM policy chain (500..1000) so minor-SM CUBIN fallbacks (e.g., sm_100
// on sm_103) always match CanonicalCubSortPolicySmVersion(cc.major).
template <typename Key, typename Value>
struct XlaChainedPolicy {
  struct Policy500 : cub::ChainedPolicy<500, Policy500, Policy500>,
                     XlaCubRadixSortPolicy<500, Key, Value> {};
  struct Policy600 : cub::ChainedPolicy<600, Policy600, Policy500>,
                     XlaCubRadixSortPolicy<600, Key, Value> {};
  struct Policy700 : cub::ChainedPolicy<700, Policy700, Policy600>,
                     XlaCubRadixSortPolicy<700, Key, Value> {};
  struct Policy800 : cub::ChainedPolicy<800, Policy800, Policy700>,
                     XlaCubRadixSortPolicy<800, Key, Value> {};
  struct Policy900 : cub::ChainedPolicy<900, Policy900, Policy800>,
                     XlaCubRadixSortPolicy<900, Key, Value> {};
  struct Policy1000 : cub::ChainedPolicy<1000, Policy1000, Policy900>,
                      XlaCubRadixSortPolicy<1000, Key, Value> {};

  using MaxPolicy = Policy1000;
};

// Returns a padding sentinel key for out-of-bounds lanes when a segment is
// smaller than the single-tile capacity (`segment_size < TILE_ITEMS`).
// `BlockRadixSort` always sorts a full tile, so padding with the extreme
// bit-order key (`MAX_KEY` for ascending, `LOWEST_KEY` for descending) ensures
// OOB lanes sort to the tail of the tile and are skipped by the guarded store.
// `std::memcpy` from `UnsignedBits` preserves exact bit patterns (including
// IEEE-754 NaN payloads).
template <SortDirection kDirection, typename Key>
__device__ __forceinline__ Key MakeDefaultOobKey() {
  using Traits = cub::Traits<Key>;
  typename Traits::UnsignedBits default_key_bits =
      kDirection == SortDirection::kDescending ? Traits::LOWEST_KEY
                                               : Traits::MAX_KEY;
  Key default_key;
  std::memcpy(&default_key, &default_key_bits, sizeof(Key));
  return default_key;
}

//===----------------------------------------------------------------------===//
// 1. Unified SingleTile kernels (unbatched and batched segments <= tile_items)
//===----------------------------------------------------------------------===//

template <typename ChainedPolicy, SortDirection kDirection, typename Key>
__launch_bounds__(ChainedPolicy::ActivePolicy::SingleTilePolicy::BLOCK_THREADS)
    __global__
    void CubRadixSortSingleTileKeysKernel(const Key* __restrict__ d_keys_in,
                                          Key* __restrict__ d_keys_out,
                                          int32_t segment_size) {
  using Policy = typename ChainedPolicy::ActivePolicy::SingleTilePolicy;
  constexpr int kBlockThreads = Policy::BLOCK_THREADS;
  constexpr int kItemsPerThread = Policy::ITEMS_PER_THREAD;

  using BlockRadixSort =
      cub::BlockRadixSort<Key, kBlockThreads, kItemsPerThread, cub::NullType,
                          Policy::RADIX_BITS,
                          (Policy::RANK_ALGORITHM == cub::RADIX_RANK_MEMOIZE),
                          Policy::SCAN_ALGORITHM>;
  using BlockLoadKeys = cub::BlockLoad<Key, kBlockThreads, kItemsPerThread,
                                       Policy::LOAD_ALGORITHM>;

  __shared__ union {
    typename BlockRadixSort::TempStorage sort;
    typename BlockLoadKeys::TempStorage load_keys;
  } temp_storage;

  const int64_t base_offset =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(segment_size);
  const Key* seg_keys_in = d_keys_in + base_offset;
  Key* seg_keys_out = d_keys_out + base_offset;

  Key keys[kItemsPerThread];
  BlockLoadKeys(temp_storage.load_keys)
      .Load(seg_keys_in, keys, segment_size,
            MakeDefaultOobKey<kDirection, Key>());
  __syncthreads();

  if constexpr (kDirection == SortDirection::kDescending) {
    BlockRadixSort(temp_storage.sort)
        .SortDescendingBlockedToStriped(keys, /*begin_bit=*/0,
                                        /*end_bit=*/sizeof(Key) * 8);
  } else {
    BlockRadixSort(temp_storage.sort)
        .SortBlockedToStriped(keys, /*begin_bit=*/0,
                              /*end_bit=*/sizeof(Key) * 8);
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

template <typename ChainedPolicy, SortDirection kDirection, typename Key,
          typename Value>
__launch_bounds__(ChainedPolicy::ActivePolicy::SingleTilePolicy::BLOCK_THREADS)
    __global__ void CubRadixSortSingleTilePairsKernel(
        const Key* __restrict__ d_keys_in, Key* __restrict__ d_keys_out,
        const Value* __restrict__ d_values_in, Value* __restrict__ d_values_out,
        int32_t segment_size) {
  using Policy = typename ChainedPolicy::ActivePolicy::SingleTilePolicy;
  constexpr int kBlockThreads = Policy::BLOCK_THREADS;
  constexpr int kItemsPerThread = Policy::ITEMS_PER_THREAD;

  using BlockRadixSort =
      cub::BlockRadixSort<Key, kBlockThreads, kItemsPerThread, Value,
                          Policy::RADIX_BITS,
                          (Policy::RANK_ALGORITHM == cub::RADIX_RANK_MEMOIZE),
                          Policy::SCAN_ALGORITHM>;
  using BlockLoadKeys = cub::BlockLoad<Key, kBlockThreads, kItemsPerThread,
                                       Policy::LOAD_ALGORITHM>;
  using BlockLoadValues = cub::BlockLoad<Value, kBlockThreads, kItemsPerThread,
                                         Policy::LOAD_ALGORITHM>;

  __shared__ union {
    typename BlockRadixSort::TempStorage sort;
    typename BlockLoadKeys::TempStorage load_keys;
    typename BlockLoadValues::TempStorage load_values;
  } temp_storage;

  const int64_t base_offset =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(segment_size);
  const Key* seg_keys_in = d_keys_in + base_offset;
  Key* seg_keys_out = d_keys_out + base_offset;
  const Value* seg_values_in = d_values_in + base_offset;
  Value* seg_values_out = d_values_out + base_offset;

  Key keys[kItemsPerThread];
  Value values[kItemsPerThread];

  BlockLoadKeys(temp_storage.load_keys)
      .Load(seg_keys_in, keys, segment_size,
            MakeDefaultOobKey<kDirection, Key>());
  __syncthreads();

  // Mirror CUB's register-pressure workaround: moving segment_size through shfl
  // prevents ptxas from holding prior guarded-load predicates across loads.
  static_assert(kBlockThreads % 32 == 0);
  const int32_t valid_items =
      cub::ShuffleIndex<32>(segment_size, 0, 0xffffffff);
  BlockLoadValues(temp_storage.load_values)
      .Load(seg_values_in, values, valid_items);
  __syncthreads();

  if constexpr (kDirection == SortDirection::kDescending) {
    BlockRadixSort(temp_storage.sort)
        .SortDescendingBlockedToStriped(keys, values, /*begin_bit=*/0,
                                        /*end_bit=*/sizeof(Key) * 8);
  } else {
    BlockRadixSort(temp_storage.sort)
        .SortBlockedToStriped(keys, values, /*begin_bit=*/0,
                              /*end_bit=*/sizeof(Key) * 8);
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

template <typename ChainedPolicy, SortDirection kDirection, typename Key>
__launch_bounds__(ChainedPolicy::ActivePolicy::HistogramPolicy::BLOCK_THREADS)
    __global__
    void CubRadixSortHistogramKernel(CubGlobalOffset* __restrict__ d_bins_out,
                                     const Key* __restrict__ d_keys_in,
                                     CubGlobalOffset num_items,
                                     int32_t begin_bit, int32_t end_bit) {
  using HistogramPolicy = typename ChainedPolicy::ActivePolicy::HistogramPolicy;
  using Agent = cub::detail::radix_sort::AgentRadixSortHistogram<
      HistogramPolicy, kDirection == SortDirection::kDescending, Key,
      CubGlobalOffset>;
  __shared__ typename Agent::TempStorage temp_storage;
  Agent agent(temp_storage, d_bins_out, d_keys_in, num_items, begin_bit,
              end_bit);
  agent.Process();
}

template <typename ChainedPolicy, SortDirection kDirection, typename Key>
__launch_bounds__(ChainedPolicy::ActivePolicy::OnesweepPolicy::BLOCK_THREADS)
    __global__ void CubRadixSortOnesweepKeysKernel(
        int32_t* __restrict__ d_lookback, int32_t* __restrict__ d_ctrs,
        CubGlobalOffset* __restrict__ d_bins_out,
        const CubGlobalOffset* __restrict__ d_bins_in,
        Key* __restrict__ d_keys_out, const Key* __restrict__ d_keys_in,
        int64_t portion_item_offset, int32_t portion_num_items,
        int32_t current_bit, int32_t num_bits) {
  using OnesweepPolicy = typename ChainedPolicy::ActivePolicy::OnesweepPolicy;
  using Agent = cub::detail::radix_sort::AgentRadixSortOnesweep<
      OnesweepPolicy, kDirection == SortDirection::kDescending, Key,
      cub::NullType, CubGlobalOffset, int32_t>;
  __shared__ typename Agent::TempStorage temp_storage;
  Agent agent(temp_storage, d_lookback, d_ctrs, d_bins_out, d_bins_in,
              d_keys_out, d_keys_in + portion_item_offset,
              /*d_values_out=*/nullptr, /*d_values_in=*/nullptr,
              portion_num_items, current_bit, num_bits);
  agent.Process();
}

template <typename ChainedPolicy, SortDirection kDirection, typename Key,
          typename Value>
__launch_bounds__(ChainedPolicy::ActivePolicy::OnesweepPolicy::BLOCK_THREADS)
    __global__ void CubRadixSortOnesweepPairsKernel(
        int32_t* __restrict__ d_lookback, int32_t* __restrict__ d_ctrs,
        CubGlobalOffset* __restrict__ d_bins_out,
        const CubGlobalOffset* __restrict__ d_bins_in,
        Key* __restrict__ d_keys_out, const Key* __restrict__ d_keys_in,
        Value* __restrict__ d_values_out, const Value* __restrict__ d_values_in,
        int64_t portion_item_offset, int32_t portion_num_items,
        int32_t current_bit, int32_t num_bits) {
  using OnesweepPolicy = typename ChainedPolicy::ActivePolicy::OnesweepPolicy;
  using Agent = cub::detail::radix_sort::AgentRadixSortOnesweep<
      OnesweepPolicy, kDirection == SortDirection::kDescending, Key, Value,
      CubGlobalOffset, int32_t>;
  __shared__ typename Agent::TempStorage temp_storage;
  Agent agent(temp_storage, d_lookback, d_ctrs, d_bins_out, d_bins_in,
              d_keys_out, d_keys_in + portion_item_offset, d_values_out,
              d_values_in + portion_item_offset, portion_num_items, current_bit,
              num_bits);
  agent.Process();
}

template <typename Return, typename... Args>
KernelHostRef MakeKernelHostRef(Return (*fn)(Args...)) {
  return {reinterpret_cast<const void*>(fn), sizeof...(Args)};
}

template <typename Key>
CubSortTypeKernelRefs GetKeysRefsImpl(SortDirection direction) {
  using Policy = typename XlaChainedPolicy<Key, cub::NullType>::MaxPolicy;
  if (direction == SortDirection::kDescending) {
    return {
        /*single_tile=*/MakeKernelHostRef(
            &CubRadixSortSingleTileKeysKernel<Policy,
                                              SortDirection::kDescending, Key>),
        /*onesweep=*/
        MakeKernelHostRef(
            &CubRadixSortOnesweepKeysKernel<Policy, SortDirection::kDescending,
                                            Key>),
    };
  }
  return {
      /*single_tile=*/MakeKernelHostRef(
          &CubRadixSortSingleTileKeysKernel<Policy, SortDirection::kAscending,
                                            Key>),
      /*onesweep=*/
      MakeKernelHostRef(
          &CubRadixSortOnesweepKeysKernel<Policy, SortDirection::kAscending,
                                          Key>),
  };
}

template <typename Key>
KernelHostRef GetHistogramRefImpl(SortDirection direction) {
  using Policy = typename XlaChainedPolicy<Key, cub::NullType>::MaxPolicy;
  if (direction == SortDirection::kDescending) {
    return MakeKernelHostRef(
        &CubRadixSortHistogramKernel<Policy, SortDirection::kDescending, Key>);
  }
  return MakeKernelHostRef(
      &CubRadixSortHistogramKernel<Policy, SortDirection::kAscending, Key>);
}

template <typename Key, typename Value>
CubSortTypeKernelRefs GetPairsRefsImpl(SortDirection direction) {
  using Policy = typename XlaChainedPolicy<Key, Value>::MaxPolicy;
  if (direction == SortDirection::kDescending) {
    return {
        /*single_tile=*/MakeKernelHostRef(
            &CubRadixSortSingleTilePairsKernel<
                Policy, SortDirection::kDescending, Key, Value>),
        /*onesweep=*/
        MakeKernelHostRef(
            &CubRadixSortOnesweepPairsKernel<Policy, SortDirection::kDescending,
                                             Key, Value>),
    };
  }
  return {
      /*single_tile=*/MakeKernelHostRef(
          &CubRadixSortSingleTilePairsKernel<Policy, SortDirection::kAscending,
                                             Key, Value>),
      /*onesweep=*/
      MakeKernelHostRef(
          &CubRadixSortOnesweepPairsKernel<Policy, SortDirection::kAscending,
                                           Key, Value>),
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
CubSortTypeKernelRefs GetCubSortKeysRefs(SortDirection direction) {
  return GetKeysRefsImpl<typename CudaKeyType<kKeyType>::type>(direction);
}

template <PrimitiveType kKeyType>
KernelHostRef GetCubSortHistogramRef(SortDirection direction) {
  return GetHistogramRefImpl<typename CudaKeyType<kKeyType>::type>(direction);
}

template <PrimitiveType kKeyType, int kValueBitWidth>
CubSortTypeKernelRefs GetCubSortPairsRefs(SortDirection direction) {
  return GetPairsRefsImpl<typename CudaKeyType<kKeyType>::type,
                          typename CudaValueType<kValueBitWidth>::type>(
      direction);
}

// Floating point keys.
#ifdef CUB_TYPE_BF16
template CubSortTypeKernelRefs GetCubSortKeysRefs<BF16>(SortDirection);
template KernelHostRef GetCubSortHistogramRef<BF16>(SortDirection);
#endif
#ifdef CUB_TYPE_F16
template CubSortTypeKernelRefs GetCubSortKeysRefs<F16>(SortDirection);
template KernelHostRef GetCubSortHistogramRef<F16>(SortDirection);
#endif
#ifdef CUB_TYPE_F32
template CubSortTypeKernelRefs GetCubSortKeysRefs<F32>(SortDirection);
template KernelHostRef GetCubSortHistogramRef<F32>(SortDirection);
#endif
#ifdef CUB_TYPE_F64
template CubSortTypeKernelRefs GetCubSortKeysRefs<F64>(SortDirection);
template KernelHostRef GetCubSortHistogramRef<F64>(SortDirection);
#endif

// Signed integer keys.
#ifdef CUB_TYPE_S8
template CubSortTypeKernelRefs GetCubSortKeysRefs<S8>(SortDirection);
template KernelHostRef GetCubSortHistogramRef<S8>(SortDirection);
#endif
#ifdef CUB_TYPE_S16
template CubSortTypeKernelRefs GetCubSortKeysRefs<S16>(SortDirection);
template KernelHostRef GetCubSortHistogramRef<S16>(SortDirection);
#endif
#ifdef CUB_TYPE_S32
template CubSortTypeKernelRefs GetCubSortKeysRefs<S32>(SortDirection);
template KernelHostRef GetCubSortHistogramRef<S32>(SortDirection);
#endif
#ifdef CUB_TYPE_S64
template CubSortTypeKernelRefs GetCubSortKeysRefs<S64>(SortDirection);
template KernelHostRef GetCubSortHistogramRef<S64>(SortDirection);
#endif

// Unsigned integer keys.
#ifdef CUB_TYPE_U8
template CubSortTypeKernelRefs GetCubSortKeysRefs<U8>(SortDirection);
template KernelHostRef GetCubSortHistogramRef<U8>(SortDirection);
#endif
#ifdef CUB_TYPE_U16
template CubSortTypeKernelRefs GetCubSortKeysRefs<U16>(SortDirection);
template KernelHostRef GetCubSortHistogramRef<U16>(SortDirection);
#endif
#ifdef CUB_TYPE_U32
template CubSortTypeKernelRefs GetCubSortKeysRefs<U32>(SortDirection);
template KernelHostRef GetCubSortHistogramRef<U32>(SortDirection);
#endif
#ifdef CUB_TYPE_U64
template CubSortTypeKernelRefs GetCubSortKeysRefs<U64>(SortDirection);
template KernelHostRef GetCubSortHistogramRef<U64>(SortDirection);
#endif

// Pairs with 8-bit key.
#ifdef CUB_TYPE_U8_B16
template CubSortTypeKernelRefs GetCubSortPairsRefs<U8, 16>(SortDirection);
#endif
#ifdef CUB_TYPE_U8_B32
template CubSortTypeKernelRefs GetCubSortPairsRefs<U8, 32>(SortDirection);
#endif
#ifdef CUB_TYPE_U8_B64
template CubSortTypeKernelRefs GetCubSortPairsRefs<U8, 64>(SortDirection);
#endif

// Pairs with 16-bit key.
#ifdef CUB_TYPE_U16_B16
template CubSortTypeKernelRefs GetCubSortPairsRefs<U16, 16>(SortDirection);
#endif
#ifdef CUB_TYPE_U16_B32
template CubSortTypeKernelRefs GetCubSortPairsRefs<U16, 32>(SortDirection);
#endif
#ifdef CUB_TYPE_U16_B64
template CubSortTypeKernelRefs GetCubSortPairsRefs<U16, 64>(SortDirection);
#endif

// Pairs with signed 32-bit key.
#ifdef CUB_TYPE_S32_B16
template CubSortTypeKernelRefs GetCubSortPairsRefs<S32, 16>(SortDirection);
#endif
#ifdef CUB_TYPE_S32_B32
template CubSortTypeKernelRefs GetCubSortPairsRefs<S32, 32>(SortDirection);
#endif
#ifdef CUB_TYPE_S32_B64
template CubSortTypeKernelRefs GetCubSortPairsRefs<S32, 64>(SortDirection);
#endif

// Pairs with unsigned 32-bit key.
#ifdef CUB_TYPE_U32_B16
template CubSortTypeKernelRefs GetCubSortPairsRefs<U32, 16>(SortDirection);
#endif
#ifdef CUB_TYPE_U32_B32
template CubSortTypeKernelRefs GetCubSortPairsRefs<U32, 32>(SortDirection);
#endif
#ifdef CUB_TYPE_U32_B64
template CubSortTypeKernelRefs GetCubSortPairsRefs<U32, 64>(SortDirection);
#endif

// Pairs with 64-bit key.
#ifdef CUB_TYPE_U64_B16
template CubSortTypeKernelRefs GetCubSortPairsRefs<U64, 16>(SortDirection);
#endif
#ifdef CUB_TYPE_U64_B32
template CubSortTypeKernelRefs GetCubSortPairsRefs<U64, 32>(SortDirection);
#endif
#ifdef CUB_TYPE_U64_B64
template CubSortTypeKernelRefs GetCubSortPairsRefs<U64, 64>(SortDirection);
#endif

// Pairs with f32 key.
#ifdef CUB_TYPE_F32_B16
template CubSortTypeKernelRefs GetCubSortPairsRefs<F32, 16>(SortDirection);
#endif
#ifdef CUB_TYPE_F32_B32
template CubSortTypeKernelRefs GetCubSortPairsRefs<F32, 32>(SortDirection);
#endif
#ifdef CUB_TYPE_F32_B64
template CubSortTypeKernelRefs GetCubSortPairsRefs<F32, 64>(SortDirection);
#endif

}  // namespace xla::gpu::internal
