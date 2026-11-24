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

#include "xla/backends/gpu/libraries/cub/cub_sort_kernels.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_policy.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_elf_utils.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

// Reference comparison matching CUB's radix sort ordering:
// - Integers: numeric `<` / `>`.
// - Floats: total-order-like except -0.0 and +0.0 compare equal (preserving
//   input order stably via std::stable_sort).
template <typename KeyT>
bool KeyLessThan(KeyT a, KeyT b, bool descending) {
  if constexpr (std::is_integral_v<KeyT>) {
    return descending ? (a > b) : (a < b);
  } else {
    auto to_ordered = [](KeyT v) -> uint64_t {
      if constexpr (sizeof(KeyT) == 2) {
        uint16_t bits = absl::bit_cast<uint16_t>(v);
        if ((bits & 0x7FFFu) == 0) {
          bits = 0;  // Equate -0.0 and +0.0.
        }
        uint16_t mask =
            (bits & 0x8000u) ? uint16_t{0xFFFFu} : uint16_t{0x8000u};
        return static_cast<uint64_t>(bits ^ mask);
      } else if constexpr (sizeof(KeyT) == 4) {
        uint32_t bits = absl::bit_cast<uint32_t>(v);
        if ((bits & 0x7FFFFFFFu) == 0) {
          bits = 0;
        }
        uint32_t mask = (bits & 0x80000000u) ? uint32_t{0xFFFFFFFFu}
                                             : uint32_t{0x80000000u};
        return static_cast<uint64_t>(bits ^ mask);
      } else {
        uint64_t bits = absl::bit_cast<uint64_t>(v);
        if ((bits & 0x7FFFFFFFFFFFFFFFULL) == 0) {
          bits = 0;
        }
        uint64_t mask = (bits & 0x8000000000000000ULL) ? 0xFFFFFFFFFFFFFFFFULL
                                                       : 0x8000000000000000ULL;
        return bits ^ mask;
      }
    };
    uint64_t oa = to_ordered(a);
    uint64_t ob = to_ordered(b);
    return descending ? (oa > ob) : (oa < ob);
  }
}

template <typename KeyT>
std::vector<KeyT> GenerateKeys(int64_t count, uint32_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<KeyT> keys(count);
  if constexpr (std::is_integral_v<KeyT>) {
    std::uniform_int_distribution<uint64_t> dist(0, 1000000);
    for (int64_t i = 0; i < count; ++i) {
      // Mix wide range and duplicates to exercise tie-breaking and stability.
      uint64_t raw = (i % 7 == 0) ? (dist(rng) % 16) : rng();
      keys[i] = static_cast<KeyT>(raw);
    }
  } else {
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (int64_t i = 0; i < count; ++i) {
      double val = (i % 11 == 0) ? static_cast<double>((i % 5) - 2) : dist(rng);
      if (i % 37 == 0) {
        val = -0.0;
      } else if (i % 41 == 0) {
        val = 0.0;
      }
      keys[i] = static_cast<KeyT>(val);
    }
  }
  return keys;
}

template <typename KeyT>
void ExpectKeysBitEqual(const std::vector<KeyT>& actual,
                        const std::vector<KeyT>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    if constexpr (sizeof(KeyT) == 1) {
      const auto a = absl::bit_cast<uint8_t>(actual[i]);
      const auto e = absl::bit_cast<uint8_t>(expected[i]);
      if (a != e) {
        ADD_FAILURE() << "Mismatch at index " << i << ": actual=" << int{a}
                      << " expected=" << int{e};
        return;
      }
    } else if constexpr (sizeof(KeyT) == 2) {
      const auto a = absl::bit_cast<uint16_t>(actual[i]);
      const auto e = absl::bit_cast<uint16_t>(expected[i]);
      if (a != e) {
        ADD_FAILURE() << "Mismatch at index " << i << ": actual=0x" << std::hex
                      << a << " expected=0x" << e << std::dec;
        return;
      }
    } else if constexpr (sizeof(KeyT) == 4) {
      const auto a = absl::bit_cast<uint32_t>(actual[i]);
      const auto e = absl::bit_cast<uint32_t>(expected[i]);
      if (a != e) {
        ADD_FAILURE() << "Mismatch at index " << i << ": actual=0x" << std::hex
                      << a << " expected=0x" << e << std::dec;
        return;
      }
    } else {
      const auto a = absl::bit_cast<uint64_t>(actual[i]);
      const auto e = absl::bit_cast<uint64_t>(expected[i]);
      if (a != e) {
        ADD_FAILURE() << "Mismatch at index " << i << ": actual=0x" << std::hex
                      << a << " expected=0x" << e << std::dec;
        return;
      }
    }
  }
}

class CubSortKernelsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                         se::PlatformManager::PlatformWithName("CUDA"));
    ASSERT_OK_AND_ASSIGN(executor_, platform->ExecutorForDevice(0));
    ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream());
    cc_ = executor_->GetDeviceDescription().cuda_compute_capability();
    sm_version_ = CanonicalCubSortPolicySmVersion(cc_.major);
  }

  template <typename KeyT>
  void RunSingleTileKeysWithInput(PrimitiveType key_type,
                                  const std::vector<KeyT>& h_in,
                                  int32_t segment_size, int64_t batch_size,
                                  bool descending, bool split_chunks = false) {
    ASSERT_OK_AND_ASSIGN(se::KernelLoaderSpec spec,
                         GetCubSortSingleTileKernelSpec(key_type, std::nullopt,
                                                        descending, cc_));
    using KernelFactory =
        se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                               int32_t>;
    ASSERT_OK_AND_ASSIGN(auto kernel, KernelFactory::Create(executor_, spec));

    const CubRadixSortPolicySpec policy = GetCubRadixSortPolicySpec(
        sm_version_, sizeof(KeyT), /*value_byte_size=*/0,
        std::is_same_v<KeyT, float>);
    ASSERT_LE(segment_size, policy.single_tile.tile_items());

    const int64_t total_items = batch_size * segment_size;
    ASSERT_EQ(h_in.size(), total_items);
    std::vector<KeyT> h_expected = h_in;
    for (int64_t b = 0; b < batch_size; ++b) {
      auto begin = h_expected.begin() + b * segment_size;
      std::stable_sort(begin, begin + segment_size,
                       [descending](KeyT a, KeyT b) {
                         return KeyLessThan(a, b, descending);
                       });
    }

    const int64_t byte_size = total_items * sizeof(KeyT);
    const int64_t alloc_bytes = std::max<int64_t>(byte_size, sizeof(KeyT));
    se::DeviceAddressBase d_in = executor_->Allocate(alloc_bytes, 0);
    se::DeviceAddressBase d_out = executor_->Allocate(alloc_bytes, 0);
    absl::Cleanup free_bufs = [&] {
      executor_->Deallocate(&d_in);
      executor_->Deallocate(&d_out);
    };
    if (byte_size > 0) {
      ASSERT_OK(stream_->Memcpy(&d_in, h_in.data(), byte_size));
    }
    ASSERT_OK(stream_->MemZero(&d_out, alloc_bytes));

    // Pass explicit 0 dynamic shared memory: CUB kernels use static __shared__
    // storage only, and TypedKernel::Launch without an explicit shmem_bytes
    // argument forwards static shared memory metadata as dynamic shared memory
    // (doubling shared memory usage and exceeding T4's 64 KiB opt-in limit).
    if (split_chunks && batch_size >= 2) {
      const int64_t first = batch_size / 2;
      const int64_t first_bytes = first * segment_size * sizeof(KeyT);
      se::DeviceAddressBase d_in_second(
          static_cast<char*>(d_in.opaque()) + first_bytes,
          byte_size - first_bytes);
      se::DeviceAddressBase d_out_second(
          static_cast<char*>(d_out.opaque()) + first_bytes,
          byte_size - first_bytes);
      ASSERT_OK(kernel.Launch(se::ThreadDim(policy.single_tile.block_threads),
                              se::BlockDim(first), /*shmem_bytes=*/0,
                              stream_.get(), d_in, d_out, segment_size));
      ASSERT_OK(kernel.Launch(se::ThreadDim(policy.single_tile.block_threads),
                              se::BlockDim(batch_size - first),
                              /*shmem_bytes=*/0, stream_.get(), d_in_second,
                              d_out_second, segment_size));
    } else {
      ASSERT_OK(kernel.Launch(se::ThreadDim(policy.single_tile.block_threads),
                              se::BlockDim(batch_size), /*shmem_bytes=*/0,
                              stream_.get(), d_in, d_out, segment_size));
    }

    std::vector<KeyT> h_out(total_items);
    if (byte_size > 0) {
      ASSERT_OK(stream_->Memcpy(h_out.data(), d_out, byte_size));
    }
    ASSERT_OK(stream_->BlockHostUntilDone());

    ExpectKeysBitEqual(h_out, h_expected);
  }

  template <typename KeyT>
  void RunSingleTileKeys(PrimitiveType key_type, int32_t segment_size,
                         int64_t batch_size, bool descending,
                         bool split_chunks = false) {
    RunSingleTileKeysWithInput<KeyT>(
        key_type, GenerateKeys<KeyT>(batch_size * segment_size, 12345),
        segment_size, batch_size, descending, split_chunks);
  }

  template <typename KeyT, typename ValT>
  void RunSingleTilePairs(PrimitiveType key_type, PrimitiveType val_type,
                          const std::vector<KeyT>& h_keys_in,
                          int32_t segment_size, int64_t batch_size,
                          bool descending) {
    ASSERT_OK_AND_ASSIGN(
        se::KernelLoaderSpec spec,
        GetCubSortSingleTileKernelSpec(key_type, val_type, descending, cc_));
    using KernelFactory =
        se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                               se::DeviceAddressBase, se::DeviceAddressBase,
                               int32_t>;
    ASSERT_OK_AND_ASSIGN(auto kernel, KernelFactory::Create(executor_, spec));

    const CubRadixSortPolicySpec policy = GetCubRadixSortPolicySpec(
        sm_version_, sizeof(KeyT), sizeof(ValT), std::is_same_v<KeyT, float>);
    ASSERT_LE(segment_size, policy.single_tile.tile_items());

    const int64_t total_items = batch_size * segment_size;
    ASSERT_EQ(h_keys_in.size(), total_items);
    std::vector<ValT> h_vals_in(total_items);
    for (int64_t i = 0; i < total_items; ++i) {
      h_vals_in[i] = static_cast<ValT>(i);
    }

    std::vector<std::pair<KeyT, ValT>> pairs(total_items);
    for (int64_t i = 0; i < total_items; ++i) {
      pairs[i] = {h_keys_in[i], h_vals_in[i]};
    }
    for (int64_t b = 0; b < batch_size; ++b) {
      auto begin = pairs.begin() + b * segment_size;
      std::stable_sort(begin, begin + segment_size,
                       [descending](const auto& a, const auto& b) {
                         return KeyLessThan(a.first, b.first, descending);
                       });
    }
    std::vector<KeyT> expected_keys(total_items);
    std::vector<ValT> expected_vals(total_items);
    for (int64_t i = 0; i < total_items; ++i) {
      expected_keys[i] = pairs[i].first;
      expected_vals[i] = pairs[i].second;
    }

    const int64_t key_bytes = total_items * sizeof(KeyT);
    const int64_t val_bytes = total_items * sizeof(ValT);
    const int64_t key_alloc = std::max<int64_t>(key_bytes, sizeof(KeyT));
    const int64_t val_alloc = std::max<int64_t>(val_bytes, sizeof(ValT));
    se::DeviceAddressBase d_keys_in = executor_->Allocate(key_alloc, 0);
    se::DeviceAddressBase d_keys_out = executor_->Allocate(key_alloc, 0);
    se::DeviceAddressBase d_vals_in = executor_->Allocate(val_alloc, 0);
    se::DeviceAddressBase d_vals_out = executor_->Allocate(val_alloc, 0);
    absl::Cleanup free_bufs = [&] {
      executor_->Deallocate(&d_keys_in);
      executor_->Deallocate(&d_keys_out);
      executor_->Deallocate(&d_vals_in);
      executor_->Deallocate(&d_vals_out);
    };
    if (key_bytes > 0) {
      ASSERT_OK(stream_->Memcpy(&d_keys_in, h_keys_in.data(), key_bytes));
      ASSERT_OK(stream_->Memcpy(&d_vals_in, h_vals_in.data(), val_bytes));
    }

    ASSERT_OK(kernel.Launch(se::ThreadDim(policy.single_tile.block_threads),
                            se::BlockDim(batch_size), /*shmem_bytes=*/0,
                            stream_.get(), d_keys_in, d_keys_out, d_vals_in,
                            d_vals_out, segment_size));

    std::vector<KeyT> h_keys_out(total_items);
    std::vector<ValT> h_vals_out(total_items);
    if (key_bytes > 0) {
      ASSERT_OK(stream_->Memcpy(h_keys_out.data(), d_keys_out, key_bytes));
      ASSERT_OK(stream_->Memcpy(h_vals_out.data(), d_vals_out, val_bytes));
    }
    ASSERT_OK(stream_->BlockHostUntilDone());

    ExpectKeysBitEqual(h_keys_out, expected_keys);
    EXPECT_EQ(h_vals_out, expected_vals);
  }

  template <typename KeyT>
  void RunOnesweepKeys(PrimitiveType key_type,
                       const std::vector<KeyT>& h_keys_in, bool descending,
                       int32_t portion_size_override = 0) {
    ASSERT_OK_AND_ASSIGN(
        CubSortOnesweepKernelSpecs specs,
        GetCubSortOnesweepKernelSpecs(key_type, std::nullopt, descending, cc_));
    if (se::cuda::IsCudaElf(
            specs.histogram.cuda_cubin_in_memory()->cubin_bytes)) {
      ASSERT_TRUE(specs.histogram_attrs.has_value());
      EXPECT_GT(specs.histogram_attrs->num_regs, 0);
      EXPECT_GT(specs.histogram_attrs->static_shared_size_bytes, 0);
    }

    using HistoFactory =
        se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                               CubGlobalOffset, int32_t, int32_t>;
    using ExclSumFactory =
        se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                               se::DeviceAddressBase, int32_t>;
    using OnesweepFactory =
        se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                               se::DeviceAddressBase, se::DeviceAddressBase,
                               se::DeviceAddressBase, se::DeviceAddressBase,
                               int64_t, int32_t, int32_t, int32_t>;

    ASSERT_OK_AND_ASSIGN(auto histo_kernel,
                         HistoFactory::Create(executor_, specs.histogram));
    ASSERT_OK_AND_ASSIGN(
        auto excl_sum_kernel,
        ExclSumFactory::Create(executor_, specs.exclusive_sum));
    ASSERT_OK_AND_ASSIGN(auto onesweep_kernel,
                         OnesweepFactory::Create(executor_, specs.onesweep));

    const CubRadixSortPolicySpec policy = GetCubRadixSortPolicySpec(
        sm_version_, sizeof(KeyT), /*value_byte_size=*/0,
        std::is_same_v<KeyT, float>);
    const int64_t num_items = h_keys_in.size();
    const int num_passes = static_cast<int>(sizeof(KeyT));
    const int32_t tile_items = policy.onesweep.tile_items();
    const int32_t portion_size =
        portion_size_override > 0
            ? portion_size_override
            : (kCubSortMaxPortionLimit / tile_items) * tile_items;
    const int64_t num_portions = (num_items + portion_size - 1) / portion_size;
    const int32_t max_blocks =
        (std::min<int64_t>(num_items, portion_size) + tile_items - 1) /
        tile_items;

    const int64_t key_bytes = num_items * sizeof(KeyT);
    const int64_t bins_pass_bytes =
        num_passes * kCubSortOnesweepRadixDigits * sizeof(CubGlobalOffset);
    const int64_t lookback_bytes = static_cast<int64_t>(max_blocks) *
                                   kCubSortOnesweepRadixDigits *
                                   sizeof(int32_t);
    const int64_t ctrs_all_bytes = static_cast<int64_t>(num_passes) *
                                   num_portions * kCubSortScratchAlignmentBytes;

    se::DeviceAddressBase d_in = executor_->Allocate(key_bytes, 0);
    se::DeviceAddressBase d_out = executor_->Allocate(key_bytes, 0);
    se::DeviceAddressBase d_tmp = executor_->Allocate(key_bytes, 0);
    se::DeviceAddressBase d_bins_raw = executor_->Allocate(bins_pass_bytes, 0);
    std::vector<se::DeviceAddressBase> d_bins_portions(num_portions);
    for (int64_t p = 0; p < num_portions; ++p) {
      d_bins_portions[p] = executor_->Allocate(bins_pass_bytes, 0);
    }
    se::DeviceAddressBase d_lookback = executor_->Allocate(lookback_bytes, 0);
    se::DeviceAddressBase d_ctrs_all = executor_->Allocate(ctrs_all_bytes, 0);
    absl::Cleanup free_bufs = [&] {
      executor_->Deallocate(&d_in);
      executor_->Deallocate(&d_out);
      executor_->Deallocate(&d_tmp);
      executor_->Deallocate(&d_bins_raw);
      for (auto& buf : d_bins_portions) {
        executor_->Deallocate(&buf);
      }
      executor_->Deallocate(&d_lookback);
      executor_->Deallocate(&d_ctrs_all);
    };

    ASSERT_OK(stream_->Memcpy(&d_in, h_keys_in.data(), key_bytes));
    ASSERT_OK(stream_->MemZero(&d_bins_raw, bins_pass_bytes));
    // Poison d_ctrs_all with non-zero bytes to verify ExclusiveSum zeroes all
    // counter slots.
    ASSERT_OK(stream_->Memset32(&d_ctrs_all, 0x5a5a5a5au, ctrs_all_bytes));

    const int64_t histo_grid =
        std::min<int64_t>(executor_->GetDeviceDescription().core_count() * 2,
                          (num_items + policy.histogram.tile_items() - 1) /
                              policy.histogram.tile_items());
    ASSERT_OK(histo_kernel.Launch(
        se::ThreadDim(policy.histogram.block_threads), se::BlockDim(histo_grid),
        /*shmem_bytes=*/0, stream_.get(), d_bins_raw, d_in,
        static_cast<CubGlobalOffset>(num_items), int32_t{0},
        static_cast<int32_t>(sizeof(KeyT) * 8)));
    ASSERT_OK(excl_sum_kernel.Launch(
        se::ThreadDim(policy.exclusive_sum.block_threads),
        se::BlockDim(num_passes), /*shmem_bytes=*/0, stream_.get(), d_bins_raw,
        d_bins_portions[0], d_ctrs_all, static_cast<int32_t>(num_portions)));

    for (int pass = 0; pass < num_passes; ++pass) {
      const int rem = num_passes - 1 - pass;
      se::DeviceAddressBase dst = (rem % 2 == 0) ? d_out : d_tmp;
      se::DeviceAddressBase src =
          (pass == 0) ? d_in : ((rem % 2 == 0) ? d_tmp : d_out);
      const int64_t pass_bin_byte_offset = static_cast<int64_t>(pass) *
                                           kCubSortOnesweepRadixDigits *
                                           sizeof(CubGlobalOffset);

      for (int64_t portion = 0; portion < num_portions; ++portion) {
        const int64_t item_offset = portion * portion_size;
        const int32_t portion_items = static_cast<int32_t>(
            std::min<int64_t>(num_items - item_offset, portion_size));
        const int32_t num_blocks =
            (portion_items + tile_items - 1) / tile_items;

        ASSERT_OK(stream_->MemZero(&d_lookback, lookback_bytes));

        const int64_t ctr_slot = portion * num_passes + pass;
        se::DeviceAddressBase d_ctr(
            static_cast<char*>(d_ctrs_all.opaque()) +
                ctr_slot * kCubSortScratchAlignmentBytes,
            sizeof(int32_t));
        se::DeviceAddressBase bins_in(
            static_cast<char*>(d_bins_portions[portion].opaque()) +
                pass_bin_byte_offset,
            kCubSortOnesweepRadixDigits * sizeof(CubGlobalOffset));
        se::DeviceAddressBase bins_out =
            (portion + 1 < num_portions)
                ? se::DeviceAddressBase(
                      static_cast<char*>(
                          d_bins_portions[portion + 1].opaque()) +
                          pass_bin_byte_offset,
                      kCubSortOnesweepRadixDigits * sizeof(CubGlobalOffset))
                : se::DeviceAddressBase(nullptr, 0);

        ASSERT_OK(onesweep_kernel.Launch(
            se::ThreadDim(policy.onesweep.block_threads),
            se::BlockDim(num_blocks), /*shmem_bytes=*/0, stream_.get(),
            d_lookback, d_ctr, bins_out, bins_in, dst, src, item_offset,
            portion_items, pass * kCubSortOnesweepRadixBits,
            kCubSortOnesweepRadixBits));
      }
    }

    std::vector<KeyT> h_out(num_items);
    ASSERT_OK(stream_->Memcpy(h_out.data(), d_out, key_bytes));
    ASSERT_OK(stream_->BlockHostUntilDone());

    std::vector<KeyT> expected = h_keys_in;
    absl::c_stable_sort(expected, [descending](KeyT a, KeyT b) {
      return KeyLessThan(a, b, descending);
    });
    ExpectKeysBitEqual(h_out, expected);
  }

  template <typename KeyT, typename ValT>
  void RunOnesweepPairs(PrimitiveType key_type, PrimitiveType val_type,
                        const std::vector<KeyT>& h_keys_in, bool descending,
                        int32_t portion_size_override = 0) {
    ASSERT_OK_AND_ASSIGN(
        CubSortOnesweepKernelSpecs specs,
        GetCubSortOnesweepKernelSpecs(key_type, val_type, descending, cc_));
    if (se::cuda::IsCudaElf(
            specs.histogram.cuda_cubin_in_memory()->cubin_bytes)) {
      ASSERT_TRUE(specs.histogram_attrs.has_value());
      EXPECT_GT(specs.histogram_attrs->num_regs, 0);
      EXPECT_GT(specs.histogram_attrs->static_shared_size_bytes, 0);
    }

    using HistoFactory =
        se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                               CubGlobalOffset, int32_t, int32_t>;
    using ExclSumFactory =
        se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                               se::DeviceAddressBase, int32_t>;
    using OnesweepFactory =
        se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                               se::DeviceAddressBase, se::DeviceAddressBase,
                               se::DeviceAddressBase, se::DeviceAddressBase,
                               se::DeviceAddressBase, se::DeviceAddressBase,
                               int64_t, int32_t, int32_t, int32_t>;

    ASSERT_OK_AND_ASSIGN(auto histo_kernel,
                         HistoFactory::Create(executor_, specs.histogram));
    ASSERT_OK_AND_ASSIGN(
        auto excl_sum_kernel,
        ExclSumFactory::Create(executor_, specs.exclusive_sum));
    ASSERT_OK_AND_ASSIGN(auto onesweep_kernel,
                         OnesweepFactory::Create(executor_, specs.onesweep));

    const CubRadixSortPolicySpec policy = GetCubRadixSortPolicySpec(
        sm_version_, sizeof(KeyT), sizeof(ValT), std::is_same_v<KeyT, float>);
    const int64_t num_items = h_keys_in.size();
    std::vector<ValT> h_vals_in(num_items);
    for (int64_t i = 0; i < num_items; ++i) {
      h_vals_in[i] = static_cast<ValT>(i);
    }

    const int num_passes = static_cast<int>(sizeof(KeyT));
    const int32_t tile_items = policy.onesweep.tile_items();
    const int32_t portion_size =
        portion_size_override > 0
            ? portion_size_override
            : (kCubSortMaxPortionLimit / tile_items) * tile_items;
    const int64_t num_portions = (num_items + portion_size - 1) / portion_size;
    const int32_t max_blocks =
        (std::min<int64_t>(num_items, portion_size) + tile_items - 1) /
        tile_items;

    const int64_t key_bytes = num_items * sizeof(KeyT);
    const int64_t val_bytes = num_items * sizeof(ValT);
    const int64_t bins_pass_bytes =
        num_passes * kCubSortOnesweepRadixDigits * sizeof(CubGlobalOffset);
    const int64_t lookback_bytes = static_cast<int64_t>(max_blocks) *
                                   kCubSortOnesweepRadixDigits *
                                   sizeof(int32_t);
    const int64_t ctrs_all_bytes = static_cast<int64_t>(num_passes) *
                                   num_portions * kCubSortScratchAlignmentBytes;

    se::DeviceAddressBase d_keys_in = executor_->Allocate(key_bytes, 0);
    se::DeviceAddressBase d_keys_out = executor_->Allocate(key_bytes, 0);
    se::DeviceAddressBase d_keys_tmp = executor_->Allocate(key_bytes, 0);
    se::DeviceAddressBase d_vals_in = executor_->Allocate(val_bytes, 0);
    se::DeviceAddressBase d_vals_out = executor_->Allocate(val_bytes, 0);
    se::DeviceAddressBase d_vals_tmp = executor_->Allocate(val_bytes, 0);
    se::DeviceAddressBase d_bins_raw = executor_->Allocate(bins_pass_bytes, 0);
    std::vector<se::DeviceAddressBase> d_bins_portions(num_portions);
    for (int64_t p = 0; p < num_portions; ++p) {
      d_bins_portions[p] = executor_->Allocate(bins_pass_bytes, 0);
    }
    se::DeviceAddressBase d_lookback = executor_->Allocate(lookback_bytes, 0);
    se::DeviceAddressBase d_ctrs_all = executor_->Allocate(ctrs_all_bytes, 0);
    absl::Cleanup free_bufs = [&] {
      executor_->Deallocate(&d_keys_in);
      executor_->Deallocate(&d_keys_out);
      executor_->Deallocate(&d_keys_tmp);
      executor_->Deallocate(&d_vals_in);
      executor_->Deallocate(&d_vals_out);
      executor_->Deallocate(&d_vals_tmp);
      executor_->Deallocate(&d_bins_raw);
      for (auto& buf : d_bins_portions) {
        executor_->Deallocate(&buf);
      }
      executor_->Deallocate(&d_lookback);
      executor_->Deallocate(&d_ctrs_all);
    };

    ASSERT_OK(stream_->Memcpy(&d_keys_in, h_keys_in.data(), key_bytes));
    ASSERT_OK(stream_->Memcpy(&d_vals_in, h_vals_in.data(), val_bytes));
    ASSERT_OK(stream_->MemZero(&d_bins_raw, bins_pass_bytes));
    ASSERT_OK(stream_->Memset32(&d_ctrs_all, 0x5a5a5a5au, ctrs_all_bytes));

    const int64_t histo_grid =
        std::min<int64_t>(executor_->GetDeviceDescription().core_count() * 2,
                          (num_items + policy.histogram.tile_items() - 1) /
                              policy.histogram.tile_items());
    ASSERT_OK(histo_kernel.Launch(
        se::ThreadDim(policy.histogram.block_threads), se::BlockDim(histo_grid),
        /*shmem_bytes=*/0, stream_.get(), d_bins_raw, d_keys_in,
        static_cast<CubGlobalOffset>(num_items), int32_t{0},
        static_cast<int32_t>(sizeof(KeyT) * 8)));
    ASSERT_OK(excl_sum_kernel.Launch(
        se::ThreadDim(policy.exclusive_sum.block_threads),
        se::BlockDim(num_passes), /*shmem_bytes=*/0, stream_.get(), d_bins_raw,
        d_bins_portions[0], d_ctrs_all, static_cast<int32_t>(num_portions)));

    for (int pass = 0; pass < num_passes; ++pass) {
      const int rem = num_passes - 1 - pass;
      se::DeviceAddressBase k_dst = (rem % 2 == 0) ? d_keys_out : d_keys_tmp;
      se::DeviceAddressBase k_src =
          (pass == 0) ? d_keys_in : ((rem % 2 == 0) ? d_keys_tmp : d_keys_out);
      se::DeviceAddressBase v_dst = (rem % 2 == 0) ? d_vals_out : d_vals_tmp;
      se::DeviceAddressBase v_src =
          (pass == 0) ? d_vals_in : ((rem % 2 == 0) ? d_vals_tmp : d_vals_out);
      const int64_t pass_bin_byte_offset = static_cast<int64_t>(pass) *
                                           kCubSortOnesweepRadixDigits *
                                           sizeof(CubGlobalOffset);

      for (int64_t portion = 0; portion < num_portions; ++portion) {
        const int64_t item_offset = portion * portion_size;
        const int32_t portion_items = static_cast<int32_t>(
            std::min<int64_t>(num_items - item_offset, portion_size));
        const int32_t num_blocks =
            (portion_items + tile_items - 1) / tile_items;

        ASSERT_OK(stream_->MemZero(&d_lookback, lookback_bytes));

        const int64_t ctr_slot = portion * num_passes + pass;
        se::DeviceAddressBase d_ctr(
            static_cast<char*>(d_ctrs_all.opaque()) +
                ctr_slot * kCubSortScratchAlignmentBytes,
            sizeof(int32_t));
        se::DeviceAddressBase bins_in(
            static_cast<char*>(d_bins_portions[portion].opaque()) +
                pass_bin_byte_offset,
            kCubSortOnesweepRadixDigits * sizeof(CubGlobalOffset));
        se::DeviceAddressBase bins_out =
            (portion + 1 < num_portions)
                ? se::DeviceAddressBase(
                      static_cast<char*>(
                          d_bins_portions[portion + 1].opaque()) +
                          pass_bin_byte_offset,
                      kCubSortOnesweepRadixDigits * sizeof(CubGlobalOffset))
                : se::DeviceAddressBase(nullptr, 0);

        ASSERT_OK(onesweep_kernel.Launch(
            se::ThreadDim(policy.onesweep.block_threads),
            se::BlockDim(num_blocks), /*shmem_bytes=*/0, stream_.get(),
            d_lookback, d_ctr, bins_out, bins_in, k_dst, k_src, v_dst, v_src,
            item_offset, portion_items, pass * kCubSortOnesweepRadixBits,
            kCubSortOnesweepRadixBits));
      }
    }

    std::vector<KeyT> h_keys_out(num_items);
    std::vector<ValT> h_vals_out(num_items);
    ASSERT_OK(stream_->Memcpy(h_keys_out.data(), d_keys_out, key_bytes));
    ASSERT_OK(stream_->Memcpy(h_vals_out.data(), d_vals_out, val_bytes));
    ASSERT_OK(stream_->BlockHostUntilDone());

    std::vector<std::pair<KeyT, ValT>> pairs(num_items);
    for (int64_t i = 0; i < num_items; ++i) {
      pairs[i] = {h_keys_in[i], h_vals_in[i]};
    }
    absl::c_stable_sort(pairs, [descending](const auto& a, const auto& b) {
      return KeyLessThan(a.first, b.first, descending);
    });
    std::vector<KeyT> expected_keys(num_items);
    std::vector<ValT> expected_vals(num_items);
    for (int64_t i = 0; i < num_items; ++i) {
      expected_keys[i] = pairs[i].first;
      expected_vals[i] = pairs[i].second;
    }
    ExpectKeysBitEqual(h_keys_out, expected_keys);
    EXPECT_EQ(h_vals_out, expected_vals);
  }

  template <typename KeyT>
  void RunSegmentedMultiPassKeys(PrimitiveType key_type,
                                 const std::vector<KeyT>& h_keys_in,
                                 int32_t segment_size, int64_t batch_size,
                                 bool descending) {
    ASSERT_OK_AND_ASSIGN(CubSortSegmentedKernelSpecs specs,
                         GetCubSortSegmentedKernelSpecs(key_type, std::nullopt,
                                                        descending, cc_));

    using SegFactory =
        se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                               int32_t, int32_t, int32_t>;
    ASSERT_OK_AND_ASSIGN(auto primary_kernel,
                         SegFactory::Create(executor_, specs.primary));
    ASSERT_OK_AND_ASSIGN(auto alt_kernel,
                         SegFactory::Create(executor_, specs.alt));

    const CubRadixSortPolicySpec policy = GetCubRadixSortPolicySpec(
        sm_version_, sizeof(KeyT), /*value_byte_size=*/0,
        std::is_same_v<KeyT, float>);
    const int total_bits = static_cast<int>(sizeof(KeyT) * 8);
    const int primary_bits = policy.segmented.radix_bits;
    const int alt_bits = policy.alt_segmented.radix_bits;
    const int num_passes = (total_bits + primary_bits - 1) / primary_bits;
    const int max_alt_passes = num_passes * primary_bits - total_bits;
    const int alt_end_bit = std::min(total_bits, max_alt_passes * alt_bits);

    const int64_t total_items = batch_size * segment_size;
    ASSERT_EQ(h_keys_in.size(), total_items);
    const int64_t key_bytes = total_items * sizeof(KeyT);
    se::DeviceAddressBase d_keys_in = executor_->Allocate(key_bytes, 0);
    se::DeviceAddressBase d_keys_out = executor_->Allocate(key_bytes, 0);
    se::DeviceAddressBase d_keys_tmp = executor_->Allocate(key_bytes, 0);
    absl::Cleanup free_bufs = [&] {
      executor_->Deallocate(&d_keys_in);
      executor_->Deallocate(&d_keys_out);
      executor_->Deallocate(&d_keys_tmp);
    };
    ASSERT_OK(stream_->Memcpy(&d_keys_in, h_keys_in.data(), key_bytes));
    ASSERT_OK(stream_->MemZero(&d_keys_out, key_bytes));
    ASSERT_OK(stream_->MemZero(&d_keys_tmp, key_bytes));

    int current_bit = 0;
    for (int pass = 0; pass < num_passes; ++pass) {
      const bool use_alt = current_bit < alt_end_bit;
      const int radix_bits = use_alt ? alt_bits : primary_bits;
      const int pass_bits = std::min(radix_bits, total_bits - current_bit);
      const int block_threads = use_alt ? policy.alt_segmented.block_threads
                                        : policy.segmented.block_threads;
      auto& kernel = use_alt ? alt_kernel : primary_kernel;

      const int rem = num_passes - 1 - pass;
      se::DeviceAddressBase k_dst = (rem % 2 == 0) ? d_keys_out : d_keys_tmp;
      se::DeviceAddressBase k_src =
          (pass == 0) ? d_keys_in : ((rem % 2 == 0) ? d_keys_tmp : d_keys_out);

      const int64_t first_chunk = std::max<int64_t>(1, batch_size / 2);
      ASSERT_OK(kernel.Launch(se::ThreadDim(block_threads),
                              se::BlockDim(first_chunk), /*shmem_bytes=*/0,
                              stream_.get(), k_src, k_dst, segment_size,
                              current_bit, pass_bits));
      if (batch_size > first_chunk) {
        const int64_t key_off = first_chunk * segment_size * sizeof(KeyT);
        se::DeviceAddressBase k_src_2(
            static_cast<char*>(k_src.opaque()) + key_off, key_bytes - key_off);
        se::DeviceAddressBase k_dst_2(
            static_cast<char*>(k_dst.opaque()) + key_off, key_bytes - key_off);
        ASSERT_OK(kernel.Launch(se::ThreadDim(block_threads),
                                se::BlockDim(batch_size - first_chunk),
                                /*shmem_bytes=*/0, stream_.get(), k_src_2,
                                k_dst_2, segment_size, current_bit, pass_bits));
      }
      current_bit += pass_bits;
    }

    std::vector<KeyT> h_keys_out(total_items);
    std::vector<KeyT> h_keys_in_after(total_items);
    ASSERT_OK(stream_->Memcpy(h_keys_out.data(), d_keys_out, key_bytes));
    ASSERT_OK(stream_->Memcpy(h_keys_in_after.data(), d_keys_in, key_bytes));
    ASSERT_OK(stream_->BlockHostUntilDone());

    // Verify input buffer was not mutated.
    ExpectKeysBitEqual(h_keys_in_after, h_keys_in);

    std::vector<KeyT> expected_keys = h_keys_in;
    for (int64_t b = 0; b < batch_size; ++b) {
      auto begin = expected_keys.begin() + b * segment_size;
      std::stable_sort(begin, begin + segment_size,
                       [descending](KeyT a, KeyT b) {
                         return KeyLessThan(a, b, descending);
                       });
    }
    ExpectKeysBitEqual(h_keys_out, expected_keys);
  }

  template <typename KeyT, typename ValT>
  void RunSegmentedMultiPassPairs(PrimitiveType key_type,
                                  PrimitiveType val_type,
                                  const std::vector<KeyT>& h_keys_in,
                                  int32_t segment_size, int64_t batch_size,
                                  bool descending) {
    ASSERT_OK_AND_ASSIGN(
        CubSortSegmentedKernelSpecs specs,
        GetCubSortSegmentedKernelSpecs(key_type, val_type, descending, cc_));

    using SegFactory =
        se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                               se::DeviceAddressBase, se::DeviceAddressBase,
                               int32_t, int32_t, int32_t>;
    ASSERT_OK_AND_ASSIGN(auto primary_kernel,
                         SegFactory::Create(executor_, specs.primary));
    ASSERT_OK_AND_ASSIGN(auto alt_kernel,
                         SegFactory::Create(executor_, specs.alt));

    const CubRadixSortPolicySpec policy = GetCubRadixSortPolicySpec(
        sm_version_, sizeof(KeyT), sizeof(ValT), std::is_same_v<KeyT, float>);
    const int total_bits = static_cast<int>(sizeof(KeyT) * 8);
    const int primary_bits = policy.segmented.radix_bits;
    const int alt_bits = policy.alt_segmented.radix_bits;
    const int num_passes = (total_bits + primary_bits - 1) / primary_bits;
    const int max_alt_passes = num_passes * primary_bits - total_bits;
    const int alt_end_bit = std::min(total_bits, max_alt_passes * alt_bits);

    const int64_t total_items = batch_size * segment_size;
    ASSERT_EQ(h_keys_in.size(), total_items);
    std::vector<ValT> h_vals_in(total_items);
    for (int64_t i = 0; i < total_items; ++i) {
      h_vals_in[i] = static_cast<ValT>(i);
    }

    const int64_t key_bytes = total_items * sizeof(KeyT);
    const int64_t val_bytes = total_items * sizeof(ValT);
    se::DeviceAddressBase d_keys_in = executor_->Allocate(key_bytes, 0);
    se::DeviceAddressBase d_keys_out = executor_->Allocate(key_bytes, 0);
    se::DeviceAddressBase d_keys_tmp = executor_->Allocate(key_bytes, 0);
    se::DeviceAddressBase d_vals_in = executor_->Allocate(val_bytes, 0);
    se::DeviceAddressBase d_vals_out = executor_->Allocate(val_bytes, 0);
    se::DeviceAddressBase d_vals_tmp = executor_->Allocate(val_bytes, 0);
    absl::Cleanup free_bufs = [&] {
      executor_->Deallocate(&d_keys_in);
      executor_->Deallocate(&d_keys_out);
      executor_->Deallocate(&d_keys_tmp);
      executor_->Deallocate(&d_vals_in);
      executor_->Deallocate(&d_vals_out);
      executor_->Deallocate(&d_vals_tmp);
    };
    ASSERT_OK(stream_->Memcpy(&d_keys_in, h_keys_in.data(), key_bytes));
    ASSERT_OK(stream_->Memcpy(&d_vals_in, h_vals_in.data(), val_bytes));
    ASSERT_OK(stream_->MemZero(&d_keys_out, key_bytes));
    ASSERT_OK(stream_->MemZero(&d_vals_out, val_bytes));

    int current_bit = 0;
    for (int pass = 0; pass < num_passes; ++pass) {
      const bool use_alt = current_bit < alt_end_bit;
      const int radix_bits = use_alt ? alt_bits : primary_bits;
      const int pass_bits = std::min(radix_bits, total_bits - current_bit);
      const int block_threads = use_alt ? policy.alt_segmented.block_threads
                                        : policy.segmented.block_threads;
      auto& kernel = use_alt ? alt_kernel : primary_kernel;

      const int rem = num_passes - 1 - pass;
      se::DeviceAddressBase k_dst = (rem % 2 == 0) ? d_keys_out : d_keys_tmp;
      se::DeviceAddressBase k_src =
          (pass == 0) ? d_keys_in : ((rem % 2 == 0) ? d_keys_tmp : d_keys_out);
      se::DeviceAddressBase v_dst = (rem % 2 == 0) ? d_vals_out : d_vals_tmp;
      se::DeviceAddressBase v_src =
          (pass == 0) ? d_vals_in : ((rem % 2 == 0) ? d_vals_tmp : d_vals_out);

      // Exercise chunked grid launching with host-offset slices.
      const int64_t first_chunk = std::max<int64_t>(1, batch_size / 2);
      ASSERT_OK(kernel.Launch(se::ThreadDim(block_threads),
                              se::BlockDim(first_chunk), /*shmem_bytes=*/0,
                              stream_.get(), k_src, k_dst, v_src, v_dst,
                              segment_size, current_bit, pass_bits));
      if (batch_size > first_chunk) {
        const int64_t key_off = first_chunk * segment_size * sizeof(KeyT);
        const int64_t val_off = first_chunk * segment_size * sizeof(ValT);
        se::DeviceAddressBase k_src_2(
            static_cast<char*>(k_src.opaque()) + key_off, key_bytes - key_off);
        se::DeviceAddressBase k_dst_2(
            static_cast<char*>(k_dst.opaque()) + key_off, key_bytes - key_off);
        se::DeviceAddressBase v_src_2(
            static_cast<char*>(v_src.opaque()) + val_off, val_bytes - val_off);
        se::DeviceAddressBase v_dst_2(
            static_cast<char*>(v_dst.opaque()) + val_off, val_bytes - val_off);
        ASSERT_OK(kernel.Launch(se::ThreadDim(block_threads),
                                se::BlockDim(batch_size - first_chunk),
                                /*shmem_bytes=*/0, stream_.get(), k_src_2,
                                k_dst_2, v_src_2, v_dst_2, segment_size,
                                current_bit, pass_bits));
      }
      current_bit += pass_bits;
    }

    std::vector<KeyT> h_keys_out(total_items);
    std::vector<ValT> h_vals_out(total_items);
    std::vector<KeyT> h_keys_in_after(total_items);
    std::vector<ValT> h_vals_in_after(total_items);
    ASSERT_OK(stream_->Memcpy(h_keys_out.data(), d_keys_out, key_bytes));
    ASSERT_OK(stream_->Memcpy(h_vals_out.data(), d_vals_out, val_bytes));
    ASSERT_OK(stream_->Memcpy(h_keys_in_after.data(), d_keys_in, key_bytes));
    ASSERT_OK(stream_->Memcpy(h_vals_in_after.data(), d_vals_in, val_bytes));
    ASSERT_OK(stream_->BlockHostUntilDone());

    // Verify input buffers were not mutated across odd or even pass counts.
    ExpectKeysBitEqual(h_keys_in_after, h_keys_in);
    EXPECT_EQ(h_vals_in_after, h_vals_in);

    std::vector<std::pair<KeyT, ValT>> pairs(total_items);
    for (int64_t i = 0; i < total_items; ++i) {
      pairs[i] = {h_keys_in[i], h_vals_in[i]};
    }
    for (int64_t b = 0; b < batch_size; ++b) {
      auto begin = pairs.begin() + b * segment_size;
      std::stable_sort(begin, begin + segment_size,
                       [descending](const auto& a, const auto& b) {
                         return KeyLessThan(a.first, b.first, descending);
                       });
    }
    std::vector<KeyT> expected_keys(total_items);
    std::vector<ValT> expected_vals(total_items);
    for (int64_t i = 0; i < total_items; ++i) {
      expected_keys[i] = pairs[i].first;
      expected_vals[i] = pairs[i].second;
    }
    ExpectKeysBitEqual(h_keys_out, expected_keys);
    EXPECT_EQ(h_vals_out, expected_vals);
  }

  se::StreamExecutor* executor_ = nullptr;
  std::unique_ptr<se::Stream> stream_;
  se::CudaComputeCapability cc_;
  int sm_version_ = 800;
};

TEST_F(CubSortKernelsTest, SingleTileKeysAllTypes) {
  for (bool descending : {false, true}) {
    RunSingleTileKeys<bfloat16>(BF16, 257, 1, descending);
    RunSingleTileKeys<half>(F16, 257, 1, descending);
    RunSingleTileKeys<float>(F32, 1024, 4, descending, /*split_chunks=*/true);
    RunSingleTileKeys<double>(F64, 513, 2, descending);
    RunSingleTileKeys<int8_t>(S8, 300, 1, descending);
    RunSingleTileKeys<int16_t>(S16, 300, 1, descending);
    RunSingleTileKeys<int32_t>(S32, 4864, 1, descending);
    RunSingleTileKeys<int64_t>(S64, 2304, 1, descending);
    RunSingleTileKeys<uint8_t>(U8, 300, 3, descending);
    RunSingleTileKeys<uint16_t>(U16, 300, 3, descending);
    RunSingleTileKeys<uint32_t>(U32, 512, 2, descending);
    RunSingleTileKeys<uint64_t>(U64, 256, 2, descending);
  }
}

TEST_F(CubSortKernelsTest, SingleTilePairsAllTypes) {
  for (bool descending : {false, true}) {
    RunSingleTilePairs<uint8_t, uint16_t>(
        U8, U16, GenerateKeys<uint8_t>(300, 1), 300, 1, descending);
    RunSingleTilePairs<uint8_t, uint32_t>(
        U8, U32, GenerateKeys<uint8_t>(600, 2), 300, 2, descending);
    RunSingleTilePairs<uint8_t, uint64_t>(
        U8, U64, GenerateKeys<uint8_t>(300, 3), 300, 1, descending);

    RunSingleTilePairs<uint16_t, uint16_t>(
        U16, U16, GenerateKeys<uint16_t>(300, 4), 300, 1, descending);
    RunSingleTilePairs<uint16_t, uint32_t>(
        U16, U32, GenerateKeys<uint16_t>(600, 5), 300, 2, descending);
    RunSingleTilePairs<uint16_t, uint64_t>(
        U16, U64, GenerateKeys<uint16_t>(300, 6), 300, 1, descending);

    RunSingleTilePairs<int32_t, uint16_t>(
        S32, U16, GenerateKeys<int32_t>(300, 7), 300, 1, descending);
    RunSingleTilePairs<int32_t, uint32_t>(
        S32, U32, GenerateKeys<int32_t>(1024, 8), 512, 2, descending);
    RunSingleTilePairs<int32_t, uint64_t>(
        S32, U64, GenerateKeys<int32_t>(300, 9), 300, 1, descending);

    RunSingleTilePairs<uint32_t, uint16_t>(
        U32, U16, GenerateKeys<uint32_t>(300, 10), 300, 1, descending);
    RunSingleTilePairs<uint32_t, uint32_t>(
        U32, U32, GenerateKeys<uint32_t>(300, 11), 300, 1, descending);
    RunSingleTilePairs<uint32_t, uint64_t>(
        U32, U64, GenerateKeys<uint32_t>(300, 12), 300, 1, descending);

    RunSingleTilePairs<uint64_t, uint16_t>(
        U64, U16, GenerateKeys<uint64_t>(300, 13), 300, 1, descending);
    RunSingleTilePairs<uint64_t, uint32_t>(
        U64, U32, GenerateKeys<uint64_t>(300, 14), 300, 1, descending);
    RunSingleTilePairs<uint64_t, uint64_t>(
        U64, U64, GenerateKeys<uint64_t>(300, 15), 300, 1, descending);

    RunSingleTilePairs<float, uint16_t>(F32, U16, GenerateKeys<float>(300, 16),
                                        300, 1, descending);
    RunSingleTilePairs<float, uint32_t>(F32, U32, GenerateKeys<float>(1024, 17),
                                        512, 2, descending);
    RunSingleTilePairs<float, uint64_t>(F32, U64, GenerateKeys<float>(300, 18),
                                        300, 1, descending);
  }
}

TEST_F(CubSortKernelsTest, OnesweepKeysAndPairsBoundarySizes) {
  const CubRadixSortPolicySpec f32_policy =
      GetCubRadixSortPolicySpec(sm_version_, 4, 0, /*is_float32_key=*/true);
  const int32_t tile = f32_policy.onesweep.tile_items();
  for (int32_t n : {f32_policy.single_tile.tile_items() + 1, tile - 1, tile,
                    tile + 1, 2 * tile + 17}) {
    RunOnesweepKeys<float>(F32, GenerateKeys<float>(n, n),
                           /*descending=*/false);
    RunOnesweepKeys<float>(F32, GenerateKeys<float>(n, n + 1),
                           /*descending=*/true);
  }

  // Exercise all 12 keys-only and all 18 key-value pair Onesweep shards.
  for (bool descending : {false, true}) {
    RunOnesweepKeys<uint8_t>(U8, GenerateKeys<uint8_t>(10000, 101), descending);
    RunOnesweepKeys<int8_t>(S8, GenerateKeys<int8_t>(10000, 102), descending);
    RunOnesweepKeys<bfloat16>(BF16, GenerateKeys<bfloat16>(10000, 103),
                              descending);
    RunOnesweepKeys<half>(F16, GenerateKeys<half>(10000, 104), descending);
    RunOnesweepKeys<int16_t>(S16, GenerateKeys<int16_t>(10000, 107),
                             descending);
    RunOnesweepKeys<uint16_t>(U16, GenerateKeys<uint16_t>(10000, 108),
                              descending);
    RunOnesweepKeys<int32_t>(S32, GenerateKeys<int32_t>(16384, 105),
                             descending);
    RunOnesweepKeys<uint32_t>(U32, GenerateKeys<uint32_t>(16384, 109),
                              descending);
    RunOnesweepKeys<int64_t>(S64, GenerateKeys<int64_t>(10000, 110),
                             descending);
    RunOnesweepKeys<uint64_t>(U64, GenerateKeys<uint64_t>(10000, 111),
                              descending);
    RunOnesweepKeys<double>(F64, GenerateKeys<double>(10000, 106), descending);

    RunOnesweepPairs<uint8_t, uint16_t>(
        U8, U16, GenerateKeys<uint8_t>(10000, 200), descending);
    RunOnesweepPairs<uint8_t, uint32_t>(
        U8, U32, GenerateKeys<uint8_t>(10000, 201), descending);
    RunOnesweepPairs<uint8_t, uint64_t>(
        U8, U64, GenerateKeys<uint8_t>(10000, 206), descending);

    RunOnesweepPairs<uint16_t, uint16_t>(
        U16, U16, GenerateKeys<uint16_t>(10000, 207), descending);
    RunOnesweepPairs<uint16_t, uint32_t>(
        U16, U32, GenerateKeys<uint16_t>(10000, 208), descending);
    RunOnesweepPairs<uint16_t, uint64_t>(
        U16, U64, GenerateKeys<uint16_t>(10000, 202), descending);

    RunOnesweepPairs<int32_t, uint16_t>(
        S32, U16, GenerateKeys<int32_t>(10000, 209), descending);
    RunOnesweepPairs<int32_t, uint32_t>(
        S32, U32, GenerateKeys<int32_t>(16384, 203), descending);
    RunOnesweepPairs<int32_t, uint64_t>(
        S32, U64, GenerateKeys<int32_t>(10000, 210), descending);

    RunOnesweepPairs<uint32_t, uint16_t>(
        U32, U16, GenerateKeys<uint32_t>(10000, 211), descending);
    RunOnesweepPairs<uint32_t, uint32_t>(
        U32, U32, GenerateKeys<uint32_t>(10000, 212), descending);
    RunOnesweepPairs<uint32_t, uint64_t>(
        U32, U64, GenerateKeys<uint32_t>(10000, 213), descending);

    RunOnesweepPairs<uint64_t, uint16_t>(
        U64, U16, GenerateKeys<uint64_t>(10000, 214), descending);
    RunOnesweepPairs<uint64_t, uint32_t>(
        U64, U32, GenerateKeys<uint64_t>(10000, 215), descending);
    RunOnesweepPairs<uint64_t, uint64_t>(
        U64, U64, GenerateKeys<uint64_t>(10000, 205), descending);

    RunOnesweepPairs<float, uint16_t>(F32, U16, GenerateKeys<float>(10000, 216),
                                      descending);
    RunOnesweepPairs<float, uint32_t>(F32, U32, GenerateKeys<float>(16384, 204),
                                      descending);
    RunOnesweepPairs<float, uint64_t>(F32, U64, GenerateKeys<float>(10000, 217),
                                      descending);
  }
}

TEST_F(CubSortKernelsTest, OnesweepMultiPortionChaining) {
  // CUB's Onesweep requires every non-final portion size to be an exact
  // multiple of that configuration's Onesweep tile size (only the final
  // portion, where d_bins_out == nullptr, may contain a partial tail tile).
  const CubRadixSortPolicySpec s32_keys_policy =
      GetCubRadixSortPolicySpec(sm_version_, 4, 0, /*is_float32_key=*/false);
  const int32_t s32_portion = 2 * s32_keys_policy.onesweep.tile_items();
  const int32_t s32_num_items = 2 * s32_portion + 137;

  const CubRadixSortPolicySpec f32_pair_policy =
      GetCubRadixSortPolicySpec(sm_version_, 4, 4, /*is_float32_key=*/true);
  const int32_t f32_portion = 2 * f32_pair_policy.onesweep.tile_items();
  const int32_t f32_num_items = 2 * f32_portion + 137;

  for (bool descending : {false, true}) {
    RunOnesweepKeys<int32_t>(S32, GenerateKeys<int32_t>(s32_num_items, 301),
                             descending, /*portion_size_override=*/s32_portion);
    RunOnesweepPairs<float, uint32_t>(
        F32, U32, GenerateKeys<float>(f32_num_items, 302), descending,
        /*portion_size_override=*/f32_portion);
    // All-equal keys exercise AgentRadixSortOnesweep::ShortCircuitCopy +
    // UpdateBinsGlobal across portions.
    RunOnesweepKeys<int32_t>(S32, std::vector<int32_t>(s32_num_items, 42),
                             descending, /*portion_size_override=*/s32_portion);
    RunOnesweepPairs<float, uint32_t>(
        F32, U32, std::vector<float>(f32_num_items, -3.5f), descending,
        /*portion_size_override=*/f32_portion);
  }
}

TEST_F(CubSortKernelsTest, ExclusiveSumMultiIterationAnd64BitBins) {
  ASSERT_OK_AND_ASSIGN(CubSortOnesweepKernelSpecs specs,
                       GetCubSortOnesweepKernelSpecs(
                           U64, std::nullopt, /*descending=*/false, cc_));
  using ExclSumFactory =
      se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                             se::DeviceAddressBase, int32_t>;
  ASSERT_OK_AND_ASSIGN(auto excl_sum_kernel,
                       ExclSumFactory::Create(executor_, specs.exclusive_sum));

  constexpr int kNumPasses = 8;
  constexpr int32_t kNumPortions = 300;  // > 256 exercises loop stride.
  constexpr int64_t kTotalBins = kNumPasses * kCubSortOnesweepRadixDigits;
  constexpr int64_t kBinsBytes = kTotalBins * sizeof(CubGlobalOffset);
  constexpr int64_t kCtrsBytes = static_cast<int64_t>(kNumPasses) *
                                 kNumPortions * kCubSortScratchAlignmentBytes;

  std::vector<CubGlobalOffset> h_bins_in(kTotalBins);
  std::vector<CubGlobalOffset> expected_bins_out(kTotalBins);
  for (int pass = 0; pass < kNumPasses; ++pass) {
    CubGlobalOffset running = 0;
    for (int bin = 0; bin < kCubSortOnesweepRadixDigits; ++bin) {
      const int idx = pass * kCubSortOnesweepRadixDigits + bin;
      // Values exceed UINT32_MAX to verify 64-bit BlockScan accumulation.
      h_bins_in[idx] = (CubGlobalOffset{1} << 32) +
                       static_cast<CubGlobalOffset>(pass * 1000 + bin);
      expected_bins_out[idx] = running;
      running += h_bins_in[idx];
    }
  }

  se::DeviceAddressBase d_bins_in = executor_->Allocate(kBinsBytes, 0);
  se::DeviceAddressBase d_bins_out = executor_->Allocate(kBinsBytes, 0);
  se::DeviceAddressBase d_ctrs_out = executor_->Allocate(kCtrsBytes, 0);
  absl::Cleanup free_bufs = [&] {
    executor_->Deallocate(&d_bins_in);
    executor_->Deallocate(&d_bins_out);
    executor_->Deallocate(&d_ctrs_out);
  };
  ASSERT_OK(stream_->Memcpy(&d_bins_in, h_bins_in.data(), kBinsBytes));
  ASSERT_OK(stream_->MemZero(&d_bins_out, kBinsBytes));
  ASSERT_OK(stream_->Memset32(&d_ctrs_out, 0x5a5a5a5au, kCtrsBytes));

  ASSERT_OK(excl_sum_kernel.Launch(
      se::ThreadDim(kCubSortExclusiveSumBlockThreads), se::BlockDim(kNumPasses),
      /*shmem_bytes=*/0, stream_.get(), d_bins_in, d_bins_out, d_ctrs_out,
      kNumPortions));

  std::vector<CubGlobalOffset> h_bins_out(kTotalBins);
  std::vector<int32_t> h_ctrs_out(kCtrsBytes / sizeof(int32_t));
  ASSERT_OK(stream_->Memcpy(h_bins_out.data(), d_bins_out, kBinsBytes));
  ASSERT_OK(stream_->Memcpy(h_ctrs_out.data(), d_ctrs_out, kCtrsBytes));
  ASSERT_OK(stream_->BlockHostUntilDone());

  EXPECT_EQ(h_bins_out, expected_bins_out);
  constexpr int64_t kCtrStrideInt32 =
      kCubSortScratchAlignmentBytes / static_cast<int64_t>(sizeof(int32_t));
  for (int32_t p = 0; p < kNumPortions; ++p) {
    for (int pass = 0; pass < kNumPasses; ++pass) {
      const int64_t slot = static_cast<int64_t>(p) * kNumPasses + pass;
      ASSERT_EQ(h_ctrs_out[slot * kCtrStrideInt32], 0)
          << "portion=" << p << " pass=" << pass;
    }
  }
}

TEST_F(CubSortKernelsTest, FloatEdgeCasesAndNegativeZeroStability) {
  const float neg_nan = -std::numeric_limits<float>::quiet_NaN();
  const float pos_nan = std::numeric_limits<float>::quiet_NaN();
  const float max_oob_f32 = absl::bit_cast<float>(uint32_t{0x7FFFFFFFu});
  const float min_oob_f32 = absl::bit_cast<float>(uint32_t{0xFFFFFFFFu});
  const float neg_inf = -std::numeric_limits<float>::infinity();
  const float pos_inf = std::numeric_limits<float>::infinity();

  // Include exact MakeDefaultOobKey sentinels (0x7FFFFFFF / 0xFFFFFFFF) in a
  // partial tile to verify valid sentinel keys stay ahead of OOB lanes.
  std::vector<float> base = {
      pos_nan,     max_oob_f32, 3.0f,    -0.0f,   +0.0f, neg_inf, -1.5f, +0.0f,
      min_oob_f32, -0.0f,       pos_inf, neg_nan, -0.0f, 0.0f,    3.0f,  -1.5f};

  // SingleTile stability with -0.0 / +0.0 / NaNs / Infs / OOB sentinels.
  for (bool descending : {false, true}) {
    RunSingleTilePairs<float, uint32_t>(F32, U32, base, base.size(), 1,
                                        descending);
  }
  // Integer OOB sentinel ties in partial tiles.
  const std::vector<int32_t> s32_sentinels = {
      std::numeric_limits<int32_t>::max(),
      std::numeric_limits<int32_t>::min(),
      0,
      -1,
      1,
      std::numeric_limits<int32_t>::max(),
      std::numeric_limits<int32_t>::min()};
  const std::vector<uint32_t> u32_sentinels = {
      std::numeric_limits<uint32_t>::max(), 0, 42,
      std::numeric_limits<uint32_t>::max(), 0};
  for (bool descending : {false, true}) {
    RunSingleTilePairs<int32_t, uint32_t>(S32, U32, s32_sentinels,
                                          s32_sentinels.size(), 1, descending);
    RunSingleTilePairs<uint32_t, uint32_t>(U32, U32, u32_sentinels,
                                           u32_sentinels.size(), 1, descending);
  }

  // BF16, F16, and F64 edge cases (NaN sentinels, Inf, subnormals, +/-0.0).
  const std::vector<bfloat16> bf16_edge = {
      absl::bit_cast<bfloat16>(uint16_t{0x7FFFu}),
      absl::bit_cast<bfloat16>(uint16_t{0xFFFFu}),
      absl::bit_cast<bfloat16>(uint16_t{0x7F80u}),
      absl::bit_cast<bfloat16>(uint16_t{0xFF80u}),
      absl::bit_cast<bfloat16>(uint16_t{0x0001u}),
      absl::bit_cast<bfloat16>(uint16_t{0x8001u}),
      absl::bit_cast<bfloat16>(uint16_t{0x8000u}),
      absl::bit_cast<bfloat16>(uint16_t{0x0000u}),
      bfloat16(1.5f),
      bfloat16(-2.5f),
  };
  const std::vector<half> f16_edge = {
      absl::bit_cast<half>(uint16_t{0x7FFFu}),
      absl::bit_cast<half>(uint16_t{0xFFFFu}),
      absl::bit_cast<half>(uint16_t{0x7C00u}),
      absl::bit_cast<half>(uint16_t{0xFC00u}),
      absl::bit_cast<half>(uint16_t{0x0001u}),
      absl::bit_cast<half>(uint16_t{0x8001u}),
      absl::bit_cast<half>(uint16_t{0x8000u}),
      absl::bit_cast<half>(uint16_t{0x0000u}),
      half(1.5f),
      half(-2.5f),
  };
  const std::vector<double> f64_edge = {
      absl::bit_cast<double>(uint64_t{0x7FFFFFFFFFFFFFFFULL}),
      absl::bit_cast<double>(uint64_t{0xFFFFFFFFFFFFFFFFULL}),
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::denorm_min(),
      -std::numeric_limits<double>::denorm_min(),
      -0.0,
      +0.0,
      3.25,
      -7.5,
  };
  for (bool descending : {false, true}) {
    RunSingleTileKeysWithInput<bfloat16>(BF16, bf16_edge, bf16_edge.size(), 1,
                                         descending);
    RunSingleTileKeysWithInput<half>(F16, f16_edge, f16_edge.size(), 1,
                                     descending);
    RunSingleTileKeysWithInput<double>(F64, f64_edge, f64_edge.size(), 1,
                                       descending);
  }

  // Onesweep stability (repeat pattern past SingleTile threshold).
  std::vector<float> large;
  large.reserve(8192);
  while (large.size() < 8192) {
    for (float v : base) {
      if (large.size() < 8192) {
        large.push_back(v);
      }
    }
  }
  for (bool descending : {false, true}) {
    RunOnesweepPairs<float, uint32_t>(F32, U32, large, descending);
  }

  // SegmentedMultiPass stability (2 segments of 8000 floats each, exceeding
  // SegmentedPolicy::tile_items() = 7488 so primary passes span multiple
  // tiles).
  std::vector<float> segmented;
  segmented.reserve(16000);
  while (segmented.size() < 16000) {
    for (float v : base) {
      if (segmented.size() < 16000) {
        segmented.push_back(v);
      }
    }
  }
  for (bool descending : {false, true}) {
    RunSegmentedMultiPassPairs<float, uint32_t>(F32, U32, segmented, 8000, 2,
                                                descending);
  }
}

TEST_F(CubSortKernelsTest, SegmentedMultiPassKeysAllTypes) {
  for (bool descending : {false, true}) {
    RunSegmentedMultiPassKeys<bfloat16>(
        BF16, GenerateKeys<bfloat16>(3 * 8000, 411), 8000, 3, descending);
    RunSegmentedMultiPassKeys<half>(F16, GenerateKeys<half>(3 * 8000, 412),
                                    8000, 3, descending);
    RunSegmentedMultiPassKeys<float>(F32, GenerateKeys<float>(3 * 8000, 413),
                                     8000, 3, descending);
    RunSegmentedMultiPassKeys<double>(F64, GenerateKeys<double>(3 * 4000, 414),
                                      4000, 3, descending);
    RunSegmentedMultiPassKeys<int8_t>(S8, GenerateKeys<int8_t>(3 * 5200, 415),
                                      5200, 3, descending);
    RunSegmentedMultiPassKeys<int16_t>(
        S16, GenerateKeys<int16_t>(3 * 8000, 416), 8000, 3, descending);
    RunSegmentedMultiPassKeys<int32_t>(
        S32, GenerateKeys<int32_t>(3 * 8000, 417), 8000, 3, descending);
    RunSegmentedMultiPassKeys<int64_t>(
        S64, GenerateKeys<int64_t>(3 * 4000, 418), 4000, 3, descending);
    RunSegmentedMultiPassKeys<uint8_t>(U8, GenerateKeys<uint8_t>(3 * 5200, 419),
                                       5200, 3, descending);
    RunSegmentedMultiPassKeys<uint16_t>(
        U16, GenerateKeys<uint16_t>(3 * 8000, 420), 8000, 3, descending);
    RunSegmentedMultiPassKeys<uint32_t>(
        U32, GenerateKeys<uint32_t>(3 * 8000, 421), 8000, 3, descending);
    RunSegmentedMultiPassKeys<uint64_t>(
        U64, GenerateKeys<uint64_t>(3 * 4000, 422), 4000, 3, descending);
  }
}

TEST_F(CubSortKernelsTest, SegmentedMultiPassPairsAndTileBoundaries) {
  const CubRadixSortPolicySpec f32_u32_policy =
      GetCubRadixSortPolicySpec(sm_version_, 4, 4, /*is_float32_key=*/true);
  const int32_t tile_4b = f32_u32_policy.segmented.tile_items();
  const CubRadixSortPolicySpec f32_u64_policy =
      GetCubRadixSortPolicySpec(sm_version_, 4, 8, /*is_float32_key=*/true);
  const int32_t tile_8b = f32_u64_policy.segmented.tile_items();

  // Exercise exact full-tile and multi-tile boundaries (ProcessFullTile +
  // bin_offset carry-over) for both 4B and 8B dominant types.
  for (int32_t seg : {tile_4b - 1, tile_4b, tile_4b + 1, 2 * tile_4b + 17}) {
    RunSegmentedMultiPassKeys<int32_t>(
        S32, GenerateKeys<int32_t>(2 * seg, 430 + seg), seg, 2,
        /*descending=*/false);
    RunSegmentedMultiPassPairs<float, uint32_t>(
        F32, U32, GenerateKeys<float>(2 * seg, 440 + seg), seg, 2,
        /*descending=*/true);
  }
  for (int32_t seg : {tile_8b - 1, tile_8b, tile_8b + 1, 2 * tile_8b + 17}) {
    RunSegmentedMultiPassPairs<float, uint64_t>(
        F32, U64, GenerateKeys<float>(2 * seg, 450 + seg), seg, 2,
        /*descending=*/false);
  }

  // Exercise AgentRadixSortUpsweep's UNROLLED_ELEMENTS counter-flush loop
  // (6 * 7488 = 44,928 on SM60..SM100) and AgentRadixSortDownsweep::Copy
  // short-circuit path (all-equal keys).
  for (bool descending : {false, true}) {
    RunSegmentedMultiPassPairs<float, uint32_t>(
        F32, U32, GenerateKeys<float>(2 * 50000, 460), 50000, 2, descending);
    RunSegmentedMultiPassKeys<int32_t>(S32, std::vector<int32_t>(2 * 8000, 77),
                                       8000, 2, descending);
    RunSegmentedMultiPassPairs<float, uint32_t>(
        F32, U32, std::vector<float>(2 * 8000, 1.25f), 8000, 2, descending);

    RunSegmentedMultiPassPairs<uint8_t, uint32_t>(
        U8, U32, GenerateKeys<uint8_t>(5 * 8000, 401), 8000, 5, descending);
    RunSegmentedMultiPassPairs<uint16_t, uint16_t>(
        U16, U16, GenerateKeys<uint16_t>(4 * 8000, 402), 8000, 4, descending);
    RunSegmentedMultiPassPairs<int32_t, uint32_t>(
        S32, U32, GenerateKeys<int32_t>(3 * 8000, 403), 8000, 3, descending);
    RunSegmentedMultiPassPairs<uint32_t, uint32_t>(
        U32, U32, GenerateKeys<uint32_t>(3 * 8000, 406), 8000, 3, descending);
    RunSegmentedMultiPassPairs<float, uint64_t>(
        F32, U64, GenerateKeys<float>(3 * 4000, 404), 4000, 3, descending);
    RunSegmentedMultiPassPairs<uint64_t, uint32_t>(
        U64, U32, GenerateKeys<uint64_t>(3 * 4000, 405), 4000, 3, descending);
  }
}

TEST_F(CubSortKernelsTest, SingleTileBoundarySizesAndValueDispatch) {
  // Boundary segment sizes: 0 (no OOB access), 1, and full asymmetric 8B-value
  // tile (U8 keys + U64 values -> DominantType is 8 bytes -> tile_items 2304).
  for (bool descending : {false, true}) {
    RunSingleTileKeys<uint32_t>(U32, /*segment_size=*/0, /*batch_size=*/2,
                                descending);
    RunSingleTileKeys<uint32_t>(U32, /*segment_size=*/1, /*batch_size=*/3,
                                descending);
    RunSingleTilePairs<uint32_t, uint32_t>(U32, U32, {}, /*segment_size=*/0,
                                           /*batch_size=*/2, descending);
    RunSingleTilePairs<uint32_t, uint32_t>(U32, U32, {7, 3}, /*segment_size=*/1,
                                           /*batch_size=*/2, descending);
    RunSingleTilePairs<uint8_t, uint64_t>(
        U8, U64, GenerateKeys<uint8_t>(2304, 77), /*segment_size=*/2304,
        /*batch_size=*/1, descending);
  }

  // Non-unsigned value types exercising 16/32/64-bit value-width dispatch.
  RunSingleTilePairs<float, float>(F32, F32, GenerateKeys<float>(128, 19), 128,
                                   1, /*descending=*/false);
  RunSingleTilePairs<uint32_t, bfloat16>(
      U32, BF16, GenerateKeys<uint32_t>(128, 20), 128, 1, /*descending=*/true);
  RunSingleTilePairs<uint32_t, int64_t>(
      U32, S64, GenerateKeys<uint32_t>(128, 21), 128, 1, /*descending=*/false);
}

TEST_F(CubSortKernelsTest, SingleArchCubinExtractionAndUnsupportedArchError) {
  ASSERT_OK_AND_ASSIGN(se::KernelLoaderSpec spec,
                       GetCubSortSingleTileKernelSpec(
                           F32, std::nullopt, /*descending=*/false, cc_));
  ASSERT_TRUE(spec.cuda_cubin_in_memory().has_value());
  // Native Forge GPU targets (sm_80/90/100) embed an uncompressed single-arch
  // CUBIN ELF rather than falling back to the full multi-arch fatbinary.
  if (cc_.major <= 10) {
    EXPECT_TRUE(se::cuda::IsCudaElf(spec.cuda_cubin_in_memory()->cubin_bytes));
  }

  // An unsupported old arch with neither CUBIN nor compatible PTX must fail
  // rather than silently returning an un-runnable fatbinary.
  EXPECT_THAT(
      GetCubSortSingleTileKernelSpec(F32, std::nullopt, /*descending=*/false,
                                     se::CudaComputeCapability(3, 0)),
      StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      GetCubSortOnesweepKernelSpecs(F32, std::nullopt, /*descending=*/false,
                                    se::CudaComputeCapability(3, 0)),
      StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      GetCubSortSegmentedKernelSpecs(F32, std::nullopt, /*descending=*/false,
                                     se::CudaComputeCapability(3, 0)),
      StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(CubSortKernelsTest, UnsupportedTypesReturnInvalidArgument) {
  EXPECT_THAT(GetCubSortSingleTileKernelSpec(PRED, std::nullopt, false, cc_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported key type")));
  EXPECT_THAT(GetCubSortSingleTileKernelSpec(F16, U32, false, cc_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported key type")));
  EXPECT_THAT(GetCubSortSingleTileKernelSpec(U32, TUPLE, false, cc_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported value type")));
  EXPECT_THAT(GetCubSortSingleTileKernelSpec(U32, U8, false, cc_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported value bit width")));
  EXPECT_THAT(GetCubSortOnesweepKernelSpecs(PRED, std::nullopt, false, cc_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported key type")));
  EXPECT_THAT(GetCubSortOnesweepKernelSpecs(F16, U32, false, cc_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported key type")));
  EXPECT_THAT(GetCubSortOnesweepKernelSpecs(U32, TUPLE, false, cc_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported value type")));
  EXPECT_THAT(GetCubSortOnesweepKernelSpecs(U32, U8, false, cc_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported value bit width")));
  EXPECT_THAT(GetCubSortSegmentedKernelSpecs(PRED, std::nullopt, false, cc_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported key type")));
  EXPECT_THAT(GetCubSortSegmentedKernelSpecs(F16, U32, false, cc_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported key type")));
  EXPECT_THAT(GetCubSortSegmentedKernelSpecs(U32, TUPLE, false, cc_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported value type")));
  EXPECT_THAT(GetCubSortSegmentedKernelSpecs(U32, U8, false, cc_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported value bit width")));
}

}  // namespace
}  // namespace xla::gpu
