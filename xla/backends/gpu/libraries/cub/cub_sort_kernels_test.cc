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
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_policy.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_types.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_elf_utils.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;
using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

// Reference comparison matching CUB's radix sort ordering:
// - Integers: numeric `<` / `>`.
// - Floats: total-order-like except -0.0 and +0.0 compare equal (preserving
//   input order stably via std::stable_sort).
template <typename Key>
bool KeyLessThan(Key a, Key b, bool descending) {
  if constexpr (std::is_integral_v<Key>) {
    return descending ? (a > b) : (a < b);
  } else {
    auto to_ordered = [](Key v) -> uint64_t {
      if constexpr (sizeof(Key) == 2) {
        uint16_t bits = absl::bit_cast<uint16_t>(v);
        if ((bits & 0x7FFFu) == 0) {
          bits = 0;  // Equate -0.0 and +0.0.
        }
        uint16_t mask =
            (bits & 0x8000u) ? uint16_t{0xFFFFu} : uint16_t{0x8000u};
        return static_cast<uint64_t>(bits ^ mask);
      } else if constexpr (sizeof(Key) == 4) {
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

template <typename Key>
std::vector<Key> GenerateKeys(int64_t count, uint32_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<Key> keys(count);
  if constexpr (std::is_integral_v<Key>) {
    std::uniform_int_distribution<uint64_t> dist(0, 1000000);
    for (int64_t i = 0; i < count; ++i) {
      // Mix wide range and duplicates to exercise tie-breaking and stability.
      uint64_t raw = (i % 7 == 0) ? (dist(rng) % 16) : rng();
      keys[i] = static_cast<Key>(raw);
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
      keys[i] = static_cast<Key>(val);
    }
  }
  return keys;
}

template <typename Key>
void ExpectKeysBitEqual(const std::vector<Key>& actual,
                        const std::vector<Key>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    uint64_t a = 0;
    uint64_t e = 0;
    if constexpr (sizeof(Key) == 1) {
      a = absl::bit_cast<uint8_t>(actual[i]);
      e = absl::bit_cast<uint8_t>(expected[i]);
    } else if constexpr (sizeof(Key) == 2) {
      a = absl::bit_cast<uint16_t>(actual[i]);
      e = absl::bit_cast<uint16_t>(expected[i]);
    } else if constexpr (sizeof(Key) == 4) {
      a = absl::bit_cast<uint32_t>(actual[i]);
      e = absl::bit_cast<uint32_t>(expected[i]);
    } else {
      a = absl::bit_cast<uint64_t>(actual[i]);
      e = absl::bit_cast<uint64_t>(expected[i]);
    }
    if (a != e) {
      ADD_FAILURE() << "Mismatch at index " << i << ": actual=0x" << std::hex
                    << a << " expected=0x" << e << std::dec;
      return;
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
    allocator_ =
        std::make_unique<se::StreamExecutorAddressAllocator>(executor_);
    cc_ = executor_->GetDeviceDescription().cuda_compute_capability();
    sm_version_ = CanonicalCubSortPolicySmVersion(cc_.major);
  }

  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> Allocate(int64_t bytes) {
    return allocator_->Allocate(executor_->device_ordinal(), bytes);
  }

  template <typename Key>
  void RunSingleTileKeysWithInput(PrimitiveType key_type,
                                  const std::vector<Key>& h_in,
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
        sm_version_, sizeof(Key), /*value_byte_size=*/0,
        std::is_same_v<Key, float>);
    ASSERT_LE(segment_size, policy.single_tile.tile_items());

    const int64_t total_items = batch_size * segment_size;
    ASSERT_EQ(h_in.size(), total_items);
    std::vector<Key> h_expected = h_in;
    for (int64_t b = 0; b < batch_size; ++b) {
      auto begin = h_expected.begin() + b * segment_size;
      std::stable_sort(begin, begin + segment_size, [descending](Key a, Key b) {
        return KeyLessThan(a, b, descending);
      });
    }

    const int64_t byte_size = total_items * sizeof(Key);
    const int64_t alloc_bytes = std::max<int64_t>(byte_size, sizeof(Key));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> d_in,
                         Allocate(alloc_bytes));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> d_out,
                         Allocate(alloc_bytes));
    if (byte_size > 0) {
      ASSERT_OK(stream_->Memcpy(d_in.ptr(), h_in.data(), byte_size));
    }
    ASSERT_OK(stream_->MemZero(d_out.ptr(), alloc_bytes));

    // Pass explicit 0 dynamic shared memory: CUB kernels use static __shared__
    // storage only, and TypedKernel::Launch without an explicit shmem_bytes
    // argument forwards static shared memory metadata as dynamic shared memory
    // (doubling shared memory usage and exceeding T4's 64 KiB opt-in limit).
    if (split_chunks && batch_size >= 2) {
      const int64_t first = batch_size / 2;
      const int64_t first_bytes = first * segment_size * sizeof(Key);
      se::DeviceAddressBase d_in_second(
          static_cast<char*>(d_in->opaque()) + first_bytes,
          byte_size - first_bytes);
      se::DeviceAddressBase d_out_second(
          static_cast<char*>(d_out->opaque()) + first_bytes,
          byte_size - first_bytes);
      ASSERT_OK(kernel.Launch(se::ThreadDim(policy.single_tile.block_threads),
                              se::BlockDim(first), /*shmem_bytes=*/0,
                              stream_.get(), *d_in, *d_out, segment_size));
      ASSERT_OK(kernel.Launch(se::ThreadDim(policy.single_tile.block_threads),
                              se::BlockDim(batch_size - first),
                              /*shmem_bytes=*/0, stream_.get(), d_in_second,
                              d_out_second, segment_size));
    } else {
      ASSERT_OK(kernel.Launch(se::ThreadDim(policy.single_tile.block_threads),
                              se::BlockDim(batch_size), /*shmem_bytes=*/0,
                              stream_.get(), *d_in, *d_out, segment_size));
    }

    std::vector<Key> h_out(total_items);
    if (byte_size > 0) {
      ASSERT_OK(stream_->Memcpy(h_out.data(), *d_out, byte_size));
    }
    ASSERT_OK(stream_->BlockHostUntilDone());

    ExpectKeysBitEqual(h_out, h_expected);
  }

  template <typename Key>
  void RunSingleTileKeys(PrimitiveType key_type, int32_t segment_size,
                         int64_t batch_size, bool descending,
                         bool split_chunks = false) {
    RunSingleTileKeysWithInput<Key>(
        key_type, GenerateKeys<Key>(batch_size * segment_size, 12345),
        segment_size, batch_size, descending, split_chunks);
  }

  template <typename Key, typename Value>
  void RunSingleTilePairs(PrimitiveType key_type, PrimitiveType val_type,
                          const std::vector<Key>& h_keys_in,
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
        sm_version_, sizeof(Key), sizeof(Value), std::is_same_v<Key, float>);
    ASSERT_LE(segment_size, policy.single_tile.tile_items());

    const int64_t total_items = batch_size * segment_size;
    ASSERT_EQ(h_keys_in.size(), total_items);
    std::vector<Value> h_vals_in(total_items);
    for (int64_t i = 0; i < total_items; ++i) {
      h_vals_in[i] = static_cast<Value>(i);
    }

    std::vector<std::pair<Key, Value>> pairs(total_items);
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
    std::vector<Key> expected_keys(total_items);
    std::vector<Value> expected_vals(total_items);
    for (int64_t i = 0; i < total_items; ++i) {
      expected_keys[i] = pairs[i].first;
      expected_vals[i] = pairs[i].second;
    }

    const int64_t key_bytes = total_items * sizeof(Key);
    const int64_t val_bytes = total_items * sizeof(Value);
    const int64_t key_alloc = std::max<int64_t>(key_bytes, sizeof(Key));
    const int64_t val_alloc = std::max<int64_t>(val_bytes, sizeof(Value));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> d_keys_in,
                         Allocate(key_alloc));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> d_keys_out,
                         Allocate(key_alloc));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> d_vals_in,
                         Allocate(val_alloc));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> d_vals_out,
                         Allocate(val_alloc));
    if (key_bytes > 0) {
      ASSERT_OK(stream_->Memcpy(d_keys_in.ptr(), h_keys_in.data(), key_bytes));
      ASSERT_OK(stream_->Memcpy(d_vals_in.ptr(), h_vals_in.data(), val_bytes));
    }

    ASSERT_OK(kernel.Launch(se::ThreadDim(policy.single_tile.block_threads),
                            se::BlockDim(batch_size), /*shmem_bytes=*/0,
                            stream_.get(), *d_keys_in, *d_keys_out, *d_vals_in,
                            *d_vals_out, segment_size));

    std::vector<Key> h_keys_out(total_items);
    std::vector<Value> h_vals_out(total_items);
    if (key_bytes > 0) {
      ASSERT_OK(stream_->Memcpy(h_keys_out.data(), *d_keys_out, key_bytes));
      ASSERT_OK(stream_->Memcpy(h_vals_out.data(), *d_vals_out, val_bytes));
    }
    ASSERT_OK(stream_->BlockHostUntilDone());

    ExpectKeysBitEqual(h_keys_out, expected_keys);
    EXPECT_EQ(h_vals_out, expected_vals);
  }

  template <typename Key>
  void RunOnesweepKeys(PrimitiveType key_type, const std::vector<Key>& keys_in,
                       bool descending, int32_t portion_size_override = 0) {
    ASSERT_OK_AND_ASSIGN(
        CubSortOnesweepKernelSpecs specs,
        GetCubSortOnesweepKernelSpecs(key_type, std::nullopt, descending, cc_));
    const std::optional<se::CudaCubinInMemory> cubin =
        specs.histogram.cuda_cubin_in_memory();
    ASSERT_TRUE(cubin.has_value());
    if (se::cuda::IsCudaElf(cubin->cubin_bytes)) {
      ASSERT_TRUE(specs.histogram_attrs.has_value());
      EXPECT_GT(specs.histogram_attrs->num_regs, 0);
      EXPECT_GT(specs.histogram_attrs->static_shared_size_bytes, 0);
    }

    using HistogramFactory =
        se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                               CubGlobalOffset, int32_t, int32_t>;
    using ExclusiveSumFactory =
        se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                               se::DeviceAddressBase, int32_t>;
    using OnesweepFactory =
        se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                               se::DeviceAddressBase, se::DeviceAddressBase,
                               se::DeviceAddressBase, se::DeviceAddressBase,
                               int64_t, int32_t, int32_t, int32_t>;

    ASSERT_OK_AND_ASSIGN(auto histogram_kernel,
                         HistogramFactory::Create(executor_, specs.histogram));
    ASSERT_OK_AND_ASSIGN(
        auto exclusive_sum_kernel,
        ExclusiveSumFactory::Create(executor_, specs.exclusive_sum));
    ASSERT_OK_AND_ASSIGN(auto onesweep_kernel,
                         OnesweepFactory::Create(executor_, specs.onesweep));

    const CubRadixSortPolicySpec policy = GetCubRadixSortPolicySpec(
        sm_version_, sizeof(Key), /*value_byte_size=*/0,
        std::is_same_v<Key, float>);
    const int64_t num_items = keys_in.size();
    const int num_passes = static_cast<int>(sizeof(Key));
    const int32_t tile_items = policy.onesweep.tile_items();
    const int32_t portion_size =
        portion_size_override > 0
            ? portion_size_override
            : (kCubSortMaxPortionLimit / tile_items) * tile_items;
    const int64_t num_portions = (num_items + portion_size - 1) / portion_size;
    const int32_t max_blocks =
        (std::min<int64_t>(num_items, portion_size) + tile_items - 1) /
        tile_items;

    const int64_t key_bytes = num_items * sizeof(Key);
    const int64_t bins_pass_bytes =
        num_passes * kCubSortOnesweepRadixDigits * sizeof(CubGlobalOffset);
    const int64_t lookback_bytes = static_cast<int64_t>(max_blocks) *
                                   kCubSortOnesweepRadixDigits *
                                   sizeof(int32_t);
    const int64_t counters_all_bytes = static_cast<int64_t>(num_passes) *
                                       num_portions *
                                       kCubSortScratchAlignmentBytes;

    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> keys_in_buffer,
                         Allocate(key_bytes));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> keys_out_buffer,
                         Allocate(key_bytes));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> keys_tmp_buffer,
                         Allocate(key_bytes));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> bins_raw_buffer,
                         Allocate(bins_pass_bytes));
    std::vector<se::ScopedDeviceAddress<uint8_t>> bins_portion_buffers;
    bins_portion_buffers.reserve(num_portions);
    for (int64_t p = 0; p < num_portions; ++p) {
      ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> buf,
                           Allocate(bins_pass_bytes));
      bins_portion_buffers.push_back(std::move(buf));
    }
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> lookback_buffer,
                         Allocate(lookback_bytes));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> counters_buffer,
                         Allocate(counters_all_bytes));

    ASSERT_OK(stream_->Memcpy(keys_in_buffer.ptr(), keys_in.data(), key_bytes));
    ASSERT_OK(stream_->MemZero(bins_raw_buffer.ptr(), bins_pass_bytes));
    // Poison counters_buffer with non-zero bytes to verify ExclusiveSum zeroes
    // all counter slots.
    ASSERT_OK(stream_->Memset32(counters_buffer.ptr(), 0x5a5a5a5au,
                                counters_all_bytes));

    const int64_t histogram_grid =
        std::min<int64_t>(executor_->GetDeviceDescription().core_count() * 2,
                          (num_items + policy.histogram.tile_items() - 1) /
                              policy.histogram.tile_items());
    ASSERT_OK(histogram_kernel.Launch(
        se::ThreadDim(policy.histogram.block_threads),
        se::BlockDim(histogram_grid), /*shmem_bytes=*/0, stream_.get(),
        *bins_raw_buffer, *keys_in_buffer,
        static_cast<CubGlobalOffset>(num_items), int32_t{0},
        static_cast<int32_t>(sizeof(Key) * 8)));
    ASSERT_OK(exclusive_sum_kernel.Launch(
        se::ThreadDim(policy.exclusive_sum.block_threads),
        se::BlockDim(num_passes), /*shmem_bytes=*/0, stream_.get(),
        *bins_raw_buffer, *bins_portion_buffers[0], *counters_buffer,
        static_cast<int32_t>(num_portions)));

    for (int pass = 0; pass < num_passes; ++pass) {
      const int remaining_passes = num_passes - 1 - pass;
      se::DeviceAddressBase dst =
          (remaining_passes % 2 == 0) ? *keys_out_buffer : *keys_tmp_buffer;
      se::DeviceAddressBase src =
          (pass == 0) ? *keys_in_buffer
                      : ((remaining_passes % 2 == 0) ? *keys_tmp_buffer
                                                     : *keys_out_buffer);
      const int64_t pass_bin_byte_offset = static_cast<int64_t>(pass) *
                                           kCubSortOnesweepRadixDigits *
                                           sizeof(CubGlobalOffset);

      for (int64_t portion = 0; portion < num_portions; ++portion) {
        const int64_t item_offset = portion * portion_size;
        const int32_t portion_items = static_cast<int32_t>(
            std::min<int64_t>(num_items - item_offset, portion_size));
        const int32_t num_blocks =
            (portion_items + tile_items - 1) / tile_items;

        ASSERT_OK(stream_->MemZero(lookback_buffer.ptr(), lookback_bytes));

        const int64_t counter_slot = portion * num_passes + pass;
        se::DeviceAddressBase counter_slice(
            static_cast<char*>(counters_buffer->opaque()) +
                counter_slot * kCubSortScratchAlignmentBytes,
            sizeof(int32_t));
        se::DeviceAddressBase bins_in_slice(
            static_cast<char*>(bins_portion_buffers[portion]->opaque()) +
                pass_bin_byte_offset,
            kCubSortOnesweepRadixDigits * sizeof(CubGlobalOffset));
        se::DeviceAddressBase bins_out_slice =
            (portion + 1 < num_portions)
                ? se::DeviceAddressBase(
                      static_cast<char*>(
                          bins_portion_buffers[portion + 1]->opaque()) +
                          pass_bin_byte_offset,
                      kCubSortOnesweepRadixDigits * sizeof(CubGlobalOffset))
                : se::DeviceAddressBase(nullptr, 0);

        ASSERT_OK(onesweep_kernel.Launch(
            se::ThreadDim(policy.onesweep.block_threads),
            se::BlockDim(num_blocks), /*shmem_bytes=*/0, stream_.get(),
            *lookback_buffer, counter_slice, bins_out_slice, bins_in_slice, dst,
            src, item_offset, portion_items, pass * kCubSortOnesweepRadixBits,
            kCubSortOnesweepRadixBits));
      }
    }

    std::vector<Key> keys_out(num_items);
    ASSERT_OK(stream_->Memcpy(keys_out.data(), *keys_out_buffer, key_bytes));
    ASSERT_OK(stream_->BlockHostUntilDone());

    std::vector<Key> expected = keys_in;
    absl::c_stable_sort(expected, [descending](Key a, Key b) {
      return KeyLessThan(a, b, descending);
    });
    ExpectKeysBitEqual(keys_out, expected);
  }

  template <typename Key, typename Value>
  void RunOnesweepPairs(PrimitiveType key_type, PrimitiveType val_type,
                        const std::vector<Key>& keys_in, bool descending,
                        int32_t portion_size_override = 0) {
    ASSERT_OK_AND_ASSIGN(
        CubSortOnesweepKernelSpecs specs,
        GetCubSortOnesweepKernelSpecs(key_type, val_type, descending, cc_));
    const std::optional<se::CudaCubinInMemory> cubin =
        specs.histogram.cuda_cubin_in_memory();
    ASSERT_TRUE(cubin.has_value());
    if (se::cuda::IsCudaElf(cubin->cubin_bytes)) {
      ASSERT_TRUE(specs.histogram_attrs.has_value());
      EXPECT_GT(specs.histogram_attrs->num_regs, 0);
      EXPECT_GT(specs.histogram_attrs->static_shared_size_bytes, 0);
    }

    using HistogramFactory =
        se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                               CubGlobalOffset, int32_t, int32_t>;
    using ExclusiveSumFactory =
        se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                               se::DeviceAddressBase, int32_t>;
    using OnesweepFactory =
        se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                               se::DeviceAddressBase, se::DeviceAddressBase,
                               se::DeviceAddressBase, se::DeviceAddressBase,
                               se::DeviceAddressBase, se::DeviceAddressBase,
                               int64_t, int32_t, int32_t, int32_t>;

    ASSERT_OK_AND_ASSIGN(auto histogram_kernel,
                         HistogramFactory::Create(executor_, specs.histogram));
    ASSERT_OK_AND_ASSIGN(
        auto exclusive_sum_kernel,
        ExclusiveSumFactory::Create(executor_, specs.exclusive_sum));
    ASSERT_OK_AND_ASSIGN(auto onesweep_kernel,
                         OnesweepFactory::Create(executor_, specs.onesweep));

    const CubRadixSortPolicySpec policy = GetCubRadixSortPolicySpec(
        sm_version_, sizeof(Key), sizeof(Value), std::is_same_v<Key, float>);
    const int64_t num_items = keys_in.size();
    std::vector<Value> values_in(num_items);
    for (int64_t i = 0; i < num_items; ++i) {
      values_in[i] = static_cast<Value>(i);
    }

    const int num_passes = static_cast<int>(sizeof(Key));
    const int32_t tile_items = policy.onesweep.tile_items();
    const int32_t portion_size =
        portion_size_override > 0
            ? portion_size_override
            : (kCubSortMaxPortionLimit / tile_items) * tile_items;
    const int64_t num_portions = (num_items + portion_size - 1) / portion_size;
    const int32_t max_blocks =
        (std::min<int64_t>(num_items, portion_size) + tile_items - 1) /
        tile_items;

    const int64_t key_bytes = num_items * sizeof(Key);
    const int64_t val_bytes = num_items * sizeof(Value);
    const int64_t bins_pass_bytes =
        num_passes * kCubSortOnesweepRadixDigits * sizeof(CubGlobalOffset);
    const int64_t lookback_bytes = static_cast<int64_t>(max_blocks) *
                                   kCubSortOnesweepRadixDigits *
                                   sizeof(int32_t);
    const int64_t counters_all_bytes = static_cast<int64_t>(num_passes) *
                                       num_portions *
                                       kCubSortScratchAlignmentBytes;

    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> keys_in_buffer,
                         Allocate(key_bytes));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> keys_out_buffer,
                         Allocate(key_bytes));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> keys_tmp_buffer,
                         Allocate(key_bytes));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> values_in_buffer,
                         Allocate(val_bytes));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> values_out_buffer,
                         Allocate(val_bytes));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> values_tmp_buffer,
                         Allocate(val_bytes));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> bins_raw_buffer,
                         Allocate(bins_pass_bytes));
    std::vector<se::ScopedDeviceAddress<uint8_t>> bins_portion_buffers;
    bins_portion_buffers.reserve(num_portions);
    for (int64_t p = 0; p < num_portions; ++p) {
      ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> buf,
                           Allocate(bins_pass_bytes));
      bins_portion_buffers.push_back(std::move(buf));
    }
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> lookback_buffer,
                         Allocate(lookback_bytes));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> counters_buffer,
                         Allocate(counters_all_bytes));

    ASSERT_OK(stream_->Memcpy(keys_in_buffer.ptr(), keys_in.data(), key_bytes));
    ASSERT_OK(
        stream_->Memcpy(values_in_buffer.ptr(), values_in.data(), val_bytes));
    ASSERT_OK(stream_->MemZero(bins_raw_buffer.ptr(), bins_pass_bytes));
    ASSERT_OK(stream_->Memset32(counters_buffer.ptr(), 0x5a5a5a5au,
                                counters_all_bytes));

    const int64_t histogram_grid =
        std::min<int64_t>(executor_->GetDeviceDescription().core_count() * 2,
                          (num_items + policy.histogram.tile_items() - 1) /
                              policy.histogram.tile_items());
    ASSERT_OK(histogram_kernel.Launch(
        se::ThreadDim(policy.histogram.block_threads),
        se::BlockDim(histogram_grid), /*shmem_bytes=*/0, stream_.get(),
        *bins_raw_buffer, *keys_in_buffer,
        static_cast<CubGlobalOffset>(num_items), int32_t{0},
        static_cast<int32_t>(sizeof(Key) * 8)));
    ASSERT_OK(exclusive_sum_kernel.Launch(
        se::ThreadDim(policy.exclusive_sum.block_threads),
        se::BlockDim(num_passes), /*shmem_bytes=*/0, stream_.get(),
        *bins_raw_buffer, *bins_portion_buffers[0], *counters_buffer,
        static_cast<int32_t>(num_portions)));

    for (int pass = 0; pass < num_passes; ++pass) {
      const int remaining_passes = num_passes - 1 - pass;
      se::DeviceAddressBase keys_dst =
          (remaining_passes % 2 == 0) ? *keys_out_buffer : *keys_tmp_buffer;
      se::DeviceAddressBase keys_src =
          (pass == 0) ? *keys_in_buffer
                      : ((remaining_passes % 2 == 0) ? *keys_tmp_buffer
                                                     : *keys_out_buffer);
      se::DeviceAddressBase values_dst =
          (remaining_passes % 2 == 0) ? *values_out_buffer : *values_tmp_buffer;
      se::DeviceAddressBase values_src =
          (pass == 0) ? *values_in_buffer
                      : ((remaining_passes % 2 == 0) ? *values_tmp_buffer
                                                     : *values_out_buffer);
      const int64_t pass_bin_byte_offset = static_cast<int64_t>(pass) *
                                           kCubSortOnesweepRadixDigits *
                                           sizeof(CubGlobalOffset);

      for (int64_t portion = 0; portion < num_portions; ++portion) {
        const int64_t item_offset = portion * portion_size;
        const int32_t portion_items = static_cast<int32_t>(
            std::min<int64_t>(num_items - item_offset, portion_size));
        const int32_t num_blocks =
            (portion_items + tile_items - 1) / tile_items;

        ASSERT_OK(stream_->MemZero(lookback_buffer.ptr(), lookback_bytes));

        const int64_t counter_slot = portion * num_passes + pass;
        se::DeviceAddressBase counter_slice(
            static_cast<char*>(counters_buffer->opaque()) +
                counter_slot * kCubSortScratchAlignmentBytes,
            sizeof(int32_t));
        se::DeviceAddressBase bins_in_slice(
            static_cast<char*>(bins_portion_buffers[portion]->opaque()) +
                pass_bin_byte_offset,
            kCubSortOnesweepRadixDigits * sizeof(CubGlobalOffset));
        se::DeviceAddressBase bins_out_slice =
            (portion + 1 < num_portions)
                ? se::DeviceAddressBase(
                      static_cast<char*>(
                          bins_portion_buffers[portion + 1]->opaque()) +
                          pass_bin_byte_offset,
                      kCubSortOnesweepRadixDigits * sizeof(CubGlobalOffset))
                : se::DeviceAddressBase(nullptr, 0);

        ASSERT_OK(onesweep_kernel.Launch(
            se::ThreadDim(policy.onesweep.block_threads),
            se::BlockDim(num_blocks), /*shmem_bytes=*/0, stream_.get(),
            *lookback_buffer, counter_slice, bins_out_slice, bins_in_slice,
            keys_dst, keys_src, values_dst, values_src, item_offset,
            portion_items, pass * kCubSortOnesweepRadixBits,
            kCubSortOnesweepRadixBits));
      }
    }

    std::vector<Key> keys_out(num_items);
    std::vector<Value> values_out(num_items);
    ASSERT_OK(stream_->Memcpy(keys_out.data(), *keys_out_buffer, key_bytes));
    ASSERT_OK(
        stream_->Memcpy(values_out.data(), *values_out_buffer, val_bytes));
    ASSERT_OK(stream_->BlockHostUntilDone());

    std::vector<std::pair<Key, Value>> pairs(num_items);
    for (int64_t i = 0; i < num_items; ++i) {
      pairs[i] = {keys_in[i], values_in[i]};
    }
    absl::c_stable_sort(pairs, [descending](const auto& a, const auto& b) {
      return KeyLessThan(a.first, b.first, descending);
    });
    std::vector<Key> expected_keys(num_items);
    std::vector<Value> expected_values(num_items);
    for (int64_t i = 0; i < num_items; ++i) {
      expected_keys[i] = pairs[i].first;
      expected_values[i] = pairs[i].second;
    }
    ExpectKeysBitEqual(keys_out, expected_keys);
    EXPECT_EQ(values_out, expected_values);
  }

  se::StreamExecutor* executor_ = nullptr;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<se::StreamExecutorAddressAllocator> allocator_;
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
  using ExclusiveSumFactory =
      se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                             se::DeviceAddressBase, int32_t>;
  ASSERT_OK_AND_ASSIGN(
      auto exclusive_sum_kernel,
      ExclusiveSumFactory::Create(executor_, specs.exclusive_sum));

  constexpr int kNumPasses = 8;
  constexpr int32_t kNumPortions = 300;  // > 256 exercises loop stride.
  constexpr int64_t kTotalBins = kNumPasses * kCubSortOnesweepRadixDigits;
  constexpr int64_t kBinsBytes = kTotalBins * sizeof(CubGlobalOffset);
  constexpr int64_t kCountersBytes = static_cast<int64_t>(kNumPasses) *
                                     kNumPortions *
                                     kCubSortScratchAlignmentBytes;

  std::vector<CubGlobalOffset> bins_in(kTotalBins);
  std::vector<CubGlobalOffset> expected_bins_out(kTotalBins);
  for (int pass = 0; pass < kNumPasses; ++pass) {
    CubGlobalOffset running = 0;
    for (int bin = 0; bin < kCubSortOnesweepRadixDigits; ++bin) {
      const int idx = pass * kCubSortOnesweepRadixDigits + bin;
      // Values exceed UINT32_MAX to verify 64-bit BlockScan accumulation.
      bins_in[idx] = (CubGlobalOffset{1} << 32) +
                     static_cast<CubGlobalOffset>(pass * 1000 + bin);
      expected_bins_out[idx] = running;
      running += bins_in[idx];
    }
  }

  ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> bins_in_buffer,
                       Allocate(kBinsBytes));
  ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> bins_out_buffer,
                       Allocate(kBinsBytes));
  ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> counters_out_buffer,
                       Allocate(kCountersBytes));
  ASSERT_OK(stream_->Memcpy(bins_in_buffer.ptr(), bins_in.data(), kBinsBytes));
  ASSERT_OK(stream_->MemZero(bins_out_buffer.ptr(), kBinsBytes));
  ASSERT_OK(stream_->Memset32(counters_out_buffer.ptr(), 0x5a5a5a5au,
                              kCountersBytes));

  ASSERT_OK(exclusive_sum_kernel.Launch(
      se::ThreadDim(kCubSortExclusiveSumBlockThreads), se::BlockDim(kNumPasses),
      /*shmem_bytes=*/0, stream_.get(), *bins_in_buffer, *bins_out_buffer,
      *counters_out_buffer, kNumPortions));

  std::vector<CubGlobalOffset> bins_out(kTotalBins);
  std::vector<int32_t> counters_out(kCountersBytes / sizeof(int32_t));
  ASSERT_OK(stream_->Memcpy(bins_out.data(), *bins_out_buffer, kBinsBytes));
  ASSERT_OK(stream_->Memcpy(counters_out.data(), *counters_out_buffer,
                            kCountersBytes));
  ASSERT_OK(stream_->BlockHostUntilDone());

  EXPECT_EQ(bins_out, expected_bins_out);
  constexpr int64_t kCounterStrideInt32 =
      kCubSortScratchAlignmentBytes / static_cast<int64_t>(sizeof(int32_t));
  for (int32_t p = 0; p < kNumPortions; ++p) {
    for (int pass = 0; pass < kNumPasses; ++pass) {
      const int64_t slot = static_cast<int64_t>(p) * kNumPasses + pass;
      ASSERT_EQ(counters_out[slot * kCounterStrideInt32], 0)
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
  // Architectures compiled natively into the fatbinary (SM <= 10.x in standard
  // CUDA builds) extract an uncompressed single-arch CUBIN ELF rather than
  // falling back to the full multi-arch fatbinary.
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
}

TEST_F(CubSortKernelsTest, KernelSpecSupportMatchesCubSortTypesAcrossAllTypes) {
  auto verify_types = [&](PrimitiveType key_type,
                          std::optional<PrimitiveType> val_type) {
    const bool supported = IsCubSortSupported(key_type, val_type);
    for (bool descending : {false, true}) {
      auto check_support = [&](const auto& result, absl::string_view kind) {
        if (supported) {
          EXPECT_THAT(result, IsOk())
              << kind << ": " << PrimitiveType_Name(key_type) << ", "
              << (val_type ? PrimitiveType_Name(*val_type) : "keys-only")
              << ", descending=" << descending;
        } else {
          EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument))
              << kind << ": " << PrimitiveType_Name(key_type) << ", "
              << (val_type ? PrimitiveType_Name(*val_type) : "keys-only")
              << ", descending=" << descending;
        }
      };
      check_support(
          GetCubSortSingleTileKernelSpec(key_type, val_type, descending, cc_),
          "single_tile");
      check_support(
          GetCubSortOnesweepKernelSpecs(key_type, val_type, descending, cc_),
          "onesweep");
    }
  };

  for (int k = PrimitiveType_MIN; k <= PrimitiveType_MAX; ++k) {
    if (!PrimitiveType_IsValid(k)) {
      continue;
    }
    const auto key_type = static_cast<PrimitiveType>(k);
    verify_types(key_type, std::nullopt);
    for (int v = PrimitiveType_MIN; v <= PrimitiveType_MAX; ++v) {
      if (!PrimitiveType_IsValid(v)) {
        continue;
      }
      verify_types(key_type, static_cast<PrimitiveType>(v));
    }
  }
}

}  // namespace
}  // namespace xla::gpu
