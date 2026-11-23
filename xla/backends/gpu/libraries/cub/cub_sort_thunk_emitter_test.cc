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

#include "xla/backends/gpu/libraries/cub/cub_sort_thunk_emitter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_planner.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_policy.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/backends/gpu/runtime/thunk_test_utils.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;
using ::absl_testing::StatusIs;

template <typename KeyT>
bool KeyLessThan(KeyT a, KeyT b, bool descending) {
  if constexpr (std::is_integral_v<KeyT>) {
    return descending ? (a > b) : (a < b);
  } else {
    auto to_ordered = [](KeyT v) -> uint64_t {
      uint32_t bits = absl::bit_cast<uint32_t>(v);
      if ((bits & 0x7FFFFFFFu) == 0) {
        bits = 0;
      }
      uint32_t mask =
          (bits & 0x80000000u) ? uint32_t{0xFFFFFFFFu} : uint32_t{0x80000000u};
      return static_cast<uint64_t>(bits ^ mask);
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
    for (int64_t i = 0; i < count; ++i) {
      keys[i] = static_cast<KeyT>((i % 7 == 0) ? (rng() % 16) : rng());
    }
  } else {
    std::uniform_real_distribution<float> dist(-500.0f, 500.0f);
    for (int64_t i = 0; i < count; ++i) {
      if (i % 37 == 0) {
        keys[i] = -0.0f;
      } else if (i % 41 == 0) {
        keys[i] = +0.0f;
      } else {
        keys[i] = dist(rng);
      }
    }
  }
  return keys;
}

se::DeviceDescription MakeSyntheticCudaDevice(int major, int minor,
                                              int core_count) {
  se::DeviceDescription desc;
  desc.set_gpu_compute_capability(se::CudaComputeCapability(major, minor));
  desc.set_core_count(core_count);
  desc.set_threads_per_block_limit(1024);
  desc.set_threads_per_core_limit(2048);
  desc.set_threads_per_warp(32);
  desc.set_registers_per_block_limit(65536);
  desc.set_registers_per_core_limit(65536);
  desc.set_shared_memory_per_block(48 * 1024);
  desc.set_shared_memory_per_core(major >= 9 ? 228 * 1024 : 164 * 1024);
  desc.set_shared_memory_per_block_optin(major >= 9 ? 227 * 1024 : 163 * 1024);
  desc.set_reserved_shared_memory_per_block(1024);
  desc.set_max_blocks_per_multiprocessor(32);
  return desc;
}

// Verifies that every emitted CustomKernelThunk satisfies the pure-Read /
// pure-Write invariant (no overlapping read and write slices within a single
// thunk) and round-trips cleanly through ToProto / FromProto.
void VerifyInvariantsAndRoundTrip(
    ThunkSequence& thunks, absl::Span<const BufferAllocation> allocations) {
  ThunkSequence rebuilt;
  rebuilt.reserve(thunks.size());
  for (const std::unique_ptr<Thunk>& thunk : thunks) {
    // Every emitted thunk must be a flat Command (CustomKernel or Copy),
    // never a nested SequentialThunk.
    ASSERT_NE(dynamic_cast<Command*>(thunk.get()), nullptr);
    EXPECT_TRUE(thunk->kind() == Thunk::kCustomKernel ||
                thunk->kind() == Thunk::kCopy);

    // Check no read slice overlaps any write slice inside the same thunk.
    Thunk::BufferUses uses = thunk->buffer_uses();
    for (size_t i = 0; i < uses.size(); ++i) {
      for (size_t j = i + 1; j < uses.size(); ++j) {
        if (uses[i].access() == BufferUse::MemoryAccess::kWrite ||
            uses[j].access() == BufferUse::MemoryAccess::kWrite) {
          EXPECT_FALSE(uses[i].slice().OverlapsWith(uses[j].slice()))
              << "Thunk " << thunk->ToString(0) << " has overlapping uses "
              << uses[i].slice().ToString() << " and "
              << uses[j].slice().ToString();
        }
      }
    }

    ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk->ToProto());
    if (thunk->kind() == Thunk::kCustomKernel) {
      ASSERT_TRUE(proto.has_custom_kernel_thunk());
      ASSERT_OK_AND_ASSIGN(
          std::unique_ptr<CustomKernelThunk> round_tripped,
          CustomKernelThunk::FromProto(thunk->thunk_info(),
                                       proto.custom_kernel_thunk(), allocations,
                                       /*devices_in_process=*/1));
      rebuilt.push_back(std::move(round_tripped));
    } else {
      ASSERT_TRUE(proto.has_device_to_device_copy_thunk());
      ASSERT_OK_AND_ASSIGN(
          std::unique_ptr<DeviceToDeviceCopyThunk> round_tripped,
          DeviceToDeviceCopyThunk::FromProto(
              thunk->thunk_info(), proto.device_to_device_copy_thunk(),
              allocations));
      rebuilt.push_back(std::move(round_tripped));
    }
  }
  thunks = std::move(rebuilt);
}

class CubSortThunkEmitterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                         se::PlatformManager::PlatformWithName("CUDA"));
    ASSERT_OK_AND_ASSIGN(executor_, platform->ExecutorForDevice(0));
    ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream());
    allocator_ =
        std::make_unique<se::StreamExecutorAddressAllocator>(executor_);
  }

  // Emits, round-trips through proto, and executes a CUB sort both eagerly
  // (`ExecuteOnStream`) and inside a concurrent `CommandBufferThunk` (testing
  // RecordCreate, Replay, and RecordUpdate with swapped buffers).
  template <typename KeyT, typename ValT = uint32_t>
  void RunAndVerify(PrimitiveType key_type,
                    std::optional<PrimitiveType> val_type, int64_t batch_size,
                    int64_t segment_size, bool descending,
                    std::optional<int32_t> portion_size_override = std::nullopt,
                    std::optional<int32_t> max_grid_x_override = std::nullopt) {
    const bool is_pairs = val_type.has_value();
    const int64_t num_items = batch_size * segment_size;
    const se::DeviceDescription& desc = executor_->GetDeviceDescription();
    ASSERT_OK_AND_ASSIGN(
        CubSortExecutionPlan plan,
        PlanCubSortExecution(desc.cuda_compute_capability(), key_type, val_type,
                             num_items, batch_size, portion_size_override,
                             max_grid_x_override));

    const Shape key_shape =
        batch_size > 1
            ? ShapeUtil::MakeShape(key_type, {batch_size, segment_size})
            : ShapeUtil::MakeShape(key_type, {num_items});
    const int64_t key_bytes = std::max<int64_t>(1, num_items * sizeof(KeyT));
    const int64_t val_bytes =
        is_pairs ? std::max<int64_t>(1, num_items * sizeof(ValT)) : 1;
    const int64_t scratch_bytes = std::max<int64_t>(1, plan.scratch_bytes);
    const int64_t scratch_alloc_bytes = RoundUpTo<int64_t>(scratch_bytes, 4);

    std::vector<BufferAllocation> allocations;
    allocations.emplace_back(/*index=*/0, key_bytes, /*color=*/0);
    allocations.emplace_back(/*index=*/1, key_bytes, /*color=*/0);
    allocations.emplace_back(/*index=*/2, scratch_alloc_bytes, /*color=*/0);
    if (is_pairs) {
      allocations.emplace_back(/*index=*/3, val_bytes, /*color=*/0);
      allocations.emplace_back(/*index=*/4, val_bytes, /*color=*/0);
    }

    ShapedSlice keys_in{
        BufferAllocation::Slice(&allocations[0], 0, num_items * sizeof(KeyT)),
        key_shape};
    ShapedSlice keys_out{
        BufferAllocation::Slice(&allocations[1], 0, num_items * sizeof(KeyT)),
        key_shape};
    ShapedSlice scratch{
        BufferAllocation::Slice(&allocations[2], 0, scratch_bytes),
        ShapeUtil::MakeShape(U8, {scratch_bytes})};

    std::optional<ShapedSlice> vals_in;
    std::optional<ShapedSlice> vals_out;
    if (is_pairs) {
      const Shape val_shape =
          batch_size > 1
              ? ShapeUtil::MakeShape(*val_type, {batch_size, segment_size})
              : ShapeUtil::MakeShape(*val_type, {num_items});
      vals_in = ShapedSlice{
          BufferAllocation::Slice(&allocations[3], 0, num_items * sizeof(ValT)),
          val_shape};
      vals_out = ShapedSlice{
          BufferAllocation::Slice(&allocations[4], 0, num_items * sizeof(ValT)),
          val_shape};
    }

    int64_t next_id = 1;
    auto make_info = [&]() {
      Thunk::ThunkInfo info;
      info.thunk_id = ThunkId(next_id++);
      info.profile_annotation = "test_cub_sort";
      return info;
    };

    CubSortThunkEmitParams emit_params{
        /*make_thunk_info=*/make_info,
        /*device_description=*/desc,
        /*devices_in_process=*/1,
        /*key_type=*/key_type,
        /*value_type=*/val_type,
        /*descending=*/descending,
        /*batch_size=*/batch_size,
        /*keys_in=*/keys_in,
        /*keys_out=*/keys_out,
        /*values_in=*/vals_in,
        /*values_out=*/vals_out,
        /*scratch=*/scratch,
        /*portion_size_override=*/portion_size_override,
        /*max_grid_x_override=*/max_grid_x_override,
    };

    ASSERT_OK_AND_ASSIGN(ThunkSequence thunks,
                         EmitCudaCubSortThunkSequence(emit_params));
    VerifyInvariantsAndRoundTrip(thunks, allocations);

    if (num_items == 0) {
      EXPECT_TRUE(thunks.empty());
      return;
    }

    auto make_expected = [&](const std::vector<KeyT>& h_keys,
                             const std::vector<ValT>& h_vals) {
      std::vector<KeyT> exp_keys(num_items);
      std::vector<ValT> exp_vals(is_pairs ? num_items : 0);
      if (is_pairs) {
        std::vector<std::pair<KeyT, ValT>> pairs(num_items);
        for (int64_t i = 0; i < num_items; ++i) {
          pairs[i] = {h_keys[i], h_vals[i]};
        }
        for (int64_t b = 0; b < batch_size; ++b) {
          auto begin = pairs.begin() + b * segment_size;
          std::stable_sort(begin, begin + segment_size,
                           [descending](const auto& a, const auto& c) {
                             return KeyLessThan(a.first, c.first, descending);
                           });
        }
        for (int64_t i = 0; i < num_items; ++i) {
          exp_keys[i] = pairs[i].first;
          exp_vals[i] = pairs[i].second;
        }
      } else {
        exp_keys = h_keys;
        for (int64_t b = 0; b < batch_size; ++b) {
          auto begin = exp_keys.begin() + b * segment_size;
          std::stable_sort(begin, begin + segment_size,
                           [descending](KeyT a, KeyT c) {
                             return KeyLessThan(a, c, descending);
                           });
        }
      }
      return std::make_pair(std::move(exp_keys), std::move(exp_vals));
    };

    auto alloc_device_set = [&]() {
      std::vector<se::DeviceAddressBase> bufs;
      bufs.push_back(executor_->Allocate(key_bytes, 0));
      bufs.push_back(executor_->Allocate(key_bytes, 0));
      bufs.push_back(executor_->Allocate(scratch_alloc_bytes, 0));
      if (is_pairs) {
        bufs.push_back(executor_->Allocate(val_bytes, 0));
        bufs.push_back(executor_->Allocate(val_bytes, 0));
      }
      return bufs;
    };
    auto free_device_set = [&](std::vector<se::DeviceAddressBase>& bufs) {
      for (auto& b : bufs) {
        executor_->Deallocate(&b);
      }
    };
    auto poison_scratch_and_zero_outputs =
        [&](std::vector<se::DeviceAddressBase>& bufs) {
          ASSERT_OK(stream_->MemZero(&bufs[1], key_bytes));
          ASSERT_OK(
              stream_->Memset32(&bufs[2], 0xDEADBEEFu, scratch_alloc_bytes));
          if (is_pairs) {
            ASSERT_OK(stream_->MemZero(&bufs[4], val_bytes));
          }
        };

    auto check_outputs = [&](const std::vector<se::DeviceAddressBase>& bufs,
                             const std::vector<KeyT>& orig_keys,
                             const std::vector<ValT>& orig_vals,
                             const std::vector<KeyT>& exp_keys,
                             const std::vector<ValT>& exp_vals) {
      std::vector<KeyT> actual_keys(num_items);
      std::vector<KeyT> actual_keys_in(num_items);
      ASSERT_OK(stream_->Memcpy(actual_keys.data(), bufs[1],
                                num_items * sizeof(KeyT)));
      ASSERT_OK(stream_->Memcpy(actual_keys_in.data(), bufs[0],
                                num_items * sizeof(KeyT)));
      std::vector<ValT> actual_vals(is_pairs ? num_items : 0);
      std::vector<ValT> actual_vals_in(is_pairs ? num_items : 0);
      if (is_pairs) {
        ASSERT_OK(stream_->Memcpy(actual_vals.data(), bufs[4],
                                  num_items * sizeof(ValT)));
        ASSERT_OK(stream_->Memcpy(actual_vals_in.data(), bufs[3],
                                  num_items * sizeof(ValT)));
      }
      ASSERT_OK(stream_->BlockHostUntilDone());
      for (int64_t i = 0; i < num_items; ++i) {
        if constexpr (sizeof(KeyT) == 4) {
          ASSERT_EQ(absl::bit_cast<uint32_t>(actual_keys[i]),
                    absl::bit_cast<uint32_t>(exp_keys[i]))
              << "key mismatch at index " << i;
          ASSERT_EQ(absl::bit_cast<uint32_t>(actual_keys_in[i]),
                    absl::bit_cast<uint32_t>(orig_keys[i]))
              << "input key mutated at index " << i;
        } else {
          ASSERT_EQ(actual_keys[i], exp_keys[i])
              << "key mismatch at index " << i;
          ASSERT_EQ(actual_keys_in[i], orig_keys[i])
              << "input key mutated at index " << i;
        }
      }
      if (is_pairs) {
        EXPECT_EQ(actual_vals, exp_vals);
        EXPECT_EQ(actual_vals_in, orig_vals);
      }
    };

    // Dataset 1.
    std::vector<KeyT> h_keys_1 = GenerateKeys<KeyT>(num_items, 1001);
    std::vector<ValT> h_vals_1(is_pairs ? num_items : 0);
    for (int64_t i = 0; i < static_cast<int64_t>(h_vals_1.size()); ++i) {
      h_vals_1[i] = static_cast<ValT>(i);
    }
    auto [exp_keys_1, exp_vals_1] = make_expected(h_keys_1, h_vals_1);

    std::vector<se::DeviceAddressBase> bufs_1 = alloc_device_set();
    absl::Cleanup free_bufs_1 = [&] { free_device_set(bufs_1); };
    ASSERT_OK(
        stream_->Memcpy(&bufs_1[0], h_keys_1.data(), num_items * sizeof(KeyT)));
    if (is_pairs) {
      ASSERT_OK(stream_->Memcpy(&bufs_1[3], h_vals_1.data(),
                                num_items * sizeof(ValT)));
    }
    poison_scratch_and_zero_outputs(bufs_1);

    // 1. Direct ExecuteOnStream over the flat ThunkSequence.
    {
      BufferAllocations buffer_allocs(bufs_1, 0, allocator_.get());
      TestThunkParams tparams(executor_, &buffer_allocs, stream_.get());
      for (auto& thunk : thunks) {
        ASSERT_OK(thunk->Initialize(tparams.InitializeParams()));
      }
      Thunk::ExecuteParams exec_params = tparams.ExecuteParams();
      for (auto& thunk : thunks) {
        ASSERT_OK(thunk->ExecuteOnStream(exec_params));
      }
      check_outputs(bufs_1, h_keys_1, h_vals_1, exp_keys_1, exp_vals_1);
    }

    // 2. Concurrent CommandBufferThunk (CUDA Graph Create + Replay + Update).
    CommandSequence commands;
    for (auto& thunk : thunks) {
      Command* cmd = dynamic_cast<Command*>(thunk.release());
      ASSERT_NE(cmd, nullptr);
      commands.Append(std::unique_ptr<Command>(cmd));
    }
    ASSERT_OK_AND_ASSIGN(
        CommandExecutor cmd_executor,
        CommandExecutor::Create(
            std::move(commands),
            CommandExecutor::SynchronizationMode::kConcurrent));
    CommandBufferThunk cmd_buffer_thunk(std::move(cmd_executor),
                                        Thunk::ThunkInfo(),
                                        /*devices_in_process=*/1);

    // 2a. RecordCreate + Execute.
    poison_scratch_and_zero_outputs(bufs_1);
    {
      BufferAllocations buffer_allocs(bufs_1, 0, allocator_.get());
      TestThunkParams tparams(executor_, &buffer_allocs, stream_.get());
      tparams.WithCommandBufferTraceStream(stream_.get());
      tparams.WithPersistentAllocIndices({});
      ASSERT_OK(cmd_buffer_thunk.Initialize(tparams.InitializeParams()));
      ASSERT_OK(cmd_buffer_thunk.ExecuteOnStream(tparams.ExecuteParams()));
      check_outputs(bufs_1, h_keys_1, h_vals_1, exp_keys_1, exp_vals_1);

      // 2b. Replay with same buffers (after poisoning scratch and zeroing
      // outputs).
      poison_scratch_and_zero_outputs(bufs_1);
      ASSERT_OK(cmd_buffer_thunk.ExecuteOnStream(tparams.ExecuteParams()));
      check_outputs(bufs_1, h_keys_1, h_vals_1, exp_keys_1, exp_vals_1);
    }

    // 2c. RecordUpdate with brand new buffer addresses and new dataset!
    std::vector<KeyT> h_keys_2 = GenerateKeys<KeyT>(num_items, 2002);
    std::vector<ValT> h_vals_2(is_pairs ? num_items : 0);
    for (int64_t i = 0; i < static_cast<int64_t>(h_vals_2.size()); ++i) {
      h_vals_2[i] = static_cast<ValT>(i + 7);
    }
    auto [exp_keys_2, exp_vals_2] = make_expected(h_keys_2, h_vals_2);

    std::vector<se::DeviceAddressBase> bufs_2 = alloc_device_set();
    absl::Cleanup free_bufs_2 = [&] { free_device_set(bufs_2); };
    ASSERT_OK(
        stream_->Memcpy(&bufs_2[0], h_keys_2.data(), num_items * sizeof(KeyT)));
    if (is_pairs) {
      ASSERT_OK(stream_->Memcpy(&bufs_2[3], h_vals_2.data(),
                                num_items * sizeof(ValT)));
    }
    poison_scratch_and_zero_outputs(bufs_2);
    {
      BufferAllocations buffer_allocs_2(bufs_2, 0, allocator_.get());
      TestThunkParams tparams_2(executor_, &buffer_allocs_2, stream_.get());
      tparams_2.WithCommandBufferTraceStream(stream_.get());
      tparams_2.WithPersistentAllocIndices({});
      ASSERT_OK(cmd_buffer_thunk.ExecuteOnStream(tparams_2.ExecuteParams()));
      check_outputs(bufs_2, h_keys_2, h_vals_2, exp_keys_2, exp_vals_2);
    }
  }

  se::StreamExecutor* executor_ = nullptr;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<se::StreamExecutorAddressAllocator> allocator_;
};

TEST_F(CubSortThunkEmitterTest, DevicelessEmissionAcrossGpuArchitectures) {
  for (const auto& [major, minor, sms] : std::vector<std::tuple<int, int, int>>{
           {8, 0, 108}, {8, 6, 84}, {8, 9, 128}, {9, 0, 132}, {10, 0, 148}}) {
    se::DeviceDescription desc = MakeSyntheticCudaDevice(major, minor, sms);
    ASSERT_OK_AND_ASSIGN(
        int64_t scratch_bytes,
        ComputeCudaCubSortScratchSize(desc.cuda_compute_capability(), F32, U32,
                                      65536, 1));
    std::vector<BufferAllocation> allocations;
    allocations.emplace_back(0, 65536 * 4, 0);
    allocations.emplace_back(1, 65536 * 4, 0);
    allocations.emplace_back(2, scratch_bytes, 0);
    allocations.emplace_back(3, 65536 * 4, 0);
    allocations.emplace_back(4, 65536 * 4, 0);

    const Shape shape = ShapeUtil::MakeShape(F32, {65536});
    const Shape val_shape = ShapeUtil::MakeShape(U32, {65536});
    int64_t next_id = 1;
    auto make_info = [&]() {
      Thunk::ThunkInfo info;
      info.thunk_id = ThunkId(next_id++);
      return info;
    };
    CubSortThunkEmitParams params{
        /*make_thunk_info=*/make_info,
        /*device_description=*/desc,
        /*devices_in_process=*/1,
        /*key_type=*/F32,
        /*value_type=*/U32,
        /*descending=*/false,
        /*batch_size=*/1,
        /*keys_in=*/
        {BufferAllocation::Slice(&allocations[0], 0, 65536 * 4), shape},
        /*keys_out=*/
        {BufferAllocation::Slice(&allocations[1], 0, 65536 * 4), shape},
        /*values_in=*/
        ShapedSlice{BufferAllocation::Slice(&allocations[3], 0, 65536 * 4),
                    val_shape},
        /*values_out=*/
        ShapedSlice{BufferAllocation::Slice(&allocations[4], 0, 65536 * 4),
                    val_shape},
        /*scratch=*/
        {BufferAllocation::Slice(&allocations[2], 0, scratch_bytes),
         ShapeUtil::MakeShape(U8, {scratch_bytes})},
    };
    ASSERT_OK_AND_ASSIGN(ThunkSequence thunks,
                         EmitCudaCubSortThunkSequence(params));
    // Histogram + ExclusiveSum + 4 Onesweep passes = 6 CustomKernelThunks.
    ASSERT_EQ(thunks.size(), 6) << "sm_" << major << minor;
    const auto* histo_thunk =
        dynamic_cast<const CustomKernelThunk*>(thunks[0].get());
    ASSERT_NE(histo_thunk, nullptr);
    const CubRadixSortPolicySpec policy = GetCubRadixSortPolicySpec(
        CanonicalCubSortPolicySmVersion(major), 4, 4, /*is_float32_key=*/true);
    const int64_t max_histo_tiles =
        CeilOfRatio<int64_t>(65536, policy.histogram.tile_items());
    EXPECT_GT(histo_thunk->custom_kernel().block_dims().x, 0);
    EXPECT_LE(histo_thunk->custom_kernel().block_dims().x, max_histo_tiles);
    VerifyInvariantsAndRoundTrip(thunks, allocations);
  }

  // Verify uninitialized negative reserved_shared_memory_per_block (-1, as in
  // TestGpuDeviceInfo) matches 0 instead of wrapping to SIZE_MAX, and missing
  // occupancy fields fall back conservatively to <= 4 blocks/SM.
  {
    constexpr int64_t kLargeN = 4 * 1024 * 1024;
    se::DeviceDescription desc_zero = MakeSyntheticCudaDevice(9, 0, 10);
    desc_zero.set_reserved_shared_memory_per_block(0);
    se::DeviceDescription desc_uninit = MakeSyntheticCudaDevice(9, 0, 10);
    desc_uninit.set_reserved_shared_memory_per_block(-1);
    desc_uninit.set_shared_memory_per_block_optin(-1);
    se::DeviceDescription desc_incomplete = MakeSyntheticCudaDevice(9, 0, 10);
    desc_incomplete.set_registers_per_core_limit(0);

    ASSERT_OK_AND_ASSIGN(
        int64_t scratch_bytes,
        ComputeCudaCubSortScratchSize(desc_zero.cuda_compute_capability(), F32,
                                      std::nullopt, kLargeN, 1));
    std::vector<BufferAllocation> allocations;
    allocations.emplace_back(0, kLargeN * 4, 0);
    allocations.emplace_back(1, kLargeN * 4, 0);
    allocations.emplace_back(2, scratch_bytes, 0);
    const Shape shape = ShapeUtil::MakeShape(F32, {kLargeN});
    auto emit_histo_grid = [&](const se::DeviceDescription& d) -> int64_t {
      int64_t next_id = 1;
      auto make_info = [&]() {
        Thunk::ThunkInfo info;
        info.thunk_id = ThunkId(next_id++);
        return info;
      };
      CubSortThunkEmitParams params{
          /*make_thunk_info=*/make_info,
          /*device_description=*/d,
          /*devices_in_process=*/1,
          /*key_type=*/F32,
          /*value_type=*/std::nullopt,
          /*descending=*/false,
          /*batch_size=*/1,
          /*keys_in=*/
          {BufferAllocation::Slice(&allocations[0], 0, kLargeN * 4), shape},
          /*keys_out=*/
          {BufferAllocation::Slice(&allocations[1], 0, kLargeN * 4), shape},
          /*values_in=*/std::nullopt,
          /*values_out=*/std::nullopt,
          /*scratch=*/
          {BufferAllocation::Slice(&allocations[2], 0, scratch_bytes),
           ShapeUtil::MakeShape(U8, {scratch_bytes})},
      };
      auto thunks_or = EmitCudaCubSortThunkSequence(params);
      EXPECT_OK(thunks_or);
      if (!thunks_or.ok() || thunks_or->empty()) {
        return -1;
      }
      const auto* histo =
          dynamic_cast<const CustomKernelThunk*>((*thunks_or)[0].get());
      return histo ? static_cast<int64_t>(histo->custom_kernel().block_dims().x)
                   : -1;
    };
    const int64_t grid_zero = emit_histo_grid(desc_zero);
    const int64_t grid_uninit = emit_histo_grid(desc_uninit);
    const int64_t grid_incomplete = emit_histo_grid(desc_incomplete);
    EXPECT_GT(grid_zero, 0);
    EXPECT_EQ(grid_uninit, grid_zero);
    EXPECT_GT(grid_incomplete, 0);
    EXPECT_LE(grid_incomplete, 4 * 10);
  }
}

TEST_F(CubSortThunkEmitterTest, EmptyAndTrivialCopyExecuteAndCapture) {
  RunAndVerify<float>(F32, std::nullopt, /*batch_size=*/1, /*segment_size=*/0,
                      /*descending=*/false);
  RunAndVerify<float, uint32_t>(F32, U32, /*batch_size=*/1, /*segment_size=*/1,
                                /*descending=*/false);
  RunAndVerify<int32_t>(S32, std::nullopt, /*batch_size=*/16,
                        /*segment_size=*/1, /*descending=*/true);
}

TEST_F(CubSortThunkEmitterTest, SingleTileExecuteAndCapture) {
  RunAndVerify<float>(F32, std::nullopt, /*batch_size=*/1,
                      /*segment_size=*/1024, /*descending=*/false);
  RunAndVerify<float, uint32_t>(F32, U32, /*batch_size=*/7,
                                /*segment_size=*/512, /*descending=*/true,
                                std::nullopt, /*max_grid_x_override=*/3);
}

TEST_F(CubSortThunkEmitterTest,
       OnesweepSingleAndMultiPortionExecuteAndCapture) {
  RunAndVerify<uint8_t>(U8, std::nullopt, /*batch_size=*/1,
                        /*segment_size=*/12000, /*descending=*/false);
  RunAndVerify<float, uint32_t>(F32, U32, /*batch_size=*/1,
                                /*segment_size=*/20000, /*descending=*/true);

  const int sm_version = CanonicalCubSortPolicySmVersion(
      executor_->GetDeviceDescription().cuda_compute_capability().major);
  const CubRadixSortPolicySpec policy =
      GetCubRadixSortPolicySpec(sm_version, 4, 4, /*is_float32_key=*/true);
  const int32_t small_portion = 2 * policy.onesweep.tile_items();
  RunAndVerify<float, uint32_t>(F32, U32, /*batch_size=*/1,
                                /*segment_size=*/2 * small_portion + 137,
                                /*descending=*/false, small_portion);
}

TEST_F(CubSortThunkEmitterTest, SegmentedMultiPassExecuteAndCapture) {
  RunAndVerify<int32_t>(S32, std::nullopt, /*batch_size=*/5,
                        /*segment_size=*/8000, /*descending=*/false,
                        std::nullopt, /*max_grid_x_override=*/2);
  // Odd pass count (U16 -> 3 passes) exercises pass 0 writing directly to kOut.
  RunAndVerify<uint16_t>(U16, std::nullopt, /*batch_size=*/4,
                         /*segment_size=*/8000, /*descending=*/true,
                         std::nullopt, /*max_grid_x_override=*/2);
  RunAndVerify<float, uint32_t>(F32, U32, /*batch_size=*/4,
                                /*segment_size=*/8000, /*descending=*/true,
                                std::nullopt, /*max_grid_x_override=*/2);
}

TEST_F(CubSortThunkEmitterTest,
       NonZeroSharedAllocationOffsetsAndAliasedInputs) {
  // Pack keys_in (= values_in, testing read-only input aliasing), keys_out,
  // vals_out, and scratch at non-zero 256-byte-aligned offsets inside a single
  // shared BufferAllocation.
  constexpr int64_t kNumItems = 16384;
  constexpr int64_t kDataBytes = kNumItems * sizeof(float);
  const se::DeviceDescription& desc = executor_->GetDeviceDescription();
  ASSERT_OK_AND_ASSIGN(
      CubSortExecutionPlan plan,
      PlanCubSortExecution(desc.cuda_compute_capability(), F32, F32, kNumItems,
                           /*batch_size=*/1));
  ASSERT_EQ(plan.mode, CubSortMode::kOnesweep);

  const int64_t off_in = 256;
  const int64_t off_keys_out =
      RoundUpTo<int64_t>(off_in + kDataBytes, kCubSortScratchAlignmentBytes);
  const int64_t off_vals_out = RoundUpTo<int64_t>(
      off_keys_out + kDataBytes, kCubSortScratchAlignmentBytes);
  const int64_t off_scratch = RoundUpTo<int64_t>(off_vals_out + kDataBytes,
                                                 kCubSortScratchAlignmentBytes);
  const int64_t total_bytes =
      RoundUpTo<int64_t>(off_scratch + plan.scratch_bytes, 256);

  std::vector<BufferAllocation> allocations;
  allocations.emplace_back(/*index=*/0, total_bytes, /*color=*/0);
  const Shape shape = ShapeUtil::MakeShape(F32, {kNumItems});
  ShapedSlice in_slice{
      BufferAllocation::Slice(&allocations[0], off_in, kDataBytes), shape};
  ShapedSlice keys_out_slice{
      BufferAllocation::Slice(&allocations[0], off_keys_out, kDataBytes),
      shape};
  ShapedSlice vals_out_slice{
      BufferAllocation::Slice(&allocations[0], off_vals_out, kDataBytes),
      shape};
  ShapedSlice scratch_slice{
      BufferAllocation::Slice(&allocations[0], off_scratch, plan.scratch_bytes),
      ShapeUtil::MakeShape(U8, {plan.scratch_bytes})};

  int64_t next_id = 1;
  auto make_info = [&]() {
    Thunk::ThunkInfo info;
    info.thunk_id = ThunkId(next_id++);
    return info;
  };
  CubSortThunkEmitParams params{
      /*make_thunk_info=*/make_info,
      /*device_description=*/desc,
      /*devices_in_process=*/1,
      /*key_type=*/F32,
      /*value_type=*/F32,
      /*descending=*/false,
      /*batch_size=*/1,
      /*keys_in=*/in_slice,
      /*keys_out=*/keys_out_slice,
      /*values_in=*/in_slice,
      /*values_out=*/vals_out_slice,
      /*scratch=*/scratch_slice,
  };
  ASSERT_OK_AND_ASSIGN(ThunkSequence thunks,
                       EmitCudaCubSortThunkSequence(params));
  VerifyInvariantsAndRoundTrip(thunks, allocations);

  std::vector<float> h_in = GenerateKeys<float>(kNumItems, 4242);
  std::vector<float> expected = h_in;
  absl::c_stable_sort(expected, [](float a, float b) {
    return KeyLessThan(a, b, /*descending=*/false);
  });

  se::DeviceAddressBase d_shared = executor_->Allocate(total_bytes, 0);
  absl::Cleanup free_shared = [&] { executor_->Deallocate(&d_shared); };
  ASSERT_OK(stream_->Memset32(&d_shared, 0xDEADBEEFu, total_bytes));
  se::DeviceAddressBase d_in(static_cast<char*>(d_shared.opaque()) + off_in,
                             kDataBytes);
  se::DeviceAddressBase d_keys_out(
      static_cast<char*>(d_shared.opaque()) + off_keys_out, kDataBytes);
  se::DeviceAddressBase d_vals_out(
      static_cast<char*>(d_shared.opaque()) + off_vals_out, kDataBytes);
  ASSERT_OK(stream_->Memcpy(&d_in, h_in.data(), kDataBytes));

  CommandSequence commands;
  for (auto& thunk : thunks) {
    Command* cmd = dynamic_cast<Command*>(thunk.release());
    ASSERT_NE(cmd, nullptr);
    commands.Append(std::unique_ptr<Command>(cmd));
  }
  ASSERT_OK_AND_ASSIGN(CommandExecutor cmd_executor,
                       CommandExecutor::Create(
                           std::move(commands),
                           CommandExecutor::SynchronizationMode::kConcurrent));
  CommandBufferThunk cmd_buffer_thunk(std::move(cmd_executor),
                                      Thunk::ThunkInfo(),
                                      /*devices_in_process=*/1);
  BufferAllocations buffer_allocs({d_shared}, 0, allocator_.get());
  TestThunkParams tparams(executor_, &buffer_allocs, stream_.get());
  tparams.WithCommandBufferTraceStream(stream_.get());
  tparams.WithPersistentAllocIndices({});
  ASSERT_OK(cmd_buffer_thunk.Initialize(tparams.InitializeParams()));
  ASSERT_OK(cmd_buffer_thunk.ExecuteOnStream(tparams.ExecuteParams()));

  std::vector<float> actual_keys(kNumItems);
  std::vector<float> actual_vals(kNumItems);
  std::vector<float> actual_in(kNumItems);
  ASSERT_OK(stream_->Memcpy(actual_keys.data(), d_keys_out, kDataBytes));
  ASSERT_OK(stream_->Memcpy(actual_vals.data(), d_vals_out, kDataBytes));
  ASSERT_OK(stream_->Memcpy(actual_in.data(), d_in, kDataBytes));
  ASSERT_OK(stream_->BlockHostUntilDone());

  for (int64_t i = 0; i < kNumItems; ++i) {
    ASSERT_EQ(absl::bit_cast<uint32_t>(actual_keys[i]),
              absl::bit_cast<uint32_t>(expected[i]))
        << "at index " << i;
    ASSERT_EQ(absl::bit_cast<uint32_t>(actual_vals[i]),
              absl::bit_cast<uint32_t>(expected[i]))
        << "at index " << i;
    ASSERT_EQ(absl::bit_cast<uint32_t>(actual_in[i]),
              absl::bit_cast<uint32_t>(h_in[i]))
        << "input mutated at index " << i;
  }
}

TEST_F(CubSortThunkEmitterTest, InvalidParamsReturnErrors) {
  const se::DeviceDescription desc = MakeSyntheticCudaDevice(9, 0, 132);
  constexpr int64_t kN = 16384;
  ASSERT_OK_AND_ASSIGN(
      int64_t scratch_bytes,
      ComputeCudaCubSortScratchSize(desc.cuda_compute_capability(), F32,
                                    std::nullopt, kN, 1));
  std::vector<BufferAllocation> allocs;
  allocs.emplace_back(0, kN * 4, 0);
  allocs.emplace_back(1, kN * 4, 0);
  allocs.emplace_back(2, scratch_bytes, 0);
  allocs.emplace_back(3, kN * 4, 0);
  allocs.emplace_back(4, kN * 4, 0);

  const Shape f32_shape = ShapeUtil::MakeShape(F32, {kN});
  const Shape u32_shape = ShapeUtil::MakeShape(U32, {kN});
  auto make_info = []() { return Thunk::ThunkInfo(); };
  auto base_params = [&](const se::DeviceDescription& d) {
    return CubSortThunkEmitParams{
        /*make_thunk_info=*/make_info,
        /*device_description=*/d,
        /*devices_in_process=*/1,
        /*key_type=*/F32,
        /*value_type=*/std::nullopt,
        /*descending=*/false,
        /*batch_size=*/1,
        /*keys_in=*/{BufferAllocation::Slice(&allocs[0], 0, kN * 4), f32_shape},
        /*keys_out=*/
        {BufferAllocation::Slice(&allocs[1], 0, kN * 4), f32_shape},
        /*values_in=*/std::nullopt,
        /*values_out=*/std::nullopt,
        /*scratch=*/
        {BufferAllocation::Slice(&allocs[2], 0, scratch_bytes),
         ShapeUtil::MakeShape(U8, {scratch_bytes})},
    };
  };

  // Non-CUDA device description.
  {
    se::DeviceDescription rocm_desc;
    rocm_desc.set_gpu_compute_capability(
        se::GpuComputeCapability(se::RocmComputeCapability("gfx942")));
    CubSortThunkEmitParams p = base_params(rocm_desc);
    EXPECT_THAT(EmitCudaCubSortThunkSequence(p),
                StatusIs(absl::StatusCode::kInternal));
  }
  // Mismatched value_type vs values_in/values_out presence.
  {
    CubSortThunkEmitParams p = base_params(desc);
    p.value_type = U32;
    EXPECT_THAT(EmitCudaCubSortThunkSequence(p),
                StatusIs(absl::StatusCode::kInternal));
  }
  // Non-array shape.
  {
    CubSortThunkEmitParams p = base_params(desc);
    p.keys_in.shape = ShapeUtil::MakeTupleShape({f32_shape});
    EXPECT_THAT(EmitCudaCubSortThunkSequence(p),
                StatusIs(absl::StatusCode::kInternal));
  }
  // Mismatched element_type.
  {
    CubSortThunkEmitParams p = base_params(desc);
    p.keys_out.shape = u32_shape;
    EXPECT_THAT(EmitCudaCubSortThunkSequence(p),
                StatusIs(absl::StatusCode::kInternal));
  }
  // Undersized output slice.
  {
    CubSortThunkEmitParams p = base_params(desc);
    p.keys_out.slice = BufferAllocation::Slice(&allocs[1], 0, kN * 2);
    EXPECT_THAT(EmitCudaCubSortThunkSequence(p),
                StatusIs(absl::StatusCode::kInternal));
  }
  // Undersized scratch buffer.
  {
    CubSortThunkEmitParams p = base_params(desc);
    p.scratch.slice = BufferAllocation::Slice(&allocs[2], 0, scratch_bytes - 1);
    EXPECT_THAT(EmitCudaCubSortThunkSequence(p),
                StatusIs(absl::StatusCode::kInternal));
  }
  // Overlapping keys_in / keys_out (violates __restrict__).
  {
    CubSortThunkEmitParams p = base_params(desc);
    p.keys_out.slice = p.keys_in.slice;
    EXPECT_THAT(EmitCudaCubSortThunkSequence(p),
                StatusIs(absl::StatusCode::kInternal));
  }
  // Overlapping scratch / keys_out.
  {
    CubSortThunkEmitParams p = base_params(desc);
    p.keys_out.slice = BufferAllocation::Slice(&allocs[2], 0, kN * 4);
    EXPECT_THAT(EmitCudaCubSortThunkSequence(p),
                StatusIs(absl::StatusCode::kInternal));
  }
}

}  // namespace
}  // namespace xla::gpu
