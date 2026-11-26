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
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_planner.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_policy.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_testlib.h"
#include "xla/backends/gpu/runtime/buffer_allocations.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/backends/gpu/runtime/thunk_test_utils.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_elf_utils.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/kernel_spec.h"
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
using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::AnyOf;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Not;

template <typename Key>
bool KeyLessThan(Key a, Key b, bool descending) {
  if constexpr (std::is_integral_v<Key>) {
    return descending ? (a > b) : (a < b);
  } else {
    auto to_ordered = [](Key v) -> uint64_t {
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

template <typename Key>
std::vector<Key> GenerateKeys(int64_t count, uint32_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<Key> keys(count);
  if constexpr (std::is_integral_v<Key>) {
    for (int64_t i = 0; i < count; ++i) {
      keys[i] = static_cast<Key>((i % 7 == 0) ? (rng() % 16) : rng());
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

absl::StatusOr<se::DeviceDescription> LoadGpuModelDescription(GpuModel model) {
  ABSL_ASSIGN_OR_RETURN(se::GpuTargetConfigProto proto,
                        GetGpuTargetConfig(model));
  ABSL_ASSIGN_OR_RETURN(GpuTargetConfig config,
                        GpuTargetConfig::FromProto(proto));
  return std::move(config.device_description);
}

// Verifies that every emitted CustomKernelThunk satisfies the pure-Read /
// pure-Write invariant (no overlapping read and write slices within a single
// thunk) and round-trips cleanly through ToProto / FromProto.
absl::StatusOr<ThunkSequence> VerifyInvariantsAndRoundTrip(
    const ThunkSequence& thunks,
    absl::Span<const BufferAllocation> allocations) {
  ThunkSequence rebuilt;
  rebuilt.reserve(thunks.size());
  for (const std::unique_ptr<Thunk>& thunk : thunks) {
    // Every emitted thunk must be a flat CustomKernelThunk Command, never a
    // nested SequentialThunk.
    if (dynamic_cast<const Command*>(thunk.get()) == nullptr) {
      return absl::InternalError(
          absl::StrCat("Emitted thunk is not a Command: ", thunk->ToString(0)));
    }
    if (thunk->kind() != Thunk::kCustomKernel) {
      return absl::InternalError(
          absl::StrCat("Expected kCustomKernel thunk, got kind ", thunk->kind(),
                       ": ", thunk->ToString(0)));
    }

    // Check no read slice overlaps any write slice inside the same thunk.
    Thunk::BufferUses uses = thunk->buffer_uses();
    for (size_t i = 0; i < uses.size(); ++i) {
      for (size_t j = i + 1; j < uses.size(); ++j) {
        if ((uses[i].access() == BufferUse::MemoryAccess::kWrite ||
             uses[j].access() == BufferUse::MemoryAccess::kWrite) &&
            uses[i].slice().OverlapsWith(uses[j].slice())) {
          return absl::InternalError(absl::StrCat(
              "Thunk ", thunk->ToString(0), " has overlapping uses ",
              uses[i].slice().ToString(), " and ", uses[j].slice().ToString()));
        }
      }
    }

    ABSL_ASSIGN_OR_RETURN(ThunkProto proto, thunk->ToProto());
    if (!proto.has_custom_kernel_thunk()) {
      return absl::InternalError(
          "Serialized ThunkProto is missing custom_kernel_thunk payload");
    }
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<CustomKernelThunk> round_tripped,
        CustomKernelThunk::FromProto(thunk->thunk_info(),
                                     proto.custom_kernel_thunk(), allocations,
                                     /*devices_in_process=*/1));
    rebuilt.push_back(std::move(round_tripped));
  }
  return rebuilt;
}

// Wraps a flat ThunkSequence of Command thunks into a concurrent
// CommandBufferThunk without transferring ownership (`thunks` must outlive the
// returned CommandBufferThunk).
absl::StatusOr<std::unique_ptr<CommandBufferThunk>>
BuildConcurrentCommandBufferThunk(const ThunkSequence& thunks) {
  CommandSequence commands;
  for (const std::unique_ptr<Thunk>& thunk : thunks) {
    Command* cmd = dynamic_cast<Command*>(thunk.get());
    if (cmd == nullptr) {
      return absl::InternalError(
          absl::StrCat("Thunk is not a Command: ", thunk->ToString(0)));
    }
    commands.Append(cmd);
  }
  ABSL_ASSIGN_OR_RETURN(CommandExecutor cmd_executor,
                        CommandExecutor::Create(
                            std::move(commands),
                            CommandExecutor::SynchronizationMode::kConcurrent));
  return std::make_unique<CommandBufferThunk>(
      std::move(cmd_executor), Thunk::ThunkInfo(), /*devices_in_process=*/1);
}

struct SortDeviceBuffers {
  se::ScopedDeviceAddress<uint8_t> keys_in;
  se::ScopedDeviceAddress<uint8_t> keys_out;
  se::ScopedDeviceAddress<uint8_t> scratch;
  std::optional<se::ScopedDeviceAddress<uint8_t>> vals_in;
  std::optional<se::ScopedDeviceAddress<uint8_t>> vals_out;

  std::vector<se::DeviceAddressBase> RawAddresses() const {
    std::vector<se::DeviceAddressBase> raw = {*keys_in, *keys_out, *scratch};
    if (vals_in.has_value()) {
      raw.push_back(**vals_in);
      raw.push_back(**vals_out);
    }
    return raw;
  }
};

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
  template <typename Key, typename Val = uint32_t>
  void RunAndVerify(PrimitiveType key_type,
                    std::optional<PrimitiveType> val_type, int64_t batch_size,
                    int64_t segment_size, bool descending,
                    CubSortPlanOptions plan_options = {}) {
    SCOPED_TRACE(testing::Message()
                 << "key=" << PrimitiveType_Name(key_type) << " val="
                 << (val_type ? PrimitiveType_Name(*val_type) : "none")
                 << " batch=" << batch_size << " seg=" << segment_size
                 << " desc=" << descending);

    const bool is_pairs = val_type.has_value();
    const int64_t num_items = batch_size * segment_size;
    const se::DeviceDescription& desc = executor_->GetDeviceDescription();
    ASSERT_OK_AND_ASSIGN(
        CubSortExecutionPlan plan,
        PlanCubSortExecution(desc.cuda_compute_capability(), key_type, val_type,
                             num_items, batch_size, plan_options));

    const Shape key_shape =
        batch_size > 1
            ? ShapeUtil::MakeShape(key_type, {batch_size, segment_size})
            : ShapeUtil::MakeShape(key_type, {num_items});
    const int64_t key_bytes = std::max<int64_t>(1, num_items * sizeof(Key));
    const int64_t val_bytes =
        is_pairs ? std::max<int64_t>(1, num_items * sizeof(Val)) : 1;
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

    CubSortBufferPair keys{
        .in = ShapedSlice{BufferAllocation::Slice(&allocations[0], 0,
                                                  num_items * sizeof(Key)),
                          key_shape},
        .out = ShapedSlice{BufferAllocation::Slice(&allocations[1], 0,
                                                   num_items * sizeof(Key)),
                           key_shape},
    };
    ShapedSlice scratch{
        BufferAllocation::Slice(&allocations[2], 0, scratch_bytes),
        ShapeUtil::MakeShape(U8, {scratch_bytes})};

    std::optional<CubSortBufferPair> values;
    if (is_pairs) {
      const Shape val_shape =
          batch_size > 1
              ? ShapeUtil::MakeShape(*val_type, {batch_size, segment_size})
              : ShapeUtil::MakeShape(*val_type, {num_items});
      values = CubSortBufferPair{
          .in = ShapedSlice{BufferAllocation::Slice(&allocations[3], 0,
                                                    num_items * sizeof(Val)),
                            val_shape},
          .out = ShapedSlice{BufferAllocation::Slice(&allocations[4], 0,
                                                     num_items * sizeof(Val)),
                             val_shape},
      };
    }

    int64_t next_id = 1;
    auto make_info = [&]() {
      Thunk::ThunkInfo info;
      info.thunk_id = ThunkId(next_id++);
      info.profile_annotation = "test_cub_sort";
      return info;
    };

    CubSortThunkEmitParams emit_params{
        .devices_in_process = 1,
        .descending = descending,
        .batch_size = batch_size,
        .keys = keys,
        .values = values,
        .scratch = scratch,
    };

    ASSERT_OK_AND_ASSIGN(ThunkSequence thunks,
                         EmitCudaCubSortThunkSequence(desc, emit_params,
                                                      make_info, plan_options));
    ASSERT_OK_AND_ASSIGN(thunks,
                         VerifyInvariantsAndRoundTrip(thunks, allocations));

    auto make_expected = [&](const std::vector<Key>& input_keys,
                             const std::vector<Val>& input_values) {
      std::vector<Key> exp_keys(num_items);
      std::vector<Val> exp_vals(is_pairs ? num_items : 0);
      if (is_pairs) {
        std::vector<std::pair<Key, Val>> pairs(num_items);
        for (int64_t i = 0; i < num_items; ++i) {
          pairs[i] = {input_keys[i], input_values[i]};
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
        exp_keys = input_keys;
        for (int64_t b = 0; b < batch_size; ++b) {
          auto begin = exp_keys.begin() + b * segment_size;
          std::stable_sort(begin, begin + segment_size,
                           [descending](Key a, Key c) {
                             return KeyLessThan(a, c, descending);
                           });
        }
      }
      return std::pair(std::move(exp_keys), std::move(exp_vals));
    };

    auto alloc_device_set = [&]() -> absl::StatusOr<SortDeviceBuffers> {
      const int ordinal = executor_->device_ordinal();
      ABSL_ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> d_keys_in,
                            allocator_->Allocate(ordinal, key_bytes));
      ABSL_ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> d_keys_out,
                            allocator_->Allocate(ordinal, key_bytes));
      ABSL_ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> d_scratch,
                            allocator_->Allocate(ordinal, scratch_alloc_bytes));
      SortDeviceBuffers bufs{
          .keys_in = std::move(d_keys_in),
          .keys_out = std::move(d_keys_out),
          .scratch = std::move(d_scratch),
      };
      if (is_pairs) {
        absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> d_vals_in =
            allocator_->Allocate(ordinal, val_bytes);
        if (!d_vals_in.ok()) {
          return d_vals_in.status();
        }
        bufs.vals_in = *std::move(d_vals_in);
        absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> d_vals_out =
            allocator_->Allocate(ordinal, val_bytes);
        if (!d_vals_out.ok()) {
          return d_vals_out.status();
        }
        bufs.vals_out = *std::move(d_vals_out);
      }
      return bufs;
    };

    auto poison_scratch_and_zero_outputs =
        [&](SortDeviceBuffers& bufs) -> absl::Status {
      ABSL_RETURN_IF_ERROR(stream_->MemZero(bufs.keys_out.ptr(), key_bytes));
      ABSL_RETURN_IF_ERROR(stream_->Memset32(bufs.scratch.ptr(), 0xDEADBEEFu,
                                             scratch_alloc_bytes));
      if (is_pairs) {
        ABSL_RETURN_IF_ERROR(stream_->MemZero(bufs.vals_out->ptr(), val_bytes));
      }
      return absl::OkStatus();
    };

    auto check_outputs =
        [&](absl::string_view phase, const SortDeviceBuffers& bufs,
            const std::vector<Key>& orig_keys,
            const std::vector<Val>& orig_vals, const std::vector<Key>& exp_keys,
            const std::vector<Val>& exp_vals) {
          SCOPED_TRACE(phase);
          std::vector<Key> actual_keys(num_items);
          std::vector<Key> actual_keys_in(num_items);
          ASSERT_OK(stream_->Memcpy(actual_keys.data(), *bufs.keys_out,
                                    num_items * sizeof(Key)));
          ASSERT_OK(stream_->Memcpy(actual_keys_in.data(), *bufs.keys_in,
                                    num_items * sizeof(Key)));
          std::vector<Val> actual_vals(is_pairs ? num_items : 0);
          std::vector<Val> actual_vals_in(is_pairs ? num_items : 0);
          if (is_pairs) {
            ASSERT_OK(stream_->Memcpy(actual_vals.data(), **bufs.vals_out,
                                      num_items * sizeof(Val)));
            ASSERT_OK(stream_->Memcpy(actual_vals_in.data(), **bufs.vals_in,
                                      num_items * sizeof(Val)));
          }
          ASSERT_OK(stream_->BlockHostUntilDone());
          for (int64_t i = 0; i < num_items; ++i) {
            if constexpr (sizeof(Key) == 4) {
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
    std::vector<Key> keys_1 = GenerateKeys<Key>(num_items, 1001);
    std::vector<Val> values_1(is_pairs ? num_items : 0);
    for (int64_t i = 0; i < static_cast<int64_t>(values_1.size()); ++i) {
      values_1[i] = static_cast<Val>(i);
    }
    auto [exp_keys_1, exp_vals_1] = make_expected(keys_1, values_1);

    ASSERT_OK_AND_ASSIGN(SortDeviceBuffers bufs_1, alloc_device_set());
    ASSERT_OK(stream_->Memcpy(bufs_1.keys_in.ptr(), keys_1.data(),
                              num_items * sizeof(Key)));
    if (is_pairs) {
      ASSERT_OK(stream_->Memcpy(bufs_1.vals_in->ptr(), values_1.data(),
                                num_items * sizeof(Val)));
    }
    ASSERT_OK(poison_scratch_and_zero_outputs(bufs_1));

    // 1. Direct ExecuteOnStream over the flat ThunkSequence.
    {
      BufferAllocations buffer_allocs(bufs_1.RawAddresses(), 0,
                                      allocator_.get());
      TestThunkParams tparams(executor_, &buffer_allocs, stream_.get());
      for (const std::unique_ptr<Thunk>& thunk : thunks) {
        ASSERT_OK(thunk->Initialize(tparams.InitializeParams()));
      }
      Thunk::ExecuteParams exec_params = tparams.ExecuteParams();
      for (const std::unique_ptr<Thunk>& thunk : thunks) {
        ASSERT_OK(thunk->ExecuteOnStream(exec_params));
      }
      ASSERT_NO_FATAL_FAILURE(check_outputs("ExecuteOnStream", bufs_1, keys_1,
                                            values_1, exp_keys_1, exp_vals_1));
    }

    // 2. Concurrent CommandBufferThunk (CUDA Graph Create + Replay + Update).
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBufferThunk> cmd_buffer_thunk,
                         BuildConcurrentCommandBufferThunk(thunks));

    // 2a. RecordCreate + Execute.
    ASSERT_OK(poison_scratch_and_zero_outputs(bufs_1));
    {
      BufferAllocations buffer_allocs(bufs_1.RawAddresses(), 0,
                                      allocator_.get());
      TestThunkParams tparams(executor_, &buffer_allocs, stream_.get());
      tparams.WithCommandBufferTraceStream(stream_.get());
      tparams.WithPersistentAllocIndices({});
      ASSERT_OK(cmd_buffer_thunk->Initialize(tparams.InitializeParams()));
      ASSERT_OK(cmd_buffer_thunk->ExecuteOnStream(tparams.ExecuteParams()));
      ASSERT_NO_FATAL_FAILURE(check_outputs("CommandBufferCreate", bufs_1,
                                            keys_1, values_1, exp_keys_1,
                                            exp_vals_1));

      // 2b. Replay with same buffers (after poisoning scratch and zeroing
      // outputs).
      ASSERT_OK(poison_scratch_and_zero_outputs(bufs_1));
      ASSERT_OK(cmd_buffer_thunk->ExecuteOnStream(tparams.ExecuteParams()));
      ASSERT_NO_FATAL_FAILURE(check_outputs("CommandBufferReplay", bufs_1,
                                            keys_1, values_1, exp_keys_1,
                                            exp_vals_1));
    }

    // 2c. RecordUpdate with brand new buffer addresses and new dataset.
    std::vector<Key> keys_2 = GenerateKeys<Key>(num_items, 2002);
    std::vector<Val> values_2(is_pairs ? num_items : 0);
    for (int64_t i = 0; i < static_cast<int64_t>(values_2.size()); ++i) {
      values_2[i] = static_cast<Val>(i + 7);
    }
    auto [exp_keys_2, exp_vals_2] = make_expected(keys_2, values_2);

    ASSERT_OK_AND_ASSIGN(SortDeviceBuffers bufs_2, alloc_device_set());
    ASSERT_OK(stream_->Memcpy(bufs_2.keys_in.ptr(), keys_2.data(),
                              num_items * sizeof(Key)));
    if (is_pairs) {
      ASSERT_OK(stream_->Memcpy(bufs_2.vals_in->ptr(), values_2.data(),
                                num_items * sizeof(Val)));
    }
    ASSERT_OK(poison_scratch_and_zero_outputs(bufs_2));
    {
      BufferAllocations buffer_allocs_2(bufs_2.RawAddresses(), 0,
                                        allocator_.get());
      TestThunkParams tparams_2(executor_, &buffer_allocs_2, stream_.get());
      tparams_2.WithCommandBufferTraceStream(stream_.get());
      tparams_2.WithPersistentAllocIndices({});
      ASSERT_OK(cmd_buffer_thunk->ExecuteOnStream(tparams_2.ExecuteParams()));
      ASSERT_NO_FATAL_FAILURE(check_outputs("CommandBufferUpdate", bufs_2,
                                            keys_2, values_2, exp_keys_2,
                                            exp_vals_2));
    }
  }

  se::StreamExecutor* executor_ = nullptr;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<se::StreamExecutorAddressAllocator> allocator_;
};

struct ArchTestCase {
  std::string name;
  // Nullopt loads `TestGpuDeviceInfo::RTXA6000DeviceInfo()` (SM 8.9, which has
  // no dedicated `GpuModel` entry).
  std::optional<GpuModel> gpu_model;
};

class CubSortDevicelessArchTest
    : public ::testing::TestWithParam<ArchTestCase> {};

TEST_P(CubSortDevicelessArchTest, EmitsValidOnesweepThunks) {
  const ArchTestCase& tc = GetParam();
  se::DeviceDescription desc;
  if (tc.gpu_model.has_value()) {
    ASSERT_OK_AND_ASSIGN(desc, LoadGpuModelDescription(*tc.gpu_model));
  } else {
    desc = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  }
  const se::CudaComputeCapability cc = desc.cuda_compute_capability();
  const int64_t sms = desc.core_count();
  ASSERT_GT(sms, 0);

  constexpr int64_t kNumItems = 16 * 1024 * 1024;
  ASSERT_OK_AND_ASSIGN(
      int64_t scratch_bytes,
      ComputeCudaCubSortScratchSize(cc, F32, U32, kNumItems, /*batch_size=*/1));
  std::vector<BufferAllocation> allocations;
  allocations.emplace_back(0, kNumItems * 4, 0);
  allocations.emplace_back(1, kNumItems * 4, 0);
  allocations.emplace_back(2, scratch_bytes, 0);
  allocations.emplace_back(3, kNumItems * 4, 0);
  allocations.emplace_back(4, kNumItems * 4, 0);

  const Shape key_shape = ShapeUtil::MakeShape(F32, {kNumItems});
  const Shape val_shape = ShapeUtil::MakeShape(U32, {kNumItems});
  int64_t next_id = 1;
  auto make_info = [&]() {
    Thunk::ThunkInfo info;
    info.thunk_id = ThunkId(next_id++);
    return info;
  };
  CubSortThunkEmitParams params{
      .devices_in_process = 1,
      .descending = false,
      .batch_size = 1,
      .keys =
          CubSortBufferPair{
              .in = {BufferAllocation::Slice(&allocations[0], 0, kNumItems * 4),
                     key_shape},
              .out = {BufferAllocation::Slice(&allocations[1], 0,
                                              kNumItems * 4),
                      key_shape},
          },
      .values =
          CubSortBufferPair{
              .in = {BufferAllocation::Slice(&allocations[3], 0, kNumItems * 4),
                     val_shape},
              .out = {BufferAllocation::Slice(&allocations[4], 0,
                                              kNumItems * 4),
                      val_shape},
          },
      .scratch = {BufferAllocation::Slice(&allocations[2], 0, scratch_bytes),
                  ShapeUtil::MakeShape(U8, {scratch_bytes})},
  };
  ASSERT_OK_AND_ASSIGN(ThunkSequence thunks,
                       EmitCudaCubSortThunkSequence(desc, params, make_info));
  // Histogram + ExclusiveSum + 4 Onesweep passes = 6 CustomKernelThunks.
  ASSERT_EQ(thunks.size(), 6) << tc.name;
  const auto* histogram_thunk =
      dynamic_cast<const CustomKernelThunk*>(thunks[0].get());
  ASSERT_NE(histogram_thunk, nullptr);
  const CubRadixSortPolicySpec policy = GetCubRadixSortPolicySpec(
      CanonicalCubSortPolicySmVersion(cc.major), 4, 4, /*is_float32_key=*/true);
  const int64_t max_histogram_tiles =
      CeilOfRatio<int64_t>(kNumItems, policy.histogram.tile_items());
  const int64_t grid_x = histogram_thunk->custom_kernel().block_dims().x;
  EXPECT_GE(grid_x, sms);
  EXPECT_LT(grid_x, max_histogram_tiles);
  EXPECT_EQ(grid_x % sms, 0);
  EXPECT_OK(VerifyInvariantsAndRoundTrip(thunks, allocations));
}

INSTANTIATE_TEST_SUITE_P(
    GpuArchitectures, CubSortDevicelessArchTest,
    ::testing::Values(ArchTestCase{"A100", GpuModel::A100_SXM_80},
                      ArchTestCase{"A6000", GpuModel::A6000},
                      ArchTestCase{"AdaSm89", std::nullopt},
                      ArchTestCase{"H100", GpuModel::H100_SXM},
                      ArchTestCase{"B200", GpuModel::B200}),
    [](const ::testing::TestParamInfo<ArchTestCase>& info) {
      return info.param.name;
    });

TEST(CubSortThunkEmitterDevicelessTest,
     HistogramOccupancyClampsNegativeSharedMemAndFallsBack) {
  // Verify uninitialized negative reserved_shared_memory_per_block (-1, as in
  // TestGpuDeviceInfo) matches 0 instead of wrapping to SIZE_MAX, tight SMEM
  // limits reduce static occupancy when CUBIN ELF attributes are embedded, and
  // missing occupancy fields fall back conservatively to <= 4 blocks/SM.
  constexpr int64_t kLargeN = 4 * 1024 * 1024;
  ASSERT_OK_AND_ASSIGN(se::DeviceDescription desc_base,
                       LoadGpuModelDescription(GpuModel::H100_SXM));
  desc_base.set_core_count(10);

  se::DeviceDescription desc_zero = desc_base;
  desc_zero.set_reserved_shared_memory_per_block(0);

  se::DeviceDescription desc_uninit = desc_base;
  desc_uninit.set_reserved_shared_memory_per_block(-1);
  desc_uninit.set_shared_memory_per_block_optin(-1);

  se::DeviceDescription desc_tight_smem = desc_zero;
  desc_tight_smem.set_shared_memory_per_core(8 * 1024);

  se::DeviceDescription desc_incomplete = desc_base;
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

  struct HistoInfo {
    int64_t grid_x = -1;
    bool has_embedded_cubin = false;
  };
  auto emit_histo_info =
      [&](const se::DeviceDescription& d) -> absl::StatusOr<HistoInfo> {
    int64_t next_id = 1;
    CubSortThunkEmitParams params{
        .devices_in_process = 1,
        .descending = false,
        .batch_size = 1,
        .keys =
            CubSortBufferPair{
                .in = {BufferAllocation::Slice(&allocations[0], 0, kLargeN * 4),
                       shape},
                .out = {BufferAllocation::Slice(&allocations[1], 0,
                                                kLargeN * 4),
                        shape},
            },
        .scratch = {BufferAllocation::Slice(&allocations[2], 0, scratch_bytes),
                    ShapeUtil::MakeShape(U8, {scratch_bytes})},
    };
    ABSL_ASSIGN_OR_RETURN(ThunkSequence thunks,
                          EmitCudaCubSortThunkSequence(d, params, [&]() {
                            Thunk::ThunkInfo info;
                            info.thunk_id = ThunkId(next_id++);
                            return info;
                          }));
    if (thunks.empty()) {
      return absl::InternalError("Expected non-empty thunk sequence");
    }
    const auto* histo = dynamic_cast<const CustomKernelThunk*>(thunks[0].get());
    if (histo == nullptr) {
      return absl::InternalError("Expected CustomKernelThunk at index 0");
    }
    const std::optional<se::CudaCubinInMemory> cubin =
        histo->custom_kernel().kernel_loader_spec().cuda_cubin_in_memory();
    return HistoInfo{
        .grid_x = static_cast<int64_t>(histo->custom_kernel().block_dims().x),
        .has_embedded_cubin =
            cubin.has_value() && se::cuda::IsCudaElf(cubin->cubin_bytes),
    };
  };

  ASSERT_OK_AND_ASSIGN(HistoInfo info_zero, emit_histo_info(desc_zero));
  ASSERT_OK_AND_ASSIGN(HistoInfo info_uninit, emit_histo_info(desc_uninit));
  ASSERT_OK_AND_ASSIGN(HistoInfo info_tight_smem,
                       emit_histo_info(desc_tight_smem));
  ASSERT_OK_AND_ASSIGN(HistoInfo info_incomplete,
                       emit_histo_info(desc_incomplete));

  EXPECT_GT(info_zero.grid_x, 0);
  EXPECT_EQ(info_uninit.grid_x, info_zero.grid_x);
  EXPECT_GT(info_tight_smem.grid_x, 0);
  // PTX-only builds have nullopt histogram_attrs and fall back to 4 blocks/SM
  // for both configs; when an embedded CUBIN ELF is present, tightening SMEM
  // must measurably reduce static occupancy.
  if (info_zero.has_embedded_cubin) {
    EXPECT_LT(info_tight_smem.grid_x, info_zero.grid_x);
  }
  EXPECT_GT(info_incomplete.grid_x, 0);
  EXPECT_LE(info_incomplete.grid_x, 4 * 10);
}

TEST_F(CubSortThunkEmitterTest, SingleTileExecuteAndCapture) {
  RunAndVerify<float>(F32, std::nullopt, /*batch_size=*/1,
                      /*segment_size=*/1024, /*descending=*/false);
  RunAndVerify<float, uint32_t>(F32, U32, /*batch_size=*/7,
                                /*segment_size=*/512, /*descending=*/true,
                                CubSortPlanOptions{.max_grid_x_override = 3});
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
  RunAndVerify<float, uint32_t>(
      F32, U32, /*batch_size=*/1,
      /*segment_size=*/2 * small_portion + 137, /*descending=*/false,
      CubSortPlanOptions{.portion_size_override = small_portion});
}

TEST_F(CubSortThunkEmitterTest, SegmentedMultiPassExecuteAndCapture) {
  RunAndVerify<int32_t>(S32, std::nullopt, /*batch_size=*/5,
                        /*segment_size=*/8000, /*descending=*/false,
                        CubSortPlanOptions{.max_grid_x_override = 2});
  // Odd pass count (U16 -> 3 passes) exercises pass 0 writing directly to kOut.
  RunAndVerify<uint16_t>(U16, std::nullopt, /*batch_size=*/4,
                         /*segment_size=*/8000, /*descending=*/true,
                         CubSortPlanOptions{.max_grid_x_override = 2});
  RunAndVerify<float, uint32_t>(F32, U32, /*batch_size=*/4,
                                /*segment_size=*/8000, /*descending=*/true,
                                CubSortPlanOptions{.max_grid_x_override = 2});
}

TEST_F(CubSortThunkEmitterTest,
       NonZeroSharedAllocationOffsetsAndAliasedInputs) {
  // Pack keys.in (= values->in, testing read-only input aliasing), keys.out,
  // values->out, and scratch at non-zero 256-byte-aligned offsets inside a
  // single shared BufferAllocation.
  constexpr int64_t kNumItems = 16384;
  constexpr int64_t kDataBytes = kNumItems * sizeof(float);
  const se::DeviceDescription& desc = executor_->GetDeviceDescription();
  ASSERT_OK_AND_ASSIGN(
      CubSortExecutionPlan plan,
      PlanCubSortExecution(desc.cuda_compute_capability(), F32, F32, kNumItems,
                           /*batch_size=*/1));
  ASSERT_TRUE(std::holds_alternative<CubSortOnesweepPlan>(plan.mode_plan));

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
  CubSortThunkEmitParams params{
      .devices_in_process = 1,
      .descending = false,
      .batch_size = 1,
      .keys = CubSortBufferPair{.in = in_slice, .out = keys_out_slice},
      .values = CubSortBufferPair{.in = in_slice, .out = vals_out_slice},
      .scratch = scratch_slice,
  };
  ASSERT_OK_AND_ASSIGN(ThunkSequence thunks,
                       EmitCudaCubSortThunkSequence(desc, params, [&]() {
                         Thunk::ThunkInfo info;
                         info.thunk_id = ThunkId(next_id++);
                         return info;
                       }));
  ASSERT_OK_AND_ASSIGN(thunks,
                       VerifyInvariantsAndRoundTrip(thunks, allocations));

  std::vector<float> input_keys = GenerateKeys<float>(kNumItems, 4242);
  std::vector<float> expected = input_keys;
  absl::c_stable_sort(expected, [](float a, float b) {
    return KeyLessThan(a, b, /*descending=*/false);
  });

  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> shared_buffer,
      allocator_->Allocate(executor_->device_ordinal(), total_bytes));
  ASSERT_OK(stream_->Memset32(shared_buffer.ptr(), 0xDEADBEEFu, total_bytes));
  const se::DeviceAddressBase shared_base = *shared_buffer;
  se::DeviceAddressBase input_buffer =
      shared_base.GetByteSlice(off_in, kDataBytes);
  se::DeviceAddressBase keys_out_buffer =
      shared_base.GetByteSlice(off_keys_out, kDataBytes);
  se::DeviceAddressBase values_out_buffer =
      shared_base.GetByteSlice(off_vals_out, kDataBytes);
  ASSERT_OK(stream_->Memcpy(&input_buffer, input_keys.data(), kDataBytes));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBufferThunk> cmd_buffer_thunk,
                       BuildConcurrentCommandBufferThunk(thunks));
  BufferAllocations buffer_allocs({*shared_buffer}, 0, allocator_.get());
  TestThunkParams tparams(executor_, &buffer_allocs, stream_.get());
  tparams.WithCommandBufferTraceStream(stream_.get());
  tparams.WithPersistentAllocIndices({});
  ASSERT_OK(cmd_buffer_thunk->Initialize(tparams.InitializeParams()));
  ASSERT_OK(cmd_buffer_thunk->ExecuteOnStream(tparams.ExecuteParams()));

  std::vector<float> actual_keys(kNumItems);
  std::vector<float> actual_vals(kNumItems);
  std::vector<float> actual_in(kNumItems);
  ASSERT_OK(stream_->Memcpy(actual_keys.data(), keys_out_buffer, kDataBytes));
  ASSERT_OK(stream_->Memcpy(actual_vals.data(), values_out_buffer, kDataBytes));
  ASSERT_OK(stream_->Memcpy(actual_in.data(), input_buffer, kDataBytes));
  ASSERT_OK(stream_->BlockHostUntilDone());

  for (int64_t i = 0; i < kNumItems; ++i) {
    ASSERT_EQ(absl::bit_cast<uint32_t>(actual_keys[i]),
              absl::bit_cast<uint32_t>(expected[i]))
        << "at index " << i;
    ASSERT_EQ(absl::bit_cast<uint32_t>(actual_vals[i]),
              absl::bit_cast<uint32_t>(expected[i]))
        << "at index " << i;
    ASSERT_EQ(absl::bit_cast<uint32_t>(actual_in[i]),
              absl::bit_cast<uint32_t>(input_keys[i]))
        << "input mutated at index " << i;
  }
}

TEST(CubSortThunkEmitterDevicelessTest, InvalidParamsReturnErrors) {
  ASSERT_OK_AND_ASSIGN(se::DeviceDescription desc,
                       LoadGpuModelDescription(GpuModel::H100_SXM));
  constexpr int64_t kN = 16384;
  ASSERT_OK_AND_ASSIGN(int64_t pairs_scratch_bytes,
                       ComputeCudaCubSortScratchSize(
                           desc.cuda_compute_capability(), F32, U32, kN, 1));
  ASSERT_OK_AND_ASSIGN(
      int64_t keys_scratch_bytes,
      ComputeCudaCubSortScratchSize(desc.cuda_compute_capability(), F32,
                                    std::nullopt, kN, 1));
  const int64_t alloc_scratch_bytes =
      std::max(keys_scratch_bytes, pairs_scratch_bytes);
  std::vector<BufferAllocation> allocs;
  allocs.emplace_back(0, kN * 4, 0);
  allocs.emplace_back(1, kN * 4, 0);
  allocs.emplace_back(2, alloc_scratch_bytes, 0);
  allocs.emplace_back(3, kN * 4, 0);
  allocs.emplace_back(4, kN * 4, 0);

  const Shape f32_shape = ShapeUtil::MakeShape(F32, {kN});
  const Shape u32_shape = ShapeUtil::MakeShape(U32, {kN});
  auto make_info = []() { return Thunk::ThunkInfo(); };

  auto base_keys_params = [&]() {
    return CubSortThunkEmitParams{
        .devices_in_process = 1,
        .descending = false,
        .batch_size = 1,
        .keys =
            CubSortBufferPair{
                .in = {BufferAllocation::Slice(&allocs[0], 0, kN * 4),
                       f32_shape},
                .out = {BufferAllocation::Slice(&allocs[1], 0, kN * 4),
                        f32_shape},
            },
        .scratch = {BufferAllocation::Slice(&allocs[2], 0, keys_scratch_bytes),
                    ShapeUtil::MakeShape(U8, {keys_scratch_bytes})},
    };
  };
  auto base_pairs_params = [&]() {
    CubSortThunkEmitParams p = base_keys_params();
    p.values = CubSortBufferPair{
        .in = {BufferAllocation::Slice(&allocs[3], 0, kN * 4), u32_shape},
        .out = {BufferAllocation::Slice(&allocs[4], 0, kN * 4), u32_shape},
    };
    p.scratch = {BufferAllocation::Slice(&allocs[2], 0, pairs_scratch_bytes),
                 ShapeUtil::MakeShape(U8, {pairs_scratch_bytes})};
    return p;
  };

  // Both baselines must succeed so negative assertions cannot pass vacuously.
  ASSERT_OK(EmitCudaCubSortThunkSequence(desc, base_keys_params(), make_info));
  ASSERT_OK(EmitCudaCubSortThunkSequence(desc, base_pairs_params(), make_info));

  // Non-CUDA device description.
  {
    se::DeviceDescription rocm_desc;
    rocm_desc.set_gpu_compute_capability(
        se::GpuComputeCapability(se::RocmComputeCapability("gfx942")));
    EXPECT_THAT(
        EmitCudaCubSortThunkSequence(rocm_desc, base_keys_params(), make_info),
        StatusIs(absl::StatusCode::kFailedPrecondition,
                 HasSubstr("requires a CUDA device description")));
  }
  // Non-array shape.
  {
    CubSortThunkEmitParams p = base_keys_params();
    p.keys.in.shape = ShapeUtil::MakeTupleShape({f32_shape});
    EXPECT_THAT(EmitCudaCubSortThunkSequence(desc, p, make_info),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("must be an array")));
  }
  // Dynamic shape.
  {
    CubSortThunkEmitParams p = base_keys_params();
    Shape dynamic_shape = f32_shape;
    dynamic_shape.set_dynamic_dimension(0, true);
    p.keys.in.shape = dynamic_shape;
    EXPECT_THAT(
        EmitCudaCubSortThunkSequence(desc, p, make_info),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("dynamic")));
  }
  // Non-dim-0-major physical layout.
  {
    CubSortThunkEmitParams p = base_keys_params();
    Shape col_major =
        ShapeUtil::MakeShapeWithDenseLayout(F32, {4, kN / 4}, {0, 1});
    p.batch_size = 4;
    p.keys.in.shape = col_major;
    p.keys.out.shape = col_major;
    EXPECT_THAT(
        EmitCudaCubSortThunkSequence(desc, p, make_info),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("dim-0-major")));
  }
  // Mismatched key element_type.
  {
    CubSortThunkEmitParams p = base_keys_params();
    p.keys.out.shape = u32_shape;
    EXPECT_THAT(EmitCudaCubSortThunkSequence(desc, p, make_info),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("element type mismatch")));
  }
  // Undersized output slice.
  {
    CubSortThunkEmitParams p = base_keys_params();
    p.keys.out.slice = BufferAllocation::Slice(&allocs[1], 0, kN * 2);
    EXPECT_THAT(EmitCudaCubSortThunkSequence(desc, p, make_info),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("keys.out slice")));
  }
  // Undersized scratch buffer.
  {
    CubSortThunkEmitParams p = base_keys_params();
    p.scratch.slice =
        BufferAllocation::Slice(&allocs[2], 0, keys_scratch_bytes - 1);
    EXPECT_THAT(EmitCudaCubSortThunkSequence(desc, p, make_info),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("scratch buffer")));
  }
  // Overlapping keys.in / keys.out (violates __restrict__).
  {
    CubSortThunkEmitParams p = base_keys_params();
    p.keys.out.slice = p.keys.in.slice;
    EXPECT_THAT(EmitCudaCubSortThunkSequence(desc, p, make_info),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("keys.out and keys.in")));
  }
  // Overlapping scratch / keys.out.
  {
    CubSortThunkEmitParams p = base_keys_params();
    p.keys.out.slice = BufferAllocation::Slice(&allocs[2], 0, kN * 4);
    EXPECT_THAT(EmitCudaCubSortThunkSequence(desc, p, make_info),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("scratch and keys.out")));
  }

  // Pairs-mode validation: element type mismatch, undersized slice, overlaps.
  {
    CubSortThunkEmitParams p = base_pairs_params();
    p.values->out.shape = f32_shape;
    EXPECT_THAT(EmitCudaCubSortThunkSequence(desc, p, make_info),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("values input/output element type")));
  }
  {
    CubSortThunkEmitParams p = base_pairs_params();
    p.values->in.slice = BufferAllocation::Slice(&allocs[3], 0, kN * 2);
    EXPECT_THAT(EmitCudaCubSortThunkSequence(desc, p, make_info),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("values.in slice")));
  }
  {
    CubSortThunkEmitParams p = base_pairs_params();
    p.values->out.slice = BufferAllocation::Slice(&allocs[4], 0, kN * 2);
    EXPECT_THAT(EmitCudaCubSortThunkSequence(desc, p, make_info),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("values.out slice")));
  }
  {
    CubSortThunkEmitParams p = base_pairs_params();
    p.keys.out.slice = p.values->in.slice;
    EXPECT_THAT(EmitCudaCubSortThunkSequence(desc, p, make_info),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("keys.out and values.in")));
  }
  {
    CubSortThunkEmitParams p = base_pairs_params();
    p.keys.out.slice = p.values->out.slice;
    EXPECT_THAT(EmitCudaCubSortThunkSequence(desc, p, make_info),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("keys.out and values.out")));
  }
  {
    CubSortThunkEmitParams p = base_pairs_params();
    p.values->out.slice = p.keys.in.slice;
    EXPECT_THAT(EmitCudaCubSortThunkSequence(desc, p, make_info),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("values.out and keys.in")));
  }
  {
    CubSortThunkEmitParams p = base_pairs_params();
    p.values->out.slice = p.values->in.slice;
    EXPECT_THAT(EmitCudaCubSortThunkSequence(desc, p, make_info),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("values.out and values.in")));
  }
  {
    CubSortThunkEmitParams p = base_pairs_params();
    p.values->in.slice = BufferAllocation::Slice(&allocs[2], 0, kN * 4);
    EXPECT_THAT(EmitCudaCubSortThunkSequence(desc, p, make_info),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("scratch and values.in")));
  }
  {
    CubSortThunkEmitParams p = base_pairs_params();
    p.values->out.slice = BufferAllocation::Slice(&allocs[2], 0, kN * 4);
    EXPECT_THAT(EmitCudaCubSortThunkSequence(desc, p, make_info),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("scratch and values.out")));
  }

  // Empty and single-element sorts are rejected by the planner.
  {
    CubSortThunkEmitParams p = base_keys_params();
    const Shape empty_shape = ShapeUtil::MakeShape(F32, {0});
    p.keys.in = {BufferAllocation::Slice(&allocs[0], 0, 0), empty_shape};
    p.keys.out = {BufferAllocation::Slice(&allocs[1], 0, 0), empty_shape};
    EXPECT_THAT(EmitCudaCubSortThunkSequence(desc, p, make_info),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    CubSortThunkEmitParams p = base_keys_params();
    const Shape one_shape = ShapeUtil::MakeShape(F32, {1});
    p.keys.in = {BufferAllocation::Slice(&allocs[0], 0, 4), one_shape};
    p.keys.out = {BufferAllocation::Slice(&allocs[1], 0, 4), one_shape};
    EXPECT_THAT(EmitCudaCubSortThunkSequence(desc, p, make_info),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    CubSortThunkEmitParams p = base_keys_params();
    const Shape batched_one_shape = ShapeUtil::MakeShape(F32, {16, 1});
    p.batch_size = 16;
    p.keys.in = {BufferAllocation::Slice(&allocs[0], 0, 16 * 4),
                 batched_one_shape};
    p.keys.out = {BufferAllocation::Slice(&allocs[1], 0, 16 * 4),
                  batched_one_shape};
    EXPECT_THAT(EmitCudaCubSortThunkSequence(desc, p, make_info),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(CubSortThunkEmitterDevicelessTest,
     NativeCustomCallHandlerRegisteredForKeysAndPairs) {
  // Sort keys with a bitcasted input (underlying buffer is s32, logical HLO
  // operand shape is f32) to verify the handler pairs allocation slices with
  // logical HLO shapes rather than the defining buffer's shape.
  {
    constexpr absl::string_view kKeysHlo = R"hlo(
      HloModule m

      ENTRY main {
        %raw = s32[1024]{0} parameter(0)
        %keys = f32[1024]{0} bitcast(%raw)
        ROOT %sort = (f32[1024]{0}, u8[10240]{0}) custom-call(%keys),
            custom_call_target="xla.gpu.ext.cub_sort_keys",
            api_version=API_VERSION_TYPED_FFI,
            backend_config={descending = false, batch_size = 1 : i64}
      }
    )hlo";
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<NativeCustomCallHandlerTester> tester,
                         NativeCustomCallHandlerTester::Create(
                             kKeysHlo, {.gpu_model = GpuModel::H100_SXM}));
    ASSERT_OK_AND_ASSIGN(ThunkSequence thunks, tester->EmitThunks());
    ASSERT_EQ(thunks.size(), 1);
    const auto* single_tile_thunk =
        dynamic_cast<const CustomKernelThunk*>(thunks[0].get());
    ASSERT_NE(single_tile_thunk, nullptr);
    EXPECT_EQ(single_tile_thunk->custom_kernel_name(),
              "cub_sort_single_tile_f32_asc");
  }

  // Sort pairs (Onesweep multi-thunk sequence).
  {
    constexpr absl::string_view kPairsHlo = R"hlo(
      HloModule m

      ENTRY main {
        %keys = f32[65536]{0} parameter(0)
        %values = u32[65536]{0} parameter(1)
        ROOT %sort = (f32[65536]{0}, u32[65536]{0}, u8[1048576]{0})
            custom-call(%keys, %values),
            custom_call_target="xla.gpu.ext.cub_sort_pairs",
            api_version=API_VERSION_TYPED_FFI,
            backend_config={descending = true, batch_size = 1 : i64}
      }
    )hlo";
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<NativeCustomCallHandlerTester> tester,
                         NativeCustomCallHandlerTester::Create(
                             kPairsHlo, {.gpu_model = GpuModel::H100_SXM}));
    ASSERT_OK_AND_ASSIGN(ThunkSequence thunks, tester->EmitThunks());
    ASSERT_EQ(thunks.size(), 6);
    std::vector<absl::string_view> kernel_names;
    kernel_names.reserve(thunks.size());
    for (const std::unique_ptr<Thunk>& thunk : thunks) {
      const auto* kernel_thunk =
          dynamic_cast<const CustomKernelThunk*>(thunk.get());
      ASSERT_NE(kernel_thunk, nullptr);
      kernel_names.push_back(kernel_thunk->custom_kernel_name());
    }
    EXPECT_THAT(kernel_names, ElementsAre("cub_sort_histogram_f32_b32_desc",
                                          "cub_sort_exclusive_sum",
                                          "cub_sort_onesweep_f32_b32_desc",
                                          "cub_sort_onesweep_f32_b32_desc",
                                          "cub_sort_onesweep_f32_b32_desc",
                                          "cub_sort_onesweep_f32_b32_desc"));
  }

  // Batched segmented sort (`batch_size > 1`).
  {
    constexpr absl::string_view kSegmentedHlo = R"hlo(
      HloModule m

      ENTRY main {
        %keys = f32[4,8000]{1,0} parameter(0)
        ROOT %sort = (f32[4,8000]{1,0}, u8[262144]{0}) custom-call(%keys),
            custom_call_target="xla.gpu.ext.cub_sort_keys",
            api_version=API_VERSION_TYPED_FFI,
            backend_config={descending = false, batch_size = 4 : i64}
      }
    )hlo";
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<NativeCustomCallHandlerTester> tester,
                         NativeCustomCallHandlerTester::Create(
                             kSegmentedHlo, {.gpu_model = GpuModel::H100_SXM}));
    ASSERT_OK_AND_ASSIGN(ThunkSequence thunks, tester->EmitThunks());
    ASSERT_GE(thunks.size(), 2);
    std::vector<absl::string_view> kernel_names;
    kernel_names.reserve(thunks.size());
    for (const std::unique_ptr<Thunk>& thunk : thunks) {
      const auto* kernel_thunk =
          dynamic_cast<const CustomKernelThunk*>(thunk.get());
      ASSERT_NE(kernel_thunk, nullptr);
      kernel_names.push_back(kernel_thunk->custom_kernel_name());
    }
    EXPECT_THAT(kernel_names, Each(AnyOf("cub_sort_segmented_primary_f32_asc",
                                         "cub_sort_segmented_alt_f32_asc")));
  }
}

TEST(CubSortThunkEmitterDevicelessTest,
     NativeCustomCallRejectsInvalidHloAndNonCuda) {
  // Non-CUDA target device.
  {
    constexpr absl::string_view kKeysHlo = R"hlo(
      HloModule m

      ENTRY main {
        %keys = f32[1024]{0} parameter(0)
        ROOT %sort = (f32[1024]{0}, u8[10240]{0}) custom-call(%keys),
            custom_call_target="xla.gpu.ext.cub_sort_keys",
            api_version=API_VERSION_TYPED_FFI,
            backend_config={descending = false, batch_size = 1 : i64}
      }
    )hlo";
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<NativeCustomCallHandlerTester> tester,
                         NativeCustomCallHandlerTester::Create(
                             kKeysHlo, {.gpu_model = GpuModel::MI200}));
    EXPECT_THAT(tester->EmitThunks(),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         HasSubstr("requires a CUDA device description")));
  }

  // Unexpected custom_call_target invoked directly.
  {
    constexpr absl::string_view kWrongTargetHlo = R"hlo(
      HloModule m

      ENTRY main {
        %keys = f32[1024]{0} parameter(0)
        ROOT %sort = (f32[1024]{0}, u8[10240]{0}) custom-call(%keys),
            custom_call_target="xla.gpu.ext.not_cub_sort",
            api_version=API_VERSION_TYPED_FFI,
            backend_config={descending = false, batch_size = 1 : i64}
      }
    )hlo";
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<NativeCustomCallHandlerTester> tester,
        NativeCustomCallHandlerTester::Create(
            kWrongTargetHlo, {.gpu_model = GpuModel::H100_SXM}));
    EXPECT_THAT(tester->EmitThunksWith(EmitCudaCubSortNativeCustomCall),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Unexpected custom call target")));
  }

  // Wrong operand count (2 operands for keys-only target).
  {
    constexpr absl::string_view kBadOperandsHlo = R"hlo(
      HloModule m

      ENTRY main {
        %keys = f32[1024]{0} parameter(0)
        %vals = u32[1024]{0} parameter(1)
        ROOT %sort = (f32[1024]{0}, u8[10240]{0}) custom-call(%keys, %vals),
            custom_call_target="xla.gpu.ext.cub_sort_keys",
            api_version=API_VERSION_TYPED_FFI,
            backend_config={descending = false, batch_size = 1 : i64}
      }
    )hlo";
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<NativeCustomCallHandlerTester> tester,
        NativeCustomCallHandlerTester::Create(
            kBadOperandsHlo, {.gpu_model = GpuModel::H100_SXM}));
    EXPECT_THAT(tester->EmitThunks(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Unexpected operand count")));
  }

  // Wrong result tuple arity (3-tuple for keys-only target).
  {
    constexpr absl::string_view kBadResultArityHlo = R"hlo(
      HloModule m

      ENTRY main {
        %keys = f32[1024]{0} parameter(0)
        ROOT %sort = (f32[1024]{0}, u32[1024]{0}, u8[10240]{0})
            custom-call(%keys),
            custom_call_target="xla.gpu.ext.cub_sort_keys",
            api_version=API_VERSION_TYPED_FFI,
            backend_config={descending = false, batch_size = 1 : i64}
      }
    )hlo";
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<NativeCustomCallHandlerTester> tester,
        NativeCustomCallHandlerTester::Create(
            kBadResultArityHlo, {.gpu_model = GpuModel::H100_SXM}));
    EXPECT_THAT(tester->EmitThunks(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Unexpected result shape")));
  }

  // Missing FFI attributes (`descending` / `batch_size`).
  {
    constexpr absl::string_view kMissingDescendingHlo = R"hlo(
      HloModule m

      ENTRY main {
        %keys = f32[1024]{0} parameter(0)
        ROOT %sort = (f32[1024]{0}, u8[10240]{0}) custom-call(%keys),
            custom_call_target="xla.gpu.ext.cub_sort_keys",
            api_version=API_VERSION_TYPED_FFI,
            backend_config={batch_size = 1 : i64}
      }
    )hlo";
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<NativeCustomCallHandlerTester> tester,
        NativeCustomCallHandlerTester::Create(
            kMissingDescendingHlo, {.gpu_model = GpuModel::H100_SXM}));
    EXPECT_THAT(tester->EmitThunks(), Not(IsOk()));
  }
  {
    constexpr absl::string_view kMissingBatchSizeHlo = R"hlo(
      HloModule m

      ENTRY main {
        %keys = f32[1024]{0} parameter(0)
        ROOT %sort = (f32[1024]{0}, u8[10240]{0}) custom-call(%keys),
            custom_call_target="xla.gpu.ext.cub_sort_keys",
            api_version=API_VERSION_TYPED_FFI,
            backend_config={descending = false}
      }
    )hlo";
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<NativeCustomCallHandlerTester> tester,
        NativeCustomCallHandlerTester::Create(
            kMissingBatchSizeHlo, {.gpu_model = GpuModel::H100_SXM}));
    EXPECT_THAT(tester->EmitThunks(), Not(IsOk()));
  }

  // Mismatched key element types.
  {
    constexpr absl::string_view kBadKeyTypeHlo = R"hlo(
      HloModule m

      ENTRY main {
        %keys = f32[1024]{0} parameter(0)
        ROOT %sort = (u32[1024]{0}, u8[10240]{0}) custom-call(%keys),
            custom_call_target="xla.gpu.ext.cub_sort_keys",
            api_version=API_VERSION_TYPED_FFI,
            backend_config={descending = false, batch_size = 1 : i64}
      }
    )hlo";
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<NativeCustomCallHandlerTester> tester,
        NativeCustomCallHandlerTester::Create(
            kBadKeyTypeHlo, {.gpu_model = GpuModel::H100_SXM}));
    EXPECT_THAT(tester->EmitThunks(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Mismatched key element types")));
  }

  // Mismatched value element types.
  {
    constexpr absl::string_view kBadValueTypeHlo = R"hlo(
      HloModule m

      ENTRY main {
        %keys = f32[1024]{0} parameter(0)
        %vals = u32[1024]{0} parameter(1)
        ROOT %sort = (f32[1024]{0}, u64[1024]{0}, u8[10240]{0})
            custom-call(%keys, %vals),
            custom_call_target="xla.gpu.ext.cub_sort_pairs",
            api_version=API_VERSION_TYPED_FFI,
            backend_config={descending = false, batch_size = 1 : i64}
      }
    )hlo";
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<NativeCustomCallHandlerTester> tester,
        NativeCustomCallHandlerTester::Create(
            kBadValueTypeHlo, {.gpu_model = GpuModel::H100_SXM}));
    EXPECT_THAT(tester->EmitThunks(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Mismatched value element types")));
  }

  // Non-U8 scratch buffer.
  {
    constexpr absl::string_view kBadScratchHlo = R"hlo(
      HloModule m

      ENTRY main {
        %keys = f32[1024]{0} parameter(0)
        ROOT %sort = (f32[1024]{0}, f32[1024]{0}) custom-call(%keys),
            custom_call_target="xla.gpu.ext.cub_sort_keys",
            api_version=API_VERSION_TYPED_FFI,
            backend_config={descending = false, batch_size = 1 : i64}
      }
    )hlo";
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<NativeCustomCallHandlerTester> tester,
        NativeCustomCallHandlerTester::Create(
            kBadScratchHlo, {.gpu_model = GpuModel::H100_SXM}));
    EXPECT_THAT(tester->EmitThunks(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("U8 scratch buffer")));
  }

  // Non-dim-0-major physical layout.
  {
    constexpr absl::string_view kColMajorHlo = R"hlo(
      HloModule m

      ENTRY main {
        %keys = f32[4,1024]{0,1} parameter(0)
        ROOT %sort = (f32[4,1024]{0,1}, u8[10240]{0}) custom-call(%keys),
            custom_call_target="xla.gpu.ext.cub_sort_keys",
            api_version=API_VERSION_TYPED_FFI,
            backend_config={descending = false, batch_size = 4 : i64}
      }
    )hlo";
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<NativeCustomCallHandlerTester> tester,
                         NativeCustomCallHandlerTester::Create(
                             kColMajorHlo, {.gpu_model = GpuModel::H100_SXM}));
    EXPECT_THAT(
        tester->EmitThunks(),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("dim-0-major")));
  }
}

}  // namespace
}  // namespace xla::gpu
