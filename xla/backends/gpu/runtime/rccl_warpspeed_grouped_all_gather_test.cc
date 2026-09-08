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

// ROCm integration test for XLA's grouped multi-buffer AllGather through RCCL.

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk_multigpu_test_utils.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

constexpr int kNumDevices = 8;
constexpr int kNumBuffers = 2;
constexpr int64_t kElementsPerRank = 524'288;

using Word = uint16_t;
constexpr int64_t kElementBytes = sizeof(Word);
constexpr int64_t kSourceBytes = kElementsPerRank * kElementBytes;
constexpr int64_t kGatheredElements = kElementsPerRank * kNumDevices;
constexpr int64_t kGatheredBytes = kGatheredElements * kElementBytes;

constexpr int64_t kGuardBytes = 1 << 20;
constexpr int64_t kGuardElements = kGuardBytes / kElementBytes;
constexpr Word kGuardWord = 0xbaa5;
constexpr Word kUnwrittenWord = 0xdead;

constexpr int64_t SourceAllocationIndex(int buffer_index) {
  return 2 * buffer_index;
}

constexpr int64_t DestinationAllocationIndex(int buffer_index) {
  return 2 * buffer_index + 1;
}

constexpr int64_t AllocationBytes(int64_t payload_bytes) {
  return kGuardBytes + payload_bytes + kGuardBytes;
}

Word SourceWord(int buffer_index, int source_rank, int64_t index) {
  uint32_t mixed = static_cast<uint32_t>(index) * 2654435761u;
  mixed ^= static_cast<uint32_t>(buffer_index + 1) * 0x9e37u;
  mixed ^= static_cast<uint32_t>(source_rank + 1) * 0x85ebu;
  mixed ^= mixed >> 15;
  Word word = static_cast<Word>(mixed & 0xffffu);
  if (word == kGuardWord || word == kUnwrittenWord) {
    word ^= 0x00ffu;
  }
  return word;
}

std::vector<int64_t> DeviceBufferSizes() {
  std::vector<int64_t> sizes;
  sizes.reserve(2 * kNumBuffers);
  for (int buffer_index = 0; buffer_index < kNumBuffers; ++buffer_index) {
    sizes.push_back(AllocationBytes(kSourceBytes));
    sizes.push_back(AllocationBytes(kGatheredBytes));
  }
  return sizes;
}

CollectiveConfig MakeAllGatherConfig() {
  ReplicaGroup replica_group;
  for (int device = 0; device < kNumDevices; ++device) {
    replica_group.add_replica_ids(device);
  }

  CollectiveConfig config;
  config.operand_element_type.assign(kNumBuffers, BF16);
  config.replica_groups = {replica_group};
  config.group_mode = COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
  config.use_symmetric_buffer = false;
  return config;
}

AllGatherThunk MakeThunk(absl::Span<const BufferAllocation> allocations) {
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(kNumBuffers);
  for (int buffer_index = 0; buffer_index < kNumBuffers; ++buffer_index) {
    const BufferAllocation& source =
        allocations[SourceAllocationIndex(buffer_index)];
    const BufferAllocation& destination =
        allocations[DestinationAllocationIndex(buffer_index)];

    ShapedSlice source_slice{
        BufferAllocation::Slice(&source, kGuardBytes, kSourceBytes),
        ShapeUtil::MakeShape(BF16, {kElementsPerRank})};
    ShapedSlice destination_slice{
        BufferAllocation::Slice(&destination, kGuardBytes, kGatheredBytes),
        ShapeUtil::MakeShape(BF16, {kGatheredElements})};

    buffers.push_back(
        CollectiveThunk::Buffer{.element_count = kElementsPerRank,
                                .source_buffer = source_slice,
                                .destination_buffer = destination_slice,
                                .source_memory_space = 0,
                                .destination_memory_space = 0});
  }

  return AllGatherThunk(Thunk::ThunkInfo(), MakeAllGatherConfig(),
                        std::move(buffers));
}

absl::Status CopyToDevice(se::Stream& stream, se::DeviceAddressBase destination,
                          absl::Span<const Word> contents) {
  ABSL_RETURN_IF_ERROR(stream.Memcpy(&destination, contents.data(),
                                     contents.size() * sizeof(Word)));
  return stream.BlockHostUntilDone();
}

absl::StatusOr<std::vector<Word>> CopyFromDevice(se::Stream& stream,
                                                 se::DeviceAddressBase source) {
  std::vector<Word> contents(source.size() / sizeof(Word));
  ABSL_RETURN_IF_ERROR(stream.Memcpy(contents.data(), source, source.size()));
  ABSL_RETURN_IF_ERROR(stream.BlockHostUntilDone());
  return contents;
}

std::vector<Word> SourceImage(int buffer_index, int rank) {
  std::vector<Word> image(AllocationBytes(kSourceBytes) / sizeof(Word),
                          kGuardWord);
  for (int64_t i = 0; i < kElementsPerRank; ++i) {
    image[kGuardElements + i] = SourceWord(buffer_index, rank, i);
  }
  return image;
}

std::vector<Word> DestinationImage() {
  std::vector<Word> image(AllocationBytes(kGatheredBytes) / sizeof(Word),
                          kGuardWord);
  for (int64_t i = 0; i < kGatheredElements; ++i) {
    image[kGuardElements + i] = kUnwrittenWord;
  }
  return image;
}

absl::Status CheckGuards(absl::Span<const Word> image, int64_t payload_elements,
                         std::string_view buffer_name) {
  for (int64_t i = 0; i < kGuardElements; ++i) {
    if (image[i] != kGuardWord) {
      return absl::DataLossError(
          absl::StrFormat("%s: leading guard[%d] = 0x%04x, expected 0x%04x",
                          buffer_name, i, static_cast<unsigned int>(image[i]),
                          static_cast<unsigned int>(kGuardWord)));
    }
  }

  const int64_t trailing_guard = kGuardElements + payload_elements;
  for (int64_t i = 0; i < kGuardElements; ++i) {
    if (image[trailing_guard + i] != kGuardWord) {
      return absl::DataLossError(absl::StrFormat(
          "%s: trailing guard[%d] = 0x%04x, expected 0x%04x", buffer_name, i,
          static_cast<unsigned int>(image[trailing_guard + i]),
          static_cast<unsigned int>(kGuardWord)));
    }
  }
  return absl::OkStatus();
}

absl::Status PrepareBuffers(se::Stream& stream,
                            absl::Span<const se::DeviceAddressBase> allocations,
                            int rank) {
  for (int buffer_index = 0; buffer_index < kNumBuffers; ++buffer_index) {
    std::vector<Word> source = SourceImage(buffer_index, rank);
    ABSL_RETURN_IF_ERROR(CopyToDevice(
        stream, allocations[SourceAllocationIndex(buffer_index)], source));

    std::vector<Word> destination = DestinationImage();
    ABSL_RETURN_IF_ERROR(CopyToDevice(
        stream, allocations[DestinationAllocationIndex(buffer_index)],
        destination));
  }
  return absl::OkStatus();
}

absl::Status VerifySource(se::Stream& stream,
                          absl::Span<const se::DeviceAddressBase> allocations,
                          int buffer_index, int rank) {
  ABSL_ASSIGN_OR_RETURN(
      std::vector<Word> image,
      CopyFromDevice(stream, allocations[SourceAllocationIndex(buffer_index)]));
  const std::string name =
      absl::StrFormat("rank %d buffer %d source", rank, buffer_index);
  ABSL_RETURN_IF_ERROR(CheckGuards(image, kElementsPerRank, name));

  for (int64_t i = 0; i < kElementsPerRank; ++i) {
    const Word expected = SourceWord(buffer_index, rank, i);
    const Word actual = image[kGuardElements + i];
    if (actual != expected) {
      return absl::DataLossError(
          absl::StrFormat("%s payload[%d] = 0x%04x, expected 0x%04x", name, i,
                          static_cast<unsigned int>(actual),
                          static_cast<unsigned int>(expected)));
    }
  }
  return absl::OkStatus();
}

absl::Status VerifyDestination(
    se::Stream& stream, absl::Span<const se::DeviceAddressBase> allocations,
    int buffer_index, int destination_rank) {
  ABSL_ASSIGN_OR_RETURN(
      std::vector<Word> image,
      CopyFromDevice(stream,
                     allocations[DestinationAllocationIndex(buffer_index)]));
  const std::string name = absl::StrFormat("rank %d buffer %d destination",
                                           destination_rank, buffer_index);
  ABSL_RETURN_IF_ERROR(CheckGuards(image, kGatheredElements, name));

  for (int source_rank = 0; source_rank < kNumDevices; ++source_rank) {
    for (int64_t i = 0; i < kElementsPerRank; ++i) {
      const int64_t output_index = source_rank * kElementsPerRank + i;
      const Word expected = SourceWord(buffer_index, source_rank, i);
      const Word actual = image[kGuardElements + output_index];
      if (actual != expected) {
        return absl::DataLossError(absl::StrFormat(
            "%s payload[%d] = 0x%04x, expected 0x%04x from rank %d "
            "element %d",
            name, output_index, static_cast<unsigned int>(actual),
            static_cast<unsigned int>(expected), source_rank, i));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status RunAndVerify(CollectiveThunkMultiGpuTestState& state,
                          AllGatherThunk& thunk, int rank) {
  ABSL_RETURN_IF_ERROR(
      PrepareBuffers(*state.stream, state.create_buffers, rank));

  BufferAllocations allocations =
      MakeBufferAllocations(state, state.create_buffers);
  Thunk::ExecuteParams execute_params = MakeExecuteParams(state, allocations);
  ABSL_RETURN_IF_ERROR(ExecuteOnStreamAndBlock(thunk, execute_params));

  for (int buffer_index = 0; buffer_index < kNumBuffers; ++buffer_index) {
    ABSL_RETURN_IF_ERROR(VerifyDestination(*state.stream, state.create_buffers,
                                           buffer_index, rank));
    ABSL_RETURN_IF_ERROR(
        VerifySource(*state.stream, state.create_buffers, buffer_index, rank));
  }
  return absl::OkStatus();
}

TEST(RcclWarpSpeedGroupedAllGatherTest, TwoBuffersProduceExactResults) {
  ASSERT_TRUE(HasEnoughGpus(kNumDevices))
      << "Test requires at least " << kNumDevices << " visible GPUs";

  se::StreamExecutor* executor = GetGpuExecutor(0);
  ASSERT_NE(executor, nullptr);
  const auto* rocm_capability = executor->GetDeviceDescription()
                                    .gpu_compute_capability()
                                    .rocm_compute_capability();
  ASSERT_NE(rocm_capability, nullptr);
  ASSERT_EQ(rocm_capability->gfx_version(), "gfx950");
  ASSERT_GT(executor->GetDeviceDescription().core_count(), 128);

  // Populate the executor cache before the device threads start.
  for (int device = 1; device < kNumDevices; ++device) {
    ASSERT_NE(GetGpuExecutor(device), nullptr);
  }

  const std::vector<int64_t> buffer_sizes = DeviceBufferSizes();
  std::vector<BufferAllocation> buffer_allocations;
  buffer_allocations.reserve(buffer_sizes.size());
  for (int64_t index = 0; index < static_cast<int64_t>(buffer_sizes.size());
       ++index) {
    buffer_allocations.emplace_back(index, buffer_sizes[index], /*color=*/0);
  }

  AllGatherThunk thunk = MakeThunk(buffer_allocations);
  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  std::vector<CollectiveThunkMultiGpuTestState> states(kNumDevices);

  ASSERT_OK(RunOnDevices(kNumDevices, "rccl_warpspeed_grouped_all_gather",
                         [&](int device) -> absl::Status {
                           ABSL_RETURN_IF_ERROR(SetupCollectiveThunkDevice(
                               device, kNumDevices, buffer_sizes, thunk,
                               device_assignment, states[device]));
                           return RunAndVerify(states[device], thunk, device);
                         }));
}

}  // namespace
}  // namespace xla::gpu
