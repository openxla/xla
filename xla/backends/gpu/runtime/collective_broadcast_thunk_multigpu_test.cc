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

// Multi-GPU integration tests for CollectiveBroadcastThunk through NCCL.
// Requires at least kNumDevices GPUs. Command-buffer tests additionally require
// CUDA 12.9+ driver/toolkit for CreateChildCommand / UpdateChildCommand support.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/runtime/collective_broadcast_thunk.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/scratch_memory_requests.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/future.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

static constexpr int kNumDevices = 2;
static constexpr int64_t kLength = 4;
static constexpr int64_t kByteLength = sizeof(float) * kLength;

se::StreamExecutor* GetGpuExecutor(int ordinal) {
  auto* platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()).value();
  return platform->ExecutorForDevice(ordinal).value();
}

static bool IsAtLeastCuda12900(const se::StreamExecutor* executor) {
  const auto& desc = executor->GetDeviceDescription();
  const auto* cuda_cc = desc.gpu_compute_capability().cuda_compute_capability();
  if (cuda_cc == nullptr) {
    return false;
  }
  return std::min(desc.driver_version(), desc.compile_time_toolkit_version()) >=
         se::SemanticVersion(12, 9, 0);
}

static bool HasEnoughGpus() {
  auto platform = se::PlatformManager::PlatformWithName(se::GpuPlatformName());
  if (!platform.ok()) {
    return false;
  }
  return (*platform)->VisibleDeviceCount() >= kNumDevices;
}

static CollectiveConfig MakeCollectiveBroadcastConfig() {
  CollectiveConfig config;
  config.operand_element_type = {F32};
  config.group_mode = COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;

  ReplicaGroup replica_group;
  for (int i = 0; i < kNumDevices; ++i) {
    replica_group.add_replica_ids(i);
  }
  config.replica_groups = {replica_group};
  return config;
}

static CollectiveBroadcastThunk MakeThunk(const BufferAllocation& alloc_src,
                                          const BufferAllocation& alloc_dst) {
  ShapedSlice src_slice{BufferAllocation::Slice(&alloc_src, 0, kByteLength),
                        ShapeUtil::MakeShape(F32, {kLength})};
  ShapedSlice dst_slice{BufferAllocation::Slice(&alloc_dst, 0, kByteLength),
                        ShapeUtil::MakeShape(F32, {kLength})};
  CollectiveThunk::Buffer buffer{.element_count = kLength,
                                 .source_buffer = src_slice,
                                 .destination_buffer = dst_slice,
                                 .source_memory_space = 0,
                                 .destination_memory_space = 0};
  return CollectiveBroadcastThunk(Thunk::ThunkInfo(),
                                  MakeCollectiveBroadcastConfig(), {buffer});
}

struct DeviceTestSlot {
  se::StreamExecutor* executor = nullptr;
  std::unique_ptr<se::Stream> stream;
  std::unique_ptr<se::StreamExecutorAddressAllocator> allocator;

  se::DeviceAddressBase src1, dst1;
  se::DeviceAddressBase src2, dst2;

  GpuExecutableRunOptions gpu_run_options;
  ServiceExecutableRunOptions run_options;
  std::optional<CollectiveParams> collective_params;
  CollectiveCliques collective_cliques;

  CommandStateManager state_manager;
  std::unique_ptr<se::CommandBuffer> command_buffer;
  const se::CommandBuffer::Command* cmd = nullptr;
};

static std::vector<se::DeviceAddressBase> MakeDeviceAddresses(
    se::DeviceAddressBase src, se::DeviceAddressBase dst) {
  return {src, dst};
}

static std::vector<float> SourceValues(int device_ordinal, int phase) {
  std::vector<float> values(kLength);
  for (int i = 0; i < kLength; ++i) {
    values[i] = static_cast<float>(phase * 100 + device_ordinal * 10 + i);
  }
  return values;
}

static absl::Status FillSourceBuffer(se::Stream& stream,
                                     se::DeviceAddressBase buf,
                                     int device_ordinal, int phase) {
  std::vector<float> data = SourceValues(device_ordinal, phase);
  RETURN_IF_ERROR(stream.Memcpy(&buf, data.data(), kByteLength));
  return stream.BlockHostUntilDone();
}

static absl::Status FillDestinationBuffer(se::Stream& stream,
                                          se::DeviceAddressBase buf,
                                          float value) {
  std::vector<float> data(kLength, value);
  RETURN_IF_ERROR(stream.Memcpy(&buf, data.data(), kByteLength));
  return stream.BlockHostUntilDone();
}

static absl::StatusOr<std::vector<float>> ReadDeviceBuffer(
    se::Stream& stream, se::DeviceAddressBase buf) {
  std::vector<float> data(kLength);
  RETURN_IF_ERROR(stream.Memcpy(data.data(), buf, kByteLength));
  RETURN_IF_ERROR(stream.BlockHostUntilDone());
  return data;
}

static absl::Status VerifyOutput(se::Stream& stream, se::DeviceAddressBase dst,
                                 int device_ordinal, int phase) {
  ASSIGN_OR_RETURN(std::vector<float> output, ReadDeviceBuffer(stream, dst));
  std::vector<float> expected = SourceValues(/*device_ordinal=*/0, phase);
  for (int i = 0; i < kLength; ++i) {
    if (output[i] != expected[i]) {
      return absl::InternalError(
          absl::StrFormat("device %d output[%d] = %g, expected %g",
                          device_ordinal, i, output[i], expected[i]));
    }
  }
  return absl::OkStatus();
}

static absl::Status SetupDeviceSlot(int device_ordinal, DeviceTestSlot& slot,
                                    CollectiveBroadcastThunk& thunk,
                                    const DeviceAssignment& device_assignment) {
  slot.executor = GetGpuExecutor(device_ordinal);
  ASSIGN_OR_RETURN(slot.stream, slot.executor->CreateStream());
  slot.allocator =
      std::make_unique<se::StreamExecutorAddressAllocator>(slot.executor);

  slot.src1 = slot.executor->AllocateArray<float>(kLength, 0);
  slot.dst1 = slot.executor->AllocateArray<float>(kLength, 0);
  slot.src2 = slot.executor->AllocateArray<float>(kLength, 0);
  slot.dst2 = slot.executor->AllocateArray<float>(kLength, 0);

  GpuExecutableRunOptions::DeviceIdMap id_map;
  for (int i = 0; i < kNumDevices; ++i) {
    id_map[LocalDeviceId(i)] = GlobalDeviceId(i);
  }
  slot.gpu_run_options.set_gpu_global_device_ids(std::move(id_map));
  slot.run_options.mutable_run_options()->set_stream(slot.stream.get());
  slot.run_options.mutable_run_options()->set_device_assignment(
      &device_assignment);
  slot.run_options.mutable_run_options()->set_gpu_executable_run_options(
      &slot.gpu_run_options);
  slot.run_options.mutable_run_options()->set_local_device_count(kNumDevices);

  ASSIGN_OR_RETURN(
      CollectiveParams params,
      CollectiveParams::Create(slot.run_options, /*async_streams=*/{},
                               LocalDeviceId(device_ordinal)));

  std::vector<se::DeviceAddressBase> addresses =
      MakeDeviceAddresses(slot.src1, slot.dst1);
  BufferAllocations allocations(addresses, slot.executor->device_ordinal(),
                                slot.allocator.get());
  CollectiveCliqueRequests clique_requests;
  CollectiveMemoryRequests memory_requests(allocations);
  ScratchMemoryRequests scratch_requests;
  Thunk::PrepareParams prepare_params{&params,          &clique_requests,
                                      &memory_requests, &scratch_requests,
                                      slot.executor,    &allocations};
  RETURN_IF_ERROR(thunk.Prepare(prepare_params));

  ASSIGN_OR_RETURN(slot.collective_cliques,
                   AcquireCollectiveCliques(params, clique_requests));

  Thunk::InitializeParams init_params;
  init_params.executor = slot.executor;
  init_params.stream = slot.stream.get();
  init_params.command_buffer_trace_stream = slot.stream.get();
  init_params.buffer_allocations = &allocations;
  init_params.collective_params = &params;
  init_params.collective_cliques = &slot.collective_cliques;
  init_params.local_device_count = kNumDevices;
  RETURN_IF_ERROR(thunk.Initialize(init_params));

  slot.collective_params = std::move(params);
  return absl::OkStatus();
}

static absl::Status RunExecuteOnStreamPhase(DeviceTestSlot& slot,
                                            CollectiveBroadcastThunk& thunk,
                                            int device_ordinal, int phase) {
  RETURN_IF_ERROR(
      FillSourceBuffer(*slot.stream, slot.src1, device_ordinal, phase));
  RETURN_IF_ERROR(FillDestinationBuffer(*slot.stream, slot.dst1, -1.0f));

  std::vector<se::DeviceAddressBase> addresses =
      MakeDeviceAddresses(slot.src1, slot.dst1);
  BufferAllocations allocations(addresses, slot.executor->device_ordinal(),
                                slot.allocator.get());
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      slot.run_options, allocations, slot.stream.get(),
      /*command_buffer_trace_stream=*/slot.stream.get(),
      &*slot.collective_params, &slot.collective_cliques,
      /*collective_memory=*/nullptr);

  RETURN_IF_ERROR(thunk.ExecuteOnStream(execute_params));
  RETURN_IF_ERROR(slot.stream->BlockHostUntilDone());
  return VerifyOutput(*slot.stream, slot.dst1, device_ordinal, phase);
}

static absl::Status RunCreatePhase(DeviceTestSlot& slot,
                                   CollectiveBroadcastThunk& thunk,
                                   int device_ordinal, int phase) {
  RETURN_IF_ERROR(
      FillSourceBuffer(*slot.stream, slot.src1, device_ordinal, phase));
  RETURN_IF_ERROR(FillDestinationBuffer(*slot.stream, slot.dst1, -1.0f));

  std::vector<se::DeviceAddressBase> addresses =
      MakeDeviceAddresses(slot.src1, slot.dst1);
  BufferAllocations allocations(addresses, slot.executor->device_ordinal(),
                                slot.allocator.get());
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      slot.run_options, allocations, slot.stream.get(),
      /*command_buffer_trace_stream=*/slot.stream.get(),
      &*slot.collective_params, &slot.collective_cliques,
      /*collective_memory=*/nullptr);

  // Warm up NCCL outside stream capture. Reset destination buffers afterward so
  // correctness is verified from command-buffer execution, not from warm-up.
  RETURN_IF_ERROR(thunk.ExecuteOnStream(execute_params));
  RETURN_IF_ERROR(slot.stream->BlockHostUntilDone());
  RETURN_IF_ERROR(FillDestinationBuffer(*slot.stream, slot.dst1, -1.0f));

  ASSIGN_OR_RETURN(slot.command_buffer, slot.executor->CreateCommandBuffer(
                                            se::CommandBuffer::Mode::kPrimary));

  Command::RecordParams record_params = {slot.state_manager};
  ASSIGN_OR_RETURN(slot.cmd,
                   thunk.Record(execute_params, record_params,
                                Command::RecordCreate{/*dependencies=*/{}},
                                slot.command_buffer.get()));
  if (slot.cmd == nullptr) {
    return absl::InternalError("Record(create) returned null command node");
  }

  RETURN_IF_ERROR(slot.command_buffer->Finalize());
  RETURN_IF_ERROR(slot.command_buffer->Submit(slot.stream.get()));
  RETURN_IF_ERROR(slot.stream->BlockHostUntilDone());
  return VerifyOutput(*slot.stream, slot.dst1, device_ordinal, phase);
}

static absl::Status RunUpdatePhase(DeviceTestSlot& slot,
                                   CollectiveBroadcastThunk& thunk,
                                   int device_ordinal, int phase) {
  RETURN_IF_ERROR(
      FillSourceBuffer(*slot.stream, slot.src2, device_ordinal, phase));
  RETURN_IF_ERROR(FillDestinationBuffer(*slot.stream, slot.dst2, -1.0f));

  std::vector<se::DeviceAddressBase> addresses =
      MakeDeviceAddresses(slot.src2, slot.dst2);
  BufferAllocations allocations(addresses, slot.executor->device_ordinal(),
                                slot.allocator.get());
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      slot.run_options, allocations, slot.stream.get(),
      /*command_buffer_trace_stream=*/slot.stream.get(),
      &*slot.collective_params, &slot.collective_cliques,
      /*collective_memory=*/nullptr);

  std::vector<BufferAllocation::Index> updated_allocs = {0, 1};
  Command::RecordParams record_params = {slot.state_manager,
                                         std::move(updated_allocs)};

  RETURN_IF_ERROR(slot.command_buffer->Update());
  ASSIGN_OR_RETURN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk.Record(execute_params, record_params,
                   Command::RecordUpdate{slot.cmd}, slot.command_buffer.get()));
  if (updated_cmd != slot.cmd) {
    return absl::InternalError(
        "Update returned a different command node; expected reuse");
  }

  RETURN_IF_ERROR(slot.command_buffer->Finalize());
  RETURN_IF_ERROR(slot.command_buffer->Submit(slot.stream.get()));
  RETURN_IF_ERROR(slot.stream->BlockHostUntilDone());
  return VerifyOutput(*slot.stream, slot.dst2, device_ordinal, phase);
}

static absl::StatusOr<int> SetupAndExecute(
    int d, DeviceTestSlot* slots, CollectiveBroadcastThunk* thunk,
    const DeviceAssignment* device_assignment) {
  RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], *thunk, *device_assignment));
  RETURN_IF_ERROR(RunExecuteOnStreamPhase(slots[d], *thunk, d, /*phase=*/1));
  return d;
}

static absl::StatusOr<int> SetupAndCreate(
    int d, DeviceTestSlot* slots, CollectiveBroadcastThunk* thunk,
    const DeviceAssignment* device_assignment) {
  RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], *thunk, *device_assignment));
  RETURN_IF_ERROR(RunCreatePhase(slots[d], *thunk, d, /*phase=*/2));
  return d;
}

static absl::StatusOr<int> RunUpdate(int d, DeviceTestSlot* slots,
                                     CollectiveBroadcastThunk* thunk) {
  RETURN_IF_ERROR(RunUpdatePhase(slots[d], *thunk, d, /*phase=*/3));
  return d;
}

static DeviceAssignment MakeDeviceAssignment() {
  DeviceAssignment device_assignment(kNumDevices, /*computation_count=*/1);
  for (int i = 0; i < kNumDevices; ++i) {
    device_assignment(i, 0) = i;
  }
  return device_assignment;
}

TEST(CollectiveBroadcastThunkMultiGpuTest, ExecuteOnStream) {
  if (!HasEnoughGpus()) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment();
  BufferAllocation alloc_src(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, kByteLength, /*color=*/0);
  CollectiveBroadcastThunk thunk = MakeThunk(alloc_src, alloc_dst);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  tsl::thread::ThreadPool pool(tsl::Env::Default(),
                               "collective_broadcast_execute", kNumDevices);
  std::vector<tsl::Future<int>> futures(kNumDevices);
  for (int d = 0; d < kNumDevices; ++d) {
    futures[d] = tsl::MakeFutureOn<int>(
        *pool.AsExecutor(), [d, &slots, &thunk, &device_assignment]() {
          return SetupAndExecute(d, slots.data(), &thunk, &device_assignment);
        });
  }
  ASSERT_OK(JoinFutures<int>(futures).Await());
}

TEST(CollectiveBroadcastThunkMultiGpuTest, RecordCommandBufferCreate) {
  if (!HasEnoughGpus()) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment();
  BufferAllocation alloc_src(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, kByteLength, /*color=*/0);
  CollectiveBroadcastThunk thunk = MakeThunk(alloc_src, alloc_dst);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  tsl::thread::ThreadPool pool(tsl::Env::Default(),
                               "collective_broadcast_create", kNumDevices);
  std::vector<tsl::Future<int>> futures(kNumDevices);
  for (int d = 0; d < kNumDevices; ++d) {
    futures[d] = tsl::MakeFutureOn<int>(
        *pool.AsExecutor(), [d, &slots, &thunk, &device_assignment]() {
          return SetupAndCreate(d, slots.data(), &thunk, &device_assignment);
        });
  }
  ASSERT_OK(JoinFutures<int>(futures).Await());
}

TEST(CollectiveBroadcastThunkMultiGpuTest, RecordCommandBufferUpdate) {
  if (!HasEnoughGpus()) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment();
  BufferAllocation alloc_src(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, kByteLength, /*color=*/0);
  CollectiveBroadcastThunk thunk = MakeThunk(alloc_src, alloc_dst);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(),
                                 "collective_broadcast_create", kNumDevices);
    std::vector<tsl::Future<int>> futures(kNumDevices);
    for (int d = 0; d < kNumDevices; ++d) {
      futures[d] = tsl::MakeFutureOn<int>(
          *pool.AsExecutor(), [d, &slots, &thunk, &device_assignment]() {
            return SetupAndCreate(d, slots.data(), &thunk, &device_assignment);
          });
    }
    ASSERT_OK(JoinFutures<int>(futures).Await());
  }

  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(),
                                 "collective_broadcast_update", kNumDevices);
    std::vector<tsl::Future<int>> futures(kNumDevices);
    for (int d = 0; d < kNumDevices; ++d) {
      futures[d] = tsl::MakeFutureOn<int>(
          *pool.AsExecutor(),
          [d, &slots, &thunk]() { return RunUpdate(d, slots.data(), &thunk); });
    }
    ASSERT_OK(JoinFutures<int>(futures).Await());
  }
}

}  // namespace
}  // namespace xla::gpu
