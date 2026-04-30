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

// Multi-GPU integration tests for RaggedAllToAllThunk command-buffer Record().
// Requires two GPUs and CUDA 12.9+ driver/toolkit for CreateChildCommand /
// UpdateChildCommand support.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"
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
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

static constexpr int kNumDevices = 2;
static constexpr int64_t kNumUpdates = 4;
static constexpr int64_t kNumInputRows = 4;
static constexpr int64_t kNumRowElements = 1;
static constexpr int64_t kNumElements = kNumInputRows * kNumRowElements;
static constexpr int64_t kNumBuffers = 6;

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

static RaggedAllToAllConfig MakeConfig() {
  ReplicaGroup replica_group;
  for (int i = 0; i < kNumDevices; ++i) {
    replica_group.add_replica_ids(i);
  }

  RaggedAllToAllConfig config;
  config.config.operand_element_type = {F32, F32, S64, S64, S64, S64};
  config.config.replica_groups = {replica_group};
  config.config.group_mode = COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
  config.num_total_updates = kNumUpdates;
  config.num_input_rows = kNumInputRows;
  config.num_row_elements = kNumRowElements;
  config.one_shot_kernel_enabled = true;
  return config;
}

static CollectiveThunk::Buffer MakeBuffer(const BufferAllocation& allocation,
                                          PrimitiveType element_type,
                                          int64_t element_count) {
  Shape shape = ShapeUtil::MakeShape(element_type, {element_count});
  ShapedSlice slice{BufferAllocation::Slice(
                        &allocation, 0, ShapeUtil::ByteSizeOfElements(shape)),
                    shape};
  return CollectiveThunk::Buffer{.element_count = element_count,
                                 .source_buffer = slice,
                                 .destination_buffer = slice,
                                 .source_memory_space = 0,
                                 .destination_memory_space = 0};
}

static std::vector<BufferAllocation> MakeBufferAllocations() {
  std::vector<BufferAllocation> allocations;
  allocations.reserve(kNumBuffers);
  allocations.emplace_back(/*index=*/0, sizeof(float) * kNumElements,
                           /*color=*/0);
  allocations.emplace_back(/*index=*/1, sizeof(float) * kNumElements,
                           /*color=*/0);
  for (int i = 2; i < kNumBuffers; ++i) {
    allocations.emplace_back(i, sizeof(int64_t) * kNumUpdates, /*color=*/0);
  }
  return allocations;
}

static RaggedAllToAllThunk MakeThunk(
    const std::vector<BufferAllocation>& allocations) {
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(kNumBuffers);
  buffers.push_back(MakeBuffer(allocations[0], F32, kNumElements));
  buffers.push_back(MakeBuffer(allocations[1], F32, kNumElements));
  for (int i = 2; i < kNumBuffers; ++i) {
    buffers.push_back(MakeBuffer(allocations[i], S64, kNumUpdates));
  }
  return RaggedAllToAllThunk(Thunk::ThunkInfo(), MakeConfig(),
                             std::move(buffers));
}

static std::vector<se::DeviceAddressBase> AllocateDeviceBuffers(
    se::StreamExecutor* executor) {
  std::vector<se::DeviceAddressBase> buffers;
  buffers.reserve(kNumBuffers);
  buffers.push_back(executor->AllocateArray<float>(kNumElements, 0));
  buffers.push_back(executor->AllocateArray<float>(kNumElements, 0));
  for (int i = 2; i < kNumBuffers; ++i) {
    buffers.push_back(executor->AllocateArray<int64_t>(kNumUpdates, 0));
  }
  return buffers;
}

static std::vector<float> InputValues(int device_ordinal, int phase) {
  std::vector<float> values(kNumElements);
  for (int i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(phase * 100 + device_ordinal * 10 + i);
  }
  return values;
}

static std::vector<int64_t> OutputOffsets(int device_ordinal) {
  int64_t base = 2 * device_ordinal;
  return {base, base + 1, base, base + 1};
}

static std::vector<float> ExpectedValues(int device_ordinal, int phase) {
  std::vector<float> rank0 = InputValues(/*device_ordinal=*/0, phase);
  std::vector<float> rank1 = InputValues(/*device_ordinal=*/1, phase);
  if (device_ordinal == 0) {
    return {rank0[0], rank0[1], rank1[0], rank1[1]};
  }
  return {rank0[2], rank0[3], rank1[2], rank1[3]};
}

static absl::Status WriteBuffer(se::Stream& stream,
                                se::DeviceAddressBase buffer,
                                const std::vector<float>& data) {
  RETURN_IF_ERROR(
      stream.Memcpy(&buffer, data.data(), data.size() * sizeof(float)));
  return absl::OkStatus();
}

static absl::Status WriteBuffer(se::Stream& stream,
                                se::DeviceAddressBase buffer,
                                const std::vector<int64_t>& data) {
  RETURN_IF_ERROR(
      stream.Memcpy(&buffer, data.data(), data.size() * sizeof(int64_t)));
  return absl::OkStatus();
}

static absl::Status PrepareInputs(
    se::Stream& stream, absl::Span<const se::DeviceAddressBase> buffers,
    int device_ordinal, int phase) {
  RETURN_IF_ERROR(
      WriteBuffer(stream, buffers[0], InputValues(device_ordinal, phase)));
  RETURN_IF_ERROR(
      WriteBuffer(stream, buffers[1], std::vector<float>(kNumElements, -1.0f)));
  RETURN_IF_ERROR(
      WriteBuffer(stream, buffers[2], std::vector<int64_t>{0, 1, 2, 3}));
  RETURN_IF_ERROR(
      WriteBuffer(stream, buffers[3], std::vector<int64_t>{1, 1, 1, 1}));
  RETURN_IF_ERROR(
      WriteBuffer(stream, buffers[4], OutputOffsets(device_ordinal)));
  RETURN_IF_ERROR(
      WriteBuffer(stream, buffers[5], std::vector<int64_t>{1, 1, 1, 1}));
  return stream.BlockHostUntilDone();
}

static absl::StatusOr<std::vector<float>> ReadOutput(
    se::Stream& stream, se::DeviceAddressBase buffer) {
  std::vector<float> output(kNumElements);
  RETURN_IF_ERROR(
      stream.Memcpy(output.data(), buffer, output.size() * sizeof(float)));
  RETURN_IF_ERROR(stream.BlockHostUntilDone());
  return output;
}

static absl::Status VerifyOutput(se::Stream& stream,
                                 se::DeviceAddressBase buffer,
                                 int device_ordinal, int phase) {
  ASSIGN_OR_RETURN(std::vector<float> output, ReadOutput(stream, buffer));
  std::vector<float> expected = ExpectedValues(device_ordinal, phase);
  for (int i = 0; i < output.size(); ++i) {
    if (output[i] != expected[i]) {
      return absl::InternalError(
          absl::StrFormat("device %d output[%d] = %g, expected %g",
                          device_ordinal, i, output[i], expected[i]));
    }
  }
  return absl::OkStatus();
}

struct DeviceTestSlot {
  se::StreamExecutor* executor = nullptr;
  std::unique_ptr<se::Stream> stream;
  std::unique_ptr<se::StreamExecutorAddressAllocator> allocator;
  std::vector<se::DeviceAddressBase> create_buffers;
  std::vector<se::DeviceAddressBase> update_buffers;

  GpuExecutableRunOptions gpu_run_options;
  ServiceExecutableRunOptions run_options;
  std::optional<CollectiveParams> collective_params;
  CollectiveCliques collective_cliques;

  CommandStateManager state_manager;
  std::unique_ptr<se::CommandBuffer> command_buffer;
  const se::CommandBuffer::Command* command = nullptr;
};

static absl::Status SetupDeviceSlot(int device_ordinal, DeviceTestSlot& slot,
                                    RaggedAllToAllThunk& thunk,
                                    const DeviceAssignment& device_assignment) {
  slot.executor = GetGpuExecutor(device_ordinal);
  ASSIGN_OR_RETURN(slot.stream, slot.executor->CreateStream());
  slot.allocator =
      std::make_unique<se::StreamExecutorAddressAllocator>(slot.executor);
  slot.create_buffers = AllocateDeviceBuffers(slot.executor);
  slot.update_buffers = AllocateDeviceBuffers(slot.executor);

  GpuExecutableRunOptions::DeviceIdMap id_map;
  for (int i = 0; i < kNumDevices; ++i) {
    id_map[LocalDeviceId(i)] = GlobalDeviceId(i);
  }
  slot.gpu_run_options.set_gpu_global_device_ids(std::move(id_map));
  slot.run_options.mutable_run_options()->set_stream(slot.stream.get());
  slot.run_options.mutable_run_options()->set_local_device_count(kNumDevices);
  slot.run_options.mutable_run_options()->set_device_assignment(
      &device_assignment);
  slot.run_options.mutable_run_options()->set_gpu_executable_run_options(
      &slot.gpu_run_options);

  ASSIGN_OR_RETURN(
      CollectiveParams params,
      CollectiveParams::Create(slot.run_options, /*async_streams=*/{},
                               LocalDeviceId(device_ordinal)));

  BufferAllocations allocations(slot.create_buffers, device_ordinal,
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
  init_params.buffer_allocations = &allocations;
  init_params.collective_params = &params;
  init_params.collective_cliques = &slot.collective_cliques;
  init_params.local_device_count = kNumDevices;
  RETURN_IF_ERROR(thunk.Initialize(init_params));

  slot.collective_params = std::move(params);
  return absl::OkStatus();
}

static absl::Status RunExecuteOnStreamPhase(DeviceTestSlot& slot,
                                            RaggedAllToAllThunk& thunk,
                                            int device_ordinal, int phase) {
  RETURN_IF_ERROR(
      PrepareInputs(*slot.stream, slot.create_buffers, device_ordinal, phase));

  BufferAllocations allocations(slot.create_buffers, device_ordinal,
                                slot.allocator.get());
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      slot.run_options, allocations, slot.stream.get(),
      /*command_buffer_trace_stream=*/slot.stream.get(),
      &*slot.collective_params, &slot.collective_cliques,
      /*collective_memory=*/nullptr);

  RETURN_IF_ERROR(thunk.ExecuteOnStream(execute_params));
  RETURN_IF_ERROR(slot.stream->BlockHostUntilDone());
  return VerifyOutput(*slot.stream, slot.create_buffers[1], device_ordinal,
                      phase);
}

static absl::Status RunCreatePhase(DeviceTestSlot& slot,
                                   RaggedAllToAllThunk& thunk,
                                   int device_ordinal, int phase) {
  RETURN_IF_ERROR(
      PrepareInputs(*slot.stream, slot.create_buffers, device_ordinal, phase));

  BufferAllocations allocations(slot.create_buffers, device_ordinal,
                                slot.allocator.get());
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      slot.run_options, allocations, slot.stream.get(),
      /*command_buffer_trace_stream=*/slot.stream.get(),
      &*slot.collective_params, &slot.collective_cliques,
      /*collective_memory=*/nullptr);

  RETURN_IF_ERROR(thunk.ExecuteOnStream(execute_params));
  RETURN_IF_ERROR(slot.stream->BlockHostUntilDone());
  RETURN_IF_ERROR(
      PrepareInputs(*slot.stream, slot.create_buffers, device_ordinal, phase));

  ASSIGN_OR_RETURN(slot.command_buffer, slot.executor->CreateCommandBuffer(
                                            se::CommandBuffer::Mode::kPrimary));

  Command::RecordParams record_params = {slot.state_manager};
  ASSIGN_OR_RETURN(slot.command,
                   thunk.Record(execute_params, record_params,
                                Command::RecordCreate{/*dependencies=*/{}},
                                slot.command_buffer.get()));
  if (slot.command == nullptr) {
    return absl::InternalError("Record(create) returned null command node");
  }

  RETURN_IF_ERROR(slot.command_buffer->Finalize());
  RETURN_IF_ERROR(slot.command_buffer->Submit(slot.stream.get()));
  RETURN_IF_ERROR(slot.stream->BlockHostUntilDone());
  return VerifyOutput(*slot.stream, slot.create_buffers[1], device_ordinal,
                      phase);
}

static absl::Status RunUpdatePhase(DeviceTestSlot& slot,
                                   RaggedAllToAllThunk& thunk,
                                   int device_ordinal, int phase) {
  RETURN_IF_ERROR(
      PrepareInputs(*slot.stream, slot.update_buffers, device_ordinal, phase));

  BufferAllocations allocations(slot.update_buffers, device_ordinal,
                                slot.allocator.get());
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      slot.run_options, allocations, slot.stream.get(),
      /*command_buffer_trace_stream=*/slot.stream.get(),
      &*slot.collective_params, &slot.collective_cliques,
      /*collective_memory=*/nullptr);

  std::vector<BufferAllocation::Index> updated_allocs;
  updated_allocs.reserve(kNumBuffers);
  for (int i = 0; i < kNumBuffers; ++i) {
    updated_allocs.push_back(i);
  }
  Command::RecordParams record_params = {slot.state_manager,
                                         std::move(updated_allocs)};

  RETURN_IF_ERROR(slot.command_buffer->Update());
  ASSIGN_OR_RETURN(const se::CommandBuffer::Command* updated_command,
                   thunk.Record(execute_params, record_params,
                                Command::RecordUpdate{slot.command},
                                slot.command_buffer.get()));
  if (updated_command != slot.command) {
    return absl::InternalError(
        "Update returned a different command node than create");
  }

  RETURN_IF_ERROR(slot.command_buffer->Finalize());
  RETURN_IF_ERROR(slot.command_buffer->Submit(slot.stream.get()));
  RETURN_IF_ERROR(slot.stream->BlockHostUntilDone());
  return VerifyOutput(*slot.stream, slot.update_buffers[1], device_ordinal,
                      phase);
}

static absl::StatusOr<int> SetupAndExecute(
    int d, DeviceTestSlot* slots, RaggedAllToAllThunk* thunk,
    const DeviceAssignment* device_assignment) {
  RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], *thunk, *device_assignment));
  RETURN_IF_ERROR(RunExecuteOnStreamPhase(slots[d], *thunk, d, /*phase=*/1));
  return d;
}

static absl::StatusOr<int> SetupAndCreate(
    int d, DeviceTestSlot* slots, RaggedAllToAllThunk* thunk,
    const DeviceAssignment* device_assignment) {
  RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], *thunk, *device_assignment));
  RETURN_IF_ERROR(RunCreatePhase(slots[d], *thunk, d, /*phase=*/1));
  return d;
}

static absl::StatusOr<int> RunUpdate(int d, DeviceTestSlot* slots,
                                     RaggedAllToAllThunk* thunk) {
  RETURN_IF_ERROR(RunUpdatePhase(slots[d], *thunk, d, /*phase=*/2));
  return d;
}

TEST(RaggedAllToAllThunkMultiGpuTest, ExecuteOnStream) {
  if (!HasEnoughGpus()) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }

  DeviceAssignment device_assignment(kNumDevices, /*computation_count=*/1);
  for (int i = 0; i < kNumDevices; ++i) {
    device_assignment(i, 0) = i;
  }
  std::vector<BufferAllocation> buffer_allocations = MakeBufferAllocations();
  RaggedAllToAllThunk thunk = MakeThunk(buffer_allocations);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "ragged_execute",
                               kNumDevices);
  std::vector<tsl::Future<int>> futures(kNumDevices);
  for (int d = 0; d < kNumDevices; ++d) {
    futures[d] = tsl::MakeFutureOn<int>(
        *pool.AsExecutor(), [d, &slots, &thunk, &device_assignment]() {
          return SetupAndExecute(d, slots.data(), &thunk, &device_assignment);
        });
  }
  ASSERT_OK(JoinFutures<int>(futures).Await());
}

TEST(RaggedAllToAllThunkMultiGpuTest, RecordCommandBufferCreate) {
  if (!HasEnoughGpus()) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment(kNumDevices, /*computation_count=*/1);
  for (int i = 0; i < kNumDevices; ++i) {
    device_assignment(i, 0) = i;
  }
  std::vector<BufferAllocation> buffer_allocations = MakeBufferAllocations();
  RaggedAllToAllThunk thunk = MakeThunk(buffer_allocations);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "ragged_create",
                               kNumDevices);
  std::vector<tsl::Future<int>> futures(kNumDevices);
  for (int d = 0; d < kNumDevices; ++d) {
    futures[d] = tsl::MakeFutureOn<int>(
        *pool.AsExecutor(), [d, &slots, &thunk, &device_assignment]() {
          return SetupAndCreate(d, slots.data(), &thunk, &device_assignment);
        });
  }
  ASSERT_OK(JoinFutures<int>(futures).Await());
}

TEST(RaggedAllToAllThunkMultiGpuTest, RecordCommandBufferUpdate) {
  if (!HasEnoughGpus()) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment(kNumDevices, /*computation_count=*/1);
  for (int i = 0; i < kNumDevices; ++i) {
    device_assignment(i, 0) = i;
  }
  std::vector<BufferAllocation> buffer_allocations = MakeBufferAllocations();
  RaggedAllToAllThunk thunk = MakeThunk(buffer_allocations);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "ragged_create",
                                 kNumDevices);
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
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "ragged_update",
                                 kNumDevices);
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
