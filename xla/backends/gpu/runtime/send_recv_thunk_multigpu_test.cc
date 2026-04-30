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

// Multi-GPU integration tests for SendThunk and RecvThunk.
// Requires two GPUs. Command-buffer tests also require CUDA 12.9+ for child
// command create/update support.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/recv_thunk.h"
#include "xla/backends/gpu/runtime/scratch_memory_requests.h"
#include "xla/backends/gpu/runtime/send_thunk.h"
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

static P2PConfig MakeSendRecvConfig() {
  ReplicaGroup replica_group;
  for (int i = 0; i < kNumDevices; ++i) {
    replica_group.add_replica_ids(i);
  }

  P2PConfig config;
  config.config.operand_element_type = {F32};
  config.config.replica_groups = {replica_group};
  config.config.group_mode = COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
  config.id_to_source_target[0].target = 1;
  config.id_to_source_target[1].source = 0;
  return config;
}

static CollectiveThunk::Buffer MakeBuffer(const BufferAllocation& allocation) {
  ShapedSlice slice{BufferAllocation::Slice(&allocation, 0, kByteLength),
                    ShapeUtil::MakeShape(F32, {kLength})};
  return CollectiveThunk::Buffer{.element_count = kLength,
                                 .source_buffer = slice,
                                 .destination_buffer = slice,
                                 .source_memory_space = 0,
                                 .destination_memory_space = 0};
}

static SendThunk MakeSendThunk(const BufferAllocation& send_allocation) {
  return SendThunk(Thunk::ThunkInfo(), MakeSendRecvConfig(),
                   MakeBuffer(send_allocation), "send");
}

static RecvThunk MakeRecvThunk(const BufferAllocation& recv_allocation) {
  return RecvThunk(Thunk::ThunkInfo(), MakeSendRecvConfig(),
                   MakeBuffer(recv_allocation), "recv");
}

struct DeviceTestSlot {
  se::StreamExecutor* executor = nullptr;
  std::unique_ptr<se::Stream> stream;
  std::unique_ptr<se::StreamExecutorAddressAllocator> allocator;

  se::DeviceAddressBase send_src1, recv_dst1;
  se::DeviceAddressBase send_src2, recv_dst2;

  GpuExecutableRunOptions gpu_run_options;
  ServiceExecutableRunOptions run_options;
  std::optional<CollectiveParams> collective_params;
  CollectiveCliques collective_cliques;

  CommandStateManager state_manager;
  std::unique_ptr<se::CommandBuffer> command_buffer;
  const se::CommandBuffer::Command* send_cmd = nullptr;
  const se::CommandBuffer::Command* recv_cmd = nullptr;
};

static std::vector<float> SourceValues(int device_ordinal, int phase) {
  std::vector<float> data(kLength);
  for (int i = 0; i < kLength; ++i) {
    data[i] = static_cast<float>(phase * 100 + device_ordinal * 10 + i);
  }
  return data;
}

static std::vector<float> ExpectedRecvValues(int device_ordinal, int phase) {
  if (device_ordinal == 1) {
    return SourceValues(/*device_ordinal=*/0, phase);
  }
  return std::vector<float>(kLength, 0.0f);
}

static absl::Status WriteDeviceBuffer(se::Stream& stream,
                                      se::DeviceAddressBase buffer,
                                      const std::vector<float>& data) {
  RETURN_IF_ERROR(stream.Memcpy(&buffer, data.data(), kByteLength));
  return stream.BlockHostUntilDone();
}

static absl::Status FillDeviceBuffer(se::Stream& stream,
                                     se::DeviceAddressBase buffer,
                                     float value) {
  return WriteDeviceBuffer(stream, buffer, std::vector<float>(kLength, value));
}

static absl::StatusOr<std::vector<float>> ReadDeviceBuffer(
    se::Stream& stream, se::DeviceAddressBase buffer) {
  std::vector<float> data(kLength);
  RETURN_IF_ERROR(stream.Memcpy(data.data(), buffer, kByteLength));
  RETURN_IF_ERROR(stream.BlockHostUntilDone());
  return data;
}

static absl::Status PreparePhaseInputs(DeviceTestSlot& slot, int device_ordinal,
                                       int phase,
                                       se::DeviceAddressBase send_src,
                                       se::DeviceAddressBase recv_dst) {
  RETURN_IF_ERROR(WriteDeviceBuffer(*slot.stream, send_src,
                                    SourceValues(device_ordinal, phase)));
  return FillDeviceBuffer(*slot.stream, recv_dst, -1.0f);
}

static absl::Status VerifyRecvOutput(DeviceTestSlot& slot, int device_ordinal,
                                     int phase,
                                     se::DeviceAddressBase recv_dst) {
  ASSIGN_OR_RETURN(std::vector<float> output,
                   ReadDeviceBuffer(*slot.stream, recv_dst));
  std::vector<float> expected = ExpectedRecvValues(device_ordinal, phase);
  for (int i = 0; i < kLength; ++i) {
    if (output[i] != expected[i]) {
      return absl::InternalError(
          absl::StrFormat("device %d output[%d] = %g, expected %g",
                          device_ordinal, i, output[i], expected[i]));
    }
  }
  return absl::OkStatus();
}

static Thunk::ExecuteParams MakeExecuteParams(DeviceTestSlot& slot,
                                              BufferAllocations& allocations) {
  return Thunk::ExecuteParams::Create(
      slot.run_options, allocations, slot.stream.get(),
      /*command_buffer_trace_stream=*/slot.stream.get(),
      &*slot.collective_params, &slot.collective_cliques,
      /*collective_memory=*/nullptr);
}

static absl::Status SetupDeviceSlot(int device_ordinal, DeviceTestSlot& slot,
                                    SendThunk& send_thunk,
                                    RecvThunk& recv_thunk,
                                    const DeviceAssignment& device_assignment) {
  slot.executor = GetGpuExecutor(device_ordinal);
  ASSIGN_OR_RETURN(slot.stream, slot.executor->CreateStream());
  slot.allocator =
      std::make_unique<se::StreamExecutorAddressAllocator>(slot.executor);

  slot.send_src1 = slot.executor->AllocateArray<float>(kLength,
                                                       /*memory_space=*/0);
  slot.recv_dst1 = slot.executor->AllocateArray<float>(kLength, 0);
  slot.send_src2 = slot.executor->AllocateArray<float>(kLength, 0);
  slot.recv_dst2 = slot.executor->AllocateArray<float>(kLength, 0);

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

  ASSIGN_OR_RETURN(
      CollectiveParams params,
      CollectiveParams::Create(slot.run_options, /*async_streams=*/{},
                               LocalDeviceId(device_ordinal)));

  BufferAllocations allocations1({slot.send_src1, slot.recv_dst1}, 0,
                                 slot.allocator.get());
  CollectiveCliqueRequests clique_requests;
  CollectiveMemoryRequests memory_requests(allocations1);
  ScratchMemoryRequests scratch_requests;
  Thunk::PrepareParams prepare_params{&params,          &clique_requests,
                                      &memory_requests, &scratch_requests,
                                      slot.executor,    &allocations1};
  RETURN_IF_ERROR(send_thunk.Prepare(prepare_params));
  RETURN_IF_ERROR(recv_thunk.Prepare(prepare_params));

  ASSIGN_OR_RETURN(slot.collective_cliques,
                   AcquireCollectiveCliques(params, clique_requests));

  Thunk::InitializeParams init_params;
  init_params.executor = slot.executor;
  init_params.stream = slot.stream.get();
  init_params.buffer_allocations = &allocations1;
  init_params.collective_params = &params;
  init_params.collective_cliques = &slot.collective_cliques;
  RETURN_IF_ERROR(send_thunk.Initialize(init_params));
  RETURN_IF_ERROR(recv_thunk.Initialize(init_params));

  slot.collective_params = std::move(params);
  return absl::OkStatus();
}

static absl::Status ExecuteSendRecv(DeviceTestSlot& slot, SendThunk& send_thunk,
                                    RecvThunk& recv_thunk,
                                    BufferAllocations& allocations) {
  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);
  RETURN_IF_ERROR(send_thunk.ExecuteOnStream(execute_params));
  RETURN_IF_ERROR(recv_thunk.ExecuteOnStream(execute_params));
  return slot.stream->BlockHostUntilDone();
}

static absl::Status RunExecuteOnStreamPhase(int device_ordinal,
                                            DeviceTestSlot& slot,
                                            SendThunk& send_thunk,
                                            RecvThunk& recv_thunk) {
  constexpr int kPhase = 1;
  RETURN_IF_ERROR(PreparePhaseInputs(slot, device_ordinal, kPhase,
                                     slot.send_src1, slot.recv_dst1));
  BufferAllocations allocations({slot.send_src1, slot.recv_dst1}, 0,
                                slot.allocator.get());
  RETURN_IF_ERROR(ExecuteSendRecv(slot, send_thunk, recv_thunk, allocations));
  return VerifyRecvOutput(slot, device_ordinal, kPhase, slot.recv_dst1);
}

static absl::Status RunCreatePhase(int device_ordinal, DeviceTestSlot& slot,
                                   SendThunk& send_thunk,
                                   RecvThunk& recv_thunk) {
  constexpr int kPhase = 1;
  RETURN_IF_ERROR(PreparePhaseInputs(slot, device_ordinal, kPhase,
                                     slot.send_src1, slot.recv_dst1));

  BufferAllocations allocations({slot.send_src1, slot.recv_dst1}, 0,
                                slot.allocator.get());
  RETURN_IF_ERROR(ExecuteSendRecv(slot, send_thunk, recv_thunk, allocations));
  RETURN_IF_ERROR(FillDeviceBuffer(*slot.stream, slot.recv_dst1, -1.0f));

  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);
  ASSIGN_OR_RETURN(slot.command_buffer, slot.executor->CreateCommandBuffer(
                                            se::CommandBuffer::Mode::kPrimary));

  Command::RecordParams record_params = {slot.state_manager};
  ASSIGN_OR_RETURN(slot.send_cmd,
                   send_thunk.Record(execute_params, record_params,
                                     Command::RecordCreate{/*dependencies=*/{}},
                                     slot.command_buffer.get()));

  std::vector<const se::CommandBuffer::Command*> dependencies;
  if (slot.send_cmd != nullptr) {
    dependencies.push_back(slot.send_cmd);
  }
  ASSIGN_OR_RETURN(slot.recv_cmd,
                   recv_thunk.Record(execute_params, record_params,
                                     Command::RecordCreate{dependencies},
                                     slot.command_buffer.get()));
  if (slot.recv_cmd == nullptr) {
    return absl::InternalError("RecvThunk returned null command node");
  }

  RETURN_IF_ERROR(slot.command_buffer->Finalize());
  RETURN_IF_ERROR(slot.command_buffer->Submit(slot.stream.get()));
  RETURN_IF_ERROR(slot.stream->BlockHostUntilDone());
  return VerifyRecvOutput(slot, device_ordinal, kPhase, slot.recv_dst1);
}

static absl::Status RunUpdatePhase(int device_ordinal, DeviceTestSlot& slot,
                                   SendThunk& send_thunk,
                                   RecvThunk& recv_thunk) {
  constexpr int kPhase = 2;
  RETURN_IF_ERROR(PreparePhaseInputs(slot, device_ordinal, kPhase,
                                     slot.send_src2, slot.recv_dst2));

  BufferAllocations allocations({slot.send_src2, slot.recv_dst2}, 0,
                                slot.allocator.get());
  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);
  Command::RecordParams record_params = {
      slot.state_manager,
      /*updated_allocs=*/std::vector<BufferAllocation::Index>{0, 1}};

  RETURN_IF_ERROR(slot.command_buffer->Update());
  ASSIGN_OR_RETURN(const se::CommandBuffer::Command* updated_send_cmd,
                   send_thunk.Record(execute_params, record_params,
                                     Command::RecordUpdate{slot.send_cmd},
                                     slot.command_buffer.get()));
  if (updated_send_cmd != slot.send_cmd) {
    return absl::InternalError("SendThunk update returned a new command node");
  }

  ASSIGN_OR_RETURN(const se::CommandBuffer::Command* updated_recv_cmd,
                   recv_thunk.Record(execute_params, record_params,
                                     Command::RecordUpdate{slot.recv_cmd},
                                     slot.command_buffer.get()));
  if (updated_recv_cmd != slot.recv_cmd) {
    return absl::InternalError("RecvThunk update returned a new command node");
  }

  RETURN_IF_ERROR(slot.command_buffer->Finalize());
  RETURN_IF_ERROR(slot.command_buffer->Submit(slot.stream.get()));
  RETURN_IF_ERROR(slot.stream->BlockHostUntilDone());
  return VerifyRecvOutput(slot, device_ordinal, kPhase, slot.recv_dst2);
}

static absl::StatusOr<int> SetupAndExecute(
    int d, DeviceTestSlot* slots, SendThunk* send_thunk, RecvThunk* recv_thunk,
    const DeviceAssignment* device_assignment) {
  RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], *send_thunk, *recv_thunk,
                                  *device_assignment));
  RETURN_IF_ERROR(
      RunExecuteOnStreamPhase(d, slots[d], *send_thunk, *recv_thunk));
  return d;
}

static absl::StatusOr<int> SetupAndCreate(
    int d, DeviceTestSlot* slots, SendThunk* send_thunk, RecvThunk* recv_thunk,
    const DeviceAssignment* device_assignment) {
  RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], *send_thunk, *recv_thunk,
                                  *device_assignment));
  RETURN_IF_ERROR(RunCreatePhase(d, slots[d], *send_thunk, *recv_thunk));
  return d;
}

static absl::StatusOr<int> RunUpdate(int d, DeviceTestSlot* slots,
                                     SendThunk* send_thunk,
                                     RecvThunk* recv_thunk) {
  RETURN_IF_ERROR(RunUpdatePhase(d, slots[d], *send_thunk, *recv_thunk));
  return d;
}

static DeviceAssignment MakeDeviceAssignment() {
  DeviceAssignment device_assignment(kNumDevices, /*computation_count=*/1);
  for (int i = 0; i < kNumDevices; ++i) {
    device_assignment(i, 0) = i;
  }
  return device_assignment;
}

TEST(SendRecvThunkMultiGpuTest, ExecuteOnStream) {
  if (!HasEnoughGpus()) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment();
  BufferAllocation send_alloc(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation recv_alloc(/*index=*/1, kByteLength, /*color=*/0);
  SendThunk send_thunk = MakeSendThunk(send_alloc);
  RecvThunk recv_thunk = MakeRecvThunk(recv_alloc);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "sendrecv_execute",
                               kNumDevices);
  std::vector<tsl::Future<int>> futures(kNumDevices);
  for (int d = 0; d < kNumDevices; ++d) {
    futures[d] = tsl::MakeFutureOn<int>(
        *pool.AsExecutor(),
        [d, &slots, &send_thunk, &recv_thunk, &device_assignment]() {
          return SetupAndExecute(d, slots.data(), &send_thunk, &recv_thunk,
                                 &device_assignment);
        });
  }
  ASSERT_OK(JoinFutures<int>(futures).Await());
}

TEST(SendRecvThunkMultiGpuTest, RecordCommandBufferCreate) {
  if (!HasEnoughGpus()) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment();
  BufferAllocation send_alloc(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation recv_alloc(/*index=*/1, kByteLength, /*color=*/0);
  SendThunk send_thunk = MakeSendThunk(send_alloc);
  RecvThunk recv_thunk = MakeRecvThunk(recv_alloc);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "sendrecv_create",
                               kNumDevices);
  std::vector<tsl::Future<int>> futures(kNumDevices);
  for (int d = 0; d < kNumDevices; ++d) {
    futures[d] = tsl::MakeFutureOn<int>(
        *pool.AsExecutor(),
        [d, &slots, &send_thunk, &recv_thunk, &device_assignment]() {
          return SetupAndCreate(d, slots.data(), &send_thunk, &recv_thunk,
                                &device_assignment);
        });
  }
  ASSERT_OK(JoinFutures<int>(futures).Await());
}

TEST(SendRecvThunkMultiGpuTest, RecordCommandBufferUpdate) {
  if (!HasEnoughGpus()) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment();
  BufferAllocation send_alloc(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation recv_alloc(/*index=*/1, kByteLength, /*color=*/0);
  SendThunk send_thunk = MakeSendThunk(send_alloc);
  RecvThunk recv_thunk = MakeRecvThunk(recv_alloc);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "sendrecv_create",
                                 kNumDevices);
    std::vector<tsl::Future<int>> futures(kNumDevices);
    for (int d = 0; d < kNumDevices; ++d) {
      futures[d] = tsl::MakeFutureOn<int>(
          *pool.AsExecutor(),
          [d, &slots, &send_thunk, &recv_thunk, &device_assignment]() {
            return SetupAndCreate(d, slots.data(), &send_thunk, &recv_thunk,
                                  &device_assignment);
          });
    }
    ASSERT_OK(JoinFutures<int>(futures).Await());
  }

  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "sendrecv_update",
                                 kNumDevices);
    std::vector<tsl::Future<int>> futures(kNumDevices);
    for (int d = 0; d < kNumDevices; ++d) {
      futures[d] = tsl::MakeFutureOn<int>(
          *pool.AsExecutor(), [d, &slots, &send_thunk, &recv_thunk]() {
            return RunUpdate(d, slots.data(), &send_thunk, &recv_thunk);
          });
    }
    ASSERT_OK(JoinFutures<int>(futures).Await());
  }
}

}  // namespace
}  // namespace xla::gpu
