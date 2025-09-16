/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/all_reduce_thunk.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/service/rendezvous.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

// Contains the values that are passed between host threads with rendezvous.
struct RendezvousValue {
  RankId rank;
  se::DeviceMemoryBase input_buffer;
  se::Event* start_event;
  se::Event* end_event;

  bool operator<(const RendezvousValue& other) const {
    return rank < other.rank;
  }
};

// Executes the rendezvous before the kernel start.
// Inserts CUDA events into the stream to ensure that all devices have reached
// the start event before the kernel starts.
absl::StatusOr<std::shared_ptr<std::vector<RendezvousValue>>>
RendezvousBeforeKernelStart(const GpuCliqueKey& clique_key, RankId rank,
                            int64_t num_ranks,
                            const se::DeviceMemoryBase& input_buffer,
                            se::Stream& stream, se::Event* start_event,
                            se::Event* end_event) {
  RendezvousValue rendezvous_value;
  rendezvous_value.rank = rank;
  rendezvous_value.input_buffer = input_buffer;
  rendezvous_value.start_event = start_event;
  rendezvous_value.end_event = end_event;

  // Record that this device has started executing the kernel. We do
  // this before the rendezvous to make sure that RecordEvent is called before
  // WaitFor on another stream.
  TF_RETURN_IF_ERROR(stream.RecordEvent(start_event));

  auto rendezvous_fn = [](absl::Span<const RendezvousValue* const> values) {
    std::vector<RendezvousValue> values_copy;
    for (const auto& value : values) {
      values_copy.push_back(*value);
    }
    // Sort to make sure that values are in the same order as the devices are
    // ordered in the communicator.
    absl::c_sort(values_copy);
    return values_copy;
  };

  std::string start_rendezvous_key =
      absl::StrFormat("start one-shot all-reduce for rank %d, clique %s",
                      rank.value(), clique_key.ToString());
  VLOG(1) << "##### " << __func__ << start_rendezvous_key;
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<std::vector<RendezvousValue>> rendezvous_values,
      Rendezvous<std::vector<RendezvousValue>>(
          /*name=*/start_rendezvous_key, /*key=*/clique_key,
          /*value=*/rendezvous_value, /*num_threads=*/num_ranks,
          rendezvous_fn));

  // Wait for all devices to reach the start event. This indicates that all
  // output buffers are ready for transfer.
  for (auto& value : *rendezvous_values) {
    VLOG(1) << "##### " << __func__ << " Wait on stream ";
    TF_RETURN_IF_ERROR(stream.WaitFor(value.start_event));
  }

  VLOG(1) << "##### " << __func__ << " Done";
  return rendezvous_values;
}

// Executes the rendezvous after the kernel finish. Waits for all devices to
// reach the end event.
absl::Status RendezvousAfterKernelFinish(
    const GpuCliqueKey& clique_key, RankId rank, int64_t num_ranks,
    se::Stream& stream, se::Event* end_event,
    const std::shared_ptr<std::vector<RendezvousValue>>& rendezvous_values) {
  // Record that this device has finished executing the kernel.
  VLOG(1) << "##### " << __func__ << " Start ";
  TF_RETURN_IF_ERROR(stream.RecordEvent(end_event));

  // Do another rendezvous to make sure that we call RecordEvent for end_event
  // before WaitFor on another stream.
  std::string finish_rendezvous_key =
      absl::StrFormat("finish one-shot all-reduce for rank %d, clique %s",
                      rank.value(), clique_key.ToString());
  VLOG(1) << "##### " << __func__ << " " << finish_rendezvous_key;
  TF_RETURN_IF_ERROR(Rendezvous(/*name=*/finish_rendezvous_key,
                                /*key=*/clique_key,
                                /*num_threads=*/num_ranks));

  // Wait for all devices to reach the end event. This indicates that all
  // updates from other devices have arrived.
  for (auto& value : *rendezvous_values) {
    TF_RETURN_IF_ERROR(stream.WaitFor(value.end_event));
  }

  VLOG(1) << "##### " << __func__ << " Done";
  return absl::OkStatus();
}

absl::Status CheckImplementableInst(const HloInstruction* inst,
                                    Thunk::Kind reduction_op) {
  for (HloInstruction* operand : inst->operands()) {
    TF_RETURN_IF_ERROR(IsValidOperand(operand->shape(), reduction_op));
  }

  if (!MatchReductionComputation(inst->called_computations().front())
           .has_value()) {
    return absl::UnimplementedError("Unrecognized reduction computation");
  }

  return absl::OkStatus();
}

template <typename HloInstType>
AllReduceConfig GetAllReduceConfigInst(HloInstType* inst) {
  std::optional<ReductionKind> reduction_kind =
      MatchReductionComputation(inst->called_computations().front());
  CHECK(reduction_kind.has_value());

  AllReduceConfig config;
  config.config = GetCollectiveConfig(inst, inst->use_global_device_ids());
  config.reduction_kind = *reduction_kind;
  return config;
}

template <typename HloInstType>
CollectiveOpGroupMode GetGroupModeInst(HloInstType* inst) {
  return GetAllReduceConfigInst(inst).config.group_mode;
}

}  // namespace

absl::Status RunAllReduce(GpuCollectives* collectives,
                          ReductionKind reduction_kind,
                          std::vector<DeviceBufferPair>& buffers,
                          se::Stream& stream, Communicator* comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(1) << "##### " << __func__ << " Start ";
  VLOG(3) << "Performing all-reduce from device ordinal: " << device_ordinal;
  TF_RETURN_IF_ERROR(
      MaybeRegisterBuffers(collectives, stream.parent(), buffers, comm));

  TF_RETURN_IF_ERROR(collectives->GroupStart());
  for (DeviceBufferPair& buffer : buffers) {
    TF_RETURN_IF_ERROR(comm->AllReduce(
        buffer.source_buffer, buffer.destination_buffer, buffer.element_type,
        buffer.element_count, reduction_kind, GpuCollectives::On(stream)));
  }

  auto result = collectives->GroupEnd();
  VLOG(1) << "##### " << __func__ << " Done " << result.ToString();
  return result;
}

AllReduceReduceScatterThunkBase::AllReduceReduceScatterThunkBase(
    Thunk::Kind kind, ThunkInfo thunk_info, AllReduceConfig config,
    std::vector<Buffer> buffers, bool is_sync)
    : CollectiveThunk(kind, thunk_info, is_sync, AsyncStreamKind::kCollective),
      config_(std::move(config)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

AllReduceStartThunk::AllReduceStartThunk(ThunkInfo thunk_info,
                                         const HloAllReduceInstruction* inst,
                                         std::vector<Buffer> buffers,
                                         bool p2p_memcpy_enabled)
    : AllReduceReduceScatterThunkBase(
          Thunk::kAllReduceStart, thunk_info, GetAllReduceConfigInst(inst),
          std::move(buffers), IsGPUSyncCollective(*inst)),
      collective_kernel_thunk_{
          thunk_info,
          config_.config,
          config_.reduction_kind,
          IsAsync(),
          buffers_,
          /*is_collective_kernel_enabled=*/
          inst->GetModule()
              ->config()
              .debug_options()
              .xla_gpu_unsupported_use_all_reduce_one_shot_kernel(),
      } {}

absl::Status AllReduceStartThunk::CheckImplementable(
    const HloAllReduceInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<AllReduceStartThunk>(
      CheckImplementableInst(inst, Thunk::kAllReduceStart), inst, replica_count,
      partition_count);
}

CollectiveOpGroupMode AllReduceStartThunk::GetGroupMode(
    const HloAllReduceInstruction* inst) {
  return GetGroupModeInst(inst);
}

absl::Status AllReduceStartThunk::Initialize(const InitializeParams& params) {
  TF_RETURN_IF_ERROR(CollectiveThunk::Initialize(params));
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, config()));
  TF_ASSIGN_OR_RETURN(bool use_collective_kernel,
                      collective_kernel_thunk_.IsSupported(
                          clique_key, params.collective_cliques));
  if (use_collective_kernel) {
    TF_RETURN_IF_ERROR(collective_kernel_thunk_.Initialize(params));
  }
  return absl::OkStatus();
}

absl::Status AllReduceStartThunk::RunCollective(
    const ExecuteParams& params, se::Stream& stream,
    CommunicatorHandle comm_handle) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));

  TF_ASSIGN_OR_RETURN(bool use_collective_kernel,
                      collective_kernel_thunk_.IsSupported(
                          comm_handle.clique_key, params.collective_cliques));

  if (use_collective_kernel) {
    VLOG(3) << "Custom AllReduce is used for intra-node communication";
    return collective_kernel_thunk_.ExecuteOnStream(params);
  }

  VLOG(3) << "NCCL AllReduce is used for intra-node communication";
  return RunAllReduce(collectives, config_.reduction_kind, device_buffers, stream,
                      comm_handle.comm);
}

ReduceScatterStartThunk::ReduceScatterStartThunk(
    ThunkInfo thunk_info, const HloReduceScatterInstruction* inst,
    std::vector<Buffer> buffers, bool p2p_memcpy_enabled)
    : AllReduceReduceScatterThunkBase(
          Thunk::kReduceScatterStart, thunk_info, GetAllReduceConfigInst(inst),
          std::move(buffers), IsGPUSyncCollective(*inst)) {}

/*static*/ absl::Status ReduceScatterStartThunk::CheckImplementable(
    const HloReduceScatterInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<ReduceScatterStartThunk>(
      CheckImplementableInst(inst, Thunk::kReduceScatterStart), inst,
      replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode ReduceScatterStartThunk::GetGroupMode(
    const HloReduceScatterInstruction* inst) {
  return GetGroupModeInst(inst);
}

absl::Status ReduceScatterStartThunk::RunCollective(
    const ExecuteParams& params, se::Stream& stream,
    CommunicatorHandle comm_handle) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));
  return RunReduceScatter(collectives, config_.reduction_kind, device_buffers,
                          stream, comm_handle.comm);
}

absl::Status RunReduceScatter(GpuCollectives* collectives,
                              ReductionKind reduction_kind,
                              std::vector<DeviceBufferPair>& buffers,
                              se::Stream& stream, Communicator* comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing reduce-scatter from device ordinal: "
          << device_ordinal;
  TF_RETURN_IF_ERROR(
      MaybeRegisterBuffers(collectives, stream.parent(), buffers, comm));

  TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm->NumRanks());

  TF_RETURN_IF_ERROR(collectives->GroupStart());

  for (DeviceBufferPair& buffer : buffers) {
    // buffer.element_count is the source buffers element count. For
    // ncclReduceScatter, we need the destination buffers element count.
    TF_RET_CHECK(buffer.element_count % num_ranks == 0)
        << "Source buffer was not an exact multiple of the number of "
           "participants.";

    TF_RETURN_IF_ERROR(comm->ReduceScatter(
        buffer.source_buffer, buffer.destination_buffer, buffer.element_type,
        buffer.element_count / num_ranks, reduction_kind,
        GpuCollectives::On(stream)));
  }

  return collectives->GroupEnd();
}

}  // namespace gpu
}  // namespace xla
