/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/command_buffer_cmd_emitter.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/service/gpu/runtime/command_buffer_cmd.h"
#include "xla/service/gpu/runtime/conditional_thunk.h"
#include "xla/service/gpu/runtime/copy_thunk.h"
#include "xla/service/gpu/runtime/cudnn_thunk.h"
#include "xla/service/gpu/runtime/custom_call_thunk.h"
#include "xla/service/gpu/runtime/gemm_thunk.h"
#include "xla/service/gpu/runtime/kernel_thunk.h"
#include "xla/service/gpu/runtime/memset_thunk.h"
#include "xla/service/gpu/runtime/nccl_all_gather_thunk.h"
#include "xla/service/gpu/runtime/nccl_all_reduce_thunk.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"
#include "xla/service/gpu/runtime/replica_id_thunk.h"
#include "xla/service/gpu/runtime/sequential_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/gpu/runtime/while_thunk.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

// Appends command(s) converted from `thunk` to `cmd_sequence`.
static absl::Status AppendCommands(
    CommandBufferCmdSequence& cmd_sequence, AllocationCmdMap& alloc_to_cmd,
    const Thunk& thunk,
    CommandBufferCmdSequence::SynchronizationMode synchronization_mode);

// Appends command(s) converted from `sequence` to `cmd_sequence`.
static absl::Status AppendCommands(
    CommandBufferCmdSequence& cmd_sequence, AllocationCmdMap& alloc_to_cmd,
    const ThunkSequence& sequence,
    CommandBufferCmdSequence::SynchronizationMode synchronization_mode);

//===----------------------------------------------------------------------===//
// Conversions from Thunk to Command
//===----------------------------------------------------------------------===//

using Command = std::unique_ptr<CommandBufferCmd>;

static auto ArgsAccess(const std::vector<bool>& written) {
  absl::InlinedVector<CommandBufferCmd::MemoryAccess, 4> args_access;
  args_access.reserve(written.size());
  for (bool w : written) {
    args_access.push_back(w ? CommandBufferCmd::MemoryAccess::kWrite
                            : CommandBufferCmd::MemoryAccess::kRead);
  }
  return args_access;
}

static absl::StatusOr<Command> Convert(const KernelThunk& thunk,
                                       AllocationCmdMap& alloc_to_cmd) {
  auto cmd = std::make_unique<LaunchCmd>(
      thunk.execution_stream_id(), thunk.kernel_name(), thunk.arguments(),
      ArgsAccess(thunk.written()), thunk.launch_dimensions(),
      thunk.shmem_bytes());
  for (auto slice : thunk.arguments()) {
    TF_RETURN_IF_ERROR(alloc_to_cmd.InsertBufferUse(slice.index(), cmd.get()));
  }
  return cmd;
}

static absl::StatusOr<Command> Convert(const CustomKernelThunk& thunk,
                                       AllocationCmdMap& alloc_to_cmd) {
  auto cmd = std::make_unique<CustomKernelLaunchCmd>(
      thunk.execution_stream_id(), thunk.arguments(),
      ArgsAccess(thunk.written()), thunk.custom_kernel());
  for (auto slice : thunk.arguments()) {
    TF_RETURN_IF_ERROR(alloc_to_cmd.InsertBufferUse(slice.index(), cmd.get()));
  }
  return cmd;
}

static absl::StatusOr<Command> Convert(const DeviceToDeviceCopyThunk& thunk,
                                       AllocationCmdMap& alloc_to_cmd) {
  auto cmd = std::make_unique<MemcpyDeviceToDeviceCmd>(
      thunk.execution_stream_id(), thunk.destination(), thunk.source(),
      thunk.size_bytes());
  TF_RETURN_IF_ERROR(
      alloc_to_cmd.InsertBufferUse(thunk.source().index(), cmd.get()));
  TF_RETURN_IF_ERROR(
      alloc_to_cmd.InsertBufferUse(thunk.destination().index(), cmd.get()));
  return cmd;
}

static absl::StatusOr<Command> Convert(const MemzeroThunk& thunk,
                                       AllocationCmdMap& alloc_to_cmd) {
  auto cmd = std::make_unique<MemzeroCmd>(thunk.execution_stream_id(),
                                          thunk.destination());
  TF_RETURN_IF_ERROR(
      alloc_to_cmd.InsertBufferUse(thunk.destination().index(), cmd.get()));
  return cmd;
}

static absl::StatusOr<Command> Convert(const Memset32BitValueThunk& thunk,
                                       AllocationCmdMap& alloc_to_cmd) {
  auto cmd = std::make_unique<Memset32Cmd>(thunk.execution_stream_id(),
                                           thunk.destination(), thunk.value());
  TF_RETURN_IF_ERROR(
      alloc_to_cmd.InsertBufferUse(thunk.destination().index(), cmd.get()));
  return cmd;
}

static absl::StatusOr<Command> Convert(
    const WhileThunk& thunk, AllocationCmdMap& alloc_to_cmd,
    CommandBufferCmdSequence::SynchronizationMode synchronization_mode) {
  TF_ASSIGN_OR_RETURN(
      CommandBufferCmdSequence cond_cmds,
      ConvertToCommands(thunk.condition_thunk_sequence()->thunks(),
                        alloc_to_cmd, synchronization_mode));
  TF_ASSIGN_OR_RETURN(CommandBufferCmdSequence body_cmds,
                      ConvertToCommands(thunk.body_thunk_sequence()->thunks(),
                                        alloc_to_cmd, synchronization_mode));
  return std::make_unique<WhileCmd>(thunk.execution_stream_id(),
                                    thunk.condition_result_buffer(),
                                    std::move(cond_cmds), std::move(body_cmds));
}

static absl::StatusOr<Command> Convert(const GemmThunk& thunk,
                                       AllocationCmdMap& alloc_to_cmd) {
  if (!thunk.workspace().has_value()) {
    return absl::InternalError(
        "Gemm thunk does not contain a workspace buffer");
  }
  auto cmd = std::make_unique<GemmCmd>(
      thunk.execution_stream_id(), thunk.config(), thunk.lhs_buffer(),
      thunk.rhs_buffer(), thunk.output_buffer(), thunk.workspace().value(),
      thunk.deterministic());
  TF_RETURN_IF_ERROR(alloc_to_cmd.InsertBufferUse(
      thunk.workspace().value().index(), cmd.get()));
  TF_RETURN_IF_ERROR(
      alloc_to_cmd.InsertBufferUse(thunk.lhs_buffer().index(), cmd.get()));
  TF_RETURN_IF_ERROR(
      alloc_to_cmd.InsertBufferUse(thunk.rhs_buffer().index(), cmd.get()));
  TF_RETURN_IF_ERROR(
      alloc_to_cmd.InsertBufferUse(thunk.output_buffer().index(), cmd.get()));
  return cmd;
}

static absl::StatusOr<Command> Convert(
    const ConditionalThunk& thunk, AllocationCmdMap& alloc_to_cmd,
    CommandBufferCmdSequence::SynchronizationMode synchronization_mode) {
  std::vector<CommandBufferCmdSequence> branch_cmds;
  branch_cmds.reserve(thunk.branch_thunks().size());
  for (auto& branch_thunk : thunk.branch_thunks()) {
    TF_ASSIGN_OR_RETURN(CommandBufferCmdSequence cmds,
                        ConvertToCommands(branch_thunk->thunks(), alloc_to_cmd,
                                          synchronization_mode));
    branch_cmds.emplace_back(std::move(cmds));
  }
  auto cmd = std::make_unique<CaseCmd>(thunk.execution_stream_id(),
                                       thunk.branch_index_buffer(),
                                       std::move(branch_cmds));

  TF_RETURN_IF_ERROR(alloc_to_cmd.InsertBufferUse(
      thunk.branch_index_buffer().index(), cmd.get()));
  return cmd;
}

static absl::StatusOr<Command> Convert(const NcclAllReduceStartThunk& thunk,
                                       AllocationCmdMap& alloc_to_cmd) {
  auto cmd = std::make_unique<AllReduceCmd>(
      thunk.nccl_execution_stream_id(), thunk.execution_stream_id(),
      thunk.nccl_api(), thunk.config(), thunk.reduction_kind(),
      thunk.buffers());
  for (auto buffer : thunk.buffers()) {
    TF_RETURN_IF_ERROR(
        alloc_to_cmd.InsertBufferUse(buffer.source_buffer.index(), cmd.get()));
    TF_RETURN_IF_ERROR(alloc_to_cmd.InsertBufferUse(
        buffer.destination_buffer.index(), cmd.get()));
  }
  return cmd;
}

static absl::StatusOr<Command> Convert(const NcclReduceScatterStartThunk& thunk,
                                       AllocationCmdMap& alloc_to_cmd) {
  auto cmd = std::make_unique<ReduceScatterCmd>(
      thunk.nccl_execution_stream_id(), thunk.execution_stream_id(),
      thunk.nccl_api(), thunk.config(), thunk.reduction_kind(),
      thunk.buffers());
  for (auto buffer : thunk.buffers()) {
    TF_RETURN_IF_ERROR(
        alloc_to_cmd.InsertBufferUse(buffer.source_buffer.index(), cmd.get()));
    TF_RETURN_IF_ERROR(alloc_to_cmd.InsertBufferUse(
        buffer.destination_buffer.index(), cmd.get()));
  }
  return cmd;
}

static absl::StatusOr<Command> Convert(const NcclAllGatherStartThunk& thunk,
                                       AllocationCmdMap& alloc_to_cmd) {
  auto cmd = std::make_unique<AllGatherCmd>(
      thunk.nccl_execution_stream_id(), thunk.execution_stream_id(),
      thunk.nccl_api(), thunk.config(), thunk.buffers());
  for (auto buffer : thunk.buffers()) {
    TF_RETURN_IF_ERROR(
        alloc_to_cmd.InsertBufferUse(buffer.source_buffer.index(), cmd.get()));
    TF_RETURN_IF_ERROR(alloc_to_cmd.InsertBufferUse(
        buffer.destination_buffer.index(), cmd.get()));
  }
  return cmd;
}

static absl::StatusOr<Command> Convert(const NcclCollectiveDoneThunk& thunk,
                                       AllocationCmdMap& alloc_to_cmd) {
  return std::make_unique<BarrierCmd>(thunk.execution_stream_id(),
                                      thunk.nccl_execution_stream_id());
}

static absl::StatusOr<Command> Convert(const PartitionIdThunk& thunk,
                                       AllocationCmdMap& alloc_to_cmd) {
  auto cmd = std::make_unique<ComputationIdCmd>(
      thunk.execution_stream_id(), thunk.dest(),
      ComputationIdCmd::Kind::kPartition);
  TF_RETURN_IF_ERROR(
      alloc_to_cmd.InsertBufferUse(thunk.dest().index(), cmd.get()));
  return cmd;
}

static absl::StatusOr<Command> Convert(const ReplicaIdThunk& thunk,
                                       AllocationCmdMap& alloc_to_cmd) {
  auto cmd = std::make_unique<ComputationIdCmd>(
      thunk.execution_stream_id(), thunk.dest(),
      ComputationIdCmd::Kind::kReplica);
  TF_RETURN_IF_ERROR(
      alloc_to_cmd.InsertBufferUse(thunk.dest().index(), cmd.get()));
  return cmd;
}

static absl::StatusOr<Command> Convert(const CustomCallThunk& thunk,
                                       AllocationCmdMap& alloc_to_cmd) {
  auto cmd = std::make_unique<CustomCallCmd>(
      thunk.execution_stream_id(), thunk.call_target(), thunk.operands(),
      thunk.results(), thunk.opaque());
  for (auto slice : thunk.operands()) {
    if (slice.has_value()) {
      TF_RETURN_IF_ERROR(
          alloc_to_cmd.InsertBufferUse(slice.value().slice.index(), cmd.get()));
    }
  }
  for (auto slice : thunk.results()) {
    if (slice.has_value()) {
      TF_RETURN_IF_ERROR(
          alloc_to_cmd.InsertBufferUse(slice.value().slice.index(), cmd.get()));
    }
  }
  return cmd;
}

static absl::StatusOr<Command> Convert(const CuDnnThunk& thunk,
                                       AllocationCmdMap& alloc_to_cmd) {
  auto cmd = std::make_unique<CuDnnCmd>(thunk.execution_stream_id(),
                                        thunk.arguments(), thunk.graph());
  for (auto slice : thunk.arguments()) {
    TF_RETURN_IF_ERROR(alloc_to_cmd.InsertBufferUse(slice.index(), cmd.get()));
  }
  return cmd;
}

//===----------------------------------------------------------------------===//
static absl::StatusOr<Command> CopyMetadata(absl::StatusOr<Command> cmd,
                                            const Thunk& thunk) {
  if (cmd.ok()) {
    (*cmd)->set_profile_annotation(thunk.profile_annotation());
    return cmd;
  }
  return cmd;
}

template <typename ThunkType>
static absl::StatusOr<Command> Convert(const Thunk& thunk,
                                       AllocationCmdMap& alloc_to_cmd) {
  return CopyMetadata(
      Convert(static_cast<const ThunkType&>(thunk), alloc_to_cmd), thunk);
}

template <typename ThunkType>
static absl::StatusOr<Command> Convert(
    const Thunk& thunk, AllocationCmdMap& alloc_to_cmd,
    CommandBufferCmdSequence::SynchronizationMode synchronization_mode) {
  return Convert(static_cast<const ThunkType&>(thunk), alloc_to_cmd,
                 synchronization_mode);
}

static absl::Status AppendCommands(
    CommandBufferCmdSequence& cmd_sequence, AllocationCmdMap& alloc_to_cmd,
    const Thunk& thunk,
    CommandBufferCmdSequence::SynchronizationMode synchronization_mode) {
  auto append = [&](absl::StatusOr<Command> command) -> absl::Status {
    if (command.ok()) {
      cmd_sequence.Append(std::move(*command));
      return absl::OkStatus();
    }
    return command.status();
  };

  switch (thunk.kind()) {
    case Thunk::Kind::kConditional:
      return append(
          Convert<ConditionalThunk>(thunk, alloc_to_cmd, synchronization_mode));
    case Thunk::Kind::kCopy:
      return append(Convert<DeviceToDeviceCopyThunk>(thunk, alloc_to_cmd));
    case Thunk::Kind::kCustomCall:
      return append(Convert<CustomCallThunk>(thunk, alloc_to_cmd));
    case Thunk::Kind::kCustomKernel:
      return append(Convert<CustomKernelThunk>(thunk, alloc_to_cmd));
    case Thunk::Kind::kKernel:
      return append(Convert<KernelThunk>(thunk, alloc_to_cmd));
    case Thunk::Kind::kGemm:
      return append(Convert<GemmThunk>(thunk, alloc_to_cmd));
    case Thunk::Kind::kMemset32BitValue:
      return append(Convert<Memset32BitValueThunk>(thunk, alloc_to_cmd));
    case Thunk::Kind::kMemzero:
      return append(Convert<MemzeroThunk>(thunk, alloc_to_cmd));
    case Thunk::Kind::kNcclAllGatherStart:
      return append(Convert<NcclAllGatherStartThunk>(thunk, alloc_to_cmd));
    case Thunk::Kind::kNcclAllReduceStart:
      return append(Convert<NcclAllReduceStartThunk>(thunk, alloc_to_cmd));
    case Thunk::Kind::kNcclReduceScatterStart:
      return append(Convert<NcclReduceScatterStartThunk>(thunk, alloc_to_cmd));
    case Thunk::Kind::kPartitionId:
      return append(Convert<PartitionIdThunk>(thunk, alloc_to_cmd));
    case Thunk::Kind::kReplicaId:
      return append(Convert<ReplicaIdThunk>(thunk, alloc_to_cmd));
    case Thunk::Kind::kWhile:
      return append(
          Convert<WhileThunk>(thunk, alloc_to_cmd, synchronization_mode));
    case Thunk::Kind::kCuDnn:
      return append(Convert<CuDnnThunk>(thunk, alloc_to_cmd));

    // Sequential thunk does not have any special semantics and we simply inline
    // all nested thunks into command buffer.
    case Thunk::Kind::kSequential:
      return AppendCommands(cmd_sequence, alloc_to_cmd,
                            static_cast<const SequentialThunk&>(thunk).thunks(),
                            synchronization_mode);

    case Thunk::Kind::kNcclAllGatherDone:
    case Thunk::Kind::kNcclAllReduceDone:
    case Thunk::Kind::kNcclReduceScatterDone:
      return append(Convert<NcclCollectiveDoneThunk>(thunk, alloc_to_cmd));

    // Currently all collective operations recorded on the tracing stream and do
    // not need to have a separate done command.
    case Thunk::Kind::kWaitForStreams:
      return absl::OkStatus();

    default:
      return Internal("Unsupported thunk kind: %s",
                      Thunk::KindToString(thunk.kind()));
  }
}

static absl::Status AppendCommands(
    CommandBufferCmdSequence& cmd_sequence, AllocationCmdMap& alloc_to_cmd,
    const ThunkSequence& sequence,
    CommandBufferCmdSequence::SynchronizationMode synchronization_mode) {
  for (const std::unique_ptr<Thunk>& thunk : sequence)
    TF_RETURN_IF_ERROR(AppendCommands(cmd_sequence, alloc_to_cmd, *thunk,
                                      synchronization_mode));
  return absl::OkStatus();
}

// TODO(vuson): Add unit tests.
absl::StatusOr<CommandBufferCmdSequence> ConvertToCommands(
    const ThunkSequence& sequence, AllocationCmdMap& alloc_to_cmd,
    CommandBufferCmdSequence::SynchronizationMode synchronization_mode) {
  CommandBufferCmdSequence cmd_sequence(synchronization_mode);
  TF_RETURN_IF_ERROR(AppendCommands(cmd_sequence, alloc_to_cmd, sequence,
                                    synchronization_mode));
  return cmd_sequence;
}

}  // namespace xla::gpu
