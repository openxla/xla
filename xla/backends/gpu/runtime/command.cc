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

#include "xla/backends/gpu/runtime/command.h"

#include <string>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/command_buffer.h"

namespace xla::gpu {

std::string CommandTypeString(CommandType type) {
  switch (type) {
#define CASE_CMD_STRING(enum_name, cmd_name, ...) \
  case CommandType::enum_name:                    \
    return cmd_name;
    XLA_GPU_COMMAND_LIST(CASE_CMD_STRING)
#undef CASE_CMD_STRING
  }
}

absl::StatusOr<const se::CommandBuffer::Command*> Command::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& /*record_params*/, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  if (construction_mode_ == ConstructionMode::kExplicit) {
    return absl::UnimplementedError("Record is not implemented");
  }

  // kCapture mode
  if (std::holds_alternative<RecordUpdate>(record_action)) {
    return absl::UnimplementedError(
        "RecordUpdate is not supported in kCapture construction mode.");
  }

  // RecordCreate: capture ExecuteOnStream into the command buffer via Trace.
  auto& create = std::get<RecordCreate>(record_action);
  se::Stream* trace_stream = execute_params.command_buffer_trace_stream;

  TF_ASSIGN_OR_RETURN(
      std::vector<const se::CommandBuffer::Command*> sinks,
      command_buffer->Trace(
          trace_stream,
          [this, &execute_params]() -> absl::Status {
            return ExecuteOnStream(execute_params);
          },
          create.dependencies));

  if (sinks.empty()) {
    return absl::InternalError("Trace returned no commands");
  }
  return sinks[0];
}

bool IsCollectiveCommand(CommandType type) {
  switch (type) {
    case CommandType::kAllGatherCmd:
    case CommandType::kAllReduceCmd:
    case CommandType::kAllToAllCmd:
    case CommandType::kCollectiveBroadcastCmd:
    case CommandType::kCollectivePermuteCmd:
    case CommandType::kRaggedAllToAllCmd:
    case CommandType::kReduceScatterCmd:
    case CommandType::kRecvCmd:
    case CommandType::kSendCmd:
      return true;
    default:
      return false;
  }
}

}  // namespace xla::gpu
