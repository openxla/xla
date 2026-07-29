/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/traced_command.h"

#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/traced_command_buffer.h"
#include "xla/debug_options_flags.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"

namespace xla::gpu {

namespace {

// Marks a traced command that was traced directly into the parent command
// buffer (without a child node and a nested command buffer). Such commands
// can never be updated, see `ShouldInlineTracedCommand`.
struct InlinedTracedCommandState : public CommandState {};

}  // namespace

//===----------------------------------------------------------------------===//
// Inlined traced commands
//===----------------------------------------------------------------------===//

bool ShouldInlineTracedCommand(const Command& command,
                               const Thunk::ExecuteParams& execute_params) {
  // Command parameters can change even when buffer allocation addresses are
  // stable, so the command might require updates and can't be inlined.
  if (command.requires_update_on_execute()) {
    return false;
  }

  if (!execute_params.persistent_alloc_indices.has_value()) {
    return false;
  }

  // Inline only if every buffer allocation used by the command is persistent,
  // i.e. its command-buffer-visible address never changes, so the recorded
  // commands will never need an update (see
  // `CommandExecutor::RecordUpdate` that skips updates for such commands).
  absl::btree_set<BufferAllocation::Index> allocs_indices;
  for (const BufferUse& buffer_use : command.buffer_uses()) {
    allocs_indices.insert(buffer_use.slice().index());
  }
  return absl::c_includes(*execute_params.persistent_alloc_indices,
                          allocs_indices);
}

absl::StatusOr<std::optional<const se::CommandBuffer::Command*>>
TryRecordInlinedTracedCommand(
    Command& command, const Thunk::ExecuteParams& execute_params,
    const Command::RecordParams& record_params,
    const Command::RecordAction& record_action,
    se::CommandBuffer* command_buffer,
    absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  if (auto* create = std::get_if<Command::RecordCreate>(&record_action);
      create != nullptr && ShouldInlineTracedCommand(command, execute_params)) {
    // Trace the command directly into the parent command buffer: the command
    // will never need an update, so the child node indirection and a nested
    // command buffer are unnecessary.
    absl::StatusOr<const se::CommandBuffer::Command*> inlined_cmd =
        command_buffer->CreateTracedCommand(
            execute_params.command_buffer_trace_stream, create->dependencies,
            [&](se::Stream* stream) { return trace(stream); });

    if (inlined_cmd.ok()) {
      VLOG(5) << "Traced command " << *inlined_cmd
              << " directly into parent command buffer: " << command_buffer
              << " (CreateTracedCommand)";
      record_params.state.GetOrCreate<InlinedTracedCommandState>(
          &command, command_buffer);
      return std::optional<const se::CommandBuffer::Command*>(*inlined_cmd);
    }

    if (!absl::IsUnimplemented(inlined_cmd.status())) {
      return inlined_cmd.status();
    }

    // Fall back to the child node recording path on platforms that do not
    // support tracing into an existing command buffer.
    return std::nullopt;
  }

  if (auto* update = std::get_if<Command::RecordUpdate>(&record_action);
      update != nullptr &&
      record_params.state.GetOrNull<InlinedTracedCommandState>(
          &command, command_buffer) != nullptr) {
    // The command was traced directly into the parent command buffer and can
    // not be updated. This is valid only while the command still qualifies
    // for inlining (i.e. all buffer allocation addresses are still stable),
    // in which case the update is a no-op.
    if (!ShouldInlineTracedCommand(command, execute_params)) {
      return Internal(
          "Traced command was traced directly into the parent command buffer "
          "but no longer qualifies for it and can't be updated");
    }
    return std::optional<const se::CommandBuffer::Command*>(update->command);
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// TracedCommand
//===----------------------------------------------------------------------===//

TracedCommand::TracedCommand(Thunk::Kind kind) : Command(kind) {}

TracedCommand::TracedCommand(Thunk::Kind thunk_kind, ThunkInfo thunk_info)
    : Command(thunk_kind, std::move(thunk_info)) {}

absl::StatusOr<const se::CommandBuffer::Command*> TracedCommand::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  return RecordTracedCommand(execute_params, record_params,
                             std::move(record_action), command_buffer,
                             [&](se::Stream* stream) {
                               ExecuteParams trace_params = execute_params;
                               trace_params.stream = stream;
                               return ExecuteOnStream(trace_params);
                             });
}

absl::StatusOr<const se::CommandBuffer::Command*>
TracedCommand::RecordTracedCommand(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer,
    absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  ASSIGN_OR_RETURN(
      std::optional<const se::CommandBuffer::Command*> inlined_cmd,
      TryRecordInlinedTracedCommand(*this, execute_params, record_params,
                                    record_action, command_buffer, trace));
  if (inlined_cmd.has_value()) {
    return *inlined_cmd;
  }

  auto traced_cmd = record_params.state.GetOrCreate<TracedCommandBuffer>(
      this, command_buffer, [&] {
        const auto& debug_options = xla::GetDebugOptionsFromFlags();
        return std::make_unique<TracedCommandBuffer>(
            this, buffer_uses(),
            debug_options.xla_cmd_buffer_trace_cache_size());
      });

  ASSIGN_OR_RETURN(
      auto nested_cmd,
      traced_cmd->GetOrTraceCommandBuffer(
          execute_params.buffer_allocations, execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, trace, priority()));

  if (auto* create = std::get_if<RecordCreate>(&record_action)) {
    VLOG(5) << "Record traced command " << nested_cmd
            << " into parent command buffer: " << command_buffer
            << " (CreateChildCommand)";
    return command_buffer->CreateChildCommand(*nested_cmd,
                                              create->dependencies);
  }

  if (auto* update = std::get_if<RecordUpdate>(&record_action)) {
    VLOG(5) << "Record traced command " << nested_cmd
            << " into parent command buffer: " << command_buffer
            << " (UpdateChildCommand)";
    RETURN_IF_ERROR(
        command_buffer->UpdateChildCommand(update->command, *nested_cmd));
    return update->command;
  }

  return Internal("Invalid record action");
}

}  // namespace xla::gpu
