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

#ifndef XLA_BACKENDS_GPU_RUNTIME_TRACED_COMMAND_H_
#define XLA_BACKENDS_GPU_RUNTIME_TRACED_COMMAND_H_

#include <optional>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// Inlined traced commands
//===----------------------------------------------------------------------===//

// Returns true if `command` can be traced directly into the parent command
// buffer instead of being recorded as a child node backed by a nested command
// buffer. This is the case only when the command is guaranteed to never
// require an update: every buffer allocation used by the command is
// persistent (its command-buffer-visible address never changes) and command
// parameters can't change for any other reason.
bool ShouldInlineTracedCommand(const Command& command,
                               const Thunk::ExecuteParams& execute_params);

// Tries to record a traced region produced by the `trace` function directly
// into the parent command buffer (without a child node and a nested command
// buffer). Returns:
//  - the recorded command on success: on create the region is traced inline
//    and the command is marked as inlined; on update of a previously inlined
//    command the update is validated to be a no-op,
//  - std::nullopt if the caller must record with its own child node recording
//    path: the command is not eligible for inlining, the platform does not
//    support tracing into an existing command buffer, or the command was not
//    previously inlined.
//
// Commands recorded by this function can never be updated, see
// `ShouldInlineTracedCommand` above. The traced region is always recorded at
// default priority: the command's priority setting is not propagated to the
// captured nodes.
absl::StatusOr<std::optional<const se::CommandBuffer::Command*>>
TryRecordInlinedTracedCommand(
    Command& command, const Thunk::ExecuteParams& execute_params,
    const Command::RecordParams& record_params,
    const Command::RecordAction& record_action,
    se::CommandBuffer* command_buffer,
    absl::FunctionRef<absl::Status(se::Stream*)> trace);

//===----------------------------------------------------------------------===//
// TracedCommand
//===----------------------------------------------------------------------===//

// A base class for commands implemented as tracing of stream activities.
// Subclasses may override Record() for custom behavior; the default
// implementation traces ExecuteOnStream() on the command_buffer_trace_stream.
class TracedCommand : public Command {
 public:
  bool IsTracedCommand() const override { return true; }

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

 protected:
  explicit TracedCommand(Thunk::Kind kind);

  // Constructor for Thunk subclasses that are also TracedCommands.
  // Preserves the caller's Thunk::Kind and ThunkInfo.
  TracedCommand(Thunk::Kind thunk_kind, ThunkInfo thunk_info);

  // Creates a command buffer by calling a user-provided `trace` function and
  // adds it as a nested command to `command_buffer`. Traced command buffers
  // cached and reused in an instance of `TracedCommandBuffer` kept in `state`.
  //
  // If the command is guaranteed to never require an update (all buffer
  // allocations used by the command have stable addresses, see
  // `ShouldInlineTracedCommand` above), the `trace` function is instead traced
  // directly into `command_buffer`, without a child node and a nested command
  // buffer.
  absl::StatusOr<const se::CommandBuffer::Command*> RecordTracedCommand(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer,
      absl::FunctionRef<absl::Status(se::Stream*)> trace);
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_TRACED_COMMAND_H_
