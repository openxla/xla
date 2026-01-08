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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COMMAND_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_COMMAND_THUNK_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_params.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/execution_graph.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

// Forward declaration.
class CommandBufferCmdExecutor;

//===----------------------------------------------------------------------===//
// CommandThunk
//===----------------------------------------------------------------------===//

// An external manager for a state attached to commands recorded into command
// buffers (same command can be recorded into multiple command buffers).
using ResourceUseVector = absl::InlinedVector<ResourceUse, 1>;

// CommandThunk is a Thunk counterpart that instead of launching operations
// directly on the underlying device records them into command buffers.
//
// Commands have the same execution stages as thunks as they are executed by a
// command buffer thunk: Prepare, Initialize and Record (Execute). See Thunk
// documentation for details.
//
// Commands must be thread safe as they can be recorded into multiple command
// buffers concurrently on different stream executors.
//
// IMPORTANT: In contrast to GPU thunks, commands MUST be stateless. Thunk state
// typically belongs to the Thunk instance itself, and tends to be kept in
// synchronized hash maps keyed by `se::StreamExecutor*` pointer. Commands on
// the other hand should attach state to the underlying command buffer, and
// because the number of command buffers that can be instantiated from a command
// sequence is unbounded (as we have an eviction policy for command buffers),
// keeping a state in a map inside the command will lead to memory leaks.
//
// Commands have an external state manager, which is responsible for managing
// the lifetime of command state. See `CommandState` and
// `CommandStateManager` classes below.
//
// To make command stateful, it needs a `params.state` indirection:
//
//   class MyCommand : public CommandThunk {
//     public:
//
//     // Container for mutable state required for command execution.
//     struct MyState : CommandState {
//       ...
//     };
//
//     absl::StatusOr<Command*> Record(...) override {
//       // Attach a new instance of `MyState` to the `command_buffer`. When
//       // command buffer will be destroyed, the state will be destroyed as
//       // well automatically by XLA runtime. If this command will be recorded
//       // into another command buffer, the state will be re-created
//       // automatically using the provided callback.
//       MyState* my_state = record_params.state.GetOrCreate<MyState>(this,
//         command_buffer, [&] { // create MyState for a `command_buffer` });
//       ...
//     }
//
//   };
//
class CommandThunk : public Thunk {
 public:
  explicit CommandThunk(
      Thunk::Kind kind, Thunk::ThunkInfo thunk_info,
      se::StreamPriority priority = se::StreamPriority::Default)
      : Thunk(kind, thunk_info), priority_(priority) {
    token_ = Resource::Create(Resource::kToken);
    resources_.push_back(ResourceUse::Write(token_));
  }

  virtual ~CommandThunk() = default;

  using BufferUseVector = absl::InlinedVector<BufferUse, 4>;

  using State = ::xla::gpu::CommandState;

  using CreateCommand =
      absl::FunctionRef<absl::StatusOr<const se::CommandBuffer::Command*>()>;

  using UpdateCommand = absl::FunctionRef<absl::Status(
      const se::CommandBuffer::Command* command)>;

  absl::Status HandleCmdCreateOrUpdate(CommandBufferParams& record_params,
                                       CreateCommand create_command,
                                       UpdateCommand update_command);

  // Returns true if command requires initialization (has to be recorded at
  // command buffer thunk initialization).
  //
  // Today this is only true for collective commands that might use NCCL for
  // communication. With NCCL, all participating ranks must record collective
  // commands at the same time, if some ranks will skip command updates (because
  // they got lucky and got the same buffer allocations), it will lead to
  // deadlocks. By forcing the command update at thunk initialization time, we
  // ensure that all ranks execute NCCL command update.
  virtual bool command_buffer_requires_initialization() const { return false; }

  // Returns true if command supports loop unroll, the while loop can be
  // unrolled only if it has pre-known trip count and also all commands from the
  // body commands are unrollable..
  virtual bool command_buffer_support_loop_unroll() const { return true; }

  // This is only true for DynamicSliceCopyFusionCmd when offset is dependents
  // on loop iteration. As the command of slice operation is access the sliced
  // memory region that varies across loop iterations, so even the original
  // buffer allocation is the same, it still requires to do update.
  virtual bool command_buffer_force_update() const { return false; }

  // Returns async events for async commands (CollectiveCmd, AsyncDoneCmd).
  // Returns nullptr for non-async commands.
  virtual std::shared_ptr<CollectiveThunk::AsyncEvents> async_events() const {
    return nullptr;
  }

  std::shared_ptr<Resource> token() const { return token_; }

  void add_resouce_use(ResourceUse resource_use) {
    resources_.push_back(resource_use);
  }
  ResourceUseVector resources() const { return resources_; }

  absl::string_view profile_annotation() const { return profile_annotation_; }
  void set_profile_annotation(absl::string_view profile_annotation) {
    profile_annotation_ = profile_annotation;
  }

  se::StreamPriority command_buffer_priority() const { return priority_; }
  void set_command_buffer_priority(se::StreamPriority priority) {
    priority_ = priority;
  }

  // Return the dependencies of the command from within the executor, if the
  // command is a source command, it will return the executor dependencies
  // specified in record_params.
  std::vector<const se::CommandBuffer::Command*> CommandBufferDependencies(
      const CommandBufferParams& record_params) const;

 private:
  std::string profile_annotation_;

  ResourceUseVector resources_;

  // The token resource is used to specify additional dependency across
  // commands, like control dependency across HLO operators, and LHS
  // scheduling dependency.
  std::shared_ptr<Resource> token_;

  // Command priority, currently only support default, lowest and highest
  // priority.
  se::StreamPriority priority_ = se::StreamPriority::Default;
};

//===----------------------------------------------------------------------===//
// CommandBufferCmdSequence
//===----------------------------------------------------------------------===//

// A sequence of commands (corresponds to a ThunkSequence from the Thunk API).
class CommandBufferCmdSequence
    : public std::vector<std::unique_ptr<CommandThunk>> {
 public:
  template <typename CmdT, typename... Args>
  void Emplace(Args&&... args) {
    this->emplace_back(std::make_unique<CmdT>(std::forward<Args>(args)...));
  }

  std::string ToString(int indent = 0) const;
};

//===----------------------------------------------------------------------===//
// CommandBufferCmdExecutor
//===----------------------------------------------------------------------===//

// Command executor is responsible for recording commands sequence into the
// underlying command buffer and setting up dependencies between commands.
class CommandBufferCmdExecutor {
 public:
  CommandBufferCmdExecutor() = default;
  CommandBufferCmdExecutor(CommandBufferCmdExecutor&&) = default;
  CommandBufferCmdExecutor& operator=(CommandBufferCmdExecutor&&) = default;

  // Synchronization mode defines how much concurrency is allowed between
  // commands in the sequence.
  enum class SynchronizationMode {
    // Serializes execution of all commands recorded into the command buffer
    // by adding a dependency between them.
    kSerialize,

    // Relies on execution graph to insert dependencies between commands
    // that have buffer of resource conflicts, and building a DAG of commands.
    kConcurrent,

    // Uses the same latency hidden scheduling results used in the thunk
    // scheduling.
    kLHS,
  };

  template <typename Sink>
  friend void AbslStringify(Sink& sink, SynchronizationMode mode) {
    switch (mode) {
      case SynchronizationMode::kSerialize:
        sink.Append("serialize");
        break;
      case SynchronizationMode::kConcurrent:
        sink.Append("concurrent");
        break;
      case SynchronizationMode::kLHS:
        sink.Append("lhs");
        break;
    }
  }

  // Creates a command executor from a sequence of commands using given
  // synchronization mode.
  static absl::StatusOr<CommandBufferCmdExecutor> Create(
      CommandBufferCmdSequence commands,
      SynchronizationMode synchronization_mode);

  // Prepares all commands added to a sequence.
  absl::Status Prepare(const Thunk::PrepareParams& params);

  // Initializes all commands added to a sequence.
  absl::Status Initialize(const Thunk::InitializeParams& params);

  // Records commands into the command buffer. This method automatically
  // switches between `RecordCreate` or `RecordUpdate` depending on the
  // command buffer state.

  // This Record function allows multiple CommandbufferCmdEXecutor to be
  // recorded into a single command buffer. e.g. we can have Executor A, B, C
  // to be recorded into the same command buffer in the order of A -> B -> C.
  // In this pattern, B's source commands will depend on A's sink commands,
  // and C's source commands will also depend on B's sink commands.

  // If record_action is `RecordCreate`, it will set up initial
  // dependencies for recorded commands by the `dependencies` parameter.
  // If record_action is `RecordUpdate`, it will only update previously
  // recorded commands' dependencies, no other actions.

  // Records commands into the command buffer. Uses record_params.is_finalize
  // to determine whether to finalize the command buffer after recording.
  // The CommandBufferParams is accessed via
  // execute_params.command_buffer_params.
  absl::Status ExecuteOnStream(
      const Thunk::ExecuteParams& execute_params) const;

  // Returns buffers referenced by commands in this sequence.
  const absl::flat_hash_set<BufferUse>& buffer_uses() const;

  // Returns buffer allocations indices referenced by commands in this
  // sequence.
  absl::Span<const BufferAllocation::Index> allocs_indices() const;

  bool empty() const { return commands_.empty(); }
  size_t size() const { return commands_.size(); }

  bool command_buffer_requires_initialization() const;

  bool command_buffer_force_update() const;

  bool command_buffer_support_loop_unroll() const;

  // Returns all commands associated with the given ids on a certain unroll
  // iteration.
  std::vector<const se::CommandBuffer::Command*> SourceCommands(
      const CommandBufferParams& record_params) const;

  std::vector<const se::CommandBuffer::Command*> SinkCommands(
      const CommandBufferParams& record_params) const;

  // Renders the execution graph using default renderer. Returns url of the
  // rendered graph, or an error if rendering failed.
  absl::StatusOr<std::string> RenderExecutionGraph();

  // Returns true if the given command is a source command (has no
  // dependencies within this executor).
  bool IsSource(const CommandThunk* cmd) const;

  // Returns dependencies of the given command.
  std::vector<const se::CommandBuffer::Command*> Dependencies(
      const CommandBufferParams& record_params, const CommandThunk* cmd) const;

  using CreateCommand =
      absl::FunctionRef<absl::StatusOr<const se::CommandBuffer::Command*>()>;

  using UpdateCommand = absl::FunctionRef<absl::Status(
      const se::CommandBuffer::Command* absl_nonnull command)>;

  absl::Status HandleCmdCreateOrUpdate(CommandBufferParams& record_params,
                                       const CommandThunk* cmd,
                                       CreateCommand create_command,
                                       UpdateCommand update_command) const;

 private:
  // We use index into the `commands_` vector as a command id.
  using CommandId = int64_t;

  // A state associated with commands in the sequence. We rely on this state
  // to efficiently update command recorded into the command buffer.
  struct RecordState : public CommandState {
    const se::CommandBuffer::Command* command;
  };

  CommandBufferCmdExecutor(SynchronizationMode synchronization_mode,
                           CommandBufferCmdSequence commands,
                           std::optional<ExecutionGraph> execution_graph);

  absl::Status CheckCommandBufferState(
      se::CommandBuffer* absl_nonnull command_buffer,
      se::CommandBuffer::State expected_state) const;

  // Returns true if command has no dependencies.
  bool IsSource(CommandId id) const;

  // Returns true if command is not a dependency of any other commands.
  bool IsSink(CommandId id) const;

  // Returns dependencies of the command with the given id.
  std::vector<const se::CommandBuffer::Command*> Dependencies(
      const CommandBufferParams& record_params, CommandId id) const;

  SynchronizationMode synchronization_mode_;
  CommandBufferCmdSequence commands_;

  // In automatic synchronization mode we build an execution graph for the
  // sequence of commands and use it to set up dependencies between commands.
  std::optional<ExecutionGraph> execution_graph_;

  // Buffers referenced by commands in this sequence.
  absl::flat_hash_set<BufferUse> buffers_;

  // Unique buffer allocations indices referenced by all commands in this
  // sequence (sorted by the buffer allocation index).
  std::vector<BufferAllocation::Index> allocs_indices_;

  // A mapping from command id to unique buffer allocations indices referenced
  // by the command (sorted by the buffer allocation index).
  std::vector<std::vector<BufferAllocation::Index>> cmd_allocs_indices_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COMMAND_THUNK_H_
