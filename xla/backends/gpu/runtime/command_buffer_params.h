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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_PARAMS_H_
#define XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_PARAMS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla::gpu {

// Forward declarations to avoid circular dependencies.
class CommandBufferCmd;
class CommandBufferCmdExecutor;
class Thunk;

// A base class for externally managed command state.
//
// Commands can be executed concurrently for many stream executors (underlying
// devices) and command buffers. Managing per-executor state can become
// expensive as it requires synchronization. Furthermore the number of command
// buffers command is recorded into is unbounded as they come and go (command
// buffers evicted and reconstructed) which makes it hard to manage the
// lifetime of resources attached to command buffers.
//
// Externally managed state (owned and synchronized by CommandBufferThunk)
// allows commands to attach a piece of information to command buffer in a
// safe and performant way.
class CommandBufferState {
 public:
  virtual ~CommandBufferState() = default;
};

// CommandBufferStateManager is an external manager for a state attached to
// commands recorded into command buffers (same command can be recorded into
// multiple command buffers).
class CommandBufferStateManager {
 public:
  virtual ~CommandBufferStateManager() = default;

  template <typename ConcreteState>
  ConcreteState* absl_nullable GetOrNull(
      std::variant<const CommandBufferCmd*, const Thunk*> cmd,
      const se::CommandBuffer* absl_nonnull command_buffer,
      int64_t unroll_iteration = 0) {
    static_assert(std::is_base_of_v<CommandBufferState, ConcreteState>);
    return static_cast<ConcreteState*>(
        GetOrNull(cmd, command_buffer, GetTypeId<ConcreteState>(),
                  unroll_iteration));
  }

  template <typename ConcreteState>
  ConcreteState* absl_nonnull GetOrCreate(
      std::variant<const CommandBufferCmd*, const Thunk*> cmd,
      const se::CommandBuffer* absl_nonnull command_buffer,
      absl::FunctionRef<std::unique_ptr<ConcreteState>()> create,
      int64_t unroll_iteration = 0) {
    static_assert(std::is_base_of_v<CommandBufferState, ConcreteState>);
    return static_cast<ConcreteState*>(
        GetOrCreate(cmd, command_buffer, GetTypeId<ConcreteState>(),
                    unroll_iteration, [&] { return create(); }));
  }

  template <typename ConcreteState>
  ConcreteState* absl_nonnull GetOrCreate(
      std::variant<const CommandBufferCmd*, const Thunk*> cmd,
      const se::CommandBuffer* absl_nonnull command_buffer,
      int64_t unroll_iteration = 0) {
    return GetOrCreate<ConcreteState>(
        cmd, command_buffer, [] { return std::make_unique<ConcreteState>(); },
        unroll_iteration);
  }

 private:
  // We use TypeId to distinguish between different state types.
  TSL_LIB_GTL_DEFINE_INT_TYPE(TypeId, int64_t);

  template <typename F>
  static TypeId GetTypeId() {
    static const TypeId id = GetNextTypeId();
    return id;
  }

  static TypeId GetNextTypeId();

  CommandBufferState* absl_nullable GetOrNull(
      std::variant<const CommandBufferCmd*, const Thunk*> cmd,
      const se::CommandBuffer* absl_nonnull command_buffer, TypeId type_id,
      int64_t unroll_iteration);

  CommandBufferState* absl_nonnull GetOrCreate(
      std::variant<const CommandBufferCmd*, const Thunk*> cmd,
      const se::CommandBuffer* absl_nonnull command_buffer, TypeId type_id,
      int64_t unroll_iteration,
      absl::FunctionRef<std::unique_ptr<CommandBufferState>()> create);

  using Key = std::tuple<std::variant<const CommandBufferCmd*, const Thunk*>,
                         const se::CommandBuffer*, TypeId, int64_t>;
  absl::flat_hash_map<Key, std::unique_ptr<CommandBufferState>> state_;
};

// Parameters for recording commands into the command buffer.
struct CommandBufferParams {
  // An external state manager that gives efficient access to per-device state
  // to commands without a need to add expensive synchronization.
  CommandBufferStateManager& state;

  // Buffer allocations that changed since the last call to `Record`. Buffer
  // allocation indices are sorted. CommandBufferCmdExecutor and individual
  // commands rely on this information to skip unnecessary updates.
  std::optional<std::vector<BufferAllocation::Index>> updated_allocs;

  // A flag indicating whether we record comands at command buffer thunk
  // initialization time.
  bool is_initialization = false;

  // The command sequence might be recorded in the loop unrolling pattern, so
  // the command sequence might be instantiated multiple times, we uses
  // unroll_iteration to locate the commands for current unroll iteration.
  int64_t unroll_iteration = 0;

  // The command buffer that is recording the commands. Must be set before
  // calling Record().
  se::CommandBuffer* command_buffer = nullptr;

  // A flag indicating whether we finalize the command buffer after recording.
  bool is_finalize = true;

  // The executor that is recording the commands. Used for dependency
  // resolution within the executor. May be null when recording commands
  // outside of an executor context.
  const CommandBufferCmdExecutor* executor = nullptr;

  // External dependencies that source commands in this executor must wait on.
  // When multiple CommandBufferCmdExecutors are recorded into the same
  // command buffer (e.g., A -> B -> C), the source commands of executor B
  // must depend on the sink commands of executor A. These external
  // dependencies are passed in via this field to establish cross-executor
  // ordering.
  //
  // Example: WhileCmd loop unrolling. When a WhileCmd has a known trip count,
  // we unroll the loop by recording cond_commands and body_commands executors
  // multiple times into a single command buffer:
  //   [cond_0] -> [body_0] -> [cond_1] -> [body_1] -> ... -> [cond_N] ->
  //   [body_N]
  // Each iteration's body_commands must wait on the preceding cond_commands,
  // and each cond_commands (except the first) must wait on the preceding
  // body_commands. The external_dependencies field carries these
  // cross-executor dependencies between unrolled iterations.
  std::vector<const se::CommandBuffer::Command*> external_dependencies;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_PARAMS_H_
