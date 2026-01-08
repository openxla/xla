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
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla::gpu {

// Parameters for recording commands into the command buffer.

class CommandBufferCmdExecutor;
struct CommandBufferParams {
  // An external state manager that gives efficient access to per-device state
  // to commands without a need to add expensive synchronization.
  CommandStateManager& state;

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
