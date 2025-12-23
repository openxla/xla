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

#include "xla/backends/gpu/runtime/command_buffer_params.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <variant>

#include "absl/functional/function_ref.h"
#include "xla/stream_executor/command_buffer.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// CommandBufferStateManager
//===----------------------------------------------------------------------===//

CommandBufferStateManager::TypeId CommandBufferStateManager::GetNextTypeId() {
  static auto* counter = new std::atomic<int64_t>(1);
  return TypeId(counter->fetch_add(1));
}

CommandBufferState* CommandBufferStateManager::GetOrNull(
    std::variant<const CommandBufferCmd*, const Thunk*> cmd,
    const se::CommandBuffer* command_buffer, TypeId type_id,
    int64_t unroll_iteration) {
  Key key = {cmd, command_buffer, type_id, unroll_iteration};
  if (auto it = state_.find(key); it != state_.end()) {
    return it->second.get();
  }
  return nullptr;
}

CommandBufferState* CommandBufferStateManager::GetOrCreate(
    std::variant<const CommandBufferCmd*, const Thunk*> cmd,
    const se::CommandBuffer* command_buffer, TypeId type_id,
    int64_t unroll_iteration,
    absl::FunctionRef<std::unique_ptr<CommandBufferState>()> create) {
  Key key = {cmd, command_buffer, type_id, unroll_iteration};
  if (auto it = state_.find(key); it != state_.end()) {
    return it->second.get();
  }
  return state_.try_emplace(key, create()).first->second.get();
}

}  // namespace xla::gpu
