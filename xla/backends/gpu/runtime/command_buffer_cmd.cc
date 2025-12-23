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

#include "xla/backends/gpu/runtime/command_buffer_cmd.h"

#include "xla/backends/gpu/runtime/command_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_params.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/collective_broadcast_thunk.h"
#include "xla/backends/gpu/runtime/collective_execution.h"
#include "xla/backends/gpu/runtime/collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_slice_thunk.h"
#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/shaped_slice.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/debug_options_flags.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/execution_graph.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/tensor_map.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/util.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla::gpu {

namespace {
// Indvar is a thread-local map that stores the induction variable for each
// dynamic slice thunk. The same thunk object in the memory is shared by
// multiple replicas of the same computation. So, each replica should have its
// own tracking of the induction variable (threadlocal). With threadlocal, we
// cannot embed this inside the dynamic slice thunk object, and so we have a
// static map. There could be multiple dynamic slice thunks in the same module,
// and so we need a map to store the induction variable for each thunk. The
// usage of threadlocal in this context is similar to `LoopCounters` in
// while_thunk.cc (b/343294327).
Literal& Indvar(DynamicSliceFusionCmd* cmd) {
  static thread_local absl::flat_hash_map<DynamicSliceFusionCmd*, Literal>
      indvar_map;
  return indvar_map[cmd];
}
}  // namespace

using MemoryAccess = BufferUse::MemoryAccess;

static absl::string_view ReductionKindString(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::MAX:
      return "max";
    case ReductionKind::MIN:
      return "min";
    case ReductionKind::PRODUCT:
      return "product";
    case ReductionKind::SUM:
      return "sum";
  }
}

// Create a callback to create a command buffer from a command sequence.
static se::CommandBuffer::CreateCommands CreateCommands(
    const CommandBufferCmdExecutor* commands,
    const Thunk::ExecuteParams* execute_params) {
  return [=](se::CommandBuffer* command_buffer,
             absl::Span<const se::CommandBuffer::Command* const> dependencies)
             -> absl::StatusOr<std::vector<const se::CommandBuffer::Command*>> {
    CommandBufferParams nest_record_params =
        *execute_params->command_buffer_params;
    nest_record_params.command_buffer = command_buffer;
    nest_record_params.external_dependencies.assign(dependencies.begin(),
                                                    dependencies.end());
    // Don't finalize from within CreateCommands callbacks - the parent
    // (CreateWhile/CreateCase) handles finalization of nested command buffers.
    nest_record_params.is_finalize = false;
    // Set the executor for dependency resolution within this executor.
    nest_record_params.executor = commands;
    // Automatically restores original record_params when scope exits.
    Thunk::ExecuteParams::ScopedCommandBufferParams scoped(*execute_params,
                                                           &nest_record_params);
    TF_RETURN_IF_ERROR(commands->ExecuteOnStream(*execute_params));
    return commands->SinkCommands(nest_record_params);
  };
}

// Create callbacks to create a command buffer from command sequences.
static std::vector<se::CommandBuffer::CreateCommands> CreateCommands(
    absl::Span<const CommandBufferCmdExecutor> commands,
    const Thunk::ExecuteParams* execute_params) {
  std::vector<se::CommandBuffer::CreateCommands> create_commands;
  for (const CommandBufferCmdExecutor& cmd : commands) {
    create_commands.push_back(CreateCommands(&cmd, execute_params));
  }
  return create_commands;
}

// Create a callback to update a command buffer with command sequence.
static se::CommandBuffer::UpdateCommands UpdateCommands(
    const CommandBufferCmdExecutor* commands,
    const Thunk::ExecuteParams* execute_params) {
  return [=](se::CommandBuffer* command_buffer) -> absl::Status {
    CommandBufferParams nest_record_params =
        *execute_params->command_buffer_params;
    nest_record_params.command_buffer = command_buffer;
    // Don't finalize from within UpdateCommands callbacks - the parent
    // (UpdateWhile/UpdateCase) handles the nested command buffer lifecycle.
    nest_record_params.is_finalize = false;
    // Set the executor for dependency resolution within this executor.
    nest_record_params.executor = commands;
    // Automatically restores original record_params when scope exits.
    Thunk::ExecuteParams::ScopedCommandBufferParams scoped(*execute_params,
                                                           &nest_record_params);
    TF_RETURN_IF_ERROR(commands->ExecuteOnStream(*execute_params));
    return absl::OkStatus();
  };
}

// Create callbacks to update a command buffer with command sequence.
static std::vector<se::CommandBuffer::UpdateCommands> UpdateCommands(
    absl::Span<const CommandBufferCmdExecutor> commands,
    const Thunk::ExecuteParams* execute_params) {
  std::vector<se::CommandBuffer::UpdateCommands> update_commands;
  for (const CommandBufferCmdExecutor& cmd : commands) {
    update_commands.push_back(UpdateCommands(&cmd, execute_params));
  }
  return update_commands;
}

//===----------------------------------------------------------------------===//
// TracedCommandBuffer
//===----------------------------------------------------------------------===//

TracedCommandBuffer::TracedCommandBuffer(const CommandThunk* trace_cmd,
                                         CommandThunk::BufferUseVector buffers,
                                         int64_t capacity)
    : trace_cmd_(trace_cmd), capacity_(capacity), entries_(capacity) {
  CHECK_GT(capacity, 0) << "capacity must be larger than 0";  // NOLINT
  // Collect unique buffer allocation indices in a set first and convert to
  // vector as flat hash set iteration has measurable overheads.
  absl::flat_hash_set<BufferAllocation::Index> allocs_indices;
  for (auto& buffer : buffers) {
    allocs_indices.insert(buffer.slice().index());
  }
  allocs_indices_.assign(allocs_indices.begin(), allocs_indices.end());
}

absl::StatusOr<se::CommandBuffer*> TracedCommandBuffer::GetOrTraceCommandBuffer(
    const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
    se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace,
    se::StreamPriority priority) {
  // Collect memory addresses for relevant allocations.
  absl::InlinedVector<se::DeviceAddressBase, 4> allocs;
  allocs.reserve(allocs_indices_.size());
  for (auto& index : allocs_indices_) {
    allocs.emplace_back(buffer_allocation->GetDeviceAddress(index));
  }

  // Moves entry at `i` position to front and moves entries in `[0, i)` range
  // one element to the right. Returns reference to the first entry.
  auto shift_right = [&](size_t i) -> Entry& {
    if (i == 0) {
      return entries_[0];
    }

    Entry entry = std::move(entries_[i]);
    do {
      entries_[i] = std::move(entries_[i - 1]);
    } while (--i > 0);

    return entries_[0] = std::move(entry);
  };

  for (size_t i = 0; i < capacity_; ++i) {
    // Found entry for a given allocations, move it to front and return a
    // pointer to cached command buffer.
    if (ABSL_PREDICT_TRUE(absl::c_equal(entries_[i].recorded_allocs, allocs) &&
                          entries_[i].command_buffer)) {
      VLOG(6) << "Command buffer trace cache hit for command "
              << trace_cmd_->ToString();
      return shift_right(i).command_buffer.get();
    }

    // Create a new entry by calling a user-provided tracing function, move it
    // to front and return a pointer to cached command buffer.
    if (entries_[i].command_buffer == nullptr) {
      TF_ASSIGN_OR_RETURN(
          entries_[i].command_buffer,
          se::TraceCommandBufferFactory::Create(executor, stream, trace));
      entries_[i].recorded_allocs.assign(allocs.begin(), allocs.end());
      if (priority != se::StreamPriority::Default) {
        TF_RETURN_IF_ERROR(entries_[i].command_buffer->SetPriority(priority));
      }
      VLOG(6) << "Command buffer trace cache create new item for command "
              << trace_cmd_->ToString();
      return shift_right(i).command_buffer.get();
    }
  }

  // Create a new entry by calling a user-provided tracing function, replace
  // the last entry with it, move it to front and return a pointer to cached
  // command buffer.
  TF_ASSIGN_OR_RETURN(
      entries_[capacity_ - 1].command_buffer,
      se::TraceCommandBufferFactory::Create(executor, stream, trace));
  entries_[capacity_ - 1].recorded_allocs.assign(allocs.begin(), allocs.end());
  VLOG(6) << "Command buffer trace cache does replacement for command "
          << trace_cmd_->ToString();
  return shift_right(capacity_ - 1).command_buffer.get();
}

//===----------------------------------------------------------------------===//
// TracedCommandBufferCmd
//===----------------------------------------------------------------------===//

TracedCommandBufferCmd::TracedCommandBufferCmd(Thunk::Kind kind)
    : CommandThunk(kind, Thunk::ThunkInfo()) {}

absl::Status TracedCommandBufferCmd::RecordTracedCommand(
    const Thunk::ExecuteParams& execute_params,
    absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  CommandBufferParams* record_params = execute_params.command_buffer_params;
  auto traced_cmd = record_params->state.GetOrCreate<TracedCommandBuffer>(
      this, record_params->command_buffer,
      [&] {
        const auto& debug_options = xla::GetDebugOptionsFromFlags();
        return std::make_unique<TracedCommandBuffer>(
            this, buffer_uses(),
            debug_options.xla_cmd_buffer_trace_cache_size());
      },
      record_params->unroll_iteration);

  TF_ASSIGN_OR_RETURN(
      auto nested_cmd,
      traced_cmd->GetOrTraceCommandBuffer(
          execute_params.buffer_allocations, execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, trace,
          command_buffer_priority()));

  VLOG(5) << "Record traced command into command buffer: "
          << record_params->command_buffer;
  return HandleCmdCreateOrUpdate(
      *record_params,
      [&] {
        return record_params->command_buffer->CreateChildCommand(
            se::CommandBuffer::ChildCommandType::kCloned, *nested_cmd,
            CommandBufferDependencies(*record_params));
      },
      [&](const se::CommandBuffer::Command* cmd) {
        return record_params->command_buffer->UpdateChildCommand(
            se::CommandBuffer::ChildCommandType::kCloned, cmd, *nested_cmd);
      });
}

//===----------------------------------------------------------------------===//
// EmptyCmd
//===----------------------------------------------------------------------===//

EmptyCmd::EmptyCmd()
    : CommandThunk(Thunk::Kind::kSequential, Thunk::ThunkInfo()) {}

absl::Status EmptyCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  CommandBufferParams* record_params = execute_params.command_buffer_params;
  return HandleCmdCreateOrUpdate(
      *record_params,
      [&]() -> absl::StatusOr<const se::CommandBuffer::Command*> {
        return record_params->command_buffer->CreateEmptyCmd(
            CommandBufferDependencies(*record_params),
            command_buffer_priority());
      },
      [&](const se::CommandBuffer::Command* command) {
        // Empty command is not updatable.
        return absl::OkStatus();
      });
}

//===----------------------------------------------------------------------===//
// AsyncDoneCmd
//===----------------------------------------------------------------------===//

AsyncDoneCmd::AsyncDoneCmd(
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : CommandThunk(Thunk::Kind::kGroupDone, Thunk::ThunkInfo()),
      async_events_(std::move(async_events)) {}

absl::Status AsyncDoneCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  CommandBufferParams* record_params = execute_params.command_buffer_params;
  return HandleCmdCreateOrUpdate(
      *record_params,
      [&]() -> absl::StatusOr<const se::CommandBuffer::Command*> {
        return record_params->command_buffer->CreateEmptyCmd(
            CommandBufferDependencies(*record_params),
            command_buffer_priority());
      },
      [&](const se::CommandBuffer::Command* command) {
        return absl::OkStatus();
      });
}

//===----------------------------------------------------------------------===//
// ComputationId
//===----------------------------------------------------------------------===//

ComputationIdCmd::ComputationIdCmd(BufferAllocation::Slice dest, Kind kind)
    : CommandThunk(kind == Kind::kReplica ? Thunk::Kind::kReplicaId
                                          : Thunk::Kind::kPartitionId,
                   Thunk::ThunkInfo()),
      dest_(dest),
      kind_(kind) {}

CommandThunk::BufferUseVector ComputationIdCmd::buffer_uses() const {
  return {BufferUse::Write(dest_)};
}

absl::Status ComputationIdCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  CommandBufferParams* record_params = execute_params.command_buffer_params;
  se::DeviceAddressBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dest_);

  GlobalDeviceId global_device_id =
      execute_params.collective_params->global_device_id;
  TF_ASSIGN_OR_RETURN(
      const DeviceAssignment::LogicalID logical_id,
      execute_params.collective_params->device_assn->LogicalIdForDevice(
          global_device_id));

  uint32_t value = kind_ == Kind::kReplica ? logical_id.replica_id
                                           : logical_id.computation_id;

  VLOG(5) << "ComputationIdCmd"
          << ": kind=" << (kind_ == Kind::kReplica ? "replica" : "partition")
          << "; value=" << value;
  VLOG(5) << "  Id: " << dest_ << " (" << dst.opaque() << ")";

  return HandleCmdCreateOrUpdate(
      *record_params,
      [&] {
        return record_params->command_buffer->CreateMemset(
            &dst, value, /*num_elements=*/1,
            CommandBufferDependencies(*record_params));
      },
      [&](const se::CommandBuffer::Command* command) {
        return record_params->command_buffer->UpdateMemset(command, &dst, value,
                                                           /*num_elements=*/1);
      });
}

//===----------------------------------------------------------------------===//
// LaunchCmd
//===----------------------------------------------------------------------===//

LaunchCmd::LaunchCmd(
    std::string kernel_name, absl::Span<const BufferAllocation::Slice> args,
    absl::Span<const MemoryAccess> args_access, LaunchDimensions dims,
    int64_t shmem_bytes,
    std::optional<stream_executor::gpu::TmaMetadata> tma_metadata)
    : CommandThunk(Thunk::Kind::kKernel, Thunk::ThunkInfo()),
      kernel_name_(std::move(kernel_name)),
      args_(args.begin(), args.end()),
      args_access_(args_access.begin(), args_access.end()),
      dims_(dims),
      shmem_bytes_(shmem_bytes),
      tma_metadata_(std::move(tma_metadata)) {}

absl::Status LaunchCmd::Initialize(const Thunk::InitializeParams& params) {
  {
    absl::MutexLock lock(mutex_);
    if (kernels_.contains(params.executor)) {
      return absl::OkStatus();
    }
  }

  std::unique_ptr<se::Kernel> kernel;
  if (!params.src.binary.empty()) {
    TF_ASSIGN_OR_RETURN(
        kernel, CreateKernel(kernel_name_, args_.size(), params.src.binary,
                             params.executor, shmem_bytes_));

  } else {
    TF_ASSIGN_OR_RETURN(
        kernel, CreateKernel(kernel_name_, args_.size(), params.src.text,
                             params.executor, shmem_bytes_));
  }

  absl::MutexLock lock(mutex_);
  kernels_.emplace(params.executor, std::move(kernel));
  return absl::OkStatus();
}

absl::Status LaunchCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  CommandBufferParams* record_params = execute_params.command_buffer_params;
  VLOG(5) << "LaunchCmd: kernel=" << kernel_name_
          << "; shmem_bytes=" << shmem_bytes_;

  se::StreamExecutor* executor = execute_params.stream->parent();
  se::Kernel* kernel = [&] {
    absl::MutexLock lock(mutex_);
    return kernels_[executor].get();
  }();

  if (kernel == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Kernel not loaded on a command buffer executor: ", kernel_name_));
  }

  absl::InlinedVector<se::KernelArgument, 4> kernel_args_variant;
  stream_executor::gpu::TmaMetadata tma_metadata =
      tma_metadata_.value_or(se::gpu::TmaMetadata{});
  for (int idx = 0; idx < args_.size(); ++idx) {
    const BufferAllocation::Slice& arg = args_[idx];
    se::DeviceAddressBase buf =
        execute_params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(5) << "  Arg: " << arg << ": " << buf.opaque();

    if (auto it = tma_metadata.arg_index_to_tma_info.find(idx);
        it != tma_metadata.arg_index_to_tma_info.end()) {
      // TMA descriptor argument.
      stream_executor::gpu::TmaDescriptor tma_desc = it->second;
      TF_ASSIGN_OR_RETURN(se::TensorMap tensor_map,
                          executor->CreateTensorMap(tma_desc, buf.opaque()));
      VLOG(5) << "  Using TensorMap for arg #" << idx << ": "
              << tma_desc.ToString();
      kernel_args_variant.push_back(std::move(tensor_map));
    } else {
      // Buffer argument.
      kernel_args_variant.push_back(buf);
    }
  }

  TF_ASSIGN_OR_RETURN(
      auto kernel_args,
      se::PackKernelArgs(absl::MakeConstSpan(kernel_args_variant),
                         shmem_bytes_));

  return HandleCmdCreateOrUpdate(
      *record_params,
      [&] {
        return record_params->command_buffer->CreateLaunch(
            dims_.thread_counts_per_block(), dims_.block_counts(), *kernel,
            *kernel_args, CommandBufferDependencies(*record_params),
            command_buffer_priority());
      },
      [&](const se::CommandBuffer::Command* command) {
        return record_params->command_buffer->UpdateLaunch(
            command, dims_.thread_counts_per_block(), dims_.block_counts(),
            *kernel, *kernel_args);
      });
}

CommandThunk::BufferUseVector LaunchCmd::buffer_uses() const {
  BufferUseVector buffers;
  for (int32_t i = 0; i < args_.size(); ++i) {
    buffers.emplace_back(args_[i], args_access_[i]);
  }
  return buffers;
}

//===----------------------------------------------------------------------===//
// CustomKernelLaunchCmd
//===----------------------------------------------------------------------===//

CustomKernelLaunchCmd::CustomKernelLaunchCmd(
    absl::Span<const BufferAllocation::Slice> args,
    absl::Span<const MemoryAccess> args_access, CustomKernel custom_kernel)
    : CommandThunk(Thunk::Kind::kCustomKernel, Thunk::ThunkInfo()),
      args_(args.begin(), args.end()),
      args_access_(args_access.begin(), args_access.end()),
      custom_kernel_(std::move(custom_kernel)) {}

absl::Status CustomKernelLaunchCmd::Initialize(
    const Thunk::InitializeParams& params) {
  {
    absl::MutexLock lock(mutex_);
    if (kernels_.contains(params.executor)) {
      return absl::OkStatus();
    }
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::Kernel> kernel,
      params.executor->LoadKernel(custom_kernel_.kernel_spec()));

  absl::MutexLock lock(mutex_);
  kernels_.emplace(params.executor, std::move(kernel));
  return absl::OkStatus();
}

absl::Status CustomKernelLaunchCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  CommandBufferParams* record_params = execute_params.command_buffer_params;
  VLOG(5) << "CustomKernelLaunchCmd: custom_kernel=" << custom_kernel_.name();

  se::Kernel* kernel = [&] {
    absl::MutexLock lock(mutex_);
    return kernels_[execute_params.stream->parent()].get();
  }();

  if (kernel == nullptr) {
    return absl::InternalError(
        absl::StrCat("Custom kernel not loaded on a command buffer executor: ",
                     custom_kernel_.name()));
  }

  absl::InlinedVector<se::DeviceAddressBase, 4> buffers;
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceAddressBase buf =
        execute_params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(5) << "  Arg: " << arg << ": " << buf.opaque();
    buffers.push_back(buf);
  }

  se::KernelArgsDeviceMemoryArray kernel_args(
      buffers, custom_kernel_.shared_memory_bytes());

  return HandleCmdCreateOrUpdate(
      *record_params,
      [&] {
        return record_params->command_buffer->CreateLaunch(
            custom_kernel_.thread_dims(), custom_kernel_.block_dims(), *kernel,
            kernel_args, CommandBufferDependencies(*record_params),
            command_buffer_priority());
      },
      [&](const se::CommandBuffer::Command* command) {
        return record_params->command_buffer->UpdateLaunch(
            command, custom_kernel_.thread_dims(), custom_kernel_.block_dims(),
            *kernel, kernel_args);
      });
}

CommandThunk::BufferUseVector CustomKernelLaunchCmd::buffer_uses() const {
  BufferUseVector buffers;
  for (int32_t i = 0; i < args_.size(); ++i) {
    buffers.emplace_back(args_[i], args_access_[i]);
  }
  return buffers;
}

//===----------------------------------------------------------------------===//
// MemcpyDeviceToDeviceCmd
//===----------------------------------------------------------------------===//

MemcpyDeviceToDeviceCmd::MemcpyDeviceToDeviceCmd(ShapedSlice dst,
                                                 ShapedSlice src,
                                                 int64_t num_bytes)
    : CommandThunk(Thunk::Kind::kCopy, Thunk::ThunkInfo()),
      dst_(dst),
      src_(src),
      num_bytes_(num_bytes) {
  CHECK_EQ(ShapeUtil::ByteSizeOfElements(src_.shape),
           ShapeUtil::ByteSizeOfElements(dst_.shape));
  CHECK_LE(num_bytes, dst_.slice.size());
  CHECK_LE(num_bytes, src_.slice.size());
  CHECK_GE(src_.slice.size(), ShapeUtil::ByteSizeOf(src_.shape));
}

absl::Status MemcpyDeviceToDeviceCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  CommandBufferParams* record_params = execute_params.command_buffer_params;
  se::DeviceAddressBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_.slice);
  se::DeviceAddressBase src =
      execute_params.buffer_allocations->GetDeviceAddress(src_.slice);

  VLOG(5) << "MemcpyDeviceToDeviceCmd: num_bytes = " << num_bytes_;
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";
  VLOG(5) << "  Src: " << src_ << " (" << src.opaque() << ")";

  if (num_bytes_ == 0) {
    VLOG(5) << "Skip recording MemcpyDeviceToDeviceCmd command of 0 bytes";
    return absl::OkStatus();
  }

  return HandleCmdCreateOrUpdate(
      *record_params,
      [&] {
        return record_params->command_buffer->CreateMemcpyD2D(
            &dst, src, num_bytes_, CommandBufferDependencies(*record_params));
      },
      [&](const se::CommandBuffer::Command* command) {
        return record_params->command_buffer->UpdateMemcpyD2D(command, &dst,
                                                              src, num_bytes_);
      });
}

CommandThunk::BufferUseVector MemcpyDeviceToDeviceCmd::buffer_uses() const {
  return {BufferUse::Write(dst_.slice, dst_.shape),
          BufferUse::Read(src_.slice, src_.shape)};
}

//===----------------------------------------------------------------------===//
// MemzeroCmd
//===----------------------------------------------------------------------===//

MemzeroCmd::MemzeroCmd(ShapedSlice dst)
    : CommandThunk(Thunk::Kind::kMemzero, Thunk::ThunkInfo()), dst_(dst) {}

absl::Status MemzeroCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  CommandBufferParams* record_params = execute_params.command_buffer_params;
  se::DeviceAddressBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_.slice);

  VLOG(5) << "MemzeroCmd:";
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";

  if (dst_.slice.size() == 0) {
    VLOG(5) << "Skip recording MemzeroCmd command of 0 bytes";
    return absl::OkStatus();
  }

  return HandleCmdCreateOrUpdate(
      *record_params,
      [&] {
        return record_params->command_buffer->CreateMemset(
            &dst, uint8_t{0},
            /*num_elements=*/dst_.slice.size(),
            CommandBufferDependencies(*record_params));
      },
      [&](const se::CommandBuffer::Command* command) {
        return record_params->command_buffer->UpdateMemset(
            command, &dst, uint8_t{0},
            /*num_elements=*/dst_.slice.size());
      });
}

CommandThunk::BufferUseVector MemzeroCmd::buffer_uses() const {
  return {BufferUse::Write(dst_.slice, dst_.shape)};
}

//===----------------------------------------------------------------------===//
// Memset32Cmd
//===----------------------------------------------------------------------===//

Memset32Cmd::Memset32Cmd(BufferAllocation::Slice dst, uint32_t bit_pattern)
    : CommandThunk(Thunk::Kind::kMemset32BitValue, Thunk::ThunkInfo()),
      dst_(dst),
      bit_pattern_(bit_pattern) {}

absl::Status Memset32Cmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  CommandBufferParams* record_params = execute_params.command_buffer_params;
  se::DeviceAddressBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_);

  VLOG(5) << "Memset32Cmd: bit_pattern=" << bit_pattern_;
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";

  if (dst_.size() == 0) {
    VLOG(5) << "Skip recording Memset32Cmd command of 0 bytes";
    return absl::OkStatus();
  }

  return HandleCmdCreateOrUpdate(
      *record_params,
      [&] {
        return record_params->command_buffer->CreateMemset(
            &dst, bit_pattern_,
            /*num_elements=*/dst_.size() / sizeof(uint32_t),
            CommandBufferDependencies(*record_params));
      },
      [&](const se::CommandBuffer::Command* command) {
        return record_params->command_buffer->UpdateMemset(
            command, &dst, bit_pattern_,
            /*num_elements=*/dst_.size() / sizeof(uint32_t));
      });
}

CommandThunk::BufferUseVector Memset32Cmd::buffer_uses() const {
  return {BufferUse::Write(dst_)};
}

//===----------------------------------------------------------------------===//
// ChildCmd
//===----------------------------------------------------------------------===//

ChildCmd::ChildCmd(CommandBufferCmdExecutor child_commands)
    : CommandThunk(Thunk::Kind::kSequential, Thunk::ThunkInfo()),
      child_commands_(std::move(child_commands)) {}

bool ChildCmd::command_buffer_requires_initialization() const {
  return child_commands_.command_buffer_requires_initialization();
}

bool ChildCmd::command_buffer_force_update() const {
  return child_commands_.command_buffer_force_update();
}

CommandThunk::BufferUseVector ChildCmd::buffer_uses() const {
  return {child_commands_.buffer_uses().begin(),
          child_commands_.buffer_uses().end()};
}

absl::Status ChildCmd::Initialize(const Thunk::InitializeParams& params) {
  TF_RETURN_IF_ERROR(child_commands_.Initialize(params));
  return absl::OkStatus();
}

absl::Status ChildCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  CommandBufferParams* record_params = execute_params.command_buffer_params;
  VLOG(5) << "Record ChildCmd " << child_commands_.size() << " commands";
  auto record_fn = [&](se::CommandBuffer* command_buffer) -> absl::Status {
    CommandBufferParams child_record_params = *record_params;
    child_record_params.command_buffer = command_buffer;
    Thunk::ExecuteParams::ScopedCommandBufferParams scoped(
        execute_params, &child_record_params);
    TF_RETURN_IF_ERROR(child_commands_.ExecuteOnStream(execute_params));
    return absl::OkStatus();
  };
  return HandleCmdCreateOrUpdate(
      *record_params,
      [&] {
        return record_params->command_buffer->CreateChildCommand(
            se::CommandBuffer::ChildCommandType::kMoved,
            execute_params.stream->parent(), record_fn,
            CommandBufferDependencies(*record_params));
      },
      [&](const se::CommandBuffer::Command* command) {
        return record_params->command_buffer->UpdateChildCommand(
            se::CommandBuffer::ChildCommandType::kMoved, command, record_fn);
      });
}

//===----------------------------------------------------------------------===//
// CaseCmd
//===----------------------------------------------------------------------===//

CaseCmd::CaseCmd(ShapedSlice index,
                 std::vector<CommandBufferCmdExecutor> branches)
    : CommandThunk(Thunk::Kind::kConditional, Thunk::ThunkInfo()),
      index_(index),
      index_is_bool_(index.shape.element_type() == PRED),
      branches_(std::move(branches)) {}

absl::Status CaseCmd::Initialize(const Thunk::InitializeParams& params) {
  for (auto& branch : branches_) {
    TF_RETURN_IF_ERROR(branch.Initialize(params));
  }
  return absl::OkStatus();
}

absl::Status CaseCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  CommandBufferParams* record_params = execute_params.command_buffer_params;
  se::DeviceAddressBase index =
      execute_params.buffer_allocations->GetDeviceAddress(index_.slice);

  VLOG(5) << "CaseCmd:";
  VLOG(5) << "  index: " << index_ << " (" << index.opaque() << ")";

  return HandleCmdCreateOrUpdate(
      *record_params,
      [&] {
        if (index_is_bool_) {
          return record_params->command_buffer->CreateCase(
              se::DeviceAddress<bool>(index),
              CreateCommands(branches_, &execute_params),
              CommandBufferDependencies(*record_params));
        }
        return record_params->command_buffer->CreateCase(
            se::DeviceAddress<int32_t>(index),
            CreateCommands(branches_, &execute_params),
            CommandBufferDependencies(*record_params));
      },
      [&](const se::CommandBuffer::Command* command) {
        if (index_is_bool_) {
          return record_params->command_buffer->UpdateCase(
              command, se::DeviceAddress<bool>(index),
              UpdateCommands(branches_, &execute_params));
        }
        return record_params->command_buffer->UpdateCase(
            command, se::DeviceAddress<int32_t>(index),
            UpdateCommands(branches_, &execute_params));
      });
}

bool CaseCmd::command_buffer_requires_initialization() const {
  return absl::c_any_of(branches_, [](const auto& seq) {
    return seq.command_buffer_requires_initialization();
  });
}

bool CaseCmd::command_buffer_force_update() const {
  return absl::c_any_of(branches_, [](const auto& seq) {
    return seq.command_buffer_force_update();
  });
}

CommandThunk::BufferUseVector CaseCmd::buffer_uses() const {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(BufferUse::Read(index_.slice, index_.shape));
  for (auto& branch : branches_) {
    buffers.insert(branch.buffer_uses().begin(), branch.buffer_uses().end());
  }
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// WhileCmd
//===----------------------------------------------------------------------===//

WhileCmd::WhileCmd(BufferAllocation::Slice pred,
                   CommandBufferCmdExecutor cond_commands,
                   CommandBufferCmdExecutor body_commands,
                   std::optional<int64_t> trip_count, bool enable_loop_unroll)
    : CommandThunk(Thunk::Kind::kWhile, Thunk::ThunkInfo()),
      pred_(pred),
      cond_commands_(std::move(cond_commands)),
      body_commands_(std::move(body_commands)),
      trip_count_(trip_count),
      enable_loop_unroll_(enable_loop_unroll) {}

absl::Status WhileCmd::Initialize(const Thunk::InitializeParams& params) {
  TF_RETURN_IF_ERROR(cond_commands_.Initialize(params));
  TF_RETURN_IF_ERROR(body_commands_.Initialize(params));
  enable_loop_unroll_ = true;
  if (enable_loop_unroll_ &&
      body_commands_.command_buffer_support_loop_unroll() &&
      cond_commands_.command_buffer_support_loop_unroll() &&
      trip_count_ != std::nullopt) {
    is_unrolled_loop_ = true;
  }
  VLOG(3) << "while command trip_count: " << trip_count_.value_or(-1);
  return absl::OkStatus();
}

absl::Status WhileCmd::Prepare(const Thunk::PrepareParams& params) {
  TF_RETURN_IF_ERROR(cond_commands_.Prepare(params));
  TF_RETURN_IF_ERROR(body_commands_.Prepare(params));
  return absl::OkStatus();
}

absl::Status WhileCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  CommandBufferParams* record_params = execute_params.command_buffer_params;
  se::DeviceAddressBase pred =
      execute_params.buffer_allocations->GetDeviceAddress(pred_);

  VLOG(5) << "WhileCmd: cond_commands=" << cond_commands_.size()
          << " body_commands=" << body_commands_.size();
  VLOG(5) << "  pred: " << pred_ << " (" << pred.opaque() << ")";
  if (is_unrolled_loop_) {
    auto record_fn =
        [&](se::CommandBuffer* child_command_buffer) -> absl::Status {
      // When the loop is unrolled, we need to record the body commands for
      // `trip_count` times into child_command_buffer, and implement the While
      // command as a child command.
      VLOG(3) << "Recording unrolled loop with trip_count: "
              << trip_count_.value();

      // Unroll the while loop body for `trip_count` times.
      // Unrolled execution sequence: cond -> body -> cond -> body -> ...
      // In the unrolled pattern, we still need to run the cond commands because
      // body commands might depends on the value of index variable that is
      // updated by condition commands.
      CommandBufferParams unroll_record_params = *record_params;
      unroll_record_params.command_buffer = child_command_buffer;
      unroll_record_params.external_dependencies = {};
      Thunk::ExecuteParams::ScopedCommandBufferParams scoped(
          execute_params, &unroll_record_params);
      for (int64_t i = 0; i < trip_count_.value(); ++i) {
        unroll_record_params.unroll_iteration = i;
        unroll_record_params.is_finalize = false;
        TF_RETURN_IF_ERROR(cond_commands_.ExecuteOnStream(execute_params));
        unroll_record_params.external_dependencies =
            cond_commands_.SinkCommands(unroll_record_params);
        unroll_record_params.is_finalize = (i == trip_count_.value() - 1);
        TF_RETURN_IF_ERROR(body_commands_.ExecuteOnStream(execute_params));
        unroll_record_params.external_dependencies =
            body_commands_.SinkCommands(unroll_record_params);
      }
      return absl::OkStatus();
    };
    return HandleCmdCreateOrUpdate(
        *record_params,
        [&] {
          return record_params->command_buffer->CreateChildCommand(
              se::CommandBuffer::ChildCommandType::kMoved,
              execute_params.stream->parent(), record_fn,
              CommandBufferDependencies(*record_params));
        },
        [&](const se::CommandBuffer::Command* command) {
          return record_params->command_buffer->UpdateChildCommand(
              se::CommandBuffer::ChildCommandType::kMoved, command, record_fn);
        });
  }
  return HandleCmdCreateOrUpdate(
      *record_params,
      [&] {
        return record_params->command_buffer->CreateWhile(
            se::DeviceAddress<bool>(pred),
            CreateCommands(&cond_commands_, &execute_params),
            CreateCommands(&body_commands_, &execute_params),
            CommandBufferDependencies(*record_params));
      },
      [&](const se::CommandBuffer::Command* command) {
        return record_params->command_buffer->UpdateWhile(
            command, se::DeviceAddress<bool>(pred),
            UpdateCommands(&cond_commands_, &execute_params),
            UpdateCommands(&body_commands_, &execute_params));
      });
}

bool WhileCmd::command_buffer_requires_initialization() const {
  return (cond_commands_.command_buffer_requires_initialization() ||
          body_commands_.command_buffer_requires_initialization());
}

bool WhileCmd::command_buffer_force_update() const {
  return cond_commands_.command_buffer_force_update() ||
         body_commands_.command_buffer_force_update();
}

CommandThunk::BufferUseVector WhileCmd::buffer_uses() const {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(BufferUse::Read(pred_, ShapeUtil::MakeShape(PRED, {})));
  buffers.insert(cond_commands_.buffer_uses().begin(),
                 cond_commands_.buffer_uses().end());
  buffers.insert(body_commands_.buffer_uses().begin(),
                 body_commands_.buffer_uses().end());
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// GemmCmd
//===----------------------------------------------------------------------===//

GemmCmd::GemmCmd(GemmConfig config, const BufferAllocation::Slice& lhs_buffer,
                 const BufferAllocation::Slice& rhs_buffer,
                 const BufferAllocation::Slice& output_buffer,
                 std::optional<BufferAllocation::Slice> workspace,
                 bool deterministic)
    : TracedCommandBufferCmd(Thunk::Kind::kGemm),
      config_(std::move(config)),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      output_buffer_(output_buffer),
      workspace_(workspace),
      deterministic_(deterministic) {}

absl::Status GemmCmd::Initialize(const Thunk::InitializeParams& params) {
  if (!params.stream->parent()->AsBlas()) {
    return absl::InternalError("Failed to initialize BLAS support for GemmCmd");
  }
  return absl::OkStatus();
}

absl::Status GemmCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  se::DeviceAddressBase lhs =
      execute_params.buffer_allocations->GetDeviceAddress(lhs_buffer_);
  se::DeviceAddressBase rhs =
      execute_params.buffer_allocations->GetDeviceAddress(rhs_buffer_);
  se::DeviceAddressBase out =
      execute_params.buffer_allocations->GetDeviceAddress(output_buffer_);

  se::DeviceAddressBase workspace(/*opaque=*/nullptr, /*size=*/0);
  if (workspace_.has_value()) {
    workspace =
        execute_params.buffer_allocations->GetDeviceAddress(workspace_.value());
  }

  VLOG(5) << "GemmCmd: deterministic=" << deterministic_;
  VLOG(5) << "  Lhs: " << lhs_buffer_ << " (" << lhs.opaque() << ")";
  VLOG(5) << "  Lhs: " << rhs_buffer_ << " (" << rhs.opaque() << ")";
  VLOG(5) << "  Out: " << output_buffer_ << " (" << out.opaque() << ")";
  VLOG(5) << "  Workspace: " << workspace.opaque();

  return RecordTracedCommand(execute_params, [&](se::Stream* stream) {
    return RunGemm(config_, lhs, rhs, out, workspace, deterministic_, stream);
  });
}

CommandThunk::BufferUseVector GemmCmd::buffer_uses() const {
  CommandThunk::BufferUseVector res{
      BufferUse::Read(lhs_buffer_, config_.lhs_layout.ToShape()),
      BufferUse::Read(rhs_buffer_, config_.rhs_layout.ToShape()),
      BufferUse::Write(output_buffer_, config_.output_layout.ToShape()),
  };
  if (workspace_.has_value()) {
    res.push_back(BufferUse::Write(
        *workspace_, ShapeUtil::MakeShape(S8, {workspace_->size()})));
  }
  return res;
}

//===----------------------------------------------------------------------===//
// CublasLtCmd
//===----------------------------------------------------------------------===//

CublasLtCmd::CublasLtCmd(const CublasLtMatmulThunk& matmul_thunk)
    : TracedCommandBufferCmd(Thunk::Kind::kCublasLtMatmul),
      CublasLtMatmulThunk(matmul_thunk) {}

absl::Status CublasLtCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  // This call is required to make sure matmul plan is already created and
  // cached before recording the command buffer.
  TF_RETURN_IF_ERROR(GetCachedMatmulPlan(execute_params).status());

  VLOG(5) << "CublasLtCmd:";
  VLOG(5) << "  a_buffer: " << a_.ToString();
  VLOG(5) << "  b_buffer: " << b_.ToString();
  VLOG(5) << "  c_buffer: " << c_.ToString();
  VLOG(5) << "  d_buffer: " << d_.ToString();
  VLOG(5) << "  bias_buffer: " << bias_.ToString();
  VLOG(5) << "  aux_buffer: " << aux_.ToString();
  VLOG(5) << "  a_scale_buffer: " << a_scale_.ToString();
  VLOG(5) << "  b_scale_buffer: " << b_scale_.ToString();
  VLOG(5) << "  c_scale_buffer: " << c_scale_.ToString();
  VLOG(5) << "  d_scale_buffer: " << d_scale_.ToString();
  VLOG(5) << "  d_amax_buffer: " << d_amax_.ToString();
  // workspace buffer is guaranteed to be non-null here.
  VLOG(5) << "  workspace_buffer: " << workspace_->ToString();

  return RecordTracedCommand(execute_params, [&](se::Stream* stream) {
    return ExecuteOnStreamInternal(stream, execute_params);
  });
}

CommandThunk::BufferUseVector CublasLtCmd::buffer_uses() const {
  BufferUseVector buffer_usage;
  buffer_usage.reserve(13);
  buffer_usage.push_back(BufferUse::Read(a_));
  buffer_usage.push_back(BufferUse::Read(b_));
  buffer_usage.push_back(BufferUse::Read(c_));
  buffer_usage.push_back(BufferUse::Write(d_));
  buffer_usage.push_back(BufferUse::Write(*workspace_));

  if (bias_.allocation() != nullptr) {
    buffer_usage.push_back(BufferUse::Read(bias_));
  }
  if (a_scale_.allocation() != nullptr) {
    buffer_usage.push_back(BufferUse::Read(a_scale_));
  }
  if (b_scale_.allocation() != nullptr) {
    buffer_usage.push_back(BufferUse::Read(b_scale_));
  }
  if (c_scale_.allocation() != nullptr) {
    buffer_usage.push_back(BufferUse::Read(c_scale_));
  }
  if (d_scale_.allocation() != nullptr) {
    buffer_usage.push_back(BufferUse::Read(d_scale_));
  }
  if (aux_.allocation() != nullptr) {
    buffer_usage.push_back(BufferUse::Write(aux_));
  }
  if (d_amax_.allocation() != nullptr) {
    buffer_usage.push_back(BufferUse::Read(d_amax_));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CuDnnCmd
//===----------------------------------------------------------------------===//

CuDnnCmd::CuDnnCmd(absl::Span<const BufferAllocation::Slice> args,
                   const std::shared_ptr<se::dnn::LazyDnnGraph> graph)
    : TracedCommandBufferCmd(Thunk::Kind::kCuDnn),
      args_(args.cbegin(), args.cend()),
      graph_(graph) {}

absl::Status CuDnnCmd::Initialize(const Thunk::InitializeParams& params) {
  if (!params.stream->parent()->AsDnn()) {
    return absl::InternalError("Failed to initialize DNN support for CuDnnCmd");
  }
  return absl::OkStatus();
}

absl::Status CuDnnCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  CommandBufferParams* record_params = execute_params.command_buffer_params;
  CHECK(graph_ != nullptr);
  std::vector<se::DeviceAddressBase> operands;
  operands.reserve(args_.size());
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceAddressBase buf =
        execute_params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(5) << "  Arg: " << arg << ": " << buf.opaque();
    operands.push_back(buf);
  }
  TF_ASSIGN_OR_RETURN(
      const bool supports_explicit,
      graph_->get()->SupportsExplicitCommandBufferConstruction());
  if (supports_explicit) {
    return HandleCmdCreateOrUpdate(
        *record_params,
        [&] {
          return record_params->command_buffer->CreateDnnGraphCommand(
              *graph_->get(), *execute_params.stream,
              absl::Span<se::DeviceAddressBase>(operands),
              CommandBufferDependencies(*record_params));
        },
        [&](const se::CommandBuffer::Command* command) {
          return record_params->command_buffer->UpdateDnnGraphCommand(
              command, *graph_->get(), *execute_params.stream,
              absl::Span<se::DeviceAddressBase>(operands));
        });
  }
  return RecordTracedCommand(execute_params, [&](se::Stream* stream) {
    return graph_->get()->Execute(
        *stream, absl::Span<se::DeviceAddressBase>(operands),
        execute_params.collective_params->local_device_id.value());
  });
}

CommandThunk::BufferUseVector CuDnnCmd::buffer_uses() const {
  CommandThunk::BufferUseVector buffer_usage;
  buffer_usage.reserve(args_.size());
  for (int i = 0; i < args_.size() - 1; ++i) {
    buffer_usage.push_back(BufferUse::Read(args_[i]));
  }
  buffer_usage.push_back(BufferUse::Write(args_.back()));
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CustomCallCmd
//===----------------------------------------------------------------------===//

absl::Status CustomCallCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  if (handler_ == nullptr) {
    return RecordLegacyCustomCall(execute_params);
  }
  return RecordXlaFfiCall(execute_params);
}

namespace {
// Records each buffer associated with each slice into the provided vector.
// Returns an error if any of the slices is missing a buffer allocation.
absl::Status GetBuffers(const Thunk::ExecuteParams& execute_params,
                        absl::Span<const NullableShapedSlice> slices,
                        std::vector<void*>& buffers, absl::string_view label) {
  for (int i = 0; i < slices.size(); ++i) {
    if (!slices[i].has_value()) {
      buffers.push_back(nullptr);
      VLOG(5) << label << i << ": null";
      continue;
    }

    if (!slices[i]->slice.allocation()) {
      return absl::InternalError("custom call input missing buffer allocation");
    }

    auto buffer =
        execute_params.buffer_allocations->GetDeviceAddress(slices[i]->slice)
            .opaque();
    VLOG(5) << label << i << ": " << slices[i]->slice << " (" << buffer << ")";
    buffers.push_back(buffer);
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status CustomCallCmd::RecordLegacyCustomCall(
    const Thunk::ExecuteParams& execute_params) {
  CommandBufferParams* record_params = execute_params.command_buffer_params;
  std::vector<void*> buffers;
  buffers.reserve(operands_.size() + results_.size());

  VLOG(5) << "CustomCallCmd: target_name=" << target_name_;
  TF_RETURN_IF_ERROR(
      GetBuffers(execute_params, operands_, buffers, "  Operand "));
  TF_RETURN_IF_ERROR(
      GetBuffers(execute_params, results_, buffers, "  Result "));

  TF_ASSIGN_OR_RETURN(
      auto nested_cmd,
      se::TraceCommandBufferFactory::Create(
          execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, [&](se::Stream* stream) {
            XlaCustomCallStatus custom_call_status;
            call_target_(stream, buffers.data(), opaque_.data(), opaque_.size(),
                         &custom_call_status);
            auto message = CustomCallStatusGetMessage(&custom_call_status);
            if (message) {
              return absl::InternalError(
                  absl::StrCat("CustomCall failed: ", *message));
            }
            return absl::OkStatus();
          }));

  return HandleCmdCreateOrUpdate(
      *record_params,
      [&] {
        return record_params->command_buffer->CreateChildCommand(
            se::CommandBuffer::ChildCommandType::kCloned, *nested_cmd,
            CommandBufferDependencies(*record_params));
      },
      [&](const se::CommandBuffer::Command* command) {
        return record_params->command_buffer->UpdateChildCommand(
            se::CommandBuffer::ChildCommandType::kCloned, command, *nested_cmd);
      });
}

absl::Status CustomCallCmd::RecordXlaFfiCall(
    const Thunk::ExecuteParams& execute_params) {
  CommandBufferParams* record_params = execute_params.command_buffer_params;
  // TODO(ezhulenev): This is not the most optimal approach, as we'll be doing
  // a lot of extra allocation on every call. We have to keep attributes
  // separate from arguments, as they do not change after thunk is
  // constructed.
  ffi::CallFrameBuilder builder(operands_.size(), results_.size());

  VLOG(5) << "CustomCallCmd: target_name=" << target_name_;

  absl::InlinedVector<se::DeviceAddressBase, 4> arguments;
  arguments.reserve(operands_.size());

  for (int i = 0; i < operands_.size(); ++i) {
    const NullableShapedSlice& slice = operands_[i];
    if (!slice.has_value()) {
      arguments.push_back(se::DeviceAddressBase{});
      continue;
    }

    se::DeviceAddressBase buffer =
        execute_params.buffer_allocations->GetDeviceAddress(slice->slice);
    VLOG(5) << "  Operand " << i << ": " << slice->slice << " ("
            << buffer.opaque() << ")";
    arguments.push_back(buffer);
  }

  absl::InlinedVector<se::DeviceAddressBase, 4> results;
  results.reserve(results_.size());

  for (int i = 0; i < results_.size(); ++i) {
    const NullableShapedSlice& slice = results_[i];
    if (!slice.has_value()) {
      results.push_back(se::DeviceAddressBase{});
      continue;
    }

    se::DeviceAddressBase buffer =
        execute_params.buffer_allocations->GetDeviceAddress(slice->slice);
    VLOG(5) << "  Result " << i << ": " << slice->slice << " ("
            << buffer.opaque() << ")";
    results.push_back(buffer);
  }

  // Borrow the FFI call frame from the object pool and update with the actual
  // device memory addresses.
  TF_ASSIGN_OR_RETURN(auto call_frame, call_frames_->GetOrCreate());
  TF_RETURN_IF_ERROR(call_frame->UpdateWithBuffers(arguments, results));

  RunId run_id = execute_params.collective_params->run_id;

  TF_ASSIGN_OR_RETURN(
      auto nested_cmd,
      se::TraceCommandBufferFactory::Create(
          execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, [&](se::Stream* stream) {
            ffi::CallOptions options = {
                run_id,
                execute_params.buffer_allocations->device_ordinal(),
                ffi::CallOptions::GpuOptions{
                    stream,
                    execute_params.buffer_allocations->memory_allocator()},
                /*called_computation=*/nullptr,  // TODO(b/342285364)
                execute_params.ffi_execution_context,
                execution_state_.get()};
            return ffi::Call(handler_, *call_frame, options);
          }));

  return HandleCmdCreateOrUpdate(
      *record_params,
      [&] {
        return record_params->command_buffer->CreateChildCommand(
            se::CommandBuffer::ChildCommandType::kCloned, *nested_cmd,
            CommandBufferDependencies(*record_params));
      },
      [&](const se::CommandBuffer::Command* command) {
        return record_params->command_buffer->UpdateChildCommand(
            se::CommandBuffer::ChildCommandType::kCloned, command, *nested_cmd);
      });
}

CommandThunk::BufferUseVector CustomCallCmd::buffer_uses() const {
  CommandThunk::BufferUseVector buffer_usage;
  for (auto& slices : {operands_, results_}) {
    for (const std::optional<ShapedSlice>& slice : slices) {
      if (slice.has_value()) {
        buffer_usage.push_back(BufferUse::Write(slice->slice));
      }
    }
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CollectiveCmd
//===----------------------------------------------------------------------===//

CollectiveCmd::CollectiveCmd(
    Thunk::Kind kind, CollectiveConfig config,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : CommandThunk(kind, Thunk::ThunkInfo(), se::StreamPriority::Highest),
      config_(std::move(config)),
      async_events_(std::move(async_events)) {}

absl::Status CollectiveCmd::Prepare(const Thunk::PrepareParams& params) {
  TF_RET_CHECK(params.collective_params != nullptr);
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*params.collective_params, config().replica_groups,
                      config().group_mode,
                      AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));
  return params.clique_requests->RequestClique(clique_key);
}

absl::Status CollectiveCmd::RecordTracedCommand(
    const Thunk::ExecuteParams& execute_params,
    absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  CommandBufferParams* record_params = execute_params.command_buffer_params;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::CommandBuffer> nested_cmd,
                      se::TraceCommandBufferFactory::Create(
                          execute_params.stream->parent(),
                          execute_params.command_buffer_trace_stream, trace));

  if (command_buffer_priority() != se::StreamPriority::Default) {
    TF_RETURN_IF_ERROR(nested_cmd->SetPriority(command_buffer_priority()));
  }

  return HandleCmdCreateOrUpdate(
      *record_params,
      [&] {
        return record_params->command_buffer->CreateChildCommand(
            se::CommandBuffer::ChildCommandType::kCloned, *nested_cmd,
            CommandBufferDependencies(*record_params));
      },
      [&](const se::CommandBuffer::Command* command) {
        return record_params->command_buffer->UpdateChildCommand(
            se::CommandBuffer::ChildCommandType::kCloned, command, *nested_cmd);
      });
}

//===----------------------------------------------------------------------===//
// AllReduceCmd
//===----------------------------------------------------------------------===//

AllReduceCmd::AllReduceCmd(
    CollectiveConfig config, ReductionKind reduction_kind,
    absl::Span<const CollectiveThunk::Buffer> buffers,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : CollectiveCmd(Thunk::Kind::kAllReduceStart, std::move(config),
                    std::move(async_events)),
      reduction_kind_(reduction_kind),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status AllReduceCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal)
      << "AllReduceCmd: reduction=" << ReductionKindString(reduction_kind_);

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Src: " << buffers_[i].source_buffer << " ("
        << device_buffers[i].source_buffer.opaque() << ")";
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Dst: " << buffers_[i].destination_buffer << " ("
        << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "AllReduceCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*execute_params.collective_params,
                      config().replica_groups, config().group_mode,
                      AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  return RecordTracedCommand(execute_params, [&](se::Stream* stream) {
    return RunAllReduce(reduction_kind_, device_buffers, *stream, *comm,
                        config().use_symmetric_buffer);
  });
}

CommandThunk::BufferUseVector AllReduceCmd::buffer_uses() const {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer));
    buffer_usage.emplace_back(BufferUse::Write(buffer.destination_buffer));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// ReduceScatterCmd
//===----------------------------------------------------------------------===//

ReduceScatterCmd::ReduceScatterCmd(
    CollectiveConfig config, ReductionKind reduction_kind,
    absl::Span<const CollectiveThunk::Buffer> buffers,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : CollectiveCmd(Thunk::Kind::kReduceScatterStart, std::move(config),
                    std::move(async_events)),
      reduction_kind_(reduction_kind),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status ReduceScatterCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal)
      << "ReduceScatterCmd: reduction=" << ReductionKindString(reduction_kind_);

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Src: " << buffers_[i].source_buffer << " ("
        << device_buffers[i].source_buffer.opaque() << ")";
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Dst: " << buffers_[i].destination_buffer << " ("
        << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "ReduceScatterCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*execute_params.collective_params,
                      config().replica_groups, config().group_mode,
                      AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  return RecordTracedCommand(execute_params, [&](se::Stream* stream) {
    return RunReduceScatter(reduction_kind_, device_buffers, *stream, *comm,
                            config().use_symmetric_buffer);
  });
}

CommandThunk::BufferUseVector ReduceScatterCmd::buffer_uses() const {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer));
    buffer_usage.emplace_back(BufferUse::Write(buffer.destination_buffer));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// AllToAllCmd
//===----------------------------------------------------------------------===//

AllToAllCmd::AllToAllCmd(
    CollectiveConfig config, bool has_split_dimension,
    absl::Span<const CollectiveThunk::Buffer> buffers,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : CollectiveCmd(Thunk::Kind::kAllToAllStart, std::move(config),
                    std::move(async_events)),
      has_split_dimension_(has_split_dimension),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status AllToAllCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal)
      << "AllToAllCmd, has_split_dimension=" << has_split_dimension_;

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Src: " << buffers_[i].source_buffer << " ("
        << device_buffers[i].source_buffer.opaque() << ")";
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Dst: " << buffers_[i].destination_buffer << " ("
        << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "AllToAllCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*execute_params.collective_params,
                      config().replica_groups, config().group_mode,
                      AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  // MemCpy case is not currently supported in CommandBuffer.
  return RecordTracedCommand(execute_params, [&](se::Stream* stream) {
    return RunAllToAll(has_split_dimension_, device_buffers, *stream, *comm,
                       config().use_symmetric_buffer);
  });
}

CommandThunk::BufferUseVector AllToAllCmd::buffer_uses() const {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer));
    buffer_usage.emplace_back(BufferUse::Write(buffer.destination_buffer));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// AllGatherCmd
//===----------------------------------------------------------------------===//

AllGatherCmd::AllGatherCmd(
    CollectiveConfig config, absl::Span<const CollectiveThunk::Buffer> buffers,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : CollectiveCmd(Thunk::Kind::kAllGatherStart, std::move(config),
                    std::move(async_events)),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status AllGatherCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal) << "AllGatherCmd:";

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Src: " << buffers_[i].source_buffer << " ("
        << device_buffers[i].source_buffer.opaque() << ")";
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Dst: " << buffers_[i].destination_buffer << " ("
        << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "AllGatherCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*execute_params.collective_params,
                      config().replica_groups, config().group_mode,
                      AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  return RecordTracedCommand(execute_params, [&](se::Stream* stream) {
    return RunAllGather(device_buffers, *stream, *comm,
                        config().use_symmetric_buffer);
  });
}

CommandThunk::BufferUseVector AllGatherCmd::buffer_uses() const {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer));
    buffer_usage.emplace_back(BufferUse::Write(buffer.destination_buffer));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CollectiveBroadcastCmd
//===----------------------------------------------------------------------===//

CollectiveBroadcastCmd::CollectiveBroadcastCmd(
    CollectiveConfig config, absl::Span<const CollectiveThunk::Buffer> buffers,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : CollectiveCmd(Thunk::Kind::kCollectiveBroadcastStart, std::move(config),
                    std::move(async_events)),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status CollectiveBroadcastCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal) << "CollectiveBroadcastCmd:";

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Src: " << buffers_[i].source_buffer << " ("
        << device_buffers[i].source_buffer.opaque() << ")";
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Dst: " << buffers_[i].destination_buffer << " ("
        << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "CollectiveBroadcastCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*execute_params.collective_params,
                      config().replica_groups, config().group_mode,
                      AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  return RecordTracedCommand(execute_params, [&](se::Stream* stream) {
    return RunCollectiveBroadcast(device_buffers, *stream, *comm);
  });
}

CommandThunk::BufferUseVector CollectiveBroadcastCmd::buffer_uses() const {
  BufferUseVector buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer));
    buffer_usage.emplace_back(BufferUse::Write(buffer.destination_buffer));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CollectivePermuteCmd
//===----------------------------------------------------------------------===//

CollectivePermuteCmd::CollectivePermuteCmd(
    CollectiveConfig config, P2PConfig p2p_config,
    absl::Span<const CollectiveThunk::Buffer> buffers,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : CollectiveCmd(Thunk::Kind::kCollectivePermuteStart, std::move(config),
                    std::move(async_events)),
      p2p_config_(std::move(p2p_config)),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status CollectivePermuteCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal) << "CollectivePermuteCmd:";

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Src: " << buffers_[i].source_buffer << " ("
        << device_buffers[i].source_buffer.opaque() << ")";
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Dst: " << buffers_[i].destination_buffer << " ("
        << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "CollectivePermuteCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*execute_params.collective_params,
                      config().replica_groups, config().group_mode,
                      AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  std::string device_string =
      CollectiveThunk::GetDeviceString(*execute_params.collective_params);
  bool use_symmetric_buffer = config().use_symmetric_buffer;

  TF_ASSIGN_OR_RETURN(
      const int64_t current_id,
      GetCollectiveCurrentId(execute_params.collective_params, p2p_config_));

  const P2PConfig::SourceTargetMapEntry source_target =
      P2PConfig::GetSourceTarget(p2p_config_.id_to_source_target, current_id);

  // MemCpy case is not currently supported in CommandBuffer.
  return RecordTracedCommand(execute_params, [&](se::Stream* stream) {
    return RunCollectivePermute(source_target, device_buffers, *stream, *comm,
                                device_string, current_id,
                                /*use_memcpy=*/false,
                                /*recv_ptr_map=*/nullptr, use_symmetric_buffer);
  });
}

CommandThunk::BufferUseVector CollectivePermuteCmd::buffer_uses() const {
  BufferUseVector buffer_usage;
  for (const CollectiveThunk::Buffer& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer));
    buffer_usage.emplace_back(BufferUse::Write(buffer.destination_buffer));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// DynamicSliceFusionCmd
//===----------------------------------------------------------------------===//

DynamicSliceFusionCmd::DynamicSliceFusionCmd(
    CommandBufferCmdExecutor embedded_commands,
    std::vector<std::optional<BufferAllocation::Slice>> arguments,
    std::vector<BufferAllocation> fake_allocations,
    std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>> offsets,
    std::vector<std::optional<Shape>> orig_shapes,
    std::vector<std::optional<Shape>> sliced_shapes,
    std::vector<std::optional<PrimitiveType>> offset_primitive_types,
    std::optional<
        const DynamicSliceThunk::OffsetAsFunctionOfIndvarModulesMetadata*>
        offset_as_function_of_indvar_metadata)
    : CommandThunk(Thunk::Kind::kDynamicSlice, Thunk::ThunkInfo()),
      embedded_commands_(std::move(embedded_commands)),
      fake_allocations_(std::move(fake_allocations)),
      offset_as_function_of_indvar_metadata_(
          std::move(offset_as_function_of_indvar_metadata)) {
  // Zip all arguments together to create a list of SliceDef.
  for (auto [arg, offset, orig_shape, sliced_shape, offset_primitive_type] :
       llvm::zip_equal(arguments, offsets, orig_shapes, sliced_shapes,
                       offset_primitive_types)) {
    slices_.push_back(DynamicSliceThunk::SliceDef{
        std::move(arg),
        std::move(offset),
        std::move(orig_shape),
        std::move(sliced_shape),
        std::move(offset_primitive_type),
    });
  }

  for (auto [argument_idx, slice] : llvm::enumerate(slices_)) {
    embeded_to_origin_slice_map_[argument_idx] = slice.embedded_thunk_argument;
  }

  // Find how many offsets we might have to transfer from device to host and
  // pre-compute host allocation requirements.
  for (DynamicSliceThunk::SliceDef& slice : slices_) {
    offsets_allocs_base_.push_back(offsets_allocs_size_);
    if (slice.sliced_shape.has_value()) {
      offsets_allocs_size_ +=
          slice.sliced_shape->dimensions().size() * sizeof(int64_t);
    }
  }
}

// Force update the command when there is any non-constant value slice offset,
// because the memory address might changed if the offset is loop
// iterator or operator outputs even if the parent command's memory pointers
// do not change.
bool DynamicSliceFusionCmd::command_buffer_requires_initialization() const {
  return !absl::c_all_of(slices_, [](const DynamicSliceThunk::SliceDef& slice) {
    if (!slice.offsets.has_value()) {
      return true;
    }
    return absl::c_all_of(slice.offsets.value(),
                          [](DynamicSliceThunk::Offset offset) {
                            return std::holds_alternative<int64_t>(offset);
                          });
  });
}

absl::Status DynamicSliceFusionCmd::Initialize(
    const Thunk::InitializeParams& params) {
  TF_RETURN_IF_ERROR(embedded_commands_.Initialize(params));
  absl::MutexLock lock(mutex_);
  if (offsets_allocs_.contains(params.executor)) {
    return absl::OkStatus();
  }

  XLA_VLOG_DEVICE(2, params.executor->device_ordinal())
      << "Allocate " << offsets_allocs_size_
      << " bytes for transferring offsets on executor: " << params.executor;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::MemoryAllocation> allocation,
      params.executor->HostMemoryAllocate(offsets_allocs_size_));
  offsets_allocs_.emplace(params.executor, std::move(allocation));
  return absl::OkStatus();
}

absl::Status DynamicSliceFusionCmd::Prepare(
    const Thunk::PrepareParams& params) {
  for (DynamicSliceThunk::SliceDef& slice : slices_) {
    VLOG(3) << "DynamicSliceFusionCmd: slice: " << slice.ToString();
    if (slice.offsets.has_value()) {
      TF_RET_CHECK(slice.embedded_thunk_argument.has_value());
      TF_RET_CHECK(slice.orig_shape.has_value());
      TF_RET_CHECK(slice.sliced_shape.has_value());
      TF_RET_CHECK(slice.offset_primitive_type.has_value());
      TF_RET_CHECK(slice.orig_shape->IsArray());
      TF_RET_CHECK(slice.sliced_shape->IsArray());
      TF_RET_CHECK(slice.offsets->size() ==
                   slice.orig_shape->dimensions().size());
      TF_RET_CHECK(slice.sliced_shape->dimensions().size() ==
                   slice.orig_shape->dimensions().size());
    }
  }
  TF_RETURN_IF_ERROR(embedded_commands_.Prepare(params));
  if (offset_as_function_of_indvar_metadata_ != std::nullopt) {
    Indvar(this) =
        HloEvaluator()
            .Evaluate(
                *offset_as_function_of_indvar_metadata_.value()->indvar_init,
                {})
            .value();
    VLOG(3) << "Indvar init module: "
            << offset_as_function_of_indvar_metadata_.value()
                   ->indvar_init->ToString();
    VLOG(3) << "Indvar update module: "
            << offset_as_function_of_indvar_metadata_.value()
                   ->indvar_update->ToString();
    VLOG(3) << "Indvar value initialized to :" << Indvar(this).ToString();
  }
  return absl::OkStatus();
}

absl::Status DynamicSliceFusionCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  CommandBufferParams& record_params = *execute_params.command_buffer_params;
  se::Stream& stream = *execute_params.stream;

  const BufferAllocations& orig_allocations =
      *execute_params.buffer_allocations;
  absl::InlinedVector<se::DeviceAddressBase, 8> slice_buffers(
      slices_.size(), se::DeviceAddressBase());

  // Get memory allocation for copying offsets from device.
  int64_t* offsets_alloc = [&] {
    absl::MutexLock lock(mutex_);
    return reinterpret_cast<int64_t*>(
        offsets_allocs_.at(stream.parent())->opaque());
  }();

  auto offset_value = [&](int64_t arg_idx, int64_t offset_idx) -> int64_t& {
    return offsets_alloc[offsets_allocs_base_.at(arg_idx) + offset_idx];
  };

  VLOG(2) << "Execute dynamic slice thunk: slices=" << slices_.size();
  for (auto [argument_idx, slice] : llvm::enumerate(slices_)) {
    // Skip arguments that do not have buffer slices (tokens).
    if (!slice.embedded_thunk_argument.has_value()) {
      continue;
    }

    // `argument_buffer` will contain the original offset for slice
    // `argument_slice` within `orig_allocations`
    se::DeviceAddressBase argument_buffer =
        orig_allocations.GetDeviceAddress(*slice.embedded_thunk_argument);

    // If argument is not sliced, just use the original buffer.
    if (!slice.offsets.has_value()) {
      slice_buffers[argument_idx] = argument_buffer;
      continue;
    }

    const Shape& src_shape = *slice.orig_shape;
    const Shape& dst_shape = *slice.sliced_shape;

    absl::InlinedVector<int64_t, 4> slice_starts;
    slice_starts.reserve(dst_shape.dimensions().size());

    // Number of issues d2h transfers to copy offset values from device to
    // host.
    int64_t num_transfers = 0;

    // Get offset for `argument_idx`-th argument, which has
    // `dst_shape.dimensions_size()` components.
    for (auto [offset_idx, values] : llvm::enumerate(llvm::zip(
             *slice.offsets, src_shape.dimensions(), dst_shape.dimensions()))) {
      auto [offset, src_dim, dst_dim] = values;
      if (int64_t* const_offset = std::get_if<int64_t>(&offset)) {
        // Forward slice offsets that are known constant values
        VLOG(2) << "  - arg " << argument_idx << "[" << offset_idx
                << "]: constant offset = " << *const_offset;
        offset_value(argument_idx, offset_idx) = *const_offset;

      } else if (HloModule** offset_module = std::get_if<HloModule*>(&offset)) {
        TF_ASSIGN_OR_RETURN(
            Literal offset,
            HloEvaluator().Evaluate(**offset_module, {&Indvar(this)}));
        auto offset_int = LiteralUtil::LiteralAsScalarInt64(offset);
        if (offset_int.has_value()) {
          offset_value(argument_idx, offset_idx) = *offset_int;
        } else {
          return absl::InternalError(
              absl::StrFormat("Unhandled type returned from offset module: %s",
                              offset.shape().ToString()));
        }
        VLOG(2) << "Offset value = " << offset_value(argument_idx, offset_idx);
      } else {
        // Transfer slice offset value from device to host.
        auto alloc_slice = std::get<BufferAllocation::Slice>(offset);
        VLOG(2) << "  - arg " << argument_idx << "[" << offset_idx
                << "]: transfer offset from device " << alloc_slice.ToString();

        se::DeviceAddressBase offset_src =
            orig_allocations.GetDeviceAddress(alloc_slice);
        int64_t* offset_dst = &offset_value(argument_idx, offset_idx);

        // Copy the `offset_idx`-th component of the offset for the
        // `argument_idx`-th argument from device to host.
        TF_RETURN_IF_ERROR(stream.Memcpy(
            offset_dst, offset_src,
            ShapeUtil::ByteSizeOfPrimitiveType(*slice.offset_primitive_type)));
        ++num_transfers;
      }
    }

    // Wait for the completion of all transfers.
    if (num_transfers > 0) {
      VLOG(2) << "Wait for completion of " << num_transfers << " transfer";
      TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());
    }

    // Clamp start indices:
    // start_indices[i] = min(max(start_indices[i], 0),
    //                        operand.dimension_size[i] - size_indices[i])
    for (auto [offset_idx, values] : llvm::enumerate(
             llvm::zip(src_shape.dimensions(), dst_shape.dimensions()))) {
      auto [src_dim, dst_dim] = values;
      int64_t start_index =
          std::min(std::max(offset_value(argument_idx, offset_idx), int64_t{0}),
                   src_dim - dst_dim);
      VLOG(2) << "arg idx: " << argument_idx << " offset_idx " << offset_idx
              << " with offset_value " << offset_value(argument_idx, offset_idx)
              << " start_idx: " << start_index << " src_dim: " << src_dim
              << " dst_dim:" << dst_dim;
      slice_starts.push_back(start_index);
    }

    // Compute new slice. No need to copy the content to new buffers as we can
    // reuse the original buffers since slices are contiguous.
    int64_t new_size = ShapeUtil::ByteSizeOf(dst_shape);

    int64_t new_offset = 0;
    for (auto [start, stride] :
         llvm::zip(slice_starts, *ShapeUtil::ByteStrides(src_shape))) {
      new_offset += start * stride;
    }

    VLOG(3) << "Create sliced argument " << argument_idx << " of shape "
            << slice.sliced_shape->ToString()
            << " by slicing argument of shape " << slice.orig_shape->ToString()
            << " at offset " << new_offset << " with " << new_size;
    slice_buffers[argument_idx] =
        argument_buffer.GetByteSlice(new_offset, new_size);
  }

  // Safe to create a local BufferAllocations here since buffers are only
  // slices of bigger ones allocated elsewhere.
  BufferAllocations slice_allocations(slice_buffers,
                                      orig_allocations.device_ordinal(),
                                      orig_allocations.memory_allocator());

  VLOG(3) << "DynamicSliceFusionCmd: new slice_allocations: "
          << slice_allocations.ToString();

  Thunk::ExecuteParams new_params =
      Thunk::ExecuteParams::CloneWithNewAllocations(execute_params,
                                                    slice_allocations);

  // TODO(b/406370928): Instead of creating a nested command buffer on every
  // call we should create it once and update it. CommandBufferThunk state
  // manager relies on command buffer pointer as an identity for command
  // buffers, and it means that command buffer commands sequence should not
  // create ephemeral command buffers at run time.
  TF_ASSIGN_OR_RETURN(auto nested_command_buffer,
                      execute_params.stream->parent()->CreateCommandBuffer(
                          se::CommandBuffer::Mode::kNested));

  CommandStateManager state;
  CommandBufferParams nested_record_params = {
      .state = state,
      .updated_allocs = std::nullopt,
      .is_initialization = false,
      .unroll_iteration = record_params.unroll_iteration,
      .command_buffer = nested_command_buffer.get(),
      .is_finalize = true,
      .executor = &embedded_commands_,
      .external_dependencies = {},
  };
  Thunk::ExecuteParams::ScopedCommandBufferParams scoped(new_params,
                                                         &nested_record_params);
  TF_RETURN_IF_ERROR(embedded_commands_.ExecuteOnStream(new_params));

  // For command buffer instantiation ran by CommandBufferThunk::Initialize, we
  // must not step the Indvar, because it is not a real run.
  if (offset_as_function_of_indvar_metadata_ != std::nullopt &&
      record_params.command_buffer->state() ==
          se::CommandBuffer::State::kUpdate) {
    Indvar(this) =
        HloEvaluator()
            .Evaluate(
                *offset_as_function_of_indvar_metadata_.value()->indvar_update,
                {&Indvar(this)})
            .value();
    VLOG(2) << "Update Indvar = " << Indvar(this).ToString();
  }

  return HandleCmdCreateOrUpdate(
      record_params,
      [&] {
        return record_params.command_buffer->CreateChildCommand(
            se::CommandBuffer::ChildCommandType::kCloned,
            *nested_command_buffer, CommandBufferDependencies(record_params));
      },
      [&](const se::CommandBuffer::Command* command) {
        return record_params.command_buffer->UpdateChildCommand(
            se::CommandBuffer::ChildCommandType::kCloned, command,
            *nested_command_buffer);
      });
}

CommandThunk::BufferUseVector DynamicSliceFusionCmd::buffer_uses() const {
  CommandThunk::BufferUseVector buffers;
  auto embed_buffers = embedded_commands_.buffer_uses();
  for (const auto& buffer_usage : embed_buffers) {
    buffers.emplace_back(
        *embeded_to_origin_slice_map_.at(buffer_usage.slice().index()),
        buffer_usage.access());
  }
  return buffers;
}

//===----------------------------------------------------------------------===//
// DynamicSliceCopyFusionCmd
//===----------------------------------------------------------------------===//

DynamicSliceCopyFusionCmd::DynamicSliceCopyFusionCmd(
    const BufferAllocation::Slice& source_buffer,
    const BufferAllocation::Slice& destination_buffer, uint64_t mem_size,
    DynamicMemcpyThunk::Offsets offsets)
    : CommandThunk(Thunk::Kind::kCopy, Thunk::ThunkInfo()),
      source_buffer_(source_buffer),
      destination_buffer_(destination_buffer),
      mem_size_(mem_size),
      offsets_(offsets) {}

absl::Status DynamicSliceCopyFusionCmd::ExecuteOnStream(
    const Thunk::ExecuteParams& execute_params) {
  CommandBufferParams& record_params = *execute_params.command_buffer_params;
  se::DeviceAddressBase src_data =
      execute_params.buffer_allocations->GetDeviceAddress(source_buffer_);
  se::DeviceAddressBase dst_data =
      execute_params.buffer_allocations->GetDeviceAddress(destination_buffer_);

  return HandleCmdCreateOrUpdate(
      record_params,
      [&]() -> absl::StatusOr<const se::CommandBuffer::Command*> {
        int64_t src_offset = offsets_.src_offsets[0];
        int64_t dst_offset = offsets_.dst_offsets[0];
        auto src_with_offset = src_data.GetByteSlice(src_offset, mem_size_);
        auto dst_with_offset = dst_data.GetByteSlice(dst_offset, mem_size_);
        VLOG(3) << "Create DynamicSliceCopyFusionCmd with Memcpy of size "
                << mem_size_ << " from " << src_with_offset.opaque()
                << " (offset " << src_offset << ") to "
                << dst_with_offset.opaque() << " (offset " << dst_offset
                << "), dependends_on_loop: " << offsets_.depends_on_loop;
        return record_params.command_buffer->CreateMemcpyD2D(
            &dst_with_offset, src_with_offset, mem_size_,
            CommandBufferDependencies(record_params));
      },
      [&](const se::CommandBuffer::Command* command) {
        int64_t iteration_index = 0;
        if (offsets_.depends_on_loop) {
          if (WhileThunk::RunningWhileThunkLoop()) {
            TF_ASSIGN_OR_RETURN(iteration_index,
                                WhileThunk::CurrentLoopIteration());
          } else {
            iteration_index = record_params.unroll_iteration;
          }
        }
        int64_t src_offset = offsets_.src_offsets[iteration_index];
        int64_t dst_offset = offsets_.dst_offsets[iteration_index];
        auto src_with_offset = src_data.GetByteSlice(src_offset, mem_size_);
        auto dst_with_offset = dst_data.GetByteSlice(dst_offset, mem_size_);

        VLOG(3) << "Update DynamicSliceCopyFusionCmd with Memcpy of size "
                << mem_size_ << " from " << src_with_offset.opaque()
                << " (offset " << src_offset << ") to "
                << dst_with_offset.opaque() << " (offset " << dst_offset
                << "), iteration_index: " << iteration_index;
        return record_params.command_buffer->UpdateMemcpyD2D(
            command, &dst_with_offset, src_with_offset, mem_size_);
      });
}

CommandThunk::BufferUseVector DynamicSliceCopyFusionCmd::buffer_uses() const {
  CommandThunk::BufferUseVector buffers;
  buffers.emplace_back(BufferUse::Read(source_buffer_));
  buffers.emplace_back(BufferUse::Write(destination_buffer_));
  return buffers;
}

}  // namespace xla::gpu
