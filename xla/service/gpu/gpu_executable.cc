/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_executable.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "riegeli/bytes/string_writer.h"
#include "riegeli/bytes/writer.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_cache.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_conversion_pass.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/scratch_memory.h"
#include "xla/backends/gpu/runtime/scratch_memory_requests.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_buffer_debug_pass.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/backends/gpu/runtime/thunk_proto_deserialization.h"
#include "xla/client/executable_build_options.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/map_util.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/hang_watchdog.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_value.h"
#include "xla/service/llvm_ir/buffer_assignment_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/service/rendezvous.h"
#include "xla/service/riegeli_dump_writer.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/stream_pool.h"
#include "xla/service/xla_debug_info_manager.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/kernel_stats.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_id.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/scoped_module_handle.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/stream_executor/vmm_device_address_allocator.h"
#include "xla/tsl/platform/env_time.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/util/sorted_range.h"
#include "xla/util.h"
#include "xla/util/split_proto/split_executable_and_options_writer.h"
#include "xla/util/split_proto/split_gpu_executable_writer.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/random.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::gpu {
namespace {

// Chooses the correct allocations to be used within the GpuExecutable code.
std::vector<const BufferAllocation*> GatherAllocationPtrs(
    const std::optional<std::vector<BufferAllocation>>& mlir_allocations,
    const BufferAssignment* buffer_assignment,
    const std::deque<BufferAllocation>& thunk_pass_allocations) {
  const std::vector<BufferAllocation>* allocation_vec = nullptr;
  if (mlir_allocations.has_value()) {
    allocation_vec = &mlir_allocations.value();
  } else if (buffer_assignment != nullptr) {
    allocation_vec = &buffer_assignment->Allocations();
  }

  std::vector<const BufferAllocation*> alloc_ptrs;
  if (allocation_vec != nullptr) {
    alloc_ptrs.reserve(allocation_vec->size());
    for (const BufferAllocation& alloc : *allocation_vec) {
      alloc_ptrs.push_back(&alloc);
    }
  }

  if (!thunk_pass_allocations.empty()) {
    alloc_ptrs.reserve(alloc_ptrs.size() + thunk_pass_allocations.size());
    for (const BufferAllocation& alloc : thunk_pass_allocations) {
      alloc_ptrs.push_back(&alloc);
    }
  }

  return alloc_ptrs;
}

class GpuExecutableThunkPassBufferAllocator : public ThunkPassBufferAllocator {
 public:
  ~GpuExecutableThunkPassBufferAllocator() override = default;

  explicit GpuExecutableThunkPassBufferAllocator(
      BufferAllocation::Index start_idx)
      : next_idx_(start_idx) {}

  absl::StatusOr<BufferAllocation * absl_nonnull> NewEmptyAllocation(
      int64_t size) override {
    allocations_.push_back(BufferAllocation(next_idx_++, size, /*color=*/0));
    return &allocations_.back();
  }

  std::deque<BufferAllocation>& MutableAllocations() { return allocations_; }

 private:
  BufferAllocation::Index next_idx_ = 0;
  // std::deque is used to ensure pointer stability.
  std::deque<BufferAllocation> allocations_;
};

}  // namespace

using ::tsl::profiler::ScopedAnnotation;

constexpr int kAsyncStreamTotal =
    static_cast<int>(AsyncStreamKind::ASYNC_STREAM_KIND_MEMCPYP2P) + 1;

// Returns the number of additional streams to allocate for a `GpuExecutable`.
static GpuExecutable::NumAdditionalStreams GetNumAdditionalStreams(
    ThunkExecutor& executor, const DebugOptions& opts) {
  // First initialize based on what was requested via the DebugOptions.
  int compute = opts.xla_gpu_executable_num_compute_streams();
  int comm = opts.xla_gpu_executable_num_communication_streams();

  // Clamp it to minimum number of required streams.
  compute = std::max(0, compute);
  comm = std::max(kAsyncStreamTotal, comm);

  // Then traverse all thunks to see if anyone requested more streams.
  for (const auto& thunk : executor.thunks()) {
    thunk->Walk([&](Thunk* nested) {
      if (auto* async_start = dynamic_cast<AsyncStartThunk*>(nested)) {
        ExecutionStreamId id = async_start->execution_stream_id();
        if (id.is_computation()) {
          compute = std::max<int>(compute, id.computation_id().value() + 1);
        } else {
          comm = std::max<int>(comm, id.communication_id().value() + 1);
        }
      }
    });
  }

  return {compute, comm};
}

static absl::Status RunThunkPasses(const DebugOptions& debug_options,
                                   const se::DeviceDescription& device_info,
                                   SequentialThunk* root_thunk,
                                   HloModule* hlo_module,
                                   ThunkPassBufferAllocator& allocator) {
  ThunkPassPipeline pipeline("thunk-passes");
  if (debug_options.xla_gpu_experimental_enable_checksum_tracing_on_thunks()) {
    pipeline.AddPass(std::make_unique<ThunkBufferDebugPass>(
        ThunkBufferDebugPass::Mode::kChecksum));
  }
  if (debug_options.xla_gpu_experimental_enable_buffer_saver_on_thunks()) {
    pipeline.AddPass(std::make_unique<ThunkBufferDebugPass>(
        ThunkBufferDebugPass::Mode::kBufferSaver));
  }
  if ((debug_options.xla_gpu_detect_nan() !=
       DebugOptions::DETECTION_MODE_NONE) ||
      (debug_options.xla_gpu_detect_inf() !=
       DebugOptions::DETECTION_MODE_NONE) ||
      debug_options.xla_gpu_log_minmax()) {
    LOG(ERROR) << "Adding ThunkBufferDebugPass for nan/inf/minmax checking";
    pipeline.AddPass(std::make_unique<ThunkBufferDebugPass>(
        ThunkBufferDebugPass::Mode::kFloatChecker));
  }
  pipeline.AddPass(std::make_unique<CommandBufferConversionPass>(
      hlo_module ? hlo_module->name() : "Anonymous"));

  ASSIGN_OR_RETURN(bool changed,
                   pipeline.Run(&root_thunk->thunks(), debug_options,
                                hlo_module, device_info, allocator));
  if (changed) {
    VLOG(3) << "Thunk passes changed the thunk tree.";
    if (hlo_module && DumpingEnabledForHloModule(*hlo_module)) {
      DumpToFileInDirOrStdout(
          *hlo_module, "",
          absl::StrCat("thunk_sequence_after_thunk_passes", ".txt"),
          root_thunk->ToString(/*indent=*/0));
    }
  }

  if (hlo_module && DumpingEnabledForHloModule(*hlo_module)) {
    ThunkMetadataListProto metadata_list_proto =
        GetMetadataListProtoFromThunkGraph(root_thunk->executor().thunks());
    DumpPerModuleProtobufToFile(*hlo_module, metadata_list_proto, debug_options,
                                "thunk_metadata");
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<GpuExecutable>> GpuExecutable::Create(
    Params params) {
  if (params.buffer_assignment_proto.has_value() &&
      params.buffer_assignment != nullptr) {
    return absl::InvalidArgumentError(
        "Cannot set both buffer_assignment_proto and buffer_assignment.");
  }

  int64_t next_idx = 0;
  if (params.mlir_allocations.has_value()) {
    next_idx = params.mlir_allocations->size();
  } else if (params.buffer_assignment != nullptr) {
    next_idx = params.buffer_assignment->Allocations().size();
  }

  GpuExecutableThunkPassBufferAllocator allocator(next_idx);

  // TODO(b/461380690): Remove this once we have a better way to distinguish
  // between compiler-generated and runtime-loaded GPU executables.
  absl::StatusOr<std::vector<ThunkProto>> thunk_sequence_proto =
      [&]() -> absl::StatusOr<std::vector<ThunkProto>> {
    std::vector<ThunkProto> protos;
    protos.reserve(params.executable->thunks().size());
    for (const auto& thunk : params.executable->thunks()) {
      ASSIGN_OR_RETURN(ThunkProto proto, thunk->ToProto());
      protos.push_back(std::move(proto));
    }
    return protos;
  }();

  // Wrap the ThunkExecutor's thunks in a temporary SequentialThunk for running
  // thunk passes (which operate on SequentialThunk).
  auto seq_thunk = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo(), std::move(params.executable->thunks()));
  RETURN_IF_ERROR(RunThunkPasses(params.debug_options,
                                 params.device_description, seq_thunk.get(),
                                 params.debug_module.get(), allocator));
  // Extract modified thunks back into a ThunkExecutor.
  auto executor =
      std::make_unique<ThunkExecutor>(std::move(seq_thunk->thunks()));

  return std::unique_ptr<GpuExecutable>(new GpuExecutable(
      std::move(params.debug_module), std::move(params.asm_text),
      std::move(params.binary), std::move(params.dnn_compiled_graphs),
      std::move(params.device_description), std::move(executor),
      std::move(params.module_name), std::move(params.program_shape),
      std::move(params.mlir_allocations), std::move(params.buffer_assignment),
      std::move(allocator.MutableAllocations()), std::move(params.alias_info),
      std::move(params.debug_options), std::move(params.constants),
      std::move(params.output_info), params.enable_debug_info_manager,
      std::move(params.module_stats), std::move(thunk_sequence_proto),
      std::move(params.executable_abi_version),
      std::move(params.cpu_target_machine_options),
      std::move(params.buffer_assignment_proto)));
}

// Implementation note: HLO profiling is always enabled for GPU executables,
// since we can use timers around thunks.
GpuExecutable::GpuExecutable(
    std::unique_ptr<HloModule> debug_module, std::string asm_text,
    std::vector<uint8_t> binary, BinaryMap dnn_compiled_graphs,
    se::DeviceDescription device_description,
    std::unique_ptr<ThunkExecutor> executable, std::string module_name,
    ProgramShape program_shape,
    std::optional<std::vector<BufferAllocation>> mlir_allocations,
    std::unique_ptr<const BufferAssignment> buffer_assignment,
    std::deque<BufferAllocation> thunk_pass_allocations,
    std::unique_ptr<GpuAliasInfo> alias_info, DebugOptions debug_options,
    std::vector<ConstantInfo> constants,
    absl::flat_hash_map<ShapeIndex, OutputInfo> output_info,
    bool enable_debug_info_manager, ModuleStats module_stats,
    absl::StatusOr<std::vector<ThunkProto>> thunk_sequence_proto,
    se::ExecutableAbiVersion executable_abi_version,
    std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options,
    std::optional<BufferAssignmentProto> buffer_assignment_proto)
    : Executable(std::move(debug_module)),
      text_(std::move(asm_text)),
      binary_(std::move(binary)),
      dnn_compiled_graphs_(std::move(dnn_compiled_graphs)),
      gpu_version_(device_description.gpu_compute_capability()),
      thunk_executor_(std::move(executable)),
      num_additional_streams_(
          GetNumAdditionalStreams(*thunk_executor_, debug_options)),
      module_name_(std::move(module_name)),
      program_shape_(std::move(program_shape)),
      allocation_ptrs_(GatherAllocationPtrs(
          mlir_allocations, buffer_assignment.get(), thunk_pass_allocations)),
      allocations_(std::move(mlir_allocations)),
      buffer_assignment_(std::move(buffer_assignment)),
      buffer_assignment_proto_(std::move(buffer_assignment_proto)),
      thunk_pass_allocations_(std::move(thunk_pass_allocations)),
      alias_info_(std::move(alias_info)),
      debug_buffer_assignment_show_max_(
          debug_options.xla_debug_buffer_assignment_show_max()),
      constants_(std::move(constants)),
      output_info_(std::move(output_info)),
      enable_debug_info_manager_(enable_debug_info_manager),
      thunk_sequence_proto_(std::move(thunk_sequence_proto)),
      executable_abi_version_(std::move(executable_abi_version)),
      cpu_target_machine_options_(std::move(cpu_target_machine_options)) {
  if (gpu_version_.IsRocm()) {
    // ROCm uses hsaco hashes to distinguish between modules.
    // Bad things happen if multiple modules with identical code are loaded.
    binary_.resize(binary_.size() + 16);
    *(uint64_t*)(&binary_[binary_.size() - 16]) = tsl::EnvTime::NowNanos();
    *(uint64_t*)(&binary_[binary_.size() - 8]) = tsl::random::New64();
  }
  if (has_module() && enable_debug_info_manager_) {
    std::optional<BufferAssignmentProto> proto;
    if (buffer_assignment_proto_.has_value()) {
      proto = buffer_assignment_proto_;
    } else if (buffer_assignment_ != nullptr) {
      proto = buffer_assignment_->ToProto();
    }
    XlaDebugInfoManager::Get()->RegisterModule(shared_module(),
                                               std::move(proto));
  }
  set_module_stats(std::move(module_stats));

  CollectCommandBufferAllocationIndexes();
}

GpuExecutable::~GpuExecutable() {
  if (has_module() && enable_debug_info_manager_) {
    XlaDebugInfoManager::Get()->UnregisterModule(module().unique_id());
  }
}

absl::Status GpuExecutable::CheckCompatibilityWithServiceExecutableRunOptions(
    const ServiceExecutableRunOptions* run_options) {
  se::Stream* main_stream = run_options->stream();

  se::PlatformId platform_id = main_stream->parent()->GetPlatform()->id();
  if (platform_id == se::rocm::kROCmPlatformId) {
    auto cc = main_stream->GetRocmComputeCapability();
    std::string stream_arch = cc.gcn_arch_name();
    std::string gpu_exec_arch =
        gpu_version_.rocm_compute_capability()->gcn_arch_name();
    TF_RET_CHECK(stream_arch == gpu_exec_arch)
        << "AMDGPU GCN ISA version mismatch; expected {" << gpu_exec_arch
        << ", but was " << stream_arch;
  } else if (platform_id == se::cuda::kCudaPlatformId) {
    se::CudaComputeCapability cc = main_stream->GetCudaComputeCapability();
    TF_RET_CHECK(cc == *gpu_version_.cuda_compute_capability())
        << "Compute capability mismatch; expected {" << gpu_version_.ToString()
        << "}, but was {" << cc.ToString() << "}";
  } else if (platform_id == se::sycl::kSyclPlatformId) {
    // TODO: Add check.
  } else {
    return Internal("Unknown platform");
  }

  return absl::OkStatus();
}

absl::StatusOr<ExecutionOutput> GpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments) {
  return ExecuteAsyncOnStreamImpl(run_options, absl::MakeSpan(arguments));
}

absl::StatusOr<ScopedShapedBuffer> GpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments) {
  ASSIGN_OR_RETURN(ExecutionOutput out,
                   ExecuteAsyncOnStreamImpl(run_options, arguments));
  return out.ConsumeResult();
}

absl::StatusOr<ExecutionOutput> GpuExecutable::ExecuteAsyncOnStreamImpl(
    const ServiceExecutableRunOptions* run_options,
    VariantArguments arguments) {
  XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
      "GpuExecutable::ExecuteAsyncOnStreamImpl(", module_name_, ")"));
  se::DeviceAddressAllocator* const memory_allocator = run_options->allocator();
  se::StreamExecutor* executor = run_options->stream()->parent();

  // GpuExecutable always bound to a single GpuContext during its execution, so
  // we activate it once to skip expensive context activations later.
  auto activation = executor->Activate();

  // Lock the GPU with a shared lock so that we don't interfere with autotuning
  // that may be running during JIT compilation while allowing multiple XLA
  // computations to use the same GPU simultaneously. We do not add locking for
  // "recursive" invocations, which are done when holding a lock already.
  std::variant<absl::ReaderMutexLock, absl::WriterMutexLock> gpu_lock(
      std::in_place_index_t<0>{}, GetGpuMutex(executor));

  // Maybe update to a writer lock to get exclusive access to underlying GPU.
  if (auto* gpu_opts = run_options->run_options().gpu_executable_run_options();
      gpu_opts && gpu_opts->requires_exclusive_lock_on_gpu()) {
    gpu_lock.emplace<1>(GetGpuMutex(executor));
  }

  const GpuExecutable::BufferAllocToDeviceMemoryMap* globals;
  {
    tsl::profiler::TraceMe hlo_module_activity(
        [&] { return std::string("Resolve constant globals"); },
        tsl::profiler::TraceMeLevel::kInfo);

    ASSIGN_OR_RETURN(globals, ResolveConstantGlobals(run_options->stream()));
  }

  // Use the `device_ordinal` from the `run_options` if it is provided. This is
  // the ordinal of the logical devices (e.g., virtual GPUs). If it is not
  // provided, the ordinals of the logical and physical devices are the same.
  const int device_ordinal = run_options->device_ordinal() != -1
                                 ? run_options->device_ordinal()
                                 : executor->device_ordinal();
  ExecutionOutput result(/*on_device_shape=*/program_shape_.result(),
                         memory_allocator, device_ordinal,
                         executor->device_ordinal());

  auto get_parameter_buffer = [&](const BufferAllocation& allocation)
      -> absl::StatusOr<ParameterBuffer> {
    int64_t param_no = allocation.parameter_number();
    if (auto unowned_shapedbuffers =
            std::get_if<absl::Span<const ShapedBuffer* const>>(&arguments)) {
      return ParameterBuffer{
          (*unowned_shapedbuffers)[param_no]->buffers().element(
              allocation.param_shape_index()),
          param_no};
    }
    return ParameterBuffer{
        std::get<absl::Span<ExecutionInput>>(arguments)[param_no]
            .Buffer(allocation.param_shape_index())
            .AsDeviceAddress(),
        param_no};
  };
  ASSIGN_OR_RETURN(
      BufferAllocations buffer_allocations,
      GenerateBufferAllocations(run_options, get_parameter_buffer, globals,
                                memory_allocator, device_ordinal));
  XLA_VLOG_DEVICE(3, device_ordinal) << buffer_allocations.ToString();
  absl::Span<const BufferAllocation* const> allocations = GetAllocations();

  std::set<se::DeviceAddressBase> buffers_in_result;

  const bool is_entire_tuple_contents_aliased = [&] {
    for (auto& p : result.MutableResult()->buffers().leaves()) {
      if (!output_info_.contains(p.first)) {
        continue;
      }
      const OutputInfo& output_info = output_info_.at(p.first);
      if (!output_info.alias_config.has_value()) {
        return false;
      }
    }
    return true;
  }();

  for (auto& p : result.MutableResult()->buffers()) {
    const ShapeIndex& index = p.first;
    if (!output_info_.contains(index)) {
      continue;
    }
    const OutputInfo& output_info = output_info_.at(index);
    const BufferAllocation* allocation =
        allocations[output_info.allocation_index];
    se::DeviceAddressBase& result_buffer = p.second;

    XLA_VLOG_DEVICE(4, device_ordinal)
        << "Looking at: allocation " << output_info.allocation_index
        << " @ index: " << index.ToString();

    if (output_info.alias_config) {
      MaybeOwningDeviceAddress* maybe_owning_memory =
          [&]() -> xla::MaybeOwningDeviceAddress* {
        // ShapedBuffer is never an owned buffer.
        if (std::holds_alternative<absl::Span<const ShapedBuffer* const>>(
                arguments)) {
          return nullptr;
        }
        auto unowned_execution_input =
            std::get<absl::Span<ExecutionInput>>(arguments);
        ExecutionInput& input =
            unowned_execution_input[allocation->parameter_number()];
        return input.MutableBuffer(allocation->param_shape_index());
      }();
      if (output_info.alias_config->must_alias() && maybe_owning_memory &&
          !maybe_owning_memory->HasOwnership()) {
        return InvalidArgument(
            "An input was configured to be must-alias at "
            "compile time but not donated at runtime: allocation %d",
            output_info.allocation_index);
      }
      if (maybe_owning_memory && maybe_owning_memory->HasOwnership()) {
        std::optional<tensorflow::se::ScopedDeviceAddress<uint8_t>> owning =
            maybe_owning_memory->Release();
        // If the caller passes the ownership of the device memory, reuse it
        // as the output buffer. It is up to the caller whether or not to
        // donate a buffer; the aliasing information describes which buffers
        // may alias, not buffers that must alias.
        se::DeviceAddressBase argument_buffer = owning->Release();
        *maybe_owning_memory = argument_buffer;
        result_buffer = argument_buffer;
        // The caller is giving us the
        // input buffer, but in case of error from the execute call, we should
        // not be releasing it as it contains valid data (for example, it is a
        // parameter which the user wants us to alias, in a gradient update
        // computation). So we store the index into the result in the aliased
        // vector, which will be fed to the ExecutionOutput, which will use
        // the indices to drop the addresses from its own ScopedShapedBuffer
        // result, if the ExecutionOutput is not committed.
        result.AddAliasedIndex(index);
      } else if (!output_info.passthrough &&
                 !ShapeUtil::GetSubshape(program_shape_.result(), index)
                      .IsTuple()) {
        ASSIGN_OR_RETURN(result_buffer,
                         AllocateCopyProtectedOutputBuffer(
                             run_options, buffer_allocations, index,
                             *allocation, device_ordinal, memory_allocator));
      }
    }

    if (result_buffer.is_null()) {
      // The source instruction should have a non-parameter buffer
      // assigned.
      result_buffer =
          buffer_allocations.GetDeviceAddress(output_info.allocation_index);

      // If the entire tuple contents is aliased, the copy insertion will *not*
      // materialize a new tuple, so we mark it as aliased as well.
      if (is_entire_tuple_contents_aliased) {
        result.AddAliasedIndex(index);
      }
    }
    buffers_in_result.insert(result_buffer);
  }

  absl::Status execute_status = ExecuteThunks(buffer_allocations, run_options);

  absl::Status teardown_status =
      buffer_allocations.TearDown(buffers_in_result, GetAllocations());

  RETURN_IF_ERROR(execute_status);
  RETURN_IF_ERROR(teardown_status);

  // Free allocations for arguments.
  if (auto args = std::get_if<absl::Span<ExecutionInput>>(&arguments)) {
    MarkToBeReleasedArguments(*args, result);
  }
  return std::move(result);
}

int64_t GpuExecutable::SizeOfGeneratedCodeInBytes() const {
  // Non-empty PTX but empty cubin: compilation must have failed, return
  // "unknown".
  if (binary().empty() && !text_.empty()) {
    return -1;
  }
  int64_t size = binary().size();
  for (const BufferAllocation* allocation : GetAllocations()) {
    if (allocation->is_constant()) {
      size += allocation->size();
    }
  }
  return size;
}

}  // namespace xla::gpu
