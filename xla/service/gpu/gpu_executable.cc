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
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_conversion_pass.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_buffer_debug_pass.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/executable_run_options.h"
#include "xla/map_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/xla_debug_info_manager.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
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
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/random.h"
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

std::unique_ptr<CommandBufferVaRemapping> CreateCommandBufferVaRemapping(
    DebugOptions::CommandBufferUpdateMode update_mode,
    ThunkExecutor* thunk_executor,
    absl::Span<const BufferAllocation* const> allocations,
    absl::string_view module_name) {
  absl::StatusOr<std::unique_ptr<CommandBufferVaRemapping>> remapping =
      CommandBufferVaRemapping::Create(update_mode, thunk_executor, allocations,
                                       module_name);
  CHECK_OK(remapping.status());
  return std::move(remapping).value();
}
}  // namespace

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
      command_buffer_va_remapping_(CreateCommandBufferVaRemapping(
          has_module() ? debug_options.xla_gpu_command_buffer_update_mode()
                       : DebugOptions::ALWAYS_UPDATE,
          thunk_executor_.get(), allocation_ptrs_, module_name_)),
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

absl::StatusOr<const GpuExecutable::BufferAllocToDeviceMemoryMap*>
GpuExecutable::ResolveConstantGlobals(se::Stream* stream) {
  se::StreamExecutor* executor = stream->parent();

  absl::MutexLock lock(module_handle_mutex_);
  auto it = module_globals_.find(executor);
  if (it != module_globals_.end()) {
    return it->second.get();
  }

  se::MultiModuleLoaderSpec module_spec;
  if (!binary().empty()) {
    module_spec.AddCudaCubinInMemory(binary());
  }
  module_spec.AddCudaPtxInMemory(text().c_str());

  auto globals = std::make_unique<BufferAllocToDeviceMemoryMap>();
  se::ModuleHandle module_handle;
  // The CUDA driver isn't able to load a PTX and a binary which are both empty.
  // It's okay if we skip loading in this case; if the module isn't loaded, all
  // symbol lookups will fail, just as they should for an empty module.
  if (!(executor->GetPlatform()->id() == se::cuda::kCudaPlatformId &&
        binary().empty() && text().empty())) {
    ASSIGN_OR_RETURN(module_handle, executor->LoadModule(module_spec));
  }

  // A flag signalling if constant initialization submitted memcpy operations
  // to the `stream`.
  int submitted_mem_copies = 0;

  for (const ConstantInfo& info : constants_) {
    absl::StatusOr<se::DeviceAddressBase> global_status;
    if (static_cast<bool>(module_handle)) {
      global_status = executor->GetSymbol(info.symbol_name, module_handle);
    }

    se::DeviceAddressBase global;

    CHECK(static_cast<bool>(module_handle) && global_status.ok());
    // The constant was defined in the PTX and has been allocated by the CUDA
    // driver.
    global = *global_status;
    XLA_VLOG_DEVICE(3, executor->device_ordinal()) << absl::StreamFormat(
        "Resolved global %s to %p", info.symbol_name, global.opaque());

    if (!info.content.span().empty()) {
      // This means the constant did not have an initializer in the PTX and
      // therefore must be initialized by XLA here.
      RETURN_IF_ERROR(stream->Memcpy(&global, info.content.span().data(),
                                     info.content.span().size()));
      submitted_mem_copies = true;
    }

    if (info.allocation_index != -1) {
      InsertOrDie(globals.get(), info.allocation_index, global);
    }
  }

  // Wait for the completion of all host->device transfers, to guarantee that
  // destructor will not race with any operations in flight (deallocate
  // xla::Literal owned by the HLO module).
  if (submitted_mem_copies) {
    CHECK_OK(stream->BlockHostUntilDone());
  }

  module_handles_.emplace(executor,
                          se::ScopedModuleHandle(executor, module_handle));
  return module_globals_.emplace(executor, std::move(globals))
      .first->second.get();
}

absl::StatusOr<se::DeviceAddressBase> GpuExecutable::BufferForAllocation(
    ParameterBufferResolver get_parameter_buffer,
    const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
    const BufferAllocation& allocation,
    se::DeviceAddressAllocator* const memory_allocator, int device_ordinal,
    int64_t arg_idx,
    const absl::flat_hash_map<LogicalBuffer::Color, int64_t>&
        allocate_granularity,
    const absl::flat_hash_set<BufferAllocation::Index>&
        returned_output_allocations,
    CommandBufferVaRemapping::ScopedExecution* va_remap_execution) {
  if (allocation.is_thread_local()) {
    return se::DeviceAddressBase{};
  }
  if (allocation.is_entry_computation_parameter()) {
    ASSIGN_OR_RETURN(ParameterBuffer registered_buffer,
                     get_parameter_buffer(allocation));
    if (registered_buffer.buffer.is_null() &&
        registered_buffer.buffer.size() > 0) {
      return FailedPrecondition(
          "Cannot run XLA computation because pointer to (sub-)buffer at "
          "index %s of parameter %d was null.  All pointers to "
          "(sub-)buffers must not be null, unless the (sub-)buffer has "
          "zero elements.",
          allocation.param_shape_index().ToString(),
          registered_buffer.parameter_number);
    }
    return registered_buffer.buffer;
  }
  if (allocation.is_constant()) {
    auto it = globals->find(arg_idx);
    if (it == globals->end()) {
      return se::DeviceAddressBase();
    }
    return it->second;
  }

  // Allocate each allocation that might escape, or is the temp buffer.
  CHECK(allocation.maybe_live_out() || allocation.IsPreallocatedTempBuffer());
  int64_t buffer_size = allocation.size();
  se::DeviceAddressBase buffer_address;
  if (buffer_size > 0) {
    // Maybe round up buffer allocation size to the requested granularity.
    if (auto it = allocate_granularity.find(allocation.color());
        it != allocate_granularity.end()) {
      buffer_size = RoundUpTo(buffer_size, it->second);
    }
    absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> buffer;
    if (command_buffer_va_remapping_->ShouldRemapAllocation(
            allocation.index(), va_remap_execution)) {
      bool return_reservation_address =
          !(allocation.maybe_live_out() &&
            returned_output_allocations.contains(allocation.index()));
      buffer = command_buffer_va_remapping_->Allocate(
          device_ordinal, allocation, buffer_size, return_reservation_address,
          *va_remap_execution);
    } else {
      buffer = memory_allocator->Allocate(device_ordinal, buffer_size,
                                          /*retry_on_failure=*/true,
                                          /*memory_space=*/allocation.color());
    }
    ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> scoped_buffer,
                     std::move(buffer));
    buffer_address = scoped_buffer.Release();
  }
  return buffer_address;
}

absl::Status CheckAlignment(const BufferAllocation& allocation,
                            se::DeviceAddressBase buffer, int arg_idx) {
  const int64_t expected_alignment = [&] {
    if (allocation.is_entry_computation_parameter()) {
      return kEntryParameterAlignBytes;
    }
    if (allocation.is_constant()) {
      return kConstantBufferAlignBytes;
    }
    return kXlaAllocatedBufferAlignBytes;
  }();
  if (!buffer.is_null() &&
      reinterpret_cast<uintptr_t>(buffer.opaque()) % expected_alignment != 0) {
    return Internal(
        "Address of buffer %d must be a multiple of %x, but "
        "was %p",
        arg_idx, expected_alignment, buffer.opaque());
  }
  return absl::OkStatus();
}

// Resolve GpuCollectives instance that we should use for the run.
// TODO(ezhulenev): We have almost identical method in `collective_params.cc`,
// this one has to be removed.
static GpuCollectives* ResolveGpuCollectives(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options) {
  auto* gpu_options = run_options->run_options().gpu_executable_run_options();
  if (gpu_options && gpu_options->collectives()) {
    return gpu_options->collectives();
  }

  absl::string_view platform_name =
      run_options->run_options().stream()->parent()->GetPlatform()->Name();

  // If debug options specify a collectives implementation by name, look it up
  // in the registry. Otherwise, use the default (highest-priority) one.
  if (debug_options &&
      !debug_options->xla_gpu_collectives_implementation().empty()) {
    absl::StatusOr<Collectives*> collectives = CollectivesRegistry::Get(
        platform_name, debug_options->xla_gpu_collectives_implementation());
    CHECK_OK(collectives)  // Crash OK
        << "Failed to get GPU collectives implementation: "
        << debug_options->xla_gpu_collectives_implementation();
    return absl::down_cast<GpuCollectives*>(*collectives);
  }

  return GpuCollectives::Default(platform_name);
}

absl::StatusOr<BufferAllocations> GpuExecutable::GenerateBufferAllocations(
    const ServiceExecutableRunOptions* run_options,
    ParameterBufferResolver get_parameter_buffer,
    const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
    se::DeviceAddressAllocator* const memory_allocator, int device_ordinal,
    const absl::flat_hash_set<BufferAllocation::Index>&
        returned_output_allocations,
    CommandBufferVaRemapping::ScopedExecution* va_remap_execution) {
  tsl::profiler::TraceMe hlo_module_activity(
      [&] { return std::string("Build buffer allocations"); },
      tsl::profiler::TraceMeLevel::kInfo);

  const DebugOptions* debug_options =
      has_module() ? &module_config().debug_options() : nullptr;

  absl::flat_hash_map<LogicalBuffer::Color, int64_t> allocate_granularity;
  if (auto* collectives = ResolveGpuCollectives(run_options, debug_options)) {
    // BFC allocator ignores memory alignment and always allocates 256 byte
    // aligned buffers, however for collective memory underlying libraries
    // require larger alignment. We conservatively round up all allocation
    // sizes to the alignment requirement. Proper fix must be done in BFC
    // allocator and all the other allocator adaptors that we have in XLA, but
    // this is left as an exercise for curious reader. The raw memory allocator
    // that backs the BFC allocator uses correct granularity and alignment.
    static constexpr int64_t kCollectiveMemoryColor = 1;
    allocate_granularity[kCollectiveMemoryColor] =
        collectives->SymmetricMemoryAlignment();
  }

  // Tag allocations made in this invocation as multi-device for VMM reuse.
  se::DeviceAddressVmmAllocator::DeviceAssignmentScope
      vmm_device_assignment_scope(
          run_options->run_options().device_assignment());

  absl::Span<const BufferAllocation* const> allocations = GetAllocations();
  const int64_t num_buffers = allocations.size();
  RETURN_IF_ERROR(command_buffer_va_remapping_->PrepareReservation(
      run_options, device_ordinal, allocations, allocate_granularity,
      va_remap_execution));

  std::vector<se::DeviceAddressBase> buffers;
  buffers.reserve(num_buffers);
  for (int64_t i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = *allocations[i];
    ASSIGN_OR_RETURN(
        buffers.emplace_back(),
        BufferForAllocation(get_parameter_buffer, globals, allocation,
                            memory_allocator, device_ordinal, i,
                            allocate_granularity, returned_output_allocations,
                            va_remap_execution));
    RETURN_IF_ERROR(CheckAlignment(allocation, buffers.back(), i));
  }
  return {{buffers, device_ordinal, memory_allocator}};
}

absl::StatusOr<se::DeviceAddressBase>
GpuExecutable::AllocateCopyProtectedOutputBuffer(
    const ServiceExecutableRunOptions* run_options,
    BufferAllocations& buffer_allocations, const ShapeIndex& index,
    const BufferAllocation& allocation, int device_ordinal,
    se::DeviceAddressAllocator* const memory_allocator,
    CommandBufferVaRemapping::ScopedExecution* va_remap_execution) {
  // The caller guards this against aliasing pass-through params, as we do not
  // need to write into the output buffer in that case.
  XLA_VLOG_DEVICE(3, device_ordinal)
      << "Using copy-protection: aliasing is specified, but the "
         "buffer is not donated; allocating a fresh buffer";
  int64_t allocation_size =
      ShapeUtil::ByteSizeOf(ShapeUtil::GetSubshape(result_shape(), index));
  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> allocated_buffer;
  if (command_buffer_va_remapping_->ShouldRemapAllocation(allocation.index(),
                                                          va_remap_execution)) {
    allocated_buffer = command_buffer_va_remapping_->Allocate(
        device_ordinal, allocation, allocation_size,
        /*return_reservation_address=*/false, *va_remap_execution);
  } else {
    allocated_buffer = memory_allocator->Allocate(
        device_ordinal, allocation_size, /*retry_on_failure=*/true,
        /*memory_space=*/allocation.color());
  }
  if (!allocated_buffer.ok()) {
    return VerboseAllocationError(allocated_buffer.status());
  }
  se::DeviceAddressBase result_buffer = allocated_buffer->Release();
  se::DeviceAddressBase& aliased_buffer =
      buffer_allocations.GetMutableDeviceAddress(allocation.index());
  CHECK_EQ(aliased_buffer.size(), result_buffer.size());
  RETURN_IF_ERROR(run_options->stream()->MemcpyD2D(
      &result_buffer, aliased_buffer, aliased_buffer.size()));
  aliased_buffer = result_buffer;
  return result_buffer;
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

  absl::flat_hash_set<BufferAllocation::Index> returned_output_allocations;
  for (const auto& [_, output_info] : output_info_) {
    returned_output_allocations.insert(output_info.allocation_index);
  }

  ASSIGN_OR_RETURN(std::unique_ptr<CommandBufferVaRemapping::ScopedExecution>
                       va_remap_execution,
                   command_buffer_va_remapping_->BeginExecution(
                       run_options, memory_allocator, device_ordinal));

  ASSIGN_OR_RETURN(BufferAllocations owning_buffer_allocations,
                   GenerateBufferAllocations(
                       run_options, get_parameter_buffer, globals,
                       memory_allocator, device_ordinal,
                       returned_output_allocations, va_remap_execution.get()));
  XLA_VLOG_DEVICE(3, device_ordinal) << owning_buffer_allocations.ToString();
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
        ASSIGN_OR_RETURN(
            result_buffer,
            AllocateCopyProtectedOutputBuffer(
                run_options, owning_buffer_allocations, index, *allocation,
                device_ordinal, memory_allocator, va_remap_execution.get()));
      }
    }

    if (result_buffer.is_null()) {
      // The source instruction should have a non-parameter buffer
      // assigned.
      result_buffer = owning_buffer_allocations.GetDeviceAddress(
          output_info.allocation_index);

      // If the entire tuple contents is aliased, the copy insertion will *not*
      // materialize a new tuple, so we mark it as aliased as well.
      if (is_entire_tuple_contents_aliased) {
        result.AddAliasedIndex(index);
      }
    }
    buffers_in_result.insert(result_buffer);
  }

  std::optional<BufferAllocations> execution_buffer_allocations;
  const BufferAllocations* execution_buffers = &owning_buffer_allocations;
  std::optional<Thunk::CommandBufferUpdateInfo> command_buffer_update_info;
  if (va_remap_execution != nullptr) {
    RETURN_IF_ERROR(command_buffer_va_remapping_->UpdateAllocationPolicy(
        *va_remap_execution));
    command_buffer_update_info.emplace(
        command_buffer_va_remapping_->GetCommandBufferUpdateInfo(
            *va_remap_execution));
    absl::StatusOr<BufferAllocations> execution_buffer_allocations_or =
        command_buffer_va_remapping_->BuildBufferAllocations(
            owning_buffer_allocations, device_ordinal, GetAllocations(),
            *va_remap_execution);
    if (!execution_buffer_allocations_or.ok()) {
      absl::Status build_status = execution_buffer_allocations_or.status();
      absl::Status cleanup_status = command_buffer_va_remapping_->UnmapAliases(
          device_ordinal, *va_remap_execution);
      absl::Status teardown_status = owning_buffer_allocations.TearDown(
          buffers_in_result, GetAllocations());
      RETURN_IF_ERROR(build_status);
      RETURN_IF_ERROR(cleanup_status);
      RETURN_IF_ERROR(teardown_status);
    }
    execution_buffer_allocations =
        std::move(execution_buffer_allocations_or).value();
    execution_buffers = &*execution_buffer_allocations;
    XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
        "VA remapping: module %s executing with %d command buffer "
        "allocation(s)",
        module_name_,
        command_buffer_va_remapping_->allocation_indices().size());
  }

  absl::Status execute_status = ExecuteThunks(
      *execution_buffers, run_options,
      command_buffer_update_info ? &*command_buffer_update_info : nullptr);

  absl::Status unmap_status = va_remap_execution == nullptr
                                  ? absl::OkStatus()
                                  : command_buffer_va_remapping_->UnmapAliases(
                                        device_ordinal, *va_remap_execution);

  absl::Status teardown_status =
      owning_buffer_allocations.TearDown(buffers_in_result, GetAllocations());

  RETURN_IF_ERROR(execute_status);
  RETURN_IF_ERROR(unmap_status);
  RETURN_IF_ERROR(teardown_status);

  // Free allocations for arguments.
  if (auto args = std::get_if<absl::Span<ExecutionInput>>(&arguments)) {
    MarkToBeReleasedArguments(*args, result);
  }
  return std::move(result);
}

absl::Status GpuExecutable::VerboseAllocationError(absl::Status s) {
  return ResourceExhausted(
      "%s\n%s\n", s.message(),
      buffer_assignment_->ToVerboseString(alias_info_.get(),
                                          debug_buffer_assignment_show_max_));
}

std::optional<BufferAssignmentProto> GpuExecutable::buffer_assignment_proto()
    const {
  if (buffer_assignment_ != nullptr) {
    return buffer_assignment_->ToProto();
  }
  return buffer_assignment_proto_;
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
