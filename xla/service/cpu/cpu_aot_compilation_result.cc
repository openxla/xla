/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/cpu/cpu_aot_compilation_result.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"
#include "xla/backends/cpu/buffer_allocation_info.h"
#include "xla/backends/cpu/buffer_allocation_info_util.h"
#include "xla/backends/cpu/constant_allocation.h"
#include "xla/backends/cpu/nanort/nanort_executable.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/thread_pool_task_runner.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_layout.h"
#include "xla/service/cpu/cpu_aot_loader.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {
namespace {

absl::StatusOr<std::unique_ptr<CompiledModule>> LoadCompilationResultFromProto(
    const xla::cpu::CompilationResultProto& aot_result_proto) {
  VLOG(3) << "AOT result target machine options: "
          << aot_result_proto.target_machine_options().DebugString();

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProtoWithConfig(aot_result_proto.hlo_module()));

  ABSL_ASSIGN_OR_RETURN(TargetMachineOptions target_machine_options,
                        TargetMachineOptions::FromProto(
                            aot_result_proto.target_machine_options()));
  llvm::Triple host_triple(llvm::sys::getDefaultTargetTriple());
  llvm::Triple expected_triple(target_machine_options.triple());
  if (host_triple.getArchName() != expected_triple.getArchName()) {
    return Internal("Target arch mismatch expected %s got %s.",
                    expected_triple.getArchName(), host_triple.getArchName());
  }

  const llvm::StringMap<bool> host_machine_features =
      llvm::sys::getHostCPUFeatures();
  const std::vector<std::string> compile_machine_features =
      target_machine_options.GetTargetMachineFeaturesVector();
  std::vector<std::string> host_machine_features_vector;
  for (const auto& [feature, supported] : host_machine_features) {
    if (supported) {
      host_machine_features_vector.push_back(feature.str());
    }
  }

  VLOG(3) << "Host machine options:"
          << "\nHost CPU: " << llvm::sys::getHostCPUName().str()
          << "\nHost triple: " << host_triple.str() << "\nHost features: "
          << absl::StrJoin(host_machine_features_vector, ",");

  for (const absl::string_view feature : compile_machine_features) {
    if (!absl::StartsWith(feature, "+")) {
      continue;
    }
    absl::string_view feature_name = feature.substr(1);
    if (absl::StartsWith(feature_name, "prefer-") ||
        absl::StartsWith(feature_name, "fast-")) {
      continue;
    }
    if (!host_machine_features.lookup(
            llvm::StringRef(feature_name.data(), feature_name.size()))) {
      LOG(ERROR)
          << "Loading XLA:CPU AOT result. Target machine feature " << feature
          << " is not  supported on the host machine. Machine type used for "
             "XLA:CPU compilation doesn't match the machine type for "
             "execution. Compile machine features: ["
          << absl::StrJoin(compile_machine_features, ",")
          << "] vs host machine features: ["
          << absl::StrJoin(host_machine_features_vector, ",") << "]"
          << ". This could lead to execution errors such as SIGILL.";
    }
  }

  ABSL_ASSIGN_OR_RETURN(auto function_library,
                        CpuAotLoader::LoadFunctionLibrary(aot_result_proto));

  return CpuAotCompilationResult::FromProto(aot_result_proto,
                                            std::move(function_library));
}

}  // namespace

absl::StatusOr<std::unique_ptr<Executable>> CpuAotLoader::LoadExecutable(
    const std::string& serialized_aot_result) {
  xla::cpu::CompilationResultProto proto;
  if (!proto.ParseFromString(serialized_aot_result)) {
    return Internal("Failed to parse serialized CpuAotCompilationResult.");
  }
  return LoadExecutable(proto);
}

absl::StatusOr<std::unique_ptr<Executable>> CpuAotLoader::LoadExecutable(
    const xla::cpu::CompilationResultProto& aot_result_proto) {
  ABSL_ASSIGN_OR_RETURN(auto aot_result,
                        LoadAotCompilationResult(aot_result_proto));
  return LoadExecutable(std::move(*aot_result));
}

absl::StatusOr<std::unique_ptr<Executable>> CpuAotLoader::LoadExecutable(
    CompiledModule&& compilation_result) {
  return std::move(compilation_result).LoadExecutable();
}

absl::StatusOr<std::unique_ptr<CompiledModule>>
CpuAotLoader::LoadAotCompilationResult(
    const std::string& serialized_aot_result) {
  xla::cpu::CompilationResultProto proto;
  if (!proto.ParseFromString(serialized_aot_result)) {
    return Internal("Failed to parse serialized CpuAotCompilationResult.");
  }
  return LoadAotCompilationResult(proto);
}

absl::StatusOr<std::unique_ptr<CompiledModule>>
CpuAotLoader::LoadAotCompilationResult(
    const xla::cpu::CompilationResultProto& aot_result_proto) {
  return LoadCompilationResultFromProto(aot_result_proto);
}

CpuAotCompilationOptions::CpuAotCompilationOptions(
    std::string triple, std::string cpu_name, std::string features,
    std::string entry_point_name, RelocationModel relocation_model)
    : triple_(std::move(triple)),
      cpu_name_(std::move(cpu_name)),
      features_(std::move(features)),
      entry_point_name_(std::move(entry_point_name)),
      relocation_model_(relocation_model) {}

CpuAotCompilationOptions::~CpuAotCompilationOptions() = default;

se::Platform::Id CpuAotCompilationOptions::PlatformId() const {
  return se::host::kHostPlatformId;
}

absl::StatusOr<std::unique_ptr<CpuAotCompilationResult>>
CpuAotCompilationResult::Create(
    const HloModule* hlo_module, const BufferAssignment* buffer_assignment,
    absl::string_view function_name, std::vector<ObjFileProto> obj_files,
    std::vector<SymbolProto> symbols, const ThunkSequence& thunks,
    std::unique_ptr<FunctionLibrary> function_library,
    TargetMachineOptionsProto target_machine_options, std::string data_layout) {
  ThunkSequenceSerDesProtobuf thunk_sequence_serdes(
      hlo_module, &buffer_assignment->Allocations());
  ABSL_ASSIGN_OR_RETURN(ThunkSequenceProto thunk_proto,
                        thunk_sequence_serdes.ToProto(thunks));

  std::vector<cpu::BufferAllocationInfo> buffer_allocation_infos;
  std::optional<size_t> temp_allocation_index;

  if (buffer_assignment) {
    buffer_allocation_infos =
        CreateBufferAllocationInfos(*hlo_module, *buffer_assignment);

    // Find temp allocation index if it exists
    for (const BufferAllocation& allocation :
         buffer_assignment->Allocations()) {
      if (allocation.IsPreallocatedTempBuffer()) {
        if (temp_allocation_index.has_value()) {
          return Internal("Multiple temp buffer allocations found");
        }
        temp_allocation_index = allocation.index();
      }
    }
  }

  return absl::WrapUnique(new CpuAotCompilationResult(
      hlo_module, buffer_assignment, function_name, std::move(obj_files),
      std::move(symbols), thunk_proto, std::move(temp_allocation_index),
      std::move(buffer_allocation_infos), std::move(function_library),
      std::move(target_machine_options), std::move(data_layout)));
}

CpuAotCompilationResult::CpuAotCompilationResult(
    const HloModule* hlo_module, const BufferAssignment* buffer_assignment,
    absl::string_view function_name, std::vector<ObjFileProto> obj_files,
    std::vector<SymbolProto> symbols, const ThunkSequenceProto& thunks,
    std::optional<size_t> temp_allocation_index,
    std::vector<BufferAllocationInfo> buffer_allocation_infos,
    std::unique_ptr<FunctionLibrary> function_library,
    TargetMachineOptionsProto target_machine_options, std::string data_layout)
    : temp_allocation_index_(temp_allocation_index),
      buffer_allocation_infos_(std::move(buffer_allocation_infos)),
      function_library_(std::move(function_library)) {
  *proto_.mutable_hlo_module()->mutable_hlo_module() = hlo_module->ToProto();
  *proto_.mutable_hlo_module()->mutable_config() =
      hlo_module->config().ToProto();
  *proto_.mutable_buffer_assignment() = buffer_assignment->ToProto();
  proto_.set_entry_function_name(function_name);
  *proto_.mutable_target_machine_options() = std::move(target_machine_options);
  proto_.set_data_layout(std::move(data_layout));
  for (ObjFileProto& obj_file : obj_files) {
    *proto_.add_object_files() = std::move(obj_file);
  }

  for (const auto& symbol : symbols) {
    auto* symbol_proto = proto_.add_compiled_symbols();
    *symbol_proto = symbol;
  }
  proto_.set_obj_files_kind(CompilationResultProto::KERNELS);
  module_ = hlo_module->Clone();

  ThunkSequenceSerDesProtobuf thunk_sequence_serdes(
      hlo_module, &buffer_assignment->Allocations());
  *proto_.mutable_thunk_sequence() = thunks;
}

absl::StatusOr<std::unique_ptr<CpuAotCompilationResult>>
CpuAotCompilationResult::FromProto(
    CompilationResultProto proto,
    std::unique_ptr<FunctionLibrary> function_library) {
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      HloModule::CreateFromProtoWithConfig(proto.hlo_module()));

  std::vector<BufferAllocationInfo> buffer_allocation_infos =
      CreateBufferAllocationInfos(proto.hlo_module().hlo_module(),
                                  proto.buffer_assignment());
  std::optional<size_t> temp_allocation_index;
  for (size_t i = 0; i < buffer_allocation_infos.size(); ++i) {
    if (buffer_allocation_infos[i].is_temp()) {
      temp_allocation_index = i;
      break;
    }
  }

  return std::unique_ptr<CpuAotCompilationResult>(new CpuAotCompilationResult(
      std::move(proto), std::move(module), temp_allocation_index,
      std::move(buffer_allocation_infos), std::move(function_library)));
}

absl::StatusOr<std::unique_ptr<Executable>>
CpuAotCompilationResult::LoadExecutable() && {
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      HloModule::CreateFromProtoWithConfig(proto_.hlo_module()));

  VLOG(2) << "Load XLA:CPU executable for module: " << module->name();

  if (proto_.obj_files_kind() != CompilationResultProto::KERNELS) {
    return Internal("AOT compilation result does not have thunks.");
  }

  std::vector<BufferAllocation> allocations;
  allocations.reserve(proto_.buffer_assignment().buffer_allocations_size());
  for (const BufferAllocationProto& alloc_proto :
       proto_.buffer_assignment().buffer_allocations()) {
    allocations.push_back(BufferAllocation::FromProto(alloc_proto));
  }

  ABSL_ASSIGN_OR_RETURN(
      ShapeTree<BufferAllocation::Index> result_allocation_indices,
      CreateResultAllocationIndexTree(proto_.hlo_module().hlo_module(),
                                      proto_.buffer_assignment()));

  ThunkSequenceSerDesProtobuf thunk_sequence_serdes(module.get(), &allocations);
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<ThunkSequence> thunks,
      thunk_sequence_serdes.FromProto(proto_.thunk_sequence()));

  VLOG(3) << "Loaded " << thunks->size() << " thunks.";

  // Create constant allocations from the buffer assignment proto.
  ABSL_ASSIGN_OR_RETURN(
      std::vector<ConstantAllocation> constants,
      CreateConstantAllocations(proto_.buffer_assignment(),
                                proto_.hlo_module().hlo_module()));

  ABSL_ASSIGN_OR_RETURN(
      TargetMachineOptions target_machine_options,
      TargetMachineOptions::FromProto(proto_.target_machine_options()));

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<CpuExecutable> cpu_executable,
      CpuExecutable::Create(
          std::move(function_library_), std::move(allocations),
          std::move(result_allocation_indices), proto_.buffer_assignment(),
          std::move(module), std::move(*thunks), std::move(constants),
          target_machine_options, proto_.data_layout()));

  // Dump computation proto state and buffer assignment for
  // GetCompiledMemoryStats results.
  auto hlo_proto = std::make_unique<HloProto>();
  *hlo_proto->mutable_hlo_module() = proto_.hlo_module().hlo_module();
  *hlo_proto->mutable_buffer_assignment() = proto_.buffer_assignment();
  cpu_executable->set_hlo_proto(std::move(hlo_proto));

  return std::unique_ptr<Executable>(std::move(cpu_executable));
}

absl::StatusOr<std::unique_ptr<Executable>>
CpuAotCompilationResult::LoadExecutable(
    se::Platform::Id platform_id,
    const se::DeviceDescription& device_description,
    const DebugOptions& debug_options) && {
  return std::move((*this)).LoadExecutable();
}

namespace {

using ::tsl::profiler::TraceMe;
using ::tsl::profiler::TraceMeEncode;

using ArgumentIndex = std::pair<size_t, ShapeIndex>;

absl::StatusOr<std::vector<size_t>> ResolveArgumentsMapping(
    const HloModule& module, const BufferAssignment& buffer_assignment) {
  const ComputationLayout& entry_layout = module.entry_computation_layout();

  absl::flat_hash_map<ArgumentIndex, size_t> executable_arg_index;
  for (size_t i = 0; i < entry_layout.parameter_count(); ++i) {
    ShapeUtil::ForEachLeafShape(
        entry_layout.parameter_shape(i),
        [&](const Shape& shape, const ShapeIndex& index) {
          if (shape.IsToken()) {
            return;
          }
          size_t arg_index = executable_arg_index.size();
          executable_arg_index[ArgumentIndex{i, index}] = arg_index;
        });
  }

  std::vector<size_t> argument_to_allocation_index(executable_arg_index.size());
  for (const BufferAllocation& allocation : buffer_assignment.Allocations()) {
    if (allocation.is_entry_computation_parameter()) {
      ArgumentIndex idx{allocation.parameter_number(),
                        allocation.param_shape_index()};
      auto arg_idx = executable_arg_index.find(idx);
      if (arg_idx == executable_arg_index.end()) continue;
      argument_to_allocation_index[arg_idx->second] = allocation.index();
    }
  }

  return argument_to_allocation_index;
}

absl::StatusOr<std::vector<size_t>> ResolveResultMapping(
    const HloModule& module, const BufferAssignment& buffer_assignment) {
  const ComputationLayout& entry_layout = module.entry_computation_layout();

  absl::flat_hash_map<ShapeIndex, size_t> executable_res_index;
  ShapeUtil::ForEachLeafShape(entry_layout.result_shape(),
                              [&](const Shape& shape, const ShapeIndex& index) {
                                if (shape.IsToken()) {
                                  return;
                                }
                                size_t res_index = executable_res_index.size();
                                executable_res_index[index] = res_index;
                              });

  std::vector<size_t> result_to_allocation_index(executable_res_index.size());
  for (const auto& [index, res_idx] : executable_res_index) {
    ABSL_ASSIGN_OR_RETURN(
        BufferAllocation::Slice slice,
        buffer_assignment.GetUniqueSlice(
            module.entry_computation()->root_instruction(), index));
    result_to_allocation_index[res_idx] =
        static_cast<size_t>(slice.allocation()->index());
  }
  return result_to_allocation_index;
}

absl::StatusOr<std::optional<size_t>> ResolveTempAllocationIndex(
    const BufferAssignment& buffer_assignment) {
  std::optional<size_t> temp_allocation_index;
  for (const BufferAllocation& allocation : buffer_assignment.Allocations()) {
    if (allocation.IsPreallocatedTempBuffer()) {
      if (temp_allocation_index.has_value()) {
        return Internal("Multiple temp buffer allocations found");
      }
      temp_allocation_index = allocation.index();
    }
  }
  return temp_allocation_index;
}

class CpuExecutableRunner final : public NanoRtExecutable::ExecutableRunner {
 public:
  explicit CpuExecutableRunner(std::unique_ptr<Executable> executable)
      : executable_(std::move(executable)) {}

  Executable* executable() const override { return executable_.get(); }

  const HloModuleConfig& module_config() const override {
    return executable_->module_config();
  }

  tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> Execute(
      std::vector<se::DeviceAddressBase> buffers,
      const NanoRtExecutable::ExecuteOptions& options) override {
    TraceMe trace([&] {
      return TraceMeEncode("NanoRtExecutable::Execute",
                           {{"name", executable_->module().name()}});
    });

    auto* executable = absl::down_cast<CpuExecutable*>(executable_.get());
    for (const auto& constant : executable->constants()) {
      if (constant.index >= 0) {
        buffers[constant.index] = constant.AsDeviceAddress();
      }
    }

    struct ExecutionContext {
      ExecutionContext(std::vector<se::DeviceAddressBase> buffers,
                       FunctionLibrary* function_library,
                       const NanoRtExecutable::ExecuteOptions& options)
          : allocations(std::move(buffers)),
            task_runner(options.intra_op_thread_pool()
                            ? std::make_optional<ThreadPoolTaskRunner>(
                                  options.intra_op_thread_pool()->getPool())
                            : std::nullopt),
            execute_params(Thunk::ExecuteParams{
                function_library, &allocations,
                /*xfeed=*/nullptr, options.intra_op_thread_pool(),
                task_runner.has_value() ? &*task_runner : nullptr}),
            collective_execute_params(
                RunId(options.launch_id()), options.local_device_id().value(),
                GlobalDeviceId(options.global_device_id()),
                options.device_assignment(), /*collectives=*/nullptr),
            custom_call_execute_params(
                RunId(options.launch_id()), options.local_device_id().value(),
                options.intra_op_thread_pool(), options.ffi_context()) {
        execute_params.collective_params = &collective_execute_params;
        execute_params.custom_call_params = &custom_call_execute_params;
      }

      cpu::BufferAllocations allocations;
      std::optional<ThreadPoolTaskRunner> task_runner;
      Thunk::ExecuteParams execute_params;
      Thunk::CollectiveExecuteParams collective_execute_params;
      Thunk::CustomCallExecuteParams custom_call_execute_params;
    };

    if (options.intra_op_thread_pool() || options.ffi_context() ||
        options.device_assignment()) {
      auto execution_context = std::make_unique<ExecutionContext>(
          std::move(buffers), executable->function_library(), options);

      auto execute_event =
          executable->thunks().Execute(execution_context->execute_params);

      execute_event.AndThen(
          [execution_context = std::move(execution_context)] {});

      return execute_event;
    }

    cpu::BufferAllocations allocations(std::move(buffers));
    Thunk::ExecuteParams execute_params{
        executable->function_library(), &allocations,
        /*xfeed=*/nullptr, options.intra_op_thread_pool(),
        /*task_runner=*/nullptr};
    return executable->thunks().Execute(execute_params);
  }

 private:
  std::unique_ptr<Executable> executable_;
};

absl::StatusOr<std::unique_ptr<NanoRtExecutable>> CreateFromExecutable(
    std::unique_ptr<Executable> executable,
    std::optional<ProgramShape> program_shape) {
  const HloModule& module = executable->module();
  auto* cpu_executable = absl::down_cast<CpuExecutable*>(executable.get());
  if (cpu_executable == nullptr) {
    return Internal("NanoRtExecutable requires CPU executable");
  }
  if (!cpu_executable->has_thunks()) {
    return Internal("NanoRtExecutable requires CPU executable to use thunks");
  }

  ABSL_ASSIGN_OR_RETURN(
      std::vector<size_t> argument_to_allocation_index,
      ResolveArgumentsMapping(module, cpu_executable->buffer_assignment()));
  ABSL_ASSIGN_OR_RETURN(
      std::vector<size_t> result_to_allocation_index,
      ResolveResultMapping(module, cpu_executable->buffer_assignment()));
  ABSL_ASSIGN_OR_RETURN(
      std::optional<size_t> temp_allocation_index,
      ResolveTempAllocationIndex(cpu_executable->buffer_assignment()));

  const auto& buffer_assignment = cpu_executable->buffer_assignment();
  std::vector<size_t> allocation_sizes(buffer_assignment.Allocations().size());
  for (const BufferAllocation& allocation : buffer_assignment.Allocations()) {
    allocation_sizes[allocation.index()] = allocation.size();
  }

  auto runner = std::make_unique<CpuExecutableRunner>(std::move(executable));
  return std::make_unique<NanoRtExecutable>(
      std::move(runner), std::move(allocation_sizes),
      std::move(argument_to_allocation_index),
      std::move(result_to_allocation_index), temp_allocation_index,
      std::move(program_shape));
}

absl::StatusOr<std::unique_ptr<NanoRtExecutable>> CreateFromAotResult(
    CompilationResultProto aot_compilation_result,
    std::optional<ProgramShape> program_shape) {
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      CpuAotLoader::LoadExecutable(std::move(aot_compilation_result)));
  return CreateFromExecutable(std::move(executable), std::move(program_shape));
}

const bool kRegisterNanoRtImporters = [] {
  NanoRtExecutable::RegisterAotImporter(&CreateFromAotResult);
  return true;
}();

}  // namespace

absl::StatusOr<std::unique_ptr<NanoRtExecutable>> NanoRtExecutable::Create(
    std::unique_ptr<Executable> executable,
    std::optional<ProgramShape> program_shape) {
  return CreateFromExecutable(std::move(executable), std::move(program_shape));
}

}  // namespace xla::cpu
