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

#ifndef XLA_SERVICE_CPU_CPU_COMPILER_H_
#define XLA_SERVICE_CPU_CPU_COMPILER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes.h"
#include "xla/cpu_function_runtime.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/buffer_info_util.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_profile_printer_data.pb.h"
#include "xla/service/llvm_compiler.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace mlir {
class DialectRegistry;
}  // namespace mlir

namespace xla {
namespace cpu {

class CpuExecutable;

// This class wraps the configurability options that LLVM exposes including: the
// target triple, the target cpu and the target features.  It also includes the
// desired linkage name for the computation entry point.
class CpuAotCompilationOptions : public AotCompilationOptions {
 public:
  // Relocation models available for compilation.
  enum class RelocationModel {
    // Corresponds to the -fno-pic compiler option.
    Static,
    // Corresponds to the -fpic compiler option.
    SmallPic,
    // Corresponds to the -fPIC compiler option.
    BigPic,
    // Corresponds to the -fpie compiler option.
    SmallPie,
    // Corresponds to the -fPIE compiler option.
    BigPie
  };

  CpuAotCompilationOptions(std::string triple, std::string cpu_name,
                           std::string features, std::string entry_point_name,
                           RelocationModel relocation_model);

  ~CpuAotCompilationOptions() override;

  se::Platform::Id PlatformId() const override;

  // The triple used for compilation, similar to clang's -target flag.
  const std::string& triple() const { return triple_; }
  // The CPU name used for compilation, similar to clang's -mcpu flag.
  const std::string& cpu_name() const { return cpu_name_; }
  // The target features used for compilation ("+avx2", "+neon", etc).
  const std::string& features() const { return features_; }
  // The name to be used for the compiled code's entry point.
  const std::string& entry_point_name() const { return entry_point_name_; }
  // The relocation model used for compilation.
  RelocationModel relocation_model() const { return relocation_model_; }

 private:
  const std::string triple_;
  const std::string cpu_name_;
  const std::string features_;
  const std::string entry_point_name_;
  const RelocationModel relocation_model_;
};

class CpuAotCompilationResult : public AotCompilationResult {
 public:
  CpuAotCompilationResult(
      ObjectFileData object_file_data,
      std::vector<cpu_function_runtime::BufferInfo> buffer_infos,
      int64_t result_buffer_index, std::unique_ptr<HloModule> module,
      std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data);
  ~CpuAotCompilationResult() override = default;

  HloProfilePrinterData* hlo_profile_printer_data() const {
    return hlo_profile_printer_data_.get();
  }

  const ObjectFileData& object_file_data() const { return object_file_data_; }
  const std::vector<cpu_function_runtime::BufferInfo>& buffer_infos() const {
    return buffer_infos_;
  }
  int64_t result_buffer_index() const { return result_buffer_index_; }

  const HloModule* optimized_module() const override;
  std::unique_ptr<HloModule> consume_optimized_module() override;

 private:
  // Contains the compiled computation: an object file.
  const ObjectFileData object_file_data_;

  // A list of BufferInfo objects describing the buffers used by the XLA
  // computation.
  const std::vector<cpu_function_runtime::BufferInfo> buffer_infos_;

  // Contains which buffer index into |buffer_sizes| was designated to the
  // result of the computation.  This buffer should be passed into the output
  // parameter when calling the compiled computation.
  const int64_t result_buffer_index_;

  // Contains the optimized HLO module.
  std::unique_ptr<HloModule> module_;

  // Contains an instance of HloProfilePrinterData if HLO profiling is enabled,
  // otherwise is nullptr.
  std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data_;
};

// This is a result of exporting JIT compiled CpuExecutable to AOT compilation
// result that can be saved on disk and shipped over the wire.
class CpuExecutableAotCompilationResult : public AotCompilationResult {
 public:
  static absl::StatusOr<std::unique_ptr<CpuExecutableAotCompilationResult>>
  Create(const HloModule* hlo_module, const BufferAssignment* buffer_assignment,
         absl::string_view function_name, std::vector<std::string> obj_files,
         std::vector<SymbolProto> symbols, const ThunkSequence* thunks,
         CompilationResultProto::ObjFileKind obj_file_kind,
         std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data) {
    std::optional<ThunkSequenceProto> thunk_proto;

    if (thunks != nullptr) {
      ThunkSequenceSerDesProtobuf thunk_sequence_serdes(
          &buffer_assignment->Allocations());
      TF_ASSIGN_OR_RETURN(thunk_proto, thunk_sequence_serdes.ToProto(*thunks));
    }

    std::vector<cpu_function_runtime::BufferInfo> buffer_infos;
    std::optional<size_t> temp_allocation_index;

    if (buffer_assignment) {
      buffer_infos = CreateBufferInfosFromBufferAssignment(
          *hlo_module,
          *buffer_assignment);  // Find temp allocation index if it exists
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

    return absl::WrapUnique(new CpuExecutableAotCompilationResult(
        hlo_module, buffer_assignment, function_name, std::move(obj_files),
        std::move(symbols), thunk_proto, obj_file_kind,
        std::move(temp_allocation_index), std::move(buffer_infos),
        std::move(hlo_profile_printer_data)));
  }

  absl::StatusOr<std::string> SerializeAsString() const override {
    return proto_.SerializeAsString();
  }

  absl::StatusOr<std::vector<std::byte>> SerializeToByteVector() const {
    size_t serialized_size = proto_.ByteSizeLong();
    std::vector<std::byte> buffer(serialized_size);
    if (!proto_.SerializeToArray(reinterpret_cast<void*>(buffer.data()),
                                 serialized_size)) {
      return Internal(
          "Failed to serialize CpuExecutableAotCompilationResult to byte "
          "vector.");
    }

    return buffer;
  }

  const CompilationResultProto& proto() const { return proto_; }

  std::optional<size_t> temp_allocation_index() const {
    return temp_allocation_index_;
  }

  const std::vector<cpu_function_runtime::BufferInfo>& buffer_infos() const {
    return buffer_infos_;
  }

  const HloProfilePrinterData* hlo_profile_printer_data() const {
    return hlo_profile_printer_data_.get();
  }

  static absl::StatusOr<std::unique_ptr<CpuExecutableAotCompilationResult>>
  FromString(const std::string& serialized) {
    CompilationResultProto proto;
    if (!proto.ParseFromString(serialized)) {
      return Internal(
          "Failed to parse serialized CpuExecutableAotCompilationResult.");
    }

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> module,
        HloModule::CreateFromProtoWithConfig(proto.hlo_module()));

    return std::unique_ptr<CpuExecutableAotCompilationResult>(
        new CpuExecutableAotCompilationResult(proto, std::move(module)));
  }

  static absl::StatusOr<std::unique_ptr<CpuExecutableAotCompilationResult>>
  FromByteVector(const std::vector<std::byte>& serialized) {
    CompilationResultProto proto;
    if (!proto.ParseFromArray(serialized.data(), serialized.size())) {
      return Internal(
          "Failed to parse serialized CpuExecutableAotCompilationResult.");
    }

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> module,
        HloModule::CreateFromProtoWithConfig(proto.hlo_module()));

    return std::unique_ptr<CpuExecutableAotCompilationResult>(
        new CpuExecutableAotCompilationResult(proto, std::move(module)));
  }

  absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      Compiler* compiler, const se::StreamExecutor* stream_exec) const override;

  const HloModule* optimized_module() const override { return module_.get(); }

  std::unique_ptr<HloModule> consume_optimized_module() override {
    return std::move(module_);
  }

 private:
  CpuExecutableAotCompilationResult(
      const HloModule* hlo_module, const BufferAssignment* buffer_assignment,
      absl::string_view function_name, std::vector<std::string> obj_files,
      std::vector<SymbolProto> symbols,
      const std::optional<ThunkSequenceProto>& thunks,
      CompilationResultProto::ObjFileKind obj_file_kind,
      std::optional<size_t> temp_allocation_index,
      std::vector<cpu_function_runtime::BufferInfo> buffer_infos,
      std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data)
      : temp_allocation_index_(temp_allocation_index),
        buffer_infos_(std::move(buffer_infos)),
        hlo_profile_printer_data_(std::move(hlo_profile_printer_data)) {
    *proto_.mutable_hlo_module()->mutable_hlo_module() = hlo_module->ToProto();
    *proto_.mutable_hlo_module()->mutable_config() =
        hlo_module->config().ToProto();
    *proto_.mutable_buffer_assignment() = buffer_assignment->ToProto();
    proto_.set_entry_function_name(std::string(function_name));
    for (std::string& obj_file : obj_files) {
      proto_.add_obj_files(std::move(obj_file));
    }

    for (const auto& symbol : symbols) {
      auto* symbol_proto = proto_.add_compiled_symbols();
      *symbol_proto = symbol;
    }
    proto_.set_obj_files_kind(obj_file_kind);
    module_ = hlo_module->Clone();

    if (thunks.has_value()) {
      ThunkSequenceSerDesProtobuf thunk_sequence_serdes(
          &buffer_assignment->Allocations());
      *proto_.mutable_thunk_sequence() = *thunks;
    }
  }

  explicit CpuExecutableAotCompilationResult(CompilationResultProto proto,
                                             std::unique_ptr<HloModule> module)
      : proto_(std::move(proto)), module_(std::move(module)) {}

  CompilationResultProto proto_;
  std::unique_ptr<HloModule> module_;
  std::optional<size_t> temp_allocation_index_;
  std::vector<cpu_function_runtime::BufferInfo> buffer_infos_;
  // Contains an instance of HloProfilePrinterData if HLO profiling is enabled,
  // otherwise is nullptr.
  std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data_;
};

// CPU-targeting implementation of the XLA Compiler interface.
//
// The compiler translates XLA HLO code into LLVM IR and uses LLVM's JIT
// infrastructure to create an executable "blob" that can then be returned
// wrapped in CpuExecutable and actually invoked.
class CpuCompiler : public LLVMCompiler {
 public:
  CpuCompiler();
  ~CpuCompiler() override = default;

  absl::StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_execs,
      const CompileOptions& options) override;

  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  absl::StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     const AotCompilationOptions& options) override;

  se::Platform::Id PlatformId() const override;

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

  absl::StatusOr<std::unique_ptr<AotCompilationResult>> Export(
      Executable* executable) const override;

  // Returns a (deserialized) AotCompilationResult from a serialized
  // AotCompilationResult.
  absl::StatusOr<std::unique_ptr<AotCompilationResult>>
  LoadAotCompilationResult(const std::string& serialized_aot_result) override;

  // Returns a (deserialized) AotCompilationResult from a serialized
  // AotCompilationResult.
  absl::StatusOr<std::unique_ptr<AotCompilationResult>>
  LoadAotCompilationResult(const std::vector<std::byte>& serialized_aot_result);

  absl::StatusOr<HloSchedule> CreateHloSchedule(
      const HloModule& hlo_module) const;

  absl::StatusOr<std::unique_ptr<BufferAssignment>> CreateBufferAssignment(
      const HloModule& module) const;

 private:
  absl::StatusOr<std::unique_ptr<CpuExecutable>>
  CompileCpuExecutableAheadOfTime(
      std::unique_ptr<HloModule> module,
      std::shared_ptr<llvm::TargetMachine> target_machine,
      const CpuAotCompilationOptions& aot_options, const llvm::Triple& triple,
      const llvm::PICLevel::Level& pic_level,
      const llvm::PIELevel::Level& pie_level);

  // Initialize the LLVM target.
  static void InitializeLLVMTarget();

  // Runs the HLO passes which are necessary for both optimizations and
  // correctness.
  absl::Status RunHloPasses(HloModule* module, bool is_aot_compile,
                            llvm::TargetMachine* target_machine,
                            const CompileOptions& compile_options);

  // Runs HLO passes up to and including layout assignment.
  absl::Status RunHloPassesThroughLayoutAssn(
      HloModule* module, bool /*is_aot_compile*/,
      TargetMachineFeatures* target_machine_features);

  // Runs HLO passes after layout assignment.
  absl::Status RunHloPassesAfterLayoutAssn(
      HloModule* module, bool is_aot_compile,
      TargetMachineFeatures* target_machine_features,
      const CompileOptions& compile_options);

  absl::StatusOr<std::unique_ptr<CpuExecutable>> CompileCpuExecutable(
      std::unique_ptr<HloModule> module);

  CpuCompiler(const CpuCompiler&) = delete;
  CpuCompiler& operator=(const CpuCompiler&) = delete;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_CPU_COMPILER_H_
