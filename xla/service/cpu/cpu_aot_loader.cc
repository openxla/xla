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

#include "xla/service/cpu/cpu_aot_loader.h"

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/codegen/aot_compiled_function_library.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

absl::StatusOr<std::vector<FunctionLibrary::Symbol>>
GetCompiledSymbolsFromProto(
    absl::Span<const SymbolProto> compiled_symbols_proto) {
  std::vector<FunctionLibrary::Symbol> compiled_symbols;
  compiled_symbols.reserve(compiled_symbols_proto.size());
  for (const auto& symbol_proto : compiled_symbols_proto) {
    switch (symbol_proto.function_type_id()) {
      case SymbolProto::KERNEL:
        compiled_symbols.push_back(
            FunctionLibrary::Sym<FunctionLibrary::Kernel>(symbol_proto.name()));
        break;
      case SymbolProto::COMPARATOR:
        compiled_symbols.push_back(
            FunctionLibrary::Sym<FunctionLibrary::Comparator>(
                symbol_proto.name()));
        break;
      default:
        return Internal(
            "Unknown function type id %s",
            SymbolProto_FunctionTypeId_Name(symbol_proto.function_type_id()));
    }
  }
  VLOG(3) << "Collected " << compiled_symbols.size() << " compiled symbols";
  for (const auto& symbol : compiled_symbols) {
    VLOG(3) << " Symbol: " << symbol.name;
  }

  return compiled_symbols;
}

absl::StatusOr<std::unique_ptr<FunctionLibrary>> LoadFunctionLibrary(
    const std::vector<FunctionLibrary::Symbol>& compiled_symbols,
    absl::Span<const ObjFileProto> obj_files,
    absl::string_view data_layout_str) {
  std::vector<std::string> raw_obj_files;
  raw_obj_files.reserve(obj_files.size());
  for (const auto& obj_file : obj_files) {
    raw_obj_files.push_back(obj_file.contents());
  }
  return AotObjectLoader::LoadFunctionLibrary(compiled_symbols, raw_obj_files);
}

absl::StatusOr<std::unique_ptr<FunctionLibrary>> LoadFunctionLibrary(
    const std::vector<FunctionLibrary::Symbol>& compiled_symbols,
    absl::Span<const ObjFileProto> obj_files, const HloModule* hlo_module,
    const TargetMachineOptions& target_machine_options,
    absl::string_view data_layout_str) {
  return LoadFunctionLibrary(compiled_symbols, obj_files, data_layout_str);
}

absl::StatusOr<std::unique_ptr<FunctionLibrary>>
CpuAotLoader::LoadFunctionLibrary(
    const xla::cpu::CompilationResultProto& aot_result_proto) {
  std::vector<SymbolProto> compiled_symbols_proto(
      aot_result_proto.compiled_symbols().begin(),
      aot_result_proto.compiled_symbols().end());
  ABSL_ASSIGN_OR_RETURN(auto compiled_symbols,
                        GetCompiledSymbolsFromProto(compiled_symbols_proto));

  std::vector<std::string> raw_obj_files;
  raw_obj_files.reserve(aot_result_proto.object_files_size());
  for (const auto& obj_file : aot_result_proto.object_files()) {
    raw_obj_files.push_back(obj_file.contents());
  }
  return AotObjectLoader::LoadFunctionLibrary(compiled_symbols, raw_obj_files);
}

}  // namespace xla::cpu
