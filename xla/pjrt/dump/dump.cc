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
#include "xla/pjrt/dump/dump.h"

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/pjrt/dump/mlir.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/path.h"

namespace pjrt {

absl::StatusOr<std::string> ResolveTestingDumpPath(absl::string_view dump_to) {
  std::string dump_to_lower = absl::AsciiStrToLower(dump_to);
  if (dump_to_lower == "sponge" ||
      dump_to_lower == "test_undeclared_outputs_dir") {
    if (tsl::io::GetTestUndeclaredOutputsDir(&dump_to_lower)) {
      return dump_to_lower;
    }
    return absl::InvalidArgumentError(
        "Failed to get test undeclared outputs directory.");
  }
  return std::string(dump_to);
}

absl::StatusOr<std::string> GetDumpSubdirPath(absl::string_view dump_to_path,
                                              absl::string_view module_name) {
  if (dump_to_path.empty()) {
    return "";
  }
  absl::Time now = absl::Now();
  std::string timestamp = std::to_string(absl::ToUnixMillis(now));
  std::string dump_subdir = tsl::io::JoinPath(
      dump_to_path, absl::StrCat(module_name, "_", timestamp));
  TF_RETURN_IF_ERROR(tsl::Env::Default()->RecursivelyCreateDir(dump_subdir));
  return dump_subdir;
}

absl::Status DumpCompileInputs(absl::string_view dump_to_path,
                               xla::CompileOptions compile_options,
                               mlir::ModuleOp module,
                               const xla::PjRtTopologyDescription& topology) {
  TF_ASSIGN_OR_RETURN(std::string path, ResolveTestingDumpPath(dump_to_path));
  std::string module_name = module.getName().has_value()
                                ? std::string(module.getName().value())
                                : "unknown_module";
  TF_ASSIGN_OR_RETURN(std::string dump_sub_dir,
                      GetDumpSubdirPath(path, module_name));

  if (dump_sub_dir.empty()) {
    return absl::OkStatus();
  }

  // Dump module to file.
  std::string module_file_name = tsl::io::JoinPath(dump_sub_dir, "module.mlir");
  LOG(INFO) << "Dumping module to " << module_file_name;
  TF_RETURN_IF_ERROR(pjrt::MlirModuleToFile(module, module_file_name));

  // Dump compile options to file.
  std::string options_file_name =
      tsl::io::JoinPath(dump_sub_dir, "compile_options.pb");
  LOG(INFO) << "Dumping compile options to " << options_file_name;
  TF_RETURN_IF_ERROR(tsl::WriteStringToFile(
      tsl::Env::Default(), options_file_name,
      compile_options.ToProto().value().SerializeAsString()));

  std::string topology_file_name =
      tsl::io::JoinPath(dump_sub_dir, "topology.pb");

  LOG(INFO) << "Dumping topology to " << topology_file_name;
  TF_ASSIGN_OR_RETURN(auto topology_proto, topology.ToProto());
  TF_RETURN_IF_ERROR(
      tsl::WriteStringToFile(tsl::Env::Default(), topology_file_name,
                             topology_proto.SerializeAsString()));
  return absl::OkStatus();
}

}  // namespace pjrt
