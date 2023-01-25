/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tools/multihost_hlo_runner/hlo_runner_utils.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/tests/test_utils.h"

namespace xla::multihost_hlo_runner::hlo_runner_utils {

StatusOr<std::unique_ptr<HloModule>> ReadHloModuleFromFile(
    absl::string_view hlo_file, InputFormat input_format) {
  switch (input_format) {
    case InputFormat::kText: {
      return MultiHostHloRunner::ReadModuleFromHloTextFile(hlo_file);
    }
    case InputFormat::kProtoBinary: {
      return MultiHostHloRunner::ReadModuleFromBinaryProtoFile(hlo_file);
    }
    case InputFormat::kProtoText: {
      return MultiHostHloRunner::ReadModuleFromTextProtoFile(hlo_file);
    }
    case InputFormat::kSnapshotProtoBinary: {
      TF_ASSIGN_OR_RETURN(
          MultiHostHloRunner::HloModuleAndArguments hlo_module_and_arguments,
          MultiHostHloRunner::LoadHloModuleAndArguments(hlo_file,
                                                        input_format));
      std::unique_ptr<HloModule> hlo_module =
          std::move(hlo_module_and_arguments.hlo_module);
      return hlo_module;
    }
  }
}

StatusOr<HloModuleAndMetadata> LoadHloModule(
    absl::string_view hlo_file, InputFormat input_format,
    absl::string_view device_assignment_proto_path) {
  CHECK(!hlo_file.empty()) << "No valid source for HLO module.";

  HloModuleAndMetadata hlo_module_and_metadata;
  TF_ASSIGN_OR_RETURN(hlo_module_and_metadata.hlo_module,
                      ReadHloModuleFromFile(hlo_file, input_format));
  XLA_VLOG_LINES(3, "Loaded HLO module: " +
                        hlo_module_and_metadata.hlo_module->ToString());
  TF_RETURN_IF_ERROR(VerifyHloModule(hlo_module_and_metadata.hlo_module.get(),
                                     /*layout_sensitive=*/false,
                                     /*allow_mixed_precision=*/true));

  if (!device_assignment_proto_path.empty()) {
    xla::DeviceAssignmentProto device_assignment_proto;
    TF_RETURN_IF_ERROR(tsl::ReadBinaryProto(
        tsl::Env::Default(), std::string(device_assignment_proto_path),
        &device_assignment_proto));
    xla::StatusOr<std::unique_ptr<xla::DeviceAssignment>> device_assignment =
        xla::DeviceAssignment::Deserialize(device_assignment_proto);
    TF_RETURN_IF_ERROR(device_assignment.status());
    hlo_module_and_metadata.device_assignment = std::move(**device_assignment);
  }
  return hlo_module_and_metadata;
}

ExecutionOptions LoadExecutionOptions(
    absl::string_view execution_options_path) {
  ExecutionOptions execution_options;
  TF_QCHECK_OK(tsl::ReadTextOrBinaryProto(tsl::Env::Default(),
                                          std::string(execution_options_path),
                                          &execution_options));
  return execution_options;
}

}  // namespace xla::multihost_hlo_runner::hlo_runner_utils
