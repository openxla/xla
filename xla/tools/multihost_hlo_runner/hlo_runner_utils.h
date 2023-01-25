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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_MULTIHOST_HLO_RUNNER_HLO_RUNNER_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_MULTIHOST_HLO_RUNNER_HLO_RUNNER_UTILS_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "xla/service/computation_placer.h"
#include "xla/tools/multihost_hlo_runner/hlo_runner.h"

namespace xla::multihost_hlo_runner::hlo_runner_utils {

struct HloModuleAndMetadata {
  std::unique_ptr<HloModule> hlo_module;
  std::optional<DeviceAssignment> device_assignment;
  std::optional<ExecutionOptions> execution_options;
  std::optional<std::string> flagfile;
};
// Load HLO module and metadata from a file.
StatusOr<HloModuleAndMetadata> LoadHloModule(
    absl::string_view hlo_file, InputFormat input_format,
    absl::string_view device_assignment_proto_path);

ExecutionOptions LoadExecutionOptions(absl::string_view execution_options_path);

}  // namespace xla::multihost_hlo_runner::hlo_runner_utils

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_MULTIHOST_HLO_RUNNER_HLO_RUNNER_UTILS_H_
