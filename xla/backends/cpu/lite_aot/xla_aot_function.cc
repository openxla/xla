/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/cpu/lite_aot/xla_aot_function.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/nanort/nanort_executable.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

namespace {

struct ExecutableAndSupportingBuffers {
  std::unique_ptr<NanoRtExecutable> nanort_executable;
  std::vector<XlaAotFunction::AlignedBuffer> results_buffers;
  XlaAotFunction::AlignedBuffer temp_buffer;
};

absl::StatusOr<ProgramShape> GetProgramShape(
    const NanoRtExecutable& nanort_executable) {
  auto maybe_nanort_program_shape = nanort_executable.program_shape();
  if (!maybe_nanort_program_shape.has_value()) {
    return absl::InternalError(
        "Program shape was not set in the NanoRtExecutable.");
  }
  return maybe_nanort_program_shape.value();
}

absl::StatusOr<ExecutableAndSupportingBuffers>
CreateExecutableAndSupportingBuffers(
    const CompilationResultProto& compilation_result) {
  ABSL_ASSIGN_OR_RETURN(
      ProgramShape program_shape,
      ProgramShape::FromProto(
          compilation_result.hlo_module().hlo_module().host_program_shape()));

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<NanoRtExecutable> nanort_executable,
      NanoRtExecutable::Create(compilation_result, program_shape));

  std::vector<XlaAotFunction::AlignedBuffer> results_buffers;

  ABSL_ASSIGN_OR_RETURN(auto nanort_program_shape,
                        GetProgramShape(*nanort_executable));
  if (nanort_program_shape.result().IsTuple()) {
    auto tuple_shapes = nanort_program_shape.result().tuple_shapes();
    results_buffers.reserve(tuple_shapes.size());
    for (const Shape& shape : tuple_shapes) {
      results_buffers.emplace_back(
          static_cast<size_t>(ShapeUtil::ByteSizeOf(shape)));
    }
  } else {
    results_buffers.emplace_back(static_cast<size_t>(
        ShapeUtil::ByteSizeOf(nanort_program_shape.result())));
  }

  XlaAotFunction::AlignedBuffer temp_buffer(
      nanort_executable->temp_buffer_size());

  return ExecutableAndSupportingBuffers{std::move(nanort_executable),
                                        std::move(results_buffers),
                                        std::move(temp_buffer)};
}

bool AreStringsInVectorUnique(const std::vector<std::string>& strings) {
  absl::flat_hash_set<absl::string_view> unique_strings(strings.begin(),
                                                        strings.end());
  return unique_strings.size() == strings.size();
}
}  // namespace

absl::StatusOr<std::unique_ptr<XlaAotFunction>> XlaAotFunction::Create(
    const CompilationResultProto& compilation_result,
    std::vector<std::string> arg_names, std::vector<std::string> result_names) {
  if (!AreStringsInVectorUnique(arg_names)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Argument names must be unique. Got ", absl::StrJoin(arg_names, ",")));
  }
  if (!AreStringsInVectorUnique(result_names)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Result names must be unique. Got ", absl::StrJoin(result_names, ",")));
  }

  ABSL_ASSIGN_OR_RETURN(
      auto executable_and_supporting_buffers,
      CreateExecutableAndSupportingBuffers(compilation_result));

  ABSL_ASSIGN_OR_RETURN(
      auto program_shape,
      GetProgramShape(*executable_and_supporting_buffers.nanort_executable));

  if (program_shape.parameters_size() != arg_names.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Argument names size does not match the number "
                     "of arguments in the program shape. Got ",
                     arg_names.size(), " argument names but program shape has ",
                     program_shape.parameters_size(),
                     " arguments. Program shape: ", program_shape.ToString()));
  }

  if (executable_and_supporting_buffers.results_buffers.size() !=
      result_names.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Result names size does not match the number "
        "of results in the program shape. Got ",
        result_names.size(), " result names but program shape has ",
        executable_and_supporting_buffers.results_buffers.size(),
        " results. Program shape: ", program_shape.ToString()));
  }

  return absl::WrapUnique(new XlaAotFunction(
      std::move(executable_and_supporting_buffers.nanort_executable),
      std::move(executable_and_supporting_buffers.results_buffers),
      std::move(executable_and_supporting_buffers.temp_buffer),
      std::move(arg_names), std::move(result_names)));
}

absl::StatusOr<std::unique_ptr<XlaAotFunction>> XlaAotFunction::Create(
    const CompilationResultProto& compilation_result) {
  ABSL_ASSIGN_OR_RETURN(
      auto executable_and_supporting_buffers,
      CreateExecutableAndSupportingBuffers(compilation_result));

  auto& nanort_executable = executable_and_supporting_buffers.nanort_executable;
  auto& results_buffers = executable_and_supporting_buffers.results_buffers;
  auto& temp_buffer = executable_and_supporting_buffers.temp_buffer;

  const HloModuleProto& hlo_module_proto =
      compilation_result.hlo_module().hlo_module();
  const HloComputationProto* entry_computation = nullptr;
  for (const HloComputationProto& computation :
       hlo_module_proto.computations()) {
    if (computation.id() == hlo_module_proto.entry_computation_id()) {
      entry_computation = &computation;
      break;
    }
  }
  if (entry_computation == nullptr &&
      !hlo_module_proto.computations().empty()) {
    entry_computation = &hlo_module_proto.computations(
        hlo_module_proto.computations_size() - 1);
  }
  if (entry_computation == nullptr) {
    return absl::InternalError(
        "Cannot infer argument and result names because HLO module has no "
        "entry computation.");
  }

  ABSL_ASSIGN_OR_RETURN(auto program_shape,
                        GetProgramShape(*nanort_executable));
  std::vector<std::string> arg_names(program_shape.parameters_size());
  absl::string_view root_name;
  for (const HloInstructionProto& instr : entry_computation->instructions()) {
    if (instr.opcode() == "parameter" && instr.parameter_number() >= 0 &&
        instr.parameter_number() < static_cast<int64_t>(arg_names.size())) {
      arg_names[instr.parameter_number()] = instr.name();
    }
    if (instr.id() == entry_computation->root_id()) {
      root_name = instr.name();
    }
  }
  if (root_name.empty() && !entry_computation->instructions().empty()) {
    root_name = entry_computation
                    ->instructions(entry_computation->instructions_size() - 1)
                    .name();
  }

  std::vector<std::string> result_names;
  if (program_shape.result().IsTuple()) {
    auto tuple_shapes = program_shape.result().tuple_shapes();
    result_names.reserve(tuple_shapes.size());
    for (int index = 0; index < tuple_shapes.size(); ++index) {
      result_names.push_back(absl::StrCat(root_name, "_", index));
    }

  } else {
    result_names.push_back(std::string(root_name));
  }

  CHECK(AreStringsInVectorUnique(arg_names))
      << "Argument names must be unique. Got " << absl::StrJoin(arg_names, ",");
  CHECK(AreStringsInVectorUnique(result_names))
      << "Result names must be unique. Got "
      << absl::StrJoin(result_names, ",");

  return absl::WrapUnique(new XlaAotFunction(
      std::move(nanort_executable), std::move(results_buffers),
      std::move(temp_buffer), std::move(arg_names), std::move(result_names)));
}

XlaAotFunction::XlaAotFunction(std::unique_ptr<NanoRtExecutable> executable,
                               std::vector<AlignedBuffer> results_buffers,
                               AlignedBuffer temp_buffer,
                               std::vector<std::string> argument_names,
                               std::vector<std::string> result_names)
    : executable_(std::move(executable)),
      results_buffers_(std::move(results_buffers)),
      temp_buffer_(std::move(temp_buffer)) {
  VLOG(2) << "Creating XlaAotFunction with " << argument_names.size()
          << " arguments and " << result_names.size() << " results.";
  VLOG(5) << "Argument names: " << absl::StrJoin(argument_names, ",");
  VLOG(5) << "Result names: " << absl::StrJoin(result_names, ",");
  auto program_shape = executable_->program_shape().value();
  arguments_.reserve(program_shape.parameters_size());
  for (size_t i = 0; i < program_shape.parameters_size(); ++i) {
    argument_sizes_.push_back(
        ShapeUtil::ByteSizeOfElements(program_shape.parameters(i)));
    arguments_.emplace_back(nullptr, 0);
    name_to_argument_index_[argument_names[i]] = i;
  }

  results_.reserve(results_buffers_.size());
  for (size_t i = 0; i < results_buffers_.size(); ++i) {
    auto& result_buffer = results_buffers_[i];
    results_.emplace_back(result_buffer.untyped_data(),
                          result_buffer.size_bytes());
    name_to_result_index_[result_names[i]] = i;
  }

  temp_ = NanoRtExecutable::PreallocatedTemp(
      static_cast<std::byte*>(temp_buffer_.untyped_data()),
      temp_buffer_.size_bytes());
}

absl::Status XlaAotFunction::Execute() {
  auto event = executable_->Execute(arguments_, results_, temp_);
  tsl::BlockUntilReady(event);
  if (event.IsError()) {
    return event.GetError();
  }
  return absl::OkStatus();
}

}  // namespace xla::cpu
