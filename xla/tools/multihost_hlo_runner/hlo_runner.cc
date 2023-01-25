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

#include "xla/tools/multihost_hlo_runner/hlo_runner.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/global_device_id.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_parser.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/test_utils.h"
#include "xla/tools/hlo_control_flow_flattening.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {
// Creates an HloModule from the given proto.
StatusOr<std::unique_ptr<HloModule>> HloTextToModule(
    absl::string_view hlo_text) {
  return ParseAndReturnUnverifiedModule(hlo_text);
}

// Creates an HloModule from the given proto.
StatusOr<std::unique_ptr<HloModule>> HloProtoToModule(
    const HloModuleProto& proto) {
  TF_ASSIGN_OR_RETURN(
      HloModuleConfig config,
      HloModule::CreateModuleConfigFromProto(proto, DebugOptions()));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      HloModule::CreateFromProto(proto, config));
  return std::move(module);
}

template <typename ElementType>
void PopulateWithSameValue(Literal* literal, ElementType val) {
  for (ElementType& element : literal->data<ElementType>()) {
    element = static_cast<ElementType>(val);
  }
}

StatusOr<Literal> MakeFakeLiteralWithSameValue(const Shape& shape, int value) {
  if (!shape.IsArray()) {
    return InvalidArgument(
        "MakeFakeLiteralWithSameValue does not support non-array type");
  }
  Shape new_shape = shape;
  new_shape.mutable_layout()->clear_tiles();
  Literal literal(new_shape);
  switch (new_shape.element_type()) {
    case BF16:
      PopulateWithSameValue(&literal, bfloat16(static_cast<float>(value)));
      break;
    case F16:
      PopulateWithSameValue(&literal, static_cast<half>(value));
      break;
    case F32:
      PopulateWithSameValue(&literal, static_cast<float>(value));
      break;
    case F64:
      PopulateWithSameValue(&literal, static_cast<double>(value));
      break;
    case S8:
      PopulateWithSameValue(&literal, static_cast<int8_t>(value));
      break;
    case U8:
      PopulateWithSameValue(&literal, static_cast<uint8_t>(value));
      break;
    case S16:
      PopulateWithSameValue(&literal, static_cast<int16_t>(value));
      break;
    case U16:
      PopulateWithSameValue(&literal, static_cast<uint16_t>(value));
      break;
    case S32:
      PopulateWithSameValue(&literal, static_cast<int32_t>(value));
      break;
    case U32:
      PopulateWithSameValue(&literal, static_cast<uint32_t>(value));
      break;
    case S64:
      PopulateWithSameValue(&literal, static_cast<int64_t>(value));
      break;
    case U64:
      PopulateWithSameValue(&literal, static_cast<uint64_t>(value));
      break;
    case C64:
      PopulateWithSameValue(&literal,
                            static_cast<complex64>(complex64(value, 0.0)));
      break;
    case C128:
      PopulateWithSameValue(&literal,
                            static_cast<complex128>(complex128(value, 0.0)));
      break;
    case PRED:
      PopulateWithSameValue(&literal, (value % 2) == 0);
      break;
    default:
      return Unimplemented("Unsupported type for fake literal generation: %s",
                           ShapeUtil::HumanString(shape));
  }
  return literal;
}

void AddShardingAnnotationsToSpmdPartitionedModule(HloModule* hlo_module) {
  auto set_manual_sharding = [](HloInstruction* hlo) {
    if (!hlo->has_sharding()) {
      hlo->set_sharding(
          HloSharding::Manual().NormalizeTupleSharding(hlo->shape()));
    }
  };
  for (int64_t i = 0; i < hlo_module->entry_computation()->num_parameters();
       ++i) {
    HloInstruction* param =
        hlo_module->entry_computation()->parameter_instruction(i);
    set_manual_sharding(param);
  }

  HloInstruction* entry_root =
      hlo_module->entry_computation()->root_instruction();
  set_manual_sharding(entry_root);
}

}  // namespace

bool AbslParseFlag(absl::string_view text, InputFormat* input_format,
                   std::string* error) {
  if (text == "text") {
    *input_format = InputFormat::kText;
    return true;
  }
  if (text == "proto_text") {
    *input_format = InputFormat::kProtoText;
    return true;
  }
  if (text == "proto_binary") {
    *input_format = InputFormat::kProtoBinary;
    return true;
  }
  if (text == "snapshot_proto_binary") {
    *input_format = InputFormat::kSnapshotProtoBinary;
    return true;
  }

  *error = "unknown value for enumeration";
  return false;
}

std::string AbslUnparseFlag(InputFormat input_format) {
  switch (input_format) {
    case InputFormat::kText:
      return "text";
    case InputFormat::kProtoText:
      return "proto_text";
    case InputFormat::kProtoBinary:
      return "proto_binary";
    case InputFormat::kSnapshotProtoBinary:
      return "snapshot_proto_binary";
    default:
      return absl::StrCat(input_format);
  }
}

bool AbslParseFlag(absl::string_view text,
                   MultiHostHloRunner::DeviceType* device_type,
                   std::string* error) {
  if (text == "gpu") {
    *device_type = xla::MultiHostHloRunner::DeviceType::kGpu;
    return true;
  }
  *error = "Unrecognized device type specified. Expected tpu or gpu";
  return false;
}

std::string AbslUnparseFlag(MultiHostHloRunner::DeviceType device_type) {
  switch (device_type) {
    case xla::MultiHostHloRunner::DeviceType::kGpu:
      return "gpu";
  }
}

bool AbslParseFlag(absl::string_view text,
                   MultiHostHloRunner::ModuleArgumentMode* argument_mode,
                   std::string* error) {
  if (text == "use_device_id_as_input") {
    *argument_mode =
        MultiHostHloRunner::ModuleArgumentMode::kUseDeviceIdAsInput;
    return true;
  }
  if (text == "use_random_inputs") {
    *argument_mode = MultiHostHloRunner::ModuleArgumentMode::kUseRandomInputs;
    return true;
  }
  if (text == "use_shared_random_inputs") {
    *argument_mode =
        MultiHostHloRunner::ModuleArgumentMode::kUseSharedRandomInputs;
    return true;
  }
  *error =
      "Unrecognized module argument mode specified. Expect "
      "\"use_device_id_as_input\", \"use_random_inputs\", or "
      "\"use_shared_random_inputs\".";
  return false;
}

std::string AbslUnparseFlag(
    MultiHostHloRunner::ModuleArgumentMode argument_mode) {
  switch (argument_mode) {
    case MultiHostHloRunner::ModuleArgumentMode::kUseDeviceIdAsInput:
      return "use_device_id_as_input";
    case MultiHostHloRunner::ModuleArgumentMode::kUseRandomInputs:
      return "use_random_inputs";
    case MultiHostHloRunner::ModuleArgumentMode::kUseSharedRandomInputs:
      return "use_shared_random_inputs";
    default:
      LOG(FATAL) << "Unexpected argument mode.";
  }
}

MultiHostHloRunner::MultiHostHloRunner(const Options& options,
                                       std::shared_ptr<PjRtClient> client)
    : client_(std::move(client)),
      log_output_(options.log_output_mode == LogOutputMode::kLogOutput),
      hlo_passes_mode_(options.hlo_passes_mode),
      use_spmd_partitioning_(options.spmd_mode ==
                             SpmdMode::kUseSpmdPartitioning),
      is_spmd_partitioned_module_(
          options.spmd_partitioned_mode ==
          SpmdPartitionedMode::kIsSpmdPartitionedModule),
      module_argument_mode_(options.module_argument_mode),
      module_output_mode_(options.module_output_mode),
      flatten_while_loop_(options.flatten_while_loop),
      while_execution_count_(options.while_execution_count),
      remove_infeed_outfeed_(options.remove_infeed_outfeed),
      num_repeats_(options.num_repeats),
      task_id_(options.task_id) {}

StatusOr<std::unique_ptr<PjRtClient>> MultiHostHloRunner::GetDeviceClient(
    DeviceType device_type, bool use_tfrt_client) {
  switch (device_type) {
    case DeviceType::kGpu:
      return GetStreamExecutorGpuClient(
          /*asynchronous=*/true, GpuAllocatorConfig(),
          /*distributed_client=*/nullptr, /*node_id=*/0);
    default:
      return xla::InvalidArgument("Unknown device type #%d", device_type);
  }
}

namespace {
void OverrideExecutableBuildOptionsFromExecutionOptions(
    const ExecutionOptions& execution_options,
    ExecutableBuildOptions& build_options) {
  if (execution_options.has_debug_options()) {
    *build_options.mutable_debug_options() = execution_options.debug_options();
    build_options.mutable_debug_options()->set_xla_dump_to("");
  }
  if (execution_options.has_shape_with_output_layout()) {
    build_options.set_result_layout(
        Shape(execution_options.shape_with_output_layout()));
  }
  build_options.set_num_replicas(execution_options.num_replicas());
  build_options.set_num_partitions(execution_options.num_partitions());
  build_options.set_use_spmd_partitioning(
      execution_options.use_spmd_partitioning());
  build_options.set_use_auto_spmd_partitioning(
      execution_options.use_auto_spmd_partitioning());
  build_options.set_deduplicate_hlo(execution_options.deduplicate_hlo());
  build_options.set_allow_spmd_sharding_propagation_to_output(
      execution_options.allow_spmd_sharding_propagation_to_output());
  if (execution_options.has_device_assignment()) {
    StatusOr<std::unique_ptr<DeviceAssignment>> device_assignment =
        DeviceAssignment::Deserialize(execution_options.device_assignment());
    TF_CHECK_OK(device_assignment.status());
    build_options.set_device_assignment(**device_assignment);
  }
  build_options.set_alias_passthrough_params(
      execution_options.alias_passthrough_params());
}

}  // namespace

StatusOr<std::unique_ptr<MultiHostHloRunner>>
MultiHostHloRunner::CreateMultiHostHloRunner(
    const Options& options, std::shared_ptr<PjRtClient> client) {
  auto hlo_runner_ptr =
      absl::WrapUnique(new MultiHostHloRunner(options, std::move(client)));

  if (options.execution_options.has_value()) {
    OverrideExecutableBuildOptionsFromExecutionOptions(
        *options.execution_options,
        hlo_runner_ptr->compile_options_.executable_build_options);
  }
  DebugOptions* debug_options =
      hlo_runner_ptr->compile_options_.executable_build_options
          .mutable_debug_options();
  VLOG(1) << "Create MultiHostHloRunner with task_id: "
          << hlo_runner_ptr->task_id_;
  if (hlo_runner_ptr->task_id_ == 0) {
    debug_options->set_xla_dump_to(options.xla_dump_to);
    debug_options->set_xla_dump_hlo_as_text(options.xla_text_dump_mode ==
                                            XlaTextDumpMode::kDumpAsText);
    debug_options->set_xla_dump_hlo_as_proto(options.xla_proto_dump_mode ==
                                             XlaProtoDumpMode::kDumpAsProto);
  }
  ExecutableBuildOptions& build_options =
      hlo_runner_ptr->compile_options_.executable_build_options;
  if (hlo_runner_ptr->use_spmd_partitioning_) {
    build_options.set_use_spmd_partitioning(true);
  }
  switch (hlo_runner_ptr->hlo_passes_mode_) {
    case HloPassesMode::kRunXLABackendOnly:
      build_options.set_run_backend_only(true);
      break;
    case HloPassesMode::kDisableAllHloPasses:
      debug_options->set_xla_disable_all_hlo_passes(true);
      break;
    case HloPassesMode::kStandardCompile:
      // Just use the default.
      break;
  }

  int my_num_replicas = -1;
  int my_num_partitions = -1;
  if (!options.num_replicas.has_value() ||
      !options.num_partitions.has_value()) {
    if (options.execution_options.has_value()) {
      my_num_replicas = options.execution_options->num_replicas();
      my_num_partitions = options.execution_options->num_partitions();
    } else {
      if (!options.num_replicas.has_value() &&
          !options.num_partitions.has_value()) {
        my_num_replicas = hlo_runner_ptr->client_->device_count();
        my_num_partitions = 1;
      } else if (!options.num_replicas.has_value()) {
        CHECK(options.num_partitions.has_value());
        my_num_replicas =
            hlo_runner_ptr->client_->device_count() / *options.num_partitions;
        my_num_partitions = *(options.num_partitions);
      } else if (!options.num_partitions.has_value()) {
        CHECK(options.num_replicas.has_value());
        my_num_partitions =
            hlo_runner_ptr->client_->device_count() / *options.num_replicas;
        my_num_replicas = *(options.num_replicas);
      }
    }
  } else {
    my_num_replicas = *(options.num_replicas);
    my_num_partitions = *(options.num_partitions);
  }
  CHECK_GE(my_num_replicas, 1);
  CHECK_GE(my_num_partitions, 1);
  build_options.set_num_replicas(my_num_replicas);
  build_options.set_num_partitions(my_num_partitions);

  if (!build_options.has_device_assignment()) {
    DeviceAssignment device_assignment =
        hlo_runner_ptr->client_
            ->GetDefaultDeviceAssignment(my_num_replicas, my_num_partitions)
            .value();
    build_options.set_device_assignment(device_assignment);
  }
  return hlo_runner_ptr;
}

StatusOr<std::unique_ptr<MultiHostHloRunner>>
MultiHostHloRunner::CreateMultiHostHloRunner(const Options& options,
                                             DeviceType device_type) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                      GetDeviceClient(device_type, options.use_tfrt_client));

  return CreateMultiHostHloRunner(options, std::move(client));
}

absl::Span<PjRtDevice* const> MultiHostHloRunner::local_devices() const {
  return client_->addressable_devices();
}

absl::Span<PjRtDevice* const> MultiHostHloRunner::devices() const {
  return client_->devices();
}

PjRtClient* MultiHostHloRunner::client() const { return client_.get(); }

StatusOr<MultiHostHloRunner::PerDeviceLiteralVecType>
MultiHostHloRunner::ParseAndRun(
    absl::string_view hlo_text,
    const MultiHostHloRunner::PerDeviceLiteralVecType& arguments) const {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      ReadModuleFromString(hlo_text));
  return CompileAndRun(hlo_module.get(), arguments);
}

StatusOr<MultiHostHloRunner::PerDeviceLiteralVecType>
MultiHostHloRunner::ParseAndRun(
    absl::string_view hlo_text, const LiteralVec& argument_literals,
    const PerDeviceIndexVecType& per_device_index_vec) const {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      ReadModuleFromString(hlo_text));
  return CompileAndRun(hlo_module.get(), argument_literals,
                       per_device_index_vec);
}

StatusOr<MultiHostHloRunner::HloModuleAndArguments>
MultiHostHloRunner::LoadHloModuleAndArguments(absl::string_view hlo_file,
                                              InputFormat input_format) {
  HloModuleAndArguments hlo_module_and_arguments;
  switch (input_format) {
    case InputFormat::kText: {
      std::string hlo_text;
      TF_ASSIGN_OR_RETURN(hlo_module_and_arguments.hlo_module,
                          ReadModuleFromHloTextFile(hlo_file));
    } break;
    case InputFormat::kProtoText: {
      TF_ASSIGN_OR_RETURN(hlo_module_and_arguments.hlo_module,
                          ReadModuleFromTextProtoFile(hlo_file));
    } break;
    case InputFormat::kProtoBinary: {
      TF_ASSIGN_OR_RETURN(hlo_module_and_arguments.hlo_module,
                          ReadModuleFromBinaryProtoFile(hlo_file));
    } break;
    case InputFormat::kSnapshotProtoBinary: {
      TF_ASSIGN_OR_RETURN(hlo_module_and_arguments,
                          ReadModuleFromSnapshotBinaryProtoFile(hlo_file));
    } break;
    default:
      LOG(FATAL) << "Cannot process input format: "
                 << AbslUnparseFlag(input_format);
  }
  return hlo_module_and_arguments;
}

StatusOr<MultiHostHloRunner::PerDeviceLiteralVecType>
MultiHostHloRunner::LoadAndRun(
    absl::Span<const std::string> hlo_files, InputFormat input_format,
    const MultiHostHloRunner::PerDeviceLiteralVecType& arguments) const {
  // We only support SPMD as of now, i.e., all devices are supposed
  // to execute the same HLO module.
  // Currently there is no mechanism to map the loaded arguments to
  // proper device ID, so loading and executing from HLO snapshot might not
  // replay the original execution.
  HloModuleAndArguments hlo_module_and_arguments;
  PerDeviceLiteralVecType loaded_arguments;
  for (int i = 0; i < hlo_files.size(); ++i) {
    TF_ASSIGN_OR_RETURN(hlo_module_and_arguments,
                        LoadHloModuleAndArguments(hlo_files[i], input_format));
    if (input_format == InputFormat::kSnapshotProtoBinary) {
      loaded_arguments[client_->devices()[i]->id()] =
          std::move(hlo_module_and_arguments.arguments);
    }
  }
  if (!arguments.empty()) {
    return CompileAndRun(hlo_module_and_arguments.hlo_module.get(), arguments);
  }
  return CompileAndRun(hlo_module_and_arguments.hlo_module.get(),
                       loaded_arguments);
}

StatusOr<MultiHostHloRunner::PerDeviceLiteralVecType>
MultiHostHloRunner::LoadAndRun(
    absl::Span<const std::string> hlo_files, InputFormat input_format,
    const LiteralVec& argument_literals,
    const PerDeviceIndexVecType& per_device_index_vec) const {
  CHECK(!hlo_files.empty());
  // We only support SPMD as of now, i.e., all devices are supposed
  // to execute the same HLO module.
  HloModuleAndArguments hlo_module_and_arguments;
  TF_ASSIGN_OR_RETURN(hlo_module_and_arguments,
                      LoadHloModuleAndArguments(hlo_files[0], input_format));
  return CompileAndRun(hlo_module_and_arguments.hlo_module.get(),
                       argument_literals, per_device_index_vec);
}

StatusOr<std::unique_ptr<HloModule>>
MultiHostHloRunner::ReadModuleFromHloTextFile(absl::string_view hlo_file) {
  std::string hlo_string;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(),
                                           std::string(hlo_file), &hlo_string));
  return ParseAndReturnUnverifiedModule(hlo_string);
}

StatusOr<std::unique_ptr<HloModule>>
MultiHostHloRunner::ReadModuleFromBinaryProtoFile(absl::string_view hlo_file) {
  HloProto proto;
  TF_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), std::string(hlo_file), &proto));
  return HloProtoToModule(proto.hlo_module());
}

StatusOr<std::unique_ptr<HloModule>>
MultiHostHloRunner::ReadModuleFromTextProtoFile(absl::string_view hlo_file) {
  HloProto proto;
  TF_RETURN_IF_ERROR(
      tsl::ReadTextProto(tsl::Env::Default(), std::string(hlo_file), &proto));
  return HloProtoToModule(proto.hlo_module());
}

StatusOr<MultiHostHloRunner::HloModuleAndArguments>
MultiHostHloRunner::ReadModuleFromSnapshotBinaryProtoFile(
    absl::string_view hlo_file) {
  HloSnapshot proto;
  HloModuleAndArguments hlo_module_and_arguments;
  TF_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), std::string(hlo_file), &proto));
  hlo_module_and_arguments.arguments.resize(proto.arguments_size());
  for (int i = 0; i < proto.arguments_size(); i++) {
    TF_ASSIGN_OR_RETURN(hlo_module_and_arguments.arguments[i],
                        Literal::CreateFromProto(proto.arguments()[i]));
  }
  TF_ASSIGN_OR_RETURN(hlo_module_and_arguments.hlo_module,
                      HloProtoToModule(proto.hlo().hlo_module()));
  return hlo_module_and_arguments;
}

StatusOr<std::unique_ptr<HloModule>> MultiHostHloRunner::ReadModuleFromString(
    absl::string_view hlo_text) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      HloTextToModule(hlo_text));
  return hlo_module;
}

StatusOr<std::unique_ptr<HloModule>> MultiHostHloRunner::ReadModuleFromProto(
    const HloModuleProto& proto) {
  return HloProtoToModule(proto);
}

StatusOr<MultiHostHloRunner::PerDeviceLiteralVecType>
MultiHostHloRunner::CompileAndRun(
    HloModule* hlo_module,
    const MultiHostHloRunner::PerDeviceLiteralVecType& arguments) const {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtLoadedExecutable> executable,
                      Compile(hlo_module));
  return Run(executable.get(), arguments, num_repeats_);
}

StatusOr<MultiHostHloRunner::PerDeviceLiteralVecType>
MultiHostHloRunner::CompileAndRun(
    HloModule* hlo_module, const LiteralVec& argument_literals,
    const PerDeviceIndexVecType& argument_indices) const {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtLoadedExecutable> executable,
                      Compile(hlo_module));
  return Run(executable.get(), argument_literals, argument_indices,
             num_repeats_);
}

void MultiHostHloRunner::SetDeviceAssignment(
    const DeviceAssignment& device_assignment) {
  compile_options_.executable_build_options.set_device_assignment(
      device_assignment);
}

const DeviceAssignment& MultiHostHloRunner::GetDeviceAssignment() const {
  return compile_options_.executable_build_options.device_assignment();
}

void MultiHostHloRunner::SetArgumentLayouts(const HloModule* hlo_module) {
  CHECK(compile_options_.executable_build_options.run_backend_only())
      << "No need to set argument layouts if it is not backend compilation.";
  std::vector<Shape> parameter_shapes;
  parameter_shapes.reserve(
      hlo_module->entry_computation_layout().parameter_count());
  for (const ShapeLayout& shape_layout :
       hlo_module->entry_computation_layout().parameter_layouts()) {
    parameter_shapes.push_back(shape_layout.shape());
  }
  compile_options_.argument_layouts = parameter_shapes;
}

namespace {

// Argument buffers are created on device at the first time an HLO module
// is executed. We reuse argument buffers in the following repeated
// executions whenever possible. We take the following strategy to
// maximally reuse on-device argument buffers which compiles and executes
// the HLO module differently depending on the number of parameters and the
// shape of the parameters of the HLO module. We have the following 3 cases.
// 1. The number of parameters is 1 and it has a shape of tuple of arrays.
// 2. The number of parameters is 1 or many and they are all arrays.
// 3. The rest: this should be rare and we don't expect this to happen with
// JAX.
//
// Case 1: the HLO module is compiled with
// CompileOptions::parameter_is_tupled_arguments = true
// and the HLO module is executed with
// ExecuteOptions::arguments_are_tupled = false.
// This enables PjRtClient::Execute to assemble the tupled arguments from
// a flat list of buffers.
// Additionally, we set ExecuteOptions::untuple_result = true if the module's
// output is a tuple. Thus we can use the aliased output buffer as input
// arguments and reuse the non-aliased argument buffers. In this mode, users may
// provide the argument literals as a list of tuples (for the convenience of
// future use cases) or a tuple literal (to support existing use cases).
//
// Case 2: the HLO module is compiled with
// CompileOptions::parameter_is_tupled_arguments = false
// and the HLO module is executed with
// ExecuteOptions::arguments_are_tupled = false.
// Same as above, we set ExecuteOptions::untuple_result = true if the module's
// output is a tuple. This allows us to reuse on-device buffers in the same way
// as case 1.
//
// Case 3: the HLO module is compiled with
// CompileOptions::parameter_is_tupled_arguments = false
// and the HLO module is executed with
// ExecuteOptions::arguments_are_tupled = false.
// Additionally, we set ExecuteOptions::untuple_result = false.
// We will create new on-device buffers for each repeated execution.

enum class ParameterType {
  kOneTupleOfArrays = 0,
  kOneListOfArrays = 1,
  kOther = 2
};

ParameterType GetParameterType(const HloModule& module) {
  int num_parameters = module.entry_computation()->num_parameters();
  if (num_parameters == 1) {
    const Shape& shape =
        module.entry_computation()->parameter_instruction(0)->shape();
    if (shape.IsTuple()) {
      bool is_tuple_of_arrays = absl::c_all_of(
          shape.tuple_shapes(),
          [](const Shape& subshape) { return subshape.IsArray(); });
      if (is_tuple_of_arrays) {
        return ParameterType::kOneTupleOfArrays;
      }
      return ParameterType::kOther;
    }
  }
  bool is_list_of_arrays =
      absl::c_all_of(module.entry_computation()->parameter_instructions(),
                     [](const HloInstruction* parameter) {
                       return parameter->shape().IsArray();
                     });
  return is_list_of_arrays ? ParameterType::kOneListOfArrays
                           : ParameterType::kOther;
}

}  // namespace

Status MultiHostHloRunner::PrepareHloModuleForCompilation(
    HloModule* hlo_module) const {
  if (is_spmd_partitioned_module_) {
    // If the module has already been partitioned by SPMD, add sharding
    // annotations (replicated) to module parameters and result.
    AddShardingAnnotationsToSpmdPartitionedModule(hlo_module);
  }

  if (flatten_while_loop_ || remove_infeed_outfeed_) {
    HloControlFlowFlattening hlo_control_flow_flattening(
        HloControlFlowFlattening::Options{
            /*while_execution_count=*/while_execution_count_,
            /*max_outer_loop_count=*/while_execution_count_,
            /*max_loop_count=*/while_execution_count_,
            /*remove_infeed_outfeed=*/remove_infeed_outfeed_,
            /*flatten_while_loop=*/flatten_while_loop_,
            /*remove_comm=*/false, /*remove_host_transfer=*/true});
    TF_RETURN_IF_ERROR(hlo_control_flow_flattening.Run(hlo_module).status());
  }
  return OkStatus();
}

CompileOptions MultiHostHloRunner::GetCompileOptions(
    const HloModule& hlo_module) const {
  ParameterType parameter_type = GetParameterType(hlo_module);
  CompileOptions compile_options = compile_options_;
  compile_options.parameter_is_tupled_arguments =
      (parameter_type == ParameterType::kOneTupleOfArrays);
  return compile_options;
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>> MultiHostHloRunner::Compile(
    HloModule* hlo_module) const {
  TF_RETURN_IF_ERROR(PrepareHloModuleForCompilation(hlo_module));
  CompileOptions compile_options = GetCompileOptions(*hlo_module);
  XlaComputation computation(hlo_module->ToProto());
  VLOG(1) << "MultiHostHloRunner: compilation started.";
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtLoadedExecutable> executable,
                      client_->Compile(computation, compile_options));
  VLOG(1) << "MultiHostHloRunner: compile succeeded.";
  return executable;
}

// Runs the executable and may repeat for multiple times.
// Since the input buffers may be donated by the PjrtClient, we re-create the
// input PjrtBuffers for each repetition.
StatusOr<MultiHostHloRunner::PerDeviceLiteralVecType> MultiHostHloRunner::Run(
    PjRtLoadedExecutable* executable, const PerDeviceLiteralVecType& arguments,
    int num_repeats) const {
  auto create_argument_buffers_on_device = [this, &executable, &arguments](
                                               bool flatten_tupled_arguments) {
    if (arguments.empty()) {
      return CreateArgumentsOnDevice(executable, flatten_tupled_arguments);
    }

    if (flatten_tupled_arguments && arguments.begin()->second.size() == 1 &&
        arguments.begin()->second.front().shape().IsTuple()) {
      PerDeviceLiteralVecType flattened_arguments;
      for (const auto& device_id_and_arguments : arguments) {
        Literal tupled_argument =
            device_id_and_arguments.second.front().Clone();
        LiteralVec flattened_argument = tupled_argument.DecomposeTuple();
        int device_id = device_id_and_arguments.first;
        flattened_arguments.insert({device_id, std::move(flattened_argument)});
      }
      return CopyArgumentsToDevice(executable->addressable_devices(),
                                   flattened_arguments);
    }
    // If the per-device argument is not a single tuple, we ignore the
    // flatten_tupled_arguments parameter and assume the provided arguments have
    // already been flattened.
    return CopyArgumentsToDevice(executable->addressable_devices(), arguments);
  };
  return RunInternal(executable, create_argument_buffers_on_device,
                     num_repeats);
}

// Runs the executable and may repeat for multiple times.
// Since the input buffers may be donated by the PjrtClient, we re-create the
// input PjrtBuffers for each repetition.
StatusOr<MultiHostHloRunner::PerDeviceLiteralVecType> MultiHostHloRunner::Run(
    PjRtLoadedExecutable* executable, const LiteralVec& argument_literals,
    const PerDeviceIndexVecType& argument_indices, int num_repeats) const {
  auto create_argument_buffers_on_device = [this, &executable,
                                            &argument_literals,
                                            &argument_indices](
                                               bool flatten_arguments) {
    CHECK_GE(argument_literals.size(), 1);
    bool arguments_can_be_flattened = absl::c_all_of(
        argument_literals,
        [](const Literal& literal) { return literal.shape().IsTuple(); });
    arguments_can_be_flattened &= absl::c_all_of(
        argument_indices, [](PerDeviceIndexVecType::const_reference
                                 device_id_and_argument_indices) {
          return device_id_and_argument_indices.second.size() == 1;
        });
    if (flatten_arguments && arguments_can_be_flattened) {
      int tuple_shape_size =
          argument_literals.front().shape().tuple_shapes_size();
      LiteralVec flattened_argument_literals;
      for (const Literal& tupled_argument : argument_literals) {
        LiteralVec flattened_arguments =
            tupled_argument.Clone().DecomposeTuple();
        for (Literal& flattened_argument : flattened_arguments) {
          flattened_argument_literals.push_back(std::move(flattened_argument));
        }
      }
      PerDeviceIndexVecType flattened_per_device_index_vec;
      for (const auto& device_id_and_argument_indices : argument_indices) {
        std::vector<int> flattened_argument_indices(tuple_shape_size);
        int tupled_argument_index =
            device_id_and_argument_indices.second.front();
        for (int i = 0; i < tuple_shape_size; i++) {
          flattened_argument_indices[i] =
              tupled_argument_index * tuple_shape_size + i;
        }
        int device_id = device_id_and_argument_indices.first;
        flattened_per_device_index_vec.insert(
            {device_id, std::move(flattened_argument_indices)});
      }
      return CopyArgumentsToDevice(executable->addressable_devices(),
                                   flattened_argument_literals,
                                   flattened_per_device_index_vec);
    }
    return CopyArgumentsToDevice(executable->addressable_devices(),
                                 argument_literals, argument_indices);
  };
  return RunInternal(executable, create_argument_buffers_on_device,
                     num_repeats);
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>> MultiHostHloRunner::Load(
    std::unique_ptr<PjRtExecutable> executable,
    const LoadOptions& load_options) const {
  return client_->Load(std::move(executable), load_options);
}

namespace {

std::vector<std::vector<PjRtBuffer*>> CreateArgumentPointersFromDeviceBuffers(
    absl::Span<const std::vector<std::unique_ptr<PjRtBuffer>>> device_buffers) {
  std::vector<std::vector<PjRtBuffer*>> argument_ptrs(device_buffers.size());
  for (int i = 0; i < device_buffers.size(); i++) {
    argument_ptrs[i].resize(device_buffers[i].size());
    for (int j = 0; j < device_buffers[i].size(); j++) {
      argument_ptrs[i][j] = device_buffers[i][j].get();
    }
  }
  return argument_ptrs;
}

std::vector<std::vector<PjRtBuffer*>> CreateArgumentPointersBasedOnAliasing(
    absl::Span<const std::vector<std::unique_ptr<PjRtBuffer>>> output_buffers,
    absl::Span<const std::vector<std::unique_ptr<PjRtBuffer>>> input_buffers,
    std::function<std::optional<int64_t>(int64_t)> get_output_buffer_index) {
  int num_arguments = input_buffers.front().size();
  std::vector<std::vector<PjRtBuffer*>> argument_ptrs(output_buffers.size());
  for (int i = 0; i < input_buffers.size(); i++) {
    argument_ptrs[i].resize(num_arguments);
    for (int argument_index = 0; argument_index < num_arguments;
         argument_index++) {
      std::optional<int> output_buffer_index =
          get_output_buffer_index(argument_index);
      if (!output_buffer_index.has_value()) {
        argument_ptrs[i][argument_index] =
            input_buffers[i][argument_index].get();
      } else {
        argument_ptrs[i][argument_index] =
            output_buffers[i][*output_buffer_index].get();
      }
    }
  }
  return argument_ptrs;
}

}  // namespace

StatusOr<MultiHostHloRunner::PerDeviceLiteralVecType>
MultiHostHloRunner::RunInternal(
    PjRtLoadedExecutable* executable,
    std::function<
        StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>(bool)>
        create_argument_buffers_on_device,
    int num_repeats) const {
  if (num_repeats <= 0) {
    num_repeats = num_repeats_;
  }
  ExecuteOptions execute_options;
  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> hlo_modules,
                      executable->GetHloModules());
  CHECK_EQ(hlo_modules.size(), 1);
  const HloModule& module = *(hlo_modules.front());
  ParameterType parameter_type = GetParameterType(module);
  bool flatten_arguments = parameter_type == ParameterType::kOneTupleOfArrays;
  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> device_buffers,
      create_argument_buffers_on_device(flatten_arguments));
  auto get_output_index_for_one_tuple_of_arrays =
      [&module](int64_t parameter_index) -> std::optional<int64_t> {
    const HloInputOutputAliasConfig& alias_config =
        module.input_output_alias_config();
    std::optional<ShapeIndex> output_index =
        alias_config.GetAliasedOutput(0, {parameter_index});
    if (!output_index.has_value()) {
      return std::nullopt;
    }
    // If the HLO module output is a tuple, it should have been untupled by
    // PjRt. Therefore, we return the tuple index of the buffer.
    if (module.entry_computation()->root_instruction()->shape().IsTuple()) {
      return std::optional<int64_t>(output_index->front());
    }
    CHECK(output_index->empty());
    return 0;
  };
  auto get_output_index_for_one_list_of_arrays =
      [&module](int64_t parameter_index) -> std::optional<int64_t> {
    const HloInputOutputAliasConfig& alias_config =
        module.input_output_alias_config();
    std::optional<ShapeIndex> output_index =
        alias_config.GetAliasedOutput(parameter_index, {});
    if (!output_index.has_value()) {
      return std::nullopt;
    }
    if (module.entry_computation()->root_instruction()->shape().IsTuple()) {
      return std::optional<int64_t>(output_index->front());
    }
    CHECK(output_index->empty());
    return 0;
  };

  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> output_buffers;
  std::vector<std::vector<PjRtBuffer*>> argument_ptrs =
      CreateArgumentPointersFromDeviceBuffers(device_buffers);
  bool default_untuple_result = execute_options.untuple_result;
  switch (parameter_type) {
    case ParameterType::kOneTupleOfArrays:
      execute_options.arguments_are_tupled = false;
      execute_options.untuple_result =
          module.entry_computation()->root_instruction()->shape().IsTuple();
      break;
    case ParameterType::kOneListOfArrays:
      execute_options.arguments_are_tupled = false;
      execute_options.untuple_result =
          module.entry_computation()->root_instruction()->shape().IsTuple();
      break;
    case ParameterType::kOther:
      execute_options.arguments_are_tupled = false;
      execute_options.untuple_result = false;
      break;
  }
  for (int repeat = 0; repeat < num_repeats; ++repeat) {
    VLOG(1) << "MultiHostHloRunner: ExecuteOnDevices started (repeat = "
            << repeat << ").";
    if (repeat == num_repeats - 1) {
      execute_options.untuple_result = default_untuple_result;
    }
    TF_ASSIGN_OR_RETURN(output_buffers,
                        executable->Execute(argument_ptrs, execute_options));
    VLOG(1) << "MultiHostHloRunner: ExecuteOnDevices succeeded (repeat = "
            << repeat << ")";
    if (repeat < num_repeats - 1) {
      switch (parameter_type) {
        case ParameterType::kOneTupleOfArrays:
          argument_ptrs = CreateArgumentPointersBasedOnAliasing(
              output_buffers, device_buffers,
              get_output_index_for_one_tuple_of_arrays);
          break;
        case ParameterType::kOneListOfArrays:
          argument_ptrs = CreateArgumentPointersBasedOnAliasing(
              output_buffers, device_buffers,
              get_output_index_for_one_list_of_arrays);
          break;
        case ParameterType::kOther:
          argument_ptrs =
              CreateArgumentPointersFromDeviceBuffers(device_buffers);
          break;
      }
    }
  }
  TF_ASSIGN_OR_RETURN(PerDeviceLiteralVecType results,
                      FetchAndLogOutput(output_buffers));
  return results;
}

StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
MultiHostHloRunner::CreateArgumentsOnDevice(
    const PjRtLoadedExecutable* executable, bool flatten_arguments) const {
  absl::Span<PjRtDevice* const> addressable_devices =
      executable->addressable_devices();
  size_t num_addressable_devices = addressable_devices.size();

  PerDeviceLiteralVecType per_device_argument_literals;
  absl::Span<const PjRtLoadedExecutable::LogicalDeviceIds>
      addressable_device_logical_ids =
          executable->addressable_device_logical_ids();
  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> hlo_modules,
                      executable->GetHloModules());
  VLOG(1) << "MultiHostHloRunner: local_executable count = "
          << hlo_modules.size();

  for (int i = 0; i < num_addressable_devices; ++i) {
    VLOG(3) << "Creating fake argument for device " << i;
    LiteralVec& argument_literals =
        per_device_argument_literals[addressable_devices[i]->id()];
    int executable_idx = hlo_modules.size() == 1
                             ? 0
                             : addressable_device_logical_ids[i].partition;
    HloModule* my_hlo_module = hlo_modules[executable_idx].get();
    if (flatten_arguments) {
      if (my_hlo_module->entry_computation()->num_parameters() != 1) {
        return InvalidArgument(
            "Flattening arguments requires the number of parameters to be 1. "
            "The actual number of parameters is %d",
            my_hlo_module->entry_computation()->num_parameters());
      }
      if (!my_hlo_module->entry_computation()
               ->parameter_instructions()
               .front()
               ->shape()
               .IsTuple()) {
        return InvalidArgument(
            "Flattening arguments requires the module parameter to be a single "
            "tuple. But the acutal parameter shape is %s",
            my_hlo_module->entry_computation()
                ->parameter_instructions()
                .front()
                ->shape()
                .ToString());
      }
    }
    if (module_argument_mode_ == ModuleArgumentMode::kUseDeviceIdAsInput) {
      const auto params =
          my_hlo_module->entry_computation()->parameter_instructions();
      if (flatten_arguments) {
        CHECK_EQ(params.size(), 1);
        CHECK(params.front()->shape().IsTuple());
        argument_literals.reserve(params.front()->shape().tuple_shapes_size());
      } else {
        argument_literals.reserve(params.size());
      }
      for (int j = 0; j < params.size(); ++j) {
        TF_ASSIGN_OR_RETURN(
            Literal argument_literal_j,
            MakeFakeLiteralWithSameValue(params[j]->shape(),
                                         addressable_devices[i]->id()));
        if (flatten_arguments) {
          std::vector<Literal> decomposed_argument_literals =
              argument_literal_j.DecomposeTuple();
          for (auto& literal : decomposed_argument_literals) {
            argument_literals.push_back(std::move(literal));
          }
        } else {
          argument_literals.push_back(std::move(argument_literal_j));
        }
      }
    } else {
      if (flatten_arguments) {
        TF_ASSIGN_OR_RETURN(LiteralVec tupled_argument_literals,
                            MakeFakeArguments(my_hlo_module));
        CHECK_EQ(tupled_argument_literals.size(), 1);
        CHECK(tupled_argument_literals.front().shape().IsTuple());
        argument_literals = tupled_argument_literals.front().DecomposeTuple();
      } else {
        TF_ASSIGN_OR_RETURN(argument_literals,
                            MakeFakeArguments(my_hlo_module));
      }
      if (module_argument_mode_ == ModuleArgumentMode::kUseSharedRandomInputs) {
        break;
      }
    }
  }

  if (module_argument_mode_ == ModuleArgumentMode::kUseSharedRandomInputs) {
    PerDeviceIndexVecType per_device_index_vec;
    std::vector<int> argument_indices;
    argument_indices.resize(
        per_device_argument_literals[addressable_devices[0]->id()].size());
    absl::c_iota(argument_indices, 0);
    for (int i = 0; i < num_addressable_devices; ++i) {
      per_device_index_vec[addressable_devices[i]->id()] = argument_indices;
    }
    return CopyArgumentsToDevice(
        addressable_devices,
        per_device_argument_literals[addressable_devices[0]->id()],
        per_device_index_vec);
  }
  return CopyArgumentsToDevice(addressable_devices,
                               per_device_argument_literals);
}

StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
MultiHostHloRunner::CopyArgumentsToDevice(
    absl::Span<PjRtDevice* const> addressable_devices,
    const PerDeviceLiteralVecType& arguments) const {
  size_t num_addressable_devices = addressable_devices.size();
  if (num_addressable_devices != arguments.size()) {
    return InvalidArgument(
        "The number of provided arguments does not match "
        "the number of logical devices.");
  }
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> argument_buffers;
  argument_buffers.resize(num_addressable_devices);

  for (int i = 0; i < num_addressable_devices; ++i) {
    PjRtDevice* curr_device = addressable_devices[i];
    int curr_device_id = curr_device->id();
    if (!arguments.contains(curr_device_id)) {
      return InvalidArgument(
          "The provided argument map does not contain arguments "
          "for device: %d",
          curr_device_id);
    }

    const std::vector<Literal>& curr_device_arguments =
        arguments.at(curr_device_id);

    argument_buffers[i].reserve(curr_device_arguments.size());
    for (const Literal& literal : curr_device_arguments) {
      if (log_output_) {
        LOG(INFO) << "device_id=" << curr_device_id
                  << ", input = " << literal.ToString();
      }
      TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> argument_buffer,
                          client_->BufferFromHostLiteral(literal, curr_device));
      argument_buffers[i].push_back(std::move(argument_buffer));
    }
  }
  for (const auto& device_argument_buffers : argument_buffers) {
    for (const auto& device_buffer : device_argument_buffers) {
      TF_RETURN_IF_ERROR(device_buffer->BlockHostUntilReady());
    }
  }
  return argument_buffers;
}

StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
MultiHostHloRunner::CopyArgumentsToDevice(
    absl::Span<PjRtDevice* const> addressable_devices,
    const LiteralVec& argument_literals,
    const PerDeviceIndexVecType& argument_indices) const {
  size_t num_addressable_devices = addressable_devices.size();
  if (num_addressable_devices != argument_indices.size()) {
    return InvalidArgument(
        "The number of provided arguments does not match "
        "the number of logical devices.");
  }
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> argument_buffers;
  argument_buffers.resize(num_addressable_devices);

  for (int i = 0; i < num_addressable_devices; ++i) {
    PjRtDevice* curr_device = addressable_devices[i];
    int curr_device_id = curr_device->id();
    if (!argument_indices.contains(curr_device_id)) {
      return InvalidArgument(
          "The provided argument map does not contain arguments "
          "for device: %d",
          curr_device_id);
    }

    const std::vector<int> curr_device_arguments_indices =
        argument_indices.at(curr_device_id);

    argument_buffers[i].reserve(curr_device_arguments_indices.size());
    for (int index : curr_device_arguments_indices) {
      const Literal& literal = argument_literals[index];
      if (log_output_) {
        LOG(INFO) << "device_id=" << curr_device_id
                  << ", input = " << literal.ToString();
      }
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<PjRtBuffer> argument_buffer,
          client_->BufferFromHostLiteral(literal, addressable_devices[i]));
      argument_buffers[i].push_back(std::move(argument_buffer));
    }
  }
  for (const auto& device_argument_buffers : argument_buffers) {
    for (const auto& device_buffer : device_argument_buffers) {
      TF_RETURN_IF_ERROR(device_buffer->BlockHostUntilReady());
    }
  }
  return argument_buffers;
}

StatusOr<MultiHostHloRunner::PerDeviceLiteralVecType>
MultiHostHloRunner::FetchAndLogOutput(
    const std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>& output_buffers)
    const {
  CHECK(!output_buffers.empty());
  absl::Mutex mu;
  Status status;
  size_t num_pending_transfers = 0;
  bool device_0_is_local = false;
  for (PjRtDevice* device : local_devices()) {
    if (device->id() == 0) {
      device_0_is_local = true;
    }
  }

  if (module_output_mode_ == ModuleOutputMode::kReturnDevice0Outputs &&
      device_0_is_local) {
    num_pending_transfers = output_buffers[0].size();
  } else if (module_output_mode_ == ModuleOutputMode::kReturnOutputs) {
    for (const auto& bs : output_buffers) {
      num_pending_transfers += bs.size();
    }
  }

  PerDeviceLiteralVecType outputs;
  for (int i = 0; i < output_buffers.size(); ++i) {
    if (output_buffers[i].empty()) {
      continue;
    }
    const int device_id = output_buffers[i][0]->device()->id();
    std::vector<Literal>& output_slice = outputs[device_id];
    if (module_output_mode_ == ModuleOutputMode::kReturnOutputs ||
        (module_output_mode_ == ModuleOutputMode::kReturnDevice0Outputs &&
         device_id == 0)) {
      output_slice.reserve(output_buffers[i].size());
      for (const auto& buffer : output_buffers[i]) {
        TF_RET_CHECK(buffer->device() == output_buffers[i][0]->device())
            << "All outputs from a given vector of outputs should be for the "
               "same device";
        output_slice.emplace_back(
            ShapeUtil::DeviceShapeToHostShape(buffer->on_device_shape()));
        buffer->ToLiteral(&output_slice.back(), [&](Status s) {
          absl::MutexLock lock(&mu);
          --num_pending_transfers;
          status.Update(s);
        });
      }
    } else {
      for (const auto& buffer : output_buffers[i]) {
        TF_RET_CHECK(buffer->device() == output_buffers[i][0]->device())
            << "All outputs from a given vector of outputs should be for the "
               "same device";
        TF_RETURN_IF_ERROR(buffer->BlockHostUntilReady());
      }
    }
  }
  if (module_output_mode_ == ModuleOutputMode::kReturnOutputs ||
      (module_output_mode_ == ModuleOutputMode::kReturnDevice0Outputs &&
       device_0_is_local)) {
    auto cond = [&]() { return !status.ok() || num_pending_transfers == 0; };
    absl::MutexLock lock(&mu);
    mu.Await(absl::Condition(&cond));
    TF_RETURN_IF_ERROR(status);
    if (log_output_) {
      for (const PjRtDevice* device : local_devices()) {
        int device_id = device->id();
        if (module_output_mode_ == ModuleOutputMode::kReturnDevice0Outputs &&
            device_id != 0) {
          continue;
        }
        LOG(INFO) << "Outputs for device_id: " << device_id;
        const std::vector<Literal>& output_slice = outputs[device_id];
        for (int i = 0; i < output_slice.size(); ++i) {
          LOG(INFO) << "output[" << i << "]: " << output_slice[i].ToString();
        }
      }
    }
  }
  return outputs;
}

StatusOr<MultiHostHloRunner::LogicalIdToDeviceIdMap>
MultiHostHloRunner::CreateLogicalIdToDeviceIdMap(
    const DeviceAssignment& device_assignment,
    absl::Span<const int> device_ids) {
  LogicalIdToDeviceIdMap id_map(device_assignment.replica_count(),
                                device_assignment.computation_count());
  for (int device_id : device_ids) {
    auto logical_id =
        device_assignment.LogicalIdForDevice(GlobalDeviceId(device_id));
    if (!logical_id.ok()) {
      continue;
    }
    int replica_id = logical_id->replica_id;
    int partition_id = logical_id->computation_id;
    id_map(replica_id, partition_id) = device_id;
  }
  return id_map;
}

StatusOr<MultiHostHloRunner::LogicalIdToDeviceIdMap>
MultiHostHloRunner::CreateLogicalIdToDeviceIdMap() {
  const DeviceAssignment& device_assignment =
      compile_options_.executable_build_options.device_assignment();
  LogicalIdToDeviceIdMap id_map(device_assignment.replica_count(),
                                device_assignment.computation_count());
  std::vector<int> device_ids;
  device_ids.reserve(devices().size());
  absl::c_for_each(devices(), [&device_ids](const PjRtDevice* device) {
    device_ids.push_back(device->id());
  });
  return CreateLogicalIdToDeviceIdMap(device_assignment, device_ids);
}

}  // namespace xla
