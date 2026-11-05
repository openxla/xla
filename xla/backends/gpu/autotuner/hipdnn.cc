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

#include "xla/backends/gpu/autotuner/hipdnn.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/engine_options.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

using HipdnnBackendConfig = stream_executor::dnn::AlgorithmProto;

namespace {

// Rebuilds the custom call with a workspace large enough for the chosen plan.
// Mirrors the MIOpen backend: the result tuple keeps its arity, only the
// trailing scratch shape grows, and the original users see an equivalent tuple
// whose scratch element is empty.
absl::Status ApplyConfigAndUpdateWorkspaceInOutputTuple(
    HloInstruction& instr, const HipdnnBackendConfig& config) {
  HloComputation* computation = instr.parent();
  absl::InlinedVector<Shape, 2> new_call_element_shapes;
  for (int i = 0; i < instr.shape().tuple_shapes().size() - 1; ++i) {
    new_call_element_shapes.emplace_back(instr.shape().tuple_shapes(i));
  }
  const int64_t workspace_size = config.workspace_size().value();
  new_call_element_shapes.emplace_back(
      ShapeUtil::MakeShape(U8, {workspace_size}));
  HloInstruction* new_call =
      computation->AddInstruction(instr.CloneWithNewOperands(
          ShapeUtil::MakeTupleShape(new_call_element_shapes),
          instr.operands()));
  new_call->SetAndSanitizeName(instr.name());

  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                      instr.backend_config<GpuBackendConfig>());
  *gpu_backend_config.mutable_cudnn_conv_backend_config()
       ->mutable_algorithm() = config;
  TF_RETURN_IF_ERROR(new_call->set_backend_config(gpu_backend_config));

  absl::InlinedVector<HloInstruction*, 2> new_tuple_elements;
  for (int i = 0; i < new_call->shape().tuple_shapes().size() - 1; ++i) {
    new_tuple_elements.emplace_back(
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_call->shape().tuple_shapes(i), new_call, i)));
  }
  new_tuple_elements.emplace_back(computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<uint8_t>({}))));

  HloInstruction* new_tuple = computation->AddInstruction(
      HloInstruction::CreateTuple(new_tuple_elements));
  return computation->ReplaceInstruction(&instr, new_tuple);
}

}  // namespace

std::string HipdnnBackend::version() const {
  // hipDNN's own version is not reachable from here without linking the ROCm
  // stream_executor plugin, so the device's DNN version stands in. Both are
  // constant for a given ROCm install, which is what the autotune cache key
  // needs them for.
  return absl::StrCat(
      "hipdnn-",
      target_config().device_description.dnn_version().ToString());
}

bool HipdnnBackend::IsSupported(const HloInstruction& instr) {
  return instr.opcode() == HloOpcode::kCustomCall &&
         instr.custom_call_target() == kCudnnConvForwardGraphCallTarget;
}

absl::StatusOr<std::unique_ptr<BackendConfig>> HipdnnBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  return absl::UnimplementedError(
      "hipDNN backend doesn't support getting a default config");
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
HipdnnBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::vector<std::unique_ptr<BackendConfig>>();
  }
  se::StreamExecutor* executor = stream_executor();
  if (executor == nullptr) {
    return absl::InvalidArgumentError("Null stream executor is not supported.");
  }

  const auto* custom_call = Cast<HloCustomCallInstruction>(&instr);
  TF_ASSIGN_OR_RETURN(GpuConvConfig gpu_conv_config,
                      GetGpuConvConfig(custom_call));
  TF_ASSIGN_OR_RETURN(
      se::dnn::DataType input_type,
      GetDNNDataTypeFromPrimitiveType(gpu_conv_config.input_type));
  TF_ASSIGN_OR_RETURN(
      se::dnn::DataType output_type,
      GetDNNDataTypeFromPrimitiveType(gpu_conv_config.output_type));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
                      executor->CreateStream());

  const bool allow_tf32 = absl::c_all_of(
      custom_call->precision_config().operand_precision(),
      [](int precision) { return precision <= PrecisionConfig::HIGH; });
  const se::EngineOptions engine_options{
      RequireDeterminism(custom_call->GetModule()->config()), allow_tf32,
      /*require_command_buffer=*/false};

  // Graph construction needs only the descriptors, so unlike the MIOpen path
  // there are no scratch buffers to allocate here.
  std::vector<std::unique_ptr<const se::dnn::GraphConvRunner>> runners;
  TF_RETURN_IF_ERROR(executor->AsDnn()->GetGraphConvolveRunners(
      se::dnn::ConvolutionKind::FORWARD_GRAPH, input_type, output_type,
      stream.get(), gpu_conv_config.input_descriptor,
      gpu_conv_config.filter_descriptor, gpu_conv_config.output_descriptor,
      gpu_conv_config.conv_desc,
      /*use_fallback=*/do_not_autotune_, engine_options, &runners,
      gpu_conv_config.serialized_graph));

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.reserve(runners.size());
  for (const auto& runner : runners) {
    TF_ASSIGN_OR_RETURN(se::dnn::AlgorithmDesc desc, runner->ToAlgorithmDesc());
    auto config = std::make_unique<BackendConfig>();
    *config->mutable_algorithm() = desc.ToProto();
    configs.push_back(std::move(config));
  }
  return configs;
}

absl::Status HipdnnBackend::ApplyConfig(HloInstruction& instr,
                                        const BackendConfig& config) {
  if (!config.has_algorithm()) {
    return absl::InvalidArgumentError(
        "Expected AlgorithmProto config for HipdnnBackend.");
  }
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "hipDNN backend doesn't support this instruction.");
  }
  const HipdnnBackendConfig& algorithm_config = config.algorithm();
  if (algorithm_config.has_workspace_size() &&
      algorithm_config.workspace_size().value() > 0) {
    return ApplyConfigAndUpdateWorkspaceInOutputTuple(instr, algorithm_config);
  }
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      instr.backend_config<GpuBackendConfig>());
  *gpu_config.mutable_cudnn_conv_backend_config()->mutable_algorithm() =
      algorithm_config;
  return instr.set_backend_config(std::move(gpu_config));
}

}  // namespace gpu
}  // namespace xla
