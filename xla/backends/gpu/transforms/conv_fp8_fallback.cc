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

#include "xla/backends/gpu/transforms/conv_fp8_fallback.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/engine_options.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

namespace {

bool IsFp8ConvCustomCall(const HloCustomCallInstruction* instr) {
  return absl::c_any_of(instr->operands(), [](const HloInstruction* op) {
    return primitive_util::IsF8Type(op->shape().element_type());
  });
}

absl::StatusOr<se::dnn::ConvolutionKind> GetBf16FallbackConvolutionKind(
    se::dnn::ConvolutionKind conv_kind) {
  switch (conv_kind) {
    case se::dnn::ConvolutionKind::FORWARD:
    case se::dnn::ConvolutionKind::BACKWARD_DATA:
    case se::dnn::ConvolutionKind::BACKWARD_FILTER:
    case se::dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION:
      return conv_kind;
    case se::dnn::ConvolutionKind::FORWARD_GRAPH:
      return se::dnn::ConvolutionKind::FORWARD;
    default:
      return absl::InvalidArgumentError(
          "BF16 fallback is unsupported for convolution kind.");
  }
}

absl::StatusOr<absl::string_view> GetBf16FallbackCustomCallTarget(
    const HloCustomCallInstruction& instr) {
  TF_ASSIGN_OR_RETURN(CudnnConvKind conv_kind, GetCudnnConvKind(&instr));
  switch (conv_kind) {
    case CudnnConvKind::kForward:
    case CudnnConvKind::kForwardGraph:
      return kCudnnConvForwardCallTarget;
    case CudnnConvKind::kBackwardInput:
      return kCudnnConvBackwardInputCallTarget;
    case CudnnConvKind::kBackwardFilter:
      return kCudnnConvBackwardFilterCallTarget;
    case CudnnConvKind::kForwardActivation:
      return kCudnnConvBiasActivationForwardCallTarget;
  }
  return absl::InvalidArgumentError(
      "BF16 fallback target mapping is unavailable for convolution kind.");
}

// Probes cuDNN for available execution plans for the given convolution config.
absl::StatusOr<bool> HasCudnnPlans(
    se::dnn::DnnSupport* dnn, se::dnn::ConvolutionKind conv_kind,
    se::dnn::DataType input_type, se::dnn::DataType output_type,
    se::Stream* stream, const GpuConvConfig& gpu_conv_config,
    const se::EngineOptions& engine_options) {
  std::vector<std::unique_ptr<const se::dnn::ConvRunner>> conv_runners;
  std::vector<std::unique_ptr<const se::dnn::FusedConvRunner>>
      fused_conv_runners;
  std::vector<std::unique_ptr<const se::dnn::GraphConvRunner>>
      graph_conv_runners;
  switch (conv_kind) {
    case se::dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION: {
      if (!gpu_conv_config.fusion) {
        return absl::InvalidArgumentError(
            "GpuConvConfig had fusion ConvolutionKind but no FusionConfig.");
      }
      TF_RETURN_IF_ERROR(dnn->GetFusedConvolveRunners(
          se::dnn::ConvolutionKind::FORWARD, input_type,
          BiasTypeForInputType(input_type), output_type,
          gpu_conv_config.conv_result_scale,
          gpu_conv_config.fusion->side_input_scale,
          gpu_conv_config.fusion->leakyrelu_alpha, stream,
          gpu_conv_config.input_descriptor, gpu_conv_config.filter_descriptor,
          gpu_conv_config.bias_descriptor, gpu_conv_config.output_descriptor,
          gpu_conv_config.conv_desc, /*use_fallback=*/false,
          gpu_conv_config.fusion->mode, engine_options,
          &fused_conv_runners));
      if (fused_conv_runners.empty()) {
        TF_RETURN_IF_ERROR(dnn->GetFusedConvolveRunners(
            se::dnn::ConvolutionKind::FORWARD, input_type,
            BiasTypeForInputType(input_type), output_type,
            gpu_conv_config.conv_result_scale,
            gpu_conv_config.fusion->side_input_scale,
            gpu_conv_config.fusion->leakyrelu_alpha, stream,
            gpu_conv_config.input_descriptor,
            gpu_conv_config.filter_descriptor,
            gpu_conv_config.bias_descriptor,
            gpu_conv_config.output_descriptor, gpu_conv_config.conv_desc,
            /*use_fallback=*/true, gpu_conv_config.fusion->mode,
            engine_options, &fused_conv_runners));
      }
      return !fused_conv_runners.empty();
    }
    case se::dnn::ConvolutionKind::FORWARD_GRAPH: {
      TF_RETURN_IF_ERROR(dnn->GetGraphConvolveRunners(
          conv_kind, input_type, output_type, stream,
          gpu_conv_config.input_descriptor, gpu_conv_config.filter_descriptor,
          gpu_conv_config.output_descriptor, gpu_conv_config.conv_desc,
          /*use_fallback=*/false, engine_options, &graph_conv_runners,
          gpu_conv_config.serialized_graph));
      if (graph_conv_runners.empty()) {
        TF_RETURN_IF_ERROR(dnn->GetGraphConvolveRunners(
            conv_kind, input_type, output_type, stream,
            gpu_conv_config.input_descriptor, gpu_conv_config.filter_descriptor,
            gpu_conv_config.output_descriptor, gpu_conv_config.conv_desc,
            /*use_fallback=*/true, engine_options, &graph_conv_runners,
            gpu_conv_config.serialized_graph));
      }
      return !graph_conv_runners.empty();
    }
    case se::dnn::ConvolutionKind::FORWARD:
    case se::dnn::ConvolutionKind::BACKWARD_DATA:
    case se::dnn::ConvolutionKind::BACKWARD_FILTER: {
      TF_RETURN_IF_ERROR(dnn->GetConvolveRunners(
          conv_kind, input_type, output_type, stream,
          gpu_conv_config.input_descriptor,
          /*input_data=*/se::DeviceAddressBase(nullptr),
          gpu_conv_config.filter_descriptor,
          /*filter_data=*/se::DeviceAddressBase(nullptr),
          gpu_conv_config.output_descriptor,
          /*output_data=*/se::DeviceAddressBase(nullptr),
          gpu_conv_config.conv_desc, /*use_fallback=*/false,
          /*scratch_allocator=*/nullptr, engine_options, &conv_runners));
      if (conv_runners.empty()) {
        TF_RETURN_IF_ERROR(dnn->GetConvolveRunners(
            conv_kind, input_type, output_type, stream,
            gpu_conv_config.input_descriptor,
            /*input_data=*/se::DeviceAddressBase(nullptr),
            gpu_conv_config.filter_descriptor,
            /*filter_data=*/se::DeviceAddressBase(nullptr),
            gpu_conv_config.output_descriptor,
            /*output_data=*/se::DeviceAddressBase(nullptr),
            gpu_conv_config.conv_desc, /*use_fallback=*/true,
            /*scratch_allocator=*/nullptr, engine_options, &conv_runners));
      }
      return !conv_runners.empty();
    }
    default:
      return absl::InvalidArgumentError(
          "Unsupported convolution kind for FP8 fallback probing.");
  }
}

// Rewrites an FP8 conv custom call to use BF16 types instead.
absl::Status RewriteToBf16(HloCustomCallInstruction* instr) {
  if (!instr->shape().IsTuple() || instr->shape().tuple_shapes_size() < 2) {
    return absl::InvalidArgumentError(
        "Expected (result, workspace) tuple from conv custom call");
  }
  HloComputation* computation = instr->parent();
  TF_ASSIGN_OR_RETURN(absl::string_view bf16_target,
                      GetBf16FallbackCustomCallTarget(*instr));

  // 1. Convert FP8 operands to BF16.
  std::vector<HloInstruction*> new_operands;
  new_operands.reserve(instr->operand_count());
  for (HloInstruction* operand : instr->operands()) {
    if (primitive_util::IsF8Type(operand->shape().element_type())) {
      Shape bf16_shape = ShapeUtil::ChangeElementType(operand->shape(), BF16);
      HloInstruction* convert = computation->AddInstruction(
          HloInstruction::CreateConvert(bf16_shape, operand));
      new_operands.push_back(convert);
    } else {
      new_operands.push_back(operand);
    }
  }

  // ForwardGraph→Forward: strip extra graph operands (scales, etc.) that
  // the plain Forward runner does not use.  Keep only input and filter.
  bool is_graph_to_forward =
      instr->custom_call_target() == kCudnnConvForwardGraphCallTarget;
  if (is_graph_to_forward && new_operands.size() > 2) {
    new_operands.resize(2);
  }

  // 2. Build new output shape with BF16 result element(s).
  //    The last tuple element is the workspace (U8).
  std::vector<Shape> new_call_element_shapes;
  new_call_element_shapes.reserve(instr->shape().tuple_shapes().size());
  for (int i = 0; i < instr->shape().tuple_shapes().size() - 1; ++i) {
    Shape elem = instr->shape().tuple_shapes(i);
    if (primitive_util::IsF8Type(elem.element_type())) {
      elem = ShapeUtil::ChangeElementType(elem, BF16);
    }
    new_call_element_shapes.push_back(elem);
  }
  // Preserve workspace size from backend config.
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                      instr->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& cudnn_conv_config =
      gpu_backend_config.cudnn_conv_backend_config();
  int64_t workspace_size =
      cudnn_conv_config.algorithm().has_workspace_size()
          ? cudnn_conv_config.algorithm().workspace_size().value()
          : 0;
  new_call_element_shapes.push_back(
      ShapeUtil::MakeShape(U8, {workspace_size}));
  Shape new_call_shape = ShapeUtil::MakeTupleShape(new_call_element_shapes);

  // 3. Create the BF16 custom call (FORWARD_GRAPH → plain FORWARD).
  HloInstruction* new_call = computation->AddInstruction(
      instr->CloneWithNewOperands(new_call_shape, new_operands));
  new_call->SetAndSanitizeName(instr->name());
  auto* new_custom_call = Cast<HloCustomCallInstruction>(new_call);
  new_custom_call->set_custom_call_target(bf16_target);

  // 4. Clear graph state if the original was a ForwardGraph call.
  CudnnConvBackendConfig* mutable_cudnn_conv_config =
      gpu_backend_config.mutable_cudnn_conv_backend_config();
  if (is_graph_to_forward) {
    mutable_cudnn_conv_config->clear_serialized_graph();
  }
  TF_RETURN_IF_ERROR(new_call->set_backend_config(gpu_backend_config));

  // 5. Extract results and convert BF16 back to original FP8 types.
  std::vector<HloInstruction*> new_tuple_elements;
  new_tuple_elements.reserve(instr->shape().tuple_shapes().size());
  for (int i = 0; i < instr->shape().tuple_shapes().size() - 1; ++i) {
    HloInstruction* gte =
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_call->shape().tuple_shapes(i), new_call, i));
    Shape original_shape = instr->shape().tuple_shapes(i);
    if (gte->shape().element_type() != original_shape.element_type()) {
      gte = computation->AddInstruction(
          HloInstruction::CreateConvert(original_shape, gte));
    }
    new_tuple_elements.push_back(gte);
  }
  // Empty workspace placeholder.
  new_tuple_elements.push_back(computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<uint8_t>({}))));

  // 6. Replace the original instruction with the BF16 version.
  HloInstruction* new_tuple = computation->AddInstruction(
      HloInstruction::CreateTuple(new_tuple_elements));
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(instr, new_tuple));
  VLOG(1) << "FP8→BF16 fallback applied to conv: " << new_call->name();
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> ConvFp8Fallback::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (stream_exec_ == nullptr) {
    return false;
  }

  se::dnn::DnnSupport* dnn = stream_exec_->AsDnn();
  if (dnn == nullptr) {
    return false;
  }

  auto allocator =
      std::make_unique<se::StreamExecutorAddressAllocator>(stream_exec_);
  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      allocator->GetStream(stream_exec_->device_ordinal()));

  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (!IsCustomCallToDnnConvolution(*instr)) continue;
      auto* custom_call = Cast<HloCustomCallInstruction>(instr);
      if (!IsFp8ConvCustomCall(custom_call)) continue;

      TF_ASSIGN_OR_RETURN(GpuConvConfig gpu_conv_config,
                          GetGpuConvConfig(custom_call));
      se::dnn::ConvolutionKind conv_kind =
          CudnnConvKindToProto(gpu_conv_config.kind);
      TF_ASSIGN_OR_RETURN(
          se::dnn::DataType input_type,
          GetDNNDataTypeFromPrimitiveType(gpu_conv_config.input_type));
      TF_ASSIGN_OR_RETURN(
          se::dnn::DataType output_type,
          GetDNNDataTypeFromPrimitiveType(gpu_conv_config.output_type));

      bool allow_tf32 = absl::c_all_of(
          custom_call->precision_config().operand_precision(),
          [](int precision) { return precision <= PrecisionConfig::HIGH; });
      const se::EngineOptions engine_options{
          RequireDeterminism(module->config()), allow_tf32,
          /*require_command_buffer=*/false};

      TF_ASSIGN_OR_RETURN(
          bool has_fp8_plans,
          HasCudnnPlans(dnn, conv_kind, input_type, output_type, stream,
                        gpu_conv_config, engine_options));

      if (has_fp8_plans) continue;

      // No FP8 plans — check if BF16 plans exist before rewriting.
      TF_ASSIGN_OR_RETURN(se::dnn::ConvolutionKind bf16_conv_kind,
                          GetBf16FallbackConvolutionKind(conv_kind));
      TF_ASSIGN_OR_RETURN(se::dnn::DataType bf16_type,
                          GetDNNDataTypeFromPrimitiveType(BF16));
      TF_ASSIGN_OR_RETURN(
          bool has_bf16_plans,
          HasCudnnPlans(dnn, bf16_conv_kind, bf16_type, bf16_type, stream,
                        gpu_conv_config, engine_options));

      if (!has_bf16_plans) {
        LOG(WARNING) << "FP8 conv " << custom_call->name()
                     << " has no cuDNN plans for either FP8 or BF16.";
        continue;
      }

      LOG(WARNING) << "FP8 conv " << custom_call->name()
                   << " has no cuDNN FP8 plans; rewriting to BF16. "
                   << "Try different convolution dimensions/group counts "
                   << "to regain FP8.";
      TF_RETURN_IF_ERROR(RewriteToBf16(custom_call));
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
