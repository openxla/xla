/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/transforms/cudnn_fused_conv_decomposer.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/dnn.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

class CustomCallVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleCustomCall(HloInstruction* instr) override {
    if (instr->custom_call_target() !=
        kCudnnConvBiasActivationForwardCallTarget) {
      return absl::OkStatus();
    }
    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        instr->backend_config<GpuBackendConfig>());
    CudnnConvBackendConfig& backend_config =
        *gpu_config.mutable_cudnn_conv_backend_config();
    if (backend_config.algorithm().is_cudnn_frontend()) {
      return absl::OkStatus();
    }

    CHECK(backend_config.conv_result_scale() == 1.0 &&
          backend_config.side_input_scale() == 0.0 &&
          instr->operands().size() == 3);

    HloInstruction* bias = instr->operands().back();
    const std::vector<HloInstruction*> users(instr->users().begin(),
                                             instr->users().end());
    HloInstruction* conv_result =
        instr->AddInstruction(HloInstruction::CreateGetTupleElement(instr, 0));
    HloInstruction* bcast_bias = instr->AddInstruction(
        HloInstruction::CreateBroadcast(conv_result->shape(), bias,
                                        {instr->convolution_dimension_numbers()
                                             .output_feature_dimension()}));
    HloInstruction* conv_bias =
        instr->AddInstruction(HloInstruction::CreateBinary(
            conv_result->shape(), HloOpcode::kAdd, conv_result, bcast_bias));

    HloInstruction* conv_bias_act = nullptr;

    switch (backend_config.activation_mode()) {
      case se::dnn::ActivationMode::kNone:
        conv_bias_act = conv_bias;
        break;
      case se::dnn::ActivationMode::kRelu:
        conv_bias_act = instr->AddInstruction(HloInstruction::CreateBinary(
            conv_bias->shape(), HloOpcode::kMaximum,
            BroadcastZeros(instr->parent(), conv_bias->shape()), conv_bias));
        break;
      case se::dnn::ActivationMode::kElu:
        conv_bias_act = instr->AddInstruction(HloInstruction::CreateTernary(
            conv_bias->shape(), HloOpcode::kSelect,
            instr->AddInstruction(HloInstruction::CreateCompare(
                ShapeUtil::ChangeElementType(conv_bias->shape(), PRED),
                conv_bias, BroadcastZeros(instr->parent(), conv_bias->shape()),
                ComparisonDirection::kGt)),
            conv_bias,
            instr->AddInstruction(HloInstruction::CreateUnary(
                conv_bias->shape(), HloOpcode::kExpm1, conv_bias))));
        break;
      case se::dnn::ActivationMode::kRelu6:
        conv_bias_act = instr->AddInstruction(HloInstruction::CreateTernary(
            conv_bias->shape(), HloOpcode::kClamp,
            BroadcastZeros(instr->parent(), conv_bias->shape()), conv_bias,
            instr->AddInstruction(HloInstruction::CreateBroadcast(
                conv_bias->shape(),
                instr->AddInstruction(
                    HloInstruction::CreateConstant(LiteralUtil::CreateR0(
                        conv_bias->shape().element_type(), 6))),
                {}))));
        break;
      case se::dnn::ActivationMode::kLeakyRelu:
        conv_bias_act = instr->AddInstruction(HloInstruction::CreateTernary(
            conv_bias->shape(), HloOpcode::kSelect,
            instr->AddInstruction(HloInstruction::CreateCompare(
                ShapeUtil::ChangeElementType(conv_bias->shape(), PRED),
                conv_bias, BroadcastZeros(instr->parent(), conv_bias->shape()),
                ComparisonDirection::kGt)),
            conv_bias,
            instr->AddInstruction(HloInstruction::CreateBinary(
                conv_bias->shape(), HloOpcode::kMultiply,
                instr->AddInstruction(HloInstruction::CreateBroadcast(
                    conv_bias->shape(),
                    instr->AddInstruction(
                        HloInstruction::CreateConstant(LiteralUtil::CreateR0(
                            conv_bias->shape().element_type(),
                            backend_config.leakyrelu_alpha()))),
                    {})),
                conv_bias))));
        break;
    }

    CHECK_NE(conv_bias_act, nullptr);

    HloInstruction* new_result =
        instr->AddInstruction(HloInstruction::CreateTuple(
            {conv_bias_act,
             instr->AddInstruction(
                 HloInstruction::CreateGetTupleElement(instr, 1))}));

    for (auto user : users) {
      TF_RETURN_IF_ERROR(instr->ReplaceUseWith(user, new_result));
    }

    backend_config.set_activation_mode(se::dnn::ActivationMode::kNone);
    absl::InlinedVector<HloInstruction*, 3> new_operands(
        instr->operands().begin(), instr->operands().end());
    new_operands.pop_back();

    HloInstruction* new_conv = instr->AddInstruction(
        instr->CloneWithNewOperands(instr->shape(), new_operands));
    new_conv->set_custom_call_target(kCudnnConvForwardCallTarget);
    // Preserve old name to make it obvious that we decomposed fused conv
    new_conv->SetAndSanitizeName(instr->name());
    TF_RETURN_IF_ERROR(ReplaceInstruction(instr, new_conv));
    ;
    return absl::OkStatus();
  }
};

}  // namespace

absl::StatusOr<bool> CudnnFusedConvDecomposer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_SCOPED_LOGGING_TIMER("cuDNN fused conv decomposer");
  return CustomCallVisitor().RunOnModule(module, execution_threads);
}

}  // namespace gpu
}  // namespace xla
