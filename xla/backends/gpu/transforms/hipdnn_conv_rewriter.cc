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

#include "xla/backends/gpu/transforms/hipdnn_conv_rewriter.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

// Element types hipDNN resolves a plain convolution graph for. Narrower types
// reach the graph API on cuDNN only as part of an fp8 scaling pattern, which
// hipDNN has no engine for.
bool IsSupportedGraphConvType(PrimitiveType type) {
  return type == F16 || type == F32 || type == BF16;
}

bool IsPromotableConv(const HloInstruction* conv) {
  if (conv->opcode() != HloOpcode::kCustomCall ||
      conv->custom_call_target() != kCudnnConvForwardCallTarget) {
    return false;
  }
  // A convolution that already carries a bias or side input, or an activation,
  // belongs to one of the fused patterns hipDNN cannot serve.
  if (conv->operand_count() != 2) {
    return false;
  }
  // A reversed window is a true convolution rather than a cross-correlation.
  // The graph API can describe one, but hipDNN ships no engine for it, and
  // promoting it here would strand it: MIOpen does not serve the graph target,
  // so there would be nothing left to fall back to.
  for (const WindowDimension& dim : conv->window().dimensions()) {
    if (dim.window_reversal()) {
      return false;
    }
  }
  const Shape& input_shape = conv->operand(0)->shape();
  const Shape& result_shape =
      ShapeUtil::GetTupleElementShape(conv->shape(), 0);
  return IsSupportedGraphConvType(input_shape.element_type()) &&
         input_shape.element_type() == result_shape.element_type();
}

}  // namespace

absl::StatusOr<bool> HipdnnConvRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (!IsPromotableConv(instr)) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                          instr->backend_config<GpuBackendConfig>());
      CudnnConvBackendConfig& config =
          *gpu_config.mutable_cudnn_conv_backend_config();
      if (config.activation_mode() != se::dnn::kNone ||
          config.conv_result_scale() != 1) {
        continue;
      }

      // The single-node form of the serialization CudnnFusedConvRewriter emits
      // for fused graphs: "<uid>:[<output type>]conv();". The UID only has to
      // be consistent within the string, so the instruction's own id is used,
      // matching what GraphString does for the fused case.
      const PrimitiveType result_type =
          ShapeUtil::GetTupleElementShape(instr->shape(), 0).element_type();
      config.set_serialized_graph(absl::StrCat(
          instr->unique_id(), ":[",
          primitive_util::LowercasePrimitiveTypeName(result_type), "]conv();"));

      TF_RETURN_IF_ERROR(instr->set_backend_config(gpu_config));
      instr->set_custom_call_target(kCudnnConvForwardGraphCallTarget);
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
