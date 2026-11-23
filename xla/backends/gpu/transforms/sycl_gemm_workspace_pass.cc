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

#include "xla/backends/gpu/transforms/sycl_gemm_workspace_pass.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/sycl/sycl_conv_utils.h"
#include "xla/stream_executor/sycl/sycl_matmul_utils.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

absl::StatusOr<size_t> ComputeMatmulWorkspaceSize(
    const HloInstruction* matmul_instr,
    const se::GpuComputeCapability& gpu_version) {
  ABSL_ASSIGN_OR_RETURN(GemmConfig gemm_config,
                        GemmConfig::For(matmul_instr, gpu_version));
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        matmul_instr->backend_config<GpuBackendConfig>());
  const GemmBackendConfig& config = gpu_config.gemm_backend_config();
  ABSL_ASSIGN_OR_RETURN(
      auto sycl_epilogue,
      stream_executor::sycl::sycl_gemm::AsSYCLEpilogue(config.epilogue()));
  ABSL_ASSIGN_OR_RETURN(
      auto prim_desc, stream_executor::sycl::CreateMatMulPrimDescFromGemmConfig(
                          gemm_config, sycl_epilogue));
  return prim_desc->scratchpad_desc().get_size();
}

// Builds the same oneDNN convolution config that the runtime runner builds
// from this custom call, so that both agree on the workspace size.
absl::StatusOr<size_t> ComputeConvWorkspaceSize(
    const HloCustomCallInstruction* conv_instr) {
  ABSL_ASSIGN_OR_RETURN(GpuConvConfig gpu_config, GetGpuConvConfig(conv_instr));
  ABSL_ASSIGN_OR_RETURN(se::dnn::DataType input_type,
                        GetDNNDataTypeFromPrimitiveType(gpu_config.input_type));
  ABSL_ASSIGN_OR_RETURN(
      se::dnn::DataType output_type,
      GetDNNDataTypeFromPrimitiveType(gpu_config.output_type));
  se::sycl::OneDnnConvConfig config{CudnnConvKindToProto(gpu_config.kind),
                                    input_type,
                                    output_type,
                                    gpu_config.input_descriptor,
                                    gpu_config.filter_descriptor,
                                    gpu_config.output_descriptor,
                                    gpu_config.conv_desc,
                                    gpu_config.conv_result_scale};
  if (gpu_config.fusion.has_value()) {
    config.fusion = se::sycl::OneDnnConvConfig::Fusion{
        gpu_config.fusion->mode, gpu_config.fusion->side_input_scale,
        gpu_config.fusion->leakyrelu_alpha};
  }
  return se::sycl::ComputeOneDnnConvWorkspaceSize(config);
}

// Matmul workspaces are never read after the call, so their shape is resized
// in place.
void ResizeMatmulWorkspace(HloInstruction* matmul_instr, int64_t new_bytes) {
  Shape* shape = matmul_instr->mutable_shape();
  *shape->mutable_tuple_shapes(shape->tuple_shapes().size() - 1) =
      ShapeUtil::MakeShape(S8, {new_bytes});
}

// A conv workspace can be read, e.g. by a root tuple from layout
// normalization. As the cuDNN autotuner does, clone the call with the new
// workspace and repackage it with the original shape, (conv_result, u8[0]).
absl::Status ResizeConvWorkspace(HloInstruction* conv_instr,
                                 int64_t new_bytes) {
  HloComputation* computation = conv_instr->parent();
  const Shape& old_shape = conv_instr->shape();
  const int64_t workspace_index = old_shape.tuple_shapes().size() - 1;
  Shape new_shape = old_shape;
  *new_shape.mutable_tuple_shapes(workspace_index) =
      ShapeUtil::MakeShape(U8, {new_bytes});
  HloInstruction* new_call = computation->AddInstruction(
      conv_instr->CloneWithNewOperands(new_shape, conv_instr->operands()));
  new_call->SetAndSanitizeName(conv_instr->name());

  std::vector<HloInstruction*> new_tuple_elements;
  for (int64_t i = 0; i < workspace_index; ++i) {
    new_tuple_elements.push_back(
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_shape.tuple_shapes(i), new_call, i)));
  }
  new_tuple_elements.push_back(computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<uint8_t>({}))));
  HloInstruction* new_tuple = computation->AddInstruction(
      HloInstruction::CreateTuple(new_tuple_elements));
  return computation->ReplaceInstruction(conv_instr, new_tuple);
}

}  // namespace

absl::StatusOr<bool> OneDnnWorkspacePass::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (IsCublasLtMatmul(*instr)) {
        const Shape& shape = instr->shape();
        if (!shape.IsTuple() || shape.tuple_shapes().size() != 2) {
          continue;
        }
        // oneDNN's matmul primitive does not support complex element types.
        if (primitive_util::IsComplexType(
                shape.tuple_shapes(0).element_type())) {
          continue;
        }
        absl::StatusOr<size_t> size =
            ComputeMatmulWorkspaceSize(instr, gpu_version_);
        if (!size.ok()) {
          VLOG(1) << "Failed to compute oneDNN workspace size for "
                  << instr->name() << ": " << size.status().message();
          continue;
        }
        ResizeMatmulWorkspace(instr, *size);
        changed = true;
      } else if (IsCustomCallToDnnConvolution(*instr)) {
        // Graph convolutions are not supported by the oneDNN runner.
        if (instr->custom_call_target() == kCudnnConvForwardGraphCallTarget) {
          continue;
        }
        const Shape& shape = instr->shape();
        if (!shape.IsTuple() || shape.tuple_shapes().size() != 2) {
          continue;
        }
        absl::StatusOr<size_t> size =
            ComputeConvWorkspaceSize(Cast<HloCustomCallInstruction>(instr));
        if (!size.ok()) {
          VLOG(1) << "Failed to compute oneDNN workspace size for "
                  << instr->name() << ": " << size.status().message();
          continue;
        }
        // ConvRewriter already creates the workspace as u8[0].
        if (*size == 0) {
          continue;
        }
        ABSL_RETURN_IF_ERROR(ResizeConvWorkspace(instr, *size));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
