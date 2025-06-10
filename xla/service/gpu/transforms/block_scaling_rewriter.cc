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

#include "xla/service/gpu/transforms/block_scaling_rewriter.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/transforms/block_scaling_matcher.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu {
namespace block_scaling {
namespace {

// Verify that the operation is supported by the cuDNN kernel.
bool IsCudnnSupported(const BlockScaledDotOps& ops) {
  if (!ops.IsSupported()) {
    return false;
  }

  // cuDNN kernel doesn't support the configuration where both LHS and RHS have
  // E5M2 input type.
  if (ops.lhs.input->operand(0)->shape().element_type() == F8E5M2 &&
      ops.rhs.input->operand(0)->shape().element_type() == F8E5M2) {
    return false;
  }

  // Scaling tensors have to map to the input parameters directly.
  // Support pre-padded scaling tensors (allow slicing).
  auto is_parameter_or_slice = [](const HloInstruction* inst) {
    if (inst->opcode() == HloOpcode::kSlice) {
      inst = inst->operand(0);
    }
    return inst->opcode() == HloOpcode::kParameter;
  };
  return is_parameter_or_slice(ops.lhs.scale->operand(0)) &&
         is_parameter_or_slice(ops.rhs.scale->operand(0));
}

// Prepare the scaling tensor, which has to be padded to the 128x4 tile and
// swizzled in a way required by the cuDNN kernel, which expects the layout to
// be compatible with `scale_vec::1X` TMEM layout.
// https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
//
// - `dot_dims` argument contains the dimension index in the tensor shape:
//   [batch, noncontracting, contracting], with batch dimension being optional;
absl::StatusOr<XlaOp> PrepareScalingOperand(XlaOp scale_op,
                                            std::vector<int64_t> dot_dims) {
  // Get input shape dimension numbers for noncontracting and contracting
  // dimensions, the batch dimension being optional.
  CHECK(dot_dims.size() == 2 || dot_dims.size() == 3);
  bool has_batch = dot_dims.size() == 3;
  int noncontracting_dim = dot_dims[has_batch ? 1 : 0];
  int contracting_dim = dot_dims[has_batch ? 2 : 1];

  // Get input shape sizes for noncontracting and contracting dimensions.
  XlaBuilder& builder = *scale_op.builder();
  TF_ASSIGN_OR_RETURN(Shape scale_shape, builder.GetShape(scale_op));
  int64_t batch_dim_size = has_batch ? scale_shape.dimensions(dot_dims[0]) : 1;
  int64_t noncontracting_dim_size = scale_shape.dimensions(noncontracting_dim);
  int64_t contracting_dim_size = scale_shape.dimensions(contracting_dim);

  // If the scaling tensor is not tileable, insert a padding op.
  if (noncontracting_dim_size % kScaleNoncontractingTileSize != 0 ||
      contracting_dim_size % kScaleContractingTileSize != 0) {
    // Calculate padding sizes.
    int64_t tileable_noncontracting =
        RoundUpTo(noncontracting_dim_size, kScaleNoncontractingTileSize);
    int64_t tileable_contracting =
        RoundUpTo(contracting_dim_size, kScaleContractingTileSize);
    int64_t pad_noncontracting =
        tileable_noncontracting - noncontracting_dim_size;
    int64_t pad_contracting = tileable_contracting - contracting_dim_size;

    // Build pad op.
    PaddingConfig padding_config =
        MakeNoPaddingConfig(/*rank=*/dot_dims.size());
    padding_config.mutable_dimensions(noncontracting_dim)
        ->set_edge_padding_high(pad_noncontracting);
    padding_config.mutable_dimensions(contracting_dim)
        ->set_edge_padding_high(pad_contracting);
    scale_op = Pad(scale_op, Zero(&builder, scale_shape.element_type()),
                   padding_config);

    // Update shape sizes.
    noncontracting_dim_size = tileable_noncontracting;
    contracting_dim_size = tileable_contracting;
  }

  // Generate the sequence of ops to implement swizzle:
  // 1. reshape [B,N*128,C*4] -> [B,N,4,32,C,4]
  // 2. transpose [B,N,4,32,C,4] -> [B,N,C,32,4,4]
  // 3. reshape [B,N,C,32,4,4] -> [B,N*128,C*4]
  // The last reshape has no semantic meaning, and is only needed to make the
  // tensor compatible with cuDNN frontend, which expects 2D/3D input.

  // The scale shape becomes 6D, where the noncontracting dimension is split
  // into three parts, and the contracting dimension is split into two parts.
  // Update the dimension numbers to the start index of respective spans.
  // Examples: [0,1,2] -> [0,1,4]; [0,2,1] -> [0,3,1].
  if (dot_dims.size() == 2) {
    dot_dims = {0, dot_dims[0] + 1, dot_dims[1] + 1};
  }
  dot_dims[0] += (dot_dims[1] < dot_dims[0]) * 2 + (dot_dims[2] < dot_dims[0]);
  dot_dims[1] += (dot_dims[2] < dot_dims[1]);
  dot_dims[2] += (dot_dims[1] < dot_dims[2]) * 2;

  // Build reshape op.
  // Example: [8,256,64] -> [8,2,4,32,16,4]
  std::vector<int64_t> reshape_sizes(6);
  reshape_sizes[dot_dims[0]] = batch_dim_size;
  reshape_sizes[dot_dims[1]] =
      noncontracting_dim_size / kScaleNoncontractingTileSize;
  reshape_sizes[dot_dims[1] + 1] = kSwizzleHorizontalSize;
  reshape_sizes[dot_dims[1] + 2] = kSwizzleVerticalSize;
  reshape_sizes[dot_dims[2]] = contracting_dim_size / kScaleContractingTileSize;
  reshape_sizes[dot_dims[2] + 1] = kScaleContractingTileSize;
  scale_op = Reshape(scale_op, reshape_sizes);

  // Build transpose op.
  // Example: [8,2,4,32,16,4] -> [8,2,16,32,4,4]
  std::vector<int64_t> transpose_dims = {dot_dims[0],     dot_dims[1],
                                         dot_dims[2],     dot_dims[1] + 2,
                                         dot_dims[1] + 1, dot_dims[2] + 1};
  scale_op = Transpose(scale_op, transpose_dims);

  // Reshape back to [batch, noncontracting, contracting].
  std::vector<int64_t> result_dims = {noncontracting_dim_size,
                                      contracting_dim_size};
  if (has_batch) {
    result_dims.insert(result_dims.begin(), batch_dim_size);
  }
  scale_op = Reshape(scale_op, result_dims);

  return scale_op;
}

// The generated transpose shape layout makes it a bitcast, but we need an
// actual transpose.
void FixTransposeLayout(HloComputation* computation) {
  for (HloInstruction* instruction : computation->instructions()) {
    if (instruction->opcode() == HloOpcode::kTranspose) {
      LayoutUtil::SetToDefaultLayout(instruction->mutable_shape());
    }
  }
}

// Make dot dimension numbers array (helper).
std::vector<int64_t> MakeDotDims(const BlockScaledDotOps& ops, bool is_lhs) {
  std::vector<int64_t> dot_dims;
  auto batch_dim = is_lhs ? ops.lhs_batch_dim() : ops.rhs_batch_dim();
  if (batch_dim.has_value()) {
    dot_dims.push_back(batch_dim.value());
  }
  dot_dims.push_back(is_lhs ? ops.lhs_noncontracting_dim()
                            : ops.rhs_noncontracting_dim());
  dot_dims.push_back(is_lhs ? ops.lhs_contracting_dim()
                            : ops.rhs_contracting_dim());
  return dot_dims;
}

// Expand builder into a new instruction that will replace the old one.
absl::StatusOr<std::unique_ptr<HloInstruction>> ExpandInstructionUsingBuilder(
    XlaBuilder& builder, HloInstruction* old_instruction) {
  TF_ASSIGN_OR_RETURN(XlaComputation xla_computation, builder.Build());
  TF_ASSIGN_OR_RETURN(
      HloComputation * computation,
      XlaComputationToHloComputation(xla_computation,
                                     old_instruction->parent()->parent()));
  return HloInstruction::CreateCall(old_instruction->shape(),
                                    old_instruction->operands(), computation);
}

// Slice the input parameter of the composite to the new shape.
absl::Status SliceInputParameter(HloComputation* computation, int param_idx,
                                 const Shape& old_shape, const Shape& new_shape,
                                 std::vector<int64_t> transpose_dims) {
  // Update the parameter to the new shape.
  HloInstruction* new_param = computation->ReplaceParameter(
      param_idx, HloInstruction::CreateParameter(
                     param_idx, new_shape,
                     computation->parameter_instruction(param_idx)->name()));

  // Set limit indices for the slice operation.
  int rank = old_shape.dimensions_size();
  std::vector<int64_t> limit_indices(rank);
  bool slice_needed = false, transpose_needed = false;
  for (int i = 0; i < rank; ++i) {
    limit_indices[i] = old_shape.dimensions(i);
    slice_needed |= limit_indices[i] != new_shape.dimensions(transpose_dims[i]);
    transpose_needed |= i != transpose_dims[i];
  }

  // Slice scale tensor, if needed.
  if (slice_needed) {
    std::vector<int64_t> start_indices(rank, 0);
    std::vector<int64_t> strides(rank, 1);
    HloInstruction* slice =
        computation->AddInstruction(HloInstruction::CreateSlice(
            old_shape, new_param, start_indices, limit_indices, strides));
    TF_RETURN_IF_ERROR(new_param->ReplaceAllUsesWithDifferentShape(slice));
  }

  // Transpose scale tensor, if needed.
  if (transpose_needed) {
    Shape transpose_shape = new_shape;
    for (int i = 0; i < transpose_dims.size(); ++i) {
      transpose_shape.set_dimensions(i,
                                     new_shape.dimensions(transpose_dims[i]));
    }
    HloInstruction* transpose =
        computation->AddInstruction(HloInstruction::CreateTranspose(
            transpose_shape, new_param, absl::MakeSpan(transpose_dims)));
    TF_RETURN_IF_ERROR(new_param->ReplaceAllUsesWithDifferentShape(transpose));
  }
  return absl::OkStatus();
}

// Create a computation for padding and swizzling the scale tensors.
absl::Status AddScalePaddingAndSwizzle(HloInstruction* hlo) {
  HloComputation* computation = hlo->called_computations()[0];
  auto ops = BlockScaledDotOps::Match(computation->root_instruction());
  CHECK(ops.has_value());

  // Build LHS/RHS scale ops.
  XlaBuilder builder(std::string(hlo->name()));

  int lhs_scale_param_idx = ops->lhs.GetScaleParameter()->parameter_number();
  const Shape& lhs_scale_shape = hlo->operand(lhs_scale_param_idx)->shape();
  TF_ASSIGN_OR_RETURN(XlaOp lhs_scale_op,
                      PrepareScalingOperand(
                          Parameter(&builder, 0, lhs_scale_shape, "lhs_scale"),
                          MakeDotDims(*ops, /*is_lhs=*/true)));

  int rhs_scale_param_idx = ops->rhs.GetScaleParameter()->parameter_number();
  const Shape& rhs_scale_shape = hlo->operand(rhs_scale_param_idx)->shape();
  TF_ASSIGN_OR_RETURN(XlaOp rhs_scale_op,
                      PrepareScalingOperand(
                          Parameter(&builder, 1, rhs_scale_shape, "rhs_scale"),
                          MakeDotDims(*ops, /*is_lhs=*/false)));

  // Slice the scale tensors that were padded.
  TF_ASSIGN_OR_RETURN(Shape lhs_scale_new_shape,
                      builder.GetShape(lhs_scale_op));
  if (lhs_scale_new_shape != lhs_scale_shape) {
    TF_RETURN_IF_ERROR(SliceInputParameter(computation, lhs_scale_param_idx,
                                           lhs_scale_shape, lhs_scale_new_shape,
                                           MakeDotDims(*ops, /*is_lhs=*/true)));
  }

  TF_ASSIGN_OR_RETURN(Shape rhs_scale_new_shape,
                      builder.GetShape(rhs_scale_op));
  if (rhs_scale_new_shape != rhs_scale_shape) {
    TF_RETURN_IF_ERROR(SliceInputParameter(
        computation, rhs_scale_param_idx, rhs_scale_shape, rhs_scale_new_shape,
        MakeDotDims(*ops, /*is_lhs=*/false)));
  }

  // Create scale swizzle computation.
  XlaOp tuple_op = Tuple(&builder, {lhs_scale_op, rhs_scale_op});
  TF_ASSIGN_OR_RETURN(Shape tuple_shape, builder.GetShape(tuple_op));
  TF_ASSIGN_OR_RETURN(XlaComputation xla_computation, builder.Build());
  TF_ASSIGN_OR_RETURN(
      HloComputation * swizzle_computation,
      XlaComputationToHloComputation(xla_computation, hlo->parent()->parent()));
  FixTransposeLayout(swizzle_computation);

  auto operands = hlo->mutable_operands();
  auto call_op = HloInstruction::CreateCall(
      tuple_shape,
      {operands[lhs_scale_param_idx], operands[rhs_scale_param_idx]},
      swizzle_computation);
  HloInstruction* swizzle_call =
      hlo->parent()->AddInstruction(std::move(call_op));

  // Create cuDNN fusion call.
  operands[lhs_scale_param_idx] = hlo->parent()->AddInstruction(
      HloInstruction::CreateGetTupleElement(swizzle_call, 0));
  operands[rhs_scale_param_idx] = hlo->parent()->AddInstruction(
      HloInstruction::CreateGetTupleElement(swizzle_call, 1));

  auto fusion = HloInstruction::CreateFusion(
      hlo->shape(), HloInstruction::FusionKind::kCustom, operands, computation);
  TF_ASSIGN_OR_RETURN(auto backend_config,
                      hlo->backend_config<GpuBackendConfig>());
  TF_RETURN_IF_ERROR(fusion->set_backend_config(backend_config));

  return hlo->parent()->ReplaceWithNewInstruction(hlo, std::move(fusion));
}

// Build composite computation for a custom call.
absl::StatusOr<HloInstruction*> TransformCustomCall(
    HloInstruction* custom_call) {
  XlaBuilder builder(std::string(custom_call->name()));

  // Build LHS/RHS parameters.
  TF_RET_CHECK(custom_call->operand_count() == 4);
  XlaOp lhs_input_op =
      Parameter(&builder, 0, custom_call->operand(0)->shape(), "lhs");
  XlaOp rhs_input_op =
      Parameter(&builder, 1, custom_call->operand(1)->shape(), "rhs");
  XlaOp lhs_scale_op =
      Parameter(&builder, 2, custom_call->operand(2)->shape(), "lhs_scale");
  XlaOp rhs_scale_op =
      Parameter(&builder, 3, custom_call->operand(3)->shape(), "rhs_scale");

  // Dequantize LHS/RHS.
  auto dequantize = [&](XlaOp input_op,
                        XlaOp scale_op) -> absl::StatusOr<XlaOp> {
    // Calculate block size.
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder.GetShape(input_op));
    TF_ASSIGN_OR_RETURN(Shape scale_shape, builder.GetShape(scale_op));
    int64_t block_size =
        input_shape.dimensions().back() / scale_shape.dimensions().back();

    // Convert input/scale to the same type.
    PrimitiveType result_type = custom_call->shape().element_type();
    input_op = ConvertElementType(input_op, result_type);
    scale_op = ConvertElementType(scale_op, result_type);

    // Reshape scale to the same shape as input.
    std::vector<int64_t> new_dims(scale_shape.dimensions().begin(),
                                  scale_shape.dimensions().end());
    new_dims.push_back(block_size);
    std::vector<int64_t> broadcast_dims(scale_shape.dimensions_size());
    absl::c_iota(broadcast_dims, 0);
    scale_op = BroadcastInDim(scale_op, new_dims, broadcast_dims);
    new_dims.pop_back();
    new_dims.back() *= block_size;
    scale_op = Reshape(scale_op, new_dims);

    return Mul(input_op, scale_op);
  };

  // Build dot dimension numbers.
  int rank = custom_call->operand(0)->shape().dimensions_size();
  TF_RET_CHECK(rank == 2 || rank == 3);
  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(rank - 1);
  dnums.add_rhs_contracting_dimensions(rank - 1);
  if (rank == 3) {
    dnums.add_lhs_batch_dimensions(0);
    dnums.add_rhs_batch_dimensions(0);
  }

  // Build dot operation.
  TF_ASSIGN_OR_RETURN(XlaOp lhs_op, dequantize(lhs_input_op, lhs_scale_op));
  TF_ASSIGN_OR_RETURN(XlaOp rhs_op, dequantize(rhs_input_op, rhs_scale_op));
  DotGeneral(lhs_op, rhs_op, dnums);

  // Replace custom call with composite call.
  TF_ASSIGN_OR_RETURN(auto composite_call,
                      ExpandInstructionUsingBuilder(builder, custom_call));
  HloInstruction* result = composite_call.get();
  TF_RETURN_IF_ERROR(custom_call->parent()->ReplaceWithNewInstruction(
      custom_call, std::move(composite_call)));

  result->set_is_composite(true);
  result->set_frontend_attribute("composite.version", "1");
  result->set_frontend_attribute("composite.name",
                                 std::string(kBlockScaledDotCompositeName));
  return result;
}

// Convert composite call to a fusion.
absl::Status ReplaceCompositeWithFusion(HloInstruction* hlo,
                                        absl::string_view fusion_kind) {
  auto fusion = HloInstruction::CreateFusion(
      hlo->shape(), HloInstruction::FusionKind::kCustom, hlo->operands(),
      hlo->to_apply());
  GpuBackendConfig backend_config;
  *backend_config.mutable_fusion_backend_config()->mutable_kind() =
      std::string(fusion_kind);
  TF_RETURN_IF_ERROR(fusion->set_backend_config(backend_config));

  return hlo->parent()->ReplaceWithNewInstruction(hlo, std::move(fusion));
}

}  // namespace
}  // namespace block_scaling

absl::StatusOr<bool> BlockScalingRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  namespace m = match;

  // Collect matching composites and custom calls in all computations.
  std::vector<HloInstruction*> matching_instructions;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (Match(instruction,
                m::Call().WithPredicate([](const HloInstruction* call) {
                  return call->is_composite() &&
                         call->get_frontend_attribute("composite.name") ==
                             block_scaling::kBlockScaledDotCompositeName;
                }))) {
        VLOG(2) << "Matched block scaled dot composite: "
                << instruction->name();
        matching_instructions.push_back(instruction);
      }
      if (Match(instruction,
                m::CustomCall({kBlockScaledDotCustomCallTarget}))) {
        VLOG(2) << "Matched block scaled dot custom call: "
                << instruction->name();
        matching_instructions.push_back(instruction);
      }
    }
  }

  // Transform to cuDNN fusion (if allowed), which could be autotuned.
  bool changed = false;
  for (HloInstruction* instruction : matching_instructions) {
    if (Match(instruction, m::CustomCall())) {
      VLOG(3) << "Replacing custom call with composite: "
              << instruction->name();
      TF_ASSIGN_OR_RETURN(instruction,
                          block_scaling::TransformCustomCall(instruction));
      changed = true;
    }
    auto ops = block_scaling::BlockScaledDotOps::Match(
        instruction->to_apply()->root_instruction());
    if (ops.has_value()) {
      absl::string_view fusion_kind;
      if (allow_cudnn_ && block_scaling::IsCudnnSupported(*ops)) {
        fusion_kind = kCuDnnFusionKind;
      } else {
        continue;
      }
      VLOG(3) << "Replacing block scaled dot with fusion: "
              << instruction->name();
      TF_RETURN_IF_ERROR(
          block_scaling::ReplaceCompositeWithFusion(instruction, fusion_kind));
      changed = true;
    }
  }
  return changed;
}

absl::StatusOr<bool> CudnnBlockScalingRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Collect matching cuDNN fusions containing block scaled dot operations.
  std::vector<HloInstruction*> matching_instructions;
  for (HloComputation* computation : module->computations()) {
    if (!computation->IsFusionComputation()) {
      continue;
    }
    for (HloInstruction* caller :
         computation->caller_instructions(HloOpcode::kFusion)) {
      if (caller->get_frontend_attribute("composite.name") ==
          block_scaling::kBlockScaledDotCompositeName) {
        auto backend_config = caller->backend_config<GpuBackendConfig>();
        if (backend_config.ok() &&
            backend_config->fusion_backend_config().kind() ==
                kCuDnnFusionKind) {
          VLOG(2) << "Matched block scaled dot cuDNN fusion: "
                  << caller->name();
          matching_instructions.push_back(caller);
        }
      }
    }
  }

  // Prepend a computation that pads and swizzles the scaling factors.
  for (HloInstruction* instruction : matching_instructions) {
    if (!IsCudnnSupported(
            instruction->called_computations()[0]->root_instruction())) {
      return absl::InvalidArgumentError(
          "Block scaled dot operation is not supported by cuDNN.");
    }
    VLOG(3) << "Adding scale padding and swizzling: " << instruction->name();
    TF_RETURN_IF_ERROR(block_scaling::AddScalePaddingAndSwizzle(instruction));
  }
  return !matching_instructions.empty();
}

/*static*/ bool CudnnBlockScalingRewriter::IsCudnnSupported(
    const HloInstruction* root) {
  auto ops = block_scaling::BlockScaledDotOps::Match(root);
  return ops.has_value() && block_scaling::IsCudnnSupported(*ops);
}

}  // namespace xla::gpu
