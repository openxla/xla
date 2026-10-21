/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/algebraic_simplifier.h"

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace m = ::xla::match;

namespace {

// For a reshape from `shape_pre` to `shape_post`, returns for each dimension
// of `shape_post` the dimension of `shape_pre` it corresponds to, or -1 if the
// dimension is inserted, merged or split by the reshape.
std::vector<int64_t> MapReshapeDimensionsBackward(const Shape& shape_pre,
                                                  const Shape& shape_post) {
  std::vector<int64_t> result(shape_post.dimensions().size(), -1);
  if (std::optional<ShapeUtil::ShapeEqualityDescriptor> degenerate =
          ShapeUtil::InsertedOrDeleted1SizedDimensions(shape_pre, shape_post);
      degenerate.has_value()) {
    int64_t pre_dim = 0;
    for (int64_t post_dim = 0; post_dim < result.size(); ++post_dim) {
      if (absl::c_linear_search(degenerate->inserted_dimensions, post_dim)) {
        continue;
      }
      while (absl::c_linear_search(degenerate->deleted_dimensions, pre_dim)) {
        ++pre_dim;
      }
      result[post_dim] = pre_dim++;
    }
    return result;
  }
  for (const auto& [pre_dim, post_dim] :
       ShapeUtil::DimensionsUnmodifiedByReshape(shape_pre, shape_post)) {
    result[post_dim] = pre_dim;
  }
  return result;
}

struct DotWithDimensionMap {
  HloDotInstruction* dot;
  // For each dimension of the instruction the search started from, the
  // dimension of the dot output it corresponds to, or -1 if none.
  std::vector<int64_t> dot_dims;
};

// Walks from `instr` down through single-user transposes and reshapes to a
// single-user dot.
std::optional<DotWithDimensionMap> FindDotThroughTransposesAndReshapes(
    HloInstruction* instr) {
  std::vector<HloInstruction*> chain;
  while (HloPredicateIsOp<HloOpcode::kTranspose, HloOpcode::kReshape>(instr)) {
    if (instr->user_count() != 1) {
      return std::nullopt;
    }
    chain.push_back(instr);
    instr = instr->mutable_operand(0);
  }
  if (HloPredicateIsNotOp<HloOpcode::kDot>(instr) || instr->user_count() != 1) {
    return std::nullopt;
  }
  std::vector<int64_t> dot_dims(instr->shape().dimensions().size());
  absl::c_iota(dot_dims, 0);
  // Apply the chain from the dot upwards.
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    HloInstruction* op = *it;
    std::vector<int64_t> new_dot_dims(op->shape().dimensions().size(), -1);
    if (op->opcode() == HloOpcode::kTranspose) {
      for (int64_t i = 0; i < new_dot_dims.size(); ++i) {
        new_dot_dims[i] = dot_dims[op->dimensions(i)];
      }
    } else {
      std::vector<int64_t> reshape_map =
          MapReshapeDimensionsBackward(op->operand(0)->shape(), op->shape());
      for (int64_t i = 0; i < new_dot_dims.size(); ++i) {
        if (reshape_map[i] >= 0) {
          new_dot_dims[i] = dot_dims[reshape_map[i]];
        }
      }
    }
    dot_dims = std::move(new_dot_dims);
  }
  return DotWithDimensionMap{Cast<HloDotInstruction>(instr),
                             std::move(dot_dims)};
}

// Returns true if `instr` is a dequantization of a quantized tensor, i.e. a
// convert from a narrower element type, possibly followed by single-user
// reshapes and transposes. All instructions in the chain must have a single
// user, so that scaling the result does not affect other users.
bool IsDequantizedOperand(const HloInstruction* instr) {
  while (HloPredicateIsOp<HloOpcode::kReshape, HloOpcode::kTranspose>(instr) &&
         instr->user_count() == 1) {
    instr = instr->operand(0);
  }
  return instr->opcode() == HloOpcode::kConvert && instr->user_count() == 1 &&
         primitive_util::BitWidth(instr->operand(0)->shape().element_type()) <
             primitive_util::BitWidth(instr->shape().element_type());
}

}  // namespace

absl::StatusOr<bool>
GpuAlgebraicSimplifierVisitor::TryFoldDotOutputScaleIntoDequantizedOperand(
    HloInstruction* multiply) {
  if (!options_.enable_dot_output_scale_folding() ||
      options_.is_layout_sensitive()) {
    return false;
  }
  HloInstruction *value, *broadcast;
  if (!Match(multiply,
             m::MultiplyAnyOrder(m::Op(&value), m::Broadcast(&broadcast)))) {
    return false;
  }
  std::optional<DotWithDimensionMap> found =
      FindDotThroughTransposesAndReshapes(value);
  if (!found.has_value()) {
    return false;
  }
  HloDotInstruction* dot = found->dot;
  // Sparse dots carry extra metadata operands.
  if (dot->operand_count() != 2) {
    return false;
  }
  const PrecisionConfig::Algorithm algorithm =
      dot->precision_config().algorithm();
  if (algorithm != PrecisionConfig::ALG_UNSET &&
      algorithm != PrecisionConfig::ALG_DOT_BF16_BF16_F32) {
    return false;
  }

  // Map the broadcast dimensions of the scale to dot output dimensions.
  // Degenerate dimensions of the scale do not constrain the fold and are
  // dropped.
  HloInstruction* scale = broadcast->mutable_operand(0);
  std::vector<int64_t> scale_dot_dims;
  scale_dot_dims.reserve(broadcast->dimensions().size());
  for (int64_t i = 0; i < broadcast->dimensions().size(); ++i) {
    if (scale->shape().dimensions(i) == 1) {
      continue;
    }
    const int64_t dot_dim = found->dot_dims[broadcast->dimensions(i)];
    if (dot_dim < 0) {
      return false;
    }
    scale_dot_dims.push_back(dot_dim);
  }

  using DimsPerOperand = std::array<DotOperandDims, 2>;
  ABSL_ASSIGN_OR_RETURN(DimsPerOperand operand_dims,
                        DotOperandDims::FromDot(dot));
  const int64_t num_batch_dims = operand_dims[0].Rank(DotOperandDims::kBatch);
  const int64_t num_lhs_non_contracting_dims =
      operand_dims[0].Rank(DotOperandDims::kNonContracting);

  for (int64_t operand_index : {1, 0}) {
    HloInstruction* operand = dot->mutable_operand(operand_index);
    if (!IsDequantizedOperand(operand) ||
        operand->shape().element_type() != multiply->shape().element_type()) {
      continue;
    }
    const DotOperandDims& dims = operand_dims[operand_index];
    // The dot output is [batch..., lhs non-contracting..., rhs
    // non-contracting...]; every scale dimension must map to a batch or
    // non-contracting dimension of this operand.
    const int64_t non_contracting_start =
        num_batch_dims +
        (operand_index == 1 ? num_lhs_non_contracting_dims : 0);
    const int64_t non_contracting_end =
        non_contracting_start + dims.Rank(DotOperandDims::kNonContracting);
    std::vector<int64_t> operand_broadcast_dims;
    operand_broadcast_dims.reserve(scale_dot_dims.size());
    bool all_dims_mapped = true;
    for (int64_t dot_dim : scale_dot_dims) {
      if (dot_dim < num_batch_dims) {
        operand_broadcast_dims.push_back(
            dims.Indices(DotOperandDims::kBatch)[dot_dim]);
      } else if (dot_dim >= non_contracting_start &&
                 dot_dim < non_contracting_end) {
        operand_broadcast_dims.push_back(dims.Indices(
            DotOperandDims::kNonContracting)[dot_dim - non_contracting_start]);
      } else {
        all_dims_mapped = false;
        break;
      }
    }
    if (!all_dims_mapped || !absl::c_is_sorted(operand_broadcast_dims)) {
      continue;
    }

    if (ShapeUtil::HasDegenerateDimensions(scale->shape())) {
      scale = dot->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::DropDegenerateDimensions(scale->shape()), scale));
    }
    HloInstruction* operand_broadcast =
        dot->AddInstruction(HloInstruction::CreateBroadcast(
            operand->shape(), scale, operand_broadcast_dims));
    HloInstruction* scaled_operand = dot->AddInstruction(
        HloInstruction::CreateBinary(operand->shape(), HloOpcode::kMultiply,
                                     operand, operand_broadcast));
    ABSL_RETURN_IF_ERROR(
        dot->ReplaceOperandWith(operand_index, scaled_operand));
    ABSL_RETURN_IF_ERROR(ReplaceInstruction(multiply, value));
    return true;
  }
  return false;
}

absl::Status GpuAlgebraicSimplifierVisitor::HandleMultiply(
    HloInstruction* multiply) {
  ABSL_ASSIGN_OR_RETURN(bool replaced,
                        TryFoldDotOutputScaleIntoDequantizedOperand(multiply));
  if (replaced) {
    return absl::OkStatus();
  }
  return AlgebraicSimplifierVisitor::HandleMultiply(multiply);
}

absl::StatusOr<bool>
GpuAlgebraicSimplifierVisitor::TryToSinkBroadcastOperandsOfChainedAdds(
    HloInstruction* add) {
  if (!options_.enable_sink_broadcast()) {
    return false;
  }

  HloInstruction *conv, *constant_0, *broadcast_0, *add_0, *constant_1,
      *broadcast_1;
  if (!Match(add, m::AddAnyOrder(
                      m::AddAnyOrder(
                          &add_0, m::Convolution(&conv, m::Op(), m::Op()),
                          m::Broadcast(&broadcast_0, m::Constant(&constant_0))),
                      m::Broadcast(&broadcast_1, m::Constant(&constant_1))))) {
    return false;
  }

  // Skip when the broadcast shapes and dimensions don't match.
  if (!ShapeUtil::Equal(constant_0->shape(), constant_1->shape()) ||
      broadcast_0->dimensions() != broadcast_1->dimensions()) {
    return false;
  }

  HloInstruction* new_constant_add =
      add->AddInstruction(HloInstruction::CreateBinary(
          constant_0->shape(), HloOpcode::kAdd, constant_0, constant_1));
  HloInstruction* new_bcast =
      add->AddInstruction(HloInstruction::CreateBroadcast(
          broadcast_0->shape(), new_constant_add, broadcast_0->dimensions()));
  ABSL_RETURN_IF_ERROR(ReplaceWithNewInstruction(
      add, HloInstruction::CreateBinary(add->shape(), HloOpcode::kAdd,
                                        new_bcast, conv)));
  return true;
}

absl::Status GpuAlgebraicSimplifierVisitor::HandleAdd(HloInstruction* add) {
  ABSL_ASSIGN_OR_RETURN(bool replaced,
                        TryToSinkBroadcastOperandsOfChainedAdds(add));
  if (replaced) {
    return absl::OkStatus();
  }

  return AlgebraicSimplifierVisitor::HandleAdd(add);
}

bool GpuAlgebraicSimplifierVisitor::SupportedDotPrecisionConfig(
    const PrecisionConfig& config, bool has_contracting_dim) {
  if (!has_contracting_dim) {
    return config.algorithm() == PrecisionConfig::ALG_UNSET ||
           config.algorithm() == PrecisionConfig::ALG_DOT_F32_F32_F32;
  }
  return config.algorithm() == PrecisionConfig::ALG_UNSET ||
         config.algorithm() == PrecisionConfig::ALG_DOT_BF16_BF16_F32 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_TF32_TF32_F32 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_F32_F32_F32;
}

absl::StatusOr<HloInstruction*>
GpuAlgebraicSimplifierVisitor::MakeMultiplyForPrecisionAlgorithm(
    HloInstruction* dot, HloInstruction* lhs, HloInstruction* rhs) {
  return MakeMultiplyForDotPrecisionAlgorithm(
      lhs, rhs, dot->precision_config().algorithm());
}

absl::StatusOr<bool> GpuAlgebraicSimplifier::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      2, "GpuAlgebraicSimplifier::RunImpl(), before:\n" + module->ToString());
  bool changed = false;
  GpuAlgebraicSimplifierVisitor visitor(options_, compute_capability_, this);
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    if (visitor.Run(comp, options_, this)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(
      2, "GpuAlgebraicSimplifier::RunImpl(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla::gpu
