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

#include "xla/service/gpu/transforms/block_scaling_matcher.h"

#include <optional>

#include "xla/service/pattern_matcher.h"

namespace xla::gpu {
namespace block_scaling {
namespace {

namespace m = match;

// Verify that the dot op is compatible (assuming it is valid).
// There should be exactly one contracting dimension, one non-contracting
// dimension and at most one batch dimension.
bool IsCompatibleDot(const HloInstruction* dot) {
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  int batch_dims = dnums.lhs_batch_dimensions_size();
  int contracting_dims = dnums.lhs_contracting_dimensions_size();
  int lhs_noncontracting_dims = dot->operand(0)->shape().dimensions_size() -
                                batch_dims - contracting_dims;
  int rhs_noncontracting_dims = dot->operand(1)->shape().dimensions_size() -
                                batch_dims - contracting_dims;
  return batch_dims <= 1 && contracting_dims == 1 &&
         lhs_noncontracting_dims == 1 && rhs_noncontracting_dims == 1;
}

// Verify that the reshape op is compatible, i.e. reshapes back to the same
// number of dimensions as the broadcast input (convert->broadcast->reshape
// operations are matched at this point; the broadcast expands the contracting
// subchannel scales to the input shape).
bool IsCompatibleReshape(const HloInstruction* reshape) {
  const Shape& input_shape = reshape->operand(0)->operand(0)->shape();
  const Shape& output_shape = reshape->shape();
  if (output_shape.dimensions_size() != input_shape.dimensions_size()) {
    return false;
  }
  for (int i = 0; i < output_shape.dimensions_size(); ++i) {
    if (output_shape.dimensions(i) % input_shape.dimensions(i) != 0) {
      return false;
    }
  }
  return true;
}

}  // namespace

/*static*/ std::optional<BlockScaledDequantizeOps>
BlockScaledDequantizeOps::Match(const HloInstruction* instruction) {
  BlockScaledDequantizeOps ops;
  if (::xla::Match(instruction,
                   m::Multiply(&ops.result, m::Convert(&ops.input),
                               m::Reshape(&ops.reshape,
                                          m::Broadcast(&ops.broadcast,
                                                       m::Convert(&ops.scale)))
                                   .WithPredicate(IsCompatibleReshape)))) {
    return ops;
  }
  return std::nullopt;
}

int64_t BlockScaledDequantizeOps::GetBlockSize(int64_t dimension) const {
  // The matcher validates that the reshape op is compatible.
  return reshape->shape().dimensions(dimension) /
         reshape->operand(0)->operand(0)->shape().dimensions(dimension);
}

const HloInstruction* BlockScaledDequantizeOps::GetScaleParameter() const {
  // Find parameter corresponding to the scale tensor, if possible.
  const HloInstruction* current = scale->operand(0);
  while (current->operand_count() == 1) {
    current = current->operand(0);
  }
  return current->opcode() == HloOpcode::kParameter ? current : nullptr;
}

/*static*/ std::optional<BlockScaledDotOps> BlockScaledDotOps::Match(
    const HloInstruction* instruction) {
  if (::xla::Match(instruction, m::Dot().WithPredicate(IsCompatibleDot))) {
    if (auto lhs = BlockScaledDequantizeOps::Match(instruction->operand(0));
        lhs.has_value()) {
      if (auto rhs = BlockScaledDequantizeOps::Match(instruction->operand(1));
          rhs.has_value()) {
        return BlockScaledDotOps{lhs.value(), rhs.value(), instruction};
      }
    }
  }
  return std::nullopt;
}

int64_t BlockScaledDotOps::lhs_contracting_dim() const {
  return result->dot_dimension_numbers().lhs_contracting_dimensions(0);
}

int64_t BlockScaledDotOps::lhs_noncontracting_dim() const {
  const DotDimensionNumbers& dnums = result->dot_dimension_numbers();
  return GetNonContractingDims(lhs.result->shape().dimensions_size(),
                               dnums.lhs_contracting_dimensions(),
                               dnums.lhs_batch_dimensions())
      .front();
}

std::optional<int64_t> BlockScaledDotOps::lhs_batch_dim() const {
  auto batch = result->dot_dimension_numbers().lhs_batch_dimensions();
  return batch.empty() ? std::nullopt : std::make_optional(batch[0]);
}

int64_t BlockScaledDotOps::rhs_contracting_dim() const {
  return result->dot_dimension_numbers().rhs_contracting_dimensions(0);
}

int64_t BlockScaledDotOps::rhs_noncontracting_dim() const {
  const DotDimensionNumbers& dnums = result->dot_dimension_numbers();
  return GetNonContractingDims(rhs.result->shape().dimensions_size(),
                               dnums.rhs_contracting_dimensions(),
                               dnums.rhs_batch_dimensions())
      .front();
}

std::optional<int64_t> BlockScaledDotOps::rhs_batch_dim() const {
  auto batch = result->dot_dimension_numbers().rhs_batch_dimensions();
  return batch.empty() ? std::nullopt : std::make_optional(batch[0]);
}

bool BlockScaledDotOps::IsSupported() const {
  // LHS/RHS block sizes must be the same.
  int64_t block_size = lhs.GetBlockSize(lhs_contracting_dim());
  if (block_size != rhs.GetBlockSize(rhs_contracting_dim())) {
    return false;
  }

  // Match MXFP8 configuration.
  auto is_mxfp8 = [](const BlockScaledDequantizeOps& dq) -> bool {
    PrimitiveType input_type = dq.input->operand(0)->shape().element_type();
    PrimitiveType scale_type = dq.scale->operand(0)->shape().element_type();
    return (input_type == PrimitiveType::F8E4M3FN ||
            input_type == PrimitiveType::F8E5M2) &&
           scale_type == PrimitiveType::F8E8M0FNU;
  };
  if (is_mxfp8(lhs) && is_mxfp8(rhs) && block_size == kBlockSizeMXFP8) {
    return true;
  }

  // Match NVFP4 configuration.
  auto is_nvfp4 = [](const BlockScaledDequantizeOps& dq) -> bool {
    PrimitiveType input_type = dq.input->operand(0)->shape().element_type();
    PrimitiveType scale_type = dq.scale->operand(0)->shape().element_type();
    return input_type == PrimitiveType::F4E2M1FN &&
           scale_type == PrimitiveType::F8E4M3FN;
  };
  if (is_nvfp4(lhs) && is_nvfp4(rhs) && block_size == kBlockSizeNVFP4) {
    return true;
  }

  // Unsupported configuration.
  return false;
}

bool BlockScaledDotOps::IsMXFP8() const {
  CHECK(IsSupported());
  return lhs.scale->operand(0)->shape().element_type() ==
         PrimitiveType::F8E8M0FNU;
}

bool BlockScaledDotOps::IsNVFP4() const {
  CHECK(IsSupported());
  return lhs.scale->operand(0)->shape().element_type() ==
         PrimitiveType::F8E4M3FN;
}

}  // namespace block_scaling
}  // namespace xla::gpu
