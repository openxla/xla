/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/convert_operand_folder.h"

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "tsl/platform/errors.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

bool IsUpcastConvert(const HloInstruction* hlo) {
  if (!hlo->shape().IsArray()) {
    return false;
  }
  switch (hlo->opcode()) {
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kGather:
    case HloOpcode::kReshape:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose: {
      return IsUpcastConvert(hlo->operand(0));
    }
    case HloOpcode::kReduce: {
      if (ShapeUtil::ElementsIn(hlo->shape()) ==
          ShapeUtil::ElementsIn(hlo->operand(0)->shape())) {
        return IsUpcastConvert(hlo->operand(0));
      }
      return false;
    }
    case HloOpcode::kConvert:
      return primitive_util::CastPreservesValues(
          hlo->operand(0)->shape().element_type(), hlo->shape().element_type());
    default:
      return false;
  }
}

// Returns the element type of `operand` once its upcast, if any, is folded.
PrimitiveType FoldedElementType(const HloInstruction* operand) {
  if (!IsUpcastConvert(operand)) {
    return operand->shape().element_type();
  }
  while (operand->opcode() != HloOpcode::kConvert) {
    operand = operand->operand(0);
  }
  return operand->operand(0)->shape().element_type();
}

// Returns whether folding would leave `instruction` with two operands of at
// most 8 bits, at least one of them a float. Hardware can accumulate such
// contractions less precisely than the wider one the converts ask for.
bool FoldingCreatesNarrowFloatContraction(const HloInstruction* instruction) {
  const PrimitiveType lhs = FoldedElementType(instruction->operand(0));
  const PrimitiveType rhs = FoldedElementType(instruction->operand(1));
  return primitive_util::BitWidth(lhs) <= 8 &&
         primitive_util::BitWidth(rhs) <= 8 &&
         (primitive_util::IsFloatingPointType(lhs) ||
          primitive_util::IsFloatingPointType(rhs));
}

HloInstruction* EffectiveOperand(HloInstruction* hlo) {
  switch (hlo->opcode()) {
    case HloOpcode::kBroadcast:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kGather:
    case HloOpcode::kReshape:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose: {
      HloInstruction* operand = EffectiveOperand(hlo->mutable_operand(0));
      HloInstruction* clone = hlo->AddInstruction(hlo->Clone());
      *(clone->mutable_shape()) = ShapeUtil::ChangeElementType(
          clone->shape(), operand->shape().element_type());
      clone->ReplaceOperandWithDifferentShape(0, operand).IgnoreError();
      return clone;
    }
    case HloOpcode::kReduce: {
      // Reduce is a reshape in the case the the hlo chain was an upcast.
      HloInstruction* operand = EffectiveOperand(hlo->mutable_operand(0));
      return hlo->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::ChangeElementType(hlo->shape(),
                                       operand->shape().element_type()),
          operand));
    }
    case HloOpcode::kConvert:
      return hlo->mutable_operand(0);
    default:
      return nullptr;
  }
}

}  // namespace

bool ConvertOperandFolding::InstructionMatchesPattern(
    HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kDot &&
      instruction->opcode() != HloOpcode::kConvolution) {
    return false;
  }
  if (instruction->GetModule()
          ->config()
          .debug_options()
          .xla_keep_converts_of_narrow_float_dots() &&
      FoldingCreatesNarrowFloatContraction(instruction)) {
    return false;
  }
  for (auto* operand : instruction->operands()) {
    if (IsUpcastConvert(operand)) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<HloInstruction*> ConvertOperandFolding::ExpandInstruction(
    HloInstruction* instruction) {
  for (int i = 0; i < instruction->operand_count(); ++i) {
    auto* operand = instruction->mutable_operand(i);
    if (IsUpcastConvert(operand)) {
      ABSL_RETURN_IF_ERROR(instruction->ReplaceOperandWithDifferentShape(
          i, EffectiveOperand(operand)));
    }
  }
  return nullptr;
}

}  // namespace xla
