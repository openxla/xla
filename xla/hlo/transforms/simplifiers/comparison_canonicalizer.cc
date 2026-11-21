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

#include "xla/hlo/transforms/simplifiers/comparison_canonicalizer.h"

#include <optional>

#include "Eigen/Core"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

const Literal* GetScalarConstant(const HloInstruction* instr,
                                 PrimitiveType expected_type) {
  const auto* constant = DynCast<HloConstantInstruction>(instr);
  if (constant == nullptr || !constant->HasLiteral() ||
      !ShapeUtil::IsScalar(constant->shape()) ||
      constant->shape().element_type() != expected_type) {
    return nullptr;
  }
  return &constant->literal();
}

bool IsPositiveZero(const HloInstruction* instr, PrimitiveType expected_type) {
  const Literal* literal = GetScalarConstant(instr, expected_type);
  if (literal == nullptr) {
    return false;
  }
  return primitive_util::FloatingPointTypeSwitch(
      [&](auto primitive_type) -> bool {
        using NativeT = primitive_util::NativeTypeOf<primitive_type>;
        NativeT val = literal->GetFirstElement<NativeT>();
        return val == NativeT{0} && SignAndMagnitude(val).first == 0;
      },
      expected_type);
}

const Literal* GetPositiveNanLiteral(const HloInstruction* instr,
                                     PrimitiveType expected_type) {
  const Literal* literal = GetScalarConstant(instr, expected_type);
  if (literal == nullptr) {
    return nullptr;
  }
  bool is_pos_nan = primitive_util::FloatingPointTypeSwitch(
      [&](auto primitive_type) -> bool {
        using NativeT = primitive_util::NativeTypeOf<primitive_type>;
        NativeT val = literal->GetFirstElement<NativeT>();
        return Eigen::numext::isnan(val) && SignAndMagnitude(val).first == 0;
      },
      expected_type);
  return is_pos_nan ? literal : nullptr;
}

struct CanonicalizedOperand {
  HloInstruction* value;
  const Literal* nan_literal;
};

// Matches `select(compare(x, x, NE, PARTIAL), +nan,
//                 select(compare(x, +0.0, EQ, PARTIAL), +0.0, x))`.
std::optional<CanonicalizedOperand> MatchCanonicalizedOperand(
    HloInstruction* op, PrimitiveType element_type) {
  if (op->opcode() != HloOpcode::kSelect ||
      op->shape().element_type() != element_type) {
    return std::nullopt;
  }
  HloInstruction* inner = op->mutable_operand(2);
  if (inner->opcode() != HloOpcode::kSelect) {
    return std::nullopt;
  }
  HloInstruction* x = inner->mutable_operand(2);
  if (x->shape().element_type() != element_type) {
    return std::nullopt;
  }

  const HloInstruction* nan_cond = op->operand(0);
  if (nan_cond->opcode() != HloOpcode::kCompare ||
      nan_cond->comparison_direction() != ComparisonDirection::kNe ||
      nan_cond->comparison_order() != ComparisonOrder::kPartial ||
      nan_cond->operand(0) != x || nan_cond->operand(1) != x) {
    return std::nullopt;
  }

  const HloInstruction* zero_cond = inner->operand(0);
  if (zero_cond->opcode() != HloOpcode::kCompare ||
      zero_cond->comparison_direction() != ComparisonDirection::kEq ||
      zero_cond->comparison_order() != ComparisonOrder::kPartial ||
      zero_cond->operand(0) != x ||
      !IsPositiveZero(zero_cond->operand(1), element_type) ||
      !IsPositiveZero(inner->operand(1), element_type)) {
    return std::nullopt;
  }

  const Literal* nan_literal =
      GetPositiveNanLiteral(op->operand(1), element_type);
  if (nan_literal == nullptr) {
    return std::nullopt;
  }
  return CanonicalizedOperand{x, nan_literal};
}

}  // namespace

absl::StatusOr<bool> ComparisonCanonicalizer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (computation->caller_instructions(HloOpcode::kSort).empty()) {
      continue;
    }
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() != HloOpcode::kCompare ||
          instr->comparison_order() != ComparisonOrder::kTotal ||
          !ShapeUtil::IsScalar(instr->shape())) {
        continue;
      }
      const PrimitiveType element_type =
          instr->operand(0)->shape().element_type();
      if (!primitive_util::HasNegativeZero(element_type)) {
        continue;
      }
      std::optional<CanonicalizedOperand> lhs =
          MatchCanonicalizedOperand(instr->mutable_operand(0), element_type);
      if (!lhs.has_value()) {
        continue;
      }
      std::optional<CanonicalizedOperand> rhs =
          MatchCanonicalizedOperand(instr->mutable_operand(1), element_type);
      if (!rhs.has_value() || *lhs->nan_literal != *rhs->nan_literal) {
        continue;
      }
      ABSL_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
          instr,
          HloInstruction::CreateCompare(instr->shape(), lhs->value, rhs->value,
                                        instr->comparison_direction(),
                                        ComparisonOrder::kWeak),
          /*preserve_sharding=*/false, /*relay_control_dependency=*/true));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
