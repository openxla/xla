/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/transforms/while_loop_trip_count_annotator.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/numeric/int128.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/host_offload_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

constexpr int kMaxLinearExprDepth = 16;

// Returns true iff `v` fits in both signed `int64_t` and integral type `ty`
// without modular wraparound.
bool FitsInSignedAndType(absl::int128 v, PrimitiveType ty) {
  if (!primitive_util::IsIntegralType(ty)) {
    return false;
  }
  const int bits = primitive_util::BitWidth(ty);
  if (primitive_util::IsSignedIntegralType(ty)) {
    const absl::int128 lo = -(absl::int128{1} << (bits - 1));
    const absl::int128 hi = (absl::int128{1} << (bits - 1)) - 1;
    return v >= lo && v <= hi;
  }
  const absl::int128 hi =
      bits == 64 ? absl::int128{std::numeric_limits<int64_t>::max()}
                 : (absl::int128{1} << bits) - 1;
  return v >= 0 && v <= hi;
}

// Verifies that `while_op`'s condition compares the induction variable as an
// upper-bounded forward loop (`i < N`, `i <= N`, `N > i`, or `N >= i`) and is
// satisfied on the initial iteration (`init_val < N` or `init_val <= N`), so
// the loop executes at least once without `ConstantValue` bound underflow.
bool IsPositiveForwardLoopCondition(const HloInstruction* while_op,
                                    int64_t indvar_idx, int64_t init_val) {
  const HloComputation* cond = while_op->while_condition();
  const HloInstruction* root = cond->root_instruction();
  if (root->opcode() != HloOpcode::kCompare || cond->num_parameters() != 1) {
    return false;
  }
  const HloInstruction* param = cond->parameter_instruction(0);
  auto is_indvar = [&](const HloInstruction* op) {
    return op->opcode() == HloOpcode::kGetTupleElement &&
           op->operand(0) == param && op->tuple_index() == indvar_idx;
  };
  const bool indvar_lhs = is_indvar(root->operand(0));
  const bool indvar_rhs = is_indvar(root->operand(1));
  if (indvar_lhs == indvar_rhs) {
    return false;
  }
  const HloInstruction* bound_op = root->operand(indvar_lhs ? 1 : 0);
  if (!bound_op->IsConstant()) {
    return false;
  }
  std::optional<int64_t> bound =
      LiteralUtil::LiteralAsScalarInt64(bound_op->literal());
  if (!bound.has_value()) {
    return false;
  }
  const ComparisonDirection dir = root->comparison_direction();
  if ((indvar_lhs && dir == ComparisonDirection::kLt) ||
      (indvar_rhs && dir == ComparisonDirection::kGt)) {
    return init_val < *bound;
  }
  if ((indvar_lhs && dir == ComparisonDirection::kLe) ||
      (indvar_rhs && dir == ComparisonDirection::kGe)) {
    return init_val <= *bound;
  }
  return false;
}

std::optional<int64_t> ScalarConstantInitOperand(const HloInstruction* while_op,
                                                 int64_t tuple_idx) {
  const HloInstruction* init = while_op->operand(0);
  if (init->opcode() != HloOpcode::kTuple || tuple_idx < 0 ||
      tuple_idx >= init->operand_count()) {
    return std::nullopt;
  }
  const HloInstruction* init_val = init->operand(tuple_idx);
  if (init_val->opcode() != HloOpcode::kConstant ||
      !ShapeUtil::IsScalar(init_val->shape()) ||
      !primitive_util::IsIntegralType(init_val->shape().element_type())) {
    return std::nullopt;
  }
  return LiteralUtil::LiteralAsScalarInt64(init_val->literal());
}

struct AffineMatch {
  int64_t source_tuple_idx;
  uint64_t step_offset;
};

struct LinearTerm {
  std::optional<int64_t> source_tuple_idx;
  uint64_t constant_offset;
};

std::optional<LinearTerm> DecomposeLinear(const HloInstruction* expr,
                                          const HloInstruction* param,
                                          int depth = 0) {
  if (depth > kMaxLinearExprDepth || !ShapeUtil::IsScalar(expr->shape()) ||
      !primitive_util::IsIntegralType(expr->shape().element_type())) {
    return std::nullopt;
  }
  if (expr->opcode() == HloOpcode::kGetTupleElement &&
      expr->operand(0) == param) {
    return LinearTerm{expr->tuple_index(), 0};
  }
  if (expr->opcode() == HloOpcode::kConstant) {
    if (expr->shape().element_type() == U64) {
      return LinearTerm{std::nullopt, expr->literal().Get<uint64_t>({})};
    }
    std::optional<int64_t> v =
        LiteralUtil::LiteralAsScalarInt64(expr->literal());
    if (!v.has_value()) {
      return std::nullopt;
    }
    return LinearTerm{std::nullopt, static_cast<uint64_t>(*v)};
  }
  if (expr->opcode() == HloOpcode::kAdd ||
      expr->opcode() == HloOpcode::kSubtract) {
    std::optional<LinearTerm> lhs =
        DecomposeLinear(expr->operand(0), param, depth + 1);
    std::optional<LinearTerm> rhs =
        DecomposeLinear(expr->operand(1), param, depth + 1);
    if (!lhs.has_value() || !rhs.has_value()) {
      return std::nullopt;
    }
    const bool is_subtract = expr->opcode() == HloOpcode::kSubtract;
    if (lhs->source_tuple_idx.has_value() &&
        rhs->source_tuple_idx.has_value()) {
      return std::nullopt;
    }
    if (rhs->source_tuple_idx.has_value() && is_subtract) {
      return std::nullopt;
    }
    const uint64_t offset = is_subtract
                                ? lhs->constant_offset - rhs->constant_offset
                                : lhs->constant_offset + rhs->constant_offset;
    return LinearTerm{lhs->source_tuple_idx.has_value() ? lhs->source_tuple_idx
                                                        : rhs->source_tuple_idx,
                      offset};
  }
  return std::nullopt;
}

std::optional<AffineMatch> MatchAffineBodyUpdate(const HloInstruction* root,
                                                 const HloInstruction* param,
                                                 int64_t tuple_idx) {
  if (root->opcode() != HloOpcode::kTuple || tuple_idx < 0 ||
      tuple_idx >= root->operand_count()) {
    return std::nullopt;
  }
  std::optional<LinearTerm> term =
      DecomposeLinear(root->operand(tuple_idx), param);
  if (!term.has_value() || !term->source_tuple_idx.has_value()) {
    return std::nullopt;
  }
  return AffineMatch{*term->source_tuple_idx, term->constant_offset};
}

// Recovers exact non-wrapping affine `(init, step)` parameters such that the
// tuple slot equals `init + k * step` (in both its native type and `int64_t`)
// on every executed iteration `0 <= k < known_trip_count`, matching either a
// self-affine update or a one-iteration lagged copy of the primary induction
// variable.
std::optional<std::pair<int64_t, int64_t>> ResolveAffine(
    const HloInstruction* while_op, int64_t tuple_idx,
    std::optional<int64_t> known_trip_count, std::optional<int64_t> primary_idx,
    std::optional<int64_t> primary_init, std::optional<int64_t> primary_step) {
  if (primary_idx.has_value() && tuple_idx == *primary_idx &&
      primary_init.has_value() && primary_step.has_value()) {
    return std::make_pair(*primary_init, *primary_step);
  }
  const HloComputation* body = while_op->while_body();
  const HloInstruction* root = body->root_instruction();
  const HloInstruction* param = body->parameter_instruction(0);

  std::optional<AffineMatch> match =
      MatchAffineBodyUpdate(root, param, tuple_idx);
  if (!match.has_value()) {
    return std::nullopt;
  }
  std::optional<int64_t> init = ScalarConstantInitOperand(while_op, tuple_idx);
  const PrimitiveType ty = root->operand(tuple_idx)->shape().element_type();
  if (!init.has_value() || !FitsInSignedAndType(absl::int128{*init}, ty)) {
    return std::nullopt;
  }

  const int bits = primitive_util::BitWidth(ty);
  const uint64_t mask = bits == 64 ? ~uint64_t{0} : ((uint64_t{1} << bits) - 1);
  const uint64_t r = match->step_offset & mask;

  if (match->source_tuple_idx == tuple_idx) {
    if (r == 0) {
      return std::make_pair(*init, int64_t{0});
    }
    if (!known_trip_count.has_value() || *known_trip_count <= 0) {
      return std::nullopt;
    }
    constexpr uint64_t kMaxI64 =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    int64_t candidates[2];
    int num_candidates = 0;
    if (r <= kMaxI64) {
      candidates[num_candidates++] = static_cast<int64_t>(r);
    }
    if (bits < 64) {
      candidates[num_candidates++] =
          -static_cast<int64_t>((uint64_t{1} << bits) - r);
    } else if (r > kMaxI64) {
      candidates[num_candidates++] =
          static_cast<int64_t>(absl::int128{r} - (absl::int128{1} << 64));
    }
    if (*known_trip_count == 1) {
      if (num_candidates == 2 && (r & (uint64_t{1} << (bits - 1))) != 0) {
        std::swap(candidates[0], candidates[1]);
      }
      for (int i = 0; i < num_candidates; ++i) {
        if (FitsInSignedAndType(
                absl::int128{*init} + absl::int128{candidates[i]}, ty)) {
          return std::make_pair(*init, candidates[i]);
        }
      }
      return std::make_pair(*init, int64_t{0});
    }
    for (int i = 0; i < num_candidates; ++i) {
      const absl::int128 end =
          absl::int128{*init} +
          absl::int128{*known_trip_count - 1} * absl::int128{candidates[i]};
      if (FitsInSignedAndType(end, ty)) {
        return std::make_pair(*init, candidates[i]);
      }
    }
    return std::nullopt;
  }

  if (primary_idx.has_value() && match->source_tuple_idx == *primary_idx &&
      primary_init.has_value() && primary_step.has_value() &&
      known_trip_count.has_value() && *known_trip_count > 0) {
    const uint64_t lhs_mod =
        (static_cast<uint64_t>(*init) + static_cast<uint64_t>(*primary_step)) &
        mask;
    const uint64_t rhs_mod =
        (static_cast<uint64_t>(*primary_init) + match->step_offset) & mask;
    if (lhs_mod != rhs_mod) {
      return std::nullopt;
    }
    if (*known_trip_count >= 2 &&
        !FitsInSignedAndType(
            absl::int128{*init} + absl::int128{*known_trip_count - 1} *
                                      absl::int128{*primary_step},
            ty)) {
      return std::nullopt;
    }
    return std::make_pair(*init, *primary_step);
  }

  return std::nullopt;
}

bool WhileLoopBackendConfigsEqual(const WhileLoopBackendConfig& a,
                                  const WhileLoopBackendConfig& b) {
  if (a.has_known_trip_count() != b.has_known_trip_count() ||
      (a.has_known_trip_count() &&
       a.known_trip_count().n() != b.known_trip_count().n())) {
    return false;
  }
  if (a.has_known_init_step() != b.has_known_init_step() ||
      (a.has_known_init_step() &&
       (a.known_init_step().init() != b.known_init_step().init() ||
        a.known_init_step().step() != b.known_init_step().step()))) {
    return false;
  }
  if (a.has_known_induction_variable() != b.has_known_induction_variable() ||
      (a.has_known_induction_variable() &&
       a.known_induction_variable().tuple_index() !=
           b.known_induction_variable().tuple_index())) {
    return false;
  }
  if (a.dynamic_variables_size() != b.dynamic_variables_size()) {
    return false;
  }
  for (int i = 0; i < a.dynamic_variables_size(); ++i) {
    const WhileLoopBackendConfig::DynamicVariable& da = a.dynamic_variables(i);
    const WhileLoopBackendConfig::DynamicVariable& db = b.dynamic_variables(i);
    if (da.tuple_index() != db.tuple_index() ||
        da.has_init() != db.has_init() || da.has_step() != db.has_step() ||
        (da.has_init() && da.init() != db.init()) ||
        (da.has_step() && da.step() != db.step())) {
      return false;
    }
  }
  // Reject equality if either message or any submessage carries unknown fields.
  return a.ByteSizeLong() == b.ByteSizeLong();
}

// Replaces non-control-dependent scalar `get-tuple-element` users extracting
// `indvar_index` from `while_instr` with a constant equal to `final_value`,
// preserving layout, sharding, and derived instruction metadata.
absl::StatusOr<bool> ForwardInductionVarToConstants(HloInstruction* while_instr,
                                                    int64_t indvar_index,
                                                    absl::int128 final_value) {
  std::vector<HloInstruction*> indvar_gtes;
  for (HloInstruction* user : while_instr->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement &&
        user->tuple_index() == indvar_index &&
        !user->HasControlDependencies() && ShapeUtil::IsScalar(user->shape()) &&
        FitsInSignedAndType(final_value, user->shape().element_type())) {
      indvar_gtes.push_back(user);
    }
  }

  bool changed = false;
  for (HloInstruction* gte : indvar_gtes) {
    HloComputation* comp = gte->parent();
    Literal literal = LiteralUtil::CreateR0(gte->shape().element_type(),
                                            static_cast<int64_t>(final_value));
    std::unique_ptr<HloInstruction> constant_ptr =
        gte->shape().has_layout()
            ? std::make_unique<HloConstantInstruction>(std::move(literal),
                                                       gte->shape())
            : HloInstruction::CreateConstant(std::move(literal));
    HloInstruction* constant = comp->AddInstruction(std::move(constant_ptr));
    gte->SetupDerivedInstruction(constant);
    ABSL_RETURN_IF_ERROR(gte->ReplaceAllUsesWith(constant));
    ABSL_RETURN_IF_ERROR(comp->RemoveInstruction(gte));
    changed = true;
  }

  return changed;
}

}  // namespace

absl::StatusOr<bool> WhileLoopTripCountAnnotator::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp : module->computations(execution_threads)) {
    std::vector<HloInstruction*> while_ops;
    for (HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kWhile) {
        while_ops.push_back(instr);
      }
    }
    for (HloInstruction* instr : while_ops) {
      std::optional<int64_t> induction_variable_index =
          GetLoopInductionVarTupleIdx(instr);
      if (!induction_variable_index.has_value()) {
        continue;
      }

      ABSL_ASSIGN_OR_RETURN(WhileLoopBackendConfig existing_config,
                            instr->backend_config<WhileLoopBackendConfig>());

      WhileLoopBackendConfig config;
      config.mutable_known_induction_variable()->set_tuple_index(
          *induction_variable_index);

      absl::flat_hash_set<int64_t> dynamic_variable_tuple_indices =
          host_offload_utils::CollectDynamicVariableTupleIndices(instr);
      std::vector<int64_t> sorted_dynamic_indices(
          dynamic_variable_tuple_indices.begin(),
          dynamic_variable_tuple_indices.end());
      std::sort(sorted_dynamic_indices.begin(), sorted_dynamic_indices.end());
      for (int64_t tuple_idx : sorted_dynamic_indices) {
        config.add_dynamic_variables()->set_tuple_index(tuple_idx);
      }

      std::optional<int64_t> known_trip_count;
      std::optional<int64_t> primary_init;
      std::optional<int64_t> primary_step;
      if (std::optional<Range> range = MatchTrivialLoopRange(instr);
          range.has_value() && range->IsBounded() && range->IsStepKnown() &&
          range->min().GetSignedValue() >= 0 &&
          range->max()->GetSignedValue() >= range->min().GetSignedValue() &&
          range->step()->GetSignedValue() > 0 &&
          IsPositiveForwardLoopCondition(instr, *induction_variable_index,
                                         range->min().GetSignedValue())) {
        const int64_t min = range->min().GetSignedValue();
        const int64_t max = range->max()->GetSignedValue();
        const int64_t step = range->step()->GetSignedValue();
        const int64_t quot = (max - min) / step;
        if (quot < std::numeric_limits<int64_t>::max()) {
          const int64_t trip_count = quot + 1;
          known_trip_count = trip_count;
          config.mutable_known_trip_count()->set_n(trip_count);
          config.mutable_known_init_step()->set_init(min);
          config.mutable_known_init_step()->set_step(step);
          primary_init = min;
          primary_step = step;

          const absl::int128 final_value =
              absl::int128{min} + absl::int128{step} * absl::int128{trip_count};
          ABSL_ASSIGN_OR_RETURN(
              bool forwarded,
              ForwardInductionVarToConstants(instr, *induction_variable_index,
                                             final_value));
          changed |= forwarded;
        }
      }
      if (!known_trip_count.has_value()) {
        if (std::optional<int64_t> trip_count =
                ComputeWhileLoopTripCount(instr);
            trip_count.has_value() && *trip_count >= 0) {
          known_trip_count = *trip_count;
          config.mutable_known_trip_count()->set_n(*trip_count);
          if (*trip_count == 0) {
            if (std::optional<int64_t> init =
                    ScalarConstantInitOperand(instr, *induction_variable_index);
                init.has_value()) {
              ABSL_ASSIGN_OR_RETURN(
                  bool forwarded,
                  ForwardInductionVarToConstants(
                      instr, *induction_variable_index, absl::int128{*init}));
              changed |= forwarded;
            }
          }
        }
      }

      for (WhileLoopBackendConfig::DynamicVariable& dv :
           *config.mutable_dynamic_variables()) {
        std::optional<std::pair<int64_t, int64_t>> resolved =
            ResolveAffine(instr, dv.tuple_index(), known_trip_count,
                          induction_variable_index, primary_init, primary_step);
        if (!resolved.has_value()) {
          continue;
        }
        dv.set_init(resolved->first);
        dv.set_step(resolved->second);
      }

      if (!WhileLoopBackendConfigsEqual(existing_config, config)) {
        if (existing_config.ByteSizeLong() != 0) {
          LOG(WARNING) << absl::StrFormat(
              "WhileLoopTripCountAnnotator is overwriting an existing "
              "non-empty WhileLoopBackendConfig on %s: %s",
              instr->name(), existing_config.ShortDebugString());
        }
        ABSL_RETURN_IF_ERROR(instr->set_backend_config(config));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
