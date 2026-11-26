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

#include "xla/hlo/transforms/simplifiers/slice_sinker.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

constexpr int64_t kMaxLiftedTensorBytes = int64_t{8} * 1024 * 1024;
constexpr int64_t kMinDisjointSubrangeGroupSize = 4;
constexpr size_t kMaxCycleDfsVisited = 8192;

bool IsPreLayoutDefaultArray(const Shape& shape) {
  if (!shape.IsArray() || !shape.is_static() ||
      ShapeUtil::IsZeroElementArray(shape)) {
    return false;
  }
  const PrimitiveType ty = shape.element_type();
  if (primitive_util::IsComplexType(ty) ||
      primitive_util::IsSubByteNonPredType(ty)) {
    return false;
  }
  if (!shape.has_layout()) {
    return true;
  }
  const Layout& layout = shape.layout();
  return layout.tiles().empty() &&
         layout.memory_space() == Layout::kDefaultMemorySpace &&
         layout.element_size_in_bits() == 0 &&
         layout.tail_padding_alignment_in_elements() == 1 &&
         layout == LayoutUtil::GetDefaultLayoutForShape(shape);
}

bool HasCleanMetadata(const HloInstruction* inst) {
  return !inst->HasControlDependencies() && !inst->has_frontend_attributes() &&
         (!inst->has_sharding() || inst->sharding().IsReplicated());
}

bool SameResultAccuracy(const HloInstruction* a, const HloInstruction* b) {
  const bool ha = a->has_result_accuracy();
  const bool hb = b->has_result_accuracy();
  if (ha != hb) {
    return false;
  }
  if (!ha) {
    return true;
  }
  const ResultAccuracy& ra = a->result_accuracy();
  const ResultAccuracy& rb = b->result_accuracy();
  if (ra.mode() != rb.mode() || ra.has_tolerance() != rb.has_tolerance()) {
    return false;
  }
  if (!ra.has_tolerance()) {
    return true;
  }
  const double atol_a = ra.tolerance().atol();
  const double atol_b = rb.tolerance().atol();
  const double rtol_a = ra.tolerance().rtol();
  const double rtol_b = rb.tolerance().rtol();
  return ra.tolerance().ulps() == rb.tolerance().ulps() &&
         std::memcmp(&atol_a, &atol_b, sizeof(double)) == 0 &&
         std::memcmp(&rtol_a, &rtol_b, sizeof(double)) == 0;
}

bool SameScalarOperandOrBitwiseConstant(const HloInstruction* a,
                                        const HloInstruction* b) {
  if (a == b) {
    return true;
  }
  if (a->opcode() != HloOpcode::kConstant ||
      b->opcode() != HloOpcode::kConstant || !HasCleanMetadata(a) ||
      !HasCleanMetadata(b) || !ShapeUtil::IsScalar(a->shape()) ||
      !ShapeUtil::IsScalar(b->shape()) ||
      a->shape().element_type() != b->shape().element_type()) {
    return false;
  }
  const Literal& la = a->literal();
  const Literal& lb = b->literal();
  return la.size_bytes() == lb.size_bytes() &&
         (la.size_bytes() == 0 ||
          std::memcmp(la.untyped_data(), lb.untyped_data(),
                      static_cast<size_t>(la.size_bytes())) == 0);
}

bool IsAllowedScalarLift(const HloInstruction* inst) {
  auto wide_enough = [](PrimitiveType ty) {
    return ty == PRED || primitive_util::BitWidth(ty) >= 32;
  };
  const PrimitiveType dst_ty = inst->shape().element_type();
  if (!wide_enough(dst_ty)) {
    return false;
  }
  for (const HloInstruction* op : inst->operands()) {
    if (!wide_enough(op->shape().element_type())) {
      return false;
    }
  }
  switch (inst->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kSubtract:
    case HloOpcode::kMultiply:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kAbs:
    case HloOpcode::kNegate:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kNot:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kClz:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kMulhi:
    case HloOpcode::kSelect:
    case HloOpcode::kCompare:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kLog:
    case HloOpcode::kExp:
      return true;
    case HloOpcode::kConvert: {
      const PrimitiveType src_ty = inst->operand(0)->shape().element_type();
      if (primitive_util::IsFloatingPointType(src_ty) &&
          primitive_util::IsUnsignedIntegralType(dst_ty)) {
        return false;
      }
      if (primitive_util::IsFloatingPointType(src_ty) &&
          primitive_util::IsFloatingPointType(dst_ty) &&
          primitive_util::BitWidth(dst_ty) < primitive_util::BitWidth(src_ty)) {
        return false;
      }
      return true;
    }
    case HloOpcode::kDivide:
    case HloOpcode::kSqrt:
    case HloOpcode::kRsqrt:
      return dst_ty == F32 && !inst->has_result_accuracy() &&
             absl::c_all_of(inst->operands(), [](const HloInstruction* op) {
               return op->opcode() != HloOpcode::kConvert;
             });
    default:
      return false;
  }
}

bool IsEligibleElementwiseInst(const HloInstruction* inst) {
  if (!inst->IsElementwise() || inst->operand_count() == 0 ||
      inst->user_count() == 0 || inst->HasSideEffect() ||
      !HasCleanMetadata(inst) || !IsPreLayoutDefaultArray(inst->shape())) {
    return false;
  }
  if (inst->opcode() == HloOpcode::kCopy ||
      inst->opcode() == HloOpcode::kStochasticConvert ||
      inst->opcode() == HloOpcode::kReducePrecision) {
    return false;
  }
  for (const HloInstruction* op : inst->operands()) {
    if (!IsPreLayoutDefaultArray(op->shape())) {
      return false;
    }
  }
  if (inst->opcode() == HloOpcode::kBitcastConvert &&
      primitive_util::BitWidth(inst->operand(0)->shape().element_type()) !=
          primitive_util::BitWidth(inst->shape().element_type())) {
    return false;
  }
  if (inst->opcode() == HloOpcode::kCompare &&
      primitive_util::IsFloatingPointType(
          inst->operand(0)->shape().element_type()) &&
      inst->comparison_type() == Comparison::Type::kTotalOrder) {
    return false;
  }
  if (ShapeUtil::IsEffectiveScalar(inst->shape()) &&
      !IsAllowedScalarLift(inst)) {
    return false;
  }
  return true;
}

bool SameElementwiseOp(const HloInstruction* ref, const HloInstruction* cand) {
  return IsEligibleElementwiseInst(cand) && cand->SameOp(*ref) &&
         cand->shape().element_type() == ref->shape().element_type() &&
         cand->has_sharding() == ref->has_sharding() &&
         SameResultAccuracy(ref, cand);
}

bool IsCleanSlice(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kSlice && HasCleanMetadata(inst) &&
         IsPreLayoutDefaultArray(inst->shape()) &&
         IsPreLayoutDefaultArray(inst->operand(0)->shape());
}

bool SameSliceConfiguration(const HloInstruction* s1,
                            const HloInstruction* s2) {
  return s1->operand(0)->shape().dimensions() ==
             s2->operand(0)->shape().dimensions() &&
         s1->slice_starts() == s2->slice_starts() &&
         s1->slice_limits() == s2->slice_limits() &&
         s1->slice_strides() == s2->slice_strides();
}

bool IsUnitStrideFullExcept(const HloInstruction* slice, int64_t axis) {
  const Shape& src = slice->operand(0)->shape();
  const int64_t rank = src.dimensions().size();
  if (axis < 0 || axis >= rank || slice->slice_strides(axis) != 1) {
    return false;
  }
  for (int64_t k = 0; k < rank; ++k) {
    if (k == axis) {
      continue;
    }
    if (slice->slice_starts(k) != 0 ||
        slice->slice_limits(k) != src.dimensions(k) ||
        slice->slice_strides(k) != 1) {
      return false;
    }
  }
  return true;
}

// Returns true when `reshape` drops solely unit-sized axis `axis` of `slice`.
bool DropsOnlyUnitAxis(const HloInstruction* reshape,
                       const HloInstruction* slice, int64_t axis) {
  const auto in_dims = slice->shape().dimensions();
  const auto out_dims = reshape->shape().dimensions();
  if (axis < 0 || static_cast<size_t>(axis) >= in_dims.size() ||
      in_dims.size() != out_dims.size() + 1 ||
      in_dims[static_cast<size_t>(axis)] != 1) {
    return false;
  }
  size_t out_i = 0;
  for (size_t in_i = 0; in_i < in_dims.size(); ++in_i) {
    if (static_cast<int64_t>(in_i) == axis) {
      continue;
    }
    if (out_i >= out_dims.size() || in_dims[in_i] != out_dims[out_i]) {
      return false;
    }
    ++out_i;
  }
  return true;
}

struct PrimarySliceView {
  HloInstruction* source = nullptr;
  const HloInstruction* slice = nullptr;
  int64_t axis = -1;
  bool drops_unit_axis = false;
  int64_t start = 0;
  int64_t limit = 0;
};

std::optional<PrimarySliceView> MatchSingleAxisSlice(
    HloInstruction* op, std::optional<int64_t> required_axis = std::nullopt,
    std::optional<bool> required_drop = std::nullopt) {
  if (op->opcode() == HloOpcode::kSlice) {
    if ((required_drop.has_value() && *required_drop) || !IsCleanSlice(op)) {
      return std::nullopt;
    }
    const Shape& src = op->operand(0)->shape();
    const int64_t rank = src.dimensions().size();
    int64_t axis = -1;
    if (required_axis.has_value()) {
      if (!IsUnitStrideFullExcept(op, *required_axis)) {
        return std::nullopt;
      }
      axis = *required_axis;
    } else {
      for (int64_t k = 0; k < rank; ++k) {
        if (op->slice_starts(k) != 0 ||
            op->slice_limits(k) != src.dimensions(k) ||
            op->slice_strides(k) != 1) {
          if (axis >= 0 || op->slice_strides(k) != 1) {
            return std::nullopt;
          }
          axis = k;
        }
      }
      if (axis < 0) {
        return std::nullopt;
      }
    }
    return PrimarySliceView{
        op->mutable_operand(0), op, axis, false, op->slice_starts(axis),
        op->slice_limits(axis)};
  }
  if (op->opcode() == HloOpcode::kReshape) {
    if ((required_drop.has_value() && !*required_drop) ||
        !HasCleanMetadata(op) || !IsPreLayoutDefaultArray(op->shape())) {
      return std::nullopt;
    }
    HloInstruction* sl = op->mutable_operand(0);
    if (!IsCleanSlice(sl)) {
      return std::nullopt;
    }
    const int64_t rank = sl->shape().dimensions().size();
    if (rank != static_cast<int64_t>(op->shape().dimensions().size()) + 1) {
      return std::nullopt;
    }
    for (int64_t d = 0; d < rank; ++d) {
      if (required_axis.has_value() && d != *required_axis) {
        continue;
      }
      if (IsUnitStrideFullExcept(sl, d) && DropsOnlyUnitAxis(op, sl, d)) {
        return PrimarySliceView{
            sl->mutable_operand(0), sl, d, true, sl->slice_starts(d),
            sl->slice_limits(d)};
      }
    }
  }
  return std::nullopt;
}

std::optional<std::pair<HloInstruction*, int64_t>> MatchPerSliceScalarBroadcast(
    HloInstruction* op) {
  if (op->opcode() != HloOpcode::kBroadcast || !HasCleanMetadata(op) ||
      !op->dimensions().empty()) {
    return std::nullopt;
  }
  HloInstruction* inner = op->mutable_operand(0);
  if (!ShapeUtil::IsScalar(inner->shape())) {
    return std::nullopt;
  }
  auto sv = MatchSingleAxisSlice(inner, /*required_axis=*/0,
                                 /*required_drop=*/true);
  if (!sv.has_value() || sv->source->shape().dimensions().size() != 1 ||
      sv->limit - sv->start != 1) {
    return std::nullopt;
  }
  return std::make_pair(sv->source, sv->start);
}

enum class OperandRole {
  kPrimarySlice,
  kPerSliceScalarBroadcast,
  kInvariantScalar,
  kInvariantBroadcast,
};

struct OperandSpec {
  OperandRole role = OperandRole::kPrimarySlice;
  HloInstruction* source = nullptr;
  std::vector<int64_t> bcast_dims;
  int64_t ref_delta = 0;  // op_start - anchor_start for slice-backed roles.
};

bool NonAxisDimsMatch(const Shape& a, const Shape& b, int64_t axis) {
  if (a.dimensions().size() != b.dimensions().size()) {
    return false;
  }
  for (int64_t k = 0; k < static_cast<int64_t>(a.dimensions().size()); ++k) {
    if (k != axis && a.dimensions(k) != b.dimensions(k)) {
      return false;
    }
  }
  return true;
}

bool WouldCreateCycle(absl::Span<HloInstruction* const> dep_roots,
                      absl::Span<HloInstruction* const> group) {
  absl::flat_hash_set<const HloInstruction*> group_set(group.begin(),
                                                       group.end());
  absl::flat_hash_set<const HloInstruction*> visited;
  std::vector<const HloInstruction*> stack;
  for (const HloInstruction* r : dep_roots) {
    if (r != nullptr && visited.insert(r).second) {
      stack.push_back(r);
    }
  }
  while (!stack.empty()) {
    if (visited.size() > kMaxCycleDfsVisited) {
      return true;
    }
    const HloInstruction* cur = stack.back();
    stack.pop_back();
    if (group_set.contains(cur)) {
      return true;
    }
    for (const HloInstruction* op : cur->operands()) {
      if (visited.insert(op).second) {
        stack.push_back(op);
      }
    }
    for (const HloInstruction* cp : cur->control_predecessors()) {
      if (visited.insert(cp).second) {
        stack.push_back(cp);
      }
    }
  }
  return false;
}

Shape MakeDefaultShape(PrimitiveType ty, absl::Span<const int64_t> dims,
                       bool with_layout) {
  Shape s = ShapeUtil::MakeShape(ty, dims);
  if (!with_layout) {
    s.clear_layout();
  }
  return s;
}

HloInstruction* SliceAxisIfNeeded(HloComputation* comp, HloInstruction* src,
                                  int64_t axis, int64_t start, int64_t limit,
                                  bool with_layout) {
  const Shape& s = src->shape();
  if (start == 0 && limit == s.dimensions(axis)) {
    return src;
  }
  const int64_t rank = s.dimensions().size();
  std::vector<int64_t> starts(static_cast<size_t>(rank), 0);
  std::vector<int64_t> limits(s.dimensions().begin(), s.dimensions().end());
  std::vector<int64_t> strides(static_cast<size_t>(rank), 1);
  std::vector<int64_t> out_dims = limits;
  starts[static_cast<size_t>(axis)] = start;
  limits[static_cast<size_t>(axis)] = limit;
  out_dims[static_cast<size_t>(axis)] = limit - start;
  Shape out_shape = MakeDefaultShape(s.element_type(), out_dims, with_layout);
  return comp->AddInstruction(
      HloInstruction::CreateSlice(out_shape, src, starts, limits, strides));
}

// Tries single-axis unit-stride sinking (covering sub-range bounding boxes,
// multi-use unit-axis-drop reshapes, per-slice scalar broadcasts, and
// group-invariant scalars/broadcasts).
absl::StatusOr<bool> TrySinkSingleAxisGroup(
    HloInstruction* ref, absl::flat_hash_set<const HloInstruction*>* rejected) {
  int64_t anchor_idx = -1;
  PrimarySliceView anchor;
  for (int64_t i = 0; i < ref->operand_count(); ++i) {
    if (auto sv = MatchSingleAxisSlice(ref->mutable_operand(i));
        sv.has_value()) {
      anchor_idx = i;
      anchor = *sv;
      break;
    }
  }
  if (anchor_idx < 0) {
    return false;
  }

  const int64_t axis = anchor.axis;
  const bool drop = anchor.drops_unit_axis;
  const int64_t anchor_len = anchor.limit - anchor.start;
  std::vector<OperandSpec> specs(static_cast<size_t>(ref->operand_count()));
  bool uses_per_slice_bcast = false;

  for (int64_t i = 0; i < ref->operand_count(); ++i) {
    HloInstruction* op = ref->mutable_operand(i);
    OperandSpec& sp = specs[static_cast<size_t>(i)];
    if (auto sv = MatchSingleAxisSlice(op, axis, drop);
        sv.has_value() &&
        NonAxisDimsMatch(sv->source->shape(), anchor.source->shape(), axis) &&
        sv->limit - sv->start == anchor_len) {
      sp.role = OperandRole::kPrimarySlice;
      sp.source = sv->source;
      sp.ref_delta = sv->start - anchor.start;
      continue;
    }
    if (anchor_len == 1) {
      if (auto psb = MatchPerSliceScalarBroadcast(op); psb.has_value()) {
        sp.role = OperandRole::kPerSliceScalarBroadcast;
        sp.source = psb->first;
        sp.ref_delta = psb->second - anchor.start;
        uses_per_slice_bcast = true;
        continue;
      }
    }
    if (op->opcode() == HloOpcode::kBroadcast && HasCleanMetadata(op) &&
        IsPreLayoutDefaultArray(op->operand(0)->shape()) &&
        (drop || !absl::c_linear_search(op->dimensions(), axis))) {
      sp.role = OperandRole::kInvariantBroadcast;
      sp.source = op->mutable_operand(0);
      sp.bcast_dims.assign(op->dimensions().begin(), op->dimensions().end());
      continue;
    }
    if (ShapeUtil::IsScalar(ref->shape()) && ShapeUtil::IsScalar(op->shape())) {
      sp.role = OperandRole::kInvariantScalar;
      sp.source = op;
      continue;
    }
    return false;
  }

  struct MemberInfo {
    HloInstruction* inst = nullptr;
    int64_t start = 0;
    int64_t limit = 0;
  };
  auto match_member = [&](HloInstruction* cand) -> std::optional<MemberInfo> {
    if (!SameElementwiseOp(ref, cand) ||
        cand->operand_count() != ref->operand_count()) {
      return std::nullopt;
    }
    auto cand_anchor =
        MatchSingleAxisSlice(cand->mutable_operand(anchor_idx), axis, drop);
    if (!cand_anchor.has_value() || cand_anchor->source != anchor.source) {
      return std::nullopt;
    }
    const int64_t c_len = cand_anchor->limit - cand_anchor->start;
    if (uses_per_slice_bcast && c_len != 1) {
      return std::nullopt;
    }
    for (int64_t i = 0; i < cand->operand_count(); ++i) {
      HloInstruction* op = cand->mutable_operand(i);
      const OperandSpec& sp = specs[static_cast<size_t>(i)];
      switch (sp.role) {
        case OperandRole::kPrimarySlice: {
          auto sv = MatchSingleAxisSlice(op, axis, drop);
          if (!sv.has_value() || sv->source != sp.source ||
              sv->limit - sv->start != c_len ||
              sv->start - cand_anchor->start != sp.ref_delta) {
            return std::nullopt;
          }
          break;
        }
        case OperandRole::kPerSliceScalarBroadcast: {
          auto psb = MatchPerSliceScalarBroadcast(op);
          if (!psb.has_value() || psb->first != sp.source ||
              psb->second - cand_anchor->start != sp.ref_delta) {
            return std::nullopt;
          }
          break;
        }
        case OperandRole::kInvariantBroadcast:
          if (op->opcode() != HloOpcode::kBroadcast || !HasCleanMetadata(op) ||
              !absl::c_equal(op->dimensions(), sp.bcast_dims) ||
              !SameScalarOperandOrBitwiseConstant(op->operand(0), sp.source)) {
            return std::nullopt;
          }
          break;
        case OperandRole::kInvariantScalar:
          if (!SameScalarOperandOrBitwiseConstant(op, sp.source)) {
            return std::nullopt;
          }
          break;
      }
    }
    return MemberInfo{cand, cand_anchor->start, cand_anchor->limit};
  };

  std::vector<MemberInfo> members;
  std::vector<HloInstruction*> member_insts;
  absl::flat_hash_set<const HloInstruction*> seen;
  auto consider_user = [&](HloInstruction* user) {
    if (seen.contains(user)) {
      return;
    }
    if (auto mi = match_member(user); mi.has_value()) {
      seen.insert(user);
      members.push_back(*mi);
      member_insts.push_back(user);
    }
  };
  for (HloInstruction* u : anchor.source->users()) {
    if (u->opcode() != HloOpcode::kSlice || !IsCleanSlice(u) ||
        !IsUnitStrideFullExcept(u, axis)) {
      continue;
    }
    if (!drop) {
      for (HloInstruction* su : u->users()) {
        consider_user(su);
      }
    } else {
      for (HloInstruction* ru : u->users()) {
        if (ru->opcode() == HloOpcode::kReshape && HasCleanMetadata(ru) &&
            DropsOnlyUnitAxis(ru, u, axis)) {
          for (HloInstruction* su : ru->users()) {
            consider_user(su);
          }
        }
      }
    }
  }

  auto reject_group = [&]() -> bool {
    for (const HloInstruction* m : member_insts) {
      rejected->insert(m);
    }
    return false;
  };
  if (members.size() < 2) {
    return reject_group();
  }

  int64_t min_start = std::numeric_limits<int64_t>::max();
  int64_t max_limit = 0;
  int64_t sum_elems = 0;
  for (const MemberInfo& mi : members) {
    min_start = std::min(min_start, mi.start);
    max_limit = std::max(max_limit, mi.limit);
    sum_elems += ShapeUtil::ElementsIn(mi.inst->shape());
  }
  const int64_t bbox_len = max_limit - min_start;
  if (bbox_len <= 0) {
    return reject_group();
  }

  // Validate operand slice offsets (either aligned `delta == 0` or already
  // cropped `[0, bbox_len)` with `delta == -min_start` when `min_start > 0`).
  for (const OperandSpec& sp : specs) {
    if (sp.role != OperandRole::kPrimarySlice &&
        sp.role != OperandRole::kPerSliceScalarBroadcast) {
      continue;
    }
    const int64_t src_axis = sp.role == OperandRole::kPrimarySlice ? axis : 0;
    const int64_t src_extent = sp.source->shape().dimensions(src_axis);
    const bool unshifted = (sp.ref_delta == 0 && max_limit <= src_extent);
    const bool pre_cropped =
        (min_start > 0 && sp.ref_delta == -min_start && src_extent == bbox_len);
    if (!unshifted && !pre_cropped) {
      return reject_group();
    }
  }

  const bool with_layout = ref->shape().has_layout();
  std::vector<int64_t> lifted_dims(anchor.source->shape().dimensions().begin(),
                                   anchor.source->shape().dimensions().end());
  lifted_dims[static_cast<size_t>(axis)] = bbox_len;
  const Shape lifted_result_shape =
      MakeDefaultShape(ref->shape().element_type(), lifted_dims, with_layout);
  const int64_t bbox_elems = ShapeUtil::ElementsIn(lifted_result_shape);
  const bool is_full_source =
      (min_start == 0 && max_limit == anchor.source->shape().dimensions(axis));
  if (sum_elems < bbox_elems ||
      (!is_full_source &&
       static_cast<int64_t>(members.size()) < kMinDisjointSubrangeGroupSize &&
       sum_elems == bbox_elems)) {
    return reject_group();
  }

  // Enforce <= 8 MiB fusion-safety byte cap on lifted result and operands.
  if (ShapeUtil::ByteSizeOf(lifted_result_shape) > kMaxLiftedTensorBytes) {
    return reject_group();
  }
  std::vector<HloInstruction*> dep_roots;
  dep_roots.reserve(specs.size());
  for (size_t i = 0; i < specs.size(); ++i) {
    const PrimitiveType op_ty =
        ref->operand(static_cast<int64_t>(i))->shape().element_type();
    const Shape lifted_op_shape =
        MakeDefaultShape(op_ty, lifted_dims, with_layout);
    if (ShapeUtil::ByteSizeOf(lifted_op_shape) > kMaxLiftedTensorBytes) {
      return reject_group();
    }
    dep_roots.push_back(specs[i].source);
  }
  if (WouldCreateCycle(dep_roots, member_insts)) {
    return reject_group();
  }

  HloComputation* comp = ref->parent();
  std::vector<HloInstruction*> lifted_operands;
  lifted_operands.reserve(specs.size());
  for (size_t i = 0; i < specs.size(); ++i) {
    const OperandSpec& sp = specs[i];
    const PrimitiveType op_ty =
        ref->operand(static_cast<int64_t>(i))->shape().element_type();
    const Shape lifted_op_shape =
        MakeDefaultShape(op_ty, lifted_dims, with_layout);
    switch (sp.role) {
      case OperandRole::kPrimarySlice: {
        const int64_t st = min_start + sp.ref_delta;
        lifted_operands.push_back(SliceAxisIfNeeded(
            comp, sp.source, axis, st, st + bbox_len, with_layout));
        break;
      }
      case OperandRole::kPerSliceScalarBroadcast: {
        const int64_t st = min_start + sp.ref_delta;
        HloInstruction* vec = SliceAxisIfNeeded(comp, sp.source, /*axis=*/0, st,
                                                st + bbox_len, with_layout);
        lifted_operands.push_back(comp->AddInstruction(
            HloInstruction::CreateBroadcast(lifted_op_shape, vec, {axis})));
        break;
      }
      case OperandRole::kInvariantBroadcast: {
        std::vector<int64_t> dims;
        dims.reserve(sp.bcast_dims.size());
        for (int64_t d : sp.bcast_dims) {
          dims.push_back((drop && d >= axis) ? d + 1 : d);
        }
        lifted_operands.push_back(comp->AddInstruction(
            HloInstruction::CreateBroadcast(lifted_op_shape, sp.source, dims)));
        break;
      }
      case OperandRole::kInvariantScalar:
        lifted_operands.push_back(comp->AddInstruction(
            HloInstruction::CreateBroadcast(lifted_op_shape, sp.source, {})));
        break;
    }
  }

  HloInstruction* lifted = comp->AddInstruction(
      ref->CloneWithNewOperands(lifted_result_shape, lifted_operands));
  if (ref->has_sharding()) {
    lifted->set_sharding(ref->sharding());
  }

  const int64_t rank = lifted_dims.size();
  const std::vector<int64_t> unit_strides(static_cast<size_t>(rank), 1);
  for (const MemberInfo& mi : members) {
    std::vector<int64_t> starts(static_cast<size_t>(rank), 0);
    std::vector<int64_t> limits = lifted_dims;
    std::vector<int64_t> sl_dims = lifted_dims;
    starts[static_cast<size_t>(axis)] = mi.start - min_start;
    limits[static_cast<size_t>(axis)] = mi.limit - min_start;
    sl_dims[static_cast<size_t>(axis)] = mi.limit - mi.start;
    Shape sl_shape =
        MakeDefaultShape(mi.inst->shape().element_type(), sl_dims, with_layout);
    HloInstruction* repl = comp->AddInstruction(HloInstruction::CreateSlice(
        sl_shape, lifted, starts, limits, unit_strides));
    if (ref->has_sharding()) {
      repl->set_sharding(ref->sharding());
    }
    if (drop) {
      repl = comp->AddInstruction(
          HloInstruction::CreateReshape(mi.inst->shape(), repl));
      if (ref->has_sharding()) {
        repl->set_sharding(ref->sharding());
      }
    }
    ABSL_RETURN_IF_ERROR(mi.inst->ReplaceAllUsesWith(repl));
  }
  return true;
}

// Preserves full-source multi-axis and non-unit-stride slice sinking under the
// same pre-layout, 0-ULP, cycle, determinism, and <= 8 MiB guards.
absl::StatusOr<bool> TrySinkFullSourceGroup(
    HloInstruction* ref, absl::flat_hash_set<const HloInstruction*>* rejected) {
  if (absl::c_any_of(ref->operands(), [](const HloInstruction* op) {
        return !IsCleanSlice(op);
      })) {
    return false;
  }
  const HloInstruction* s0 = ref->operand(0);
  const Shape& src0_shape = s0->operand(0)->shape();
  for (int64_t i = 1; i < ref->operand_count(); ++i) {
    const HloInstruction* si = ref->operand(i);
    if (!ShapeUtil::CompatibleIgnoringElementType(src0_shape,
                                                  si->operand(0)->shape()) ||
        !SameSliceConfiguration(s0, si)) {
      return false;
    }
  }

  std::vector<HloInstruction*> group;
  absl::flat_hash_set<const HloInstruction*> seen;
  for (HloInstruction* sl : s0->operand(0)->users()) {
    if (!IsCleanSlice(sl)) {
      continue;
    }
    for (HloInstruction* cand : sl->users()) {
      if (seen.contains(cand) || !SameElementwiseOp(ref, cand) ||
          cand->operand_count() != ref->operand_count()) {
        continue;
      }
      const HloInstruction* cs0 = cand->operand(0);
      if (!IsCleanSlice(cs0) || cs0->operand(0) != s0->operand(0)) {
        continue;
      }
      bool ok = true;
      for (int64_t i = 1; i < cand->operand_count(); ++i) {
        const HloInstruction* csi = cand->operand(i);
        if (!IsCleanSlice(csi) ||
            csi->operand(0) != ref->operand(i)->operand(0) ||
            !SameSliceConfiguration(cs0, csi)) {
          ok = false;
          break;
        }
      }
      if (ok) {
        seen.insert(cand);
        group.push_back(cand);
      }
    }
  }

  auto reject_group = [&]() -> bool {
    for (const HloInstruction* m : group) {
      rejected->insert(m);
    }
    return false;
  };
  if (group.size() < 2) {
    return reject_group();
  }
  int64_t sum = 0;
  for (const HloInstruction* m : group) {
    sum += ShapeUtil::ElementsIn(m->shape());
  }
  if (sum < ShapeUtil::ElementsIn(src0_shape)) {
    return reject_group();
  }
  Shape new_shape =
      MakeDefaultShape(ref->shape().element_type(), src0_shape.dimensions(),
                       ref->shape().has_layout());
  if (ShapeUtil::ByteSizeOf(new_shape) > kMaxLiftedTensorBytes) {
    return reject_group();
  }
  std::vector<HloInstruction*> sources;
  sources.reserve(ref->operand_count());
  for (HloInstruction* sl : ref->mutable_operands()) {
    HloInstruction* src = sl->mutable_operand(0);
    if (ShapeUtil::ByteSizeOf(src->shape()) > kMaxLiftedTensorBytes) {
      return reject_group();
    }
    sources.push_back(src);
  }
  if (WouldCreateCycle(sources, group)) {
    return reject_group();
  }

  HloComputation* comp = ref->parent();
  HloInstruction* lifted =
      comp->AddInstruction(ref->CloneWithNewOperands(new_shape, sources));
  if (ref->has_sharding()) {
    lifted->set_sharding(ref->sharding());
  }
  for (HloInstruction* user : group) {
    const HloInstruction* op_sl = user->operand(0);
    HloInstruction* repl = comp->AddInstruction(
        op_sl->CloneWithNewOperands(user->shape(), {lifted}));
    if (ref->has_sharding()) {
      repl->set_sharding(ref->sharding());
    }
    VLOG(10) << "Adding new slice: " << repl->ToString()
             << " to replace: " << user->ToString();
    ABSL_RETURN_IF_ERROR(user->ReplaceAllUsesWith(repl));
  }
  return true;
}

}  // namespace

absl::StatusOr<bool> SliceSinker::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation : module->computations(execution_threads)) {
    absl::flat_hash_set<const HloInstruction*> rejected_full;
    absl::flat_hash_set<const HloInstruction*> rejected_axis;
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (!IsEligibleElementwiseInst(instruction)) {
        continue;
      }
      if (!rejected_full.contains(instruction)) {
        ABSL_ASSIGN_OR_RETURN(
            const bool sunk_full,
            TrySinkFullSourceGroup(instruction, &rejected_full));
        if (sunk_full) {
          changed = true;
          continue;
        }
      }
      if (!rejected_axis.contains(instruction)) {
        ABSL_ASSIGN_OR_RETURN(
            const bool sunk_axis,
            TrySinkSingleAxisGroup(instruction, &rejected_axis));
        if (sunk_axis) {
          changed = true;
        }
      }
    }
  }

  return changed;
}

}  // namespace xla
