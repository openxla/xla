/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/while_loop_simplifier.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/bit_set.h"
#include "xla/comparison_util.h"
#include "xla/frontend_attributes.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/ir/hlo_original_value_util.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/pass/hlo_pass_registry.h"
#include "xla/hlo/transforms/call_inliner.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/overflow_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/while_util.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/logging.h"
#include "xla/union_find.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace m = match;
using hlo_query::ContainsInstrWithOpcode;
using std::optional;

namespace {

constexpr int64_t kMaxRowPackLeaves = 8;
constexpr int64_t kMinRowPackLeafBytes = 512;
constexpr int64_t kMaxRowPackLeafRowBytes = 32 * 1024;
constexpr int64_t kMaxRowPackRowInflation = 8;
constexpr int64_t kMaxRowPackLoopGrowthBytes = int64_t{16} << 20;
constexpr int64_t kRowPackLaneWidth = 128;

bool IsAllZeroBytesLiteral(const Literal& literal) {
  if (!literal.shape().IsArray() || !ShapeUtil::IsScalar(literal.shape())) {
    return false;
  }
  const int64_t nbytes = literal.size_bytes();
  if (nbytes <= 0) {
    return false;
  }
  const auto* bytes = static_cast<const uint8_t*>(literal.untyped_data());
  return absl::c_all_of(absl::MakeSpan(bytes, static_cast<size_t>(nbytes)),
                        [](uint8_t b) { return b == 0; });
}

bool IsZeroByteElementType(PrimitiveType type) {
  if (!primitive_util::IsArrayType(type) || type == PRED || type == F8E8M0FNU) {
    return false;
  }
  const int bit_width = primitive_util::BitWidth(type);
  return bit_width >= 8 && (bit_width % 8) == 0 &&
         IsAllZeroBytesLiteral(LiteralUtil::Zero(type));
}

bool IsAllZeroScalarConstant(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kConstant &&
         !inst->HasControlDependencies() &&
         (!inst->has_sharding() ||
          (!inst->sharding().IsTuple() &&
           (inst->sharding().IsReplicated() || inst->sharding().IsManual()))) &&
         IsZeroByteElementType(inst->shape().element_type()) &&
         IsAllZeroBytesLiteral(inst->literal());
}

bool IsCleanRowMajorArray(const Shape& shape, bool strict_dim0_major) {
  if (!shape.IsArray() || !shape.is_static() ||
      ShapeUtil::IsZeroElementArray(shape) ||
      !IsZeroByteElementType(shape.element_type())) {
    return false;
  }
  if (!shape.has_layout()) {
    return true;
  }
  const Layout& layout = shape.layout();
  if (!layout.tiles().empty() ||
      layout.memory_space() != Layout::kDefaultMemorySpace ||
      layout.element_size_in_bits() != 0 ||
      layout.tail_padding_alignment_in_elements() > 1) {
    return false;
  }
  if (strict_dim0_major) {
    return LayoutUtil::IsMonotonicWithDim0Major(layout);
  }
  int64_t prev_non_unit_dim = shape.dimensions().size();
  for (int64_t d : layout.minor_to_major()) {
    if (shape.dimensions(d) > 1) {
      if (d >= prev_non_unit_dim) {
        return false;
      }
      prev_non_unit_dim = d;
    }
  }
  return true;
}

bool IsSimpleNonTiledSharding(const HloSharding& sharding) {
  return !sharding.IsTuple() &&
         (sharding.IsReplicated() || sharding.IsManual());
}

bool HasAllowedArraySharding(const HloInstruction& inst) {
  return !inst.has_sharding() || IsSimpleNonTiledSharding(inst.sharding());
}

bool HasAllowedTupleSharding(const HloInstruction& inst) {
  if (!inst.has_sharding()) {
    return true;
  }
  const HloSharding& sharding = inst.sharding();
  return sharding.IsTuple() ? absl::c_all_of(sharding.tuple_elements(),
                                             IsSimpleNonTiledSharding)
                            : IsSimpleNonTiledSharding(sharding);
}

bool AllUsersAreGte(const HloInstruction& inst) {
  return absl::c_all_of(inst.users(),
                        HloPredicateIsOp<HloOpcode::kGetTupleElement>);
}

bool HasAnySlotGte(const HloInstruction& tuple_inst, int64_t slot) {
  return absl::c_any_of(tuple_inst.users(), [slot](const HloInstruction* u) {
    return u->opcode() == HloOpcode::kGetTupleElement &&
           u->tuple_index() == slot;
  });
}

HloInstruction* FindUniqueSlotGte(const HloInstruction& tuple_inst,
                                  int64_t slot) {
  HloInstruction* found = nullptr;
  for (HloInstruction* user : tuple_inst.users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement &&
        user->tuple_index() == slot) {
      if (found != nullptr) {
        return nullptr;
      }
      found = user;
    }
  }
  return found;
}

bool IsEligibleRowPackWhileShell(const HloInstruction& w) {
  if (w.opcode() != HloOpcode::kWhile || !w.shape().IsTuple() ||
      &w == w.parent()->root_instruction() || w.HasControlDependencies() ||
      frontend::HasDisableWhileLoopDceAttr(&w) || !HasAllowedTupleSharding(w) ||
      !AllUsersAreGte(w)) {
    return false;
  }
  const HloInstruction* init = w.operand(0);
  const HloComputation* body = w.while_body();
  const HloComputation* cond = w.while_condition();
  if (init->opcode() != HloOpcode::kTuple || init->user_count() != 1 ||
      init->HasControlDependencies() || !HasAllowedTupleSharding(*init) ||
      body == cond || body->num_parameters() != 1 ||
      cond->num_parameters() != 1 || body->caller_instructions().size() != 1 ||
      cond->caller_instructions().size() != 1) {
    return false;
  }
  const HloInstruction* bp = body->parameter_instruction(0);
  const HloInstruction* cp = cond->parameter_instruction(0);
  const HloInstruction* root = body->root_instruction();
  return bp->shape().IsTuple() && cp->shape().IsTuple() &&
         HasAllowedTupleSharding(*bp) && HasAllowedTupleSharding(*cp) &&
         AllUsersAreGte(*bp) && AllUsersAreGte(*cp) &&
         root->opcode() == HloOpcode::kTuple && root->user_count() == 0 &&
         !root->HasControlDependencies() && HasAllowedTupleSharding(*root);
}

struct WriteOnlyRowDusChain {
  HloInstruction* outer_in_gte = nullptr;
  HloInstruction* inner_while = nullptr;
  int64_t inner_slot = -1;
  HloInstruction* inner_in_gte = nullptr;
  HloInstruction* inner_out_gte = nullptr;
  HloInstruction* cond = nullptr;
  int64_t cond_out_slot = -1;
  HloInstruction* cond_out_gte = nullptr;
  int updating_branch = -1;
  std::vector<int64_t> branch_in_slots;
  std::vector<HloInstruction*> branch_in_gtes;
  HloInstruction* dus = nullptr;
};

bool MatchFullRowDus(HloInstruction* in_gte, HloInstruction* dus,
                     const Shape& carry_shape) {
  const int64_t rank = carry_shape.dimensions().size();
  if (dus->opcode() != HloOpcode::kDynamicUpdateSlice ||
      in_gte->user_count() != 1 || dus->user_count() != 1 ||
      dus->HasControlDependencies() || dus->has_frontend_attributes() ||
      !HasAllowedArraySharding(*dus) ||
      !ShapeUtil::Compatible(dus->shape(), carry_shape) ||
      !IsCleanRowMajorArray(dus->shape(), /*strict_dim0_major=*/true) ||
      dus->operand_count() != 2 + rank || dus->operand(0) != in_gte ||
      dus->OperandIndices(in_gte).size() != 1) {
    return false;
  }
  const HloInstruction* upd = dus->operand(1);
  if (upd->user_count() != 1 || dus->OperandIndices(upd).size() != 1 ||
      upd->shape().element_type() != carry_shape.element_type() ||
      static_cast<int64_t>(upd->shape().dimensions().size()) != rank ||
      upd->shape().dimensions(0) != 1 ||
      !IsCleanRowMajorArray(upd->shape(), /*strict_dim0_major=*/false)) {
    return false;
  }
  for (int64_t d = 1; d < rank; ++d) {
    if (upd->shape().dimensions(d) != carry_shape.dimensions(d)) {
      return false;
    }
  }
  const HloInstruction* idx0 = dus->operand(2);
  const PrimitiveType idx_ty = idx0->shape().element_type();
  if (!idx0->shape().IsArray() || !ShapeUtil::IsScalar(idx0->shape()) ||
      !primitive_util::IsIntegralType(idx_ty)) {
    return false;
  }
  for (int64_t d = 1; d < rank; ++d) {
    const HloInstruction* zk = dus->operand(2 + d);
    if (zk->opcode() != HloOpcode::kConstant || zk->HasControlDependencies() ||
        zk->shape().element_type() != idx_ty ||
        !IsAllZeroBytesLiteral(zk->literal())) {
      return false;
    }
  }
  return true;
}

bool MatchWriteOnlyRowDusBody(HloInstruction* in_gte,
                              HloInstruction* root_tuple, int64_t out_slot,
                              const Shape& carry_shape, bool allow_inner_while,
                              WriteOnlyRowDusChain* chain) {
  if (in_gte == nullptr || in_gte->HasControlDependencies() ||
      !HasAllowedArraySharding(*in_gte) ||
      out_slot >= root_tuple->operand_count()) {
    return false;
  }
  HloInstruction* out_val = root_tuple->mutable_operand(out_slot);
  if (out_val == in_gte || root_tuple->OperandIndices(out_val).size() != 1 ||
      !root_tuple->OperandIndices(in_gte).empty()) {
    return false;
  }
  if (MatchFullRowDus(in_gte, out_val, carry_shape)) {
    chain->dus = out_val;
    return true;
  }
  if (out_val->opcode() != HloOpcode::kGetTupleElement ||
      out_val->user_count() != 1 || out_val->HasControlDependencies() ||
      !HasAllowedArraySharding(*out_val)) {
    return false;
  }
  HloInstruction* producer = out_val->mutable_operand(0);
  const int64_t prod_slot = out_val->tuple_index();
  if (FindUniqueSlotGte(*producer, prod_slot) != out_val) {
    return false;
  }
  if (allow_inner_while && producer->opcode() == HloOpcode::kWhile) {
    if (!IsEligibleRowPackWhileShell(*producer)) {
      return false;
    }
    HloInstruction* inner_init = producer->mutable_operand(0);
    if (in_gte->user_count() != 1 || in_gte->users()[0] != inner_init ||
        prod_slot >= inner_init->operand_count() ||
        inner_init->operand(prod_slot) != in_gte ||
        inner_init->OperandIndices(in_gte).size() != 1 ||
        HasAnySlotGte(*producer->while_condition()->parameter_instruction(0),
                      prod_slot)) {
      return false;
    }
    HloInstruction* inner_gte = FindUniqueSlotGte(
        *producer->while_body()->parameter_instruction(0), prod_slot);
    if (inner_gte == nullptr) {
      return false;
    }
    chain->inner_while = producer;
    chain->inner_slot = prod_slot;
    chain->inner_in_gte = inner_gte;
    chain->inner_out_gte = out_val;
    return MatchWriteOnlyRowDusBody(
        inner_gte, producer->while_body()->root_instruction(), prod_slot,
        carry_shape, /*allow_inner_while=*/false, chain);
  }
  if (producer->opcode() != HloOpcode::kConditional ||
      !producer->shape().IsTuple() ||
      producer == producer->parent()->root_instruction() ||
      producer->HasControlDependencies() ||
      !HasAllowedTupleSharding(*producer) || !AllUsersAreGte(*producer)) {
    return false;
  }
  const int bcount = producer->branch_count();
  if (in_gte->user_count() != bcount) {
    return false;
  }
  chain->branch_in_slots.assign(bcount, -1);
  chain->branch_in_gtes.assign(bcount, nullptr);
  int updating = -1;
  for (int b = 0; b < bcount; ++b) {
    HloInstruction* arg = producer->mutable_operand(b + 1);
    HloComputation* br = producer->branch_computation(b);
    if (arg->opcode() != HloOpcode::kTuple || arg->user_count() != 1 ||
        producer->OperandIndices(arg).size() != 1 ||
        arg->HasControlDependencies() || !HasAllowedTupleSharding(*arg) ||
        br->num_parameters() != 1 || br->caller_instructions().size() != 1 ||
        absl::c_count(producer->branch_computations(), br) != 1) {
      return false;
    }
    auto indices = arg->OperandIndices(in_gte);
    if (indices.size() != 1) {
      return false;
    }
    const int64_t in_slot = indices[0];
    HloInstruction* bp = br->parameter_instruction(0);
    HloInstruction* br_root = br->root_instruction();
    if (!bp->shape().IsTuple() || !HasAllowedTupleSharding(*bp) ||
        !AllUsersAreGte(*bp) || br_root->opcode() != HloOpcode::kTuple ||
        br_root->user_count() != 0 || br_root->HasControlDependencies() ||
        !HasAllowedTupleSharding(*br_root) ||
        prod_slot >= br_root->operand_count()) {
      return false;
    }
    HloInstruction* br_gte = FindUniqueSlotGte(*bp, in_slot);
    if (br_gte == nullptr || br_gte->HasControlDependencies() ||
        !HasAllowedArraySharding(*br_gte)) {
      return false;
    }
    HloInstruction* br_out = br_root->mutable_operand(prod_slot);
    if (br_root->OperandIndices(br_out).size() != 1) {
      return false;
    }
    if (br_out == br_gte) {
      if (br_gte->user_count() != 1) {
        return false;
      }
    } else if (updating == -1 && br_root->OperandIndices(br_gte).empty() &&
               MatchFullRowDus(br_gte, br_out, carry_shape)) {
      updating = b;
      chain->dus = br_out;
    } else {
      return false;
    }
    chain->branch_in_slots[b] = in_slot;
    chain->branch_in_gtes[b] = br_gte;
  }
  if (updating < 0) {
    return false;
  }
  chain->cond = producer;
  chain->cond_out_slot = prod_slot;
  chain->cond_out_gte = out_val;
  chain->updating_branch = updating;
  return true;
}

bool IsEligibleRowPackInteriorOp(const HloInstruction& inst,
                                 const HloInstruction* carry_gte) {
  if (&inst == carry_gte || &inst == inst.parent()->root_instruction() ||
      inst.HasControlDependencies() || inst.has_frontend_attributes() ||
      !HasAllowedArraySharding(inst) ||
      !IsCleanRowMajorArray(inst.shape(), /*strict_dim0_major=*/false)) {
    return false;
  }
  switch (inst.opcode()) {
    case HloOpcode::kSlice:
      return !inst.shape().dimensions().empty() &&
             IsCleanRowMajorArray(inst.operand(0)->shape(),
                                  /*strict_dim0_major=*/false);
    case HloOpcode::kConcatenate:
      return inst.operand_count() >= 1 &&
             absl::c_all_of(inst.operands(), [](const HloInstruction* op) {
               return IsCleanRowMajorArray(op->shape(),
                                           /*strict_dim0_major=*/false);
             });
    case HloOpcode::kPad:
      return IsCleanRowMajorArray(inst.operand(0)->shape(),
                                  /*strict_dim0_major=*/false) &&
             IsAllZeroScalarConstant(inst.operand(1));
    case HloOpcode::kReshape:
    case HloOpcode::kBitcast:
    case HloOpcode::kCopy:
      return inst.operand(0)->shape().element_type() ==
                 inst.shape().element_type() &&
             ShapeUtil::ElementsIn(inst.operand(0)->shape()) ==
                 ShapeUtil::ElementsIn(inst.shape()) &&
             IsCleanRowMajorArray(inst.operand(0)->shape(),
                                  /*strict_dim0_major=*/false);
    case HloOpcode::kBitcastConvert:
      return inst.operand(0)->shape().dimensions() ==
                 inst.shape().dimensions() &&
             primitive_util::BitWidth(
                 inst.operand(0)->shape().element_type()) ==
                 primitive_util::BitWidth(inst.shape().element_type()) &&
             IsCleanRowMajorArray(inst.operand(0)->shape(),
                                  /*strict_dim0_major=*/false);
    case HloOpcode::kBroadcast:
      return (inst.dimensions().empty() &&
              IsAllZeroScalarConstant(inst.operand(0))) ||
             IsCleanRowMajorArray(inst.operand(0)->shape(),
                                  /*strict_dim0_major=*/false);
    default:
      return false;
  }
}

struct RowPackDag {
  std::vector<HloInstruction*> topo_ops;
  std::vector<HloInstruction*> leaves;
  int64_t net_growth_bytes = 0;
};

std::optional<RowPackDag> CollectZeroPreservingRowPackDag(
    HloInstruction* dus, const HloInstruction* carry_gte, int64_t num_rows,
    int64_t remaining_growth_bytes) {
  HloInstruction* upd = dus->mutable_operand(1);
  if (!IsEligibleRowPackInteriorOp(*upd, carry_gte)) {
    return std::nullopt;
  }
  HloComputation* comp = dus->parent();
  std::vector<HloInstruction*> post_order = comp->MakeInstructionPostOrder();
  absl::flat_hash_set<const HloInstruction*> in_pack;
  for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
    HloInstruction* inst = *it;
    if (inst == comp->root_instruction() ||
        !IsEligibleRowPackInteriorOp(*inst, carry_gte)) {
      continue;
    }
    if (inst == upd ||
        (!inst->users().empty() &&
         absl::c_all_of(inst->users(), [&](const HloInstruction* u) {
           return in_pack.contains(u);
         }))) {
      in_pack.insert(inst);
    }
  }
  for (HloInstruction* inst : post_order) {
    if (in_pack.contains(inst) && inst->opcode() == HloOpcode::kBroadcast &&
        !(inst->dimensions().empty() &&
          IsAllZeroScalarConstant(inst->operand(0))) &&
        !in_pack.contains(inst->operand(0))) {
      in_pack.erase(inst);
    }
  }
  if (!in_pack.contains(upd)) {
    return std::nullopt;
  }

  auto data_operands =
      [](HloInstruction* inst) -> absl::InlinedVector<HloInstruction*, 4> {
    if (inst->opcode() == HloOpcode::kPad) {
      return {inst->mutable_operand(0)};
    }
    if (inst->opcode() == HloOpcode::kBroadcast && inst->dimensions().empty() &&
        IsAllZeroScalarConstant(inst->operand(0))) {
      return {};
    }
    return {inst->operands().begin(), inst->operands().end()};
  };

  absl::flat_hash_set<const HloInstruction*> reachable = {upd};
  std::vector<HloInstruction*> worklist = {upd};
  while (!worklist.empty()) {
    HloInstruction* cur = worklist.back();
    worklist.pop_back();
    for (HloInstruction* op : data_operands(cur)) {
      if (in_pack.contains(op) && reachable.insert(op).second) {
        worklist.push_back(op);
      }
    }
  }

  RowPackDag dag;
  absl::flat_hash_set<const HloInstruction*> seen_leaves;
  int64_t nontrivial_ops = 0;
  int64_t sublane_slices = 0;
  for (HloInstruction* inst : post_order) {
    if (!reachable.contains(inst)) {
      continue;
    }
    dag.topo_ops.push_back(inst);
    const HloOpcode op = inst->opcode();
    if (op == HloOpcode::kSlice || op == HloOpcode::kConcatenate ||
        op == HloOpcode::kPad) {
      ++nontrivial_ops;
    }
    if (op == HloOpcode::kSlice &&
        inst->shape().dimensions().back() < kRowPackLaneWidth) {
      ++sublane_slices;
    }
    for (HloInstruction* operand : data_operands(inst)) {
      if (!reachable.contains(operand) && seen_leaves.insert(operand).second) {
        dag.leaves.push_back(operand);
      }
    }
  }

  const int64_t m = static_cast<int64_t>(dag.leaves.size());
  if (m < 1 || m > kMaxRowPackLeaves || nontrivial_ops < 2 * m + 2 ||
      (m > 1 && sublane_slices < m + 4)) {
    return std::nullopt;
  }
  const int64_t upd_bytes = ShapeUtil::ByteSizeOfElements(upd->shape());
  const int64_t max_row_bytes = std::min<int64_t>(
      kMaxRowPackLeafRowBytes,
      upd_bytes <= kMaxRowPackLeafRowBytes / kMaxRowPackRowInflation
          ? upd_bytes * kMaxRowPackRowInflation
          : kMaxRowPackLeafRowBytes);
  int64_t sum_leaf_bytes = 0;
  for (const HloInstruction* leaf : dag.leaves) {
    const Shape& ls = leaf->shape();
    const int64_t rank = ls.dimensions().size();
    if (leaf == carry_gte || rank < 1 || !HasAllowedArraySharding(*leaf) ||
        !IsCleanRowMajorArray(
            ls, /*strict_dim0_major=*/rank >= 2 && ls.dimensions(0) > 1)) {
      return std::nullopt;
    }
    const int64_t lb = ShapeUtil::ByteSizeOfElements(ls);
    const int bw = primitive_util::BitWidth(ls.element_type());
    if (lb < kMinRowPackLeafBytes ||
        (bw < 32 && (rank < 2 || (ls.dimensions(rank - 2) % (32 / bw)) != 0)) ||
        (m > 1 && (ls.dimensions(rank - 1) % kRowPackLaneWidth) != 0) ||
        lb > max_row_bytes - sum_leaf_bytes) {
      return std::nullopt;
    }
    sum_leaf_bytes += lb;
  }
  if (sum_leaf_bytes > upd_bytes) {
    const int64_t row_delta = sum_leaf_bytes - upd_bytes;
    if (num_rows > remaining_growth_bytes / row_delta) {
      return std::nullopt;
    }
    dag.net_growth_bytes = num_rows * row_delta;
  }
  return dag;
}

Shape MakeRowPackShape(PrimitiveType type, absl::Span<const int64_t> dims,
                       bool with_default_layout) {
  return with_default_layout
             ? ShapeUtil::MakeShapeWithDescendingLayout(type, dims)
             : ShapeUtil::MakeShape(type, dims);
}

void ApplyWidenedRowPackSharding(HloInstruction* inst, const Shape& old_shape,
                                 const std::optional<HloSharding>& old_sharding,
                                 int64_t slot) {
  if (!old_sharding.has_value()) {
    return;
  }
  if (!old_sharding->IsTuple()) {
    inst->set_sharding(*old_sharding);
    return;
  }
  const int64_t old_count = old_shape.tuple_shapes().size();
  const HloSharding slot_sharding =
      old_sharding->GetSubSharding(old_shape, {slot});
  ShapeTree<HloSharding> tree(inst->shape(), slot_sharding);
  tree.ForEachMutableElement([&](const ShapeIndex& idx, HloSharding* sub) {
    if (!idx.empty() && !ShapeUtil::GetSubshape(inst->shape(), idx).IsTuple() &&
        idx[0] < old_count && idx[0] != slot) {
      *sub = old_sharding->GetSubSharding(old_shape, idx);
    }
  });
  inst->set_sharding(HloSharding::Tuple(tree));
}

void WidenTupleInstInPlace(HloInstruction* inst, int64_t slot,
                           absl::Span<const Shape> collector_shapes) {
  const Shape old_shape = inst->shape();
  const std::optional<HloSharding> old_sharding =
      inst->has_sharding() ? std::optional<HloSharding>(inst->sharding())
                           : std::nullopt;
  *inst->mutable_shape()->mutable_tuple_shapes(slot) = collector_shapes[0];
  for (size_t j = 1; j < collector_shapes.size(); ++j) {
    *inst->mutable_shape()->add_tuple_shapes() = collector_shapes[j];
  }
  inst->set_original_value(nullptr);
  ApplyWidenedRowPackSharding(inst, old_shape, old_sharding, slot);
}

HloInstruction* BuildWidenedTuple(HloInstruction* old_tuple, int64_t slot,
                                  absl::Span<HloInstruction* const> new_vals) {
  const Shape old_shape = old_tuple->shape();
  const std::optional<HloSharding> old_sharding =
      old_tuple->has_sharding()
          ? std::optional<HloSharding>(old_tuple->sharding())
          : std::nullopt;
  std::vector<HloInstruction*> ops(old_tuple->operands().begin(),
                                   old_tuple->operands().end());
  ops[slot] = new_vals[0];
  ops.insert(ops.end(), new_vals.begin() + 1, new_vals.end());
  HloInstruction* widened =
      old_tuple->parent()->AddInstruction(HloInstruction::CreateTuple(ops));
  widened->set_metadata(old_tuple->metadata());
  ApplyWidenedRowPackSharding(widened, old_shape, old_sharding, slot);
  return widened;
}

std::vector<HloInstruction*> EmitCollectorGtes(
    HloInstruction* tuple_inst, int64_t slot, int64_t old_count,
    const HloInstruction& tmpl, absl::Span<const Shape> collector_shapes) {
  std::vector<HloInstruction*> gtes;
  gtes.reserve(collector_shapes.size());
  HloComputation* comp = tuple_inst->parent();
  for (size_t j = 0; j < collector_shapes.size(); ++j) {
    const int64_t idx = j == 0 ? slot : old_count + static_cast<int64_t>(j - 1);
    HloInstruction* gte =
        comp->AddInstruction(HloInstruction::CreateGetTupleElement(
            collector_shapes[j], tuple_inst, idx));
    gte->set_frontend_attributes(tmpl.frontend_attributes());
    gte->set_metadata(tmpl.metadata());
    if (tmpl.has_sharding()) {
      gte->set_sharding(tmpl.sharding());
    }
    gtes.push_back(gte);
  }
  return gtes;
}

absl::StatusOr<bool> TryHoistOneWriteOnlyDusCarry(HloInstruction* outer_while,
                                                  int64_t* loop_growth_bytes) {
  if (!IsEligibleRowPackWhileShell(*outer_while)) {
    return false;
  }
  HloComputation* preheader = outer_while->parent();
  HloInstruction* outer_init = outer_while->mutable_operand(0);
  HloComputation* outer_body = outer_while->while_body();
  HloComputation* outer_cond = outer_while->while_condition();

  for (int64_t slot = 0; slot < outer_init->operand_count(); ++slot) {
    HloInstruction* c_init = outer_init->mutable_operand(slot);
    const Shape carry_shape = c_init->shape();
    if (c_init->opcode() != HloOpcode::kBroadcast ||
        !c_init->dimensions().empty() || c_init->HasControlDependencies() ||
        !HasAllowedArraySharding(*c_init) ||
        !IsCleanRowMajorArray(carry_shape, /*strict_dim0_major=*/true) ||
        carry_shape.dimensions().size() < 2 || carry_shape.dimensions(0) < 2 ||
        !IsAllZeroScalarConstant(c_init->operand(0)) ||
        HasAnySlotGte(*outer_cond->parameter_instruction(0), slot)) {
      continue;
    }
    std::vector<HloInstruction*> post_gtes;
    bool post_gtes_ok = true;
    for (HloInstruction* u : outer_while->users()) {
      if (u->tuple_index() == slot) {
        if (u->HasControlDependencies() || !HasAllowedArraySharding(*u)) {
          post_gtes_ok = false;
          break;
        }
        post_gtes.push_back(u);
      }
    }
    if (!post_gtes_ok || post_gtes.empty()) {
      continue;
    }
    HloInstruction* outer_gte =
        FindUniqueSlotGte(*outer_body->parameter_instruction(0), slot);
    if (outer_gte == nullptr) {
      continue;
    }

    WriteOnlyRowDusChain chain;
    chain.outer_in_gte = outer_gte;
    if (!MatchWriteOnlyRowDusBody(outer_gte, outer_body->root_instruction(),
                                  slot, carry_shape, /*allow_inner_while=*/true,
                                  &chain)) {
      continue;
    }
    const int64_t num_rows = carry_shape.dimensions(0);
    const HloInstruction* leaf_carry_gte =
        chain.cond != nullptr
            ? chain.branch_in_gtes[chain.updating_branch]
            : (chain.inner_while != nullptr ? chain.inner_in_gte
                                            : chain.outer_in_gte);
    std::optional<RowPackDag> dag = CollectZeroPreservingRowPackDag(
        chain.dus, leaf_carry_gte, num_rows,
        kMaxRowPackLoopGrowthBytes - *loop_growth_bytes);
    if (!dag.has_value()) {
      continue;
    }

    const bool with_layout = carry_shape.has_layout();
    const size_t m = dag->leaves.size();
    std::vector<Shape> collector_shapes;
    std::vector<Shape> row_upd_shapes;
    std::vector<HloInstruction*> init_vals;
    collector_shapes.reserve(m);
    row_upd_shapes.reserve(m);
    init_vals.reserve(m);
    for (const HloInstruction* leaf : dag->leaves) {
      const PrimitiveType ty = leaf->shape().element_type();
      absl::InlinedVector<int64_t, 6> c_dims = {num_rows};
      absl::InlinedVector<int64_t, 6> u_dims = {1};
      if (leaf->shape().dimensions().size() == 1) {
        c_dims.push_back(1);
        u_dims.push_back(1);
      }
      c_dims.insert(c_dims.end(), leaf->shape().dimensions().begin(),
                    leaf->shape().dimensions().end());
      u_dims.insert(u_dims.end(), leaf->shape().dimensions().begin(),
                    leaf->shape().dimensions().end());
      collector_shapes.push_back(MakeRowPackShape(ty, c_dims, with_layout));
      row_upd_shapes.push_back(MakeRowPackShape(ty, u_dims, with_layout));
      HloInstruction* zero = preheader->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(ty)));
      HloInstruction* bcast = preheader->AddInstruction(
          HloInstruction::CreateBroadcast(collector_shapes.back(), zero, {}));
      bcast->set_metadata(c_init->metadata());
      if (c_init->has_sharding()) {
        bcast->set_sharding(c_init->sharding());
      }
      init_vals.push_back(bcast);
    }

    std::vector<HloInstruction*> dead_tuples;
    std::vector<HloInstruction*> dead_gtes = {chain.outer_in_gte};

    // Widen outer_while init, shape, and parameters in place.
    const int64_t outer_old_count = outer_while->shape().tuple_shapes().size();
    ABSL_RETURN_IF_ERROR(outer_while->ReplaceOperandWithDifferentShape(
        0, BuildWidenedTuple(outer_init, slot, init_vals)));
    dead_tuples.push_back(outer_init);
    WidenTupleInstInPlace(outer_while, slot, collector_shapes);
    WidenTupleInstInPlace(outer_body->parameter_instruction(0), slot,
                          collector_shapes);
    WidenTupleInstInPlace(outer_cond->parameter_instruction(0), slot,
                          collector_shapes);
    std::vector<HloInstruction*> cur_in_gtes = EmitCollectorGtes(
        outer_body->parameter_instruction(0), slot, outer_old_count,
        *chain.outer_in_gte, collector_shapes);

    if (chain.inner_while != nullptr) {
      HloInstruction* iw = chain.inner_while;
      const int64_t iw_old_count = iw->shape().tuple_shapes().size();
      HloInstruction* iw_init = iw->mutable_operand(0);
      ABSL_RETURN_IF_ERROR(iw->ReplaceOperandWithDifferentShape(
          0, BuildWidenedTuple(iw_init, chain.inner_slot, cur_in_gtes)));
      dead_tuples.push_back(iw_init);
      WidenTupleInstInPlace(iw, chain.inner_slot, collector_shapes);
      WidenTupleInstInPlace(iw->while_body()->parameter_instruction(0),
                            chain.inner_slot, collector_shapes);
      WidenTupleInstInPlace(iw->while_condition()->parameter_instruction(0),
                            chain.inner_slot, collector_shapes);
      cur_in_gtes = EmitCollectorGtes(
          iw->while_body()->parameter_instruction(0), chain.inner_slot,
          iw_old_count, *chain.inner_in_gte, collector_shapes);
      dead_gtes.push_back(chain.inner_in_gte);
    }

    auto emit_leaf_duses = [&](absl::Span<HloInstruction* const> base_gtes) {
      HloComputation* b_star = chain.dus->parent();
      HloInstruction* idx0 = chain.dus->mutable_operand(2);
      HloInstruction* z_idx = chain.dus->mutable_operand(3);
      std::vector<HloInstruction*> duses;
      duses.reserve(m);
      for (size_t j = 0; j < m; ++j) {
        HloInstruction* upd_j = b_star->AddInstruction(
            HloInstruction::CreateReshape(row_upd_shapes[j], dag->leaves[j]));
        upd_j->set_metadata(chain.dus->metadata());
        std::vector<HloInstruction*> starts(
            collector_shapes[j].dimensions().size(), z_idx);
        starts[0] = idx0;
        HloInstruction* dus_j =
            b_star->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
                collector_shapes[j], base_gtes[j], upd_j, starts));
        dus_j->set_metadata(chain.dus->metadata());
        if (chain.dus->has_sharding()) {
          dus_j->set_sharding(chain.dus->sharding());
        }
        duses.push_back(dus_j);
      }
      return duses;
    };

    std::vector<HloInstruction*> cur_out_vals;
    if (chain.cond != nullptr) {
      HloInstruction* cond = chain.cond;
      const int64_t cond_old_count = cond->shape().tuple_shapes().size();
      WidenTupleInstInPlace(cond, chain.cond_out_slot, collector_shapes);
      for (int b = 0; b < cond->branch_count(); ++b) {
        HloInstruction* old_arg = cond->mutable_operand(b + 1);
        const int64_t arg_old_count = old_arg->shape().tuple_shapes().size();
        ABSL_RETURN_IF_ERROR(cond->ReplaceOperandWithDifferentShape(
            b + 1,
            BuildWidenedTuple(old_arg, chain.branch_in_slots[b], cur_in_gtes)));
        dead_tuples.push_back(old_arg);
        HloComputation* br = cond->branch_computation(b);
        WidenTupleInstInPlace(br->parameter_instruction(0),
                              chain.branch_in_slots[b], collector_shapes);
        std::vector<HloInstruction*> br_gtes = EmitCollectorGtes(
            br->parameter_instruction(0), chain.branch_in_slots[b],
            arg_old_count, *chain.branch_in_gtes[b], collector_shapes);
        dead_gtes.push_back(chain.branch_in_gtes[b]);
        std::vector<HloInstruction*> br_outs =
            b == chain.updating_branch ? emit_leaf_duses(br_gtes) : br_gtes;
        HloInstruction* old_br_root = br->root_instruction();
        br->set_root_instruction(
            BuildWidenedTuple(old_br_root, chain.cond_out_slot, br_outs),
            /*accept_different_shape=*/true);
        dead_tuples.push_back(old_br_root);
      }
      cur_out_vals =
          EmitCollectorGtes(cond, chain.cond_out_slot, cond_old_count,
                            *chain.cond_out_gte, collector_shapes);
      dead_gtes.push_back(chain.cond_out_gte);
    } else {
      cur_out_vals = emit_leaf_duses(cur_in_gtes);
    }

    if (chain.inner_while != nullptr) {
      HloInstruction* iw = chain.inner_while;
      HloInstruction* old_iw_root = iw->while_body()->root_instruction();
      iw->while_body()->set_root_instruction(
          BuildWidenedTuple(old_iw_root, chain.inner_slot, cur_out_vals),
          /*accept_different_shape=*/true);
      dead_tuples.push_back(old_iw_root);
      cur_out_vals = EmitCollectorGtes(
          iw, chain.inner_slot,
          iw->shape().tuple_shapes().size() - static_cast<int64_t>(m - 1),
          *chain.inner_out_gte, collector_shapes);
      dead_gtes.push_back(chain.inner_out_gte);
    }

    HloInstruction* old_outer_root = outer_body->root_instruction();
    outer_body->set_root_instruction(
        BuildWidenedTuple(old_outer_root, slot, cur_out_vals),
        /*accept_different_shape=*/true);
    dead_tuples.push_back(old_outer_root);

    // Replay V_pack once after outer_while batched along leading axis N.
    std::vector<HloInstruction*> exit_gtes = EmitCollectorGtes(
        outer_while, slot, outer_old_count, *post_gtes[0], collector_shapes);
    absl::flat_hash_map<const HloInstruction*, HloInstruction*> replay;
    for (size_t j = 0; j < m; ++j) {
      HloInstruction* val = exit_gtes[j];
      if (dag->leaves[j]->shape().dimensions().size() == 1) {
        const Shape squeezed = MakeRowPackShape(
            val->shape().element_type(),
            {num_rows, dag->leaves[j]->shape().dimensions(0)}, with_layout);
        val = preheader->AddInstruction(
            HloInstruction::CreateReshape(squeezed, val));
        val->set_metadata(post_gtes[0]->metadata());
      }
      replay.emplace(dag->leaves[j], val);
    }

    for (HloInstruction* inst : dag->topo_ops) {
      absl::InlinedVector<int64_t, 6> bdims = {num_rows};
      bdims.insert(bdims.end(), inst->shape().dimensions().begin(),
                   inst->shape().dimensions().end());
      const Shape out_shape =
          MakeRowPackShape(inst->shape().element_type(), bdims, with_layout);
      HloInstruction* batched = nullptr;
      switch (inst->opcode()) {
        case HloOpcode::kReshape:
        case HloOpcode::kBitcast:
        case HloOpcode::kCopy:
          batched = preheader->AddInstruction(HloInstruction::CreateReshape(
              out_shape, replay.at(inst->operand(0))));
          break;
        case HloOpcode::kBitcastConvert:
          batched =
              preheader->AddInstruction(HloInstruction::CreateBitcastConvert(
                  out_shape, replay.at(inst->operand(0))));
          break;
        case HloOpcode::kSlice: {
          std::vector<int64_t> starts = {0};
          std::vector<int64_t> limits = {num_rows};
          std::vector<int64_t> strides = {1};
          starts.insert(starts.end(), inst->slice_starts().begin(),
                        inst->slice_starts().end());
          limits.insert(limits.end(), inst->slice_limits().begin(),
                        inst->slice_limits().end());
          strides.insert(strides.end(), inst->slice_strides().begin(),
                         inst->slice_strides().end());
          batched = preheader->AddInstruction(HloInstruction::CreateSlice(
              out_shape, replay.at(inst->operand(0)), starts, limits, strides));
          break;
        }
        case HloOpcode::kConcatenate: {
          std::vector<HloInstruction*> ops;
          ops.reserve(inst->operand_count());
          for (const HloInstruction* op : inst->operands()) {
            ops.push_back(replay.at(op));
          }
          batched = preheader->AddInstruction(HloInstruction::CreateConcatenate(
              out_shape, ops, inst->concatenate_dimension() + 1));
          break;
        }
        case HloOpcode::kPad: {
          HloInstruction* pad_val =
              preheader->AddInstruction(inst->operand(1)->Clone());
          PaddingConfig pc;
          auto* d0 = pc.add_dimensions();
          d0->set_edge_padding_low(0);
          d0->set_edge_padding_high(0);
          d0->set_interior_padding(0);
          for (const auto& dim : inst->padding_config().dimensions()) {
            *pc.add_dimensions() = dim;
          }
          batched = preheader->AddInstruction(HloInstruction::CreatePad(
              out_shape, replay.at(inst->operand(0)), pad_val, pc));
          break;
        }
        case HloOpcode::kBroadcast: {
          auto it = replay.find(inst->operand(0));
          if (it != replay.end()) {
            std::vector<int64_t> dims = {0};
            for (int64_t d : inst->dimensions()) {
              dims.push_back(d + 1);
            }
            batched = preheader->AddInstruction(
                HloInstruction::CreateBroadcast(out_shape, it->second, dims));
          } else {
            HloInstruction* zc =
                preheader->AddInstruction(inst->operand(0)->Clone());
            batched = preheader->AddInstruction(
                HloInstruction::CreateBroadcast(out_shape, zc, {}));
          }
          break;
        }
        default:
          LOG(FATAL) << "Unexpected row-pack opcode: " << inst->opcode();
      }
      batched->set_metadata(inst->metadata());
      if (post_gtes[0]->has_sharding()) {
        batched->set_sharding(post_gtes[0]->sharding());
      }
      replay.emplace(inst, batched);
    }

    HloInstruction* restored =
        preheader->AddInstruction(HloInstruction::CreateReshape(
            carry_shape, replay.at(chain.dus->operand(1))));
    restored->set_metadata(post_gtes[0]->metadata());
    if (post_gtes[0]->has_sharding()) {
      restored->set_sharding(post_gtes[0]->sharding());
    }
    for (HloInstruction* pg : post_gtes) {
      if (pg->original_value() != nullptr) {
        restored->CopyOriginalValue(pg);
        break;
      }
    }
    for (HloInstruction* pg : post_gtes) {
      ABSL_RETURN_IF_ERROR(pg->ReplaceAllUsesWith(restored));
      if (preheader->root_instruction() == pg) {
        preheader->set_root_instruction(restored);
      }
      ABSL_RETURN_IF_ERROR(preheader->RemoveInstruction(pg));
    }

    absl::flat_hash_set<const HloInstruction*> pruned;
    auto prune_dead =
        [&](absl::Span<HloInstruction* const> insts) -> absl::Status {
      for (auto it = insts.rbegin(); it != insts.rend(); ++it) {
        HloInstruction* inst = *it;
        if (!pruned.contains(inst) && inst->IsDead()) {
          pruned.insert(inst);
          ABSL_RETURN_IF_ERROR(inst->parent()->RemoveInstruction(inst));
        }
      }
      return absl::OkStatus();
    };
    ABSL_RETURN_IF_ERROR(prune_dead(dead_tuples));
    ABSL_RETURN_IF_ERROR(prune_dead({chain.dus}));
    ABSL_RETURN_IF_ERROR(prune_dead(dag->topo_ops));
    ABSL_RETURN_IF_ERROR(prune_dead(dead_gtes));
    ABSL_RETURN_IF_ERROR(prune_dead({c_init}));

    *loop_growth_bytes += dag->net_growth_bytes;
    return true;
  }
  return false;
}

absl::StatusOr<bool> TryHoistRowPackFromWriteOnlyDusCarries(
    HloInstruction* while_op) {
  if (while_op->GetModule()->has_schedule()) {
    return false;
  }
  int64_t loop_growth_bytes = 0;
  bool any_changed = false;
  while (true) {
    ABSL_ASSIGN_OR_RETURN(
        bool step, TryHoistOneWriteOnlyDusCarry(while_op, &loop_growth_bytes));
    if (!step) {
      break;
    }
    any_changed = true;
  }
  return any_changed;
}

}  // namespace

// This function removes trivial compare hlo instructions inside the while body.
// Assuming a while loop with known trip count k, loop induction variable i with
// initial value c and constant step s (so i takes the values c, c + s, ...,
// c + (k - 1) * s), a compare(i,x) instruction with constant x is trivial if:
//   1) x <= c: (i < x) is false; and if x < c, (i > x) is true.
//   2) x >= c + (k - 1) * s: (i > x) is false; and if x > c + (k - 1) * s,
//      (i < x) is true.
static absl::StatusOr<bool> TryRemoveTrivialCompare(HloInstruction* while_op) {
  std::optional<int64_t> indvar_index = GetLoopInductionVarTupleIdx(while_op);
  if (indvar_index.has_value()) {
    if (while_op->operand(0)->operand(*indvar_index)->IsConstant()) {
      const HloConstantInstruction* init_value_hlo =
          Cast<HloConstantInstruction>(
              while_op->operand(0)->operand(*indvar_index));
      std::optional<int64_t> trip_count = MatchTrivialLoopTripCount(
          while_op, indvar_index.value(), init_value_hlo->literal());
      std::optional<int64_t> step =
          MatchTrivialLoopInductionStep(while_op, indvar_index.value());

      if (trip_count.has_value() && step.has_value() && *trip_count > 0) {
        std::optional<int64_t> init_value =
            LiteralUtil::LiteralAsScalarInt64(init_value_hlo->literal());
        // The last value of i is init + (trip_count - 1) * step. Give up if
        // that does not fit in an int64_t rather than fold with a wrong bound.
        std::optional<int64_t> last_value;
        if (init_value.has_value()) {
          auto [last_offset, overflow] =
              OverflowSafeMultiply(*trip_count - 1, *step);
          if (overflow) {
            return false;
          }
          last_value = OverflowSafeAdd(*init_value, last_offset);
          if (!last_value.has_value()) {
            return false;
          }
        }
        for (HloInstruction* body_instr :
             while_op->while_body()->instructions()) {
          HloInstruction* constant;
          if (Match(body_instr,
                    m::Compare(m::GetTupleElement(m::Parameter(),
                                                  indvar_index.value()),
                               m::Constant(&constant).IsConstantScalar()))) {
            std::optional<int64_t> constant_value =
                LiteralUtil::LiteralAsScalarInt64(constant->literal());
            if (constant_value.has_value() && init_value.has_value()) {
              const int64_t min_i = *init_value;
              const int64_t max_i = *last_value;
              const int64_t rhs = *constant_value;
              if (body_instr->comparison_direction() ==
                  ComparisonDirection::kLt) {
                if (min_i >= rhs) {
                  // i >= min_i >= rhs, so (i < rhs) is always false.
                  ABSL_RETURN_IF_ERROR(
                      while_op->while_body()->ReplaceInstruction(
                          body_instr, MakeScalarLike(body_instr, false)));
                  return true;
                }
                if (max_i < rhs) {
                  // i <= max_i < rhs, so (i < rhs) is always true.
                  ABSL_RETURN_IF_ERROR(
                      while_op->while_body()->ReplaceInstruction(
                          body_instr, MakeScalarLike(body_instr, true)));
                  return true;
                }
              } else if (body_instr->comparison_direction() ==
                         ComparisonDirection::kGt) {
                if (min_i > rhs) {
                  // i >= min_i > rhs, so (i > rhs) is always true.
                  ABSL_RETURN_IF_ERROR(
                      while_op->while_body()->ReplaceInstruction(
                          body_instr, MakeScalarLike(body_instr, true)));
                  return true;
                }
                if (max_i <= rhs) {
                  // i <= max_i <= rhs, so (i > rhs) is always false.
                  ABSL_RETURN_IF_ERROR(
                      while_op->while_body()->ReplaceInstruction(
                          body_instr, MakeScalarLike(body_instr, false)));
                  return true;
                }
              }
            }
          }
        }
      }
    }
  }
  return false;
}

// A helper function that copy the frontend attributes from the old while op to
// the new one.
void CopyFrontendAttributes(HloInstruction* old_while_op,
                            HloInstruction* new_while_op) {
  new_while_op->add_frontend_attributes(old_while_op->frontend_attributes());
}

// A helper function that copy the metadata from the old while op to
// the new one.
void CopyMetadata(HloInstruction* old_while_op, HloInstruction* new_while_op) {
  new_while_op->set_metadata(old_while_op->metadata());
}

// This is a utility function that removes the tuple indices whose bit is not
// set in used_tuple_indices from the while loop init, body, and condition. The
// final shape returned is still the same as before. If set index_for_replaced
// will replace any use of the removed indices in the final shape with a copy of
// the removed index.
static absl::StatusOr<HloInstruction*> RemoveDeadTupleIndices(
    HloInstruction* while_op, const InlinedBitSet<>& used_tuple_indices,
    std::optional<absl::flat_hash_map<int32_t, int32_t>>
        dead_to_surviving_index = std::nullopt) {
  HloModule* module = while_op->GetModule();
  HloComputation* computation = while_op->parent();
  HloInstruction* while_init = while_op->mutable_operand(0);
  HloComputation* while_cond = while_op->while_condition();
  HloComputation* while_body = while_op->while_body();
  HloInstruction* while_body_root = while_body->root_instruction();

  // Build up maps from the old/new to the new/old tuple indices.
  std::vector<int64_t> new_to_old_tuple_idx;
  const int64_t old_tuple_size = while_init->shape().tuple_shapes().size();
  for (int64_t old_idx = 0; old_idx < old_tuple_size; ++old_idx) {
    if (used_tuple_indices.Test(old_idx)) {
      new_to_old_tuple_idx.push_back(old_idx);
    }
  }

  auto print_no_metadata = HloPrintOptions().set_print_metadata(false);

  absl::flat_hash_map<int64_t, int64_t> old_to_new_tuple_idx;
  for (int64_t new_idx = 0; new_idx < new_to_old_tuple_idx.size(); ++new_idx) {
    int64_t old_idx = new_to_old_tuple_idx[new_idx];
    old_to_new_tuple_idx[old_idx] = new_idx;
    VLOG(2) << "Remapping tuple index " << old_idx << " to " << new_idx;
  }

  // Compute the shape of the while op after we remove the dead indices.
  std::vector<const Shape*> new_while_tuple_elem_shapes;
  new_while_tuple_elem_shapes.reserve(new_to_old_tuple_idx.size());
  for (int64_t old_idx : new_to_old_tuple_idx) {
    new_while_tuple_elem_shapes.push_back(
        &while_init->shape().tuple_shapes(old_idx));
  }
  Shape new_while_shape =
      ShapeUtil::MakeTupleShapeWithPtrs(new_while_tuple_elem_shapes);

  // Returns a map from elements in the computation to new instructions which
  // replace the old instructions after we remove unused elements from the while
  // tuple.
  auto make_while_computation_replacements = [&](const HloComputation* comp) {
    absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
        replacements;

    auto* param = comp->parameter_instruction(0);
    replacements.emplace(param, HloInstruction::CreateParameter(
                                    0, new_while_shape, param->name()));

    // Materialize param's users, since we're about to add new ones below.
    std::vector<HloInstruction*> materialized_users(param->users().begin(),
                                                    param->users().end());
    for (const auto* user : materialized_users) {
      // The while body root is handled separately.
      if (user == while_body_root) {
        continue;
      }
      CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement)
          << user->ToString(print_no_metadata);

      int64_t old_idx = user->tuple_index();
      auto new_idx_iter = old_to_new_tuple_idx.find(old_idx);
      if (new_idx_iter != old_to_new_tuple_idx.end()) {
        // This is a GTE of an index that survives.  Replace it.
        replacements.emplace(
            user, HloInstruction::CreateGetTupleElement(user->shape(), param,
                                                        new_idx_iter->second));
      } else {
        // This is a GTE of an index that we've removed.  Remove it from the
        // cloned computation.
        replacements.emplace(user, nullptr);
      }
    }
    // Remove instructions that depend on removed parameters.
    for (const auto* hlo : comp->MakeInstructionPostOrder()) {
      if (hlo == comp->root_instruction() || replacements.contains(hlo)) {
        continue;
      }
      for (const auto* operand : hlo->operands()) {
        auto op_it = replacements.find(operand);
        if (op_it != replacements.end() && op_it->second == nullptr) {
          replacements[hlo] = nullptr;
          break;
        }
      }
    }
    return replacements;
  };

  // Create the new while condition, body, and init value.
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      while_cond_replacements = make_while_computation_replacements(while_cond);
  std::unique_ptr<HloComputation> new_while_cond =
      while_cond->CloneWithReplacements(&while_cond_replacements);

  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      while_body_replacements = make_while_computation_replacements(while_body);
  std::vector<HloInstruction*> new_while_body_root_elems;
  new_while_body_root_elems.reserve(new_to_old_tuple_idx.size());
  for (int64_t old_idx : new_to_old_tuple_idx) {
    new_while_body_root_elems.push_back(
        while_body_root->mutable_operand(old_idx));
  }
  while_body_replacements.emplace(
      while_body_root, HloInstruction::CreateTuple(new_while_body_root_elems));
  std::unique_ptr<HloComputation> new_while_body =
      while_body->CloneWithReplacements(&while_body_replacements);

  std::vector<HloInstruction*> new_while_init_elems;
  new_while_init_elems.reserve(new_to_old_tuple_idx.size());
  for (int64_t old_idx : new_to_old_tuple_idx) {
    if (while_init->opcode() == HloOpcode::kTuple) {
      new_while_init_elems.push_back(while_init->mutable_operand(old_idx));
    } else {
      new_while_init_elems.push_back(
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              while_init->shape().tuple_shapes(old_idx), while_init, old_idx)));
    }
  }
  auto* new_while_init = computation->AddInstruction(
      HloInstruction::CreateTuple(new_while_init_elems));

  // Create the new while op.
  auto* new_while_op = computation->AddInstruction(HloInstruction::CreateWhile(
      new_while_shape,
      module->AddEmbeddedComputation(std::move(new_while_cond)),
      module->AddEmbeddedComputation(std::move(new_while_body)),
      new_while_init));
  new_while_op->CopyBackendConfigFrom(while_op);
  CopyFrontendAttributes(while_op, new_while_op);
  CopyMetadata(while_op, new_while_op);

  CopyOriginalValue(while_init, new_while_init, old_to_new_tuple_idx);
  CopyOriginalValue(while_op, new_while_op, old_to_new_tuple_idx);
  CopyOriginalValue(while_body->parameter_instruction(0),
                    new_while_op->while_body()->parameter_instruction(0),
                    old_to_new_tuple_idx);
  CopyOriginalValue(while_cond->parameter_instruction(0),
                    new_while_op->while_condition()->parameter_instruction(0),
                    old_to_new_tuple_idx);
  CopyOriginalValue(while_body_root,
                    new_while_op->while_body()->root_instruction(),
                    old_to_new_tuple_idx);

  // Create a tuple op that recreates the output of the old while op.  That is,
  // we transform to
  //
  //  new_while_init   while_init
  //       |              |
  //       V              |
  //   new_while          |
  //       |              |
  //       -------|   |----
  //              V   V
  //            new_tuple
  //                |
  //                V
  //    (orig. users of while op)
  //
  // The tuple simplifier will then simplify this if possible, removing
  // new_tuple and while_init.
  std::vector<HloInstruction*> new_tuple_elems;
  const int64_t tuple_size = ShapeUtil::TupleElementCount(while_init->shape());
  for (int64_t old_idx = 0; old_idx < tuple_size; ++old_idx) {
    auto new_tuple_idx_it = old_to_new_tuple_idx.find(old_idx);
    if (new_tuple_idx_it != old_to_new_tuple_idx.end() ||
        dead_to_surviving_index.has_value()) {
      int64_t gte_idx =
          new_tuple_idx_it != old_to_new_tuple_idx.end()
              ? new_tuple_idx_it->second
              : old_to_new_tuple_idx.find((*dead_to_surviving_index)[old_idx])
                    ->second;
      new_tuple_elems.push_back(
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              new_while_op->shape().tuple_shapes(gte_idx), new_while_op,
              gte_idx)));
    } else {
      new_tuple_elems.push_back(
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              while_init->shape().tuple_shapes(old_idx), while_init, old_idx)));
    }
  }
  HloInstruction* new_tuple =
      computation->AddInstruction(HloInstruction::CreateTuple(new_tuple_elems));
  ABSL_RETURN_IF_ERROR(computation->ReplaceInstruction(while_op, new_tuple));

  return new_while_op;
}

namespace {

bool AllWhileParamConsumersGte(const HloInstruction* while_op) {
  HloComputation* while_cond = while_op->while_condition();
  HloComputation* while_body = while_op->while_body();

  for (const HloInstruction* instr : {while_body->parameter_instruction(0),
                                      while_cond->parameter_instruction(0)}) {
    for (const HloInstruction* user : instr->users()) {
      if (user->opcode() != HloOpcode::kGetTupleElement) {
        VLOG(2) << "Cowardly refusing to analyze while loop with "
                << instr->ToString(HloPrintOptions().set_print_metadata(false))
                << " used by non-GTE instruction "
                << user->ToString(HloPrintOptions().set_print_metadata(false))
                << " in computation " << instr->parent()->name();
        return false;
      }
    }
  }
  return true;
}

// The tuple indices of a while loop that are read after the loop: all of them
// when the loop is the root of its computation or has a user other than
// get-tuple-element.
InlinedBitSet<> TupleIndicesUsedAfterLoop(const HloInstruction* while_op,
                                          int64_t tuple_size) {
  InlinedBitSet<> used(tuple_size);
  if (while_op == while_op->parent()->root_instruction()) {
    used.SetAll(tuple_size);
    return used;
  }
  for (const HloInstruction* user : while_op->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      used.SetAll(tuple_size);
      return used;
    }
    used.Set(user->tuple_index());
  }
  return used;
}

// Whether the body passes the tuple element at 'index' of its parameter
// through unmodified.
bool IsPassedThroughUnmodified(const HloComputation* while_body,
                               int64_t index) {
  const HloInstruction* output = while_body->root_instruction()->operand(index);
  return output->opcode() == HloOpcode::kGetTupleElement &&
         output->operand(0) == while_body->parameter_instruction(0) &&
         output->tuple_index() == index;
}

// How a while loop's body and condition depend on the loop's candidate tuple
// indices, the indices TryRemoveDeadWhileParams may remove: candidate_bit
// numbers them and holds -1 for every other index. Per instruction, the
// candidates it depends on within one iteration; across instructions, the
// connected components of the operand graph.
//
// The instructions of the body and the condition are addressed by slot, the
// body's local ids first, then the condition's. The slots map the instructions
// to a dense index range, so the per instruction state lives in flat arrays
// instead of hash maps keyed by instruction. Local ids stay stable during the
// analysis: nothing mutates the graph before RemoveDeadTupleIndices.
class WhileInputDependencies {
 public:
  WhileInputDependencies(const HloInstruction* while_op,
                         absl::Span<const int> candidate_bit,
                         int num_candidates)
      : while_body_(while_op->while_body()),
        while_cond_(while_op->while_condition()),
        while_body_root_(while_body_->root_instruction()),
        candidate_bit_(candidate_bit),
        body_slots_(while_body_->next_unique_instruction_internal_id()),
        num_slots_(body_slots_ +
                   while_cond_->next_unique_instruction_internal_id()),
        side_effecting_row_(num_slots_),
        affecting_others_row_(num_slots_ + 1),
        input_deps_(num_slots_ + 2, num_candidates),
        sets_(num_slots_) {
    PropagateThrough(while_body_);
    PropagateThrough(while_cond_);
    AccumulateOutputDependencies();
  }

  int64_t tuple_size() const { return candidate_bit_.size(); }

  // Whether the index may be removed at all.
  bool IsCandidate(int64_t index) const { return candidate_bit_[index] >= 0; }

  // Whether a side effecting instruction or the loop condition depends on the
  // candidate index.
  bool ReachesSink(int64_t index) const {
    DCHECK(IsCandidate(index));
    return input_deps_.Test(side_effecting_row_, candidate_bit_[index]);
  }

  // Whether a body output other than its own depends on the candidate index.
  bool AffectsOtherOutputs(int64_t index) const {
    DCHECK(IsCandidate(index));
    return input_deps_.Test(affecting_others_row_, candidate_bit_[index]);
  }

  // The connected component of the body output at the index, numbered below
  // num_slots(). The get-tuple-elements of the index belong to the same
  // component, so indices that depend on each other share it.
  int ComponentOf(int64_t index) {
    return sets_.Find(SlotOf(while_body_root_->operand(index)));
  }

  int num_slots() const { return num_slots_; }

 private:
  int SlotOf(const HloInstruction* inst) const {
    DCHECK(inst->parent() == while_body_ || inst->parent() == while_cond_);
    return (inst->parent() == while_body_ ? 0 : body_slots_) + inst->local_id();
  }

  // The parameters and the body root take no part in the dependencies.
  bool IsTracked(const HloInstruction* inst) const {
    return inst != while_body_->parameter_instruction(0) &&
           inst != while_cond_->parameter_instruction(0) &&
           inst != while_body_root_;
  }

  // One post order pass over a computation. Operands come before their users,
  // so the dependencies of an instruction are the union of its operands' by
  // the time it is visited. A get-tuple-element of the parameter seeds its own
  // bit and joins the component of the output it feeds back into. A side
  // effecting instruction and the condition root are sinks: the inputs they
  // depend on must stay.
  void PropagateThrough(const HloComputation* comp) {
    const HloInstruction* while_input = comp->parameter_instruction(0);
    comp->ForEachInstructionPostOrder([&](HloInstruction* inst) {
      if (!IsTracked(inst)) {
        return;
      }
      const int slot = SlotOf(inst);
      if (inst->opcode() == HloOpcode::kGetTupleElement &&
          inst->operand(0) == while_input) {
        if (candidate_bit_[inst->tuple_index()] >= 0) {
          input_deps_.Set(slot, candidate_bit_[inst->tuple_index()]);
        }
        const HloInstruction* output =
            while_body_root_->operand(inst->tuple_index());
        if (output != inst) {
          sets_.Merge(SlotOf(output), slot);
        }
      } else {
        for (const HloInstruction* operand : inst->operands()) {
          if (!IsTracked(operand)) {
            continue;
          }
          sets_.Merge(SlotOf(operand), slot);
          if (const std::optional<int64_t> index =
                  ParameterIndex(operand, while_input)) {
            // The row of a parameter get-tuple-element holds only its own
            // bit, so one Set replaces the OR of a whole row.
            if (candidate_bit_[*index] >= 0) {
              input_deps_.Set(slot, candidate_bit_[*index]);
            }
          } else {
            input_deps_.Or(slot, SlotOf(operand));
          }
        }
      }
      if (inst->HasSideEffect() || inst == while_cond_->root_instruction()) {
        input_deps_.Or(side_effecting_row_, slot);
      }
    });
  }

  // The tuple index when inst is a get-tuple-element of the given parameter.
  static std::optional<int64_t> ParameterIndex(const HloInstruction* inst,
                                               const HloInstruction* param) {
    if (inst->opcode() == HloOpcode::kGetTupleElement &&
        inst->operand(0) == param) {
      return inst->tuple_index();
    }
    return std::nullopt;
  }

  // Collects the inputs that reach some body output other than their own; a
  // self update does not count.
  void AccumulateOutputDependencies() {
    const HloInstruction* while_input = while_body_->parameter_instruction(0);
    for (int64_t i = 0; i < while_body_root_->operand_count(); ++i) {
      const HloInstruction* output = while_body_root_->operand(i);
      if (const std::optional<int64_t> index =
              ParameterIndex(output, while_input)) {
        // The output copies input *index; the row holds only that bit, and a
        // pass through (*index == i) is a self update.
        if (*index != i && candidate_bit_[*index] >= 0) {
          input_deps_.Set(affecting_others_row_, candidate_bit_[*index]);
        }
        continue;
      }
      const int output_slot = SlotOf(output);
      if (candidate_bit_[i] >= 0) {
        input_deps_.OrIgnoringBit(affecting_others_row_, output_slot,
                                  candidate_bit_[i]);
      } else {
        input_deps_.Or(affecting_others_row_, output_slot);
      }
    }
  }

  const HloComputation* const while_body_;
  const HloComputation* const while_cond_;
  const HloInstruction* const while_body_root_;
  const absl::Span<const int> candidate_bit_;
  const int body_slots_;
  const int num_slots_;
  // Two extra rows of input_deps_: the inputs that reach a side effect or the
  // condition, and the inputs that reach another output.
  const int side_effecting_row_;
  const int affecting_others_row_;
  DenseBitSets input_deps_;
  DenseUnionFind sets_;
};

// Clears index i of the surviving indices; returns 1 when it was still set.
int64_t EraseIndex(int64_t i, InlinedBitSet<>& used_tuple_indices) {
  if (!used_tuple_indices.Test(i)) {
    return 0;
  }
  used_tuple_indices.Clear(i);
  return 1;
}

// Case 1: an index that is unused after the loop can go on its own when no
// other output, side effect or condition depends on its input. Returns the
// number of indices newly removed from used_tuple_indices.
int64_t EraseIndicesDeadOnTheirOwn(const InlinedBitSet<>& used_after_loop,
                                   const WhileInputDependencies& deps,
                                   InlinedBitSet<>& used_tuple_indices) {
  int64_t num_erased = 0;
  for (int64_t i = 0; i < deps.tuple_size(); ++i) {
    if (!used_after_loop.Test(i) && !deps.AffectsOtherOutputs(i) &&
        !deps.ReachesSink(i)) {
      VLOG(2) << "Remove with dependencies " << i;
      num_erased += EraseIndex(i, used_tuple_indices);
    }
  }
  return num_erased;
}

// Case 2: a connected group of indices can go when none of its indices blocks
// it. An index blocks its group when it is not a candidate, that is used after
// the loop and not passed through, or when a side effect or the condition
// depends on it. Returns the number of indices newly removed from
// used_tuple_indices.
int64_t EraseIndicesDeadAsGroup(WhileInputDependencies& deps,
                                InlinedBitSet<>& used_tuple_indices) {
  InlinedBitSet<> blocked_components(deps.num_slots());
  for (int64_t i = 0; i < deps.tuple_size(); ++i) {
    if (!deps.IsCandidate(i) || deps.ReachesSink(i)) {
      blocked_components.Set(deps.ComponentOf(i));
    }
  }
  int64_t num_erased = 0;
  for (int64_t i = 0; i < deps.tuple_size(); ++i) {
    if (!blocked_components.Test(deps.ComponentOf(i))) {
      VLOG(2) << "Remove with groups: index " << i;
      num_erased += EraseIndex(i, used_tuple_indices);
    }
  }
  return num_erased;
}

}  // namespace

absl::StatusOr<bool> TryRemoveDeadWhileParams(HloInstruction* while_op) {
  CHECK_EQ(while_op->opcode(), HloOpcode::kWhile);
  if (frontend::HasDisableWhileLoopDceAttr(while_op)) {
    return false;
  }

  // Don't try this transformation if the while loop isn't removable, since if
  // it succeeds ultimately we're going to have to replace the old while loop
  // with a new one.
  if (!while_op->parent()->IsSafelyRemovable(while_op)) {
    VLOG(2) << "Can't remove dead parameters from non-removable while op.";
    return false;
  }

  HloInstruction* while_init = while_op->mutable_operand(0);
  HloComputation* while_body = while_op->while_body();
  HloInstruction* while_body_root = while_body->root_instruction();

  if (!while_init->shape().IsTuple()) {
    VLOG(2) << "While op's carried value isn't tuple shaped.";
    return false;
  }

  if (while_body_root->opcode() != HloOpcode::kTuple) {
    VLOG(2) << "While body's root is not a tuple(...) instruction.";
    return false;
  }

  const int64_t tuple_size = ShapeUtil::TupleElementCount(while_init->shape());
  auto print_no_metadata = HloPrintOptions().set_print_metadata(false);

  // Bail if param0 of while_cond or while_body has users which aren't of type
  // get-tuple-element.
  if (!AllWhileParamConsumersGte(while_op)) {
    return false;
  }

  if (tuple_size == 0) {
    VLOG(2) << "Can't remove elements from while loop's tuple -- it's already "
               "empty.";
    return false;
  }
  const InlinedBitSet<> used_after_loop =
      TupleIndicesUsedAfterLoop(while_op, tuple_size);

  // We identify dead loop parameters in two cases (provided they do not cause
  // side effects or affect the loop condition):
  // 1) There is no use after the loop, and the input does not affect other
  // outputs (though it may update itself).
  // 2) A group of mutually dependent elements whose outputs are either unused
  // after the loop or are passed through unmodified. We use a union find data
  // structure to approximate connected components (which has false negatives
  // for directed dependencies).

  // Only an index that is unused after the loop or passed through unmodified
  // can be removed, in either case. The dependency analysis tracks just those
  // candidate indices, numbered by candidate_bit; every other index keeps -1
  // there.
  std::vector<int> candidate_bit(tuple_size, -1);
  int num_candidates = 0;
  for (int64_t i = 0; i < tuple_size; ++i) {
    if (!used_after_loop.Test(i) || IsPassedThroughUnmodified(while_body, i)) {
      candidate_bit[i] = num_candidates++;
    }
  }
  if (num_candidates == 0) {
    VLOG(2) << "Loop " << while_op->ToString(print_no_metadata)
            << " has no removable candidate index.";
    return false;
  }
  WhileInputDependencies deps(while_op, candidate_bit, num_candidates);
  // The indices that survive: all of them until a case removes some.
  InlinedBitSet<> used_tuple_indices(tuple_size);
  used_tuple_indices.SetAll(tuple_size);
  const int64_t num_removed =
      EraseIndicesDeadOnTheirOwn(used_after_loop, deps, used_tuple_indices) +
      EraseIndicesDeadAsGroup(deps, used_tuple_indices);

  if (num_removed == 0) {
    VLOG(2) << "Loop " << while_op->ToString(print_no_metadata)
            << " uses all of its inputs; no simplification possible.";
    return false;
  }

  VLOG(1) << "Eliminating " << num_removed << " elements from tuple of "
          << while_op->ToString(print_no_metadata);

  ABSL_ASSIGN_OR_RETURN(while_op,
                        RemoveDeadTupleIndices(while_op, used_tuple_indices));

  return true;
}

// Returns if this instruction looks like an insertion inside a variable of a
// while loop.
static bool IsDynamicUpdateSliceWhileInsertion(
    const HloInstruction* instr, const HloComputation* while_body) {
  return instr->opcode() == HloOpcode::kDynamicUpdateSlice &&
         instr->operand(0)->opcode() == HloOpcode::kGetTupleElement &&
         instr->operand(0)->operand(0) == while_body->parameter_instruction(0);
}

namespace {
template <typename KeyToIndicesMap>
absl::StatusOr<HloInstruction*> RemoveRepeatedWhileTupleIndices(
    HloInstruction* while_op, const KeyToIndicesMap& key_to_indices,
    bool replace_with_init) {
  // We want to do two things here:
  // a) Make sure that all of the Param -> GTE(i) -> ROOT Tuple chains where the
  // indices belong to the same equivalence set use the same index.
  // b) Remove the newly dead indices.

  // First, find all the indices that belong to a non-trivial (size >= 2) set,
  // and decide, for each set, which index to keep, and which will be removed.
  absl::flat_hash_map<int32_t, int32_t> dead_to_surviving_index;
  for (const auto& [key, indices] : key_to_indices) {
    if (indices.size() < 2) {
      continue;
    }
    int32_t surviving_index = indices[0];
    for (int i = 1; i < indices.size(); ++i) {
      int32_t dead_index = indices[i];
      dead_to_surviving_index[dead_index] = surviving_index;
    }
  }

  // Return early if we didn't find any non-trivial sets.
  if (dead_to_surviving_index.empty()) {
    return while_op;
  }

  // For each set, replace GTEs for the indices being removed with one common
  // GTE for the surviving index.
  HloComputation* while_cond = while_op->while_condition();
  HloComputation* while_body = while_op->while_body();
  for (HloComputation* comp : {while_body, while_cond}) {
    absl::flat_hash_map<int32_t, HloInstruction*> surviving_indices_to_gte;
    HloInstruction* param = comp->parameter_instruction(0);
    // Make a copy of the input's users, since we're going to be mutating it in
    // the loop.
    std::vector<HloInstruction*> all_gtes(param->users().begin(),
                                          param->users().end());
    for (HloInstruction* gte : all_gtes) {
      int gte_index = gte->tuple_index();
      auto it = dead_to_surviving_index.find(gte_index);
      if (it != dead_to_surviving_index.end()) {
        int32_t surviving_index = it->second;
        // We could track down the original GTE for the surviving index, but
        // it's not worth it, CSE will clean this up.
        HloInstruction* surviving_gte;
        auto it = surviving_indices_to_gte.find(surviving_index);
        if (it == surviving_indices_to_gte.end()) {
          surviving_gte =
              comp->AddInstruction(HloInstruction::CreateGetTupleElement(
                  param->shape().tuple_shapes(surviving_index),
                  comp->parameter_instruction(0), surviving_index));
          surviving_indices_to_gte[surviving_index] = surviving_gte;
        } else {
          surviving_gte = it->second;
        }
        ABSL_RETURN_IF_ERROR(comp->ReplaceInstruction(gte, surviving_gte));
      }
    }
  }

  // And finally, remove the dead indices.
  const int64_t tuple_size =
      while_body->root_instruction()->shape().tuple_shapes().size();
  InlinedBitSet<> used_tuple_indices(tuple_size);
  for (int64_t index = 0; index < tuple_size; ++index) {
    if (!dead_to_surviving_index.contains(index)) {
      used_tuple_indices.Set(index);
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      while_op,
      RemoveDeadTupleIndices(
          while_op, used_tuple_indices,
          replace_with_init ? std::nullopt
                            : std::make_optional(dead_to_surviving_index)));

  return while_op;
}

struct RepeatedWhileTupleIndicesKey {
  const HloInstruction* init_value;
  const HloInstruction* dynamic_update_slice_value;
  std::vector<const HloInstruction*> dynamic_update_slice_indices;
};

bool operator==(const RepeatedWhileTupleIndicesKey& lhs,
                const RepeatedWhileTupleIndicesKey& rhs) {
  return lhs.init_value == rhs.init_value &&
         lhs.dynamic_update_slice_value == rhs.dynamic_update_slice_value &&
         absl::c_equal(lhs.dynamic_update_slice_indices,
                       rhs.dynamic_update_slice_indices);
}

template <typename H>
H AbslHashValue(H h, const RepeatedWhileTupleIndicesKey& key) {
  return H::combine(std::move(h), key.init_value,
                    key.dynamic_update_slice_value,
                    key.dynamic_update_slice_indices);
}

const HloPassFactory* const kWhileLoopSimplifierRegistration =
    RegisterHloPassFactory<DefaultHloPassFactory<WhileLoopSimplifier>>(
        "simplify-while-loops");

}  // namespace

// If the while loop init passes the same values to several tuple indices, and
// if the body keeps on passing them through, we can remove the duplicates.
static absl::StatusOr<bool> TryRemoveRepeatedWhileTupleIndices(
    HloInstruction* while_op) {
  CHECK_EQ(while_op->opcode(), HloOpcode::kWhile);
  if (frontend::HasDisableWhileLoopDceAttr(while_op)) {
    return false;
  }

  // Don't try this transformation if the while loop isn't removable, since if

  // it succeeds ultimately we're going to have to replace the old while loop
  // with a new one.
  if (!while_op->parent()->IsSafelyRemovable(while_op)) {
    VLOG(2) << "Can't remove dead parameters from non-removable while op.";
    return false;
  }

  HloInstruction* while_init = while_op->mutable_operand(0);
  HloComputation* while_body = while_op->while_body();
  HloInstruction* while_body_root = while_body->root_instruction();

  if (!while_init->shape().IsTuple() ||
      while_init->opcode() != HloOpcode::kTuple) {
    VLOG(2) << "While op's carried value isn't tuple shaped.";
    return false;
  }

  if (while_body_root->opcode() != HloOpcode::kTuple) {
    VLOG(2) << "While body's root is not a tuple(...) instruction.";
    return false;
  }

  if (!AllWhileParamConsumersGte(while_op)) {
    VLOG(2) << "While body or condition parameter consumers are not all "
               "get-tuple-element instructions.";
    return false;
  }

  absl::flat_hash_map<const HloInstruction*, std::vector<int32_t>>
      init_to_indices;
  absl::flat_hash_map<RepeatedWhileTupleIndicesKey, std::vector<int32_t>>
      dus_key_to_indices;
  const auto gte_at_index = [](const HloInstruction* maybe_gte, int32_t index) {
    return maybe_gte->opcode() == HloOpcode::kGetTupleElement &&
           maybe_gte->operand(0) ==
               maybe_gte->parent()->parameter_instruction(0) &&
           maybe_gte->tuple_index() == index;
  };

  // For each index in the input tuple, find all the indices where the only use
  // in the while body is either `Param -> GTE(index) -> ROOT Tuple`, or the
  // dynamic-update-slice equivalent, and split such indices into equivalence
  // sets based on the input value into the while.
  for (int index = 0; index < while_init->shape().tuple_shapes().size();
       ++index) {
    const HloInstruction* body_elem = while_body_root->operand(index);
    const HloInstruction* init_elem = while_init->operand(index);
    if (gte_at_index(body_elem, index)) {
      init_to_indices[init_elem].push_back(index);
    }
  }

  // Only keep one index for each equivalence set.
  HloInstruction* original_while_op = while_op;
  ABSL_ASSIGN_OR_RETURN(
      while_op, RemoveRepeatedWhileTupleIndices(while_op, init_to_indices,
                                                /*replace_with_init=*/true));

  // In theory, we could handle the "simple" case and the "dynamic-update-slice"
  // case in one go, but it's probably not worth the added complexity, so do it
  // separately.
  while_init = while_op->mutable_operand(0);
  while_body = while_op->while_body();
  while_body_root = while_body->root_instruction();
  for (int index = 0; index < while_init->shape().tuple_shapes().size();
       ++index) {
    const HloInstruction* body_elem = while_body_root->operand(index);
    const HloInstruction* init_elem = while_init->operand(index);
    if (body_elem->opcode() == HloOpcode::kDynamicUpdateSlice &&
        gte_at_index(body_elem->operand(0), index)) {
      const HloDynamicIndexInstruction* dus =
          Cast<HloDynamicIndexInstruction>(body_elem);
      dus_key_to_indices[RepeatedWhileTupleIndicesKey{
                             init_elem, body_elem->operand(1),
                             std::vector<const HloInstruction*>(
                                 dus->index_operands().begin(),
                                 dus->index_operands().end())}]
          .push_back(index);
    }
  }
  ABSL_ASSIGN_OR_RETURN(
      while_op, RemoveRepeatedWhileTupleIndices(while_op, dus_key_to_indices,
                                                /*replace_with_init=*/false));

  return while_op != original_while_op;
}

// Removes each loop parameter (i.e. member of the while loop tuple) that is a
// constant and is the same in the while loop body and the while loop init.
static absl::StatusOr<bool> TryRemoveConstantParams(HloInstruction* while_op) {
  if (frontend::HasDisableWhileLoopDceAttr(while_op)) {
    return false;
  }
  HloModule* module = while_op->GetModule();

  HloComputation* computation = while_op->parent();
  auto* while_init = while_op->mutable_operand(0);
  auto* while_body = while_op->while_body();
  auto* while_cond = while_op->while_condition();
  auto* while_body_root = while_body->root_instruction();
  if (while_init->opcode() != HloOpcode::kTuple ||
      while_body_root->opcode() != HloOpcode::kTuple) {
    return false;
  }

  TF_RET_CHECK(while_cond->num_parameters() == 1);
  TF_RET_CHECK(while_body->num_parameters() == 1);
  TF_RET_CHECK(
      ShapeUtil::Compatible(while_init->shape(), while_body_root->shape()));

  absl::flat_hash_set<int64_t> constant_tuple_indices;
  const auto& while_shape = while_init->shape();
  for (int i = 0; i < while_shape.tuple_shapes().size(); ++i) {
    auto* init_elem = while_init->operand(i);
    auto* body_elem = while_body_root->operand(i);
    if (init_elem->opcode() == HloOpcode::kConstant &&
        body_elem->opcode() == HloOpcode::kConstant &&
        init_elem->literal() == body_elem->literal()) {
      constant_tuple_indices.insert(i);
    }
  }

  if (constant_tuple_indices.empty()) {
    return false;
  }

  // OK, we found some constant elements of the while parameter!  Eliminate
  // them.
  std::vector<const Shape*> new_while_shape_elems;
  for (int i = 0; i < while_shape.tuple_shapes().size(); ++i) {
    if (!constant_tuple_indices.count(i)) {
      new_while_shape_elems.push_back(&while_shape.tuple_shapes(i));
    }
  }
  Shape new_while_shape =
      ShapeUtil::MakeTupleShapeWithPtrs(new_while_shape_elems);

  // `new_instrs` holds instructions created outside of a computation for
  // cloning.  Elements added here just need to live until the end of the
  // relevant CloneWithReplacement call.
  std::vector<std::unique_ptr<HloInstruction>> new_instrs;
  auto add_new_instr = [&](std::unique_ptr<HloInstruction> instr) {
    new_instrs.push_back(std::move(instr));
    return new_instrs.back().get();
  };

  // Returns a new tuple without the elements of constant_tuple_indices.
  auto remove_constant_elems =
      [&](HloInstruction* instr) -> std::unique_ptr<HloInstruction> {
    CHECK(ShapeUtil::Compatible(instr->shape(), while_shape));

    std::vector<HloInstruction*> tuple_elems;
    for (int i = 0; i < while_shape.tuple_shapes().size(); ++i) {
      if (!constant_tuple_indices.count(i)) {
        tuple_elems.push_back(
            add_new_instr(HloInstruction::CreateGetTupleElement(
                while_shape.tuple_shapes(i), instr, i)));
      }
    }
    std::unique_ptr<HloInstruction> new_tuple =
        HloInstruction::CreateTuple(tuple_elems);
    if (instr->original_value()) {
      auto new_ov = std::make_shared<OriginalValue>(new_tuple->shape());
      int64_t new_i = 0;
      for (int i = 0; i < while_shape.tuple_shapes().size(); ++i) {
        if (!constant_tuple_indices.count(i)) {
          CHECK_OK(new_ov->mutable_tree()->CopyCompatibleSubtreeFrom(
              instr->original_value()->tree(), {i}, {new_i++}));
        }
      }
      new_tuple->set_original_value(new_ov);
    }
    return new_tuple;
  };

  auto add_constant_elems =
      [&](HloInstruction* instr,
          const HloInstruction* old_instr =
              nullptr) -> std::unique_ptr<HloInstruction> {
    CHECK(ShapeUtil::Compatible(instr->shape(), new_while_shape));

    std::vector<HloInstruction*> tuple_elems;
    int64_t j = 0;
    for (int i = 0; i < while_shape.tuple_shapes().size(); ++i) {
      if (constant_tuple_indices.count(i)) {
        tuple_elems.push_back(while_init->mutable_operand(i));
      } else {
        tuple_elems.push_back(
            add_new_instr(HloInstruction::CreateGetTupleElement(
                while_shape.tuple_shapes(i), instr, j)));
        ++j;
      }
    }
    auto tuple = HloInstruction::CreateTuple(tuple_elems);
    if (old_instr != nullptr) {
      tuple->CopyOriginalValue(old_instr);
    }
    return tuple;
  };

  // Special case: constant_tuple_indices covers the whole while parameter, so
  // the new while shape is the empty tuple.  In this case, the value of the
  // while loop is simply equal to the value of `init`.
  //
  // It's unfortunate to special-case this, but it's simpler than the
  // alternative.  The problem is that if our while parameter has no
  // non-constant elems, the tuple returned by `add_constant_elems` won't depend
  // on instr (the loop body/cond parameter), and therefore
  // CloneWithReplacementPairs will *leave the parameter out entirely*, creating
  // invalid HLO.
  if (ShapeUtil::IsEmptyTuple(new_while_shape)) {
    ABSL_RETURN_IF_ERROR(computation->ReplaceInstruction(while_op, while_init));
    return true;
  }

  std::unique_ptr<HloComputation> new_while_cond =
      while_cond->CloneWithReplacementPairs({
          while_cond->parameter_instruction(0),
          add_constant_elems(add_new_instr(HloInstruction::CreateParameter(
                                 0, new_while_shape,
                                 while_cond->parameter_instruction(0)->name())),
                             while_cond->parameter_instruction(0)),
      });

  std::unique_ptr<HloComputation> new_while_body =
      while_body->CloneWithReplacementPairs(
          {
              while_body->parameter_instruction(0),
              add_constant_elems(
                  add_new_instr(HloInstruction::CreateParameter(
                      0, new_while_shape,
                      while_cond->parameter_instruction(0)->name())),
                  while_body->parameter_instruction(0)),
          },
          {
              while_body->root_instruction(),
              remove_constant_elems(
                  add_new_instr(while_body->root_instruction()->Clone())),
          });

  // Create the final while loop, and add any new instructions created to
  // `computation`.
  new_instrs.clear();
  auto* new_while_op = computation->AddInstruction(HloInstruction::CreateWhile(
      new_while_shape,
      module->AddEmbeddedComputation(std::move(new_while_cond)),
      module->AddEmbeddedComputation(std::move(new_while_body)),
      add_new_instr(remove_constant_elems(while_init))));
  absl::flat_hash_map<int64_t, int64_t> old_to_new_tuple_idx;
  for (int64_t i = 0, new_i = 0; i < while_shape.tuple_shapes().size(); ++i) {
    if (!constant_tuple_indices.count(i)) {
      old_to_new_tuple_idx[i] = new_i++;
    }
  }
  CopyOriginalValue(while_init, new_while_op->mutable_operand(0),
                    old_to_new_tuple_idx);
  CopyOriginalValue(while_op, new_while_op, old_to_new_tuple_idx);
  CopyOriginalValue(while_body->parameter_instruction(0),
                    new_while_op->while_body()->parameter_instruction(0),
                    old_to_new_tuple_idx);
  CopyOriginalValue(while_cond->parameter_instruction(0),
                    new_while_op->while_condition()->parameter_instruction(0),
                    old_to_new_tuple_idx);
  CopyOriginalValue(while_body->root_instruction(),
                    new_while_op->while_body()->root_instruction(),
                    old_to_new_tuple_idx);
  new_while_op->CopyBackendConfigFrom(while_op);
  CopyFrontendAttributes(while_op, new_while_op);
  CopyMetadata(while_op, new_while_op);
  ABSL_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
      while_op, add_constant_elems(new_while_op, while_op)));
  for (auto& instr : new_instrs) {
    computation->AddInstruction(std::move(instr));
  }
  return true;
}

// Tries to remove a while loop from the graph.
//
//  - Loops with trip count of 0 can be replaced by the loop's "init" value.
//  - Loops with trip count of 1 can be replaced by the loop's body, with the
//    loop itself removed.
//
// Returns true if it made a change to the graph.
static absl::StatusOr<bool> TryRemoveWhileLoop(HloInstruction* while_op) {
  if (frontend::HasDisableWhileLoopDceAttr(while_op)) {
    return false;
  }
  // Cowardly refuse to remove loops that are not removable.  In practice, this
  // means that we can't remove loops that have control predecessors/successors.
  if (!while_op->parent()->IsSafelyRemovable(while_op)) {
    VLOG(2) << "Not attempting to remove while loop that is not removable: "
            << while_op->ToShortString();
    return false;
  }

  // Refuse to remove while loops with a condition that contain side-effects,
  // because removing a while loop is tantamount to removing its condition.
  //
  // TODO(jlebar): This is conservative: We could instead just run the while
  // condition once (trip-count == 0) or twice (trip-count == 1).
  if (while_op->while_condition()->HasSideEffect()) {
    VLOG(2) << "Not attempting to remove while loop whose condition contains "
               "side-effecting instructions: "
            << while_op->ToShortString();
    return false;
  }

  // Remove while loops with static trip count of 0.
  optional<int64_t> trip_count =
      ComputeWhileLoopTripCount(while_op, /*max_brute_force_iters=*/1);
  if (trip_count && *trip_count == 0) {
    // The loop never executes, so the value of the loop is the value of its
    // "init" operand.
    auto computation = while_op->parent();

    // Remove while_op (i.e., call ReplaceInstruction rather than
    // ReplaceUsesWithInstruction) so that if the algebraic simplifier is run in
    // a loop without an intervening DCE, we don't try to re-remove the loop.
    ABSL_RETURN_IF_ERROR(computation->ReplaceInstruction(
        while_op, while_op->mutable_operand(0)));
    return true;
  }

  // Transform while loops with static trip count of 1 into a call op, then
  // inline the call.
  bool skip_trip_count_one_simplification =
      while_op->get_frontend_attribute(
          "skip-simplify-while-loops_trip-count-one") == "true";
  if (trip_count && *trip_count == 1 && !skip_trip_count_one_simplification &&
      CallInliner::InlineInstructionAllowed(while_op)) {
    // Do not simplify the loop away when there is a side-effectful op,
    // otherwise the infeed op may not inherit the data dependency from
    // the while loop.
    //
    // Example: while_body (param_a) {
    //   param_a = parameter(0)
    //   infeed2 = infeed()
    // }
    //
    // infeed1 = ...
    // while = while(infeed1), body=while_body // infeed2 has implicit
    // dependency on infeed1.
    //
    // After simplification:
    //
    // infeed1 = ...
    // infeed2 = infeed() // no dependency between infeed1 and infeed2. infeed1
    //                    // can be scheduled after infeed2.
    //
    bool has_side_effects = absl::c_any_of(
        while_op->called_computations(), [](const HloComputation* computation) {
          return computation->HasSideEffect();
        });
    if (!has_side_effects) {
      auto computation = while_op->parent();
      auto call_op = computation->AddInstruction(HloInstruction::CreateCall(
          while_op->shape(), while_op->operands(), while_op->while_body()));
      call_op->set_original_value(while_op->original_value());
      ABSL_RETURN_IF_ERROR(computation->ReplaceInstruction(while_op, call_op));
      call_op->set_metadata_op_name("");
      ABSL_ASSIGN_OR_RETURN(auto inlined_instructions_map,
                            CallInliner::Inline(call_op));
      (void)inlined_instructions_map;
      return true;
    } else {
      VLOG(2) << "Not attempting to simplify while loop because it contains a "
                 "side-effecting node: "
              << while_op->ToShortString();
    }
  }
  return false;
}

static absl::StatusOr<bool> TryPropagateConstant(HloInstruction* while_op) {
  auto while_init = while_op->operand(0);
  if (while_init->opcode() != HloOpcode::kTuple) {
    return false;
  }

  auto while_body = while_op->while_body();
  auto while_body_root = while_body->root_instruction();
  if (while_body_root->opcode() != HloOpcode::kTuple) {
    return false;
  }

  auto while_body_param = while_body->parameter_instruction(0);
  const HloInstruction::InstructionVector& root_operands =
      while_body_root->operands();

  // Find the loop invariant tuple elements with scalar constant init value and
  // build a map from the tuple element index to the constant value. Limit this
  // to scalar constant values because propagating array constants can regress
  // performance by forcing us to copy constants.
  absl::flat_hash_map<int, const HloInstruction*> index_to_constant;
  for (int i = 0; i < root_operands.size(); i++) {
    const HloInstruction* init_tuple_elem = nullptr;
    if (Match(root_operands[i],
              m::GetTupleElement(m::Op().Is(while_body_param), i)
                  .WithShape(m::Shape().IsScalar())) &&
        Match(while_init->operand(i), m::Constant(&init_tuple_elem))) {
      VLOG(3) << "Found loop invariant tuple element " << i << " "
              << init_tuple_elem->ToString();
      index_to_constant[i] = init_tuple_elem;
    }
  }

  if (index_to_constant.empty()) {
    return false;
  }

  // Replace the use of each constant tuple element in the loop_condition and
  // loop_body with the corresponding constant value.
  auto propagate_constant =
      [&](HloComputation* computation) -> absl::StatusOr<bool> {
    HloInstruction* param = computation->parameter_instruction(0);
    bool changed = false;
    for (auto instr : param->users()) {
      // Since only a while-loop with a tuple result reaches here, we can safely
      // assume that `param` is a tuple and the first operand of the
      // GetTupleElement instruction is a use of `param`.
      if (instr->opcode() == HloOpcode::kGetTupleElement) {
        VLOG(3) << "tuple index " << instr->tuple_index() << " "
                << instr->ToString();
        auto iter = index_to_constant.find(instr->tuple_index());
        if (iter != index_to_constant.end()) {
          const HloInstruction* hlo_constant = (*iter).second;
          VLOG(3) << "Replace use of " << instr->ToString() << " with "
                  << hlo_constant->ToString();
          ABSL_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(
              computation->AddInstruction(hlo_constant->Clone())));
          changed = true;
        }
      }
    }
    return changed;
  };

  ABSL_ASSIGN_OR_RETURN(bool changed_cond,
                        propagate_constant(while_op->while_condition()));
  ABSL_ASSIGN_OR_RETURN(bool changed_body, propagate_constant(while_body));

  return changed_cond || changed_body;
}

// Converts a flat list of instructions into a tuple of the desired shape.  For
// example, given a tuple shape ((x, x), x) and instructions {A, B, C}, returns
// a tuple of value ((A, B), C).
//
// desired_shape must be a tuple.  (This precondition allows us to return a
// unique_ptr rather than a raw ptr.)
static std::unique_ptr<HloInstruction> UnflattenTupleInstr(
    absl::Span<HloInstruction*> instrs, const Shape& desired_shape,
    std::vector<std::unique_ptr<HloInstruction>>* new_instrs) {
  CHECK(desired_shape.IsTuple()) << ShapeUtil::HumanString(desired_shape);

  // For each child shape in `desired_shape`, slice out the correct number of
  // `instrs` and call UnflattenTupleInstr recursively.  At each step we remove
  // elements from `instrs` so that it only contains instructions we have not
  // yet processed.
  std::vector<HloInstruction*> elems;
  for (int i = 0; i < desired_shape.tuple_shapes().size(); ++i) {
    const Shape& subshape = desired_shape.tuple_shapes(i);
    if (!subshape.IsTuple()) {
      elems.push_back(instrs[0]);
      instrs.remove_prefix(1);
      continue;
    }

    // Count the number of leaf nodes underneath desired_shape[i].
    int64_t num_leaves = 0;
    ShapeUtil::ForEachSubshape(
        subshape, [&](const Shape& s, const ShapeIndex& /*index*/) {
          if (!s.IsTuple()) {
            ++num_leaves;
          }
        });

    std::unique_ptr<HloInstruction> subinstr =
        UnflattenTupleInstr(instrs.subspan(0, num_leaves),
                            desired_shape.tuple_shapes(i), new_instrs);
    elems.push_back(subinstr.get());
    new_instrs->push_back(std::move(subinstr));
    instrs.remove_prefix(num_leaves);
  }
  return HloInstruction::CreateTuple(elems);
}

// Builds a vector whose elements are the values in the flattened tuple for
// `instr`.  For example, if `instr` is a tuple of form ((A, B), C), returns the
// vector {A, B, C} (or kGetTupleElement ops which point to A, B, and C).
static std::vector<HloInstruction*> GetFlatTupleElems(
    HloInstruction* instr,
    std::vector<std::unique_ptr<HloInstruction>>* new_instrs) {
  const auto& shape = instr->shape();
  if (!shape.IsTuple()) {
    return {instr};
  }
  std::vector<HloInstruction*> elems;
  for (int i = 0; i < shape.tuple_shapes().size(); ++i) {
    const Shape& subshape = shape.tuple_shapes(i);
    new_instrs->push_back(
        HloInstruction::CreateGetTupleElement(subshape, instr, i));
    auto* gte = new_instrs->back().get();
    auto flattened_subshape = GetFlatTupleElems(gte, new_instrs);
    elems.insert(elems.end(), flattened_subshape.begin(),
                 flattened_subshape.end());
  }
  return elems;
}

static absl::StatusOr<bool> TryFlattenNestedTuples(HloInstruction* while_op) {
  if (frontend::HasDisableWhileLoopDceAttr(while_op)) {
    return false;
  }
  auto flatten_original_value = [&](const HloInstruction* old_instr,
                                    HloInstruction* new_instr) {
    if (old_instr->original_value()) {
      auto new_original_value = std::make_shared<OriginalValue>(
          new_instr->shape(), old_instr->original_value()->call_hierarchy());
      int64_t i = 0;
      for (auto& [shape_index, original_array] :
           old_instr->original_value()->tree().leaves()) {
        *new_original_value->mutable_tree()->mutable_element({i++}) =
            original_array;
      }
      new_instr->set_original_value(new_original_value);
    }
  };
  HloModule* module = while_op->GetModule();
  HloComputation* computation = while_op->parent();
  auto* while_init = while_op->mutable_operand(0);
  auto* while_body = while_op->while_body();
  auto* while_cond = while_op->while_condition();
  auto* while_body_root = while_body->root_instruction();
  if (while_init->opcode() != HloOpcode::kTuple ||
      while_body_root->opcode() != HloOpcode::kTuple) {
    return false;
  }

  TF_RET_CHECK(while_cond->num_parameters() == 1);
  TF_RET_CHECK(while_body->num_parameters() == 1);
  TF_RET_CHECK(
      ShapeUtil::Compatible(while_init->shape(), while_body_root->shape()));
  Shape while_shape = while_init->shape();
  if (!ShapeUtil::IsNestedTuple(while_shape)) {
    return false;
  }

  std::vector<const Shape*> flattened_shape_elems;
  ShapeUtil::ForEachSubshape(while_shape,
                             [&](const Shape& s, const ShapeIndex& /*index*/) {
                               if (!s.IsTuple()) {
                                 flattened_shape_elems.push_back(&s);
                               }
                             });
  Shape flattened_shape =
      ShapeUtil::MakeTupleShapeWithPtrs(flattened_shape_elems);

  // `new_instrs` holds instructions created outside of a computation for
  // cloning.  Elements added here just need to live until the end of the
  // relevant CloneWithReplacement call.
  std::vector<std::unique_ptr<HloInstruction>> new_instrs;
  auto add_new_instr = [&](std::unique_ptr<HloInstruction> instr) {
    new_instrs.push_back(std::move(instr));
    return new_instrs.back().get();
  };

  auto nested = [&](HloInstruction* instr,
                    const HloInstruction* old_instr = nullptr) {
    std::vector<HloInstruction*> gtes;
    const Shape& flat_shape = instr->shape();
    gtes.reserve(flat_shape.tuple_shapes().size());
    for (int i = 0; i < flat_shape.tuple_shapes().size(); ++i) {
      gtes.push_back(add_new_instr(HloInstruction::CreateGetTupleElement(
          flat_shape.tuple_shapes(i), instr, i)));
    }
    auto nested_instr =
        UnflattenTupleInstr(absl::MakeSpan(gtes), while_shape, &new_instrs);
    CHECK(ShapeUtil::Compatible(nested_instr->shape(), while_shape))
        << ShapeUtil::HumanString(nested_instr->shape()) << " vs "
        << ShapeUtil::HumanString(while_shape);
    if (old_instr != nullptr) {
      nested_instr->CopyOriginalValue(old_instr);
    }
    return nested_instr;
  };

  auto flattened = [&](HloInstruction* instr) {
    return HloInstruction::CreateTuple(GetFlatTupleElems(instr, &new_instrs));
  };

  // Create a new while-condition computation, where parameter 0 has flat shape
  // but all uses of it go through the nested shape.
  std::unique_ptr<HloComputation> new_while_cond =
      while_cond->CloneWithReplacementPairs({
          while_cond->parameter_instruction(0),
          nested(add_new_instr(HloInstruction::CreateParameter(
                     0, flattened_shape,
                     while_cond->parameter_instruction(0)->name())),
                 while_cond->parameter_instruction(0)),
      });

  // Create a new while-body computation, where parameter 0 has a flat shape and
  // all uses of it go through the nested shape, and where the root has a flat
  // shape constructed from the old nested root.
  std::unique_ptr<HloComputation> new_while_body =
      while_body->CloneWithReplacementPairs(
          {
              while_body->parameter_instruction(0),
              nested(add_new_instr(HloInstruction::CreateParameter(
                         0, flattened_shape,
                         while_body->parameter_instruction(0)->name())),
                     while_body->parameter_instruction(0)),
          },
          {
              while_body->root_instruction(),
              flattened(add_new_instr(while_body->root_instruction()->Clone())),
          });

  // Create the final while loop, and add any new instructions created to
  // `computation`.
  new_instrs.clear();
  auto* new_while_op = computation->AddInstruction(HloInstruction::CreateWhile(
      flattened_shape,
      module->AddEmbeddedComputation(std::move(new_while_cond)),
      module->AddEmbeddedComputation(std::move(new_while_body)),
      computation->AddInstruction(flattened(while_init))));
  new_while_op->CopyBackendConfigFrom(while_op);
  CopyFrontendAttributes(while_op, new_while_op);
  CopyMetadata(while_op, new_while_op);
  flatten_original_value(while_init, new_while_op->mutable_operand(0));
  flatten_original_value(while_op, new_while_op);
  flatten_original_value(while_body->parameter_instruction(0),
                         new_while_op->while_body()->parameter_instruction(0));
  flatten_original_value(
      while_cond->parameter_instruction(0),
      new_while_op->while_condition()->parameter_instruction(0));
  flatten_original_value(while_body_root,
                         new_while_op->while_body()->root_instruction());
  ABSL_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
      while_op, nested(new_while_op, while_op)));
  for (auto& instr : new_instrs) {
    computation->AddInstruction(std::move(instr));
  }

  return true;
}

// Tries to merge loop induction variables of a given type.
//
// In this pass we're only concerned with elements of the loop's tuple that
// are effective-scalars of type `elem_ty`.  Some terminology:
//
//  - The trip counter is the first element of the loop's tuple that starts at
//    0 and does x++ on each iteration.
//
//  - An induction variable is an element of the loop's tuple that is not the
//    trip counter and does `x += <constant>` on each iteration of the loop.
//    Negative constants are OK.
//
// This pass adds a trip counter if one isn't already present, then replaces
// each induction variable with
//
//   <initial_value> + <trip_count> * <constant>.
//
// This reduces the number of scalar operations in the loop, which is important
// e.g. on GPUs, where each scalar operation is nontrivially expensive because
// it's a separate kernel launch.
//
// Returns the new loop if a change was made, or null if no change was made.
// Note that the new loop is not a valid replacement for the old loop; it may
// need to be wrapped in a tuple that changes its shape.  We return the loop
// itself so that you can call TryMergeInductionVariables in a loop, once for
// each integral type elem_ty.
static absl::StatusOr<HloInstruction*> TryMergeInductionVariables(
    HloInstruction* while_op, PrimitiveType elem_ty) {
  CHECK(primitive_util::IsIntegralType(elem_ty)) << PrimitiveType_Name(elem_ty);
  HloModule* module = while_op->GetModule();
  HloComputation* computation = while_op->parent();
  auto* while_init = while_op->mutable_operand(0);
  auto* while_body = while_op->while_body();
  auto* while_cond = while_op->while_condition();
  auto* while_body_root = while_body->root_instruction();
  if (while_init->opcode() != HloOpcode::kTuple ||
      while_body_root->opcode() != HloOpcode::kTuple) {
    return nullptr;
  }

  TF_RET_CHECK(while_cond->num_parameters() == 1);
  TF_RET_CHECK(while_body->num_parameters() == 1);
  TF_RET_CHECK(
      ShapeUtil::Compatible(while_init->shape(), while_body_root->shape()));
  Shape while_shape = while_init->shape();

  // The tuple index of the trip counter, if one is present.
  std::optional<int64_t> trip_counter;
  // Maps the tuple index of each induction variable to its constant increment.
  absl::flat_hash_map<int64_t, const HloConstantInstruction*> induction_vars;
  for (int64_t i = 0; i < while_body_root->operand_count(); ++i) {
    HloInstruction* constant;
    if (!Match(while_body_root->mutable_operand(i),
               m::AddAnyOrder(m::GetTupleElement(m::Parameter(), i),
                              m::ConstantScalar(&constant))
                   .WithShape(m::Shape().WithElementType(elem_ty)))) {
      continue;
    }
    if (!trip_counter && constant->literal().IsAll(1) &&
        while_init->operand(i)->IsConstant() &&
        while_init->operand(i)->literal().IsAll(0)) {
      VLOG(10) << "Found existing trip counter at index " << i;
      trip_counter = i;
    } else {
      VLOG(10) << "Found induction variable at index " << i;
      induction_vars.emplace(i, Cast<HloConstantInstruction>(constant));
    }
  }

  // There's only something to simplify if we can either:
  //
  //  - combine one or more induction vars with an existing trip counter, or
  //  - replace two or more induction variables with a new trip counter.
  //
  // Put another way, there's only something to simplify if the number of
  // induction vars plus the number of existing trip counters (0 or 1) is >= 2.
  if (induction_vars.size() + (trip_counter.has_value() ? 1 : 0) < 2) {
    return nullptr;
  }

  // OK, we're going to do the transformation!  Set up some helpers.

  // `new_instrs` holds instructions created outside of a computation for
  // cloning.  Elements added here just need to live until the end of the
  // relevant CloneWithReplacement call.
  std::vector<std::unique_ptr<HloInstruction>> new_instrs;
  auto add_new_instr = [&](std::unique_ptr<HloInstruction> instr) {
    new_instrs.push_back(std::move(instr));
    return new_instrs.back().get();
  };

  auto add_binary_op = [&](const Shape& shape, HloOpcode opcode,
                           HloInstruction* lhs, HloInstruction* rhs) {
    // Reshape lhs/rhs to the output shape if necessary.  This deals with the
    // fact that induction variables need only be effective scalars, not true
    // scalars.
    if (!ShapeUtil::Compatible(shape, lhs->shape())) {
      lhs = add_new_instr(HloInstruction::CreateReshape(shape, lhs));
    }
    if (!ShapeUtil::Compatible(shape, rhs->shape())) {
      rhs = add_new_instr(HloInstruction::CreateReshape(shape, rhs));
    }
    return add_new_instr(HloInstruction::CreateBinary(shape, opcode, lhs, rhs));
  };

  auto add_gte = [&](HloInstruction* src, int64_t idx) {
    return add_new_instr(HloInstruction::CreateGetTupleElement(
        src->shape().tuple_shapes(idx), src, idx));
  };

  // Our new while loop will have the same shape as the old while loop, except
  // we'll add a trip counter to the end if it wasn't originally present.
  Shape new_while_shape = while_shape;
  bool added_trip_counter = false;
  if (!trip_counter) {
    VLOG(10) << "Adding new trip counter to end of loop's tuple.";
    trip_counter = new_while_shape.tuple_shapes().size();
    *new_while_shape.add_tuple_shapes() =
        ShapeUtil::MakeShape(elem_ty, /*dimensions=*/{});
    added_trip_counter = true;
  }

  // Converts `instr` into a tuple of the "old" form -- that is, to a tuple with
  // shape `while_body->shape()` and where the induction variables are "reified"
  // (i.e. they have value <init> + <counter> * <constant>).
  auto convert_to_old_form = [&](HloInstruction* instr,
                                 const HloInstruction* old_instr = nullptr) {
    CHECK(ShapeUtil::Compatible(instr->shape(), new_while_shape));
    std::vector<HloInstruction*> tuple_elems;
    for (int i = 0; i < while_shape.tuple_shapes().size(); ++i) {
      const auto& elem_shape = while_shape.tuple_shapes(i);
      if (!induction_vars.count(i)) {
        tuple_elems.push_back(add_gte(instr, i));
        continue;
      }
      tuple_elems.push_back(add_binary_op(
          elem_shape, HloOpcode::kAdd, add_gte(instr, i),
          add_binary_op(elem_shape, HloOpcode::kMultiply,
                        add_gte(instr, *trip_counter),
                        add_new_instr(induction_vars.at(i)->Clone()))));
      // Copy the original value of the induction variable to its replacement.
      tuple_elems.back()->CopyOriginalValue(while_body_root->operand(i));
    }
    auto tuple = HloInstruction::CreateTuple(tuple_elems);
    if (old_instr != nullptr) {
      tuple->CopyOriginalValue(old_instr);
    }
    return tuple;
  };

  // Converts `root` into a tuple of the "new" form -- that is, to a tuple with
  // shape `new_while_shape` and where the induction variables (but not trip
  // counters) are replaced with their unchanging <loop_body_param> values.
  auto convert_to_new_form = [&](HloInstruction* old_root,
                                 HloParameterInstruction* loop_body_param) {
    CHECK(ShapeUtil::Compatible(old_root->shape(), while_shape));
    std::vector<HloInstruction*> tuple_elems;

    // In the new form, induction variables come from `init`, everything else
    // (including the trip counter if it's not one we created ourselves) comes
    // from the `root` tuple unmodified.
    tuple_elems.reserve(while_shape.tuple_shapes().size());
    for (int i = 0; i < while_shape.tuple_shapes().size(); ++i) {
      tuple_elems.push_back(
          add_gte((induction_vars.count(i) ? loop_body_param : old_root), i));
    }
    // If we created a trip counter ourselves, add 1 to it in the next
    // iteration.
    if (added_trip_counter) {
      tuple_elems.push_back(add_binary_op(
          new_while_shape.tuple_shapes(*trip_counter), HloOpcode::kAdd,
          add_gte(loop_body_param, *trip_counter),
          add_new_instr(
              HloInstruction::CreateConstant(LiteralUtil::One(elem_ty)))));
    }

    return HloInstruction::CreateTuple(tuple_elems);
  };

  // Creates a new init tuple, which is the same as the old init tuple except if
  // we added a trip counter, it's set to 0.
  auto get_new_while_init = [&](HloInstruction* init) {
    CHECK(ShapeUtil::Compatible(init->shape(), while_shape));
    if (!added_trip_counter) {
      return init;
    }
    std::vector<HloInstruction*> tuple_elems;
    tuple_elems.reserve(while_shape.tuple_shapes().size());
    for (int i = 0; i < while_shape.tuple_shapes().size(); ++i) {
      tuple_elems.push_back(add_gte(init, i));
    }
    tuple_elems.push_back(add_new_instr(
        HloInstruction::CreateConstant(LiteralUtil::Zero(elem_ty))));
    return add_new_instr(HloInstruction::CreateTuple(tuple_elems));
  };

  std::unique_ptr<HloComputation> new_while_cond =
      while_cond->CloneWithReplacementPairs({
          while_cond->parameter_instruction(0),
          convert_to_old_form(
              add_new_instr(HloInstruction::CreateParameter(
                  0, new_while_shape,
                  while_cond->parameter_instruction(0)->name())),
              while_cond->parameter_instruction(0)),
      });

  // Creating the new while body proceeds in two steps.  First we convert the
  // users of the parameter to the old form.  Then as a second
  // CloneWithReplacement operation we convert the root to the new form.  We
  // have to do this in two steps because the new root needs to use the new
  // param0, and during the first clone operation, only the *old-form* param0 is
  // accessible.
  //
  // We have to add temp_new_while_body to the module because cloning a
  // computation touches the module (to get its NameUniquer).
  HloComputation* temp_new_while_body =
      module->AddEmbeddedComputation(while_body->CloneWithReplacementPairs({
          while_body->parameter_instruction(0),
          convert_to_old_form(
              add_new_instr(HloInstruction::CreateParameter(
                  0, new_while_shape,
                  while_body->parameter_instruction(0)->name())),
              while_body->parameter_instruction(0)),
      }));
  std::unique_ptr<HloComputation> new_while_body =
      temp_new_while_body->CloneWithReplacementPairs({
          temp_new_while_body->root_instruction(),
          convert_to_new_form(
              add_new_instr(temp_new_while_body->root_instruction()->Clone()),
              Cast<HloParameterInstruction>(
                  temp_new_while_body->parameter_instruction(0))),
      });
  ABSL_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(temp_new_while_body));

  // Create the final while loop, and add any new instructions created to
  // `computation`.
  new_instrs.clear();
  auto* new_while = computation->AddInstruction(HloInstruction::CreateWhile(
      new_while_shape,
      module->AddEmbeddedComputation(std::move(new_while_cond)),
      module->AddEmbeddedComputation(std::move(new_while_body)),
      get_new_while_init(while_init)));
  new_while->CopyBackendConfigFrom(while_op);
  if (auto original_value = while_init->original_value()) {
    new_while->while_init()->set_original_value(original_value);
  }
  if (auto original_value = while_op->original_value()) {
    new_while->set_original_value(original_value);
  }
  if (auto original_value =
          while_op->while_body()->parameter_instruction(0)->original_value()) {
    new_while->while_body()->parameter_instruction(0)->set_original_value(
        original_value);
  }
  if (auto original_value = while_op->while_condition()
                                ->parameter_instruction(0)
                                ->original_value()) {
    new_while->while_condition()->parameter_instruction(0)->set_original_value(
        original_value);
  }
  if (auto original_value =
          while_op->while_body()->root_instruction()->original_value()) {
    new_while->while_body()->root_instruction()->set_original_value(
        original_value);
  }
  if (added_trip_counter) {
    AppendToWhileLoopOriginalValue(new_while, {nullptr});
  }
  CopyFrontendAttributes(while_op, new_while);
  CopyMetadata(while_op, new_while);
  ABSL_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
      while_op, convert_to_old_form(new_while, while_op)));
  for (auto& instr : new_instrs) {
    computation->AddInstruction(std::move(instr));
  }
  return new_while;
}

absl::StatusOr<bool> WhileLoopSimplifier::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      3, "WhileLoopSimplifier::RunImpl(), before:\n" + module->ToString());
  bool changed = false;

  // Gather all the while ops in our module.  We do this ahead of time so we
  // don't have to worry about mutating the lists of computations or
  // instructions while we iterate.
  std::vector<HloInstruction*> while_ops;
  for (auto* comp : module->computations(execution_threads)) {
    for (auto* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kWhile) {
        while_ops.push_back(instr);
      }
    }
  }

  for (HloInstruction* while_op : while_ops) {
    // TODO(b/260601110) : Bail if any of the computations the while_op uses are
    // used by something else.  In theory, we could handle this case more
    // cleanly, but that'd require restructuring the pass, and there's no need
    // to do it right now, so just add a bail-out to be conservative.
    if (while_op->while_body()->caller_instructions().size() > 1 ||
        while_op->while_condition()->caller_instructions().size() > 1 ||
        frontend::HasDisableWhileLoopDceAttr(while_op)) {
      continue;
    }

    // Each of the optimizations below modifies the while loop itself if it's
    // successful, meaning that `while_op` is no longer valid after one of these
    // transformations returns true.
    // These optimizations should be fine even with send/recv nodes within the
    // loop.

    ABSL_ASSIGN_OR_RETURN(bool result,
                          TryRemoveRepeatedWhileTupleIndices(while_op));
    changed |= result;
    if (result) {
      continue;
    }

    ABSL_ASSIGN_OR_RETURN(result, TryFlattenNestedTuples(while_op));
    changed |= result;
    if (result) {
      continue;
    }

    ABSL_ASSIGN_OR_RETURN(result, TryRemoveDeadWhileParams(while_op));
    changed |= result;
    if (result) {
      continue;
    }

    ABSL_ASSIGN_OR_RETURN(result, TryRemoveConstantParams(while_op));
    changed |= result;
    if (result) {
      continue;
    }

    if (hoist_write_only_row_packs_) {
      ABSL_ASSIGN_OR_RETURN(result,
                            TryHoistRowPackFromWriteOnlyDusCarries(while_op));
      changed |= result;
      if (result) {
        continue;
      }
    }

    if (simplify_compare_instrs_) {
      ABSL_ASSIGN_OR_RETURN(result, TryRemoveTrivialCompare(while_op));
      changed |= result;
      if (result) {
        continue;
      }
    }

    // We can't remove while loops that contain send/recv nodes, because we rely
    // on the particular loop structure around the node matching on the send and
    // recv sides.  Other while simplifications require us to remove the loop
    // and replace it with a new one, so we can't do that either.
    if (ContainsInstrWithOpcode(while_op->while_body(),
                                {HloOpcode::kSend, HloOpcode::kSendDone,
                                 HloOpcode::kRecv, HloOpcode::kRecvDone}) ||
        ContainsInstrWithOpcode(while_op->while_condition(),
                                {HloOpcode::kSend, HloOpcode::kSendDone,
                                 HloOpcode::kRecv, HloOpcode::kRecvDone})) {
      VLOG(2) << "Not attempting to simplify while loop because it contains a "
                 "send/recv node: "
              << while_op->ToShortString();
      continue;
    }

    ABSL_ASSIGN_OR_RETURN(result, TryPropagateConstant(while_op));
    changed |= result;

    ABSL_ASSIGN_OR_RETURN(result, TryRemoveWhileLoop(while_op));
    changed |= result;

    if (result) {
      // Don't continue simplifying after successfully removing the while loop
      // -- that would result in use-after-free nastiness.
      continue;
    }

    bool merged_induction_vars = false;
    // Notably missing from this list are S16 and U16.  These don't currently
    // work because S/U16 literals are not implemented.
    for (auto elem_ty : {S8, U8, S32, U32, S64, U64}) {
      ABSL_ASSIGN_OR_RETURN(auto* new_while_op,
                            TryMergeInductionVariables(while_op, elem_ty));
      if (new_while_op) {
        while_op = new_while_op;
        changed = true;
        merged_induction_vars = true;
      }
    }
    if (merged_induction_vars) {
      continue;
    }
  }
  if (changed) {
    HloDCE dce;
    ABSL_RETURN_IF_ERROR(dce.Run(module, execution_threads).status());
  }
  XLA_VLOG_LINES(
      3, "WhileLoopSimplifier::RunImpl(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
