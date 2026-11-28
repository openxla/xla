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

#include "xla/backends/gpu/hlo/transforms/dot_bf16_demotion.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tsl/platform/tensor_float_32_utils.h"
#include "xla/backends/gpu/hlo/transforms/dot_operand_converter.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

constexpr int64_t kMaxExactConstantElements = 1024;

// Returns true if `value` round-trips through BF16 with identical IEEE-754 F32
// bits (i.e. its lower 16 mantissa bits are zero).
bool IsBf16ExactFloat(float value) {
  return (absl::bit_cast<uint32_t>(value) & 0xFFFFu) == 0;
}

bool IsBf16ExactConstant(const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kConstant || !instr->shape().IsArray() ||
      instr->shape().element_type() != F32 ||
      ShapeUtil::ElementsIn(instr->shape()) > kMaxExactConstantElements) {
    return false;
  }
  const Literal& literal = instr->literal();
  if (!literal.shape().IsArray() || literal.shape().element_type() != F32) {
    return false;
  }
  return absl::c_all_of(literal.data<float>(), IsBf16ExactFloat);
}

// Returns true if `instr` is an F32 instruction whose exact BF16 equivalent can
// be obtained in the same computation with zero new HBM allocation:
//  - `convert<f32>(bf16)` with identical non-type shape (the BF16 operand
//    already exists in memory), or
//  - a scalar BF16-exact constant (materialized as an immediate, not a buffer).
bool IsFreeBf16Source(const HloInstruction* instr) {
  if (!instr->shape().IsArray() || instr->shape().element_type() != F32) {
    return false;
  }
  if (instr->opcode() == HloOpcode::kConvert &&
      instr->operand(0)->shape().element_type() == BF16 &&
      ShapeUtil::Equal(ShapeUtil::ChangeElementType(instr->shape(), BF16),
                       instr->operand(0)->shape())) {
    return true;
  }
  return instr->opcode() == HloOpcode::kConstant &&
         ShapeUtil::IsScalar(instr->shape()) && IsBf16ExactConstant(instr);
}

// Returns true if `instr` is a base F32 producer whose output values are always
// exactly representable in BF16 regardless of its operands.
bool IsBaseBf16ExactProducer(const HloInstruction* instr) {
  if (!instr->shape().IsArray() || instr->shape().element_type() != F32) {
    return false;
  }
  switch (instr->opcode()) {
    case HloOpcode::kConvert:
      return primitive_util::CastPreservesValues(
          instr->operand(0)->shape().element_type(), BF16);
    case HloOpcode::kReducePrecision: {
      const auto* reduce_precision = Cast<HloReducePrecisionInstruction>(instr);
      return reduce_precision->exponent_bits() <=
                 primitive_util::ExponentWidth(BF16) &&
             reduce_precision->mantissa_bits() <=
                 primitive_util::SignificandWidth(BF16) - 1;
    }
    case HloOpcode::kConstant:
      return IsBf16ExactConstant(instr);
    default:
      return false;
  }
}

// Returns the operand indices whose elements flow verbatim into `instr`'s
// array output without arithmetic. Returns empty if `instr` is not a
// value-preserving forwarder of its own element type.
absl::InlinedVector<int64_t, 2> GetForwarderDataOperands(
    const HloInstruction* instr) {
  if (!instr->shape().IsArray()) {
    return {};
  }
  switch (instr->opcode()) {
    case HloOpcode::kBitcast:
      if (instr->operand(0)->shape().element_type() ==
          instr->shape().element_type()) {
        return {0};
      }
      return {};
    case HloOpcode::kBroadcast:
    case HloOpcode::kCopy:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kGather:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
      return {0};
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kPad:
      return {0, 1};
    case HloOpcode::kSelect:
      return {1, 2};
    case HloOpcode::kConcatenate: {
      absl::InlinedVector<int64_t, 2> indices;
      indices.reserve(instr->operand_count());
      for (int64_t i = 0; i < instr->operand_count(); ++i) {
        indices.push_back(i);
      }
      return indices;
    }
    default:
      return {};
  }
}

bool IsEligibleF32Dot(const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kDot || instr->operand_count() != 2) {
    return false;
  }
  const PrimitiveType out_type = instr->shape().element_type();
  if (out_type != F32 && out_type != BF16) {
    return false;
  }
  if (instr->operand(0)->shape().element_type() != F32 ||
      instr->operand(1)->shape().element_type() != F32) {
    return false;
  }
  const PrecisionConfig& precision_config = instr->precision_config();
  return precision_config.algorithm() == PrecisionConfig::ALG_UNSET &&
         absl::c_all_of(precision_config.operand_precision(),
                        [](int p) { return p <= PrecisionConfig::HIGH; });
}

bool IsStructuralOrControlFlowOp(
    const HloInstruction* instr,
    const absl::flat_hash_set<const HloComputation*>& control_flow_comps) {
  switch (instr->opcode()) {
    case HloOpcode::kTuple:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kAddDependency:
      return true;
    case HloOpcode::kCopy:
      return instr->shape().IsTuple();
    case HloOpcode::kParameter:
      return !instr->parent()->IsEntryComputation();
    case HloOpcode::kConditional:
    case HloOpcode::kWhile:
    case HloOpcode::kCall:
      return absl::c_all_of(instr->called_computations(),
                            [&](const HloComputation* comp) {
                              return control_flow_comps.contains(comp);
                            });
    default:
      return false;
  }
}

template <typename Fn>
void ForEachF32Leaf(const HloInstruction* instr, Fn&& fn) {
  ShapeUtil::ForEachSubshape(
      instr->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.IsArray() && subshape.element_type() == F32) {
          fn(index);
        }
      });
}

ShapeIndex PrependIndex(int64_t prefix, const ShapeIndex& suffix) {
  ShapeIndex result;
  result.reserve(suffix.size() + 1);
  result.push_back(prefix);
  result.insert(result.end(), suffix.begin(), suffix.end());
  return result;
}

absl::StatusOr<HloInstruction*> GetOrCreateFreeBf16Source(
    HloInstruction* f32_source,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>& free_bf16_cache) {
  auto it = free_bf16_cache.find(f32_source);
  if (it != free_bf16_cache.end()) {
    return it->second;
  }
  if (f32_source->opcode() == HloOpcode::kConvert) {
    HloInstruction* bf16_operand = f32_source->mutable_operand(0);
    CHECK_EQ(bf16_operand->shape().element_type(), BF16);
    free_bf16_cache.emplace(f32_source, bf16_operand);
    return bf16_operand;
  }
  CHECK_EQ(f32_source->opcode(), HloOpcode::kConstant);
  ABSL_ASSIGN_OR_RETURN(Literal bf16_literal,
                        f32_source->literal().Convert(BF16));
  HloInstruction* bf16_constant = f32_source->parent()->AddInstruction(
      HloInstruction::CreateConstant(std::move(bf16_literal)));
  f32_source->SetupDerivedInstruction(bf16_constant);
  free_bf16_cache.emplace(f32_source, bf16_constant);
  return bf16_constant;
}

// Phase 1: Forward BF16-exactness analysis over F32 HloValues.
// Optimistically assumes candidate producers and value-preserving forwarders
// are BF16-exact, seeds all non-BF16-exact producers, and propagates
// non-exactness forward via BFS (resolving while-loop cycles in linear time).
absl::flat_hash_set<const HloValue*> FindNotBf16ExactValues(
    const HloDataflowAnalysis& dataflow,
    const absl::flat_hash_set<const HloComputation*>& control_flow_comps) {
  absl::flat_hash_map<const HloValue*, std::vector<const HloValue*>>
      forward_edges;
  absl::flat_hash_set<const HloValue*> not_bf16_exact;
  std::deque<const HloValue*> queue;
  auto mark_not_exact = [&](const HloValue* value) {
    if (not_bf16_exact.insert(value).second) {
      queue.push_back(value);
    }
  };

  for (const HloValue* value : dataflow.values()) {
    if (!value->shape().IsArray() || value->shape().element_type() != F32) {
      continue;
    }
    const HloInstruction* def = value->defining_instruction();
    if (!control_flow_comps.contains(def->parent())) {
      mark_not_exact(value);
      continue;
    }
    if (IsBaseBf16ExactProducer(def)) {
      continue;
    }
    const absl::InlinedVector<int64_t, 2> data_operands =
        GetForwarderDataOperands(def);
    if (data_operands.empty()) {
      mark_not_exact(value);
      continue;
    }
    bool missing_source = false;
    for (int64_t op_idx : data_operands) {
      const HloValueSet& value_set = dataflow.GetValueSet(def->operand(op_idx));
      if (value_set.values().empty()) {
        missing_source = true;
        break;
      }
      for (const HloValue* src : value_set.values()) {
        forward_edges[src].push_back(value);
      }
    }
    if (missing_source) {
      mark_not_exact(value);
    }
  }

  while (!queue.empty()) {
    const HloValue* value = queue.front();
    queue.pop_front();
    if (auto it = forward_edges.find(value); it != forward_edges.end()) {
      for (const HloValue* dst : it->second) {
        mark_not_exact(dst);
      }
    }
  }
  return not_bf16_exact;
}

struct DemotionPlan {
  std::vector<HloInstruction*> surviving_dots;
  absl::flat_hash_set<HloPosition> positions_to_narrow;
};

// Phase 2: Build the F32 constraint graph, propagate MUST_STAY_F32 via a single
// BFS, and collect the exact connected components feeding surviving dots.
// Any materialized F32 position that must stay F32 prevents its consumers from
// narrowing (avoiding duplicate F32 + BF16 buffers), while any consumer that
// stays F32 forces its F32 operands to stay F32. Free BF16 sources
// (`convert<f32>(bf16)` and scalar BF16-exact constants) omit the forward edge
// so direct consumers can bypass them with zero allocation.
DemotionPlan BuildDemotionPlan(
    absl::Span<HloComputation* const> ordered_control_flow_comps,
    const absl::flat_hash_set<const HloComputation*>& control_flow_comps,
    const absl::flat_hash_set<const HloValue*>& not_bf16_exact,
    absl::Span<HloInstruction* const> candidate_dots) {
  absl::flat_hash_set<const HloInstruction*> candidate_dot_set(
      candidate_dots.begin(), candidate_dots.end());
  absl::flat_hash_map<HloPosition, std::vector<HloPosition>> f32_edges;
  absl::flat_hash_map<HloPosition, std::vector<HloPosition>> narrow_edges;
  absl::flat_hash_map<HloPosition, std::vector<HloInstruction*>> pos_to_dots;
  absl::flat_hash_set<HloPosition> f32_positions;
  absl::flat_hash_set<HloInstruction*> f32_dots;
  std::deque<HloPosition> pos_queue;
  std::deque<HloInstruction*> dot_queue;

  auto constrain_pos = [&](const HloPosition& pos) {
    if (f32_positions.insert(pos).second) {
      pos_queue.push_back(pos);
    }
  };
  auto constrain_dot = [&](HloInstruction* dot) {
    if (f32_dots.insert(dot).second) {
      dot_queue.push_back(dot);
    }
  };
  auto constrain_all_f32_leaves = [&](HloInstruction* instr) {
    ForEachF32Leaf(instr, [&](const ShapeIndex& idx) {
      constrain_pos(HloPosition{instr, idx});
    });
  };
  auto add_consumer_producer_edge = [&](const HloPosition& consumer,
                                        const HloPosition& producer) {
    f32_edges[consumer].push_back(producer);
    if (!(producer.index.empty() && IsFreeBf16Source(producer.instruction))) {
      f32_edges[producer].push_back(consumer);
    }
    narrow_edges[consumer].push_back(producer);
    narrow_edges[producer].push_back(consumer);
  };
  auto add_bidirectional_edge = [&](const HloPosition& a,
                                    const HloPosition& b) {
    f32_edges[a].push_back(b);
    f32_edges[b].push_back(a);
    narrow_edges[a].push_back(b);
    narrow_edges[b].push_back(a);
  };

  for (const HloValue* value : not_bf16_exact) {
    for (const HloPosition& pos : value->positions()) {
      if (control_flow_comps.contains(pos.instruction->parent())) {
        constrain_pos(pos);
      }
    }
  }

  for (HloComputation* comp : ordered_control_flow_comps) {
    const bool is_entry = comp->IsEntryComputation();
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (is_entry && instr == comp->root_instruction()) {
        constrain_all_f32_leaves(instr);
      }
      if (is_entry && instr->opcode() == HloOpcode::kParameter) {
        constrain_all_f32_leaves(instr);
        continue;
      }
      if (candidate_dot_set.contains(instr)) {
        constrain_all_f32_leaves(instr);
        for (int i : {0, 1}) {
          HloInstruction* op = instr->mutable_operand(i);
          if (!IsFreeBf16Source(op)) {
            pos_to_dots[HloPosition{op, {}}].push_back(instr);
          }
        }
        continue;
      }
      if (IsBaseBf16ExactProducer(instr)) {
        if (instr->opcode() == HloOpcode::kReducePrecision) {
          constrain_pos(HloPosition{instr->mutable_operand(0), {}});
        }
        continue;
      }
      if (instr->shape().IsArray() && instr->shape().element_type() == F32) {
        const absl::InlinedVector<int64_t, 2> data_operands =
            GetForwarderDataOperands(instr);
        if (!data_operands.empty()) {
          for (int64_t op_idx : data_operands) {
            add_consumer_producer_edge(
                HloPosition{instr, {}},
                HloPosition{instr->mutable_operand(op_idx), {}});
          }
          continue;
        }
      }
      if (instr->opcode() == HloOpcode::kConvert &&
          instr->shape().element_type() == BF16) {
        continue;
      }
      if (IsStructuralOrControlFlowOp(instr, control_flow_comps)) {
        switch (instr->opcode()) {
          case HloOpcode::kTuple:
            for (int64_t i = 0; i < instr->operand_count(); ++i) {
              HloInstruction* op = instr->mutable_operand(i);
              ForEachF32Leaf(op, [&](const ShapeIndex& sub) {
                add_consumer_producer_edge(
                    HloPosition{instr, PrependIndex(i, sub)},
                    HloPosition{op, sub});
              });
            }
            break;
          case HloOpcode::kGetTupleElement: {
            HloInstruction* op = instr->mutable_operand(0);
            const int64_t tuple_index = instr->tuple_index();
            ForEachF32Leaf(instr, [&](const ShapeIndex& sub) {
              add_consumer_producer_edge(
                  HloPosition{instr, sub},
                  HloPosition{op, PrependIndex(tuple_index, sub)});
            });
            break;
          }
          case HloOpcode::kOptimizationBarrier:
          case HloOpcode::kAddDependency:
          case HloOpcode::kCopy: {
            HloInstruction* op = instr->mutable_operand(0);
            ForEachF32Leaf(instr, [&](const ShapeIndex& idx) {
              add_consumer_producer_edge(HloPosition{instr, idx},
                                         HloPosition{op, idx});
            });
            break;
          }
          case HloOpcode::kConditional:
            for (int j = 0; j < instr->branch_count(); ++j) {
              HloInstruction* arg = instr->mutable_operand(j + 1);
              HloInstruction* param =
                  instr->branch_computation(j)->parameter_instruction(0);
              ForEachF32Leaf(arg, [&](const ShapeIndex& idx) {
                add_consumer_producer_edge(HloPosition{param, idx},
                                           HloPosition{arg, idx});
              });
              HloInstruction* root =
                  instr->branch_computation(j)->root_instruction();
              ForEachF32Leaf(instr, [&](const ShapeIndex& idx) {
                add_bidirectional_edge(HloPosition{instr, idx},
                                       HloPosition{root, idx});
              });
            }
            break;
          case HloOpcode::kWhile: {
            HloInstruction* init = instr->mutable_operand(0);
            HloInstruction* body_param =
                instr->while_body()->parameter_instruction(0);
            HloInstruction* body_root = instr->while_body()->root_instruction();
            HloInstruction* cond_param =
                instr->while_condition()->parameter_instruction(0);
            ForEachF32Leaf(instr, [&](const ShapeIndex& idx) {
              add_consumer_producer_edge(HloPosition{body_param, idx},
                                         HloPosition{init, idx});
              add_bidirectional_edge(HloPosition{instr, idx},
                                     HloPosition{body_param, idx});
              add_bidirectional_edge(HloPosition{instr, idx},
                                     HloPosition{body_root, idx});
              add_bidirectional_edge(HloPosition{instr, idx},
                                     HloPosition{cond_param, idx});
            });
            break;
          }
          case HloOpcode::kCall:
            for (int64_t i = 0; i < instr->operand_count(); ++i) {
              HloInstruction* arg = instr->mutable_operand(i);
              HloInstruction* param =
                  instr->to_apply()->parameter_instruction(i);
              ForEachF32Leaf(arg, [&](const ShapeIndex& idx) {
                add_consumer_producer_edge(HloPosition{param, idx},
                                           HloPosition{arg, idx});
              });
            }
            ForEachF32Leaf(instr, [&](const ShapeIndex& idx) {
              add_bidirectional_edge(
                  HloPosition{instr, idx},
                  HloPosition{instr->to_apply()->root_instruction(), idx});
            });
            break;
          case HloOpcode::kParameter:
          default:
            break;
        }
        continue;
      }

      constrain_all_f32_leaves(instr);
      for (HloInstruction* op : instr->operands()) {
        constrain_all_f32_leaves(op);
      }
    }
  }

  while (!pos_queue.empty() || !dot_queue.empty()) {
    if (!dot_queue.empty()) {
      HloInstruction* dot = dot_queue.front();
      dot_queue.pop_front();
      constrain_pos(HloPosition{dot->mutable_operand(0), {}});
      constrain_pos(HloPosition{dot->mutable_operand(1), {}});
      continue;
    }
    const HloPosition pos = pos_queue.front();
    pos_queue.pop_front();
    if (auto it = f32_edges.find(pos); it != f32_edges.end()) {
      for (const HloPosition& next : it->second) {
        constrain_pos(next);
      }
    }
    if (auto it = pos_to_dots.find(pos); it != pos_to_dots.end()) {
      for (HloInstruction* dot : it->second) {
        constrain_dot(dot);
      }
    }
  }

  DemotionPlan plan;
  for (HloInstruction* dot : candidate_dots) {
    if (!f32_dots.contains(dot)) {
      plan.surviving_dots.push_back(dot);
    }
  }
  if (plan.surviving_dots.empty()) {
    return plan;
  }

  std::deque<HloPosition> narrow_queue;
  auto mark_to_narrow = [&](const HloPosition& pos) {
    if (!f32_positions.contains(pos) &&
        plan.positions_to_narrow.insert(pos).second) {
      narrow_queue.push_back(pos);
    }
  };
  for (HloInstruction* dot : plan.surviving_dots) {
    mark_to_narrow(HloPosition{dot->mutable_operand(0), {}});
    mark_to_narrow(HloPosition{dot->mutable_operand(1), {}});
  }
  while (!narrow_queue.empty()) {
    const HloPosition pos = narrow_queue.front();
    narrow_queue.pop_front();
    if (auto it = narrow_edges.find(pos); it != narrow_edges.end()) {
      for (const HloPosition& next : it->second) {
        mark_to_narrow(next);
      }
    }
  }
  return plan;
}

// Phase 3: Apply in-place narrowing, replace reduce-precision/constants,
// rewire free BF16 source operands, and remove no-op BF16 conversions.
absl::Status ApplyDemotionPlan(
    absl::Span<HloComputation* const> ordered_control_flow_comps,
    const DemotionPlan& plan) {
  absl::flat_hash_map<HloInstruction*, std::vector<ShapeIndex>> narrow_by_inst;
  for (const HloPosition& pos : plan.positions_to_narrow) {
    narrow_by_inst[pos.instruction].push_back(pos.index);
  }

  // 3a. Mutate shapes in-place except reduce-precision and constant.
  for (HloComputation* comp : ordered_control_flow_comps) {
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      auto it = narrow_by_inst.find(instr);
      if (it == narrow_by_inst.end() ||
          instr->opcode() == HloOpcode::kReducePrecision ||
          instr->opcode() == HloOpcode::kConstant) {
        continue;
      }
      for (const ShapeIndex& idx : it->second) {
        Shape* subshape =
            ShapeUtil::GetMutableSubshape(instr->mutable_shape(), idx);
        CHECK_EQ(subshape->element_type(), F32);
        subshape->set_element_type(BF16);
      }
    }
  }

  // 3b. Replace narrowed reduce-precision and constant instructions.
  std::vector<HloInstruction*> replaced_dead_insts;
  for (HloComputation* comp : ordered_control_flow_comps) {
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (!narrow_by_inst.contains(instr)) {
        continue;
      }
      if (instr->opcode() == HloOpcode::kReducePrecision) {
        const std::vector<HloInstruction*> users = instr->users();
        const bool is_root = instr == comp->root_instruction();
        HloInstruction* conv =
            comp->AddInstruction(HloInstruction::CreateConvert(
                ShapeUtil::ChangeElementType(instr->shape(), BF16), instr));
        instr->SetupDerivedInstruction(conv);
        for (HloInstruction* user : users) {
          ABSL_RETURN_IF_ERROR(instr->ReplaceUseWithDifferentShape(user, conv));
        }
        if (is_root) {
          comp->set_root_instruction(conv, /*accept_different_shape=*/true);
        }
      } else if (instr->opcode() == HloOpcode::kConstant) {
        ABSL_ASSIGN_OR_RETURN(Literal bf16_literal,
                              instr->literal().Convert(BF16));
        const bool is_root = instr == comp->root_instruction();
        HloInstruction* bf16_constant = comp->AddInstruction(
            HloInstruction::CreateConstant(std::move(bf16_literal)));
        instr->SetupDerivedInstruction(bf16_constant);
        ABSL_RETURN_IF_ERROR(
            instr->ReplaceAllUsesWithDifferentShape(bf16_constant));
        if (is_root) {
          comp->set_root_instruction(bf16_constant,
                                     /*accept_different_shape=*/true);
        }
        replaced_dead_insts.push_back(instr);
      }
    }
  }

  // 3c. Rewire remaining F32 free-source operands.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> free_bf16_cache;
  auto rewire_if_f32_free_source = [&](HloInstruction* consumer,
                                       int64_t operand_index) -> absl::Status {
    HloInstruction* op = consumer->mutable_operand(operand_index);
    if (!op->shape().IsArray() || op->shape().element_type() != F32) {
      return absl::OkStatus();
    }
    CHECK(IsFreeBf16Source(op))
        << "Expected free BF16 source for operand " << operand_index << " of "
        << consumer->ToString() << ", got " << op->ToString();
    ABSL_ASSIGN_OR_RETURN(HloInstruction * bf16_source,
                          GetOrCreateFreeBf16Source(op, free_bf16_cache));
    return consumer->ReplaceOperandWithDifferentShape(operand_index,
                                                      bf16_source);
  };

  for (HloInstruction* dot : plan.surviving_dots) {
    ABSL_RETURN_IF_ERROR(rewire_if_f32_free_source(dot, 0));
    ABSL_RETURN_IF_ERROR(rewire_if_f32_free_source(dot, 1));
  }

  for (HloComputation* comp : ordered_control_flow_comps) {
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kConditional) {
        for (int j = 0; j < instr->branch_count(); ++j) {
          const Shape& param_shape =
              instr->branch_computation(j)->parameter_instruction(0)->shape();
          if (param_shape.IsArray() && param_shape.element_type() == BF16) {
            ABSL_RETURN_IF_ERROR(rewire_if_f32_free_source(instr, j + 1));
          }
        }
        continue;
      }
      if (instr->opcode() == HloOpcode::kWhile) {
        const Shape& param_shape =
            instr->while_body()->parameter_instruction(0)->shape();
        if (param_shape.IsArray() && param_shape.element_type() == BF16) {
          ABSL_RETURN_IF_ERROR(rewire_if_f32_free_source(instr, 0));
        }
        continue;
      }
      if (instr->opcode() == HloOpcode::kCall) {
        for (int64_t i = 0; i < instr->operand_count(); ++i) {
          const Shape& param_shape =
              instr->to_apply()->parameter_instruction(i)->shape();
          if (param_shape.IsArray() && param_shape.element_type() == BF16) {
            ABSL_RETURN_IF_ERROR(rewire_if_f32_free_source(instr, i));
          }
        }
        continue;
      }

      auto it = narrow_by_inst.find(instr);
      if (it == narrow_by_inst.end()) {
        continue;
      }
      if (instr->opcode() == HloOpcode::kTuple) {
        for (const ShapeIndex& idx : it->second) {
          if (idx.size() == 1) {
            ABSL_RETURN_IF_ERROR(rewire_if_f32_free_source(instr, idx[0]));
          }
        }
        continue;
      }
      if (instr->opcode() == HloOpcode::kOptimizationBarrier ||
          instr->opcode() == HloOpcode::kAddDependency) {
        if (instr->shape().IsArray()) {
          ABSL_RETURN_IF_ERROR(rewire_if_f32_free_source(instr, 0));
        }
        continue;
      }
      if (instr->opcode() == HloOpcode::kBitcast) {
        ABSL_RETURN_IF_ERROR(rewire_if_f32_free_source(instr, 0));
        continue;
      }
      for (int64_t op_idx : GetForwarderDataOperands(instr)) {
        ABSL_RETURN_IF_ERROR(rewire_if_f32_free_source(instr, op_idx));
      }
    }
  }

  // 3d. Eliminate no-op BF16->BF16 conversions.
  for (HloComputation* comp : ordered_control_flow_comps) {
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (instr->opcode() != HloOpcode::kConvert ||
          instr->shape().element_type() != BF16 ||
          !ShapeUtil::Equal(instr->shape(), instr->operand(0)->shape())) {
        continue;
      }
      const bool is_root = instr == comp->root_instruction();
      HloInstruction* src = instr->mutable_operand(0);
      ABSL_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(src));
      if (is_root) {
        comp->set_root_instruction(src);
      }
      replaced_dead_insts.push_back(instr);
    }
  }

  for (HloInstruction* dead : replaced_dead_insts) {
    if (dead->IsDead()) {
      ABSL_RETURN_IF_ERROR(dead->parent()->RemoveInstruction(dead));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> DotBf16Demotion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!DotOperandConverter::SupportsBf16Dot(gpu_version_) ||
      !tsl::tensor_float_32_execution_enabled()) {
    return false;
  }

  bool has_eligible_dot = false;
  for (const HloComputation* computation :
       module->computations(execution_threads)) {
    if (absl::c_any_of(computation->instructions(), IsEligibleF32Dot)) {
      has_eligible_dot = true;
      break;
    }
  }
  if (!has_eligible_dot) {
    return false;
  }

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<HloDataflowAnalysis> dataflow,
      HloDataflowAnalysis::Run(
          *module, /*ssa_form=*/false, /*bitcast_defines_value=*/true,
          execution_threads, /*propagate_through_calls=*/true,
          /*precompute_uses=*/[](const HloValue& value) {
            return value.shape().element_type() == F32;
          }));

  const CallGraph& call_graph = dataflow->call_graph();
  std::vector<HloComputation*> ordered_control_flow_comps;
  absl::flat_hash_set<const HloComputation*> control_flow_comps;
  for (HloComputation* comp :
       module->MakeComputationPostOrder(execution_threads)) {
    const CallGraphNode& node = call_graph.GetNode(comp);
    if (node.context() != CallContext::kControlFlow ||
        comp->IsAsyncComputation()) {
      continue;
    }
    const bool only_standard_callers =
        absl::c_all_of(node.caller_callsites(), [](const CallSite& callsite) {
          const HloOpcode op = callsite.instruction()->opcode();
          return op == HloOpcode::kConditional || op == HloOpcode::kWhile ||
                 op == HloOpcode::kCall;
        });
    if (!only_standard_callers) {
      continue;
    }
    ordered_control_flow_comps.push_back(comp);
    control_flow_comps.insert(comp);
  }

  const absl::flat_hash_set<const HloValue*> not_bf16_exact =
      FindNotBf16ExactValues(*dataflow, control_flow_comps);

  auto is_position_bf16_exact = [&](const HloInstruction* instr) {
    const HloValueSet& value_set = dataflow->GetValueSet(instr);
    return !value_set.values().empty() &&
           absl::c_none_of(value_set.values(), [&](const HloValue* value) {
             return not_bf16_exact.contains(value);
           });
  };

  std::vector<HloInstruction*> candidate_dots;
  for (HloComputation* comp : ordered_control_flow_comps) {
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (IsEligibleF32Dot(instr) &&
          is_position_bf16_exact(instr->operand(0)) &&
          is_position_bf16_exact(instr->operand(1))) {
        candidate_dots.push_back(instr);
      }
    }
  }
  if (candidate_dots.empty()) {
    return false;
  }

  const DemotionPlan plan =
      BuildDemotionPlan(ordered_control_flow_comps, control_flow_comps,
                        not_bf16_exact, candidate_dots);
  if (plan.surviving_dots.empty()) {
    return false;
  }

  // Drop dataflow analysis before mutating the module.
  dataflow.reset();
  ABSL_RETURN_IF_ERROR(ApplyDemotionPlan(ordered_control_flow_comps, plan));
  return true;
}

}  // namespace xla::gpu
