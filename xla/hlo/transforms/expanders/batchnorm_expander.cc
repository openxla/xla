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

#include "xla/hlo/transforms/expanders/batchnorm_expander.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

using AddFn =
    absl::FunctionRef<HloInstruction*(std::unique_ptr<HloInstruction>)>;
constexpr int64_t kMaxS32 = std::numeric_limits<int32_t>::max();

std::optional<int64_t> MaxFiniteElementCount(PrimitiveType ptype) {
  constexpr std::pair<PrimitiveType, int64_t> kLimits[] = {
      {F8E5M3FNU, 114688}, {F16, 65504},        {F8E5M2, 57344},
      {F8E5M2FNUZ, 57344}, {F8E4M3FN, 448},     {F8E4M3, 240},
      {F8E4M3FNUZ, 240},   {F8E4M3B11FNUZ, 30}, {F6E3M2FN, 28},
      {F8E3M4, 15},        {F6E2M3FN, 7},       {F4E2M1FN, 6}};
  for (auto [t, limit] : kLimits) {
    if (t == ptype) {
      return limit;
    }
  }
  return std::nullopt;
}

Shape MakeFeatureShape(const Shape& op_shape, int64_t feature_index) {
  Shape s = ShapeUtil::MakeShape(
      op_shape.element_type(), {op_shape.dimensions(feature_index)},
      {op_shape.is_dynamic_dimension(feature_index)});
  if (op_shape.has_layout()) {
    s.mutable_layout()->set_element_size_in_bits(
        op_shape.layout().element_size_in_bits());
  }
  return s;
}

HloInstruction* ConvertIfNeeded(HloInstruction* inst, const Shape& ref,
                                AddFn add) {
  if (inst->shape().element_type() == ref.element_type()) {
    return inst;
  }
  Shape s = ShapeUtil::ChangeElementType(inst->shape(), ref.element_type());
  if (s.has_layout() && ref.has_layout()) {
    s.mutable_layout()->set_element_size_in_bits(
        ref.layout().element_size_in_bits());
  }
  return add(HloInstruction::CreateConvert(s, inst));
}

HloInstruction* NormalizeOperand(HloInstruction* inst, const Shape& ref,
                                 AddFn add) {
  inst = ConvertIfNeeded(inst, ref, add);
  if (inst->shape().has_layout() && ref.has_layout() &&
      !Layout::Equal().IgnoreTiles().IgnoreMemorySpace()(inst->shape().layout(),
                                                         ref.layout())) {
    Shape s = inst->shape();
    *s.mutable_layout() = ref.layout();
    s.mutable_layout()->clear_tiles();
    s.mutable_layout()->set_memory_space(inst->shape().layout().memory_space());
    inst = add(HloInstruction::CreateUnary(s, HloOpcode::kCopy, inst));
  }
  return inst;
}

absl::StatusOr<HloInstruction*> AlignOutputShape(HloInstruction* inst,
                                                 const Shape& target,
                                                 HloInstruction* bn,
                                                 HloInstruction* alt,
                                                 AddFn add) {
  HloInstruction* op = bn->mutable_operand(0);
  const int64_t fi = bn->feature_index(), rank = target.dimensions().size();
  const Shape s32 = ShapeUtil::MakeShape(S32, {});
  for (int64_t d = 0; d < rank; ++d) {
    const bool want = target.is_dynamic_dimension(d);
    if (inst->shape().is_dynamic_dimension(d) == want) {
      continue;
    }
    if (target.dimensions(d) > kMaxS32) {
      return InvalidArgument("BatchNorm dynamic bound %d exceeds INT32_MAX.",
                             target.dimensions(d));
    }
    Shape next = inst->shape();
    next.set_dynamic_dimension(d, want);
    HloInstruction* sz = nullptr;
    if (!want) {
      sz = add(HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(
          static_cast<int32_t>(target.dimensions(d)))));
    } else {
      int64_t sd = (rank == 1 && op->shape().dimensions().size() != 1) ? fi : d;
      HloInstruction* src =
          (!op->shape().is_dynamic_dimension(sd) && alt != nullptr &&
           alt->shape().is_dynamic_dimension(sd))
              ? alt
              : op;
      for (int64_t k = 1; !src->shape().is_dynamic_dimension(sd) && sd == fi &&
                          k < bn->operand_count();
           ++k) {
        if (bn->operand(k)->shape().dimensions().size() == 1 &&
            bn->operand(k)->shape().is_dynamic_dimension(0)) {
          src = bn->mutable_operand(k);
          sd = 0;
        }
      }
      sz = add(HloInstruction::CreateGetDimensionSize(s32, src, sd));
    }
    inst = add(HloInstruction::CreateSetDimensionSize(next, inst, sz, d));
  }
  inst = ConvertIfNeeded(inst, target, add);
  if (inst->shape().has_layout() && target.has_layout() &&
      inst->shape().layout() != target.layout()) {
    const int64_t sm = inst->shape().layout().memory_space();
    const int64_t dm = target.layout().memory_space();
    if (sm != dm &&
        (sm == Layout::kHostMemorySpace || dm == Layout::kHostMemorySpace) &&
        !Shape::Equal().IgnoreMemorySpaceInLayout()(inst->shape(), target)) {
      Shape mid = target;
      mid.mutable_layout()->set_memory_space(sm);
      inst = add(HloInstruction::CreateUnary(mid, HloOpcode::kCopy, inst));
    }
    inst = add(HloInstruction::CreateUnary(target, HloOpcode::kCopy, inst));
  }
  return inst;
}

void AssignBatchNormSharding(const HloInstruction* bn, const Shape& op_shape,
                             const Shape& feat_shape, int64_t count_before,
                             absl::Span<HloInstruction* const> added,
                             absl::Span<HloInstruction* const> outputs,
                             HloInstruction* repl) {
  CHECK_EQ(bn->parent()->instruction_count(),
           count_before + static_cast<int64_t>(added.size()));
  if (!bn->has_sharding()) {
    return;
  }
  const HloSharding& sh = bn->sharding();
  const bool tup = bn->shape().IsTuple() && sh.IsTuple();
  auto leaf = [&](int64_t i) {
    return tup ? sh.GetSubSharding(bn->shape(), {i}) : sh;
  };
  const HloSharding op_sh = leaf(0);
  const auto dev = bn->sharding_unique_device();
  const HloSharding def_sh =
      dev ? HloSharding::SingleDevice(*dev)
          : ((op_sh.IsSingleDevice() || op_sh.IsManual() || op_sh.IsUnknown() ||
              op_sh.IsUnreduced())
                 ? op_sh
                 : HloSharding::Replicate());
  const std::optional<HloSharding> feat_sh =
      (tup && bn->shape().tuple_shapes().size() >= 3 && leaf(1) == leaf(2))
          ? std::optional<HloSharding>(leaf(1))
          : std::nullopt;
  auto set = [&](HloInstruction* h, const HloSharding& s) {
    h->set_sharding(s.Validate(h->shape()).ok() ? s : def_sh);
  };
  for (HloInstruction* h : added) {
    if (h->shape().IsArray() &&
        h->shape().dimensions() == op_shape.dimensions()) {
      set(h, op_sh);
    } else if (feat_sh && h->shape().IsArray() &&
               h->shape().dimensions() == feat_shape.dimensions()) {
      set(h, *feat_sh);
    } else {
      h->set_sharding(def_sh);
    }
  }
  for (int64_t i = 0; i < static_cast<int64_t>(outputs.size()); ++i) {
    set(outputs[i], leaf(i));
  }
  repl->set_sharding(sh);
}

struct ExpandBuilder {
  HloInstruction* bn;
  Shape op_shape;
  Shape static_op_shape;
  Shape feature_shape;
  int64_t fi;
  int64_t count_before;
  std::vector<int64_t> non_feature_dims;
  std::vector<HloInstruction*> added;

  struct AddOp {
    ExpandBuilder* self;
    HloInstruction* operator()(std::unique_ptr<HloInstruction> h) const {
      return self->Add(std::move(h));
    }
  } add_fn{this};

  explicit ExpandBuilder(HloInstruction* bn)
      : bn(bn),
        op_shape(bn->operand(0)->shape()),
        static_op_shape(ShapeUtil::MakeStaticShape(op_shape)),
        feature_shape(MakeFeatureShape(op_shape, bn->feature_index())),
        fi(bn->feature_index()),
        count_before(bn->parent()->instruction_count()) {
    if (static_op_shape.has_layout() &&
        static_op_shape.layout().memory_space() == Layout::kHostMemorySpace) {
      static_op_shape.mutable_layout()->set_memory_space(
          Layout::kDefaultMemorySpace);
    }
    for (int64_t i = 0; i < op_shape.dimensions().size(); ++i) {
      if (i != fi) {
        non_feature_dims.push_back(i);
      }
    }
  }
  HloInstruction* Add(std::unique_ptr<HloInstruction> inst) {
    HloInstruction* h = bn->parent()->AddInstruction(std::move(inst));
    h->set_metadata(bn->metadata());
    added.push_back(h);
    return h;
  }
  HloInstruction* Bin(HloOpcode op, HloInstruction* a, HloInstruction* b) {
    Shape s = a->shape();
    if (s.has_layout() &&
        s.layout().memory_space() == Layout::kHostMemorySpace) {
      s.mutable_layout()->set_memory_space(Layout::kDefaultMemorySpace);
    }
    for (int64_t i = 0; i < s.dimensions().size(); ++i) {
      if (b->shape().is_dynamic_dimension(i)) {
        s.set_dynamic_dimension(i, true);
      }
    }
    return Add(HloInstruction::CreateBinary(s, op, a, b));
  }
  HloInstruction* BcastFeature(HloInstruction* h) {
    Shape s = static_op_shape;
    s.set_dynamic_dimension(fi, h->shape().is_dynamic_dimension(0));
    return Add(HloInstruction::CreateBroadcast(s, h, {fi}));
  }
  HloInstruction* Reduce(HloInstruction* src, HloInstruction* zero,
                         HloComputation* fn) {
    Shape s = feature_shape;
    s.set_dynamic_dimension(0, src->shape().is_dynamic_dimension(fi));
    return Add(
        HloInstruction::CreateReduce(s, src, zero, non_feature_dims, fn));
  }
  absl::StatusOr<HloInstruction*> Const(float v) {
    ABSL_ASSIGN_OR_RETURN(
        Literal lit, LiteralUtil::CreateR0(v).Convert(op_shape.element_type()));
    return Add(HloInstruction::CreateConstant(std::move(lit)));
  }
  AddFn Fn() const { return add_fn; }
};

// BatchNormExpanderVisitor traverses the HLO computation and rewrites BatchNorm
// operations into smaller operations.
class BatchNormExpanderVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleBatchNormTraining(HloInstruction* batch_norm) override;
  absl::Status HandleBatchNormInference(HloInstruction* batch_norm) override;
  absl::Status HandleBatchNormGrad(HloInstruction* batch_norm) override;

  static absl::StatusOr<bool> Run(HloComputation* computation,
                                  bool rewrite_training_op,
                                  bool rewrite_inference_op,
                                  bool rewrite_grad_op);

  ~BatchNormExpanderVisitor() override = default;

 private:
  explicit BatchNormExpanderVisitor(HloComputation* computation,
                                    bool rewrite_training_op,
                                    bool rewrite_inference_op,
                                    bool rewrite_grad_op)
      : computation_(computation),
        rewrite_training_op_(rewrite_training_op),
        rewrite_inference_op_(rewrite_inference_op),
        rewrite_grad_op_(rewrite_grad_op) {}

  HloComputation* GetOrCreateScalarAddComputation(
      PrimitiveType primitive_type) {
    HloComputation::Builder b("scalar_add_computation");
    Shape shape = ShapeUtil::MakeShape(primitive_type, {});
    auto lhs = b.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "scalar_lhs"));
    auto rhs = b.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "scalar_rhs"));
    auto op = b.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, lhs, rhs));
    HloComputation* comp =
        computation_->parent()->AddEmbeddedComputation(b.Build(op));
    comp->SetExecutionThread(computation_->execution_thread());
    return comp;
  }

  std::unique_ptr<HloInstruction> Rsqrt(HloInstruction* operand) {
    return HloInstruction::CreateUnary(operand->shape(), HloOpcode::kRsqrt,
                                       operand);
  }

  std::unique_ptr<HloInstruction> Mean(HloInstruction* element_count,
                                       HloInstruction* operand,
                                       AddFn add_instruction) {
    const PrimitiveType ctype = element_count->shape().element_type();
    if (ctype != operand->shape().element_type()) {
      Shape wide = ShapeUtil::ChangeElementType(operand->shape(), ctype);
      if (wide.has_layout()) {
        wide.mutable_layout()->clear_tiles();
        wide.mutable_layout()->set_element_size_in_bits(0);
      }
      auto up = add_instruction(HloInstruction::CreateConvert(wide, operand));
      auto bc = add_instruction(HloInstruction::CreateBroadcast(
          ShapeUtil::MakeStaticShape(wide), element_count, {}));
      return HloInstruction::CreateConvert(
          operand->shape(), add_instruction(HloInstruction::CreateBinary(
                                wide, HloOpcode::kDivide, up, bc)));
    }
    auto broadcast = add_instruction(HloInstruction::CreateBroadcast(
        ShapeUtil::MakeStaticShape(operand->shape()), element_count, {}));
    return HloInstruction::CreateBinary(operand->shape(), HloOpcode::kDivide,
                                        operand, broadcast);
  }

  absl::StatusOr<HloInstruction*> DynamicElementCountPerFeature(
      HloInstruction* operand, int64_t feature_index, HloInstruction* alt,
      AddFn add) {
    const Shape& shape = operand->shape();
    const PrimitiveType ptype = shape.element_type();
    bool any_zero = false, any_gt_s32 = false, prod_gt_s64 = false;
    int64_t prod = 1;
    auto use_alt = [&](int64_t i) {
      return !shape.is_dynamic_dimension(i) && alt != nullptr &&
             alt->shape().is_dynamic_dimension(i);
    };
    for (int64_t i = 0; i < shape.dimensions().size(); ++i) {
      if (i == feature_index) {
        continue;
      }
      const int64_t d = shape.dimensions(i);
      if (d > kMaxS32) {
        if (shape.is_dynamic_dimension(i) || use_alt(i)) {
          return InvalidArgument(
              "BatchNorm dynamic dimension %d bound %d exceeds INT32_MAX.", i,
              d);
        }
        any_gt_s32 = true;
      }
      if (d == 0) {
        any_zero = true;
      } else if (!prod_gt_s64 &&
                 d > std::numeric_limits<int64_t>::max() / prod) {
        prod_gt_s64 = true;
      } else if (!prod_gt_s64) {
        prod *= d;
      }
    }
    const auto max_n = MaxFiniteElementCount(ptype);
    const PrimitiveType count_ftype =
        (!any_zero && max_n && (prod_gt_s64 || prod > *max_n)) ? F32 : ptype;
    const Shape s32 = ShapeUtil::MakeShape(S32, {});
    HloInstruction* count = nullptr;
    if (!any_gt_s32 && !prod_gt_s64 && prod <= kMaxS32) {
      count = add(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1)));
      for (int64_t i = 0; i < shape.dimensions().size(); ++i) {
        if (i != feature_index) {
          auto dim = add(HloInstruction::CreateGetDimensionSize(
              s32, use_alt(i) ? alt : operand, i));
          count = add(HloInstruction::CreateBinary(s32, HloOpcode::kMultiply,
                                                   dim, count));
        }
      }
    } else if (any_zero) {
      count = add(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(0)));
    } else {
      const PrimitiveType aty = !prod_gt_s64 ? S64 : (ptype == F64 ? F64 : F32);
      const Shape ash = ShapeUtil::MakeShape(aty, {});
      auto c = [&](int64_t v) {
        return add(HloInstruction::CreateConstant(
            aty == S64 ? LiteralUtil::CreateR0<int64_t>(v)
            : aty == F64
                ? LiteralUtil::CreateR0<double>(static_cast<double>(v))
                : LiteralUtil::CreateR0<float>(static_cast<float>(v))));
      };
      count = c(1);
      for (int64_t i = 0; i < shape.dimensions().size(); ++i) {
        if (i != feature_index) {
          const int64_t d = shape.dimensions(i);
          HloInstruction* term =
              d > kMaxS32 ? c(d)
                          : add(HloInstruction::CreateConvert(
                                ash, add(HloInstruction::CreateGetDimensionSize(
                                         s32, use_alt(i) ? alt : operand, i))));
          count = add(HloInstruction::CreateBinary(ash, HloOpcode::kMultiply,
                                                   term, count));
        }
      }
    }
    return count->shape().element_type() == count_ftype
               ? count
               : add(HloInstruction::CreateConvert(
                     ShapeUtil::MakeShape(count_ftype, {}), count));
  }

  absl::Status ReplaceTuple(ExpandBuilder& eb, HloInstruction* alt,
                            HloInstruction* o0, HloInstruction* o1,
                            HloInstruction* o2) {
    HloInstruction* outs[3] = {o0, o1, o2};
    for (int64_t i = 0; i < 3; ++i) {
      ABSL_ASSIGN_OR_RETURN(
          outs[i], AlignOutputShape(outs[i], eb.bn->shape().tuple_shapes(i),
                                    eb.bn, alt, eb.Fn()));
    }
    auto tuple =
        HloInstruction::CreateVariadic(eb.bn->shape(), HloOpcode::kTuple, outs);
    AssignBatchNormSharding(eb.bn, eb.op_shape, eb.feature_shape,
                            eb.count_before, eb.added, outs, tuple.get());
    return ReplaceWithNewInstruction(eb.bn, std::move(tuple),
                                     /*preserve_sharding=*/false,
                                     /*relay_control_dependency=*/true);
  }

  HloComputation* computation_;
  bool rewrite_training_op_;
  bool rewrite_inference_op_;
  bool rewrite_grad_op_;
};

}  // namespace

absl::StatusOr<bool> BatchNormExpanderVisitor::Run(HloComputation* computation,
                                                   bool rewrite_training_op,
                                                   bool rewrite_inference_op,
                                                   bool rewrite_grad_op) {
  BatchNormExpanderVisitor visitor(computation, rewrite_training_op,
                                   rewrite_inference_op, rewrite_grad_op);
  ABSL_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

absl::Status BatchNormExpanderVisitor::HandleBatchNormTraining(
    HloInstruction* batch_norm) {
  if (!rewrite_training_op_) {
    return absl::OkStatus();
  }
  HloInstruction* operand = batch_norm->mutable_operand(0);
  const PrimitiveType ptype = operand->shape().element_type();
  if (ptype == F8E8M0FNU) {
    return Unimplemented("BatchNormExpander does not support F8E8M0FNU.");
  }
  ExpandBuilder eb(batch_norm);
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * elements_per_feature,
      DynamicElementCountPerFeature(operand, eb.fi,
                                    /*alt=*/nullptr, eb.Fn()));
  HloInstruction* scale = NormalizeOperand(batch_norm->mutable_operand(1),
                                           eb.feature_shape, eb.Fn());
  HloInstruction* offset = NormalizeOperand(batch_norm->mutable_operand(2),
                                            eb.feature_shape, eb.Fn());
  ABSL_ASSIGN_OR_RETURN(HloInstruction * zero, eb.Const(0.0f));
  ABSL_ASSIGN_OR_RETURN(HloInstruction * eps_scalar,
                        eb.Const(batch_norm->epsilon()));
  auto epsilon = eb.Add(
      HloInstruction::CreateBroadcast(eb.static_op_shape, eps_scalar, {}));
  auto scale_broadcasted = eb.BcastFeature(scale);
  auto offset_broadcasted = eb.BcastFeature(offset);
  HloComputation* add_fn = GetOrCreateScalarAddComputation(ptype);

  // Compute E[X], E[X^2], and Var[X] clamped to >= 0 against rounding error.
  auto operand_squared = eb.Bin(HloOpcode::kMultiply, operand, operand);
  auto sum = eb.Reduce(operand, zero, add_fn);
  auto squared_sum = eb.Reduce(operand_squared, zero, add_fn);
  auto mean = eb.Add(Mean(elements_per_feature, sum, eb.Fn()));
  auto mean_broadcasted = eb.BcastFeature(mean);
  auto square_mean = eb.Add(Mean(elements_per_feature, squared_sum, eb.Fn()));
  auto mean_square = eb.Bin(HloOpcode::kMultiply, mean, mean);
  auto raw_var = eb.Bin(HloOpcode::kSubtract, square_mean, mean_square);
  auto zero_feature = eb.Add(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeStaticShape(eb.feature_shape), zero, {}));
  auto var = eb.Bin(HloOpcode::kMaximum, raw_var, zero_feature);

  // Compute (X - E[X]) * rsqrt(Var[X] + epsilon) * scale + offset.
  auto rsqrt_var =
      eb.Add(Rsqrt(eb.Bin(HloOpcode::kAdd, eb.BcastFeature(var), epsilon)));
  auto normalized = eb.Bin(
      HloOpcode::kMultiply,
      eb.Bin(HloOpcode::kSubtract, operand, mean_broadcasted), rsqrt_var);
  auto shifted_normalized =
      eb.Bin(HloOpcode::kAdd,
             eb.Bin(HloOpcode::kMultiply, normalized, scale_broadcasted),
             offset_broadcasted);
  return ReplaceTuple(eb, /*alt=*/nullptr, shifted_normalized, mean, var);
}

absl::Status BatchNormExpanderVisitor::HandleBatchNormInference(
    HloInstruction* batch_norm) {
  if (!rewrite_inference_op_) {
    return absl::OkStatus();
  }
  HloInstruction* operand = batch_norm->mutable_operand(0);
  if (operand->shape().element_type() == F8E8M0FNU) {
    return Unimplemented("BatchNormExpander does not support F8E8M0FNU.");
  }
  ExpandBuilder eb(batch_norm);
  HloInstruction* scale = NormalizeOperand(batch_norm->mutable_operand(1),
                                           eb.feature_shape, eb.Fn());
  HloInstruction* offset = NormalizeOperand(batch_norm->mutable_operand(2),
                                            eb.feature_shape, eb.Fn());
  HloInstruction* mean = NormalizeOperand(batch_norm->mutable_operand(3),
                                          eb.feature_shape, eb.Fn());
  HloInstruction* var = NormalizeOperand(batch_norm->mutable_operand(4),
                                         eb.feature_shape, eb.Fn());
  ABSL_ASSIGN_OR_RETURN(HloInstruction * eps_scalar,
                        eb.Const(batch_norm->epsilon()));
  auto epsilon = eb.Add(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeStaticShape(eb.feature_shape), eps_scalar, {}));
  auto true_scale =
      eb.Bin(HloOpcode::kMultiply, scale,
             eb.Add(Rsqrt(eb.Bin(HloOpcode::kAdd, var, epsilon))));
  auto true_shift = eb.Bin(HloOpcode::kSubtract, offset,
                           eb.Bin(HloOpcode::kMultiply, mean, true_scale));
  auto shifted =
      eb.Bin(HloOpcode::kAdd,
             eb.Bin(HloOpcode::kMultiply, operand, eb.BcastFeature(true_scale)),
             eb.BcastFeature(true_shift));
  ABSL_ASSIGN_OR_RETURN(
      shifted, AlignOutputShape(shifted, batch_norm->shape(), batch_norm,
                                /*alt=*/nullptr, eb.Fn()));
  AssignBatchNormSharding(batch_norm, eb.op_shape, eb.feature_shape,
                          eb.count_before, eb.added, {shifted}, shifted);
  ABSL_RETURN_IF_ERROR(ReplaceInstruction(batch_norm, shifted));
  return absl::OkStatus();
}

absl::Status BatchNormExpanderVisitor::HandleBatchNormGrad(
    HloInstruction* batch_norm) {
  if (!rewrite_grad_op_) {
    return absl::OkStatus();
  }
  HloInstruction* activation = batch_norm->mutable_operand(0);
  const PrimitiveType ptype = activation->shape().element_type();
  if (ptype == F8E8M0FNU) {
    return Unimplemented("BatchNormExpander does not support F8E8M0FNU.");
  }
  ExpandBuilder eb(batch_norm);
  HloInstruction* raw_grad = batch_norm->mutable_operand(4);
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * elements_per_feature,
      DynamicElementCountPerFeature(activation, eb.fi, raw_grad, eb.Fn()));
  HloInstruction* scale = NormalizeOperand(batch_norm->mutable_operand(1),
                                           eb.feature_shape, eb.Fn());
  HloInstruction* mean = NormalizeOperand(batch_norm->mutable_operand(2),
                                          eb.feature_shape, eb.Fn());
  HloInstruction* variance = NormalizeOperand(batch_norm->mutable_operand(3),
                                              eb.feature_shape, eb.Fn());
  HloInstruction* grad_output =
      NormalizeOperand(raw_grad, eb.op_shape, eb.Fn());
  ABSL_ASSIGN_OR_RETURN(HloInstruction * zero, eb.Const(0.0f));
  ABSL_ASSIGN_OR_RETURN(HloInstruction * eps_scalar,
                        eb.Const(batch_norm->epsilon()));
  auto eps_act = eb.Add(
      HloInstruction::CreateBroadcast(eb.static_op_shape, eps_scalar, {}));
  auto eps_feat = eb.Add(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeStaticShape(eb.feature_shape), eps_scalar, {}));
  for (int64_t i : eb.non_feature_dims) {
    if (eb.op_shape.is_dynamic_dimension(i) &&
        !grad_output->shape().is_dynamic_dimension(i)) {
      Shape dyn = grad_output->shape();
      dyn.set_dynamic_dimension(i, true);
      auto sz = eb.Add(HloInstruction::CreateGetDimensionSize(
          ShapeUtil::MakeShape(S32, {}), activation, i));
      grad_output = eb.Add(
          HloInstruction::CreateSetDimensionSize(dyn, grad_output, sz, i));
    }
  }

  auto scale_bc = eb.BcastFeature(scale);
  auto var_bc = eb.BcastFeature(variance);
  auto mean_bc = eb.BcastFeature(mean);
  auto rsqrt_bc = eb.Add(Rsqrt(eb.Bin(HloOpcode::kAdd, var_bc, eps_act)));
  auto rsqrt_feat = eb.Add(Rsqrt(eb.Bin(HloOpcode::kAdd, variance, eps_feat)));
  auto centered = eb.Bin(HloOpcode::kSubtract, activation, mean_bc);
  auto dy_centered = eb.Bin(HloOpcode::kMultiply, grad_output, centered);
  HloComputation* add_fn = GetOrCreateScalarAddComputation(ptype);
  auto sum_dy_centered = eb.Reduce(dy_centered, zero, add_fn);
  auto grad_beta = eb.Reduce(grad_output, zero, add_fn);
  auto grad_scale = eb.Bin(HloOpcode::kMultiply, sum_dy_centered, rsqrt_feat);

  // When N is widened to avoid narrow-float overflow, divide 1D reductions by N
  // before broadcasting and skip forming grad_output * N.
  const bool wide_count = elements_per_feature->shape().element_type() != ptype;
  auto i2 = eb.BcastFeature(
      wide_count ? eb.Add(Mean(elements_per_feature, grad_beta, eb.Fn()))
                 : grad_beta);
  auto i3 = eb.BcastFeature(
      wide_count ? eb.Add(Mean(elements_per_feature, sum_dy_centered, eb.Fn()))
                 : sum_dy_centered);
  auto i5 =
      eb.Bin(HloOpcode::kDivide, eb.Bin(HloOpcode::kMultiply, i3, centered),
             eb.Bin(HloOpcode::kAdd, var_bc, eps_act));
  auto factor = eb.Bin(HloOpcode::kMultiply, scale_bc, rsqrt_bc);
  HloInstruction* i1 = grad_output;
  if (!wide_count) {
    factor = eb.Add(Mean(elements_per_feature, factor, eb.Fn()));
    i1 = eb.Bin(HloOpcode::kMultiply, grad_output,
                eb.Add(HloInstruction::CreateBroadcast(
                    eb.static_op_shape, elements_per_feature, {})));
  }
  auto i6 =
      eb.Bin(HloOpcode::kSubtract, eb.Bin(HloOpcode::kSubtract, i1, i2), i5);
  auto grad_activation = eb.Bin(HloOpcode::kMultiply, factor, i6);
  return ReplaceTuple(eb, raw_grad, grad_activation, grad_scale, grad_beta);
}

absl::StatusOr<bool> BatchNormExpander::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      2, "BatchNormExpander::RunImpl(), before:\n" + module->ToString());
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    ABSL_ASSIGN_OR_RETURN(
        bool comp_changed,
        BatchNormExpanderVisitor::Run(computation, rewrite_training_op_,
                                      rewrite_inference_op_, rewrite_grad_op_));
    changed |= comp_changed;
  }
  XLA_VLOG_LINES(2,
                 "BatchNormExpander::RunImpl(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
