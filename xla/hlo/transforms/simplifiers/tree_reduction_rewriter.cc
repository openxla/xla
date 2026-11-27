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

#include "xla/hlo/transforms/simplifiers/tree_reduction_rewriter.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/builder/padding.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

bool IsSafeForTreeReduction(const HloComputation* to_apply,
                            const HloInstruction* initial_value,
                            PrimitiveType elem_type) {
  if (to_apply == nullptr || to_apply->num_parameters() != 2 ||
      to_apply->instruction_count() != 3) {
    return false;
  }
  const HloInstruction* p0 = to_apply->parameter_instruction(0);
  const HloInstruction* p1 = to_apply->parameter_instruction(1);
  const HloInstruction* root = to_apply->root_instruction();
  if (!ShapeUtil::IsScalarWithElementType(p0->shape(), elem_type) ||
      !ShapeUtil::IsScalarWithElementType(p1->shape(), elem_type) ||
      !ShapeUtil::IsScalarWithElementType(root->shape(), elem_type) ||
      root->operand_count() != 2) {
    return false;
  }
  const bool matches_params =
      (root->operand(0) == p0 && root->operand(1) == p1) ||
      (root->operand(0) == p1 && root->operand(1) == p0);
  if (!matches_params) {
    return false;
  }
  const bool is_constant_init =
      initial_value->opcode() == HloOpcode::kConstant &&
      ShapeUtil::IsScalarWithElementType(initial_value->shape(), elem_type);
  const bool is_pred_or_integral =
      elem_type == PRED || primitive_util::IsIntegralType(elem_type);
  switch (root->opcode()) {
    case HloOpcode::kMinimum:
    case HloOpcode::kMaximum:
      return !primitive_util::IsComplexType(elem_type);
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
      return is_pred_or_integral;
    case HloOpcode::kAdd:
      return elem_type != PRED && elem_type != F8E8M0FNU && is_constant_init &&
             initial_value->literal().IsZero({});
    case HloOpcode::kMultiply:
      return elem_type != PRED && !primitive_util::IsComplexType(elem_type) &&
             is_constant_init && initial_value->literal().IsAll(1);
    case HloOpcode::kXor:
      return is_pred_or_integral && is_constant_init &&
             initial_value->literal().IsZero({});
    default:
      return false;
  }
}

}  // namespace

class ReductionRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  ReductionRewriterVisitor(int64_t reduce_window_size, HloPredicate filter)
      : reduce_window_size_(reduce_window_size), filter_(std::move(filter)) {}

  absl::Status HandleReduce(HloInstruction* hlo) override {
    if (filter_ && !filter_(hlo)) {
      return absl::OkStatus();
    }
    if (hlo->operand_count() != 2 || !hlo->shape().IsArray()) {
      VLOG(1) << "Skipping rewrite for variadic reduction";
      return absl::OkStatus();
    }

    HloInstruction* reduced_op = hlo->mutable_operand(0);
    HloInstruction* initial_value = hlo->mutable_operand(1);
    const Shape& input_shape = reduced_op->shape();
    const Shape& reduce_shape = hlo->shape();
    if (!input_shape.IsArray() || input_shape.is_unbounded_dynamic() ||
        hlo->dimensions().empty()) {
      return absl::OkStatus();
    }
    const PrimitiveType elem_type = input_shape.element_type();
    if (reduce_shape.element_type() != elem_type ||
        !ShapeUtil::IsScalarWithElementType(initial_value->shape(),
                                            elem_type)) {
      return absl::OkStatus();
    }

    bool has_dim_exceeding_window = false;
    for (int64_t reduced_dim : hlo->dimensions()) {
      if (input_shape.is_dynamic_dimension(reduced_dim) ||
          input_shape.dimensions(reduced_dim) <= 0) {
        return absl::OkStatus();
      }
      if (input_shape.dimensions(reduced_dim) > reduce_window_size_) {
        has_dim_exceeding_window = true;
      }
    }
    if (!has_dim_exceeding_window) {
      VLOG(1) << "Skipping tree reduction rewrite: all reduced dimensions are "
                 "smaller than the window size.";
      return absl::OkStatus();
    }
    if (!IsSafeForTreeReduction(hlo->to_apply(), initial_value, elem_type)) {
      return absl::OkStatus();
    }

    std::vector<int64_t> window_dimensions;
    std::vector<int64_t> window_strides;
    window_dimensions.reserve(input_shape.dimensions().size());
    window_strides.reserve(input_shape.dimensions().size());
    for (int64_t dim_idx = 0; dim_idx < input_shape.dimensions().size();
         dim_idx++) {
      if (!absl::c_linear_search(hlo->dimensions(), dim_idx)) {
        window_dimensions.push_back(1);
        window_strides.push_back(1);
        continue;
      }

      int64_t window_size_for_dim =
          std::min(input_shape.dimensions(dim_idx), reduce_window_size_);

      window_dimensions.push_back(window_size_for_dim);
      window_strides.push_back(window_size_for_dim);
    }

    std::vector<std::pair<int64_t, int64_t>> padding =
        MakePadding(input_shape.dimensions(), window_dimensions, window_strides,
                    Padding::kSame);

    ABSL_ASSIGN_OR_RETURN(
        Window window, ShapeInference::InferWindowFromDimensions(
                           window_dimensions, window_strides, padding, {}, {}));

    ABSL_ASSIGN_OR_RETURN(Shape intermediate_shape,
                          ShapeInference::InferReduceWindowShape(
                              input_shape, initial_value->shape(), window));
    if (input_shape.has_layout() &&
        input_shape.layout().element_size_in_bits() != 0 &&
        intermediate_shape.has_layout()) {
      intermediate_shape.mutable_layout()->set_element_size_in_bits(
          input_shape.layout().element_size_in_bits());
    }

    HloInstruction* reduce_window = hlo->parent()->AddInstruction(
        HloInstruction::CreateReduceWindow(intermediate_shape, reduced_op,
                                           initial_value, window,
                                           hlo->to_apply()),
        &hlo->metadata());
    for (HloInstruction* control_pred : hlo->control_predecessors()) {
      ABSL_RETURN_IF_ERROR(control_pred->AddControlDependencyTo(reduce_window));
    }

    std::unique_ptr<HloInstruction> new_output =
        HloInstruction::CreateReduce(reduce_shape, reduce_window, initial_value,
                                     hlo->dimensions(), hlo->to_apply());

    return ReplaceWithNewInstruction(hlo, std::move(new_output),
                                     /*preserve_sharding=*/false,
                                     /*relay_control_dependency=*/true);
  }

 private:
  int64_t reduce_window_size_;
  HloPredicate filter_;
};

absl::StatusOr<bool> TreeReductionRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (reduce_window_size_ <= 1) {
    return false;
  }
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    ReductionRewriterVisitor visitor(reduce_window_size_, filter_);
    ABSL_RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }

  return changed;
}

}  // end namespace xla
