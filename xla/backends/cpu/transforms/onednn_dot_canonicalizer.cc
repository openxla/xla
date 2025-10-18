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

#include "xla/backends/cpu/transforms/onednn_dot_canonicalizer.h"

#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/pattern_matcher.h"
#include "xla/status_macros.h"

namespace xla {
namespace cpu {

class OneDnnDotCanonicalizerVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleDot(HloInstruction* dot_instr) override {
    HloInstruction* lhs = dot_instr->mutable_operand(0);
    HloInstruction* rhs = dot_instr->mutable_operand(1);
    DotDimensionNumbers dim_numbers = dot_instr->dot_dimension_numbers();

    auto lhs_batch_dims = dim_numbers.lhs_batch_dimensions();
    auto lhs_contraction_dims = dim_numbers.lhs_contracting_dimensions();
    bool is_lhs_vector = lhs->shape().dimensions_size() ==
                         (lhs_batch_dims.size() + lhs_contraction_dims.size());

    auto rhs_batch_dims = dim_numbers.rhs_batch_dimensions();
    auto rhs_contraction_dims = dim_numbers.rhs_contracting_dimensions();
    bool is_rhs_vector = rhs->shape().dimensions_size() ==
                         (rhs_batch_dims.size() + rhs_contraction_dims.size());

    if (!is_lhs_vector && !is_rhs_vector) return absl::OkStatus();

    std::vector<int64_t> adjusted_lhs_dims(lhs->shape().dimensions().begin(),
                                           lhs->shape().dimensions().end());
    std::vector<int64_t> adjusted_rhs_dims(rhs->shape().dimensions().begin(),
                                           rhs->shape().dimensions().end());
    std::vector<int64_t> adjusted_dot_dims(
        dot_instr->shape().dimensions().begin(),
        dot_instr->shape().dimensions().end());

    if (is_lhs_vector) {
      auto lhs_it = adjusted_lhs_dims.begin() + lhs_batch_dims.size();
      adjusted_lhs_dims.insert(lhs_it, 1, 1);
      auto result_it = adjusted_dot_dims.begin() + lhs_batch_dims.size();
      adjusted_dot_dims.insert(result_it, 1, 1);
      auto lhs_contraction_dim =
          dot_instr->dot_dimension_numbers().lhs_contracting_dimensions(0);
      dim_numbers.set_lhs_contracting_dimensions(0, lhs_contraction_dim + 1);
      lhs = lhs->AddInstruction(HloInstruction::CreateBitcast(
          ShapeUtil::MakeShape(lhs->shape().element_type(), adjusted_lhs_dims),
          lhs));
    }

    if (is_rhs_vector) {
      auto it = adjusted_rhs_dims.end();
      adjusted_rhs_dims.insert(it, 1, 1);
      auto result_it = adjusted_dot_dims.end();
      adjusted_dot_dims.insert(result_it, 1, 1);
      rhs = rhs->AddInstruction(HloInstruction::CreateBitcast(
          ShapeUtil::MakeShape(rhs->shape().element_type(), adjusted_rhs_dims),
          rhs));
    }

    HloInstruction* adjusted_dot =
        dot_instr->AddInstruction(HloInstruction::CreateDot(
            ShapeUtil::MakeShape(dot_instr->shape().element_type(),
                                 adjusted_dot_dims),
            lhs, rhs, dim_numbers, dot_instr->precision_config()));

    HloInstruction* replacement_instr = adjusted_dot->AddInstruction(
        HloInstruction::CreateBitcast(dot_instr->shape(), adjusted_dot));

    TF_RETURN_IF_ERROR(ReplaceInstruction(dot_instr, replacement_instr));
    return absl::OkStatus();
  }
};

absl::StatusOr<bool> OneDnnDotCanonicalizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      3, "OneDnnDotCanonicalizer::Run(), before:\n" + module->ToString());
  OneDnnDotCanonicalizerVisitor visitor;
  TF_ASSIGN_OR_RETURN(auto result,
                      visitor.RunOnModule(module, execution_threads));
  XLA_VLOG_LINES(
      3, "OneDnnDotCanonicalizer::Run(), after:\n" + module->ToString());
  return result;
}

}  // namespace cpu
}  // namespace xla
