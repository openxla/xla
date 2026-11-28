/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/reduce_decomposer.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {

namespace {

bool HasAllMinorToMajorLayouts(const HloReduceInstruction* reduce) {
  if (!LayoutUtil::HasMinorToMajorSetInLayout(reduce->shape())) {
    return false;
  }
  for (const HloInstruction* input : reduce->inputs()) {
    if (!LayoutUtil::HasMinorToMajorSetInLayout(input->shape())) {
      return false;
    }
  }
  return true;
}

HloInstruction* MakeLayoutCopy(HloInstruction* from, const Shape& to) {
  HloInstruction* copy = MakeCopyHlo(from, to);
  copy->set_frontend_attributes({});
  return copy;
}

// Enforces that all inputs to a variadic reduction share the same
// minor-to-major dimension order while preserving output layouts.
class VariadicReductionLayoutEqualizer : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleReduce(HloInstruction* hlo) override {
    auto* reduce = Cast<HloReduceInstruction>(hlo);
    if (reduce->input_count() <= 1 || !HasAllMinorToMajorLayouts(reduce)) {
      return absl::OkStatus();
    }
    const Layout& first_layout = reduce->inputs()[0]->shape().layout();
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(reduce->operand_count());
    bool changed = false;
    for (HloInstruction* input : reduce->inputs()) {
      const Shape& input_s = input->shape();
      if (!Layout::Equal().MinorToMajorOnly()(first_layout, input_s.layout())) {
        Shape new_input_s = input_s;
        Layout* layout = new_input_s.mutable_layout();
        layout->mutable_minor_to_major()->assign(
            first_layout.minor_to_major().begin(),
            first_layout.minor_to_major().end());
        layout->clear_tiles();
        layout->clear_split_configs();
        layout->clear_physical_shape();
        layout->set_tail_padding_alignment_in_elements(1);
        new_operands.push_back(MakeLayoutCopy(input, new_input_s));
        changed = true;
      } else {
        new_operands.push_back(input);
      }
    }

    if (changed) {
      new_operands.insert(new_operands.end(), reduce->init_values().begin(),
                          reduce->init_values().end());
      HloInstruction* new_reduce = reduce->parent()->AddInstruction(
          reduce->CloneWithNewOperands(reduce->shape(), new_operands));
      ABSL_RETURN_IF_ERROR(ReplaceInstruction(
          reduce, new_reduce, /*preserve_frontend_attributes=*/false));
    }

    return absl::OkStatus();
  }
};

class ReduceDecomposerVisitor : public DfsHloRewriteVisitor {
 public:
  explicit ReduceDecomposerVisitor(HloPredicate custom_layout_allowed)
      : custom_layout_allowed_(std::move(custom_layout_allowed)) {}

  absl::Status HandleReduce(HloInstruction* hlo) override {
    auto* reduce = Cast<HloReduceInstruction>(hlo);
    const Shape& shape = reduce->shape();
    if (!HasAllMinorToMajorLayouts(reduce) ||
        (custom_layout_allowed_ && custom_layout_allowed_(reduce))) {
      return absl::OkStatus();
    }

    std::vector<Shape> output_shapes;
    if (shape.IsTuple()) {
      output_shapes.assign(shape.tuple_shapes().begin(),
                           shape.tuple_shapes().end());
    } else {
      output_shapes.push_back(shape);
    }
    TF_RET_CHECK(!output_shapes.empty());
    TF_RET_CHECK(static_cast<int64_t>(output_shapes.size()) ==
                 reduce->input_count());

    std::vector<Shape> expected_shapes(reduce->input_count());
    const Layout& first_layout = reduce->inputs()[0]->shape().layout();
    for (int64_t i = 0; i < reduce->input_count(); ++i) {
      TF_RET_CHECK(Layout::Equal().MinorToMajorOnly()(
          reduce->inputs()[i]->shape().layout(), first_layout));
      expected_shapes[i] = ExpectedOutputShape(reduce, output_shapes[i], i);
    }

    Shape expected_tuple_shape;
    if (shape.IsTuple()) {
      ABSL_ASSIGN_OR_RETURN(
          expected_tuple_shape,
          ShapeUtil::MakeValidatedTupleShape(expected_shapes));
    } else {
      expected_tuple_shape = expected_shapes[0];
    }
    if (expected_tuple_shape == shape) {
      return absl::OkStatus();  // Nothing to do here.
    }

    HloComputation* comp = reduce->parent();
    HloInstruction* r_prime = comp->AddInstruction(
        reduce->CloneWithNewOperands(expected_tuple_shape, reduce->operands()));

    // Handle non-variadic reductions.
    if (!shape.IsTuple()) {
      if (reduce->user_count() == 1 && reduce != comp->root_instruction() &&
          reduce->users()[0]->opcode() == HloOpcode::kConvert &&
          LayoutUtil::HasMinorToMajorSetInLayout(reduce->users()[0]->shape()) &&
          reduce->users()[0]->shape().layout().memory_space() ==
              r_prime->shape().layout().memory_space()) {
        // Keep reduction+convert ops adjacent and insert the copy after them.
        HloInstruction* convert = reduce->users()[0];
        for (HloInstruction* pred : reduce->control_predecessors()) {
          ABSL_RETURN_IF_ERROR(pred->AddControlDependencyTo(r_prime));
        }
        for (HloInstruction* succ : reduce->control_successors()) {
          if (succ != convert) {
            ABSL_RETURN_IF_ERROR(r_prime->AddControlDependencyTo(succ));
          }
        }
        ABSL_RETURN_IF_ERROR(reduce->DropAllControlDeps());

        Shape conv_shape = ShapeUtil::ChangeElementType(
            r_prime->shape(), convert->shape().element_type());
        conv_shape.mutable_layout()->set_element_size_in_bits(
            convert->shape().layout().element_size_in_bits());
        HloInstruction* to_copy = comp->AddInstruction(
            convert->CloneWithNewOperands(conv_shape, {r_prime}));
        HloInstruction* out = (to_copy->shape() == convert->shape())
                                  ? to_copy
                                  : MakeLayoutCopy(to_copy, convert->shape());
        ABSL_RETURN_IF_ERROR(convert->CopyAllControlDepsTo(to_copy, out));
        ABSL_RETURN_IF_ERROR(convert->DropAllControlDeps());
        return ReplaceInstruction(convert, out,
                                  /*preserve_frontend_attributes=*/false);
      }
      HloInstruction* copy = MakeLayoutCopy(r_prime, shape);
      ABSL_RETURN_IF_ERROR(reduce->CopyAllControlDepsTo(r_prime, copy));
      ABSL_RETURN_IF_ERROR(reduce->DropAllControlDeps());
      return ReplaceInstruction(reduce, copy,
                                /*preserve_frontend_attributes=*/false);
    }

    // Handle tuple and variadic reductions.
    std::optional<HloSharding> norm_sharding;
    if (reduce->has_sharding()) {
      norm_sharding = reduce->sharding().NormalizeTupleSharding(shape);
    }
    std::vector<HloInstruction*> outputs;
    outputs.reserve(output_shapes.size());
    for (int64_t i = 0; i < reduce->input_count(); ++i) {
      ABSL_ASSIGN_OR_RETURN(
          HloInstruction * from,
          MakeGetTupleElementHlo(r_prime, i, &reduce->metadata()));
      if (norm_sharding.has_value() &&
          static_cast<int64_t>(norm_sharding->tuple_elements().size()) ==
              reduce->input_count()) {
        from->set_sharding(norm_sharding->GetSubSharding(shape, {i}));
      }
      outputs.push_back(output_shapes[i] == expected_shapes[i]
                            ? from
                            : MakeLayoutCopy(from, output_shapes[i]));
    }
    HloInstruction* out =
        comp->AddInstruction(HloInstruction::CreateTuple(outputs));
    if (reduce->has_sharding()) {
      out->set_sharding(reduce->sharding());
    }
    ABSL_RETURN_IF_ERROR(reduce->CopyAllControlDepsTo(r_prime, out));
    ABSL_RETURN_IF_ERROR(reduce->DropAllControlDeps());
    return ReplaceInstruction(reduce, out,
                              /*preserve_frontend_attributes=*/false);
  }

 private:
  Shape ExpectedOutputShape(const HloReduceInstruction* reduce,
                            const Shape& output_shape, int64_t input_idx) {
    Shape expected = ShapeUtil::DeleteDimensions(
        reduce->dimensions(), reduce->inputs()[input_idx]->shape());
    if (expected.element_type() != output_shape.element_type()) {
      expected =
          ShapeUtil::ChangeElementType(expected, output_shape.element_type());
      expected.mutable_layout()->set_element_size_in_bits(
          output_shape.layout().element_size_in_bits());
    }
    return expected;
  }

  HloPredicate custom_layout_allowed_;
};

}  // namespace

absl::StatusOr<bool> ReduceDecomposer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  ABSL_ASSIGN_OR_RETURN(bool changed1,
                        VariadicReductionLayoutEqualizer{}.RunOnModule(
                            module, execution_threads));
  ABSL_ASSIGN_OR_RETURN(
      bool changed2,
      ReduceDecomposerVisitor{custom_layout_allowed_}.RunOnModule(
          module, execution_threads));
  return changed1 || changed2;
}

}  // namespace xla
