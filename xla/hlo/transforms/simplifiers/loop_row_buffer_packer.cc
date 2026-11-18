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

#include "xla/hlo/transforms/simplifiers/loop_row_buffer_packer.h"

#include <cstdint>
#include <iterator>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/memory_annotations.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/while_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace m = ::xla::match;

// Wider rows keep their own update: packing copies every row once more.
constexpr int64_t kMaxRowElements = 128;
// Smaller groups keep their updates: the concatenate would not pay for itself.
constexpr int64_t kMinBuffersPerGroup = 8;

// Whether `value` is annotated to move to host memory. HostOffloader accepts
// only these operations between the annotation and the write it offloads.
bool IsMovedToHost(const HloInstruction* value) {
  while (value->opcode() == HloOpcode::kReshape ||
         value->opcode() == HloOpcode::kBitcast ||
         value->opcode() == HloOpcode::kCopy) {
    value = value->operand(0);
  }
  return value->IsCustomCall(memory_annotations::kMoveToHostCustomCallTarget);
}

// Whether `instruction` writes one whole row into a static [N] or [N, m]
// buffer, at a scalar row index. The row lands at column 0 whatever the column
// index: dynamic-update-slice clamps the start so that the update fits.
bool IsRowWrite(const HloInstruction* instruction) {
  // A packed write would not carry the attributes or the ordering of the
  // writes it replaces, and would part a host memory annotation from its
  // write.
  if (instruction->opcode() != HloOpcode::kDynamicUpdateSlice ||
      !instruction->frontend_attributes().map().empty() ||
      instruction->HasControlDependencies() ||
      IsMovedToHost(instruction->operand(1))) {
    return false;
  }
  const Shape& update = instruction->operand(1)->shape();
  Shape row = instruction->shape();
  if (row.is_dynamic() || update.is_dynamic() || row.dimensions().empty() ||
      row.dimensions().size() > 2) {
    return false;
  }
  row.set_dimensions(0, 1);
  return ShapeUtil::Compatible(update, row) &&
         ShapeUtil::IsScalar(instruction->operand(2)->shape());
}

// For each element of the loop state: the row write that is all the loop does
// with the element, or null. That write is the only instruction of the body or
// the condition that uses the element, and the same element of the next state
// is the only use of the write, so no other value depends on the buffer.
std::vector<HloInstruction*> FindRowWrites(const HloInstruction* while_instr) {
  HloInstruction* root = while_instr->while_body()->root_instruction();
  // The next state must be a tuple instruction that only the loop reads:
  // another user would see the old buffers, which the packed loop no longer
  // writes.
  if (root->opcode() != HloOpcode::kTuple || root->user_count() != 0) {
    return {};
  }
  // How many instructions use each element. The root counts like any other:
  // it would hand the old buffer to another element of the next state.
  std::vector<int64_t> uses(root->operand_count(), 0);
  for (const HloComputation* computation :
       {while_instr->while_body(), while_instr->while_condition()}) {
    for (const HloInstruction* element :
         computation->parameter_instruction(0)->users()) {
      if (element->opcode() != HloOpcode::kGetTupleElement) {
        return {};
      }
      uses[element->tuple_index()] += element->user_count();
    }
  }
  std::vector<HloInstruction*> writes(root->operand_count(), nullptr);
  for (int64_t index = 0; index < root->operand_count(); ++index) {
    HloInstruction* write = root->mutable_operand(index);
    // `write` is the only user of the element, which it updates, and this
    // operand of the root the only use of `write`.
    if (IsRowWrite(write) && uses[index] == 1 &&
        Match(write->operand(0), m::GetTupleElement(m::Parameter(0), index)) &&
        write->user_count() == 1 &&
        absl::c_count(root->operands(), write) == 1) {
      writes[index] = write;
    }
  }
  return writes;
}

// `value` with tuple element selections of tuple instructions resolved.
HloInstruction* LookThroughTuples(HloInstruction* value) {
  while (value->opcode() == HloOpcode::kGetTupleElement &&
         value->operand(0)->opcode() == HloOpcode::kTuple) {
    value = value->mutable_operand(0)->mutable_operand(value->tuple_index());
  }
  return value;
}

// Whether the initial value `buffer` is annotated to live in host memory, as
// it is or by a row written to it.
bool IsHostBuffer(HloInstruction* buffer) {
  buffer = LookThroughTuples(buffer);
  if (buffer->opcode() == HloOpcode::kDynamicUpdateSlice) {
    return IsMovedToHost(buffer->operand(1)) ||
           IsHostBuffer(buffer->mutable_operand(0));
  }
  return IsMovedToHost(buffer);
}

// `values`, arrays of `rows` rows each, side by side as the columns of one
// array.
HloInstruction* Concatenate(HloComputation* computation,
                            absl::Span<HloInstruction* const> values,
                            int64_t rows) {
  const PrimitiveType type = values.front()->shape().element_type();
  std::vector<HloInstruction*> pieces;
  pieces.reserve(values.size());
  int64_t columns = 0;
  for (HloInstruction* value : values) {
    const int64_t width = ShapeUtil::ElementsIn(value->shape()) / rows;
    pieces.push_back(
        value->shape().dimensions().size() == 2
            ? value
            : computation->AddInstruction(HloInstruction::CreateReshape(
                  ShapeUtil::MakeShape(type, {rows, width}), value)));
    columns += width;
  }
  return computation->AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeShape(type, {rows, columns}), pieces, 1));
}

// `buffer` with `row` written at row `index`.
HloInstruction* WriteRow(HloInstruction* buffer, HloInstruction* row,
                         HloInstruction* index) {
  HloComputation* computation = buffer->parent();
  HloInstruction* zero =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(index->shape().element_type())));
  return computation->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      buffer->shape(), buffer, row, {index, zero}));
}

// The buffer of shape `packed_shape` that holds the buffers `values` as its
// columns.
HloInstruction* PackValues(HloComputation* computation,
                           std::vector<HloInstruction*> values,
                           const Shape& packed_shape) {
  for (HloInstruction*& value : values) {
    value = LookThroughTuples(value);
  }
  HloInstruction* first = values.front();
  // Buffers filled with one scalar pack into a buffer filled with it.
  if (absl::c_all_of(values, [&](const HloInstruction* value) {
        return value->opcode() == HloOpcode::kBroadcast &&
               ShapeUtil::IsScalar(value->operand(0)->shape()) &&
               value->operand(0) == first->operand(0);
      })) {
    return computation->AddInstruction(HloInstruction::CreateBroadcast(
        packed_shape, first->mutable_operand(0), {}));
  }
  // Buffers with one row written at the same index, such as the first
  // iteration of a peeled loop, pack into the packed bases with the packed row
  // written.
  if (absl::c_all_of(values, [&](const HloInstruction* value) {
        return IsRowWrite(value) && value->operand(2) == first->operand(2);
      })) {
    std::vector<HloInstruction*> bases, rows;
    for (HloInstruction* value : values) {
      bases.push_back(value->mutable_operand(0));
      rows.push_back(value->mutable_operand(1));
    }
    return WriteRow(PackValues(computation, std::move(bases), packed_shape),
                    Concatenate(computation, rows, /*rows=*/1),
                    first->mutable_operand(2));
  }
  return Concatenate(computation, values, packed_shape.dimensions(0));
}

absl::StatusOr<bool> PackLoop(HloInstruction* while_instr) {
  // WhileUtil rebuilds the loop on the main thread, without its compute type.
  if (!while_instr->parent()->IsMainThread() ||
      while_instr->frontend_attributes().map().contains(kXlaComputeTypeAttr)) {
    return false;
  }
  HloComputation* parent = while_instr->parent();
  HloInstruction* init = while_instr->mutable_operand(0);
  const std::vector<HloInstruction*> row_writes = FindRowWrites(while_instr);
  const int64_t num_elements = row_writes.size();

  // The buffers of a group share the element type, the row count and the row
  // index instruction, so they take the same row in every iteration.
  std::map<std::tuple<PrimitiveType, int64_t, int64_t>, std::vector<int64_t>>
      candidates;
  for (int64_t index = 0; index < num_elements; ++index) {
    const HloInstruction* write = row_writes[index];
    // HostOffloader offloads the row writes of a host memory buffer one by
    // one.
    if (write == nullptr ||
        ShapeUtil::ElementsIn(write->operand(1)->shape()) > kMaxRowElements ||
        (init->opcode() == HloOpcode::kTuple &&
         IsHostBuffer(init->mutable_operand(index)))) {
      continue;
    }
    candidates[{write->shape().element_type(), write->shape().dimensions(0),
                write->operand(2)->unique_id()}]
        .push_back(index);
  }
  std::vector<std::vector<int64_t>> groups;
  for (auto& [key, group] : candidates) {
    if (group.size() >= kMinBuffersPerGroup) {
      groups.push_back(std::move(group));
    }
  }
  if (groups.empty()) {
    return false;
  }

  std::vector<HloInstruction*> packed_inits;
  for (const std::vector<int64_t>& group : groups) {
    const Shape& first = while_instr->shape().tuple_shapes(group.front());
    int64_t columns = 0;
    std::vector<HloInstruction*> inits;
    inits.reserve(group.size());
    for (int64_t index : group) {
      const Shape& shape = while_instr->shape().tuple_shapes(index);
      columns += ShapeUtil::ElementsIn(shape) / shape.dimensions(0);
      inits.push_back(
          init->opcode() == HloOpcode::kTuple
              ? init->mutable_operand(index)
              : parent->AddInstruction(
                    HloInstruction::CreateGetTupleElement(init, index)));
    }
    packed_inits.push_back(
        PackValues(parent, std::move(inits),
                   ShapeUtil::MakeShape(first.element_type(),
                                        {first.dimensions(0), columns})));
  }

  // Each packed buffer becomes a new element of the loop state.
  ABSL_ASSIGN_OR_RETURN(
      WhileUtil::MakeInstructionsLiveInResult result,
      WhileUtil::MakeInstructionsLiveIn(while_instr, packed_inits));
  HloInstruction* new_while = result.new_while_instr;
  HloComputation* new_body = new_while->while_body();
  const auto& body_map = result.while_body_instruction_map;

  for (int64_t g = 0; g < groups.size(); ++g) {
    // The body writes one packed row per iteration.
    std::vector<HloInstruction*> writes, rows;
    for (int64_t index : groups[g]) {
      writes.push_back(body_map.at(row_writes[index]));
      rows.push_back(writes.back()->mutable_operand(1));
    }
    ABSL_RETURN_IF_ERROR(new_body->root_instruction()->ReplaceOperandWith(
        num_elements + g, WriteRow(result.while_body_live_in_values[g],
                                   Concatenate(new_body, rows, /*rows=*/1),
                                   writes.front()->mutable_operand(2))));

    // Each old buffer passes through the loop unwritten, for
    // WhileLoopSimplifier to remove, and each loop output becomes its columns
    // of the packed buffer.
    HloInstruction* packed = parent->AddInstruction(
        HloInstruction::CreateGetTupleElement(new_while, num_elements + g));
    const int64_t num_rows = packed->shape().dimensions(0);
    int64_t column = 0;
    for (int64_t k = 0; k < groups[g].size(); ++k) {
      ABSL_RETURN_IF_ERROR(
          writes[k]->ReplaceAllUsesWith(writes[k]->mutable_operand(0)));
      ABSL_RETURN_IF_ERROR(new_body->RemoveInstruction(writes[k]));
      HloInstruction* output =
          result.replacement_instr->mutable_operand(groups[g][k]);
      const int64_t width = ShapeUtil::ElementsIn(output->shape()) / num_rows;
      HloInstruction* columns =
          parent->AddInstruction(HloInstruction::CreateSlice(
              ShapeUtil::MakeShape(packed->shape().element_type(),
                                   {num_rows, width}),
              packed, {0, column}, {num_rows, column + width}, {1, 1}));
      if (output->shape().dimensions().size() == 1) {
        columns = parent->AddInstruction(
            HloInstruction::CreateReshape(output->shape(), columns));
      }
      ABSL_RETURN_IF_ERROR(parent->ReplaceInstruction(output, columns));
      column += width;
    }
    VLOG(1) << "Packed " << groups[g].size() << " row buffers of "
            << new_while->name() << " into "
            << ShapeUtil::HumanString(packed->shape());
  }
  return true;
}

}  // namespace

absl::StatusOr<bool> LoopRowBufferPacker::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Loops of callees come first: packing a loop clones its body, and with it
  // the loops inside.
  std::vector<HloInstruction*> loops;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    absl::c_copy_if(computation->instructions(), std::back_inserter(loops),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }
  bool changed = false;
  for (HloInstruction* loop : loops) {
    ABSL_ASSIGN_OR_RETURN(bool packed, PackLoop(loop));
    changed |= packed;
  }
  return changed;
}

}  // namespace xla
