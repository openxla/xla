/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/hlo_value.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/map_util.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/types.h"
#include "xla/util.h"
#include "third_party/tsl/platform/errors.h"
#include "third_party/tsl/platform/logging.h"

namespace xla {

using absl::StrAppend;
using absl::StrCat;

const Shape& HloPosition::shape() const {
  return ShapeUtil::GetSubshape(instruction->shape(), index);
}

std::string HloPosition::ToString() const {
  std::string index_str =
      instruction->shape().IsTuple() ? (" " + index.ToString()) : "";
  return StrCat(instruction->name(), index_str);
}

std::ostream& operator<<(std::ostream& out, const HloPosition& position) {
  out << position.ToString();
  return out;
}

std::string HloUse::ToString() const {
  std::string index_str =
      instruction->operand(operand_number)->shape().IsTuple()
          ? (" " + operand_index.ToString())
          : "";
  return StrCat(instruction->name(), ", operand ", operand_number, index_str);
}

std::ostream& operator<<(std::ostream& out, const HloUse& use) {
  out << use.ToString();
  return out;
}

HloValue::HloValue(HloValue::Id id, HloInstruction* instruction,
                   const ShapeIndex& index, bool is_phi)
    : BufferValue(instruction, index, id),
      uses_([this] { return ComputeUses(); }),
      is_phi_(is_phi) {
  // The defining position is always the first element in the positions_ vector.
  positions_.push_back(HloPosition{instruction, index});
}

std::string HloValue::ToShortString() const {
  return absl::StrFormat(
      "<%d %s%s%s%s>", id(), instruction()->name(),
      instruction()->shape().IsTuple() ? index().ToString() : "",
      is_phi() ? " (phi)" : "", has_color() ? StrCat(" @", color()) : "");
}

std::string HloValue::ToString(int indent) const {
  std::string indentation(indent, ' ');
  std::string out =
      StrCat(indentation, ToShortString(), "\n", indentation, " positions:\n");
  for (const HloPosition& position : positions()) {
    StrAppend(&out, indentation, "  ", position.ToString(), "\n");
  }
  StrAppend(&out, indentation, " uses:\n");
  for (const HloUse& use : GetUses()) {
    StrAppend(&out, indentation, "  ", use.ToString(), "\n");
  }
  StrAppend(&out, indentation, " from instruction:", instruction()->ToString(),
            "\n");
  return out;
}

namespace {

// Returns true if the instruction 'user' may use the value at the given
// ShapeIndex in the given operand. Generally, instruction which pass through
// values transparently without reading the value are not considered to use the
// value.
bool MayUseOperandValue(int64_t operand_number, const ShapeIndex& index,
                        const HloInstruction* user) {
  switch (user->opcode()) {
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kCopy:
      // These instructions only access the top-level values of their
      // operand. Non-top-level (nested) values are passed through
      // transparently.
      CHECK_EQ(operand_number, 0);
      return index.empty();
    case HloOpcode::kDomain:
    case HloOpcode::kTuple:
      // These instructions always pass through their operands transparently.
      return false;

    case HloOpcode::kCall:
    case HloOpcode::kWhile:
      // Although call and while instructions pass through their operands, they
      // are considered uses.
      return true;

    default:
      return true;
  }
}

}  // namespace

void HloValue::SetPositions(absl::Span<const HloPosition> positions) {
  CHECK_EQ(positions_.size(), 1) << "SetPositions should only be called once.";

  // The positions must be unique and should not contain the defining position
  // as this is added at construction time.
  for (const HloPosition& position_a : positions) {
    DCHECK_NE(position_a, defining_position());
    for (const HloPosition& position_b : positions) {
      if (&position_a != &position_b) {
        DCHECK_NE(position_a, position_b);
      }
    }
  }

  positions_.insert(positions_.end(), positions.begin(), positions.end());
  // Update liveout status of this HloValue.
  live_out_of_module_ |=
      IsRootOf(defining_instruction()->GetModule()->entry_computation());
}

std::vector<HloUse> HloValue::ComputeUses() const {
  // Gather the computation roots at which this value appears.
  absl::flat_hash_set<HloInstruction*> root_positions;
  for (const HloPosition& position : positions_) {
    if (position.instruction->IsRoot()) {
      root_positions.insert(position.instruction);
    }
  }

  std::vector<HloUse> uses;
  // Build vector of HloUses for the value.
  for (const HloPosition& position : positions_) {
    for (HloInstruction* user : position.instruction->users()) {
      for (int64_t i = 0; i < user->operand_count(); ++i) {
        if (user->operand(i) != position.instruction) {
          continue;
        }

        // Root instructions of computations are considered to be uses whether
        // or not the root instruction itself actually uses the value.
        if (MayUseOperandValue(i, position.index, user) ||
            root_positions.contains(user)) {
          HloUse new_use{user, i, position.index};

          // The new use must not already exist in uses.
          for (const HloUse& use : uses) {
            DCHECK_NE(use, new_use);
          }

          uses.push_back(std::move(new_use));
        }
      }
    }
  }
  return uses;
}

bool HloValue::IsRootOf(const HloComputation* computation) const {
  return absl::c_any_of(positions_, [&](const HloPosition& position) {
    return position.instruction->IsRoot() &&
           position.instruction->parent() == computation;
  });
}

std::ostream& operator<<(std::ostream& out, const HloValue& value) {
  out << value.ToShortString();
  return out;
}

/*static*/
HloValueSet HloValueSet::UnionOf(absl::Span<const HloValueSet* const> inputs) {
  if (inputs.size() == 1) {
    return *inputs[0];
  }

  HloValueSet union_set;
  if (inputs.size() == 2) {
    union_set.reserve(std::max(inputs[0]->size(), inputs[1]->size()));
    absl::c_set_union(*inputs[0], *inputs[1],
                      std::back_inserter(union_set.values_),
                      HloValue::IdLessThan());
  } else {
    // TODO(cjfj): Take advantage of the fact that the inputs are sorted?
    for (const HloValueSet* input : inputs) {
      union_set.values_.insert(union_set.values_.end(), input->begin(),
                               input->end());
    }
    union_set.Sort();
    union_set.Deduplicate();
  }
  return union_set;
}

bool HloValueSet::AssignUnionOf(absl::Span<const HloValueSet* const> inputs) {
  HloValueSet union_set(UnionOf(inputs));
  bool changed = (*this != union_set);
  if (changed) *this = union_set;
  return changed;
}

std::string HloValueSet::ToString() const {
  return StrCat("HloValueSet: ",
                absl::StrJoin(*this, ", ",
                              [](std::string* result, const HloValue* value) {
                                result->append(value->ToShortString());
                              }));
}

std::ostream& operator<<(std::ostream& out, const HloValueSet& value_set) {
  out << value_set.ToString();
  return out;
}

bool InstructionValueSet::IsAmbiguous() const {
  return absl::c_any_of(
      *this, [](const auto& entry) { return entry.second.size() > 1; });
}

bool InstructionValueSet::AssignUnionOf(
    absl::Span<const InstructionValueSet* const> inputs) {
  CHECK_GT(inputs.size(), 0);
  for (int i = 1; i < inputs.size(); ++i) {
    DCHECK(ShapeUtil::Compatible(inputs[0]->shape(), inputs[i]->shape()));
  }

  bool changed = false;
  for (auto& pair : *this) {
    const ShapeIndex& index = pair.first;
    HloValueSet& value_set = pair.second;

    std::vector<const HloValueSet*> input_value_sets;
    input_value_sets.reserve(inputs.size());
    for (const InstructionValueSet* input : inputs) {
      input_value_sets.push_back(&input->element(index));
    }
    changed |= value_set.AssignUnionOf(input_value_sets);
  }

  return changed;
}

bool InstructionValueSet::AssignUnionOf(const InstructionValueSet& input,
                                        ShapeIndexView input_index) {
  bool changed = false;
  for (auto& [index, value_set] : *this) {
    ShapeIndex source_index(input_index);
    for (auto i : index) {
      source_index.push_back(i);
    }
    changed |= value_set.AssignUnionOf({&input.element(source_index)});
  }

  return changed;
}

std::ostream& operator<<(std::ostream& out,
                         const InstructionValueSet& instruction_value_set) {
  out << instruction_value_set.ToString();
  return out;
}

std::string InstructionValueSet::ToString() const {
  std::string out =
      StrCat("InstructionValueSet(", ShapeUtil::HumanString(shape()), ")\n");
  ForEachElement([&out](const ShapeIndex& index, const HloValueSet& value_set) {
    StrAppend(&out, "  ", index.ToString(), " : ", value_set.ToString(), "\n");
  });
  return out;
}

}  // namespace xla
