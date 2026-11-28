/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/tuple_util.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tsl/platform/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/ir/hlo_original_value_util.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"

namespace xla {

namespace {

// Returns the original value of `instruction` if it can be used to derive the
// original values of instructions built from `instruction`, i.e. if it is set,
// is not a synthetic call, and has the same tuple structure as `instruction`'s
// shape. Returns nullptr otherwise, e.g. if `instruction`'s shape was changed
// in place without updating its original value. Instructions derived from such
// an instruction are left without an original value, and it is up to the caller
// that changed the shape to keep the original values in sync.
//
// REQUIRES: `instruction != nullptr`.
const OriginalValue* GetUsableOriginalValue(const HloInstruction* instruction) {
  const std::shared_ptr<OriginalValue>& original_value =
      instruction->original_value();
  if (original_value == nullptr || original_value->is_synthetic_call() ||
      !original_value->IsCompatibleWith(instruction->shape())) {
    return nullptr;
  }
  return original_value.get();
}

// Returns the subtree at `index` of `original_value`, which must be compatible
// with a tuple shape whose element `index` is `subshape`, as an original value
// for an instruction of shape `subshape`. Returns nullptr if the subtree is
// empty.
std::shared_ptr<OriginalValue> GetElementOriginalValue(
    const OriginalValue& original_value, int64_t index, const Shape& subshape) {
  auto element_original_value = std::make_shared<OriginalValue>(subshape);
  CHECK_OK(element_original_value->mutable_tree()->CopyCompatibleSubtreeFrom(
      original_value.tree(), {index}, {}));
  if (element_original_value->IsEmpty()) {
    return nullptr;
  }
  return element_original_value;
}

// Sets the original value of the new get-tuple-element `gte` to the original
// value of the element of its operand that it extracts, if any. If the operand
// has no usable original value, falls back to the element at `gte`'s tuple
// index of `fallback_original_value` (the original value of the operand
// according to an enclosing tuple), if not null.
//
// REQUIRES: `gte != nullptr`, `gte->opcode() == HloOpcode::kGetTupleElement`,
// and `fallback_original_value`, if not null, is compatible with the shape of
// `gte`'s operand.
void SetGetTupleElementOriginalValue(
    HloInstruction* gte,
    const OriginalValue* fallback_original_value = nullptr) {
  if (GetUsableOriginalValue(gte->operand(0)) != nullptr) {
    gte->set_original_value(OriginalValue::CreateFromInstruction(gte));
  } else if (fallback_original_value != nullptr) {
    gte->set_original_value(GetElementOriginalValue(
        *fallback_original_value, gte->tuple_index(), gte->shape()));
  }
}

// Sets the original value of the new tuple instruction `new_tuple`, which is
// derived from `source_tuple`.
//
// Each element of `new_tuple` takes the original value of the corresponding
// operand of `new_tuple` if that operand has a usable one, since the operand is
// the instruction that produces the element. Otherwise, if `is_from_source(i)`
// is true, element i of `new_tuple` holds the same value as element i of
// `source_tuple`, so it falls back to the original value of that element of
// `source_tuple`. If `source_tuple` has no usable original value, the fallback
// is `fallback_original_value` (the original value of `source_tuple` according
// to an enclosing tuple), if not null. `new_tuple` takes the call hierarchy of
// the original value used as the fallback. Leaves `new_tuple` without an
// original value if none of its elements has one.
//
// REQUIRES: `new_tuple != nullptr` and `new_tuple->opcode() ==
// HloOpcode::kTuple`. If `is_from_source(i)` is true for an operand index i,
// then `source_tuple` is not null and has a tuple shape with an element i whose
// shape is that of operand i of `new_tuple`. `fallback_original_value`, if not
// null, is compatible with the shape of `source_tuple`.
template <typename IsFromSource>
void SetTupleOriginalValue(
    HloInstruction* new_tuple, const HloInstruction* source_tuple,
    IsFromSource is_from_source,
    const OriginalValue* fallback_original_value = nullptr) {
  const OriginalValue* source_original_value =
      source_tuple != nullptr ? GetUsableOriginalValue(source_tuple) : nullptr;
  if (source_original_value == nullptr) {
    source_original_value = fallback_original_value;
  }
  auto original_value = std::make_shared<OriginalValue>(
      new_tuple->shape(), source_original_value != nullptr
                              ? source_original_value->call_hierarchy()
                              : std::nullopt);
  for (int64_t i = 0; i < new_tuple->operand_count(); ++i) {
    // The compatibility of the original values used below with the shapes of
    // their instructions guarantees that these copies succeed.
    if (const OriginalValue* operand_original_value =
            GetUsableOriginalValue(new_tuple->operand(i))) {
      CHECK_OK(original_value->mutable_tree()->CopyCompatibleSubtreeFrom(
          operand_original_value->tree(), {}, {i}));
    } else if (is_from_source(i) && source_original_value != nullptr) {
      CHECK_OK(original_value->mutable_tree()->CopyCompatibleSubtreeFrom(
          source_original_value->tree(), {i}, {i}));
    }
  }
  if (!original_value->IsEmpty()) {
    new_tuple->set_original_value(std::move(original_value));
  }
}

// Implements TupleUtil::ReplaceTupleWith. `fallback_original_value`, if not
// null, is the original value of `tuple` according to an enclosing tuple. It is
// used in place of `tuple`'s own original value if `tuple` doesn't have a
// usable one, when deriving the original values of the instructions created
// here (see SetTupleOriginalValue).
absl::StatusOr<HloInstruction*> ReplaceTupleWithImpl(
    HloInstruction* new_instruction, HloInstruction* tuple,
    ShapeIndex shape_index, bool insert_bitcast_if_different_shape,
    const OriginalValue* fallback_original_value) {
  const Shape& tuple_shape = tuple->shape();
  CHECK(tuple->shape().IsTuple())
      << "ReplaceTupleWith was called for a non-tuple. Tuple = "
      << tuple->ToString()
      << ", new_instruction = " << new_instruction->ToString()
      << ", shape_index = " << shape_index.ToString();
  // Check if the new instruction is a get-tuple-element of the correct index of
  // the tuple, and if so, simply return tuple.
  const HloInstruction* instruction = new_instruction;
  bool equivalent = true;
  for (int i = shape_index.size() - 1; i >= 0; --i) {
    int index = shape_index[i];
    if (instruction->opcode() != HloOpcode::kGetTupleElement ||
        instruction->tuple_index() != index) {
      equivalent = false;
      break;
    }
    instruction = instruction->operand(0);
  }
  if (equivalent && instruction == tuple) {
    VLOG(4) << "Instruction " << new_instruction->ToShortString()
            << " already exists at index " << shape_index.ToString() << " of "
            << tuple->ToShortString();
    return tuple;
  }

  HloComputation* computation = new_instruction->parent();
  std::vector<HloInstruction*> tuple_args(tuple_shape.tuple_shapes().size());
  CHECK_GE(tuple_shape.tuple_shapes().size(), shape_index[0]);
  for (int i = 0; i < tuple_shape.tuple_shapes().size(); ++i) {
    const Shape& subshape = tuple_shape.tuple_shapes(i);
    // If tuple is a tuple instruction, we can get the tuple instruction's
    // operand to construct the new tuple to improve compilation time
    // performance.
    auto get_operand = [&]() {
      if (tuple->opcode() == HloOpcode::kTuple) {
        return tuple->mutable_operand(i);
      } else {
        HloInstruction* gte = computation->AddInstruction(
            HloInstruction::CreateGetTupleElement(subshape, tuple, i));
        SetGetTupleElementOriginalValue(gte, fallback_original_value);
        return gte;
      }
    };
    if (i == shape_index[0]) {
      // If the subshape is still a tuple, recurse and pass a new shape index
      // for the one level deeper.
      if (subshape.IsTuple()) {
        // The enclosing original value of the nested tuple is element i of
        // `tuple`'s original value (or of its fallback).
        const OriginalValue* tuple_original_value =
            GetUsableOriginalValue(tuple);
        if (tuple_original_value == nullptr) {
          tuple_original_value = fallback_original_value;
        }
        std::shared_ptr<OriginalValue> element_original_value =
            tuple_original_value != nullptr
                ? GetElementOriginalValue(*tuple_original_value, i, subshape)
                : nullptr;
        ABSL_ASSIGN_OR_RETURN(
            tuple_args[i],
            ReplaceTupleWithImpl(
                new_instruction, get_operand(),
                ShapeIndex(shape_index.begin() + 1, shape_index.end()),
                /*insert_bitcast_if_different_shape=*/true,
                element_original_value.get()));
      } else {
        if (subshape != new_instruction->shape() &&
            insert_bitcast_if_different_shape) {
          VLOG(4) << "Old shape = " << subshape.ToString()
                  << ", new shape = " << new_instruction->shape().ToString()
                  << "; inserting a bitcast.";
          HloInstruction* bitcast = computation->AddInstruction(
              HloInstruction::CreateBitcast(subshape, new_instruction));
          // The bitcast holds the value of `new_instruction` if their shapes
          // are compatible (e.g. differ only in layout).
          CopyOriginalValue(new_instruction, bitcast, /*clone=*/false,
                            /*issue_warning=*/false);
          new_instruction = bitcast;
        } else if (tuple->opcode() == HloOpcode::kTuple &&
                   tuple->operand(i) == new_instruction) {
          // If the tuple element is the same as the new instruction, we
          // actually don't have to create a new tuple, just return the original
          // tuple.
          VLOG(4) << "Tuple already contains the new instruction = "
                  << new_instruction->ToShortString()
                  << " tuple = " << tuple->ToShortString();
          return tuple;
        }
        tuple_args[i] = new_instruction;
      }
    } else {
      tuple_args[i] = get_operand();
    }
  }
  if (shape_index[0] == tuple_shape.tuple_shapes().size()) {
    // If shape_index[0] is equal to the tuple shape size, add the new
    // instruction as an additional argument.
    tuple_args.push_back(new_instruction);
  }
  HloInstruction* new_tuple =
      computation->AddInstruction(HloInstruction::CreateTuple(tuple_args));
  // Every element of `new_tuple` other than the replaced (or appended) one
  // holds the same value as the element of `tuple` at the same index.
  SetTupleOriginalValue(
      new_tuple, tuple, [&](int64_t i) { return i != shape_index[0]; },
      fallback_original_value);
  return new_tuple;
}

}  // namespace

/*static*/ HloInstruction* TupleUtil::ExtractPrefix(HloInstruction* input_tuple,
                                                    int64_t elements,
                                                    absl::string_view name) {
  CHECK(input_tuple->shape().IsTuple());

  HloComputation* computation = input_tuple->parent();
  const Shape& input_shape = input_tuple->shape();

  std::vector<HloInstruction*> tuple_elements;
  tuple_elements.reserve(elements);
  for (int i = 0; i < elements; i++) {
    std::string element_name;
    if (!name.empty()) {
      element_name = absl::StrCat(name, ".element.", i);
    }
    tuple_elements.push_back(computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(input_shape.tuple_shapes(i),
                                              input_tuple, i),
        element_name));
    SetGetTupleElementOriginalValue(tuple_elements.back());
  }

  HloInstruction* prefix = computation->AddInstruction(
      HloInstruction::CreateTuple(tuple_elements), name);
  SetTupleOriginalValue(prefix, input_tuple, [](int64_t) { return true; });
  return prefix;
}

/*static*/ HloInstruction* TupleUtil::AppendSuffix(
    HloInstruction* input_tuple,
    absl::Span<HloInstruction* const> trailing_values) {
  CHECK(input_tuple->shape().IsTuple());

  HloComputation* computation = input_tuple->parent();
  const Shape& input_shape = input_tuple->shape();
  const int64_t num_input_elements = input_shape.tuple_shapes().size();
  std::vector<HloInstruction*> tuple_elements;
  tuple_elements.reserve(num_input_elements);
  for (int i = 0; i < num_input_elements; i++) {
    tuple_elements.push_back(
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            input_shape.tuple_shapes(i), input_tuple, i)));
    SetGetTupleElementOriginalValue(tuple_elements.back());
  }
  tuple_elements.insert(tuple_elements.end(), trailing_values.begin(),
                        trailing_values.end());
  HloInstruction* with_suffix =
      computation->AddInstruction(HloInstruction::CreateTuple(tuple_elements));
  SetTupleOriginalValue(with_suffix, input_tuple,
                        [&](int64_t i) { return i < num_input_elements; });
  return with_suffix;
}

/*static*/ absl::StatusOr<HloInstruction*> TupleUtil::ReplaceTupleWith(
    HloInstruction* new_instruction, HloInstruction* tuple,
    ShapeIndex shape_index, bool insert_bitcast_if_different_shape) {
  return ReplaceTupleWithImpl(new_instruction, tuple, std::move(shape_index),
                              insert_bitcast_if_different_shape,
                              /*fallback_original_value=*/nullptr);
}

/*static*/ HloInstruction* TupleUtil::AddGetTupleElements(
    const HloPosition& position) {
  HloInstruction* instruction = position.instruction;
  HloComputation* computation = instruction->parent();

  // If the instruction we're processing is a tuple, we (recursively) search or
  // create kGetTupleElement instructions and copy that value.
  //
  // `enclosing_original_value` is the original value of `instruction`
  // according to `position.instruction`. A new get-tuple-element falls back to
  // it if `instruction` (e.g. a reused get-tuple-element) has no usable
  // original value of its own.
  std::shared_ptr<OriginalValue> enclosing_original_value;
  for (int64_t index : position.index) {
    const OriginalValue* parent_original_value =
        GetUsableOriginalValue(instruction);
    if (parent_original_value == nullptr) {
      parent_original_value = enclosing_original_value.get();
    }
    std::shared_ptr<OriginalValue> element_original_value =
        parent_original_value != nullptr
            ? GetElementOriginalValue(*parent_original_value, index,
                                      instruction->shape().tuple_shapes(index))
            : nullptr;
    // We first search if there already is a get-tuple-element with the correct
    // index. If there is no such get-tuple-element, we create one.
    auto gte_it = absl::c_find_if(
        instruction->users(), [index](const HloInstruction* use) {
          return use != use->parent()->root_instruction() &&
                 use->opcode() == HloOpcode::kGetTupleElement &&
                 use->tuple_index() == index;
        });
    if (gte_it != instruction->users().end()) {
      instruction = *gte_it;
    } else {
      instruction =
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              instruction->shape().tuple_shapes(index), instruction, index));
      SetGetTupleElementOriginalValue(instruction,
                                      enclosing_original_value.get());
    }
    enclosing_original_value = std::move(element_original_value);
  }
  return instruction;
}

ShapeTree<HloInstruction*> TupleUtil::DisassembleTupleInstruction(
    HloInstruction* tuple) {
  const Shape& shape = tuple->shape();
  ShapeTree<HloInstruction*> result(shape);
  result.ForEachMutableElement([&](ShapeIndexView index,
                                   HloInstruction** element) {
    if (index.empty()) {
      *element = tuple;
    } else {
      ShapeIndexView parent_index = index.subspan(0, index.size() - 1);
      HloInstruction* parent = result.element(parent_index);
      std::string name = absl::StrCat(tuple->name(), ".disassembled.",
                                      absl::StrJoin(index, "."));
      *element = tuple->parent()->AddInstruction(
          HloInstruction::CreateGetTupleElement(parent, index.back()), name);
      // Elements are visited in pre-order, so `parent` already has its
      // original value, if any.
      SetGetTupleElementOriginalValue(*element);
    }
  });
  return result;
}

HloInstruction* TupleUtil::AssembleTupleInstruction(
    HloComputation* computation, ShapeTree<HloInstruction*> elements,
    absl::string_view name) {
  elements.ForEachMutableElementPostOrder(
      [&](const ShapeIndex& index, HloInstruction** element) {
        const Shape& subshape = ShapeUtil::GetSubshape(elements.shape(), index);
        if (subshape.IsTuple()) {
          absl::InlinedVector<HloInstruction*, 2> children;
          ShapeIndex child_index = index;
          for (int i = 0; i < subshape.tuple_shapes().size(); ++i) {
            child_index.push_back(i);
            children.push_back(elements.element(child_index));
            child_index.pop_back();
          }
          std::string new_name;
          if (!name.empty()) {
            if (index.empty()) {
              new_name = std::string(name);
            } else {
              new_name =
                  absl::StrCat(name, ".assembled.", absl::StrJoin(index, "."));
            }
          }
          *element = computation->AddInstruction(
              HloInstruction::CreateTuple(children), new_name);
          // Elements are visited in post-order, so `children` already have
          // their original values, if any. The tuple is assembled from its
          // elements, so it takes their original values.
          SetTupleOriginalValue(*element, /*source_tuple=*/nullptr,
                                [](int64_t) { return false; });
        }
      });
  return elements.element({});
}

HloInstruction* TupleUtil::GetTupleInstructionAtIndex(
    HloInstruction& tuple, const ShapeIndex& target_index) {
  HloInstruction* target_index_instr = &tuple;

  for (int32_t tuple_index : target_index) {
    bool found = false;
    for (HloInstruction* user : target_index_instr->users()) {
      if (user->opcode() == HloOpcode::kGetTupleElement &&
          user->tuple_index() == tuple_index) {
        target_index_instr = user;
        found = true;
        break;
      }
    }

    if (!found) {
      // No GTE found at the target index.
      return nullptr;
    }
  }

  return target_index_instr;
}

}  // namespace xla
