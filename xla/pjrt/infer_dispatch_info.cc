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

#include "xla/pjrt/infer_dispatch_info.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/pjrt/dynamic_shapes.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/utils.h"
#include "xla/service/computation_layout.h"
#include "xla/service/device_assignment.h"
#include "xla/service/hlo_value.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

std::vector<Shape> GetParameterShapes(const ComputationLayout& layout) {
  // For now, XLA programs compiled with multiple arguments for PJRT cannot use
  // tuples for any of their arguments, so we can assume that a tuple can only
  // arise when there is a single argument.
  std::vector<Shape> shapes;
  if (layout.parameter_count() == 1 && layout.parameter_shape(0).IsTuple()) {
    shapes.reserve(layout.parameter_shape(0).tuple_shapes().size());
    absl::c_copy(layout.parameter_shape(0).tuple_shapes(),
                 std::back_inserter(shapes));
  } else {
    shapes.reserve(layout.parameter_count());
    for (const ShapeLayout& sl : layout.parameter_layouts()) {
      shapes.push_back(sl.shape());
    }
  }
  return shapes;
}

namespace {

bool IsPadRealToStaticOp(mlir::Operation* op) {
  auto attr = op->getAttrOfType<mlir::StringAttr>("call_target_name");
  return attr != nullptr && attr.getValue() == "PadRealToStatic";
}

bool ShapeIndexOverlaps(const ShapeIndex& a, const ShapeIndex& b) {
  const size_t min_size = std::min(a.size(), b.size());
  for (size_t i = 0; i < min_size; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

bool HasOutputToOperandAlias(const HloInstruction* instruction,
                             int64_t operand_number,
                             const ShapeIndex& operand_index) {
  return absl::c_any_of(
      instruction->output_operand_aliasing(), [&](const auto& alias) {
        return alias.second.first == operand_number &&
               ShapeIndexOverlaps(alias.second.second, operand_index);
      });
}

// Returns true if `instruction` is an embedded custom-call dispatcher at the
// root of an offloaded async wrapper computation (for example, after an
// async-wrapped subcomputation `kCall` is embedded as a `kCustomCall` with
// `to_apply()`).
bool IsAsyncWrappedSubcomputationCall(const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kCustomCall ||
      instruction->IsCustomCall("PadRealToStatic") ||
      instruction->called_computations().size() != 1) {
    return false;
  }
  const HloComputation* wrapper = instruction->parent();
  if (wrapper == nullptr || wrapper->root_instruction() != instruction ||
      wrapper->execution_thread() == HloInstruction::kMainExecutionThread) {
    return false;
  }
  const HloComputation* callee = instruction->to_apply();
  if (callee == nullptr ||
      instruction->operand_count() != callee->num_parameters()) {
    return false;
  }
  const absl::InlinedVector<HloInstruction*, 1> all_callers =
      wrapper->caller_instructions();
  const absl::InlinedVector<HloInstruction*, 1> start_callers =
      wrapper->caller_instructions(HloOpcode::kAsyncStart);
  return !start_callers.empty() &&
         absl::c_all_of(all_callers, [&](const HloInstruction* caller) {
           return caller->IsAsynchronous() &&
                  caller->async_wrapped_instruction() == instruction;
         });
}

// Maps a data operand `(async_user, operand_number)` on a `kAsyncStart` or
// `kAsyncUpdate` instruction to the bound parameter index in
// `async_user->async_wrapped_computation()`, or `std::nullopt` if the chain is
// incomplete or `operand_number` is not a bound data operand.
std::optional<int64_t> GetAsyncBoundParameterNumber(
    const HloInstruction* async_user, int64_t operand_number) {
  if (async_user->opcode() != HloOpcode::kAsyncStart &&
      async_user->opcode() != HloOpcode::kAsyncUpdate) {
    return std::nullopt;
  }
  const int64_t user_start_idx =
      async_user->opcode() == HloOpcode::kAsyncStart ? 0 : 1;
  if (operand_number < user_start_idx ||
      operand_number >= async_user->operand_count()) {
    return std::nullopt;
  }
  const HloInstruction* start = async_user->async_chain_start();
  const HloInstruction* done = async_user->async_chain_done();
  const HloComputation* wrapped = async_user->async_wrapped_computation();
  const HloAsyncInstruction* done_async =
      done != nullptr ? DynCast<HloAsyncInstruction>(done) : nullptr;
  const HloAsyncStartInstruction* start_async =
      start != nullptr ? DynCast<HloAsyncStartInstruction>(start) : nullptr;
  if (start_async == nullptr || done_async == nullptr || wrapped == nullptr) {
    return std::nullopt;
  }
  const std::vector<const HloInstruction*> bound_operands =
      hlo_instruction_utils::async::GetAsyncBoundOperands(done_async);
  if (bound_operands.size() != static_cast<size_t>(wrapped->num_parameters())) {
    return std::nullopt;
  }
  int64_t offset = 0;
  for (const HloInstruction* instr : start_async->GetAsyncChain()) {
    const int64_t start_idx = instr->opcode() == HloOpcode::kAsyncStart ? 0 : 1;
    if (instr == async_user) {
      const int64_t param_no = offset + (operand_number - user_start_idx);
      if (param_no < 0 || param_no >= wrapped->num_parameters()) {
        return std::nullopt;
      }
      return param_no;
    }
    if (instr == done) {
      break;
    }
    if (instr->operand_count() > start_idx) {
      offset += instr->operand_count() - start_idx;
    }
  }
  return std::nullopt;
}

// Returns the caller operand passed to `callee->parameter_instruction(
// parameter_number)` when `caller` invokes `callee` with positional argument
// forwarding, or `nullptr` otherwise. Output-to-operand aliasing on `caller`
// is rejected separately by `IsAllowedPadRealToStaticUse`.
const HloInstruction* GetSubcomputationCallerOperand(
    const HloInstruction* caller, const HloComputation* callee,
    int64_t parameter_number) {
  if (parameter_number < 0 || parameter_number >= callee->num_parameters()) {
    return nullptr;
  }
  if (caller->opcode() == HloOpcode::kCall) {
    if (caller->to_apply() != callee ||
        caller->operand_count() != callee->num_parameters()) {
      return nullptr;
    }
    return caller->operand(parameter_number);
  }
  if (IsAsyncWrappedSubcomputationCall(caller)) {
    if (caller->to_apply() != callee) {
      return nullptr;
    }
    return caller->operand(parameter_number);
  }
  if (caller->opcode() == HloOpcode::kAsyncStart) {
    if (caller->async_wrapped_computation() != callee ||
        caller->async_chain_done() == nullptr) {
      return nullptr;
    }
    const HloAsyncInstruction* done_async =
        DynCast<HloAsyncInstruction>(caller->async_chain_done());
    if (done_async == nullptr) {
      return nullptr;
    }
    const std::vector<const HloInstruction*> bound_operands =
        hlo_instruction_utils::async::GetAsyncBoundOperands(done_async);
    if (bound_operands.size() !=
        static_cast<size_t>(callee->num_parameters())) {
      return nullptr;
    }
    return bound_operands[parameter_number];
  }
  return nullptr;
}

bool IsAllowedPadRealToStaticUse(
    const HloDataflowAnalysis& dataflow,
    const absl::flat_hash_set<HloValue::Id>& allowed_embedded_param_values,
    const HloValue& value, const HloUse& use) {
  const HloInstruction* user = use.instruction;
  switch (user->opcode()) {
    case HloOpcode::kCustomCall: {
      if (user->IsCustomCall("PadRealToStatic")) {
        return use.operand_number == 0 && use.operand_index.empty();
      }
      if (!IsAsyncWrappedSubcomputationCall(user) ||
          HasOutputToOperandAlias(user, use.operand_number,
                                  use.operand_index)) {
        return false;
      }
      const HloInstruction* param =
          user->to_apply()->parameter_instruction(use.operand_number);
      if (!ShapeUtil::IndexIsValid(param->shape(), use.operand_index)) {
        return false;
      }
      const std::vector<const HloValue*>& param_values =
          dataflow.GetValueSet(param, use.operand_index).values();
      return !param_values.empty() &&
             absl::c_all_of(param_values, [&](const HloValue* v) {
               return allowed_embedded_param_values.contains(v->id());
             });
    }
    case HloOpcode::kBitcast:
    case HloOpcode::kOptimizationBarrier:
      return true;
    case HloOpcode::kAddDependency:
      return use.operand_number == 0;
    case HloOpcode::kCall: {
      const HloComputation* callee = user->to_apply();
      if (callee == nullptr || use.operand_number >= callee->num_parameters() ||
          HasOutputToOperandAlias(user, use.operand_number,
                                  use.operand_index)) {
        return false;
      }
      const HloInstruction* param =
          callee->parameter_instruction(use.operand_number);
      if (!ShapeUtil::IndexIsValid(param->shape(), use.operand_index)) {
        return false;
      }
      const std::vector<const HloValue*>& param_values =
          dataflow.GetValueSet(param, use.operand_index).values();
      return absl::c_linear_search(param_values, &value) ||
             (!param_values.empty() &&
              absl::c_all_of(param_values, [&](const HloValue* v) {
                return allowed_embedded_param_values.contains(v->id());
              }));
    }
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate: {
      if (user->async_chain_start() == nullptr ||
          user->async_chain_done() == nullptr ||
          user->async_wrapped_computation() == nullptr) {
        return false;
      }
      if (user->opcode() == HloOpcode::kAsyncUpdate &&
          use.operand_number == 0) {
        return !use.operand_index.empty() && use.operand_index.front() == 0 &&
               !HasOutputToOperandAlias(user, 0, use.operand_index);
      }
      const std::optional<int64_t> param_no =
          GetAsyncBoundParameterNumber(user, use.operand_number);
      if (!param_no.has_value() ||
          HasOutputToOperandAlias(user, use.operand_number,
                                  use.operand_index) ||
          HasOutputToOperandAlias(user->async_chain_start(), *param_no,
                                  use.operand_index)) {
        return false;
      }
      const HloInstruction* param =
          user->async_wrapped_computation()->parameter_instruction(*param_no);
      if (!ShapeUtil::IndexIsValid(param->shape(), use.operand_index)) {
        return false;
      }
      const std::vector<const HloValue*>& param_values =
          dataflow.GetValueSet(param, use.operand_index).values();
      return absl::c_linear_search(param_values, &value) ||
             (!param_values.empty() &&
              absl::c_all_of(param_values, [&](const HloValue* v) {
                return allowed_embedded_param_values.contains(v->id());
              }));
    }
    case HloOpcode::kAsyncDone:
      return user->async_chain_start() != nullptr && use.operand_number == 0 &&
             !use.operand_index.empty() && use.operand_index.front() == 0;
    default:
      return false;
  }
}

absl::Status VerifyOnlyPadRealToStaticReadsValue(
    const HloDataflowAnalysis& dataflow,
    const absl::flat_hash_set<HloValue::Id>& allowed_embedded_param_values,
    const HloValue& value, const HloValue& entry_parameter_value) {
  const HloInstruction* parameter =
      entry_parameter_value.defining_instruction();
  const ShapeIndex& leaf_index = entry_parameter_value.defining_index();
  auto format_error = [&](absl::string_view detail,
                          absl::string_view via_name) {
    return InvalidArgument(
        "Entry parameter %d%s of %s feeds PadRealToStatic and is also %s "
        "(through %s). A dynamic parameter consumed by PadRealToStatic is "
        "passed as a compact buffer without dynamic-shape metadata, which no "
        "other instruction can read, so it must have no other users.",
        parameter->parameter_number(),
        leaf_index.empty() ? "" : absl::StrCat(" ", leaf_index.ToString()),
        parameter->GetModule()->name(), detail, via_name);
  };

  for (const HloPosition& pos : value.positions()) {
    if (pos.instruction->IsRoot()) {
      return format_error(absl::StrCat("returned as a result of computation ",
                                       pos.instruction->parent()->name()),
                          pos.instruction->name());
    }
  }
  for (const HloUse& use : value.GetUses()) {
    if (!IsAllowedPadRealToStaticUse(dataflow, allowed_embedded_param_values,
                                     value, use)) {
      const HloInstruction* via = use.instruction->operand(use.operand_number);
      return format_error(absl::StrCat("used by ", use.instruction->ToString()),
                          via->name());
    }
  }
  return absl::OkStatus();
}

struct EmbeddedParamOrigin {
  const HloValue* embedded_value;
  const HloValue* entry_parameter_value;
};

absl::StatusOr<std::vector<const HloValue*>> ResolvePadRealToStaticOriginValues(
    const HloModule& hlo_module, const HloInstruction* pad,
    const HloDataflowAnalysis& dataflow, const HloValue* value,
    absl::flat_hash_map<HloValue::Id, std::vector<const HloValue*>>& memo,
    absl::flat_hash_set<HloValue::Id>& active_stack,
    absl::flat_hash_set<HloValue::Id>& allowed_embedded_param_values,
    std::vector<EmbeddedParamOrigin>& embedded_origins,
    absl::flat_hash_set<HloValue::Id>& seen_entry_values,
    std::vector<const HloValue*>& entry_parameter_values) {
  if (auto it = memo.find(value->id()); it != memo.end()) {
    return it->second;
  }

  const HloComputation* entry_comp = hlo_module.entry_computation();
  const HloInstruction* def = value->defining_instruction();
  auto non_entry_error = [&]() {
    return InvalidArgument(
        "Dynamic operand of PadRealToStatic (%s) in %s must read an entry "
        "computation parameter directly without intermediate copies or "
        "reshards, but resolves to %s.",
        pad->ToString(), hlo_module.name(), def->ToString());
  };

  if (def->opcode() != HloOpcode::kParameter) {
    return non_entry_error();
  }
  if (def->parent() == entry_comp) {
    if (seen_entry_values.insert(value->id()).second) {
      entry_parameter_values.push_back(value);
    }
    std::vector<const HloValue*> result = {value};
    memo.emplace(value->id(), result);
    return result;
  }

  const HloComputation* sub_comp = def->parent();
  absl::InlinedVector<HloInstruction*, 1> callers =
      sub_comp->caller_instructions();
  if (callers.empty() || !active_stack.insert(value->id()).second) {
    return non_entry_error();
  }
  absl::c_sort(callers, [](const HloInstruction* a, const HloInstruction* b) {
    return a->unique_id() < b->unique_id();
  });

  std::vector<const HloValue*> resolved_entry_values;
  absl::flat_hash_set<HloValue::Id> seen_local_entry_values;
  for (const HloInstruction* caller : callers) {
    if ((caller->opcode() == HloOpcode::kAsyncUpdate ||
         caller->opcode() == HloOpcode::kAsyncDone) &&
        caller->async_chain_start() != nullptr &&
        caller->async_wrapped_computation() == sub_comp) {
      continue;
    }
    const HloInstruction* arg = GetSubcomputationCallerOperand(
        caller, sub_comp, def->parameter_number());
    if (arg == nullptr ||
        !ShapeUtil::IndexIsValid(arg->shape(), value->defining_index())) {
      return non_entry_error();
    }
    const std::vector<const HloValue*>& caller_values =
        dataflow.GetValueSet(arg, value->defining_index()).values();
    if (caller_values.empty()) {
      return non_entry_error();
    }
    for (const HloValue* caller_value : caller_values) {
      ABSL_ASSIGN_OR_RETURN(
          const std::vector<const HloValue*> entry_vals,
          ResolvePadRealToStaticOriginValues(
              hlo_module, pad, dataflow, caller_value, memo, active_stack,
              allowed_embedded_param_values, embedded_origins,
              seen_entry_values, entry_parameter_values));
      for (const HloValue* entry_val : entry_vals) {
        if (seen_local_entry_values.insert(entry_val->id()).second) {
          resolved_entry_values.push_back(entry_val);
        }
      }
    }
  }
  active_stack.erase(value->id());

  if (resolved_entry_values.empty()) {
    return non_entry_error();
  }
  if (allowed_embedded_param_values.insert(value->id()).second) {
    for (const HloValue* entry_val : resolved_entry_values) {
      const Shape& leaf_shape =
          ShapeUtil::GetSubshape(entry_val->defining_instruction()->shape(),
                                 entry_val->defining_index());
      if (leaf_shape.IsArray() && leaf_shape.is_dynamic()) {
        embedded_origins.push_back({value, entry_val});
        break;
      }
    }
  }
  memo.emplace(value->id(), resolved_entry_values);
  return resolved_entry_values;
}

}  // namespace

absl::Status ClearPadRealToStaticParameterMetadataPrefixes(
    const HloModule& hlo_module, absl::Span<Shape> parameter_shapes) {
  const HloComputation* entry_comp = hlo_module.entry_computation();
  if (entry_comp == nullptr) {
    return absl::OkStatus();
  }

  std::vector<const HloInstruction*> pad_calls;
  for (const HloComputation* comp : hlo_module.computations()) {
    for (const HloInstruction* instruction : comp->instructions()) {
      if (instruction->IsCustomCall("PadRealToStatic") &&
          instruction->operand_count() > 0 &&
          instruction->operand(0)->shape().is_dynamic()) {
        pad_calls.push_back(instruction);
      }
    }
  }
  if (pad_calls.empty()) {
    return absl::OkStatus();
  }

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloDataflowAnalysis> dataflow,
                        HloDataflowAnalysis::Run(hlo_module));

  const auto& layout = hlo_module.entry_computation_layout();
  const bool flattened_tuple_parameter =
      layout.parameter_count() == 1 && layout.parameter_shape(0).IsTuple();

  absl::flat_hash_map<HloValue::Id, std::vector<const HloValue*>> memo;
  absl::flat_hash_set<HloValue::Id> allowed_embedded_param_values;
  std::vector<EmbeddedParamOrigin> embedded_origins;
  absl::flat_hash_set<HloValue::Id> seen_entry_values;
  std::vector<const HloValue*> entry_parameter_values;
  for (const HloInstruction* pad : pad_calls) {
    const std::vector<const HloValue*>& values =
        dataflow->GetValueSet(pad->operand(0), {}).values();
    if (values.empty()) {
      return InvalidArgument(
          "Dynamic operand of PadRealToStatic (%s) in %s does not resolve to "
          "an entry computation parameter.",
          pad->ToString(), hlo_module.name());
    }
    for (const HloValue* value : values) {
      absl::flat_hash_set<HloValue::Id> active_stack;
      ABSL_RETURN_IF_ERROR(ResolvePadRealToStaticOriginValues(
                               hlo_module, pad, *dataflow, value, memo,
                               active_stack, allowed_embedded_param_values,
                               embedded_origins, seen_entry_values,
                               entry_parameter_values)
                               .status());
    }
  }

  for (const EmbeddedParamOrigin& origin : embedded_origins) {
    const HloInstruction* entry_def =
        origin.entry_parameter_value->defining_instruction();
    const Shape& entry_leaf_shape = ShapeUtil::GetSubshape(
        entry_def->shape(), origin.entry_parameter_value->defining_index());
    if (!entry_leaf_shape.IsArray() || !entry_leaf_shape.is_dynamic()) {
      continue;
    }
    ABSL_RETURN_IF_ERROR(VerifyOnlyPadRealToStaticReadsValue(
        *dataflow, allowed_embedded_param_values, *origin.embedded_value,
        *origin.entry_parameter_value));
  }

  std::vector<const HloValue*> parameter_values;
  parameter_values.reserve(entry_parameter_values.size());
  for (const HloValue* value : entry_parameter_values) {
    const HloInstruction* def = value->defining_instruction();
    const Shape& leaf_shape =
        ShapeUtil::GetSubshape(def->shape(), value->defining_index());
    if (!leaf_shape.IsArray() || !leaf_shape.is_dynamic()) {
      continue;
    }
    ABSL_RETURN_IF_ERROR(VerifyOnlyPadRealToStaticReadsValue(
        *dataflow, allowed_embedded_param_values, *value, *value));
    if (std::optional<ShapeIndex> aliased_output =
            hlo_module.input_output_alias_config().GetAliasedOutput(
                def->parameter_number(), value->defining_index());
        aliased_output.has_value()) {
      return InvalidArgument(
          "Entry parameter %d%s of %s feeds PadRealToStatic and is aliased "
          "to output %s. A dynamic parameter consumed by PadRealToStatic is "
          "passed as a compact runtime-sized buffer without dynamic-shape "
          "metadata, so it cannot back an aliased or donated output.",
          def->parameter_number(),
          value->defining_index().empty()
              ? ""
              : absl::StrCat(" ", value->defining_index().ToString()),
          hlo_module.name(), aliased_output->ToString());
    }
    parameter_values.push_back(value);
  }

  // Mutate `parameter_shapes` only after all validations succeed so an error
  // leaves `parameter_shapes` untouched.
  for (const HloValue* value : parameter_values) {
    int64_t shape_no = value->defining_instruction()->parameter_number();
    ShapeIndex index = value->defining_index();
    if (flattened_tuple_parameter) {
      if (shape_no != 0 || index.empty()) {
        continue;
      }
      shape_no = index.front();
      index = ShapeIndex(index.begin() + 1, index.end());
    }
    if (shape_no < 0 || shape_no >= parameter_shapes.size() ||
        !ShapeUtil::IndexIsValid(parameter_shapes[shape_no], index)) {
      continue;
    }
    Shape* subshape =
        ShapeUtil::GetMutableSubshape(&parameter_shapes[shape_no], index);
    if (subshape->IsArray() && subshape->has_layout()) {
      subshape->mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(0);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<PjRtLoadedExecutableDispatchInfo> InferDispatchInfo(
    const PjRtTopologyDescription* topology,
    std::vector<Shape> parameter_device_shapes, Shape output_device_shape,
    const HloInputOutputAliasConfig& alias_config,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
        addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices,
    std::unique_ptr<PjRtLoadedExecutableDispatchInfo::Extras> extras,
    bool tuple_inputs,
    std::unique_ptr<PjRtLoadedExecutableDispatchInfo::InputHloSnapshotBits>
        input_hlo_snapshot_bits) {
  PjRtLoadedExecutableDispatchInfo result{
      .parameter_device_shapes = std::move(parameter_device_shapes),
      .output_device_shape =
          std::make_shared<const Shape>(std::move(output_device_shape)),
      .addressable_devices = std::move(addressable_devices),
      .addressable_device_logical_ids =
          std::move(addressable_device_logical_ids),
      .device_assignment = std::move(device_assignment),
      .input_hlo_snapshot_bits = std::move(input_hlo_snapshot_bits),
      .extras = std::move(extras),
  };
  for (const auto& shape : result.parameter_device_shapes) {
    ABSL_ASSIGN_OR_RETURN(int kind,
                          topology->GetMemorySpaceKindForShape(shape));
    result.parameter_memory_space_kind_ids.push_back(kind);
  }
  {
    absl::Span<const Shape> shapes =
        result.output_device_shape->IsTuple()
            ? absl::MakeSpan(result.output_device_shape->tuple_shapes())
            : absl::MakeSpan(&*result.output_device_shape, 1);
    result.output_memory_space_kind_ids.reserve(shapes.size());
    for (const auto& shape : shapes) {
      ABSL_ASSIGN_OR_RETURN(int kind,
                            topology->GetMemorySpaceKindForShape(shape));
      result.output_memory_space_kind_ids.push_back(kind);
    }
  }
  // Initializes information about which arguments to which executables may
  // be donated due to aliases that were specified by the computation.
  ABSL_ASSIGN_OR_RETURN(
      result.parameters_that_may_be_donated,
      ComputeParametersThatMayBeDonated(
          alias_config, result.parameter_device_shapes.size(), tuple_inputs));
  result.input_buffer_sizes_in_bytes.reserve(
      result.parameter_device_shapes.size());
  for (const Shape& shape : result.parameter_device_shapes) {
    DCHECK(!shape.IsTuple());
    ABSL_ASSIGN_OR_RETURN(int64_t size_in_bytes,
                          PjRtGetOnDeviceBytesCount(shape));
    result.input_buffer_sizes_in_bytes.push_back(size_in_bytes);
  }
  return result;
}

namespace {

absl::StatusOr<std::vector<int64_t>> GetShardShape(
    const xla::HloSharding& sharding, llvm::ArrayRef<int64_t> shape,
    size_t num_devices) {
  if (sharding.IsReplicatedOrSingleDevice() || sharding.IsManual() ||
      sharding.IsUnreduced() || sharding.IsUnknown()) {
    return std::vector<int64_t>{shape.begin(), shape.end()};
  }
  if (sharding.TotalNumTiles() != num_devices) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "sharding's tile count and device count does not "
        "match: %d vs. %d; shape=%s, sharding=%s",
        sharding.TotalNumTiles(), num_devices, "??", sharding.ToString()));
  }
  if (shape.size() != sharding.TiledDataRank()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Numbers of dimensions don't match. From Shape %d vs from "
        "HloSharding %d",
        shape.size(), sharding.TiledDataRank()));
  }
  const absl::Span<const int64_t> sharding_dims = sharding.dimensions();
  std::vector<int64_t> tile_shape;
  tile_shape.reserve(shape.size());
  for (int64_t i = 0; i < shape.size(); ++i) {
    tile_shape.push_back(shape[i] == Shape::kUnboundedSize
                             ? Shape::kUnboundedSize
                             : xla::CeilOfRatio(shape[i], sharding_dims[i]));
  }
  return tile_shape;
}

}  // namespace

absl::StatusOr<PjRtLoadedExecutableDispatchInfo> InferDispatchInfo(
    const PjRtTopologyDescription* topology, mlir::ModuleOp mlir_module,
    const CompileOptions& options,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
        addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, bool tuple_inputs) {
  if (!device_assignment) {
    return absl::UnimplementedError(
        "Async compilation requires a device_assignment");
  }
  auto extras = std::make_unique<PjRtLoadedExecutableDispatchInfo::Extras>();
  extras->name =
      std::string(mlir_module.getSymName().value_or("?unknown program name?"));
  extras->num_partitions = device_assignment->replica_count();
  extras->num_replicas = device_assignment->replica_count();
  auto mesh = mlir_module.getOps<mlir::sdy::MeshOp>();
  if (mesh.empty() || std::next(mesh.begin()) != mesh.end()) {
    return absl::UnimplementedError("Exactly one shardy mesh is required.");
  }
  auto main = mlir_module.lookupSymbol<mlir::func::FuncOp>("main");
  if (main == nullptr) {
    return absl::UnimplementedError(
        "InferAvalFromModule called on module without main function.");
  }
  auto sharding_fn = [&](mlir::sdy::TensorShardingAttr sharding) {
    return sharding.getMesh(main);
  };
  auto get_xla_shape = [&](xla::HloSharding sharding, mlir::Type type)
      -> absl::StatusOr<
          std::tuple<xla::Shape, std::shared_ptr<const xla::PjRtLayout>>> {
    if (mlir::isa<mlir::UnrankedTensorType>(type)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported type ", mlir::debugString(type)));
    }
    std::vector<int64_t> shard_shape;
    xla::PrimitiveType primitive_type;
    std::vector<bool> dynamic_dimensions;
    if (auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
      xla::Shape unsharded_shape = xla::TypeToShape(tensor_type);
      if (unsharded_shape.element_type() == PRIMITIVE_TYPE_INVALID) {
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported type ", mlir::debugString(type)));
      }
      llvm::ArrayRef<int64_t> dims(unsharded_shape.dimensions().data(),
                                   unsharded_shape.dimensions().size());
      ABSL_ASSIGN_OR_RETURN(
          shard_shape,
          GetShardShape(sharding, dims,
                        extras->num_replicas * extras->num_partitions));
      primitive_type = unsharded_shape.element_type();
      dynamic_dimensions.assign(unsharded_shape.dynamic_dimensions().begin(),
                                unsharded_shape.dynamic_dimensions().end());
    } else {
      primitive_type = xla::ConvertMlirTypeToPrimitiveType(type);
    }
    ABSL_ASSIGN_OR_RETURN(auto* memory_space,
                          addressable_devices[0]->default_memory_space());
    xla::Shape xla_shard_shape =
        dynamic_dimensions.empty()
            ? xla::ShapeUtil::MakeShape(primitive_type, shard_shape)
            : xla::ShapeUtil::MakeShape(primitive_type, shard_shape,
                                        dynamic_dimensions);
    // TODO(parkers): Fix the nullptr layout.
    ABSL_ASSIGN_OR_RETURN(
        auto xla_shape, topology->MakeCanonicalShapeForMemorySpace(
                            memory_space->kind_id(), xla_shard_shape, nullptr));
    auto layout = std::make_shared<PjRtLayout>(xla_shape.layout());
    return std::make_tuple(xla_shape, layout);
  };
  std::vector<xla::Shape> parameter_device_shapes;
  extras->parameter_shardings.emplace();
  extras->parameter_layouts = std::vector<std::shared_ptr<const PjRtLayout>>();
  for (size_t i = 0; i < main.getNumArguments(); ++i) {
    auto sharding_attr = mlir::sdy::getSharding(main.getArgument(i));
    if (sharding_attr == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Arg[%d] is missing a sharding", i));
    }
    xla::HloSharding hlo_sharding =
        xla::sdy::convertToHloSharding(sharding_attr, sharding_fn,
                                       /*manualAxes=*/{});
    xla::Shape shape;
    std::shared_ptr<const xla::PjRtLayout> layout;
    ABSL_ASSIGN_OR_RETURN(
        (std::tie(shape, layout)),
        get_xla_shape(hlo_sharding, main.getArgumentTypes()[i]));

    // Inputs feeding PadRealToStatic custom calls are bounded dynamic tensors
    // supplied without the TPU hardware dynamic metadata prefix (0 bytes).
    // On the async-compile path, CommonPjRtClient::CompileAndLoad builds
    // DispatchInfo directly from this MLIR pass without calling LoadInternal,
    // whereas sync/loaded executables validate and clear prefixes post-compile
    // in ClearPadRealToStaticParameterMetadataPrefixes.
    if (shape.is_dynamic()) {
      mlir::BlockArgument arg = main.getArgument(i);
      const bool feeds_pad_real_to_static =
          llvm::any_of(arg.getUses(), [](mlir::OpOperand& use) {
            return use.getOperandNumber() == 0 &&
                   IsPadRealToStaticOp(use.getOwner());
          });
      if (feeds_pad_real_to_static) {
        for (mlir::OpOperand& use : arg.getUses()) {
          if (use.getOperandNumber() != 0 ||
              !IsPadRealToStaticOp(use.getOwner())) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Entry argument %d feeds PadRealToStatic and has another use "
                "(%s); PadRealToStatic dynamic inputs have no metadata prefix "
                "and must have no other users.",
                i, use.getOwner()->getName().getStringRef().str()));
          }
        }
        if (main.getArgAttr(i, "tf.aliasing_output") != nullptr) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Entry argument %d feeds PadRealToStatic and is aliased via "
              "tf.aliasing_output; compact PadRealToStatic buffers cannot back "
              "aliased or donated outputs.",
              i));
        }
        shape.mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(0);
        layout = std::make_shared<PjRtLayout>(shape.layout());
      }
    }

    parameter_device_shapes.push_back(shape);
    extras->parameter_shardings->push_back(hlo_sharding.ToProto());
    extras->parameter_layouts->push_back(std::move(layout));
  }
  std::vector<xla::Shape> result_shapes;
  extras->output_shardings.emplace();
  extras->output_layouts = std::vector<std::shared_ptr<const PjRtLayout>>();
  for (size_t i = 0; i < main.getNumResults(); ++i) {
    auto sharding_attr = mlir::sdy::getFuncResultSharding(main, i);
    // sharding_attr.dump();
    if (sharding_attr == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Result[%d] is missing a sharding", i));
    }
    xla::HloSharding hlo_sharding =
        xla::sdy::convertToHloSharding(sharding_attr, sharding_fn,
                                       /*manualAxes=*/{});

    xla::Shape shape;
    std::shared_ptr<const xla::PjRtLayout> layout;
    ABSL_ASSIGN_OR_RETURN(
        (std::tie(shape, layout)),
        get_xla_shape(hlo_sharding, main.getResultTypes()[i]));

    result_shapes.push_back(std::move(shape));
    extras->output_shardings->push_back(hlo_sharding.ToProto());
    extras->output_layouts->push_back(std::move(layout));
  }
  Shape output_device_shape = (result_shapes.size() == 1)
                                  ? std::move(result_shapes[0])
                                  : xla::Shape(std::move(result_shapes));

  // TODO(parkers): implement aliasing.
  extras->input_output_alias_config =
      HloInputOutputAliasConfig(output_device_shape);

  const auto& input_output_alias_config = extras->input_output_alias_config;
  ABSL_ASSIGN_OR_RETURN(
      auto result,
      InferDispatchInfo(topology, std::move(parameter_device_shapes),
                        std::move(output_device_shape),
                        input_output_alias_config, std::move(device_assignment),
                        std::move(addressable_device_logical_ids),
                        std::move(addressable_devices), std::move(extras),
                        tuple_inputs));

  result.extras->fingerprint = "";
  result.extras->parameter_memory_kinds.reserve(
      result.parameter_memory_space_kind_ids.size());
  for (size_t i = 0; i < result.parameter_memory_space_kind_ids.size(); ++i) {
    auto* device = result.addressable_devices[0];
    ABSL_ASSIGN_OR_RETURN(auto* memory_space, device->default_memory_space());
    for (auto* ms : device->memory_spaces()) {
      if (ms->kind_id() == result.parameter_memory_space_kind_ids[i]) {
        memory_space = ms;
      }
    }
    result.extras->parameter_memory_kinds.push_back(memory_space->kind());
  }
  result.extras->output_memory_kinds.reserve(
      result.output_memory_space_kind_ids.size());
  for (size_t i = 0; i < result.output_memory_space_kind_ids.size(); ++i) {
    auto* device = result.addressable_devices[0];
    ABSL_ASSIGN_OR_RETURN(auto* memory_space, device->default_memory_space());
    for (auto* ms : device->memory_spaces()) {
      if (ms->kind_id() == result.output_memory_space_kind_ids[i]) {
        memory_space = ms;
      }
    }
    result.extras->output_memory_kinds.push_back(memory_space->kind());
  }
  return result;
}

}  // namespace xla
