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

#include "xla/backends/gpu/transforms/custom_call_scratch_assigner.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/ffi/ffi_attributes_from_backend_config.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_constants.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registry.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_scratch_context.h"
#include "xla/ffi/attributes.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/gpu_memory_space_assignment.h"
#include "xla/service/gpu_topology.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/logging.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

class ScratchContextImpl : public NativeCustomCallScratchContext {
 public:
  ScratchContextImpl(const HloCustomCallInstruction& instr,
                     const GpuTopology& topology,
                     const DebugOptions& debug_options,
                     mlir::MLIRContext* mlir_context)
      : instr_(instr),
        topology_(topology),
        debug_options_(debug_options),
        mlir_context_(mlir_context) {}

  const GpuTopology& GetTargetTopology() const override { return topology_; }

  const stream_executor::DeviceDescription& GetDeviceDescription()
      const override {
    return topology_.gpu_target_config().device_description;
  }

  const DebugOptions& GetDebugOptions() const override {
    return debug_options_;
  }

  absl::StatusOr<xla::ffi::Attributes> GetFfiAttributes() const override {
    return FfiAttributesFromBackendConfig(instr_, *mlir_context_);
  }

 private:
  const HloCustomCallInstruction& instr_;
  const GpuTopology& topology_;
  const DebugOptions& debug_options_;
  mlir::MLIRContext* mlir_context_;
};

// Checks that `shape` is a valid scratch buffer shape and fills in the default
// layout if the handler didn't specify one.
absl::StatusOr<Shape> ValidateScratchShape(
    Shape shape, const HloCustomCallInstruction& custom_call, int64_t index) {
  auto error = [&](absl::string_view what) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Scratch buffer %d of custom call '%s' (target '%s') %s: %s", index,
        custom_call.name(), custom_call.custom_call_target(), what,
        shape.ToString(/*print_layout=*/true)));
  };
  if (!shape.IsArray()) {
    return error("must be an array shape");
  }
  if (!shape.is_static()) {
    return error("must have static dimensions");
  }
  if (!shape.has_layout()) {
    LayoutUtil::SetToDefaultLayout(&shape);
  }
  absl::StatusOr<MemorySpaceColor> color =
      AsMemorySpaceColor(shape.layout().memory_space());
  if (!color.ok() || (*color != MemorySpaceColor::kDefault &&
                      *color != MemorySpaceColor::kCollective)) {
    return error(absl::StrFormat(
        "has an unsupported memory space %d (supported are %d and %d)",
        shape.layout().memory_space(), MemorySpaceColor::kDefault,
        MemorySpaceColor::kCollective));
  }
  return shape;
}

}  // namespace

absl::StatusOr<bool> CustomCallScratchAssigner::RunOnCustomCall(
    HloCustomCallInstruction* custom_call) {
  if (custom_call->get_frontend_attribute(
          kNativeCustomCallNumScratchBuffersAttr)) {
    return false;
  }
  std::optional<NativeCustomCallScratchHandlerRef> handler =
      NativeCustomCallHandlerRegistry::GetGlobal().LookupScratchHandler(
          custom_call->custom_call_target());
  if (!handler.has_value()) {
    return false;
  }

  const DebugOptions& debug_options =
      custom_call->GetModule()->config().debug_options();
  ScratchContextImpl context(*custom_call, *gpu_topology_, debug_options,
                             mlir_context_);
  ABSL_ASSIGN_OR_RETURN(std::vector<Shape> scratch_shapes,
                        (*handler)(*custom_call, context));
  if (scratch_shapes.empty()) {
    return false;
  }

  // The new result is the original result as element 0 followed by the scratch
  // buffers. We do not flatten the original return type when it is already a
  // tuple: keeping it as element 0 avoids having to rebuild the tuple from
  // individual elements, which prevents unnecessary tuple table materialization
  // when consumers consume the whole tuple.
  const Shape& original_shape = custom_call->shape();
  std::vector<Shape> result_shapes;
  result_shapes.reserve(1 + scratch_shapes.size());
  result_shapes.push_back(original_shape);
  for (int64_t i = 0; i < scratch_shapes.size(); ++i) {
    ABSL_ASSIGN_OR_RETURN(
        Shape scratch_shape,
        ValidateScratchShape(std::move(scratch_shapes[i]), *custom_call, i));
    result_shapes.push_back(std::move(scratch_shape));
  }
  Shape new_shape = ShapeUtil::MakeTupleShape(result_shapes);

  HloComputation* computation = custom_call->parent();
  auto* new_custom_call = Cast<HloCustomCallInstruction>(
      computation->AddInstruction(custom_call->CloneWithNewOperands(
          new_shape, custom_call->operands())));
  new_custom_call->add_frontend_attribute(
      kNativeCustomCallNumScratchBuffersAttr,
      absl::StrCat(scratch_shapes.size()));

  // If the custom call has a sharding, preserve it on the new custom call by
  // placing the original sharding on element 0 and replicating the scratch
  // buffers.
  if (custom_call->has_sharding()) {
    ShapeTree<HloSharding> new_sharding_tree(new_shape,
                                             HloSharding::Replicate());
    ABSL_ASSIGN_OR_RETURN(ShapeTree<HloSharding> orig_tree,
                          custom_call->sharding().AsShapeTree(original_shape));
    new_sharding_tree.CopySubtreeFrom(orig_tree, /*src_index=*/{},
                                      /*dst_index=*/{0});
    new_custom_call->set_sharding(HloSharding::Tuple(new_sharding_tree));
  }

  // The original result is now element 0 of the tuple. Update output-to-operand
  // aliasing by prepending 0 to every output shape index.
  if (!custom_call->output_operand_aliasing().empty()) {
    std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>> aliasing;
    aliasing.reserve(custom_call->output_operand_aliasing().size());
    for (const auto& [output_index, operand] :
         custom_call->output_operand_aliasing()) {
      ShapeIndex new_output_index = output_index;
      new_output_index.push_front(0);
      aliasing.push_back({new_output_index, operand});
    }
    new_custom_call->set_output_to_operand_aliasing(std::move(aliasing));
  }

  // Replace all uses of the original custom call with element 0 of the new
  // tuple so that users (and the computation root) keep seeing the original
  // shape.
  HloInstruction* replacement = computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(new_custom_call, 0));

  // Control dependencies belong to the custom call itself, not to the
  // get-tuple-element that replaces it in the data flow.
  ABSL_RETURN_IF_ERROR(new_custom_call->CopyAllControlDepsFrom(custom_call));
  ABSL_RETURN_IF_ERROR(custom_call->DropAllControlDeps());

  std::string name(custom_call->name());
  // The frontend attributes (including the marker attribute set above) belong
  // to the custom call and must not be copied to the replacement.
  ABSL_ASSIGN_OR_RETURN(
      bool replaced, computation->ReplaceInstruction(
                         custom_call, replacement, /*preserve_sharding=*/false,
                         /*relay_control_dependency=*/false,
                         /*remove_unused_operands=*/true,
                         /*preserve_frontend_attributes=*/false));
  TF_RET_CHECK(replaced);
  // Keep the name so that the custom call stays easy to find in dumps and
  // tests.
  new_custom_call->SetAndSanitizeName(name);
  return true;
}

absl::StatusOr<bool> CustomCallScratchAssigner::RunOnComputation(
    HloComputation* computation) {
  // Collect first, since RunOnCustomCall mutates the instruction list.
  std::vector<HloCustomCallInstruction*> custom_calls;
  for (HloInstruction* instr : computation->instructions()) {
    if (auto* custom_call = DynCast<HloCustomCallInstruction>(instr)) {
      custom_calls.push_back(custom_call);
    }
  }
  bool changed = false;
  for (HloCustomCallInstruction* custom_call : custom_calls) {
    ABSL_ASSIGN_OR_RETURN(bool result, RunOnCustomCall(custom_call));
    changed |= result;
  }
  return changed;
}

absl::StatusOr<bool> CustomCallScratchAssigner::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(3, "CustomCallScratchAssigner::RunImpl(), before:\n" +
                        module->ToString());
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    ABSL_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  XLA_VLOG_LINES(
      3, "CustomCallScratchAssigner::RunImpl(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla::gpu
