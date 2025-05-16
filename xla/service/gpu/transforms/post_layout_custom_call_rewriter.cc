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

#include "xla/service/gpu/transforms/post_layout_custom_call_rewriter.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/ffi/attribute_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/memory_annotations.h"
#include "xla/service/gpu/gpu_memory_space_assignment.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"

namespace xla::gpu {
namespace {

using ::xla::ffi::CallFrameBuilder;

absl::StatusOr<bool> RewriteAllocatePersistentBuffer(
    HloCustomCallInstruction* custom_call) {
  CHECK(custom_call->api_version() ==
        CustomCallApiVersion::API_VERSION_TYPED_FFI);
  CHECK_EQ(custom_call->operand_count(), 1);
  const auto& backend_config_str = custom_call->raw_backend_config_string();

  mlir::MLIRContext context;
  mlir::Attribute attr = mlir::parseAttribute(backend_config_str, &context);
  CallFrameBuilder::AttributesMap attributes;
  if (auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr)) {
    // Convert the MLIR dictionary to FFI attributes.
    TF_ASSIGN_OR_RETURN(attributes, ffi::BuildAttributesMap(dict));
  } else {
    return Internal(
        "Unsupported backend config. Expected a string parsable into "
        "dictionary attribute");
  }
  auto it = attributes.find("memory_space");
  if (it == attributes.end()) {
    return Internal("memory_space attribute not found in backend config.");
  }

  const auto& memory_space_str = std::get<std::string>(it->second);
  auto* operand = custom_call->mutable_operand(0);
  if (memory_space_str == "persistent_temp") {
    operand->mutable_shape()->mutable_layout()->set_memory_space(
        kTempBufferMemorySpaceColor);
  } else if (memory_space_str == "host") {
    operand->mutable_shape()->mutable_layout()->set_memory_space(
        static_cast<int64_t>(stream_executor::MemoryType::kHost));
  } else {
    return Internal("Unsupported memory space.");
  }
  TF_RETURN_IF_ERROR(custom_call->ReplaceAllUsesWith(operand));
  return true;
}
}  // namespace

absl::StatusOr<bool> PostLayoutCustomCallRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->IsCustomCall(memory_annotations::kAnnotateMemorySpace)) {
        TF_ASSIGN_OR_RETURN(bool rewrited,
                            RewriteAllocatePersistentBuffer(
                                Cast<HloCustomCallInstruction>(instruction)));
        changed |= rewrited;
      }
    }
  }

  if (changed) {
    HloDCE hlo_dce;
    TF_ASSIGN_OR_RETURN(bool dced, hlo_dce.Run(module, execution_threads));
  }

  return changed;
}
}  // namespace xla::gpu
