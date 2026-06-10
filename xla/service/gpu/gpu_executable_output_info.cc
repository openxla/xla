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

#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

absl::StatusOr<absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>>
GetOutputInfo(const HloModule& hlo_module, const BufferAssignment& assignment) {
  const HloInstruction* root =
      hlo_module.entry_computation()->root_instruction();

  InstructionValueSet root_value_set =
      assignment.dataflow_analysis().GetInstructionValueSet(root);

  if (root_value_set.IsAmbiguous()) {
    return Unimplemented("Points-to set of root instruction is ambiguous");
  }

  using OutputInfoMap =
      absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>;
  OutputInfoMap output;
  RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      root->shape(),
      [&](const Shape& /*sub_shape*/, const ShapeIndex& index) -> absl::Status {
        const auto& sources = root_value_set.element(index);
        // The points-to set is unambiguous so the set should be a
        // singleton. That is, we know exactly which instruction
        // produced the array at this element.
        CHECK_EQ(1, sources.values().size());
        HloInstruction* src_hlo = sources.values()[0]->instruction();

        GpuExecutable::OutputInfo& info = output[index];
        info.passthrough = src_hlo->opcode() == HloOpcode::kParameter;
        ASSIGN_OR_RETURN(
            const BufferAllocation::Slice slice,
            assignment.GetUniqueSlice(src_hlo, sources.values()[0]->index()));
        CHECK_EQ(slice.offset(), 0) << "Parameter should get its own slice";
        info.allocation_index = slice.index();

        output[index].alias_config =
            hlo_module.input_output_alias_config().GetAliasedParameter(index);

        return absl::OkStatus();
      }));
  return output;
}

GpuExecutableProto::OutputInfoProto GpuExecutable::OutputInfo::ToProto() const {
  GpuExecutableProto::OutputInfoProto proto;
  proto.set_allocation_index(allocation_index);
  proto.set_passthrough(passthrough);

  if (alias_config.has_value()) {
    proto.mutable_alias_config()->set_parameter_number(
        alias_config->parameter_number);
    proto.mutable_alias_config()->mutable_parameter_shape_index()->Assign(
        alias_config->parameter_index.begin(),
        alias_config->parameter_index.end());

    switch (alias_config->kind) {
      case xla::HloInputOutputAliasConfig::AliasKind::kMayAlias:
        proto.mutable_alias_config()->set_kind(Kind::MAY_ALIAS);
        break;
      case xla::HloInputOutputAliasConfig::AliasKind::kMustAlias:
        proto.mutable_alias_config()->set_kind(Kind::MUST_ALIAS);
        break;
    }
  }

  return proto;
}

absl::StatusOr<GpuExecutable::OutputInfo> GpuExecutable::OutputInfo::FromProto(
    const GpuExecutableProto::OutputInfoProto& proto) {
  OutputInfo output_info;
  output_info.allocation_index = proto.allocation_index();
  output_info.passthrough = proto.passthrough();
  if (proto.has_alias_config()) {
    xla::HloInputOutputAliasConfig::AliasKind alias_kind;
    switch (proto.alias_config().kind()) {
      case Kind::MAY_ALIAS:
        alias_kind = xla::HloInputOutputAliasConfig::AliasKind::kMayAlias;
        break;
      case Kind::MUST_ALIAS:
        alias_kind = xla::HloInputOutputAliasConfig::AliasKind::kMustAlias;
        break;
      default:
        return absl::InvalidArgumentError("Given alias kind is invalid");
    }
    const auto& parameter_shape_index =
        proto.alias_config().parameter_shape_index();
    output_info.alias_config.emplace(
        proto.alias_config().parameter_number(),
        ShapeIndex{parameter_shape_index.begin(), parameter_shape_index.end()},
        alias_kind);
  }
  return output_info;
}

}  // namespace xla::gpu
