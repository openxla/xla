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

#include "xla/service/gpu/gpu_executable_types.h"

#include <cstdint>
#include <string>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape_tree.h"

namespace xla::gpu {

GpuExecutableProto::OutputInfoProto GpuExecutableOutputInfo::ToProto() const {
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

absl::StatusOr<GpuExecutableOutputInfo> GpuExecutableOutputInfo::FromProto(
    const GpuExecutableProto::OutputInfoProto& proto) {
  GpuExecutableOutputInfo output_info;
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

GpuExecutableProto::ConstantInfoProto GpuExecutableConstantInfo::ToProto()
    const {
  GpuExecutableProto::ConstantInfoProto proto;
  proto.set_symbol_name(symbol_name);
  *proto.mutable_content() = content.ToProto();
  proto.set_allocation_index(allocation_index);
  return proto;
}

absl::StatusOr<GpuExecutableConstantInfo> GpuExecutableConstantInfo::FromProto(
    const GpuExecutableProto::ConstantInfoProto& proto,
    const absl::flat_hash_map<std::string, const HloInstruction*>* absl_nullable
        content_overrides) {
  if (content_overrides) {
    auto it = content_overrides->find(proto.symbol_name());
    if (it == content_overrides->end()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Instruction for ", proto.symbol_name(), " constant missing."));
    }
    const HloInstruction* instr = it->second;
    const Literal& literal = instr->literal();
    auto base = static_cast<const uint8_t*>(literal.untyped_data());
    return GpuExecutableConstantInfo{
        proto.symbol_name(),
        DenseDataIntermediate::Alias(
            absl::MakeSpan(base, base + literal.size_bytes())),
        static_cast<int>(proto.allocation_index())};
  }
  return GpuExecutableConstantInfo{
      proto.symbol_name(), DenseDataIntermediate::FromProto(proto.content()),
      static_cast<int>(proto.allocation_index())};
}

}  // namespace xla::gpu
