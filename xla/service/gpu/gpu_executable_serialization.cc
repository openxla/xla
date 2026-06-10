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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "riegeli/bytes/string_writer.h"
#include "riegeli/bytes/writer.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_proto_deserialization.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_value.h"
#include "xla/service/llvm_ir/buffer_assignment_util.h"
#include "xla/service/riegeli_dump_writer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/tsl/util/sorted_range.h"
#include "xla/util.h"
#include "xla/util/split_proto/split_executable_and_options_writer.h"
#include "xla/util/split_proto/split_gpu_executable_writer.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

std::optional<absl::flat_hash_map<std::string, const HloInstruction*>>
MakeConstantsMap(HloModule* absl_nullable debug_module) {
  if (!debug_module) {
    return std::nullopt;
  }
  absl::flat_hash_map<std::string, const HloInstruction*> constants;
  for (const HloComputation* computation :
       debug_module->MakeComputationSorted()) {
    for (const HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() != HloOpcode::kConstant) {
        continue;
      }
      if (llvm_ir::SanitizeConstantName(*instr) != instr->name()) {
        continue;
      }
      auto [it, inserted] = constants.try_emplace(
          llvm_ir::ConstantHloToGlobalName(*instr), instr);
      CHECK(inserted) << "Duplicate constant global name found: " << it->first;
    }
  }
  return constants;
}

}  // namespace

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

GpuExecutableProto::ConstantInfoProto GpuExecutable::ConstantInfo::ToProto()
    const {
  GpuExecutableProto::ConstantInfoProto proto;
  proto.set_symbol_name(symbol_name);
  *proto.mutable_content() = content.ToProto();
  proto.set_allocation_index(allocation_index);
  return proto;
}

absl::StatusOr<GpuExecutable::ConstantInfo>
GpuExecutable::ConstantInfo::FromProto(
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
    return ConstantInfo{proto.symbol_name(),
                        DenseDataIntermediate::Alias(
                            absl::MakeSpan(base, base + literal.size_bytes())),
                        static_cast<int>(proto.allocation_index())};
  }
  return ConstantInfo{proto.symbol_name(),
                      DenseDataIntermediate::FromProto(proto.content()),
                      static_cast<int>(proto.allocation_index())};
}

absl::StatusOr<GpuExecutableProto> GpuExecutable::ToProto() const {
  GpuExecutableProto proto;
  proto.set_binary(binary_.data(), binary_.size());
  proto.set_asm_text(text_);
  proto.mutable_dnn_compiled_graphs()->insert(dnn_compiled_graphs_.cbegin(),
                                              dnn_compiled_graphs_.cend());

  *proto.mutable_gpu_compute_capability() = gpu_version_.ToProto();

  // TODO(b/461380690): Generate the proto on-the-fly once we have a better way
  // to distinguish between compiler-generated and runtime-loaded GPU
  // executables.
  ASSIGN_OR_RETURN(const auto& thunk_sequence_proto, thunk_sequence_proto_);
  proto.mutable_thunks()->Reserve(thunk_sequence_proto.size());
  for (const auto& thunk_proto : thunk_sequence_proto) {
    *proto.add_thunks() = thunk_proto;
  }

  proto.set_module_name(module_name_);
  *proto.mutable_program_shape() = program_shape_.ToProto();

  absl::Span<const BufferAllocation* const> allocations = GetAllocations();
  proto.mutable_buffer_allocations()->mutable_values()->Reserve(
      allocations.size());
  for (const auto& allocation : allocations) {
    proto.mutable_buffer_allocations()->mutable_values()->Add(
        allocation->ToProto());
  }

  if (buffer_assignment_ != nullptr) {
    *proto.mutable_buffer_assignment() = buffer_assignment_->ToProto();
  } else if (buffer_assignment_proto_.has_value()) {
    *proto.mutable_buffer_assignment() = buffer_assignment_proto_.value();
  }

  if (has_module()) {
    *proto.mutable_hlo_module_with_config() = module().ToProtoWithConfig(
        HloProtoOptions{/*deduplicate_backend_config=*/true});
  }

  proto.mutable_output_info_map()->Reserve(output_info_.size());
  for (const auto& [shape_index, output_info] :
       tsl::KeySortedRange(output_info_)) {
    auto map_entry = proto.add_output_info_map();
    *map_entry->mutable_shape_index() = shape_index.ToProto();
    *map_entry->mutable_output_info() = output_info.ToProto();
  }

  proto.mutable_constants()->Reserve(constants_.size());
  for (const auto& constant : constants_) {
    *proto.add_constants() = constant.ToProto();
  }

  *proto.mutable_executable_abi_version() = executable_abi_version_.proto();

  if (cpu_target_machine_options_.has_value()) {
    *proto.mutable_cpu_target_machine_options() =
        cpu_target_machine_options_->ToProto();
  }

  return proto;
}

absl::StatusOr<std::unique_ptr<GpuExecutable>> GpuExecutable::FromProto(
    const GpuExecutableProto& proto,
    const se::DeviceDescription& device_description,
    absl::string_view platform_name, DebugOptions debug_options,
    const std::optional<se::KernelLoaderSpec::SymbolResolver>&
        symbol_resolver) {
  Params params;
  params.debug_options = std::move(debug_options);
  params.enable_debug_info_manager =
      params.debug_options.xla_gpu_executable_embed_debug_info();
  params.asm_text = proto.asm_text();
  const std::string& binary = proto.binary();
  params.binary.assign(binary.begin(), binary.end());
  params.buffer_assignment = nullptr;
  if (proto.has_hlo_module_with_config()) {
    ASSIGN_OR_RETURN(params.debug_module, HloModule::CreateFromProtoWithConfig(
                                              proto.hlo_module_with_config()));
    // The HLO module deserialized from the proto carries xla_dump_to from the
    // process that originally compiled it. Override with the current process's
    // dump path so that runtime dumps (checksum logs, etc.) land in the correct
    // per-process directory.
    if (params.debug_options.has_xla_dump_to()) {
      params.debug_module->mutable_config()
          .mutable_debug_options()
          .set_xla_dump_to(params.debug_options.xla_dump_to());
    }
  }
  if (proto.has_buffer_assignment()) {
    params.buffer_assignment_proto.emplace(proto.buffer_assignment());
  }

  params.mlir_allocations.emplace();
  params.mlir_allocations->reserve(proto.buffer_allocations().values_size());
  for (const BufferAllocationProto& allocation_proto :
       proto.buffer_allocations().values()) {
    params.mlir_allocations->push_back(
        BufferAllocation::FromProto(allocation_proto));
  }

  for (const auto& [key, value] : proto.dnn_compiled_graphs()) {
    params.dnn_compiled_graphs.emplace(key, value);
  }

  ASSIGN_OR_RETURN(
      se::GpuComputeCapability gpu_compute_capability,
      se::GpuComputeCapability::FromProto(proto.gpu_compute_capability()));

  if (gpu_compute_capability != device_description.gpu_compute_capability()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "GPU compute capability of serialized executable doesn't match target "
        "device capability. (serialized: %s, target: %s)",
        gpu_compute_capability.ToString(),
        device_description.gpu_compute_capability().ToString()));
  }

  params.device_description = device_description;

  if (proto.has_cpu_target_machine_options()) {
    ASSIGN_OR_RETURN(params.cpu_target_machine_options,
                     xla::cpu::TargetMachineOptions::FromProto(
                         proto.cpu_target_machine_options()));
  }

  ThunkSequenceProto thunk_sequence_proto;
  *thunk_sequence_proto.mutable_thunks() = proto.thunks();
  ASSIGN_OR_RETURN(
      ThunkSequence thunk_sequence,
      DeserializeThunkSequenceProto(
          thunk_sequence_proto, params.mlir_allocations.value(),
          params.debug_module.get(), platform_name, gpu_compute_capability,
          symbol_resolver, params.cpu_target_machine_options));

  params.executable =
      std::make_unique<ThunkExecutor>(std::move(thunk_sequence));

  std::optional<absl::flat_hash_map<std::string, const HloInstruction*>>
      name_to_const = MakeConstantsMap(params.debug_module.get());

  params.constants.reserve(proto.constants().size());
  for (const auto& constant_proto : proto.constants()) {
    ASSIGN_OR_RETURN(
        params.constants.emplace_back(),
        ConstantInfo::FromProto(constant_proto, name_to_const.has_value()
                                                    ? &*name_to_const
                                                    : nullptr));
  }

  params.output_info.reserve(proto.output_info_map().size());
  for (const auto& output_info_proto : proto.output_info_map()) {
    ShapeIndex shape_index =
        ShapeIndex::FromProto(output_info_proto.shape_index());
    ASSIGN_OR_RETURN(OutputInfo output_info,
                     OutputInfo::FromProto(output_info_proto.output_info()));
    params.output_info.emplace(std::move(shape_index), std::move(output_info));
  }

  params.module_name = proto.module_name();
  ASSIGN_OR_RETURN(params.program_shape,
                   ProgramShape::FromProto(proto.program_shape()));

  ASSIGN_OR_RETURN(
      params.executable_abi_version,
      se::ExecutableAbiVersion::FromProto(proto.executable_abi_version()));

  return Create(std::move(params));
}

static absl::StatusOr<ExecutableBuildOptionsProto>
CreateSerializableBuildOptionsProto(const ExecutableBuildOptions& options) {
  ExecutableBuildOptions serializable_opts = options;
  // These fields are not serializable, and the toProto will fail if they are
  // set, but we also don't need them for the dump so just clear them.
  serializable_opts.set_layout_canonicalization_callback(nullptr);
  serializable_opts.set_compile_thread_pool(nullptr);

  return serializable_opts.ToProto();
}

absl::Status GpuExecutable::DumpExecutableIfEnabled(
    const ExecutableBuildOptions& options,
    const DebugOptions& debug_options) const {
  if (!debug_options.has_xla_dump_to() ||
      !debug_options.xla_gpu_experimental_dump_gpu_executable()) {
    return absl::OkStatus();
  }

  ASSIGN_OR_RETURN(GpuExecutableProto gpu_executable_proto, ToProto());
  ExecutableAndOptionsProto dump_proto;
  RETURN_IF_ERROR(
      WriteSplitGpuExecutable(std::move(gpu_executable_proto),
                              std::make_unique<riegeli::StringWriter<>>(
                                  dump_proto.mutable_serialized_executable())));
  ASSIGN_OR_RETURN(
      *dump_proto.mutable_compile_options()->mutable_executable_build_options(),
      CreateSerializableBuildOptionsProto(options));

  constexpr absl::string_view kDumpFilename = "gpu_executable.riegeli";
  ASSIGN_OR_RETURN(std::unique_ptr<riegeli::Writer> writer,
                   CreateRiegeliDumpWriter(debug_options, kDumpFilename,
                                           has_module() ? &module() : nullptr));
  RETURN_IF_ERROR(
      WriteSplitExecutableAndOptions(dump_proto, std::move(writer)));

  return absl::OkStatus();
}

}  // namespace xla::gpu
