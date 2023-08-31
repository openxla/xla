/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/pjrt/pjrt_hlo_module_metadata.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "third_party/protobuf/repeated_ptr_field.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_hlo_module_metadata.pb.h"
#include "xla/service/computation_layout.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/name_uniquer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace {

// Precomputes sharded program shapes and saves in PjrtHloModuleMetadataProto.
Status SetShardedProgramShape(
    const ProgramShape& host_program_shape, int entry_computation_id,
    const ::proto2::RepeatedPtrField<::xla::HloComputationProto>& computations,
    PjrtHloModuleMetadataProto* metadata_proto_out) {
  // Pre-compute sharded program shapes
  auto result = GetShardedProgramShapesHelper(
      computations, entry_computation_id, host_program_shape);
  if (!result.ok()) {
    return result.status();
  }
  auto pair = result.value();
  ShardedProgramShapeProto sharded_program_shape;
  for (const Shape& shape : pair.first) {
    *sharded_program_shape.add_argument_shapes() = shape.ToProto();
  }
  *(sharded_program_shape.mutable_result_shape()) = pair.second.ToProto();
  *(metadata_proto_out->mutable_sharded_program_shape()) =
      sharded_program_shape;
  return OkStatus();
}

StatusOr<Shape> GetShardedShape(const Shape& shape,
                                const OpSharding& sharding) {
  TF_ASSIGN_OR_RETURN(HloSharding hlo_sharding,
                      HloSharding::FromProto(sharding));
  if (shape.IsTuple()) {
    Shape sharded_shape = shape;
    ShapeUtil::ForEachMutableSubshape(
        &sharded_shape, [&](Shape* subshape, const ShapeIndex& index) {
          if (!subshape->IsTuple()) {
            HloSharding subsharding = hlo_sharding.GetSubSharding(shape, index);
            *subshape = subsharding.TileShape(*subshape);
          }
        });
    return sharded_shape;
  } else {
    return hlo_sharding.TileShape(shape);
  }
}

StatusOr<Shape> GetShardedShape(const HloInstructionProto& instr) {
  const Shape unsharded_shape(instr.shape());
  Shape sharded_shape;
  if (instr.has_sharding()) {
    TF_ASSIGN_OR_RETURN(sharded_shape,
                        GetShardedShape(unsharded_shape, instr.sharding()));
  } else {
    sharded_shape = unsharded_shape;
  }
  LayoutUtil::ClearLayout(&sharded_shape);
  return sharded_shape;
}
}  // namespace

StatusOr<std::pair<std::vector<Shape>, Shape>> GetShardedProgramShapesHelper(
    const ::proto2::RepeatedPtrField<::xla::HloComputationProto>& computations,
    int entry_computation_id, const ProgramShape& program_shape) {
  std::vector<Shape> arg_shapes;
  arg_shapes.resize(program_shape.parameters_size());
  Shape result_shape;
  for (const HloComputationProto& comp : computations) {
    if (comp.id() != entry_computation_id) {
      continue;
    }
    for (const HloInstructionProto& instr : comp.instructions()) {
      if (instr.opcode() == HloOpcodeString(HloOpcode::kParameter)) {
        if (instr.parameter_number() >= program_shape.parameters_size()) {
          return InvalidArgument(
              "Got invalid parameter number %d, expected %d parameters",
              instr.parameter_number(), program_shape.parameters_size());
        }
        TF_ASSIGN_OR_RETURN(arg_shapes[instr.parameter_number()],
                            GetShardedShape(instr));
      }
      if (instr.id() == comp.root_id()) {
        if (result_shape.element_type() != PRIMITIVE_TYPE_INVALID) {
          return InvalidArgument("Found multiple root instructions");
        }
        TF_ASSIGN_OR_RETURN(result_shape, GetShardedShape(instr));
      }
    }
  }
  for (int i = 0; i < arg_shapes.size(); ++i) {
    if (arg_shapes[i].element_type() == PRIMITIVE_TYPE_INVALID) {
      return InvalidArgument("Couldn't find parameter %d", i);
    }
  }
  if (result_shape.element_type() == PRIMITIVE_TYPE_INVALID) {
    return InvalidArgument("Couldn't find root instruction");
  }
  return std::make_pair(arg_shapes, result_shape);
}

StatusOr<std::unique_ptr<PjrtHloModuleMetadata>>
PjrtHloModuleMetadata::CreateFromProto(
    const PjrtHloModuleMetadataProto& proto) {
  std::shared_ptr<Shape> hlo_first_parameter_instruction_shape;

  TF_RET_CHECK(proto.has_host_program_shape());
  auto entry_computation_layout =
      ComputationLayout(ProgramShape(proto.host_program_shape()),
                        /*ignore_layouts=*/false);

  TF_ASSIGN_OR_RETURN(auto input_output_alias_config,
                      HloInputOutputAliasConfig::CreateFromProto(
                          xla::Shape(entry_computation_layout.result_shape()),
                          proto.input_output_alias_config()));

  if (proto.has_hlo_first_parameter_instruction_shape()) {
    hlo_first_parameter_instruction_shape =
        std::make_unique<Shape>(proto.hlo_first_parameter_instruction_shape());
  }
  Shape hlo_first_root_instruction_shape =
      Shape(proto.hlo_first_root_instruction_shape());

  return std::make_unique<PjrtHloModuleMetadata>(
      NameUniquer::GetSanitizedName(proto.name()), proto,
      std::move(entry_computation_layout),
      hlo_first_parameter_instruction_shape,
      std::move(hlo_first_root_instruction_shape),
      std::move(input_output_alias_config));
}

StatusOr<std::unique_ptr<PjrtHloModuleMetadata>>
PjrtHloModuleMetadata::CreateFromHloModule(const HloModule& hlo_module) {
  PjrtHloModuleMetadataProto hlo_module_metadata;
  hlo_module_metadata.set_replica_count(hlo_module.config().replica_count());
  hlo_module_metadata.set_num_partitions(hlo_module.config().num_partitions());

  if (!hlo_module.has_entry_computation()) {
    return InvalidArgument("HloModule has no entry computation");
  }
  HloComputation* hlo_computation = hlo_module.entry_computation();
  ProgramShape host_program_shape =
      hlo_module.entry_computation_layout().ComputeProgramShape();
  int entry_computation_id = hlo_module.entry_computation()->unique_id();

  *(hlo_module_metadata.mutable_hlo_first_root_instruction_shape()) =
      hlo_module.entry_computation()->root_instruction()->shape().ToProto();

  hlo_module_metadata.set_num_parameters(hlo_computation->num_parameters());
  if (hlo_computation->num_parameters() > 0) {
    *(hlo_module_metadata.mutable_hlo_first_parameter_instruction_shape()) =
        hlo_computation->parameter_instruction(0)->shape().ToProto();
  }

  hlo_module_metadata.set_name(hlo_module.name());
  hlo_module_metadata.set_entry_computation_id(entry_computation_id);
  *hlo_module_metadata.mutable_input_output_alias_config() =
      hlo_module.input_output_alias_config().ToProto();
  *(hlo_module_metadata.mutable_host_program_shape()) =
      host_program_shape.ToProto();

  ::proto2::RepeatedPtrField<::xla::HloComputationProto> computations;
  for (const HloComputation* computation :
       hlo_module.MakeComputationPostOrder()) {
    HloComputationProto computation_proto = computation->ToProto();
    *computations.Add() = computation_proto;
  }
  TF_RETURN_IF_ERROR(SetShardedProgramShape(host_program_shape,
                                            entry_computation_id, computations,
                                            &hlo_module_metadata));

  return CreateFromProto(hlo_module_metadata);
}

StatusOr<PjrtHloModuleMetadataProto> CreatePjrtHloModuleMetadataProto(
    const HloModule& hlo_module) {
  PjrtHloModuleMetadataProto hlo_module_metadata;
  hlo_module_metadata.set_replica_count(hlo_module.config().replica_count());
  hlo_module_metadata.set_num_partitions(hlo_module.config().num_partitions());

  if (!hlo_module.has_entry_computation()) {
    return InvalidArgument("HloModule has no entry computation");
  }
  HloComputation* hlo_computation = hlo_module.entry_computation();
  ProgramShape host_program_shape =
      hlo_module.entry_computation_layout().ComputeProgramShape();
  int entry_computation_id = hlo_module.entry_computation()->unique_id();

  *(hlo_module_metadata.mutable_hlo_first_root_instruction_shape()) =
      hlo_module.entry_computation()->root_instruction()->shape().ToProto();

  hlo_module_metadata.set_num_parameters(hlo_computation->num_parameters());
  if (hlo_computation->num_parameters() > 0) {
    *(hlo_module_metadata.mutable_hlo_first_parameter_instruction_shape()) =
        hlo_computation->parameter_instruction(0)->shape().ToProto();
  }

  hlo_module_metadata.set_name(hlo_module.name());
  hlo_module_metadata.set_entry_computation_id(entry_computation_id);
  *hlo_module_metadata.mutable_input_output_alias_config() =
      hlo_module.input_output_alias_config().ToProto();
  *(hlo_module_metadata.mutable_host_program_shape()) =
      host_program_shape.ToProto();

  ::proto2::RepeatedPtrField<::xla::HloComputationProto> computations;
  for (const HloComputation* computation :
       hlo_module.MakeComputationPostOrder()) {
    HloComputationProto computation_proto = computation->ToProto();
    *computations.Add() = computation_proto;
  }
  TF_RETURN_IF_ERROR(SetShardedProgramShape(host_program_shape,
                                            entry_computation_id, computations,
                                            &hlo_module_metadata));

  return hlo_module_metadata;
}

const ComputationLayout& PjrtHloModuleMetadata::entry_computation_layout()
    const {
  return entry_computation_layout_;
}

absl::string_view PjrtHloModuleMetadata::name() const { return name_; }
int PjrtHloModuleMetadata::launch_id() const { return proto_.launch_id(); }
int PjrtHloModuleMetadata::num_replicas() const {
  return proto_.replica_count();
}
int PjrtHloModuleMetadata::num_partitions() const {
  return proto_.num_partitions();
}
int PjrtHloModuleMetadata::num_parameters() const {
  return proto_.num_parameters();
}

const Shape& PjrtHloModuleMetadata::hlo_first_parameter_instruction_shape()
    const {
  CHECK(hlo_first_parameter_instruction_shape_ != nullptr);
  return *hlo_first_parameter_instruction_shape_;
}

const Shape& PjrtHloModuleMetadata::result_shape() const {
  return hlo_first_root_instruction_shape_;
}

const HloInputOutputAliasConfig&
PjrtHloModuleMetadata::hlo_input_output_alias_config() const {
  return input_output_alias_config_;
}

const PjrtHloModuleMetadataProto& PjrtHloModuleMetadata::proto() const {
  return proto_;
}

}  // namespace xla
