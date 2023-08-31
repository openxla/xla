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
#ifndef XLA_PJRT_PJRT_HLO_MODULE_METADATA_H_
#define XLA_PJRT_PJRT_HLO_MODULE_METADATA_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_hlo_module_metadata.pb.h"
#include "xla/service/computation_layout.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/statusor.h"

namespace xla {
// This class manages essential hlo module information that will
// be consumed by PJRT client for serialization. PjrtHloModuleMetadataProto is
// used as an alternative of HloModuleProto in serialization.
// PjrtHloModuleMetadata stores objects that needs to be created from
// PjrtHloModuleMetadataProto and shared with Pjrt client.
class PjrtHloModuleMetadata {
 public:
  // Creates a PjrtHloModuleMetadata from a PjrtHloModuleMetadataProto.
  static StatusOr<std::unique_ptr<PjrtHloModuleMetadata>> CreateFromProto(
      const PjrtHloModuleMetadataProto& proto);

  // Creates a PjrtHloModuleMetadata from from a HloModule.
  static StatusOr<std::unique_ptr<PjrtHloModuleMetadata>> CreateFromHloModule(
      const HloModule& hlo_module);

  explicit PjrtHloModuleMetadata(
      absl::string_view name, const PjrtHloModuleMetadataProto& proto,
      const ComputationLayout& entry_computation_layout,
      std::shared_ptr<Shape> hlo_first_parameter_instruction_shape,
      Shape hlo_first_root_instruction_shape,
      const HloInputOutputAliasConfig& input_output_alias_config)
      : name_(name),
        proto_(std::move(proto)),
        entry_computation_layout_(std::move(entry_computation_layout)),
        hlo_first_parameter_instruction_shape_(
            hlo_first_parameter_instruction_shape),
        hlo_first_root_instruction_shape_(
            std::move(hlo_first_root_instruction_shape)),
        input_output_alias_config_(std::move(input_output_alias_config)) {}

  const PjrtHloModuleMetadataProto& proto() const;
  const ComputationLayout& entry_computation_layout() const;
  absl::string_view name() const;
  int num_replicas() const;
  int num_partitions() const;
  int num_parameters() const;
  int launch_id() const;
  const Shape& hlo_first_parameter_instruction_shape() const;
  const Shape& result_shape() const;
  const HloInputOutputAliasConfig& hlo_input_output_alias_config() const;

 private:
  std::string name_;
  PjrtHloModuleMetadataProto proto_;
  ComputationLayout entry_computation_layout_;
  std::shared_ptr<Shape> hlo_first_parameter_instruction_shape_;
  Shape hlo_first_root_instruction_shape_;
  HloInputOutputAliasConfig input_output_alias_config_;
};

// Creates a PjrtHloModuleMetadataProto from a HloModule.
StatusOr<PjrtHloModuleMetadataProto> CreatePjrtHloModuleMetadataProto(
    const HloModule& hlo_module);

// Computes sharded (argument shapes, result shape) without layouts.
StatusOr<std::pair<std::vector<Shape>, Shape>> GetShardedProgramShapesHelper(
    const ::proto2::RepeatedPtrField<::xla::HloComputationProto>& computations,
    int entry_computation_id, const ProgramShape& program_shape);

}  // namespace xla

#endif  // XLA_PJRT_PJRT_HLO_MODULE_METADATA_H_
