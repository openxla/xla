/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/pjrt/utils.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_hlo_module_metadata.h"
#include "xla/pjrt/pjrt_hlo_module_metadata.pb.h"
#include "xla/primitive_util.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/statusor.h"

namespace xla {

namespace {
StatusOr<std::pair<std::vector<Shape>, Shape>> GetShardedProgramShapes(
    const XlaComputation& computation, const ProgramShape& program_shape) {
  return GetShardedProgramShapesHelper(
      computation.proto().computations(),
      computation.proto().entry_computation_id(), program_shape);
}

std::pair<std::vector<Shape>, Shape> CreateShardedProgramShape(
    const PjrtHloModuleMetadataProto& metadata) {
  std::vector<Shape> argument_shapes;
  for (const ShapeProto& shape :
       metadata.sharded_program_shape().argument_shapes()) {
    argument_shapes.push_back(Shape(shape));
  }
  return std::make_pair(argument_shapes,
                        Shape(metadata.sharded_program_shape().result_shape()));
}

Status PrepareArgumentLayout(
    ProgramShape program_shape,
    std::optional<std::vector<Shape>>& argument_layouts) {
  if (!argument_layouts) {
    argument_layouts.emplace(program_shape.parameters());
    for (Shape& shape : *argument_layouts) {
      LayoutUtil::ClearLayout(&shape);
    }
  } else if (argument_layouts->size() != program_shape.parameters_size()) {
    return InvalidArgument(
        "CompileOptions specify %d argument layouts, but computation has %d "
        "arguments",
        argument_layouts->size(), program_shape.parameters_size());
  }
  return OkStatus();
}

Status DetermineArgumentLayoutsFromCompileOptions(
    ProgramShape program_shape,
    StatusOr<std::pair<std::vector<Shape>, Shape>> sharded_program_shapes,
    std::function<StatusOr<Shape>(Shape)>
        choose_compact_layout_for_shape_function,
    std::optional<std::vector<Shape>>& argument_layouts,
    ExecutableBuildOptions* build_options,
    std::vector<const Shape*>* argument_layout_pointers) {
  argument_layout_pointers->reserve(argument_layouts->size());

  // Assign a default layout based on `sharded_shape` to any array subshapes in
  // `dst_shape` that are missing layouts.
  auto assign_layouts = [&choose_compact_layout_for_shape_function](
                            const Shape& sharded_shape, Shape* dst_shape) {
    return ShapeUtil::ForEachMutableSubshapeWithStatus(
        dst_shape, [&](Shape* subshape, const ShapeIndex& idx) {
          if (subshape->IsArray() && !subshape->has_layout()) {
            CHECK(ShapeUtil::IndexIsValid(sharded_shape, idx));
            const Shape& sharded_subshape =
                ShapeUtil::GetSubshape(sharded_shape, idx);
            LayoutUtil::SetToDefaultLayout(subshape);
            TF_ASSIGN_OR_RETURN(
                Shape layout,
                choose_compact_layout_for_shape_function(sharded_subshape));
            *subshape->mutable_layout() = layout.layout();
          }
          return OkStatus();
        });
  };
  TF_ASSIGN_OR_RETURN(auto sharded_shapes, sharded_program_shapes);

  CHECK_EQ(sharded_shapes.first.size(), argument_layouts->size());
  for (int i = 0; i < argument_layouts->size(); ++i) {
    Shape* layout = &(*argument_layouts)[i];
    argument_layout_pointers->push_back(layout);
    TF_RETURN_IF_ERROR(assign_layouts(sharded_shapes.first[i], layout));
  }

  Shape result_layout;
  if (build_options->result_layout()) {
    result_layout = *build_options->result_layout();
  } else {
    result_layout = program_shape.result();
    LayoutUtil::ClearLayout(&result_layout);
  }
  TF_RETURN_IF_ERROR(assign_layouts(sharded_shapes.second, &result_layout));
  build_options->set_result_layout(result_layout);
  return OkStatus();
}
int DetermineNumberOfParameters(
    const PjrtHloModuleMetadata& hlo_module_metadata, bool tuple_inputs) {
  if (tuple_inputs) {
    CHECK_EQ(hlo_module_metadata.num_parameters(), 1);
    const Shape& input_tuple_shape =
        hlo_module_metadata.hlo_first_parameter_instruction_shape();
    CHECK(input_tuple_shape.IsTuple());
    return input_tuple_shape.tuple_shapes_size();
  } else {
    return hlo_module_metadata.num_parameters();
  }
}

int DetermineNumberOfParameters(HloComputation* computation,
                                bool tuple_inputs) {
  if (tuple_inputs) {
    CHECK_EQ(computation->num_parameters(), 1);
    const Shape& input_tuple_shape =
        computation->parameter_instruction(0)->shape();
    CHECK(input_tuple_shape.IsTuple());
    return input_tuple_shape.tuple_shapes_size();
  } else {
    return computation->num_parameters();
  }
}

StatusOr<std::vector<int>> ComputeParametersThatMustBeDonated(
    int num_parameters, int determined_number_of_parameters,
    const HloInputOutputAliasConfig& config, bool tuple_inputs) {
  // If any buffer in a parameter is aliased we will donate the entire input
  // parameter.
  std::vector<int> parameters_to_donate;
  parameters_to_donate.reserve(num_parameters);
  TF_RETURN_IF_ERROR(config.ForEachAliasWithStatus(
      [&](const ShapeIndex& output_index,
          const HloInputOutputAliasConfig::Alias& alias) {
        if (tuple_inputs) {
          if (alias.parameter_number != 0) {
            return InvalidArgument(
                "Unexpected parameter number %d in alias config with tupled "
                "inputs",
                alias.parameter_number);
          }
          const ShapeIndex& index = alias.parameter_index;
          if (!index.empty()) {
            int this_parameter = index.data()[0];
            if (this_parameter >= determined_number_of_parameters) {
              return InvalidArgument(
                  "Unexpected parameter index %s in alias config with tupled "
                  "inputs and %d parameters",
                  index.ToString(), determined_number_of_parameters);
            }
            parameters_to_donate.push_back(this_parameter);
          }
        } else {
          int this_parameter = alias.parameter_number;
          if (this_parameter >= determined_number_of_parameters) {
            return InvalidArgument(
                "Unexpected parameter number %d in alias config without tupled "
                "inputs and %d parameters",
                this_parameter, determined_number_of_parameters);
          }
          parameters_to_donate.push_back(this_parameter);
        }
        return OkStatus();
      }));
  absl::c_sort(parameters_to_donate);
  return parameters_to_donate;
}

}  // namespace

Status ParseDeviceAssignmentCompileOptions(
    bool compile_portable_executable, ExecutableBuildOptions* build_options,
    std::function<StatusOr<DeviceAssignment>(int, int)>
        GetDefaultDeviceAssignmentFunction,
    int* num_replicas, int* num_partitions,
    std::shared_ptr<DeviceAssignment>* device_assignment) {
  if (compile_portable_executable) {
    if (build_options->has_device_assignment()) {
      return InvalidArgument(
          "CompileOptions requests portable executable but "
          "ExecutableBuildOptions includes a device assignment");
    }
    if (build_options->num_replicas() != 1 ||
        build_options->num_partitions() != 1) {
      return InvalidArgument(
          "CompileOptions requests portable executable but "
          "ExecutableBuildOptions includes num_replicas %d  and num_partitions "
          "%d.",
          build_options->num_replicas(), build_options->num_partitions());
    }
    *num_replicas = 1;
    *num_partitions = 1;
  } else {
    if (!build_options->has_device_assignment()) {
      VLOG(2) << "Compile using default device_assignment.";
      TF_ASSIGN_OR_RETURN(
          DeviceAssignment device_assignment,
          GetDefaultDeviceAssignmentFunction(build_options->num_replicas(),
                                             build_options->num_partitions()));
      build_options->set_device_assignment(device_assignment);
    }
    VLOG(2) << "Compile device_assignment:\n"
            << build_options->device_assignment().ToString();
    *num_replicas = build_options->device_assignment().replica_count();
    *num_partitions = build_options->device_assignment().computation_count();
    *device_assignment =
        std::make_shared<DeviceAssignment>(build_options->device_assignment());
  }
  return OkStatus();
}

Status DetermineArgumentLayoutsFromCompileOptions(
    const XlaComputation& computation,
    std::function<StatusOr<Shape>(Shape)>
        choose_compact_layout_for_shape_function,
    std::optional<std::vector<Shape>>& argument_layouts,
    ExecutableBuildOptions* build_options,
    std::vector<const Shape*>* argument_layout_pointers) {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());
  TF_RETURN_IF_ERROR(PrepareArgumentLayout(program_shape, argument_layouts));
  return DetermineArgumentLayoutsFromCompileOptions(
      program_shape, GetShardedProgramShapes(computation, program_shape),
      choose_compact_layout_for_shape_function, argument_layouts, build_options,
      argument_layout_pointers);
}

Status DetermineArgumentLayoutsFromCompileOptions(
    const PjrtHloModuleMetadataProto& metadata,
    std::function<StatusOr<Shape>(Shape)>
        choose_compact_layout_for_shape_function,
    std::optional<std::vector<Shape>>& argument_layouts,
    ExecutableBuildOptions* build_options,
    std::vector<const Shape*>* argument_layout_pointers) {
  TF_RET_CHECK(metadata.has_host_program_shape());
  const ProgramShape& program_shape =
      ProgramShape(metadata.host_program_shape());
  TF_RETURN_IF_ERROR(PrepareArgumentLayout(program_shape, argument_layouts));
  return DetermineArgumentLayoutsFromCompileOptions(
      program_shape, CreateShardedProgramShape(metadata),
      choose_compact_layout_for_shape_function, argument_layouts, build_options,
      argument_layout_pointers);
}

StatusOr<std::vector<int>> ComputeParametersThatMustBeDonated(
    const PjrtHloModuleMetadata& hlo_module_metadata, bool tuple_inputs) {
  int determined_number_of_parameters =
      DetermineNumberOfParameters(hlo_module_metadata, tuple_inputs);
  return ComputeParametersThatMustBeDonated(
      hlo_module_metadata.num_parameters(), determined_number_of_parameters,
      hlo_module_metadata.hlo_input_output_alias_config(), tuple_inputs);
}

StatusOr<std::vector<int>> ComputeParametersThatMustBeDonated(
    const HloModule& module, bool tuple_inputs) {
  HloComputation* computation = module.entry_computation();
  int determined_number_of_parameters =
      DetermineNumberOfParameters(computation, tuple_inputs);
  return ComputeParametersThatMustBeDonated(
      computation->num_parameters(), determined_number_of_parameters,
      module.input_output_alias_config(), tuple_inputs);
}

int DefaultThreadPoolSize() {
  // Google's CI system exposes an environment variable NPROC that describes
  // a CPU reservation for tests.
  // TODO(phawkins): expose a better thought-out set of knobs to control
  // parallelism.
  for (const char* nproc_env : {"PJRT_NPROC", "NPROC"}) {
    const char* nproc_str = std::getenv(nproc_env);
    int nproc = 0;
    if (nproc_str && absl::SimpleAtoi(nproc_str, &nproc)) {
      return std::max(0, nproc);
    }
  }
  return tsl::port::MaxParallelism();
}

bool HasMajorToMinorLayout(PrimitiveType type, absl::Span<int64_t const> dims,
                           absl::Span<int64_t const> byte_strides) {
  CHECK_EQ(dims.size(), byte_strides.size());
  // If the array is size 0, the strides are irrelevant.
  if (absl::c_find(dims, 0) != dims.end()) {
    return true;
  }
  int64_t stride = primitive_util::ByteWidth(type);
  for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
    // If a dimension is of size 1, its stride is irrelevant.
    if (dims[i] != 1) {
      if (byte_strides[i] != stride) {
        return false;
      }
      stride *= dims[i];
    }
  }
  return true;
}

StatusOr<Shape> MakeShapeWithTrivialByteStrides(
    PrimitiveType element_type, absl::Span<const int64_t> dimensions,
    absl::Span<const int64_t> byte_strides) {
  TF_RET_CHECK(dimensions.size() == byte_strides.size());
  std::vector<int64_t> minor_to_major(dimensions.size());
  // Begin with a major-to-minor layout that is likey the most common.
  std::iota(minor_to_major.rbegin(), minor_to_major.rend(), 0);
  // Find minor-to-major only if there is no zero dimension size because
  // minor-to-major is irrelevant with any zero dimension size.
  if (absl::c_find(dimensions, 0) == dimensions.end()) {
    absl::c_sort(minor_to_major, [&](int a, int b) {
      if (byte_strides[a] < byte_strides[b]) {
        return true;
      }
      if (byte_strides[a] > byte_strides[b]) {
        return false;
      }
      return dimensions[a] == 1 && dimensions[b] != 1;
    });
    int64_t byte_stride = ShapeUtil::ByteSizeOfPrimitiveType(element_type);
    for (int64_t d : minor_to_major) {
      if (dimensions[d] != 1 && byte_strides[d] != byte_stride) {
        return Unimplemented(
            "Only trivial (compact) byte strides are supported; i.e., byte "
            "striding represents a transposition of the underlying dense "
            "buffer but not broadcasting. Dimensions were: [%s], byte strides "
            "were [%s].",
            absl::StrJoin(dimensions, ","), absl::StrJoin(byte_strides, ","));
      }
      byte_stride *= dimensions[d];
    }
  }
  return ShapeUtil::MakeShapeWithDenseLayout(element_type, dimensions,
                                             minor_to_major);
}

Status TestBufferDonationClashes(
    void* opaque_key,
    absl::flat_hash_map<const void*, std::pair<bool, int>>& donation_clashes,
    bool is_donated, int arg_idx, int replica, int partition) {
  auto [donation_clash_it, first_use] =
      donation_clashes.emplace(opaque_key, std::make_pair(is_donated, arg_idx));
  if (!first_use && (is_donated || donation_clash_it->second.first)) {
    auto [prev_is_donated, prev_arg_idx] = donation_clash_it->second;
    if (is_donated && prev_is_donated) {
      return InvalidArgument(
          "Attempt to donate the same buffer twice in Execute() ("
          "flattened argument %d, replica %d, partition %d, first use: %d). "
          "Toy "
          "example for this bug: `f(donate(a), donate(a))`.",
          arg_idx, replica, partition, prev_arg_idx);
    } else if (is_donated) {
      return InvalidArgument(
          "Attempt to donate a buffer which is also used by the same call "
          "to Execute() (flattened argument %d, replica %d, partition %d, "
          "first use: %d). Toy example for this bug: `f(a, donate(a))`.",
          arg_idx, replica, partition, prev_arg_idx);
    } else {
      return InvalidArgument(
          "Attempt to use a buffer that was previously donated in the same "
          "call to Execute() (flattened argument %d, replica %d, partition "
          "%d, first use: %d). Toy example for this bug: `f(donate(a), "
          "a)`.",
          arg_idx, replica, partition, prev_arg_idx);
    }
  }
  return absl::OkStatus();
}

}  // namespace xla
