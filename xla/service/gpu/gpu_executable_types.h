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

#ifndef XLA_SERVICE_GPU_GPU_EXECUTABLE_TYPES_H_
#define XLA_SERVICE_GPU_GPU_EXECUTABLE_TYPES_H_

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/device_address.h"

namespace xla::gpu {

struct GpuExecutableNumAdditionalStreams {
  int compute = 0;
  int communication = 0;
};

struct GpuExecutableConstantInfo {
  std::string symbol_name;
  DenseDataIntermediate content;
  int allocation_index = -1;

  GpuExecutableProto::ConstantInfoProto ToProto() const;

  static absl::StatusOr<GpuExecutableConstantInfo> FromProto(
      const GpuExecutableProto::ConstantInfoProto& proto,
      const absl::flat_hash_map<std::string, const HloInstruction*>*
          absl_nullable content_overrides = nullptr);
};

struct GpuExecutableOutputInfo {
  // Corresponding allocation index.
  int allocation_index;

  // Output is passed-through from a parameter.
  bool passthrough = false;

  // Whether this output is hinted to alias a parameter (BufferAllocation* would
  // indicate the aliased parameter), and what kind of alias it is.
  std::optional<HloInputOutputAliasConfig::Alias> alias_config;

  GpuExecutableProto::OutputInfoProto ToProto() const;
  static absl::StatusOr<GpuExecutableOutputInfo> FromProto(
      const GpuExecutableProto::OutputInfoProto& proto);

  friend bool operator==(const GpuExecutableOutputInfo& lhs,
                         const GpuExecutableOutputInfo& rhs) {
    return std::tie(lhs.allocation_index, lhs.passthrough, lhs.alias_config) ==
           std::tie(rhs.allocation_index, rhs.passthrough, rhs.alias_config);
  }

  friend bool operator!=(const GpuExecutableOutputInfo& lhs,
                         const GpuExecutableOutputInfo& rhs) {
    return !(lhs == rhs);
  }
};

struct GpuExecutableParameterBuffer {
  se::DeviceAddressBase buffer;
  int64_t parameter_number;
};

using GpuExecutableBufferAllocToDeviceMemoryMap =
    absl::flat_hash_map<BufferAllocation::Index, se::DeviceAddressBase>;

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_TYPES_H_
