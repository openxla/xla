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

#include "xla/stream_executor/sycl/oneapi_compute_capability.h"

#include <cstdint>
#include <string>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.pb.h"

namespace stream_executor {

XeArch OneAPIComputeCapability::InferDeviceArch(XeGeneration generation,
                                                uint8_t version) {
  for (const OneAPIArchInfo& info : kXeArchs) {
    if (info.generation == generation && version >= info.min_version &&
        version <= info.max_version) {
      return info.arch;
    }
  }
  // Unknown or not-yet-supported generation.
  return XeArch::kUnknown;
}

OneAPIComputeCapability::OneAPIComputeCapability(XeGeneration generation,
                                                 uint8_t version)
    : generation_(generation), arch_(InferDeviceArch(generation, version)) {}

OneAPIComputeCapability::OneAPIComputeCapability(absl::string_view arch) {
  for (const OneAPIArchInfo& info : kXeArchs) {
    if (absl::EqualsIgnoreCase(info.name, arch)) {
      generation_ = info.generation;
      arch_ = info.arch;
      return;
    }
  }
}

std::string OneAPIComputeCapability::architecture() const {
  for (const OneAPIArchInfo& info : kXeArchs) {
    if (arch_ == info.arch) {
      return info.name;
    }
  }
  return "Unknown";
}

OneAPIComputeCapability OneAPIComputeCapability::FromProto(
    const OneAPIComputeCapabilityProto& proto) {
  return OneAPIComputeCapability{proto.architecture()};
}

OneAPIComputeCapabilityProto OneAPIComputeCapability::ToProto() const {
  OneAPIComputeCapabilityProto proto;
  proto.set_architecture(architecture());
  return proto;
}

std::string OneAPIComputeCapability::ToString() const { return architecture(); }

}  // namespace stream_executor
