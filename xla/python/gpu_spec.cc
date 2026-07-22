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

#include <string>

#include "absl/strings/str_cat.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/pjrt/exceptions.h"
#include "xla/stream_executor/device_description.pb.h"

namespace xla {

namespace nb = nanobind;

namespace {

class GpuTargetConfigWrapper {
 public:
  explicit GpuTargetConfigWrapper(stream_executor::GpuTargetConfigProto proto)
      : proto_(std::move(proto)), info_(proto_.gpu_device_info()) {}

  const std::string& platform_name() const { return proto_.platform_name(); }
  const std::string& device_description_str() const {
    return proto_.device_description_str();
  }

  std::string arch_name() const {
    if (info_.has_cuda_compute_capability()) {
      const auto& cc = info_.cuda_compute_capability();
      return absl::StrCat(cc.major(), ".", cc.minor());
    }
    if (info_.has_rocm_compute_capability()) {
      return std::string(info_.rocm_compute_capability().gcn_arch_name());
    }
    return "";
  }

  int64_t compute_capability() const {
    if (info_.has_cuda_compute_capability()) {
      const auto& cc = info_.cuda_compute_capability();
      return cc.major() * 10 + cc.minor();
    }
    return 0;
  }

  int64_t core_count() const {
    return info_.core_count();
  }

  int64_t shared_memory_per_core() const {
    return info_.shared_memory_per_core();
  }

 private:
  stream_executor::GpuTargetConfigProto proto_;
  const stream_executor::GpuDeviceInfoProto& info_;
};

}  // namespace

NB_MODULE(_gpu_spec, m) {
  nb::class_<GpuTargetConfigWrapper> gpu_target_config_class(
      m, "GpuTargetConfig");
  gpu_target_config_class
      .def_prop_ro("platform_name", &GpuTargetConfigWrapper::platform_name)
      .def_prop_ro("device_description_str",
                   &GpuTargetConfigWrapper::device_description_str)
      .def_prop_ro("arch_name", &GpuTargetConfigWrapper::arch_name)
      .def_prop_ro("compute_capability",
                   &GpuTargetConfigWrapper::compute_capability)
      .def_prop_ro("core_count",
                   &GpuTargetConfigWrapper::core_count)
      .def_prop_ro("shared_memory_per_core",
                   &GpuTargetConfigWrapper::shared_memory_per_core);

  m.def(
      "get_gpu_spec",
      [](const std::string& device_kind) -> GpuTargetConfigWrapper {
        auto maybe_proto =
            gpu::GetGpuTargetConfigFromDeviceKind(device_kind);
        if (!maybe_proto.ok()) {
          throw xla::XlaRuntimeError(maybe_proto.status().message().data());
        }
        return GpuTargetConfigWrapper(std::move(*maybe_proto));
      },
      nb::arg("device_kind"));
}

}  // namespace xla
