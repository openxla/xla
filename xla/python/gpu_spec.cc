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

class GpuDeviceInfoWrapper {
 public:
  explicit GpuDeviceInfoWrapper(stream_executor::GpuDeviceInfoProto proto)
      : proto_(std::move(proto)) {}

  const std::string& device_vendor() const { return proto_.device_vendor(); }
  const std::string& platform_version() const {
    return proto_.platform_version();
  }
  const std::string& pci_bus_id() const { return proto_.pci_bus_id(); }
  const std::string& name() const { return proto_.name(); }
  const std::string& model_str() const { return proto_.model_str(); }
  int32_t threads_per_block_limit() const {
    return proto_.threads_per_block_limit();
  }
  int32_t threads_per_warp() const { return proto_.threads_per_warp(); }
  int32_t shared_memory_per_block() const {
    return proto_.shared_memory_per_block();
  }
  int32_t shared_memory_per_core() const {
    return proto_.shared_memory_per_core();
  }
  int32_t threads_per_core_limit() const {
    return proto_.threads_per_core_limit();
  }
  int32_t core_count() const { return proto_.core_count(); }
  int64_t fpus_per_core() const { return proto_.fpus_per_core(); }
  int64_t block_dim_limit_x() const { return proto_.block_dim_limit_x(); }
  int64_t block_dim_limit_y() const { return proto_.block_dim_limit_y(); }
  int64_t block_dim_limit_z() const { return proto_.block_dim_limit_z(); }
  int64_t memory_bandwidth() const { return proto_.memory_bandwidth(); }
  int64_t l2_cache_size() const { return proto_.l2_cache_size(); }
  float clock_rate_ghz() const { return proto_.clock_rate_ghz(); }
  int64_t device_memory_size() const { return proto_.device_memory_size(); }
  int32_t shared_memory_per_block_optin() const {
    return proto_.shared_memory_per_block_optin();
  }
  int64_t registers_per_core_limit() const {
    return proto_.registers_per_core_limit();
  }
  int64_t registers_per_block_limit() const {
    return proto_.registers_per_block_limit();
  }
  const std::string& driver_version() const { return proto_.driver_version(); }
  const std::string& kernel_mode_driver_version() const {
    return proto_.kernel_mode_driver_version();
  }
  const std::string& runtime_version() const {
    return proto_.runtime_version();
  }
  const std::string& compile_time_toolkit_version() const {
    return proto_.compile_time_toolkit_version();
  }
  const std::string& dnn_version() const { return proto_.dnn_version(); }
  const std::string& cub_version() const { return proto_.cub_version(); }
  int32_t numa_node() const { return proto_.numa_node(); }
  int64_t thread_dim_limit_x() const { return proto_.thread_dim_limit_x(); }
  int64_t thread_dim_limit_y() const { return proto_.thread_dim_limit_y(); }
  int64_t thread_dim_limit_z() const { return proto_.thread_dim_limit_z(); }
  int64_t device_address_bits() const { return proto_.device_address_bits(); }
  int64_t pcie_bandwidth() const { return proto_.pcie_bandwidth(); }
  bool ecc_enabled() const { return proto_.ecc_enabled(); }
  float mem_clock_ghz() const { return proto_.mem_clock_ghz(); }
  int64_t reserved_shared_memory_per_block() const {
    return proto_.reserved_shared_memory_per_block();
  }
  int64_t max_blocks_per_multiprocessor() const {
    return proto_.max_blocks_per_multiprocessor();
  }
  int64_t collective_memory_granularity() const {
    return proto_.collective_memory_granularity();
  }

  const stream_executor::GpuDeviceInfoProto& proto() const {
    return proto_;
  }

 private:
  stream_executor::GpuDeviceInfoProto proto_;
};

class GpuTargetConfigWrapper {
 public:
  explicit GpuTargetConfigWrapper(stream_executor::GpuTargetConfigProto proto)
      : proto_(std::move(proto)),
        device_info_(proto_.gpu_device_info()) {}

  const GpuDeviceInfoWrapper& gpu_device_info() const { return device_info_; }
  const std::string& platform_name() const { return proto_.platform_name(); }
  const std::string& device_description_str() const {
    return proto_.device_description_str();
  }

  std::string arch_name() const {
    const auto& info = proto_.gpu_device_info();
    if (info.has_cuda_compute_capability()) {
      const auto& cc = info.cuda_compute_capability();
      return absl::StrCat(cc.major(), ".", cc.minor());
    }
    if (info.has_rocm_compute_capability()) {
      return std::string(info.rocm_compute_capability().gcn_arch_name());
    }
    return "";
  }

  int64_t compute_capability() const {
    const auto& info = proto_.gpu_device_info();
    if (info.has_cuda_compute_capability()) {
      const auto& cc = info.cuda_compute_capability();
      return cc.major() * 10 + cc.minor();
    }
    return 0;
  }

  const stream_executor::GpuTargetConfigProto& proto() const {
    return proto_;
  }

 private:
  stream_executor::GpuTargetConfigProto proto_;
  GpuDeviceInfoWrapper device_info_;
};

}  // namespace

NB_MODULE(_gpu_spec, m) {
  nb::class_<GpuDeviceInfoWrapper> gpu_device_info_class(m, "GpuDeviceInfo");
  gpu_device_info_class
      .def_prop_ro("device_vendor", &GpuDeviceInfoWrapper::device_vendor)
      .def_prop_ro("platform_version", &GpuDeviceInfoWrapper::platform_version)
      .def_prop_ro("pci_bus_id", &GpuDeviceInfoWrapper::pci_bus_id)
      .def_prop_ro("name", &GpuDeviceInfoWrapper::name)
      .def_prop_ro("model_str", &GpuDeviceInfoWrapper::model_str)
      .def_prop_ro("threads_per_block_limit",
                   &GpuDeviceInfoWrapper::threads_per_block_limit)
      .def_prop_ro("threads_per_warp", &GpuDeviceInfoWrapper::threads_per_warp)
      .def_prop_ro("shared_memory_per_block",
                   &GpuDeviceInfoWrapper::shared_memory_per_block)
      .def_prop_ro("shared_memory_per_core",
                   &GpuDeviceInfoWrapper::shared_memory_per_core)
      .def_prop_ro("threads_per_core_limit",
                   &GpuDeviceInfoWrapper::threads_per_core_limit)
      .def_prop_ro("core_count", &GpuDeviceInfoWrapper::core_count)
      .def_prop_ro("fpus_per_core", &GpuDeviceInfoWrapper::fpus_per_core)
      .def_prop_ro("block_dim_limit_x",
                   &GpuDeviceInfoWrapper::block_dim_limit_x)
      .def_prop_ro("block_dim_limit_y",
                   &GpuDeviceInfoWrapper::block_dim_limit_y)
      .def_prop_ro("block_dim_limit_z",
                   &GpuDeviceInfoWrapper::block_dim_limit_z)
      .def_prop_ro("memory_bandwidth",
                   &GpuDeviceInfoWrapper::memory_bandwidth)
      .def_prop_ro("l2_cache_size", &GpuDeviceInfoWrapper::l2_cache_size)
      .def_prop_ro("clock_rate_ghz", &GpuDeviceInfoWrapper::clock_rate_ghz)
      .def_prop_ro("device_memory_size",
                   &GpuDeviceInfoWrapper::device_memory_size)
      .def_prop_ro("shared_memory_per_block_optin",
                   &GpuDeviceInfoWrapper::shared_memory_per_block_optin)
      .def_prop_ro("registers_per_core_limit",
                   &GpuDeviceInfoWrapper::registers_per_core_limit)
      .def_prop_ro("registers_per_block_limit",
                   &GpuDeviceInfoWrapper::registers_per_block_limit)
      .def_prop_ro("driver_version", &GpuDeviceInfoWrapper::driver_version)
      .def_prop_ro("kernel_mode_driver_version",
                   &GpuDeviceInfoWrapper::kernel_mode_driver_version)
      .def_prop_ro("runtime_version", &GpuDeviceInfoWrapper::runtime_version)
      .def_prop_ro("compile_time_toolkit_version",
                   &GpuDeviceInfoWrapper::compile_time_toolkit_version)
      .def_prop_ro("dnn_version", &GpuDeviceInfoWrapper::dnn_version)
      .def_prop_ro("cub_version", &GpuDeviceInfoWrapper::cub_version)
      .def_prop_ro("numa_node", &GpuDeviceInfoWrapper::numa_node)
      .def_prop_ro("thread_dim_limit_x",
                   &GpuDeviceInfoWrapper::thread_dim_limit_x)
      .def_prop_ro("thread_dim_limit_y",
                   &GpuDeviceInfoWrapper::thread_dim_limit_y)
      .def_prop_ro("thread_dim_limit_z",
                   &GpuDeviceInfoWrapper::thread_dim_limit_z)
      .def_prop_ro("device_address_bits",
                   &GpuDeviceInfoWrapper::device_address_bits)
      .def_prop_ro("pcie_bandwidth", &GpuDeviceInfoWrapper::pcie_bandwidth)
      .def_prop_ro("ecc_enabled", &GpuDeviceInfoWrapper::ecc_enabled)
      .def_prop_ro("mem_clock_ghz", &GpuDeviceInfoWrapper::mem_clock_ghz)
      .def_prop_ro("reserved_shared_memory_per_block",
                   &GpuDeviceInfoWrapper::reserved_shared_memory_per_block)
      .def_prop_ro("max_blocks_per_multiprocessor",
                   &GpuDeviceInfoWrapper::max_blocks_per_multiprocessor)
      .def_prop_ro("collective_memory_granularity",
                   &GpuDeviceInfoWrapper::collective_memory_granularity);

  nb::class_<GpuTargetConfigWrapper> gpu_target_config_class(
      m, "GpuTargetConfig");
  gpu_target_config_class
      .def_prop_ro("gpu_device_info", &GpuTargetConfigWrapper::gpu_device_info)
      .def_prop_ro("platform_name", &GpuTargetConfigWrapper::platform_name)
      .def_prop_ro("device_description_str",
                   &GpuTargetConfigWrapper::device_description_str)
      .def_prop_ro("arch_name", &GpuTargetConfigWrapper::arch_name)
      .def_prop_ro("compute_capability",
                   &GpuTargetConfigWrapper::compute_capability);

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
