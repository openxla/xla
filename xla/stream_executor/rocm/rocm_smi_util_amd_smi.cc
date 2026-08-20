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

// amd_smi backend for the SMI queries declared in rocm_smi_util.h. Compiled in
// from ROCm 7.13 on; rocm_smi_util_rocm_smi.cc takes its place below that.

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "rocm/include/amd_smi/amdsmi.h"
#include "xla/stream_executor/rocm/rocm_smi_util.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {
namespace {

amdsmi_processor_handle ToProcessorHandle(SmiDeviceHandle device) {
  return reinterpret_cast<amdsmi_processor_handle>(device.value);
}

SmiDeviceHandle ToDeviceHandle(amdsmi_processor_handle processor) {
  return SmiDeviceHandle{reinterpret_cast<uintptr_t>(processor)};
}

}  // namespace

ABSL_CONST_INIT const absl::string_view kSmiLibraryName = "amd_smi";

bool InitRocmSmi() {
  static bool initialized = []() {
    amdsmi_status_t status = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
    if (status != AMDSMI_STATUS_SUCCESS) {
      const char* err_str = nullptr;
      amdsmi_status_code_to_string(status, &err_str);
      LOG(WARNING) << "amdsmi_init failed: "
                   << (err_str ? err_str : "unknown error");
      return false;
    }
    return true;
  }();
  return initialized;
}

std::vector<SmiDeviceHandle> EnumerateDevices() {
  uint32_t num_sockets = 0;
  if (amdsmi_get_socket_handles(&num_sockets, nullptr) !=
          AMDSMI_STATUS_SUCCESS ||
      num_sockets == 0) {
    return {};
  }

  std::vector<amdsmi_socket_handle> sockets(num_sockets);
  if (amdsmi_get_socket_handles(&num_sockets, sockets.data()) !=
      AMDSMI_STATUS_SUCCESS) {
    return {};
  }

  std::vector<SmiDeviceHandle> devices;
  for (amdsmi_socket_handle socket : sockets) {
    uint32_t num_processors = 0;
    if (amdsmi_get_processor_handles(socket, &num_processors, nullptr) !=
            AMDSMI_STATUS_SUCCESS ||
        num_processors == 0) {
      continue;
    }

    std::vector<amdsmi_processor_handle> processors(num_processors);
    if (amdsmi_get_processor_handles(socket, &num_processors,
                                     processors.data()) !=
        AMDSMI_STATUS_SUCCESS) {
      continue;
    }

    // amdsmi_init(AMDSMI_INIT_AMD_GPUS) already restricts enumeration to
    // sockets holding AMD GPUs, but a socket can still expose processors that
    // are not GPUs, so filter by type.
    for (amdsmi_processor_handle processor : processors) {
      processor_type_t type = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
      if (amdsmi_get_processor_type(processor, &type) ==
              AMDSMI_STATUS_SUCCESS &&
          type == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
        devices.push_back(ToDeviceHandle(processor));
      }
    }
  }

  return devices;
}

std::optional<SmiDeviceHandle> FindDevice(const BdfComponents& target_bdf) {
  amdsmi_bdf_t bdf = {};
  bdf.bdf.domain_number = target_bdf.domain;
  bdf.bdf.bus_number = target_bdf.bus;
  bdf.bdf.device_number = target_bdf.device;
  bdf.bdf.function_number = target_bdf.function;

  amdsmi_processor_handle handle = nullptr;
  if (amdsmi_get_processor_handle_from_bdf(bdf, &handle) !=
          AMDSMI_STATUS_SUCCESS ||
      handle == nullptr) {
    return std::nullopt;
  }

  return ToDeviceHandle(handle);
}

std::optional<PcieLinkStatus> QueryPcieLinkStatus(
    SmiDeviceHandle device, absl::string_view pci_bus_id) {
  amdsmi_pcie_info_t pcie_info = {};
  amdsmi_status_t status =
      amdsmi_get_pcie_info(ToProcessorHandle(device), &pcie_info);
  if (status != AMDSMI_STATUS_SUCCESS) {
    const char* err_str = nullptr;
    amdsmi_status_code_to_string(status, &err_str);
    LOG(WARNING) << "amdsmi_get_pcie_info failed for " << pci_bus_id << ": "
                 << (err_str ? err_str : "unknown error");
    return std::nullopt;
  }

  // amdsmi.h documents pcie_metric.pcie_speed as "current PCIe speed in MT/s",
  // so no scaling here, unlike rocm_smi's 0.1 GT/s field.
  return PcieLinkStatus{pcie_info.pcie_metric.pcie_speed,
                        pcie_info.pcie_metric.pcie_width};
}

std::optional<uint64_t> QueryHiveId(SmiDeviceHandle device) {
  amdsmi_xgmi_info_t xgmi_info = {};
  if (amdsmi_get_xgmi_info(ToProcessorHandle(device), &xgmi_info) !=
      AMDSMI_STATUS_SUCCESS) {
    return std::nullopt;
  }
  return xgmi_info.xgmi_hive_id;
}

bool IsXgmiPeer(SmiDeviceHandle src, SmiDeviceHandle dst) {
  // The API rejects a null hops pointer; only the link type is used.
  uint64_t hops = 0;
  amdsmi_link_type_t link_type = AMDSMI_LINK_TYPE_UNKNOWN;
  if (amdsmi_topo_get_link_type(ToProcessorHandle(src), ToProcessorHandle(dst),
                                &hops, &link_type) != AMDSMI_STATUS_SUCCESS) {
    return false;
  }
  return link_type == AMDSMI_LINK_TYPE_XGMI;
}

}  // namespace stream_executor::gpu
