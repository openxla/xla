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

// rocm_smi backend for the SMI queries declared in rocm_smi_util.h. Compiled
// in below ROCm 7.13; rocm_smi_util_amd_smi.cc takes its place from 7.13 on.

#include "rocm/include/rocm_smi/rocm_smi.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/rocm/rocm_smi_util.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {
namespace {

uint32_t ToDeviceIndex(SmiDeviceHandle device) {
  return static_cast<uint32_t>(device.value);
}

SmiDeviceHandle ToDeviceHandle(uint32_t device_index) {
  return SmiDeviceHandle{device_index};
}

}  // namespace

ABSL_CONST_INIT const absl::string_view kSmiLibraryName = "rocm_smi";

bool InitRocmSmi() {
  static bool initialized = []() {
    rsmi_status_t status = rsmi_init(0);
    if (status != RSMI_STATUS_SUCCESS) {
      const char* err_str = nullptr;
      rsmi_status_string(status, &err_str);
      LOG(WARNING) << "rsmi_init failed: "
                   << (err_str ? err_str : "unknown error");
      return false;
    }
    return true;
  }();
  return initialized;
}

std::vector<SmiDeviceHandle> EnumerateDevices() {
  uint32_t num_devices = 0;
  rsmi_status_t status = rsmi_num_monitor_devices(&num_devices);
  if (status != RSMI_STATUS_SUCCESS || num_devices == 0) {
    return {};
  }

  std::vector<SmiDeviceHandle> devices;
  devices.reserve(num_devices);
  for (uint32_t i = 0; i < num_devices; ++i) {
    devices.push_back(ToDeviceHandle(i));
  }
  return devices;
}

std::optional<SmiDeviceHandle> FindDevice(const BdfComponents& target_bdf) {
  for (SmiDeviceHandle device : EnumerateDevices()) {
    uint64_t bdfid = 0;
    if (rsmi_dev_pci_id_get(ToDeviceIndex(device), &bdfid) !=
        RSMI_STATUS_SUCCESS) {
      continue;
    }

    // Unpack rocm_smi's 64-bit BDF format into individual fields.
    // See
    // rocm-systems/projects/rocm-smi-lib/src/rocm_smi.cc:rsmi_dev_pci_id_get
    // for details on the packing.
    uint32_t domain = (bdfid >> 32) & 0xFFFFFFFF;
    uint8_t bus = (bdfid >> 8) & 0xFF;
    uint8_t device_number = (bdfid >> 3) & 0x1F;
    uint8_t function = bdfid & 0x7;

    if (domain == target_bdf.domain && bus == target_bdf.bus &&
        device_number == target_bdf.device && function == target_bdf.function) {
      return device;
    }
  }

  return std::nullopt;
}

std::optional<PcieLinkStatus> QueryPcieLinkStatus(
    SmiDeviceHandle device, absl::string_view pci_bus_id) {
  rsmi_gpu_metrics_t gpu_metrics = {};
  rsmi_status_t status =
      rsmi_dev_gpu_metrics_info_get(ToDeviceIndex(device), &gpu_metrics);
  if (status != RSMI_STATUS_SUCCESS) {
    const char* err_str = nullptr;
    rsmi_status_string(status, &err_str);
    LOG(WARNING) << "rsmi_dev_gpu_metrics_info_get failed for " << pci_bus_id
                 << ": " << (err_str ? err_str : "unknown error");
    return std::nullopt;
  }

  // rocm_smi reports pcie_link_speed in units of 0.1 GT/s, so scale to MT/s.
  return PcieLinkStatus{
      static_cast<uint32_t>(gpu_metrics.pcie_link_speed) * 100,
      gpu_metrics.pcie_link_width};
}

std::optional<uint64_t> QueryHiveId(SmiDeviceHandle device) {
  uint64_t hive_id = 0;
  if (rsmi_dev_xgmi_hive_id_get(ToDeviceIndex(device), &hive_id) !=
      RSMI_STATUS_SUCCESS) {
    return std::nullopt;
  }
  return hive_id;
}

bool IsXgmiPeer(SmiDeviceHandle src, SmiDeviceHandle dst) {
  // The API rejects a null hops pointer; only the link type is used.
  uint64_t hops = 0;
  RSMI_IO_LINK_TYPE link_type = RSMI_IOLINK_TYPE_UNDEFINED;
  if (rsmi_topo_get_link_type(ToDeviceIndex(src), ToDeviceIndex(dst), &hops,
                              &link_type) != RSMI_STATUS_SUCCESS) {
    return false;
  }
  return link_type == RSMI_IOLINK_TYPE_XGMI;
}

}  // namespace stream_executor::gpu
