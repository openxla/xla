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

#include "xla/stream_executor/rocm/rocm_xgmi_topology.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "rocm/rocm_config.h"
#include "xla/stream_executor/rocm/rocm_smi_util.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {
namespace {

// Returns the xGMI hive ID of the device, or std::nullopt if the query fails
// (typically because the device is not part of a hive).
std::optional<uint64_t> QueryHiveId(SmiDeviceHandle device) {
#if (TF_ROCM_VERSION >= 71300)
  amdsmi_xgmi_info_t xgmi_info = {};
  if (amdsmi_get_xgmi_info(device, &xgmi_info) != AMDSMI_STATUS_SUCCESS) {
    return std::nullopt;
  }
  return xgmi_info.xgmi_hive_id;
#else
  uint64_t hive_id = 0;
  if (rsmi_dev_xgmi_hive_id_get(device, &hive_id) != RSMI_STATUS_SUCCESS) {
    return std::nullopt;
  }
  return hive_id;
#endif  // TF_ROCM_VERSION >= 71300
}

// Returns true if src reaches dst over an xGMI link.
bool IsXgmiPeer(SmiDeviceHandle src, SmiDeviceHandle dst) {
  // Both APIs reject a null hops pointer; only the link type is used.
  uint64_t hops = 0;
#if (TF_ROCM_VERSION >= 71300)
  amdsmi_link_type_t link_type = AMDSMI_LINK_TYPE_UNKNOWN;
  if (amdsmi_topo_get_link_type(src, dst, &hops, &link_type) !=
      AMDSMI_STATUS_SUCCESS) {
    return false;
  }
  return link_type == AMDSMI_LINK_TYPE_XGMI;
#else
  RSMI_IO_LINK_TYPE link_type = RSMI_IOLINK_TYPE_UNDEFINED;
  if (rsmi_topo_get_link_type(src, dst, &hops, &link_type) !=
      RSMI_STATUS_SUCCESS) {
    return false;
  }
  return link_type == RSMI_IOLINK_TYPE_XGMI;
#endif  // TF_ROCM_VERSION >= 71300
}

}  // namespace

XgmiTopologyInfo GetRocmXgmiTopology(absl::string_view pci_bus_id) {
  XgmiTopologyInfo info;

  absl::MutexLock lock(rocm_smi_mutex);

  if (!InitRocmSmi()) return info;

  std::optional<BdfComponents> bdf = ParseBdf(pci_bus_id);
  if (!bdf.has_value()) {
    LOG(WARNING) << "Failed to parse PCI bus ID for xGMI query: " << pci_bus_id;
    return info;
  }

  std::optional<SmiDeviceHandle> device = FindDeviceIndex(*bdf);
  if (!device.has_value()) {
    LOG(WARNING) << kSmiLibraryName << " could not find device for PCI bus ID "
                 << pci_bus_id << " (xGMI query)";
    return info;
  }

  std::optional<uint64_t> hive_id = QueryHiveId(*device);
  if (hive_id.has_value()) {
    info.hive_id = *hive_id;
  } else {
    VLOG(1) << "xGMI hive ID query failed for " << pci_bus_id
            << "; device may not be in an xGMI hive.";
  }

  // Count peers reachable over xGMI by querying the link type to every other
  // device. This counts peer GPUs, not physical links.
  std::vector<SmiDeviceHandle> devices = EnumerateDevices();
  if (devices.size() <= 1) return info;

  int xgmi_links = 0;
  for (SmiDeviceHandle peer : devices) {
    if (peer == *device) continue;
    if (IsXgmiPeer(*device, peer)) ++xgmi_links;
  }

  info.active_links = xgmi_links;

  VLOG(1) << "xGMI topology for " << pci_bus_id << " via " << kSmiLibraryName
          << ": " << xgmi_links << " active xGMI links"
          << " (hive_id=" << info.hive_id << ", num_devices=" << devices.size()
          << ")";

  return info;
}

}  // namespace stream_executor::gpu
