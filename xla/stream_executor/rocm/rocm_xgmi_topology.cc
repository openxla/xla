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
#include "xla/stream_executor/rocm/rocm_smi_util.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {

XgmiTopologyInfo GetRocmXgmiTopology(absl::string_view pci_bus_id) {
  XgmiTopologyInfo info;

  absl::MutexLock lock(rocm_smi_mutex);

  if (!InitRocmSmi()) return info;

  std::optional<BdfComponents> bdf = ParseBdf(pci_bus_id);
  if (!bdf.has_value()) {
    LOG(WARNING) << "Failed to parse PCI bus ID for xGMI query: " << pci_bus_id;
    return info;
  }

  std::optional<SmiDeviceHandle> device = FindDevice(*bdf);
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
