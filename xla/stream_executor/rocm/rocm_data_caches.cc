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

#include "xla/stream_executor/rocm/rocm_data_caches.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/smi_util.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {
namespace {

absl::StatusOr<std::vector<SmiDataCache>> QuerySmiDataCaches(
    absl::string_view pci_bus_id) {
  absl::MutexLock lock(smi_mutex);

  ABSL_RETURN_IF_ERROR(InitSmi());

  ABSL_ASSIGN_OR_RETURN(BdfComponents bdf, ParseBdf(pci_bus_id));
  ABSL_ASSIGN_OR_RETURN(SmiDeviceHandle device, FindDevice(bdf));
  return QueryDataCaches(device);
}

}  // namespace

std::vector<DataCacheInfo> BuildDataCacheHierarchy(
    absl::Span<const SmiDataCache> caches, int core_count, int xcc_count,
    int64_t hip_l2_cache_size, int64_t default_l1_cache_size) {
  // SMI may list several caches at one level, such as the scalar cache and
  // RDNA's GL1 next to the vector L1, each split further by harvesting. Keep
  // the one with the most instances, which is the most private.
  std::map<int, SmiDataCache> per_level;
  for (const SmiDataCache& cache : caches) {
    if (cache.level == 0 || cache.size_bytes <= 0) {
      continue;
    }
    auto it = per_level.try_emplace(cache.level, cache).first;
    if (cache.num_instances > it->second.num_instances) {
      it->second = cache;
    }
  }
  per_level.try_emplace(1, SmiDataCache{1, default_l1_cache_size, 0});
  // The L2 size stays HIP's, as autotune cache keys depend on it.
  if (hip_l2_cache_size > 0) {
    per_level[2].size_bytes = hip_l2_cache_size;
  }

  // KFD instance counts are unreliable (L2 and L3 are listed once per GPU, and
  // rocm_smi folds scalar caches into the L1 group), so take them from the
  // topology instead. One L1 per CU, one L2 per XCD, one device wide L3.
  std::vector<DataCacheInfo> hierarchy;
  hierarchy.reserve(per_level.size());
  for (const auto& [level, cache] : per_level) {
    const int num_instances = level == 1   ? core_count
                              : level == 2 ? std::max(1, xcc_count)
                                           : 1;
    hierarchy.push_back(DataCacheInfo{level, cache.size_bytes, num_instances});
  }
  return hierarchy;
}

std::vector<DataCacheInfo> GetRocmDataCaches(absl::string_view pci_bus_id,
                                             int core_count, int xcc_count,
                                             int64_t hip_l2_cache_size,
                                             int64_t default_l1_cache_size) {
  absl::StatusOr<std::vector<SmiDataCache>> caches =
      QuerySmiDataCaches(pci_bus_id);
  if (!caches.ok()) {
    if (absl::IsUnimplemented(caches.status())) {
      VLOG(1) << "Data caches of " << pci_bus_id
              << " are not queried with the rocm-smi backend, reporting only "
                 "the default L1 and the L2 from HIP.";
    } else {
      LOG(WARNING) << "Could not read the data caches of " << pci_bus_id
                   << " over SMI (" << caches.status().message()
                   << "), reporting only the default L1 and the L2 from HIP.";
    }
    caches.emplace();
  }

  std::vector<DataCacheInfo> hierarchy = BuildDataCacheHierarchy(
      *caches, core_count, xcc_count, hip_l2_cache_size, default_l1_cache_size);
  for (const DataCacheInfo& cache : hierarchy) {
    VLOG(1) << "L" << cache.level << " data cache of " << pci_bus_id << ": "
            << cache.num_instances << " x " << cache.size_bytes << " bytes.";
  }
  return hierarchy;
}

}  // namespace stream_executor::gpu
