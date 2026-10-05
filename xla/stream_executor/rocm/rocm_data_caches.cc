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
#include <vector>

#include "absl/algorithm/container.h"
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

// KFD lists each L1 instance separately, so SMI's L1 counts are exact. The L2
// and L3 are listed once per GPU, so their counts follow the topology instead:
// one L2 per XCD, one device-wide L3.
int NumInstances(const SmiDataCache& cache, int core_count, int xcc_count) {
  switch (cache.level) {
    case 1: {
      // On gfx9 rocm_smi folds orphaned scalar caches (also 16 KiB) into the
      // vector L1 group. Capping at the CU count undoes that.
      const int num_instances =
          std::max(1, static_cast<int>(cache.num_instances));
      return core_count > 0 ? std::min(num_instances, core_count)
                            : num_instances;
    }
    case 2:
      return std::max(1, xcc_count);
    default:
      return 1;
  }
}

}  // namespace

std::vector<DataCacheInfo> BuildDataCacheHierarchy(
    absl::Span<const SmiDataCache> caches, int core_count, int xcc_count,
    int64_t hip_l2_cache_size, int64_t default_l1_cache_size) {
  std::vector<DataCacheInfo> all;
  all.reserve(caches.size());
  for (const SmiDataCache& cache : caches) {
    if (cache.level == 0 || cache.size_bytes <= 0) continue;
    all.push_back(DataCacheInfo{static_cast<int>(cache.level), cache.size_bytes,
                                NumInstances(cache, core_count, xcc_count)});
  }
  absl::c_stable_sort(all, [](const DataCacheInfo& a, const DataCacheInfo& b) {
    return a.level != b.level ? a.level < b.level
                              : a.num_instances > b.num_instances;
  });

  // Level 1 also holds the scalar cache and, on RDNA, the GL1. Keep the vector
  // L1, which has the most instances and sorts first, and anything larger than
  // it (the GL1). Drop the rest: scalar caches and their harvested splits.
  std::vector<DataCacheInfo> hierarchy;
  hierarchy.reserve(all.size());
  const DataCacheInfo* vector_l1 = nullptr;
  for (const DataCacheInfo& cache : all) {
    if (cache.level == 1) {
      if (vector_l1 == nullptr) {
        vector_l1 = &cache;
      } else if (cache.size_bytes <= vector_l1->size_bytes) {
        continue;
      }
    }
    hierarchy.push_back(cache);
  }

  // No per-CU L1 from SMI, so use the legacy default.
  if (absl::c_none_of(hierarchy, [&](const DataCacheInfo& cache) {
        return cache.level == 1 && cache.num_instances == core_count;
      })) {
    hierarchy.insert(hierarchy.begin(),
                     DataCacheInfo{/*level=*/1, default_l1_cache_size,
                                   /*num_instances=*/core_count});
  }

  // The L2 size stays HIP's, as autotune cache keys depend on it. HIP also
  // covers a missing SMI L2.
  if (hip_l2_cache_size > 0) {
    auto it = absl::c_find_if(
        hierarchy, [](const DataCacheInfo& cache) { return cache.level >= 2; });
    if (it != hierarchy.end() && it->level == 2) {
      if (it->size_bytes != hip_l2_cache_size) {
        VLOG(1) << "SMI reports a " << it->size_bytes
                << " byte L2, HIP reports " << hip_l2_cache_size
                << " bytes; using HIP's.";
      }
      it->size_bytes = hip_l2_cache_size;
    } else {
      hierarchy.insert(it,
                       DataCacheInfo{/*level=*/2, hip_l2_cache_size,
                                     /*num_instances=*/std::max(1, xcc_count)});
    }
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
