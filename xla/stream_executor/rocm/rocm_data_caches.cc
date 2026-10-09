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
  ABSL_ASSIGN_OR_RETURN(BdfComponents bdf, ParseBdf(pci_bus_id));

  absl::MutexLock lock(smi_mutex);
  ABSL_RETURN_IF_ERROR(InitSmi());
  ABSL_ASSIGN_OR_RETURN(SmiDeviceHandle device, FindDevice(bdf));
  return QueryDataCaches(device);
}

int NumL2Instances(int xcc_count) { return std::max(1, xcc_count); }

// KFD lists every L1 instance but the L2 and L3 only once per GPU, so their
// counts come from the topology: one L2 per XCD, one L3.
int NumInstances(const SmiDataCache& cache, int core_count, int xcc_count) {
  switch (cache.level) {
    case 1: {
      // amd-smi may count orphaned scalar caches as vector L1s.
      const int num_instances =
          std::max(1, static_cast<int>(cache.num_instances));
      return core_count > 0 ? std::min(num_instances, core_count)
                            : num_instances;
    }
    case 2:
      return NumL2Instances(xcc_count);
    default:
      return 1;
  }
}

}  // namespace

std::vector<DataCacheInfo> BuildDataCacheHierarchy(
    absl::Span<const SmiDataCache> caches, int core_count, int xcc_count,
    int64_t hip_l2_cache_size) {
  std::vector<DataCacheInfo> all;
  all.reserve(caches.size());
  for (const SmiDataCache& cache : caches) {
    if (cache.level == 0 || cache.size_bytes <= 0) {
      continue;
    }
    all.push_back(DataCacheInfo{static_cast<int>(cache.level), cache.size_bytes,
                                NumInstances(cache, core_count, xcc_count)});
  }
  // The per-CU L1 is the level 1 cache with the most instances, the larger one
  // on a tie. Keep it and larger level 1 caches such as the GL1. Drop the
  // scalar caches.
  const DataCacheInfo* vector_l1 = nullptr;
  for (const DataCacheInfo& cache : all) {
    if (cache.level == 1 && (vector_l1 == nullptr ||
                             cache.num_instances > vector_l1->num_instances ||
                             (cache.num_instances == vector_l1->num_instances &&
                              cache.size_bytes > vector_l1->size_bytes))) {
      vector_l1 = &cache;
    }
  }
  std::vector<DataCacheInfo> hierarchy;
  hierarchy.reserve(all.size() + 1);
  for (const DataCacheInfo& cache : all) {
    if (cache.level == 1 && &cache != vector_l1 &&
        cache.size_bytes <= vector_l1->size_bytes) {
      continue;
    }
    hierarchy.push_back(cache);
  }

  // Keep HIP's L2 size, which autotune cache keys depend on.
  if (hip_l2_cache_size > 0) {
    auto l2 = absl::c_find_if(
        hierarchy, [](const DataCacheInfo& cache) { return cache.level == 2; });
    if (l2 != hierarchy.end()) {
      if (l2->size_bytes != hip_l2_cache_size) {
        VLOG(1) << "SMI reports a " << l2->size_bytes
                << " byte L2, HIP reports " << hip_l2_cache_size
                << " bytes; using HIP's.";
      }
      l2->size_bytes = hip_l2_cache_size;
    } else {
      hierarchy.push_back(DataCacheInfo{/*level=*/2, hip_l2_cache_size,
                                        NumL2Instances(xcc_count)});
    }
  }
  return hierarchy;
}

std::vector<DataCacheInfo> GetRocmDataCaches(absl::string_view pci_bus_id,
                                             int core_count, int xcc_count,
                                             int64_t hip_l2_cache_size) {
  absl::StatusOr<std::vector<SmiDataCache>> caches =
      QuerySmiDataCaches(pci_bus_id);
  if (!caches.ok()) {
    if (absl::IsUnimplemented(caches.status())) {
      VLOG(1) << "Data caches of " << pci_bus_id
              << " are not queried with the rocm-smi backend, reporting only "
                 "the L2 from HIP.";
    } else {
      LOG(WARNING) << "Could not read the data caches of " << pci_bus_id
                   << " over SMI (" << caches.status().message()
                   << "), reporting only the L2 from HIP.";
    }
    caches.emplace();
  }

  std::vector<DataCacheInfo> hierarchy = BuildDataCacheHierarchy(
      *caches, core_count, xcc_count, hip_l2_cache_size);
  for (const DataCacheInfo& cache : hierarchy) {
    VLOG(1) << "L" << cache.level << " data cache of " << pci_bus_id << ": "
            << cache.num_instances << " x " << cache.size_bytes << " bytes.";
  }
  return hierarchy;
}

}  // namespace stream_executor::gpu
