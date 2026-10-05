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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_DATA_CACHES_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_DATA_CACHES_H_

#include <cstdint>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/smi_util.h"

namespace stream_executor::gpu {

// Returns the data caches of the device at `pci_bus_id`, read over SMI where
// supported.
std::vector<DataCacheInfo> GetRocmDataCaches(absl::string_view pci_bus_id,
                                             int core_count, int xcc_count,
                                             int64_t hip_l2_cache_size);

// Returns the data caches built from already queried `caches`.
std::vector<DataCacheInfo> BuildDataCacheHierarchy(
    absl::Span<const SmiDataCache> caches, int core_count, int xcc_count,
    int64_t hip_l2_cache_size);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_DATA_CACHES_H_
