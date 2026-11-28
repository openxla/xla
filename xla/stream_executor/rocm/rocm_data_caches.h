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

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/smi_util.h"

namespace stream_executor::gpu {

// Returns the data caches of the device at `pci_bus_id` as amd-smi reports
// them, in SMI order and without instruction caches. Unimplemented with the
// rocm_smi backend (before ROCm 7.13).
absl::StatusOr<std::vector<SmiDataCache>> QueryRocmDataCaches(
    absl::string_view pci_bus_id);

// Returns the data cache hierarchy, one cache per level, built from `caches`
// (what QueryRocmDataCaches returned). Fails if `caches` has no L1 or no L2,
// so the caller can fall back to what HIP reports. Instance counts come from
// the topology, `core_count` L1s, `xcc_count` L2s and one of each higher level.
absl::StatusOr<std::vector<DataCacheInfo>> BuildDataCacheHierarchy(
    absl::Span<const SmiDataCache> caches, int core_count, int xcc_count);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_DATA_CACHES_H_
