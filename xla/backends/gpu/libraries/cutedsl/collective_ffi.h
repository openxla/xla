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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_COLLECTIVE_FFI_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_COLLECTIVE_FFI_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/libraries/cutedsl/collective_config.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/stream_executor/device_address.h"

namespace xla::gpu::cutedsl::internal {

// Resolves one absolute peer-address row for every configured region. The
// buffers span has one whole FFI argument or result buffer per peer region in
// configuration order. This seam is exposed only for focused address and
// overflow tests; it is not part of the generated-function ABI.
absl::StatusOr<std::vector<uint64_t>> ResolvePeerAddressesV3(
    const xla::gpu::GpuCliqueKey& clique_key, xla::RankId rank,
    absl::Span<const PeerRegionV3> peer_regions,
    absl::Span<const stream_executor::DeviceAddressBase> buffers,
    const xla::gpu::CollectiveMemory& collective_memory);

}  // namespace xla::gpu::cutedsl::internal

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_COLLECTIVE_FFI_H_
