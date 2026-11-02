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

#ifndef XLA_PJRT_GPU_COLLECTIVE_GRAPH_INVALIDATION_H_
#define XLA_PJRT_GPU_COLLECTIVE_GRAPH_INVALIDATION_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/process_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"

namespace xla::gpu {

// The callback must outlive all executions using this topology so that its
// per-signature invocation counters stay aligned across processes.
CollectiveGraphInvalidationCallback CreateCollectiveGraphInvalidationCallback(
    ProcessId process_id,
    absl::flat_hash_map<GlobalDeviceId, ProcessId> device_to_process,
    std::shared_ptr<KeyValueStoreInterface> kv_store);

}  // namespace xla::gpu

#endif  // XLA_PJRT_GPU_COLLECTIVE_GRAPH_INVALIDATION_H_
