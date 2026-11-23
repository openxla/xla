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

#include "xla/stream_executor/rocm/rocm_cumem.h"

#include <cstdint>

#include "xla/tsl/util/env_var.h"

namespace stream_executor {

bool IsNcclCuMemEnabled() {
  static const bool enabled = [] {
    int64_t value = 0;
    // An unparsable value leaves the allocator disabled.
    return tsl::ReadInt64FromEnvVar("NCCL_CUMEM_ENABLE", /*default_val=*/0,
                                    &value)
               .ok() &&
           value >= 1;
  }();
  return enabled;
}

}  // namespace stream_executor
