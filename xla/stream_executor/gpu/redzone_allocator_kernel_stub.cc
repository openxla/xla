/* Copyright 2025 The OpenXLA Authors.

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

#include <cstdint>

#include "absl/status/statusor.h"
#include "tsl/platform/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/redzone_allocator_kernel.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"

namespace stream_executor {

// A stub for ComparisonKernel for Redzone allocator to enable sycl build.
// Proper functionality will be implemented soon.
absl::StatusOr<ComparisonKernel *> GetComparisonKernel(StreamExecutor *executor,
                                                       GpuAsmOpts) {
  return absl::UnimplementedError("Unimplemented");
}

}  // namespace stream_executor
