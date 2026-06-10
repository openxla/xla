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

#ifndef XLA_SERVICE_GPU_GPU_EXECUTABLE_MODULE_CACHE_H_
#define XLA_SERVICE_GPU_GPU_EXECUTABLE_MODULE_CACHE_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/scoped_module_handle.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

// Owns loaded GPU modules and the resolved addresses of executable constants.
// The cache is per GpuExecutable, keyed by StreamExecutor, and keeps module
// handles alive until the executable is destroyed.
class GpuExecutableModuleCache {
 public:
  struct Constant {
    std::string symbol_name;
    absl::Span<const uint8_t> content;
    int allocation_index = -1;
  };

  using BufferAllocToDeviceMemoryMap =
      absl::flat_hash_map<BufferAllocation::Index, se::DeviceAddressBase>;

  absl::StatusOr<const BufferAllocToDeviceMemoryMap*> ResolveConstantGlobals(
      se::Stream* stream, const std::string& asm_text,
      absl::Span<const uint8_t> binary, absl::Span<const Constant> constants);

 private:
  absl::Mutex mutex_;

  // Cache of module handles. Required to keep loaded modules alive until this
  // cache is destroyed.
  absl::flat_hash_map<se::StreamExecutor*, se::ScopedModuleHandle>
      module_handles_ ABSL_GUARDED_BY(mutex_);

  // Cache of constant buffer allocation maps used by ResolveConstantGlobals.
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<BufferAllocToDeviceMemoryMap>>
      module_globals_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_MODULE_CACHE_H_
