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

#include "xla/service/gpu/gpu_module_globals.h"

#include <memory>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/map_util.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/scoped_module_handle.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"

namespace xla::gpu {

absl::StatusOr<const GpuModuleGlobals::BufferAllocToDeviceMemoryMap*>
GpuModuleGlobals::Resolve(se::Stream* stream) {
  se::StreamExecutor* executor = stream->parent();

  absl::MutexLock lock(mutex_);
  auto it = globals_.find(executor);
  if (it != globals_.end()) {
    return it->second.get();
  }

  se::MultiModuleLoaderSpec module_spec;
  if (!binary_.empty()) {
    module_spec.AddCudaCubinInMemory(binary_);
  }

  auto globals = std::make_unique<BufferAllocToDeviceMemoryMap>();
  se::ModuleHandle module_handle;
  // The CUDA driver isn't able to load a PTX and a binary which are both empty.
  // It's okay if we skip loading in this case; if the module isn't loaded, all
  // symbol lookups will fail, just as they should for an empty module.
  if (!(executor->GetPlatform()->id() == se::cuda::kCudaPlatformId &&
        binary_.empty())) {
    ASSIGN_OR_RETURN(module_handle, executor->LoadModule(module_spec));
  }

  // A flag signalling if constant initialization submitted memcpy operations
  // to the `stream`.
  int submitted_mem_copies = 0;

  for (const GpuExecutable::ConstantInfo& info : constants_) {
    absl::StatusOr<se::DeviceAddressBase> global_status;
    if (static_cast<bool>(module_handle)) {
      global_status = executor->GetSymbol(info.symbol_name, module_handle);
    }

    se::DeviceAddressBase global;

    CHECK(static_cast<bool>(module_handle) && global_status.ok());
    // The constant was defined in the PTX and has been allocated by the CUDA
    // driver.
    global = *global_status;
    XLA_VLOG_DEVICE(3, executor->device_ordinal()) << absl::StreamFormat(
        "Resolved global %s to %p", info.symbol_name, global.opaque());

    if (!info.content.span().empty()) {
      // This means the constant did not have an initializer in the PTX and
      // therefore must be initialized by XLA here.
      RETURN_IF_ERROR(stream->Memcpy(&global, info.content.span().data(),
                                     info.content.span().size()));
      submitted_mem_copies = true;
    }

    if (info.allocation_index != -1) {
      InsertOrDie(globals.get(), info.allocation_index, global);
    }
  }

  // Wait for the completion of all host->device transfers, to guarantee that
  // destructor will not race with any operations in flight (deallocate
  // xla::Literal owned by the HLO module).
  if (submitted_mem_copies) {
    CHECK_OK(stream->BlockHostUntilDone());
  }

  module_handles_.emplace(executor,
                          se::ScopedModuleHandle(executor, module_handle));
  return globals_.emplace(executor, std::move(globals)).first->second.get();
}

}  // namespace xla::gpu
