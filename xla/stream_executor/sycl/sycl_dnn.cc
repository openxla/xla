/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/stream_executor/sycl/sycl_dnn.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_timer.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_internal.h"
#include "absl/strings/string_view.h"


namespace stream_executor {
namespace gpu {

OnednnSupport::OnednnSupport(GpuExecutor* parent) : parent_(parent) {}

absl::Status OnednnSupport::Init() {
  return absl::OkStatus();
}

absl::StatusOr<stream_executor::dnn::VersionInfo> OnednnSupport::GetVersion() {
  return dnn::VersionInfo(0, 0, 0);
}

}  // namespace gpu

void initialize_onednn() {
  absl::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::DnnFactory>(
          sycl::kSyclPlatformId, "oneDNN",
          [](internal::StreamExecutorInterface* parent) -> dnn::DnnSupport* {
            gpu::GpuExecutor* sycl_executor =
                dynamic_cast<gpu::GpuExecutor*>(parent);
            if (sycl_executor == nullptr) {
              LOG(ERROR) << "Attempting to initialize an instance of the oneDNN "
                         << "support library with a non-SYCL StreamExecutor";
              return nullptr;
            }

            gpu::OnednnSupport* dnn = new gpu::OnednnSupport(sycl_executor);
            if (!dnn->Init().ok()) {
              // Note: Init() will log a more specific error.
              delete dnn;
              return nullptr;
            }
            return dnn;
          });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register oneDNN factory: " << status.message();
  }
}

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(register_onednn, {
  stream_executor::initialize_onednn();
});
