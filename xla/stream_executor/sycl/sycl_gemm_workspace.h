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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_GEMM_WORKSPACE_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_GEMM_WORKSPACE_H_

#include <cstddef>

#include "absl/status/statusor.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/matmul_utils.h"

namespace stream_executor {
namespace sycl {

// Returns the oneDNN matmul scratchpad size (in bytes) for `config` with the
// given `epilogue`. Callable on any build — when SYCL is not configured this
// returns absl::UnimplementedError, letting callers fall back to their default
// workspace estimate without a preprocessor guard at the call site.
absl::StatusOr<size_t> GetGemmScratchpadSize(
    const xla::gpu::GemmConfig& config,
    xla::gpu::GemmBackendConfig_Epilogue epilogue);

}  // namespace sycl
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_GEMM_WORKSPACE_H_
