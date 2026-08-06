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

#include "xla/stream_executor/sycl/sycl_gemm_workspace.h"

#include <cstddef>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#if TENSORFLOW_USE_SYCL
#include "xla/stream_executor/sycl/sycl_matmul_utils.h"
#endif

namespace stream_executor {
namespace sycl {

absl::StatusOr<size_t> GetGemmScratchpadSize(
    const xla::gpu::GemmConfig& config,
    xla::gpu::GemmBackendConfig_Epilogue epilogue) {
#if TENSORFLOW_USE_SYCL
  auto prim_desc_or = CreateMatMulPrimDescFromGemmConfig(config, epilogue);
  if (!prim_desc_or.ok()) {
    return prim_desc_or.status();
  }
  return (*prim_desc_or)->scratchpad_desc().get_size();
#else
  (void)config;
  (void)epilogue;
  return absl::UnimplementedError("SYCL support is not compiled in.");
#endif
}

}  // namespace sycl
}  // namespace stream_executor
