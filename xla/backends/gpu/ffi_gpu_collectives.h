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

#ifndef XLA_BACKENDS_GPU_FFI_GPU_COLLECTIVES_H_
#define XLA_BACKENDS_GPU_FFI_GPU_COLLECTIVES_H_

#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/ffi/api/ffi_gpu_collectives.h"

namespace xla::gpu {

XLA_FFI_Gpu_Collectives_Extension MakeGpuCollectivesExtension(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* collective_clique_requests,
    const CollectiveCliques* collective_cliques);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_FFI_GPU_COLLECTIVES_H_
