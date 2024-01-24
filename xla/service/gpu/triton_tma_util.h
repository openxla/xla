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

#ifndef XLA_SERVICE_GPU_TRITON_TMA_UTIL_H_
#define XLA_SERVICE_GPU_TRITON_TMA_UTIL_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "xla/service/gpu/runtime3/tma_metadata.h"
#include "triton/Target/PTX/TmaMetadata.h"

namespace xla {
namespace gpu {
namespace triton_tma_util {

// Converts TMAMetadataTy to string.
std::string ToString(const mlir::triton::gpu::TMAMetadataTy& tma_metadata);

// Converts TMAMetadataTy to XLA's representation.
//
// Please don't pass an empty TMA metadata.
absl::StatusOr<std::unique_ptr<TmaMetadata>> ToTmaMetadata(
    const mlir::triton::gpu::TMAMetadataTy& tma_metadata);

}  // namespace triton_tma_util
}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRITON_TMA_UTIL_H_
