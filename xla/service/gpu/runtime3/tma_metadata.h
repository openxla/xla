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

#ifndef XLA_SERVICE_GPU_RUNTIME3_TMA_METADATA_H_
#define XLA_SERVICE_GPU_RUNTIME3_TMA_METADATA_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif  // GOOGLE_CUDA

namespace xla {
namespace gpu {

// TmaMetadata interface.
//
// This allows passing around TmaMetadata without depending on a specific GPU
// library (such as CUDA).
struct TmaMetadata {
  TmaMetadata() = default;
  TmaMetadata(const TmaMetadata&) = delete;
  TmaMetadata& operator=(const TmaMetadata&) = delete;
  virtual ~TmaMetadata() = default;

  virtual std::unique_ptr<TmaMetadata> Clone() const = 0;
  virtual std::string ToString() const = 0;
};

#if GOOGLE_CUDA

// Information describing a CUDA tensor map - to be used with
// cuTensorMapEncodeTiled.
struct CudaTensorMapInfo {
  CUtensorMapDataType tensor_data_type;
  uint32_t tensor_rank;
  // The index of the kernel argument used for the pointer of the tensor.
  int global_address_arg_index;
  absl::InlinedVector<uint64_t, 4> global_dim;
  absl::InlinedVector<uint64_t, 4> global_strides;
  absl::InlinedVector<uint32_t, 4> box_dim;
  absl::InlinedVector<uint32_t, 4> element_strides;
  CUtensorMapInterleave interleave;
  CUtensorMapSwizzle swizzle;
  CUtensorMapL2promotion l2_promotion;
  CUtensorMapFloatOOBfill oob_fill;

  std::string ToString() const;
};

// CUDA-specific TmaMetadata.
struct CudaTmaMetadata : public TmaMetadata {
  explicit CudaTmaMetadata(std::vector<CudaTensorMapInfo> tensor_map_infos);
  ~CudaTmaMetadata() override;

  std::unique_ptr<TmaMetadata> Clone() const override;
  std::string ToString() const override;

  std::vector<CudaTensorMapInfo> tensor_map_infos;
};

#endif  // GOOGLE_CUDA

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME3_TMA_METADATA_H_
