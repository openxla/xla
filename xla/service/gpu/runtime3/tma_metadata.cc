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

#include "xla/service/gpu/runtime3/tma_metadata.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace xla {
namespace gpu {

#if GOOGLE_CUDA

std::string CudaTensorMapInfo::ToString() const {
  return absl::StrCat(
      "{tensor_data_type:", tensor_data_type, ",tensor_rank:", tensor_rank,
      ",global_address_arg_index:", global_address_arg_index, ",global_dim:[",
      absl::StrJoin(global_dim, ","), "],global_strides:[",
      absl::StrJoin(global_strides, ","), "],box_dim:[",
      absl::StrJoin(box_dim, ","), "],element_strides:[",
      absl::StrJoin(element_strides, ","), "],interleave:", interleave,
      ",swizzle:", swizzle, ",l2_promotion:", l2_promotion,
      ",oob_fill:", oob_fill, "}");
}

CudaTmaMetadata::CudaTmaMetadata(
    std::vector<CudaTensorMapInfo> tensor_map_infos)
    : tensor_map_infos(std::move(tensor_map_infos)) {}

CudaTmaMetadata::~CudaTmaMetadata() = default;

std::unique_ptr<TmaMetadata> CudaTmaMetadata::Clone() const {
  return std::make_unique<CudaTmaMetadata>(tensor_map_infos);
}

std::string CudaTmaMetadata::ToString() const {
  return absl::StrCat(
      "[",
      absl::StrJoin(tensor_map_infos, ",\n ",
                    [&](std::string* s, const CudaTensorMapInfo& info) {
                      absl::StrAppend(s, info.ToString());
                    }),
      "]");
}

#endif  // GOOGLE_CUDA

}  // namespace gpu
}  // namespace xla
