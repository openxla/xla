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

#include "xla/service/gpu/triton_tma_util.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/service/gpu/runtime3/tma_metadata.h"
#include "xla/status_macros.h"
#include "tsl/platform/statusor.h"
#include "triton/Target/PTX/TmaMetadata.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif

namespace xla {
namespace gpu {
namespace triton_tma_util {

namespace {

std::string ToString(const mlir::triton::gpu::TMAInfo& tma_info) {
  return absl::StrCat(
      "{tensorDataType:", tma_info.tensorDataType,
      ",tensorRank:", tma_info.tensorRank,
      ",globalAddressArgIdx:", tma_info.globalAddressArgIdx,
      ",globalDimsArgIdx:[", absl::StrJoin(tma_info.globalDimsArgIdx, ","),
      "],globalStridesArgIdx:[",
      absl::StrJoin(tma_info.globalStridesArgIdx, ","), "],boxDims:[",
      absl::StrJoin(tma_info.boxDims, ","), "],elementStrides:[",
      absl::StrJoin(tma_info.elementStrides, ","),
      "],interleave:", tma_info.interleave, ",swizzle:", tma_info.swizzle,
      ",l2_promotion:", tma_info.l2Promotion, ",oob_fill:", tma_info.oobFill,
      ",TMADescArgIdx:", tma_info.TMADescArgIdx, "}");
}

#if GOOGLE_CUDA
absl::StatusOr<int64_t> GetSizeInBytes(CUtensorMapDataType type) {
  switch (type) {
    case CU_TENSOR_MAP_DATA_TYPE_UINT8:
      return 1;
    case CU_TENSOR_MAP_DATA_TYPE_UINT16:
      return 2;
    case CU_TENSOR_MAP_DATA_TYPE_UINT32:
      return 4;
    case CU_TENSOR_MAP_DATA_TYPE_INT32:
      return 4;
    case CU_TENSOR_MAP_DATA_TYPE_UINT64:
      return 8;
    case CU_TENSOR_MAP_DATA_TYPE_INT64:
      return 8;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
      return 2;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
      return 4;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT64:
      return 8;
    case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
      return 2;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:
      return 4;
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:
      return 4;
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ:
      return 4;
    default:
      TF_RET_CHECK(false) << "Unsupported tensor data type: "
                          << static_cast<int>(type);
  }
}

absl::StatusOr<CudaTensorMapInfo> ToCudaTensorMapInfo(
    const mlir::triton::gpu::TMAInfo& tma_info) {
  // Code origin: triton/compiler/utils.py "def tensormap"

  CudaTensorMapInfo result;
  result.tensor_data_type =
      static_cast<CUtensorMapDataType>(tma_info.tensorDataType);
  result.tensor_rank = tma_info.tensorRank;
  result.global_address_arg_index = tma_info.globalAddressArgIdx;

  for (int32_t arg_index : tma_info.globalDimsArgIdx) {
    uint64_t size = 0;
    if (arg_index == -1) {
      size = 1;
    } else if (arg_index < 0 && arg_index != -1) {
      size = -arg_index - 1;
    } else {
      TF_RET_CHECK(false) << "Currently shapes should not be kernel arguments.";
    }
    result.global_dim.push_back(size);
  }

  // Skip dim_index=0.
  for (uint32_t dim_index = 1; dim_index < tma_info.tensorRank; dim_index++) {
    uint64_t stride = 0;
    int32_t arg_index = tma_info.globalStridesArgIdx.at(dim_index);
    if (arg_index == -1) {
      stride = 1;
      for (uint32_t dim_index2 = 0; dim_index2 < dim_index; dim_index2++) {
        stride *= result.global_dim.at(dim_index2);
      }
    } else if (arg_index < 0) {
      stride = -1 - arg_index;
    } else {
      TF_RET_CHECK(false)
          << "Currently strides should not be kernel arguments.";
    }
    TF_ASSIGN_OR_RETURN(int element_size,
                        GetSizeInBytes(static_cast<CUtensorMapDataType>(
                            result.tensor_data_type)));
    result.global_strides.push_back(stride * element_size);
  }

  result.box_dim.insert(result.box_dim.begin(), tma_info.boxDims.begin(),
                        tma_info.boxDims.end());
  result.element_strides.insert(result.element_strides.begin(),
                                tma_info.elementStrides.begin(),
                                tma_info.elementStrides.end());

  result.interleave = static_cast<CUtensorMapInterleave>(tma_info.interleave);
  result.swizzle = static_cast<CUtensorMapSwizzle>(tma_info.swizzle);
  result.l2_promotion =
      static_cast<CUtensorMapL2promotion>(tma_info.l2Promotion);
  result.oob_fill = static_cast<CUtensorMapFloatOOBfill>(tma_info.oobFill);

  return result;
}

#endif  // GOOGLE_CUDA

}  // namespace

std::string ToString(const mlir::triton::gpu::TMAMetadataTy& tma_metadata) {
  return absl::StrCat(
      "[",
      absl::StrJoin(
          tma_metadata, ",\n ",
          [&](std::string* s, const mlir::triton::gpu::TMAInfo& info) {
            absl::StrAppend(s, ToString(info));
          }),
      "]");
}

absl::StatusOr<std::unique_ptr<TmaMetadata>> ToTmaMetadata(
    const mlir::triton::gpu::TMAMetadataTy& tma_metadata) {
#ifdef GOOGLE_CUDA
  TF_RET_CHECK(!tma_metadata.empty());
  TF_RET_CHECK(std::is_sorted(tma_metadata.begin(), tma_metadata.end(),
                              [](const mlir::triton::gpu::TMAInfo& a,
                                 const mlir::triton::gpu::TMAInfo& b) {
                                return a.TMADescArgIdx < b.TMADescArgIdx;
                              }));
  std::vector<CudaTensorMapInfo> infos;
  infos.reserve(tma_metadata.size());
  for (const mlir::triton::gpu::TMAInfo& tma_info : tma_metadata) {
    TF_ASSIGN_OR_RETURN(CudaTensorMapInfo info, ToCudaTensorMapInfo(tma_info));
    infos.push_back(info);
  }

  return std::make_unique<CudaTmaMetadata>(std::move(infos));
#else   // GOOGLE_CUDA
  return absl::UnimplementedError("Not implemented.");
#endif  // GOOGLE_CUDA
}

}  // namespace triton_tma_util
}  // namespace gpu
}  // namespace xla
