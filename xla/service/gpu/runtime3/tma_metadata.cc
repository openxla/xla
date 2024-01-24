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
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_pimpl.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_driver.h"
#endif

namespace xla {
namespace gpu {

namespace se = ::stream_executor;  // NOLINT

#if GOOGLE_CUDA

namespace {

#define RETURN_IF_CUDA_RES_ERROR(expr, ...)                            \
  do {                                                                 \
    CUresult _res = (expr);                                            \
    if (ABSL_PREDICT_FALSE(_res != CUDA_SUCCESS)) {                    \
      return absl::InternalError(absl::StrCat(                         \
          __VA_ARGS__, ": ", ::stream_executor::gpu::ToString(_res))); \
    }                                                                  \
  } while (0)

absl::Status CreateCudaTensorMap(const ConcreteCudaTensorMapInfo& info,
                                 CUtensorMap& tensor_map) {
  VLOG(1) << "CreateCudaTensorMap: " << info.ToString();

  RETURN_IF_CUDA_RES_ERROR(
      cuTensorMapEncodeTiled(&tensor_map, info.tensor_data_type,
                             info.tensor_rank, info.global_address,
                             info.global_dim.data(), info.global_strides.data(),
                             info.box_dim.data(), info.element_strides.data(),
                             info.interleave, info.swizzle, info.l2_promotion,
                             info.oob_fill),
      "Failed to encode TensorMap.");

  return absl::OkStatus();
}

std::string PointerToString(void* pointer) {
  std::stringstream stream;
  stream << pointer;
  return stream.str();
}

}  // namespace

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

ConcreteCudaTensorMapInfo::ConcreteCudaTensorMapInfo() = default;
ConcreteCudaTensorMapInfo::ConcreteCudaTensorMapInfo(CudaTensorMapInfo info,
                                                     void* global_address)
    : tensor_data_type(info.tensor_data_type),
      tensor_rank(info.tensor_rank),
      global_address(global_address),
      global_dim(std::move(info.global_dim)),
      global_strides(std::move(info.global_strides)),
      box_dim(std::move(info.box_dim)),
      element_strides(std::move(info.element_strides)),
      interleave(info.interleave),
      swizzle(info.swizzle),
      l2_promotion(info.l2_promotion),
      oob_fill(info.oob_fill) {}

std::string ConcreteCudaTensorMapInfo::ToString() const {
  return absl::StrCat(
      "{tensor_data_type:", tensor_data_type, ",tensor_rank:", tensor_rank,
      ",global_address:", PointerToString(global_address), ",global_dim:[",
      absl::StrJoin(global_dim, ","), "],global_strides:[",
      absl::StrJoin(global_strides, ","), "],box_dim:[",
      absl::StrJoin(box_dim, ","), "],element_strides:[",
      absl::StrJoin(element_strides, ","), "],interleave:", interleave,
      ",swizzle:", swizzle, ",l2_promotion:", l2_promotion,
      ",oob_fill:", oob_fill, "}");
}

bool ConcreteCudaTensorMapInfo::operator==(
    const ConcreteCudaTensorMapInfo& other) const {
  return tensor_data_type == other.tensor_data_type &&
         tensor_rank == other.tensor_rank &&
         global_address == other.global_address &&
         global_dim == other.global_dim &&
         global_strides == other.global_strides && box_dim == other.box_dim &&
         element_strides == other.element_strides &&
         interleave == other.interleave && swizzle == other.swizzle &&
         l2_promotion == other.l2_promotion && oob_fill == other.oob_fill;
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

CudaTensorMapManager::CudaTensorMapManager() = default;

/*static*/ CudaTensorMapManager& CudaTensorMapManager::GetInstance() {
  static CudaTensorMapManager* instance = new CudaTensorMapManager();
  return *instance;
}

absl::StatusOr<se::DeviceMemoryBase>
CudaTensorMapManager::GetOrCreateDeviceTensorMap(ConcreteCudaTensorMapInfo info,
                                                 se::Stream& stream) {
  absl::MutexLock lock(&mu_);

  se::StreamExecutor& executor = *stream.parent();
  se::StreamExecutorMemoryAllocator& allocator = *executor.GetAllocator();

  std::pair<ConcreteCudaTensorMapInfo, int> key = {info,
                                                   executor.device_ordinal()};
  auto it = tensor_maps_.find(key);
  if (it == tensor_maps_.end()) {
    CUtensorMap host_tensor_map;
    TF_RETURN_IF_ERROR(CreateCudaTensorMap(info, host_tensor_map));
    TF_ASSIGN_OR_RETURN(
        se::OwningDeviceMemory device_tensor_map,
        allocator.Allocate(executor.device_ordinal(), sizeof(host_tensor_map)));
    stream.ThenMemcpy(device_tensor_map.ptr(), &host_tensor_map,
                      sizeof(host_tensor_map));
    // TODO(tdanyluk): This can be probably done with less blocking.
    TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());

    it = tensor_maps_.insert({std::move(key), std::move(device_tensor_map)})
             .first;
  }

  return *it->second.ptr();
}

#endif  // GOOGLE_CUDA

}  // namespace gpu
}  // namespace xla
