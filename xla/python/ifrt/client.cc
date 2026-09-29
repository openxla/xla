/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/ifrt/client.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/pjrt/transpose.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/util.h"

namespace xla {
namespace ifrt {

char Client::ID = 0;

absl::StatusOr<ArrayRef> Client::MakeArrayFromHostChunkedArray(
    const ChunkedArray& chunked_array, ShardingRef sharding, LayoutRef layout,
    HostBufferSemantics semantics,
    std::function<void()> on_done_with_host_buffer) {
  absl::Span<const int64_t> dims = chunked_array.shape.dims();
  if (dims.empty()) {
    return absl::InvalidArgumentError(
        "ChunkedArray passed to MakeArrayFromHostChunkedArray must have rank "
        ">= 1");
  }
  if (static_cast<int64_t>(chunked_array.chunks.size()) != dims[0]) {
    return absl::InvalidArgumentError(
        absl::StrFormat("ChunkedArray chunks size (%d) must match dims[0] (%d)",
                        chunked_array.chunks.size(), dims[0]));
  }
  std::optional<int> bit_size = chunked_array.dtype.bit_size();
  if (!bit_size.has_value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "MakeArrayFromHostChunkedArray does not support dtype %v",
        chunked_array.dtype));
  }
  int64_t elem_size_in_bytes = CeilOfRatio<int64_t>(*bit_size, 8);
  int64_t num_elems = chunked_array.shape.num_elements();
  auto staging =
      std::make_shared<std::vector<uint8_t>>(num_elems * elem_size_in_bytes);
  if (!staging->empty()) {
    TransposePlan::Options options;
    options.elem_size_in_bytes = elem_size_in_bytes;
    options.dims = dims;
    absl::InlinedVector<int64_t, 4> permutation(dims.size());
    absl::c_iota(permutation, 0);
    options.permutation = permutation;
    if (chunked_array.byte_strides.has_value()) {
      options.input_striding =
          TransposePlan::Striding{*chunked_array.byte_strides};
    }
    options.input_dim0_is_chunked = true;
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<TransposePlan> plan,
                          TransposePlan::Create(options));
    plan->ExecuteChunked(chunked_array.chunks, staging->data());
  }
  if (on_done_with_host_buffer) {
    std::move(on_done_with_host_buffer)();
  }
  const void* staging_ptr = staging->data();
  return MakeArrayFromHostBuffer(
      staging_ptr, chunked_array.dtype, chunked_array.shape,
      /*byte_strides=*/std::nullopt, std::move(sharding), std::move(layout),
      HostBufferSemantics::kImmutableZeroCopy,
      [staging = std::move(staging)]() {});
}

absl::StatusOr<CustomLayoutRef> Client::GetDefaultLayout(
    DType dtype, absl::Span<const int64_t> shard_dims, Device* device,
    xla::ifrt::MemoryKind memory_kind) const {
  return GetDefaultLayout(dtype, Shape(shard_dims),
                          SingleDeviceSharding::Create(device, memory_kind));
}

}  // namespace ifrt
}  // namespace xla
