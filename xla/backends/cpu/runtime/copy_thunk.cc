/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/copy_thunk.h"

#include "absl/log/vlog_is_on.h"
#include "absl/status/status_macros.h"

#define EIGEN_USE_THREADS

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/pjrt/transpose.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {
namespace {

bool IsDenseUntiledByteArray(const Shape& shape) {
  if (!shape.IsArray() || !shape.has_layout()) {
    return false;
  }
  const auto& layout = shape.layout();
  const int64_t elem_bits = layout.element_size_in_bits();
  const int64_t byte_bits =
      ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type()) * 8;
  return layout.tiles().empty() &&
         layout.tail_padding_alignment_in_elements() == 1 &&
         layout.dynamic_shape_metadata_prefix_bytes() == 0 &&
         layout.minor_to_major().size() == shape.dimensions().size() &&
         (elem_bits == 0 || elem_bits == byte_bits) &&
         ShapeUtil::ByteStrides(shape).has_value();
}

bool RangesOverlap(const void* a, const void* b, size_t size_in_bytes) {
  if (size_in_bytes == 0) {
    return false;
  }
  const uintptr_t ua = reinterpret_cast<uintptr_t>(a);
  const uintptr_t ub = reinterpret_cast<uintptr_t>(b);
  return (ua < ub ? (ub - ua) : (ua - ub)) < size_in_bytes;
}

}  // namespace

absl::StatusOr<std::unique_ptr<CopyThunk>> CopyThunk::Create(
    Info info, BufferAllocation::Slice src_buffer, const Shape& src_shape,
    BufferAllocation::Slice dst_buffer, const Shape& dst_shape) {
  if (!src_shape.IsArray() || !dst_shape.IsArray() || !src_shape.has_layout() ||
      !dst_shape.has_layout() || src_shape.is_unbounded_dynamic() ||
      dst_shape.is_unbounded_dynamic()) {
    return InvalidArgument(
        "CopyThunk requires bounded array shapes with layouts, got %s and %s",
        src_shape.ToString(true), dst_shape.ToString(true));
  }
  ABSL_RETURN_IF_ERROR(ShapeUtil::ValidateShape(src_shape));
  ABSL_RETURN_IF_ERROR(ShapeUtil::ValidateShape(dst_shape));
  if (!ShapeUtil::Compatible(src_shape, dst_shape)) {
    return InvalidArgument(
        "Source shape %s must be compatible with destination shape %s",
        src_shape.ToString(true), dst_shape.ToString(true));
  }
  std::unique_ptr<TransposePlan> transpose_plan;
  if (src_shape != dst_shape) {
    if (!IsDenseUntiledByteArray(src_shape) ||
        !IsDenseUntiledByteArray(dst_shape)) {
      return InvalidArgument(
          "Layout-changing copy requires dense untiled byte layouts, got %s -> "
          "%s",
          src_shape.ToString(true), dst_shape.ToString(true));
    }
    auto byte_strides = ShapeUtil::ByteStrides(src_shape);
    if (!byte_strides.has_value()) {
      return InvalidArgument("Unsupported source layout in CopyThunk: %s",
                             src_shape.ToString(true));
    }
    TransposePlan::Options options;
    options.elem_size_in_bytes =
        ShapeUtil::ByteSizeOfPrimitiveType(src_shape.element_type());
    options.dims = src_shape.dimensions();
    options.input_striding = TransposePlan::Striding{*byte_strides};
    absl::InlinedVector<int64_t, 4> permutation(options.dims.size());
    absl::c_reverse_copy(dst_shape.layout().minor_to_major(),
                         permutation.begin());
    options.permutation = permutation;
    ABSL_ASSIGN_OR_RETURN(transpose_plan, TransposePlan::Create(options));
  }
  const int64_t src_bytes = ShapeUtil::ByteSizeOf(src_shape);
  const int64_t dst_bytes = ShapeUtil::ByteSizeOf(dst_shape);
  if (src_bytes != dst_bytes || src_buffer.size() < src_bytes ||
      dst_buffer.size() < dst_bytes) {
    return InvalidArgument(
        "CopyThunk buffer slices (%d, %d) incompatible with shape byte sizes "
        "(%d, %d)",
        src_buffer.size(), dst_buffer.size(), src_bytes, dst_bytes);
  }
  return absl::WrapUnique(new CopyThunk(std::move(info), src_buffer, src_shape,
                                        dst_buffer, dst_shape,
                                        std::move(transpose_plan)));
}

CopyThunk::CopyThunk(Info info, BufferAllocation::Slice src_buffer,
                     const Shape& src_shape, BufferAllocation::Slice dst_buffer,
                     const Shape& dst_shape,
                     std::unique_ptr<TransposePlan> transpose_plan)
    : Thunk(Kind::kCopy, std::move(info)),
      src_buffer_(src_buffer),
      src_shape_(src_shape),
      dst_buffer_(dst_buffer),
      dst_shape_(dst_shape),
      parallel_block_params_(ComputeParallelBlockParams(src_shape_)),
      transpose_plan_(std::move(transpose_plan)) {}

static std::tuple<void*, void*, int64_t> GetBlockCopyParameters(
    const CopyThunk::ParallelBlockParams& params, int64_t block_index,
    se::DeviceAddressBase destination, se::DeviceAddressBase source) {
  CHECK_LT(block_index, params.block_count);

  int64_t offset = block_index * params.block_size;
  CHECK_LT(offset, params.size_in_bytes);

  int64_t size = std::min(params.block_size, params.size_in_bytes - offset);
  CHECK_LE(size, params.block_size);

  std::byte* dst = reinterpret_cast<std::byte*>(destination.opaque());
  std::byte* src = reinterpret_cast<std::byte*>(source.opaque());

  return {dst + offset, src + offset, size};
}

CopyThunk::ParallelBlockParams CopyThunk::ComputeParallelBlockParams(
    const Shape& shape) {
  // Prefer single-threaded memcpy for small copies.
  static constexpr int64_t kMinParallelCopySize = 1024 * 1024;
  // Make block size a multiple of 1024 to match AVX2/AVX512 vector sizes.
  static constexpr int64_t kBlockSizeAlign = 1024;
  // Do not run more than 8 parallel copy blocks at a time.
  static constexpr int64_t kMaxParallelCopyBlocks = 8;

  int64_t size_in_bytes = ShapeUtil::ByteSizeOf(shape);
  if (size_in_bytes == 0) {
    return ParallelBlockParams{0, 0, 0};
  }

  int64_t num_blocks = std::min(
      kMaxParallelCopyBlocks, CeilOfRatio(size_in_bytes, kMinParallelCopySize));

  if (num_blocks == 1) {
    return ParallelBlockParams{size_in_bytes, size_in_bytes, 1};
  }

  int64_t block_size = RoundUpTo(size_in_bytes / num_blocks, kBlockSizeAlign);
  return ParallelBlockParams{size_in_bytes, block_size,
                             CeilOfRatio(size_in_bytes, block_size)};
}

tsl::AsyncValueRef<Thunk::ExecuteEvent> CopyThunk::Execute(
    const ExecuteParams& params) {
  const BufferAllocations* allocations = params.buffer_allocations;

  se::DeviceAddressBase src_data;
  se::DeviceAddressBase dst_data;

  if constexpr (ShouldCheckBufferSlices()) {
    ABSL_ASSIGN_OR_RETURN(src_data, allocations->GetDeviceAddress(src_buffer_));
    ABSL_ASSIGN_OR_RETURN(dst_data, allocations->GetDeviceAddress(dst_buffer_));
  } else {
    src_data = allocations->GetDeviceAddressUnchecked(src_buffer_);
    dst_data = allocations->GetDeviceAddressUnchecked(dst_buffer_);
  }

  // TODO(ezhulenev): Add extra checks for buffer aliasing, as we rely on the
  // fact that buffers do not alias each other to run copy in parallel.

  // ToString operands to operator<< allocate even when VLOG is off.
  if (ABSL_PREDICT_FALSE(VLOG_IS_ON(3))) {
    VLOG(3) << absl::StreamFormat("Copy buffer: use_transpose=%s",
                                  transpose_plan_ ? "true" : "false");
    VLOG(3) << absl::StreamFormat("  src: %s in slice %s (%p)",
                                  src_shape_.ToString(true),
                                  src_buffer_.ToString(), src_data.opaque());
    VLOG(3) << absl::StreamFormat("  dst: %s in slice %s (%p)",
                                  dst_shape_.ToString(true),
                                  dst_buffer_.ToString(), dst_data.opaque());
  }

  // Skip no-op copy operations.
  if (ABSL_PREDICT_FALSE(parallel_block_params_.block_count == 0)) {
    return OkExecuteEvent();
  }

  const size_t size_in_bytes =
      static_cast<size_t>(parallel_block_params_.size_in_bytes);
  const bool overlaps =
      RangesOverlap(src_data.opaque(), dst_data.opaque(), size_in_bytes);

  // Use prepared transpose plan to copy data if copy requires changing layout.
  if (ABSL_PREDICT_FALSE(transpose_plan_)) {
    if (ABSL_PREDICT_FALSE(overlaps)) {
      auto scratch = std::make_unique<std::byte[]>(size_in_bytes);
      transpose_plan_->Execute(src_data.opaque(), scratch.get(),
                               [](std::function<void()> fn) { fn(); });
      std::memcpy(dst_data.opaque(), scratch.get(), size_in_bytes);
      return OkExecuteEvent();
    }
    transpose_plan_->Execute(src_data.opaque(), dst_data.opaque(),
                             [](std::function<void()> fn) { fn(); });
    return OkExecuteEvent();
  }

  if (ABSL_PREDICT_FALSE(overlaps)) {
    if (src_data.opaque() != dst_data.opaque()) {
      std::memmove(dst_data.opaque(), src_data.opaque(), size_in_bytes);
    }
    return OkExecuteEvent();
  }

  // For a single block, use std::memcpy to copy data from source to
  // destination.
  if (ABSL_PREDICT_TRUE(params.intra_op_threadpool == nullptr ||
                        parallel_block_params_.block_count == 1)) {
    std::memcpy(dst_data.opaque(), src_data.opaque(),
                parallel_block_params_.size_in_bytes);
    return OkExecuteEvent();
  }

  // Use intra-op thread pool to run copy operation in parallel.
  auto event = tsl::MakeConstructedAsyncValueRef<ExecuteEvent>();
  auto counter = std::make_shared<std::atomic<int64_t>>(
      parallel_block_params_.block_count);

  // Executes copy operation for a single block.
  auto execute = [this, event, counter, dst_data,
                  src_data](int64_t block_index) {
    auto [dst, src, size] = GetBlockCopyParameters(
        parallel_block_params_, block_index, dst_data, src_data);

    std::memcpy(dst, src, size);

    if (counter->load() == 1 || counter->fetch_sub(1) == 1) {
      event.SetStateConcrete();
    }
  };

  // Launch parallel copy operations in the intra-op thread pool.
  for (int64_t i = 1; i < parallel_block_params_.block_count; ++i) {
    params.intra_op_threadpool->getPool()->Schedule(
        [i, execute] { execute(i); });
  }

  // Execute the first memory copy task in the caller thread.
  execute(0);

  return event;
}

}  // namespace xla::cpu
