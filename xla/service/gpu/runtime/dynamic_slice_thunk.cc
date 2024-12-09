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

#include "xla/service/gpu/runtime/dynamic_slice_thunk.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/runtime/sequential_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/gpu/runtime/while_thunk.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

std::unique_ptr<Literal>& Indvar(DynamicSliceThunk* thunk) {
  static thread_local absl::flat_hash_map<DynamicSliceThunk*,
                                          std::unique_ptr<Literal>>
      indvar_map;
  return indvar_map[thunk];
}

DynamicSliceThunk::DynamicSliceThunk(
    ThunkInfo thunk_info, std::unique_ptr<ThunkSequence> embedded_thunk,
    std::vector<std::optional<BufferAllocation::Slice>> arguments,
    std::vector<std::unique_ptr<BufferAllocation>> fake_allocations,
    std::vector<std::optional<std::vector<Offset>>> offsets,
    std::vector<std::optional<Shape>> orig_shapes,
    std::vector<std::optional<Shape>> sliced_shapes,
    std::vector<std::optional<uint64_t>> offset_byte_sizes,
    std::vector<std::unique_ptr<HloModule>> fake_modules,
    std::unique_ptr<HloModule> indvar_init,
    std::unique_ptr<HloModule> indvar_update)
    : Thunk(Kind::kDynamicSlice, thunk_info),
      embedded_thunk_(std::make_unique<SequentialThunk>(
          ThunkInfo(), std::move(*embedded_thunk))),
      arguments_(arguments),
      fake_allocations_(std::move(fake_allocations)),
      offsets_(offsets),
      orig_shapes_(orig_shapes),
      sliced_shapes_(sliced_shapes),
      offset_byte_sizes_(offset_byte_sizes),
      fake_modules_(std::move(fake_modules)),
      indvar_init_(std::move(indvar_init)),
      indvar_update_(std::move(indvar_update)) {
  // Zip all arguments together to create a list of SliceDef.
  for (auto [arg, offsets, orig_shape, sliced_shape, offset_byte_size] :
       llvm::zip_equal(arguments, offsets, orig_shapes, sliced_shapes,
                       offset_byte_sizes)) {
    slices_.push_back(SliceDef{
        std::move(arg),
        std::move(offsets),
        std::move(orig_shape),
        std::move(sliced_shape),
        std::move(offset_byte_size),
    });
  }

  // Find how many offsets we might have to transfer from device to host and
  // pre-compute host allocation requirements.
  for (SliceDef& slice : slices_) {
    offsets_allocs_base_.push_back(offsets_allocs_size_);
    if (slice.sliced_shape.has_value()) {
      offsets_allocs_size_ += slice.sliced_shape->rank() * sizeof(int64_t);
    }
  }
}

absl::Status DynamicSliceThunk::Prepare(const PrepareParams& params,
                                        ResourceRequests& resource_requests) {
  for (SliceDef& slice : slices_) {
    if (slice.offsets.has_value()) {
      TF_RET_CHECK(slice.embedded_thunk_argument.has_value());
      TF_RET_CHECK(slice.orig_shape.has_value());
      TF_RET_CHECK(slice.sliced_shape.has_value());
      TF_RET_CHECK(slice.offset_byte_size.has_value());

      TF_RET_CHECK(slice.orig_shape->IsArray());
      TF_RET_CHECK(slice.sliced_shape->IsArray());

      TF_RET_CHECK(slice.offsets->size() == slice.orig_shape->rank());
      TF_RET_CHECK(slice.sliced_shape->rank() == slice.orig_shape->rank());
    }
  }

  TF_RETURN_IF_ERROR(embedded_thunk_->Prepare(params, resource_requests));
  if (indvar_init_ != nullptr) {
    Indvar(this) = HloEvaluator().Evaluate(*indvar_init_, {})->CloneToUnique();
  }

  return absl::OkStatus();
}

absl::Status DynamicSliceThunk::Initialize(const InitializeParams& params) {
  TF_RETURN_IF_ERROR(embedded_thunk_->Initialize(params));

  absl::MutexLock lock(&mutex_);
  if (offsets_allocs_.contains(params.executor)) return absl::OkStatus();

  VLOG(2) << "Allocate " << offsets_allocs_size_
          << " bytes for transferring offsets on executor: " << params.executor;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::MemoryAllocation> allocation,
      params.executor->HostMemoryAllocate(offsets_allocs_size_));
  offsets_allocs_.emplace(params.executor, std::move(allocation));

  return absl::OkStatus();
}

absl::Status DynamicSliceThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream& stream = *params.stream;
  const BufferAllocations& orig_allocations = *params.buffer_allocations;

  absl::InlinedVector<se::DeviceMemoryBase, 8> slice_buffers(
      slices_.size(), se::DeviceMemoryBase());

  // Get memory allocation for copying offsets from device.
  int64_t* offsets_alloc = [&] {
    absl::MutexLock lock(&mutex_);
    return reinterpret_cast<int64_t*>(
        offsets_allocs_.at(stream.parent())->opaque());
  }();

  auto offset_value = [&](int64_t arg_idx, int64_t offset_idx) -> int64_t& {
    return offsets_alloc[offsets_allocs_base_.at(arg_idx) + offset_idx];
  };

  VLOG(2) << "Execute address computation thunk: slices=" << slices_.size();
  for (auto [argument_idx, slice] : llvm::enumerate(slices_)) {
    // Skip arguments that do not have buffer slices (tokens).
    if (!slice.embedded_thunk_argument.has_value()) {
      continue;
    }

    // `argument_buffer` will contain the original offset for slice
    // `argument_slice` within `orig_allocations`
    se::DeviceMemoryBase argument_buffer =
        orig_allocations.GetDeviceAddress(*slice.embedded_thunk_argument);

    // If argument is not sliced, just use the original buffer.
    if (!slice.offsets.has_value()) {
      slice_buffers[argument_idx] = argument_buffer;
      continue;
    }

    const Shape& src_shape = *slice.orig_shape;
    const Shape& dst_shape = *slice.sliced_shape;

    absl::InlinedVector<int64_t, 4> slice_starts;
    slice_starts.reserve(dst_shape.rank());

    // Number of issues d2h transfers to copy offset values from device to
    // host.
    int64_t num_transfers = 0;

    // Get offset for `argument_idx`-th argument, which has `dst_shape.rank()`
    // components.
    for (auto [offset_idx, values] : llvm::enumerate(llvm::zip(
             *slice.offsets, src_shape.dimensions(), dst_shape.dimensions()))) {
      auto [offset, src_dim, dst_dim] = values;

      if (int64_t* const_offset = std::get_if<int64_t>(&offset)) {
        // Forward slice offsets that are known constant values
        VLOG(2) << "  - arg " << argument_idx << "[" << offset_idx
                << "]: constant offset = " << *const_offset;
        offset_value(argument_idx, offset_idx) = *const_offset;
      } else if (HloModule** offset_module = std::get_if<HloModule*>(&offset)) {
        VLOG(3) << "Offset module: " << (*offset_module)->ToString();
        TF_ASSIGN_OR_RETURN(
            Literal offset,
            HloEvaluator().Evaluate(**offset_module, {Indvar(this)->Clone()}));
        std::optional<int64_t> offset_literal =
            LiteralUtil::LiteralAsScalarInt64(offset);
        CHECK(offset_literal != std::nullopt)
            << "Offset value is expected to be integer. Found "
            << offset.ToString();
        offset_value(argument_idx, offset_idx) = *offset_literal;
        VLOG(2) << "  - arg " << argument_idx << "[" << offset_idx
                << "]: loop induction variable dependent offset = "
                << *offset_literal;
      } else {
        // Transfer slice offset value from device to host.
        auto alloc_slice = std::get<BufferAllocation::Slice>(offset);
        VLOG(2) << "  - arg " << argument_idx << "[" << offset_idx
                << "]: transfer offset from device " << alloc_slice.ToString();

        se::DeviceMemoryBase offset_src =
            orig_allocations.GetDeviceAddress(alloc_slice);
        int64_t* offset_dst = &offset_value(argument_idx, offset_idx);

        // Copy the `offset_idx`-th component of the offset for the
        // `argument_idx`-th argument from device to host.
        TF_RETURN_IF_ERROR(
            stream.Memcpy(offset_dst, offset_src, *slice.offset_byte_size));
        ++num_transfers;
      }
    }

    // Wait for the completion of all transfers.
    if (num_transfers > 0) {
      VLOG(2) << "Wait for completion of " << num_transfers << " transfer";
      TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());
    }

    // Clamp start indices:
    // start_indices[i] = min(max(start_indices[i], 0),
    //                        operand.dimension_size[i] - size_indices[i])
    for (auto [offset_idx, values] : llvm::enumerate(
             llvm::zip(src_shape.dimensions(), dst_shape.dimensions()))) {
      auto [src_dim, dst_dim] = values;
      int64_t start_index =
          std::min(std::max(offset_value(argument_idx, offset_idx), int64_t{0}),
                   src_dim - dst_dim);
      VLOG(2) << "arg idx: " << argument_idx << " offset_idx " << offset_idx
              << " with offset_value " << offset_value(argument_idx, offset_idx)
              << " start_idx: " << start_index << " src_dim: " << src_dim
              << " dst_dim:" << dst_dim;
      slice_starts.push_back(start_index);
    }

    // Compute new slice. No need to copy the content to new buffers as we can
    // reuse the original buffers since slices are contiguous.
    int64_t new_size = ShapeUtil::ByteSizeOf(dst_shape);

    int64_t new_offset = 0;
    for (auto [start, stride] :
         llvm::zip(slice_starts, *ShapeUtil::ByteStrides(src_shape))) {
      new_offset += start * stride;
    }

    VLOG(2) << "Create sliced argument " << argument_idx << " of shape "
            << slice.sliced_shape->ToString()
            << " by slicing argument of shape " << slice.orig_shape->ToString()
            << " at offset " << new_offset << " with " << new_size;
    slice_buffers[argument_idx] =
        argument_buffer.GetByteSlice(new_offset, new_size);
  }

  // Safe to create a local BufferAllocations here since buffers are only slices
  // of bigger ones allocated elsewhere.
  BufferAllocations slice_allocations(slice_buffers,
                                      orig_allocations.device_ordinal(),
                                      orig_allocations.memory_allocator());

  Thunk::ExecuteParams new_params =
      Thunk::ExecuteParams::CloneWithNewAllocations(params, slice_allocations);

  // Execute the underlying custom call thunk with the new buffers.
  TF_RETURN_IF_ERROR(embedded_thunk_->ExecuteOnStream(new_params));

  // Before ending, we need to update the induction variable.
  if (indvar_update_ != nullptr) {
    Indvar(this) = HloEvaluator()
                       .Evaluate(*indvar_update_, {Indvar(this)->Clone()})
                       ->CloneToUnique();
    VLOG(1) << "induction variable = " << Indvar(this)->ToString();
  }

  return absl::OkStatus();
}

void DynamicSliceThunk::ForAllThunks(
    absl::FunctionRef<void(const Thunk*)> fn) const {
  fn(this);
  embedded_thunk_->ForAllThunks(fn);
}

std::string DynamicSliceThunk::ToString(int indent) const {
  std::string arguments = absl::StrJoin(
      arguments_, ",",
      [](std::string* out, const std::optional<BufferAllocation::Slice>& arg) {
        out->append(arg.has_value() ? arg->ToString() : "std::nullopt");
      });
  std::string fake_allocations =
      absl::StrJoin(fake_allocations_, ",",
                    [](std::string* out,
                       const std::unique_ptr<BufferAllocation>& allocation) {
                      out->append(allocation->ToString());
                    });
  std::string offsets = absl::StrJoin(
      offsets_, ",",
      [](std::string* out, const std::optional<std::vector<Offset>>& offsets) {
        out->append(
            offsets.has_value()
                ? "{" +
                      absl::StrJoin(
                          offsets.value(), ",",
                          [](std::string* out, const Offset& offset) {
                            if (const int64_t* constant =
                                    std::get_if<int64_t>(&offset)) {
                              out->append(std::to_string(*constant));
                            } else if (const BufferAllocation::Slice* slice =
                                           std::get_if<BufferAllocation::Slice>(
                                               &offset)) {
                              out->append(slice->ToString());
                            } else if (const HloModule* const* module =
                                           std::get_if<HloModule*>(&offset)) {
                              out->append((*module)->ToString());
                            }
                          }) +
                      "}"
                : "std::nullopt");
      });
  std::string orig_shapes = absl::StrJoin(
      orig_shapes_, ",",
      [](std::string* out, const std::optional<Shape>& orig_shape) {
        out->append(orig_shape.has_value() ? orig_shape->ToString()
                                           : "std::nullopt");
      });
  std::string sliced_shapes = absl::StrJoin(
      sliced_shapes_, ",",
      [](std::string* out, const std::optional<Shape>& sliced_shape) {
        out->append(sliced_shape.has_value() ? sliced_shape->ToString()
                                             : "std::nullopt");
      });
  std::string offset_byte_sizes = absl::StrJoin(
      offset_byte_sizes_, ",",
      [](std::string* out, const std::optional<uint64_t>& offset_byte_size) {
        out->append(offset_byte_size.has_value()
                        ? std::to_string(offset_byte_size.value())
                        : "std::nullopt");
      });
  std::string indvar_init =
      indvar_init_ == nullptr ? "nullptr" : indvar_init_->ToString();
  std::string indvar_update =
      indvar_update_ == nullptr ? "nullptr" : indvar_update_->ToString();
  std::string fake_modules = absl::StrJoin(
      fake_modules_, ",",
      [](std::string* out, const std::unique_ptr<HloModule>& fake_module) {
        out->append("HloModule " + fake_module->name());
      });
  return absl::StrFormat(
      "{arguments={%s}, fake_allocations={%s}, offsets={%s}, orig_shapes={%s}, "
      "sliced_shapes={%s}, offset_byte_sizes={%s}, indvar_init={%s}, "
      "indvar_update={%s}, fake_offset_modules={%s}}",
      arguments, fake_allocations, offsets, orig_shapes, sliced_shapes,
      offset_byte_sizes, indvar_init, indvar_update, fake_modules);
}
}  // namespace gpu
}  // namespace xla
