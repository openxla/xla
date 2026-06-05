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

#include "xla/backends/gpu/runtime/dynamic_memcpy_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/copy_thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/while_loop.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/shaped_slice.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

DynamicMemcpyThunk::DynamicMemcpyThunk(ThunkInfo thunk_info,
                                       const ShapedSlice& source_buffer,
                                       const ShapedSlice& destination_buffer,
                                       uint64_t mem_size,
                                       DynamicMemcpyThunk::Offsets offsets)
    : Command(Kind::kCopy, std::move(thunk_info)),
      source_buffer_(source_buffer),
      destination_buffer_(destination_buffer),
      mem_size_(mem_size),
      offsets_(std::move(offsets)) {}

absl::StatusOr<int64_t> DynamicMemcpyThunk::GetOffsetIndex(
    const RecordParams* record_params) const {
  if (!offsets_.depends_on_loop) {
    return 0;
  }

  const WhileLoopState* state = IsInsideWhileLoop();
  if (state != nullptr) {
    return state->loop_iteration;
  }

  if (record_params != nullptr && record_params->is_initialization) {
    return 0;
  }

  return absl::FailedPreconditionError(
      "DynamicMemcpyThunk depends on loop, but no while-loop command recording "
      "context is active");
}

absl::StatusOr<DynamicMemcpyThunk::CopyAddresses>
DynamicMemcpyThunk::GetCopyAddresses(const BufferAllocations& allocations,
                                     const RecordParams* record_params) const {
  ASSIGN_OR_RETURN(int64_t offset_index, GetOffsetIndex(record_params));
  int64_t src_offset_count = static_cast<int64_t>(offsets_.src_offsets.size());
  int64_t dst_offset_count = static_cast<int64_t>(offsets_.dst_offsets.size());
  TF_RET_CHECK(offset_index >= 0);
  TF_RET_CHECK(offset_index < src_offset_count)
      << "DynamicMemcpyThunk source offset index is out of range";
  TF_RET_CHECK(offset_index < dst_offset_count)
      << "DynamicMemcpyThunk destination offset index is out of range";

  se::DeviceAddressBase src_data =
      allocations.GetDeviceAddress(source_buffer_.slice);
  se::DeviceAddressBase dst_data =
      allocations.GetDeviceAddress(destination_buffer_.slice);

  int64_t src_offset = offsets_.src_offsets[offset_index];
  int64_t dst_offset = offsets_.dst_offsets[offset_index];

  auto src_with_offset = src_data.GetByteSlice(src_offset, mem_size_);
  auto dst_with_offset = dst_data.GetByteSlice(dst_offset, mem_size_);

  VLOG(3) << "Memcpy of size " << mem_size_ << " from "
          << src_with_offset.opaque() << " (offset " << src_offset << ") to "
          << dst_with_offset.opaque() << " (offset " << dst_offset << ")";
  return CopyAddresses{dst_with_offset, src_with_offset};
}

absl::Status DynamicMemcpyThunk::ExecuteOnStream(const ExecuteParams& params) {
  ASSIGN_OR_RETURN(CopyAddresses addresses,
                   GetCopyAddresses(*params.buffer_allocations));
  return params.stream->Memcpy(&addresses.dst, addresses.src, mem_size_);
}

absl::StatusOr<const se::CommandBuffer::Command*> DynamicMemcpyThunk::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  if (mem_size_ == 0) {
    VLOG(5) << "Skip recording DynamicMemcpyThunk command of 0 bytes";
    return nullptr;
  }

  ASSIGN_OR_RETURN(
      CopyAddresses addresses,
      GetCopyAddresses(*execute_params.buffer_allocations, &record_params));

  if (auto* create = std::get_if<RecordCreate>(&record_action)) {
    return command_buffer->CreateMemcpyD2D(&addresses.dst, addresses.src,
                                           mem_size_, create->dependencies);
  }

  if (auto* update = std::get_if<RecordUpdate>(&record_action)) {
    RETURN_IF_ERROR(command_buffer->UpdateMemcpyD2D(
        update->command, &addresses.dst, addresses.src, mem_size_));
    return update->command;
  }

  return absl::InternalError("Invalid record action");
}

DynamicMemcpyThunkProto::Offsets DynamicMemcpyThunk::Offsets::ToProto() const {
  DynamicMemcpyThunkProto::Offsets proto;
  proto.set_depends_on_loop(depends_on_loop);
  proto.mutable_src_offsets()->Add(src_offsets.begin(), src_offsets.end());
  proto.mutable_dst_offsets()->Add(dst_offsets.begin(), dst_offsets.end());
  return proto;
}

absl::StatusOr<DynamicMemcpyThunk::Offsets>
DynamicMemcpyThunk::Offsets::FromProto(
    const DynamicMemcpyThunkProto::Offsets& proto) {
  Offsets offsets;
  offsets.depends_on_loop = proto.depends_on_loop();
  offsets.src_offsets = {proto.src_offsets().begin(),
                         proto.src_offsets().end()};
  offsets.dst_offsets = {proto.dst_offsets().begin(),
                         proto.dst_offsets().end()};
  return offsets;
}

absl::StatusOr<ThunkProto> DynamicMemcpyThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  DynamicMemcpyThunkProto* dynamic_memcpy_thunk_proto =
      proto.mutable_dynamic_memcpy_thunk();
  ASSIGN_OR_RETURN(*dynamic_memcpy_thunk_proto->mutable_source_buffer(),
                   source_buffer_.ToProto());
  ASSIGN_OR_RETURN(*dynamic_memcpy_thunk_proto->mutable_destination_buffer(),
                   destination_buffer_.ToProto());
  dynamic_memcpy_thunk_proto->set_mem_size(mem_size_);
  *dynamic_memcpy_thunk_proto->mutable_offsets() = offsets_.ToProto();
  return proto;
}

absl::StatusOr<std::unique_ptr<DynamicMemcpyThunk>>
DynamicMemcpyThunk::FromProto(
    ThunkInfo thunk_info, const DynamicMemcpyThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  ASSIGN_OR_RETURN(
      ShapedSlice src_slice,
      ShapedSlice::FromProto(thunk_proto.source_buffer(), buffer_allocations));
  ASSIGN_OR_RETURN(ShapedSlice dst_slice,
                   ShapedSlice::FromProto(thunk_proto.destination_buffer(),
                                          buffer_allocations));
  ASSIGN_OR_RETURN(Offsets offsets, Offsets::FromProto(thunk_proto.offsets()));
  return std::make_unique<DynamicMemcpyThunk>(std::move(thunk_info), src_slice,
                                              dst_slice, thunk_proto.mem_size(),
                                              std::move(offsets));
}

}  // namespace gpu
}  // namespace xla
