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

#include "xla/service/buffer_allocation.h"

#include <cstdint>
#include <ostream>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "tsl/platform/numbers.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape_util.h"

namespace xla {

std::string BufferAllocation::Slice::ToString() const {
  return absl::StrCat("{index:", allocation_ == nullptr ? -1 : index(),
                      ", offset:", offset_, ", size:", size_, "}");
}

absl::StatusOr<xla::buffer_assignment::BufferAllocationSliceProto>
BufferAllocation::Slice::ToProto() const {
  xla::buffer_assignment::BufferAllocationSliceProto proto;
  proto.set_offset(offset());
  proto.set_size(size());
  proto.set_buffer_allocation_index(allocation() == nullptr ? -1 : index());
  proto.set_element_type(element_type());
  return proto;
}

absl::StatusOr<BufferAllocation::Slice> BufferAllocation::Slice::FromProto(
    const xla::buffer_assignment::BufferAllocationSliceProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  if (proto.buffer_allocation_index() < 0 ||
      proto.buffer_allocation_index() >= buffer_allocations.size()) {
    return absl::OutOfRangeError(absl::StrCat("Buffer allocation index ",
                                              proto.buffer_allocation_index(),
                                              " is out of range."));
  }
  const BufferAllocation& allocation =
      buffer_allocations[proto.buffer_allocation_index()];
  if (proto.offset() < 0 || proto.size() < 0) {
    return absl::OutOfRangeError(absl::StrCat(
        "Buffer slice has negative offset/size: offset=", proto.offset(),
        " size=", proto.size()));
  }
  if (proto.size() > allocation.size() ||
      proto.offset() > allocation.size() - proto.size()) {
    return absl::OutOfRangeError(absl::StrCat(
        "Buffer slice [offset=", proto.offset(), ", size=", proto.size(),
        "] is out of range for allocation #", proto.buffer_allocation_index(),
        " of size ", allocation.size()));
  }
  return BufferAllocation::Slice(&allocation, proto.offset(), proto.size(),
                                 proto.element_type());
}

BufferAllocationProto BufferAllocation::ToProto() const {
  BufferAllocationProto proto;
  proto.set_index(index_);
  proto.set_size(size_);
  proto.set_is_thread_local(is_thread_local_);
  proto.set_is_tuple(is_tuple_);
  proto.set_color(color_);
  if (is_entry_computation_parameter_) {
    proto.set_is_entry_computation_parameter(true);
    for (int64_t idx : param_shape_index()) {
      proto.add_parameter_shape_index(idx);
    }
    proto.set_parameter_number(parameter_number_);
    proto.set_is_parameter_aliased_with_output(
        is_parameter_aliased_with_output_);
  }
  proto.set_is_constant(is_constant_);
  proto.set_maybe_live_out(maybe_live_out_);
  return proto;
}

BufferAllocation BufferAllocation::FromProto(
    const BufferAllocationProto& proto) {
  BufferAllocation allocation{proto.index(), proto.size(), proto.color()};
  allocation.set_constant(proto.is_constant());
  allocation.set_is_thread_local(proto.is_thread_local());
  allocation.set_is_tuple(proto.is_tuple());
  if (proto.is_entry_computation_parameter()) {
    ShapeIndex shape_index{proto.parameter_shape_index().begin(),
                           proto.parameter_shape_index().end()};
    allocation.set_entry_computation_parameter(
        proto.parameter_number(), std::move(shape_index),
        proto.is_parameter_aliased_with_output());
  }
  allocation.set_maybe_live_out(proto.maybe_live_out());

  return allocation;
}

std::string BufferAllocation::ToShortString(bool human_readable_size) const {
  std::string output;
  if (human_readable_size) {
    absl::StrAppendFormat(&output, "allocation %d: size %s", index_,
                          tsl::strings::HumanReadableNumBytes(size()));
  } else {
    absl::StrAppendFormat(&output, "allocation %d: size %d", index_, size());
  }
  if (color() != 0) {
    absl::StrAppend(&output, ", color ", color());
  }
  if (page_id() != -1) {
    absl::StrAppend(&output, ", page ", page_id());
  }
  if (is_entry_computation_parameter()) {
    absl::StrAppend(&output, ", parameter ", parameter_number(),
                    " at ShapeIndex ", param_shape_index().ToString());
  }
  if (is_constant()) {
    absl::StrAppend(&output, ", constant");
  }
  if (is_thread_local()) {
    absl::StrAppend(&output, ", thread-local");
  }
  if (maybe_live_out()) {
    absl::StrAppend(&output, ", maybe-live-out");
  }
  if (IsPreallocatedTempBuffer()) {
    absl::StrAppend(&output, ", preallocated-temp");
  }
  absl::StrAppend(&output, ":\n");
  return output;
}

std::string BufferAllocation::ToString() const { return ToShortString(); }

std::ostream& operator<<(std::ostream& out, const BufferAllocation& buffer) {
  out << buffer.ToString();
  return out;
}

std::ostream& operator<<(std::ostream& out, const BufferAllocation::Slice& s) {
  out << s.ToString();
  return out;
}

}  // namespace xla
