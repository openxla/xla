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

#ifndef XLA_SERVICE_BUFFER_ALLOCATION_H_
#define XLA_SERVICE_BUFFER_ALLOCATION_H_

#include <cstdint>
#include <iosfwd>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Memory space color of a buffer or buffer allocation. Differently colored
// buffers cannot be parts of the same allocation.
using BufferColor = int64_t;

// This class abstracts an allocation of contiguous memory which can hold the
// values described by LogicalBuffers. Each LogicalBuffer occupies a subrange
// of the allocation, represented by a Slice. A single BufferAllocation may hold
// LogicalBuffers with disjoint liveness, which may have overlapping Slices. A
// single BufferAllocation may also hold LogicalBuffers with overlapping
// liveness, which must have disjoint Slices.
//
// The abstraction includes information required by the backends for allocation,
// use, and deallocation of the buffer.
class BufferAllocation {
 public:
  // Holds a unique identifier for each allocation. Values are assigned
  // contiguously and can be used as array indexes.
  using Index = int64_t;

  BufferAllocation(Index index, int64_t size, BufferColor color)
      : index_(index),
        size_(size),
        color_(color),
        page_id_(0),
        is_thread_local_(false),
        is_tuple_(false),
        is_entry_computation_parameter_(false),
        is_parameter_aliased_with_output_(false),
        maybe_live_out_(false),
        is_constant_(false) {}

  BufferAllocation(const BufferAllocation&) = default;
  BufferAllocation& operator=(const BufferAllocation&) = default;
  BufferAllocation(BufferAllocation&&) = default;
  BufferAllocation& operator=(BufferAllocation&&) = default;

  // Returns the index of this allocation.
  Index index() const { return index_; }

  // Whether this allocation is used in a parallel calling context such as
  // inside of a map or reduce computation. Such allocations need to be thread
  // local.
  bool is_thread_local() const { return is_thread_local_; }
  void set_is_thread_local(bool is_thread_local) {
    is_thread_local_ = is_thread_local;
  }

  // Whether this allocation can be used by more than one logical buffer.
  bool is_reusable() const {
    // We do not reuse thread-local buffers for now, because they are
    // dynamically allocated and their lifetimes are hard to compute.
    //
    // TODO(b/34669761): Don't reuse tuple buffers because the GPU backend
    // assumes longer buffer liveness than indicated by the analysis.
    return !is_thread_local() && !is_tuple();
  }

  // Whether this allocation is read-only i.e. backed by memory we cannot write
  // to.
  bool is_readonly() const {
    // Entry parameters are generally readonly, except when they are aliased
    // with any output.
    return (is_entry_computation_parameter() &&
            !is_parameter_aliased_with_output_) ||
           is_constant();
  }

  bool is_tuple() const { return is_tuple_; }
  void set_is_tuple(bool is_tuple) { is_tuple_ = is_tuple; }

  // Whether this allocation holds a LogicalBuffer from a parameter of the entry
  // computation. These buffers have lifetimes which may be longer than the
  // XLA computation.
  bool is_entry_computation_parameter() const {
    return is_entry_computation_parameter_;
  }

  bool is_parameter_aliased_with_output() const {
    return is_parameter_aliased_with_output_;
  }

  // Whether this allocation holds a constant.  On the CPU and GPU backends
  // constant allocations are not allocated dynamically, instead we resolve
  // references to these buffer allocations to a global in the readonly section
  // of the binary.
  bool is_constant() const { return is_constant_; }

  // If this allocation holds a Buffer from a parameter of the entry
  // computation, this methods returns the parameter number. CHECKs otherwise.
  int64_t parameter_number() const {
    CHECK(is_entry_computation_parameter_);
    return parameter_number_;
  }

  // If this allocation is for a parameter of the entry computation, this
  // function returns which subshape of the parameter the allocation is for.
  const ShapeIndex& param_shape_index() const {
    CHECK(is_entry_computation_parameter_);
    return param_shape_index_;
  }

  // Returns whether this allocation is assigned a LogicalBuffer which may
  // be live out of the entry computation.
  bool maybe_live_out() const { return maybe_live_out_; }

  void set_maybe_live_out(bool value) { maybe_live_out_ = value; }

  // Returns the size of the allocation. Necessarily this must be at least as
  // large as any LogicalBuffer assigned to this allocation.
  int64_t size() const { return size_; }

  // Returns the color of the allocation. Only logical buffers with a matching
  // color can reside in this allocation.
  BufferColor color() const { return color_; }

  void set_color(BufferColor color) { color_ = color; }
  struct OffsetSize {
    int64_t offset = 0;
    int64_t size = 0;
  };

  // Returns the page id of the allocation. This is only used for multi-page
  // buffer assignment.
  int64_t page_id() const { return page_id_; }

  // Sets the page id of the allocation. This is only used for multi-page
  // buffer assignment.
  void set_page_id(int64_t page) { page_id_ = page; }

  // A Slice represents a contiguous portion of a memory allocation. It is used
  // to identify the memory range that a LogicalBuffer corresponds to.
  class Slice {
   public:
    Slice() = default;
    Slice(const BufferAllocation* allocation, int64_t offset, int64_t size,
          PrimitiveType element_type = PrimitiveType::PRIMITIVE_TYPE_INVALID)
        : allocation_(allocation),
          offset_(offset),
          size_(size),
          element_type_(element_type) {}

    Slice(const Slice&) = default;
    Slice& operator=(const Slice&) = default;
    Slice(Slice&&) = default;
    Slice& operator=(Slice&&) = default;

    const BufferAllocation* allocation() const { return allocation_; }
    Index index() const { return allocation_->index(); }
    int64_t offset() const { return offset_; }
    int64_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    PrimitiveType element_type() const { return element_type_; }

    bool operator==(const Slice& other) const {
      if (allocation_ == nullptr || other.allocation_ == nullptr) {
        return other.allocation_ == allocation_;
      }
      // We don't compare element_type_ because it's not always set, and it's
      // not relevant for the comparison here.
      return index() == other.index() && offset_ == other.offset_ &&
             size_ == other.size_;
    }
    bool operator!=(const Slice& other) const { return !(*this == other); }
    bool operator<(const Slice& other) const {
      if (index() != other.index()) {
        return index() < other.index();
      }
      if (offset_ != other.offset_) {
        return offset_ < other.offset_;
      }
      return size_ < other.size_;
    }

    // Returns true iff this slice's memory range has a non-empty intersection
    // with the other slice's memory range.
    bool OverlapsWith(const Slice& other) const {
      const int64_t end = offset_ + size_;
      const int64_t other_end = other.offset_ + other.size_;
      return index() == other.index() && offset_ < other_end &&
             end > other.offset_;
    }

    bool Contains(const Slice& other) const {
      const int64_t end = offset_ + size_;
      const int64_t other_end = other.offset_ + other.size_;
      return index() == other.index() && offset_ <= other.offset_ &&
             other_end <= end;
    }

    template <typename H>
    friend H AbslHashValue(H h, const Slice& s) {
      return H::combine(std::move(h), s.index(), s.offset(), s.size());
    }

    std::string ToString() const;

    absl::StatusOr<xla::buffer_assignment::BufferAllocationSliceProto> ToProto()
        const;

    static absl::StatusOr<BufferAllocation::Slice> FromProto(
        const xla::buffer_assignment::BufferAllocationSliceProto& proto,
        absl::Span<const BufferAllocation> buffer_allocations);

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const Slice& slice) {
      absl::Format(&sink, "%v", slice.ToString());
    }

   private:
    // The BufferAllocation that this slice is a part of. The allocation object
    // is not owned by this slice and must outlive it.
    const BufferAllocation* allocation_ = nullptr;
    int64_t offset_ = 0;
    int64_t size_ = 0;
    PrimitiveType element_type_ = PrimitiveType::PRIMITIVE_TYPE_INVALID;
  };

  std::string ToString() const;
  std::string ToShortString(bool human_readable_size = false) const;

  BufferAllocationProto ToProto() const;
  static BufferAllocation FromProto(const BufferAllocationProto&);

  // Whether the buffer is a parameter to or live out of the entry computation.
  bool IsInputOrOutput() const {
    return is_entry_computation_parameter() || maybe_live_out();
  }

  // Whether the buffer is a temporary buffer allocated before
  // Executable::ExecuteOnStream.
  bool IsPreallocatedTempBuffer() const {
    // Parameters do not need temporary buffers.
    return !is_entry_computation_parameter() &&
           // LogicalBuffers that maybe pointed to by the output should live out
           // of the computation.
           !maybe_live_out() &&
           // Thread-local buffers are allocated using `alloca`s.
           !is_thread_local() &&
           // Constant buffers are allocated as global values.
           !is_constant();
  }

  bool operator==(const BufferAllocation& other) const {
    return index_ == other.index_;
  }
  bool operator!=(const BufferAllocation& other) const {
    return !(*this == other);
  }
  bool operator<(const BufferAllocation& other) const {
    return index() < other.index();
  }

  void set_entry_computation_parameter(int64_t parameter_number,
                                       ShapeIndex param_shape_index,
                                       bool parameter_aliased_with_output) {
    is_entry_computation_parameter_ = true;
    is_parameter_aliased_with_output_ = parameter_aliased_with_output;
    parameter_number_ = parameter_number;
    param_shape_index_ = std::move(param_shape_index);
  }

  void set_constant(bool is_constant) { is_constant_ = is_constant; }

 private:
  // Only BufferAssigner and BufferAssignment can modify BufferAllocation.
  friend class BufferAssigner;
  friend class BufferAssignment;

  void set_index(Index index) { index_ = index; }
  void set_size(int64_t size) { size_ = size; }

  // The index of the allocation in the BufferAssignment.
  Index index_;

  // Size of the allocation in bytes.
  int64_t size_;

  // Color of the allocation.
  BufferColor color_;

  // The page id of the allocation, indicating in which page of a multi-page
  // assignment this allocation is located. Only set for multi-page buffer
  // assignment. If multi-page buffer assignment is not used this field will be
  // set to 0.
  int64_t page_id_;

  // If this allocation holds an entry computation parameter, this field
  // indicates the index (starting from 0) of the parameter.
  int64_t parameter_number_ = 0;

  // Whether this buffer needs to be thread-local.
  bool is_thread_local_ : 1;

  // Whether this buffer holds a tuple.
  bool is_tuple_ : 1;

  // Whether this allocation holds an entry computation parameter. Entry
  // computation parameters are special because they have lifetimes which may
  // outlast the computation.
  bool is_entry_computation_parameter_ : 1;

  // Whether this entry computation parameter is aliased with output.
  bool is_parameter_aliased_with_output_ : 1;

  // Whether the allocation contains a LogicalBuffer which may be live-out of
  // the entry computation. Note that this flag is conservatively computed by
  // TuplePointsToAnalysis.  That is, an allocation marked `maybe_live_out_`
  // might not actually escape.
  bool maybe_live_out_ : 1;

  // See comment on the is_constant() accessor.
  bool is_constant_ : 1;

  // If this buffer is for an entry computation parameter, which subshape of the
  // parameter is it for?
  ShapeIndex param_shape_index_;
};

// Add stream operators for nicer output of CHECK/RET_CHECK failures.
std::ostream& operator<<(std::ostream& out, const BufferAllocation& buffer);
std::ostream& operator<<(std::ostream& out, const BufferAllocation::Slice& s);

}  // namespace xla

#endif  // XLA_SERVICE_BUFFER_ALLOCATION_H_
