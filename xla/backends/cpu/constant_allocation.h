/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_CONSTANT_ALLOCATION_H_
#define XLA_BACKENDS_CPU_CONSTANT_ALLOCATION_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
class BufferAssignment;
class Literal;
}  // namespace xla

namespace xla::cpu {

// A storage (or an alias) for constant allocations data.
struct ConstantAllocation {
  class OwnedBuffer {
   public:
    OwnedBuffer() = default;
    explicit OwnedBuffer(size_t size) : size_(size) {
      size_t alloc_size = std::max<size_t>(size, 1);
      data_ = static_cast<uint8_t*>(
          ::operator new(alloc_size, std::align_val_t{Align()}));
      std::memset(data_, 0, alloc_size);
    }
    ~OwnedBuffer() {
      if (data_ != nullptr) {
        ::operator delete(data_, std::align_val_t{Align()});
      }
    }
    OwnedBuffer(const OwnedBuffer&) = delete;
    OwnedBuffer& operator=(const OwnedBuffer&) = delete;
    OwnedBuffer(OwnedBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0)) {}
    OwnedBuffer& operator=(OwnedBuffer&& other) noexcept {
      if (this != &other) {
        if (data_ != nullptr) {
          ::operator delete(data_, std::align_val_t{Align()});
        }
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
      }
      return *this;
    }

    uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

   private:
    uint8_t* data_ = nullptr;
    size_t size_ = 0;
  };

  se::DeviceAddressBase AsDeviceAddress() const;

  ABSL_DEPRECATE_AND_INLINE()
  se::DeviceAddressBase AsDeviceMemoryBase() const { return AsDeviceAddress(); }

  int64_t index = -1;
  std::variant<std::monostate, OwnedBuffer, absl::Span<const uint8_t>> data;
};

template <typename LiteralT = Literal>
absl::StatusOr<ConstantAllocation> LiteralToConstantAllocation(
    int64_t index, const LiteralT& literal) {
  PrimitiveType element_type = literal.shape().element_type();
  if (!primitive_util::IsArrayType(element_type)) {
    return absl::InternalError(
        "Only array literals can be converted to constant allocations");
  }

  int64_t size_bytes = literal.size_bytes();
  const void* untyped_data = literal.untyped_data();

  if (primitive_util::IsSubByteNonPredType(element_type)) {
    int bit_width = primitive_util::BitWidth(element_type);
    int64_t packed_size_bytes = CeilOfRatio<int64_t>(size_bytes, 8 / bit_width);
    ConstantAllocation::OwnedBuffer packed(packed_size_bytes);
    PackIntN(
        bit_width,
        absl::MakeSpan(reinterpret_cast<const char*>(untyped_data), size_bytes),
        absl::MakeSpan(reinterpret_cast<char*>(packed.data()), packed.size()));
    return ConstantAllocation{index, std::move(packed)};
  }

  return ConstantAllocation{
      index, absl::Span<const uint8_t>(
                 reinterpret_cast<const uint8_t*>(untyped_data), size_bytes)};
}

// Creates a vector of constant allocations from the given buffer assignment.
template <typename BufferAssignmentT = BufferAssignment>
absl::StatusOr<std::vector<ConstantAllocation>> CreateConstantAllocations(
    const BufferAssignmentT& assignment) {
  std::vector<ConstantAllocation> constants;

  for (const auto& allocation : assignment.Allocations()) {
    if (!allocation.is_constant()) {
      continue;
    }

    // Find the constant instruction defining the value for allocation.
    const auto* const_instr = static_cast<
        decltype(allocation.assigned_buffers().begin()->first->instruction())>(
        nullptr);
    for (const auto& [value, _] : allocation.assigned_buffers()) {
      // Multiple aliasing instructions can share the allocation, we need to
      // find the original constant instruction that defines the value.
      if (value->instruction()->opcode() ==
          decltype(value->instruction()->opcode())::kConstant) {
        if (const_instr != nullptr) {
          return absl::InternalError(
              absl::StrCat("Multiple constant instructions define buffer ",
                           allocation.ToString()));
        }
        const_instr = value->instruction();
      }
    }
    if (const_instr == nullptr) {
      return absl::InternalError(
          absl::StrCat("Could not find constant instruction defining buffer ",
                       allocation.ToString()));
    }

    ABSL_ASSIGN_OR_RETURN(constants.emplace_back(),
                          LiteralToConstantAllocation(allocation.index(),
                                                      const_instr->literal()));
  }

  return constants;
}

absl::StatusOr<std::vector<ConstantAllocation>> CreateConstantAllocations(
    const BufferAssignmentProto& assignment, const HloModuleProto& module);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CONSTANT_ALLOCATION_H_
