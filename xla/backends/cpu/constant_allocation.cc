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

#include "xla/backends/cpu/constant_allocation.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

template <typename T, typename RepeatedFieldT>
absl::Status CopyRepeatedFieldToBuffer(uint8_t* dst, size_t dst_bytes,
                                       const RepeatedFieldT& src) {
  if (static_cast<size_t>(src.size()) * sizeof(T) != dst_bytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("LiteralProto size mismatch: expected ",
                     dst_bytes / sizeof(T), " elements, got ", src.size()));
  }
  T* typed_dst = reinterpret_cast<T*>(dst);
  std::copy(src.begin(), src.end(), typed_dst);
  return absl::OkStatus();
}

absl::Status CopyBytesToBuffer(uint8_t* dst, size_t dst_bytes,
                               const std::string& src) {
  if (src.size() != dst_bytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("LiteralProto byte size mismatch: expected ", dst_bytes,
                     ", got ", src.size()));
  }
  if (dst_bytes > 0) {
    std::memcpy(dst, src.data(), dst_bytes);
  }
  return absl::OkStatus();
}

absl::StatusOr<ConstantAllocation> LiteralProtoToConstantAllocation(
    int64_t index, const LiteralProto& proto) {
  ABSL_ASSIGN_OR_RETURN(Shape shape, Shape::FromProto(proto.shape()));
  PrimitiveType element_type = shape.element_type();
  if (!primitive_util::IsArrayType(element_type)) {
    return absl::InternalError(
        "Only array literals can be converted to constant allocations");
  }

  int64_t num_elements = ShapeUtil::ElementsIn(shape);
  size_t unpacked_size_bytes = static_cast<size_t>(num_elements) *
                               primitive_util::ByteWidth(element_type);
  ConstantAllocation::OwnedBuffer buffer(unpacked_size_bytes);
  uint8_t* dst = buffer.data();

  switch (element_type) {
    case PRED:
      ABSL_RETURN_IF_ERROR(CopyRepeatedFieldToBuffer<bool>(
          dst, unpacked_size_bytes, proto.preds()));
      break;
    case S2:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.s2s()));
      break;
    case S4:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.s4s()));
      break;
    case S8:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.s8s()));
      break;
    case S16:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.s16s()));
      break;
    case S32:
      ABSL_RETURN_IF_ERROR(CopyRepeatedFieldToBuffer<int32_t>(
          dst, unpacked_size_bytes, proto.s32s()));
      break;
    case S64:
      ABSL_RETURN_IF_ERROR(CopyRepeatedFieldToBuffer<int64_t>(
          dst, unpacked_size_bytes, proto.s64s()));
      break;
    case U2:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.u2s()));
      break;
    case U4:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.u4s()));
      break;
    case U8:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.u8s()));
      break;
    case U16:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.u16s()));
      break;
    case U32:
      ABSL_RETURN_IF_ERROR(CopyRepeatedFieldToBuffer<uint32_t>(
          dst, unpacked_size_bytes, proto.u32s()));
      break;
    case U64:
      ABSL_RETURN_IF_ERROR(CopyRepeatedFieldToBuffer<uint64_t>(
          dst, unpacked_size_bytes, proto.u64s()));
      break;
    case F4E2M1FN:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.f4e2m1fns()));
      break;
    case F8E5M2:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.f8e5m2s()));
      break;
    case F8E4M3:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.f8e4m3s()));
      break;
    case F8E4M3FN:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.f8e4m3fns()));
      break;
    case F8E4M3B11FNUZ:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.f8e4m3b11fnuzs()));
      break;
    case F8E5M2FNUZ:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.f8e5m2fnuzs()));
      break;
    case F8E4M3FNUZ:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.f8e4m3fnuzs()));
      break;
    case F8E3M4:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.f8e3m4s()));
      break;
    case F8E8M0FNU:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.f8e8m0fnus()));
      break;
    case F16:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.f16s()));
      break;
    case BF16:
      ABSL_RETURN_IF_ERROR(
          CopyBytesToBuffer(dst, unpacked_size_bytes, proto.bf16s()));
      break;
    case F32:
      ABSL_RETURN_IF_ERROR(CopyRepeatedFieldToBuffer<float>(
          dst, unpacked_size_bytes, proto.f32s()));
      break;
    case F64:
      ABSL_RETURN_IF_ERROR(CopyRepeatedFieldToBuffer<double>(
          dst, unpacked_size_bytes, proto.f64s()));
      break;
    case C64: {
      if (proto.c64s_size() != num_elements * 2) {
        return absl::InvalidArgumentError("C64 LiteralProto size mismatch");
      }
      auto* complex_dst = reinterpret_cast<complex64*>(dst);
      for (int64_t i = 0; i < num_elements; ++i) {
        complex_dst[i] = complex64{proto.c64s(i * 2), proto.c64s(i * 2 + 1)};
      }
      break;
    }
    case C128: {
      if (proto.c128s_size() != num_elements * 2) {
        return absl::InvalidArgumentError("C128 LiteralProto size mismatch");
      }
      auto* complex_dst = reinterpret_cast<complex128*>(dst);
      for (int64_t i = 0; i < num_elements; ++i) {
        complex_dst[i] = complex128{proto.c128s(i * 2), proto.c128s(i * 2 + 1)};
      }
      break;
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported element type in LiteralProto: ", element_type));
  }

  if (primitive_util::IsSubByteNonPredType(element_type)) {
    int bit_width = primitive_util::BitWidth(element_type);
    int64_t packed_size_bytes =
        CeilOfRatio<int64_t>(unpacked_size_bytes, 8 / bit_width);
    ConstantAllocation::OwnedBuffer packed(packed_size_bytes);
    PackIntN(
        bit_width,
        absl::MakeSpan(reinterpret_cast<const char*>(buffer.data()),
                       buffer.size()),
        absl::MakeSpan(reinterpret_cast<char*>(packed.data()), packed.size()));
    return ConstantAllocation{index, std::move(packed)};
  }

  return ConstantAllocation{index, std::move(buffer)};
}

}  // namespace

se::DeviceAddressBase ConstantAllocation::AsDeviceAddress() const {
  if (auto* _ = std::get_if<std::monostate>(&data)) {
    return se::DeviceAddressBase();
  }

  if (auto* owned = std::get_if<OwnedBuffer>(&data)) {
    return se::DeviceAddressBase(owned->data(), owned->size());
  }

  auto* view = std::get_if<absl::Span<const uint8_t>>(&data);
  return se::DeviceAddressBase(
      const_cast<void*>(reinterpret_cast<const void*>(view->data())),
      view->size());
}

absl::StatusOr<std::vector<ConstantAllocation>> CreateConstantAllocations(
    const BufferAssignmentProto& assignment, const HloModuleProto& module) {
  absl::flat_hash_map<int64_t, const HloInstructionProto*> id_to_instruction;
  for (const HloComputationProto& comp : module.computations()) {
    for (const HloInstructionProto& instr : comp.instructions()) {
      id_to_instruction[instr.id()] = &instr;
    }
  }

  absl::flat_hash_map<int64_t, int64_t> buffer_id_to_instr_id;
  for (const LogicalBufferProto& lb : assignment.logical_buffers()) {
    buffer_id_to_instr_id[lb.id()] = lb.defined_at().instruction_id();
  }

  std::vector<ConstantAllocation> constants;
  for (const BufferAllocationProto& allocation :
       assignment.buffer_allocations()) {
    if (!allocation.is_constant()) {
      continue;
    }

    const HloInstructionProto* const_instr = nullptr;
    for (const auto& assigned : allocation.assigned()) {
      auto buf_it = buffer_id_to_instr_id.find(assigned.logical_buffer_id());
      if (buf_it == buffer_id_to_instr_id.end()) {
        continue;
      }
      auto instr_it = id_to_instruction.find(buf_it->second);
      if (instr_it == id_to_instruction.end()) {
        continue;
      }
      if (instr_it->second->opcode() == "constant") {
        if (const_instr != nullptr && const_instr != instr_it->second) {
          return absl::InternalError(
              absl::StrCat("Multiple constant instructions define buffer ",
                           allocation.index()));
        }
        const_instr = instr_it->second;
      }
    }
    if (const_instr == nullptr) {
      return absl::InternalError(
          absl::StrCat("Could not find constant instruction defining buffer ",
                       allocation.index()));
    }

    ABSL_ASSIGN_OR_RETURN(constants.emplace_back(),
                          LiteralProtoToConstantAllocation(
                              allocation.index(), const_instr->literal()));
  }

  return constants;
}

}  // namespace xla::cpu
