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

#include "xla/pjrt/tiled_linearizer.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/overflow_util.h"
#include "xla/pjrt/tiled_linearizer_highway.h"
#include "xla/pjrt/transpose.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tpu/noncopyable_buffer.h"
#include "xla/tsl/platform/byte_order.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using Buffer = tensorflow::tpu::NoncopyableBuffer;

bool HasLinearLayout(const Layout& layout) {
  return !layout.tiles().empty() &&
         layout.tiles()[0].dimensions() == absl::Span<const int64_t>({1});
}

template <int N>
struct ComponentTraits {};
template <>
struct ComponentTraits<64> {
  static constexpr PrimitiveType kFloatingPointComponent = F32;
  static constexpr PrimitiveType kIntegralComponent = U32;
};
template <>
struct ComponentTraits<128> {
  static constexpr PrimitiveType kFloatingPointComponent = F64;
  static constexpr PrimitiveType kIntegralComponent = U64;
};

template <int N>
constexpr PrimitiveType ComponentType(const PrimitiveType type) {
  if (!primitive_util::IsArrayType(type) ||
      primitive_util::BitWidth(type) != N) {
    return type;
  }
  if (primitive_util::IsComplexType(type)) {
    return primitive_util::ComplexComponentType(type);
  }
  if (primitive_util::IsFloatingPointType(type)) {
    return ComponentTraits<N>::kFloatingPointComponent;
  }
  return ComponentTraits<N>::kIntegralComponent;
}

template <int N>
Shape ComponentShape(const Shape& shape) {
  if (shape.IsTuple()) {
    std::vector<Shape> component_shapes;
    component_shapes.reserve(shape.tuple_shapes().size());
    for (const Shape& tuple_shape : shape.tuple_shapes()) {
      component_shapes.push_back(ComponentShape<N>(tuple_shape));
    }
    return ShapeUtil::MakeTupleShape(component_shapes);
  }
  if (!ShapeUtil::ElementHasBitWidth(shape, N)) {
    return shape;
  }
  return ShapeUtil::ChangeElementType(shape,
                                      ComponentType<N>(shape.element_type()));
}

constexpr PrimitiveType X64ComponentType(const PrimitiveType x64_type) {
  return ComponentType<64>(x64_type);
}

Shape X64ComponentShape(const Shape& shape) {
  return ComponentShape<64>(shape);
}

Shape X128ComponentShape(const Shape& shape) {
  return ComponentShape<128>(shape);
}

// All-ones bit-pattern is a quiet NaN for IEEE/OCP float types with NaN
// (F32, F16, BF16, F8E5M2, F8E4M3FN, etc.), so a single repeated byte works
// across packed element widths.
constexpr uint8_t kPadByte = 0xff;

// TensorCore lane count as constexpr used for template specialization.
template <int64_t kTcLaneCount>
using TcLaneCountConstant = std::integral_constant<int64_t, kTcLaneCount>;

template <typename F>
absl::Status TcLaneCountSwitch(F&& f, int64_t lane_count) {
  switch (lane_count) {
    case 128:
      std::invoke(f, TcLaneCountConstant<128>());
      return absl::OkStatus();
    default:
      return absl::UnimplementedError(
          absl::StrCat("unhandled lane_count ", lane_count));
  }
}

// TensorCore lane count as constexpr used for template specialization.
template <int64_t kElementPackingFactor>
using ElementPackingFactorConstant =
    std::integral_constant<int64_t, kElementPackingFactor>;

template <typename F>
auto ElementPackingFactorSwitch(F&& f, const int64_t packing_factor) {
  switch (packing_factor) {
    case 1:
      return std::invoke(f, ElementPackingFactorConstant<1>());
    case 2:
      return std::invoke(f, ElementPackingFactorConstant<2>());
    case 4:
      return std::invoke(f, ElementPackingFactorConstant<4>());
    case 8:
      return std::invoke(f, ElementPackingFactorConstant<8>());
    case 16:
      return std::invoke(f, ElementPackingFactorConstant<16>());
    case 32:
      return std::invoke(f, ElementPackingFactorConstant<32>());
    default:
      LOG(FATAL) << "unhandled packing_factor" << packing_factor;
  }
}

template <typename R, typename F>
R ElementPackingFactorSwitch(F&& f, const int64_t packing_factor) {
  return ElementPackingFactorSwitch(std::forward<F>(f), packing_factor);
}

std::string FormatLinearizedLiteral(const Shape& literal_shape,
                                    absl::Span<const uint32_t> values) {
  return absl::StrJoin(
      values, ", ", [&literal_shape](std::string* out, uint64_t value) {
        auto component_type = X64ComponentType(literal_shape.element_type());
        primitive_util::PrimitiveTypeSwitch<void>(
            [&](auto constant_type) {
              if constexpr (primitive_util::IsArrayType(constant_type) &&
                            primitive_util::BitWidth(constant_type) <= 32) {
                using NativeT = primitive_util::NativeTypeOf<constant_type>;
                using BitsT = UnsignedIntegerTypeForSizeType<sizeof(NativeT)>;
                constexpr auto kBitWidth =
                    primitive_util::BitWidth(constant_type);
                constexpr int kSubElements = 32 / kBitWidth;
                for (int i = 0; i < kSubElements; ++i) {
                  if (i > 0) {
                    absl::StrAppendFormat(out, ", ");
                  }
                  auto bits = static_cast<BitsT>(value);
                  if constexpr (primitive_util::IsFloatingPointType(
                                    constant_type)) {
                    auto native = Eigen::numext::bit_cast<NativeT>(bits);
                    absl::StrAppendFormat(out, "%g",
                                          static_cast<float>(native));
                  } else if constexpr (primitive_util::IsSignedIntegralType(
                                           constant_type)) {
                    auto native = static_cast<NativeT>(value);
                    absl::StrAppendFormat(out, "%d",
                                          static_cast<int32_t>(native));
                  } else {
                    absl::StrAppendFormat(out, "%#x", bits);
                  }
                  value >>= kBitWidth;
                }
              }
            },
            component_type);
      });
}

template <int64_t kPackingFactor>
void RepackFromSparseCoreLayout(absl::Span<const uint32_t> input,
                                absl::Span<uint8_t> output) {
  const size_t output_bytes = output.size();
  if constexpr (kPackingFactor <= 4) {
    std::memcpy(output.data(), input.data(), output_bytes);
  } else {
    constexpr int kBitsPerElement = 32 / kPackingFactor;
    const size_t num_elements = output_bytes;
    std::memset(output.data(), 0, output_bytes);
    const uint8_t* input_bytes = reinterpret_cast<const uint8_t*>(input.data());
    for (size_t i = 0; i < num_elements; ++i) {
      size_t byte_idx = (i * kBitsPerElement) / CHAR_BIT;
      size_t bit_offset = (i * kBitsPerElement) % CHAR_BIT;
      uint8_t mask = (1 << kBitsPerElement) - 1;
      output[i] = (input_bytes[byte_idx] >> bit_offset) & mask;
    }
  }
}

absl::Span<uint8_t> GetLiteralAsMutableSpan(MutableLiteralBase* literal) {
  return absl::MakeSpan(
      tsl::safe_reinterpret_cast<uint8_t*>(literal->untyped_data()),
      literal->size_bytes());
}

void CreateGarbageFilledLiteral(const Shape& shape,
                                MutableLiteralBase* literal) {
  // Use Compatible instead of Equal because host-offloaded tensors may have
  // layout/tiling differences (such as linear tiled layouts T(1)L(1024))
  // between the device shape and the host literal shape, but their logical
  // shapes (dimensions and element type) are compatible.
  DCHECK(ShapeUtil::Compatible(shape, literal->shape()))
      << "shape: " << shape.ToString(true)
      << ", literal->shape(): " << literal->shape().ToString(true);
  primitive_util::ArrayTypeSwitch(
      [&](auto ptype) {
        using NativeT = primitive_util::NativeTypeOf<ptype>;
        if constexpr (ptype == PRED) {
          literal->PopulateWithValue<bool>(false);
          return;
        }
        if constexpr (primitive_util::IsSignedIntegralType(ptype)) {
          constexpr int64_t kGarbage{0x5a5a5a5a5a5a5a5a};
          literal->PopulateWithValue<NativeT>(static_cast<NativeT>(kGarbage));
          return;
        }
        if constexpr (primitive_util::IsUnsignedIntegralType(ptype)) {
          constexpr uint64_t kGarbage{0xcdcdcdcdcdcdcdcd};
          literal->PopulateWithValue<NativeT>(static_cast<NativeT>(kGarbage));
          return;
        }
        if constexpr (primitive_util::IsFloatingPointType(ptype)) {
          const NativeT kGarbage =
              static_cast<NativeT>(std::numeric_limits<NativeT>::quiet_NaN());
          literal->PopulateWithValue<NativeT>(kGarbage);
          return;
        }
        if constexpr (primitive_util::IsComplexType(ptype)) {
          using ComponentT = typename NativeT::value_type;
          constexpr ComponentT kGarbage =
              std::numeric_limits<ComponentT>::quiet_NaN();
          literal->PopulateWithValue<NativeT>(NativeT{kGarbage, kGarbage});
          return;
        }
        LOG(FATAL) << "unhandled primitive type: "
                   << PrimitiveType_Name(ptype());
      },
      shape.element_type());
}
}  // namespace

namespace {

// Reorders a 4/8/16-bit packed input to the hardware expected layout: For
// each sublane, split the inputs into four/two parts, then zip four/two
// elements from the same lane together.
//
// `size` represents the number of 32-bit elements that need to be repacked.
//
// Precondition: Input size is a multiple of chunk lanes. If source buffer and
// target buffer aliases, then source_buffer == target_buffer must hold.
//
// Example for BF16: "|" is the pack operation for two 16 bits number.
//
// Input:  a1  |a2   a3  |a4    ....     a127|a128 (64 elements, 256 bytes)
//         b1  |b2   b3  |b4    ....     b127|b128 (64 elements, 256 bytes)
//         c1  |c2   c3  |c4    ....     c127|c128 (64 elements, 256 bytes)
//         d1  |d2   d3  |d4    ....     d127|d128 (64 elements, 256 bytes)
//
// Output: b1  |a1   b2  |a2    ....     b128|a128 (128 elements, 512 bytes)
//         d1  |c1   d2  |c2    ....     d128|c128 (128 elements, 512 bytes)
//
template <int64_t kPackingFactor, int64_t kLaneCount>
void RepackToHardwareLayout(absl::Span<const uint8_t> source_buffer,
                            uint32_t* target_buffer, bool set_padding_to_nan) {
  static_assert(kLaneCount % kPackingFactor == 0,
                "ChunkLanes not a multiple of kPackingFactor");
  constexpr int kTpuToCpuExpansionFactor =
      CeilOfRatio<size_t>(kPackingFactor, sizeof(uint32_t));
  constexpr int kExpandedLaneCount = kLaneCount * kTpuToCpuExpansionFactor;
  uint32_t staging_buffer[kExpandedLaneCount];
  constexpr int kBytesReadPerSublaneWritten = sizeof(staging_buffer);
  const size_t input_bytes = source_buffer.size();
  const size_t full_reads = input_bytes / kBytesReadPerSublaneWritten;
  const size_t partial_bytes_remaining =
      input_bytes % kBytesReadPerSublaneWritten;
  bool is_alias = source_buffer.data() ==
                  tsl::safe_reinterpret_cast<uint8_t*>(target_buffer);

  // Don't leave uninitialized memory if we are in MSAN mode.
#ifdef MEMORY_SANITIZER
  set_padding_to_nan = true;
#endif

  if constexpr (kPackingFactor == 1) {
    if (!is_alias) {
      std::memcpy(target_buffer, source_buffer.data(), input_bytes);
    }
  } else {
    for (size_t i = 0; i < full_reads; ++i) {
      // If inputs are pointer aliases, copy source to staging buffer so we can
      // modify source/target in place. Compiler actually optimizes the memcpy
      // away by prefetching a sublane into vector registers.
      // NOTE: Another way to handle pointer aliases is
      //
      // ```
      // staging = is_alias ? staging_buffer : target_sublane
      // RepackToHardwareLayoutImpl(source_sublane, staging)
      // if (is_alias) memcpy(target_sublane, staging_buffer)
      // ```
      //
      // However, compiler does not optimize the memcpy away in this case.
      if (is_alias) {
        __builtin_memcpy_inline(staging_buffer,
                                &source_buffer[kBytesReadPerSublaneWritten * i],
                                sizeof(staging_buffer));
      }
      const uint32_t* staging =
          is_alias ? staging_buffer
                   : tsl::safe_reinterpret_cast<const uint32_t*>(
                         &source_buffer[kBytesReadPerSublaneWritten * i]);

      RepackToHardwareLayoutSublaneHighway<kPackingFactor, kLaneCount>(
          staging, &target_buffer[kLaneCount * i]);
    }
    if (partial_bytes_remaining > 0) {
      uint8_t* first_uninitialized_byte = std::copy_n(
          tsl::safe_reinterpret_cast<const uint8_t*>(
              &source_buffer[kBytesReadPerSublaneWritten * full_reads]),
          partial_bytes_remaining,
          tsl::safe_reinterpret_cast<uint8_t*>(staging_buffer));
      // The caller may want to make sure uninitialized memory does not make it
      // to the linearized literal. This is important for executable
      // reproducibility as literals get baked into the TPU program data
      // segment.
      if (set_padding_to_nan) {
        // Fill the rest of the staging buffer with an all-ones bit-pattern.
        // This bit-pattern is nice because it is a NaN regardless of the width
        // of the exponent field and as such works with F32, F16, BF16, etc.
        std::fill_n(first_uninitialized_byte,
                    kBytesReadPerSublaneWritten - partial_bytes_remaining, -1);
      } else {
        std::fill_n(first_uninitialized_byte,
                    kBytesReadPerSublaneWritten - partial_bytes_remaining, 0);
      }
      RepackToHardwareLayoutSublaneHighway<kPackingFactor, kLaneCount>(
          staging_buffer, &target_buffer[kLaneCount * full_reads]);
    }
  }
}

// Reorders a 4/8/16-bit packed input to the host expected layout. This is the
// reverse transform of RepackToHardwareLayout
//
// Precondition: Input size is a multiple of chunk lanes.
//
// Example of BF16: "|" is the pack operation for two 16 bits number.
//
// Input:  b1  |a1   b2  |a2    ....     b128|a128 (128 elements, 512 bytes)
//         d1  |c1   d2  |c2    ....     d128|c128 (128 elements, 512 bytes)
//
// Output: a1  |a2   a3  |a4    ....     a127|a128 (64 elements, 256 bytes)
//         b1  |b2   b3  |b4    ....     b127|b128 (64 elements, 256 bytes)
//         c1  |c2   c3  |c4    ....     c127|c128 (64 elements, 256 bytes)
//         d1  |d2   d3  |d4    ....     d127|d128 (64 elements, 256 bytes)
//
template <int64_t kPackingFactor, int64_t kLaneCount>
void RepackToHostLayout(absl::Span<const uint32_t> input,
                        absl::Span<uint8_t> output) {
  constexpr int kTpuToCpuExpansionFactor =
      CeilOfRatio<size_t>(kPackingFactor, sizeof(uint32_t));
  constexpr int kExpandedLaneCount = kLaneCount * kTpuToCpuExpansionFactor;
  // We repack a sublane at a time, more than enough to occupy the SIMD lanes on
  // the host. While our input is a multiple of kLaneCount, our output may not
  // be because linearized data is padded to a multiple of the number of lanes.
  // Each lane delinearizes to `kTpuToCpuExpansionFactor` uint32_t values. If
  // our output is smaller, we delinearize into a staging buffer and copy the
  // needed values out.
  // N.B. Expansion occurs when we are delinearizing types which are smaller
  // than a byte because those types take up less space on the TPU (typically
  // the number of bits in the type) than on the CPU (typeically 8 bits).
  uint32_t staging_buffer[kExpandedLaneCount];
  constexpr int kBytesReadPerSublane = kLaneCount * sizeof(uint32_t);
  constexpr int kBytesWrittenPerSublane =
      kBytesReadPerSublane * kTpuToCpuExpansionFactor;
  const size_t output_bytes = output.size();
  const size_t full_writes = output_bytes / kBytesWrittenPerSublane;
  const size_t partial_bytes_remaining = output_bytes % kBytesWrittenPerSublane;
  if constexpr (kPackingFactor > 1) {
    for (size_t i = 0; i < full_writes; ++i) {
      RepackToHostLayoutSublaneHighway<kPackingFactor, kLaneCount>(
          &input[kLaneCount * i], tsl::safe_reinterpret_cast<uint32_t*>(
                                      &output[kBytesWrittenPerSublane * i]));
    }
    if (partial_bytes_remaining > 0) {
      RepackToHostLayoutSublaneHighway<kPackingFactor, kLaneCount>(
          &input[kLaneCount * full_writes], staging_buffer);
      std::memcpy(&output[kBytesWrittenPerSublane * full_writes],
                  staging_buffer, partial_bytes_remaining);
    }
  } else {
    std::memcpy(output.data(), input.data(), output.size());
  }
}

// Reorders 6-bit elements from 1-byte host strips of shape (16, kLaneCount)
// (4 * kLaneCount uint32_t words) into 3-sublane interleaved 2b/4b hardware
// strips (3 * kLaneCount uint32_t words):
// [sublane 0: 4b rows 0..7, sublane 1: 4b rows 8..15, sublane 2: 2b rows 0..15]
template <int64_t kLaneCount>
void Repack6BitToHardwareLayout(absl::Span<const uint8_t> source_buffer,
                                absl::Span<uint8_t> target_bytes,
                                bool set_padding_to_nan) {
  static_assert(kLaneCount % 16 == 0, "ChunkLanes not a multiple of 16");
  constexpr size_t kHostU32sPerStrip = 4 * kLaneCount;
  constexpr size_t kHardwareU32sPerStrip = 3 * kLaneCount;
  constexpr size_t kHardwareBytesPerStrip =
      kHardwareU32sPerStrip * sizeof(uint32_t);
  uint32_t staging_buffer[kHostU32sPerStrip];
  constexpr size_t kBytesReadPerStrip = sizeof(staging_buffer);
  const size_t input_bytes = source_buffer.size();
  const size_t full_strips = input_bytes / kBytesReadPerStrip;
  const size_t partial_bytes_remaining = input_bytes % kBytesReadPerStrip;
  const size_t required_strips =
      full_strips + (partial_bytes_remaining > 0 ? 1 : 0);
  CHECK_EQ(target_bytes.size() % kHardwareBytesPerStrip, 0);
  CHECK_GE(target_bytes.size(), required_strips * kHardwareBytesPerStrip);
  const bool use_staging =
      source_buffer.data() == target_bytes.data() ||
      reinterpret_cast<uintptr_t>(source_buffer.data()) % alignof(uint32_t) !=
          0;

#ifdef MEMORY_SANITIZER
  set_padding_to_nan = true;
#endif

  for (size_t i = 0; i < full_strips; ++i) {
    if (use_staging) {
      std::memcpy(staging_buffer, &source_buffer[kBytesReadPerStrip * i],
                  sizeof(staging_buffer));
    }
    const uint32_t* staging = use_staging
                                  ? staging_buffer
                                  : tsl::safe_reinterpret_cast<const uint32_t*>(
                                        &source_buffer[kBytesReadPerStrip * i]);
    Repack6BitStripToHardwareLayoutHighway<kLaneCount>(
        staging, tsl::safe_reinterpret_cast<uint32_t*>(
                     &target_bytes[kHardwareBytesPerStrip * i]));
  }
  if (partial_bytes_remaining > 0) {
    uint8_t* first_uninitialized_byte =
        std::copy_n(&source_buffer[kBytesReadPerStrip * full_strips],
                    partial_bytes_remaining,
                    tsl::safe_reinterpret_cast<uint8_t*>(staging_buffer));
    std::fill_n(first_uninitialized_byte,
                kBytesReadPerStrip - partial_bytes_remaining,
                set_padding_to_nan ? uint8_t{0xff} : uint8_t{0});
    Repack6BitStripToHardwareLayoutHighway<kLaneCount>(
        staging_buffer,
        tsl::safe_reinterpret_cast<uint32_t*>(
            &target_bytes[kHardwareBytesPerStrip * full_strips]));
  }
}

// Unpacks 3-sublane interleaved 2b/4b hardware strips (3 * kLaneCount
// uint32_t words) into 1-byte host strips of shape (16, kLaneCount)
// (4 * kLaneCount uint32_t words).
template <int64_t kLaneCount>
void Repack6BitToHostLayout(absl::Span<const uint32_t> input,
                            absl::Span<uint8_t> output) {
  static_assert(kLaneCount % 16 == 0, "ChunkLanes not a multiple of 16");
  constexpr size_t kHardwareU32sPerStrip = 3 * kLaneCount;
  constexpr size_t kHostU32sPerStrip = 4 * kLaneCount;
  constexpr size_t kBytesWrittenPerStrip = kHostU32sPerStrip * sizeof(uint32_t);
  uint32_t staging_buffer[kHostU32sPerStrip];
  const size_t output_bytes = output.size();
  const size_t full_writes = output_bytes / kBytesWrittenPerStrip;
  const size_t partial_bytes_remaining = output_bytes % kBytesWrittenPerStrip;
  const size_t required_strips =
      full_writes + (partial_bytes_remaining > 0 ? 1 : 0);
  CHECK_EQ(input.size() % kHardwareU32sPerStrip, 0);
  CHECK_GE(input.size(), required_strips * kHardwareU32sPerStrip);
  const bool is_output_aligned =
      reinterpret_cast<uintptr_t>(output.data()) % alignof(uint32_t) == 0;

  for (size_t i = 0; i < full_writes; ++i) {
    if (is_output_aligned) {
      Repack6BitStripToHostLayoutHighway<kLaneCount>(
          &input[kHardwareU32sPerStrip * i],
          tsl::safe_reinterpret_cast<uint32_t*>(
              &output[kBytesWrittenPerStrip * i]));
    } else {
      Repack6BitStripToHostLayoutHighway<kLaneCount>(
          &input[kHardwareU32sPerStrip * i], staging_buffer);
      std::memcpy(&output[kBytesWrittenPerStrip * i], staging_buffer,
                  kBytesWrittenPerStrip);
    }
  }
  if (partial_bytes_remaining > 0) {
    Repack6BitStripToHostLayoutHighway<kLaneCount>(
        &input[kHardwareU32sPerStrip * full_writes], staging_buffer);
    std::memcpy(&output[kBytesWrittenPerStrip * full_writes], staging_buffer,
                partial_bytes_remaining);
  }
}

struct LinearizerCache {
  LinearizerCache() : cache(1024) {}
  absl::Mutex mu;
  TransposePlanCache cache ABSL_GUARDED_BY(mu);
};

LinearizerCache* linearizer_cache = new LinearizerCache;

constexpr int64_t kSupportedFp6TcLaneCount = 128;

absl::StatusOr<size_t> CheckedMulSizeT(size_t lhs, size_t rhs,
                                       absl::string_view context) {
  constexpr uint64_t kMaxSize = std::min<uint64_t>(
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      static_cast<uint64_t>(std::numeric_limits<size_t>::max()));
  if (lhs > kMaxSize || rhs > kMaxSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Integer overflow computing %s (%d * %d).", context, lhs, rhs));
  }
  const auto [product, overflow] = OverflowSafeMultiply(
      static_cast<int64_t>(lhs), static_cast<int64_t>(rhs));
  if (overflow || static_cast<uint64_t>(product) > kMaxSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Integer overflow computing %s (%d * %d).", context, lhs, rhs));
  }
  return static_cast<size_t>(product);
}

absl::StatusOr<int64_t> CheckedRoundUpToInt64(int64_t value,
                                              int64_t alignment) {
  if (value < 0 || alignment <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid rounding arguments: value=%d, alignment=%d.",
                        value, alignment));
  }
  const int64_t quotient = CeilOfRatio<int64_t>(value, alignment);
  const auto [rounded, overflow] = OverflowSafeMultiply(quotient, alignment);
  if (overflow) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Rounded dimension overflows int64_t: value=%d, alignment=%d.", value,
        alignment));
  }
  return rounded;
}

absl::StatusOr<size_t> ComputeTiledElements(
    absl::Span<const int64_t> dims,
    std::optional<absl::Span<const int64_t>> permutation,
    absl::Span<const int64_t> tiling_vec) {
  if (permutation.has_value() && permutation->size() != dims.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Permutation size (%d) must match array rank (%d).",
                        permutation->size(), dims.size()));
  }
  if (tiling_vec.size() > dims.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tiling (%s) must have at most as many dimensions as the array (%d).",
        absl::StrJoin(tiling_vec, ","), dims.size()));
  }
  constexpr uint64_t kMaxElements = std::min<uint64_t>(
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      static_cast<uint64_t>(std::numeric_limits<size_t>::max()));
  int64_t tiled_elems = 1;
  const size_t untiled_dims = dims.size() - tiling_vec.size();
  for (size_t i = 0; i < dims.size(); ++i) {
    const int64_t dim_idx =
        permutation.has_value() ? (*permutation)[i] : static_cast<int64_t>(i);
    if (dim_idx < 0 || static_cast<size_t>(dim_idx) >= dims.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid permutation index %d for rank %d.", dim_idx, dims.size()));
    }
    const int64_t tile = i < untiled_dims ? 1 : tiling_vec[i - untiled_dims];
    ABSL_ASSIGN_OR_RETURN(const int64_t rounded_dim,
                          CheckedRoundUpToInt64(dims[dim_idx], tile));
    const auto [next_elems, overflow] =
        OverflowSafeMultiply(tiled_elems, rounded_dim);
    if (overflow || static_cast<uint64_t>(next_elems) > kMaxElements) {
      return absl::InvalidArgumentError(
          "Tiled element count overflows supported bounds.");
    }
    tiled_elems = next_elems;
  }
  return static_cast<size_t>(tiled_elems);
}

// Returns the TensorCore lane count (`kSupportedFp6TcLaneCount`) if `layout`
// matches a supported canonical TPU 6-bit 2b/4b split strip layout
// (`BP(4:(8,1))(2:(16,1))` on `(16, 128)` strips in 2D+, or
// `BP(4:(128)(8,1))(2:(128)(16,1))` on `(2048)` strips in 0D/1D), or
// `std::nullopt` otherwise.
std::optional<int64_t> GetSupportedFp6BitPlaneLaneCount(const Layout& layout,
                                                        int64_t shape_rank) {
  if (layout.bit_planes().size() != 2 || layout.tiles().empty() ||
      !layout.split_configs().empty() || layout.has_physical_shape()) {
    return std::nullopt;
  }
  const BitPlane& low_plane = layout.bit_planes(0);
  const BitPlane& high_plane = layout.bit_planes(1);
  if (low_plane.bit_offset() != 0 || low_plane.bit_width() != 4 ||
      high_plane.bit_offset() != 4 || high_plane.bit_width() != 2) {
    return std::nullopt;
  }

  const Tile& strip_tile = layout.tiles().back();
  if (strip_tile.dimensions().size() == 2) {
    if (shape_rank < 2) {
      return std::nullopt;
    }
    const int64_t strip_rows = strip_tile.dimension(0);
    const int64_t lane_count = strip_tile.dimension(1);
    if (strip_rows != 16 || lane_count != kSupportedFp6TcLaneCount) {
      return std::nullopt;
    }
    for (size_t i = 0; i + 1 < layout.tiles().size(); ++i) {
      const Tile& outer = layout.tiles(i);
      if (outer.dimensions().size() != 2 || outer.dimension(1) != lane_count) {
        return std::nullopt;
      }
      const int64_t outer_rows = outer.dimension(0);
      if (outer_rows != Tile::kMatchDimension || i != 0) {
        if (outer_rows <= 0 || outer_rows % 16 != 0) {
          return std::nullopt;
        }
      }
    }
    if (low_plane.tiles().size() != 1 || high_plane.tiles().size() != 1 ||
        low_plane.tiles(0) != Tile({8, 1}) ||
        high_plane.tiles(0) != Tile({16, 1})) {
      return std::nullopt;
    }
    return lane_count;
  }

  if (strip_tile.dimensions().size() == 1) {
    if (shape_rank < 0 || shape_rank > 1) {
      return std::nullopt;
    }
    const int64_t strip_elems = strip_tile.dimension(0);
    if (strip_elems != 16 * kSupportedFp6TcLaneCount) {
      return std::nullopt;
    }
    const int64_t lane_count = kSupportedFp6TcLaneCount;
    for (size_t i = 0; i + 1 < layout.tiles().size(); ++i) {
      const Tile& outer = layout.tiles(i);
      if (outer.dimensions().size() != 1 || outer.dimension(0) <= 0 ||
          outer.dimension(0) % strip_elems != 0) {
        return std::nullopt;
      }
    }
    if (low_plane.tiles().size() != 2 || high_plane.tiles().size() != 2 ||
        low_plane.tiles(0) != Tile({lane_count}) ||
        low_plane.tiles(1) != Tile({8, 1}) ||
        high_plane.tiles(0) != Tile({lane_count}) ||
        high_plane.tiles(1) != Tile({16, 1})) {
      return std::nullopt;
    }
    return lane_count;
  }

  return std::nullopt;
}

// Validates `device_layout.bit_planes()` for `shape`. Returns the validated
// FP6 TensorCore lane count (> 0) when `device_layout.bit_planes()` is
// non-empty, 0 when `device_layout.bit_planes()` is empty, an
// `InvalidArgumentError` if `device_layout` is invalid for `shape`, or an
// `UnimplementedError` for unsupported bit-plane layouts.
absl::StatusOr<int64_t> ValidateAndGetFp6BitPlaneLaneCount(
    const Shape& shape, const Layout& device_layout) {
  if (device_layout.bit_planes().empty()) {
    return 0;
  }
  ABSL_RETURN_IF_ERROR(
      LayoutUtil::ValidateLayoutForShape(device_layout, shape));
  if (primitive_util::IsF6Type(shape.element_type())) {
    if (std::optional<int64_t> fp6_lane_count =
            GetSupportedFp6BitPlaneLaneCount(device_layout,
                                             shape.dimensions().size());
        fp6_lane_count.has_value()) {
      return *fp6_lane_count;
    }
  }
  return absl::UnimplementedError(absl::StrFormat(
      "TiledLinearizer does not support bit-plane layout %s for shape %s.",
      device_layout.ToString(), shape.ToString()));
}

bool HasSubtilePacking(const Layout& layout, bool is_fp6_split,
                       int packing_factor) {
  if (is_fp6_split) {
    return true;
  }
  const auto tiles = layout.tiles();
  return packing_factor > 1 &&
         (tiles.size() > 1 ||
          (!tiles.empty() && tiles[0].dimensions().size() > 1));
}

// Returns the TensorCore lane count used for subtile bit-repacking.
absl::StatusOr<int64_t> GetSubtileLaneCount(const Layout& layout) {
  const auto tiles = layout.tiles();
  TF_RET_CHECK(!tiles.empty());
  const auto lane_tile_dims =
      (tiles.size() > 1 ? tiles[tiles.size() - 2] : tiles[0]).dimensions();
  TF_RET_CHECK(!lane_tile_dims.empty());
  return lane_tile_dims.back();
}

absl::Status Validate6BitHardwareStripSize(size_t hardware_bytes,
                                           size_t required_host_bytes,
                                           int64_t lane_count) {
  if (lane_count <= 0 ||
      lane_count > std::numeric_limits<int64_t>::max() / 16) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid 6-bit strip lane count: %d.", lane_count));
  }
  const size_t lane_count_u = static_cast<size_t>(lane_count);
  ABSL_ASSIGN_OR_RETURN(
      const size_t hardware_bytes_per_strip,
      CheckedMulSizeT(lane_count_u, size_t{3} * sizeof(uint32_t),
                      "6-bit hardware strip byte size"));
  if (hardware_bytes % hardware_bytes_per_strip != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "6-bit split hardware buffer size (%d bytes) must be a multiple of a "
        "full 3-sublane (16, %d) strip (%d bytes).",
        hardware_bytes, lane_count, hardware_bytes_per_strip));
  }
  ABSL_ASSIGN_OR_RETURN(
      const size_t host_bytes_per_strip,
      CheckedMulSizeT(lane_count_u, size_t{16}, "6-bit host strip byte size"));
  const size_t required_strips =
      CeilOfRatio<size_t>(required_host_bytes, host_bytes_per_strip);
  ABSL_ASSIGN_OR_RETURN(
      const size_t min_hardware_bytes,
      CheckedMulSizeT(required_strips, hardware_bytes_per_strip,
                      "required 6-bit hardware buffer size"));
  if (hardware_bytes < min_hardware_bytes) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "6-bit split hardware buffer size (%d bytes) is smaller than the %d "
        "bytes (%d strips) required for %d tiled elements.",
        hardware_bytes, min_hardware_bytes, required_strips,
        required_host_bytes));
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::InlinedVector<int64_t, 2>> GetResolvedTopLevelTiling(
    const Shape& device_shape) {
  absl::InlinedVector<int64_t, 2> tiling_vec;
  if (device_shape.layout().tiles().empty()) {
    return tiling_vec;
  }
  const auto tile_dims = device_shape.layout().tiles(0).dimensions();
  for (int i = 0; i < tile_dims.size(); ++i) {
    int64_t dim = tile_dims[i];
    if (dim == Tile::kMatchDimension) {
      int64_t minor = tile_dims.size() - 1 - i;
      if (minor >= device_shape.layout().minor_to_major().size()) {
        continue;
      }
      int64_t logical_dim = device_shape.layout().minor_to_major(minor);
      if (logical_dim < 0 || logical_dim >= device_shape.dimensions().size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Invalid minor_to_major index %d for shape rank %d.", logical_dim,
            device_shape.dimensions().size()));
      }
      int64_t dim_size = device_shape.dimensions(logical_dim);
      if (dim_size < 0) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Invalid negative dimension size %d.", dim_size));
      }
      int64_t packing =
          (device_shape.layout().tiles().size() > 1 &&
           minor < device_shape.layout().tiles(1).dimensions().size())
              ? device_shape.layout().tiles(1).dimension(
                    device_shape.layout().tiles(1).dimensions().size() - 1 -
                    minor)
              : 1;
      if (packing <= 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Invalid non-positive subtile dimension %d.", packing));
      }
      ABSL_ASSIGN_OR_RETURN(
          dim, CheckedRoundUpToInt64(std::max<int64_t>(dim_size, 1), packing));
    } else if (dim <= 0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid non-positive tile dimension %d.", dim));
    }
    tiling_vec.push_back(dim);
  }
  return tiling_vec;
}

}  // namespace

/* static */ absl::Status TiledLinearizer::LinearizeUsingTransposeInto(
    absl::Span<uint8_t> dest, const void* data, const Shape& host_shape,
    absl::Span<int64_t const> byte_strides, const Layout& device_layout,
    bool set_padding_to_nan, absl::Span<const uint32_t> dynamic_sizes,
    const TiledLinearizer::ParallelismOptions& parallelism_options) {
  ABSL_ASSIGN_OR_RETURN(
      const int64_t fp6_lane_count,
      ValidateAndGetFp6BitPlaneLaneCount(host_shape, device_layout));
  const bool is_fp6_split = fp6_lane_count > 0;
  if (host_shape.element_type() == C128) {
    TF_RET_CHECK(dest.size() % 2 == 0);
    Shape f64_shape = ShapeUtil::ChangeElementType(host_shape, F64);
    size_t half_size = dest.size() / 2;
    ABSL_RETURN_IF_ERROR(LinearizeUsingTransposeInto(
        dest.subspan(0, half_size), data, f64_shape, byte_strides,
        device_layout, set_padding_to_nan, dynamic_sizes, parallelism_options));
    return LinearizeUsingTransposeInto(
        dest.subspan(half_size),
        static_cast<const char*>(data) + sizeof(double), f64_shape,
        byte_strides, device_layout, set_padding_to_nan, dynamic_sizes,
        parallelism_options);
  }

  Shape device_shape = host_shape;
  *(device_shape.mutable_layout()) = device_layout;

  int64_t byte_width = primitive_util::ByteWidth(host_shape.element_type());
  int64_t prefix_size =
      dynamic_sizes.empty()
          ? 0
          : device_layout.dynamic_shape_metadata_prefix_bytes();
  TF_RET_CHECK(prefix_size >= 0);
  TF_RET_CHECK(prefix_size % sizeof(uint32_t) == 0);
  TF_RET_CHECK(dest.size() >= static_cast<size_t>(prefix_size));
  size_t size_in_bytes = dest.size() - static_cast<size_t>(prefix_size);
  TF_RET_CHECK(size_in_bytes % sizeof(uint32_t) == 0);
  TF_RET_CHECK(reinterpret_cast<uintptr_t>(dest.data()) % alignof(uint32_t) ==
               0);

  int ndim = host_shape.dimensions().size();

  absl::InlinedVector<int64_t, 4> input_strides(ndim);
  absl::c_copy(byte_strides, input_strides.begin());

  absl::InlinedVector<int64_t, 4> permutation;
  permutation.reserve(ndim + 1);

  absl::Span<int64_t const> dims;
  absl::InlinedVector<int64_t, 1> tmp_dims;
  int64_t transpose_byte_width = byte_width;
  TransposePlan::Transformation transformation =
      TransposePlan::Transformation::kNone;
  if (byte_width <= 4) {
    dims = host_shape.dimensions();
    // For scalars, add a size-1 dimension because the tiling will have 1
    // dimension.
    if (ndim == 0) {
      tmp_dims = {1};
      input_strides = {byte_width};
      permutation = {0};
      ++ndim;
      dims = tmp_dims;
    }
  } else {
    DCHECK_EQ(byte_width, 8);

    // For 8-byte types, change the element size to 4 bytes, add a trailing
    // input dimension [..., 2], and transpose it to an output major dimension.
    static_assert(tsl::port::kLittleEndian);
    TF_RET_CHECK(byte_width == 8) << byte_width;
    // Add a size-1 dimension for scalars, since it becomes a tiled dimension.
    if (ndim == 0) {
      tmp_dims = {1, 2};
      input_strides = {8, 4};
      permutation = {1, 0};
      ++ndim;
    } else {
      tmp_dims.resize(ndim + 1);
      absl::c_copy(host_shape.dimensions(), tmp_dims.begin());
      tmp_dims.back() = 2;
      permutation.push_back(ndim);
      input_strides.push_back(4);
    }
    transpose_byte_width = 4;
    dims = tmp_dims;
    if (host_shape.element_type() == F64) {
      transformation = TransposePlan::Transformation::kF64ToEf57;
    }
  }

  absl::Span<int64_t const> minor_to_major = device_layout.minor_to_major();
  for (auto it = minor_to_major.rbegin(); it != minor_to_major.rend(); ++it) {
    permutation.push_back(*it);
  }

  // For 1D outputs, callers may omit trailing tile padding in `dest`, so leave
  // `output_tiling_vec` empty and pad up to `dst.size()` below.
  absl::InlinedVector<int64_t, 2> output_tiling_vec;
  if (dims.size() > 1) {
    ABSL_ASSIGN_OR_RETURN(output_tiling_vec,
                          GetResolvedTopLevelTiling(device_shape));
  }

  const int bits_per_pack = is_fp6_split
                                ? 6
                                : (device_layout.element_size_in_bits() != 0
                                       ? device_layout.element_size_in_bits()
                                       : std::min<int64_t>(byte_width * 8, 32));
  if (!is_fp6_split && (bits_per_pack <= 0 || 32 % bits_per_pack != 0)) {
    return absl::UnimplementedError(absl::StrFormat(
        "TiledLinearizer does not support contiguous "
        "element_size_in_bits=%d%s.",
        bits_per_pack,
        bits_per_pack == 6 ? "; use bit_planes for split 6-bit TPU layouts"
                           : ""));
  }
  const int packing_factor = is_fp6_split ? 16 : (32 / bits_per_pack);
  const bool has_subtile_packing =
      HasSubtilePacking(device_layout, is_fp6_split, packing_factor);
  const uint8_t pad_byte = set_padding_to_nan ? kPadByte : uint8_t{0};

  TransposePlan::Options options;
  options.elem_size_in_bytes = transpose_byte_width;
  options.dims = dims;
  options.permutation = permutation;
  options.input_striding = TransposePlan::Striding{input_strides};
  options.output_tiling = TransposePlan::Tiling{output_tiling_vec};
  options.transformation = transformation;
  if (!has_subtile_packing && bits_per_pack < 8) {
    options.dest_bits_per_element = bits_per_pack;
  }
  options.num_threads = parallelism_options.max_parallelism;
  options.tile_padding = pad_byte;

  ABSL_ASSIGN_OR_RETURN(
      const size_t tiled_elems,
      ComputeTiledElements(dims, permutation, output_tiling_vec));
  size_t tiled_bytes = 0;
  if (options.dest_bits_per_element.has_value()) {
    ABSL_ASSIGN_OR_RETURN(
        const size_t total_bits,
        CheckedMulSizeT(tiled_elems,
                        static_cast<size_t>(*options.dest_bits_per_element),
                        "tiled packed bits"));
    tiled_bytes = CeilOfRatio<size_t>(total_bits, CHAR_BIT);
  } else {
    ABSL_ASSIGN_OR_RETURN(
        tiled_bytes,
        CheckedMulSizeT(tiled_elems, static_cast<size_t>(transpose_byte_width),
                        "tiled bytes"));
  }

  int64_t subtile_lane_count = 0;
  if (has_subtile_packing) {
    if (is_fp6_split) {
      subtile_lane_count = fp6_lane_count;
      ABSL_RETURN_IF_ERROR(Validate6BitHardwareStripSize(
          size_in_bytes, tiled_bytes, subtile_lane_count));
    } else {
      ABSL_ASSIGN_OR_RETURN(subtile_lane_count,
                            GetSubtileLaneCount(device_layout));
    }
  }

  std::shared_ptr<TransposePlan> plan;
  {
    absl::MutexLock lock(linearizer_cache->mu);
    ABSL_ASSIGN_OR_RETURN(plan, linearizer_cache->cache.GetOrCreate(options));
  }

  size_t elements_in_the_packed_data = size_in_bytes / sizeof(uint32_t);
  // The `data` from host memory has 1 int4 in 1 byte, then we need double size
  // for `scratch` to support the transpose Execute below. It is then shrunk
  // when repacking into `dest`.
  if (has_subtile_packing && bits_per_pack < 8) {
    ABSL_ASSIGN_OR_RETURN(
        const size_t packed_bits,
        CheckedMulSizeT(elements_in_the_packed_data,
                        static_cast<size_t>(CHAR_BIT), "packed scratch bits"));
    elements_in_the_packed_data =
        CeilOfRatio<size_t>(packed_bits, static_cast<size_t>(bits_per_pack));
    ABSL_RETURN_IF_ERROR(CheckedMulSizeT(elements_in_the_packed_data,
                                         sizeof(uint32_t),
                                         "scratch buffer bytes")
                             .status());
  }

  if (prefix_size > 0) {
    std::memset(dest.data(), pad_byte, prefix_size);
  }

  std::optional<Buffer> scratch;
  absl::Span<uint8_t> dst;
  if (has_subtile_packing && bits_per_pack < 8) {
    scratch.emplace(elements_in_the_packed_data, /*value=*/std::nullopt,
                    &tensorflow::tpu::DefaultAllocator);
    dst = scratch->mutable_data<uint8_t>();
  } else {
    dst = dest.subspan(prefix_size);
  }

  // `TransposePlan` writes and pads the tiled output `[0, tiled_bytes)`. Fill
  // any trailing padding in `dst` beyond that, such as
  // `tail_padding_alignment_in_elements` or 1D padding.
  TF_RET_CHECK(dst.size() >= tiled_bytes);
  if (dst.size() > tiled_bytes) {
    std::memset(dst.data() + tiled_bytes, pad_byte, dst.size() - tiled_bytes);
  }
  if (plan->Parallelism() <= 1) {
    plan->Execute(data, dst.data());
  } else {
    parallelism_options.run_plan(*plan, data, dst.data());
  }
  // TODO(phawkins): it would be preferable to sink the repacking into the
  // transpose computation to improve cache locality.

  // TODO(b/188059010, b/194422623): Ideally we want to check LaneCount ==
  // kTcLaneCount here, but chip variants without TC have zero LaneCount, and
  // it's not clear what that means.
  // TODO (b/274171729): For SparseCore currently the data is being packed
  // linearly.
  if (has_subtile_packing) {
    ABSL_RETURN_IF_ERROR(TcLaneCountSwitch(
        [&](auto kTcLaneCount) {
          if (bits_per_pack == 6) {
            Repack6BitToHardwareLayout<kTcLaneCount>(
                dst, dest.subspan(prefix_size), set_padding_to_nan);
            return;
          }
          ElementPackingFactorSwitch<void>(
              [&](auto kPackingFactor) {
                if constexpr (kPackingFactor > 1) {
                  RepackToHardwareLayout<kPackingFactor, kTcLaneCount>(
                      dst,
                      tsl::safe_reinterpret_cast<uint32_t*>(
                          dest.subspan(prefix_size).data()),
                      set_padding_to_nan);
                }
              },
              packing_factor);
        },
        subtile_lane_count));
  }

  if (!dynamic_sizes.empty()) {
    absl::c_copy(dynamic_sizes,
                 tsl::safe_reinterpret_cast<uint32_t*>(dest.data()));
  }

  return absl::OkStatus();
}

/* static */ absl::Status TiledLinearizer::X64FromTuple(
    const LiteralSlice& low, const LiteralSlice& high,
    MutableBorrowingLiteral literal) {
  tsl::profiler::TraceMe trace_me(
      [&] {
        return absl::StrCat("X64FromTuple ",
                            ShapeUtil::HumanStringWithLayout(literal.shape()));
      },
      /*level=*/2);
  TF_RET_CHECK(ShapeUtil::ElementHasBitWidth(literal.shape(), 64));
  Shape component_shape = X64ComponentShape(literal.shape());
  TF_RET_CHECK(ShapeUtil::Compatible(low.shape(), component_shape));
  TF_RET_CHECK(ShapeUtil::Compatible(high.shape(), component_shape));
  switch (literal.shape().element_type()) {
    case C64: {
      LiteralSlice real = low;
      LiteralSlice imag = high;
      ABSL_RETURN_IF_ERROR(
          literal.Populate<complex64>([&](absl::Span<const int64_t> indices) {
            return complex64(real.Get<float>(indices),
                             imag.Get<float>(indices));
          }));
      break;
    }
    case S64:
      ABSL_RETURN_IF_ERROR(
          literal.Populate<int64_t>([&](absl::Span<const int64_t> indices) {
            return static_cast<int64_t>(
                low.Get<uint32_t>(indices) +
                (static_cast<uint64_t>(high.Get<uint32_t>(indices)) << 32));
          }));
      break;
    case U64:
      ABSL_RETURN_IF_ERROR(
          literal.Populate<uint64_t>([&](absl::Span<const int64_t> indices) {
            return low.Get<uint32_t>(indices) +
                   (static_cast<uint64_t>(high.Get<uint32_t>(indices)) << 32);
          }));
      break;
    case F64:
      ABSL_RETURN_IF_ERROR(
          literal.Populate<double>([&](absl::Span<const int64_t> indices) {
            auto most_significant = low.Get<float>(indices);
            auto least_significant = high.Get<float>(indices);
            return double{least_significant} + double{most_significant};
          }));
      break;
    default:
      return Unimplemented("Unsupported complex type: %s",
                           PrimitiveType_Name(literal.shape().element_type()));
  }
  return absl::OkStatus();
}

/* static */ absl::Status TiledLinearizer::X128FromTuple(
    const LiteralSlice& low, const LiteralSlice& high,
    MutableBorrowingLiteral literal) {
  tsl::profiler::TraceMe trace_me(
      [&] {
        return absl::StrCat("X128FromTuple ",
                            ShapeUtil::HumanStringWithLayout(literal.shape()));
      },
      /*level=*/2);
  TF_RET_CHECK(ShapeUtil::ElementHasBitWidth(literal.shape(), 128));
  Shape component_shape = X128ComponentShape(literal.shape());
  TF_RET_CHECK(component_shape.element_type() == F64);
  TF_RET_CHECK(ShapeUtil::Compatible(low.shape(), component_shape));
  TF_RET_CHECK(ShapeUtil::Compatible(high.shape(), component_shape));
  switch (literal.shape().element_type()) {
    case C128: {
      LiteralSlice real = low;
      LiteralSlice imag = high;
      ABSL_RETURN_IF_ERROR(
          literal.Populate<complex128>([&](absl::Span<const int64_t> indices) {
            return complex128(real.Get<double>(indices),
                              imag.Get<double>(indices));
          }));
      break;
    }
    default:
      return Unimplemented("Unsupported type: %s",
                           PrimitiveType_Name(literal.shape().element_type()));
  }
  return absl::OkStatus();
}

namespace {

absl::Status DelinearizeComponents(
    int bitwidth, const Shape& literal_shape, absl::Span<const uint8_t> data,
    const Layout& device_layout, MutableLiteralBase* literal,
    const TiledLinearizer::ParallelismOptions& parallelism_options) {
  Shape device_shape = literal_shape;
  (*device_shape.mutable_layout()) = device_layout;
  Shape component_subshape;
  Shape component_subshape_device;
  if (bitwidth == 64) {
    component_subshape = X64ComponentShape(literal_shape);
    component_subshape_device = X64ComponentShape(device_shape);
  } else if (bitwidth == 128) {
    component_subshape = X128ComponentShape(literal_shape);
    component_subshape_device = X128ComponentShape(device_shape);
  } else {
    return absl::InternalError(
        "Unsupported bitwidth for tuple delinearization.");
  }

  int64_t component_size_bytes =
      ShapeUtil::ArraySize(component_subshape_device);
  Literal low_tmp(component_subshape);
  Literal high_tmp(component_subshape);
  ABSL_RETURN_IF_ERROR(TiledLinearizer::Delinearize(
      low_tmp.shape(), data.subspan(0, component_size_bytes), device_layout,
      &low_tmp, parallelism_options));

  ABSL_RETURN_IF_ERROR(TiledLinearizer::Delinearize(
      high_tmp.shape(),
      data.subspan(component_size_bytes, component_size_bytes), device_layout,
      &high_tmp, parallelism_options));

  if (bitwidth == 64) {
    return TiledLinearizer::X64FromTuple(low_tmp, high_tmp, literal);
  }
  return TiledLinearizer::X128FromTuple(low_tmp, high_tmp, literal);
}

ABSL_ATTRIBUTE_FUNC_ALIGN(64)
absl::Status DelinearizeFp6SplitR0OrR1(const Shape& literal_shape,
                                       absl::Span<const uint32_t> values,
                                       int64_t lane_count,
                                       MutableLiteralBase* literal) {
  DCHECK(literal_shape.IsArray());
  DCHECK_LE(literal_shape.dimensions().size(), 1);
#ifndef NDEBUG
  CreateGarbageFilledLiteral(literal_shape, literal);
#endif
  auto output = GetLiteralAsMutableSpan(literal);
  ABSL_RETURN_IF_ERROR(Validate6BitHardwareStripSize(
      values.size() * sizeof(uint32_t), output.size(), lane_count));
  return TcLaneCountSwitch(
      [&](auto kLaneCount) {
        Repack6BitToHostLayout<kLaneCount>(values, output);
      },
      lane_count);
}

template <int kPackingFactor>
ABSL_ATTRIBUTE_FUNC_ALIGN(64)
absl::Status
    DelinearizeR1(const Shape& literal_shape, absl::Span<const uint32_t> values,
                  const Layout& device_layout, MutableLiteralBase* literal) {
  CHECK(!HasLinearLayout(device_layout));
  DCHECK(literal_shape.IsArray());
  DCHECK_EQ(literal_shape.dimensions().size(), 1);
#ifndef NDEBUG
  CreateGarbageFilledLiteral(literal_shape, literal);
#endif
  auto output = GetLiteralAsMutableSpan(literal);
  const auto tiles = device_layout.tiles();
  if (tiles.size() <= 1) {
    // Transfer in host layout or 1D SparseCore layout. No subtile pack/unpack.
    RepackFromSparseCoreLayout<kPackingFactor>(values, output);
    return absl::OkStatus();
  }
  ABSL_ASSIGN_OR_RETURN(const int64_t lane_count,
                        GetSubtileLaneCount(device_layout));
  return TcLaneCountSwitch(
      [&](auto kLaneCount) {
        RepackToHostLayout<kPackingFactor, kLaneCount>(values, output);
      },
      lane_count);
}

absl::Status DelinearizeUsingTranspose(
    const Shape& literal_shape, absl::Span<const uint8_t> data,
    const Layout& device_layout, int64_t fp6_lane_count,
    MutableLiteralBase* literal,
    const TiledLinearizer::ParallelismOptions& parallelism_options) {
  tsl::profiler::TraceMe traceme("DelinearizeUsingTranspose");
  DCHECK_OK(LayoutUtil::ValidateLayoutForShape(device_layout, literal_shape));
  TF_RET_CHECK(!device_layout.tiles().empty());

#ifndef NDEBUG
  CreateGarbageFilledLiteral(literal_shape, literal);
#endif

  const bool is_fp6_split = fp6_lane_count > 0;

  // The fast path does not parallelize. Above a certain size it's worth using
  // the transpose plan to get parallelism.
  static const int64_t kFastPathMaxBytes = int64_t{1} << 26;
  int64_t size_bytes = literal->size_bytes();

  const int64_t element_size_in_bits =
      is_fp6_split ? 6
                   : (device_layout.element_size_in_bits() != 0
                          ? device_layout.element_size_in_bits()
                          : ShapeUtil::ByteSizeOfPrimitiveType(
                                literal_shape.element_type()) *
                                CHAR_BIT);
  if (!is_fp6_split &&
      (element_size_in_bits <= 0 || 32 % element_size_in_bits != 0)) {
    return absl::UnimplementedError(
        absl::StrFormat("TiledLinearizer does not support contiguous "
                        "element_size_in_bits=%d%s.",
                        element_size_in_bits,
                        element_size_in_bits == 6
                            ? "; use bit_planes for split 6-bit TPU layouts"
                            : ""));
  }
  int packing_factor = is_fp6_split ? 16 : (32 / element_size_in_bits);

  if (!is_fp6_split && packing_factor <= 4 && HasLinearLayout(device_layout) &&
      Layout::Equal().MinorToMajorOnly()(device_layout,
                                         literal_shape.layout()) &&
      size_bytes <= kFastPathMaxBytes) {
    tsl::profiler::TraceMe trace_me("the memcpy");

    std::memcpy(literal->untyped_data(), data.data(), size_bytes);
    return absl::OkStatus();
  }

  if (is_fp6_split && data.size() % sizeof(uint32_t) != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "6-bit split hardware buffer size (%d bytes) must be 4-byte aligned.",
        data.size()));
  }

  auto values = absl::MakeConstSpan(
      tsl::safe_reinterpret_cast<const uint32_t*>(data.data()),
      data.size() / sizeof(uint32_t));

  // Fast path for R0/R1 FP6 split tensors and R1 packed tensors.
  if (is_fp6_split && literal_shape.dimensions().size() <= 1 &&
      !HasLinearLayout(device_layout) && size_bytes <= kFastPathMaxBytes) {
    return DelinearizeFp6SplitR0OrR1(literal_shape, values, fp6_lane_count,
                                     literal);
  }
  if (literal_shape.dimensions().size() == 1 &&
      !HasLinearLayout(device_layout) && size_bytes <= kFastPathMaxBytes) {
    return ElementPackingFactorSwitch<absl::Status>(
        [&](auto kPackingFactor) {
          return DelinearizeR1<kPackingFactor>(literal_shape, values,
                                               device_layout, literal);
        },
        packing_factor);
  }

  Shape device_shape = literal_shape;
  *(device_shape.mutable_layout()) = device_layout;

  absl::InlinedVector<int64_t, 4> input_dims(literal_shape.dimensions().size());
  absl::InlinedVector<int64_t, 4> input_logical_dim_to_physical_dim(
      input_dims.size());
  auto dim_it = input_dims.rbegin();
  int physical_dim = input_dims.size();
  for (int logical_dim : device_layout.minor_to_major()) {
    --physical_dim;
    input_logical_dim_to_physical_dim[logical_dim] = physical_dim;
    *dim_it = literal_shape.dimensions(logical_dim);
    ++dim_it;
  }

  absl::InlinedVector<int64_t, 2> input_tiling_vec;
  if (literal_shape.dimensions().size() > 1) {
    ABSL_ASSIGN_OR_RETURN(input_tiling_vec,
                          GetResolvedTopLevelTiling(device_shape));
  }
  ABSL_ASSIGN_OR_RETURN(
      const size_t input_tiled_elems,
      ComputeTiledElements(input_dims, /*permutation=*/std::nullopt,
                           input_tiling_vec));

  // minor_to_major maps from physical dim to logical dim.
  // permutation maps from output physical dims to input physical dims.
  absl::InlinedVector<int64_t, 4> permutation(
      literal_shape.dimensions().size());
  auto it = permutation.rbegin();
  for (int dim : literal_shape.layout().minor_to_major()) {
    *it = input_logical_dim_to_physical_dim[dim];
    ++it;
  }

  int byte_width = primitive_util::ByteWidth(literal_shape.element_type());

  TransposePlan::Options options;
  options.elem_size_in_bytes = byte_width;
  options.dims = input_dims;
  options.permutation = permutation;
  options.input_tiling = TransposePlan::Tiling{input_tiling_vec};
  options.output_tiling = TransposePlan::Tiling{};
  options.transformation = TransposePlan::Transformation::kNone;
  options.num_threads = parallelism_options.max_parallelism;
  std::shared_ptr<TransposePlan> plan;
  {
    absl::MutexLock lock(linearizer_cache->mu);
    ABSL_ASSIGN_OR_RETURN(plan, linearizer_cache->cache.GetOrCreate(options));
  }

  // TODO(phawkins): sink unpacking code into transpose plan.
  Buffer unpacked_values;
  const bool has_subtile_packing =
      HasSubtilePacking(device_layout, is_fp6_split, packing_factor);
  // TODO (b/274171729): For SparseCore currently the data is being packed
  // linearly.
  if (has_subtile_packing) {
    int64_t lane_count = 0;
    size_t fp6_unpacked_bytes = 0;
    if (is_fp6_split) {
      lane_count = fp6_lane_count;
      ABSL_ASSIGN_OR_RETURN(
          const size_t required_host_bytes,
          CheckedMulSizeT(input_tiled_elems, static_cast<size_t>(byte_width),
                          "delinearize host bytes"));
      ABSL_RETURN_IF_ERROR(Validate6BitHardwareStripSize(
          data.size(), required_host_bytes, lane_count));
      const size_t num_strips =
          values.size() / (size_t{3} * static_cast<size_t>(lane_count));
      ABSL_ASSIGN_OR_RETURN(
          fp6_unpacked_bytes,
          CheckedMulSizeT(num_strips,
                          size_t{16} * static_cast<size_t>(lane_count),
                          "unpacked 6-bit host bytes"));
    } else {
      ABSL_ASSIGN_OR_RETURN(lane_count, GetSubtileLaneCount(device_layout));
    }
    ABSL_RETURN_IF_ERROR(TcLaneCountSwitch(
        [&](auto kTcLaneCount) {
          if (is_fp6_split) {
            unpacked_values = Buffer(fp6_unpacked_bytes);
            Repack6BitToHostLayout<kTcLaneCount>(
                values, unpacked_values.mutable_data<uint8_t>());
            values = unpacked_values.const_data<uint32_t>();
            return;
          }
          ElementPackingFactorSwitch<void>(
              [&](auto kPackingFactor) {
                if constexpr (kPackingFactor > 1) {
                  constexpr int kTpuToCpuExpansionFactor =
                      CeilOfRatio<size_t>(kPackingFactor, sizeof(uint32_t));
                  unpacked_values =
                      Buffer(values.size() * kTpuToCpuExpansionFactor *
                             sizeof(uint32_t));
                  RepackToHostLayout<kPackingFactor, kTcLaneCount>(
                      values, unpacked_values.mutable_data<uint8_t>());
                  values = unpacked_values.const_data<uint32_t>();
                }
              },
              packing_factor);
        },
        lane_count));
  } else if (const int bits_per_pack = 32 / packing_factor; bits_per_pack < 8) {
    ElementPackingFactorSwitch<void>(
        [&](auto kPackingFactor) {
          if constexpr (kPackingFactor > 4) {
            constexpr int kTpuToCpuExpansionFactor =
                CeilOfRatio<size_t>(kPackingFactor, sizeof(uint32_t));
            unpacked_values = Buffer(values.size() * kTpuToCpuExpansionFactor *
                                     sizeof(uint32_t));
            RepackFromSparseCoreLayout<kPackingFactor>(
                values, unpacked_values.mutable_data<uint8_t>());
            values = unpacked_values.const_data<uint32_t>();
          }
        },
        packing_factor);
  }

  if (plan->Parallelism() <= 1) {
    plan->Execute(values.data(), literal->untyped_data());
  } else {
    parallelism_options.run_plan(*plan, values.data(), literal->untyped_data());
  }
  return absl::OkStatus();
}

}  // namespace

/* static */ ABSL_ATTRIBUTE_FUNC_ALIGN(64) absl::Status
    TiledLinearizer::Delinearize(
        const Shape& literal_shape, absl::Span<const uint8_t> data,
        const Layout& device_layout, MutableLiteralBase* literal,
        const TiledLinearizer::ParallelismOptions& parallelism_options) {
  tsl::profiler::TraceMe trace_me(
      [&] {
        return tsl::profiler::TraceMeEncode(
            "Delinearize", {{"shape", literal_shape.ToString()},
                            {"layout", device_layout.ToString()}});
      },
      /*level=*/2);
  if (literal_shape.IsToken()) {
    return absl::OkStatus();
  }
  if (!literal_shape.IsArray()) {
    return Unimplemented("Unsupported shape for Delinearize.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const int64_t fp6_lane_count,
      ValidateAndGetFp6BitPlaneLaneCount(literal_shape, device_layout));
  TF_RET_CHECK(reinterpret_cast<uintptr_t>(data.data()) % alignof(uint32_t) ==
               0);
  VLOG(2) << "delinearized: " << ShapeUtil::HumanString(literal_shape) << " @ "
          << LayoutUtil::HumanString(device_layout) << " {"
          << FormatLinearizedLiteral(
                 literal_shape,
                 absl::MakeConstSpan(
                     tsl::safe_reinterpret_cast<const uint32_t*>(data.data()),
                     data.size() / sizeof(uint32_t)))
          << "}";
  // Zero-element arrays carry no tiles in their device layout and have nothing
  // to delinearize.
  if (ShapeUtil::IsZeroElementArray(literal_shape)) {
    return absl::OkStatus();
  }
  const int bitwidth = primitive_util::BitWidth(literal_shape.element_type());
  if (bitwidth == 64 || bitwidth == 128) {
    return DelinearizeComponents(bitwidth, literal_shape, data, device_layout,
                                 literal, parallelism_options);
  }
  return DelinearizeUsingTranspose(literal_shape, data, device_layout,
                                   fp6_lane_count, literal,
                                   parallelism_options);
}

}  // namespace xla
