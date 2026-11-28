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

#ifndef XLA_BACKENDS_CPU_COLLECTIVES_CPU_COLLECTIVE_REDUCTION_H_
#define XLA_BACKENDS_CPU_COLLECTIVES_CPU_COLLECTIVE_REDUCTION_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "Eigen/Core"
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/primitive_util.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

// Rejects non-array primitive types (TUPLE, TOKEN, OPAQUE_TYPE, INVALID) and
// sub-byte non-PRED packed array types (S1/U1/S2/U2/S4/U4/F4/F6) whose storage
// is smaller than one byte per element.
inline absl::Status ValidateByteArrayDataType(PrimitiveType dtype) {
  if (!primitive_util::IsArrayType(dtype) ||
      primitive_util::IsSubByteNonPredType(dtype)) {
    return Unimplemented("Unsupported collective datatype: %s",
                         primitive_util::LowercasePrimitiveTypeName(dtype));
  }
  return absl::OkStatus();
}

// Computes `count * byte_width` with overflow checking.
inline absl::StatusOr<size_t> CheckedCollectiveByteSize(size_t count,
                                                        size_t byte_width) {
  if (byte_width > 0 &&
      count > std::numeric_limits<size_t>::max() / byte_width) {
    return absl::InvalidArgumentError("Collective buffer byte size overflow");
  }
  return count * byte_width;
}

// Returns true iff half-open byte ranges `[a, a + a_bytes)` and
// `[b, b + b_bytes)` intersect.
inline bool ByteRangesOverlap(const void* a, size_t a_bytes, const void* b,
                              size_t b_bytes) {
  if (a_bytes == 0 || b_bytes == 0 || a == nullptr || b == nullptr) {
    return false;
  }
  const uintptr_t a0 = reinterpret_cast<uintptr_t>(a);
  const uintptr_t b0 = reinterpret_cast<uintptr_t>(b);
  return a0 < b0 ? (b0 - a0 < a_bytes) : (a0 - b0 < b_bytes);
}

// Copies `num_bytes` from `src` to `dst`, skipping empty and self copies.
inline void CopyBytesIfNeeded(void* dst, const void* src, size_t num_bytes) {
  if (num_bytes > 0 && dst != src) {
    std::memcpy(dst, src, num_bytes);
  }
}

// Computes IEEE 754-2019 `minimum` (`kIsMax == false`) or `maximum`
// (`kIsMax == true`): propagates NaNs (quieting signaling NaNs on formats that
// carry them), orders `-0.0 < +0.0`, and compares subnormals using pure integer
// bit operations so host FTZ/DAZ state cannot alter results.
template <typename T, bool kIsMax>
inline T IeeeMinMax(T a, T b) {
  if constexpr (!is_specialized_floating_point_v<T>) {
    return kIsMax ? std::max(a, b) : std::min(a, b);
  } else if constexpr (!std::numeric_limits<T>::has_quiet_NaN) {
    const uint8_t ua = absl::bit_cast<uint8_t>(a);
    const uint8_t ub = absl::bit_cast<uint8_t>(b);
    constexpr int kBits = std::is_same_v<T, tsl::float4_e2m1fn> ? 4 : 6;
    constexpr uint8_t kSignBit = uint8_t{1} << (kBits - 1);
    constexpr uint8_t kMask = static_cast<uint8_t>((uint8_t{1} << kBits) - 1);
    const uint8_t ka = (ua & kSignBit) ? static_cast<uint8_t>(~ua & kMask)
                                       : static_cast<uint8_t>(ua | kSignBit);
    const uint8_t kb = (ub & kSignBit) ? static_cast<uint8_t>(~ub & kMask)
                                       : static_cast<uint8_t>(ub | kSignBit);
    return absl::bit_cast<T>(kIsMax ? (ka < kb ? ub : ua)
                                    : (kb < ka ? ub : ua));
  } else {
    using Bits =
        typename Eigen::numext::get_integer_by_size<sizeof(T)>::unsigned_type;
    const Bits ua = absl::bit_cast<Bits>(a);
    const Bits ub = absl::bit_cast<Bits>(b);
    if constexpr (!std::numeric_limits<T>::is_signed) {
      // Unsigned exponent-only format (float8_e8m0fnu): 0xFF is the sole NaN.
      constexpr Bits kNan = std::numeric_limits<Bits>::max();
      if (ua == kNan || ub == kNan) {
        return absl::bit_cast<T>(kNan);
      }
      return absl::bit_cast<T>(kIsMax ? std::max(ua, ub) : std::min(ua, ub));
    } else {
      constexpr int kTotalBits = sizeof(Bits) * 8;
      constexpr Bits kSignBit = Bits{1} << (kTotalBits - 1);
      constexpr Bits kAbsMask = static_cast<Bits>(kSignBit - 1);
      if constexpr (!has_negative_zero_v<T>) {
        // FNUZ formats encode their sole NaN at 0x80 (sign bit set, payload 0).
        if (ua == kSignBit || ub == kSignBit) {
          return absl::bit_cast<T>(kSignBit);
        }
      } else if constexpr (!std::numeric_limits<T>::has_infinity) {
        // float8_e4m3fn encodes NaN at 0x7F and 0xFF with no signaling NaNs.
        if ((ua & kAbsMask) == kAbsMask) {
          return a;
        }
        if ((ub & kAbsMask) == kAbsMask) {
          return b;
        }
      } else {
        constexpr int kMantissaBits = std::numeric_limits<T>::digits - 1;
        static_assert(kMantissaBits >= 1 && kMantissaBits < kTotalBits - 1);
        constexpr Bits kMantissaMask =
            static_cast<Bits>((Bits{1} << kMantissaBits) - 1);
        constexpr Bits kInfBits = static_cast<Bits>(kAbsMask ^ kMantissaMask);
        constexpr Bits kQuietBit =
            static_cast<Bits>(Bits{1} << (kMantissaBits - 1));
        if ((ua & kAbsMask) > kInfBits) {
          return absl::bit_cast<T>(static_cast<Bits>(ua | kQuietBit));
        }
        if ((ub & kAbsMask) > kInfBits) {
          return absl::bit_cast<T>(static_cast<Bits>(ub | kQuietBit));
        }
      }
      const Bits ka = (ua & kSignBit) ? static_cast<Bits>(~ua)
                                      : static_cast<Bits>(ua | kSignBit);
      const Bits kb = (ub & kSignBit) ? static_cast<Bits>(~ub)
                                      : static_cast<Bits>(ub | kSignBit);
      if constexpr (kIsMax) {
        return absl::bit_cast<T>(ka < kb ? ub : ua);
      } else {
        return absl::bit_cast<T>(kb < ka ? ub : ua);
      }
    }
  }
}

// Elementwise reduction adapter matching Gloo's `ReduceFunction` signature.
template <typename T, bool kIsMax>
void IeeeMinMaxReduce(void* c, const void* a, const void* b, size_t n) {
  T* out = static_cast<T*>(c);
  const T* lhs = static_cast<const T*>(a);
  const T* rhs = static_cast<const T*>(b);
  for (size_t i = 0; i < n; ++i) {
    out[i] = IeeeMinMax<T, kIsMax>(lhs[i], rhs[i]);
  }
}

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_COLLECTIVES_CPU_COLLECTIVE_REDUCTION_H_
