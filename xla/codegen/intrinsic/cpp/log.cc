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

#if defined(__has_attribute) && __has_attribute(ext_vector_type) && \
    defined(__has_builtin) && __has_builtin(__builtin_vectorelements)

#include "xla/codegen/intrinsic/cpp/log.h"

#include <cstdint>
#include <limits>
#include <type_traits>

#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {
namespace {

template <typename VecT, typename SintVecT>
inline VecT SintToDouble(SintVecT x) {
  if constexpr (std::is_arithmetic_v<VecT>) {
    return static_cast<VecT>(x);
  } else {
    return __builtin_convertvector(x, VecT);
  }
}

// Computes natural log (kBase2 = false) or base-2 log (kBase2 = true) for
// float64 within 1 ULP across normal inputs, flushing subnormals to zero.
//
// Algorithm overview:
// 1. Subnormal flush-to-zero (FTZ): Inputs with |x| < 2^-1022 are flushed to
//    +/-0.0, which maps to -infinity.
// 2. Range reduction: We write x = 2^e * (1 + f) with 1 + f in
//    [sqrt(2)/2, sqrt(2)]. Adding (1.0 - sqrt(2)/2) in the integer bit
//    representation shifts the threshold to the exponent boundary so that both
//    [1, sqrt(2)) and [sqrt(2), 2) fold into [sqrt(2)/2, sqrt(2)] without a
//    floating-point branch. Because 1 + f is in [sqrt(2)/2, sqrt(2)],
//    f = (1 + f) - 1.0 is exact by Sterbenz's lemma.
// 3. Padé transform and Sollya minimax polynomial: Let z = f / (2 + f) and
//    h = 0.5 * f^2, so 1 + f = (1 + z) / (1 - z) with |z| <= 0.171573. Using
//    the exact identity 2*z = f - h + z * h, we evaluate:
//      ln(1 + f) = 2*z + z^3 * P(z^2) = f - (h - z * (h + z^2 * P(z^2))).
//    This keeps the division rounding error of z multiplied by h <= 0.086.
//    Sollya certifies a 7-term minimax polynomial P(z^2) to 0.041 ULP:
//      prec = 200!;
//      dom_z = [1b-40; (sqrt(2) - 1) / (sqrt(2) + 1)];
//      p_ln = fpminimax(log((1 + x) / (1 - x)) - 2*x,
//                       [|3, 5, 7, 9, 11, 13, 15|], [|D...|], dom_z);
//      dirtyinfnorm((2*x + p_ln) / log((1 + x) / (1 - x)) - 1, dom_z) * 2^53;
//      // -> 0.0406 ULP
// 4. Reconstruction: FastTwoSum(f, -r) splits f - r (accurate to ~55 bits)
//    into (ln_hi, ln_lo) without additional rounding error.
//    - For log(x) = e * ln(2) + ln(1 + f), ln(2) is split into a 36-bit high
//      part kLn2Hi and a 53-bit low part kLn2Lo so e * kLn2Hi is exact.
//    - For log2(x) = e + log2(e) * ln(1 + f), ln_hi is split into a 27-bit
//      high part ln_hi_top and 1 / ln(2) is split into a 26-bit high part
//      kLog2eHi and a 53-bit low part kLog2eLo so ln_hi_top * kLog2eHi is
//      exact even without hardware FMA. Continuous error stays under 1 ULP on
//      hardware-FMA targets (< 0.96 ULP) and reaches at most 1.003 ULP right
//      below sqrt(2) without FMA (within 1 integer ULP in both cases).
template <bool kBase2, typename VecT>
inline VecT LogBaseF64Impl(VecT x) {
  using UintVecT = typename internal::CorrespondingIntVector<VecT>::type;
  using SintVecT = typename internal::CorrespondingSignedIntVector<VecT>::type;

  constexpr uint64_t kAbsMask = 0x7fffffffffffffffULL;
  constexpr uint64_t kSignMask = 0x8000000000000000ULL;
  constexpr uint64_t kMinNormalBits = 0x0010000000000000ULL;
  constexpr uint64_t kMantissaMask = 0x000fffffffffffffULL;
  constexpr uint64_t kOneBits = 0x3ff0000000000000ULL;
  constexpr uint64_t kHalfSqrt2Bits = 0x3fe6a09e667f3bcdULL;
  constexpr uint64_t kReduceOffset = kOneBits - kHalfSqrt2Bits;

  // Step 1: Flush subnormal inputs (|x| < 2^-1022) to +/-0.0.
  UintVecT bits = __builtin_bit_cast(UintVecT, x);
  UintVecT abs_bits = bits & Splat<UintVecT>(kAbsMask);
  UintVecT ftz_bits = (abs_bits < Splat<UintVecT>(kMinNormalBits))
                          ? (bits & Splat<UintVecT>(kSignMask))
                          : bits;
  VecT x_ftz = __builtin_bit_cast(VecT, ftz_bits);

  // Step 2: Range reduction x = 2^e * (1 + f) with 1 + f in [sqrt(2)/2,
  // sqrt(2)].
  UintVecT u = ftz_bits + Splat<UintVecT>(kReduceOffset);
  SintVecT e_int =
      __builtin_bit_cast(SintVecT, u >> 52) - Splat<SintVecT>(1023);
  UintVecT m_bits =
      (u & Splat<UintVecT>(kMantissaMask)) + Splat<UintVecT>(kHalfSqrt2Bits);
  VecT m = __builtin_bit_cast(VecT, m_bits);
  VecT f = m - Splat<VecT>(1.0);
  VecT e = SintToDouble<VecT>(e_int);

  // Step 3: Padé transform z = f / (2 + f) and degree-6 polynomial in w = z^2.
  VecT z = f / (Splat<VecT>(2.0) + f);
  VecT h = Splat<VecT>(0.5) * f * f;
  VecT w = z * z;

  VecT poly = HornerPoly(w,
                         0x1.2b58abc69fe17p-3,   // z^15
                         0x1.39fe5b50fa392p-3,   // z^13
                         0x1.7462b4ab848ffp-3,   // z^11
                         0x1.c71c62e6848e5p-3,   // z^9
                         0x1.2492492deec38p-2,   // z^7
                         0x1.9999999995323p-2,   // z^5
                         0x1.5555555555558p-1);  // z^3

  // r = h - z * (h + w * poly), so ln(1 + f) ~ f - r.
  VecT r = Fmuladd(-z, Fmuladd(w, poly, h), h);
  auto [ln_hi, ln_lo] = FastTwoSum(f, -r);

  // Step 4: Reconstruct log(x) or log2(x).
  VecT result;
  if constexpr (kBase2) {
    constexpr uint64_t kHigh27BitsMask = 0xfffffffffc000000ULL;
    constexpr double kLog2eHi = 0x1.7154768p+0;
    constexpr double kLog2eLo = -0x1.6a3e80f444178p-27;
    VecT ln_hi_top =
        __builtin_bit_cast(VecT, __builtin_bit_cast(UintVecT, ln_hi) &
                                     Splat<UintVecT>(kHigh27BitsMask));
    VecT ln_lo_top = (ln_hi - ln_hi_top) + ln_lo;
    VecT prod_hi = ln_hi_top * Splat<VecT>(kLog2eHi);
    VecT prod_lo = Fmuladd(ln_lo_top, Splat<VecT>(kLog2eHi),
                           ln_hi * Splat<VecT>(kLog2eLo));
    auto [head, head_err] = FastTwoSum(e, prod_hi);
    result = head + (head_err + prod_lo);
  } else {
    constexpr double kLn2Hi = 0x1.62e42fefa0000p-1;
    constexpr double kLn2Lo = 0x1.cf79abc9e3b3ap-40;
    VecT e_ln2_hi = e * Splat<VecT>(kLn2Hi);
    auto [head, head_err] = FastTwoSum(e_ln2_hi, ln_hi);
    VecT tail = head_err + Fmuladd(e, Splat<VecT>(kLn2Lo), ln_lo);
    result = head + tail;
  }

  // Step 5: Handle special values (+/-0.0, flushed subnormals, +inf, < 0, NaN).
  constexpr double kInf = std::numeric_limits<double>::infinity();
  constexpr double kNegInf = -std::numeric_limits<double>::infinity();
  constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

  result = (x_ftz == Splat<VecT>(0.0)) ? Splat<VecT>(kNegInf) : result;
  result = (x_ftz == Splat<VecT>(kInf)) ? Splat<VecT>(kInf) : result;
  result = !(x_ftz >= Splat<VecT>(0.0)) ? Splat<VecT>(kNan) : result;
  return result;
}

}  // namespace

double log_f64(double x) { return LogBaseF64Impl<false>(x); }
Vec2d log_v2f64(Vec2d x) { return LogBaseF64Impl<false>(x); }
Vec4d log_v4f64(Vec4d x) { return LogBaseF64Impl<false>(x); }
Vec8d log_v8f64(Vec8d x) { return LogBaseF64Impl<false>(x); }

double log2_f64(double x) { return LogBaseF64Impl<true>(x); }
Vec2d log2_v2f64(Vec2d x) { return LogBaseF64Impl<true>(x); }
Vec4d log2_v4f64(Vec4d x) { return LogBaseF64Impl<true>(x); }
Vec8d log2_v8f64(Vec8d x) { return LogBaseF64Impl<true>(x); }

}  // namespace xla::codegen

#endif  // defined(__has_attribute) && __has_attribute(ext_vector_type) &&
        // defined(__has_builtin) && __has_builtin(__builtin_vectorelements)
