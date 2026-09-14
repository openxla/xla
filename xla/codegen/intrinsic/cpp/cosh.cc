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

#include "xla/codegen/intrinsic/cpp/cosh.h"

#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {

namespace {

template <typename T>
inline T CoshF32(T x_in) {
  // Computes cosh(x) = (e^x + e^-x) / 2 in float32 within 1 ULP.
  // Because cosh(-x) = cosh(x), we work with |x| and split into two cases:
  //
  // - Small inputs (|x| < ln(2)):
  //   Let x2 = fl(|x| * |x|). We approximate cosh(|x|) with a degree-8
  //   polynomial:
  //     1 + (0.5 * x2 + x2^2 * P(x2))
  //   Evaluating 0.5 * x2 + x2^2 * P(x2) before adding 1.0 keeps the x2^2
  //   terms from being lost when added to 1.0.
  //
  // - Large inputs (|x| >= ln(2)):
  //   1. Write |x| = n * ln(2) + r, where n is the integer nearest to
  //      |x| / ln(2) and |r| <= ln(2) / 2. We subtract n * ln(2) in two pieces
  //      (ln2_hi and ln2_lo) so that |x| - n * ln2_hi does not round.
  //   2. Since e^|x| = 2^n * e^r, we compute 0.25 * e^|x| = 2^(n - 2) * e^r.
  //      We approximate e^r with 1 + r + r^2 * P_exp(r), split 2^(n - 2) * e^r
  //      into a main part (head) and a small correction (tail), and compute
  //      0.25 * e^-|x| as 0.0625 / (head + tail).
  //   3. Combine them as cosh(|x|) = 2 * (head + (tail + div_term)). Working
  //      with 0.25 * e^|x| and multiplying by 2 at the end prevents e^|x| from
  //      overflowing to infinity early when |x| is near 89.4 (where e^|x|
  //      exceeds FLT_MAX, but cosh(|x|) still fits in float32).
  using IntType = typename internal::CorrespondingIntVector<T>::type;

  T abs_x = BitwiseAbs(x_in);
  auto is_nan_or_inf = abs_x != abs_x;

  // Clamp abs_x to 89.5f so that inputs in the overflow range [89.4160, 89.5]
  // still evaluate to +Inf while avoiding intermediate overflow for huge
  // finite inputs.
  T x = Clamp(abs_x, -89.5f, 89.5f);
  x = is_nan_or_inf ? abs_x : x;

  // Range reduction: x = n * ln(2) + r, where n = round(x / ln(2)) and
  // |r| <= ln(2) / 2.
  // Round-to-nearest-even for non-negative x * log2(e) in [0, 129.12]: adding
  // 0x1.8p23f (1.5 * 2^23 = 0x4B400000) shifts the unit bit to bit 0 of the
  // float32 fraction, rounding to the nearest integer with ties to even.
  constexpr float kMagicBias = 0x1.8p23f;
  T biased = Fmuladd(x, Splat<T>(0x1.715476p+0f), Splat<T>(kMagicBias));
  T n_float = biased - Splat<T>(kMagicBias);
  IntType n = __builtin_bit_cast(IntType, biased) - 0x4B400000u;

  // Split ln(2) into a 16-bit high part (ln2_hi = 0x1.62e4p-1f) and a low
  // part (ln2_lo = 0x1.7f7d1cp-20f). Because n <= 129 has at most 8 integer
  // bits and ln2_hi has 16 bits, n * ln2_hi and r_hi = |x| - n * ln2_hi do not
  // round even without FMA.
  T r_hi = Fmuladd(-n_float, Splat<T>(0x1.62e4p-1f), x);
  T r_lo = -n_float * Splat<T>(0x1.7f7d1cp-20f);
  T r = r_hi + r_lo;

  T r2 = r * r;

  // Polynomial for (exp(r) - 1 - r) / r^2 on [-ln(2)/2, ln(2)/2]:
  T p_exp = HornerPoly(r, 0x1.6ca992p-10f, 0x1.120abep-7f, 0x1.55556cp-5f,
                       0x1.5554dep-3f, 0.5f);

  // Construct scale = 2^(n - 2) via integer exponent bitcast ((n + 125) << 23).
  // Split 0.25 * exp(|x|) = scale * exp(r) into head + tail:
  //   s_r_hi          = scale * r_hi (does not round because scale is a power
  //                     of 2)
  //   (head, head_lo) = FastTwoSum(scale, s_r_hi) (the rounded sum scale +
  //                     s_r_hi and the low bits dropped by that addition)
  //   tail            = head_lo + scale * (r_lo + r^2 * P(r))
  // Then +0.25 * exp(-|x|) = +0.0625 / (head + tail). Summing the smaller
  // terms (tail + div_term) before adding head and multiplying by 2.0 keeps
  // the error below 0.89 ULP without intermediate overflow.
  T scale = __builtin_bit_cast(T, (n + 125u) << 23);
  T s_r_hi = scale * r_hi;
  auto [head, head_lo] = FastTwoSum(scale, s_r_hi);
  T tail = Fmuladd(scale, Fmuladd(r2, p_exp, r_lo), head_lo);
  T half_h = head + tail;
  T div_term = Splat<T>(0.0625f) / half_h;
  T poly_cosh = Splat<T>(2.0f) * (head + (tail + div_term));

  // For |x| < ln(2) (~0.69314718f), use a degree-8 polynomial on [-ln(2),
  // ln(2)]:
  //   cosh(x) ~= 1 + 0.5 * x2 + x2^2 * (c4 + x2 * (c6 + x2 * c8))
  // Computing 0.5 * x2 + x2^2 * P(x2) before adding 1.0 avoids rounding the
  // 0.5 * x2 term against 1.0 before the x2^2 terms are added.
  T x2 = x * x;
  T p_small = HornerPoly(x2, 0x1.a9039p-16f, 0x1.6c023ap-10f, 0x1.55557p-5f);
  T x4 = x2 * x2;
  T small_cosh = Splat<T>(1.0f) + Fmuladd(x4, p_small, Splat<T>(0.5f) * x2);

  auto use_small = abs_x < Splat<T>(0.69314718f);
  T res = use_small ? small_cosh : poly_cosh;
  return is_nan_or_inf ? abs_x : res;
}

}  // namespace

float cosh_f32(float x) { return CoshF32(x); }
Vec4f cosh_v4f32(Vec4f x) { return CoshF32(x); }
Vec8f cosh_v8f32(Vec8f x) { return CoshF32(x); }
Vec16f cosh_v16f32(Vec16f x) { return CoshF32(x); }

}  // namespace xla::codegen
