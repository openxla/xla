
#include <arm_sve.h>
#include <stdint.h>

#include "absl/base/attributes.h"
#include "absl/base/casts.h"
#include "absl/base/dynamic_annotations.h"

#include "xla/service/cpu/runtime_tanh.h"
#include "xla/service/custom_call_target_registry.h"

#pragma GCC target("+sve")
#pragma clang attribute push(__attribute__((target("sve"))), apply_to = function)
ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_Aarch64SveHyperbolicTangent(
    void* out, const void** in, const char* /*opaque*/, size_t /*opaque_len*/,
    void* /*status*/)
{
    // API_VERSION_STATUS_RETURNING_UNIFIED convention:
    // in[0] is input buffer, in[1] is size constant
    const float* input = static_cast<const float*>(in[0]);
    float* output = static_cast<float*>(out);
    const int32_t* size_ptr = static_cast<const int32_t*>(in[1]);
    int size = *size_ptr;

    // Manually-scheduled x6 unrolled SVE tanh.  The statements grouped
    // by blank lines are intended to issue in parallel across the FMA,
    // divider, and load-store pipes.  x6 fits within Neoverse V2's 32
    // Z-register budget without spills, amortises loop overhead better
    // than x4, and pipelines 6 in-flight svdivs on the divider unit.
    //
    // Speculative-degree dispatch: each outer iteration eagerly computes
    // s = x*x for all 24 lanes, then tests s < (THRESHOLD)^2.  When all
    // lanes pass, evaluate a degree-9/6 polynomial (drop alpha_11,
    // alpha_13, skip the input clamp).  Otherwise fall through to the
    // full degree-13/6 path which reuses s after clamping it via svmin
    // to clamp(x)^2 = min(s, 81) -- valid because the clamp range is
    // symmetric.  Both paths consume the eager s, so the slow path
    // pays no extra svmul cost relative to the non-dispatching kernel.

    const svfloat32_t plus_9 = svdup_f32(9.f);
    const svfloat32_t minus_9 = svdup_f32(-9.f);
    const svfloat32_t s_clamp = svdup_f32(9.f * 9.f);
    const svfloat32_t threshold_sq = svdup_f32(2.6f * 2.6f);

    // Numerator coefficients (odd powers).
    const svfloat32_t alpha_1 = svdup_f32(4.89352455891786e-03f);
    const svfloat32_t alpha_3 = svdup_f32(6.37261928875436e-04f);
    const svfloat32_t alpha_5 = svdup_f32(1.48572235717979e-05f);
    const svfloat32_t alpha_7 = svdup_f32(5.12229709037114e-08f);
    const svfloat32_t alpha_9 = svdup_f32(-8.60467152213735e-11f);
    const svfloat32_t alpha_11 = svdup_f32(2.00018790482477e-13f);
    const svfloat32_t alpha_13 = svdup_f32(-2.76076847742355e-16f);

    // Fast-path-only refit of alpha_3: BASE_ALPHA_3 + 4 ULPs in f32 space
    // (bits 0x3a270ded -> 0x3a270df1).  Required to keep max 5 ULP on
    // the full fast-path domain |x| < 2.6 in the degree-9/6 form -- with
    // the unperturbed alpha_3 the same form drifts to 6 ULP near
    // x ~= 2.42.  The slow path keeps the unperturbed alpha_3, so any
    // |x| >= 2.6 batch is bit-identical to the previous kernel.
    const svfloat32_t fast_alpha_3 = svdup_f32(6.3726218650e-04f);

    // Denominator coefficients (even powers).
    const svfloat32_t beta_0 = svdup_f32(4.89352518554385e-03f);
    const svfloat32_t beta_2 = svdup_f32(2.26843463243900e-03f);
    const svfloat32_t beta_4 = svdup_f32(1.18534705686654e-04f);
    const svfloat32_t beta_6 = svdup_f32(1.19825839466702e-06f);

    int i = 0;
    int sve_len = svcntw();
    svbool_t predicate_0 = svwhilelt_b32_s32(i+sve_len*0, size);
    svbool_t predicate_1 = svwhilelt_b32_s32(i+sve_len*1, size);
    svbool_t predicate_2 = svwhilelt_b32_s32(i+sve_len*2, size);
    svbool_t predicate_3 = svwhilelt_b32_s32(i+sve_len*3, size);
    svbool_t predicate_4 = svwhilelt_b32_s32(i+sve_len*4, size);
    svbool_t predicate_5 = svwhilelt_b32_s32(i+sve_len*5, size);
    do {
        svfloat32_t x_0, x2_0, p_0, q_0, result_0;
        svfloat32_t x_1, x2_1, p_1, q_1, result_1;
        svfloat32_t x_2, x2_2, p_2, q_2, result_2;
        svfloat32_t x_3, x2_3, p_3, q_3, result_3;
        svfloat32_t x_4, x2_4, p_4, q_4, result_4;
        svfloat32_t x_5, x2_5, p_5, q_5, result_5;

        // Load.
        x_0 = svld1_f32(predicate_0, &input[i+sve_len*0]);
        x_1 = svld1_f32(predicate_1, &input[i+sve_len*1]);
        x_2 = svld1_f32(predicate_2, &input[i+sve_len*2]);
        x_3 = svld1_f32(predicate_3, &input[i+sve_len*3]);
        x_4 = svld1_f32(predicate_4, &input[i+sve_len*4]);
        x_5 = svld1_f32(predicate_5, &input[i+sve_len*5]);

        // Eagerly compute s = x*x.  Both fast and slow paths consume s;
        // the slow path will svmin it against s_clamp to obtain
        // clamp(x)^2.  Computing it up front saves the slow path 6
        // svmul ops and lets us test on s directly (no svabs needed).
        x2_0 = svmul_x(predicate_0, x_0, x_0);
        x2_1 = svmul_x(predicate_1, x_1, x_1);
        x2_2 = svmul_x(predicate_2, x_2, x_2);
        x2_3 = svmul_x(predicate_3, x_3, x_3);
        x2_4 = svmul_x(predicate_4, x_4, x_4);
        x2_5 = svmul_x(predicate_5, x_5, x_5);

        // Speculative-degree dispatch test: any |x| >= 2.6 ?  Predicate
        // ops (svorr, svptest_any) live on a separate pipe from the FMA
        // work that immediately follows, so most of this latency is
        // hidden behind the start of the polynomial evaluation.
        svbool_t large_0 = svcmpge_f32(predicate_0, x2_0, threshold_sq);
        svbool_t large_1 = svcmpge_f32(predicate_1, x2_1, threshold_sq);
        svbool_t large_2 = svcmpge_f32(predicate_2, x2_2, threshold_sq);
        svbool_t large_3 = svcmpge_f32(predicate_3, x2_3, threshold_sq);
        svbool_t large_4 = svcmpge_f32(predicate_4, x2_4, threshold_sq);
        svbool_t large_5 = svcmpge_f32(predicate_5, x2_5, threshold_sq);
        svbool_t ptrue = svptrue_b8();
        svbool_t large_01 = svorr_z(ptrue, large_0, large_1);
        svbool_t large_23 = svorr_z(ptrue, large_2, large_3);
        svbool_t large_45 = svorr_z(ptrue, large_4, large_5);
        svbool_t large_0123 = svorr_z(ptrue, large_01, large_23);
        svbool_t large_mask = svorr_z(ptrue, large_0123, large_45);
        bool any_large = svptest_any(ptrue, large_mask);

        if (__builtin_expect(any_large, 0)) {
            // Slow path: full degree-13/6 polynomial.  Bit-identical to
            // the non-dispatching kernel.

            // Clamp x to [-9, 9].
            x_0 = svmax_x(predicate_0, svmin_x(predicate_0, x_0, plus_9), minus_9);
            x_1 = svmax_x(predicate_1, svmin_x(predicate_1, x_1, plus_9), minus_9);
            x_2 = svmax_x(predicate_2, svmin_x(predicate_2, x_2, plus_9), minus_9);
            x_3 = svmax_x(predicate_3, svmin_x(predicate_3, x_3, plus_9), minus_9);
            x_4 = svmax_x(predicate_4, svmin_x(predicate_4, x_4, plus_9), minus_9);
            x_5 = svmax_x(predicate_5, svmin_x(predicate_5, x_5, plus_9), minus_9);

            // Clamp s = x*x in place: clamp(x)^2 == min(x^2, s_clamp)
            // because the clamp range is symmetric.
            x2_0 = svmin_x(predicate_0, x2_0, s_clamp);
            x2_1 = svmin_x(predicate_1, x2_1, s_clamp);
            x2_2 = svmin_x(predicate_2, x2_2, s_clamp);
            x2_3 = svmin_x(predicate_3, x2_3, s_clamp);
            x2_4 = svmin_x(predicate_4, x2_4, s_clamp);
            x2_5 = svmin_x(predicate_5, x2_5, s_clamp);

            // Numerator p and denominator q polynomials (Horner, degree 13/6).
            p_0 = x2_0; p_1 = x2_1; p_2 = x2_2;
            p_3 = x2_3; p_4 = x2_4; p_5 = x2_5;
            q_0 = x2_0; q_1 = x2_1; q_2 = x2_2;
            q_3 = x2_3; q_4 = x2_4; q_5 = x2_5;

            p_0 = svmad_x(predicate_0, p_0,   alpha_13, alpha_11);
            p_1 = svmad_x(predicate_1, p_1,   alpha_13, alpha_11);
            p_2 = svmad_x(predicate_2, p_2,   alpha_13, alpha_11);
            q_0 = svmad_x(predicate_0, q_0,     beta_6, beta_4);
            q_1 = svmad_x(predicate_1, q_1,     beta_6, beta_4);
            q_2 = svmad_x(predicate_2, q_2,     beta_6, beta_4);

            p_3 = svmad_x(predicate_3, p_3,   alpha_13, alpha_11);
            p_4 = svmad_x(predicate_4, p_4,   alpha_13, alpha_11);
            p_5 = svmad_x(predicate_5, p_5,   alpha_13, alpha_11);
            q_3 = svmad_x(predicate_3, q_3,     beta_6, beta_4);
            q_4 = svmad_x(predicate_4, q_4,     beta_6, beta_4);
            q_5 = svmad_x(predicate_5, q_5,     beta_6, beta_4);

            p_0 = svmad_x(predicate_0, p_0,       x2_0, alpha_9);
            p_1 = svmad_x(predicate_1, p_1,       x2_1, alpha_9);
            p_2 = svmad_x(predicate_2, p_2,       x2_2, alpha_9);
            q_0 = svmad_x(predicate_0, q_0,       x2_0, beta_2);
            q_1 = svmad_x(predicate_1, q_1,       x2_1, beta_2);
            q_2 = svmad_x(predicate_2, q_2,       x2_2, beta_2);

            p_3 = svmad_x(predicate_3, p_3,       x2_3, alpha_9);
            p_4 = svmad_x(predicate_4, p_4,       x2_4, alpha_9);
            p_5 = svmad_x(predicate_5, p_5,       x2_5, alpha_9);
            q_3 = svmad_x(predicate_3, q_3,       x2_3, beta_2);
            q_4 = svmad_x(predicate_4, q_4,       x2_4, beta_2);
            q_5 = svmad_x(predicate_5, q_5,       x2_5, beta_2);

            p_0 = svmad_x(predicate_0, p_0,       x2_0, alpha_7);
            p_1 = svmad_x(predicate_1, p_1,       x2_1, alpha_7);
            p_2 = svmad_x(predicate_2, p_2,       x2_2, alpha_7);
            q_0 = svmad_x(predicate_0, q_0,       x2_0, beta_0);
            q_1 = svmad_x(predicate_1, q_1,       x2_1, beta_0);
            q_2 = svmad_x(predicate_2, q_2,       x2_2, beta_0);

            p_3 = svmad_x(predicate_3, p_3,       x2_3, alpha_7);
            p_4 = svmad_x(predicate_4, p_4,       x2_4, alpha_7);
            p_5 = svmad_x(predicate_5, p_5,       x2_5, alpha_7);
            q_3 = svmad_x(predicate_3, q_3,       x2_3, beta_0);
            q_4 = svmad_x(predicate_4, q_4,       x2_4, beta_0);
            q_5 = svmad_x(predicate_5, q_5,       x2_5, beta_0);

            p_0 = svmad_x(predicate_0, p_0,       x2_0, alpha_5);
            p_1 = svmad_x(predicate_1, p_1,       x2_1, alpha_5);
            p_2 = svmad_x(predicate_2, p_2,       x2_2, alpha_5);
            p_3 = svmad_x(predicate_3, p_3,       x2_3, alpha_5);
            p_4 = svmad_x(predicate_4, p_4,       x2_4, alpha_5);
            p_5 = svmad_x(predicate_5, p_5,       x2_5, alpha_5);

            p_0 = svmad_x(predicate_0, p_0,       x2_0, alpha_3);
            p_1 = svmad_x(predicate_1, p_1,       x2_1, alpha_3);
            p_2 = svmad_x(predicate_2, p_2,       x2_2, alpha_3);
            p_3 = svmad_x(predicate_3, p_3,       x2_3, alpha_3);
            p_4 = svmad_x(predicate_4, p_4,       x2_4, alpha_3);
            p_5 = svmad_x(predicate_5, p_5,       x2_5, alpha_3);

            p_0 = svmad_x(predicate_0, p_0,       x2_0, alpha_1);
            p_1 = svmad_x(predicate_1, p_1,       x2_1, alpha_1);
            p_2 = svmad_x(predicate_2, p_2,       x2_2, alpha_1);
            p_3 = svmad_x(predicate_3, p_3,       x2_3, alpha_1);
            p_4 = svmad_x(predicate_4, p_4,       x2_4, alpha_1);
            p_5 = svmad_x(predicate_5, p_5,       x2_5, alpha_1);

            p_0 = svmul_x(predicate_0, p_0, x_0);
            p_1 = svmul_x(predicate_1, p_1, x_1);
            p_2 = svmul_x(predicate_2, p_2, x_2);
            p_3 = svmul_x(predicate_3, p_3, x_3);
            p_4 = svmul_x(predicate_4, p_4, x_4);
            p_5 = svmul_x(predicate_5, p_5, x_5);

            result_0 = svdiv_x(predicate_0, p_0, q_0);
            result_1 = svdiv_x(predicate_1, p_1, q_1);
            result_2 = svdiv_x(predicate_2, p_2, q_2);
            result_3 = svdiv_x(predicate_3, p_3, q_3);
            result_4 = svdiv_x(predicate_4, p_4, q_4);
            result_5 = svdiv_x(predicate_5, p_5, q_5);

            svst1_f32(predicate_0, &output[i+sve_len*0], result_0);
            svst1_f32(predicate_1, &output[i+sve_len*1], result_1);
            svst1_f32(predicate_2, &output[i+sve_len*2], result_2);
            svst1_f32(predicate_3, &output[i+sve_len*3], result_3);
            svst1_f32(predicate_4, &output[i+sve_len*4], result_4);
            svst1_f32(predicate_5, &output[i+sve_len*5], result_5);
        } else {
            // Fast path: degree-9/6 polynomial.  All lanes have |x| < 2.6,
            // so we can skip the input clamp and drop alpha_11 and
            // alpha_13.  Uses fast_alpha_3 (refit) instead of alpha_3 to
            // keep max 5 ULP on the full fast-path domain.

            // Numerator: a1 + a3*s + a5*s^2 + a7*s^3 + a9*s^4 (Horner).
            p_0 = svmad_x(predicate_0, x2_0, alpha_9, alpha_7);
            p_1 = svmad_x(predicate_1, x2_1, alpha_9, alpha_7);
            p_2 = svmad_x(predicate_2, x2_2, alpha_9, alpha_7);
            p_3 = svmad_x(predicate_3, x2_3, alpha_9, alpha_7);
            p_4 = svmad_x(predicate_4, x2_4, alpha_9, alpha_7);
            p_5 = svmad_x(predicate_5, x2_5, alpha_9, alpha_7);

            // Denominator: b0 + b2*s + b4*s^2 + b6*s^3 (Horner).
            q_0 = svmad_x(predicate_0, x2_0, beta_6, beta_4);
            q_1 = svmad_x(predicate_1, x2_1, beta_6, beta_4);
            q_2 = svmad_x(predicate_2, x2_2, beta_6, beta_4);
            q_3 = svmad_x(predicate_3, x2_3, beta_6, beta_4);
            q_4 = svmad_x(predicate_4, x2_4, beta_6, beta_4);
            q_5 = svmad_x(predicate_5, x2_5, beta_6, beta_4);

            p_0 = svmad_x(predicate_0, p_0, x2_0, alpha_5);
            p_1 = svmad_x(predicate_1, p_1, x2_1, alpha_5);
            p_2 = svmad_x(predicate_2, p_2, x2_2, alpha_5);
            p_3 = svmad_x(predicate_3, p_3, x2_3, alpha_5);
            p_4 = svmad_x(predicate_4, p_4, x2_4, alpha_5);
            p_5 = svmad_x(predicate_5, p_5, x2_5, alpha_5);

            q_0 = svmad_x(predicate_0, q_0, x2_0, beta_2);
            q_1 = svmad_x(predicate_1, q_1, x2_1, beta_2);
            q_2 = svmad_x(predicate_2, q_2, x2_2, beta_2);
            q_3 = svmad_x(predicate_3, q_3, x2_3, beta_2);
            q_4 = svmad_x(predicate_4, q_4, x2_4, beta_2);
            q_5 = svmad_x(predicate_5, q_5, x2_5, beta_2);

            p_0 = svmad_x(predicate_0, p_0, x2_0, fast_alpha_3);
            p_1 = svmad_x(predicate_1, p_1, x2_1, fast_alpha_3);
            p_2 = svmad_x(predicate_2, p_2, x2_2, fast_alpha_3);
            p_3 = svmad_x(predicate_3, p_3, x2_3, fast_alpha_3);
            p_4 = svmad_x(predicate_4, p_4, x2_4, fast_alpha_3);
            p_5 = svmad_x(predicate_5, p_5, x2_5, fast_alpha_3);

            q_0 = svmad_x(predicate_0, q_0, x2_0, beta_0);
            q_1 = svmad_x(predicate_1, q_1, x2_1, beta_0);
            q_2 = svmad_x(predicate_2, q_2, x2_2, beta_0);
            q_3 = svmad_x(predicate_3, q_3, x2_3, beta_0);
            q_4 = svmad_x(predicate_4, q_4, x2_4, beta_0);
            q_5 = svmad_x(predicate_5, q_5, x2_5, beta_0);

            p_0 = svmad_x(predicate_0, p_0, x2_0, alpha_1);
            p_1 = svmad_x(predicate_1, p_1, x2_1, alpha_1);
            p_2 = svmad_x(predicate_2, p_2, x2_2, alpha_1);
            p_3 = svmad_x(predicate_3, p_3, x2_3, alpha_1);
            p_4 = svmad_x(predicate_4, p_4, x2_4, alpha_1);
            p_5 = svmad_x(predicate_5, p_5, x2_5, alpha_1);

            p_0 = svmul_x(predicate_0, p_0, x_0);
            p_1 = svmul_x(predicate_1, p_1, x_1);
            p_2 = svmul_x(predicate_2, p_2, x_2);
            p_3 = svmul_x(predicate_3, p_3, x_3);
            p_4 = svmul_x(predicate_4, p_4, x_4);
            p_5 = svmul_x(predicate_5, p_5, x_5);

            result_0 = svdiv_x(predicate_0, p_0, q_0);
            result_1 = svdiv_x(predicate_1, p_1, q_1);
            result_2 = svdiv_x(predicate_2, p_2, q_2);
            result_3 = svdiv_x(predicate_3, p_3, q_3);
            result_4 = svdiv_x(predicate_4, p_4, q_4);
            result_5 = svdiv_x(predicate_5, p_5, q_5);

            svst1_f32(predicate_0, &output[i+sve_len*0], result_0);
            svst1_f32(predicate_1, &output[i+sve_len*1], result_1);
            svst1_f32(predicate_2, &output[i+sve_len*2], result_2);
            svst1_f32(predicate_3, &output[i+sve_len*3], result_3);
            svst1_f32(predicate_4, &output[i+sve_len*4], result_4);
            svst1_f32(predicate_5, &output[i+sve_len*5], result_5);
        }

        // Loop accounting.
        i += sve_len*6;
        predicate_0 = svwhilelt_b32_s32(i+sve_len*0, size);
        predicate_1 = svwhilelt_b32_s32(i+sve_len*1, size);
        predicate_2 = svwhilelt_b32_s32(i+sve_len*2, size);
        predicate_3 = svwhilelt_b32_s32(i+sve_len*3, size);
        predicate_4 = svwhilelt_b32_s32(i+sve_len*4, size);
        predicate_5 = svwhilelt_b32_s32(i+sve_len*5, size);
    } while (svptest_any(svptrue_b8(), predicate_0));
}
#pragma clang attribute pop


// ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY float __xla_cpu_runtime_Aarch64SveHyperbolicTangent(float input)
// {
//     float output;
//     tanh_array_sve_x4(&input, &output, 1);
//     return output;
// }

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "__xla_cpu_runtime_Aarch64SveHyperbolicTangent",
    __xla_cpu_runtime_Aarch64SveHyperbolicTangent);
