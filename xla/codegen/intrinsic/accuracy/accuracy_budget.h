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

#ifndef XLA_CODEGEN_INTRINSIC_ACCURACY_ACCURACY_BUDGET_H_
#define XLA_CODEGEN_INTRINSIC_ACCURACY_ACCURACY_BUDGET_H_

namespace xla::codegen::intrinsic::accuracy {

// Accuracy budgets in ULPs (Unit in Last Place).
// These budgets are determined based on the performance of the intrinsics
// relative to a high-precision reference (e.g., mpmath).

// Exp
// Validated on random samples in range:
// F64: [-750, 750] (approx)
constexpr int kExpF64MaxUlp = 1;

// Log1p
// Validated on random samples.
constexpr int kLog1pF32MaxUlp = 1;
constexpr int kLog1pF64MaxUlp = 1;

// Rsqrt
// Validated on random samples + specialized range.
// 1/sqrt(x) should be very accurate if using Newton-Raphson.
constexpr int kRsqrtF32MaxUlp = 1;
constexpr int kRsqrtF64MaxUlp = 1;

// Tanh
constexpr int kTanhF32MaxUlp = 3;
// Relaxed to 5 ULP for F64 as per empirical results (observed max error 4 ULP)
constexpr int kTanhF64MaxUlp = 5;

// Erf
constexpr int kErfF32MaxUlp = 2;  // derived from previous defaults
constexpr int kErfF64MaxUlp = 2;

// Sqrt
constexpr int kSqrtF32MaxUlp =
    0;  // Hardware sqrt is exact usually (0.5 ULP rounded)
constexpr int kSqrtF64MaxUlp = 0;

}  // namespace xla::codegen::intrinsic::accuracy

#endif  // XLA_CODEGEN_INTRINSIC_ACCURACY_ACCURACY_BUDGET_H_
