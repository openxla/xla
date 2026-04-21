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

// End-to-end test for the class of bug fixed by the algebraic simplifier
// guard that skips `A/sqrt(B) => A*rsqrt(B)` for f64.
//
// The user-facing source spelling `1 / sqrt(x)` in f64 is built from two
// correctly-rounded StableHLO primitives (kDivide and kSqrt) and therefore
// carries a spec-level precision bound: the composition is at most 1 ULP
// from the IEEE round-to-nearest-even ideal. Concretely, for every input
// covered here, the expected bit pattern equals `1.0 / std::sqrt(x)` on
// the host.
//
// Without the simplifier guard, this HLO gets rewritten to
// `multiply(constant(1), rsqrt(parameter(0)))`, and the rsqrt emit on
// CPU+avx512f goes through a SIMD vrsqrt + Newton-Raphson refinement path
// that lands on a *different* in-range f64 value for some inputs — the
// failing elixir-nx/nx#1704 doctest is the original report. The
// `ParseAndReturnVerifiedModule + Execute` pipeline here runs the full
// set of HLO passes, so the simplifier does have a chance to fire; the
// assertion fails on main precisely because the rewrite produces a
// different bit pattern than sqrt+fdiv would.
//
// The assertion is bit-exact (`EXPECT_EQ` on the full Literal), not
// within-tolerance: the whole point is that both paths are ≤1 ULP from
// the ideal but land on *different* in-range values, so any tolerance
// that accepts one would accept the other and hide the bug.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

const char* const kDivBySqrtF64Module = R"(
    HloModule div_by_sqrt_f64

    ENTRY main {
      p0 = f64[] parameter(0)
      sqrt = f64[] sqrt(p0)
      one = f64[] constant(1)
      ROOT div = f64[] divide(one, sqrt)
    }
  )";

class DivideBySqrtF64Test : public HloPjRtTestBase {};

// Each entry is an f64 input whose 1/sqrt(x) bit pattern differs from the
// rsqrt(x) emit on CPU+avx512f, i.e., an input that triggers the bug when
// the simplifier rewrites to rsqrt. For each, the correctly-rounded
// composition result is computed from std::sqrt on the host.
TEST_F(DivideBySqrtF64Test, MatchesIeeeCorrectlyRoundedComposition) {
  const double kInputs[] = {
      2.0,   // 1/sqrt(2)  -- 0x1.6a09e667f3bcdp-1
      3.0,   // 1/sqrt(3)  -- 0x1.279a74590331dp-1 (elixir-nx#1704)
      0.5,   // sqrt(0.5) = 1/sqrt(2), so same reference structure
      1.5,   // 1/sqrt(1.5)
      7.0,   // 1/sqrt(7)
  };

  for (double x : kInputs) {
    // Re-parse per iteration: Execute consumes the module by-move, and
    // HloModule::Clone returns an unverified module while Execute expects
    // the verified type.
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(kDivBySqrtF64Module));

    // Host reference: the correctly-rounded composition result.
    double reference = 1.0 / std::sqrt(x);

    auto input = LiteralUtil::CreateR0<double>(x);

    TF_ASSERT_OK_AND_ASSIGN(
        auto result, Execute(std::move(module), {&input}));

    auto expected = LiteralUtil::CreateR0<double>(reference);

    EXPECT_EQ(result, expected)
        << "divide(1.0, sqrt(" << x << ")) produced a bit pattern "
        << "that differs from the correctly-rounded reference "
        << "1.0 / std::sqrt(" << x << ") = " << reference << ". "
        << "This usually indicates the algebraic simplifier has rewritten "
        << "the expression to multiply(1.0, rsqrt(x)), which on some CPUs "
        << "(notably avx512f hosts) emits a SIMD vrsqrt + Newton-Raphson "
        << "refinement landing on a different in-range f64 value.";
  }
}

// Diagnostic sweep: over a log-uniform sample of regular-range f64
// inputs, count how often `divide(1, sqrt(x))` through the full XLA
// pipeline lands on a different bit pattern than the host-computed
// correctly-rounded composition reference `1.0 / std::sqrt(x)`, and
// break the divergences down by ULP distance.
//
// This test always passes — it's a measurement, not an assertion. The
// summary line(s) it prints give PR reviewers a reproducible,
// domain-wide picture of the bug this file's assertions are targeted
// at.
//
// Measured on a local Intel/AMD avx512f host (early 2026) with
// kNumSamples = 1000 (both guards reverted, i.e. pre-PR emit):
//
//    0 ULP : 709 / 1000 (70.9%) — matches reference exactly
//    1 ULP : 289 / 1000 (28.9%) — rewrite lands on the neighbouring f64
//    2 ULP :   2 / 1000 (0.2%)  — rare cases where Newton-Raphson
//                                  refinement compounds with the
//                                  reference's own rounding to reach
//                                  the theoretical worst case
//
// With both guards applied (current branch), all 1000 samples show
// 0 ULP — the HLO stays as sqrt+fdiv, which matches the host's
// libm-based sqrt+fdiv reference exactly.
//
// Non-avx512f CPU builds trivially observe 0% regardless of the guards
// because the intrinsic already falls back; the test is most useful on
// avx512f hosts.
//
// The committed sample size is 100 to keep the test fast (~1s);
// increase locally for tighter statistics or to reliably populate the
// >=2 ULP tail.
TEST_F(DivideBySqrtF64Test, DiagnosticSweepAgainstReference) {
  constexpr int kNumSamples = 100;
  constexpr double kLogLo = -200.0;
  constexpr double kLogHi = 200.0;

  std::vector<double> inputs;
  inputs.reserve(kNumSamples);
  for (int i = 0; i < kNumSamples; ++i) {
    double e = kLogLo + (kLogHi - kLogLo) * i / (kNumSamples - 1);
    inputs.push_back(std::pow(10.0, e));
  }

  std::map<int64_t, int> ulp_histogram;
  std::map<int64_t, double> example_input_per_bucket;
  std::map<int64_t, double> example_got_per_bucket;
  std::map<int64_t, double> example_ref_per_bucket;
  // Track every input in the >=2 ULP tail so we can see whether those
  // rare cases are concentrated or scattered across the domain.
  std::vector<std::tuple<double, double, double, int64_t>> tail_samples;

  for (double x : inputs) {
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(kDivBySqrtF64Module));
    auto input = LiteralUtil::CreateR0<double>(x);
    TF_ASSERT_OK_AND_ASSIGN(
        auto result, Execute(std::move(module), {&input}));
    double got = result.data<double>()[0];
    double reference = 1.0 / std::sqrt(x);

    uint64_t a, b;
    std::memcpy(&a, &got, sizeof(a));
    std::memcpy(&b, &reference, sizeof(b));
    int64_t ulp = a > b ? a - b : b - a;

    ulp_histogram[ulp]++;
    if (example_input_per_bucket.find(ulp) == example_input_per_bucket.end()) {
      example_input_per_bucket[ulp] = x;
      example_got_per_bucket[ulp] = got;
      example_ref_per_bucket[ulp] = reference;
    }
    if (ulp >= 2) {
      tail_samples.emplace_back(x, got, reference, ulp);
    }
  }

  // Render doubles as hex-floats (%a) so single-ULP and double-ULP
  // differences are visible at the bit level rather than being hidden
  // by the default 6-digit printer.
  auto hex = [](double v) -> std::string {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%a", v);
    return std::string(buf);
  };

  LOG(INFO) << "DiagnosticSweep histogram over " << kNumSamples
            << " log-uniform f64 inputs:";
  for (const auto& [ulp, count] : ulp_histogram) {
    double pct = 100.0 * count / kNumSamples;
    if (ulp == 0) {
      LOG(INFO) << "  " << ulp << " ULP: " << count << " (" << pct
                << "%) — matches reference exactly.";
    } else {
      LOG(INFO) << "  " << ulp << " ULP: " << count << " (" << pct
                << "%), example x=" << example_input_per_bucket[ulp]
                << "  got=" << hex(example_got_per_bucket[ulp])
                << "  ref=" << hex(example_ref_per_bucket[ulp]);
    }
  }

  if (!tail_samples.empty()) {
    LOG(INFO) << "All inputs in the >=2 ULP tail:";
    for (auto const& [x, got, ref, ulp] : tail_samples) {
      LOG(INFO) << "  x=" << x << " (" << hex(x) << ")"
                << "  got=" << hex(got) << "  ref=" << hex(ref)
                << "  ULP=" << ulp;
    }
  }
}

}  // namespace
}  // namespace xla
