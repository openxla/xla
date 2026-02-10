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

#ifndef XLA_CODEGEN_INTRINSIC_ACCURACY_ACCURACY_TEST_FRAMEWORK_H_
#define XLA_CODEGEN_INTRINSIC_ACCURACY_ACCURACY_TEST_FRAMEWORK_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "xla/codegen/intrinsic/accuracy/golden_baselines.h"
#include "xla/fp_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test.h"

namespace xla::codegen::intrinsic::accuracy {

struct AccuracyReport {
  int max_ulp_error = 0;
  double mean_ulp_error = 0.0;
  int p99_ulp_error = 0;
  double worst_input = 0.0;
  double worst_expected = 0.0;
  double worst_actual = 0.0;
  int count = 0;
};

template <typename T>
AccuracyReport RunAccuracyTest(const std::vector<RefPoint>& golden_points,
                               std::function<T(T)> func,
                               const std::string& func_name) {
  AccuracyReport report;
  std::vector<int> ulp_errors;
  int64_t total_ulp_error = 0;

  for (const auto& point : golden_points) {
    double input_f64 = point.input;
    double expected_f64 = point.expected;

    T input = static_cast<T>(input_f64);
    T expected = static_cast<T>(expected_f64);

    // Filter out points where input overflowed to Infinity during cast
    if (std::isinf(input) && !std::isinf(input_f64)) {
      continue;
    }

    // Filter out subnormals for expected values if they cause issues
    if (std::fpclassify(expected) == FP_SUBNORMAL) {
      continue;
    }

    T actual = func(input);

    int ulp_dist = 0;

    // Special handling for NaN and Inf
    if (std::isnan(expected)) {
      if (!std::isnan(actual)) {
        // Expected NaN but got something else.
        ulp_dist = 1000000;
      } else {
        ulp_dist = 0;
      }
    } else if (std::isinf(expected)) {
      if (std::isinf(actual) &&
          (std::signbit(expected) == std::signbit(actual))) {
        ulp_dist = 0;
      } else {
        ulp_dist = 1000000;
      }
    } else {
      if (std::isnan(actual) || std::isinf(actual)) {
        ulp_dist = 1000000;
      } else {
        // Compute ULP distance for finite values
        ulp_dist = std::abs(xla::CalculateDistanceInFloats(actual, expected));
      }
    }

    ulp_errors.push_back(ulp_dist);
    total_ulp_error += ulp_dist;

    if (ulp_dist > report.max_ulp_error) {
      report.max_ulp_error = ulp_dist;
      report.worst_input = static_cast<double>(input);
      report.worst_expected = static_cast<double>(expected);
      report.worst_actual = static_cast<double>(actual);
    }
  }

  report.count = ulp_errors.size();
  if (report.count > 0) {
    report.mean_ulp_error = static_cast<double>(total_ulp_error) / report.count;
    std::sort(ulp_errors.begin(), ulp_errors.end());
    report.p99_ulp_error = ulp_errors[static_cast<int>(report.count * 0.99)];
  }

  return report;
}

template <typename T>
void LogReport(const AccuracyReport& report, const std::string& func_name) {
  LOG(INFO) << "Accuracy Report for " << func_name << ":\n"
            << "  Max ULP Error: " << report.max_ulp_error << "\n"
            << "  Mean ULP Error: " << report.mean_ulp_error << "\n"
            << "  P99 ULP Error: " << report.p99_ulp_error << "\n"
            << "  Worst Case: input=" << report.worst_input
            << ", expected=" << report.worst_expected
            << ", actual=" << report.worst_actual;
}

template <typename T>
void AssertWithinBudget(const AccuracyReport& report, int budget) {
  // We use EXPECT_LE so tests continue.
  EXPECT_LE(report.max_ulp_error, budget)
      << "Max ULP error " << report.max_ulp_error << " exceeds budget "
      << budget << ". Worst case: input=" << report.worst_input
      << ", expected=" << report.worst_expected
      << ", actual=" << report.worst_actual;
}

// Helper to convert array to vector for easier usage
template <size_t N>
std::vector<RefPoint> ToVector(const std::array<RefPoint, N>& arr) {
  return std::vector<RefPoint>(arr.begin(), arr.end());
}

}  // namespace xla::codegen::intrinsic::accuracy

#endif  // XLA_CODEGEN_INTRINSIC_ACCURACY_ACCURACY_TEST_FRAMEWORK_H_
