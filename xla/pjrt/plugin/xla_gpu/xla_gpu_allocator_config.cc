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

#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"

#include <cmath>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

namespace xla {
namespace {

constexpr absl::string_view kGrammar =
    "expected START or START-CAP as fractions of total device memory, for "
    "example \"0.75\" (preallocate 75%, grow to all memory), \"0.75-0.85\" "
    "(grow to at most 85%) or \"0.75-0.75\" (fixed, never grow)";

absl::Status InvalidSpec(absl::string_view spec, absl::string_view reason) {
  return absl::InvalidArgumentError(absl::StrCat(
      "Invalid memory fraction \"", spec, "\": ", reason, "; ", kGrammar));
}

absl::StatusOr<double> ParseFraction(absl::string_view text,
                                     absl::string_view spec) {
  double value = 0;
  if (!absl::SimpleAtod(absl::StripAsciiWhitespace(text), &value) ||
      !std::isfinite(value) || value <= 0) {
    return InvalidSpec(spec, "fractions must be positive numbers");
  }
  return value;
}

}  // namespace

absl::StatusOr<MemFraction> ParseMemFraction(absl::string_view spec) {
  const std::vector<absl::string_view> parts =
      absl::StrSplit(spec, absl::MaxSplits('-', 1));
  absl::StatusOr<double> start = ParseFraction(parts[0], spec);
  if (!start.ok()) {
    return start.status();
  }
  if (parts.size() == 1) {
    return MemFractionFromFraction(*start);
  }
  absl::StatusOr<double> cap = ParseFraction(parts[1], spec);
  if (!cap.ok()) {
    return cap.status();
  }
  if (*cap < *start) {
    return InvalidSpec(spec, "the cap must be at least the start");
  }
  if (*cap == *start) {
    return MemFraction(FixedMemFraction{*start});
  }
  if (*cap > 1) {
    return InvalidSpec(spec,
                       "a growth cap cannot exceed 1 (all of device memory)");
  }
  return MemFraction(FlexMemFraction{*start, *cap});
}

MemFraction MemFractionFromFraction(double fraction) {
  if (fraction < 1) {
    return FlexMemFraction{fraction, 1.0};
  }
  return FixedMemFraction{fraction};
}

double MemFractionStart(const MemFraction& fraction) {
  if (const auto* fixed = std::get_if<FixedMemFraction>(&fraction)) {
    return fixed->fraction;
  }
  return std::get<FlexMemFraction>(fraction).start;
}

std::string MemFractionToString(const MemFraction& fraction) {
  if (const auto* fixed = std::get_if<FixedMemFraction>(&fraction)) {
    return absl::StrCat(fixed->fraction, "-", fixed->fraction);
  }
  const auto& flex = std::get<FlexMemFraction>(fraction);
  if (flex.cap >= 1) {
    return absl::StrCat(flex.start);
  }
  return absl::StrCat(flex.start, "-", flex.cap);
}

}  // namespace xla
