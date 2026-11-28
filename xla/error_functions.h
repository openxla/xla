/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_ERROR_FUNCTIONS_H_
#define XLA_ERROR_FUNCTIONS_H_

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/source_location.h"

// Utility functions and types for constructing absl::Status error objects with
// formatted error messages.
//
// Examples:
//
//   // Formatted error message with compile-time checking:
//   return xla::InvalidArgument("Expected shape %s, but got %s",
//                               expected_shape.ToString(),
//                               actual_shape.ToString());
//
//   // Fast string concatenation:
//   return xla::InternalStrCat("Failed to compile computation: ", id,
//                              " due to missing operand.");
//
namespace xla {

// Adds some context information to the error message in a
// absl::Status. This is useful as absl::Statuses are propagated upwards.
absl::Status AddStatus(absl::Status prior, absl::string_view context);
absl::Status AppendStatus(absl::Status prior, absl::string_view context);

namespace internal {

// Factory for status-generating types.
//
// Key design points:
// 1. `Code` is on the outer struct so deduction guides don't need to deduce it
//    (C++ requires all deduction guide parameters to appear in the argument
//    list).
// 2. CTAD deduces `Args...` via the deduction guide first, allowing the
//    constructor to take a trailing default `SourceLocation` after the pack.
template <absl::StatusCode Code>
struct Error {
  // Uses absl::FormatSpec for compile-time format string validation.
  template <typename... Args>
  struct Format {
    absl::Status status;

    explicit Format(const absl::FormatSpec<Args...>& format,
                    const Args&... args,
                    absl::SourceLocation loc = absl::SourceLocation::current())
        : status(Code, absl::StrFormat(format, args...), loc) {}

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator absl::Status() const { return status; }
  };

  template <typename... Args>
  Format(const absl::FormatSpec<Args...>&, const Args&...) -> Format<Args...>;

  // Uses absl::StrCat for string concatenation.
  template <typename... Args>
  struct StrCat {
    absl::Status status;

    explicit StrCat(const Args&... concat,
                    absl::SourceLocation loc = absl::SourceLocation::current())
        : status(Code, absl::StrCat(concat...), loc) {}

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator absl::Status() const { return status; }
  };

  template <typename... Args>
  StrCat(const Args&...) -> StrCat<Args...>;
};

using AbortedError = Error<absl::StatusCode::kAborted>;
using CancelledError = Error<absl::StatusCode::kCancelled>;
using DeadlineExceededError = Error<absl::StatusCode::kDeadlineExceeded>;
using FailedPreconditionError = Error<absl::StatusCode::kFailedPrecondition>;
using InternalError = Error<absl::StatusCode::kInternal>;
using InvalidArgumentError = Error<absl::StatusCode::kInvalidArgument>;
using NotFoundError = Error<absl::StatusCode::kNotFound>;
using ResourceExhaustedError = Error<absl::StatusCode::kResourceExhausted>;
using UnavailableError = Error<absl::StatusCode::kUnavailable>;
using UnimplementedError = Error<absl::StatusCode::kUnimplemented>;
using UnknownError = Error<absl::StatusCode::kUnknown>;

}  // namespace internal

template <typename... Args>
using Aborted = internal::AbortedError::Format<Args...>;
template <typename... Args>
using AbortedStrCat = internal::AbortedError::StrCat<Args...>;

template <typename... Args>
using Cancelled = internal::CancelledError::Format<Args...>;
template <typename... Args>
using CancelledStrCat = internal::CancelledError::StrCat<Args...>;

template <typename... Args>
using DeadlineExceeded = internal::DeadlineExceededError::Format<Args...>;
template <typename... Args>
using DeadlineExceededStrCat = internal::DeadlineExceededError::StrCat<Args...>;

template <typename... Args>
using FailedPrecondition = internal::FailedPreconditionError::Format<Args...>;
template <typename... Args>
using FailedPreconditionStrCat =
    internal::FailedPreconditionError::StrCat<Args...>;

template <typename... Args>
using Internal = internal::InternalError::Format<Args...>;
template <typename... Args>
using InternalStrCat = internal::InternalError::StrCat<Args...>;

template <typename... Args>
using InvalidArgument = internal::InvalidArgumentError::Format<Args...>;
template <typename... Args>
using InvalidArgumentStrCat = internal::InvalidArgumentError::StrCat<Args...>;

template <typename... Args>
using NotFound = internal::NotFoundError::Format<Args...>;
template <typename... Args>
using NotFoundStrCat = internal::NotFoundError::StrCat<Args...>;

template <typename... Args>
using ResourceExhausted = internal::ResourceExhaustedError::Format<Args...>;
template <typename... Args>
using ResourceExhaustedStrCat =
    internal::ResourceExhaustedError::StrCat<Args...>;

template <typename... Args>
using Unavailable = internal::UnavailableError::Format<Args...>;
template <typename... Args>
using UnavailableStrCat = internal::UnavailableError::StrCat<Args...>;

template <typename... Args>
using Unimplemented = internal::UnimplementedError::Format<Args...>;
template <typename... Args>
using UnimplementedStrCat = internal::UnimplementedError::StrCat<Args...>;

template <typename... Args>
using Unknown = internal::UnknownError::Format<Args...>;
template <typename... Args>
using UnknownStrCat = internal::UnknownError::StrCat<Args...>;

}  // namespace xla

#endif  // XLA_ERROR_FUNCTIONS_H_
