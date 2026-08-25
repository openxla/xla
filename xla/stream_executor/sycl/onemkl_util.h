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

#ifndef XLA_STREAM_EXECUTOR_SYCL_ONEMKL_UTIL_H_
#define XLA_STREAM_EXECUTOR_SYCL_ONEMKL_UTIL_H_

#include <exception>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "oneapi/mkl.hpp"

namespace stream_executor {
namespace sycl {

// Helper functions to make oneMKL calls. For MKL calls that take a queue it
// returns a status if failed or an event object if it succeeds. The caller
// can then check for ok, and use the returend object as needed.
// Example to call oneapi::mkl::blas::trsm
//  absl::StatusOr<::sycl::event> status =
//      ExecMklFunc(AS_LAMBDA(oneapi::mkl::blas::trsm), q, left_right,
//                  upper_lower, trans, unit_diag, m, n, alpha, a, lda, b, ldb);
//
// if (result.ok()) {
//   result->DoSomethingCool();
// } else {
//   LOG(ERROR) << result.status();
// }
// For MKL calls that that do not take a queue of their own -- e.g. the DFT
// ones, which run on the queue their descriptor was committed to -- are
// called the same way:
//  absl::Status status =
//      ExecMklFunc(AS_LAMBDA(oneapi::mkl::dft::compute_forward), descriptor,
//                  input, output);
// It returns a plain absl::Status.

#define AS_LAMBDA(func)                                            \
  [](auto&&... args) -> decltype(func(                             \
                         std::forward<decltype(args)>(args)...)) { \
    return func(std::forward<decltype(args)>(args)...);            \
  }

// The type ExecMklFunc() returns for a oneMKL routine returning `R`.
template <typename R>
using MklResult =
    std::conditional_t<std::is_void_v<R>, absl::Status, absl::StatusOr<R>>;

template <typename Callable, typename... Args>
MklResult<std::invoke_result_t<Callable, Args...>> ExecMklFunc(
    Callable&& mkl_func, Args&&... args) {
  using Result = std::invoke_result_t<Callable, Args...>;
  try {
    if constexpr (std::is_void_v<Result>) {
      std::forward<Callable>(mkl_func)(std::forward<Args>(args)...);
      return absl::OkStatus();
    } else {
      return std::forward<Callable>(mkl_func)(std::forward<Args>(args)...);
    }
  } catch (oneapi::mkl::exception const& e) {
    return absl::InternalError(absl::StrCat("Mkl exception: ", e.what()));
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(absl::StrCat(
        "SYCL exception: ", e.what(), " [sycl_code=", e.code().value(), "]"));
  } catch (const std::exception& e) {
    return absl::InternalError(absl::StrCat("Exception: ", e.what()));
  } catch (...) {
    return absl::InternalError("Unknown (non-std::exception) thrown");
  }
}
}  // namespace sycl
}  // namespace stream_executor
#endif  // XLA_STREAM_EXECUTOR_SYCL_ONEMKL_UTIL_H_