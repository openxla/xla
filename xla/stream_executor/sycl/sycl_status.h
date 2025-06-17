/* Copyright (c) 2025 The OpenXLA Authors.
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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_STATUS_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_STATUS_H_

#include <string>
#include <sycl/sycl.hpp>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"

namespace stream_executor::gpu {

enum SYCLError_t {
    SYCL_SUCCESS,
    SYCL_ERROR_NO_DEVICE,
    SYCL_ERROR_NOT_READY,
    SYCL_ERROR_INVALID_DEVICE,
    SYCL_ERROR_INVALID_POINTER,
    SYCL_ERROR_INVALID_STREAM,
    SYCL_ERROR_DESTROY_DEFAULT_STREAM,
    SYCL_ERROR_ZE_ERROR,
};

// Returns a textual description of the given SYCLError_t.
std::string ToString(SYCLError_t result);

// Returns an absl::Status corresponding to the SYCLError_t.
inline absl::Status ToStatus(SYCLError_t result,
                             absl::string_view detail = "") {
    if (ABSL_PREDICT_TRUE(result == SYCL_SUCCESS)) {
        return absl::OkStatus();
    }
    std::string error_message = absl::StrCat(detail, ": ", ToString(result));
    return absl::InternalError(error_message);
}

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_STATUS_H_
