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

#ifndef XLA_STREAM_EXECUTOR_ROCM_HIPDNN_FRONTEND_HELPERS_H_
#define XLA_STREAM_EXECUTOR_ROCM_HIPDNN_FRONTEND_HELPERS_H_

namespace stream_executor {
namespace gpu {

#define RETURN_IF_HIPDNN_FRONTEND_ERROR(expr)                               \
  do {                                                                      \
    auto status = (expr);                                                   \
    if (ABSL_PREDICT_FALSE(status.is_bad())) {                              \
      std::ostringstream oss;                                               \
      oss << status.get_message() << "\nin " << __FILE__ << "(" << __LINE__ \
          << "): '" << #expr << "' ";                                       \
      return absl::InternalError(oss.str());                                \
    }                                                                       \
  } while (false)

#define RETURN_HIPDNN_FRONTEND_STATUS(expr) \
  do {                                      \
    RETURN_IF_HIPDNN_FRONTEND_ERROR(expr);  \
    return absl::OkStatus();                \
  } while (false)

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_HIPDNN_FRONTEND_HELPERS_H_
