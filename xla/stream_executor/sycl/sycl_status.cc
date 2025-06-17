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

#include "xla/stream_executor/sycl/sycl_status.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace stream_executor::gpu {
std::string ToString(SYCLError_t error) {
  switch (error) {
    case SYCL_SUCCESS:
      return "SYCL succeeded.";
    case SYCL_ERROR_NO_DEVICE:
      return "SYCL did not find the device.";
    case SYCL_ERROR_INVALID_DEVICE:
      return "SYCL got invalid device id.";
    case SYCL_ERROR_INVALID_POINTER:
      return "SYCL got invalid pointer.";
    case SYCL_ERROR_INVALID_STREAM:
      return "SYCL got invalid stream.";
    case SYCL_ERROR_DESTROY_DEFAULT_STREAM:
      return "SYCL cannot destroy default stream.";
    case SYCL_ERROR_NOT_READY:
      return "SYCL is not ready.";
    case SYCL_ERROR_ZE_ERROR:
      return "SYCL got ZE error.";
    default:
      return "SYCL got invalid error code.";
  }
}
}  // namespace stream_executor::gpu
