/*
 * Copyright 2023 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_PYTHON_IFRT_PROXY_COMMON_PROTO_UTIL_H_
#define XLA_PYTHON_IFRT_PROXY_COMMON_PROTO_UTIL_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"

// Utility functions to convert between the IFRT service protos and the c++
// types needed to work with the backend runtime.

namespace xla {
namespace ifrt {
namespace proxy {

// Makes an IfrtResponse proto with the given metadata.
std::unique_ptr<IfrtResponse> NewIfrtResponse(
    uint64_t op_id, absl::Status status = absl::OkStatus());

#if defined(PLATFORM_GOOGLE)
inline absl::string_view as_protobuf_string(
    absl::string_view s ABSL_ATTRIBUTE_LIFETIME_BOUND) {
  return s;
}
#else
inline std::string as_protobuf_string(absl::string_view s) {
  return std::string(s);
}
#endif

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_COMMON_PROTO_UTIL_H_
