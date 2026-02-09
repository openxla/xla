/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_FFI_FFI_API_H_
#define XLA_FFI_FFI_API_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "xla/ffi/ffi_interop.h"  // IWYU pragma: keep

// This is an implementation of XLA FFI API defined in `api/c_api.h` header. It
// should be linked statically into the "main" XLA binary, and third party FFI
// handlers can be linked and registered dynamically.
//
// FFI handlers registered statically (and built from the same XLA commit with
// the same toolchain) can also use `api/c_api_internal.h` to get access to
// various internal data structures.

namespace xla::ffi {

//===----------------------------------------------------------------------===//
// XLA FFI registry
//===----------------------------------------------------------------------===//

struct HandlerRegistration {
  XLA_FFI_Metadata metadata;
  XLA_FFI_Handler_Bundle bundle;
};

bool IsCommandBufferCompatible(const XLA_FFI_Metadata& metadata);

// Returns registered FFI handler for a given name and platform, or an error if
// it's not found in the static registry.
absl::StatusOr<HandlerRegistration> FindHandler(absl::string_view name,
                                                absl::string_view platform);

// Returns all registered calls in the static registry for a given platform.
absl::StatusOr<absl::flat_hash_map<std::string, HandlerRegistration>>
StaticRegisteredHandlers(absl::string_view platform);

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

// Decodes XLA FFI traits packed into a 32-bit integer into a vector of traits.
inline std::vector<Traits> DecodeTraits(XLA_FFI_Handler_Traits traits) {
  std::vector<Traits> result;
  if (traits & XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE) {
    result.push_back(Traits::kCmdBufferCompatible);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Pretty printinting for FFI C++ types.
//===----------------------------------------------------------------------===//

template <typename Sink>
static void AbslStringify(Sink& sink, Traits traits) {
  switch (traits) {
    case Traits::kCmdBufferCompatible:
      absl::Format(&sink, "cmd_buffer_compatible");
      break;
  }
}

}  // namespace xla::ffi

//===----------------------------------------------------------------------===//
// Pretty printinting for FFI C types.
//===----------------------------------------------------------------------===//

template <typename Sink>
static void AbslStringify(Sink& sink, const XLA_FFI_TypeId& type_id) {
  if (type_id.type_id == XLA_FFI_UNKNOWN_TYPE_ID.type_id) {
    absl::Format(&sink, "unknown");
  } else {
    absl::Format(&sink, "%d", type_id.type_id);
  }
}

template <typename Sink>
static void AbslStringify(Sink& sink, const XLA_FFI_Metadata& metadata) {
  absl::Format(&sink, "{api_version: %d.%d, traits: [%s], state: %v}",
               metadata.api_version.major_version,
               metadata.api_version.minor_version,
               absl::StrJoin(xla::ffi::DecodeTraits(metadata.traits), ", "),
               metadata.state_type_id);
}

#endif  // XLA_FFI_FFI_API_H_
