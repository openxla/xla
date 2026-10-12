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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_CONSTANTS_H_
#define XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_CONSTANTS_H_

#include "absl/strings/string_view.h"

namespace xla::gpu {

// Frontend attribute that `CustomCallScratchAssigner` sets on a custom call
// whose result it extended by scratch buffers. Its value is the number of
// appended scratch buffers; a custom call without the attribute has none.
inline constexpr absl::string_view kNativeCustomCallNumScratchBuffersAttr =
    "xla_gpu_native_custom_call_num_scratch_buffers";

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_CONSTANTS_H_
