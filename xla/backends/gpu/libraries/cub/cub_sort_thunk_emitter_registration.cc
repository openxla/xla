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

#include "xla/backends/gpu/libraries/cub/cub_sort_thunk_emitter.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registration.h"
#include "xla/service/gpu/cublas_cudnn.h"

namespace xla::gpu {
namespace {

XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER(kCubDeviceRadixSortKeysTarget,
                                            EmitCudaCubSortNativeCustomCall);
XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER(kCubDeviceRadixSortPairsTarget,
                                            EmitCudaCubSortNativeCustomCall);

}  // namespace
}  // namespace xla::gpu
