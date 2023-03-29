/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/runtime/topk_kernel.h"

#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"

namespace xla::gpu {

using ::tensorflow::se::gpu::GpuStreamHandle;
using ::xla::runtime::ffi::FfiStatus;
using ::xla::runtime::ffi::PrimitiveType;

FfiStatus RunTopk(GpuStreamHandle stream, PrimitiveType dtype, void* data,
                  size_t num_elements, void* top_elements,
                  uint32_t* top_indices, size_t k, size_t batch_size,
                  void* scratch) {
  VLOG(2) << "TopK: " << PrimitiveTypeToString(dtype) << ", n: " << num_elements
          << ", k: " << k << ", bs: " << batch_size;
  return FfiStatus::Internal("GpuTopK does not yet have an implementation.");
}

}  // namespace xla::gpu
