/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_CPU_BUFFER_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_CPU_BUFFER_UTILS_H_

#include "xla/literal.h"
#include "xla/pjrt/tracked_tfrt_cpu_device_buffer.h"

namespace xla {

void CopyCpuBufferToLiteral(const Shape& on_device_shape,
                            TrackedTfrtCpuDeviceBuffer* device_buffer,
                            MutableLiteralBase* literal);

void CopyLiteralSliceToLeafCpuBuffer(int leaf_index,
                                     TrackedTfrtCpuDeviceBuffer* device_buffer,
                                     const LiteralSlice& literal_slice);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_CPU_BUFFER_UTILS_H_
