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

#include "xla/pjrt/cpu_buffer_utils.h"

#include <cstring>
#include <memory>

namespace xla {

void CopyCpuBufferToLiteral(const Shape& on_device_shape,
                            TrackedTfrtCpuDeviceBuffer* device_buffer,
                            MutableLiteralBase* literal) {
  if (!on_device_shape.IsTuple()) {
    const std::shared_ptr<MaybeOwningCpuMemory>& b =
        device_buffer->Buffers()[0];
    std::memcpy(literal->untyped_data(), b->data(), b->size());
  } else {
    // Tuple case.
    int num_leaves = literal->shape().tuple_shapes().size();
    for (int i = 0; i < num_leaves; ++i) {
      const std::shared_ptr<MaybeOwningCpuMemory>& b =
          device_buffer->Buffers()[i];
      std::memcpy(literal->untyped_data({i}), b->data(), b->size());
    }
  }
}

void CopyLiteralSliceToLeafCpuBuffer(int leaf_index,
                                     TrackedTfrtCpuDeviceBuffer* device_buffer,
                                     const LiteralSlice& literal_slice) {
  const std::shared_ptr<MaybeOwningCpuMemory>& b =
      device_buffer->Buffers()[leaf_index];
  CHECK_EQ(literal_slice.size_bytes(), b->size());
  std::memcpy(b->data(), literal_slice.untyped_data(),
              literal_slice.size_bytes());
}

}  // namespace xla
