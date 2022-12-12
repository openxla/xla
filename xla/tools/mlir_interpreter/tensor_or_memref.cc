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

#include "xla/tools/mlir_interpreter/tensor_or_memref.h"

namespace mlir {
namespace interpreter {

int64_t BufferView::physical_index(llvm::ArrayRef<int64_t> view_indices) const {
  int64_t result = offset;
  for (int64_t i = 0; i < view_indices.size(); ++i) {
    result += view_indices[i] * strides[i];
  }
  return result;
}

bool BufferView::InBounds(llvm::ArrayRef<int64_t> view_indices) const {
  for (auto [index, size] : llvm::zip(view_indices, sizes)) {
    if (index < 0 || index >= size) return false;
  }
  return true;
}

SmallVector<int64_t> BufferView::DefaultStrides(ArrayRef<int64_t> sizes) {
  SmallVector<int64_t> result(sizes.size());
  int64_t stride = 1;
  for (int64_t i = result.size() - 1; i >= 0; --i) {
    result[i] = stride;
    stride *= sizes[i];
  }
  return result;
}

}  // namespace interpreter
}  // namespace mlir
