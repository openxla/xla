/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_TOPK_LIB_H_
#define XLA_BACKENDS_CPU_RUNTIME_TOPK_LIB_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/base/dynamic_annotations.h"

namespace xla::cpu::internal {

inline bool ByteRangesOverlap(const void* a, size_t a_bytes, const void* b,
                              size_t b_bytes) {
  if (a_bytes == 0 || b_bytes == 0) {
    return false;
  }
  const uintptr_t ua = absl::bit_cast<uintptr_t>(a);
  const uintptr_t ub = absl::bit_cast<uintptr_t>(b);
  return ua < ub ? (ub - ua < a_bytes) : (ua - ub < b_bytes);
}

template <typename T>
void TopK(int64_t batch_size, int64_t input_size, int64_t k, const T* values,
          T* out_values, int32_t* out_indices) {
  if (batch_size <= 0 || input_size <= 0 || k <= 0 || k > input_size ||
      input_size > std::numeric_limits<int32_t>::max() || values == nullptr ||
      out_values == nullptr || out_indices == nullptr) {
    return;
  }
  const size_t total_inputs =
      static_cast<size_t>(batch_size) * static_cast<size_t>(input_size);
  const size_t input_bytes = total_inputs * sizeof(T);
  const size_t total_outputs =
      static_cast<size_t>(batch_size) * static_cast<size_t>(k);

  // values is managed by the JIT code, so msan can't tell they are initialized.
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(values, input_bytes);

  std::unique_ptr<T[]> snapshot;
  if (ByteRangesOverlap(values, input_bytes, out_values,
                        total_outputs * sizeof(T)) ||
      ByteRangesOverlap(values, input_bytes, out_indices,
                        total_outputs * sizeof(int32_t))) {
    snapshot = std::make_unique<T[]>(total_inputs);
    std::memcpy(snapshot.get(), values, input_bytes);
    values = snapshot.get();
  }

  auto convert_to_int = [](T value) {
    uint32_t x = absl::bit_cast<uint32_t>(value);
    return static_cast<int32_t>(x) < 0 ? std::numeric_limits<int32_t>::max() - x
                                       : x;
  };

  std::vector<int32_t> temp_indices(input_size);
  for (int64_t batch = 0; batch != batch_size; ++batch) {
    absl::c_iota(temp_indices, 0);

    const T* values_batch = values + batch * input_size;

    auto kth_element = temp_indices.begin() + k;
    std::partial_sort(temp_indices.begin(), kth_element, temp_indices.end(),
                      [&](size_t i1, size_t i2) {
                        // Do the comparison in integers to enforce a total
                        // order of -NaN < -Inf < -0 < +0 < +Inf < +NaN.
                        int32_t v1 = convert_to_int(values_batch[i1]);
                        int32_t v2 = convert_to_int(values_batch[i2]);
                        if (v1 == v2) {
                          return i1 < i2;  // Stabilize sorting.
                        }
                        return v1 > v2;
                      });

    T* out_values_batch = out_values + batch * k;
    int32_t* out_indices_batch = out_indices + batch * k;
    std::copy(temp_indices.begin(), kth_element, out_indices_batch);
    for (int64_t i = 0; i < k; i++) {
      std::memcpy(&out_values_batch[i], &values_batch[temp_indices[i]],
                  sizeof(T));
    }
  }
}

}  // namespace xla::cpu::internal

#endif  // XLA_BACKENDS_CPU_RUNTIME_TOPK_LIB_H_
