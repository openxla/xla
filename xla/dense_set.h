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

#ifndef TENSORFLOW_COMPILER_XLA_DENSE_SET_H_
#define TENSORFLOW_COMPILER_XLA_DENSE_SET_H_

#include <algorithm>
#include <initializer_list>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"

namespace xla {

// A dense, sorted set, optimized for fast iteration.
template <typename T, typename LessThan = std::less<T>,
          typename Vector = std::vector<T>>
class DenseSet {
 public:
  using const_iterator = typename Vector::const_iterator;

  DenseSet() = default;
  DenseSet(std::initializer_list<T> values)
      : DenseSet(absl::Span<T const>(values)) {}

  explicit DenseSet(absl::Span<T const> values)
      : values_(values.begin(), values.end()) {
    Sort();
    Deduplicate();
  }

  explicit DenseSet(const absl::flat_hash_set<T>& values)
      : values_(values.begin(), values.end()) {
    Sort();  // Values are already unique, so only need to sort.
  }

  // Inserts a value into the set, if no equivalent value is present (O(n)).
  // Returns `true` if the value was inserted,; `false` otherwise..
  std::pair<const_iterator, bool> insert(T value) {
    auto it = absl::c_lower_bound(values_, value, LessThan());
    bool do_insert = (it == values_.end()) || (*it != value);
    return {do_insert ? values_.insert(it, std::move(value)) : it, do_insert};
  }

  bool empty() const { return values_.empty(); }
  size_t size() const { return values_.size(); }
  void reserve(size_t n) { values_.reserve(n); }

  // Returns whether the set contains the given value (O(log(n))).
  bool contains(const T& value) const {
    return absl::c_binary_search(values_, value, LessThan());
  }

  const_iterator begin() const { return values_.begin(); }
  const_iterator end() const { return values_.end(); }

  bool operator==(const DenseSet& other) const {
    return values_ == other.values_;
  }
  bool operator!=(const DenseSet& other) const { return !(*this == other); }

  // Takes ownership of the underlying container of values.
  Vector&& TakeValues() { return std::move(values_); }

 protected:
  void Sort() { absl::c_sort(values_, LessThan()); }
  void Deduplicate() {
    values_.erase(std::unique(values_.begin(), values_.end()), values_.end());
  }

  Vector values_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_DENSE_SET_H_
