/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/lib/random/weighted_picker.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "xla/tsl/lib/random/simple_philox.h"
#include "xla/tsl/platform/logging.h"

namespace tsl {
namespace random {
namespace {

int CalcNumLevels(int n) {
  CHECK_GE(n, 0);
  const size_t count = static_cast<size_t>(n);
  int levels = 1;
  while ((size_t{1} << (levels - 1)) < count) {
    levels++;
  }
  CHECK_LT(levels, static_cast<int>(sizeof(size_t) * 8));
  return levels;
}

}  // namespace

WeightedPicker::WeightedPicker(int N)
    : N_(N),
      num_levels_(CalcNumLevels(N)),
      level_(new int32_t[LevelSize(num_levels_)]) {
  level_[0] = 0;
  SetAllWeights(1);
}

WeightedPicker::~WeightedPicker() { delete[] level_; }

static int32_t UnbiasedUniform(SimplePhilox* r, int32_t n) {
  CHECK_LE(0, n);
  const uint32_t range = ~static_cast<uint32_t>(0);
  if (n == 0) {
    return r->Rand32() * n;
  } else if (0 == (n & (n - 1))) {
    // N is a power of two, so just mask off the lower bits.
    return r->Rand32() & (n - 1);
  } else {
    // Reject all numbers that skew the distribution towards 0.

    // Rand32's output is uniform in the half-open interval [0, 2^{32}).
    // For any interval [m,n), the number of elements in it is n-m.

    uint32_t rem = (range % n) + 1;
    uint32_t rnd;

    // rem = ((2^{32}-1) \bmod n) + 1
    // 1 <= rem <= n

    // NB: rem == n is impossible, since n is not a power of 2 (from
    // earlier check).

    do {
      rnd = r->Rand32();  // rnd uniform over [0, 2^{32})
    } while (rnd < rem);  // reject [0, rem)
    // rnd is uniform over [rem, 2^{32})
    //
    // The number of elements in the half-open interval is
    //
    //  2^{32} - rem = 2^{32} - ((2^{32}-1) \bmod n) - 1
    //               = 2^{32}-1 - ((2^{32}-1) \bmod n)
    //               = n \cdot \lfloor (2^{32}-1)/n \rfloor
    //
    // therefore n evenly divides the number of integers in the
    // interval.
    //
    // The function v \rightarrow v % n takes values from [bias,
    // 2^{32}) to [0, n).  Each integer in the range interval [0, n)
    // will have exactly \lfloor (2^{32}-1)/n \rfloor preimages from
    // the domain interval.
    //
    // Therefore, v % n is uniform over [0, n).  QED.

    return rnd % n;
  }
}

int WeightedPicker::Pick(SimplePhilox* rnd) const {
  if (total_weight() == 0) return -1;

  // using unbiased uniform distribution to avoid bias
  // toward low elements resulting from a possible use
  // of big weights.
  return PickAt(UnbiasedUniform(rnd, total_weight()));
}

int WeightedPicker::PickAt(int32_t weight_index) const {
  if (weight_index < 0 || weight_index >= total_weight()) return -1;

  int32_t position = weight_index;
  const size_t leaf_offset = LevelSize(num_levels_ - 1);
  const int32_t* const tree = level_;
  size_t node = 1;

  while (node < leaf_offset) {
    // Pick left or right child of "tree[node]"
    const size_t left = 2 * node;
    const int32_t left_weight = tree[left];
    if (position < left_weight) {
      // Descend to left child
      node = left;
    } else {
      // Descend to right child
      node = left + 1;
      position -= left_weight;
    }
  }
  const int index = static_cast<int>(node - leaf_offset);
  CHECK_GE(index, 0);
  CHECK_LT(index, N_);
  CHECK_LE(position, tree[node]);
  return index;
}

void WeightedPicker::set_weight(int index, int32_t weight) {
  assert(index >= 0);
  assert(index < N_);

  // Adjust the sums all the way up to the root
  size_t node = LevelSize(num_levels_ - 1) + static_cast<size_t>(index);
  const int32_t delta = weight - level_[node];
  for (; node >= 1; node >>= 1) {
    level_[node] += delta;
  }
}

void WeightedPicker::SetAllWeights(int32_t weight) {
  // Initialize leaves
  const size_t leaf_offset = LevelSize(num_levels_ - 1);
  const size_t count = static_cast<size_t>(N_);
  int32_t* leaves = level_ + leaf_offset;
  std::fill_n(leaves, count, weight);
  std::fill_n(leaves + count, leaf_offset - count, 0);

  // Now sum up towards the root
  RebuildTreeWeights();
}

void WeightedPicker::SetWeightsFromArray(int N, const int32_t* weights) {
  Resize(N);

  // Initialize leaves
  const size_t leaf_offset = LevelSize(num_levels_ - 1);
  const size_t count = static_cast<size_t>(N_);
  int32_t* leaves = level_ + leaf_offset;
  if (count > 0) {
    std::copy_n(weights, count, leaves);
  }
  std::fill_n(leaves + count, leaf_offset - count, 0);

  // Now sum up towards the root
  RebuildTreeWeights();
}

void WeightedPicker::RebuildTreeWeights() {
  int32_t* const tree = level_;
  for (size_t i = LevelSize(num_levels_ - 1) - 1; i >= 1; --i) {
    tree[i] = tree[2 * i] + tree[2 * i + 1];
  }
}

void WeightedPicker::Append(int32_t weight) {
  Resize(num_elements() + 1);
  set_weight(num_elements() - 1, weight);
}

void WeightedPicker::Resize(int new_size) {
  CHECK_GE(new_size, 0);
  const size_t new_count = static_cast<size_t>(new_size);
  if (new_count <= LevelSize(num_levels_ - 1)) {
    // The new picker fits in the existing levels.

    // First zero out any of the weights that are being dropped so
    // that the levels are correct (only needed when shrinking)
    for (int i = new_size; i < N_; i++) {
      set_weight(i, 0);
    }

    // We do not need to set any new weights when enlarging because
    // the unneeded entries always have weight zero.
    N_ = new_size;
    return;
  }

  assert(new_size > N_);
  const int new_num_levels = CalcNumLevels(new_size);
  const size_t new_leaf_offset = LevelSize(new_num_levels - 1);
  const size_t old_count = static_cast<size_t>(N_);
  int32_t* new_level = new int32_t[LevelSize(new_num_levels)];
  new_level[0] = 0;
  int32_t* dst = new_level + new_leaf_offset;
  const int32_t* src = level_ + LevelSize(num_levels_ - 1);
  if (old_count > 0) {
    std::copy_n(src, old_count, dst);
  }
  std::fill_n(dst + old_count, new_leaf_offset - old_count, 0);

  delete[] level_;
  level_ = new_level;
  num_levels_ = new_num_levels;
  N_ = new_size;
  RebuildTreeWeights();
}

}  // namespace random
}  // namespace tsl
