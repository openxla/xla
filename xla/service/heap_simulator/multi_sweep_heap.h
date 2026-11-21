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

#ifndef XLA_SERVICE_HEAP_SIMULATOR_MULTI_SWEEP_HEAP_H_
#define XLA_SERVICE_HEAP_SIMULATOR_MULTI_SWEEP_HEAP_H_

#include <cstdint>
#include <limits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_value.h"

namespace xla {

// A heap algorithm that packs the buffers in several ways (16 by default) and
// keeps the packing with the smallest heap. Each packing takes O(n log n) time
// for n buffers. The result is deterministic.
//
// Like FAST_MERGE (GlobalDecreasingSizeBestFitHeap::kFastMerge), it places a
// colocation group (a buffer and the buffers that share with it) as one block:
// all members get the same offset, and the block is as big as the largest
// member. Unlike FAST_MERGE, the block is not reserved for the whole time from
// the first member's start to the last member's end. While none of its members
// is live (a "gap"), its memory can hold other buffers that are freed before
// the gap ends. A buffer can also use a gap together with the free memory right
// below and above it.
//
// Each packing goes through time once (a "sweep") and places every block when
// it starts, in the smallest free space that is big enough (best fit), like
// FAST_MERGE does. The packings differ in their Direction and Side.
//
// Like FAST_MERGE, when `size_limit_per_heap` is set, blocks that don't fit in
// the current heap go into new heaps (a block larger than the limit goes into
// an empty heap). FAST_MULTI_SWEEP runs MultiSweepHeap and FAST_MERGE and
// keeps the smaller result, so it never uses more memory than FAST_MERGE.
class MultiSweepHeap : public HeapAlgorithm<HloValue> {
 public:
  // The order in which a packing places the blocks.
  enum class Direction {
    // Forward in time, like FAST_MERGE.
    kForward,
    // Backward in time.
    kBackward,
    // From a peak: the blocks that are live at the peak are stacked from
    // offset 0 first. Then a backward sweep places the blocks that end before
    // the peak, and a forward sweep places the blocks that start after it.
    // The peak is the earliest time with the most live bytes. The stack is
    // sorted by size (largest first), start (earliest first), end (latest
    // first), or lifetime (longest first).
    kFromPeakBySize,
    kFromPeakByStart,
    kFromPeakByEnd,
    kFromPeakByLifetime,
    // Like the above, but the peak is the time when the live blocks are the
    // biggest: when the sum of their squared sizes is the largest.
    kFromBigPeakByStart,
    kFromBigPeakByEnd,
  };

  // Which end of a free chunk (outside gaps) a block goes to.
  enum class Side {
    // The low end, like in FAST_MERGE.
    kLow,
    // The end next to the placed block whose release is closest in time to
    // the new block's release, or the low end if there is no such neighbor.
    kNearestRelease,
  };

  struct Packing {
    Direction direction;
    Side side;
  };

  // The 8 directions with each of the 2 sides: 16 packings.
  static std::vector<Packing> AllPackings();

  // Tries `packings` in order and keeps the first one with the smallest total
  // heap size. `alignment` must be positive.
  explicit MultiSweepHeap(
      int64_t alignment,
      uint64_t size_limit_per_heap = std::numeric_limits<uint64_t>::max(),
      std::vector<Packing> packings = AllPackings());
  ~MultiSweepHeap() override = default;

  void Alloc(const HloValue* buffer, int64_t size) override;
  // Like in GlobalDecreasingSizeBestFitHeap, `share_with` must have been
  // passed to Alloc or ShareWith before, with a non-zero size.
  void ShareWith(const HloValue* buffer, const HloValue* share_with,
                 int64_t size) override;
  void Free(const HloValue* buffer, int64_t size) override;
  absl::StatusOr<Result> Finish() override;

 private:
  // The live range of a buffer of non-zero size, in heap time: like in
  // GlobalDecreasingSizeBestFitHeap, every Alloc, ShareWith and Free call of a
  // non-zero size buffer takes one time step.
  struct Interval {
    const HloValue* buffer;
    int64_t size;
    int64_t start;
    // -1 until the buffer is freed. Buffers that are never freed live until
    // the end.
    int64_t end;
    // The index in `intervals_` of the first member of the buffer's
    // colocation group.
    int64_t group;
  };

  int64_t alignment_;
  uint64_t size_limit_per_heap_;
  std::vector<Packing> packings_;
  int64_t current_time_ = 0;
  // In the order of the Alloc and ShareWith calls, so sorted by start.
  std::vector<Interval> intervals_;
  absl::flat_hash_map<const HloValue*, int64_t> interval_index_;
  std::vector<const HloValue*> zero_size_buffers_;
};

}  // namespace xla

#endif  // XLA_SERVICE_HEAP_SIMULATOR_MULTI_SWEEP_HEAP_H_
