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

#include "xla/service/heap_simulator/multi_sweep_heap.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal_util.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

using Chunk = ::xla::HeapSimulator::Chunk;
using Result = ::xla::HeapSimulator::Result<HloValue>;

// A call to a heap algorithm.
struct HeapCall {
  enum class Kind { kAlloc, kShareWith, kFree };
  Kind kind;
  const HloValue* buffer;
  int64_t size;
  const HloValue* share_with = nullptr;  // For kShareWith.
};

void Replay(absl::Span<const HeapCall> calls, HeapAlgorithm<HloValue>& heap) {
  for (const HeapCall& call : calls) {
    switch (call.kind) {
      case HeapCall::Kind::kAlloc:
        heap.Alloc(call.buffer, call.size);
        break;
      case HeapCall::Kind::kShareWith:
        heap.ShareWith(call.buffer, call.share_with, call.size);
        break;
      case HeapCall::Kind::kFree:
        heap.Free(call.buffer, call.size);
        break;
    }
  }
}

// Checks `result` against `calls` (chunks, alignment, colocations, overlaps,
// per-heap size limits, and heap sizes) and returns any problems found.
std::vector<std::string> FindProblems(
    absl::Span<const HeapCall> calls, const Result& result, int64_t alignment,
    uint64_t size_limit_per_heap = std::numeric_limits<uint64_t>::max()) {
  std::vector<std::string> problems;
  if (result.heap_results.empty()) {
    problems.push_back("expected at least 1 heap, got 0");
    return problems;
  }
  if (size_limit_per_heap == std::numeric_limits<uint64_t>::max() &&
      result.heap_results.size() != 1) {
    problems.push_back(
        absl::StrCat("expected 1 heap, got ", result.heap_results.size()));
    return problems;
  }
  auto locate =
      [&](const HloValue* buffer) -> std::optional<std::pair<size_t, Chunk>> {
    std::optional<std::pair<size_t, Chunk>> found;
    for (size_t h = 0; h < result.heap_results.size(); ++h) {
      auto it = result.heap_results[h].chunk_map.find(buffer);
      if (it == result.heap_results[h].chunk_map.end()) {
        continue;
      }
      if (found.has_value()) {
        problems.push_back(absl::StrCat(buffer->ToShortString(),
                                        " appears in more than one heap"));
      } else {
        found = std::make_pair(h, it->second);
      }
    }
    return found;
  };
  // The live range of every buffer of non-zero size, with the call index as
  // the time.
  struct LiveRange {
    const HloValue* buffer;
    int64_t size;
    int64_t start;
    int64_t end;
    // The first buffer of the buffer's colocation group.
    const HloValue* group;
  };
  std::vector<LiveRange> live_ranges;
  absl::flat_hash_map<const HloValue*, int64_t> index;
  for (int64_t time = 0; time < static_cast<int64_t>(calls.size()); ++time) {
    const HeapCall& call = calls[time];
    if (call.size == 0) {
      if (call.kind != HeapCall::Kind::kFree) {
        const std::optional<std::pair<size_t, Chunk>> found =
            locate(call.buffer);
        if (!found.has_value() || found->first != 0 ||
            !(found->second == Chunk::FromOffsetSize(0, 0))) {
          problems.push_back(absl::StrCat(
              call.buffer->ToShortString(),
              " has size 0 but not chunk [0, 0) in the first heap"));
        }
      }
      continue;
    }
    switch (call.kind) {
      case HeapCall::Kind::kAlloc:
      case HeapCall::Kind::kShareWith: {
        const HloValue* group = call.buffer;
        if (call.kind == HeapCall::Kind::kShareWith) {
          if (auto it = index.find(call.share_with); it != index.end()) {
            group = live_ranges[it->second].group;
          }
        }
        index[call.buffer] = live_ranges.size();
        live_ranges.push_back({call.buffer, call.size, time,
                               std::numeric_limits<int64_t>::max(), group});
        break;
      }
      case HeapCall::Kind::kFree:
        live_ranges[index.at(call.buffer)].end = time;
        break;
    }
  }
  absl::flat_hash_map<const HloValue*, int64_t> group_size;
  for (const LiveRange& live_range : live_ranges) {
    int64_t& size = group_size[live_range.group];
    size = std::max(size, live_range.size);
  }
  std::vector<int64_t> max_chunk_end(result.heap_results.size(), 0);
  std::vector<int> oversized_groups(result.heap_results.size(), 0);
  absl::flat_hash_map<const HloValue*, std::pair<size_t, int64_t>>
      group_placement;
  std::vector<size_t> heap_of;
  std::vector<Chunk> chunks;
  for (const LiveRange& live_range : live_ranges) {
    const std::optional<std::pair<size_t, Chunk>> found =
        locate(live_range.buffer);
    if (!found.has_value()) {
      problems.push_back(
          absl::StrCat(live_range.buffer->ToShortString(), " has no chunk"));
      return problems;
    }
    const auto [h, chunk] = *found;
    heap_of.push_back(h);
    chunks.push_back(chunk);
    if (chunk.size != group_size.at(live_range.group) || chunk.offset < 0 ||
        chunk.offset % alignment != 0) {
      problems.push_back(absl::StrCat(live_range.buffer->ToShortString(),
                                      " has the wrong chunk ",
                                      chunk.ToString()));
    }
    max_chunk_end[h] = std::max(max_chunk_end[h], chunk.chunk_end());
    auto [group, inserted] = group_placement.try_emplace(
        live_range.group, std::make_pair(h, chunk.offset));
    if (inserted &&
        static_cast<uint64_t>(chunk.chunk_end()) > size_limit_per_heap) {
      ++oversized_groups[h];
      if (chunk.offset != 0 ||
          static_cast<uint64_t>(chunk.size) <= size_limit_per_heap ||
          oversized_groups[h] > 1) {
        problems.push_back(absl::StrCat("heap ", h, " exceeds limit ",
                                        size_limit_per_heap, " with chunk ",
                                        chunk.ToString()));
      }
    }
    if (!inserted && group->second != std::make_pair(h, chunk.offset)) {
      problems.push_back(
          absl::StrCat(live_range.buffer->ToShortString(),
                       " is not in the same heap and offset as its group"));
    }
  }
  for (size_t i = 0; i < live_ranges.size(); ++i) {
    for (size_t j = i + 1; j < live_ranges.size(); ++j) {
      const LiveRange& x = live_ranges[i];
      const LiveRange& y = live_ranges[j];
      if (heap_of[i] == heap_of[j] && x.group != y.group && x.start <= y.end &&
          y.start <= x.end && chunks[i].OverlapsWith(chunks[j])) {
        problems.push_back(absl::StrCat(
            x.buffer->ToShortString(), " and ", y.buffer->ToShortString(),
            " are live at the same time in heap ", heap_of[i], " and overlap"));
      }
    }
  }
  int64_t total_heap_size = 0;
  for (size_t h = 0; h < result.heap_results.size(); ++h) {
    if (result.heap_results[h].heap_size != max_chunk_end[h]) {
      problems.push_back(absl::StrCat("heap ", h, " has wrong size ",
                                      result.heap_results[h].heap_size,
                                      ", chunks end at ", max_chunk_end[h]));
    }
    total_heap_size += result.heap_results[h].heap_size;
  }
  if (result.heap_size != total_heap_size) {
    problems.push_back(absl::StrCat("wrong total heap size ", result.heap_size,
                                    ", sum of heaps is ", total_heap_size));
  }
  return problems;
}

absl::StatusOr<Result> RunMultiSweepHeap(
    absl::Span<const HeapCall> calls, int64_t alignment,
    std::vector<MultiSweepHeap::Packing> packings =
        MultiSweepHeap::AllPackings(),
    uint64_t size_limit_per_heap = std::numeric_limits<uint64_t>::max()) {
  MultiSweepHeap heap(alignment, size_limit_per_heap, std::move(packings));
  Replay(calls, heap);
  return heap.Finish();
}

absl::StatusOr<Result> RunFastMerge(
    absl::Span<const HeapCall> calls, int64_t alignment,
    uint64_t size_limit_per_heap = std::numeric_limits<uint64_t>::max()) {
  ConstrainedGlobalDecreasingSizeBestFitHeap heap(
      size_limit_per_heap, alignment,
      GlobalDecreasingSizeBestFitHeap<HloValue>::kFastMerge);
  Replay(calls, heap);
  return heap.Finish();
}

class MultiSweepHeapTest : public ::testing::Test {
 protected:
  MultiSweepHeapTest() : builder_("multi_sweep_heap_test") {}

  // Returns a new buffer to pass to the heap algorithms.
  const HloValue* NewBuffer() {
    const HloValue::Id id = buffers_.size();
    HloInstruction* constant = builder_.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
    buffers_.push_back(std::make_unique<HloValue>(id, constant, ShapeIndex{}));
    return buffers_.back().get();
  }

  // Returns random heap calls for `num_buffers` new buffers. Some have size 0.
  // If `share`, some share with an earlier buffer. If not `free_all`, some are
  // never freed.
  std::vector<HeapCall> RandomCalls(std::mt19937& rng, int num_buffers,
                                    bool share = true, bool free_all = false) {
    auto uniform = [&](int64_t low, int64_t high) {
      return std::uniform_int_distribution<int64_t>(low, high)(rng);
    };
    std::vector<HeapCall> calls;
    // The buffers of non-zero size that were allocated so far.
    std::vector<const HloValue*> allocated;
    // The buffers that are not freed yet, with their sizes.
    std::vector<std::pair<const HloValue*, int64_t>> live;
    int num_created = 0;
    while (num_created < num_buffers || !live.empty()) {
      if (num_created < num_buffers && (live.empty() || uniform(0, 2) > 0)) {
        const HloValue* buffer = NewBuffer();
        ++num_created;
        const int64_t size = uniform(0, 9) == 0 ? 0 : uniform(1, 100);
        if (share && !allocated.empty() && uniform(0, 3) == 0) {
          const HloValue* share_with =
              allocated[uniform(0, allocated.size() - 1)];
          calls.push_back(
              {HeapCall::Kind::kShareWith, buffer, size, share_with});
        } else {
          calls.push_back({HeapCall::Kind::kAlloc, buffer, size});
        }
        if (size > 0) {
          allocated.push_back(buffer);
        }
        live.push_back({buffer, size});
        continue;
      }
      const int64_t k = uniform(0, live.size() - 1);
      if (free_all || uniform(0, 19) > 0) {
        calls.push_back({HeapCall::Kind::kFree, live[k].first, live[k].second});
      }
      live.erase(live.begin() + k);
    }
    return calls;
  }

 private:
  HloComputation::Builder builder_;
  std::vector<std::unique_ptr<HloValue>> buffers_;
};

TEST_F(MultiSweepHeapTest, Empty) {
  MultiSweepHeap heap(/*alignment=*/1);
  ASSERT_OK_AND_ASSIGN(const Result result, heap.Finish());
  EXPECT_EQ(result.heap_size, 0);
  ASSERT_EQ(result.heap_results.size(), 1);
  EXPECT_THAT(result.heap_results[0].chunk_map, IsEmpty());
}

TEST_F(MultiSweepHeapTest, ZeroSizeBuffersGetEmptyChunksAtOffsetZero) {
  const HloValue* a = NewBuffer();
  const HloValue* b = NewBuffer();
  const HloValue* c = NewBuffer();
  const std::vector<HeapCall> calls = {
      {HeapCall::Kind::kAlloc, a, 0},        {HeapCall::Kind::kAlloc, b, 10},
      {HeapCall::Kind::kShareWith, c, 0, a}, {HeapCall::Kind::kFree, a, 0},
      {HeapCall::Kind::kFree, b, 10},        {HeapCall::Kind::kFree, c, 0},
  };
  ASSERT_OK_AND_ASSIGN(const Result result,
                       RunMultiSweepHeap(calls, /*alignment=*/1));
  EXPECT_THAT(FindProblems(calls, result, /*alignment=*/1), IsEmpty());
  EXPECT_EQ(result.heap_size, 10);
  const HeapSimulator::HeapResult<HloValue>& heap = result.heap_results.at(0);
  EXPECT_EQ(heap.chunk_map.at(a), Chunk::FromOffsetSize(0, 0));
  EXPECT_EQ(heap.chunk_map.at(b), Chunk::FromOffsetSize(0, 10));
  EXPECT_EQ(heap.chunk_map.at(c), Chunk::FromOffsetSize(0, 0));
}

using MultiSweepHeapDeathTest = MultiSweepHeapTest;

TEST_F(MultiSweepHeapDeathTest, SharingWithUnknownBufferFails) {
  const HloValue* a = NewBuffer();
  const HloValue* b = NewBuffer();
  MultiSweepHeap heap(/*alignment=*/1);
  heap.Alloc(a, 0);
  EXPECT_DEATH(heap.ShareWith(b, a, 10), "ShareWith called with the unknown");
}

// Each buffer of a chain is live together with the next one, which is bigger.
// FAST_MERGE, like the forward sweep, places every buffer at the lowest free
// offset when it starts and needs 400 bytes. A backward sweep needs only the
// 280 bytes that the two biggest buffers use together.
TEST_F(MultiSweepHeapTest, PacksGrowingChain) {
  const HloValue* a = NewBuffer();
  const HloValue* b = NewBuffer();
  const HloValue* c = NewBuffer();
  const HloValue* d = NewBuffer();
  const std::vector<HeapCall> calls = {
      {HeapCall::Kind::kAlloc, a, 40}, {HeapCall::Kind::kAlloc, b, 80},
      {HeapCall::Kind::kFree, a, 40},  {HeapCall::Kind::kAlloc, c, 120},
      {HeapCall::Kind::kFree, b, 80},  {HeapCall::Kind::kAlloc, d, 160},
      {HeapCall::Kind::kFree, c, 120}, {HeapCall::Kind::kFree, d, 160},
  };
  ASSERT_OK_AND_ASSIGN(const Result result,
                       RunMultiSweepHeap(calls, /*alignment=*/1));
  EXPECT_THAT(FindProblems(calls, result, /*alignment=*/1), IsEmpty());
  EXPECT_EQ(result.heap_size, 280);
  ASSERT_OK_AND_ASSIGN(const Result fast_merge,
                       RunFastMerge(calls, /*alignment=*/1));
  EXPECT_EQ(fast_merge.heap_size, 400);
  using Direction = MultiSweepHeap::Direction;
  using Side = MultiSweepHeap::Side;
  ASSERT_OK_AND_ASSIGN(const Result forward,
                       RunMultiSweepHeap(calls, /*alignment=*/1,
                                         {{Direction::kForward, Side::kLow}}));
  EXPECT_EQ(forward.heap_size, 400);
  ASSERT_OK_AND_ASSIGN(const Result backward,
                       RunMultiSweepHeap(calls, /*alignment=*/1,
                                         {{Direction::kBackward, Side::kLow}}));
  EXPECT_EQ(backward.heap_size, 280);
}

// `a` and `c` share a buffer, and `b` is live in the gap between them.
// FAST_MERGE keeps the shared buffer reserved during the gap.
TEST_F(MultiSweepHeapTest, UsesGapOfColocationGroup) {
  const HloValue* a = NewBuffer();
  const HloValue* b = NewBuffer();
  const HloValue* c = NewBuffer();
  const std::vector<HeapCall> calls = {
      {HeapCall::Kind::kAlloc, a, 40},        {HeapCall::Kind::kFree, a, 40},
      {HeapCall::Kind::kAlloc, b, 20},        {HeapCall::Kind::kFree, b, 20},
      {HeapCall::Kind::kShareWith, c, 40, a}, {HeapCall::Kind::kFree, c, 40},
  };
  ASSERT_OK_AND_ASSIGN(const Result result,
                       RunMultiSweepHeap(calls, /*alignment=*/1));
  EXPECT_THAT(FindProblems(calls, result, /*alignment=*/1), IsEmpty());
  EXPECT_EQ(result.heap_size, 40);
  const HeapSimulator::HeapResult<HloValue>& heap = result.heap_results.at(0);
  EXPECT_EQ(heap.chunk_map.at(a), Chunk::FromOffsetSize(0, 40));
  EXPECT_EQ(heap.chunk_map.at(b), Chunk::FromOffsetSize(0, 20));
  EXPECT_EQ(heap.chunk_map.at(c), Chunk::FromOffsetSize(0, 40));
  ASSERT_OK_AND_ASSIGN(const Result fast_merge,
                       RunFastMerge(calls, /*alignment=*/1));
  EXPECT_EQ(fast_merge.heap_size, 60);
}

// `b` is too big for the gap between `a` and `c`, but fits in the gap together
// with the free memory above it, which `x` used before.
TEST_F(MultiSweepHeapTest, UsesGapTogetherWithFreeMemoryAboveIt) {
  const HloValue* a = NewBuffer();
  const HloValue* x = NewBuffer();
  const HloValue* b = NewBuffer();
  const HloValue* c = NewBuffer();
  const std::vector<HeapCall> calls = {
      {HeapCall::Kind::kAlloc, a, 40},        {HeapCall::Kind::kAlloc, x, 40},
      {HeapCall::Kind::kFree, a, 40},         {HeapCall::Kind::kFree, x, 40},
      {HeapCall::Kind::kAlloc, b, 60},        {HeapCall::Kind::kFree, b, 60},
      {HeapCall::Kind::kShareWith, c, 40, a}, {HeapCall::Kind::kFree, c, 40},
  };
  ASSERT_OK_AND_ASSIGN(const Result result,
                       RunMultiSweepHeap(calls, /*alignment=*/1));
  EXPECT_THAT(FindProblems(calls, result, /*alignment=*/1), IsEmpty());
  EXPECT_EQ(result.heap_size, 80);
  EXPECT_EQ(result.heap_results.at(0).chunk_map.at(b),
            Chunk::FromOffsetSize(0, 60));
  ASSERT_OK_AND_ASSIGN(const Result fast_merge,
                       RunFastMerge(calls, /*alignment=*/1));
  EXPECT_EQ(fast_merge.heap_size, 100);
}

// The gap between `a` and `c` sits below the topmost free chunk. `b` fits in
// the gap together with the free memory of `f` below it, leaving `e`'s free
// chunk for `w`.
TEST_F(MultiSweepHeapTest, UsesGapBelowTopTogetherWithFreeMemoryBelowIt) {
  const HloValue* e = NewBuffer();
  const HloValue* pin = NewBuffer();
  const HloValue* f = NewBuffer();
  const HloValue* a = NewBuffer();
  const HloValue* b = NewBuffer();
  const HloValue* w = NewBuffer();
  const HloValue* c = NewBuffer();
  const std::vector<HeapCall> calls = {
      {HeapCall::Kind::kAlloc, e, 80},        {HeapCall::Kind::kAlloc, pin, 20},
      {HeapCall::Kind::kAlloc, f, 30},        {HeapCall::Kind::kAlloc, a, 40},
      {HeapCall::Kind::kFree, e, 80},         {HeapCall::Kind::kFree, f, 30},
      {HeapCall::Kind::kFree, a, 40},         {HeapCall::Kind::kAlloc, b, 60},
      {HeapCall::Kind::kAlloc, w, 80},        {HeapCall::Kind::kFree, b, 60},
      {HeapCall::Kind::kShareWith, c, 40, a}, {HeapCall::Kind::kFree, c, 40},
      {HeapCall::Kind::kFree, w, 80},         {HeapCall::Kind::kFree, pin, 20},
  };
  using Direction = MultiSweepHeap::Direction;
  using Side = MultiSweepHeap::Side;
  ASSERT_OK_AND_ASSIGN(const Result forward,
                       RunMultiSweepHeap(calls, /*alignment=*/1,
                                         {{Direction::kForward, Side::kLow}}));
  EXPECT_THAT(FindProblems(calls, forward, /*alignment=*/1), IsEmpty());
  EXPECT_EQ(forward.heap_size, 170);
  EXPECT_EQ(forward.heap_results.at(0).chunk_map.at(b),
            Chunk::FromOffsetSize(110, 60));
}

// A chain that grows to a peak and then shrinks needs 240 bytes with a forward
// or backward sweep, and 200 bytes when swept both ways from the peak.
TEST_F(MultiSweepHeapTest, PacksDiamondFromPeak) {
  const HloValue* a = NewBuffer();
  const HloValue* b = NewBuffer();
  const HloValue* p = NewBuffer();
  const HloValue* c = NewBuffer();
  const HloValue* d = NewBuffer();
  const std::vector<HeapCall> calls = {
      {HeapCall::Kind::kAlloc, a, 40}, {HeapCall::Kind::kAlloc, b, 80},
      {HeapCall::Kind::kFree, a, 40},  {HeapCall::Kind::kAlloc, p, 120},
      {HeapCall::Kind::kFree, b, 80},  {HeapCall::Kind::kAlloc, c, 80},
      {HeapCall::Kind::kFree, p, 120}, {HeapCall::Kind::kAlloc, d, 40},
      {HeapCall::Kind::kFree, c, 80},  {HeapCall::Kind::kFree, d, 40},
  };
  using Direction = MultiSweepHeap::Direction;
  using Side = MultiSweepHeap::Side;
  ASSERT_OK_AND_ASSIGN(const Result forward,
                       RunMultiSweepHeap(calls, /*alignment=*/1,
                                         {{Direction::kForward, Side::kLow}}));
  EXPECT_EQ(forward.heap_size, 240);
  ASSERT_OK_AND_ASSIGN(const Result backward,
                       RunMultiSweepHeap(calls, /*alignment=*/1,
                                         {{Direction::kBackward, Side::kLow}}));
  EXPECT_EQ(backward.heap_size, 240);
  ASSERT_OK_AND_ASSIGN(
      const Result from_peak,
      RunMultiSweepHeap(calls, /*alignment=*/1,
                        {{Direction::kFromPeakBySize, Side::kLow}}));
  EXPECT_THAT(FindProblems(calls, from_peak, /*alignment=*/1), IsEmpty());
  EXPECT_EQ(from_peak.heap_size, 200);
}

// `s1` and `s2` share a buffer whose gap contains the peak. `p1` is live at
// the peak and goes into that gap, and `after` reuses the gap once `p1` ends.
TEST_F(MultiSweepHeapTest, PlacesAtPeakGroupInStraddlingGap) {
  const HloValue* s1 = NewBuffer();
  const HloValue* s2 = NewBuffer();
  const HloValue* p1 = NewBuffer();
  const HloValue* p2 = NewBuffer();
  const HloValue* after = NewBuffer();
  const std::vector<HeapCall> calls = {
      {HeapCall::Kind::kAlloc, s1, 100},
      {HeapCall::Kind::kFree, s1, 100},
      {HeapCall::Kind::kAlloc, p2, 120},
      {HeapCall::Kind::kAlloc, p1, 60},
      {HeapCall::Kind::kFree, p1, 60},
      {HeapCall::Kind::kAlloc, after, 60},
      {HeapCall::Kind::kFree, after, 60},
      {HeapCall::Kind::kFree, p2, 120},
      {HeapCall::Kind::kShareWith, s2, 100, s1},
      {HeapCall::Kind::kFree, s2, 100},
  };
  using Direction = MultiSweepHeap::Direction;
  using Side = MultiSweepHeap::Side;
  ASSERT_OK_AND_ASSIGN(
      const Result result,
      RunMultiSweepHeap(calls, /*alignment=*/1,
                        {{Direction::kFromPeakBySize, Side::kLow}}));
  EXPECT_THAT(FindProblems(calls, result, /*alignment=*/1), IsEmpty());
  EXPECT_EQ(result.heap_size, 220);
  const HeapSimulator::HeapResult<HloValue>& heap = result.heap_results.at(0);
  EXPECT_EQ(heap.chunk_map.at(p1).offset, heap.chunk_map.at(s1).offset);
  EXPECT_EQ(heap.chunk_map.at(after).offset, heap.chunk_map.at(s1).offset);
}

// Three 70-byte buffers have more live bytes (210) than `b` and `p` (200), but
// a smaller sum of squared sizes, so `kFromBigPeakByStart` splits at `b` and
// `p` and needs 210 bytes instead of 240.
TEST_F(MultiSweepHeapTest, FromBigPeakPicksLargestSquaredWeightPeak) {
  const HloValue* s1 = NewBuffer();
  const HloValue* s2 = NewBuffer();
  const HloValue* s3 = NewBuffer();
  const HloValue* a = NewBuffer();
  const HloValue* b = NewBuffer();
  const HloValue* p = NewBuffer();
  const HloValue* c = NewBuffer();
  const HloValue* d = NewBuffer();
  const std::vector<HeapCall> calls = {
      {HeapCall::Kind::kAlloc, s1, 70}, {HeapCall::Kind::kAlloc, s2, 70},
      {HeapCall::Kind::kAlloc, s3, 70}, {HeapCall::Kind::kFree, s1, 70},
      {HeapCall::Kind::kFree, s2, 70},  {HeapCall::Kind::kFree, s3, 70},
      {HeapCall::Kind::kAlloc, a, 40},  {HeapCall::Kind::kAlloc, b, 80},
      {HeapCall::Kind::kFree, a, 40},   {HeapCall::Kind::kAlloc, p, 120},
      {HeapCall::Kind::kFree, b, 80},   {HeapCall::Kind::kAlloc, c, 80},
      {HeapCall::Kind::kFree, p, 120},  {HeapCall::Kind::kAlloc, d, 40},
      {HeapCall::Kind::kFree, c, 80},   {HeapCall::Kind::kFree, d, 40},
  };
  using Direction = MultiSweepHeap::Direction;
  using Side = MultiSweepHeap::Side;
  ASSERT_OK_AND_ASSIGN(
      const Result by_live_bytes,
      RunMultiSweepHeap(calls, /*alignment=*/1,
                        {{Direction::kFromPeakByStart, Side::kLow}}));
  EXPECT_EQ(by_live_bytes.heap_size, 240);
  ASSERT_OK_AND_ASSIGN(
      const Result by_big_peak,
      RunMultiSweepHeap(calls, /*alignment=*/1,
                        {{Direction::kFromBigPeakByStart, Side::kLow}}));
  EXPECT_THAT(FindProblems(calls, by_big_peak, /*alignment=*/1), IsEmpty());
  EXPECT_EQ(by_big_peak.heap_size, 210);
}

// In the free chunk of `mid`, `kNearestRelease` places `x` at the high end
// next to `high` (freed right after `x`), so freeing `low` leaves 60 contiguous
// bytes for `big`.
TEST_F(MultiSweepHeapTest, PlacesNextToNearestRelease) {
  const HloValue* low = NewBuffer();
  const HloValue* mid = NewBuffer();
  const HloValue* high = NewBuffer();
  const HloValue* x = NewBuffer();
  const HloValue* big = NewBuffer();
  const std::vector<HeapCall> calls = {
      {HeapCall::Kind::kAlloc, low, 20},  {HeapCall::Kind::kAlloc, mid, 80},
      {HeapCall::Kind::kAlloc, high, 20}, {HeapCall::Kind::kFree, mid, 80},
      {HeapCall::Kind::kAlloc, x, 40},    {HeapCall::Kind::kFree, low, 20},
      {HeapCall::Kind::kAlloc, big, 60},  {HeapCall::Kind::kFree, big, 60},
      {HeapCall::Kind::kFree, x, 40},     {HeapCall::Kind::kFree, high, 20},
  };
  using Direction = MultiSweepHeap::Direction;
  using Side = MultiSweepHeap::Side;
  ASSERT_OK_AND_ASSIGN(const Result low_side,
                       RunMultiSweepHeap(calls, /*alignment=*/1,
                                         {{Direction::kForward, Side::kLow}}));
  EXPECT_EQ(low_side.heap_size, 180);
  ASSERT_OK_AND_ASSIGN(
      const Result nearest,
      RunMultiSweepHeap(calls, /*alignment=*/1,
                        {{Direction::kForward, Side::kNearestRelease}}));
  EXPECT_THAT(FindProblems(calls, nearest, /*alignment=*/1), IsEmpty());
  EXPECT_EQ(nearest.heap_size, 120);
}

TEST_F(MultiSweepHeapTest, ColocatedBuffersGetTheSameChunk) {
  const HloValue* x = NewBuffer();
  const HloValue* a = NewBuffer();
  const HloValue* b = NewBuffer();
  const std::vector<HeapCall> calls = {
      {HeapCall::Kind::kAlloc, x, 30},        {HeapCall::Kind::kAlloc, a, 10},
      {HeapCall::Kind::kShareWith, b, 20, a}, {HeapCall::Kind::kFree, a, 10},
      {HeapCall::Kind::kFree, x, 30},         {HeapCall::Kind::kFree, b, 20},
  };
  ASSERT_OK_AND_ASSIGN(const Result result,
                       RunMultiSweepHeap(calls, /*alignment=*/1));
  EXPECT_THAT(FindProblems(calls, result, /*alignment=*/1), IsEmpty());
  EXPECT_EQ(result.heap_size, 50);
  const HeapSimulator::HeapResult<HloValue>& heap = result.heap_results.at(0);
  EXPECT_EQ(heap.chunk_map.at(a), heap.chunk_map.at(b));
  EXPECT_EQ(heap.chunk_map.at(a).size, 20);
}

TEST_F(MultiSweepHeapTest, OffsetsAreAligned) {
  const HloValue* a = NewBuffer();
  const HloValue* b = NewBuffer();
  const HloValue* c = NewBuffer();
  const std::vector<HeapCall> calls = {
      {HeapCall::Kind::kAlloc, a, 10}, {HeapCall::Kind::kAlloc, b, 20},
      {HeapCall::Kind::kAlloc, c, 30}, {HeapCall::Kind::kFree, b, 20},
      {HeapCall::Kind::kFree, a, 10},  {HeapCall::Kind::kFree, c, 30},
  };
  ASSERT_OK_AND_ASSIGN(const Result result,
                       RunMultiSweepHeap(calls, /*alignment=*/16));
  EXPECT_THAT(FindProblems(calls, result, /*alignment=*/16), IsEmpty());
  // `c` at 0, `a` at 32 and `b` at 48.
  EXPECT_EQ(result.heap_size, 68);
}

// With a 250-byte per-heap limit, the 300-byte buffer starts its own heap; a
// backward sweep reuses that heap for most of the chain and spills only `c`
// into a second heap (420 bytes vs. FAST_MERGE's 700).
TEST_F(MultiSweepHeapTest, SplitsAcrossHeapsWithSizeLimit) {
  const HloValue* a = NewBuffer();
  const HloValue* b = NewBuffer();
  const HloValue* c = NewBuffer();
  const HloValue* d = NewBuffer();
  const HloValue* huge = NewBuffer();
  const std::vector<HeapCall> calls = {
      {HeapCall::Kind::kAlloc, a, 40},     {HeapCall::Kind::kAlloc, b, 80},
      {HeapCall::Kind::kFree, a, 40},      {HeapCall::Kind::kAlloc, c, 120},
      {HeapCall::Kind::kFree, b, 80},      {HeapCall::Kind::kAlloc, d, 160},
      {HeapCall::Kind::kFree, c, 120},     {HeapCall::Kind::kFree, d, 160},
      {HeapCall::Kind::kAlloc, huge, 300}, {HeapCall::Kind::kFree, huge, 300},
  };
  constexpr uint64_t kLimit = 250;
  ASSERT_OK_AND_ASSIGN(
      const Result fast_merge,
      RunFastMerge(calls, /*alignment=*/1, /*size_limit_per_heap=*/kLimit));
  EXPECT_EQ(fast_merge.heap_size, 700);
  ASSERT_OK_AND_ASSIGN(
      const Result result,
      RunMultiSweepHeap(calls, /*alignment=*/1, MultiSweepHeap::AllPackings(),
                        /*size_limit_per_heap=*/kLimit));
  EXPECT_THAT(FindProblems(calls, result, /*alignment=*/1, kLimit), IsEmpty());
  EXPECT_EQ(result.heap_size, 420);
  EXPECT_EQ(result.heap_results.size(), 2);
}

TEST_F(MultiSweepHeapTest, TotalHeapSizeOverflowReturnsResourceExhausted) {
  const HloValue* a = NewBuffer();
  const HloValue* b = NewBuffer();
  constexpr int64_t kHalfPlusOne = std::numeric_limits<int64_t>::max() / 2 + 1;
  const std::vector<HeapCall> calls = {
      {HeapCall::Kind::kAlloc, a, kHalfPlusOne},
      {HeapCall::Kind::kAlloc, b, kHalfPlusOne},
      {HeapCall::Kind::kFree, a, kHalfPlusOne},
      {HeapCall::Kind::kFree, b, kHalfPlusOne},
  };
  EXPECT_THAT(
      RunMultiSweepHeap(calls, /*alignment=*/1, MultiSweepHeap::AllPackings(),
                        /*size_limit_per_heap=*/1),
      StatusIs(absl::StatusCode::kResourceExhausted,
               HasSubstr("overflows int64_t")));
}

TEST_F(MultiSweepHeapTest, EveryPackingGivesValidResults) {
  std::mt19937 rng(/*seed=*/2);
  for (int i = 0; i < 20; ++i) {
    const std::vector<HeapCall> calls = RandomCalls(rng, /*num_buffers=*/40);
    for (const MultiSweepHeap::Packing& packing :
         MultiSweepHeap::AllPackings()) {
      for (int64_t alignment : {1, 8}) {
        for (uint64_t limit : {std::numeric_limits<uint64_t>::max(),
                               uint64_t{200}, uint64_t{60}, uint64_t{0}}) {
          SCOPED_TRACE(absl::StrCat("sequence ", i, ", direction ",
                                    static_cast<int>(packing.direction),
                                    ", side ", static_cast<int>(packing.side),
                                    ", alignment ", alignment, ", limit ",
                                    limit));
          ASSERT_OK_AND_ASSIGN(
              const Result result,
              RunMultiSweepHeap(calls, alignment, {packing}, limit));
          EXPECT_THAT(FindProblems(calls, result, alignment, limit), IsEmpty());
        }
      }
    }
  }
}

// Without colocated buffers there are no gaps, so the forward sweep at the low
// end is FAST_MERGE.
TEST_F(MultiSweepHeapTest, ForwardSweepAtLowEndIsFastMergeWithoutSharing) {
  std::mt19937 rng(/*seed=*/3);
  for (int i = 0; i < 50; ++i) {
    const std::vector<HeapCall> calls =
        RandomCalls(rng, /*num_buffers=*/40, /*share=*/false,
                    /*free_all=*/true);
    for (int64_t alignment : {1, 8}) {
      for (uint64_t limit : {std::numeric_limits<uint64_t>::max(),
                             uint64_t{200}, uint64_t{60}}) {
        SCOPED_TRACE(absl::StrCat("sequence ", i, ", alignment ", alignment,
                                  ", limit ", limit));
        ASSERT_OK_AND_ASSIGN(
            const Result result,
            RunMultiSweepHeap(calls, alignment,
                              {{MultiSweepHeap::Direction::kForward,
                                MultiSweepHeap::Side::kLow}},
                              limit));
        ASSERT_OK_AND_ASSIGN(const Result fast_merge,
                             RunFastMerge(calls, alignment, limit));
        ASSERT_EQ(result.heap_results.size(), fast_merge.heap_results.size());
        for (size_t h = 0; h < result.heap_results.size(); ++h) {
          EXPECT_EQ(result.heap_results[h].chunk_map,
                    fast_merge.heap_results[h].chunk_map);
          EXPECT_EQ(result.heap_results[h].heap_size,
                    fast_merge.heap_results[h].heap_size);
        }
        EXPECT_EQ(result.heap_size, fast_merge.heap_size);
      }
    }
  }
}

TEST_F(MultiSweepHeapTest, RandomCallsGiveValidAndDeterministicResults) {
  std::mt19937 rng(/*seed=*/1);
  for (int i = 0; i < 100; ++i) {
    const std::vector<HeapCall> calls = RandomCalls(rng, /*num_buffers=*/40);
    for (int64_t alignment : {1, 8}) {
      for (uint64_t limit :
           {std::numeric_limits<uint64_t>::max(), uint64_t{150}}) {
        SCOPED_TRACE(absl::StrCat("sequence ", i, ", alignment ", alignment,
                                  ", limit ", limit));
        ASSERT_OK_AND_ASSIGN(
            const Result result,
            RunMultiSweepHeap(calls, alignment, MultiSweepHeap::AllPackings(),
                              limit));
        EXPECT_THAT(FindProblems(calls, result, alignment, limit), IsEmpty());
        ASSERT_OK_AND_ASSIGN(
            const Result again,
            RunMultiSweepHeap(calls, alignment, MultiSweepHeap::AllPackings(),
                              limit));
        ASSERT_EQ(again.heap_results.size(), result.heap_results.size());
        for (size_t h = 0; h < result.heap_results.size(); ++h) {
          EXPECT_EQ(again.heap_results[h].chunk_map,
                    result.heap_results[h].chunk_map);
        }
      }
    }
  }
}

}  // namespace
}  // namespace xla
