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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <optional>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_value.h"
#include "xla/util.h"

namespace xla {
namespace {

// The end of the topmost free chunk in FreeSpace, as in FreeChunksManager.
constexpr int64_t kHeapEnd = std::numeric_limits<int64_t>::max() - 3;
// Stands for "nothing fits" when comparing sizes, and for "no neighbor" when
// comparing times.
constexpr int64_t kInfinity = std::numeric_limits<int64_t>::max();
constexpr int64_t kMinInt64 = std::numeric_limits<int64_t>::min();
// Marks memory that is not in a hole.
constexpr int kNoHole = -1;
// Marks a region that is not in the hole of another region.
constexpr int kNoRegion = -1;

// A time range [start, end], both ends included.
struct Segment {
  int64_t start;
  int64_t end;
};

// A colocation group, placed as one block.
struct Group {
  int64_t start;     // The first start of a member.
  int64_t end;       // The last end of a member.
  int64_t size;      // The size of the largest member.
  int64_t own_size;  // The size of the first member.
  int64_t id;        // The HloValue id of the first member.
  // The times when a member is live, sorted and merged where they overlap.
  // The times between them are the group's gaps.
  std::vector<Segment> segments;
  // `segments` in the sweep time of a backward sweep (see Sweeper): negated
  // and in reverse order.
  std::vector<Segment> mirrored_segments;
};

// An ordered set of unique keys, each with an int64_t value. FindFirst returns
// the smallest key >= `min_key` with a value >= `min_value` in O(log n)
// expected time, because every node keeps the largest value in its subtree.
// The node priorities come from a fixed pseudo-random sequence, so the tree is
// the same in every run.
template <typename Key>
class MaxTreap {
 public:
  void Insert(const Key& key, int64_t value) {
    const int node = NewNode(key, value);
    const auto [left, right] = Split(root_, key);
    root_ = Merge(Merge(left, node), right);
  }

  // Does nothing if `key` is not in the set.
  void Erase(const Key& key) { root_ = Erase(root_, key); }

  std::optional<Key> FindFirst(const Key& min_key, int64_t min_value) const {
    return FindFirst(root_, min_key, min_value);
  }

 private:
  struct Node {
    Key key;
    int64_t value;
    int64_t subtree_max;
    uint64_t priority;
    int left;
    int right;
  };

  static constexpr int kNull = -1;

  int NewNode(const Key& key, int64_t value) {
    // xorshift64.
    random_state_ ^= random_state_ << 13;
    random_state_ ^= random_state_ >> 7;
    random_state_ ^= random_state_ << 17;
    const Node node{key, value, value, random_state_, kNull, kNull};
    if (!free_nodes_.empty()) {
      const int n = free_nodes_.back();
      free_nodes_.pop_back();
      nodes_[n] = node;
      return n;
    }
    nodes_.push_back(node);
    return static_cast<int>(nodes_.size()) - 1;
  }

  void UpdateSubtreeMax(int n) {
    Node& node = nodes_[n];
    node.subtree_max = node.value;
    if (node.left != kNull) {
      node.subtree_max =
          std::max(node.subtree_max, nodes_[node.left].subtree_max);
    }
    if (node.right != kNull) {
      node.subtree_max =
          std::max(node.subtree_max, nodes_[node.right].subtree_max);
    }
  }

  // Splits the subtree of `n` into the keys < `key` and the keys >= `key`.
  std::pair<int, int> Split(int n, const Key& key) {
    if (n == kNull) {
      return {kNull, kNull};
    }
    if (nodes_[n].key < key) {
      const auto [left, right] = Split(nodes_[n].right, key);
      nodes_[n].right = left;
      UpdateSubtreeMax(n);
      return {n, right};
    }
    const auto [left, right] = Split(nodes_[n].left, key);
    nodes_[n].left = right;
    UpdateSubtreeMax(n);
    return {left, n};
  }

  // Merges two subtrees. The keys of `left` are smaller than those of `right`.
  int Merge(int left, int right) {
    if (left == kNull) {
      return right;
    }
    if (right == kNull) {
      return left;
    }
    if (nodes_[left].priority > nodes_[right].priority) {
      const int merged = Merge(nodes_[left].right, right);
      nodes_[left].right = merged;
      UpdateSubtreeMax(left);
      return left;
    }
    const int merged = Merge(left, nodes_[right].left);
    nodes_[right].left = merged;
    UpdateSubtreeMax(right);
    return right;
  }

  int Erase(int n, const Key& key) {
    if (n == kNull) {
      return kNull;
    }
    if (key < nodes_[n].key) {
      const int left = Erase(nodes_[n].left, key);
      nodes_[n].left = left;
    } else if (nodes_[n].key < key) {
      const int right = Erase(nodes_[n].right, key);
      nodes_[n].right = right;
    } else {
      const int merged = Merge(nodes_[n].left, nodes_[n].right);
      free_nodes_.push_back(n);
      return merged;
    }
    UpdateSubtreeMax(n);
    return n;
  }

  std::optional<Key> FindFirst(int n, const Key& min_key,
                               int64_t min_value) const {
    if (n == kNull || nodes_[n].subtree_max < min_value) {
      return std::nullopt;
    }
    const Node& node = nodes_[n];
    if (node.key < min_key) {
      return FindFirst(node.right, min_key, min_value);
    }
    if (std::optional<Key> found = FindFirst(node.left, min_key, min_value)) {
      return found;
    }
    if (node.value >= min_value) {
      return node.key;
    }
    return FindFirst(node.right, min_key, min_value);
  }

  std::vector<Node> nodes_;
  std::vector<int> free_nodes_;
  int root_ = kNull;
  uint64_t random_state_ = 0x9E3779B97F4A7C15ULL;
};

// The free memory that is not in holes, as disjoint chunks [offset, end). As
// in FreeChunksManager, a chunk can hold `size` bytes if its aligned size,
// end - RoundUpTo(offset, alignment), is at least `size`, and a freed chunk
// merges with the free chunks next to it. The topmost chunk ends at kHeapEnd.
// Unlike FreeChunksManager, it can find the chunks right below and above a
// given offset, and it records every chunk that it adds or removes, which the
// spans need.
class FreeSpace {
 public:
  struct Chunk {
    int64_t offset;
    int64_t end;
  };

  explicit FreeSpace(int64_t alignment) : alignment_(alignment) {
    Add(0, kHeapEnd);
  }

  int64_t AlignedSize(const Chunk& chunk) const {
    return chunk.end - RoundUpTo(chunk.offset, alignment_);
  }

  // Takes [offset, end) out of the free chunk that contains it.
  void Allocate(int64_t offset, int64_t end) {
    auto it = chunks_.upper_bound(offset);
    CHECK(it != chunks_.begin());
    --it;
    const Chunk chunk = ToChunk(it);
    CHECK(chunk.offset <= offset && end <= chunk.end)
        << "[" << offset << ", " << end << ") is not in the free chunk ["
        << chunk.offset << ", " << chunk.end << ")";
    Remove(it);
    if (end < chunk.end) {
      Add(end, chunk.end);
    }
    if (chunk.offset < offset) {
      Add(chunk.offset, offset);
    }
  }

  // Gives [offset, end) back.
  void Free(int64_t offset, int64_t end) {
    int64_t merged_offset = offset;
    int64_t merged_end = end;
    if (auto above = chunks_.find(end); above != chunks_.end()) {
      merged_end = above->second;
      Remove(above);
    }
    auto it = chunks_.lower_bound(offset);
    CHECK(it == chunks_.end() || end <= it->first);
    if (it != chunks_.begin()) {
      auto below = std::prev(it);
      CHECK_LE(below->second, offset);
      if (below->second == offset) {
        merged_offset = below->first;
        Remove(below);
      }
    }
    Add(merged_offset, merged_end);
  }

  // The chunk with the smallest aligned size >= `size` (ties go to the lowest
  // offset), or nullopt if none fits.
  std::optional<Chunk> BestFit(int64_t size) const {
    auto it = by_aligned_size_.lower_bound({size, kMinInt64});
    if (it == by_aligned_size_.end()) {
      return std::nullopt;
    }
    auto chunk = chunks_.find(it->second);
    CHECK(chunk != chunks_.end());
    return ToChunk(chunk);
  }

  std::optional<Chunk> ChunkAt(int64_t offset) const {
    auto it = chunks_.find(offset);
    if (it == chunks_.end()) {
      return std::nullopt;
    }
    return ToChunk(it);
  }

  std::optional<Chunk> ChunkEndingAt(int64_t end) const {
    auto it = chunks_.lower_bound(end);
    if (it == chunks_.begin()) {
      return std::nullopt;
    }
    --it;
    if (it->second != end) {
      return std::nullopt;
    }
    return ToChunk(it);
  }

  // The chunks that were added or removed since the last ClearChanges() call.
  absl::Span<const Chunk> changes() const { return changes_; }
  void ClearChanges() { changes_.clear(); }

 private:
  using ChunkMap = absl::btree_map<int64_t, int64_t>;  // offset -> end

  static Chunk ToChunk(ChunkMap::const_iterator it) {
    return Chunk{it->first, it->second};
  }

  void Add(int64_t offset, int64_t end) {
    chunks_[offset] = end;
    by_aligned_size_.insert({AlignedSize({offset, end}), offset});
    changes_.push_back({offset, end});
  }

  void Remove(ChunkMap::iterator it) {
    const Chunk chunk = ToChunk(it);
    changes_.push_back(chunk);
    by_aligned_size_.erase({AlignedSize(chunk), chunk.offset});
    chunks_.erase(it);
  }

  int64_t alignment_;
  ChunkMap chunks_;
  // (aligned size, offset) of every chunk.
  absl::btree_set<std::pair<int64_t, int64_t>> by_aligned_size_;
  std::vector<Chunk> changes_;
};

// The free pieces of holes. A hole is the memory of a placed block during a
// gap of its group. It can hold buffers that are freed before the hole's
// deadline, the time when the gap ends, because then the group needs all of
// its memory again. Pieces of a hole merge only with pieces of the same hole.
class HoleSpace {
 public:
  struct Piece {
    int64_t offset;
    int64_t end;
    int hole;
  };

  explicit HoleSpace(int64_t alignment) : alignment_(alignment) {}

  // Returns the id of a new hole that closes at `deadline`.
  int NewHole(int64_t deadline) {
    deadlines_.push_back(deadline);
    return static_cast<int>(deadlines_.size()) - 1;
  }

  int64_t deadline(int hole) const { return deadlines_[hole]; }

  int64_t AlignedSize(const Piece& piece) const {
    return piece.end - RoundUpTo(piece.offset, alignment_);
  }

  // Gives [offset, end) to `hole`.
  void Free(int hole, int64_t offset, int64_t end) {
    int64_t merged_offset = offset;
    int64_t merged_end = end;
    if (auto above = pieces_.find(end);
        above != pieces_.end() && above->second.hole == hole) {
      merged_end = above->second.end;
      Remove(above);
    }
    auto it = pieces_.lower_bound(offset);
    CHECK(it == pieces_.end() || end <= it->first);
    if (it != pieces_.begin()) {
      auto below = std::prev(it);
      CHECK_LE(below->second.end, offset);
      if (below->second.end == offset && below->second.hole == hole) {
        merged_offset = below->first;
        Remove(below);
      }
    }
    Add(merged_offset, merged_end, hole);
  }

  // Takes [offset, end) out of the free piece of `hole` that contains it.
  void Allocate(int hole, int64_t offset, int64_t end) {
    auto it = pieces_.upper_bound(offset);
    CHECK(it != pieces_.begin());
    --it;
    const Piece piece = ToPiece(it);
    CHECK(piece.hole == hole && piece.offset <= offset && end <= piece.end)
        << "[" << offset << ", " << end << ") is not in a free piece of hole "
        << hole;
    Remove(it);
    if (end < piece.end) {
      Add(end, piece.end, hole);
    }
    if (piece.offset < offset) {
      Add(piece.offset, offset, hole);
    }
  }

  // Closes `hole` at its deadline. All of its memory, [offset, end), must be
  // free.
  void Close(int hole, int64_t offset, int64_t end) {
    auto it = pieces_.find(offset);
    CHECK(it != pieces_.end() && it->second.end == end &&
          it->second.hole == hole)
        << "hole " << hole << " [" << offset << ", " << end
        << ") is not free at its deadline";
    Remove(it);
  }

  // The piece with the smallest aligned size >= `size` whose hole closes
  // after `release`; ties go to the lowest offset.
  std::optional<Piece> BestFit(int64_t size, int64_t release) const {
    const std::optional<std::pair<int64_t, int64_t>> key =
        by_aligned_size_.FindFirst({size, kMinInt64}, release + 1);
    if (!key.has_value()) {
      return std::nullopt;
    }
    const std::optional<Piece> piece = PieceAt(key->second);
    CHECK(piece.has_value());
    return piece;
  }

  std::optional<Piece> PieceAt(int64_t offset) const {
    auto it = pieces_.find(offset);
    if (it == pieces_.end()) {
      return std::nullopt;
    }
    return ToPiece(it);
  }

  std::optional<Piece> PieceEndingAt(int64_t end) const {
    auto it = pieces_.lower_bound(end);
    if (it == pieces_.begin()) {
      return std::nullopt;
    }
    --it;
    if (it->second.end != end) {
      return std::nullopt;
    }
    return ToPiece(it);
  }

  // The offsets of the pieces that were added or removed since the last
  // ClearChanges() call.
  absl::Span<const int64_t> changes() const { return changes_; }
  void ClearChanges() { changes_.clear(); }

 private:
  struct PieceEnd {
    int64_t end;
    int hole;
  };
  using PieceMap = absl::btree_map<int64_t, PieceEnd>;  // offset -> end, hole

  static Piece ToPiece(PieceMap::const_iterator it) {
    return Piece{it->first, it->second.end, it->second.hole};
  }

  void Add(int64_t offset, int64_t end, int hole) {
    pieces_[offset] = PieceEnd{end, hole};
    by_aligned_size_.Insert({AlignedSize({offset, end, hole}), offset},
                            deadlines_[hole]);
    changes_.push_back(offset);
  }

  void Remove(PieceMap::iterator it) {
    const Piece piece = ToPiece(it);
    changes_.push_back(piece.offset);
    by_aligned_size_.Erase({AlignedSize(piece), piece.offset});
    pieces_.erase(it);
  }

  int64_t alignment_;
  std::vector<int64_t> deadlines_;
  PieceMap pieces_;
  // (aligned size, offset) of every piece, with the deadline of its hole.
  MaxTreap<std::pair<int64_t, int64_t>> by_aligned_size_;
  std::vector<int64_t> changes_;
};

using Direction = ::xla::MultiSweepHeap::Direction;
using Side = ::xla::MultiSweepHeap::Side;

// One sweep through time that places blocks when they start.
//
// It works in "sweep time", which always goes forward: heap time for a forward
// sweep, and minus heap time for a backward sweep. A block arrives at the
// start of its first segment and is released at the end of its last segment.
// Its memory (a "region") becomes free when the sweep reaches the release
// time, and is a hole during the gaps between its segments.
class Sweeper {
 public:
  // `now` is the sweep time when the sweep starts.
  Sweeper(int64_t alignment, int64_t max_heap_size, Side side, int64_t now)
      : alignment_(alignment),
        max_heap_size_(max_heap_size),
        side_(side),
        now_(now),
        space_(alignment),
        holes_(alignment) {}

  // Adds the region [offset, end), which is placed before the sweep starts,
  // for a block with the given segments (in sweep time). If `container` is not
  // kNoRegion, the region is in the hole that region `container` has when the
  // sweep starts. Returns the id of the region.
  int AddInitialRegion(int64_t offset, int64_t end,
                       absl::Span<const Segment> segments, int container) {
    Part part{offset, end, kNoHole};
    if (container != kNoRegion) {
      part.hole = regions_[container].hole;
      CHECK_NE(part.hole, kNoHole)
          << "region " << container << " has no hole when the sweep starts";
    }
    Take(part, segments.back().end);
    return AddRegion({part}, segments, now_);
  }

  // Places a block of `size` bytes with the given segments (in sweep time) and
  // returns its offset, or nullopt if it doesn't fit in `max_heap_size_`.
  // Blocks must come in the order of their arrival.
  std::optional<int64_t> TryPlace(absl::Span<const Segment> segments,
                                  int64_t size) {
    const int64_t arrival = segments.front().start;
    const int64_t release = segments.back().end;
    ReleaseUntil(arrival);
    // Best fit over hole pieces, spans and free chunks. A piece wins ties over
    // a span, and both win ties over a free chunk. The topmost chunk counts as
    // infinitely big, since it is where the heap grows.
    const std::optional<FreeSpace::Chunk> chunk = space_.BestFit(size);
    const std::optional<HoleSpace::Piece> piece = holes_.BestFit(size, release);
    const std::optional<HoleSpan> span = BestSpan(size, release);
    if (piece.has_value() || span.has_value()) {
      const int64_t chunk_size = (!chunk.has_value() || chunk->end == kHeapEnd)
                                     ? kInfinity
                                     : space_.AlignedSize(*chunk);
      const int64_t piece_size =
          piece.has_value() ? holes_.AlignedSize(*piece) : kInfinity;
      const int64_t span_size = span.has_value() ? span->size : kInfinity;
      if (piece.has_value() && piece_size <= span_size &&
          piece_size <= chunk_size) {
        const int64_t offset = RoundUpTo(piece->offset, alignment_);
        const Part part{piece->offset, offset + size, piece->hole};
        Take(part, release);
        AddRegion({part}, segments, arrival);
        return offset;
      }
      if (span.has_value() && span_size <= chunk_size) {
        return PlaceInSpan(*span, size, segments);
      }
    }
    if (!chunk.has_value()) {
      return std::nullopt;
    }
    // Only the topmost chunk fits, so the heap may grow. If a hole piece ends
    // where the topmost chunk starts, the block starts in that piece (or in the
    // free chunk below it), so that the heap grows less.
    if (chunk->end == kHeapEnd) {
      const std::optional<HoleSpace::Piece> below_top =
          holes_.PieceEndingAt(chunk->offset);
      if (below_top.has_value() && holes_.deadline(below_top->hole) > release) {
        const std::optional<FreeSpace::Chunk> below =
            space_.ChunkEndingAt(below_top->offset);
        const int64_t low =
            below.has_value() ? below->offset : below_top->offset;
        const int64_t low_aligned = RoundUpTo(low, alignment_);
        if (low_aligned > max_heap_size_ ||
            size > max_heap_size_ - low_aligned) {
          return std::nullopt;
        }
        const HoleSpan span_to_top{*below_top, below, *chunk, /*size=*/0};
        return PlaceInSpan(span_to_top, size, segments);
      }
    }
    if (side_ == Side::kNearestRelease && chunk->end != kHeapEnd &&
        ReleaseDistance(*chunk, /*above=*/true, release) <
            ReleaseDistance(*chunk, /*above=*/false, release)) {
      const int64_t offset = RoundDownTo(chunk->end - size, alignment_);
      const Part part{offset, chunk->end, kNoHole};
      Take(part, release);
      AddRegion({part}, segments, arrival);
      return offset;
    }
    const int64_t offset = RoundUpTo(chunk->offset, alignment_);
    if (!regions_.empty() &&
        (offset > max_heap_size_ || size > max_heap_size_ - offset)) {
      return std::nullopt;
    }
    const Part part{chunk->offset, offset + size, kNoHole};
    Take(part, release);
    AddRegion({part}, segments, arrival);
    return offset;
  }

 private:
  // A contiguous part of a region, taken from the free space (kNoHole) or
  // from a hole.
  struct Part {
    int64_t offset;
    int64_t end;
    int hole;
  };

  // The memory of a placed block.
  struct Region {
    int64_t offset() const { return parts.front().offset; }
    int64_t end() const { return parts.back().end; }

    std::vector<Part> parts;  // Contiguous, sorted by offset.
    int hole;                 // The region's own hole while it is open.
  };

  // A hole piece with the free chunks right below and above it.
  struct HoleSpan {
    HoleSpace::Piece piece;
    std::optional<FreeSpace::Chunk> below;
    std::optional<FreeSpace::Chunk> above;
    int64_t size;  // The aligned size of the whole span.
  };

  enum class EventKind { kRelease, kOpenHole, kCloseHole };

  struct Event {
    int64_t time;
    EventKind kind;
    int region;
    int64_t deadline;  // For kOpenHole: when the hole closes.
  };

  // Makes `events_` return the earliest event first. At the same time,
  // releases come first, then hole openings, then hole closings.
  struct EventIsLater {
    bool operator()(const Event& a, const Event& b) const {
      return std::tie(a.time, a.kind, a.region) >
             std::tie(b.time, b.kind, b.region);
    }
  };

  // A part of a region taken from the free space.
  struct TakenPart {
    int64_t end;
    int64_t release;
  };

  void Take(const Part& part, int64_t release) {
    if (part.hole == kNoHole) {
      space_.Allocate(part.offset, part.end);
      taken_[part.offset] = TakenPart{part.end, release};
    } else {
      holes_.Allocate(part.hole, part.offset, part.end);
    }
  }

  void Give(const Part& part) {
    if (part.hole == kNoHole) {
      space_.Free(part.offset, part.end);
      taken_.erase(part.offset);
    } else {
      holes_.Free(part.hole, part.offset, part.end);
    }
  }

  // Places the block in the span: at the low end of the piece if it fits there
  // with the chunk above, else at the high end of the piece if it fits there
  // with the chunk below, else at the bottom of the span. Next to the topmost
  // chunk, it uses the chunk below first, so that the heap grows less.
  int64_t PlaceInSpan(const HoleSpan& span, int64_t size,
                      absl::Span<const Segment> segments) {
    const HoleSpace::Piece& piece = span.piece;
    const int64_t low =
        span.below.has_value() ? span.below->offset : piece.offset;
    const int64_t high = span.above.has_value() ? span.above->end : piece.end;
    const bool at_top = span.above.has_value() && span.above->end == kHeapEnd;
    const int64_t up = RoundUpTo(piece.offset, alignment_);
    const int64_t down = RoundDownTo(piece.end - size, alignment_);
    const bool fits_up = span.above.has_value() && up + size <= high;
    const bool fits_down = span.below.has_value() && down >= low;
    int64_t offset;
    int64_t region_offset;
    int64_t region_end;
    if (fits_down && (at_top || !fits_up)) {
      offset = down;
      region_offset = down;
      region_end = piece.end;
    } else if (fits_up && !at_top) {
      offset = up;
      region_offset = piece.offset;
      region_end = up + size;
    } else {
      offset = RoundUpTo(low, alignment_);
      region_offset = low;
      region_end = offset + size;
      CHECK_LE(region_end, high);
    }
    std::vector<Part> parts;
    auto add_part = [&](int64_t part_offset, int64_t part_end, int hole) {
      part_offset = std::max(part_offset, region_offset);
      part_end = std::min(part_end, region_end);
      if (part_offset < part_end) {
        parts.push_back({part_offset, part_end, hole});
      }
    };
    add_part(low, piece.offset, kNoHole);
    add_part(piece.offset, piece.end, piece.hole);
    add_part(piece.end, high, kNoHole);
    const int64_t release = segments.back().end;
    for (const Part& part : parts) {
      Take(part, release);
    }
    AddRegion(std::move(parts), segments, segments.front().start);
    return offset;
  }

  // Adds a region with the given parts for a block with the given segments,
  // and schedules its hole openings and closings after `now` and its release.
  // Returns the id of the region.
  int AddRegion(std::vector<Part> parts, absl::Span<const Segment> segments,
                int64_t now) {
    const int region = static_cast<int>(regions_.size());
    regions_.push_back(Region{std::move(parts), kNoHole});
    // Skip the segments that end before `now`.
    size_t k = 0;
    while (k < segments.size() && segments[k].end < now) {
      ++k;
    }
    CHECK_LT(k, segments.size())
        << "the region is released before the sweep starts";
    if (segments[k].start > now) {
      // `now` is in a gap.
      OpenHole(region, segments[k].start);
      events_.push({segments[k].start, EventKind::kCloseHole, region, 0});
    }
    for (; k + 1 < segments.size(); ++k) {
      events_.push({segments[k].end, EventKind::kOpenHole, region,
                    segments[k + 1].start});
      events_.push({segments[k + 1].start, EventKind::kCloseHole, region, 0});
    }
    events_.push({segments.back().end, EventKind::kRelease, region, 0});
    return region;
  }

  void OpenHole(int region, int64_t deadline) {
    Region& r = regions_[region];
    const int64_t hole_end = std::min(r.end(), max_heap_size_);
    if (r.offset() >= hole_end) {
      return;
    }
    r.hole = holes_.NewHole(deadline);
    holes_.Free(r.hole, r.offset(), hole_end);
  }

  // Handles the events up to and including `time`.
  void ReleaseUntil(int64_t time) {
    while (!events_.empty() && events_.top().time <= time) {
      const Event event = events_.top();
      events_.pop();
      Region& region = regions_[event.region];
      switch (event.kind) {
        case EventKind::kRelease:
          for (const Part& part : region.parts) {
            Give(part);
          }
          break;
        case EventKind::kOpenHole:
          OpenHole(event.region, event.deadline);
          break;
        case EventKind::kCloseHole:
          if (region.hole != kNoHole) {
            holes_.Close(region.hole, region.offset(),
                         std::min(region.end(), max_heap_size_));
            region.hole = kNoHole;
          }
          break;
      }
    }
  }

  // Brings `spans_` up to date with the free chunks and hole pieces that
  // changed since the last call.
  void UpdateSpans() {
    for (const FreeSpace::Chunk& chunk : space_.changes()) {
      if (std::optional<HoleSpace::Piece> piece = holes_.PieceAt(chunk.end)) {
        UpdateSpan(piece->offset);
      }
      if (std::optional<HoleSpace::Piece> piece =
              holes_.PieceEndingAt(chunk.offset)) {
        UpdateSpan(piece->offset);
      }
    }
    for (int64_t piece_offset : holes_.changes()) {
      UpdateSpan(piece_offset);
    }
    space_.ClearChanges();
    holes_.ClearChanges();
  }

  // Updates the span of the hole piece at `piece_offset`. A piece has a span
  // if there is a free chunk right below or above it, other than the topmost
  // chunk.
  void UpdateSpan(int64_t piece_offset) {
    if (auto it = span_keys_.find(piece_offset); it != span_keys_.end()) {
      spans_.Erase(it->second);
      span_keys_.erase(it);
    }
    const std::optional<HoleSpace::Piece> piece = holes_.PieceAt(piece_offset);
    if (!piece.has_value()) {
      return;
    }
    const std::optional<FreeSpace::Chunk> below =
        space_.ChunkEndingAt(piece->offset);
    std::optional<FreeSpace::Chunk> above = space_.ChunkAt(piece->end);
    if (above.has_value() && above->end == kHeapEnd) {
      above = std::nullopt;
    }
    if (!below.has_value() && !above.has_value()) {
      return;
    }
    const int64_t low = below.has_value() ? below->offset : piece->offset;
    const int64_t high = above.has_value() ? above->end : piece->end;
    const std::pair<int64_t, int64_t> key = {high - RoundUpTo(low, alignment_),
                                             piece->offset};
    spans_.Insert(key, holes_.deadline(piece->hole));
    span_keys_[piece_offset] = key;
  }

  // The span with the smallest aligned size >= `size` whose hole closes after
  // `release`; ties go to the lowest piece offset.
  std::optional<HoleSpan> BestSpan(int64_t size, int64_t release) {
    UpdateSpans();
    const std::optional<std::pair<int64_t, int64_t>> key =
        spans_.FindFirst({size, kMinInt64}, release + 1);
    if (!key.has_value()) {
      return std::nullopt;
    }
    const std::optional<HoleSpace::Piece> piece = holes_.PieceAt(key->second);
    CHECK(piece.has_value());
    std::optional<FreeSpace::Chunk> above = space_.ChunkAt(piece->end);
    if (above.has_value() && above->end == kHeapEnd) {
      above = std::nullopt;
    }
    return HoleSpan{*piece, space_.ChunkEndingAt(piece->offset), above,
                    key->first};
  }

  // How far apart in time `release` and the release of the region right above
  // (or below) `chunk` are. kInfinity if no region from the free space is
  // there.
  int64_t ReleaseDistance(const FreeSpace::Chunk& chunk, bool above,
                          int64_t release) const {
    if (above) {
      auto it = taken_.find(chunk.end);
      if (it == taken_.end()) {
        return kInfinity;
      }
      return std::abs(it->second.release - release);
    }
    auto it = taken_.lower_bound(chunk.offset);
    if (it == taken_.begin()) {
      return kInfinity;
    }
    --it;
    if (it->second.end != chunk.offset) {
      return kInfinity;
    }
    return std::abs(it->second.release - release);
  }

  int64_t alignment_;
  int64_t max_heap_size_;
  Side side_;
  int64_t now_;
  FreeSpace space_;
  HoleSpace holes_;
  std::vector<Region> regions_;
  // The parts of regions that are taken from the free space, by offset.
  absl::btree_map<int64_t, TakenPart> taken_;
  std::priority_queue<Event, std::vector<Event>, EventIsLater> events_;
  // (aligned size of the span, piece offset) of every span, with the deadline
  // of the piece's hole.
  MaxTreap<std::pair<int64_t, int64_t>> spans_;
  // The key in `spans_` of every piece that has a span, by piece offset.
  absl::flat_hash_map<int64_t, std::pair<int64_t, int64_t>> span_keys_;
};

// A region that is placed before a sweep starts.
struct InitialRegion {
  int64_t offset;
  int64_t end;
  int64_t group;
  // The index (in the list of initial regions) of the region whose hole holds
  // this region when the sweep starts, or kNoRegion.
  int container;
};

// The groups placed in one heap and their offsets.
struct PlacedHeap {
  void Add(int64_t group, int64_t offset, int64_t group_size) {
    size = std::max(size, offset + group_size);
    placements.push_back({group, offset});
  }

  int64_t size = 0;
  // (index in `groups`, offset in this heap).
  std::vector<std::pair<int64_t, int64_t>> placements;
};

// Sweeps forward or backward from heap time `start_time`, with the `initial`
// regions already placed. Places the groups in `order` that fit in
// `max_heap_size` into `heap`, and appends the rest to `unplaced`.
void Sweep(absl::Span<const Group> groups, absl::Span<const int64_t> order,
           bool forward, absl::Span<const InitialRegion> initial,
           int64_t start_time, int64_t alignment, int64_t max_heap_size,
           Side side, PlacedHeap& heap, std::vector<int64_t>& unplaced) {
  Sweeper sweeper(alignment, max_heap_size, side,
                  forward ? start_time : -start_time);
  auto segments_of = [&](int64_t g) -> absl::Span<const Segment> {
    return forward ? groups[g].segments : groups[g].mirrored_segments;
  };
  std::vector<int> region_of(initial.size(), kNoRegion);
  for (size_t i = 0; i < initial.size(); ++i) {
    const InitialRegion& region = initial[i];
    region_of[i] = sweeper.AddInitialRegion(
        region.offset, region.end, segments_of(region.group),
        region.container == kNoRegion ? kNoRegion
                                      : region_of[region.container]);
  }
  for (int64_t g : order) {
    if (const std::optional<int64_t> offset =
            sweeper.TryPlace(segments_of(g), groups[g].size)) {
      heap.Add(g, *offset, groups[g].size);
    } else {
      unplaced.push_back(g);
    }
  }
}

// FAST_MERGE's order: by start, then by the size of the first member (largest
// first), then by id.
std::vector<int64_t> ForwardOrder(absl::Span<const Group> groups,
                                  std::vector<int64_t> subset) {
  absl::c_sort(subset, [&](int64_t a, int64_t b) {
    return std::make_tuple(groups[a].start, -groups[a].own_size, groups[a].id) <
           std::make_tuple(groups[b].start, -groups[b].own_size, groups[b].id);
  });
  return subset;
}

// The mirror of ForwardOrder: by end (latest first), then by the size of the
// first member (largest first), then by id.
std::vector<int64_t> BackwardOrder(absl::Span<const Group> groups,
                                   std::vector<int64_t> subset) {
  absl::c_sort(subset, [&](int64_t a, int64_t b) {
    return std::make_tuple(-groups[a].end, -groups[a].own_size, groups[a].id) <
           std::make_tuple(-groups[b].end, -groups[b].own_size, groups[b].id);
  });
  return subset;
}

// Which peak to split the sweeps at.
enum class Peak {
  // The earliest time with the most live bytes.
  kMostLiveBytes,
  // The time when the sum of the squared sizes of the live groups is the
  // largest, so that the big groups go into the stack.
  kBiggestGroups,
};

// The order of the groups in the stack at the peak.
enum class StackOrder { kSizeDesc, kStartAsc, kEndDesc, kLifetimeDesc };

// Returns the heap time of the peak among `subset`. `subset` must not be empty.
int64_t FindPeak(absl::Span<const Group> groups,
                 absl::Span<const int64_t> subset, Peak peak) {
  // (time, +size) for every segment start and (time, -size) for every end.
  std::vector<std::pair<int64_t, int64_t>> events;
  for (int64_t g : subset) {
    const Group& group = groups[g];
    for (const Segment& segment : group.segments) {
      events.push_back({segment.start, group.size});
      events.push_back({segment.end, -group.size});
    }
  }
  absl::c_sort(events);
  size_t peak_event = 0;
  if (peak == Peak::kMostLiveBytes) {
    absl::int128 live = 0;
    absl::int128 max_live = -1;
    for (size_t e = 0; e < events.size(); ++e) {
      live += events[e].second;
      if (events[e].second > 0 && live > max_live) {
        max_live = live;
        peak_event = e;
      }
    }
  } else {
    // Exact integers, so that the result is the same on every platform.
    absl::int128 weight = 0;
    absl::int128 max_weight = -1;
    for (size_t e = 0; e < events.size(); ++e) {
      const absl::int128 size = events[e].second;
      weight += events[e].second > 0 ? size * size : -size * size;
      if (events[e].second > 0 && weight > max_weight) {
        max_weight = weight;
        peak_event = e;
      }
    }
  }
  return events[peak_event].first;
}

bool IsLiveAt(const Group& group, int64_t time) {
  return absl::c_any_of(group.segments, [&](const Segment& segment) {
    return segment.start <= time && time <= segment.end;
  });
}

// The groups around a peak.
struct PeakGroups {
  int64_t time;
  // End before the peak. In BackwardOrder.
  std::vector<int64_t> before;
  // Have a member that is live at the peak.
  std::vector<int64_t> at_peak;
  // Start after the peak. In ForwardOrder.
  std::vector<int64_t> after;
  // Have members before and after the peak, but the peak is in a gap.
  std::vector<int64_t> straddling;
};

PeakGroups GroupsAroundPeak(absl::Span<const Group> groups,
                            absl::Span<const int64_t> subset, int64_t time) {
  PeakGroups peak_groups;
  peak_groups.time = time;
  std::vector<int64_t> before;
  std::vector<int64_t> after;
  for (int64_t g : subset) {
    if (groups[g].end < time) {
      before.push_back(g);
    } else if (groups[g].start > time) {
      after.push_back(g);
    } else if (IsLiveAt(groups[g], time)) {
      peak_groups.at_peak.push_back(g);
    } else {
      peak_groups.straddling.push_back(g);
    }
  }
  peak_groups.before = BackwardOrder(groups, std::move(before));
  peak_groups.after = ForwardOrder(groups, std::move(after));
  return peak_groups;
}

// Places groups from `at_peak` that are live only during a straddling group's
// gap into the smallest free piece of such a gap, by start time. Appends
// placed groups to `stack` and `heap`, and returns which entries were placed.
std::vector<bool> PlaceAtPeakInStraddlingGaps(
    absl::Span<const Group> groups, absl::Span<const int64_t> at_peak,
    absl::Span<const std::pair<int64_t, int64_t>> gaps, int64_t alignment,
    int64_t max_heap_size, std::vector<InitialRegion>& stack,
    PlacedHeap& heap) {
  std::vector<bool> in_gap(at_peak.size(), false);
  if (gaps.empty() || at_peak.empty()) {
    return in_gap;
  }
  std::vector<int> by_gap_start(gaps.size());
  absl::c_iota(by_gap_start, 0);
  absl::c_sort(by_gap_start, [&](int a, int b) {
    return std::make_pair(gaps[a].first, a) < std::make_pair(gaps[b].first, b);
  });
  std::vector<size_t> by_start(at_peak.size());
  absl::c_iota(by_start, 0);
  absl::c_sort(by_start, [&](size_t a, size_t b) {
    const Group& ga = groups[at_peak[a]];
    const Group& gb = groups[at_peak[b]];
    return std::make_pair(ga.start, ga.id) < std::make_pair(gb.start, gb.id);
  });
  // The free pieces in the gaps: offset -> (end, index in `gaps`), and
  // (aligned size, offset) with the end of the gap.
  absl::btree_map<int64_t, std::pair<int64_t, int>> pieces;
  MaxTreap<std::pair<int64_t, int64_t>> pieces_by_size;
  auto add_piece = [&](int64_t offset, int64_t end, int straddling_index) {
    if (offset >= end) {
      return;
    }
    pieces[offset] = {end, straddling_index};
    pieces_by_size.Insert({end - RoundUpTo(offset, alignment), offset},
                          gaps[straddling_index].second);
  };
  size_t next_gap = 0;
  for (size_t i : by_start) {
    const int64_t g = at_peak[i];
    while (next_gap < by_gap_start.size() &&
           gaps[by_gap_start[next_gap]].first < groups[g].start) {
      const int straddling_index = by_gap_start[next_gap++];
      add_piece(stack[straddling_index].offset,
                std::min(stack[straddling_index].end, max_heap_size),
                straddling_index);
    }
    const std::optional<std::pair<int64_t, int64_t>> piece_key =
        pieces_by_size.FindFirst({groups[g].size, kMinInt64},
                                 groups[g].end + 1);
    if (!piece_key.has_value()) {
      continue;
    }
    const int64_t piece_offset = piece_key->second;
    auto piece = pieces.find(piece_offset);
    CHECK(piece != pieces.end());
    const auto [piece_end, straddling_index] = piece->second;
    pieces_by_size.Erase(*piece_key);
    pieces.erase(piece);
    const int64_t offset = RoundUpTo(piece_offset, alignment);
    const int64_t end = offset + groups[g].size;
    if (end < piece_end) {
      add_piece(end, piece_end, straddling_index);
    }
    heap.Add(g, offset, groups[g].size);
    stack.push_back({piece_offset, end, g, straddling_index});
    in_gap[i] = true;
  }
  return in_gap;
}

// Stacks the groups live at the peak from offset 0 in `order`, then sweeps
// backward for `peak.before` and forward for `peak.after`. Straddling groups go
// to the bottom of the stack so `at_peak` groups can use their gaps. Groups
// that don't fit in `max_heap_size` are appended to `unplaced`.
void PlaceAroundPeak(absl::Span<const Group> groups, const PeakGroups& peak,
                     StackOrder order, int64_t alignment, int64_t max_heap_size,
                     Side side, PlacedHeap& heap,
                     std::vector<int64_t>& unplaced) {
  auto key = [&](int64_t g) -> std::pair<int64_t, int64_t> {
    const Group& group = groups[g];
    switch (order) {
      case StackOrder::kSizeDesc:
        return {-group.size, group.id};
      case StackOrder::kStartAsc:
        return {group.start, group.id};
      case StackOrder::kEndDesc:
        return {-group.end, group.id};
      case StackOrder::kLifetimeDesc:
        return {-(group.end - group.start), group.id};
    }
    return {0, group.id};
  };
  auto by_key = [&](int64_t a, int64_t b) { return key(a) < key(b); };
  std::vector<int64_t> at_peak = peak.at_peak;
  std::vector<int64_t> straddling = peak.straddling;
  absl::c_sort(at_peak, by_key);
  absl::c_sort(straddling, by_key);

  std::vector<InitialRegion> stack;
  stack.reserve(at_peak.size() + straddling.size());
  int64_t top = 0;
  auto try_push = [&](int64_t g) -> bool {
    const int64_t size = groups[g].size;
    if (!stack.empty() && (top > max_heap_size || size > max_heap_size - top)) {
      return false;
    }
    const int64_t end = top + size;
    const int64_t next =
        end > kHeapEnd - alignment ? kHeapEnd : RoundUpTo(end, alignment);
    const int64_t region_end = std::max(end, std::min(next, max_heap_size));
    heap.Add(g, top, size);
    stack.push_back({top, region_end, g, kNoRegion});
    top = next;
    return true;
  };
  // The placed straddling groups go to the bottom of the stack, so that their
  // index in `gaps` is also their index in `stack`. The gap of each one around
  // the peak: (end of the segment before, start of the segment after).
  std::vector<std::pair<int64_t, int64_t>> gaps;
  gaps.reserve(straddling.size());
  for (int64_t g : straddling) {
    if (!try_push(g)) {
      unplaced.push_back(g);
      continue;
    }
    int64_t gap_start = kMinInt64;
    int64_t gap_end = kInfinity;
    for (const Segment& segment : groups[g].segments) {
      if (segment.end < peak.time) {
        gap_start = std::max(gap_start, segment.end);
      }
      if (segment.start > peak.time) {
        gap_end = std::min(gap_end, segment.start);
      }
    }
    gaps.push_back({gap_start, gap_end});
  }
  const std::vector<bool> in_gap = PlaceAtPeakInStraddlingGaps(
      groups, at_peak, gaps, alignment, max_heap_size, stack, heap);
  for (size_t i = 0; i < at_peak.size(); ++i) {
    if (!in_gap[i] && !try_push(at_peak[i])) {
      unplaced.push_back(at_peak[i]);
    }
  }
  Sweep(groups, peak.before, /*forward=*/false, stack, peak.time, alignment,
        max_heap_size, side, heap, unplaced);
  Sweep(groups, peak.after, /*forward=*/true, stack, peak.time, alignment,
        max_heap_size, side, heap, unplaced);
}

// The peak and the stack order of a direction from a peak, or nullopt for
// kForward and kBackward.
std::optional<std::pair<Peak, StackOrder>> PeakAndStackOrder(
    Direction direction) {
  switch (direction) {
    case Direction::kForward:
    case Direction::kBackward:
      return std::nullopt;
    case Direction::kFromPeakBySize:
      return std::make_pair(Peak::kMostLiveBytes, StackOrder::kSizeDesc);
    case Direction::kFromPeakByStart:
      return std::make_pair(Peak::kMostLiveBytes, StackOrder::kStartAsc);
    case Direction::kFromPeakByEnd:
      return std::make_pair(Peak::kMostLiveBytes, StackOrder::kEndDesc);
    case Direction::kFromPeakByLifetime:
      return std::make_pair(Peak::kMostLiveBytes, StackOrder::kLifetimeDesc);
    case Direction::kFromBigPeakByStart:
      return std::make_pair(Peak::kBiggestGroups, StackOrder::kStartAsc);
    case Direction::kFromBigPeakByEnd:
      return std::make_pair(Peak::kBiggestGroups, StackOrder::kEndDesc);
  }
  return std::nullopt;
}

// Packs `groups` into one or more heaps with each packing in `packings`, and
// returns the heaps of the first packing with the smallest total size.
// `groups` and `packings` must not be empty.
std::vector<PlacedHeap> PlaceGroups(
    absl::Span<const Group> groups,
    absl::Span<const MultiSweepHeap::Packing> packings, int64_t alignment,
    uint64_t size_limit_per_heap) {
  const int64_t max_heap_size =
      size_limit_per_heap >= static_cast<uint64_t>(kHeapEnd)
          ? kHeapEnd
          : static_cast<int64_t>(size_limit_per_heap);
  std::vector<int64_t> all_groups(groups.size());
  absl::c_iota(all_groups, 0);

  // The full-set orders and peaks are computed once, when a packing first
  // needs them.
  std::optional<std::vector<int64_t>> forward_order;
  std::optional<std::vector<int64_t>> backward_order;
  std::optional<PeakGroups> most_live_bytes;
  std::optional<PeakGroups> biggest_groups;
  auto order = [&](bool forward) -> const std::vector<int64_t>& {
    std::optional<std::vector<int64_t>>& cached =
        forward ? forward_order : backward_order;
    if (!cached.has_value()) {
      cached = forward ? ForwardOrder(groups, all_groups)
                       : BackwardOrder(groups, all_groups);
    }
    return *cached;
  };
  auto initial_peak_groups = [&](Peak peak) -> const PeakGroups& {
    std::optional<PeakGroups>& cached =
        peak == Peak::kMostLiveBytes ? most_live_bytes : biggest_groups;
    if (!cached.has_value()) {
      cached = GroupsAroundPeak(groups, all_groups,
                                FindPeak(groups, all_groups, peak));
    }
    return *cached;
  };

  std::vector<PlacedHeap> best_heaps;
  std::optional<absl::int128> best_total_size;
  for (const MultiSweepHeap::Packing& packing : packings) {
    const std::optional<std::pair<Peak, StackOrder>> peak_and_order =
        PeakAndStackOrder(packing.direction);
    std::vector<PlacedHeap> heaps;
    absl::int128 total_size = 0;
    std::vector<int64_t> remaining;
    do {
      PlacedHeap heap;
      std::vector<int64_t> unplaced;
      if (peak_and_order.has_value()) {
        const auto [peak_kind, stack_order] = *peak_and_order;
        if (heaps.empty()) {
          PlaceAroundPeak(groups, initial_peak_groups(peak_kind), stack_order,
                          alignment, max_heap_size, packing.side, heap,
                          unplaced);
        } else {
          PlaceAroundPeak(
              groups,
              GroupsAroundPeak(groups, remaining,
                               FindPeak(groups, remaining, peak_kind)),
              stack_order, alignment, max_heap_size, packing.side, heap,
              unplaced);
        }
      } else {
        const bool forward = packing.direction == Direction::kForward;
        Sweep(groups,
              heaps.empty() ? absl::Span<const int64_t>(order(forward))
                            : absl::Span<const int64_t>(remaining),
              forward, /*initial=*/{}, /*start_time=*/0, alignment,
              max_heap_size, packing.side, heap, unplaced);
      }
      total_size += heap.size;
      heaps.push_back(std::move(heap));
      remaining = std::move(unplaced);
    } while (!remaining.empty() &&
             (!best_total_size.has_value() || total_size < *best_total_size));
    if (!best_total_size.has_value() || total_size < *best_total_size) {
      best_total_size = total_size;
      best_heaps = std::move(heaps);
    }
  }
  return best_heaps;
}

}  // namespace

std::vector<MultiSweepHeap::Packing> MultiSweepHeap::AllPackings() {
  std::vector<Packing> packings;
  for (Side side : {Side::kLow, Side::kNearestRelease}) {
    for (Direction direction :
         {Direction::kForward, Direction::kBackward, Direction::kFromPeakBySize,
          Direction::kFromPeakByStart, Direction::kFromPeakByEnd,
          Direction::kFromPeakByLifetime, Direction::kFromBigPeakByStart,
          Direction::kFromBigPeakByEnd}) {
      packings.push_back({direction, side});
    }
  }
  return packings;
}

MultiSweepHeap::MultiSweepHeap(int64_t alignment, uint64_t size_limit_per_heap,
                               std::vector<Packing> packings)
    : alignment_(alignment),
      size_limit_per_heap_(size_limit_per_heap),
      packings_(std::move(packings)) {
  CHECK_GT(alignment_, 0) << "Alignment (" << alignment_
                          << ") must be positive.";
  CHECK(!packings_.empty());
}

void MultiSweepHeap::Alloc(const HloValue* buffer, int64_t size) {
  CHECK_GE(size, 0);
  if (size == 0) {
    zero_size_buffers_.push_back(buffer);
    return;
  }
  const int64_t index = intervals_.size();
  CHECK(interval_index_.emplace(buffer, index).second)
      << "Alloc called twice for " << buffer->ToShortString();
  intervals_.push_back({buffer, size, current_time_, /*end=*/-1,
                        /*group=*/index});
  ++current_time_;
}

void MultiSweepHeap::ShareWith(const HloValue* buffer,
                               const HloValue* share_with, int64_t size) {
  CHECK_GE(size, 0);
  if (size == 0) {
    zero_size_buffers_.push_back(buffer);
    return;
  }
  auto it = interval_index_.find(share_with);
  CHECK(it != interval_index_.end())
      << "ShareWith called with the unknown " << share_with->ToShortString();
  const int64_t group = intervals_[it->second].group;
  const int64_t index = intervals_.size();
  CHECK(interval_index_.emplace(buffer, index).second)
      << "ShareWith called for the allocated " << buffer->ToShortString();
  intervals_.push_back({buffer, size, current_time_, /*end=*/-1, group});
  ++current_time_;
}

void MultiSweepHeap::Free(const HloValue* buffer, int64_t size) {
  if (size == 0) {
    return;
  }
  auto it = interval_index_.find(buffer);
  CHECK(it != interval_index_.end())
      << "Free called for the unknown " << buffer->ToShortString();
  Interval& interval = intervals_[it->second];
  CHECK_EQ(interval.size, size) << buffer->ToShortString();
  CHECK_EQ(interval.end, -1)
      << "Free called twice for " << buffer->ToShortString();
  interval.end = current_time_;
  ++current_time_;
}

absl::StatusOr<HeapSimulator::Result<HloValue>> MultiSweepHeap::Finish() {
  // One group per colocation group, in the order of their first members.
  std::vector<Group> groups;
  // The indices in `intervals_` of the members of each group.
  std::vector<std::vector<int64_t>> members;
  // The index in `groups` of each first member.
  std::vector<int64_t> group_index(intervals_.size(), -1);
  for (int64_t i = 0; i < static_cast<int64_t>(intervals_.size()); ++i) {
    const Interval& interval = intervals_[i];
    const int64_t end = interval.end == -1 ? current_time_ : interval.end;
    if (interval.group == i) {
      group_index[i] = static_cast<int64_t>(groups.size());
      groups.push_back(Group{interval.start, end, interval.size, interval.size,
                             interval.buffer->id(),
                             /*segments=*/{}, /*mirrored_segments=*/{}});
      members.emplace_back();
    }
    const int64_t g = group_index[interval.group];
    Group& group = groups[g];
    group.end = std::max(group.end, end);
    group.size = std::max(group.size, interval.size);
    if (group.size > kHeapEnd) {
      return absl::ResourceExhaustedError(
          "Buffer group size exceeds maximum heap size");
    }
    members[g].push_back(i);
    // The members come in the order of their starts, so the segments stay
    // sorted.
    if (!group.segments.empty() &&
        interval.start <= group.segments.back().end) {
      group.segments.back().end = std::max(group.segments.back().end, end);
    } else {
      group.segments.push_back({interval.start, end});
    }
  }
  for (Group& group : groups) {
    group.mirrored_segments.reserve(group.segments.size());
    for (auto it = group.segments.rbegin(); it != group.segments.rend(); ++it) {
      group.mirrored_segments.push_back({-it->end, -it->start});
    }
  }

  Result result;
  if (groups.empty()) {
    HeapResult heap;
    for (const HloValue* buffer : zero_size_buffers_) {
      heap.chunk_map[buffer] = Chunk::FromOffsetSize(0, 0);
    }
    result.heap_results.push_back(std::move(heap));
    return result;
  }

  const std::vector<PlacedHeap> heaps =
      PlaceGroups(groups, packings_, alignment_, size_limit_per_heap_);
  CHECK(!heaps.empty());
  result.heap_results.reserve(heaps.size());
  for (const PlacedHeap& placed_heap : heaps) {
    HeapResult heap;
    heap.heap_size = placed_heap.size;
    for (const auto [g, offset] : placed_heap.placements) {
      // Like in GlobalDecreasingSizeBestFitHeap, every member gets the
      // group's chunk (with the size of the largest member).
      const Chunk chunk = Chunk::FromOffsetSize(offset, groups[g].size);
      for (int64_t m : members[g]) {
        heap.chunk_map[intervals_[m].buffer] = chunk;
      }
    }
    if (result.heap_size >
        std::numeric_limits<int64_t>::max() - heap.heap_size) {
      return absl::ResourceExhaustedError("Total heap size overflows int64_t");
    }
    result.heap_size += heap.heap_size;
    result.heap_results.push_back(std::move(heap));
  }
  for (const HloValue* buffer : zero_size_buffers_) {
    result.heap_results[0].chunk_map[buffer] = Chunk::FromOffsetSize(0, 0);
  }
  return result;
}

}  // namespace xla
