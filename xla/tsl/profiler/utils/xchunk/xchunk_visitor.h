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

#ifndef XLA_TSL_PROFILER_UTILS_XCHUNK_XCHUNK_VISITOR_H_
#define XLA_TSL_PROFILER_UTILS_XCHUNK_XCHUNK_VISITOR_H_

// Read only visitors over the XChunk stream messages. The class and method
// names deliberately mirror tsl::profiler::XPlaneVisitor / XLineVisitor /
// XEventVisitor so that code written against XPlane can be ported to XChunk
// (and back) mechanically.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/xchunk/xchunk.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"

namespace tsl::profiler::xchunk {

namespace internal {

inline int64_t SaturateNanoToPico(int64_t ts_ns) {
  constexpr int64_t kMaxNs = std::numeric_limits<int64_t>::max() / 1000;
  constexpr int64_t kMinNs = std::numeric_limits<int64_t>::min() / 1000;
  if (ts_ns > kMaxNs) {
    return std::numeric_limits<int64_t>::max();
  }
  if (ts_ns < kMinNs) {
    return std::numeric_limits<int64_t>::min();
  }
  return ts_ns * 1000;
}

inline int64_t SaturatingAddInt64(int64_t a, int64_t b) {
  if (b > 0 && a > std::numeric_limits<int64_t>::max() - b) {
    return std::numeric_limits<int64_t>::max();
  }
  if (b < 0 && a < std::numeric_limits<int64_t>::min() - b) {
    return std::numeric_limits<int64_t>::min();
  }
  return a + b;
}

}  // namespace internal

// Mirrors tsl::profiler::XLineVisitor.
class XChunkLineVisitor {
 public:
  explicit XChunkLineVisitor(const XChunkLine* line) : line_(line) {}

  int64_t Id() const { return line_->line_id(); }
  int64_t DisplayId() const {
    return line_->display_id() != 0 ? line_->display_id() : line_->line_id();
  }
  absl::string_view Name() const { return line_->name(); }
  absl::string_view DisplayName() const {
    return line_->display_name().empty() ? line_->name()
                                         : line_->display_name();
  }
  int64_t TimestampNs() const { return line_->timestamp_ns(); }
  int64_t DurationPs() const { return line_->duration_ps(); }
  int64_t NumEvents() const { return line_->num_events(); }

  const XChunkLine& RawLine() const { return *line_; }

 private:
  const XChunkLine* line_;
};

// Mirrors tsl::profiler::XPlaneVisitor.
class XChunkPlaneVisitor {
 public:
  explicit XChunkPlaneVisitor(const XChunkPlane* plane) : plane_(plane) {
    lines_by_id_.reserve(plane->lines_size());
    for (const XChunkLine& line : plane->lines()) {
      lines_by_id_.emplace(line.line_id(), &line);
    }
  }

  uint32_t Index() const { return plane_->plane_index(); }
  int64_t Id() const { return plane_->plane_id(); }
  absl::string_view Name() const { return plane_->plane_name(); }
  uint32_t ProcessId() const { return plane_->process_id(); }
  absl::string_view ProcessName() const { return plane_->process_name(); }

  size_t NumLines() const { return plane_->lines_size(); }

  template <typename ForEachLineFunc>
  void ForEachLine(ForEachLineFunc&& for_each_line) const {
    for (const XChunkLine& line : plane_->lines()) {
      for_each_line(XChunkLineVisitor(&line));
    }
  }

  std::optional<XChunkLineVisitor> GetLine(int64_t line_id) const {
    auto it = lines_by_id_.find(line_id);
    if (it == lines_by_id_.end()) {
      return std::nullopt;
    }
    return XChunkLineVisitor(it->second);
  }

  // Dense event metadata table.
  size_t NumEventMetadata() const { return plane_->event_names_size(); }
  int64_t EventMetadataId(uint32_t index) const {
    return plane_->event_metadata_ids(index);
  }
  absl::string_view EventName(uint32_t index) const {
    return plane_->event_names(index);
  }
  bool HasEventDisplayName(uint32_t index) const {
    return index < plane_->event_display_names_size() &&
           !plane_->event_display_names(index).empty();
  }
  absl::string_view EventDisplayName(uint32_t index) const {
    return HasEventDisplayName(index)
               ? absl::string_view(plane_->event_display_names(index))
               : absl::string_view();
  }

  // Dense stat metadata table.
  size_t NumStatMetadata() const { return plane_->stat_names_size(); }
  absl::string_view StatName(uint32_t index) const {
    return plane_->stat_names(index);
  }

  const XChunkPlane& RawPlane() const { return *plane_; }

 private:
  const XChunkPlane* plane_;
  absl::flat_hash_map<int64_t, const XChunkLine*> lines_by_id_;
};

// One decoded event of an XChunkEvents message. Mirrors the parts of
// tsl::profiler::XEventVisitor that trace and analysis tools need, with the
// special stats (flow, group id, async, step name, counter value) already
// decoded.
struct XChunkEventVisitor {
  const XChunkPlaneVisitor* plane = nullptr;
  const XChunkLineVisitor* line = nullptr;
  int64_t line_base_ps = 0;
  uint32_t index = 0;  // index within the chunk
  uint32_t metadata_index = 0;
  int64_t offset_ps = 0;
  int64_t duration_ps = 0;
  std::optional<XFlow> flow;
  std::optional<int64_t> group_id;
  bool is_async = false;
  std::optional<absl::string_view> name_override;
  std::optional<uint64_t> counter_uint64;
  std::optional<double> counter_double;

  absl::string_view Name() const { return plane->EventName(metadata_index); }
  bool HasDisplayName() const {
    return plane->HasEventDisplayName(metadata_index);
  }
  absl::string_view DisplayName() const {
    return plane->EventDisplayName(metadata_index);
  }
  // The name the trace viewer shows: the kStepName override if present,
  // otherwise the display name, otherwise the name.
  absl::string_view TraceViewerName() const {
    if (name_override.has_value()) {
      return *name_override;
    }
    return HasDisplayName() ? DisplayName() : Name();
  }

  int64_t OffsetPs() const { return offset_ps; }
  int64_t TimestampPs() const {
    return internal::SaturatingAddInt64(line_base_ps, offset_ps);
  }
  int64_t DurationPs() const {
    return static_cast<int64_t>(static_cast<uint64_t>(duration_ps) &
                                kDurationValueMask);
  }
  uint64_t RawDurationPs() const { return static_cast<uint64_t>(duration_ps); }
  bool IsPhaseBegin() const {
    return (static_cast<uint64_t>(duration_ps) & kDurationPhaseBeginBit) != 0;
  }
  bool IsPhaseEnd() const {
    return (static_cast<uint64_t>(duration_ps) & kDurationPhaseEndBit) != 0;
  }
  int64_t EndTimestampPs() const {
    return internal::SaturatingAddInt64(TimestampPs(), DurationPs());
  }
};

namespace internal {

// Walks a sparse column whose event_index is strictly increasing.
template <typename Column>
class SparseCursor {
 public:
  explicit SparseCursor(const Column& column)
      : column_(column), size_(column.event_index_size()) {}

  // Returns true if the column has a value for event `event_index`. Must be
  // called with increasing event indices.
  bool Advance(uint32_t event_index) {
    if (pos_ < size_ && column_.event_index(pos_) == event_index) {
      current_ = pos_++;
      return true;
    }
    return false;
  }

  // Returns by reference for string columns (no copy, no dangling views).
  decltype(auto) Value() const { return column_.values(current_); }

 private:
  const Column& column_;
  int size_;
  int pos_ = 0;
  int current_ = 0;
};

template <typename Column>
SparseCursor(const Column&) -> SparseCursor<Column>;

}  // namespace internal

// Decodes every event of `chunk`, in order, calling
// `for_each_event(const XChunkEventVisitor&)`.
// REQUIRED: `line` is the line the chunk belongs to (see
// XChunkPlaneVisitor::GetLine(chunk.line_id())).
template <typename ForEachEventFunc>
void ForEachEvent(const XChunkPlaneVisitor& plane,
                  const XChunkLineVisitor& line, const XChunkEvents& chunk,
                  ForEachEventFunc&& for_each_event) {
  XChunkEventVisitor event;
  event.plane = &plane;
  event.line = &line;
  event.line_base_ps = internal::SaturateNanoToPico(line.TimestampNs());
  int64_t offset_ps = 0;
  const uint32_t num_events = chunk.num_events();
  if (chunk.flow().empty() && chunk.group_id().empty() &&
      chunk.async_event_index_size() == 0 && chunk.name_override().empty() &&
      chunk.counter_uint64().empty() && chunk.counter_double().empty()) {
    for (uint32_t i = 0; i < num_events; ++i) {
      offset_ps =
          internal::SaturatingAddInt64(offset_ps, chunk.offset_ps_deltas(i));
      event.index = i;
      event.metadata_index = chunk.event_metadata_index(i);
      event.offset_ps = offset_ps;
      event.duration_ps = chunk.duration_ps(i);
      for_each_event(event);
    }
    return;
  }

  internal::SparseCursor flow(chunk.flow());
  internal::SparseCursor group_id(chunk.group_id());
  internal::SparseCursor name_override(chunk.name_override());
  internal::SparseCursor counter_uint64(chunk.counter_uint64());
  internal::SparseCursor counter_double(chunk.counter_double());
  int async_pos = 0;
  const int async_size = chunk.async_event_index_size();
  for (uint32_t i = 0; i < num_events; ++i) {
    offset_ps =
        internal::SaturatingAddInt64(offset_ps, chunk.offset_ps_deltas(i));
    event.index = i;
    event.metadata_index = chunk.event_metadata_index(i);
    event.offset_ps = offset_ps;
    event.duration_ps = chunk.duration_ps(i);
    event.flow.reset();
    if (flow.Advance(i)) {
      event.flow = XFlow::FromStatValue(flow.Value());
    }
    event.group_id.reset();
    if (group_id.Advance(i)) {
      event.group_id = group_id.Value();
    }
    event.is_async = false;
    if (async_pos < async_size && chunk.async_event_index(async_pos) == i) {
      event.is_async = true;
      ++async_pos;
    }
    event.name_override.reset();
    if (name_override.Advance(i)) {
      event.name_override = absl::string_view(name_override.Value());
    }
    event.counter_uint64.reset();
    if (counter_uint64.Advance(i)) {
      event.counter_uint64 = counter_uint64.Value();
    }
    event.counter_double.reset();
    if (counter_double.Advance(i)) {
      event.counter_double = counter_double.Value();
    }
    for_each_event(event);
  }
}

}  // namespace tsl::profiler::xchunk

#endif  // XLA_TSL_PROFILER_UTILS_XCHUNK_XCHUNK_VISITOR_H_
